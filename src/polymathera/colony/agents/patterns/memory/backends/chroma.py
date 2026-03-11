"""ChromaDB-based vector storage backend for semantic search.

Provides semantic similarity search over memory entries using ChromaDB
in embedded mode (in-process, no external container needed).

Uses sentence-transformers for embeddings (all-MiniLM-L6-v2 by default).
Data persists to a configurable directory on the shared volume.

This backend is NOT a full StorageBackend — it's a secondary index used
by HybridStorageBackend for vector operations. The blackboard remains
the source of truth for all entry data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Module-level ChromaDB client singleton (one per process)
_chroma_client = None
_chroma_lock = asyncio.Lock()

DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/tmp/colony_chromadb")
DEFAULT_COLLECTION_PREFIX = "colony_memory"


def _get_text_representation(value: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Extract embeddable text from an entry's value and metadata.

    Strategy: For known data types, extract meaningful text fields.
    Fallback: JSON-serialize the value dict.
    """
    parts: list[str] = []

    data_type = metadata.get("data_type", "")

    # Action entries: action_type + parameters + result
    if data_type == "Action" or "action_type" in value:
        action_type = value.get("action_type", "")
        if action_type:
            parts.append(f"action: {action_type}")
        params = value.get("parameters")
        if params:
            if isinstance(params, dict):
                parts.append(f"parameters: {json.dumps(params, default=str)[:500]}")
            else:
                parts.append(f"parameters: {str(params)[:500]}")
        result = value.get("result")
        if result and isinstance(result, dict):
            output = result.get("output")
            if output:
                parts.append(f"result: {str(output)[:500]}")

    # MemoryRecord entries: content text
    elif data_type == "MemoryRecord" or "content" in value:
        content = value.get("content")
        if isinstance(content, dict):
            text = content.get("text", "")
            if text:
                parts.append(text)
            else:
                parts.append(json.dumps(content, default=str)[:1000])
        elif isinstance(content, str):
            parts.append(content)

    # ActionPlan entries
    elif data_type == "ActionPlan" or "goals" in value:
        goals = value.get("goals")
        if goals:
            parts.append(f"goals: {goals}")
        actions = value.get("actions")
        if actions and isinstance(actions, list):
            action_types = [a.get("action_type", "") for a in actions if isinstance(a, dict)]
            parts.append(f"plan actions: {', '.join(action_types)}")

    # Fallback: serialize the whole value
    if not parts:
        try:
            parts.append(json.dumps(value, default=str)[:1500])
        except (TypeError, ValueError):
            parts.append(str(value)[:1500])

    return "\n".join(parts)


class ChromaStorageBackend:
    """Vector storage backend using ChromaDB (embedded mode).

    Each scope_id maps to a ChromaDB collection. Collections are isolated.
    Embeddings are computed from text representation of entry values.

    This is used as the secondary backend in HybridStorageBackend —
    it only handles vector-specific operations (add, search, delete).
    """

    def __init__(self, scope_id: str, persist_dir: str | None = None):
        self._scope_id = scope_id
        self._persist_dir = persist_dir or os.environ.get(
            "COLONY_CHROMADB_DIR", DEFAULT_PERSIST_DIR
        )
        self._collection = None
        self._initialized = False

    @property
    def scope_id(self) -> str:
        return self._scope_id

    async def _ensure_initialized(self) -> None:
        """Lazily initialize the ChromaDB client and collection."""
        if self._initialized:
            return

        global _chroma_client
        async with _chroma_lock:
            if _chroma_client is None:
                # Initialize ChromaDB in a thread to avoid blocking
                def _init_client():
                    import chromadb
                    os.makedirs(self._persist_dir, exist_ok=True)
                    return chromadb.PersistentClient(path=self._persist_dir)

                _chroma_client = await asyncio.to_thread(_init_client)
                logger.info(f"ChromaDB client initialized at {self._persist_dir}")

        # Sanitize collection name: ChromaDB requires 3-63 chars, alphanumeric + underscores
        collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{self._scope_id}"
        collection_name = collection_name.replace(":", "_").replace("-", "_")
        # Truncate if needed, keeping it unique
        if len(collection_name) > 63:
            import hashlib
            suffix = hashlib.md5(collection_name.encode()).hexdigest()[:8]
            collection_name = collection_name[:54] + "_" + suffix

        def _get_collection():
            return _chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"scope_id": self._scope_id},
            )

        self._collection = await asyncio.to_thread(_get_collection)
        self._initialized = True
        logger.info(
            f"ChromaStorageBackend initialized: scope={self._scope_id}, "
            f"collection={collection_name}"
        )

    async def add(
        self,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any],
        tags: set[str],
    ) -> None:
        """Add or update a document embedding.

        Args:
            key: Entry key (used as ChromaDB document ID)
            value: Entry value dict (used to extract embeddable text)
            metadata: Entry metadata
            tags: Entry tags (stored as ChromaDB metadata for filtering)
        """
        await self._ensure_initialized()

        text = _get_text_representation(value, metadata)
        if not text or not text.strip():
            return

        # ChromaDB metadata must be flat (str, int, float, bool)
        chroma_meta: dict[str, str | int | float | bool] = {
            "scope_id": self._scope_id,
        }
        # Store tags as comma-separated string for metadata filtering
        if tags:
            chroma_meta["tags_csv"] = ",".join(sorted(tags))
            # Also store individual common tags as boolean fields for efficient filtering
            for tag in tags:
                safe_key = f"tag_{tag.replace(':', '_').replace('-', '_')}"
                if len(safe_key) <= 64:
                    chroma_meta[safe_key] = True

        if "data_type" in metadata:
            chroma_meta["data_type"] = metadata["data_type"]

        def _upsert():
            self._collection.upsert(
                ids=[key],
                documents=[text],
                metadatas=[chroma_meta],
            )

        await asyncio.to_thread(_upsert)

    async def search(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Semantic search by text similarity.

        Args:
            query_text: Natural language query
            n_results: Max results to return
            where: Optional ChromaDB where clause for metadata filtering

        Returns:
            List of (key, similarity_score) tuples, highest similarity first.
            Scores are normalized to 0-1 range.
        """
        await self._ensure_initialized()

        def _query():
            kwargs: dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": n_results,
            }
            if where:
                kwargs["where"] = where
            return self._collection.query(**kwargs)

        try:
            results = await asyncio.to_thread(_query)
        except Exception as e:
            logger.warning(f"ChromaDB search failed for scope {self._scope_id}: {e}")
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        # ChromaDB returns distances (lower = more similar for default L2)
        # Convert to similarity scores (higher = more similar)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        scored: list[tuple[str, float]] = []
        for doc_id, distance in zip(ids, distances):
            # ChromaDB default uses L2 distance. Convert to 0-1 similarity.
            # similarity = 1 / (1 + distance) gives 1.0 at distance=0, ~0 at large distance
            similarity = 1.0 / (1.0 + distance)
            scored.append((doc_id, similarity))

        return scored

    async def delete(self, key: str) -> bool:
        """Remove a document by key.

        Args:
            key: Document ID to remove

        Returns:
            True if operation completed (ChromaDB doesn't confirm existence)
        """
        await self._ensure_initialized()

        def _delete():
            try:
                self._collection.delete(ids=[key])
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_delete)

    async def count(self) -> int:
        """Count total documents in this collection."""
        await self._ensure_initialized()
        return await asyncio.to_thread(self._collection.count)

    async def clear(self) -> int:
        """Delete all documents in this collection."""
        await self._ensure_initialized()
        count = await self.count()
        if count > 0:
            def _clear():
                # Get all IDs and delete them
                all_data = self._collection.get()
                if all_data["ids"]:
                    self._collection.delete(ids=all_data["ids"])
            await asyncio.to_thread(_clear)
        return count
