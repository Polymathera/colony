"""Hybrid storage backend: BlackboardStorageBackend + ChromaStorageBackend.

Implements the full StorageBackend protocol with dual-write semantics:
- Blackboard: source of truth for all entry data (structured storage, events, OCC)
- ChromaDB: secondary index for semantic similarity search

Write operations go to both backends. Read/query operations route to the
appropriate backend based on query type. Event streaming delegates to blackboard.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from ....blackboard.types import BlackboardEntry, BlackboardEvent
from .blackboard import BlackboardStorageBackend, BlackboardStorageBackendFactory
from .chroma import ChromaStorageBackend

if TYPE_CHECKING:
    from ....base import Agent
    from ..types import ScoredEntry

logger = logging.getLogger(__name__)


class HybridStorageBackend:
    """Composite backend: BlackboardStorageBackend + ChromaStorageBackend.

    Implements the full StorageBackend protocol. Routes operations:
    - write() → both (blackboard for data, chroma for embeddings)
    - read() → blackboard (source of truth)
    - query() → blackboard (logical queries)
    - search_semantic() → chroma search + blackboard hydration
    - delete() → both
    - stream_events_to_queue() → blackboard
    """

    def __init__(
        self,
        blackboard_backend: BlackboardStorageBackend,
        chroma_backend: ChromaStorageBackend,
    ):
        self._blackboard = blackboard_backend
        self._chroma = chroma_backend

    @property
    def scope_id(self) -> str:
        return self._blackboard.scope_id

    async def write(
        self,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any],
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Write to blackboard AND index in ChromaDB."""
        # Blackboard write is the primary operation (source of truth + events)
        await self._blackboard.write(key, value, metadata, tags, ttl_seconds)

        # Index in ChromaDB for semantic search (best-effort, don't block on failure)
        try:
            await self._chroma.add(key, value, metadata, tags or set())
        except Exception as e:
            logger.warning(
                f"Failed to index in ChromaDB (key={key}): {e}. "
                "Semantic search may miss this entry."
            )

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read from blackboard (source of truth)."""
        return await self._blackboard.read(key)

    async def query(
        self,
        pattern: str | None = None,
        tags: set[str] | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
    ) -> list[BlackboardEntry]:
        """Logical query — delegates to blackboard."""
        return await self._blackboard.query(pattern, tags, time_range, limit)

    async def search_semantic(
        self,
        query_text: str,
        n_results: int = 10,
        tag_filter_where: dict | None = None,
    ) -> list["ScoredEntry"]:
        """Semantic search — ChromaDB vector search + blackboard hydration.

        Args:
            query_text: Natural language query
            n_results: Max results
            tag_filter_where: Optional ChromaDB where clause for pre-filtering

        Returns:
            List of ScoredEntry with semantic similarity scores,
            hydrated with full entry data from blackboard.
        """
        from ..types import ScoredEntry

        results = await self._chroma.search(query_text, n_results, tag_filter_where)

        scored: list[ScoredEntry] = []
        for key, similarity in results:
            entry = await self._blackboard.read(key)
            if entry is not None:
                scored.append(ScoredEntry(
                    entry=entry,
                    score=similarity,
                    components={"semantic": similarity},
                ))

        return scored

    async def delete(self, key: str) -> bool:
        """Delete from both backends."""
        # Delete from chroma first (secondary, best-effort)
        try:
            await self._chroma.delete(key)
        except Exception as e:
            logger.debug(f"ChromaDB delete failed for {key}: {e}")

        # Delete from blackboard (primary, must succeed)
        return await self._blackboard.delete(key)

    async def count(self) -> int:
        """Count entries in blackboard (source of truth)."""
        return await self._blackboard.count()

    async def clear(self) -> int:
        """Clear both backends."""
        # Clear chroma first (best-effort)
        try:
            await self._chroma.clear()
        except Exception as e:
            logger.debug(f"ChromaDB clear failed: {e}")

        return await self._blackboard.clear()

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        key_pattern: str,
        consumer_group: str | None = None,
        consumer_name: str | None = None,
    ) -> None:
        """Event streaming — delegates to blackboard."""
        await self._blackboard.stream_events_to_queue(
            event_queue, key_pattern, consumer_group, consumer_name,
        )


class HybridStorageBackendFactory:
    """Factory that creates HybridStorageBackend instances.

    Falls back to BlackboardStorageBackend if ChromaDB is not available
    (e.g., chromadb package not installed).
    """

    def __init__(self, agent: "Agent", chroma_persist_dir: str | None = None):
        self._agent = agent
        self._persist_dir = chroma_persist_dir
        self._chroma_available = self._check_chroma_available()
        if not self._chroma_available:
            logger.info(
                "ChromaDB not available — memory will use blackboard-only storage. "
                "Semantic queries will return empty results. "
                "Install chromadb to enable semantic search."
            )

    @staticmethod
    def _check_chroma_available() -> bool:
        try:
            import chromadb  # noqa: F401
            return True
        except ImportError:
            return False

    async def create_for_scope(self, scope_id: str) -> BlackboardStorageBackend | HybridStorageBackend:
        """Create a storage backend for the given scope.

        Returns HybridStorageBackend if ChromaDB is available,
        otherwise falls back to BlackboardStorageBackend.
        """
        blackboard = await self._agent.get_blackboard(
            scope="shared", scope_id=scope_id,
        )
        bb_backend = BlackboardStorageBackend(
            scope_id=scope_id,
            blackboard=blackboard,
            agent_id=self._agent.agent_id,
        )

        if not self._chroma_available:
            return bb_backend

        chroma_backend = ChromaStorageBackend(
            scope_id=scope_id,
            persist_dir=self._persist_dir,
        )
        return HybridStorageBackend(bb_backend, chroma_backend)
