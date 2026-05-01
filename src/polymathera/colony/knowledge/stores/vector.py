"""``VectorStore`` ABC + two implementations.

The corpus-wide vector store is a separate layer from the per-agent
memory backends in ``agents/patterns/memory/backends/`` (master §3.2).
Same Protocol-style interface, different scope. The store keeps every
``EmbeddedChunk`` keyed by ``chunk_id``, indexed by the source URI
prefix and ``data_type`` so the §6.4 retrieval modes can filter
without scanning the whole index.

Two implementations:

- ``InMemoryVectorStore`` — full implementation; cosine similarity in
  pure Python. Used in unit tests and small deployments.
- ``QdrantVectorStore`` — stub. Constructor accepts a
  ``qdrant_client.AsyncQdrantClient`` (or any object that exposes
  ``upsert`` / ``search``) and translates calls. The stub here
  raises ``NotImplementedError`` until C1b lands the real Docker
  wiring; the ABC + stub are present so consumers can import / type
  against the symbol now.
"""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..models import (
    Chunk,
    CorpusTier,
    EmbeddedChunk,
    RetrievalQuery,
    RetrievalHit,
)


class VectorStoreError(RuntimeError):
    """Base error for the vector store."""


class VectorStore(ABC):
    """Corpus-wide vector index with citation-aware retrieval."""

    @abstractmethod
    async def upsert(self, items: Sequence[EmbeddedChunk]) -> int:
        """Insert or replace chunks. Returns the count actually written."""

    @abstractmethod
    async def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        """Delete by chunk id. Returns the count actually deleted."""

    @abstractmethod
    async def delete_by_source(self, source_prefix: str) -> int:
        """Delete every chunk whose ``source`` starts with ``source_prefix``.
        Returns the count actually deleted."""

    @abstractmethod
    async def list_chunks_for_source(
        self, source_uri: str,
    ) -> Sequence[EmbeddedChunk]:
        """Return every chunk whose ``source`` is *exactly* equal to
        ``source_uri``.

        Required by the ``Ingestor``'s idempotency policies (see
        :class:`~polymathera.colony.knowledge.models.IngestionPolicy`).
        Distinct from :meth:`delete_by_source`, which is prefix-based —
        idempotency requires exact identity, otherwise ingesting
        ``file:///x/paper.pdf`` would clobber ``file:///x/paper.pdf.bak``.
        """

    @abstractmethod
    async def update_tier_for_source(
        self, source_uri: str, tier: CorpusTier,
    ) -> int:
        """Update the persisted ``tier`` on every chunk whose
        ``source`` matches ``source_uri`` exactly. Returns the count
        actually updated. Used by ``IngestionPolicy.UPGRADE_TIER``.
        """

    @abstractmethod
    async def search(
        self,
        *,
        query_vector: Sequence[float],
        query: RetrievalQuery,
    ) -> Sequence[RetrievalHit]:
        """Return ranked hits for ``query_vector`` honouring the
        ``RetrievalQuery`` filters (source_prefix, data_types, tiers,
        max_results, require_citations)."""

    @abstractmethod
    async def get(self, chunk_id: str) -> EmbeddedChunk | None:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...


# ---------------------------------------------------------------------------
# In-memory store (production-ready for small corpora; canonical for tests)
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """Pure-Python vector store with cosine similarity ranking.

    Indexes by ``data_type`` + ``source`` so prefix filters short-
    circuit the scan. For corpora above a few tens of thousands of
    chunks, switch to ``QdrantVectorStore``.
    """

    def __init__(self) -> None:
        self._items: dict[str, EmbeddedChunk] = {}
        self._by_data_type: dict[str, set[str]] = {}
        self._by_source: dict[str, set[str]] = {}

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> int:
        n = 0
        for it in items:
            chunk_id = it.chunk.chunk_id
            self._remove(chunk_id)
            self._items[chunk_id] = it
            self._by_data_type.setdefault(it.chunk.data_type, set()).add(chunk_id)
            self._by_source.setdefault(it.chunk.source, set()).add(chunk_id)
            n += 1
        return n

    async def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        n = 0
        for cid in chunk_ids:
            if self._remove(cid):
                n += 1
        return n

    async def delete_by_source(self, source_prefix: str) -> int:
        ids = [
            cid for cid, item in self._items.items()
            if item.chunk.source.startswith(source_prefix)
        ]
        return await self.delete_by_chunk_ids(ids)

    async def search(
        self,
        *,
        query_vector: Sequence[float],
        query: RetrievalQuery,
    ) -> Sequence[RetrievalHit]:
        candidates = self._candidates_for(query)
        scored: list[tuple[float, EmbeddedChunk]] = []
        for cid in candidates:
            item = self._items[cid]
            if query.require_citations and not item.chunk.citation.source_uri:
                continue
            score = _cosine(query_vector, item.vector)
            scored.append((score, item))
        scored.sort(key=lambda t: (-t[0], t[1].chunk.chunk_id))
        hits: list[RetrievalHit] = []
        for rank, (score, item) in enumerate(scored[: query.max_results]):
            hits.append(
                RetrievalHit(
                    chunk=item.chunk,
                    score=max(0.0, min(1.0, (score + 1.0) / 2.0)),
                    rank=rank,
                    explanation=f"cosine_score={score:.4f}",
                )
            )
        return tuple(hits)

    async def get(self, chunk_id: str) -> EmbeddedChunk | None:
        return self._items.get(chunk_id)

    async def count(self) -> int:
        return len(self._items)

    async def list_chunks_for_source(
        self, source_uri: str,
    ) -> Sequence[EmbeddedChunk]:
        ids = self._by_source.get(source_uri, set())
        return tuple(self._items[cid] for cid in ids)

    async def update_tier_for_source(
        self, source_uri: str, tier: CorpusTier,
    ) -> int:
        ids = list(self._by_source.get(source_uri, set()))
        if not ids:
            return 0
        for cid in ids:
            existing = self._items[cid]
            updated_chunk = existing.chunk.model_copy(update={"tier": tier})
            self._items[cid] = existing.model_copy(
                update={"chunk": updated_chunk},
            )
        return len(ids)

    # ---- internals --------------------------------------------------

    def _remove(self, chunk_id: str) -> bool:
        existing = self._items.pop(chunk_id, None)
        if existing is None:
            return False
        bucket = self._by_data_type.get(existing.chunk.data_type)
        if bucket is not None:
            bucket.discard(chunk_id)
            if not bucket:
                self._by_data_type.pop(existing.chunk.data_type, None)
        bucket = self._by_source.get(existing.chunk.source)
        if bucket is not None:
            bucket.discard(chunk_id)
            if not bucket:
                self._by_source.pop(existing.chunk.source, None)
        return True

    def _candidates_for(self, query: RetrievalQuery) -> Iterable[str]:
        if query.data_types:
            ids: set[str] = set()
            for dt in query.data_types:
                ids.update(self._by_data_type.get(dt, ()))
        else:
            ids = set(self._items.keys())
        if query.source_prefix:
            ids = {
                cid for cid in ids
                if self._items[cid].chunk.source.startswith(query.source_prefix)
            }
        if query.tiers:
            allowed_tiers = set(query.tiers)
            ids = {cid for cid in ids if self._items[cid].chunk.tier in allowed_tiers}
        return ids


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


async def _query_points_compat(
    client: Any,
    *,
    collection_name: str,
    query: Sequence[float],
    query_filter: Any,
    limit: int,
) -> list[Any]:
    """Call ``query_points`` (modern qdrant-client) or fall back to
    the legacy ``search``. Returns the list of scored hits.

    Each hit exposes ``id`` / ``score`` / ``payload`` regardless of
    which path the client took, so caller code reads them uniformly.
    """

    if hasattr(client, "query_points"):
        response = await client.query_points(
            collection_name=collection_name,
            query=list(query),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        # ``query_points`` returns a ``QueryResponse`` with a ``points``
        # attribute. Older variants returned a list directly; tolerate
        # both shapes.
        return list(getattr(response, "points", response) or ())
    if hasattr(client, "search"):
        return list(
            await client.search(
                collection_name=collection_name,
                query_vector=list(query),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            or ()
        )
    raise VectorStoreError(
        "Qdrant client exposes neither query_points nor search; "
        "incompatible client version.",
    )


# ---------------------------------------------------------------------------
# Qdrant-backed store (Phase C1b — real implementation)
# ---------------------------------------------------------------------------


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store.

    Wraps an ``AsyncQdrantClient`` (or any object exposing the same
    async upsert / search / delete / scroll / get_collection /
    create_collection API) plus a collection name. The store is
    responsible for creating the collection on first use, marshalling
    chunks into Qdrant's point shape, and translating
    ``RetrievalQuery`` filters into Qdrant payload filters.

    Each Qdrant *point* carries:

    - ``id`` — a deterministic UUID5 derived from the ``chunk_id``
      (Qdrant requires UUID or int ids; the in-memory store keys by
      the free-form ``chunk_id`` directly).
    - ``vector`` — the embedding vector.
    - ``payload`` — every field downstream filters need
      (``chunk_id``, ``data_type``, ``source``, ``tier``,
      ``effective_at`` ISO, ``page_number``, ``token_count``, plus
      the chunk's full pydantic JSON dump for round-trip).

    The store imports ``qdrant_client`` lazily so the colony framework
    stays importable without it installed.
    """

    _NAMESPACE_UUID = uuid.UUID("4f7c1b96-9b2a-4f9d-b1e6-91c0e6a6c2d0")
    """A fixed namespace so the same ``chunk_id`` always maps to the
    same Qdrant point id across runs."""

    @classmethod
    async def connect(
        cls,
        *,
        url: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: str | None = None,
        collection: str,
        embedder_id: str,
        dimensions: int,
        distance: str = "Cosine",
        timeout_s: float = 10.0,
    ) -> "QdrantVectorStore":
        """Construct a store, connect to Qdrant, ensure the collection
        exists, and return the ready-to-use store.

        Either ``url`` or ``host``+``port`` must be supplied.
        """

        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VectorStoreError(
                "QdrantVectorStore requires the 'qdrant-client' package. "
                "Install via `pip install polymathera-colony[knowledge]`.",
            ) from exc

        if url is not None:
            client = AsyncQdrantClient(
                url=url, api_key=api_key, prefer_grpc=prefer_grpc,
                timeout=int(timeout_s),
            )
        else:
            client = AsyncQdrantClient(
                host=host, port=port, grpc_port=grpc_port,
                api_key=api_key, prefer_grpc=prefer_grpc,
                timeout=int(timeout_s),
            )
        store = cls(
            client=client,
            collection=collection,
            embedder_id=embedder_id,
            dimensions=dimensions,
            distance=distance,
        )
        await store._ensure_collection()
        return store

    def __init__(
        self,
        *,
        client: Any,
        collection: str,
        embedder_id: str,
        dimensions: int,
        distance: str = "Cosine",
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder_id = embedder_id
        self._dimensions = dimensions
        self._distance = distance
        self._collection_ready = False

    @property
    def collection_name(self) -> str:
        return self._collection

    # ---- VectorStore impl ---------------------------------------------

    async def upsert(self, items: Sequence[EmbeddedChunk]) -> int:
        if not items:
            return 0
        await self._ensure_collection()
        points = [self._to_point(item) for item in items]
        # qdrant_client async API:
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        return len(points)

    async def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        await self._ensure_collection()
        # Filter out chunks that aren't there to get an accurate count.
        existing = await self._count_by_chunk_ids(chunk_ids)
        if existing == 0:
            return 0
        ids = [str(self._point_id(cid)) for cid in chunk_ids]
        await self._client.delete(
            collection_name=self._collection,
            points_selector=ids,
            wait=True,
        )
        return existing

    async def delete_by_source(self, source_prefix: str) -> int:
        await self._ensure_collection()
        # Qdrant doesn't support prefix filters natively; we scroll to
        # collect ids, then issue a single delete-by-id batch.
        ids = await self._scroll_ids_for_source_prefix(source_prefix)
        if not ids:
            return 0
        await self._client.delete(
            collection_name=self._collection,
            points_selector=ids,
            wait=True,
        )
        return len(ids)

    async def search(
        self,
        *,
        query_vector: Sequence[float],
        query: RetrievalQuery,
    ) -> Sequence[RetrievalHit]:
        await self._ensure_collection()
        from qdrant_client import models as qmodels  # type: ignore[import-not-found]

        must: list[Any] = []
        if query.data_types:
            must.append(
                qmodels.FieldCondition(
                    key="data_type",
                    match=qmodels.MatchAny(any=list(query.data_types)),
                )
            )
        if query.tiers:
            must.append(
                qmodels.FieldCondition(
                    key="tier",
                    match=qmodels.MatchAny(any=[t.value for t in query.tiers]),
                )
            )
        # source_prefix isn't a native Qdrant op; we filter post-hoc
        # below. Effective-at windows are applied post-hoc as well —
        # the typical retrieval surface oversamples and the standards-
        # mode adapter does its own narrower second pass.
        flt = qmodels.Filter(must=must) if must else None

        # Oversample when we'll filter post-hoc by source_prefix.
        oversample = (
            query.max_results * 4 if query.source_prefix else query.max_results
        )
        # qdrant-client ≥ 1.10 deprecates ``search`` in favour of
        # ``query_points`` (which returns a ``QueryResponse`` whose
        # ``points`` attribute carries the scored hits).
        results = await _query_points_compat(
            self._client,
            collection_name=self._collection,
            query=list(query_vector),
            query_filter=flt,
            limit=oversample,
        )

        hits: list[RetrievalHit] = []
        for rank, p in enumerate(results):
            payload = dict(p.payload or {})
            if query.source_prefix and not str(payload.get("source", "")).startswith(
                query.source_prefix,
            ):
                continue
            chunk = self._chunk_from_payload(payload)
            if query.require_citations and not chunk.citation.source_uri:
                continue
            score = float(getattr(p, "score", 0.0))
            # Qdrant returns cosine *similarity* in [-1, 1] scaled.
            normalized = max(0.0, min(1.0, (score + 1.0) / 2.0))
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=normalized,
                    rank=len(hits),
                    explanation=f"qdrant_score={score:.4f}",
                ),
            )
            if len(hits) >= query.max_results:
                break
        return tuple(hits)

    async def get(self, chunk_id: str) -> EmbeddedChunk | None:
        await self._ensure_collection()
        result = await self._client.retrieve(
            collection_name=self._collection,
            ids=[str(self._point_id(chunk_id))],
            with_payload=True,
            with_vectors=True,
        )
        if not result:
            return None
        point = result[0]
        payload = dict(point.payload or {})
        chunk = self._chunk_from_payload(payload)
        vector = tuple(float(v) for v in (point.vector or ()))
        return EmbeddedChunk(
            chunk=chunk, vector=vector,
            embedder=self._embedder_id,
        )

    async def count(self) -> int:
        await self._ensure_collection()
        result = await self._client.count(
            collection_name=self._collection, exact=True,
        )
        return int(getattr(result, "count", result) or 0)

    # ---- Internals ----------------------------------------------------

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        from qdrant_client import models as qmodels  # type: ignore[import-not-found]
        from qdrant_client.http.exceptions import (  # type: ignore[import-not-found]
            UnexpectedResponse,
        )

        try:
            await self._client.get_collection(
                collection_name=self._collection,
            )
        except (UnexpectedResponse, ValueError):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._dimensions,
                    distance=getattr(qmodels.Distance, self._distance.upper(), qmodels.Distance.COSINE),
                ),
            )
        except Exception:  # noqa: BLE001
            # Some qdrant-client versions raise generic exceptions for
            # missing collections. Fall through to create.
            try:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=self._dimensions,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
            except Exception:  # noqa: BLE001
                # Already exists or another transient — proceed.
                pass
        self._collection_ready = True

    def _point_id(self, chunk_id: str) -> uuid.UUID:
        return uuid.uuid5(self._NAMESPACE_UUID, chunk_id)

    def _to_point(self, item: EmbeddedChunk) -> Any:
        from qdrant_client import models as qmodels  # type: ignore[import-not-found]

        chunk = item.chunk
        payload = {
            "chunk_id": chunk.chunk_id,
            "data_type": chunk.data_type,
            "source": chunk.source,
            "tier": chunk.tier.value,
            "section_path": chunk.section_path,
            "token_count": chunk.token_count,
            "language": chunk.language,
            "effective_at": (
                chunk.effective_at.isoformat()
                if chunk.effective_at is not None
                else None
            ),
            "page_number": chunk.citation.page_number,
            "chunk_json": chunk.model_dump_json(),
            "embedder": item.embedder,
        }
        if len(item.vector) != self._dimensions:
            raise VectorStoreError(
                f"Chunk {chunk.chunk_id}: vector dim {len(item.vector)} "
                f"does not match collection dim {self._dimensions}.",
            )
        return qmodels.PointStruct(
            id=str(self._point_id(chunk.chunk_id)),
            vector=list(item.vector),
            payload=payload,
        )

    @staticmethod
    def _chunk_from_payload(payload: Mapping[str, Any]) -> "Chunk":
        raw = payload.get("chunk_json")
        if isinstance(raw, str):
            try:
                return Chunk.model_validate_json(raw)
            except Exception:  # noqa: BLE001 - fall through to manual rebuild
                pass
        # Fallback: rebuild from individual fields. Sources that
        # bypassed _to_point won't have ``chunk_json``; we still
        # surface what we can.
        from ..models import CitationSpan, CorpusTier, Chunk as _Chunk

        return _Chunk(
            chunk_id=str(payload.get("chunk_id", "")),
            text="",
            token_count=int(payload.get("token_count", 0) or 0),
            section_path=str(payload.get("section_path", "")),
            citation=CitationSpan(
                source_uri=str(payload.get("source", "")),
                section_path=str(payload.get("section_path", "")),
                page_number=payload.get("page_number"),
            ),
            data_type=str(payload.get("data_type", "untyped")),
            source=str(payload.get("source", "")),
            tier=CorpusTier(str(payload.get("tier", "untiered"))),
            language=payload.get("language"),
        )

    async def _count_by_chunk_ids(
        self, chunk_ids: Sequence[str],
    ) -> int:
        if not chunk_ids:
            return 0
        ids = [str(self._point_id(cid)) for cid in chunk_ids]
        result = await self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
        return len(result or ())

    async def _scroll_ids_for_source_prefix(
        self, source_prefix: str,
    ) -> list[str]:
        """Walk the collection and collect point ids whose payload
        ``source`` starts with the prefix."""

        ids: list[str] = []
        offset: Any = None
        while True:
            chunk, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in chunk or ():
                payload = getattr(point, "payload", {}) or {}
                if str(payload.get("source", "")).startswith(source_prefix):
                    ids.append(str(point.id))
            if offset is None:
                break
        return ids

    async def list_chunks_for_source(
        self, source_uri: str,
    ) -> Sequence[EmbeddedChunk]:
        await self._ensure_collection()
        from qdrant_client import models as qmodels  # type: ignore[import-not-found]

        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source", match=qmodels.MatchValue(value=source_uri),
                ),
            ],
        )
        out: list[EmbeddedChunk] = []
        offset: Any = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points or ():
                payload = dict(getattr(point, "payload", {}) or {})
                chunk = self._chunk_from_payload(payload)
                vector = tuple(float(v) for v in (getattr(point, "vector", ()) or ()))
                out.append(
                    EmbeddedChunk(
                        chunk=chunk, vector=vector,
                        embedder=str(payload.get("embedder", self._embedder_id)),
                    ),
                )
            if offset is None:
                break
        return tuple(out)

    async def update_tier_for_source(
        self, source_uri: str, tier: CorpusTier,
    ) -> int:
        existing = await self.list_chunks_for_source(source_uri)
        if not existing:
            return 0
        # Round-trip: rebuild chunks with the new tier and re-upsert.
        # Cheaper than a per-point payload-set call when ``tier`` lives
        # both inside ``chunk_json`` and as a top-level payload field
        # (we have to rewrite both for retrieval filters to stay
        # consistent).
        updated = tuple(
            EmbeddedChunk(
                chunk=item.chunk.model_copy(update={"tier": tier}),
                vector=item.vector,
                embedder=item.embedder,
            )
            for item in existing
        )
        await self.upsert(updated)
        return len(updated)


__all__ = (
    "VectorStore",
    "VectorStoreError",
    "InMemoryVectorStore",
    "QdrantVectorStore",
)
