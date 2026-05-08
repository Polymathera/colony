"""Knowledge-Base endpoints — read-only inspection + ad-hoc ingestion.

The dashboard's KB tab calls these to surface the corpus the agents
share via the process-singleton ``RetrievalDeps`` from
``polymathera.colony.knowledge.deps``. Same backend (Qdrant when
``QDRANT_URL`` is set, in-memory otherwise) the agents see — the tab
is a window onto live state, not a separate cache.

All endpoints are ``Ring.USER`` and gated by ``require_auth``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth


logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class KBBackendInfo(BaseModel):
    """What the dashboard process is currently bound to."""

    vector_store: str
    """Concrete class name of the live VectorStore (``InMemoryVectorStore``
    / ``QdrantVectorStore``). Operators read this to confirm the
    Qdrant wiring took effect."""

    embedder_id: str
    embedder_dimensions: int
    qdrant_url: str | None = None
    qdrant_collection: str | None = None


class KBStatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    total_tokens: int
    by_tier: dict[str, int]
    by_data_type: dict[str, int]
    backend: KBBackendInfo


class KBSourceRow(BaseModel):
    source: str
    chunk_count: int
    total_tokens: int
    data_types: list[str]
    tiers: list[str]


class KBSourcesResponse(BaseModel):
    sources: list[KBSourceRow]


class KBChunkRow(BaseModel):
    chunk_id: str
    section_path: str
    data_type: str
    tier: str
    token_count: int
    page_number: int | None = None
    text_preview: str
    """First ~400 chars of the chunk text. Full text is available via
    the future ``GET /kb/chunks/{chunk_id}`` endpoint when needed."""


class KBChunksResponse(BaseModel):
    source: str
    chunks: list[KBChunkRow]


class KBSearchRequest(BaseModel):
    text: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=100)
    source_prefix: str | None = None
    data_types: list[str] = Field(default_factory=list)


class KBSearchHit(BaseModel):
    chunk_id: str
    score: float
    rank: int
    source: str
    section_path: str
    data_type: str
    tier: str
    text_preview: str


class KBSearchResponse(BaseModel):
    hits: list[KBSearchHit]


class KBIngestRequest(BaseModel):
    """Operator-driven ingestion. Either ``path`` or ``text`` is set.

    ``path`` is interpreted on the dashboard's filesystem (so it must
    live under a volume the dashboard container can read — typically
    ``/mnt/shared``). ``text`` skips the reader pipeline and stores the
    payload as plain text.
    """

    path: str | None = None
    text: str | None = None
    source_uri: str | None = None
    tier: str = "untiered"


class KBIngestResponse(BaseModel):
    record_id: str
    source_uri: str
    status: str
    chunks_produced: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend_info() -> KBBackendInfo:
    import os

    from polymathera.colony.knowledge.deps import get_knowledge_deps

    deps = get_knowledge_deps()
    return KBBackendInfo(
        vector_store=type(deps.vector_store).__name__,
        embedder_id=deps.embedder.embedder_id,
        embedder_dimensions=deps.embedder.dimensions,
        qdrant_url=os.environ.get("QDRANT_URL"),
        qdrant_collection=(
            os.environ.get("QDRANT_COLLECTION", "colony_knowledge")
            if os.environ.get("QDRANT_URL")
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/kb/stats", response_model=KBStatsResponse)
async def kb_stats(
    _user: dict = Depends(require_auth),
) -> KBStatsResponse:
    """Aggregate stats for the live corpus.

    Walks ``list_source_summaries()`` once and folds it into per-tier
    and per-data_type counts. Operator-driven; not on the agent
    retrieval path.
    """

    from polymathera.colony.knowledge.deps import get_knowledge_deps

    deps = get_knowledge_deps()
    summaries = await deps.vector_store.list_source_summaries()

    total_chunks = 0
    total_tokens = 0
    by_tier: dict[str, int] = {}
    by_data_type: dict[str, int] = {}
    for s in summaries:
        total_chunks += s.chunk_count
        total_tokens += s.total_tokens
        for tier in s.tiers:
            by_tier[tier.value] = by_tier.get(tier.value, 0) + s.chunk_count
        for dt in s.data_types:
            by_data_type[dt] = by_data_type.get(dt, 0) + s.chunk_count
    return KBStatsResponse(
        total_chunks=total_chunks,
        total_sources=len(summaries),
        total_tokens=total_tokens,
        by_tier=by_tier,
        by_data_type=by_data_type,
        backend=_backend_info(),
    )


@router.get("/kb/sources", response_model=KBSourcesResponse)
async def kb_sources(
    _user: dict = Depends(require_auth),
) -> KBSourcesResponse:
    """List every distinct source URI in the corpus."""

    from polymathera.colony.knowledge.deps import get_knowledge_deps

    deps = get_knowledge_deps()
    summaries = await deps.vector_store.list_source_summaries()
    return KBSourcesResponse(
        sources=[
            KBSourceRow(
                source=s.source,
                chunk_count=s.chunk_count,
                total_tokens=s.total_tokens,
                data_types=list(s.data_types),
                tiers=[t.value for t in s.tiers],
            )
            for s in summaries
        ],
    )


@router.get("/kb/sources/chunks", response_model=KBChunksResponse)
async def kb_chunks_for_source(
    source_uri: str = Query(min_length=1),
    limit: int = Query(default=200, ge=1, le=1000),
    _user: dict = Depends(require_auth),
) -> KBChunksResponse:
    """List chunks for one source, oldest-first by section path.

    ``source_uri`` is taken as a query param so URL-encoded ``file:///``
    paths come through cleanly without path-segment surprises.
    """

    from polymathera.colony.knowledge.deps import get_knowledge_deps

    deps = get_knowledge_deps()
    chunks = await deps.vector_store.list_chunks_for_source(source_uri)
    rows = [
        KBChunkRow(
            chunk_id=c.chunk.chunk_id,
            section_path=c.chunk.section_path,
            data_type=c.chunk.data_type,
            tier=c.chunk.tier.value,
            token_count=c.chunk.token_count,
            page_number=c.chunk.citation.page_number,
            text_preview=c.chunk.text[:400],
        )
        for c in chunks[:limit]
    ]
    rows.sort(key=lambda r: (r.section_path, r.chunk_id))
    return KBChunksResponse(source=source_uri, chunks=rows)


@router.post("/kb/search", response_model=KBSearchResponse)
async def kb_search(
    payload: KBSearchRequest,
    _user: dict = Depends(require_auth),
) -> KBSearchResponse:
    """Embedding-similarity search across the corpus.

    Embeds ``payload.text`` with the bound embedder, runs a vector
    search with the supplied filters, and returns ranked previews.
    """

    from polymathera.colony.knowledge.deps import get_knowledge_deps
    from polymathera.colony.knowledge.models import RetrievalQuery

    deps = get_knowledge_deps()
    vectors = await deps.embedder.embed([payload.text])
    if not vectors:
        return KBSearchResponse(hits=[])
    query = RetrievalQuery(
        text=payload.text,
        max_results=payload.max_results,
        source_prefix=payload.source_prefix,
        data_types=tuple(payload.data_types),
    )
    hits = await deps.vector_store.search(
        query_vector=vectors[0], query=query,
    )
    return KBSearchResponse(
        hits=[
            KBSearchHit(
                chunk_id=h.chunk.chunk_id,
                score=h.score,
                rank=h.rank,
                source=h.chunk.source,
                section_path=h.chunk.section_path,
                data_type=h.chunk.data_type,
                tier=h.chunk.tier.value,
                text_preview=h.chunk.text[:400],
            )
            for h in hits
        ],
    )


@router.post("/kb/ingest", response_model=KBIngestResponse)
async def kb_ingest(
    payload: KBIngestRequest,
    _user: dict = Depends(require_auth),
) -> KBIngestResponse:
    """Ad-hoc ingestion of a file or a text blob.

    Provided for operator-driven smoke tests of the ingestion pipeline
    from the KB tab; the routine ingestion path remains the
    ``materialize_knowledge_routing`` action driven by ``repo_map.yaml``.
    """

    from polymathera.colony.knowledge.deps import get_default_ingestor
    from polymathera.colony.knowledge.models import CorpusTier

    if not payload.path and not payload.text:
        raise HTTPException(
            status_code=400,
            detail="kb_ingest requires either ``path`` or ``text``.",
        )
    if payload.path and payload.text:
        raise HTTPException(
            status_code=400,
            detail="kb_ingest accepts ``path`` xor ``text``, not both.",
        )

    try:
        tier = CorpusTier(payload.tier)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown tier: {payload.tier}",
        ) from exc

    ingestor = get_default_ingestor()
    if payload.path:
        path_obj = Path(payload.path)
        if not path_obj.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"File not found on dashboard host: {path_obj}",
            )
        record = await ingestor.ingest_file(
            path_obj, tier=tier, source_uri=payload.source_uri,
        )
    else:
        record = await ingestor.ingest_text(
            payload.text or "", tier=tier, source_uri=payload.source_uri,
        )

    return KBIngestResponse(
        record_id=record.record_id,
        source_uri=record.source_uri,
        status=record.status.value,
        chunks_produced=record.chunks_produced,
        error=record.error,
    )
