"""Knowledge-Base endpoints — read-only inspection + ad-hoc ingestion.

The dashboard's KB tab calls these to surface the corpus the agents
share via the process-singleton ``RetrievalDeps`` from
``polymathera.colony.knowledge.deps``. Same backend (Qdrant when
``knowledge.qdrant.url`` is set in the operator YAML, in-memory
otherwise) the agents see — the tab is a window onto live state, not
a separate cache.

All endpoints are ``Ring.USER`` and gated by ``require_auth``.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
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
    """Chunk text (capped at 16 KB). For multimodal chunks this is
    Markdown — the dashboard's KB tab renders it with figure URI
    rewriting so embedded ``colony-image://`` references resolve via
    ``GET /kb/images/<sha>``."""

    figure_ids: list[str] = Field(default_factory=list)
    """IDs of figures the chunk references, copied from
    ``Chunk.extra["figure_ids"]``. Lets the dashboard show a "N
    figures" badge per chunk and the agent's planner pull image URIs
    without re-parsing the chunk text."""

    metadata_origin: str | None = None
    """Provenance hint copied from ``Chunk.extra["metadata_origin"]``
    so the KB tab can label which extractor produced a given chunk
    (``mistral_ocr`` / ``anthropic`` / ``marker`` / …)."""


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
    extractor_override: str | None = None
    """Force a specific PDF extractor for this single ingest call —
    one of ``mistral_ocr`` / ``anthropic`` / ``marker`` / ``docling``
    / ``mineru``. Useful for A/B tests from the KB tab without
    redeploying or editing ``polymathera_cluster.knowledge.pdf_extractor``
    in the operator YAML. ``None`` (the default) uses the colony's
    configured extractor. Ignored for non-PDF ingests."""


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
    from polymathera.colony.distributed.config import get_component_or_default
    from polymathera.colony.knowledge.cluster_config import KnowledgeConfig
    from polymathera.colony.knowledge.deps import get_knowledge_deps

    deps = get_knowledge_deps()
    qdrant_cfg = get_component_or_default("knowledge", KnowledgeConfig).qdrant
    return KBBackendInfo(
        vector_store=type(deps.vector_store).__name__,
        embedder_id=deps.embedder.embedder_id,
        embedder_dimensions=deps.embedder.dimensions,
        qdrant_url=qdrant_cfg.url or None,
        qdrant_collection=qdrant_cfg.collection if qdrant_cfg.url else None,
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

    ``text_preview`` carries the full chunk text (capped at 16 KB)
    rather than the previous 400-char preview so a markdown-format
    chunk (Mistral / Anthropic / Marker / …) is rendered intact in
    the KB tab — partial markdown breaks figure references and
    table layouts. Operators can still navigate the chunk via the
    chat UI's ``CollapsiblePre`` on the client side.
    """

    from polymathera.colony.knowledge.deps import get_knowledge_deps

    _MAX_CHUNK_TEXT = 16_384

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
            text_preview=c.chunk.text[:_MAX_CHUNK_TEXT],
            figure_ids=list(c.chunk.extra.get("figure_ids") or ()),
            metadata_origin=c.chunk.extra.get("metadata_origin"),
        )
        for c in chunks[:limit]
    ]
    rows.sort(key=lambda r: (r.section_path, r.chunk_id))
    return KBChunksResponse(source=source_uri, chunks=rows)


@router.get("/kb/images/{sha}")
async def kb_image_resolve(
    sha: str,
    _user: dict = Depends(require_auth),
) -> Response:
    """Serve raw figure bytes from the active :class:`ImageStore`.

    The chunk text emitted by the multimodal readers carries
    ``colony-image://<sha>`` URIs; the KB tab's markdown renderer
    rewrites those to ``/api/v1/kb/images/<sha>`` so a browser ``<img>``
    tag resolves them via this endpoint. The mime is read from the
    store's sidecar so the right ``Content-Type`` flows back without
    sniffing magic bytes here.

    Returns 404 when the URI is not present (operator-deleted figure,
    fresh worker that never ran ingest, …) — the dashboard renders
    a placeholder rather than crashing the chat panel.
    """

    # Sha sanity: the store's URI scheme uses hex; reject anything
    # else so a bogus path can't traverse out of the shard tree.
    if not sha or not all(ch in "0123456789abcdef" for ch in sha.lower()):
        raise HTTPException(
            status_code=400, detail="invalid image sha (expected hex)",
        )

    from polymathera.colony.knowledge.deps import get_knowledge_deps
    from polymathera.colony.knowledge.stores.image import _build_uri

    image_store = get_knowledge_deps().image_store
    if image_store is None:
        raise HTTPException(
            status_code=503, detail="no image store configured on this colony",
        )
    uri = _build_uri(sha.lower())
    payload = await image_store.get(uri)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"image not found: {sha}")
    info = await image_store.stat(uri)
    media_type = (info or {}).get("mime") or "application/octet-stream"
    # Cache aggressively — content-addressed bytes never change for
    # a given sha. ``immutable`` tells the browser not to revalidate.
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


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
    from the KB tab; the routine bulk-ingest path is the
    ``/kb/ingest-repo-map`` endpoint (Design Monorepo tab) and the
    SessionAgent's ``ingest_repo_map_literature`` action.
    """

    from polymathera.colony.knowledge.deps import (
        get_default_ingestor, get_knowledge_deps,
    )
    from polymathera.colony.knowledge.models import CorpusTier
    from polymathera.colony.knowledge.readers import (
        default_registry_with_pdf_extractor,
    )

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

    # If the operator overrode the extractor for this single call,
    # build a one-shot Ingestor that shares the singleton's
    # embedder + vector store + image store but swaps the reader
    # registry. This avoids touching the process-wide ingestor
    # (so the override doesn't leak into concurrent ingests) while
    # still landing chunks in the same Qdrant collection. Ignored
    # for ``text`` ingests since those don't go through a PDF
    # reader.
    if payload.extractor_override and payload.path:
        from polymathera.colony.knowledge.ingestion import Ingestor

        try:
            override_registry = default_registry_with_pdf_extractor(
                backend=payload.extractor_override,
                image_store=get_knowledge_deps().image_store,
            )
        except (NotImplementedError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"extractor_override={payload.extractor_override!r} "
                       f"not available: {exc}",
            ) from exc
        deps = get_knowledge_deps()
        ingestor = Ingestor(
            readers=override_registry,
            embedder=deps.embedder,
            vector_store=deps.vector_store,
            graph_store=deps.graph_store,
            image_store=deps.image_store,
        )

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


# ---------------------------------------------------------------------------
# /kb/ingest-repo-map — bulk ingest from ``knowledge_sources:`` rows
# ---------------------------------------------------------------------------
#
# Mirrors the VCM ``/vcm/map`` endpoint shape: a fire-and-forget POST
# with operation tracking via a polled GET. The two endpoints are
# orthogonal — the operator picks ``vcm_sources`` row names for
# ``/vcm/map`` and ``knowledge_sources`` row names for this one.


class IngestRepoMapRequest(BaseModel):
    """Bulk-ingest the literature declared in a design monorepo's
    ``.colony/repo_map.yaml`` ``knowledge_sources:`` block.

    The operator's per-row selection lives in the colony's persisted
    source-selection state (see ``design_monorepo.source_selection``);
    the dashboard writes it on every checkbox toggle, the materialiser
    reads it inside ``_run_ingest_repo_map``. No request-body filter —
    single source of truth.
    """

    origin_url: str = Field(description="Git repo URL (https:// or file://)")
    branch: str = Field(default="main")
    commit: str = Field(default="HEAD")


class IngestRepoMapOpStatus(BaseModel):
    op_id: str
    status: str = Field(description="pending | running | completed | error")
    origin_url: str
    started_at: float
    completed_at: float | None = None
    message: str = ""
    ingested: int = 0
    failed: int = 0


# In-memory op log — same pattern as ``vcm.py:_mapping_ops``. Survives
# the lifetime of the dashboard process; not persisted.
_ingest_ops: dict[str, dict[str, Any]] = {}


@router.post("/kb/ingest-repo-map", response_model=IngestRepoMapOpStatus)
async def kb_ingest_repo_map(
    request: IngestRepoMapRequest,
    background_tasks: BackgroundTasks,
    _user: dict = Depends(require_auth),
) -> IngestRepoMapOpStatus:
    """Start bulk KB ingestion from a design monorepo's
    ``knowledge_sources:`` block. Returns immediately. Poll GET
    ``/kb/ingest-repo-map/operations`` for progress."""

    op_id = f"ingest_{uuid.uuid4().hex[:12]}"
    op = {
        "op_id": op_id,
        "status": "pending",
        "origin_url": request.origin_url,
        "started_at": time.time(),
        "completed_at": None,
        "message": "",
        "ingested": 0,
        "failed": 0,
    }
    _ingest_ops[op_id] = op
    background_tasks.add_task(_run_ingest_repo_map, op_id, request)
    return IngestRepoMapOpStatus(**op)


@router.get(
    "/kb/ingest-repo-map/operations",
    response_model=list[IngestRepoMapOpStatus],
)
async def kb_ingest_repo_map_operations(
    _user: dict = Depends(require_auth),
) -> list[IngestRepoMapOpStatus]:
    return [IngestRepoMapOpStatus(**op) for op in _ingest_ops.values()]


async def _run_ingest_repo_map(
    op_id: str, request: IngestRepoMapRequest,
) -> None:
    op = _ingest_ops.get(op_id)
    if not op:
        return
    op["status"] = "running"
    op["message"] = f"Cloning {request.origin_url}..."
    try:
        from polymathera.colony.design_monorepo.materialize import (
            materialize_knowledge_sources,
        )
        from polymathera.colony.design_monorepo.repo_map import RepoMap
        from polymathera.colony.design_monorepo.source_selection import (
            list_enabled_knowledge_sources,
        )
        from polymathera.colony.distributed import get_polymathera
        from polymathera.colony.distributed.ray_utils import serving
        from polymathera.colony.knowledge.models import IngestionStatus

        polymathera = get_polymathera()
        storage = await polymathera.get_storage()
        repo_path = await storage.git_storage.clone_or_retrieve_repository(
            origin_url=request.origin_url,
            branch=request.branch,
            commit=request.commit,
        )
        repo_root = Path(str(repo_path))
        repo_map = RepoMap.load(repo_root)

        colony_id = serving.get_colony_id() or ""
        enabled_list = await list_enabled_knowledge_sources(colony_id)
        enabled = set(enabled_list) if enabled_list is not None else None
        op["message"] = "Ingesting matching files..."
        records = await materialize_knowledge_sources(
            repo_map=repo_map,
            repo_root=repo_root,
            enabled_sources=enabled,
        )
        ingested = sum(
            1 for r in records if r.status == IngestionStatus.COMPLETED
        )
        failed = sum(
            1 for r in records if r.status == IngestionStatus.FAILED
        )
        op["status"] = "completed"
        op["ingested"] = ingested
        op["failed"] = failed
        op["message"] = f"{ingested} ingested, {failed} failed"
    except Exception as e:  # noqa: BLE001
        logger.exception("kb_ingest_repo_map op %s failed", op_id)
        op["status"] = "error"
        op["message"] = str(e)
    op["completed_at"] = time.time()
