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

import asyncio
import contextlib
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony

if TYPE_CHECKING:
    from ..services.colony_connection import ColonyConnection


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
    redeploying or editing ``knowledge.pdf_extractor``
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


def _snapshot_execution_context() -> dict[str, Any]:
    """Capture the live request's execution-context fields so a
    BackgroundTask can re-establish them. Starlette runs background
    tasks AFTER the response — the auth middleware's
    ``execution_context`` block has already exited by then, so any
    task touching context-requiring machinery (InferenceRequest,
    deployment handles) must re-enter the context itself
    (2026-08-05: the vocab revision pass would otherwise crash at its
    first judge call; the bulk-ingest and rehydrate tasks carried the
    same latent gap)."""

    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_execution_context,
    )

    ctx = get_execution_context()
    if ctx is None:
        return {}
    return {
        "colony_id": ctx.colony_id,
        "tenant_id": ctx.tenant_id,
        "session_id": ctx.session_id,
        "origin": ctx.origin or "dashboard",
    }


@contextlib.contextmanager
def _reenter_execution_context(snapshot: dict[str, Any]):
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    with execution_context(ring=Ring.USER, **snapshot):
        yield


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
    background_tasks.add_task(
        _run_ingest_repo_map, op_id, request,
        _snapshot_execution_context(),
    )
    return IngestRepoMapOpStatus(**op)


@router.get(
    "/kb/ingest-repo-map/operations",
    response_model=list[IngestRepoMapOpStatus],
)
async def kb_ingest_repo_map_operations(
    _user: dict = Depends(require_auth),
) -> list[IngestRepoMapOpStatus]:
    return [IngestRepoMapOpStatus(**op) for op in _ingest_ops.values()]


class RehydrateRequest(BaseModel):
    """Rehydrate one branch's (or every active branch's) KG snapshot
    from the design monorepo into the shared Kùzu store."""

    origin_url: str = Field(description="Git repo URL (https:// or file://)")
    branch: str = Field(
        default="main",
        description=(
            "Branch to rehydrate from origin. Pass the literal "
            '"__all__" to iterate every remote branch.'
        ),
    )


class RehydrateOpStatus(BaseModel):
    op_id: str
    status: str = Field(description="pending | running | completed | error")
    origin_url: str
    branch: str
    started_at: float
    completed_at: float | None = None
    message: str = ""
    branches_rehydrated: int = 0
    claims_in_file: int = 0
    claims_newly_added: int = 0
    claims_newly_tagged: int = 0
    claims_already_present: int = 0


_rehydrate_ops: dict[str, dict[str, Any]] = {}


@router.post("/kb/rehydrate", response_model=RehydrateOpStatus)
async def kb_rehydrate(
    request: RehydrateRequest,
    background_tasks: BackgroundTasks,
    _user: dict = Depends(require_auth),
) -> RehydrateOpStatus:
    """Rehydrate the shared KG from a design-monorepo snapshot.
    Returns immediately; poll ``GET /kb/rehydrate/operations`` for
    progress."""

    op_id = f"rehydrate_{uuid.uuid4().hex[:12]}"
    op: dict[str, Any] = {
        "op_id": op_id,
        "status": "pending",
        "origin_url": request.origin_url,
        "branch": request.branch,
        "started_at": time.time(),
        "completed_at": None,
        "message": "",
        "branches_rehydrated": 0,
        "claims_in_file": 0,
        "claims_newly_added": 0,
        "claims_newly_tagged": 0,
        "claims_already_present": 0,
    }
    _rehydrate_ops[op_id] = op
    background_tasks.add_task(
        _run_rehydrate, op_id, request,
        _snapshot_execution_context(),
    )
    return RehydrateOpStatus(**op)


@router.get(
    "/kb/rehydrate/operations",
    response_model=list[RehydrateOpStatus],
)
async def kb_rehydrate_operations(
    _user: dict = Depends(require_auth),
) -> list[RehydrateOpStatus]:
    return [RehydrateOpStatus(**op) for op in _rehydrate_ops.values()]


async def _run_rehydrate(
    op_id: str, request: RehydrateRequest,
    ctx_snapshot: dict[str, Any],
) -> None:
    op = _rehydrate_ops.get(op_id)
    if not op:
        return
    op["status"] = "running"
    try:
        from git import Repo

        from polymathera.colony.distributed import get_polymathera
        from polymathera.colony.knowledge.persistence import (
            list_remote_branches, rehydrate_branch_from_repo,
        )

        polymathera = get_polymathera()
        storage = await polymathera.get_storage()
        async with _repo_git_lock(request.origin_url):
            repo_path = await storage.git_storage.clone_or_retrieve_repository(
                origin_url=request.origin_url,
                branch="main" if request.branch == "__all__" else request.branch,
                commit="HEAD",
            )
        repo = Repo(str(repo_path))

        branch_names = []

        if request.branch == "__all__":
            op["message"] = "Discovering branches..."
            branch_names = await list_remote_branches(repo)
        else:
            op["message"] = f"Rehydrating {request.branch}..."
            branch_names = [request.branch]

        for branch_name in branch_names:
            op["message"] = f"Rehydrating {branch_name}..."
            with _reenter_execution_context(ctx_snapshot):
                result = await rehydrate_branch_from_repo(repo, branch_name)
            op["branches_rehydrated"] += 1
            for k in (
                "claims_in_file", "claims_newly_added",
                "claims_newly_tagged", "claims_already_present",
            ):
                op[k] += int(result.get(k, 0))

        op["message"] = (
            f"Rehydrated {op['branches_rehydrated']} branch(es); "
            f"{op['claims_newly_added']} new + "
            f"{op['claims_newly_tagged']} retagged"
        )
        op["status"] = "completed"
    except Exception as e:  # noqa: BLE001
        logger.exception("kb_rehydrate op %s failed", op_id)
        op["status"] = "error"
        op["message"] = str(e)
    op["completed_at"] = time.time()


async def _run_ingest_repo_map(
    op_id: str, request: IngestRepoMapRequest,
    ctx_snapshot: dict[str, Any],
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

        polymathera = get_polymathera()
        storage = await polymathera.get_storage()
        async with _repo_git_lock(request.origin_url):
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
        with _reenter_execution_context(ctx_snapshot):
            report = await materialize_knowledge_sources(
                repo_map=repo_map,
                repo_root=repo_root,
                enabled_sources=enabled,
            )
        # NOTE: the dashboard direct path doesn't commit — its clone
        # is a colony-level cache, not an agent's working tree. The
        # KB index is populated (the entire reason for this button)
        # but the acquired files + sidecars stay uncommitted on this
        # shared clone. The agent's ``ingest_repo_map_literature``
        # action does commit, since it operates on the per-agent
        # clone with an identity. Aligning these is a follow-up.
        op["status"] = "completed"
        op["ingested"] = report.ingested_count
        op["failed"] = report.failed_count
        op["message"] = (
            f"{report.ingested_count} ingested, "
            f"{report.failed_count} failed"
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("kb_ingest_repo_map op %s failed", op_id)
        op["status"] = "error"
        op["message"] = str(e)
    op["completed_at"] = time.time()


# ---------------------------------------------------------------------------
# /kb/vocab — predicate-vocabulary registry + revision passes
# (colony/predicate_vocabulary_plan.md; UI surface in KnowledgeBaseTab)
# ---------------------------------------------------------------------------


class VocabStatsResponse(BaseModel):
    """Decision-support payload for the KB tab's Vocabulary panel —
    everything the operator needs to decide whether a revision pass
    is worth its cost."""

    stats: dict[str, Any]
    estimated_clusters: int
    """How many candidate clusters a pass would judge (≈ one LLM call
    each) — the pass's cost, surfaced BEFORE the operator commits."""


class VocabProposeRequest(BaseModel):
    origin_url: str = Field(description="Git repo URL (https:// or file://)")
    branch: str = "main"
    max_clusters: int | None = Field(
        default=200,
        description="Judge at most this many clusters (cost cap).",
    )
    effort: str | None = Field(
        default="low",
        description=(
            "Effort level for judge LLM calls (low | medium | high | "
            "xhigh | max; None = provider default). Judging synonym "
            "clusters is batch classification — the provider's "
            "canonical low-effort workload; raise it if merge "
            "decisions look shallow."
        ),
    )


class VocabOpPayload(BaseModel):
    """One operation as reviewed in the UI."""

    op_id: str
    op_type: str
    term: str
    target: str | None = None
    rationale: str = ""
    confidence: float = 1.0
    proposed_by: str = ""


class VocabApplyRequest(BaseModel):
    origin_url: str
    branch: str = "main"
    operations: list[VocabOpPayload]
    approved_by: str = Field(
        description=(
            "Operator identity signing the destructive operations — "
            "apply refuses destructive ops without it."
        ),
    )


class VocabOpStatus(BaseModel):
    op_id: str
    status: str
    origin_url: str
    started_at: float
    completed_at: float | None = None
    message: str = ""
    proposals: list[dict[str, Any]] = Field(default_factory=list)


_vocab_ops: dict[str, dict[str, Any]] = {}


# Git operations on the SHARED cache clone (one working tree per
# origin, used by stats/propose/apply/ingest/rehydrate AND polled by
# the UI) must be serialized within this process: concurrent index
# mutations collide on .git/index.lock (2026-08-05: the vocab apply's
# commit raced the stats endpoint's clone_or_retrieve and 500'd with
# FileExistsError on index.lock, stranding applied-but-uncommitted
# state). Critical sections are kept SHORT (clone resolution, file
# reads, the commit block) — long phases (materialize, judging) run
# outside the lock. Cross-process collisions remain possible and stay
# loud; this removes the observed intra-process race.
_repo_git_locks: dict[str, asyncio.Lock] = {}


def _repo_git_lock(origin_url: str) -> asyncio.Lock:
    lock = _repo_git_locks.get(origin_url)
    if lock is None:
        lock = asyncio.Lock()
        _repo_git_locks[origin_url] = lock
    return lock


async def _load_vocab_and_kg(origin_url: str, branch: str):
    from polymathera.colony.distributed import get_polymathera
    from polymathera.colony.knowledge.persistence import (
        KG_FILE_RELATIVE_PATH, KgFile,
    )
    from polymathera.colony.knowledge.vocabulary import (
        VOCAB_FILE_RELATIVE_PATH, VocabFile,
    )

    polymathera = get_polymathera()
    storage = await polymathera.get_storage()
    async with _repo_git_lock(origin_url):
        repo_path = Path(str(await storage.git_storage.clone_or_retrieve_repository(
            origin_url=origin_url, branch=branch, commit="HEAD",
        )))
        kg_path = repo_path / KG_FILE_RELATIVE_PATH
        vocab_path = repo_path / VOCAB_FILE_RELATIVE_PATH
        kg = (
            KgFile.from_json(kg_path.read_text(encoding="utf-8"))
            if kg_path.is_file() else KgFile()
        )
        vocab = (
            VocabFile.from_json(vocab_path.read_text(encoding="utf-8"))
            if vocab_path.is_file() else VocabFile()
        )
    return repo_path, vocab, kg


@router.get("/kb/vocab/stats", response_model=VocabStatsResponse)
async def kb_vocab_stats(
    origin_url: str,
    branch: str = "main",
    _user: dict = Depends(require_auth),
) -> VocabStatsResponse:
    from collections import Counter

    from polymathera.colony.knowledge.vocabulary import (
        register_provisional, vocab_stats,
    )
    from polymathera.colony.knowledge.vocabulary_revision import (
        MIN_CLUSTER_USAGE, dedupe_clusters, lexical_clusters,
        type_signature_clusters,
    )

    _, vocab, kg = await _load_vocab_and_kg(origin_url, branch)
    usage = Counter(c.predicate for c in kg.claims)
    # Stats reflect the registry AS IF current predicates were
    # registered (pre-vocabulary KGs show up fully, not as zeros).
    register_provisional(vocab, usage.keys())
    stats = vocab_stats(vocab, usage)
    clusters = dedupe_clusters(
        lexical_clusters(usage) + type_signature_clusters(kg),
    )
    estimated = sum(
        1 for c in clusters
        if sum(usage.get(m, 0) for m in c.members) >= MIN_CLUSTER_USAGE
    )
    return VocabStatsResponse(
        stats=stats.model_dump(), estimated_clusters=estimated,
    )


@router.post("/kb/vocab/propose", response_model=VocabOpStatus)
async def kb_vocab_propose(
    request: VocabProposeRequest,
    background_tasks: BackgroundTasks,
    colony: "ColonyConnection" = Depends(get_colony),
    _user: dict = Depends(require_auth),
) -> VocabOpStatus:
    """Run a revision pass (candidate generation + LLM judging) in the
    background. Proposes only — nothing is applied until the operator
    approves through ``/kb/vocab/apply``."""

    op_id = f"vocab_{uuid.uuid4().hex[:12]}"
    op: dict[str, Any] = {
        "op_id": op_id, "status": "pending",
        "origin_url": request.origin_url, "started_at": time.time(),
        "completed_at": None, "message": "", "proposals": [],
    }
    _vocab_ops[op_id] = op
    # The dashboard is NOT a deployment: handle resolution cannot read
    # POLYMATHERA_SERVING_CURRENT_APP. The connection's app_name (from
    # the operator YAML) is the authority here.
    background_tasks.add_task(
        _run_vocab_propose, op_id, request, colony.app_name,
        _snapshot_execution_context(),
    )
    return VocabOpStatus(**op)


@router.get("/kb/vocab/propose/operations", response_model=list[VocabOpStatus])
async def kb_vocab_propose_operations(
    _user: dict = Depends(require_auth),
) -> list[VocabOpStatus]:
    return [VocabOpStatus(**op) for op in _vocab_ops.values()]


async def _run_vocab_propose(
    op_id: str, request: VocabProposeRequest, app_name: str,
    ctx_snapshot: dict[str, Any],
) -> None:
    op = _vocab_ops.get(op_id)
    if not op:
        return
    op["status"] = "running"
    op["message"] = "Loading registry + KG..."
    try:
        from collections import Counter

        from polymathera.colony.knowledge.deps import (
            build_default_llm_callable, get_knowledge_deps,
        )
        from polymathera.colony.knowledge.vocabulary import (
            register_provisional,
        )
        from polymathera.colony.knowledge.vocabulary_revision import (
            propose_operations,
        )

        _, vocab, kg = await _load_vocab_and_kg(
            request.origin_url, request.branch,
        )
        usage = Counter(c.predicate for c in kg.claims)
        register_provisional(vocab, usage.keys())
        llm = build_default_llm_callable(
            max_tokens=1024, temperature=0.0, app_name=app_name,
            effort=request.effort,
        )

        def _progress(message: str) -> None:
            op["message"] = message

        with _reenter_execution_context(ctx_snapshot):
            proposals = await propose_operations(
                vocab, kg,
                llm,
                embedder=get_knowledge_deps().embedder,
                max_clusters=request.max_clusters,
                on_progress=_progress,
            )
        op["proposals"] = [p.model_dump(mode="json") for p in proposals]
        op["message"] = f"{len(proposals)} operations proposed"
        op["status"] = "completed"
    except Exception as e:  # noqa: BLE001
        logger.exception("kb_vocab_propose op %s failed", op_id)
        op["status"] = "error"
        op["message"] = str(e)
    op["completed_at"] = time.time()


@router.post("/kb/vocab/apply")
async def kb_vocab_apply(
    request: VocabApplyRequest,
    _user: dict = Depends(require_auth),
) -> dict[str, Any]:
    """Apply operator-approved operations: registry update + claim
    rewrite for merges, committed and pushed so the vocabulary change
    is canonical, not a cache-clone artifact."""

    import asyncio as _asyncio

    from git import Repo

    from polymathera.colony.knowledge.persistence import (
        rewrite_claims_for_merges,
    )
    from polymathera.colony.knowledge.vocabulary import (
        DESTRUCTIVE_OP_TYPES, VOCAB_FILE_RELATIVE_PATH, VocabError,
        VocabFile, VocabOperation, apply_operation, register_provisional,
    )

    repo_path, vocab, kg = await _load_vocab_and_kg(
        request.origin_url, request.branch,
    )
    register_provisional(
        vocab, {c.predicate for c in kg.claims},
    )
    applied, failed = [], []
    for payload in request.operations:
        op = VocabOperation(
            **payload.model_dump(),
        )
        if op.op_type in DESTRUCTIVE_OP_TYPES:
            op.approved_by = request.approved_by
        try:
            apply_operation(vocab, op)
            applied.append(op.op_id)
        except VocabError as exc:
            failed.append({"op_id": op.op_id, "error": str(exc)})
    if not applied:
        return {"applied": [], "failed": failed, "rewrite": {}}

    def _commit_and_push() -> str:
        from git import Actor

        from polymathera.colony.distributed.ray_utils import serving

        repo = Repo(str(repo_path))
        repo.index.add(
            [".colony/colony.vocab.json", ".colony/colony.kg.json"],
        )
        colony_id = serving.get_colony_id() or "colony"
        # Same synthetic principal the design-monorepo capabilities
        # commit under; the approving operator is named in the message.
        actor = Actor(
            f"colony:{colony_id}", f"{colony_id}@agent.colony.local",
        )
        commit = repo.index.commit(
            f"vocab: apply {len(applied)} operations "
            f"(approved by {request.approved_by})",
            author=actor, committer=actor,
        )
        repo.remote().push().raise_if_error()
        return commit.hexsha[:8]

    # Write + rewrite + commit under the repo's git lock — the commit
    # mutates .git/index on the SHARED cache clone and must not race
    # the other kb endpoints' git operations.
    async with _repo_git_lock(request.origin_url):
        vocab_path = repo_path / VOCAB_FILE_RELATIVE_PATH
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(vocab.to_json(), encoding="utf-8")
        rewrite = await rewrite_claims_for_merges(repo_path)
        sha = await _asyncio.to_thread(_commit_and_push)

    return {
        "applied": applied, "failed": failed,
        "rewrite": rewrite, "commit": sha,
    }
