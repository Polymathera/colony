"""Process-singleton factory for the knowledge trio's shared backends.

The :class:`Ingestor`, :class:`KnowledgeCuratorCapability`, and
:class:`KnowledgeRetrievalCapability` all need the same triple of
``embedder + vector_store + (optional) graph_store`` — bound the same
way so curation and retrieval share an embedding space and a
collection. This module owns that triple as a per-process singleton so
the dashboard / SessionAgent can wire all three capabilities without
each having to construct deps independently.

Out-of-the-box defaults are :class:`InMemoryEmbedder` and
:class:`InMemoryVectorStore` — usable for development, single-node
deployments, and chat-driven curation that does not need persistence
across processes. Production operators set
``knowledge`` in the operator YAML; every Ray
worker re-reads the same :class:`KnowledgeConfig` from its own
:class:`ConfigurationManager` (the YAML is mounted at
``POLYMATHERA_CONFIG`` in every container), so the driver and every
actor agree on the configured backends without env-var passthrough.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from ..agents.blueprint import Blueprint
from ..distributed.config import get_component_or_default
from .cluster_config import KnowledgeConfig
from .embedder import InMemoryEmbedder
from .extractors.claims import (
    ClaimExtractor,
    DeterministicClaimExtractor,
    LLMCallable,
    LLMClaimExtractor,
)
from .ingestion import Ingestor
from .models import KnowledgeFormat
from .retrieval import RetrievalDeps
from .stores.graph import InMemoryGraphStore, KuzuGraphStore
from .stores.image import InMemoryImageStore, LocalFsImageStore
from .stores.vector import InMemoryVectorStore, QdrantVectorStore

if TYPE_CHECKING:
    from .embedder import Embedder
    from .stores import GraphStore, ImageStore, VectorStore


logger = logging.getLogger(__name__)


_lock = threading.Lock()
_deps: RetrievalDeps | None = None
_ingestor: Ingestor | None = None


def _knowledge_config() -> KnowledgeConfig:
    """Resolve the active :class:`KnowledgeConfig` from the global
    :class:`ConfigurationManager`. Falls back to bare defaults when
    the manager has not been initialized — the sync-safe path
    documented at :func:`get_component_or_default`."""

    return get_component_or_default("knowledge", KnowledgeConfig)


def _default_vector_store(embedder: "Embedder") -> "VectorStore":
    """Pick the default vector store from :class:`KnowledgeConfig`.

    When ``knowledge.qdrant.url`` is set, build a
    :class:`QdrantVectorStore` bound to ``knowledge.qdrant.collection``
    so ingested chunks survive process restarts. The store is
    constructed synchronously; collection creation is lazy on first
    async use. Falls back to :class:`InMemoryVectorStore` if Qdrant
    is configured but the client library is missing.

    ``QDRANT_API_KEY`` is read directly from the environment because
    it is a secret — declaring it in YAML would persist it in repo;
    the existing API-key prefix allowlist in the serving proxy
    forwards it to actor processes.
    """

    cfg = _knowledge_config().qdrant
    if not cfg.url:
        return InMemoryVectorStore()
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "knowledge.qdrant.url=%s but qdrant-client is not installed; "
            "falling back to InMemoryVectorStore.",
            cfg.url,
        )
        return InMemoryVectorStore()
    api_key = os.environ.get("QDRANT_API_KEY") or None
    try:
        client = AsyncQdrantClient(url=cfg.url, api_key=api_key, timeout=10)
        store = QdrantVectorStore(
            client=client,
            collection=cfg.collection,
            embedder_id=embedder.embedder_id,
            dimensions=embedder.dimensions,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to construct QdrantVectorStore for %s (%s); "
            "falling back to InMemoryVectorStore.",
            cfg.url, exc,
        )
        return InMemoryVectorStore()
    logger.info(
        "knowledge.deps: using QdrantVectorStore(url=%s, collection=%s, dim=%d)",
        cfg.url, cfg.collection, embedder.dimensions,
    )
    return store


def build_default_llm_callable(
    *, max_tokens: int, temperature: float,
) -> LLMCallable:
    """Build the lazy :data:`LLMCallable` the singleton Ingestor's
    :class:`LLMClaimExtractor` invokes per chunk.

    The wrapper:

    1. Lazily imports + awaits :func:`get_llm_cluster` on every call.
       The LLMCluster handle isn't available at ``set_knowledge_deps``
       time (the Ray Serving cluster may not be deployed yet), so
       construction-time resolution would race; per-call resolution
       is safe — the handle is cached inside ``get_llm_cluster``.
    2. Builds an :class:`InferenceRequest` with the active execution
       context (carried by ``serving.require_execution_context()``
       via the field's default factory).
    3. Forwards to ``LLMCluster.infer`` and returns the generated
       text — the shape :class:`LLMClaimExtractor` expects (raw
       string of JSON-shaped claim list).

    On failure (no cluster deployed, route refused, timeout, ...)
    the call raises; :meth:`LLMClaimExtractor.extract` already wraps
    in a ``try/except`` that logs + returns 0 claims for that chunk,
    so graceful degradation is automatic — the deterministic
    extractor still runs.
    """

    import uuid

    async def _llm_call(prompt: str) -> str:
        from .._handles import get_llm_cluster
        from ..cluster.models import InferenceRequest

        handle = await get_llm_cluster()
        request = InferenceRequest(
            request_id=f"claim_extract_{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = await handle.infer(request)
        return response.generated_text

    return _llm_call


def _build_default_extractors(
    cfg: KnowledgeConfig,
) -> tuple[ClaimExtractor, ...]:
    """Compose the default extractor tuple for the singleton Ingestor.

    Order: LLM extractor first (when enabled — produces the richer
    open-set claim types the design-process actions key on), then
    the deterministic ``is_a`` extractor as a complementary
    rule-based source. Both run on every chunk; their outputs are
    unioned into the Ingestor's claim list and written to the graph
    store independently.

    When ``llm_claim_extraction_enabled=False`` the LLM extractor is
    excluded — deterministic-only mode for cost-sensitive deployments
    or unit-test environments without a live LLM cluster.
    """

    extractors: list[ClaimExtractor] = []
    if cfg.llm_claim_extraction_enabled:
        llm_callable = build_default_llm_callable(
            max_tokens=cfg.llm_claim_extraction_max_tokens,
            temperature=cfg.llm_claim_extraction_temperature,
        )
        extractors.append(
            LLMClaimExtractor(
                llm_callable,
                timeout_s=cfg.llm_claim_extraction_timeout_s,
            ),
        )
    extractors.append(DeterministicClaimExtractor())
    return tuple(extractors)


def _default_graph_store() -> "GraphStore":
    """Pick the default graph store from :class:`KnowledgeConfig`.

    When ``knowledge.graph_db_path`` is set, build a
    :class:`KuzuGraphStore` rooted at that path so design-context
    + literature claims persist across process restarts and are
    shared across Ray workers / replicas on the same colony-shared
    volume. Falls back to :class:`InMemoryGraphStore` for tests and
    single-process dev runs (which works correctly for the same
    APIs but loses state on process exit).

    Falls back to in-memory if Kùzu is configured but the ``kuzu``
    package is missing, with a warning — the operator likely
    installed without the ``[knowledge]`` extra.
    """

    graph_db_path = _knowledge_config().graph_db_path
    if not graph_db_path:
        return InMemoryGraphStore()
    try:
        store = KuzuGraphStore.open(graph_db_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "knowledge.graph_db_path=%s but KuzuGraphStore.open failed "
            "(%s); falling back to InMemoryGraphStore. State will not "
            "persist across process restarts.",
            graph_db_path, exc,
        )
        return InMemoryGraphStore()
    logger.info(
        "knowledge.deps: using KuzuGraphStore(db_path=%s)", graph_db_path,
    )
    return store


def _default_image_store() -> "ImageStore":
    """Pick the default image store from :class:`KnowledgeConfig`.

    When ``knowledge.image_dir`` is set, build a
    :class:`LocalFsImageStore` rooted at that path so figures
    extracted at ingest survive process restarts and are visible to
    the dashboard's KB tab. Falls back to
    :class:`InMemoryImageStore` for tests and single-process dev runs.
    """

    image_dir = _knowledge_config().image_dir
    if not image_dir:
        return InMemoryImageStore()
    logger.info(
        "knowledge.deps: using LocalFsImageStore(root_dir=%s)", image_dir,
    )
    return LocalFsImageStore(root_dir=image_dir)


def _default_reader_registry(image_store: "ImageStore") -> Any:
    """Pick the default reader registry from :class:`KnowledgeConfig`.

    The registry's PDF reader is the layout-aware extractor wired to
    ``image_store``, selected by ``knowledge.pdf_extractor.backend``.
    Operator-supplied ``options`` flow through as ``backend_kwargs``
    so tier knobs (e.g. LlamaParse ``tier``, Gemini ``model``) reach
    the reader's constructor verbatim.
    """

    knowledge_cfg = _knowledge_config()
    cfg = knowledge_cfg.pdf_extractor
    backend = cfg.backend
    from .readers import default_registry_with_pdf_extractor

    try:
        registry = default_registry_with_pdf_extractor(
            backend=backend,
            image_store=image_store,
            backend_kwargs=cfg.options,
            fallback_backend=cfg.max_pages_fallback_backend,
            fallback_kwargs=cfg.max_pages_fallback_options,
            grobid_url=knowledge_cfg.grobid.url or None,
        )
    except (NotImplementedError, ValueError) as exc:
        logger.warning(
            "knowledge.pdf_extractor.backend=%s could not be honoured "
            "(%s); falling back to the default reader registry.",
            backend, exc,
        )
        return None
    pdf_reader = registry.reader_for(KnowledgeFormat.PDF)
    logger.info(
        "knowledge.deps: PDF reader = %s (backend=%s)",
        type(pdf_reader).__name__ if pdf_reader is not None else "<none>",
        backend,
    )
    return registry


def set_knowledge_deps(
    *,
    embedder: "Embedder | None" = None,
    vector_store: "VectorStore | None" = None,
    graph_store: "GraphStore | None" = None,
    image_store: "ImageStore | None" = None,
) -> RetrievalDeps:
    """Replace the process-wide deps. Call once during cluster bring-up.

    Any field left as ``None`` is filled from :class:`KnowledgeConfig`
    via the global :class:`ConfigurationManager`:
    ``vector_store`` from ``knowledge.qdrant`` (or in-memory if empty)
    and ``image_store`` from ``knowledge.image_dir`` (or in-memory if
    empty). Calling this again re-binds the singleton — the old
    value is discarded; live capability instances continue to hold
    their own references.
    """

    global _deps, _ingestor
    with _lock:
        resolved_embedder = embedder or InMemoryEmbedder()
        resolved_image_store = image_store or _default_image_store()
        # Graph store: explicit override wins; otherwise fall back to
        # ``knowledge.graph_db_path`` (Kuzu on disk) or
        # ``InMemoryGraphStore`` per :func:`_default_graph_store`.
        resolved_graph_store = (
            graph_store if graph_store is not None
            else _default_graph_store()
        )
        _deps = RetrievalDeps(
            embedder=resolved_embedder,
            vector_store=vector_store or _default_vector_store(resolved_embedder),
            graph_store=resolved_graph_store,
            image_store=resolved_image_store,
        )
        # Build a matching Ingestor so curation and retrieval share
        # the *same* embedder + vector store + image store + graph
        # store. Re-bound on every set. The reader registry is
        # resolved from ``knowledge.pdf_extractor`` so flipping the
        # backend in YAML is picked up on the next
        # ``set_knowledge_deps()`` call.
        #
        # ``extractors`` is the tuple of :class:`ClaimExtractor`
        # instances applied to every chunk. Built by
        # :func:`_build_default_extractors` from the active
        # :class:`KnowledgeConfig`: by default that's
        # ``(LLMClaimExtractor(_lazy_llm_callable), DeterministicClaimExtractor())``
        # — LLM first for the rich open-set claim types,
        # deterministic as a complementary rule-based source.
        # Operators wanting fully-custom extractors wire a custom
        # Ingestor via :func:`set_knowledge_deps` and pass it through
        # explicitly (today the only public switch is the
        # ``llm_claim_extraction_enabled`` flag).
        config_registry = _default_reader_registry(resolved_image_store)
        knowledge_cfg = _knowledge_config()
        _ingestor = Ingestor(
            readers=config_registry,
            extractors=_build_default_extractors(knowledge_cfg),
            embedder=_deps.embedder,
            vector_store=_deps.vector_store,
            graph_store=_deps.graph_store,
            image_store=_deps.image_store,
        )
    return _deps


def get_knowledge_deps() -> RetrievalDeps:
    """Return the process-wide :class:`RetrievalDeps`. Initialises the
    in-memory default on first call. Sync — no I/O involved."""

    if _deps is None:
        return set_knowledge_deps()
    return _deps


def get_default_ingestor() -> Ingestor:
    """Return the process-wide :class:`Ingestor`, sharing the same
    backends as the deps from :func:`get_knowledge_deps`."""

    if _ingestor is None:
        set_knowledge_deps()
    assert _ingestor is not None
    return _ingestor


def reset_knowledge_deps() -> None:
    """Drop the cached deps + ingestor — for tests only."""

    global _deps, _ingestor
    with _lock:
        _deps = None
        _ingestor = None


# ---------------------------------------------------------------------------
# Blueprint factories — for cross-Ray-boundary use
# ---------------------------------------------------------------------------
#
# The dashboard process must NOT eagerly resolve a singleton ``Ingestor``
# / ``RetrievalDeps`` and pass them into ``Capability.bind(...)`` because
# the Qdrant client carries a thread RLock and the in-memory token
# counter is a local closure — neither survives cloudpickle. Instead
# the dashboard hands each capability a ``Blueprint`` whose kwargs are
# all picklable (URLs, collection names, dimensions); the worker
# process resolves the blueprint via ``local_instance()``, building
# its own Ingestor + Qdrant client locally. Same KnowledgeConfig
# (loaded from the same YAML), same collection — different Python
# objects per process.


def _default_vector_store_blueprint(
    embedder: "Embedder",
) -> Blueprint:
    """Pick the default vector-store *blueprint* from :class:`KnowledgeConfig`.

    Mirrors :func:`_default_vector_store` but returns a picklable
    :class:`Blueprint` rather than a live store, so the result can
    cross the Ray boundary. ``QDRANT_API_KEY`` stays a direct env-var
    read because it is a secret and must not land in YAML.
    """

    cfg = _knowledge_config().qdrant
    if not cfg.url:
        return InMemoryVectorStore.bind()
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantVectorStore.bind(
        url=cfg.url,
        api_key=api_key,
        collection=cfg.collection,
        embedder_id=embedder.embedder_id,
        dimensions=embedder.dimensions,
    )


def _default_image_store_blueprint() -> Blueprint:
    """Pick the default image-store *blueprint* from :class:`KnowledgeConfig`.

    Mirrors :func:`_default_image_store` but returns a picklable
    :class:`Blueprint` so capability bind sites can ship it across
    the Ray boundary without holding the live filesystem handle.
    """

    image_dir = _knowledge_config().image_dir
    if not image_dir:
        return InMemoryImageStore.bind()
    return LocalFsImageStore.bind(root_dir=image_dir)


def default_retrieval_deps_blueprint() -> Blueprint:
    """Build a picklable :class:`RetrievalDeps` blueprint from
    :class:`KnowledgeConfig`.

    Returned blueprint resolves on the worker into a ``RetrievalDeps``
    with an :class:`InMemoryEmbedder`, either an
    :class:`InMemoryVectorStore` (empty ``knowledge.qdrant.url``) or a
    :class:`QdrantVectorStore` pointing at the operator-configured
    Qdrant, and an :class:`InMemoryImageStore` /
    :class:`LocalFsImageStore` based on ``knowledge.image_dir``. Pass
    this — not :func:`get_knowledge_deps()` — into
    :meth:`KnowledgeRetrievalCapability.bind`.
    """

    embedder = InMemoryEmbedder()
    return RetrievalDeps.bind(
        embedder=InMemoryEmbedder.bind(),
        vector_store=_default_vector_store_blueprint(embedder),
        image_store=_default_image_store_blueprint(),
    )


def default_ingestor_blueprint() -> Blueprint:
    """Build a picklable :class:`Ingestor` blueprint from
    :class:`KnowledgeConfig`.

    Companion to :func:`default_retrieval_deps_blueprint` — the two
    share the same embedder + vector-store + image-store recipe so
    curation and retrieval land in the same collection and reference
    the same image bytes. Pass this into
    :meth:`KnowledgeCuratorCapability.bind` and any other write-side
    capability that owns an ingestor.
    """

    embedder = InMemoryEmbedder()
    return Ingestor.bind(
        embedder=InMemoryEmbedder.bind(),
        vector_store=_default_vector_store_blueprint(embedder),
        image_store=_default_image_store_blueprint(),
    )


__all__ = (
    "default_ingestor_blueprint",
    "default_retrieval_deps_blueprint",
    "get_default_ingestor",
    "get_knowledge_deps",
    "reset_knowledge_deps",
    "set_knowledge_deps",
)
