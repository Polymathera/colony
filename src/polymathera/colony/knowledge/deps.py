"""Process-singleton factory for the knowledge trio's shared backends.

The :class:`Ingestor`, :class:`BulkAcquisitionCapability`, and
:class:`KnowledgeRetrievalCapability` all need the same triple of
``embedder + vector_store + (optional) graph_store`` — bound the same
way so curation and retrieval share an embedding space and a
collection. This module owns that triple as a per-process singleton so
the dashboard / SessionAgent can wire all three capabilities without
each having to construct deps independently.

Out-of-the-box defaults are :class:`InMemoryEmbedder` and
:class:`InMemoryVectorStore` — usable for development, single-node
deployments, and chat-driven curation that does not need persistence
across processes. Production users override by calling
:func:`set_knowledge_deps` once during cluster bring-up before the
SessionAgent blueprint is constructed.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from ..agents.blueprint import Blueprint
from .embedder import InMemoryEmbedder
from .ingestion import Ingestor
from .retrieval import RetrievalDeps
from .stores.vector import InMemoryVectorStore, QdrantVectorStore

if TYPE_CHECKING:
    from .embedder import Embedder
    from .stores import GraphStore, VectorStore


logger = logging.getLogger(__name__)


_lock = threading.Lock()
_deps: RetrievalDeps | None = None
_ingestor: Ingestor | None = None


def _default_vector_store(embedder: "Embedder") -> "VectorStore":
    """Pick the default vector store based on environment.

    When ``QDRANT_URL`` is set, build a :class:`QdrantVectorStore` bound
    to ``QDRANT_COLLECTION`` (default ``colony_knowledge``) so ingested
    chunks survive process restarts. The store is constructed
    synchronously; collection creation is lazy on first async use.
    Falls back to :class:`InMemoryVectorStore` if Qdrant is configured
    but the client library is missing or the URL cannot be parsed.
    """

    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        return InMemoryVectorStore()
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "QDRANT_URL=%s but qdrant-client is not installed; "
            "falling back to InMemoryVectorStore.",
            qdrant_url,
        )
        return InMemoryVectorStore()
    collection = os.environ.get("QDRANT_COLLECTION", "colony_knowledge")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    try:
        client = AsyncQdrantClient(url=qdrant_url, api_key=api_key, timeout=10)
        store = QdrantVectorStore(
            client=client,
            collection=collection,
            embedder_id=embedder.embedder_id,
            dimensions=embedder.dimensions,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to construct QdrantVectorStore for %s (%s); "
            "falling back to InMemoryVectorStore.",
            qdrant_url, exc,
        )
        return InMemoryVectorStore()
    logger.info(
        "knowledge.deps: using QdrantVectorStore(url=%s, collection=%s, dim=%d)",
        qdrant_url, collection, embedder.dimensions,
    )
    return store


def set_knowledge_deps(
    *,
    embedder: "Embedder | None" = None,
    vector_store: "VectorStore | None" = None,
    graph_store: "GraphStore | None" = None,
) -> RetrievalDeps:
    """Replace the process-wide deps. Call once during cluster bring-up.

    Any field left as ``None`` is filled with the in-memory default,
    except ``vector_store``, which honours ``QDRANT_URL`` to pick
    between the in-memory and Qdrant defaults.
    Calling this again re-binds the singleton — the old value is
    discarded; live capability instances continue to hold their own
    references.
    """

    global _deps, _ingestor
    with _lock:
        resolved_embedder = embedder or InMemoryEmbedder()
        _deps = RetrievalDeps(
            embedder=resolved_embedder,
            vector_store=vector_store or _default_vector_store(resolved_embedder),
            graph_store=graph_store,
        )
        # Build a matching Ingestor so curation and retrieval share
        # the *same* embedder + vector store. Re-bound on every set.
        _ingestor = Ingestor(
            embedder=_deps.embedder,
            vector_store=_deps.vector_store,
            graph_store=_deps.graph_store,
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
# its own Ingestor + Qdrant client locally. Same QDRANT_URL, same
# collection — different Python objects per process.


def _default_vector_store_blueprint(
    embedder: "Embedder",
) -> Blueprint:
    """Pick the default vector-store *blueprint* based on environment.

    Mirrors :func:`_default_vector_store` but returns a picklable
    :class:`Blueprint` rather than a live store, so the result can
    cross the Ray boundary.
    """

    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        return InMemoryVectorStore.bind()
    collection = os.environ.get("QDRANT_COLLECTION", "colony_knowledge")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantVectorStore.bind(
        url=qdrant_url,
        api_key=api_key,
        collection=collection,
        embedder_id=embedder.embedder_id,
        dimensions=embedder.dimensions,
    )


def default_retrieval_deps_blueprint() -> Blueprint:
    """Build a picklable :class:`RetrievalDeps` blueprint from env.

    Returned blueprint resolves on the worker into a ``RetrievalDeps``
    with an :class:`InMemoryEmbedder` and either an
    :class:`InMemoryVectorStore` (no ``QDRANT_URL``) or a
    :class:`QdrantVectorStore` pointing at the operator-configured
    Qdrant. Pass this — not :func:`get_knowledge_deps()` — into
    :meth:`KnowledgeRetrievalCapability.bind`.
    """

    embedder = InMemoryEmbedder()
    return RetrievalDeps.bind(
        embedder=InMemoryEmbedder.bind(),
        vector_store=_default_vector_store_blueprint(embedder),
    )


def default_ingestor_blueprint() -> Blueprint:
    """Build a picklable :class:`Ingestor` blueprint from env.

    Companion to :func:`default_retrieval_deps_blueprint` — the two
    share the same embedder + vector-store recipe so curation and
    retrieval land in the same collection. Pass this into
    :meth:`BulkAcquisitionCapability.bind` and
    :meth:`KnowledgeCuratorCapability.bind`.
    """

    embedder = InMemoryEmbedder()
    return Ingestor.bind(
        embedder=InMemoryEmbedder.bind(),
        vector_store=_default_vector_store_blueprint(embedder),
    )


__all__ = (
    "default_ingestor_blueprint",
    "default_retrieval_deps_blueprint",
    "get_default_ingestor",
    "get_knowledge_deps",
    "reset_knowledge_deps",
    "set_knowledge_deps",
)
