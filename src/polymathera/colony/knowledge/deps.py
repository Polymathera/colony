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
import threading
from typing import TYPE_CHECKING

from .embedder import InMemoryEmbedder
from .ingestion import Ingestor
from .retrieval import RetrievalDeps
from .stores.vector import InMemoryVectorStore

if TYPE_CHECKING:
    from .embedder import Embedder
    from .stores import GraphStore, VectorStore


logger = logging.getLogger(__name__)


_lock = threading.Lock()
_deps: RetrievalDeps | None = None
_ingestor: Ingestor | None = None


def set_knowledge_deps(
    *,
    embedder: "Embedder | None" = None,
    vector_store: "VectorStore | None" = None,
    graph_store: "GraphStore | None" = None,
) -> RetrievalDeps:
    """Replace the process-wide deps. Call once during cluster bring-up.

    Any field left as ``None`` is filled with the in-memory default.
    Calling this again re-binds the singleton — the old value is
    discarded; live capability instances continue to hold their own
    references.
    """

    global _deps, _ingestor
    with _lock:
        _deps = RetrievalDeps(
            embedder=embedder or InMemoryEmbedder(),
            vector_store=vector_store or InMemoryVectorStore(),
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


__all__ = (
    "get_default_ingestor",
    "get_knowledge_deps",
    "reset_knowledge_deps",
    "set_knowledge_deps",
)
