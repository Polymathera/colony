"""Smoke tests for the KB router.

The full HTTP surface needs a live cluster (auth middleware + the
``RetrievalDeps`` singleton bound to a real Qdrant); we exercise only
what is testable in isolation: route registration and the small
``_backend_info`` helper that drives the dashboard's "which store is
live?" badge.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.deps import (
    reset_knowledge_deps,
    set_knowledge_deps,
)
from polymathera.colony.knowledge.stores.vector import InMemoryVectorStore
from polymathera.colony.web_ui.backend.routers import kb as kb_router


@pytest.fixture(autouse=True)
def _reset_deps() -> None:
    reset_knowledge_deps()
    yield
    reset_knowledge_deps()


def test_router_registers_expected_paths() -> None:
    """The KB tab calls these five endpoints — fail fast if any go
    missing on rename / refactor."""

    paths = {route.path for route in kb_router.router.routes}
    assert paths == {
        "/kb/stats",
        "/kb/sources",
        "/kb/sources/chunks",
        "/kb/search",
        "/kb/ingest",
    }


def test_backend_info_in_memory_default(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    set_knowledge_deps()
    info = kb_router._backend_info()
    assert info.vector_store == "InMemoryVectorStore"
    assert info.qdrant_url is None
    assert info.qdrant_collection is None
    # InMemoryEmbedder ships at 64d.
    assert info.embedder_dimensions == 64


def test_backend_info_reports_qdrant_when_set(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "kb_test")
    # Force the in-memory store anyway so we don't depend on
    # qdrant-client being importable in the test env — the
    # ``backend`` field reflects the *live* store, not the env.
    set_knowledge_deps(vector_store=InMemoryVectorStore())
    info = kb_router._backend_info()
    assert info.vector_store == "InMemoryVectorStore"
    # ``qdrant_url`` mirrors the env, so operators see what was wired
    # even when the live store fell back to in-memory.
    assert info.qdrant_url == "http://qdrant:6333"
    assert info.qdrant_collection == "kb_test"
