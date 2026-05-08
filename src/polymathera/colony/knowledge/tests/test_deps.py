"""Backend-selection tests for ``knowledge.deps``.

Exercises the ``QDRANT_URL``-driven default in :func:`set_knowledge_deps`
without requiring a running Qdrant. The construction is sync and the
collection is created lazily on first async use, so we can assert on
the wired type without any network I/O.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.deps import (
    get_knowledge_deps,
    reset_knowledge_deps,
    set_knowledge_deps,
)
from polymathera.colony.knowledge.stores.vector import (
    InMemoryVectorStore,
    QdrantVectorStore,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_knowledge_deps()
    yield
    reset_knowledge_deps()


def test_default_is_in_memory_when_no_qdrant_url(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    deps = set_knowledge_deps()
    assert isinstance(deps.vector_store, InMemoryVectorStore)


def test_picks_qdrant_when_url_set(monkeypatch) -> None:
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "kb_test")
    deps = set_knowledge_deps()
    assert isinstance(deps.vector_store, QdrantVectorStore)
    # Lazy: no network I/O happened — only the client object was built.
    assert deps.vector_store.collection_name == "kb_test"


def test_falls_back_when_qdrant_client_missing(monkeypatch) -> None:
    """If QDRANT_URL is set but qdrant-client cannot be imported,
    silently fall back to InMemoryVectorStore so the dashboard
    starts cleanly with a clear log line."""

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    import builtins as _b

    real_import = _b.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise ImportError("simulated missing qdrant-client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)
    deps = set_knowledge_deps()
    assert isinstance(deps.vector_store, InMemoryVectorStore)


def test_explicit_vector_store_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    forced = InMemoryVectorStore()
    deps = set_knowledge_deps(vector_store=forced)
    assert deps.vector_store is forced


def test_get_knowledge_deps_lazy_init(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    deps = get_knowledge_deps()
    assert isinstance(deps.vector_store, InMemoryVectorStore)
    # Singleton: subsequent calls return the same object.
    assert get_knowledge_deps() is deps
