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
    """Fail fast if any of the dashboard-facing KB endpoints go
    missing on rename / refactor."""

    paths = {route.path for route in kb_router.router.routes}
    assert paths == {
        "/kb/stats",
        "/kb/sources",
        "/kb/sources/chunks",
        "/kb/search",
        "/kb/ingest",
        "/kb/ingest-repo-map",
        "/kb/ingest-repo-map/operations",
        "/kb/images/{sha}",
    }


def test_backend_info_in_memory_default(monkeypatch) -> None:
    """With the default :class:`KnowledgeConfig` (empty
    ``qdrant.url``), the backend info reports the in-memory store
    and nulls out the URL fields."""
    import polymathera.colony.distributed.config as cm_module
    from polymathera.colony.knowledge import deps as deps_module
    from polymathera.colony.knowledge.cluster_config import KnowledgeConfig

    cfg = KnowledgeConfig()
    monkeypatch.setattr(deps_module, "_knowledge_config", lambda: cfg)
    # ``_backend_info`` looks up the config via the manager helper;
    # patch at its source so the lazy import inside the function picks
    # the patched binding up.
    monkeypatch.setattr(
        cm_module, "get_component_or_default", lambda path, klass: cfg,
    )
    set_knowledge_deps()
    info = kb_router._backend_info()
    assert info.vector_store == "InMemoryVectorStore"
    assert info.qdrant_url is None
    assert info.qdrant_collection is None
    # InMemoryEmbedder ships at 64d.
    assert info.embedder_dimensions == 64


def test_backend_info_reports_qdrant_when_set(monkeypatch) -> None:
    """When ``knowledge.qdrant.url`` is configured in the typed
    config, the backend info surfaces it — the ``vector_store`` field
    still reflects the *live* store (which may have fallen back to
    in-memory if qdrant-client wasn't importable)."""
    import polymathera.colony.distributed.config as cm_module
    from polymathera.colony.knowledge import deps as deps_module
    from polymathera.colony.knowledge.cluster_config import (
        KnowledgeConfig,
        QdrantConfig,
    )

    cfg = KnowledgeConfig(
        qdrant=QdrantConfig(url="http://qdrant:6333", collection="kb_test"),
    )
    monkeypatch.setattr(deps_module, "_knowledge_config", lambda: cfg)
    monkeypatch.setattr(
        cm_module, "get_component_or_default", lambda path, klass: cfg,
    )
    # Force the in-memory store anyway so we don't depend on
    # qdrant-client being importable in the test env.
    set_knowledge_deps(vector_store=InMemoryVectorStore())
    info = kb_router._backend_info()
    assert info.vector_store == "InMemoryVectorStore"
    assert info.qdrant_url == "http://qdrant:6333"
    assert info.qdrant_collection == "kb_test"


# ---------------------------------------------------------------------------
# /kb/images/<sha> — the figure-resolve endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_endpoint_round_trips_bytes_via_image_store() -> None:
    """End-to-end: put bytes into the active ImageStore, then call
    the endpoint handler with the resulting sha and assert the
    bytes + content-type round-trip."""
    from polymathera.colony.knowledge.stores.image import (
        InMemoryImageStore, _sha_from_uri,
    )

    store = InMemoryImageStore()
    set_knowledge_deps(image_store=store)
    payload = b"\x89PNG\r\n\x1a\nfake-png-bytes"
    uri = await store.put(payload, mime="image/png")
    sha = _sha_from_uri(uri)

    # Auth middleware is bypassed in unit tests by calling the
    # endpoint coroutine directly with a sentinel ``_user``.
    response = await kb_router.kb_image_resolve(sha=sha, _user={"sub": "test"})
    assert response.status_code == 200
    assert response.body == payload
    assert response.media_type == "image/png"
    # Content-addressed bytes are immutable; the cache header must
    # mark them as such so the browser doesn't re-fetch on every
    # chunk render.
    cache = response.headers.get("cache-control") or response.headers.get("Cache-Control")
    assert cache is not None and "immutable" in cache


@pytest.mark.asyncio
async def test_image_endpoint_404_for_unknown_sha() -> None:
    """A sha that's not in the store returns 404 (not 500) so the
    KB tab can render a placeholder for an evicted figure."""
    from fastapi import HTTPException
    from polymathera.colony.knowledge.stores.image import InMemoryImageStore

    set_knowledge_deps(image_store=InMemoryImageStore())
    fake_sha = "0" * 64
    with pytest.raises(HTTPException) as exc:
        await kb_router.kb_image_resolve(sha=fake_sha, _user={"sub": "test"})
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_image_endpoint_rejects_non_hex_sha() -> None:
    """A sha that contains characters outside [0-9a-f] is rejected
    with 400 — this stops a bogus path component from traversing
    out of the shard tree on the local-FS backend."""
    from fastapi import HTTPException
    from polymathera.colony.knowledge.stores.image import InMemoryImageStore

    set_knowledge_deps(image_store=InMemoryImageStore())
    with pytest.raises(HTTPException) as exc:
        await kb_router.kb_image_resolve(
            sha="../etc/passwd", _user={"sub": "test"},
        )
    assert exc.value.status_code == 400
