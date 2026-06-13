"""Backend-selection tests for ``knowledge.deps``.

Exercises :class:`KnowledgeConfig`-driven defaults in
:func:`set_knowledge_deps` without requiring a running Qdrant. The
construction is sync and the collection is created lazily on first
async use, so we can assert on the wired type without any network
I/O.

Tests inject the typed config by patching
``knowledge.deps._knowledge_config`` — the same path the runtime
takes via :func:`get_component_or_default`, only short-circuited so
unit tests don't need to boot the global ``ConfigurationManager``.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge import deps as deps_module
from polymathera.colony.knowledge.cluster_config import (
    KnowledgeConfig,
    QdrantConfig,
)
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


def _patch_config(monkeypatch: pytest.MonkeyPatch, cfg: KnowledgeConfig) -> None:
    """Inject a :class:`KnowledgeConfig` so deps.py resolves to it
    instead of touching the global ConfigurationManager."""
    monkeypatch.setattr(deps_module, "_knowledge_config", lambda: cfg)


def test_default_is_in_memory_when_qdrant_url_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ``knowledge.qdrant.url`` (the default) selects
    :class:`InMemoryVectorStore`."""
    _patch_config(monkeypatch, KnowledgeConfig())
    deps = set_knowledge_deps()
    assert isinstance(deps.vector_store, InMemoryVectorStore)


def test_picks_qdrant_when_url_set(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("qdrant_client")
    _patch_config(monkeypatch, KnowledgeConfig(
        qdrant=QdrantConfig(url="http://qdrant:6333", collection="kb_test"),
    ))
    deps = set_knowledge_deps()
    assert isinstance(deps.vector_store, QdrantVectorStore)
    # Lazy: no network I/O happened — only the client object was built.
    assert deps.vector_store.collection_name == "kb_test"


def test_falls_back_when_qdrant_client_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``knowledge.qdrant.url`` is set but qdrant-client cannot be
    imported, silently fall back to :class:`InMemoryVectorStore` so
    the dashboard starts cleanly with a clear log line."""

    _patch_config(monkeypatch, KnowledgeConfig(
        qdrant=QdrantConfig(url="http://qdrant:6333"),
    ))
    import builtins as _b

    real_import = _b.__import__

    def fake_import(name, *args, **kwargs):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise ImportError("simulated missing qdrant-client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)
    deps = set_knowledge_deps()
    assert isinstance(deps.vector_store, InMemoryVectorStore)


def test_explicit_vector_store_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, KnowledgeConfig(
        qdrant=QdrantConfig(url="http://qdrant:6333"),
    ))
    forced = InMemoryVectorStore()
    deps = set_knowledge_deps(vector_store=forced)
    assert deps.vector_store is forced


def test_get_knowledge_deps_lazy_init(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_config(monkeypatch, KnowledgeConfig())
    deps = get_knowledge_deps()
    assert isinstance(deps.vector_store, InMemoryVectorStore)
    # Singleton: subsequent calls return the same object.
    assert get_knowledge_deps() is deps


# ---------------------------------------------------------------------------
# Phase P3a: graph_store + claim extractor wiring
# ---------------------------------------------------------------------------


def test_default_graph_store_is_in_memory_when_path_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ``knowledge.graph_db_path`` (the default) selects
    :class:`InMemoryGraphStore` — no Kùzu install needed for
    in-memory dev / tests."""

    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    _patch_config(monkeypatch, KnowledgeConfig())
    deps = set_knowledge_deps()
    assert isinstance(deps.graph_store, InMemoryGraphStore)


def test_default_graph_store_is_kuzu_when_path_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: "pytest.Path",
) -> None:
    """``knowledge.graph_db_path`` set → lazily open a
    :class:`KuzuGraphStore` rooted at that path."""

    pytest.importorskip("kuzu")
    from polymathera.colony.knowledge.stores.graph import KuzuGraphStore

    db_path = tmp_path / "design.kuzu"
    _patch_config(
        monkeypatch,
        KnowledgeConfig(graph_db_path=str(db_path)),
    )
    deps = set_knowledge_deps()
    assert isinstance(deps.graph_store, KuzuGraphStore)
    assert db_path.exists() or db_path.is_dir()


def test_default_graph_store_falls_back_when_kuzu_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: "pytest.Path",
) -> None:
    """``knowledge.graph_db_path`` set but ``kuzu`` package can't
    open / import → InMemoryGraphStore with a warning (operator
    likely missed the [knowledge] extra)."""

    from polymathera.colony.knowledge import stores as stores_module
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    db_path = tmp_path / "broken.kuzu"
    _patch_config(
        monkeypatch,
        KnowledgeConfig(graph_db_path=str(db_path)),
    )

    # Make KuzuGraphStore.open raise — simulates missing kuzu pkg.
    from polymathera.colony.knowledge.stores import graph as graph_module

    def _raise_open(_db_path):
        raise graph_module.GraphStoreError("simulated missing kuzu")

    monkeypatch.setattr(
        graph_module.KuzuGraphStore, "open", staticmethod(_raise_open),
    )
    deps = set_knowledge_deps()
    assert isinstance(deps.graph_store, InMemoryGraphStore)


def test_explicit_graph_store_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: "pytest.Path",
) -> None:
    """A graph_store passed to set_knowledge_deps wins over the config."""

    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    _patch_config(
        monkeypatch,
        KnowledgeConfig(graph_db_path=str(tmp_path / "x.kuzu")),
    )
    forced = InMemoryGraphStore()
    deps = set_knowledge_deps(graph_store=forced)
    assert deps.graph_store is forced


def test_default_ingestor_has_deterministic_claim_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase P3a wires :class:`DeterministicClaimExtractor` into the
    singleton Ingestor so claims actually populate the (in-memory or
    Kuzu) graph store. Without an extractor wired, ``_run_extractors``
    returns ``[]`` and ``add_claim`` never fires."""

    from polymathera.colony.knowledge.deps import get_default_ingestor
    from polymathera.colony.knowledge.extractors.claims import (
        DeterministicClaimExtractor,
    )

    _patch_config(monkeypatch, KnowledgeConfig())
    set_knowledge_deps()
    ingestor = get_default_ingestor()
    # ``Ingestor`` stores its extractors in ``_extractors`` (tuple).
    assert any(
        isinstance(ext, DeterministicClaimExtractor)
        for ext in ingestor._extractors
    )


# ---------------------------------------------------------------------------
# Phase P3d: LLMClaimExtractor wiring
# ---------------------------------------------------------------------------


def test_default_ingestor_includes_llm_claim_extractor_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase P3d wires :class:`LLMClaimExtractor` into the singleton
    Ingestor by default. The LLM extractor produces the open-set
    typed claims (``hypothesizes`` / ``contradicts`` / ``verifies``
    / ``constrains`` / ...) that the design-process actions key on;
    without it, those actions return empty for real-world prose."""

    from polymathera.colony.knowledge.deps import get_default_ingestor
    from polymathera.colony.knowledge.extractors.claims import (
        DeterministicClaimExtractor,
        LLMClaimExtractor,
    )

    _patch_config(monkeypatch, KnowledgeConfig())
    set_knowledge_deps()
    ingestor = get_default_ingestor()
    # Both extractors present; LLM first per :func:`_build_default_extractors`.
    extractor_types = [type(e) for e in ingestor._extractors]
    assert extractor_types[0] is LLMClaimExtractor
    assert any(t is DeterministicClaimExtractor for t in extractor_types)


def test_llm_claim_extraction_disabled_excludes_llm_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``llm_claim_extraction_enabled=False`` ships deterministic-only
    — for unit-test envs / cost-sensitive deployments without an
    LLM cluster."""

    from polymathera.colony.knowledge.deps import get_default_ingestor
    from polymathera.colony.knowledge.extractors.claims import (
        DeterministicClaimExtractor,
        LLMClaimExtractor,
    )

    _patch_config(
        monkeypatch,
        KnowledgeConfig(llm_claim_extraction_enabled=False),
    )
    set_knowledge_deps()
    ingestor = get_default_ingestor()
    extractor_types = [type(e) for e in ingestor._extractors]
    assert all(t is not LLMClaimExtractor for t in extractor_types)
    assert any(t is DeterministicClaimExtractor for t in extractor_types)


def test_llm_callable_builds_inference_request_with_config_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_default_llm_callable`` lazily resolves the LLMCluster
    handle and builds an :class:`InferenceRequest` whose ``prompt``,
    ``max_tokens``, ``temperature``, ``json_schema``, and ``deadline_s``
    mirror the config + the inbound prompt + schema. Under the typed
    contract (Change 7) the callable returns a validated pydantic
    instance; the deployment's ``generated_text`` IS the tool-use JSON
    string that round-trips through ``schema.model_validate_json``."""

    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from polymathera.colony.cluster.models import InferenceResponse
    from polymathera.colony.distributed.ray_utils import serving as _serving
    from polymathera.colony.knowledge.deps import (
        build_default_llm_callable,
    )
    from polymathera.colony.knowledge.extractors.claims import ClaimList

    # The deployment's response under structured output is a JSON
    # string that validates against the supplied schema. We return a
    # minimal ``ClaimList`` payload here.
    fake_handle = MagicMock()
    fake_handle.infer = AsyncMock(return_value=InferenceResponse(
        request_id="x",
        generated_text=(
            '{"claims": [{"subject":"s","predicate":"is_a",'
            '"object":"o","confidence":0.9}]}'
        ),
        tokens_generated=10,
        latency_ms=12.5,
    ))

    async def _stub_get_llm_cluster(_app_name=None):
        return fake_handle

    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_llm_cluster", _stub_get_llm_cluster)

    callable_ = build_default_llm_callable(
        max_tokens=4096, temperature=0.2, deadline_s=15.0,
    )

    # InferenceRequest's syscontext field default-factories from
    # ``serving.require_execution_context()`` — provide one for the
    # test. Use a synthetic context tag for clarity.
    async def _invoke() -> ClaimList:
        with _serving.execution_context(
            tenant_id="t", colony_id="c", session_id=None, origin="test",
        ):
            return await callable_("hello prompt", ClaimList)

    # Match the surrounding-test convention (deprecated but does NOT
    # close the global event loop, so subsequent tests in the same
    # session that also use ``get_event_loop().run_until_complete``
    # don't break).
    result = asyncio.get_event_loop().run_until_complete(_invoke())

    # The typed contract returns a validated ``ClaimList`` instance.
    assert isinstance(result, ClaimList)
    assert result.claims[0].predicate == "is_a"
    fake_handle.infer.assert_awaited_once()
    req = fake_handle.infer.await_args.args[0]
    assert req.prompt == "hello prompt"
    assert req.max_tokens == 4096
    assert req.temperature == 0.2
    assert req.deadline_s == 15.0
    assert req.json_schema == ClaimList.model_json_schema()
    assert req.request_id.startswith("claim_extract_")


def test_llm_claim_extractor_degrades_gracefully_when_no_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the LLM cluster is undeployed / unreachable, the LLM
    extractor logs at WARN + returns zero claims per chunk — the
    Ingestor's other extractors (deterministic) keep producing
    claims, so the path-1 pipeline degrades without crashing."""

    import asyncio
    from polymathera.colony.knowledge.deps import (
        build_default_llm_callable,
    )
    from polymathera.colony.knowledge.extractors.claims import (
        ExtractionPrompt, LLMClaimExtractor,
    )
    from polymathera.colony.knowledge.models import (
        CitationSpan, Chunk, KnowledgeFormat,
    )

    # ``get_llm_cluster`` raises (no LLM cluster up).
    async def _broken_get_llm_cluster(_app_name=None):
        raise RuntimeError("no LLM cluster deployed")

    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(
        handles_mod, "get_llm_cluster", _broken_get_llm_cluster,
    )

    callable_ = build_default_llm_callable(
        max_tokens=1024, temperature=0.0, deadline_s=5.0,
    )
    extractor = LLMClaimExtractor(
        callable_, prompt=ExtractionPrompt(),
    )
    chunk = Chunk(
        text="A is a B.",
        citation=CitationSpan(
            source_uri="design_context://test/foo.md",
            source_format=KnowledgeFormat.MARKDOWN,
            section_path="",
            char_start=0, char_end=10,
        ),
    )
    # Match surrounding-test convention — see note in the previous
    # test about not closing the global event loop.
    claims = asyncio.get_event_loop().run_until_complete(
        extractor.extract(chunk),
    )
    assert claims == ()  # graceful empty, not crash
