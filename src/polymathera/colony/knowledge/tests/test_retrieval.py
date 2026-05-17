"""Tests for the five retrieval modes (exposed as ``LocalToolCapability`` subclasses).

The retrieval capabilities are agent-mountable. Tests construct each
in detached mode (``agent=None``) inside an execution_context and
invoke the public ``retrieve()`` action directly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
from polymathera.colony.knowledge import (
    BudgetedRetrievalCapability,
    Chunk,
    CitationSpan,
    Claim,
    CorpusTier,
    EmbeddedChunk,
    GraphRetrievalCapability,
    GroundedRetrievalCapability,
    InMemoryEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    RetrievalDeps,
    ScopedRetrievalCapability,
    StandardsRetrievalCapability,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_ctx():
    """The SESSION-scoped capabilities resolve their scope from the
    execution context's session_id."""
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


@pytest.fixture
async def populated_deps() -> RetrievalDeps:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    gstore = InMemoryGraphStore()
    rows = [
        ("alpha alpha alpha", "src:tool_a", "code", CorpusTier.TIER_4_SOFTWARE_DOCS),
        ("beta beta beta", "src:tool_b", "code", CorpusTier.TIER_4_SOFTWARE_DOCS),
        ("foundational physics text", "book:wesson", "paper_section", CorpusTier.TIER_1_FOUNDATIONS),
        ("research paper extract", "arxiv:1", "paper_section", CorpusTier.TIER_3_RESEARCH),
    ]
    texts = [r[0] for r in rows]
    vectors = await embedder.embed(texts)
    items = []
    for (text, src, dt, tier), vec in zip(rows, vectors):
        items.append(
            EmbeddedChunk(
                chunk=Chunk(
                    text=text,
                    token_count=len(text.split()),
                    section_path="1",
                    citation=CitationSpan(source_uri=src, section_path="1"),
                    data_type=dt,
                    source=src,
                    tier=tier,
                ),
                vector=tuple(vec),
                embedder="inmem",
            )
        )
    await vstore.upsert(items)
    return RetrievalDeps(embedder=embedder, vector_store=vstore, graph_store=gstore)


# ---- ScopedRetrievalCapability ------------------------------------------


async def test_scoped_filters_by_source(populated_deps: RetrievalDeps) -> None:
    cap = ScopedRetrievalCapability(deps=populated_deps)
    result = await cap.retrieve(text="alpha", source_prefix="src:tool_a")
    assert result["mode"] == "scoped"
    sources = {h["chunk"]["source"] for h in result["hits"]}
    assert sources == {"src:tool_a"}


async def test_scoped_requires_source_prefix(populated_deps: RetrievalDeps) -> None:
    cap = ScopedRetrievalCapability(deps=populated_deps)
    result = await cap.retrieve(text="alpha")
    assert result["hits"] == []
    assert "source_prefix" in result["extra"]["reason"]


# ---- GroundedRetrievalCapability ----------------------------------------


async def test_grounded_drops_uncited(populated_deps: RetrievalDeps) -> None:
    bad_chunk = Chunk(
        text="uncited content",
        token_count=2,
        section_path="1",
        citation=CitationSpan(source_uri="", section_path="1"),
        data_type="paper_section",
        source="book:wesson",
        tier=CorpusTier.TIER_1_FOUNDATIONS,
    )
    bad = EmbeddedChunk(
        chunk=bad_chunk,
        vector=(await populated_deps.embedder.embed(["uncited content"]))[0],
        embedder="inmem",
    )
    await populated_deps.vector_store.upsert([bad])

    cap = GroundedRetrievalCapability(deps=populated_deps)
    result = await cap.retrieve(text="physics")
    for hit in result["hits"]:
        assert hit["chunk"]["citation"]["source_uri"]


async def test_grounded_restricts_to_tiers_1_to_3(
    populated_deps: RetrievalDeps,
) -> None:
    cap = GroundedRetrievalCapability(deps=populated_deps)
    result = await cap.retrieve(text="alpha")
    tiers = {h["chunk"]["tier"] for h in result["hits"]}
    assert "tier_4_software_docs" not in tiers


# ---- GraphRetrievalCapability -------------------------------------------


async def test_graph_query_via_text_seed(populated_deps: RetrievalDeps) -> None:
    g = populated_deps.graph_store
    await g.add_claim(
        Claim(
            subject="JET", predicate="requires", object="DT fuel",
            citation=CitationSpan(source_uri="book:x", section_path="1"),
        ),
    )
    await g.add_claim(
        Claim(
            subject="DT fuel", predicate="contains", object="tritium",
            citation=CitationSpan(source_uri="book:x", section_path="1"),
        ),
    )
    cap = GraphRetrievalCapability(deps=populated_deps)
    result = await cap.retrieve(text="JET", extra={"depth": 2})
    node_ids = set(result["extra"]["node_ids"])
    assert {"jet", "dt_fuel", "tritium"} <= node_ids


async def test_graph_query_dsl(populated_deps: RetrievalDeps) -> None:
    g = populated_deps.graph_store
    await g.add_claim(
        Claim(
            subject="A", predicate="is_a", object="thing",
            citation=CitationSpan(source_uri="x", section_path="1"),
        ),
    )
    cap = GraphRetrievalCapability(deps=populated_deps)
    result = await cap.retrieve(graph_query="MATCH (s)-[r:is_a]->(o)")
    assert any(e["predicate"] == "is_a" for e in result["extra"]["edges"])


async def test_graph_no_store_returns_empty() -> None:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore, graph_store=None)
    cap = GraphRetrievalCapability(deps=deps)
    result = await cap.retrieve(text="x")
    assert result["extra"]["reason"]


# ---- BudgetedRetrievalCapability ----------------------------------------


async def test_budgeted_truncates_at_token_budget() -> None:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    items = []
    texts = [f"chunk number {i}" for i in range(10)]
    vectors = await embedder.embed(texts)
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        items.append(
            EmbeddedChunk(
                chunk=Chunk(
                    text=text, token_count=100, section_path=str(i),
                    citation=CitationSpan(source_uri="src:x", section_path=str(i)),
                    data_type="paper_section", source="src:x",
                ),
                vector=tuple(vec), embedder="inmem",
            )
        )
    await vstore.upsert(items)
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore)
    cap = BudgetedRetrievalCapability(deps=deps)
    result = await cap.retrieve(
        text="chunk number 0",
        max_tokens=250,
        extra={"max_results": 50},
    )
    # Each chunk is 100 tokens; budget 250 fits at most 2.
    assert len(result["hits"]) <= 2
    assert result["used_tokens"] <= 250 + 100  # +100 = single-oversized-hit slack


# ---- StandardsRetrievalCapability ---------------------------------------


async def test_standards_filters_by_effective_at() -> None:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    early = datetime(2020, 1, 1, tzinfo=timezone.utc)
    later = datetime(2024, 1, 1, tzinfo=timezone.utc)
    items = []
    for i, eff in enumerate([early, later, None]):
        text = f"Clause {i}"
        vec = (await embedder.embed([text]))[0]
        items.append(
            EmbeddedChunk(
                chunk=Chunk(
                    text=text, token_count=4, section_path=f"3.{i}",
                    citation=CitationSpan(source_uri="semi:E123", section_path=f"3.{i}"),
                    data_type="standard_clause", source="semi:E123",
                    tier=CorpusTier.TIER_2_STANDARDS,
                    effective_at=eff,
                ),
                vector=tuple(vec), embedder="inmem",
            )
        )
    await vstore.upsert(items)
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore)
    cap = StandardsRetrievalCapability(deps=deps)

    as_of = datetime(2022, 6, 1, tzinfo=timezone.utc)
    result = await cap.retrieve(
        text="Clause",
        effective_at=as_of.isoformat(),
        extra={"max_results": 10},
    )
    sections = {h["chunk"]["section_path"] for h in result["hits"]}
    assert "3.1" not in sections  # Clause effective 2024 — out of force as of 2022
    assert "3.0" in sections


async def test_standards_supersedes_within_clause() -> None:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    older = datetime(2020, 1, 1, tzinfo=timezone.utc)
    newer = datetime(2023, 1, 1, tzinfo=timezone.utc)
    text = "Clause 3.4"
    vec = (await embedder.embed([text]))[0]
    items = [
        EmbeddedChunk(
            chunk=Chunk(
                text=text, token_count=4, section_path="3.4",
                citation=CitationSpan(source_uri="semi:E123", section_path="3.4"),
                data_type="standard_clause", source="semi:E123",
                tier=CorpusTier.TIER_2_STANDARDS,
                effective_at=older,
            ),
            vector=tuple(vec), embedder="inmem",
        ),
        EmbeddedChunk(
            chunk=Chunk(
                text=text + " — revised", token_count=5, section_path="3.4",
                citation=CitationSpan(source_uri="semi:E123", section_path="3.4"),
                data_type="standard_clause", source="semi:E123",
                tier=CorpusTier.TIER_2_STANDARDS,
                effective_at=newer,
            ),
            vector=tuple(vec), embedder="inmem",
        ),
    ]
    await vstore.upsert(items)
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore)
    cap = StandardsRetrievalCapability(deps=deps)

    result = await cap.retrieve(
        text="Clause",
        effective_at=datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat(),
    )
    assert len(result["hits"]) == 1
    assert "revised" in result["hits"][0]["chunk"]["text"]


# ---- Capability shape sanity ---------------------------------------------


async def test_all_five_expose_tool_and_retrieval_tags() -> None:
    """Every retrieval capability is a tool whose tags include the
    canonical ``"tool"`` + ``"retrieval"`` keys so the LLM planner can
    enumerate them via ``include_tags={"tool"}`` / ``{"retrieval"}``.
    """
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore)
    expected = {
        ScopedRetrievalCapability: "retrieve_scoped",
        GroundedRetrievalCapability: "retrieve_grounded",
        GraphRetrievalCapability: "retrieve_graph",
        BudgetedRetrievalCapability: "retrieve_budgeted",
        StandardsRetrievalCapability: "retrieve_standards",
    }
    for cls, capability_name in expected.items():
        cap = cls(deps=deps)
        tags = cap.get_capability_tags()
        assert "tool" in tags
        assert "knowledge" in tags
        assert "retrieval" in tags
        assert cap.spec.name == capability_name
        assert capability_name in cap.spec.capabilities

