"""Tests for the five retrieval modes (registered as ToolAdapters)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from polymathera.colony.knowledge import (
    BudgetedRetrievalAdapter,
    Chunk,
    CitationSpan,
    Claim,
    CorpusTier,
    EmbeddedChunk,
    GraphRetrievalAdapter,
    GroundedRetrievalAdapter,
    InMemoryEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    RetrievalDeps,
    RetrievalQuery,
    ScopedRetrievalAdapter,
    StandardsRetrievalAdapter,
)
from polymathera.colony.tools import ToolCall, ToolRegistry


pytestmark = pytest.mark.asyncio


def _embedded(
    *,
    text: str,
    source: str = "src:a",
    data_type: str = "paper_section",
    tier: CorpusTier = CorpusTier.UNTIERED,
    section_path: str = "1",
    effective_at: datetime | None = None,
    token_count: int = 50,
    citation_uri: str | None = None,
) -> EmbeddedChunk:
    chunk = Chunk(
        text=text,
        token_count=token_count,
        section_path=section_path,
        citation=CitationSpan(
            source_uri=citation_uri if citation_uri is not None else source,
            section_path=section_path,
        ),
        data_type=data_type,
        source=source,
        tier=tier,
        effective_at=effective_at,
    )
    embedder = InMemoryEmbedder()
    import asyncio

    vec = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        embedder.embed([text])
    )[0]
    return EmbeddedChunk(chunk=chunk, vector=tuple(vec), embedder="inmem")


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


# ---- ScopedRetrievalAdapter ----------------------------------------------


async def test_scoped_filters_by_source(populated_deps: RetrievalDeps) -> None:
    adapter = ScopedRetrievalAdapter(deps=populated_deps)
    call = ToolCall(
        capability="retrieve_scoped",
        parameters={"text": "alpha", "source_prefix": "src:tool_a"},
    )
    result = await adapter.invoke(call)
    assert result.success
    assert result.value["mode"] == "scoped"
    sources = {h["chunk"]["source"] for h in result.value["hits"]}
    assert sources == {"src:tool_a"}


async def test_scoped_requires_source_prefix(populated_deps: RetrievalDeps) -> None:
    adapter = ScopedRetrievalAdapter(deps=populated_deps)
    call = ToolCall(capability="retrieve_scoped", parameters={"text": "alpha"})
    result = await adapter.invoke(call)
    assert result.success
    assert result.value["hits"] == []
    assert "source_prefix" in result.value["extra"]["reason"]


# ---- GroundedRetrievalAdapter --------------------------------------------


async def test_grounded_drops_uncited(populated_deps: RetrievalDeps) -> None:
    # Insert one chunk with empty source_uri citation; the grounded
    # mode must drop it.
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

    adapter = GroundedRetrievalAdapter(deps=populated_deps)
    call = ToolCall(
        capability="retrieve_grounded",
        parameters={"text": "physics"},
    )
    result = await adapter.invoke(call)
    assert result.success
    for hit in result.value["hits"]:
        assert hit["chunk"]["citation"]["source_uri"]


async def test_grounded_restricts_to_tiers_1_to_3(
    populated_deps: RetrievalDeps,
) -> None:
    adapter = GroundedRetrievalAdapter(deps=populated_deps)
    call = ToolCall(
        capability="retrieve_grounded",
        parameters={"text": "alpha"},
    )
    result = await adapter.invoke(call)
    assert result.success
    tiers = {h["chunk"]["tier"] for h in result.value["hits"]}
    # Tier-4 entries must not surface.
    assert "tier_4_software_docs" not in tiers


# ---- GraphRetrievalAdapter -----------------------------------------------


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
    adapter = GraphRetrievalAdapter(deps=populated_deps)
    result = await adapter.invoke(
        ToolCall(capability="retrieve_graph", parameters={"text": "JET", "extra": {"depth": 2}}),
    )
    assert result.success
    node_ids = set(result.value["extra"]["node_ids"])
    assert {"jet", "dt_fuel", "tritium"} <= node_ids


async def test_graph_query_dsl(populated_deps: RetrievalDeps) -> None:
    g = populated_deps.graph_store
    await g.add_claim(
        Claim(
            subject="A", predicate="is_a", object="thing",
            citation=CitationSpan(source_uri="x", section_path="1"),
        ),
    )
    adapter = GraphRetrievalAdapter(deps=populated_deps)
    result = await adapter.invoke(
        ToolCall(
            capability="retrieve_graph",
            parameters={"graph_query": "MATCH (s)-[r:is_a]->(o)"},
        ),
    )
    assert result.success
    assert any(e["predicate"] == "is_a" for e in result.value["extra"]["edges"])


async def test_graph_no_store_returns_empty() -> None:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore, graph_store=None)
    adapter = GraphRetrievalAdapter(deps=deps)
    result = await adapter.invoke(
        ToolCall(capability="retrieve_graph", parameters={"text": "x"}),
    )
    assert result.success
    assert result.value["extra"]["reason"]


# ---- BudgetedRetrievalAdapter --------------------------------------------


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
    adapter = BudgetedRetrievalAdapter(deps=deps)
    result = await adapter.invoke(
        ToolCall(
            capability="retrieve_budgeted",
            parameters={"text": "chunk number 0", "max_tokens": 250, "max_results": 50},
        ),
    )
    assert result.success
    # Each chunk is 100 tokens; budget 250 fits at most 2.
    assert len(result.value["hits"]) <= 2
    assert result.value["used_tokens"] <= 250 + 100  # +100 = single-oversized-hit slack


# ---- StandardsRetrievalAdapter -------------------------------------------


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
    adapter = StandardsRetrievalAdapter(deps=deps)

    as_of = datetime(2022, 6, 1, tzinfo=timezone.utc)
    result = await adapter.invoke(
        ToolCall(
            capability="retrieve_standards",
            parameters={
                "text": "Clause",
                "effective_at": as_of.isoformat(),
                "max_results": 10,
            },
        ),
    )
    assert result.success
    sections = {h["chunk"]["section_path"] for h in result.value["hits"]}
    # Clause 1 (effective 2024) must be excluded; Clause 0 (2020) and
    # Clause 2 (no effective_at) survive.
    assert "3.1" not in sections
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
    adapter = StandardsRetrievalAdapter(deps=deps)

    # As of 2024 — only the newer revision survives.
    result = await adapter.invoke(
        ToolCall(
            capability="retrieve_standards",
            parameters={
                "text": "Clause",
                "effective_at": datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat(),
            },
        ),
    )
    assert result.success
    assert len(result.value["hits"]) == 1
    assert "revised" in result.value["hits"][0]["chunk"]["text"]


# ---- Registration sanity check -------------------------------------------


async def test_all_five_register_under_distinct_capabilities() -> None:
    embedder = InMemoryEmbedder()
    vstore = InMemoryVectorStore()
    gstore = InMemoryGraphStore()
    deps = RetrievalDeps(embedder=embedder, vector_store=vstore, graph_store=gstore)
    reg = ToolRegistry()
    for cls in (
        ScopedRetrievalAdapter,
        GroundedRetrievalAdapter,
        GraphRetrievalAdapter,
        BudgetedRetrievalAdapter,
        StandardsRetrievalAdapter,
    ):
        reg.register(cls(deps=deps))
    caps = set(reg.list_capabilities())
    assert {
        "retrieve_scoped",
        "retrieve_grounded",
        "retrieve_graph",
        "retrieve_budgeted",
        "retrieve_standards",
    } <= caps
