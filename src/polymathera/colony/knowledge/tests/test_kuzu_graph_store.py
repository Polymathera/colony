"""Tests for the real ``KuzuGraphStore`` (embedded Kùzu).

These tests exercise the *real* Kùzu library against a tmpdir
database — no Docker, no mocks. They verify behavioural parity with
``InMemoryGraphStore`` so callers can swap backends.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polymathera.colony.knowledge import (
    CitationSpan,
    Claim,
    GraphEdge,
    GraphNode,
    GraphStoreError,
    KuzuGraphStore,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def store(tmp_path: Path) -> KuzuGraphStore:
    db_path = tmp_path / "graph.kuzu"
    s = KuzuGraphStore.open(db_path)
    yield s
    s.close()


# ---- Schema + open --------------------------------------------------------


async def test_open_creates_db(tmp_path: Path) -> None:
    db = tmp_path / "x.kuzu"
    s = KuzuGraphStore.open(db)
    assert db.exists()
    s.close()


async def test_open_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "x.kuzu"
    s1 = KuzuGraphStore.open(db)
    s1.close()
    # Re-opening must not raise.
    s2 = KuzuGraphStore.open(db)
    s2.close()


# ---- Mutation -------------------------------------------------------------


async def test_add_node_and_count(store: KuzuGraphStore) -> None:
    await store.add_node(GraphNode(node_id="a", label="Entity"))
    nodes, edges = await store.count()
    assert (nodes, edges) == (1, 0)


async def test_add_node_merges_properties(store: KuzuGraphStore) -> None:
    await store.add_node(GraphNode(node_id="a", properties={"k1": "v1"}))
    await store.add_node(GraphNode(node_id="a", properties={"k2": "v2"}))
    n = await store.get_node("a")
    assert n is not None
    assert n.properties == {"k1": "v1", "k2": "v2"}


async def test_add_edge_requires_both_nodes(store: KuzuGraphStore) -> None:
    await store.add_node(GraphNode(node_id="a"))
    with pytest.raises(GraphStoreError):
        await store.add_edge(
            GraphEdge(
                edge_id="e", source_id="a", target_id="missing", predicate="r",
            ),
        )


async def test_add_edge_idempotent(store: KuzuGraphStore) -> None:
    await store.add_node(GraphNode(node_id="a"))
    await store.add_node(GraphNode(node_id="b"))
    edge = GraphEdge(
        edge_id="e1", source_id="a", target_id="b", predicate="r",
    )
    await store.add_edge(edge)
    await store.add_edge(edge)  # second call must be a no-op
    _, edges = await store.count()
    assert edges == 1


async def test_add_claim_round_trip(store: KuzuGraphStore) -> None:
    claim = Claim(
        subject="JET",
        predicate="requires",
        object="DT fuel",
        citation=CitationSpan(source_uri="book:wesson", section_path="1"),
    )
    s, o, e = await store.add_claim(claim)
    assert s.node_id == "jet"
    assert o.node_id == "dt_fuel"
    assert e.predicate == "requires"
    nodes, edges = await store.count()
    assert (nodes, edges) == (2, 1)


# ---- Queries --------------------------------------------------------------


async def test_get_node_returns_none_for_unknown(
    store: KuzuGraphStore,
) -> None:
    n = await store.get_node("ghost")
    assert n is None


async def test_neighbours_walks_to_depth(store: KuzuGraphStore) -> None:
    citation = CitationSpan(source_uri="x", section_path="1")
    await store.add_claim(
        Claim(subject="A", predicate="links", object="B", citation=citation),
    )
    await store.add_claim(
        Claim(subject="B", predicate="links", object="C", citation=citation),
    )
    res = await store.neighbours("a", depth=2)
    ids = sorted(n.node_id for n in res.nodes)
    assert ids == ["a", "b", "c"]


async def test_neighbours_predicate_filter(store: KuzuGraphStore) -> None:
    citation = CitationSpan(source_uri="x", section_path="1")
    await store.add_claim(
        Claim(subject="A", predicate="cites", object="B", citation=citation),
    )
    await store.add_claim(
        Claim(subject="A", predicate="contradicts", object="C", citation=citation),
    )
    res = await store.neighbours("a", predicate="cites", depth=1)
    ids = sorted(n.node_id for n in res.nodes)
    assert ids == ["a", "b"]


async def test_query_dsl_filters_by_predicate(store: KuzuGraphStore) -> None:
    citation = CitationSpan(source_uri="x", section_path="1")
    await store.add_claim(
        Claim(subject="X", predicate="is_a", object="thing", citation=citation),
    )
    await store.add_claim(
        Claim(subject="Y", predicate="is_a", object="thing", citation=citation),
    )
    await store.add_claim(
        Claim(subject="Z", predicate="requires", object="other", citation=citation),
    )
    res = await store.query("MATCH (s)-[r:is_a]->(o)")
    edges = sorted(e.source_id for e in res.edges)
    assert edges == ["x", "y"]


async def test_query_dsl_limit(store: KuzuGraphStore) -> None:
    citation = CitationSpan(source_uri="x", section_path="1")
    for i in range(5):
        await store.add_claim(
            Claim(
                subject=f"S{i}", predicate="rel", object="T",
                citation=citation,
            ),
        )
    res = await store.query("MATCH (s)-[r:rel]->(o) LIMIT 2")
    assert len(res.edges) == 2


async def test_query_invalid_dsl_raises(store: KuzuGraphStore) -> None:
    with pytest.raises(GraphStoreError):
        await store.query("DROP TABLE Entity")


# ---- Persistence ----------------------------------------------------------


async def test_data_persists_across_open(tmp_path: Path) -> None:
    db = tmp_path / "persist.kuzu"
    s = KuzuGraphStore.open(db)
    await s.add_claim(
        Claim(
            subject="A", predicate="rel", object="B",
            citation=CitationSpan(source_uri="x", section_path="1"),
        ),
    )
    s.close()
    s2 = KuzuGraphStore.open(db)
    nodes, edges = await s2.count()
    assert (nodes, edges) == (2, 1)
    s2.close()
