"""Tests for the real ``KuzuGraphStore`` (embedded Kùzu).

These tests exercise the *real* Kùzu library against a tmpdir
database — no Docker, no mocks. They verify behavioural parity with
``InMemoryGraphStore`` so callers can swap backends.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# kuzu is in the ``knowledge`` extra; this whole file tests the real
# Kùzu backend, so without that extra installed (slim ``pip install
# polymathera-colony`` envs, dev installs without ``[knowledge]``)
# none of the tests can meaningfully run. CI's matrix job installs
# the ``knowledge`` extra; this guard is the fallback for any other
# environment.
pytest.importorskip("kuzu")

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


async def test_add_claims_batch_round_trip(store: KuzuGraphStore) -> None:
    """Fix D contract: ``add_claims`` ingests N claims under a single
    ``asyncio.to_thread`` hop + a single outer lock acquisition. The
    result list is one entry per input claim, in order, and the same
    nodes/edges land in the store as if the equivalent N
    ``add_claim`` calls had been issued."""

    citation = CitationSpan(source_uri="book:wesson", section_path="1")
    claims = [
        Claim(subject="JET", predicate="requires", object="DT fuel",
              citation=citation),
        Claim(subject="DT fuel", predicate="ignites_at",
              object="100M Kelvin", citation=citation),
        Claim(subject="JET", predicate="located_in",
              object="Culham", citation=citation),
    ]
    rows = await store.add_claims(claims)
    assert len(rows) == 3
    # Order preserved.
    assert rows[0][0].node_id == "jet"  # subject of first claim
    assert rows[1][0].node_id == "dt_fuel"
    assert rows[2][0].node_id == "jet"
    # All nodes + edges land in the store.
    nodes, edges = await store.count()
    # 4 distinct subjects/objects: jet, dt_fuel, 100m_kelvin, culham.
    assert nodes == 4
    assert edges == 3


async def test_add_claims_empty_input(store: KuzuGraphStore) -> None:
    """Empty input is a no-op — no thread hop, no lock acquisition,
    returns ``()``."""

    rows = await store.add_claims([])
    assert rows == ()
    nodes, edges = await store.count()
    assert (nodes, edges) == (0, 0)


async def test_add_claims_uses_single_to_thread_hop(
    store: KuzuGraphStore, monkeypatch,
) -> None:
    """Performance contract: a 50-claim batch dispatches ONE
    ``asyncio.to_thread`` call, not N. The win the Fix D plan
    promised on the ingest path is amortising thread-pool round
    trips across the batch — pin it so a refactor that puts
    ``to_thread`` back inside the per-claim path triggers a
    regression here."""

    import asyncio as _asyncio
    from polymathera.colony.knowledge.stores import graph as _graph_mod

    hops = {"n": 0}
    real_to_thread = _asyncio.to_thread

    async def _counting_to_thread(func, *args, **kwargs):
        hops["n"] += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(_graph_mod.asyncio, "to_thread", _counting_to_thread)

    citation = CitationSpan(source_uri="x", section_path="1")
    claims = [
        Claim(subject=f"S{i}", predicate="links", object=f"O{i}",
              citation=citation)
        for i in range(50)
    ]
    rows = await store.add_claims(claims)
    assert len(rows) == 50
    # Exactly ONE to_thread hop for the whole batch (the
    # ``add_claims`` -> ``_add_claims_sync`` dispatch). Anything
    # higher means a refactor reintroduced per-claim hops.
    assert hops["n"] == 1, (
        f"Expected 1 asyncio.to_thread hop for the batch; got "
        f"{hops['n']}. A refactor likely pushed the hop back into "
        f"the per-claim path."
    )


async def test_add_claims_per_claim_failure_does_not_poison_batch(
    store: KuzuGraphStore, monkeypatch,
) -> None:
    """Recovery contract: when one claim's insertion raises, the
    failing slot is ``None`` in the returned tuple and the surviving
    claims still land. Matches the ingestor's prior per-claim
    try/except recovery semantics — now amortised under one lock."""

    citation = CitationSpan(source_uri="x", section_path="1")
    claims = [
        Claim(subject="A", predicate="links", object="B", citation=citation),
        Claim(subject="POISON", predicate="links", object="C",
              citation=citation),
        Claim(subject="D", predicate="links", object="E", citation=citation),
    ]

    real_add_claim_locked = store._add_claim_locked

    def _fragile(claim, tag):  # noqa: ANN001
        if claim.subject == "POISON":
            raise RuntimeError("synthetic insertion failure")
        return real_add_claim_locked(claim, tag)

    monkeypatch.setattr(store, "_add_claim_locked", _fragile)
    rows = await store.add_claims(claims)

    assert len(rows) == 3
    assert rows[0] is not None and rows[0][0].node_id == "a"
    assert rows[1] is None  # failed slot
    assert rows[2] is not None and rows[2][0].node_id == "d"
    # The surviving claims landed in the store; the failing one did
    # not.
    nodes, edges = await store.count()
    assert nodes == 4  # a, b, d, e
    assert edges == 2  # a-links->b, d-links->e


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
