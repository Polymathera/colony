"""Branch-annotation contract on the GraphStore ABC: enforced
contextvar, per-branch filtering, idempotent re-tagging via
:meth:`import_claims`."""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.models import Claim, CitationSpan
from polymathera.colony.knowledge.stores.graph import (
    CURRENT_BRANCH_CONTEXT,
    GraphEdge,
    GraphNode,
    GraphStoreError,
    InMemoryGraphStore,
    KuzuGraphStore,
    set_current_branch,
)


def _claim(subject: str, obj: str) -> Claim:
    return Claim(
        subject=subject, predicate="links", object=obj, confidence=0.9,
        citation=CitationSpan(source_uri="lit:1"),
        provenance={"extractor": "deterministic@v1"},
    )


@pytest.mark.asyncio
async def test_add_claim_without_branch_context_raises() -> None:
    store = InMemoryGraphStore()
    token = CURRENT_BRANCH_CONTEXT.set(None)
    try:
        with pytest.raises(GraphStoreError, match="requires a branch"):
            await store.add_claim(_claim("a", "b"))
    finally:
        CURRENT_BRANCH_CONTEXT.reset(token)


@pytest.mark.asyncio
async def test_add_claim_picks_up_contextvar() -> None:
    store = InMemoryGraphStore()
    with set_current_branch("feature/x"):
        s, o, e = await store.add_claim(_claim("a", "b"))
    assert "feature/x" in s.branches
    assert "feature/x" in o.branches
    assert "feature/x" in e.branches


@pytest.mark.asyncio
async def test_add_claim_explicit_branch_overrides_contextvar() -> None:
    store = InMemoryGraphStore()
    with set_current_branch("main"):
        _, _, edge = await store.add_claim(_claim("a", "b"), branch="dev")
    assert edge.branches == frozenset({"dev"})


@pytest.mark.asyncio
async def test_query_branch_filter_returns_only_matching_inmemory() -> None:
    store = InMemoryGraphStore()
    with set_current_branch("main"):
        await store.add_claim(_claim("a", "b"))
    with set_current_branch("dev"):
        await store.add_claim(_claim("c", "d"))
    main_only = await store.query("MATCH (s)-[r]->(o)", branch_filter="main")
    assert {e.source_id for e in main_only.edges} == {"a"}
    union = await store.query("MATCH (s)-[r]->(o)")
    assert {e.source_id for e in union.edges} == {"a", "c"}


@pytest.mark.asyncio
async def test_import_claims_added_then_tagged_then_skipped_inmemory() -> None:
    store = InMemoryGraphStore()
    claims = [_claim("a", "b"), _claim("c", "d")]
    r1 = await store.import_claims(claims, branch="main")
    assert (r1.added, r1.tagged, r1.skipped) == (2, 0, 0)
    r2 = await store.import_claims(claims, branch="main")
    assert (r2.added, r2.tagged, r2.skipped) == (0, 0, 2)
    r3 = await store.import_claims(claims, branch="dev")
    assert (r3.added, r3.tagged, r3.skipped) == (0, 2, 0)


@pytest.mark.asyncio
async def test_import_claims_rejects_empty_branch() -> None:
    store = InMemoryGraphStore()
    with pytest.raises(GraphStoreError):
        await store.import_claims([_claim("a", "b")], branch="")


@pytest.mark.asyncio
async def test_count_branch_filter_matches_filtered_query(tmp_path) -> None:
    store = KuzuGraphStore.open(str(tmp_path / "kg.db"))
    try:
        with set_current_branch("main"):
            await store.add_claim(_claim("a", "b"))
        with set_current_branch("dev"):
            await store.add_claim(_claim("c", "d"))
        n_all, e_all = await store.count()
        n_main, e_main = await store.count(branch_filter="main")
        n_dev, e_dev = await store.count(branch_filter="dev")
        assert (n_all, e_all) == (4, 2)
        assert (n_main, e_main) == (2, 1)
        assert (n_dev, e_dev) == (2, 1)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_get_node_with_branch_filter_returns_none_for_unmatched(
    tmp_path,
) -> None:
    store = KuzuGraphStore.open(str(tmp_path / "kg.db"))
    try:
        with set_current_branch("main"):
            await store.add_node(GraphNode(node_id="x", label="Entity"))
        assert await store.get_node("x", branch_filter="main") is not None
        assert await store.get_node("x", branch_filter="dev") is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_export_claims_filters_by_branch_kuzu(tmp_path) -> None:
    store = KuzuGraphStore.open(str(tmp_path / "kg.db"))
    try:
        with set_current_branch("main"):
            await store.add_claim(_claim("a", "b"))
        with set_current_branch("dev"):
            await store.add_claim(_claim("c", "d"))
        main_claims = [c async for c in store.export_claims(branch="main")]
        dev_claims = [c async for c in store.export_claims(branch="dev")]
        union_claims = [c async for c in store.export_claims()]
        assert [c.subject for c in main_claims] == ["a"]
        assert [c.subject for c in dev_claims] == ["c"]
        assert len(union_claims) == 2
    finally:
        store.close()
