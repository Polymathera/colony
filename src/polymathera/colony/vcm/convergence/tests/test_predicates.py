"""Tests for ``PageMetadataPredicate`` matching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from polymathera.colony.vcm.convergence import PageMetadataPredicate
from polymathera.colony.vcm.page_events import PageChangeEvent


def _event(**kwargs) -> PageChangeEvent:
    return PageChangeEvent.page_replaced(
        old_page_id=kwargs.pop("old_page_id", "old"),
        new_page_id=kwargs.pop("page_id", "new"),
        source=kwargs.pop("source", "git:repo:main:abc"),
        data_type=kwargs.pop("data_type", "code"),
        scope_id=kwargs.pop("scope_id", "prog"),
        extra=kwargs.pop("extra", None) or {},
    )


def test_default_predicate_matches_anything() -> None:
    assert PageMetadataPredicate().matches(_event())


def test_data_type_exact_match() -> None:
    p = PageMetadataPredicate(data_type="requirements")
    assert not p.matches(_event(data_type="code"))
    assert p.matches(_event(data_type="requirements"))


def test_source_prefix_match() -> None:
    p = PageMetadataPredicate(source_prefix="git:")
    assert p.matches(_event(source="git:remote:main:1"))
    assert not p.matches(_event(source="arxiv:2410.12345:v1"))


def test_scope_filter() -> None:
    p = PageMetadataPredicate(scope_id="program-A")
    assert not p.matches(_event(scope_id="program-B"))
    assert p.matches(_event(scope_id="program-A"))


def test_page_id_in_filter() -> None:
    p = PageMetadataPredicate(page_id_in=("p1", "p2"))
    assert p.matches(_event(page_id="p1"))
    assert not p.matches(_event(page_id="p9"))


def test_effective_at_window() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    p = PageMetadataPredicate(
        effective_at_after=now - timedelta(days=1),
        effective_at_before=now + timedelta(days=1),
    )
    inside = _event(extra={"effective_at": now.isoformat()})
    before = _event(extra={"effective_at": (now - timedelta(days=10)).isoformat()})
    no_effective = _event()
    assert p.matches(inside)
    assert not p.matches(before)
    # Pages without an effective_at when the window is set are excluded.
    assert not p.matches(no_effective)


def test_edge_reach_requires_resolver() -> None:
    p = PageMetadataPredicate(
        edge_reach_root="root",
        edge_reach_max_hops=3,
    )
    # Without a resolver — conservative deny.
    assert not p.matches(_event(page_id="p"))

    def resolver(root, page_id, max_hops, edge_types):
        return root == "root" and page_id == "p" and max_hops >= 1

    assert p.matches(
        _event(page_id="p"),
        edge_reach_resolver=resolver,
    )
    assert not p.matches(
        _event(page_id="other"),
        edge_reach_resolver=resolver,
    )


def test_edge_reach_validation() -> None:
    # max_hops > 0 without a root is a config error.
    with pytest.raises(Exception):
        PageMetadataPredicate(edge_reach_max_hops=2)
    # root without max_hops is a config error.
    with pytest.raises(Exception):
        PageMetadataPredicate(edge_reach_root="r", edge_reach_max_hops=0)


def test_indexable_flag() -> None:
    assert not PageMetadataPredicate().is_indexable
    assert PageMetadataPredicate(data_type="x").is_indexable
    assert PageMetadataPredicate(source_prefix="git:").is_indexable
    # edge-reach predicates are not indexable.
    assert not PageMetadataPredicate(
        edge_reach_root="r", edge_reach_max_hops=1, data_type="x",
    ).is_indexable
