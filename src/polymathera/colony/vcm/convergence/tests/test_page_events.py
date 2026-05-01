"""Tests for ``PageChangeEvent`` and the topic-key convention."""

from __future__ import annotations

from polymathera.colony.vcm.page_events import (
    PAGE_EVENTS_TOPIC_PREFIX,
    PageChangeEvent,
    PageChangeKind,
)


def test_page_invalidated_carries_reason() -> None:
    e = PageChangeEvent.page_invalidated(
        page_id="p1", source="git:repo:main:abc", reason="deleted",
    )
    assert e.kind is PageChangeKind.PAGE_INVALIDATED
    assert e.page_id == "p1"
    assert e.reason == "deleted"


def test_page_replaced_relates_old_id() -> None:
    e = PageChangeEvent.page_replaced(
        old_page_id="p1", new_page_id="p2",
        source="git:repo:main:def",
    )
    assert e.kind is PageChangeKind.PAGE_REPLACED
    assert e.page_id == "p2"
    assert e.related_page_ids == ("p1",)


def test_edge_added() -> None:
    e = PageChangeEvent.page_graph_edge_added(
        source_page_id="a", target_page_id="b",
        edge_type="cites", source="paper:doi",
    )
    assert e.kind is PageChangeKind.PAGE_GRAPH_EDGE_ADDED
    assert e.page_id == "a"
    assert e.related_page_ids == ("b",)
    assert e.edge_type == "cites"


def test_topic_key() -> None:
    e = PageChangeEvent.page_added(
        page_id="p", source="git:repo:main:1",
    )
    key = e.topic_key("design_monorepo:program-1")
    assert key.startswith(f"{PAGE_EVENTS_TOPIC_PREFIX}:")
    assert "design_monorepo:program-1" in key
    assert "page_added" in key


def test_round_trip_via_pydantic() -> None:
    e = PageChangeEvent.page_replaced(
        old_page_id="o", new_page_id="n", source="git:r:main:s",
        edit_diff="+1 -1", data_type="code", scope_id="prog",
    )
    payload = e.model_dump(mode="json")
    e2 = PageChangeEvent.model_validate(payload)
    assert e2 == e
