"""Tests for ``ConvergenceRuntime``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from polymathera.colony.vcm.convergence import (
    ConvergenceRuntime,
    NumericTolerance,
    PageMetadataPredicate,
    PageSubscription,
)
from polymathera.colony.vcm.page_events import PageChangeEvent


pytestmark = pytest.mark.asyncio


def _sub(**kwargs) -> PageSubscription:
    return PageSubscription(
        predicate=kwargs.pop("predicate"),
        dispatch_scope=kwargs.pop("dispatch_scope", "scope"),
        dispatch_key=kwargs.pop("dispatch_key", "k"),
        capability_key=kwargs.pop("capability_key", "Cap"),
        declared_outputs=tuple(kwargs.pop("declared_outputs", ())),
        tolerance=kwargs.pop("tolerance", None),
    )


async def test_dispatch_matching_subscription() -> None:
    fired: list[str] = []

    async def cb(sub, ev):
        fired.append(sub.subscription_id)

    rt = ConvergenceRuntime(dispatch_callback=cb)
    a = _sub(predicate=PageMetadataPredicate(data_type="code"))
    b = _sub(predicate=PageMetadataPredicate(data_type="paper_section"))
    rt.register(a)
    rt.register(b)
    await rt.feed_event(
        PageChangeEvent.page_replaced(
            old_page_id="p", new_page_id="p",
            source="git:r:main:1", data_type="code", scope_id="s",
        ),
    )
    assert fired == [a.subscription_id]


async def test_quiescence_emitted() -> None:
    quiescent: list[Any] = []

    async def cb(sub, ev):
        return None

    async def q_emit(counters):
        quiescent.append(counters)

    rt = ConvergenceRuntime(
        dispatch_callback=cb, quiescence_emit_callback=q_emit,
    )
    rt.register(_sub(predicate=PageMetadataPredicate()))
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert len(quiescent) == 1
    assert quiescent[0].dispatches_emitted == 1


async def test_status_state_transitions() -> None:
    in_flight_states = []

    async def cb(sub, ev):
        # While dispatch is in flight, status is 'converging'.
        in_flight_states.append(rt.get_status().state)

    rt = ConvergenceRuntime(dispatch_callback=cb)
    rt.register(_sub(predicate=PageMetadataPredicate()))
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert in_flight_states == ["converging"]
    assert rt.get_status().state == "converged"


async def test_damping_skips_within_tolerance() -> None:
    fired: list[float] = []

    async def cb(sub, ev):
        fired.append(ev.extra["value"])

    rt = ConvergenceRuntime(dispatch_callback=cb, rate_burst=10)
    sub = _sub(
        predicate=PageMetadataPredicate(data_type="budget"),
        tolerance=NumericTolerance(mode="absolute", value=0.5),
    )
    rt.register(sub)
    for v in (1.0, 1.2, 1.4, 2.0, 2.05):
        await rt.feed_event(
            PageChangeEvent.page_replaced(
                old_page_id="b", new_page_id="b",
                source="bb:budget", data_type="budget", scope_id="prog",
                extra={"value": v},
            ),
        )
    assert fired == [1.0, 2.0]


async def test_rate_limit_drops_repeated_writes() -> None:
    fired: list[str] = []

    async def cb(sub, ev):
        fired.append(ev.page_id)

    rt = ConvergenceRuntime(dispatch_callback=cb, rate_interval_s=10.0, rate_burst=1)
    rt.register(_sub(predicate=PageMetadataPredicate()))
    await rt.feed_event(
        PageChangeEvent.page_replaced(
            old_page_id="hot", new_page_id="hot",
            source="x:src", data_type="t", scope_id="s",
        ),
    )
    # Second write to the same page within the window — rate limited.
    await rt.feed_event(
        PageChangeEvent.page_replaced(
            old_page_id="hot", new_page_id="hot",
            source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert fired == ["hot"]
    # Different page — passes.
    await rt.feed_event(
        PageChangeEvent.page_replaced(
            old_page_id="cool", new_page_id="cool",
            source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert fired == ["hot", "cool"]


async def test_topo_sort_orders_by_declared_outputs() -> None:
    order: list[str] = []

    async def cb(sub, ev):
        order.append(sub.capability_key)

    rt = ConvergenceRuntime(dispatch_callback=cb)
    # Producer writes evidence; consumer reads evidence.
    producer = _sub(
        predicate=PageMetadataPredicate(data_type="requirements"),
        declared_outputs=[PageMetadataPredicate(data_type="evidence")],
        capability_key="Producer",
    )
    consumer = _sub(
        predicate=PageMetadataPredicate(data_type="evidence"),
        capability_key="Consumer",
    )
    # Both subscriptions match an event with data_type='evidence' through
    # the consumer's read predicate, plus a data_type='requirements'
    # event would only match producer. To test topo order we feed an
    # event matching both: data_type='evidence' should fire only consumer
    # (no producer match), so topo doesn't reorder. Instead, configure
    # both predicates to match a common event:
    producer = producer.model_copy(update={
        "predicate": PageMetadataPredicate(data_type="evidence"),
    })
    rt.register(producer)
    rt.register(consumer)
    await rt.feed_event(
        PageChangeEvent.page_replaced(
            old_page_id="e", new_page_id="e",
            source="x:src", data_type="evidence", scope_id="prog",
        ),
    )
    # Producer (writes evidence) runs before Consumer (reads evidence).
    assert order == ["Producer", "Consumer"]


async def test_subscription_lifecycle() -> None:
    fired: list[str] = []

    async def cb(sub, ev):
        fired.append(sub.subscription_id)

    rt = ConvergenceRuntime(dispatch_callback=cb)
    sub = _sub(predicate=PageMetadataPredicate())
    sid = rt.register(sub)
    assert rt.subscription_count == 1
    assert rt.unregister(sid) is True
    assert rt.subscription_count == 0
    # Subsequent events fire nothing.
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert fired == []


async def test_change_feed_records_skips() -> None:
    async def cb(sub, ev):
        return None

    rt = ConvergenceRuntime(dispatch_callback=cb, rate_interval_s=10.0, rate_burst=1)
    rt.register(_sub(predicate=PageMetadataPredicate()))
    for _ in range(3):
        await rt.feed_event(
            PageChangeEvent.page_replaced(
                old_page_id="hot", new_page_id="hot",
                source="x:src", data_type="t", scope_id="s",
            ),
        )
    feed = rt.get_change_feed()
    reasons = [e.skipped_reason for e in feed]
    # First event dispatched (no skipped_reason); the rest are
    # rate-limited (which records a 'rate_limited' entry without a
    # matching subscription).
    assert "rate_limited" in reasons


async def test_wait_for_quiescence_immediate_when_idle() -> None:
    async def cb(sub, ev):
        return None

    rt = ConvergenceRuntime(dispatch_callback=cb)
    assert await rt.wait_for_quiescence(timeout=0.1) is True


async def test_dispatch_callback_failures_do_not_break_others() -> None:
    fired: list[str] = []

    async def cb(sub, ev):
        if sub.capability_key == "Bad":
            raise RuntimeError("boom")
        fired.append(sub.subscription_id)

    rt = ConvergenceRuntime(dispatch_callback=cb)
    bad = _sub(predicate=PageMetadataPredicate(), capability_key="Bad")
    good = _sub(predicate=PageMetadataPredicate(), capability_key="Good")
    rt.register(bad)
    rt.register(good)
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert good.subscription_id in fired
