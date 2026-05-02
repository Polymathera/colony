"""Tests for ``ConvergenceRuntime``.

The runtime now persists state in ``VirtualPageTableState.convergence``
(a colony-wide ``SharedState``); the ``convergence_runtime`` fixture
provides a runtime backed by an in-memory state manager so tests
don't need Redis.
"""

from __future__ import annotations

import pytest

from polymathera.colony.vcm.convergence import (
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
        capability_key=kwargs.pop("capability_key", "Cap"),
        tolerance=kwargs.pop("tolerance", None),
    )


async def _read_state(rt):
    sm = rt._state_manager
    async for s in sm.read_transaction():
        return s.model_copy(deep=True)
    raise AssertionError("read_transaction yielded nothing")


def _capture_dispatches(rt) -> list[tuple[PageSubscription, PageChangeEvent]]:
    captured: list[tuple[PageSubscription, PageChangeEvent]] = []

    async def cb(sub, ev):
        captured.append((sub, ev))

    rt._dispatch_via_blackboard = cb
    return captured


# ---------------------------------------------------------------------------
# Subscription lifecycle
# ---------------------------------------------------------------------------


async def test_subscription_lifecycle(convergence_runtime) -> None:
    rt = convergence_runtime
    sub = _sub(predicate=PageMetadataPredicate())
    sid = await rt.register(sub)
    assert await rt.subscription_count() == 1
    assert (await rt.get(sid)).capability_key == "Cap"
    assert await rt.unregister(sid) is True
    assert await rt.subscription_count() == 0
    # Subsequent events fire nothing.
    fired = _capture_dispatches(rt)
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert fired == []


# ---------------------------------------------------------------------------
# Dispatch + matching
# ---------------------------------------------------------------------------


async def test_dispatch_matching_subscription(convergence_runtime) -> None:
    rt = convergence_runtime
    a = await rt.register(_sub(predicate=PageMetadataPredicate(data_type="code")))
    await rt.register(_sub(predicate=PageMetadataPredicate(data_type="paper_section")))
    fired = _capture_dispatches(rt)
    await rt.feed_event(
        PageChangeEvent.page_replaced(
            old_page_id="p", new_page_id="p",
            source="git:r:main:1", data_type="code", scope_id="s",
        ),
    )
    assert [s.subscription_id for s, _ in fired] == [a]


async def test_quiescence_emit_called(convergence_runtime) -> None:
    """Each ``feed_event`` writes a quiescence event onto the colony
    blackboard via ``ConvergenceQuiescenceProtocol``."""

    rt = convergence_runtime
    await rt.register(_sub(predicate=PageMetadataPredicate()))
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    rt._colony_blackboard.write.assert_called_once()
    call = rt._colony_blackboard.write.call_args
    assert call.args[0].startswith("convergence:quiescence:")


async def test_status_state_transitions(convergence_runtime) -> None:
    in_flight_states: list[str] = []
    rt = convergence_runtime
    await rt.register(_sub(predicate=PageMetadataPredicate()))

    async def cb(sub, ev):
        # While dispatch is in flight on this replica, status is 'converging'.
        status = await rt.get_status()
        in_flight_states.append(status.state)

    rt._dispatch_via_blackboard = cb
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    # The dispatch happens after the CAS commit, so the in-flight
    # episode is still set when cb runs.
    assert in_flight_states == ["converging"]
    assert (await rt.get_status()).state == "converged"


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------


async def test_damping_skips_within_tolerance(convergence_runtime) -> None:
    rt = convergence_runtime
    rt._rate_burst = 10
    await rt.register(_sub(
        predicate=PageMetadataPredicate(data_type="budget"),
        tolerance=NumericTolerance(mode="absolute", value=0.5),
    ))
    fired = _capture_dispatches(rt)
    for v in (1.0, 1.2, 1.4, 2.0, 2.05):
        await rt.feed_event(
            PageChangeEvent.page_replaced(
                old_page_id="b", new_page_id="b",
                source="bb:budget", data_type="budget", scope_id="prog",
                extra={"value": v},
            ),
        )
    assert [ev.extra["value"] for _, ev in fired] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


async def test_rate_limit_drops_repeated_writes(convergence_runtime) -> None:
    rt = convergence_runtime
    rt._rate_interval_s = 10.0
    rt._rate_burst = 1
    await rt.register(_sub(predicate=PageMetadataPredicate()))
    fired = _capture_dispatches(rt)
    for page_id in ("hot", "hot", "cool"):
        await rt.feed_event(
            PageChangeEvent.page_replaced(
                old_page_id=page_id, new_page_id=page_id,
                source="x:src", data_type="t", scope_id="s",
            ),
        )
    assert [ev.page_id for _, ev in fired] == ["hot", "cool"]


# ---------------------------------------------------------------------------
# Change feed bookkeeping
# ---------------------------------------------------------------------------


async def test_change_feed_records_skips(convergence_runtime) -> None:
    rt = convergence_runtime
    rt._rate_interval_s = 10.0
    rt._rate_burst = 1
    await rt.register(_sub(predicate=PageMetadataPredicate()))
    for _ in range(3):
        await rt.feed_event(
            PageChangeEvent.page_replaced(
                old_page_id="hot", new_page_id="hot",
                source="x:src", data_type="t", scope_id="s",
            ),
        )
    feed = await rt.get_change_feed()
    reasons = [e.skipped_reason for e in feed]
    assert "rate_limited" in reasons


async def test_dispatch_callback_failures_do_not_break_others(
    convergence_runtime,
) -> None:
    rt = convergence_runtime
    bad = await rt.register(_sub(
        predicate=PageMetadataPredicate(), capability_key="Bad",
    ))
    good = await rt.register(_sub(
        predicate=PageMetadataPredicate(), capability_key="Good",
    ))

    fired_good: list[str] = []

    async def cb(sub, ev):
        if sub.subscription_id == bad:
            raise RuntimeError("boom")
        fired_good.append(sub.subscription_id)

    rt._dispatch_via_blackboard = cb
    await rt.feed_event(
        PageChangeEvent.page_added(
            page_id="p", source="x:src", data_type="t", scope_id="s",
        ),
    )
    assert good in fired_good
