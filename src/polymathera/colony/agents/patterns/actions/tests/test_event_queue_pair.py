"""Tests for :class:`EventQueuePair` — the consume-none wait primitive
that backs ``wait_for_next_event``."""

from __future__ import annotations

import asyncio

import pytest

from polymathera.colony.agents.patterns.actions.event_queue_pair import (
    EventQueuePair,
    _WakingQueue,
)


pytestmark = pytest.mark.asyncio


async def test_put_normal_sets_nonempty_signal() -> None:
    """``put_nowait`` on the normal lane wakes a waiter."""
    pair = EventQueuePair()
    waiter = asyncio.create_task(pair.wait_nonempty())
    await asyncio.sleep(0)  # let the waiter run
    assert not waiter.done()
    pair.normal.put_nowait("event-a")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result is True
    # Consume-none: the event stays in the queue.
    assert pair.normal.qsize() == 1


async def test_put_high_sets_nonempty_signal() -> None:
    """``put_nowait`` on the high lane wakes a waiter."""
    pair = EventQueuePair()
    waiter = asyncio.create_task(pair.wait_nonempty())
    await asyncio.sleep(0)
    assert not waiter.done()
    pair.high.put_nowait("event-h")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result is True
    assert pair.high.qsize() == 1


async def test_wait_returns_immediately_when_normal_nonempty() -> None:
    """If the normal queue already has an event, the wait returns
    immediately without blocking — consume-none semantics."""
    pair = EventQueuePair()
    pair.normal.put_nowait("event-a")
    # No timeout needed: this MUST complete without waiting.
    result = await asyncio.wait_for(pair.wait_nonempty(), timeout=0.05)
    assert result is True
    assert pair.normal.qsize() == 1


async def test_wait_returns_immediately_when_high_nonempty() -> None:
    """Same for the high-priority lane."""
    pair = EventQueuePair()
    pair.high.put_nowait("event-h")
    result = await asyncio.wait_for(pair.wait_nonempty(), timeout=0.05)
    assert result is True
    assert pair.high.qsize() == 1


async def test_wait_blocks_until_put() -> None:
    """A wait task started against empty queues blocks until a put
    arrives on either lane."""
    pair = EventQueuePair()

    async def delayed_put() -> None:
        await asyncio.sleep(0.05)
        pair.normal.put_nowait("late-event")

    putter = asyncio.create_task(delayed_put())
    result = await asyncio.wait_for(pair.wait_nonempty(), timeout=1.0)
    await putter
    assert result is True


async def test_wait_times_out_returns_false() -> None:
    """The wait returns False on timeout when no event arrives."""
    pair = EventQueuePair()
    result = await pair.wait_nonempty(timeout=0.05)
    assert result is False
    # Queues stay empty after the timeout.
    assert pair.is_empty()


async def test_wait_cancellation_propagates() -> None:
    """Cancelling the waiting task raises ``CancelledError`` — the
    same primitive used by ``Agent.stop`` for ``/abort``."""
    pair = EventQueuePair()
    waiter = asyncio.create_task(pair.wait_nonempty())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


async def test_consume_none_event_remains_in_both_queues() -> None:
    """After ``wait_nonempty`` returns, BOTH queues retain whatever
    was in them — this primitive consumes nothing."""
    pair = EventQueuePair()
    pair.normal.put_nowait("n1")
    pair.normal.put_nowait("n2")
    pair.high.put_nowait("h1")
    result = await pair.wait_nonempty(timeout=0.05)
    assert result is True
    assert pair.normal.qsize() == 2
    assert pair.high.qsize() == 1


async def test_is_empty_reports_correctly() -> None:
    """``is_empty`` is True iff BOTH lanes are empty."""
    pair = EventQueuePair()
    assert pair.is_empty()
    pair.normal.put_nowait("n")
    assert not pair.is_empty()
    pair.normal.get_nowait()
    assert pair.is_empty()
    pair.high.put_nowait("h")
    assert not pair.is_empty()
    pair.high.get_nowait()
    assert pair.is_empty()


async def test_waking_queue_overrides_internal_put_only() -> None:
    """The override is on ``_put`` so it covers ``put_nowait`` and
    the async ``put`` path uniformly. Spot-check that
    ``put_nowait`` and ``put`` both wake a waiter."""
    pair = EventQueuePair()
    waiter = asyncio.create_task(pair.wait_nonempty())
    await asyncio.sleep(0)
    await pair.normal.put("via-async-put")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result is True


async def test_wait_with_timeout_returns_true_when_event_arrives() -> None:
    """A bounded wait returns True if an event arrives before the
    deadline, with the event still in the queue."""
    pair = EventQueuePair()

    async def delayed_put() -> None:
        await asyncio.sleep(0.02)
        pair.normal.put_nowait("ev")

    asyncio.create_task(delayed_put())
    result = await pair.wait_nonempty(timeout=1.0)
    assert result is True
    assert pair.normal.qsize() == 1


async def test_spurious_wakeup_does_not_falsely_return() -> None:
    """If a put happens and a separate consumer drains the queue
    before the waiter observes, the waiter must keep waiting
    (returning True only on a real arrival). Models a
    ``wait_for_next_event`` racing the high-priority loop."""
    pair = EventQueuePair()
    waiter = asyncio.create_task(pair.wait_nonempty(timeout=0.5))
    await asyncio.sleep(0)
    # Simulate "put then drain before waiter sees it": put then
    # immediately get_nowait to empty the queue. The waiter MUST
    # NOT spuriously return True.
    pair.normal.put_nowait("ev-then-drained")
    pair.normal.get_nowait()
    await asyncio.sleep(0)
    # Now genuinely arrive an event — waiter should return True
    # on this real event.
    pair.normal.put_nowait("real-ev")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result is True


def test_waking_queue_is_an_asyncio_queue_subclass() -> None:
    """Backwards-compat: existing code that types arguments as
    ``asyncio.Queue`` keeps working because the lanes ARE
    ``asyncio.Queue`` instances."""
    pair = EventQueuePair()
    assert isinstance(pair.normal, asyncio.Queue)
    assert isinstance(pair.high, asyncio.Queue)
    assert isinstance(pair.normal, _WakingQueue)
    assert isinstance(pair.high, _WakingQueue)
