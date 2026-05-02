"""Tests for ``CompositeWatcher`` (merging N child watchers)."""

from __future__ import annotations

import asyncio

import pytest

from polymathera.colony.vcm.page_events import PageChangeEvent
from polymathera.colony.vcm.watchers import CompositeWatcher


pytestmark = pytest.mark.asyncio


class _FakeWatcher:
    """Minimal watcher that yields a pre-canned list of events, then
    waits on its stop signal so the composite can drive the lifecycle."""

    static = False

    def __init__(self, scope_id: str, events: list[PageChangeEvent]) -> None:
        self.scope_id = scope_id
        self._events = list(events)
        self._stopped = asyncio.Event()
        self.stop_called = 0

    def stop(self) -> None:
        self.stop_called += 1
        self._stopped.set()

    async def watch(self):
        for ev in self._events:
            if self._stopped.is_set():
                return
            yield ev
        # Stay open until stopped, mirroring the real watchers'
        # long-running shape.
        await self._stopped.wait()


def _ev(page_id: str) -> PageChangeEvent:
    return PageChangeEvent.page_added(
        page_id=page_id, source="fake:test", scope_id="scope:test",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


async def test_requires_at_least_one_child() -> None:
    with pytest.raises(ValueError):
        CompositeWatcher((), scope_id="empty")


async def test_exposes_children_for_inspection() -> None:
    child = _FakeWatcher("child", [])
    cw = CompositeWatcher((child,), scope_id="parent")
    assert cw.children == (child,)
    assert cw.scope_id == "parent"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


async def test_yields_events_from_all_children() -> None:
    a = _FakeWatcher("a", [_ev("p_a1"), _ev("p_a2")])
    b = _FakeWatcher("b", [_ev("p_b1")])
    cw = CompositeWatcher((a, b), scope_id="parent")

    async def consume() -> set[str]:
        seen: set[str] = set()
        async for ev in cw.watch():
            seen.add(ev.page_id)
            if seen >= {"p_a1", "p_a2", "p_b1"}:
                cw.stop()
        return seen

    seen = await asyncio.wait_for(consume(), timeout=1.5)
    assert seen == {"p_a1", "p_a2", "p_b1"}


async def test_stop_propagates_to_every_child() -> None:
    a = _FakeWatcher("a", [_ev("p_a")])
    b = _FakeWatcher("b", [_ev("p_b")])
    cw = CompositeWatcher((a, b), scope_id="parent")

    async def consume() -> None:
        async for _ in cw.watch():
            cw.stop()
            break

    await asyncio.wait_for(consume(), timeout=1.5)
    assert a.stop_called >= 1
    assert b.stop_called >= 1


async def test_iterator_exit_stops_children_even_without_explicit_stop() -> None:
    """Breaking out of the iterator (e.g. caller cancellation) must
    still trigger ``stop()`` on every child via the ``finally`` cleanup."""

    a = _FakeWatcher("a", [_ev("p_a")])
    b = _FakeWatcher("b", [_ev("p_b")])
    cw = CompositeWatcher((a, b), scope_id="parent")

    async def consume() -> None:
        # Read one event then break — exercise the finally cleanup.
        async for _ in cw.watch():
            return

    await asyncio.wait_for(consume(), timeout=1.5)
    assert a.stop_called >= 1
    assert b.stop_called >= 1


async def test_one_child_crash_does_not_stop_others() -> None:
    class _CrashingWatcher:
        static = False
        scope_id = "crasher"

        def stop(self) -> None:
            return None

        async def watch(self):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    crasher = _CrashingWatcher()
    survivor = _FakeWatcher("survivor", [_ev("p_survive")])
    cw = CompositeWatcher((crasher, survivor), scope_id="parent")

    async def consume() -> str:
        async for ev in cw.watch():
            cw.stop()
            return ev.page_id
        return ""

    page = await asyncio.wait_for(consume(), timeout=1.5)
    assert page == "p_survive"
