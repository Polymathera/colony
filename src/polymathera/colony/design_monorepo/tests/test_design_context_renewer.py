"""Tests for :class:`DesignContextLockRenewer`.

The renewer's job is to call ``extend_page_lock`` on every page of
every registered scope before the lock expires. Key invariants:

- Registration starts the background task lazily (no idle task when
  nothing is pinned).
- The first registration with a short duration wakes the loop early
  (we don't wait for the previous longer duration to elapse).
- ``get_pages_for_scope`` is re-called every cycle so new pages
  materialised after the initial registration get picked up.
- ``extend_page_lock`` is preferred (cheap); ``lock_page`` is the
  fallback for pages whose lock already expired or were never
  locked (e.g. newly materialised between registration and the
  first renewal).
- ``stop()`` is idempotent and clean.

The renewer is exercised against a fake clock so tests deterministic
and fast — production uses :func:`asyncio.sleep`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.design_monorepo.design_context_renewer import (
    DesignContextLockRenewer,
)


class _FakeClock:
    """Deterministic clock + sleeper for renewer loop tests.

    ``advance(seconds)`` moves the clock forward and yields to the
    event loop enough times for any pending coroutine awaiting
    ``sleep`` to resume.

    ``settle()`` yields without advancing the clock — use this after
    creating a task to let it run to its first ``await self.sleep``
    so the test knows it's waiting on a known deadline before
    advancing.
    """

    def __init__(self) -> None:
        self._now: float = 0.0
        self._pending: list[tuple[float, asyncio.Future]] = []

    def time(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending.append((self._now + seconds, fut))
        try:
            await fut
        finally:
            # Defensive: drop ourselves from _pending if the future
            # was resolved externally without going through advance.
            self._pending = [(t, f) for (t, f) in self._pending if f is not fut]

    async def settle(self) -> None:
        """Yield enough times for any just-scheduled task to reach
        its first ``await self.sleep`` (so its deadline is registered
        against the CURRENT clock time before the next advance).
        """

        for _ in range(20):
            await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        self._now += seconds
        # Wake all sleeps whose deadline has passed.
        ready = [(t, f) for (t, f) in self._pending if t <= self._now]
        self._pending = [(t, f) for (t, f) in self._pending if t > self._now]
        for _t, fut in ready:
            if not fut.done():
                fut.set_result(None)
        # Yield several times so the just-resumed coroutines get to
        # run their next steps before the test assertion.
        for _ in range(20):
            await asyncio.sleep(0)


def _make_vcm_handle(pages_by_scope: dict[str, list[dict]] | None = None):
    """Mock VCM handle wired with predictable lock semantics.

    Pages start unlocked; ``extend_page_lock`` returns False (page
    not locked yet); the renewer's fallback ``lock_page`` records
    the call. On the second cycle the same handle returns True from
    extend (simulating the page being locked now)."""

    handle = MagicMock()
    handle.get_pages_for_scope = AsyncMock(
        side_effect=lambda *, scope_id: list(
            (pages_by_scope or {}).get(scope_id, []),
        ),
    )
    locked: set[str] = set()

    async def _extend(*, page_id, additional_duration_s):
        return page_id in locked

    async def _lock(*, page_id, locked_by, lock_duration_s, reason=""):
        locked.add(page_id)

    handle.extend_page_lock = AsyncMock(side_effect=_extend)
    handle.lock_page = AsyncMock(side_effect=_lock)
    handle._test_locked = locked  # for assertions
    return handle


@pytest.mark.asyncio
async def test_register_validates_duration() -> None:
    renewer = DesignContextLockRenewer(vcm_handle=MagicMock())
    with pytest.raises(ValueError, match="lock_duration_s"):
        await renewer.register(
            source_name="x", scope_id="s", lock_duration_s=0,
        )
    await renewer.stop()


@pytest.mark.asyncio
async def test_register_after_stop_raises() -> None:
    renewer = DesignContextLockRenewer(vcm_handle=MagicMock())
    await renewer.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await renewer.register(
            source_name="x", scope_id="s", lock_duration_s=10,
        )


@pytest.mark.asyncio
async def test_register_starts_task_lazily() -> None:
    """No background task until the first registration."""

    renewer = DesignContextLockRenewer(vcm_handle=MagicMock())
    assert renewer._task is None  # noqa: SLF001 — test introspection
    handle = _make_vcm_handle(pages_by_scope={"s1": []})
    renewer._vcm = handle
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=100.0,
    )
    assert renewer._task is not None  # noqa: SLF001
    assert not renewer._task.done()  # noqa: SLF001
    await renewer.stop()


@pytest.mark.asyncio
async def test_renewer_extends_pages_each_cycle() -> None:
    clock = _FakeClock()
    handle = _make_vcm_handle(
        pages_by_scope={"s1": [{"page_id": "p1"}, {"page_id": "p2"}]},
    )
    # Pre-mark p1/p2 as locked so extend returns True on the first
    # cycle (simulating the initial lock_page from the materialiser).
    handle._test_locked.update({"p1", "p2"})

    renewer = DesignContextLockRenewer(
        vcm_handle=handle,
        min_tick_s=1.0,
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )
    duration = 70.0  # renewal fires at 6/7 * 70 = 60 s
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=duration,
    )
    # Let the loop enter its sleep at time 0 — without this, the
    # loop computes its first sleep_for AFTER ``advance`` already
    # bumped the clock, and ends up waiting on a deadline beyond
    # what the test advances to.
    await clock.settle()

    # Advance just past the first renewal deadline.
    await clock.advance(seconds=61.0)
    assert handle.extend_page_lock.await_count == 2
    # Both extends returned True → no fallback to lock_page.
    assert handle.lock_page.await_count == 0

    # Second cycle — same settle/advance pattern.
    await clock.settle()
    await clock.advance(seconds=61.0)
    assert handle.extend_page_lock.await_count == 4

    await renewer.stop()


@pytest.mark.asyncio
async def test_renewer_falls_back_to_lock_page_when_extend_fails() -> None:
    """Pages that aren't locked yet (newly materialised between
    registration and the first renewal, OR whose lock has expired)
    get lock_page called instead of extend."""

    clock = _FakeClock()
    pages = {"s1": [{"page_id": "p_new"}]}
    handle = _make_vcm_handle(pages_by_scope=pages)
    # p_new is NOT in _test_locked → extend returns False → fallback.

    renewer = DesignContextLockRenewer(
        vcm_handle=handle,
        min_tick_s=1.0,
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=70.0,
    )
    await clock.settle()
    await clock.advance(seconds=61.0)

    handle.extend_page_lock.assert_awaited_once_with(
        page_id="p_new", additional_duration_s=70.0,
    )
    handle.lock_page.assert_awaited_once()
    lock_call = handle.lock_page.await_args
    assert lock_call.kwargs["page_id"] == "p_new"
    assert lock_call.kwargs["locked_by"] == "design_context.x"
    assert lock_call.kwargs["lock_duration_s"] == 70.0

    await renewer.stop()


@pytest.mark.asyncio
async def test_renewer_picks_up_new_pages_between_cycles() -> None:
    """``get_pages_for_scope`` is called every cycle, so a page
    appearing after the initial registration is locked on the next
    tick — no explicit "new pages" call from the materialiser."""

    clock = _FakeClock()
    initial_pages: list[dict] = []
    pages_by_scope = {"s1": initial_pages}
    handle = _make_vcm_handle(pages_by_scope=pages_by_scope)

    renewer = DesignContextLockRenewer(
        vcm_handle=handle,
        min_tick_s=1.0,
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=70.0,
    )
    await clock.settle()

    # First cycle — empty scope, no calls.
    await clock.advance(seconds=61.0)
    assert handle.extend_page_lock.await_count == 0
    assert handle.lock_page.await_count == 0

    # Pages appear before the second cycle.
    initial_pages.extend([{"page_id": "p_late_1"}, {"page_id": "p_late_2"}])
    await clock.settle()
    await clock.advance(seconds=61.0)

    # Renewer saw both new pages and locked them (extend → False →
    # fallback to lock_page).
    assert handle.lock_page.await_count == 2
    locked_ids = {
        call.kwargs["page_id"]
        for call in handle.lock_page.await_args_list
    }
    assert locked_ids == {"p_late_1", "p_late_2"}

    await renewer.stop()


@pytest.mark.asyncio
async def test_register_is_idempotent_for_same_scope() -> None:
    """Re-registering the same scope updates duration in place;
    one background task only."""

    handle = _make_vcm_handle(pages_by_scope={"s1": []})
    renewer = DesignContextLockRenewer(
        vcm_handle=handle, min_tick_s=1.0,
    )
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=100.0,
    )
    first_task = renewer._task  # noqa: SLF001
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=200.0,
    )
    assert renewer._task is first_task  # noqa: SLF001 — same task reused
    assert renewer.registered_scope_ids == ["s1"]
    await renewer.stop()


@pytest.mark.asyncio
async def test_unregister_drops_the_scope() -> None:
    handle = _make_vcm_handle(pages_by_scope={"s1": [], "s2": []})
    renewer = DesignContextLockRenewer(
        vcm_handle=handle, min_tick_s=1.0,
    )
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=10.0,
    )
    await renewer.register(
        source_name="y", scope_id="s2", lock_duration_s=10.0,
    )
    assert sorted(renewer.registered_scope_ids) == ["s1", "s2"]

    assert await renewer.unregister("s1") is True
    assert renewer.registered_scope_ids == ["s2"]
    # Unregistering an unknown scope is a no-op returning False.
    assert await renewer.unregister("missing") is False
    await renewer.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent() -> None:
    handle = _make_vcm_handle(pages_by_scope={"s1": []})
    renewer = DesignContextLockRenewer(vcm_handle=handle)
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=10.0,
    )
    await renewer.stop()
    # Second stop must not raise.
    await renewer.stop()


@pytest.mark.asyncio
async def test_renewer_survives_get_pages_for_scope_failure() -> None:
    """If get_pages_for_scope raises on one tick, the loop logs and
    skips that scope — it MUST keep firing for subsequent cycles."""

    clock = _FakeClock()

    call_count = {"n": 0}

    async def _flaky(*, scope_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient VCM hiccup")
        return [{"page_id": "p1"}]

    handle = MagicMock()
    handle.get_pages_for_scope = AsyncMock(side_effect=_flaky)
    handle.extend_page_lock = AsyncMock(return_value=False)
    handle.lock_page = AsyncMock()

    renewer = DesignContextLockRenewer(
        vcm_handle=handle,
        min_tick_s=1.0,
        sleep_fn=clock.sleep,
        time_fn=clock.time,
    )
    await renewer.register(
        source_name="x", scope_id="s1", lock_duration_s=70.0,
    )
    await clock.settle()

    # First tick fails silently.
    await clock.advance(seconds=61.0)
    assert handle.lock_page.await_count == 0  # nothing to lock — error path

    # Second tick succeeds.
    await clock.settle()
    await clock.advance(seconds=61.0)
    assert handle.lock_page.await_count == 1

    await renewer.stop()
