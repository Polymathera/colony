"""Background renewer for ``pin_in_vcm`` page locks on design-context
scopes.

``VirtualPageTable.lock_page`` takes a finite ``lock_duration_s``;
without renewal, pinned design-context pages eventually expire and
become evictable. This module owns a long-lived asyncio task that:

1. Tracks which design-context scopes have ``pin_in_vcm=True``.
2. Periodically re-lists the pages in each tracked scope (catches
   pages materialised after the initial registration when the
   underlying corpus grows).
3. Calls ``vcm.extend_page_lock`` per page, falling back to
   ``vcm.lock_page`` when extension fails (page was evicted or never
   locked — both are recoverable by issuing a fresh lock).

The renewer is owned by the ``RepoStateProvider`` instance that
called ``materialize_design_context()``. One renewer per capability
instance keeps the lifecycle bounded: when the capability is
suspended/torn down, ``stop()`` cancels the task and pages naturally
unlock when their existing duration expires.

The renewer is **not** persisted across cluster restarts. On restart
the operator (or the future-phase ``DesignProcessCapability``) calls
``materialize_design_context()`` again; pin registration is part of
that call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


# Fraction of the lock window at which the renewer fires; 6/7 keeps
# renewals well ahead of expiry without burning cycles. Constant
# rather than configurable — the per-row ``pin_lock_duration_days``
# already gives operators the cadence knob.
_RENEWAL_FRACTION: float = 6.0 / 7.0


@dataclass
class _PinnedScope:
    """One pinned design-context scope tracked by the renewer."""

    source_name: str
    scope_id: str
    locked_by: str
    lock_duration_s: float
    # Wall-clock time of the next planned renewal.
    next_renewal_at: float


class DesignContextLockRenewer:
    """Periodic refresher for ``pin_in_vcm`` design-context page locks.

    Usage::

        renewer = DesignContextLockRenewer(vcm_handle)
        await renewer.register(
            source_name="hard-constraints",
            scope_id="design_context.hard-constraints",
            lock_duration_s=7 * 86400,
        )
        # ... later, on capability shutdown:
        await renewer.stop()

    Registration is idempotent: repeated ``register`` for the same
    ``scope_id`` updates the lock duration in place (and resets the
    next renewal deadline accordingly). Pages added to the scope
    after registration are picked up on the next renewal tick — the
    renewer re-lists every cycle, so no explicit "new pages" call
    is needed from the materialiser.
    """

    def __init__(
        self,
        vcm_handle: Any,
        *,
        # The minimum sleep between scheduler wake-ups. Caps how
        # quickly a freshly-registered scope's first renewal fires;
        # the scheduler always honors per-scope deadlines, but
        # ``min_tick_s`` prevents busy-looping when many short
        # durations are registered together. Default of 60 s is well
        # below the smallest practical pin_lock_duration_days (1 day
        # → renewal every ~20 hours).
        min_tick_s: float = 60.0,
        # Test hook — defaults to ``asyncio.sleep`` so production
        # code uses real time, but tests can substitute a fake clock.
        sleep_fn: Any = asyncio.sleep,
        time_fn: Any = time.time,
    ) -> None:
        self._vcm = vcm_handle
        self._min_tick_s = min_tick_s
        self._sleep = sleep_fn
        self._time = time_fn
        self._scopes: dict[str, _PinnedScope] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._wakeup = asyncio.Event()
        self._stopped = False

    async def register(
        self,
        *,
        source_name: str,
        scope_id: str,
        lock_duration_s: float,
        locked_by: str | None = None,
    ) -> None:
        """Register a pinned scope (or update an existing registration).

        Starts the background task on first registration. Repeated
        calls for the same ``scope_id`` overwrite the previous
        registration's lock_duration_s and reset the renewal deadline.
        """

        if self._stopped:
            raise RuntimeError(
                "DesignContextLockRenewer.register: renewer has been "
                "stopped; create a new instance to resume.",
            )
        if lock_duration_s <= 0:
            raise ValueError(
                f"DesignContextLockRenewer.register: lock_duration_s "
                f"must be > 0, got {lock_duration_s!r}.",
            )

        effective_locked_by = locked_by or f"design_context.{source_name}"
        now = self._time()
        async with self._lock:
            self._scopes[scope_id] = _PinnedScope(
                source_name=source_name,
                scope_id=scope_id,
                locked_by=effective_locked_by,
                lock_duration_s=lock_duration_s,
                next_renewal_at=now + (lock_duration_s * _RENEWAL_FRACTION),
            )
            # Start the background task lazily so capabilities that
            # never register a pinned source don't spawn an idle task.
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(
                    self._renewal_loop(),
                    name="DesignContextLockRenewer",
                )
        # Wake the loop so it recomputes its next sleep with the new
        # entry in play (cheaper than waiting for the current sleep
        # to expire when a shorter duration was just added).
        self._wakeup.set()

    async def unregister(self, scope_id: str) -> bool:
        """Drop a scope from renewal. Pages stay locked until the
        currently-issued lock expires; the renewer simply stops
        refreshing it. Returns ``True`` if removed, ``False`` if the
        scope wasn't registered."""

        async with self._lock:
            removed = self._scopes.pop(scope_id, None) is not None
        if removed:
            self._wakeup.set()
        return removed

    async def stop(self) -> None:
        """Cancel the background task. Idempotent. Pages remain locked
        until their currently-issued duration expires — no explicit
        unlock issued, matching the renewer's "expiry is the natural
        end-of-life" stance."""

        self._stopped = True
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — log + swallow on shutdown
            logger.exception(
                "DesignContextLockRenewer.stop: task raised on cancel",
            )
        self._task = None

    @property
    def registered_scope_ids(self) -> list[str]:
        """For introspection / tests. Snapshot of the registry keys."""

        return list(self._scopes.keys())

    async def _renewal_loop(self) -> None:
        """Single coroutine that:

        1. Sleeps until the soonest ``next_renewal_at`` (or
           ``min_tick_s``, whichever is larger).
        2. Wakes for any ``register``/``unregister`` event via
           ``_wakeup``.
        3. Re-locks every page in every registered scope whose
           deadline has passed.
        """

        try:
            while True:
                # Compute next deadline under the lock to get a
                # consistent snapshot.
                async with self._lock:
                    if not self._scopes:
                        next_deadline: float | None = None
                    else:
                        next_deadline = min(
                            s.next_renewal_at for s in self._scopes.values()
                        )

                now = self._time()
                if next_deadline is None:
                    # No work — sleep until poked.
                    self._wakeup.clear()
                    await self._wakeup.wait()
                    continue

                sleep_for = max(self._min_tick_s, next_deadline - now)
                self._wakeup.clear()
                # Race the timed sleep against an external wakeup;
                # whichever completes first ends the wait.
                sleep_task = asyncio.create_task(self._sleep(sleep_for))
                wakeup_task = asyncio.create_task(self._wakeup.wait())
                done, pending = await asyncio.wait(
                    {sleep_task, wakeup_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                # Drain cancelled tasks so they don't show as warnings.
                for t in pending:
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass

                # Renew every scope whose deadline has elapsed.
                now = self._time()
                async with self._lock:
                    due = [
                        s for s in self._scopes.values()
                        if s.next_renewal_at <= now
                    ]
                for scope in due:
                    await self._renew_one(scope)
                    # Advance the deadline AFTER the renewal completes
                    # so a slow renewal doesn't get re-fired the same
                    # tick.
                    async with self._lock:
                        live = self._scopes.get(scope.scope_id)
                        if live is not None:
                            live.next_renewal_at = (
                                self._time()
                                + live.lock_duration_s * _RENEWAL_FRACTION
                            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — keep the loop alive
            logger.exception(
                "DesignContextLockRenewer: renewal loop crashed; "
                "task exiting (no more renewals until re-registered)",
            )

    async def _renew_one(self, scope: _PinnedScope) -> None:
        """Re-lock every page currently in ``scope.scope_id``.

        Uses ``extend_page_lock`` first (cheap if the page is already
        locked by us) and falls back to ``lock_page`` for pages that
        were never locked or whose lock has expired."""

        try:
            pages = await self._vcm.get_pages_for_scope(
                scope_id=scope.scope_id,
            )
        except Exception:  # noqa: BLE001 — log + skip this scope this tick
            logger.exception(
                "DesignContextLockRenewer: get_pages_for_scope failed "
                "for %r; will retry on next tick",
                scope.scope_id,
            )
            return

        for page in pages:
            page_id = page.get("page_id")
            if not page_id:
                logger.warning(
                    "DesignContextLockRenewer: page entry has no page_id "
                    "in scope %r; skipping (entry=%r)",
                    scope.scope_id,
                    page,
                )
                continue

            extended = False
            try:
                extended = await self._vcm.extend_page_lock(
                    page_id=page_id,
                    additional_duration_s=scope.lock_duration_s,
                )
            except Exception:  # noqa: BLE001 — fall through to lock_page
                logger.debug(
                    "DesignContextLockRenewer: extend_page_lock raised "
                    "for %r; falling back to lock_page",
                    page_id,
                    exc_info=True,
                )

            if extended:
                continue

            # Page was not previously locked (extend returns False) —
            # issue a fresh lock. This covers (a) pages materialised
            # after the initial pin call, and (b) pages whose locks
            # already expired (e.g. cluster restart).
            try:
                await self._vcm.lock_page(
                    page_id=page_id,
                    locked_by=scope.locked_by,
                    lock_duration_s=scope.lock_duration_s,
                    reason=f"design_context pin ({scope.source_name})",
                )
            except Exception:  # noqa: BLE001 — log + carry on
                logger.exception(
                    "DesignContextLockRenewer: lock_page failed for "
                    "%r in scope %r",
                    page_id,
                    scope.scope_id,
                )
