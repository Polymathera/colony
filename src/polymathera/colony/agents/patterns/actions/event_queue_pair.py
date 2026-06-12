"""EventQueuePair — two asyncio queues + a shared nonempty wakeup signal.

Sibling of ``BlockStreakTracker`` in the extracted-class pattern. Owns
the synchronization primitive that lets ``wait_for_next_event`` block on
either queue becoming nonempty without consuming anything (consume-none
semantics).

The pair is consumed by ``EventDrivenActionPolicy``, which keeps owning
the *drain* logic (in ``plan_step``'s observation path and in
``_run_high_priority_loop``) — this class owns only the queues and the
wakeup signal, nothing else. Existing code that calls
``policy.get_event_queue().put_nowait(event)`` keeps working unchanged
because the queue objects ARE ``asyncio.Queue`` instances; the wakeup
side-effect is added transparently via a one-method ``_put`` override.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...blackboard.protocol import BlackboardEvent


class _WakingQueue(asyncio.Queue):
    """asyncio.Queue that signals a shared ``asyncio.Event`` on every put.

    Used by :class:`EventQueuePair` to support consume-none wait
    semantics: a producer's ``put_nowait`` both enqueues the event AND
    wakes any task awaiting :meth:`EventQueuePair.wait_nonempty`. We
    override ``_put`` — the inner-most hook that both sync ``put_nowait``
    and async ``put`` route through — so every code path that produces
    an event triggers the wakeup signal automatically. Producers do not
    need to know the signal exists.
    """

    def __init__(
        self, wakeup_signal: asyncio.Event, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wakeup_signal = wakeup_signal

    def _put(self, item: Any) -> None:  # type: ignore[override]
        super()._put(item)
        self._wakeup_signal.set()


class EventQueuePair:
    """Two-lane event queue with a shared "nonempty" wakeup signal.

    ``normal`` and ``high`` are independent ``asyncio.Queue`` instances
    — drained by separate code paths (``plan_step`` for normal,
    ``_run_high_priority_loop`` for high). The shared ``_wakeup`` event
    is set whenever EITHER queue receives a put and is cleared
    explicitly by the consumer (``wait_nonempty``) before it sleeps.
    This canonical "event wait with predicate" idiom is robust to
    spurious wakeups arising from put + drain interleavings.

    :meth:`wait_nonempty` is the consume-none barrier used by
    ``wait_for_next_event``. It returns when at least one queue is
    nonempty, or when the timeout elapses. It never pulls events from
    either queue — that work belongs to the existing drain paths and is
    not duplicated here.
    """

    def __init__(self) -> None:
        self._wakeup: asyncio.Event = asyncio.Event()
        self.normal: asyncio.Queue["BlackboardEvent"] = _WakingQueue(
            self._wakeup,
        )
        self.high: asyncio.Queue["BlackboardEvent"] = _WakingQueue(
            self._wakeup,
        )

    def is_empty(self) -> bool:
        """True iff both lanes are empty."""
        return self.normal.empty() and self.high.empty()

    async def wait_nonempty(self, timeout: float | None = None) -> bool:
        """Block until at least one queue is nonempty.

        Returns ``True`` on wakeup (queues nonempty), ``False`` on
        timeout. Consume-none — does NOT pull events from either queue.
        Cancellation propagates as ``asyncio.CancelledError``.

        The clear-then-double-check loop is the canonical asyncio idiom
        for waiting on a condition predicate via ``asyncio.Event``: by
        clearing the wakeup signal BEFORE re-checking the predicate, we
        cannot miss a put that lands between the check and the wait
        (the put will re-set the signal, the wait will return
        immediately). Spurious wakeups — a put that has already been
        drained by another consumer before our task observes the
        queues — are absorbed by the outer loop.
        """
        if not self.is_empty():
            return True
        loop = asyncio.get_event_loop()
        deadline = (loop.time() + timeout) if timeout is not None else None
        while True:
            self._wakeup.clear()
            if not self.is_empty():
                return True
            if deadline is None:
                await self._wakeup.wait()
            else:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(
                        self._wakeup.wait(), timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    return False
            if not self.is_empty():
                return True
            # Spurious wakeup: a put + drain interleaved before we
            # observed the queues. Loop, re-clear, re-wait.
