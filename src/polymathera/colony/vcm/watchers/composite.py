"""``CompositeWatcher`` — merge N child watchers into one async iterator.

Useful when a single page source needs to combine multiple watch
transports against the same backing store. The canonical example is
``GitRepoContextPageSource``, which couples a ``LocalFsWatcher``
(for in-tree edits) with a ``GitRemoteWatcher`` (for upstream commits)
on the same cloned working tree — both watchers' events should reach
the convergence runtime through one stream.

The composite owes the same cooperative-cancellation contract as the
underlying watchers (master §5.6 watcher-contract item 1):

- ``watch()`` is a single async iterator that yields events from
  whichever child produces next; cancellation propagates to children.
- ``stop()`` stops every child.
- ``static = False`` (the composite only makes sense for live sources).

Pumps run as one ``asyncio.Task`` per child, all writing into a bounded
shared queue. Queue overflow drops the offending event with a warning;
the convergence runtime's per-page rate-limiter is the safety net for
sustained back-pressure.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from ..page_events import PageChangeEvent


logger = logging.getLogger(__name__)


class _Watcher(Protocol):
    """Structural shape every child must satisfy."""

    scope_id: str

    def stop(self) -> None: ...

    def watch(self) -> AsyncIterator[PageChangeEvent]: ...


class CompositeWatcher:
    """Merge N child watchers into one cancellable async iterator."""

    static = False

    def __init__(
        self,
        watchers: Iterable[_Watcher],
        *,
        scope_id: str,
        queue_maxsize: int = 1024,
    ) -> None:
        self._watchers: tuple[_Watcher, ...] = tuple(watchers)
        if not self._watchers:
            raise ValueError(
                "CompositeWatcher requires at least one child watcher.",
            )
        self._scope_id = scope_id
        self._queue_maxsize = queue_maxsize

    @property
    def scope_id(self) -> str:
        return self._scope_id

    @property
    def children(self) -> tuple[_Watcher, ...]:
        """Read-only view of the wrapped watchers; useful for tests."""

        return self._watchers

    def stop(self) -> None:
        """Signal every child to stop. Idempotent."""

        for w in self._watchers:
            try:
                w.stop()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "CompositeWatcher[%s]: stop() failed on %s",
                    self._scope_id, type(w).__name__,
                )

    async def watch(self) -> AsyncIterator[PageChangeEvent]:
        """Yield events from whichever child produces next.

        Each child is drained by a dedicated pump task; the pumps
        write into a shared ``asyncio.Queue``. The iterator yields
        from that queue. On exit (cancellation, exception, or natural
        end-of-stream), every child is stopped and every pump task is
        awaited to completion so resources are released cleanly.
        """

        merged: asyncio.Queue[PageChangeEvent] = asyncio.Queue(
            maxsize=self._queue_maxsize,
        )

        async def _pump(watcher: _Watcher) -> None:
            try:
                async for ev in watcher.watch():
                    try:
                        merged.put_nowait(ev)
                    except asyncio.QueueFull:
                        logger.warning(
                            "CompositeWatcher[%s]: merged queue full; "
                            "dropping %s for page %s from %s",
                            self._scope_id,
                            ev.kind.value,
                            ev.page_id,
                            type(watcher).__name__,
                        )
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                logger.exception(
                    "CompositeWatcher[%s]: pump for %s crashed",
                    self._scope_id, type(watcher).__name__,
                )

        tasks = [
            asyncio.create_task(
                _pump(w),
                name=f"composite_watcher:{self._scope_id}:{type(w).__name__}",
            )
            for w in self._watchers
        ]
        try:
            while True:
                try:
                    event = await merged.get()
                except asyncio.CancelledError:
                    return
                yield event
        finally:
            self.stop()
            for t in tasks:
                t.cancel()
                for _ in (1,):  # single iteration; await with shield
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass


__all__ = ("CompositeWatcher",)
