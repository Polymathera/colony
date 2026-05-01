"""``LocalFsWatcher`` — debounced filesystem watcher.

Implements master §5.6 transport (1) for design-monorepo working trees
and any other source backed by a local directory. Uses ``watchdog``
under the hood — wrapping ``inotify`` on Linux and ``FSEvents`` on
macOS.

Events are coalesced over a configurable debounce window: rapid
file-write bursts (e.g. an editor saving and a formatter rewriting) are
collapsed into a single ``PageChangeEvent`` per affected path. This
matches the doc's expectation that the source de-duplicates before its
events reach the runtime's rate limiter.

Without a ``watchdog`` install the watcher degrades to a polling loop
over file mtimes — same external contract, longer latency. The
distribution declares ``watchdog`` as an extra under
``design_monorepo``; it is therefore present in any deployment that
runs Phase C5/C4.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..page_events import PageChangeEvent, PageChangeKind


logger = logging.getLogger(__name__)


# Predicate that tells the watcher whether a relative path under the
# watched root is interesting. Default below ignores VCS/build noise.
PathFilter = Callable[[Path], bool]


def _default_path_filter(path: Path) -> bool:
    parts = path.parts
    if any(p == ".git" or p == "__pycache__" or p.endswith(".pyc") for p in parts):
        return False
    if any(p.startswith(".") and p not in (".gitignore", ".gitattributes") for p in parts):
        return False
    return True


# Function that maps an absolute path to a stable page id. The default
# uses the path itself; sources that group files into pages override.
PageIdFor = Callable[[Path], str]


@dataclass
class LocalFsWatcherConfig:
    """Tuning knobs for ``LocalFsWatcher``."""

    debounce_s: float = 0.5
    """How long after the last file event before emitting the page event.
    Editors that save-then-format produce two events ~50ms apart; 0.5 s
    is comfortably above that without making interactive feedback feel
    laggy."""

    poll_interval_s: float = 2.0
    """Interval used by the polling fallback when ``watchdog`` is
    unavailable."""

    data_type: str | None = None
    """Optional ``data_type`` tag carried on every emitted event."""


class LocalFsWatcher:
    """Watch a local directory and emit ``PageChangeEvent``s.

    ``static = False`` — this watcher is the canonical example of a
    live source. Use it via:

    .. code-block:: python

        watcher = LocalFsWatcher(
            root=Path("/path/to/design-repo"),
            scope_id="program-1",
            source_uri="git:git@example/foo.git:main:HEAD",
            page_id_for=lambda p: f"file:{p}",
        )
        async for event in watcher.watch():
            ...
    """

    static = False

    def __init__(
        self,
        *,
        root: Path,
        scope_id: str,
        source_uri: str,
        page_id_for: PageIdFor | None = None,
        path_filter: PathFilter | None = None,
        config: LocalFsWatcherConfig | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._scope_id = scope_id
        self._source_uri = source_uri
        self._page_id_for = page_id_for or (lambda p: f"file:{p.as_posix()}")
        self._path_filter = path_filter or _default_path_filter
        self._config = config or LocalFsWatcherConfig()
        self._stopped = asyncio.Event()
        self._pending: dict[str, _PendingChange] = {}
        self._pending_lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def scope_id(self) -> str:
        return self._scope_id

    def stop(self) -> None:
        self._stopped.set()

    async def watch(self) -> AsyncIterator[PageChangeEvent]:
        """Yield events as they arrive, until ``stop()`` is called."""

        if not self._root.is_dir():
            raise FileNotFoundError(
                f"LocalFsWatcher root {self._root} does not exist or is not a directory.",
            )
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            event_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            root = self._root
            filter_fn = self._path_filter

            def _push(kind: str, path_str: str) -> None:
                p = Path(path_str)
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    return
                if not filter_fn(rel):
                    return
                loop.call_soon_threadsafe(
                    event_queue.put_nowait, (kind, p),
                )

            class _Handler(FileSystemEventHandler):
                def on_created(self, event: Any) -> None:  # type: ignore[override]
                    if event.is_directory:
                        return
                    _push("created", event.src_path)

                def on_modified(self, event: Any) -> None:  # type: ignore[override]
                    if event.is_directory:
                        return
                    _push("modified", event.src_path)

                def on_deleted(self, event: Any) -> None:  # type: ignore[override]
                    if event.is_directory:
                        return
                    _push("deleted", event.src_path)

                def on_moved(self, event: Any) -> None:  # type: ignore[override]
                    if event.is_directory:
                        return
                    _push("deleted", event.src_path)
                    _push("created", event.dest_path)

            observer = Observer()
            observer.schedule(_Handler(), str(self._root), recursive=True)
            observer.start()
            try:
                async for event in self._consume_queue(event_queue):
                    yield event
            finally:
                observer.stop()
                observer.join(timeout=1.0)
        except ImportError:
            # Fallback: poll mtimes.
            logger.warning(
                "LocalFsWatcher: 'watchdog' not installed; falling back to "
                "%.1fs poll on %s", self._config.poll_interval_s, self._root,
            )
            async for event in self._poll_loop():
                yield event

    # ---- Streaming + debouncing ----------------------------------------

    async def _consume_queue(
        self, queue: asyncio.Queue[tuple[str, Path]],
    ) -> AsyncIterator[PageChangeEvent]:
        flush_task: asyncio.Task[None] | None = None
        try:
            while not self._stopped.is_set():
                # Wait for the next raw fs event (or stop).
                getter = asyncio.create_task(queue.get())
                stopper = asyncio.create_task(self._stopped.wait())
                done, pending = await asyncio.wait(
                    [getter, stopper], return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                if stopper in done:
                    break
                kind, path = getter.result()
                async with self._pending_lock:
                    self._record_pending(kind, path)
                if flush_task is None or flush_task.done():
                    flush_task = asyncio.create_task(
                        self._delayed_flush(self._config.debounce_s),
                    )
                # Drain anything else that's queued without waiting.
                async for event in self._drain_after_flush(flush_task):
                    yield event
        finally:
            if flush_task is not None and not flush_task.done():
                flush_task.cancel()
            # Final flush.
            async for event in self._flush_pending():
                yield event

    async def _drain_after_flush(
        self, flush_task: asyncio.Task[None],
    ) -> AsyncIterator[PageChangeEvent]:
        """Wait for the flush window to elapse, then yield emitted events."""

        try:
            await flush_task
        except asyncio.CancelledError:
            return
        async for event in self._flush_pending():
            yield event

    async def _delayed_flush(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

    async def _flush_pending(self) -> AsyncIterator[PageChangeEvent]:
        async with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for change in pending:
            yield self._to_page_event(change)

    def _record_pending(self, kind: str, path: Path) -> None:
        page_id = self._page_id_for(path.relative_to(self._root))
        existing = self._pending.get(page_id)
        ts = time.monotonic()
        if existing is None:
            self._pending[page_id] = _PendingChange(
                page_id=page_id,
                last_kind=kind,
                first_seen=ts,
                last_seen=ts,
                rel_path=path.relative_to(self._root).as_posix(),
            )
            return
        # Coalesce: a delete-then-create stays as 'modified'; a single
        # delete or create remains; modified beats nothing.
        merged = existing.last_kind
        if kind == "deleted":
            merged = "deleted" if existing.last_kind in ("deleted", "modified") else "deleted"
        elif kind == "created":
            merged = "created" if existing.last_kind == "created" else "modified"
        elif kind == "modified":
            merged = "modified" if existing.last_kind != "deleted" else "deleted"
        existing.last_kind = merged
        existing.last_seen = ts

    def _to_page_event(self, change: "_PendingChange") -> PageChangeEvent:
        if change.last_kind == "deleted":
            return PageChangeEvent.page_invalidated(
                page_id=change.page_id,
                source=self._source_uri,
                reason="source file deleted",
                data_type=self._config.data_type,
                scope_id=self._scope_id,
                extra={"relative_path": change.rel_path},
            )
        if change.last_kind == "created":
            return PageChangeEvent.page_added(
                page_id=change.page_id,
                source=self._source_uri,
                data_type=self._config.data_type,
                scope_id=self._scope_id,
                extra={"relative_path": change.rel_path},
            )
        # modified
        return PageChangeEvent.page_replaced(
            old_page_id=change.page_id,
            new_page_id=change.page_id,
            source=self._source_uri,
            data_type=self._config.data_type,
            scope_id=self._scope_id,
            extra={"relative_path": change.rel_path},
        )

    # ---- Polling fallback ---------------------------------------------

    async def _poll_loop(self) -> AsyncIterator[PageChangeEvent]:
        last_state = self._snapshot_mtimes()
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(
                    self._stopped.wait(), timeout=self._config.poll_interval_s,
                )
                break
            except asyncio.TimeoutError:
                pass
            current_state = self._snapshot_mtimes()
            for rel, mtime in current_state.items():
                prev = last_state.get(rel)
                if prev is None:
                    yield PageChangeEvent.page_added(
                        page_id=self._page_id_for(Path(rel)),
                        source=self._source_uri,
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": rel},
                    )
                elif prev != mtime:
                    pid = self._page_id_for(Path(rel))
                    yield PageChangeEvent.page_replaced(
                        old_page_id=pid,
                        new_page_id=pid,
                        source=self._source_uri,
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": rel},
                    )
            for rel in last_state.keys() - current_state.keys():
                yield PageChangeEvent.page_invalidated(
                    page_id=self._page_id_for(Path(rel)),
                    source=self._source_uri,
                    reason="source file deleted",
                    data_type=self._config.data_type,
                    scope_id=self._scope_id,
                    extra={"relative_path": rel},
                )
            last_state = current_state

    def _snapshot_mtimes(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [
                d for d in dirnames
                if self._path_filter(
                    Path(dirpath, d).relative_to(self._root)
                )
            ]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = full.relative_to(self._root)
                if not self._path_filter(rel):
                    continue
                try:
                    out[rel.as_posix()] = full.stat().st_mtime
                except OSError:
                    continue
        return out


@dataclass
class _PendingChange:
    page_id: str
    last_kind: str
    first_seen: float
    last_seen: float
    rel_path: str


__all__ = ("LocalFsWatcher", "LocalFsWatcherConfig")
