"""End-to-end smoke tests for the always-live design context chain.

These verify that the four layers documented in
``colony/docs/architecture/live-context.md`` connect:

    Watcher → PageChangeEvent → ConvergenceRuntime.feed_event →
    SubscriptionIndex.match → dispatch_callback fires.

The deployment transport (VCM's watch bridge calling the runtime
deployment's ``feed_page_event`` endpoint) is bypassed here — events
go directly from the watcher into ``feed_event``. The deployment is
independently tested, and exercising it requires Ray + Redis. What's
important here is that the contract between layers holds: a
watcher's event reaches the runtime untouched, and a matching
subscription fires.

The chain is the master design doc §5 promise. Running this test on
the wired-up tree proves we're past "infrastructure exists in
isolation" and at "infrastructure delivers a dispatch."
"""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path

import pytest

from polymathera.colony.distributed.ray_utils.serving import (
    Ring,
    execution_context,
)
from polymathera.colony.vcm.convergence import (
    PageMetadataPredicate,
    PageSubscription,
)
from polymathera.colony.vcm.page_events import PageChangeEvent
from polymathera.colony.vcm.watchers import (
    LocalFsWatcher,
    LocalFsWatcherConfig,
    SourcePollWatcher,
    SourcePollWatcherConfig,
)
from polymathera.colony.vcm.sources.context_page_source import (
    ContextPageSource,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_context():
    """``ContextPageSource.__init__`` requires an execution context.
    The test harness provides a USER-ring stub."""

    with execution_context(
        ring=Ring.USER, colony_id="c", tenant_id="t",
        session_id="sess", run_id="run", origin="test",
    ) as ctx:
        yield ctx


@pytest.fixture(autouse=True)
def _force_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide ``watchdog`` so the LocalFsWatcher exercises its polling
    fallback. Inotify behaviour varies across CI runners; the polling
    path is identical in observable contract and entirely
    deterministic."""

    real_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name.startswith("watchdog"):
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


# ---------------------------------------------------------------------------
# Layer 1 (LocalFsWatcher) → Layer 3 (ConvergenceRuntime)
# ---------------------------------------------------------------------------


async def test_local_fs_change_dispatches_subscription(
    convergence_runtime, tmp_path: Path,
) -> None:
    """A file write under a watched root reaches a matching subscription."""

    runtime = convergence_runtime
    fired_events: list[PageChangeEvent] = []

    async def cb(sub, ev):
        fired_events.append(ev)

    runtime._dispatch_via_blackboard = cb
    await runtime.register(
        PageSubscription(
            predicate=PageMetadataPredicate(data_type="design_monorepo_file"),
            dispatch_scope="capability:test",
            capability_key="TestCap",
        ),
    )

    watcher = LocalFsWatcher(
        root=tmp_path,
        scope_id="program:test",
        source_uri="git:test:main:HEAD",
        config=LocalFsWatcherConfig(
            poll_interval_s=0.05,
            debounce_s=0.05,
            data_type="design_monorepo_file",
        ),
    )

    async def pump() -> None:
        async for event in watcher.watch():
            await runtime.feed_event(event, source_id="program:test")

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0.1)
    (tmp_path / "design_decision.md").write_text("d=1\n", encoding="utf-8")
    # Give the polling fallback two cycles + the debounce window plus a
    # cushion so the event traverses watcher → runtime → callback.
    await asyncio.sleep(0.5)
    watcher.stop()
    try:
        await asyncio.wait_for(pump_task, timeout=0.5)
    except asyncio.TimeoutError:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass

    matching = [
        e for e in fired_events
        if e.extra.get("relative_path") == "design_decision.md"
    ]
    assert matching, (
        "subscription matching data_type=design_monorepo_file did not fire "
        f"for the file write; saw {len(fired_events)} events: "
        f"{[(e.kind.value, e.data_type, e.extra) for e in fired_events]}"
    )


async def test_unmatched_data_type_does_not_dispatch(
    convergence_runtime, tmp_path: Path,
) -> None:
    """A file write whose data_type does not match the subscription's
    predicate does NOT trigger a dispatch — this is the runtime's
    job and its test for the watcher chain."""

    runtime = convergence_runtime
    fired: list[str] = []

    async def cb(sub, ev):
        fired.append(sub.subscription_id)

    runtime._dispatch_via_blackboard = cb
    await runtime.register(
        PageSubscription(
            predicate=PageMetadataPredicate(data_type="paper_section"),
            dispatch_scope="capability:test",
            capability_key="TestCap",
        ),
    )

    watcher = LocalFsWatcher(
        root=tmp_path,
        scope_id="program:test",
        source_uri="git:test:main:HEAD",
        config=LocalFsWatcherConfig(
            poll_interval_s=0.05,
            debounce_s=0.05,
            data_type="design_monorepo_file",
        ),
    )

    async def pump() -> None:
        async for event in watcher.watch():
            await runtime.feed_event(event, source_id="program:test")

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0.1)
    (tmp_path / "x.md").write_text("x\n", encoding="utf-8")
    await asyncio.sleep(0.4)
    watcher.stop()
    try:
        await asyncio.wait_for(pump_task, timeout=0.5)
    except asyncio.TimeoutError:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass

    assert fired == [], (
        "subscription whose predicate does not match the watcher's "
        "data_type fired anyway"
    )


# ---------------------------------------------------------------------------
# Layer 1 (SourcePollWatcher) → Layer 3 — proves the generic-source path
# ---------------------------------------------------------------------------


class _ProgrammableSource(ContextPageSource):
    """Minimal in-memory ``ContextPageSource`` whose
    ``get_all_mapped_pages`` is a settable dict — ``SourcePollWatcher``
    diffs successive snapshots into ``PageChangeEvent``s."""

    static = False

    def __init__(self, scope_id: str = "smoke") -> None:
        from polymathera.colony.vcm.models import MmapConfig
        super().__init__(scope_id=scope_id, mmap_config=MmapConfig())
        self._pages: dict[str, list[str]] = {}

    def set_pages(self, pages: dict[str, list[str]]) -> None:
        self._pages = dict(pages)

    async def initialize(self) -> None:
        return None

    async def claim_orphaned_events(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def get_page_id_for_record(self, record_id: str) -> str | None:
        for page_id, records in self._pages.items():
            if record_id in records:
                return page_id
        return None

    async def get_record_ids_for_page(self, page_id: str) -> list[str]:
        return list(self._pages.get(page_id, ()))

    async def get_all_mapped_records(self) -> dict[str, str]:
        return {
            record_id: page_id
            for page_id, records in self._pages.items()
            for record_id in records
        }

    async def get_all_mapped_pages(self) -> dict[str, list[str]]:
        return {pid: list(rs) for pid, rs in self._pages.items()}


async def test_watch_method_is_the_uniform_contract(
    convergence_runtime, _exec_context,
) -> None:
    """A non-static ``ContextPageSource`` exposes its mutations via
    ``watch()`` — the formal ABC contract. This test demonstrates the
    shape ``VirtualContextManager._start_watch_bridge`` consumes:

        async for event in source.watch():
            await convergence_runtime.feed_page_event(event=event, ...)

    A real bridge calls the ``ConvergenceRuntimeDeployment``'s
    KERNEL-ring ``feed_page_event`` endpoint via its deployment
    handle. This test bypasses the deployment hop (which requires
    Ray + Redis) and feeds events directly into ``feed_event`` on the
    runtime instance.

    The point: every layer above the source agrees on ``watch()`` as
    the contract, regardless of whether the source achieves it via a
    background-task design (``LocalFsWatcher``-style) or by reusing
    its existing event loop (``BlackboardContextPageSource``-style).
    """

    from polymathera.colony.vcm.models import MmapConfig

    class _LiveDictSource(ContextPageSource):
        """Minimal non-static ``ContextPageSource`` whose mutations
        are pushed onto an internal queue and yielded from
        ``watch()``."""

        static = False

        def __init__(self) -> None:
            super().__init__(scope_id="live", mmap_config=MmapConfig())
            self._queue: asyncio.Queue[PageChangeEvent] = asyncio.Queue()
            self._pages: dict[str, list[str]] = {}

        async def initialize(self) -> None:
            return None

        async def claim_orphaned_events(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

        async def get_page_id_for_record(self, record_id: str) -> str | None:
            return None

        async def get_record_ids_for_page(self, page_id: str) -> list[str]:
            return []

        async def get_all_mapped_records(self) -> dict[str, str]:
            return {}

        async def get_all_mapped_pages(self) -> dict[str, list[str]]:
            return dict(self._pages)

        async def add_page(self, page_id: str, record_ids: list[str]) -> None:
            self._pages[page_id] = list(record_ids)
            await self._queue.put(
                PageChangeEvent.page_added(
                    page_id=page_id,
                    source="live:test",
                    scope_id=self.scope_id,
                    data_type="design_decision",
                ),
            )

        async def watch(self):
            while True:
                try:
                    event = await self._queue.get()
                except asyncio.CancelledError:
                    return
                yield event

    src = _LiveDictSource()
    runtime = convergence_runtime
    fired: list[PageChangeEvent] = []

    async def cb(sub, ev):
        fired.append(ev)

    runtime._dispatch_via_blackboard = cb
    await runtime.register(
        PageSubscription(
            predicate=PageMetadataPredicate(data_type="design_decision"),
            dispatch_scope="capability:test",
            capability_key="TestCap",
        ),
    )

    # Stand in for VCM._start_watch_bridge: drain watch() and feed
    # the runtime. A real bridge calls the runtime deployment's
    # ``feed_page_event`` endpoint; the contract is identical.
    bridge_task = asyncio.create_task(_run_bridge(src, runtime))

    await src.add_page("page-1", ["r1"])
    await src.add_page("page-2", ["r2"])

    for _ in range(20):
        if len(fired) >= 2:
            break
        await asyncio.sleep(0.01)

    bridge_task.cancel()
    try:
        await bridge_task
    except asyncio.CancelledError:
        pass

    page_ids = {e.page_id for e in fired}
    assert page_ids == {"page-1", "page-2"}, (
        f"watch()-driven dispatch did not deliver both events; "
        f"got {[(e.kind.value, e.page_id) for e in fired]}"
    )


async def _run_bridge(src, runtime) -> None:
    """Same shape as ``VirtualContextManager._start_watch_bridge``,
    minus the ``ConvergenceRuntimeDeployment.feed_page_event``
    deployment hop (the deployment-handle path is tested in the
    runtime deployment's own integration tests)."""

    async for event in src.watch():
        await runtime.feed_event(event, source_id=src.scope_id)


async def test_source_poll_watcher_dispatches_through_runtime(
    convergence_runtime, _exec_context,
) -> None:
    """SourcePollWatcher snapshots → diff → ConvergenceRuntime → fire."""

    src = _ProgrammableSource()
    src.set_pages({"page-a": ["r1"]})
    runtime = convergence_runtime
    runtime._rate_burst = 16  # widen so multiple events in the window pass
    fired: list[PageChangeEvent] = []

    async def cb(sub, ev):
        fired.append(ev)

    runtime._dispatch_via_blackboard = cb
    await runtime.register(
        PageSubscription(
            predicate=PageMetadataPredicate(),  # match anything
            dispatch_scope="capability:test",
            capability_key="TestCap",
        ),
    )

    watcher = SourcePollWatcher(
        source=src,
        scope_id="smoke",
        source_uri="api:smoke",
        config=SourcePollWatcherConfig(poll_interval_s=0.05),
    )

    async def pump() -> None:
        async for event in watcher.watch():
            await runtime.feed_event(event, source_id="smoke")

    pump_task = asyncio.create_task(pump())
    await asyncio.sleep(0.08)
    src.set_pages({"page-a": ["r1"], "page-b": ["r2", "r3"]})
    await asyncio.sleep(0.2)
    watcher.stop()
    try:
        await asyncio.wait_for(pump_task, timeout=0.3)
    except asyncio.TimeoutError:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass

    added = [e for e in fired if e.kind.value == "page_added"]
    assert any(e.page_id == "page-b" for e in added), (
        f"expected a page_added event for 'page-b'; saw {fired}"
    )
