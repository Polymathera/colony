"""End-to-end test for ``DesignMonorepoWatcher``.

Bootstraps a design monorepo, starts the watcher, mutates files, and
verifies that the runtime's dispatch callback fires for a subscription
that matches the resulting event.

This exercises the full Phase C5 → C4 path: file change → LocalFsWatcher
emits PageChangeEvent → publisher writes to ``vcm:page_events:*`` →
runtime's forwarder picks it up → SubscriptionIndex matches →
dispatch_callback fires.

Because we are not running inside a Ray-serving deployment, we run the
``ConvergenceRuntime`` in-process directly and feed events from the
``PageEventPublisher`` via a small relay coroutine (the same
forwarder loop the deployment uses, but without Ray).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.design_monorepo import (
    DesignMonorepoClient,
    DesignMonorepoManifest,
    AgentIdentity,
    bootstrap_design_monorepo,
)
from polymathera.colony.design_monorepo.watcher import DesignMonorepoWatcher
from polymathera.colony.vcm.convergence import (
    ConvergenceRuntime,
    PageMetadataPredicate,
    PageSubscription,
)
from polymathera.colony.vcm.page_events import (
    PAGE_EVENTS_TOPIC_PREFIX,
    PageChangeEvent,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def bootstrapped_repo(tmp_path: Path) -> DesignMonorepoClient:
    manifest = DesignMonorepoManifest(
        tenant="t", colony="c", program="p", target_system="x",
        design_repo_url="file:///dev/null",
    )
    identity = AgentIdentity(agent_id="bootstrap", role="bootstrap", colony_id="c")
    return bootstrap_design_monorepo(
        manifest, tmp_path / "repo", identity=identity,
    )


async def test_local_file_change_fires_subscriber(
    bootstrapped_repo: DesignMonorepoClient,
    tmp_path: Path,
) -> None:
    fired: list[PageChangeEvent] = []

    async def cb(sub, ev):
        fired.append(ev)

    runtime = ConvergenceRuntime(dispatch_callback=cb, rate_burst=10)
    sub = PageSubscription(
        predicate=PageMetadataPredicate(data_type="design_monorepo_file"),
        dispatch_scope="agent_a",
        dispatch_key="convergence:dispatch:a",
        capability_key="Cap",
    )
    runtime.register(sub)

    # The watcher and the relay must share a single in-memory blackboard
    # — different ``EnhancedBlackboard`` instances each get their own
    # backend, so events published by one wouldn't reach a stream open
    # on the other. The test injects one shared blackboard.
    app_name = "design_monorepo_watcher_test"
    bridge = EnhancedBlackboard(
        app_name=app_name, scope_id="colony", backend_type="memory",
    )
    await bridge.initialize()

    watcher = DesignMonorepoWatcher(
        client=bootstrapped_repo,
        app_name=app_name,
        watch_remote=False,
        colony_blackboard=bridge,
    )

    async def relay():
        try:
            async for event in bridge.stream_events(
                pattern=f"{PAGE_EVENTS_TOPIC_PREFIX}:*",
                event_types={"write"},
                until=lambda: stop_relay,
                timeout=0.5,
            ):
                value = event.value
                if not isinstance(value, dict):
                    continue
                try:
                    pce = PageChangeEvent.model_validate(value)
                except Exception:
                    continue
                await runtime.feed_event(pce, source_id="design_monorepo")
        except asyncio.CancelledError:
            return

    stop_relay = False
    relay_task = asyncio.create_task(relay())

    try:
        await watcher.start()
        # Give watchdog a moment to attach its inotify subscription
        # before we touch the working tree.
        await asyncio.sleep(0.3)
        target = bootstrapped_repo.working_dir / "design" / "decisions" / "d1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"decision_id": "d1"}', encoding="utf-8")

        # Wait for the dispatch to land. The watcher's debounce + the
        # forwarder relay introduce a small delay; 5 s is generous.
        deadline = asyncio.get_running_loop().time() + 5.0
        while not fired and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
        assert fired, "expected at least one dispatch"
        assert fired[0].data_type == "design_monorepo_file"
    finally:
        stop_relay = True
        await watcher.stop()
        relay_task.cancel()
        try:
            await relay_task
        except (asyncio.CancelledError, Exception):
            pass
        await bridge.stop()
