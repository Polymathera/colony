"""``ConvergenceRuntimeDeployment`` — Ray-serving wrapper around the runtime.

The deployment exposes the runtime over the standard
``serving.DeploymentHandle`` machinery and ties it to the colony's
blackboard:

- On startup, subscribes to ``vcm:page_events:*`` on the colony
  scope and forwards events into ``ConvergenceRuntime.feed_event``.
- Dispatches via per-subscriber blackboard scopes (each subscription
  declares its ``dispatch_scope`` + ``dispatch_key``; the deployment
  caches a ``EnhancedBlackboard`` per scope and writes the dispatch
  event there).
- Mirrors ``ConvergenceStatus`` and the change feed onto the colony
  scope so the SessionAgent's UI panel (master §5.4) can render them.

The deployment is a *singleton per colony* — there is one
``ConvergenceRuntime`` for the whole multi-agent system. Subscriptions
are not federated across colonies; that's intentional (master §5.3:
the blackboard / VCM are colony-scoped substrates).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...agents.blackboard import EnhancedBlackboard
from ...agents.scopes import BlackboardScope, get_scope_prefix
from ...distributed.ray_utils import serving
from ..page_events import (
    CONVERGENCE_CHANGE_FEED_KEY,
    CONVERGENCE_DISPATCH_PREFIX,
    CONVERGENCE_QUIESCENCE_TOPIC,
    CONVERGENCE_STATUS_KEY,
    PAGE_EVENTS_TOPIC_PREFIX,
    PageChangeEvent,
)
from .predicates import EdgeReachResolver, PageMetadataPredicate
from .runtime import (
    ChangeFeedEntry,
    ConvergenceCounters,
    ConvergenceRuntime,
    ConvergenceStatus,
)
from .subscriptions import NumericTolerance, PageSubscription


logger = logging.getLogger(__name__)


@serving.deployment
class ConvergenceRuntimeDeployment:
    """Ray-serving singleton for the convergence runtime.

    Lifecycle:

    1. ``on_app_ready`` initialises the colony-scope blackboard,
       constructs the ``ConvergenceRuntime``, and starts the event
       forwarder task.
    2. Capabilities call ``subscribe`` / ``unsubscribe`` to register
       page-graph subscriptions.
    3. Page sources / watchers write events to the
       ``vcm:page_events:*`` topic on the colony scope; the forwarder
       relays each event to ``runtime.feed_event``.
    4. The runtime dispatches matching subscriptions by writing to
       per-subscriber blackboard scopes; subscribers consume via the
       normal ``EnhancedBlackboard.stream_events_to_queue``
       machinery.
    5. ``shutdown`` cancels the forwarder and clears the runtime.
    """

    def __init__(
        self,
        *,
        episode_budget: int = ConvergenceRuntime.DEFAULT_EPISODE_BUDGET,
        change_feed_size: int = ConvergenceRuntime.DEFAULT_CHANGE_FEED_SIZE,
        rate_interval_s: float = ConvergenceRuntime.DEFAULT_RATE_INTERVAL_S,
        rate_burst: int = ConvergenceRuntime.DEFAULT_RATE_BURST,
    ) -> None:
        self._episode_budget = episode_budget
        self._change_feed_size = change_feed_size
        self._rate_interval_s = rate_interval_s
        self._rate_burst = rate_burst

        self._runtime: ConvergenceRuntime | None = None
        self._app_name: str | None = None
        self._colony_blackboard: EnhancedBlackboard | None = None
        self._dispatch_blackboards: dict[str, EnhancedBlackboard] = {}
        self._forwarder_task: asyncio.Task[None] | None = None
        self._stopped = False
        self._edge_reach_resolver: EdgeReachResolver | None = None
        # Live design-monorepo watchers, keyed by absolute working_dir.
        # Owned by this deployment so capabilities calling
        # ``register_design_monorepo`` are idempotent and do not need
        # to track watcher lifecycle themselves.
        self._design_monorepo_watchers: dict[Path, Any] = {}
        self._watcher_lock = asyncio.Lock()

    # -- Lifecycle -------------------------------------------------------

    async def on_app_ready(self, app_name: str) -> None:
        self._app_name = app_name
        self._colony_blackboard = EnhancedBlackboard(
            app_name=app_name,
            scope_id=get_scope_prefix(BlackboardScope.COLONY),
        )
        await self._colony_blackboard.initialize()

        self._runtime = ConvergenceRuntime(
            dispatch_callback=self._dispatch_via_blackboard,
            status_emit_callback=self._emit_status,
            quiescence_emit_callback=self._emit_quiescence,
            change_feed_emit_callback=self._emit_change_feed,
            edge_reach_resolver=self._edge_reach_resolver,
            episode_budget=self._episode_budget,
            change_feed_size=self._change_feed_size,
            rate_interval_s=self._rate_interval_s,
            rate_burst=self._rate_burst,
        )
        self._forwarder_task = asyncio.create_task(
            self._forward_page_events(),
            name="convergence-runtime-forwarder",
        )
        logger.info("ConvergenceRuntimeDeployment ready (app=%s)", app_name)

    async def shutdown(self) -> None:
        self._stopped = True
        async with self._watcher_lock:
            for working_dir, watcher in list(self._design_monorepo_watchers.items()):
                try:
                    await watcher.aclose()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "design-monorepo watcher stop failed for %s", working_dir,
                    )
            self._design_monorepo_watchers.clear()
        if self._forwarder_task is not None:
            self._forwarder_task.cancel()
            try:
                await self._forwarder_task
            except asyncio.CancelledError:
                pass
            self._forwarder_task = None
        for bb in self._dispatch_blackboards.values():
            try:
                await bb.stop()
            except Exception:  # noqa: BLE001
                logger.exception("dispatch blackboard stop failed")
        self._dispatch_blackboards.clear()
        if self._colony_blackboard is not None:
            try:
                await self._colony_blackboard.stop()
            except Exception:  # noqa: BLE001
                logger.exception("colony blackboard stop failed")
            self._colony_blackboard = None

    # -- Endpoints -------------------------------------------------------

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def is_ready(self) -> bool:
        return self._runtime is not None

    @serving.endpoint
    async def subscribe(
        self,
        *,
        predicate: PageMetadataPredicate,
        dispatch_scope: str,
        dispatch_key: str | None = None,
        declared_outputs: list[PageMetadataPredicate] | None = None,
        tolerance: NumericTolerance | None = None,
        capability_key: str = "",
        agent_id: str | None = None,
    ) -> str:
        runtime = self._require_runtime()
        sub = PageSubscription(
            predicate=predicate,
            dispatch_scope=dispatch_scope,
            dispatch_key=(
                dispatch_key
                or f"{CONVERGENCE_DISPATCH_PREFIX}:{capability_key or 'default'}"
            ),
            declared_outputs=tuple(declared_outputs or ()),
            tolerance=tolerance,
            capability_key=capability_key,
            agent_id=agent_id,
        )
        return runtime.register(sub)

    @serving.endpoint
    async def unsubscribe(self, subscription_id: str) -> bool:
        runtime = self._require_runtime()
        return runtime.unregister(subscription_id)

    @serving.endpoint
    async def get_subscription(
        self, subscription_id: str,
    ) -> PageSubscription | None:
        runtime = self._require_runtime()
        return runtime.get(subscription_id)

    @serving.endpoint
    async def dispatch_change(
        self, event: PageChangeEvent, *, source_id: str = "manual",
    ) -> None:
        runtime = self._require_runtime()
        await runtime.dispatch_change(event, source_id=source_id)

    @serving.endpoint
    async def get_status(self) -> ConvergenceStatus:
        runtime = self._require_runtime()
        return runtime.get_status()

    @serving.endpoint
    async def get_change_feed(self, limit: int = 50) -> list[ChangeFeedEntry]:
        runtime = self._require_runtime()
        return runtime.get_change_feed(limit)

    @serving.endpoint
    async def detect_cycle(self) -> bool:
        runtime = self._require_runtime()
        return runtime.detect_cycle()

    @serving.endpoint
    async def wait_for_quiescence(self, timeout: float | None = None) -> bool:
        runtime = self._require_runtime()
        return await runtime.wait_for_quiescence(timeout=timeout)

    # -- Live-source registration ---------------------------------------

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def register_design_monorepo(
        self,
        working_dir: str,
        *,
        watch_remote: bool = True,
    ) -> bool:
        """Idempotently start a ``DesignMonorepoWatcher`` for the repo
        rooted at ``working_dir``.

        Called by ``_DesignMonorepoCapabilityBase`` on every capability
        init; the dedup keyed on absolute ``working_dir`` makes this
        safe to call from N agents at once. Returns ``True`` if a new
        watcher was started, ``False`` if one was already running.

        The watcher publishes to the colony scope's
        ``vcm:page_events:*`` topic — the same topic this runtime's
        forwarder consumes — so the dispatch loop closes locally
        without further wiring.
        """

        from ...design_monorepo import DesignMonorepoClient
        from ...design_monorepo.watcher import DesignMonorepoWatcher

        path = Path(working_dir).resolve()
        async with self._watcher_lock:
            if path in self._design_monorepo_watchers:
                return False
            if self._app_name is None:
                raise RuntimeError(
                    "register_design_monorepo before on_app_ready",
                )
            client = await asyncio.to_thread(DesignMonorepoClient.open, path)
            watcher = DesignMonorepoWatcher(
                client=client,
                app_name=self._app_name,
                watch_remote=watch_remote,
                colony_blackboard=self._colony_blackboard,
            )
            await watcher.start()
            self._design_monorepo_watchers[path] = watcher
            logger.info(
                "ConvergenceRuntimeDeployment: registered design-monorepo "
                "watcher for %s (scope=%s)",
                path, watcher.scope_id,
            )
            return True

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def unregister_design_monorepo(self, working_dir: str) -> bool:
        """Stop the watcher for ``working_dir`` if one is running.

        Idempotent: returns ``True`` if a watcher was stopped, ``False``
        if none was registered."""

        path = Path(working_dir).resolve()
        async with self._watcher_lock:
            watcher = self._design_monorepo_watchers.pop(path, None)
        if watcher is None:
            return False
        try:
            await watcher.aclose()
        except Exception:  # noqa: BLE001
            logger.exception(
                "design-monorepo watcher stop failed for %s", path,
            )
        return True

    # -- Internals -------------------------------------------------------

    def _require_runtime(self) -> ConvergenceRuntime:
        if self._runtime is None:
            raise RuntimeError(
                "ConvergenceRuntimeDeployment not yet ready; "
                "call on_app_ready first.",
            )
        return self._runtime

    async def _forward_page_events(self) -> None:
        """Long-running task: stream page-events from the colony
        blackboard into the runtime."""

        if self._colony_blackboard is None:
            return
        try:
            async for event in self._colony_blackboard.stream_events(
                pattern=f"{PAGE_EVENTS_TOPIC_PREFIX}:*",
                event_types={"write"},
                until=lambda: self._stopped,
                timeout=1.0,
            ):
                if self._stopped:
                    break
                value = event.value
                if not isinstance(value, dict):
                    # The blackboard transport serialises pydantic
                    # models as dicts; reconstruct.
                    continue
                try:
                    pce = PageChangeEvent.model_validate(value)
                except Exception:  # noqa: BLE001 - tolerate malformed payloads
                    logger.debug(
                        "Skipping malformed PageChangeEvent at %s",
                        event.key,
                    )
                    continue
                source_id = self._extract_source_id(event.key)
                runtime = self._require_runtime()
                try:
                    await runtime.feed_event(pce, source_id=source_id)
                except Exception:  # noqa: BLE001
                    logger.exception("runtime.feed_event failed")
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.exception("convergence forwarder loop crashed")

    @staticmethod
    def _extract_source_id(event_key: str) -> str:
        # event_key format: 'vcm:page_events:<source_id>:<kind>'
        prefix = f"{PAGE_EVENTS_TOPIC_PREFIX}:"
        if not event_key.startswith(prefix):
            return "unknown"
        rest = event_key[len(prefix):]
        return rest.split(":", 1)[0] if rest else "unknown"

    async def _dispatch_via_blackboard(
        self, sub: PageSubscription, event: PageChangeEvent,
    ) -> None:
        bb = await self._get_dispatch_blackboard(sub.dispatch_scope)
        await bb.write(
            sub.dispatch_key,
            value=event.model_dump(mode="json"),
            tags={"convergence", "dispatch"},
            metadata={"subscription_id": sub.subscription_id},
        )

    async def _get_dispatch_blackboard(self, scope_id: str) -> EnhancedBlackboard:
        bb = self._dispatch_blackboards.get(scope_id)
        if bb is not None:
            return bb
        if self._app_name is None:
            raise RuntimeError("Cannot dispatch before on_app_ready.")
        bb = EnhancedBlackboard(
            app_name=self._app_name,
            scope_id=scope_id,
        )
        await bb.initialize()
        self._dispatch_blackboards[scope_id] = bb
        return bb

    async def _emit_status(self, status: ConvergenceStatus) -> None:
        if self._colony_blackboard is None:
            return
        await self._colony_blackboard.write(
            CONVERGENCE_STATUS_KEY,
            value=status.model_dump(mode="json"),
            tags={"convergence", "status"},
        )

    async def _emit_quiescence(self, counters: ConvergenceCounters) -> None:
        if self._colony_blackboard is None:
            return
        await self._colony_blackboard.write(
            f"{CONVERGENCE_QUIESCENCE_TOPIC}:{counters.episode_id}",
            value=counters.model_dump(mode="json"),
            tags={"convergence", "quiescence"},
        )

    async def _emit_change_feed(self, entries: list[ChangeFeedEntry]) -> None:
        if self._colony_blackboard is None:
            return
        await self._colony_blackboard.write(
            CONVERGENCE_CHANGE_FEED_KEY,
            value=[e.model_dump(mode="json") for e in entries],
            tags={"convergence", "change_feed"},
        )


__all__ = ("ConvergenceRuntimeDeployment",)
