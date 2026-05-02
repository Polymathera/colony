"""``ConvergenceRuntimeDeployment`` — Ray-serving wrapper around the runtime.

The deployment exposes the runtime over the standard
``serving.DeploymentHandle`` machinery:

- VCM's ``_start_watch_bridge`` calls ``feed_page_event`` directly via
  this deployment's handle (KERNEL-ring infrastructure path) — no
  intermediate blackboard topic.
- Dispatches go to per-subscriber blackboard scopes; the key shape is
  fixed by ``ConvergenceDispatchProtocol`` (the runtime is the sole
  writer; the subscribing capability is the sole reader and listens
  via ``@event_handler(pattern=ConvergenceDispatchProtocol.dispatch_pattern())``).
- Quiescence events are written onto the colony scope under
  ``ConvergenceQuiescenceProtocol`` so consumers
  (``DesignCheckpointer`` auto-tagging an
  ``auto_quiescence_<iso8601>`` checkpoint, future analogous "react
  when the design has settled" agents) can subscribe via
  ``@event_handler(pattern=ConvergenceQuiescenceProtocol.quiescence_pattern())``.
- ``ConvergenceStatus`` and the change feed are read via
  ``get_status`` / ``get_change_feed`` deployment-handle endpoints —
  the UI panel polls; we don't mirror them onto the blackboard.

The deployment is a *singleton per colony* — there is one
``ConvergenceRuntime`` for the whole multi-agent system. Subscriptions
are not federated across colonies; that's intentional (master §5.3:
the blackboard / VCM are colony-scoped substrates).
"""

from __future__ import annotations

import logging

from ...agents.blackboard import EnhancedBlackboard
from ...agents.blackboard.protocol import ConvergenceQuiescenceProtocol
from ...agents.scopes import BlackboardScope, get_scope_prefix
from ...distributed.ray_utils import serving
from ..page_events import PageChangeEvent
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
    """Ray-serving singleton for the convergence runtime."""

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
        self._edge_reach_resolver: EdgeReachResolver | None = None

    # -- Lifecycle -------------------------------------------------------

    @serving.initialize_deployment
    async def initialize(self) -> None:
        """Bring up the colony-scope blackboard and construct the runtime."""

        self._app_name = serving.get_my_app_name()
        self._colony_blackboard = EnhancedBlackboard(
            app_name=self._app_name,
            scope_id=get_scope_prefix(BlackboardScope.COLONY),
        )
        await self._colony_blackboard.initialize()

        self._runtime = ConvergenceRuntime(
            dispatch_callback=self._dispatch_via_blackboard,
            quiescence_emit_callback=self._emit_quiescence,
            edge_reach_resolver=self._edge_reach_resolver,
            episode_budget=self._episode_budget,
            change_feed_size=self._change_feed_size,
            rate_interval_s=self._rate_interval_s,
            rate_burst=self._rate_burst,
        )
        logger.info(
            "ConvergenceRuntimeDeployment ready (app=%s)", self._app_name,
        )

    @serving.cleanup_deployment
    async def cleanup(self) -> None:
        """Release blackboard handles."""

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

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def feed_page_event(
        self, event: PageChangeEvent, *, source_id: str = "unknown",
    ) -> None:
        """Privileged ingestion path used by ``VirtualContextManager``'s
        watch bridges to feed ``PageChangeEvent``s straight into the
        runtime, bypassing any blackboard-mediated topic."""

        runtime = self._require_runtime()
        await runtime.feed_event(event, source_id=source_id)

    @serving.endpoint
    async def subscribe(
        self,
        *,
        predicate: PageMetadataPredicate,
        dispatch_scope: str,
        tolerance: NumericTolerance | None = None,
        capability_key: str = "",
        agent_id: str | None = None,
    ) -> str:
        """Register a subscription. The dispatch key is derived from
        ``ConvergenceDispatchProtocol`` + the generated subscription_id;
        callers do not pick it."""

        runtime = self._require_runtime()
        sub = PageSubscription(
            predicate=predicate,
            dispatch_scope=dispatch_scope,
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
    async def wait_for_quiescence(self, timeout: float | None = None) -> bool:
        runtime = self._require_runtime()
        return await runtime.wait_for_quiescence(timeout=timeout)

    # -- Internals -------------------------------------------------------

    def _require_runtime(self) -> ConvergenceRuntime:
        if self._runtime is None:
            raise RuntimeError(
                "ConvergenceRuntimeDeployment not yet ready; "
                "wait for the @serving.initialize_deployment hook to run.",
            )
        return self._runtime

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
            raise RuntimeError(
                "Cannot dispatch before the @serving.initialize_deployment "
                "hook has run.",
            )
        bb = EnhancedBlackboard(
            app_name=self._app_name,
            scope_id=scope_id,
        )
        await bb.initialize()
        self._dispatch_blackboards[scope_id] = bb
        return bb

    async def _emit_quiescence(self, counters: ConvergenceCounters) -> None:
        if self._colony_blackboard is None:
            return
        await self._colony_blackboard.write(
            ConvergenceQuiescenceProtocol.quiescence_key(counters.episode_id),
            value=counters.model_dump(mode="json"),
            tags={"convergence", "quiescence"},
        )


__all__ = ("ConvergenceRuntimeDeployment",)
