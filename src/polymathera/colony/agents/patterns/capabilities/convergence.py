"""``ConvergenceCapability`` — agent-facing surface for the convergence runtime.

Per master §3.4 (Layer 3 capabilities), the runtime exposes:

- ``subscribe_pattern(predicate, ...)`` — register a page-graph
  subscription. The runtime writes the dispatch event onto this
  capability's blackboard scope under
  ``ConvergenceDispatchProtocol.dispatch_key(subscription_id)``; the
  capability's ``@event_handler`` (below) picks it up and surfaces it
  as planner context.
- ``unsubscribe(subscription_id)`` — drop a subscription.
- ``dispatch_change(event)`` — manually emit a page-change event into
  the runtime (used by tests and by capabilities that synthesise
  changes server-side, e.g., a deduplication step that emits
  ``PageGraphEdgeRemoved`` after detecting a stale citation).
- ``wait_for_quiescence(timeout)`` — block until the runtime is
  ``converged``.
- ``get_convergence_status()`` — current state + counters.
- ``get_change_feed(limit)`` — recent dispatches (master §5.4).

The capability resolves the ``ConvergenceRuntimeDeployment`` lazily on
first call via the ``get_convergence_runtime`` helper.
"""

from __future__ import annotations

import logging
from typing import Any

from overrides import override

from ...blackboard import (
    BlackboardEvent,
    ConvergenceDispatchProtocol,
    ConvergenceQuiescenceProtocol,
)
from ...models import AgentSuspensionState
from ...base import Agent, AgentCapability
from ..actions import action_executor
from ..events import EventProcessingResult, event_handler

from polymathera.colony.vcm.convergence import (
    ChangeFeedEntry,
    ConvergenceCounters,
    ConvergenceStatus,
    NumericTolerance,
    PageMetadataPredicate,
    PageSubscription,
)
from polymathera.colony.vcm.page_events import PageChangeEvent


logger = logging.getLogger(__name__)


class ConvergenceCapability(AgentCapability):
    """Agent-facing convergence-runtime surface.

    The capability is the standard way for any agent to participate in
    the always-live design context: declare what page-graph patterns
    you care about, fire on changes, react to cycles, wait for
    quiescence.

    The capability tracks its own subscription ids so a clean shutdown
    automatically unregisters them — agents that suspend or terminate
    do not leak subscriptions.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        input_patterns: list[str] | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        self._owned_subscription_ids: set[str] = set()

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"convergence", "design_state"})

    # ---- Subscription management --------------------------------------

    @action_executor(
        planning_summary=(
            "Subscribe to a page-graph pattern. The runtime will fire "
            "a dispatch on this capability's scope when the pattern "
            "matches; the dispatch is surfaced to the planner as a "
            "convergence:dispatch:{subscription_id} context entry."
        ),
    )
    async def subscribe_pattern(
        self,
        predicate: PageMetadataPredicate,
        *,
        tolerance: NumericTolerance | None = None,
    ) -> str:
        """Register a subscription on the convergence runtime.

        Returns the subscription id; pass it to ``unsubscribe`` to drop
        the registration. The dispatch key is owned by
        ``ConvergenceDispatchProtocol`` — callers do not pick it.
        """

        rt = await self._handle()
        sub_id = await rt.subscribe(
            predicate=predicate,
            dispatch_scope=self.scope_id,
            tolerance=tolerance,
            capability_key=self.capability_key,
            agent_id=self._agent.agent_id if self._agent is not None else None,
        )
        self._owned_subscription_ids.add(sub_id)
        return sub_id

    @action_executor(planning_summary="Drop a previously-registered subscription.")
    async def unsubscribe(self, subscription_id: str) -> bool:
        rt = await self._handle()
        ok = await rt.unsubscribe(subscription_id)
        self._owned_subscription_ids.discard(subscription_id)
        return ok

    @action_executor(
        planning_summary="Manually emit a PageChangeEvent into the runtime.",
    )
    async def dispatch_change(
        self, event: PageChangeEvent, *, source_id: str = "manual",
    ) -> None:
        rt = await self._handle()
        await rt.dispatch_change(event, source_id=source_id)

    # ---- Status surfaces ----------------------------------------------

    @action_executor(planning_summary="Snapshot the convergence runtime's state.")
    async def get_convergence_status(self) -> ConvergenceStatus:
        rt = await self._handle()
        return await rt.get_status()

    @action_executor(
        planning_summary="Return the most-recent N dispatches in the change feed.",
    )
    async def get_change_feed(self, limit: int = 50) -> list[ChangeFeedEntry]:
        rt = await self._handle()
        return await rt.get_change_feed(limit)

    # ---- Receive side -------------------------------------------------

    @event_handler(pattern=ConvergenceDispatchProtocol.dispatch_pattern())
    async def _on_dispatch(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> EventProcessingResult | None:
        """Translate a runtime dispatch into planner context.

        The runtime writes one dispatch per matched subscription onto
        this capability's blackboard scope under
        ``ConvergenceDispatchProtocol.dispatch_key(subscription_id)``.
        We only react to dispatches that target subscriptions this
        capability owns; everything else is left for sibling
        capabilities (or simply ignored).
        """

        try:
            subscription_id = ConvergenceDispatchProtocol.parse_dispatch_key(
                event.key,
            )
        except ValueError:
            return None
        if subscription_id not in self._owned_subscription_ids:
            return None
        if not isinstance(event.value, dict):
            return None
        try:
            page_event = PageChangeEvent.model_validate(event.value)
        except Exception:  # noqa: BLE001
            logger.warning(
                "ConvergenceCapability: dropping malformed dispatch %s",
                event.key,
            )
            return None
        return EventProcessingResult(
            context_key=event.key,
            context={
                "subscription_id": subscription_id,
                "page_event": page_event.model_dump(mode="json"),
            },
        )

    @event_handler(pattern=ConvergenceQuiescenceProtocol.quiescence_pattern())
    async def _on_quiescence(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> EventProcessingResult | None:
        """Surface a quiescence event as planner context so the agent
        can react when the design state has just settled."""

        try:
            episode_id = ConvergenceQuiescenceProtocol.parse_quiescence_key(
                event.key,
            )
        except ValueError:
            return None
        counters: ConvergenceCounters | None = None
        if isinstance(event.value, dict):
            try:
                counters = ConvergenceCounters.model_validate(event.value)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "ConvergenceCapability: malformed quiescence payload at %s",
                    event.key,
                )
        return EventProcessingResult(
            context_key=event.key,
            context={
                "episode_id": episode_id,
                "counters": (
                    counters.model_dump(mode="json") if counters is not None
                    else None
                ),
            },
        )

    # ---- Suspension hooks ---------------------------------------------

    _CUSTOM_DATA_KEY = "convergence_capability"

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        # Save subscription ids so the agent can re-register / clean up
        # on resume. The subscription records themselves live in the
        # runtime; the capability tracks ids for shutdown discipline.
        if self._owned_subscription_ids:
            state.custom_data[self._CUSTOM_DATA_KEY] = {
                "owned_subscription_ids": sorted(self._owned_subscription_ids),
            }
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        payload = state.custom_data.get(self._CUSTOM_DATA_KEY) or {}
        ids = payload.get("owned_subscription_ids") or []
        if ids:
            self._owned_subscription_ids.update(ids)

    async def shutdown(self) -> None:
        """Drop all subscriptions this capability owns. Idempotent."""

        if not self._owned_subscription_ids:
            return
        rt = await self._handle()
        for sub_id in list(self._owned_subscription_ids):
            try:
                await rt.unsubscribe(sub_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "ConvergenceCapability: unsubscribe %s failed", sub_id,
                )
            self._owned_subscription_ids.discard(sub_id)

    # ---- Internal -----------------------------------------------------

    async def _handle(self):
        from polymathera.colony.system import get_vcm

        return await get_vcm(self._app_name)


__all__ = ("ConvergenceCapability",)
