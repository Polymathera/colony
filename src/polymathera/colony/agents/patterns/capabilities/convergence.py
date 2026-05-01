"""``ConvergenceCapability`` — agent-facing surface for the convergence runtime.

Per master §3.4 (Layer 3 capabilities), the runtime exposes four
agent-facing primitives plus the user-visible surfaces from §5.4:

- ``subscribe_pattern(predicate, dispatch_key, ...)`` — register a
  page-graph subscription. The runtime fires the dispatch event on the
  capability's blackboard scope; the capability picks it up via its
  normal ``@event_handler`` infrastructure.
- ``unsubscribe(subscription_id)`` — drop a subscription.
- ``dispatch_change(event)`` — manually emit a page-change event into
  the runtime (used by tests and by capabilities that synthesise
  changes server-side, e.g., a deduplication step that emits
  ``PageGraphEdgeRemoved`` after detecting a stale citation).
- ``wait_for_quiescence(timeout)`` — block until the runtime is
  ``converged``.
- ``get_convergence_status()`` — current state + counters.
- ``get_change_feed(limit)`` — recent dispatches (master §5.4).
- ``detect_cycle()`` — true while a cycle break is in flight in the
  current episode.

The capability resolves the ``ConvergenceRuntimeDeployment`` lazily on
first call via the ``get_convergence_runtime`` helper. Agents do not
need to construct a deployment handle by hand.
"""

from __future__ import annotations

import logging
from typing import Any

from overrides import override

from ...models import AgentSuspensionState
from ...base import Agent, AgentCapability
from ..actions import action_executor

from polymathera.colony.vcm.convergence import (
    ChangeFeedEntry,
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
        planning_summary="Subscribe to a page-graph pattern; runtime fires "
        "the dispatch_key on this scope when the pattern matches.",
    )
    async def subscribe_pattern(
        self,
        predicate: PageMetadataPredicate,
        dispatch_key: str,
        *,
        declared_outputs: list[PageMetadataPredicate] | None = None,
        tolerance: NumericTolerance | None = None,
    ) -> str:
        """Register a subscription on the convergence runtime.

        ``dispatch_key`` is a key within the *capability's* blackboard
        scope; the runtime writes the dispatch event there, and the
        capability's existing ``@event_handler`` machinery picks it up.

        Returns the subscription id; pass it to ``unsubscribe`` to drop
        the registration.
        """

        rt = await self._handle()
        sub_id = await rt.subscribe(
            predicate=predicate,
            dispatch_scope=self.scope_id,
            dispatch_key=dispatch_key,
            declared_outputs=declared_outputs,
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

    @action_executor(planning_summary="Block until the runtime is converged.")
    async def wait_for_quiescence(self, timeout: float | None = None) -> bool:
        rt = await self._handle()
        return await rt.wait_for_quiescence(timeout=timeout)

    @action_executor(
        planning_summary="True while the current episode broke a cycle.",
    )
    async def detect_cycle(self) -> bool:
        rt = await self._handle()
        return await rt.detect_cycle()

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
        from polymathera.colony.system import get_convergence_runtime

        return get_convergence_runtime(self._app_name)


__all__ = ("ConvergenceCapability",)
