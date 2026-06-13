"""``MissionStatusCapability`` — coordinator-side narrative updates.

A single-action capability mounted on every mission coordinator. The
coordinator's planner calls ``emit_mission_status(message=...)`` to
publish a one-line narrative ("loading design context...",
"classifying issues...") that the chat UI surfaces in place of an
opaque spinner. The primitive is intentionally narrow:

- ONE action; the LLM authors the message, decides when to emit, and
  whether to emit at all. Per [[primitives-not-pipelines]] no phase
  registry or fixed sequence is baked in.
- Channel is framework-owned: the message is written via
  :class:`MissionStatusProtocol`'s canonical key on the agent's
  SESSION-scoped chat blackboard. The chat router (in ``web_ui/``)
  imports the same protocol from ``agents/blackboard/protocol.py`` —
  the dependency direction is ``web_ui/`` → ``agents/`` (downstream),
  never the reverse.
- ``mission_id`` is framework-known: it is the coordinator agent's own
  ``agent.agent_id`` — the unique identifier of a running mission
  instance. The LLM never threads it. Per
  [[no-llm-facing-framework-state]].
- The primitive itself is decorated ``emits_lifecycle=False`` so the
  ``policy:action_started`` / ``policy:action_completed`` lifecycle
  bridge does NOT produce a spinner row labelled
  "running: emit_mission_status" — that would be the absurd outcome
  the assumptions review correctly flagged.
- Status lifetime is bound to framework events (mission terminal
  state, new mission with the same ``mission_id``), NOT to LLM
  cleanup. The planner is not expected to clear; the chat router
  does, on the boundaries it owns. This is
  [[fix-the-class-not-the-instance]]: the lifecycle invariant lives
  in one place.

The capability is mounted on every coordinator (see e.g.
``ProjectPlanningCoordinator.initialize``) so the coordinator's
dispatcher discovers the action via the usual ``@action_executor``
walk. Mounting on the ``SessionAgent`` is also supported — the
agent's own ``agent_id`` becomes the ``mission_id`` and the UI
groups its narrative under the SessionAgent's row when no
coordinator is in flight.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
from overrides import override

from ...base import AgentCapability
from ...blackboard.protocol import (
    CHAT_BLACKBOARD_NAMESPACE,
    MissionStatusProtocol,
)
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor


if TYPE_CHECKING:
    from ...base import Agent


logger = logging.getLogger(__name__)


class MissionStatusCapability(AgentCapability):
    """Single-action capability that publishes narrative status for
    the mission this agent represents.

    See module docstring for the full design. The single action
    :meth:`emit_mission_status` writes a singleton key on the session-
    scoped chat blackboard via :class:`MissionStatusProtocol`;
    downstream the chat WebSocket relay streams the event to the
    frontend, which renders it in place of the spinner banner.
    """

    def __init__(
        self,
        agent: "Agent",
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = CHAT_BLACKBOARD_NAMESPACE,
        capability_key: str = "mission_status_capability",
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    @action_executor(
        emits_lifecycle=False,
        planning_summary=(
            "Emit a one-line narrative for the chat UI describing what "
            "this mission is currently doing. Does NOT terminate the "
            "turn (cf. respond_to_user). Per-mission singleton — the UI "
            "replaces the prior status with the latest. Use when the "
            "user would otherwise see opaque waiting (loading a large "
            "context, classifying a batch, awaiting an external system)."
        ),
    )
    async def emit_mission_status(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish ``message`` as the current narrative status.

        Args:
            message: Free-text narrative authored by the LLM. The
                frontend renders this verbatim as plain text — no
                markdown, no HTML interpolation. Keep it short
                (a single line).
            details: Optional structured context (e.g. counts of items
                processed). Forwarded to the frontend; intended for
                additive UI affordances. Not retained beyond the
                current singleton; on the next emit, ``details`` is
                replaced wholesale.

        Returns:
            ``{"ok": True, "mission_id": <coordinator agent_id>}``.
        """

        mission_id = self._agent.agent_id
        bb = await self._agent.get_blackboard(scope_id=self.scope_id)
        await bb.write(
            MissionStatusProtocol.status_key(mission_id),
            {
                "mission_id": mission_id,
                "agent_id": self._agent.agent_id,
                "message": message,
                "details": details or {},
                "timestamp": time.time(),
            },
            tags={"mission_status"},
            metadata={"mission_id": mission_id},
        )
        logger.debug(
            "MissionStatusCapability: emitted status for mission %s: %r",
            mission_id, message[:80],
        )
        return {"ok": True, "mission_id": mission_id}

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        """No per-mission state to suspend.

        Mission narrative is a transient signal whose current value
        already lives on the SESSION-scoped blackboard (a separate
        durability surface). The capability itself holds no in-memory
        state across emissions, so a suspend/restore cycle has nothing
        to preserve.
        """

        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        """Companion to :meth:`serialize_suspension_state`; no-op."""

        return None


__all__ = ("MissionStatusCapability",)
