"""``ProjectPlanningMissionControlCapability`` — mission-control
primitives specific to the ``project_planning`` mission's decompose
mode.

Owns the typed ``request_decompose_early_stop`` primitive and the
canonical early-stop blackboard key format that the
:class:`DecomposeCompletionValidator` reads. Both producers and the
consumer live in ``agents/missions/project_planning/`` — no
backwards layer dependency.

The capability is mounted on :class:`ProjectPlanningCoordinator` (and
ONLY on that coordinator, since the primitives are
project-planning-specific). It is NOT a generic design-process
capability — early-stop is a mission-control concern, not a
design-process operation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from overrides import override

from ...base import AgentCapability
from ...blackboard.protocol import BlackboardProtocol
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ...patterns.actions import action_executor

from typing import ClassVar


if TYPE_CHECKING:
    from ...base import Agent


logger = logging.getLogger(__name__)


class DecomposeEarlyStopProtocol(BlackboardProtocol):
    """Typed early-stop signal for one decompose mission instance.

    Producer: :meth:`ProjectPlanningMissionControlCapability.request_decompose_early_stop`.
    Consumer: :class:`DecomposeCompletionValidator`.

    Singleton per coordinator ``agent_id`` (the running coordinator IS
    the mission instance). Two simultaneous decompose missions on the
    same session do not see each other's early-stop signals because
    the key is keyed by ``agent_id``, not by ``session_id``.

    Scope: SESSION — mirrors the mission's natural lifetime.

    Key shape: ``mission:project_planning:decompose:early_stop:<agent_id>``.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    _PREFIX = "mission:project_planning:decompose:early_stop:"

    @staticmethod
    def signal_key(agent_id: str) -> str:
        return f"{DecomposeEarlyStopProtocol._PREFIX}{agent_id}"

    @staticmethod
    def signal_pattern(agent_id: str | None = None) -> str:
        if agent_id:
            return f"{DecomposeEarlyStopProtocol._PREFIX}{agent_id}"
        return f"{DecomposeEarlyStopProtocol._PREFIX}*"

    @staticmethod
    def parse_signal_key(key: str) -> str:
        if not key.startswith(DecomposeEarlyStopProtocol._PREFIX):
            raise ValueError(
                f"Not a DecomposeEarlyStopProtocol key: {key!r}",
            )
        return key[len(DecomposeEarlyStopProtocol._PREFIX):]


class ProjectPlanningMissionControlCapability(AgentCapability):
    """Mission-control primitives for the ``project_planning`` mission.

    Currently exposes one action: :meth:`request_decompose_early_stop`.
    Future mission-control primitives (cancel-with-summary, pause,
    etc.) belong here too.
    """

    def __init__(
        self,
        agent: "Agent",
        scope: BlackboardScope = BlackboardScope.SESSION,
        capability_key: str = "project_planning_mission_control",
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent),
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    @action_executor(
        planning_summary=(
            "Record an explicit user instruction to stop the decompose "
            "mission early with remaining in-scope issues unaddressed. "
            "Writes a typed early-stop record the "
            "DecomposeCompletionValidator reads as drain. Requires a "
            "verbatim user quote authorising the stop — the LLM cannot "
            "self-certify. The framework records the quote for audit. "
            "Use when the user has explicitly said the partial outcome "
            "is acceptable (e.g. 'stop after the first 3 — I'll handle "
            "the rest manually'); otherwise continue draining the "
            "in-scope backlog."
        ),
    )
    async def request_decompose_early_stop(
        self,
        *,
        user_acknowledgement_quote: str,
        remaining_in_scope: list[int],
    ) -> dict[str, Any]:
        """Record an explicit early-stop signal that the
        :class:`DecomposeCompletionValidator` reads as drain.

        Per [[no-llm-facing-framework-state]] the primitive does NOT
        ask the LLM to thread ``mission_id`` — the framework resolves
        it from the agent's own ``agent_id`` (the running coordinator
        IS the mission instance).

        Args:
            user_acknowledgement_quote: Verbatim text from the user's
                message authorising the early stop. The LLM MUST quote
                a real user utterance; the framework records it for
                postmortem audit. Empty / whitespace-only strings are
                rejected with a typed error.
            remaining_in_scope: The list of in-scope issue numbers the
                LLM is acknowledging are NOT being decomposed in this
                run. Recorded for audit and surfaced in the mission
                summary so the user sees what was deferred.

        Returns:
            ``{"ok": True, "mission_id": <agent_id>, "deferred": <list>}``
            or ``{"ok": False, "error": ...}`` on validation failure.
        """

        if self._agent is None:
            return {
                "ok": False,
                "error": (
                    "request_decompose_early_stop requires a mounted "
                    "agent context."
                ),
            }
        quote = (user_acknowledgement_quote or "").strip()
        if not quote:
            return {
                "ok": False,
                "error": (
                    "user_acknowledgement_quote must be the verbatim "
                    "user utterance authorising the early stop. Empty "
                    "or whitespace-only strings are rejected — the LLM "
                    "cannot self-certify."
                ),
            }
        bb = await self._agent.get_blackboard()
        await bb.write(
            DecomposeEarlyStopProtocol.signal_key(self._agent.agent_id),
            {
                "agent_id": self._agent.agent_id,
                "user_acknowledgement_quote": quote,
                "remaining_in_scope": [
                    int(n) for n in remaining_in_scope if isinstance(n, int)
                ],
            },
            tags={"decompose_early_stop"},
            metadata={"agent_id": self._agent.agent_id},
        )
        return {
            "ok": True,
            "mission_id": self._agent.agent_id,
            "deferred": [
                int(n) for n in remaining_in_scope if isinstance(n, int)
            ],
        }

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        """No in-memory state to suspend; the early-stop signal lives
        on the blackboard."""

        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        return None


__all__ = (
    "DecomposeEarlyStopProtocol",
    "ProjectPlanningMissionControlCapability",
)
