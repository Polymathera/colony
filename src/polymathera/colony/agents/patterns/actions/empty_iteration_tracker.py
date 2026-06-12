"""EmptyIterationTracker — soft backstop for "LLM forgot ``wait_for_next_event``".

Sibling of ``BlockStreakTracker``. Lives in its own module so
``EventDrivenActionPolicy`` stays an orchestrator with one field per
concern (see ``feedback_extract_dont_bloat.md``).

Under the unified proactive programming model, ``wait_for_next_event``
is the LLM-callable idle primitive. When the LLM has nothing to do but
forgets to call it, the agent burns LLM iterations on empty code
blocks. This tracker counts consecutive iterations where (a) NO actions
were called and (b) the event queue was empty at observation time —
the textbook "should have waited" shape — and emits ONE
``DIAGNOSTIC_EMPTY_ITERATION_STREAK`` event when the streak crosses a
small threshold. The diagnostic surfaces in the LLM's next planning
context via the existing ``agent_diagnostic`` relay (no new pipeline),
nudging the LLM to call ``wait_for_next_event``. It is a SOFT signal:
the framework never auto-injects the wait — the LLM remains in charge
([[primitives-not-pipelines]]).

Threshold is a framework-internal knob (default 3). It is intentionally
not exposed as a ``ParameterSpec`` because no operator workload has
asked for it yet; surfacing it as colony-scoped metadata would be
premature [[colony-scoped-params-propagation]].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_EMPTY_ITERATION_STREAK,
)

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class EmptyIterationTracker:
    """Counts consecutive empty iterations and emits one diagnostic
    when the streak crosses ``threshold``.

    A "streak" is the chain of iterations where the LLM neither acted
    nor observed any incoming event. ``observe_iteration`` is called
    once per planning iteration by ``EventDrivenActionPolicy``;
    streaks reset on any productive iteration. The diagnostic fires
    exactly once per streak (at ``threshold``) so a regressing LLM
    does not produce a flood of events; if the LLM continues burning
    empty iterations, the next diagnostic only fires after a reset +
    re-streak.
    """

    DIAGNOSTIC_NAMESPACE = "agent_diagnostic"

    def __init__(self, agent: "Agent", *, threshold: int = 3):
        self.agent = agent
        self._streak: int = 0
        self._threshold: int = threshold
        self._diagnostic_seq: int = 0
        self._diagnostic_fired_for_streak: bool = False

    async def observe_iteration(
        self,
        *,
        actions_called_count: int,
        queue_was_empty_at_observation: bool,
    ) -> None:
        """Account for one planning iteration.

        Increments the streak when the iteration produced no actions
        AND the event queue was empty at observation time (the shape
        of "should have waited"). Any productive iteration — at least
        one action called, or an event was observed — resets the
        streak and the per-streak fired flag, so the next streak gets
        a fresh diagnostic.
        """
        if actions_called_count == 0 and queue_was_empty_at_observation:
            self._streak += 1
            if (
                self._streak >= self._threshold
                and not self._diagnostic_fired_for_streak
            ):
                self._diagnostic_fired_for_streak = True
                await self._emit_diagnostic()
        else:
            self._streak = 0
            self._diagnostic_fired_for_streak = False

    def snapshot(self) -> dict[str, Any]:
        """Read-only summary for ``get_status_snapshot``."""
        return {
            "streak": self._streak,
            "threshold": self._threshold,
            "diagnostic_fired_for_streak": self._diagnostic_fired_for_streak,
        }

    async def _emit_diagnostic(self) -> None:
        """Publish one ``empty_iteration_streak`` diagnostic.

        Emission shape matches ``BlockStreakTracker._emit_diagnostic`` so
        the existing ``agent_diagnostic`` relay in ``SessionAgent`` and
        the LLM-prompt surfacing path require no changes. Intentional:
        no defensive try/except around ``bb.write`` — if the blackboard
        write fails, the failure must surface rather than silently mask
        the bug ([[no-bandaids-durable-solutions]]).
        """
        self._diagnostic_seq += 1
        seq = self._diagnostic_seq
        from ...scopes import BlackboardScope, get_scope_prefix
        scope_id = get_scope_prefix(
            BlackboardScope.SESSION,
            self.agent,
            namespace=self.DIAGNOSTIC_NAMESPACE,
        )
        bb = await self.agent.get_blackboard(scope_id=scope_id)
        await bb.write(
            AgentDiagnosticProtocol.event_key(
                self.agent.agent_id,
                DIAGNOSTIC_EMPTY_ITERATION_STREAK,
                seq,
            ),
            {
                "agent_id": self.agent.agent_id,
                "kind": DIAGNOSTIC_EMPTY_ITERATION_STREAK,
                "streak": self._streak,
                "threshold": self._threshold,
                "suggestion": (
                    "You have run several iterations without calling any "
                    "action and without any event arriving. If you have "
                    "no work to do right now, end your next code block "
                    "with ``await run('wait_for_next_event')`` to pause "
                    "until the next event arrives, instead of burning "
                    "empty planning iterations."
                ),
            },
            tags={"agent_diagnostic", DIAGNOSTIC_EMPTY_ITERATION_STREAK},
            metadata={
                "agent_id": self.agent.agent_id,
                "kind": DIAGNOSTIC_EMPTY_ITERATION_STREAK,
            },
        )
