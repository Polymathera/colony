"""ContinuationTracker — accounting for ``signal_continuation()`` bursts.

Sibling of ``BlockStreakTracker`` and ``LLMFailureBackoff``. Lives in
its own module to keep ``BaseActionPolicy`` an orchestrator (see
``feedback_extract_dont_bloat.md``).

One source of truth: ``state.custom["continuation_requested"]`` is
written by the ``signal_continuation`` REPL builtin and consumed by
the reactive_only gate in ``EventDrivenActionPolicy.plan_step``. This
tracker counts the consecutive-continuation burst, enforces a cap,
and emits one ``AgentDiagnosticProtocol`` event when the cap is hit.

The host policy holds ONE field (``self._continuation_tracker``); no
``_continuation_*`` attributes leak onto the policy class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED,
)

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class ContinuationTracker:
    """Counts consecutive ``signal_continuation()`` calls per event burst.

    A "burst" is the chain of LLM iterations triggered by ONE inbound
    external event (user message, child diagnostic, control command).
    Every ``record_continuation()`` increments the burst counter; any
    real external event resets it via ``reset()``.

    When the consecutive count would exceed ``max_per_burst``, the
    tracker emits ONE ``continuation_budget_exhausted`` diagnostic and
    returns ``False`` so the gate refuses to honor the signal.
    """

    DIAGNOSTIC_NAMESPACE = "agent_diagnostic"

    def __init__(self, agent: "Agent", *, max_per_burst: int = 5):
        self.agent = agent
        self._consecutive_count: int = 0
        self._max_per_burst: int = max_per_burst
        self._last_reason: str | None = None
        self._diagnostic_seq: int = 0

    async def record_continuation(self, reason: str) -> bool:
        """Account for one ``signal_continuation(reason)`` call.

        Returns ``True`` if the budget allows the next iteration to
        proceed as a continuation; ``False`` if the cap is exhausted
        (in which case the diagnostic has already been emitted).
        """
        if self._consecutive_count >= self._max_per_burst:
            await self._emit_exhaustion(reason=reason)
            return False
        self._consecutive_count += 1
        self._last_reason = reason
        return True

    def reset(self) -> None:
        """Reset the burst counter. Called when a NEW external event
        arrives (any event that is not a self-triggered continuation).
        """
        self._consecutive_count = 0
        self._last_reason = None

    def snapshot(self) -> dict[str, Any]:
        """Read-only summary for ``get_status_snapshot``."""
        return {
            "consecutive_count": self._consecutive_count,
            "max_per_burst": self._max_per_burst,
            "last_reason": self._last_reason,
            "exhausted": self._consecutive_count >= self._max_per_burst,
        }

    async def _emit_exhaustion(self, *, reason: str) -> None:
        """Publish one ``continuation_budget_exhausted`` diagnostic.

        Same emission shape as ``BlockStreakTracker._emit_diagnostic``
        so the consumer pathway in ``SessionAgent`` is uniform.

        Intentional: no ``try/except`` around ``bb.write`` (unlike
        ``BlockStreakTracker._emit_diagnostic``'s defensive wrap).
        The diagnostic is the ONLY signal the LLM receives that its
        next iteration must commit; silently swallowing a write
        failure would let the agent burn LLM calls forever with no
        visible cause. Per [[no-bandaids-durable-solutions]]: if the
        blackboard write fails, the policy raises and the failure
        surfaces — the cure for "bb.write is flaky" is to fix the
        blackboard, not to hide the breakage here.
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
                DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED,
                seq,
            ),
            {
                "agent_id": self.agent.agent_id,
                "kind": DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED,
                "consecutive_count": self._consecutive_count,
                "max_per_burst": self._max_per_burst,
                "last_reason": self._last_reason,
                "attempted_reason": reason,
                "suggestion": (
                    "Continuation budget exhausted for this event burst. "
                    "The next iteration MUST commit: dispatch an action, "
                    "respond to the user, or signal completion. A new "
                    "external event will reset the budget."
                ),
            },
            tags={"agent_diagnostic", DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED},
            metadata={
                "agent_id": self.agent.agent_id,
                "kind": DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED,
            },
        )
