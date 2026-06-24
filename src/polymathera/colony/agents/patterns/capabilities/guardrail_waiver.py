"""``GuardrailWaiverCapability`` — agent-initiated waiver requests
for semantic constraints the agent cannot pass.

When :class:`SemanticConstraintGuardrail` blocks an action the agent
judges the rule shouldn't apply to (e.g. the LLM judge keeps
misreading verified evidence), the agent calls
:meth:`request_guardrail_waiver`. That writes a typed
:class:`GuardrailWaiverProtocol` ``request`` key; the SessionAgent
mirrors it to chat as an Approve/Reject card; the user's decision
lands as a typed ``response`` key. On approve, the dashboard's
existing PR5-B path is also written
(``operator_override:semantic_constraint:<cid>``), so
:meth:`SemanticConstraintGuardrail._read_disabled_ids` sees the
override on the next ``.check()`` — no separate enforcement plumbing.

Auto-mounted by :func:`polymathera.colony.agents.patterns.actions.defaults`
on every action policy whose ``runtime_guardrail`` includes a
:class:`SemanticConstraintGuardrail`: the action only exists when
there's something to waive.

Design parallels :class:`HumanApprovalCapability`:

- Same SESSION scope, same request/response BB key shape, same
  ``@event_handler`` on the response pattern that returns a
  ``EventProcessingResult`` with a typed planner-context binding so
  the asking agent's NEXT planner iteration sees the decision.
- The agent's idle-wait happens via the existing ``wait_for_next_event``
  primitive; the BB write on the response key wakes its event queue
  through this capability's subscription.

The capability holds NO local state about granted waivers — the
override is read live from BB by ``SemanticConstraintGuardrail`` (PR5
single-source-of-truth contract). This avoids the duplicate-state
class of bug the operator-disable path already closed. The capability
DOES track in-flight ``waiver_id``s in :attr:`_outstanding_waiver_ids`
purely so :meth:`is_awaiting_event` can answer ``wait_for_next_event``'s
live-wake-source pre-check (DL2) — entries are added when the request
publishes and discarded as soon as the typed response handler observes
the matching key.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, ClassVar

from overrides import override

from ...base import Agent, AgentCapability
from ...blackboard import BlackboardEvent
from ...blackboard.protocol import GuardrailWaiverProtocol
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor
from ..events import EventProcessingResult, event_handler


logger = logging.getLogger(__name__)


class GuardrailWaiverCapability(AgentCapability):
    """Agent-callable waiver primitive + response handler.

    Mounted on agents that carry a :class:`SemanticConstraintGuardrail`
    (auto-attached in ``defaults.py``); the
    :meth:`request_guardrail_waiver` action is then available to the
    LLM planner as an escape hatch when the guardrail blocks an
    action the LLM judges as a false positive.
    """

    DEFAULT_NAMESPACE = "guardrail_waiver"

    #: Planner-context binding prefix the ``@event_handler`` emits on
    #: response. The agent's planner sees a fresh
    #: ``{RESPONSE_CONTEXT_KEY_PREFIX}{waiver_id}`` binding on its
    #: next iteration carrying ``{approved, decided_by, reason,
    #: constraint_id}`` so the LLM can branch deterministically.
    RESPONSE_CONTEXT_KEY_PREFIX: ClassVar[str] = "guardrail_waiver_response:"

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = DEFAULT_NAMESPACE,
        input_patterns: list[str] | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        if scope_id is None:
            scope_id = get_scope_prefix(scope, agent, namespace=namespace)
        if input_patterns is None:
            input_patterns = [GuardrailWaiverProtocol.response_pattern()]
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        #: Live-pending tracker: waiver_ids that have been published
        #: on BB but whose typed response key has not been observed
        #: yet. Drives :meth:`is_awaiting_event` so
        #: ``wait_for_next_event``'s pre-check counts this capability
        #: as a live wake source while a decision is outstanding.
        self._outstanding_waiver_ids: set[str] = set()

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"guardrail_waiver", "gate", "session"})

    # ---- Request side ------------------------------------------------

    @action_executor(
        planning_summary=(
            "Ask the user for a waiver on a runtime semantic-constraint "
            "rule that has BLOCKED a proposed action the agent judges "
            "the rule shouldn't apply to (LLM-judge false positive on "
            "verified evidence, rule legitimately doesn't fit the "
            "domain context, etc.). Returns immediately with "
            "``{ok: True, waiver_id: '...'}``. The user's decision "
            "surfaces later as a fresh "
            "``guardrail_waiver_response:{waiver_id}`` planner-context "
            "binding carrying ``{approved, decided_by, reason, "
            "constraint_id}`` — branch on ``approved`` in the next "
            "iteration. PRECONDITION: ``constraint_id`` MUST match a "
            "rule that's currently blocking the agent (use the rule "
            "id from the GuardrailBlockedError's reason field). "
            "``justification`` MUST explain WHY the rule's "
            "evaluation is wrong for this specific case — generic "
            "'please let me through' is not actionable for the user. "
            "End the next code block with "
            "``await run('wait_for_next_event')`` to pause until the "
            "decision arrives; the BB write on the response key "
            "wakes the agent through this capability's subscription. "
            "Do NOT poll for the decision in a loop; that wastes "
            "iteration budget and the planner-context binding only "
            "lands on a fresh planner iteration."
        ),
    )
    async def request_guardrail_waiver(
        self,
        constraint_id: str,
        justification: str,
    ) -> dict[str, Any]:
        """Post a waiver request; return ``{ok, waiver_id}``.

        Args:
            constraint_id: The rule id that has been blocking the
                agent (e.g. ``"no_unverified_agent_state_claims"``).
                Pulled from the ``GuardrailBlockedError.reason`` the
                agent received.
            justification: Human-readable explanation of WHY the rule's
                evaluation is wrong for this specific case. Surfaces
                in the chat UI for the user to evaluate.
        """

        if self._agent is None:
            return {
                "ok": False,
                "error": (
                    "request_guardrail_waiver requires a mounted agent "
                    "context."
                ),
            }
        cid = (constraint_id or "").strip()
        if not cid:
            return {
                "ok": False,
                "error": (
                    "constraint_id is required — pass the rule id from "
                    "the GuardrailBlockedError.reason field."
                ),
            }
        reason = (justification or "").strip()
        if not reason:
            return {
                "ok": False,
                "error": (
                    "justification is required — generic 'please let "
                    "me through' is not actionable for the user."
                ),
            }
        waiver_id = f"waiver_{uuid.uuid4().hex[:12]}"
        bb = await self.get_blackboard()
        await bb.write(
            GuardrailWaiverProtocol.request_key(waiver_id),
            {
                "waiver_id": waiver_id,
                "constraint_id": cid,
                "justification": reason,
                "requester_agent_id": self._agent.agent_id,
            },
            tags={"guardrail_waiver", "request"},
            metadata={
                "waiver_id": waiver_id,
                "constraint_id": cid,
                "requester_agent_id": self._agent.agent_id,
            },
        )
        self._outstanding_waiver_ids.add(waiver_id)
        logger.warning(
            "[GuardrailWaiver] request_published: waiver_id=%s "
            "constraint_id=%s requester=%s",
            waiver_id, cid, self._agent.agent_id,
        )
        return {"ok": True, "waiver_id": waiver_id}

    def is_awaiting_event(self) -> bool:
        """``GuardrailWaiverCapability`` is awaiting an event iff at
        least one ``request_guardrail_waiver`` request has been
        published whose typed response has not landed (the typed
        event handler clears the id on first observation). Used by
        ``wait_for_next_event``'s live-wake-source pre-check."""

        return bool(self._outstanding_waiver_ids)

    # ---- Receive side ------------------------------------------------

    @event_handler(pattern=GuardrailWaiverProtocol.response_pattern())
    async def _on_response(
        self,
        event: BlackboardEvent,
        _repl: Any,
    ) -> EventProcessingResult | None:
        """Surface the user's waiver decision as planner context.

        Pure translation: no local cache, no idle-wait bookkeeping
        (the agent calls ``wait_for_next_event`` after requesting; the
        BB write wakes its event queue through this capability's
        subscription; the next planner iteration sees the typed
        binding and branches). The PR5-B override is already on BB
        when ``approved=True`` because the approve-endpoint writes
        BOTH keys atomically — the asking agent's next action attempt
        sees the constraint as disabled via
        ``SemanticConstraintGuardrail._read_disabled_ids``."""

        try:
            waiver_id = GuardrailWaiverProtocol.parse_response_key(event.key)
        except ValueError:
            return None
        if not isinstance(event.value, dict):
            return None
        approved = bool(event.value.get("approved", False))
        decided_by = event.value.get("decided_by", "")
        constraint_id = event.value.get("constraint_id", "")
        reason = event.value.get("reason") or ""
        self._outstanding_waiver_ids.discard(waiver_id)
        logger.warning(
            "[GuardrailWaiver] response_received: waiver_id=%s "
            "constraint_id=%s approved=%s decided_by=%s",
            waiver_id, constraint_id, approved, decided_by,
        )
        return EventProcessingResult(
            context_key=f"{self.RESPONSE_CONTEXT_KEY_PREFIX}{waiver_id}",
            context={
                "waiver_id": waiver_id,
                "approved": approved,
                "decided_by": decided_by,
                "reason": reason,
                "constraint_id": constraint_id,
            },
        )


    # ---- Suspension hooks --------------------------------------------
    #
    # No local state to persist — the waiver request/response keys live
    # on BB and are durable across the agent's suspend/resume. On
    # resume the agent's planner sees the response binding via the
    # standard event-handler replay path.

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        return None


__all__ = ("GuardrailWaiverCapability",)
