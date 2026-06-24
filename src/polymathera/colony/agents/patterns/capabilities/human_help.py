"""Session-scoped human-help capability.

Sibling of :class:`HumanApprovalCapability`. Both are session-scoped
escalation channels but they cover semantically distinct cases:

- ``request_human_approval`` — "authorise THIS specific action". Yes/
  no/abort over a known dispatch.
- ``request_help`` (this capability) — "I'm stuck on a judgment call;
  what should I do?". The agent presents a question, what it has
  tried, and candidate next actions; the operator picks an option or
  writes free-form guidance. Distinct from ``respond_to_user``
  (fire-and-forget — operator may not be listening) because the
  agent's code calls ``wait_for_next_event`` after submitting; the
  operator's reply surfaces as a planner-context binding that the
  next iteration reads.

Per [[reactive-as-special-case-of-proactive]], the wait is a
primitive the agent calls in its own code — the framework does not
implicitly block on submit. ``request_help`` returns immediately
with a ``request_id``; the agent is responsible for pausing.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar

from overrides import override
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...base import Agent, AgentCapability
from ...blackboard.protocol import HumanHelpProtocol
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor
from ..events import EventProcessingResult, event_handler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire payloads
# ---------------------------------------------------------------------------


class HumanHelpRequest(BaseModel):
    """Payload an agent posts to ``human_help:request:{request_id}``.

    The operator's UI renders ``question`` + ``context`` and offers
    the ``options`` list as quick choices, with a free-form text box
    for additional guidance. ``options`` is advisory: the operator
    may pick one of the suggested options OR write a guidance string
    without picking.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        default_factory=lambda: f"help_{uuid.uuid4().hex[:12]}",
    )
    question: str
    """What the agent is stuck on. Concise; one or two sentences."""

    context: str = ""
    """What the agent has tried and observed so far. Multi-paragraph
    OK. Empty when the agent has nothing to report beyond the
    question (rare in practice — a help request without context
    suggests the agent should call ``request_help`` later, after
    investigating)."""

    options: tuple[str, ...] = ()
    """Candidate next actions the agent identified. Each entry is a
    short label the operator picks via a button click. Empty when
    the agent cannot enumerate options — the operator then writes
    free-form guidance."""

    requester_agent_id: str | None = None

    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class HumanHelpResponse(BaseModel):
    """Payload the operator's UI / REST endpoint posts to
    ``human_help:response:{request_id}``.

    Exactly one of ``chosen_option`` and ``guidance`` MUST be
    non-empty — the operator either picks an option, writes
    guidance, or both. A response with neither carries no signal
    for the agent and is rejected by the validator."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    chosen_option: str | None = None
    """The option the operator picked from
    :attr:`HumanHelpRequest.options`, or ``None`` for a free-form
    response."""

    guidance: str = ""
    """The operator's free-form direction. May be empty when
    ``chosen_option`` carries the answer alone."""

    decided_by: str = ""
    """User id (or display name) of the human who responded."""

    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def _require_chosen_option_or_guidance(self) -> "HumanHelpResponse":
        """At least one of ``chosen_option`` and ``guidance`` MUST be
        non-empty. A response with neither carries no signal for the
        agent — the planner-context binding would surface an empty
        envelope and the next iteration would have nothing to react
        to. Enforced at the data-shape boundary so the REST endpoint
        surfaces a 422 and the chat-UI form rejects empty submits
        before they reach the blackboard."""

        picked = self.chosen_option is not None and self.chosen_option != ""
        wrote = bool(self.guidance and self.guidance.strip())
        if not (picked or wrote):
            raise ValueError(
                "HumanHelpResponse: at least one of ``chosen_option`` "
                "(the operator picks a candidate from the request's "
                "``options``) or ``guidance`` (free-form text the "
                "operator writes when none of the options fit) must "
                "be non-empty. An empty response gives the agent "
                "nothing to react to."
            )
        return self


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class HumanHelpCapability(AgentCapability):
    """Session-scoped human-help escalation gate.

    Action surface (visible to the LLM planner):

    - :meth:`request_help` — submit a typed question. Returns a
      ``request_id`` immediately; the agent's next code block should
      call ``wait_for_next_event`` to pause until the operator
      responds.
    - :meth:`get_response` — return the cached response or fall back
      to a blackboard read. Useful when a planner step needs to
      consult a previously-asked question.

    Receive side: an ``@event_handler`` for
    :meth:`HumanHelpProtocol.response_pattern` fires when the
    operator replies and surfaces the choice + guidance as planner
    context.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    DEFAULT_NAMESPACE = "human_help"

    RESPONSE_CONTEXT_KEY_PREFIX: ClassVar[str] = "human_help_response:"
    """Single source of truth for the planner-context key the
    ``@event_handler`` writes when the operator's response lands.
    Referenced by-attribute by any guardrail / advisor that needs to
    quote it so prose and code can't drift if the prefix is renamed
    (see the analogous pattern on :class:`HumanApprovalCapability`)."""

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
            input_patterns = [HumanHelpProtocol.response_pattern()]
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        self._responses: dict[str, HumanHelpResponse] = {}
        self._requests: dict[str, HumanHelpRequest] = {}
        # Per-instance bookkeeping of which request_ids the agent is
        # CURRENTLY idle-waiting on. Used to keep
        # ``Agent.idle_wait_counter`` increment/decrement balanced
        # and idempotent across repeated polls of the same id —
        # mirrors :class:`HumanApprovalCapability`'s pattern.
        self._idle_waiting_request_ids: set[str] = set()

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"human_help", "escalation", "session"})

    # ---- Request side -------------------------------------------------

    @action_executor(
        planning_summary=(
            "Submit a typed help request to the operator. Use this "
            "for judgment calls you cannot resolve from your own "
            "context — NOT for authorising a specific dispatch (use "
            "``request_human_approval`` for that) and NOT for "
            "fire-and-forget updates the operator may not see (use "
            "``respond_to_user`` for that). Returns immediately with "
            "``{ok: True, request_id: '...'}``. Your next code block "
            "MUST end with ``await run('wait_for_next_event')`` to "
            "pause until the operator responds — the response lands "
            "as a fresh "
            "``human_help_response:{request_id}`` planner-context "
            "binding (prefix on "
            ":attr:`HumanHelpCapability.RESPONSE_CONTEXT_KEY_PREFIX`) "
            "carrying ``{chosen_option, guidance, decided_by, "
            "decided_at}``. ``chosen_option`` is one of the entries "
            "you passed in ``options`` (or ``None`` if the operator "
            "wrote free-form guidance instead); ``guidance`` is the "
            "operator's text."
        ),
    )
    async def request_help(
        self,
        question: str,
        *,
        context: str = "",
        options: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Post a help request and return ``{ok, request_id}``.

        Raises:
            ValueError: if ``question`` is empty or whitespace —
                an empty escalation has no signal for the operator.
        """

        if not question or not question.strip():
            raise ValueError(
                "request_help: question is empty or whitespace; the "
                "operator has nothing to respond to. Describe what "
                "you are stuck on in one or two sentences."
            )

        request = HumanHelpRequest(
            question=question,
            context=context,
            options=options,
            requester_agent_id=(
                self._agent.agent_id if self._agent is not None else None
            ),
        )
        self._requests[request.request_id] = request
        bb = await self.get_blackboard()
        await bb.write(
            HumanHelpProtocol.request_key(request.request_id),
            request.model_dump(mode="json"),
            tags={"human_help", "request"},
            metadata={
                "request_id": request.request_id,
                "requester_agent_id": request.requester_agent_id or "",
            },
        )
        return {"ok": True, "request_id": request.request_id}

    # ---- Receive side -------------------------------------------------

    @action_executor(
        planning_summary=(
            "On-demand lookup of a known help-request's state. NOT a "
            "wait primitive — calling it in a loop is wasted work. "
            "Returns ``{ok, state: 'pending' | 'ready', response: "
            "{request_id, chosen_option, guidance, decided_by, "
            "decided_at} | None}``. ``state='ready'`` means the "
            "operator responded. Once you read the response once for "
            "a given ``request_id``, treat it as terminal — the "
            "planner-context binding persists across iterations; "
            "calling ``get_response`` again returns the same envelope "
            "while burning an iteration. Falls back to a blackboard "
            "read so a resumed agent can recover responses that "
            "landed during suspension."
        ),
    )
    async def get_response(self, request_id: str) -> dict[str, Any]:
        cached = self._responses.get(request_id)
        if cached is not None:
            self._on_resolved(request_id)
            return _render_get_response_envelope(cached)
        bb = await self.get_blackboard()
        raw = await bb.read(HumanHelpProtocol.response_key(request_id))
        if raw is None:
            self._on_pending(request_id)
            return {"ok": True, "state": "pending", "response": None}
        try:
            response = HumanHelpResponse.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.warning(
                "HumanHelpCapability: malformed response payload at %s",
                HumanHelpProtocol.response_key(request_id),
            )
            return {
                "ok": False,
                "state": "pending",
                "response": None,
                "error": "malformed_response_payload",
            }
        self._responses[request_id] = response
        self._on_resolved(request_id)
        return _render_get_response_envelope(response)

    # ---- Idle-wait counter bookkeeping --------------------------------

    def _on_pending(self, request_id: str) -> None:
        if self._agent is None:
            return
        if request_id in self._idle_waiting_request_ids:
            return
        self._idle_waiting_request_ids.add(request_id)
        self._agent.idle_wait_counter += 1

    def _on_resolved(self, request_id: str) -> None:
        if self._agent is None:
            return
        if request_id not in self._idle_waiting_request_ids:
            return
        self._idle_waiting_request_ids.discard(request_id)
        self._agent.idle_wait_counter -= 1

    def is_awaiting_event(self) -> bool:
        """``HumanHelpCapability`` is awaiting an event iff it
        currently holds at least one help request whose response has
        not landed (typed event handler clears the id via
        :meth:`_on_resolved`). Used by
        ``wait_for_next_event``'s live-wake-source pre-check."""

        return bool(self._idle_waiting_request_ids)

    # ---- Event handler ------------------------------------------------

    @event_handler(pattern=HumanHelpProtocol.response_pattern())
    async def _on_response(
        self,
        event: Any,
        _repl: Any,
    ) -> EventProcessingResult | None:
        """Cache the operator's response and surface it as planner
        context the next iteration reads."""

        try:
            request_id = HumanHelpProtocol.parse_response_key(event.key)
        except ValueError:
            return None
        try:
            response = HumanHelpResponse.model_validate(event.value)
        except Exception:  # noqa: BLE001
            logger.warning(
                "HumanHelpCapability: dropping malformed response at %s",
                event.key,
            )
            return None
        self._responses[request_id] = response
        self._on_resolved(request_id)
        return EventProcessingResult(
            context_key=(
                f"{self.RESPONSE_CONTEXT_KEY_PREFIX}{request_id}"
            ),
            context={
                "request_id": response.request_id,
                "chosen_option": response.chosen_option,
                "guidance": response.guidance,
                "decided_by": response.decided_by,
                "decided_at": (
                    response.decided_at.isoformat()
                    if response.decided_at is not None else None
                ),
            },
        )

    # ---- AgentCapability protocol -------------------------------------

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


def _render_get_response_envelope(
    response: HumanHelpResponse,
) -> dict[str, Any]:
    return {
        "ok": True,
        "state": "ready",
        "response": {
            "request_id": response.request_id,
            "chosen_option": response.chosen_option,
            "guidance": response.guidance,
            "decided_by": response.decided_by,
            "decided_at": (
                response.decided_at.isoformat()
                if response.decided_at is not None else None
            ),
        },
    }


__all__ = (
    "HumanHelpCapability",
    "HumanHelpRequest",
    "HumanHelpResponse",
)
