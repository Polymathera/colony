"""``HumanApprovalCapability`` — typed human-in-the-loop gate over the session blackboard.

Replaces the earlier ``SupervisorCapability``-based human approval surface
(deleted in PR #1). The previous design had three structural problems
against the agent loop:

1. ``await_human_approval`` was an ``@action_executor`` whose body
   blocked on an ``asyncio.Event``, freezing the action policy
   iteration for as long as the human took to respond.
2. Suspend/resume lost the in-flight ``asyncio.Event`` — the new
   capability instance on resume created a fresh event that the
   already-fired ``record_human_approval`` could never set.
3. The Web UI had no documented path back into ``record_human_approval``
   (it was an ``@action_executor``, not a deployable HTTP handler).

The redesign matches the protocol shape of
:class:`~polymathera.colony.agents.blackboard.protocol.AgentRunProtocol`:

- The capability's :meth:`request_human_approval` writes a typed
  :class:`HumanApprovalRequest` onto the *session-scoped*
  ``human_approval:request:{request_id}`` topic and returns
  immediately. **No blocking await.** The action policy continues
  iterating.

- Multiple agents in the same session share the topic. The
  ``SessionOrchestratorCapability`` (also session-scoped) surfaces
  pending requests to the chat UI. A Web UI HTTP endpoint receives
  the user's typed response and writes it to
  ``human_approval:response:{request_id}`` on the same scope.

- An ``@event_handler`` on this capability fires when the response
  lands; it caches the response (so query-by-id is cheap) and
  returns an :class:`EventProcessingResult` whose ``context_key`` /
  ``context`` surface in the agent's next planner iteration. The LLM
  planner sees a concrete observation ("the user chose ``approve``")
  and plans the next action in the same loop it always uses.

- Suspend/resume is durable: requests AND responses live on the
  blackboard by definition. A resumed agent's
  :meth:`get_response` falls back to a blackboard read when the
  in-process cache is empty, so the response (if it arrived during
  suspension) is recovered without reissuing the request.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar

from overrides import override
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...base import Agent, AgentCapability
from ...blackboard import BlackboardEvent
from ...blackboard.protocol import HumanApprovalProtocol
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor
from ..actions.dispatcher import ActionInputViolation
from ..events import EventProcessingResult, event_handler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire payloads
# ---------------------------------------------------------------------------


class HumanApprovalRequest(BaseModel):
    """Payload an agent posts to ``human_approval:request:{request_id}``.

    When ``action_type`` is set, the UI renders four choices
    (``approve_once`` / ``approve_all`` / ``reject`` / ``abort``).
    ``approve_all`` auto-allows future dispatches whose ``action_key``
    contains ``action_type``. ``reject`` blocks the apply for this
    request but does NOT terminate the agent; the operator's
    ``explanation`` surfaces in the planner context binding so the
    next iteration can adjust. ``abort`` signals the agent to wind
    down via its mission-control surface; the explanation accompanies
    the signal. When ``action_type`` is ``None``, the legacy
    two-choice (``approve`` / ``reject``) shape applies and no
    ``explanation`` is required.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        default_factory=lambda: f"appr_{uuid.uuid4().hex[:12]}",
    )
    question: str
    options: tuple[str, ...] = ("approve", "reject")
    action_type: str | None = None
    requester_agent_id: str | None = None
    """Agent that asked. ``None`` in detached / test contexts."""

    deadline: datetime | None = None
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    extra: dict[str, Any] = Field(default_factory=dict)
    """Free-form context the requester wants the UI to render alongside
    the question (e.g., a diff, a summary, a list of affected pages)."""

    @property
    def is_typed(self) -> bool:
        return self.action_type is not None


CHOICES_REQUIRING_EXPLANATION: frozenset[str] = frozenset({"reject", "abort"})
"""``HumanApprovalResponse.explanation`` is required and must be
non-empty when ``choice`` is one of these. The validator below enforces
it at the data-shape boundary so guardrail predicates and the chat-UI
relay see the same contract."""


_HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S")
"""Match a markdown header line (e.g. ``## Proposed Decompositions``).
Headers are template scaffolding, not body content — the validator
strips them when counting substantive characters."""

_LIST_ITEM_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S")
"""Match a markdown list item: bullet (``- foo`` / ``* foo`` / ``+
foo``) or numbered (``1. foo`` / ``1) foo``). Each line that matches
is an enumerated item the operator can actually evaluate."""

_SUBSECTION_RE = re.compile(r"^\s*#{2,6}\s+\S")
"""Match a sub-section header (``## ...`` and deeper). The LLM also
enumerates items as `### #42: title` sub-sections, not just bullets."""

_SEPARATOR_RE = re.compile(r"^\s*[-=*_]{3,}\s*$")
"""Match a markdown horizontal-rule / separator line. Scaffolding."""

_MIN_BODY_CHARS = 8
"""Minimum non-scaffolding character count an approval body must
contain. Set low (admits ``"Approve?"`` and similar brief substantive
questions) because the structural template-empty check below catches
the actual forensic shape — header announcing items + footer + zero
items between."""


def _strip_scaffolding(text: str) -> str:
    """Return ``text`` with markdown headers, separators, blank lines,
    and surrounding whitespace removed. What remains is the substantive
    body the operator actually reads to make a decision."""

    out_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _HEADER_RE.match(raw):
            continue
        if _SEPARATOR_RE.match(raw):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _has_enumerated_items(text: str) -> bool:
    """Return True if the question contains at least one enumerated
    item shape. The structural test for "does this body show items":

    - Any bullet / numbered list item → an item.
    - Two or more markdown headers → the second+ are item sub-sections
      (the LLM uses ``## Item 1`` / ``## Item 2`` / ... or ``## Header``
      followed by ``### #42: title`` ``### #43: title`` ... — both
      patterns enumerate items via headers).
    """

    has_list = False
    header_count = 0
    for raw in text.splitlines():
        if _LIST_ITEM_RE.match(raw):
            has_list = True
        if _HEADER_RE.match(raw):
            header_count += 1
    return has_list or header_count >= 2


_ENUMERATION_CLAIM_RE = re.compile(
    r"\(\s*\d+\s+\w+\s*\)|"           # "(4 issues)", "(2 proposals)"
    r"\b\d+\s+(?:issues?|proposals?|"  # "4 issues", "2 proposals", etc.
    r"items?|decompositions?|tasks?|"
    r"changes?|files?|edits?)\b|"
    r"\b(?:proposed|the following|each of the following)\s+"
    r"(?:issues?|proposals?|items?|decompositions?|"
    r"tasks?|changes?|files?|edits?)\b",
    re.IGNORECASE,
)
"""Match phrases that PROMISE enumeration: count patterns like
``(4 issues)`` / ``2 proposals``, or framing phrases like ``Proposed
Decompositions`` / ``The following issues``. When a question contains
one of these AND has no enumerated items in the body, it's the
templated-empty-items shape from head4.log. A header alone (e.g. ``##
Approve this change``) without an enumeration claim is fine — the
operator can act on the descriptive text below it."""


def _claims_enumeration(text: str) -> bool:
    return bool(_ENUMERATION_CLAIM_RE.search(text))


class RequestHumanApprovalEmpty(ActionInputViolation):
    """``request_human_approval`` was called with no substantive
    content for the operator to evaluate.

    A question is rejected when, after stripping markdown headers /
    separators / blank lines, the remaining substantive body is shorter
    than :data:`_MIN_BODY_CHARS`. This catches the forensic shape where
    an LLM generates a header + footer scaffold but the loop that fills
    the body has zero items (e.g., ``successful_proposals`` was empty
    or the code was truncated by an iteration-shape limit).

    Subclass of :class:`ActionInputViolation` so the dispatcher
    re-raises it unwrapped at the cell's ``await run(...)`` site.
    Raised at the action-executor boundary so the framework converts
    the exception into ``ActionResult(success=False, error=...)``;
    the ``ErrorRewriterReflector`` rule then surfaces a typed
    recovery recipe in the next iteration's prompt — either build
    the body from existing proposals or stop the run cleanly via
    the mission's terminal-stop primitive.
    Suffix ``run()`` calls in the same cell (notably a paired
    ``wait_for_next_event``) MUST NOT execute, otherwise the cell
    deadlocks waiting on a response key that was never written.
    A forensic analysis captured the exact deadlock: the
    LLM emitted ``request_human_approval(...)`` then
    ``wait_for_next_event()`` in the same cell; the request raised
    HERE before publishing, the wait then hung forever on a
    response that could never arrive. The previous design (subclass
    of ``ValueError``) deferred recovery to the next iteration via
    ``ErrorRewriterReflector`` — but the next iteration never
    starts when the cell deadlocks inside the first one. Re-parent
    to ``ActionInputViolation`` so prevention is enforced at the
    dispatcher boundary."""


class HumanApprovalResponse(BaseModel):
    """Payload the Web UI posts to ``human_approval:response:{request_id}``."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    choice: str
    explanation: str = ""
    """Operator's justification. Required (non-empty) when ``choice``
    is in :data:`CHOICES_REQUIRING_EXPLANATION` (``reject`` / ``abort``);
    optional otherwise. Surfaces verbatim on the next planner iteration
    as ``response.explanation`` in the
    ``{RESPONSE_CONTEXT_KEY_PREFIX}{request_id}`` context binding."""

    note: str = ""
    decided_by: str = ""
    """User id (or display name) of the human who responded."""

    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @model_validator(mode="after")
    def _require_explanation_on_reject_or_abort(self) -> "HumanApprovalResponse":
        if (
            self.choice in CHOICES_REQUIRING_EXPLANATION
            and not self.explanation.strip()
        ):
            raise ValueError(
                f"HumanApprovalResponse: choice={self.choice!r} requires a "
                f"non-empty ``explanation`` (operator justification). The "
                f"chat-UI approval card surfaces a textarea when the "
                f"operator picks Reject or Abort; the explanation must "
                f"reach the agent so its next iteration can react."
            )
        return self


# ---------------------------------------------------------------------------
# Envelope renderer — single point that flattens a HumanApprovalResponse
# into the JSON-serialisable shape the action surface returns. Reading
# ``response.choice`` directly off the Pydantic model would round-trip
# through the action policy's CallRecord preview as ``str(model)`` —
# guardrail predicates that key off ``record.result["response"]
# ["choice"]`` wouldn't find a dict. Flattening here once keeps every
# downstream reader (LLM planner, guardrail predicate, chat UI relay)
# on the same dict-envelope contract.
# ---------------------------------------------------------------------------


def _render_get_response_envelope(
    response: "HumanApprovalResponse",
) -> dict[str, Any]:
    return {
        "ok": True,
        "state": "ready",
        "response": {
            "request_id": response.request_id,
            "choice": response.choice,
            "explanation": response.explanation,
            "note": response.note,
            "decided_by": response.decided_by,
            "decided_at": (
                response.decided_at.isoformat()
                if response.decided_at is not None else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class HumanApprovalCapability(AgentCapability):
    """Session-scoped human-approval gate.

    Compose this capability on any agent that needs to ask the user a
    typed question mid-flow and react to the answer through the agent's
    normal event loop. The capability is session-scoped — multiple
    agents in the same session share the topic; the Web UI surfaces
    requests; the user's response flows back through an HTTP endpoint.

    Action surface (visible to the LLM planner):

    - :meth:`request_human_approval` — submit a typed question. Returns
      a ``request_id`` immediately.
    - :meth:`get_response` — return the cached response or fall back
      to a blackboard read; ``None`` while pending. Useful when a
      planner step needs to consult a previously-asked question.
    - :meth:`list_pending` — request_ids of asks not yet resolved.

    Receive side: an ``@event_handler`` for
    :meth:`HumanApprovalProtocol.response_pattern` fires when the user
    replies and surfaces the choice as planner context.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    DEFAULT_NAMESPACE = "human_approval"

    # Single source of truth for the planner-context key the
    # ``@event_handler`` writes when the operator's response lands. The
    # ``ApprovalRequiredGuardrail`` advisory references this prefix by
    # attribute so the advisory text and the writer can't drift if the
    # prefix is ever renamed. Pairs with
    # ``[[fix-the-class-not-the-instance]]`` — canonical owner of the
    # invariant.
    RESPONSE_CONTEXT_KEY_PREFIX: ClassVar[str] = "human_approval_response:"

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
            input_patterns = [HumanApprovalProtocol.response_pattern()]
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        # Cache of responses already observed via the event handler.
        # Backed by the blackboard so a resumed agent can recover via
        # ``get_response`` without re-asking the user.
        self._responses: dict[str, HumanApprovalResponse] = {}
        # Cache of requests we've issued. Persisted across suspend/
        # resume so ``list_pending`` returns a stable view even after a
        # restart.
        self._requests: dict[str, HumanApprovalRequest] = {}
        # Per-instance bookkeeping of which request_ids the agent
        # is CURRENTLY idle-waiting on. Used to keep
        # ``Agent.idle_wait_counter`` increment/decrement balanced
        # and idempotent across repeated ``get_response`` polls for
        # the same pending request_id.
        self._idle_waiting_request_ids: set[str] = set()

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"human_approval", "gate", "session"})

    # ---- Request side -------------------------------------------------

    @action_executor(
        planning_summary=(
            "Submit a typed human-approval request. Returns immediately "
            "with ``{ok: True, request_id: '...'}``. The user's response "
            "surfaces later as a fresh "
            "``human_approval_response:{request_id}`` planner-context "
            "binding (key prefix on "
            ":attr:`HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX`). "
            "PRECONDITION: ``question`` MUST contain substantive content "
            "the operator can act on — at least one concrete description "
            "of what is being approved. A markdown header (``## ...``) "
            "plus a generic footer (``Approving will create ...``) with "
            "NOTHING ITEMIZED between them is rejected at the action "
            "boundary with ``RequestHumanApprovalEmpty``. Before "
            "calling, verify the list of items you are looping over is "
            "non-empty (e.g., ``assert len(successful_proposals) > 0``) "
            "and that each item's title / body fields are populated. "
            "If your upstream computation produced zero items, do NOT "
            "request approval for nothing — stop the run cleanly via "
            "your mission's terminal-stop primitive instead. "
            "When ``action_type`` is set the UI offers four choices: "
            "``approve_once`` (apply this dispatch), ``approve_all`` "
            "(approve every future dispatch whose action_key contains "
            "``action_type`` in this session), ``reject`` (block this "
            "dispatch; the agent stays alive and the operator's "
            "``explanation`` surfaces in the planner context so the "
            "next iteration can adjust), and ``abort`` (signal the "
            "agent to wind down via its mission-control surface; the "
            "explanation accompanies the signal). ``reject`` and "
            "``abort`` require a non-empty ``explanation`` from the "
            "operator."
        ),
    )
    async def request_human_approval(
        self,
        question: str,
        *,
        action_type: str | None = None,
        options: tuple[str, ...] | None = None,
        deadline: datetime | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Post a request and return the typed envelope ``{ok, request_id}``.

        Raises:
            RequestHumanApprovalEmpty: if the ``question`` body is
                empty / whitespace, or matches the templated-empty-
                body pattern (3+ consecutive newlines, signature of
                a header + footer with nothing between them).
        """

        question_substance = question.strip() if question else ""
        if not question_substance:
            raise RequestHumanApprovalEmpty(
                "request_human_approval: question is empty or "
                "whitespace; the operator has nothing to evaluate. "
                "Build the body from your proposals before "
                "re-requesting, or — if upstream computation failed "
                "— stop the run cleanly via your mission's "
                "terminal-stop primitive."
            )
        substantive_body = _strip_scaffolding(question)
        if len(substantive_body) < _MIN_BODY_CHARS:
            raise RequestHumanApprovalEmpty(
                f"request_human_approval: after stripping markdown "
                f"headers / separators / blank lines, only "
                f"{len(substantive_body)} chars of substantive body "
                f"remain (minimum: {_MIN_BODY_CHARS}). The operator "
                f"has no content to evaluate. Either build the body "
                f"from the real proposals you computed, or — if "
                f"upstream computation failed — stop the run cleanly "
                f"via your mission's terminal-stop primitive."
            )
        # Structural check for the forensic empty-items shape: when
        # the question PROMISES enumeration (e.g., ``Proposed
        # Decompositions``, ``(4 issues)``, ``the following items``)
        # but has NO enumerated items in the body, the operator sees
        # only a header + footer with nothing between. The forensic
        # shape from head4.log:
        # ``"## Proposed Issue Decompositions\n\nApproving will create
        # sub-issues and patch parent bodies with checklists."`` —
        # header present, footer present, ZERO list items between.
        # The triple-newline regex misses this because there are only
        # TWO consecutive newlines. Gating on ``_claims_enumeration``
        # avoids over-rejecting legitimate single-header questions
        # (e.g., ``"## Approve this change\n\nSummary: ..."``).
        if _claims_enumeration(question) and not _has_enumerated_items(question):
            raise RequestHumanApprovalEmpty(
                "request_human_approval: question promises enumerated "
                "items (e.g., 'Proposed ...', '(N items)', 'the "
                "following ...') but the body has NO enumerated items "
                "(no list bullets, no numbered items, no sub-section "
                "headers). The operator sees a header + footer with "
                "nothing between to evaluate. Common causes: the "
                "proposals list you looped over was empty; the "
                "template's interpolation produced a blank string; "
                "the IterationShapeValidator truncated the code that "
                "emits the list items. Either rebuild the body to "
                "include the actual items (one bullet or sub-section "
                "per item), or — if upstream computation produced "
                "zero items — stop the run cleanly via your mission's "
                "terminal-stop primitive. NEVER submit a header + "
                "footer with no items between."
            )

        if options is None:
            options = (
                ("approve_once", "approve_all", "reject", "abort")
                if action_type is not None
                else ("approve", "reject")
            )
        request = HumanApprovalRequest(
            question=question,
            options=options,
            action_type=action_type,
            requester_agent_id=(
                self._agent.agent_id if self._agent is not None else None
            ),
            deadline=deadline,
            extra=dict(extra or {}),
        )
        bb = await self.get_blackboard()
        await bb.write(
            HumanApprovalProtocol.request_key(request.request_id),
            request.model_dump(mode="json"),
            tags={"human_approval", "request"},
            metadata={
                "request_id": request.request_id,
                "requester_agent_id": request.requester_agent_id or "",
                "action_type": action_type or "",
            },
        )
        self._requests[request.request_id] = request
        logger.info(
            "[Approval] request_published: rid=%s action_type=%s options=%s requester=%s",
            request.request_id,
            action_type or "",
            list(options),
            request.requester_agent_id or "",
        )
        return {"ok": True, "request_id": request.request_id}

    @action_executor(
        planning_summary=(
            "On-demand lookup of a known request's current state. NOT a "
            "wait primitive — calling it in a loop is wasted work. "
            "Returns the typed envelope ``{ok, state: 'pending' | "
            "'ready', response: {request_id, choice, explanation, note, "
            "decided_by, decided_at} | None}``. ``state='ready'`` means "
            "the operator has responded — read ``response.choice`` "
            "(``approve_once`` / ``approve_all`` / ``reject`` / "
            "``abort``) and ``response.explanation`` (non-empty on "
            "``reject`` / ``abort``). ``state='pending'`` means the "
            "operator has not yet responded; instead of re-calling "
            "``get_response``, end the next code block with "
            "``await run('wait_for_next_event')`` to pause until the "
            "response arrives as planner context. Falls back to a "
            "blackboard read so a resumed agent can recover responses "
            "that landed during suspension."
        ),
    )
    async def get_response(
        self, request_id: str,
    ) -> dict[str, Any]:
        cached = self._responses.get(request_id)
        if cached is not None:
            # Cached response → terminal state. Decrement the
            # idle-wait counter on the FIRST cached-read for this
            # request_id; subsequent cached-reads are idempotent.
            self._on_resolved(request_id)
            return _render_get_response_envelope(cached)
        bb = await self.get_blackboard()
        raw = await bb.read(HumanApprovalProtocol.response_key(request_id))
        if raw is None:
            # Still pending — increment the idle-wait counter on
            # the FIRST pending poll for this request_id. Subsequent
            # pending polls for the same id are idempotent.
            self._on_pending(request_id)
            return {"ok": True, "state": "pending", "response": None}
        try:
            response = HumanApprovalResponse.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.warning(
                "HumanApprovalCapability: malformed response payload at %s",
                HumanApprovalProtocol.response_key(request_id),
            )
            return {
                "ok": False,
                "state": "pending",
                "response": None,
                "error": "malformed_response_payload",
            }
        self._responses[request_id] = response
        # Just resolved — decrement on the first observation.
        self._on_resolved(request_id)
        return _render_get_response_envelope(response)

    # ---- Idle-wait counter bookkeeping --------------------------------
    #
    # Strict per-(capability instance, request_id) accounting so the
    # agent's ``idle_wait_counter`` increment/decrement stays paired
    # even when the LLM polls the same pending request_id many times.

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
        # Defensive: never drop below zero. The Pydantic ``ge=0``
        # validator on the field would raise on assignment of a
        # negative value via ``model_validate``, but the direct
        # ``-=`` doesn't re-validate; assert here so a drift bug
        # surfaces loudly rather than wedging the agent loop.
        assert self._agent.idle_wait_counter > 0, (
            f"HumanApprovalCapability: idle_wait_counter underflow "
            f"on resolve of {request_id!r}"
        )
        self._agent.idle_wait_counter -= 1

    def is_awaiting_event(self) -> bool:
        """``HumanApprovalCapability`` is awaiting an event iff it
        currently holds at least one approval request whose response
        has not landed (typed event handler clears the id via
        :meth:`_on_resolved`). Used by
        ``wait_for_next_event``'s live-wake-source pre-check."""

        return bool(self._idle_waiting_request_ids)

    @action_executor(
        planning_summary=(
            "Return ``{ok, pending_request_ids: [...]}``. Each id is a "
            "request that has been posted but whose response has not "
            "yet landed."
        ),
    )
    async def list_pending(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pending_request_ids": [
                rid for rid in self._requests
                if rid not in self._responses
            ],
        }

    # ---- Approval-gate lookup -----------------------------------------
    #
    # ``ApprovalRequiredGuardrail`` calls ``has_active_approval_for``
    # at dispatch time. The blackboard is the source of truth; the
    # in-memory ``_responses`` is a cache primed by the event handler.

    _APPROVE_ALL_CHOICES = frozenset({"approve_all"})
    _APPROVE_ONCE_CHOICES = frozenset({"approve_once"})
    _APPROVE_COMPAT_CHOICES = frozenset(
        {"approve", "approved", "yes", "granted", "accept", "accepted"},
    )

    async def has_active_approval_for(
        self, action_key: str,
    ) -> tuple[bool, str | None]:
        """Return ``(True, request_id)`` if a non-revoked approval
        covers ``action_key``, else ``(False, None)``.

        Order: ``approve_all`` (no consumption) → unconsumed
        ``approve_once`` (writes consumption marker) → legacy untyped
        ``approve`` (no consumption; backwards compat for missions that
        haven't migrated to ``action_type``).
        """

        action_lower = action_key.lower()
        responses = await self._all_known_responses()

        for rid, resp in responses:
            req = self._requests.get(rid)
            atype = (req.action_type or "") if req is not None else ""
            if (
                resp.choice in self._APPROVE_ALL_CHOICES
                and atype
                and atype.lower() in action_lower
            ):
                logger.info(
                    "[Approval] has_active_approval_for: action_key=%s → MATCH rid=%s choice=approve_all atype=%s",
                    action_key, rid, atype,
                )
                return True, rid

        for rid, resp in responses:
            req = self._requests.get(rid)
            atype = (req.action_type or "") if req is not None else ""
            if (
                resp.choice in self._APPROVE_ONCE_CHOICES
                and atype
                and atype.lower() in action_lower
            ):
                if await self._is_consumed(rid):
                    logger.info(
                        "[Approval] has_active_approval_for: action_key=%s → SKIP rid=%s choice=approve_once already_consumed=true",
                        action_key, rid,
                    )
                    continue
                await self._mark_consumed(rid)
                logger.info(
                    "[Approval] has_active_approval_for: action_key=%s → MATCH rid=%s choice=approve_once consumed_now=true atype=%s",
                    action_key, rid, atype,
                )
                return True, rid

        for rid, resp in responses:
            req = self._requests.get(rid)
            atype = (req.action_type or "") if req is not None else ""
            if atype:
                continue  # typed responses already considered above
            if resp.choice.lower() in self._APPROVE_COMPAT_CHOICES:
                logger.info(
                    "[Approval] has_active_approval_for: action_key=%s → MATCH rid=%s choice=%s (legacy untyped)",
                    action_key, rid, resp.choice,
                )
                return True, rid

        logger.info(
            "[Approval] has_active_approval_for: action_key=%s → NOT_FOUND known_responses=%d",
            action_key, len(responses),
        )
        return False, None

    async def _all_known_responses(
        self,
    ) -> list[tuple[str, HumanApprovalResponse]]:
        """Snapshot of (request_id, response) pairs — cache plus any
        blackboard responses the event handler hasn't observed yet
        (e.g. resumed agent before its event loop catches up)."""

        out: dict[str, HumanApprovalResponse] = dict(self._responses)
        if not self._requests:
            return list(out.items())
        bb = await self.get_blackboard()
        for rid in self._requests:
            if rid in out:
                continue
            raw = await bb.read(HumanApprovalProtocol.response_key(rid))
            if raw is None:
                continue
            try:
                resp = HumanApprovalResponse.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            self._responses[rid] = resp
            out[rid] = resp
        return list(out.items())

    async def _is_consumed(self, request_id: str) -> bool:
        bb = await self.get_blackboard()
        marker = await bb.read(
            HumanApprovalProtocol.consumption_key(request_id),
        )
        return marker is not None

    async def _mark_consumed(self, request_id: str) -> None:
        bb = await self.get_blackboard()
        await bb.write(
            HumanApprovalProtocol.consumption_key(request_id),
            {
                "request_id": request_id,
                "consumed_at": datetime.now(timezone.utc).isoformat(),
            },
            tags={"human_approval", "consumed"},
            metadata={"request_id": request_id},
        )
        logger.info(
            "[Approval] consumption_marker_written: rid=%s",
            request_id,
        )

    # ---- Receive side -------------------------------------------------

    @event_handler(pattern=HumanApprovalProtocol.response_pattern())
    async def _on_response(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> EventProcessingResult | None:
        """Cache the user's typed response and surface it as planner context.

        The handler is deliberately small: it does not advance the
        agent state itself. The next planner iteration sees a fresh
        ``{RESPONSE_CONTEXT_KEY_PREFIX}{request_id}`` context binding
        (prefix on :attr:`RESPONSE_CONTEXT_KEY_PREFIX`) and plans the
        appropriate downstream action (commit, retry, abandon —
        domain-specific).
        """

        try:
            request_id = HumanApprovalProtocol.parse_response_key(event.key)
        except ValueError:
            return None
        if not isinstance(event.value, dict):
            return None
        try:
            response = HumanApprovalResponse.model_validate(event.value)
        except Exception:  # noqa: BLE001
            logger.warning(
                "HumanApprovalCapability: dropping malformed response event %s",
                event.key,
            )
            return None
        self._responses[request_id] = response
        logger.info(
            "[Approval] response_cached: rid=%s choice=%s decided_by=%s explanation_len=%d",
            request_id,
            response.choice,
            response.decided_by,
            len(response.explanation or ""),
        )
        # Resolution arrived via the event stream — decrement the
        # idle-wait counter if a prior get_response poll had
        # registered this request_id as pending.
        self._on_resolved(request_id)
        return EventProcessingResult(
            context_key=f"{self.RESPONSE_CONTEXT_KEY_PREFIX}{request_id}",
            context={
                "request_id": request_id,
                "choice": response.choice,
                "explanation": response.explanation,
                "note": response.note,
                "decided_by": response.decided_by,
            },
        )

    # ---- Suspension hooks ---------------------------------------------

    _CUSTOM_DATA_KEY = "human_approval_capability"

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        if not self._requests and not self._responses:
            return state
        state.custom_data[self._CUSTOM_DATA_KEY] = {
            "requests": [
                r.model_dump(mode="json") for r in self._requests.values()
            ],
            "responses": [
                r.model_dump(mode="json") for r in self._responses.values()
            ],
        }
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        payload = state.custom_data.get(self._CUSTOM_DATA_KEY) or {}
        for raw in payload.get("requests") or ():
            try:
                req = HumanApprovalRequest.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            self._requests[req.request_id] = req
        for raw in payload.get("responses") or ():
            try:
                resp = HumanApprovalResponse.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            self._responses[resp.request_id] = resp


__all__ = (
    "HumanApprovalCapability",
    "HumanApprovalRequest",
    "HumanApprovalResponse",
)
