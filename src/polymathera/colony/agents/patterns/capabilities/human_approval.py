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
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar

from overrides import override
from pydantic import BaseModel, ConfigDict, Field

from ...base import Agent, AgentCapability
from ...blackboard import BlackboardEvent
from ...blackboard.protocol import HumanApprovalProtocol
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor
from ..events import EventProcessingResult, event_handler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire payloads
# ---------------------------------------------------------------------------


class HumanApprovalRequest(BaseModel):
    """Payload an agent posts to ``human_approval:request:{request_id}``."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        default_factory=lambda: f"appr_{uuid.uuid4().hex[:12]}",
    )
    question: str
    options: tuple[str, ...] = ("approve", "reject")
    requester_agent_id: str | None = None
    """Agent that asked. ``None`` in detached / test contexts."""

    deadline: datetime | None = None
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    extra: dict[str, Any] = Field(default_factory=dict)
    """Free-form context the requester wants the UI to render alongside
    the question (e.g., a diff, a summary, a list of affected pages)."""


class HumanApprovalResponse(BaseModel):
    """Payload the Web UI posts to ``human_approval:response:{request_id}``."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    choice: str
    note: str = ""
    decided_by: str = ""
    """User id (or display name) of the human who responded."""

    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


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

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"human_approval", "gate", "session"})

    # ---- Request side -------------------------------------------------

    @action_executor(
        planning_summary=(
            "Submit a typed human-approval request. Returns immediately "
            "with the request_id. The user's response surfaces later "
            "via the capability's event handler — query it with "
            "get_response or wait for the planner to observe the new "
            "context."
        ),
    )
    async def request_human_approval(
        self,
        question: str,
        *,
        options: tuple[str, ...] = ("approve", "reject"),
        deadline: datetime | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Post a request and return its ``request_id``."""

        request = HumanApprovalRequest(
            question=question,
            options=options,
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
            },
        )
        self._requests[request.request_id] = request
        return request.request_id

    @action_executor(
        planning_summary=(
            "Return the recorded response for a request, or None if "
            "the user has not responded yet. Falls back to a blackboard "
            "read so a resumed agent can recover responses that landed "
            "during suspension."
        ),
    )
    async def get_response(
        self, request_id: str,
    ) -> HumanApprovalResponse | None:
        cached = self._responses.get(request_id)
        if cached is not None:
            return cached
        bb = await self.get_blackboard()
        raw = await bb.read(HumanApprovalProtocol.response_key(request_id))
        if raw is None:
            return None
        try:
            response = HumanApprovalResponse.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.warning(
                "HumanApprovalCapability: malformed response payload at %s",
                HumanApprovalProtocol.response_key(request_id),
            )
            return None
        self._responses[request_id] = response
        return response

    @action_executor(planning_summary="List request_ids still awaiting a response.")
    async def list_pending(self) -> list[str]:
        return [
            rid for rid in self._requests
            if rid not in self._responses
        ]

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
        ``human_approval_response:{request_id}`` context binding and
        plans the appropriate downstream action (commit, retry,
        abandon — domain-specific).
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
        return EventProcessingResult(
            context_key=f"human_approval_response:{request_id}",
            context={
                "request_id": request_id,
                "choice": response.choice,
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
