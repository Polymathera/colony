"""HTTP endpoints for the human-help (mid-flow clarification) gate.

Sibling of :mod:`~polymathera.colony.web_ui.backend.routers.human_approval`.
Both are session-scoped human-in-the-loop channels but they cover
semantically distinct cases:

- ``human_approval`` — *"authorise THIS specific dispatch"*. Operator
  picks one of a fixed enum (``approve_once`` / ``approve_all`` /
  ``reject`` / ``abort``) over a known, named action.
- ``human_help`` — *"I'm stuck on a judgment call; what should I do?"*.
  Operator either picks one of the agent's candidate ``options`` (a
  free-form label list — NOT a fixed enum) OR writes free-form
  ``guidance``. Used by the SessionAgent (pre-spawn) when user intent
  is incomplete relative to the chosen mission's parameters AND by
  coordinator / worker agents mid-run when new info surfaces that
  the operator must adjudicate.

Endpoints
---------

- ``POST /api/v1/sessions/{session_id}/human_help/{request_id}/respond``
  — operator submits ``{chosen_option?, guidance?}``; the handler
  auth-checks, resolves the session's tenant/colony, sets a USER
  ``ExecutionContext``, builds the session-scoped blackboard handle,
  and writes a :class:`HumanHelpResponse` to
  ``human_help:response:{request_id}``. At least one of
  ``chosen_option`` / ``guidance`` MUST be non-empty — the
  ``HumanHelpResponse`` model validator surfaces a 422 if both are
  blank.

- ``GET /api/v1/sessions/{session_id}/human_help/pending``
  — list pending request payloads. Polling fallback for SPA clients;
  the primary delivery path is the WebSocket relay in the chat router
  (which sees requests via the SessionAgent's
  ``handle_human_help_request`` translator and the session chat
  scope's existing ``chat:agent:*`` relay).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanHelpProtocol
from polymathera.colony.agents.patterns.capabilities.human_help import (
    HumanHelpCapability,
    HumanHelpResponse,
)
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection


logger = logging.getLogger(__name__)
router = APIRouter()


class HumanHelpSubmission(BaseModel):
    """Body of the POST endpoint. At least one of ``chosen_option`` /
    ``guidance`` MUST be non-empty — the ``HumanHelpResponse`` Pydantic
    model validator enforces this at construction time, so the endpoint
    surfaces a 422 if the SPA forgets both fields. ``chosen_option`` is
    one of the agent's request-time ``options`` (the agent re-validates
    on receive); ``guidance`` is free-form direction the operator
    writes when the options don't fit."""

    model_config = ConfigDict(frozen=True)

    chosen_option: str | None = Field(default=None)
    guidance: str = ""


class HumanHelpSubmissionResult(BaseModel):
    """Echo of what the handler wrote back, so the SPA can mark the
    submission optimistically without a second round-trip."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    chosen_option: str | None
    decided_by: str
    accepted: bool


@router.post(
    "/sessions/{session_id}/human_help/{request_id}/respond",
    response_model=HumanHelpSubmissionResult,
)
async def respond_to_human_help(
    session_id: str,
    request_id: str,
    body: HumanHelpSubmission,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> HumanHelpSubmissionResult:
    """Persist the operator's typed help response to the session
    blackboard. The :class:`HumanHelpCapability`'s ``@event_handler``
    picks the write up via the normal blackboard event stream and
    surfaces the response as planner context on the requesting
    agent's next iteration (key prefix
    :attr:`HumanHelpCapability.RESPONSE_CONTEXT_KEY_PREFIX`)."""

    if not colony.is_connected:
        raise HTTPException(status_code=503, detail="cluster not connected")

    session_info = await _get_session_meta(colony, session_id)
    if session_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"session {session_id!r} not found",
        )

    decided_by = (
        user.get("display_name")
        or user.get("email")
        or user.get("user_id")
        or "unknown"
    )

    try:
        response = HumanHelpResponse(
            request_id=request_id,
            chosen_option=body.chosen_option,
            guidance=body.guidance,
            decided_by=decided_by,
        )
    except ValueError as exc:
        # The ``HumanHelpResponse._require_chosen_option_or_guidance``
        # model validator raises when both are empty / whitespace.
        # Surface as 422 so the SPA can show the operator a "pick an
        # option or write guidance" hint.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with execution_context(
        ring=Ring.USER,
        tenant_id=session_info["tenant_id"],
        colony_id=session_info["colony_id"],
        session_id=session_id,
        origin="human_help_endpoint",
    ):
        bb = EnhancedBlackboard(
            app_name=colony.app_name,
            scope_id=get_scope_prefix(
                BlackboardScope.SESSION,
                namespace=HumanHelpCapability.DEFAULT_NAMESPACE,
            ),
            backend_type=None,
            enable_events=True,
        )
        await bb.initialize()
        try:
            await bb.write(
                HumanHelpProtocol.response_key(request_id),
                response.model_dump(mode="json"),
                tags={"human_help", "response"},
                metadata={
                    "request_id": request_id,
                    "decided_by": decided_by,
                },
            )
        finally:
            await bb.stop()

    return HumanHelpSubmissionResult(
        request_id=request_id,
        chosen_option=body.chosen_option,
        decided_by=decided_by,
        accepted=True,
    )


@router.get("/sessions/{session_id}/human_help/pending")
async def list_pending_human_help(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Return the help-request payloads still awaiting a response.
    Primary delivery path is the WebSocket relay; this endpoint is a
    polling fallback for SPA clients that prefer not to maintain a
    socket open. Returns the raw blackboard payloads (already JSON-
    serialisable :class:`HumanHelpRequest` shapes)."""

    if not colony.is_connected:
        raise HTTPException(status_code=503, detail="cluster not connected")

    session_info = await _get_session_meta(colony, session_id)
    if session_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"session {session_id!r} not found",
        )

    with execution_context(
        ring=Ring.USER,
        tenant_id=session_info["tenant_id"],
        colony_id=session_info["colony_id"],
        session_id=session_id,
        origin="human_help_endpoint",
    ):
        bb = EnhancedBlackboard(
            app_name=colony.app_name,
            scope_id=get_scope_prefix(
                BlackboardScope.SESSION,
                namespace=HumanHelpCapability.DEFAULT_NAMESPACE,
            ),
            backend_type=None,
            enable_events=False,
        )
        await bb.initialize()
        try:
            requests = await bb.query(namespace=HumanHelpProtocol.request_pattern())
            responses = await bb.query(namespace=HumanHelpProtocol.response_pattern())
        finally:
            await bb.stop()

    responded_ids = {
        HumanHelpProtocol.parse_response_key(entry.key)
        for entry in responses
        if entry.key.startswith("human_help:response:")
    }
    pending: list[dict[str, Any]] = []
    for entry in requests:
        if not entry.key.startswith("human_help:request:"):
            continue
        request_id = HumanHelpProtocol.parse_request_key(entry.key)
        if request_id in responded_ids:
            continue
        if isinstance(entry.value, dict):
            pending.append(entry.value)
    return pending


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _get_session_meta(
    colony: ColonyConnection, session_id: str,
) -> dict[str, str] | None:
    """Resolve the tenant_id / colony_id needed to compute the
    session-scoped blackboard scope_id."""

    try:
        with colony.kernel_execution_context(origin="human_help_endpoint"):
            sm = await colony.get_session_manager()
            session = await sm.get_session(session_id=session_id)
            if session is None:
                return None
            from polymathera.colony.agents.sessions.models import Session as SessionModel
            if isinstance(session, dict):
                session = SessionModel(**session)
            return {
                "tenant_id": session.tenant_id,
                "colony_id": session.colony_id,
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to look up session info for %s: %s", session_id, e,
        )
        return None
