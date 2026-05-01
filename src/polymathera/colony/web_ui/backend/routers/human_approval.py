"""HTTP endpoints for the human-approval gate.

The colony's :class:`~polymathera.colony.agents.patterns.capabilities.HumanApprovalCapability`
is session-scoped: an agent posts a typed
:class:`~polymathera.colony.agents.patterns.capabilities.human_approval.HumanApprovalRequest`
to the session blackboard, and a typed
:class:`~polymathera.colony.agents.patterns.capabilities.human_approval.HumanApprovalResponse`
must land on the same scope for the agent's ``@event_handler`` to fire.

This module exposes the HTTP receiver that turns a user's UI click
into the response write. It is the **only** path from the browser
back into the agent loop — the previous design had ``record_human_approval``
as an ``@action_executor`` with no callable wiring from outside the
agent process; this fixes that.

Endpoints
---------

- ``POST /api/v1/sessions/{session_id}/human_approval/{request_id}/respond``
  — the user submits ``{choice, note}``. The handler auth-checks,
  resolves the session's tenant/colony, sets a USER ``ExecutionContext``,
  constructs a session-scoped blackboard handle, and writes a
  :class:`HumanApprovalResponse` to ``human_approval:response:{request_id}``.

- ``GET /api/v1/sessions/{session_id}/human_approval/pending``
  — list pending request payloads. Useful for SPA polling fallback;
  the primary delivery path is the WebSocket relay in the chat
  router (which sees requests via the session's chat scope).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanApprovalProtocol
from polymathera.colony.agents.patterns.capabilities.human_approval import (
    HumanApprovalCapability,
    HumanApprovalResponse,
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


class HumanApprovalSubmission(BaseModel):
    """Body of the POST endpoint. The user-supplied ``choice`` MUST
    be one of the options the original request declared; the agent's
    capability re-validates on receive."""

    model_config = ConfigDict(frozen=True)

    choice: str = Field(min_length=1)
    note: str = ""


class HumanApprovalSubmissionResult(BaseModel):
    """Echo of what the handler wrote back, so the SPA can mark the
    submission optimistically without a second round-trip."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    choice: str
    decided_by: str
    accepted: bool
    """Always ``True`` on success — ``False`` is reported via HTTP
    status codes so SPAs that only inspect status work as expected."""


@router.post(
    "/sessions/{session_id}/human_approval/{request_id}/respond",
    response_model=HumanApprovalSubmissionResult,
)
async def respond_to_human_approval(
    session_id: str,
    request_id: str,
    body: HumanApprovalSubmission,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> HumanApprovalSubmissionResult:
    """Persist the user's typed response to the session blackboard.

    The capability's ``@event_handler`` picks the write up via the
    normal blackboard event stream and surfaces the choice as planner
    context on the requesting agent's next iteration.
    """

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

    response = HumanApprovalResponse(
        request_id=request_id,
        choice=body.choice,
        note=body.note,
        decided_by=decided_by,
    )

    with execution_context(
        ring=Ring.USER,
        tenant_id=session_info["tenant_id"],
        colony_id=session_info["colony_id"],
        session_id=session_id,
        origin="human_approval_endpoint",
    ):
        bb = EnhancedBlackboard(
            app_name=colony.app_name,
            scope_id=get_scope_prefix(
                BlackboardScope.SESSION,
                namespace=HumanApprovalCapability.DEFAULT_NAMESPACE,
            ),
            backend_type=None,
            enable_events=True,
        )
        await bb.initialize()
        try:
            await bb.write(
                HumanApprovalProtocol.response_key(request_id),
                response.model_dump(mode="json"),
                tags={"human_approval", "response"},
                metadata={
                    "request_id": request_id,
                    "decided_by": decided_by,
                },
            )
        finally:
            await bb.stop()

    return HumanApprovalSubmissionResult(
        request_id=request_id,
        choice=body.choice,
        decided_by=decided_by,
        accepted=True,
    )


@router.get("/sessions/{session_id}/human_approval/pending")
async def list_pending_human_approvals(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Return the request payloads still awaiting a response.

    The primary delivery path is the WebSocket relay; this endpoint
    is a polling fallback for SPA clients that prefer not to maintain
    a socket open. Returns the raw blackboard payloads (already
    JSON-serialisable :class:`HumanApprovalRequest` shapes).
    """

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
        origin="human_approval_endpoint",
    ):
        bb = EnhancedBlackboard(
            app_name=colony.app_name,
            scope_id=get_scope_prefix(
                BlackboardScope.SESSION,
                namespace=HumanApprovalCapability.DEFAULT_NAMESPACE,
            ),
            backend_type=None,
            enable_events=False,
        )
        await bb.initialize()
        try:
            requests = await bb.query(namespace=HumanApprovalProtocol.request_pattern())
            responses = await bb.query(namespace=HumanApprovalProtocol.response_pattern())
        finally:
            await bb.stop()

    responded_ids = {
        HumanApprovalProtocol.parse_response_key(entry.key)
        for entry in responses
        if entry.key.startswith("human_approval:response:")
    }
    pending: list[dict[str, Any]] = []
    for entry in requests:
        if not entry.key.startswith("human_approval:request:"):
            continue
        request_id = HumanApprovalProtocol.parse_request_key(entry.key)
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
        with colony.kernel_execution_context(origin="human_approval_endpoint"):
            sm = colony.get_session_manager()
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
