"""Session and run management endpoints.

All endpoints require authentication. The AuthMiddleware sets the
ExecutionContext (Ring.USER with tenant_id from JWT, colony_id from
X-Colony-Id header) before any endpoint code runs. Deployment handle
calls automatically propagate this context across Ray boundaries.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..models.api_models import RunSummary, SessionSummary
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    name: str | None = Field(default=None, description="Human-readable session name")
    ttl_seconds: float | None = Field(default=None, description="Session TTL (None = use default)")
    fork_from_session_id: str | None = Field(default=None, description="Fork from existing session")


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    status: str  # "created", "error"
    message: str = ""


class SessionActionResponse(BaseModel):
    """Response from session state change."""

    session_id: str
    success: bool
    message: str = ""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from object or dict — handles both Pydantic models and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------

@router.get("/sessions/", response_model=list[SessionSummary])
async def list_sessions(
    limit: int = Query(100, le=500),
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """List sessions for the authenticated user's tenant and active colony."""
    if not colony.is_connected:
        return []

    from polymathera.colony.distributed.ray_utils.serving.context import get_tenant_id, get_colony_id

    try:
        handle = colony.get_session_manager()
        sessions = await handle.list_sessions(
            tenant_id=get_tenant_id(),
            colony_id=get_colony_id(),
            include_expired=False,
            limit=limit,
        )

        return [
            SessionSummary(
                session_id=_get(s, "session_id", ""),
                tenant_id=_get(s, "tenant_id", ""),
                colony_id=_get(s, "colony_id", ""),
                state=str(_get(s, "state", "")),
                created_at=_get(s, "created_at", 0.0),
                run_count=len(_get(s, "runs", []) or []),
            )
            for s in sessions
        ]

    except Exception as e:
        logger.warning(f"Failed to list sessions: {e}")
        return []


@router.get("/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed session information."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_session_manager()
        session = await handle.get_session(session_id=session_id)
        if session is None:
            return {"error": "session not found", "session_id": session_id}

        # Verify the session belongs to this user's tenant
        from polymathera.colony.distributed.ray_utils.serving.context import get_tenant_id
        session_tenant = _get(session, "tenant_id", "")
        if session_tenant and session_tenant != get_tenant_id():
            return {"error": "session not found", "session_id": session_id}

        if isinstance(session, dict):
            return session
        if hasattr(session, "model_dump"):
            return session.model_dump()
        return {"session_id": session_id, "raw": str(session)}

    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@router.get("/sessions/{session_id}/runs", response_model=list[RunSummary])
async def get_session_runs(
    session_id: str,
    limit: int = Query(100, le=500),
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """List runs for a specific session."""
    if not colony.is_connected:
        return []

    try:
        handle = colony.get_session_manager()
        runs = await handle.get_session_runs(session_id=session_id, limit=limit)

        result = []
        for r in runs:
            ru = _get(r, "resource_usage", None)
            result.append(RunSummary(
                run_id=_get(r, "run_id", ""),
                session_id=_get(r, "session_id", session_id),
                agent_id=_get(r, "agent_id", ""),
                status=str(_get(r, "status", "")),
                started_at=_get(r, "started_at", None),
                completed_at=_get(r, "completed_at", None),
                input_tokens=_get(ru, "input_tokens", 0) if ru else 0,
                output_tokens=_get(ru, "output_tokens", 0) if ru else 0,
            ))
        return result

    except Exception as e:
        logger.warning(f"Failed to list runs for session {session_id}: {e}")
        return []


@router.get("/sessions/runs/{run_id}")
async def get_run_detail(
    run_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed run information including events and resource usage."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_session_manager()
        run = await handle.get_run(run_id=run_id)
        if run is None:
            return {"error": "run not found", "run_id": run_id}
        if isinstance(run, dict):
            return run
        if hasattr(run, "model_dump"):
            return run.model_dump()
        return {"run_id": run_id, "raw": str(run)}

    except Exception as e:
        return {"error": str(e), "run_id": run_id}


@router.get("/sessions/stats/overview")
async def get_session_stats(
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get session manager statistics."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    try:
        handle = colony.get_session_manager()
        return await handle.get_stats()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Write endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest | None = None,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> CreateSessionResponse:
    """Create a new session.

    The session is created under the authenticated user's tenant.
    The colony_id comes from the X-Colony-Id header (set by the frontend).
    """
    if not colony.is_connected:
        return CreateSessionResponse(session_id="", status="error", message="Not connected")

    request = request or CreateSessionRequest()

    try:
        from polymathera.colony.agents.sessions.models import SessionMetadata
        metadata = SessionMetadata(name=request.name) if request.name else None

        sm = colony.get_session_manager()
        session = await sm.create_session(
            metadata=metadata,
            ttl_seconds=request.ttl_seconds,
            fork_from_session_id=request.fork_from_session_id,
        )
        session_id = _get(session, "session_id", "")

        # Spawn a SessionAgent for this session.
        # The spawn needs session_id in the execution context — the AuthMiddleware
        # only sets tenant_id and colony_id. Wrap in user_execution_context.
        session_agent_id: str | None = None
        try:
            from polymathera.colony.agents import AgentMetadata, AgentHandle
            from polymathera.colony.agents.self_concept import AgentSelfConcept
            from ..chat import SessionAgent, SessionOrchestratorCapability
            from polymathera.colony.agents.patterns.capabilities.agent_pool import AgentPoolCapability
            from polymathera.colony.agents.patterns.capabilities.consciousness import ConsciousnessCapability
            from polymathera.colony.cli.polymath import ANALYSIS_REGISTRY
            from polymathera.colony.distributed.ray_utils.serving.context import get_tenant_id, get_colony_id

            # Build the available analysis types info for the LLM planner
            available_analyses = {
                atype: {
                    "label": reg["label"],
                    "description": reg.get("description", ""),
                    "coordinator_class": reg.get("coordinator_v2", ""),
                    "worker_class": reg.get("worker", ""),
                }
                for atype, reg in ANALYSIS_REGISTRY.items()
            }

            agent_metadata = AgentMetadata(
                role="session_orchestrator",
                session_id=session_id,
                goals=[
                    "Orchestrate user interactions within this session",
                    "Interpret user intent and decide whether to respond directly or spawn analysis agents",
                    "Spawn and monitor coordinator agents for code analysis tasks",
                    "Relay agent progress and results back to the user",
                ],
                self_concept=AgentSelfConcept(
                    agent_id="",  # Overwritten by ConsciousnessCapability
                    name="Session Agent",
                    role=(
                        "the session orchestrator responsible for handling all user "
                        "interactions in this session. You receive user messages, decide "
                        "how to handle them, and can either respond directly or spawn "
                        "specialized coordinator agents to perform code analysis tasks."
                    ),
                    description=(
                        "You are the primary interface between the user and the Colony's "
                        "multi-agent system. You have access to an agent pool for spawning "
                        "coordinator agents (one per analysis type), and you can respond "
                        "directly to user questions. When the user requests an analysis, "
                        "use create_agent to spawn the appropriate coordinator. When the "
                        "user asks a question or needs information, use respond_to_user."
                    ),
                ),
                parameters={
                    "available_analyses": available_analyses,
                    "session_id": session_id,
                },
            )

            bp = SessionAgent.bind(
                agent_type=SessionAgent.model_fields["agent_type"].default,
                metadata=agent_metadata,
                bound_pages=[],
                capability_blueprints=[
                    SessionOrchestratorCapability.bind(),
                    AgentPoolCapability.bind(),
                    ConsciousnessCapability.bind(),
                ],
            )

            with colony.user_execution_context(
                tenant_id=get_tenant_id(),
                colony_id=get_colony_id(),
                session_id=session_id,
                origin="dashboard_session_create",
            ):
                handle = await AgentHandle.from_blueprint(
                    agent_blueprint=bp,
                    app_name=colony.app_name,
                )
            session_agent_id = handle.agent_id
            logger.info("Spawned SessionAgent %s for session %s", session_agent_id, session_id)
        except Exception as e:
            logger.error("Failed to spawn SessionAgent for session %s: %s", session_id, e)
            # Session still works without an agent — chat will fall back to direct agent routing

        session_agent_id1 = await sm.set_session_agent_id(
            session_id=session_id,
            agent_id=session_agent_id
        )
        if session_agent_id1 != session_agent_id:
            return CreateSessionResponse(
                session_id=session_id,
                status="error",
                message=f"Session created but failed to set session agent ID. Expected {session_agent_id}, got {session_agent_id1}",
            )

        return CreateSessionResponse(session_id=session_id, status="created")

    except Exception as e:
        logger.error("Failed to create session: %s", e)
        return CreateSessionResponse(session_id="", status="error", message=str(e))


@router.put("/sessions/{session_id}/suspend", response_model=SessionActionResponse)
async def suspend_session(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Suspend an active session."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = colony.get_session_manager()
        success = await handle.suspend_session(session_id=session_id)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message="Suspended" if success else "Failed to suspend",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))


@router.put("/sessions/{session_id}/resume", response_model=SessionActionResponse)
async def resume_session(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Resume a suspended session."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = colony.get_session_manager()
        success = await handle.activate_session(session_id=session_id)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message="Resumed" if success else "Failed to resume",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionActionResponse)
async def close_session(
    session_id: str,
    archive: bool = Query(True, description="Archive session data"),
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Close and optionally archive a session."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = colony.get_session_manager()
        success = await handle.close_session(session_id=session_id, archive=archive)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message="Closed" if success else "Failed to close",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))


@router.post("/sessions/{session_id}/runs/{run_id}/cancel", response_model=SessionActionResponse)
async def cancel_run(
    session_id: str,
    run_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Cancel a running agent run."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = colony.get_session_manager()
        success = await handle.cancel_run(run_id=run_id)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message=f"Run {run_id} cancelled" if success else f"Failed to cancel run {run_id}",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))
