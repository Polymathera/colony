"""Session and run management endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..models.api_models import RunSummary, SessionSummary
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_APP_NAME = "polymathera"


@router.get("/sessions/", response_model=list[SessionSummary])
async def list_sessions(
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(100, le=500),
    colony: ColonyConnection = Depends(get_colony),
):
    """List sessions, optionally filtered by tenant."""
    if not colony.is_connected:
        return []

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "session_manager")
        sessions = await handle.list_sessions(
            tenant_id=tenant_id,
            include_expired=False,
            limit=limit,
        )

        return [
            SessionSummary(
                session_id=s.session_id,
                tenant_id=getattr(s, "tenant_id", ""),
                state=str(getattr(s, "state", "")),
                created_at=getattr(s, "created_at", 0.0),
                run_count=len(getattr(s, "runs", [])),
            )
            for s in sessions
        ]

    except Exception as e:
        logger.warning(f"Failed to list sessions: {e}")
        return []


@router.get("/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed session information."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "session_manager")
        session = await handle.get_session(session_id=session_id)
        if session is None:
            return {"error": "session not found", "session_id": session_id}
        if hasattr(session, "model_dump"):
            return session.model_dump()
        return {"session_id": session_id, "raw": str(session)}

    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@router.get("/sessions/{session_id}/runs", response_model=list[RunSummary])
async def get_session_runs(
    session_id: str,
    limit: int = Query(100, le=500),
    colony: ColonyConnection = Depends(get_colony),
):
    """List runs for a specific session."""
    if not colony.is_connected:
        return []

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "session_manager")
        runs = await handle.get_session_runs(session_id=session_id, limit=limit)

        return [
            RunSummary(
                run_id=r.run_id,
                session_id=getattr(r, "session_id", session_id),
                agent_id=getattr(r, "agent_id", ""),
                status=str(getattr(r, "status", "")),
                started_at=getattr(r, "started_at", None),
                completed_at=getattr(r, "completed_at", None),
                input_tokens=getattr(getattr(r, "resources", None), "input_tokens", 0),
                output_tokens=getattr(getattr(r, "resources", None), "output_tokens", 0),
            )
            for r in runs
        ]

    except Exception as e:
        logger.warning(f"Failed to list runs for session {session_id}: {e}")
        return []


@router.get("/sessions/runs/{run_id}")
async def get_run_detail(
    run_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed run information including events and resource usage."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "session_manager")
        run = await handle.get_run(run_id=run_id)
        if run is None:
            return {"error": "run not found", "run_id": run_id}
        if hasattr(run, "model_dump"):
            return run.model_dump()
        return {"run_id": run_id, "raw": str(run)}

    except Exception as e:
        return {"error": str(e), "run_id": run_id}


@router.get("/sessions/stats/overview")
async def get_session_stats(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get session manager statistics."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "session_manager")
        return await handle.get_stats()
    except Exception as e:
        return {"error": str(e)}
