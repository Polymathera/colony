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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from object or dict — handles both Pydantic models and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@router.get("/sessions/", response_model=list[SessionSummary])
async def list_sessions(
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    colony_id: str | None = Query(None, description="Filter by colony"),
    limit: int = Query(100, le=500),
    colony: ColonyConnection = Depends(get_colony),
):
    """List sessions, optionally filtered by tenant."""
    if not colony.is_connected:
        return []

    with colony.user_execution_context(
        tenant_id=tenant_id,
        colony_id=colony_id,
        origin="dashboard",
    ):
        try:
            handle = colony.get_session_manager()
            sessions = await handle.list_sessions(
                tenant_id=tenant_id,
                colony_id=colony_id,
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
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed session information."""
    if not colony.is_connected:
        return {"error": "not connected"}

    with colony.user_execution_context(
        session_id=session_id,
        origin="dashboard",
    ):
        try:
            handle = colony.get_session_manager()
            session = await handle.get_session(session_id=session_id)
            if session is None:
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
    colony: ColonyConnection = Depends(get_colony),
):
    """List runs for a specific session."""
    if not colony.is_connected:
        return []

    with colony.user_execution_context(
        session_id=session_id,
        origin="dashboard",
    ):
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
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed run information including events and resource usage."""
    if not colony.is_connected:
        return {"error": "not connected"}

    with colony.user_execution_context(
        run_id=run_id,
        origin="dashboard",
    ):
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
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get session manager statistics."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    with colony.kernel_execution_context(origin="dashboard"):
        try:
            handle = colony.get_session_manager()
            return await handle.get_stats()
        except Exception as e:
            return {"error": str(e)}
