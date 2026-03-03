"""Metrics and token usage endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
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


@router.get("/metrics/tokens")
async def get_token_usage(
    session_id: str | None = Query(None),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get aggregated token usage from session runs.

    Returns per-run token counts for charting.
    """
    if not colony.is_connected:
        return {"runs": [], "totals": {}, "error": "not connected"}

    try:
        handle = colony.get_session_manager()

        if session_id:
            runs = await handle.get_session_runs(session_id=session_id, limit=200)
        else:
            # Get runs across all sessions
            sessions = await handle.list_sessions(limit=20, include_expired=True)
            logger.info("Metrics: found %d sessions", len(sessions))
            runs = []
            for s in sessions:
                sid = _get(s, "session_id", "")
                session_runs = await handle.get_session_runs(
                    session_id=sid, limit=50,
                )
                logger.info("Metrics: session %s → %d runs", sid[:16], len(session_runs))
                runs.extend(session_runs)

        token_data = []
        for r in runs:
            ru = _get(r, "resource_usage", None)
            token_data.append({
                "run_id": _get(r, "run_id", ""),
                "agent_id": _get(r, "agent_id", ""),
                "status": str(_get(r, "status", "")),
                "input_tokens": _get(ru, "input_tokens", 0) if ru else 0,
                "output_tokens": _get(ru, "output_tokens", 0) if ru else 0,
                "cache_read_tokens": _get(ru, "cache_read_tokens", 0) if ru else 0,
                "cache_write_tokens": _get(ru, "cache_write_tokens", 0) if ru else 0,
                "llm_calls": _get(ru, "llm_calls", 0) if ru else 0,
                "cost_usd": _get(ru, "cost_usd", 0.0) if ru else 0.0,
                "started_at": _get(r, "started_at", None),
            })

        # Aggregate totals
        total_input = sum(d["input_tokens"] for d in token_data)
        total_output = sum(d["output_tokens"] for d in token_data)
        total_cache_read = sum(d["cache_read_tokens"] for d in token_data)
        total_cost = sum(d["cost_usd"] for d in token_data)

        # Aggregate by agent
        by_agent: dict[str, dict[str, Any]] = {}
        for d in token_data:
            aid = d["agent_id"] or "unknown"
            if aid not in by_agent:
                by_agent[aid] = {
                    "agent_id": aid,
                    "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "llm_calls": 0, "cost_usd": 0.0, "run_count": 0,
                }
            agg = by_agent[aid]
            agg["input_tokens"] += d["input_tokens"]
            agg["output_tokens"] += d["output_tokens"]
            agg["cache_read_tokens"] += d["cache_read_tokens"]
            agg["cache_write_tokens"] += d["cache_write_tokens"]
            agg["llm_calls"] += d["llm_calls"]
            agg["cost_usd"] += d["cost_usd"]
            agg["run_count"] += 1

        return {
            "runs": token_data,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cache_read_tokens": total_cache_read,
                "total_tokens": total_input + total_output,
                "cost_usd": total_cost,
                "run_count": len(token_data),
            },
            "by_agent": list(by_agent.values()),
        }

    except Exception as e:
        logger.warning("Failed to get token usage: %s", e, exc_info=True)
        return {"runs": [], "totals": {}, "error": str(e)}


@router.get("/metrics/debug")
async def get_metrics_debug(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Raw diagnostic: dump exactly what session_manager returns.

    Hit /api/v1/metrics/debug in your browser to see what's happening.
    """
    result: dict[str, Any] = {
        "connected": colony.is_connected,
        "sessions_raw": [],
        "errors": [],
    }
    if not colony.is_connected:
        result["errors"].append("Colony not connected")
        return result

    try:
        handle = colony.get_session_manager()
        result["handle_type"] = str(type(handle))
    except Exception as e:
        result["errors"].append(f"get_session_manager failed: {e}")
        return result

    try:
        sessions = await handle.list_sessions(limit=10, include_expired=True)
        result["sessions_count"] = len(sessions)
        result["sessions_type"] = str(type(sessions))

        for i, s in enumerate(sessions[:3]):
            s_info: dict[str, Any] = {
                "python_type": str(type(s)),
                "is_dict": isinstance(s, dict),
            }
            sid = _get(s, "session_id", "???")
            s_info["session_id"] = sid
            s_info["state"] = str(_get(s, "state", "???"))
            s_info["tenant_id"] = _get(s, "tenant_id", "???")

            # Get runs for this session
            try:
                runs = await handle.get_session_runs(session_id=sid, limit=5)
                s_info["runs_count"] = len(runs)
                s_info["runs_type"] = str(type(runs))
                for j, r in enumerate(runs[:2]):
                    r_info: dict[str, Any] = {
                        "python_type": str(type(r)),
                        "is_dict": isinstance(r, dict),
                        "run_id": _get(r, "run_id", "???"),
                        "agent_id": _get(r, "agent_id", "???"),
                        "status": str(_get(r, "status", "???")),
                    }
                    ru = _get(r, "resource_usage", None)
                    r_info["resource_usage_type"] = str(type(ru))
                    r_info["resource_usage_is_dict"] = isinstance(ru, dict)
                    if ru:
                        r_info["input_tokens"] = _get(ru, "input_tokens", 0)
                        r_info["output_tokens"] = _get(ru, "output_tokens", 0)
                        r_info["llm_calls"] = _get(ru, "llm_calls", 0)
                        r_info["cost_usd"] = _get(ru, "cost_usd", 0.0)
                    else:
                        r_info["resource_usage"] = None
                    s_info[f"run_{j}"] = r_info
            except Exception as e:
                s_info["runs_error"] = str(e)

            result["sessions_raw"].append(s_info)

    except Exception as e:
        result["errors"].append(f"list_sessions failed: {e}")
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    query: str = Query("up", description="PromQL query"),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Proxy a PromQL query to the Prometheus endpoint on the Ray head."""
    return await colony.query_prometheus(query)
