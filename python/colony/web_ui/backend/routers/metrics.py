"""Metrics and token usage endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_APP_NAME = "polymathera"


@router.get("/metrics/tokens")
async def get_token_usage(
    session_id: str | None = Query(None),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get aggregated token usage from session runs.

    Returns per-run token counts for charting.
    """
    if not colony.is_connected:
        return {"runs": []}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "session_manager")

        if session_id:
            runs = await handle.get_session_runs(session_id=session_id, limit=200)
        else:
            # Get runs across all sessions
            sessions = await handle.list_sessions(limit=20)
            runs = []
            for s in sessions:
                session_runs = await handle.get_session_runs(
                    session_id=s.session_id, limit=50,
                )
                runs.extend(session_runs)

        token_data = []
        for r in runs:
            resources = getattr(r, "resources", None)
            token_data.append({
                "run_id": r.run_id,
                "agent_id": getattr(r, "agent_id", ""),
                "status": str(getattr(r, "status", "")),
                "input_tokens": getattr(resources, "input_tokens", 0) if resources else 0,
                "output_tokens": getattr(resources, "output_tokens", 0) if resources else 0,
                "cache_read_tokens": getattr(resources, "cache_read_tokens", 0) if resources else 0,
                "cache_write_tokens": getattr(resources, "cache_write_tokens", 0) if resources else 0,
                "llm_calls": getattr(resources, "llm_calls", 0) if resources else 0,
                "cost_usd": getattr(resources, "cost_usd", 0.0) if resources else 0.0,
                "started_at": getattr(r, "started_at", None),
            })

        # Aggregate totals
        total_input = sum(d["input_tokens"] for d in token_data)
        total_output = sum(d["output_tokens"] for d in token_data)
        total_cache_read = sum(d["cache_read_tokens"] for d in token_data)
        total_cost = sum(d["cost_usd"] for d in token_data)

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
        }

    except Exception as e:
        logger.warning(f"Failed to get token usage: {e}")
        return {"runs": [], "totals": {}, "error": str(e)}


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    query: str = Query("up", description="PromQL query"),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Proxy a PromQL query to the Prometheus endpoint on the Ray head."""
    return await colony.query_prometheus(query)
