"""Log streaming endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/logs/actors")
async def list_log_actors(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """List available log files/actors from the Ray dashboard."""
    if not colony.is_connected or not colony._http_client:
        return {"actors": []}

    try:
        resp = await colony._http_client.get(
            f"{colony.ray_dashboard_url}/api/v0/logs"
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"actors": [], "error": str(e)}


@router.get("/logs/file")
async def get_log_file(
    node_id: str = Query(..., description="Ray node ID"),
    filename: str = Query(..., description="Log filename"),
    lines: int = Query(200, le=2000),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get log file content from a Ray node."""
    if not colony.is_connected or not colony._http_client:
        return {"lines": [], "error": "not connected"}

    try:
        resp = await colony._http_client.get(
            f"{colony.ray_dashboard_url}/api/v0/logs/file",
            params={"node_id": node_id, "filename": filename, "lines": lines},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"lines": [], "error": str(e)}
