"""Log streaming endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/logs/sources")
async def list_log_sources(
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """List available log sources (live Ray actors) with metadata.

    Uses the Ray Dashboard HTTP API (/logical/actors) instead of
    ray.util.state, which doesn't work reliably through Ray Client.
    """
    if not colony._http_client:
        return []

    try:
        # Fetch actors from Ray Dashboard
        resp = await colony._http_client.get(
            f"{colony.ray_dashboard_url}/logical/actors", timeout=10,
        )
        resp.raise_for_status()
        actors = resp.json().get("data", {}).get("actors", {})

        # Build node_id → IP mapping for display
        node_ips: dict[str, str] = {}
        try:
            nodes_resp = await colony._http_client.get(
                f"{colony.ray_dashboard_url}/nodes?view=summary", timeout=10,
            )
            nodes_resp.raise_for_status()
            for n in nodes_resp.json().get("data", {}).get("summary", []):
                nid = n.get("raylet", {}).get("nodeId", "")
                node_ips[nid] = n.get("ip", "")
        except Exception:
            pass

        sources = []
        for a in actors.values():
            if a.get("state") != "ALIVE":
                continue
            node_id = a.get("address", {}).get("nodeId", "") if a.get("address") else ""
            sources.append({
                "actor_id": a["actorId"],
                "class_name": a.get("className", ""),
                "node_id": node_id,
                "pid": a.get("pid", 0),
                "ip": node_ips.get(node_id, ""),
                "repr_name": a.get("reprName") or a.get("className", ""),
            })
        # Sort by class_name then pid for stable ordering
        sources.sort(key=lambda s: (s["class_name"], s["pid"]))
        return sources
    except Exception as e:
        logger.warning("Failed to list log sources: %s", e)
        return []


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
    node_id: str = Query(None, description="Ray node ID"),
    filename: str = Query(None, description="Log filename"),
    actor_id: str = Query(None, description="Ray actor ID"),
    lines: int = Query(200, le=2000),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get log file content from a Ray node."""
    if not colony.is_connected or not colony._http_client:
        return {"lines": [], "error": "not connected"}

    try:
        params: dict[str, Any] = {"lines": lines}
        if actor_id:
            params["actor_id"] = actor_id
        elif node_id and filename:
            params["node_id"] = node_id
            params["filename"] = filename
        else:
            return {"lines": [], "error": "provide actor_id or (node_id + filename)"}

        resp = await colony._http_client.get(
            f"{colony.ray_dashboard_url}/api/v0/logs/file",
            params=params,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"lines": [], "error": str(e)}
