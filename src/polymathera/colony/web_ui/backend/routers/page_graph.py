"""Page graph visualization endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/vcm/page-graph/scopes")
async def list_all_mapped_memory_scopes(
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """List available mapped memory scopes (tenant_id, colony_id, scope_id)."""
    if not colony.is_connected:
        return []

    with colony.kernel_execution_context(
        origin="dashboard",
    ):
        try:
            raw = await colony.get_vcm().get_all_mapped_scopes()
            # Flatten: backend returns {syscontext: {tenant_id, colony_id, ...}, scope_id}
            # Frontend expects {tenant_id, colony_id, scope_id}
            return [
                {
                    "tenant_id": item.get("syscontext", {}).get("tenant_id", ""),
                    "colony_id": item.get("syscontext", {}).get("colony_id", ""),
                    "scope_id": item.get("scope_id", ""),
                }
                for item in raw
            ]
        except Exception as e:
            logger.warning("Failed to list page graph groups: %s", e)
            return []


@router.get("/vcm/page-graph")
async def get_page_graph(
    tenant_id: str = Query(...),
    colony_id: str = Query(...),
    max_nodes: int = Query(default=5000, le=10000),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get page graph data (nodes + edges with 3D positions) for visualization."""
    if not colony.is_connected:
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

    with colony.user_execution_context(
        colony_id=colony_id,
        tenant_id=tenant_id,
        origin="dashboard",
    ):
        try:
            return await colony.get_vcm().get_page_graph_data(
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.warning("Failed to get page graph: %s", e)
            return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0, "error": str(e)}
