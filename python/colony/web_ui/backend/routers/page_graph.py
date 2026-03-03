"""Page graph visualization endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/vcm/page-graph/groups")
async def list_page_graph_groups(
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """List available page graph groups (tenant_id, group_id, scope_id)."""
    if not colony.is_connected:
        return []
    try:
        return await colony.get_vcm().get_page_graph_groups()
    except Exception as e:
        logger.warning("Failed to list page graph groups: %s", e)
        return []


@router.get("/vcm/page-graph")
async def get_page_graph(
    tenant_id: str = Query(...),
    group_id: str = Query(...),
    max_nodes: int = Query(default=5000, le=10000),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get page graph data (nodes + edges with 3D positions) for visualization."""
    if not colony.is_connected:
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}
    try:
        return await colony.get_vcm().get_page_graph_data(
            tenant_id=tenant_id,
            group_id=group_id,
            max_nodes=max_nodes,
        )
    except Exception as e:
        logger.warning("Failed to get page graph: %s", e)
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0, "error": str(e)}
