"""Agent system endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from ..dependencies import get_colony
from ..models.api_models import AgentHierarchyNode, AgentSummary
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/agents/", response_model=list[AgentSummary])
async def list_agents(
    colony: ColonyConnection = Depends(get_colony),
):
    """List all registered agents in the Colony agent system."""
    if not colony.is_connected:
        return []

    try:
        handle = colony.get_agent_system()
        agent_ids: list[str] = await handle.list_all_agents()

        summaries = []
        for agent_id in agent_ids:
            info = await handle.get_agent_info(agent_id=agent_id)
            if info is None:
                summaries.append(AgentSummary(agent_id=agent_id))
                continue
            summaries.append(AgentSummary(
                agent_id=agent_id,
                agent_type=getattr(info, "agent_type", ""),
                state=str(getattr(info, "state", "")),
                capabilities=getattr(info, "capabilities", []),
            ))
        return summaries

    except Exception as e:
        logger.warning(f"Failed to list agents: {e}")
        return []


@router.get("/agents/hierarchy", response_model=list[AgentHierarchyNode])
async def get_agent_hierarchy(
    colony: ColonyConnection = Depends(get_colony),
):
    """Get all agents with parent-child relationship info for hierarchy view."""
    if not colony.is_connected:
        return []

    try:
        handle = colony.get_agent_system()
        agent_ids: list[str] = await handle.list_all_agents()

        nodes = []
        for agent_id in agent_ids:
            info = await handle.get_agent_info(agent_id=agent_id)
            if info is None:
                nodes.append(AgentHierarchyNode(agent_id=agent_id))
                continue
            metadata = getattr(info, "metadata", None)
            nodes.append(AgentHierarchyNode(
                agent_id=agent_id,
                agent_type=getattr(info, "agent_type", ""),
                state=str(getattr(info, "state", "")),
                role=getattr(metadata, "role", None) if metadata else None,
                parent_agent_id=getattr(metadata, "parent_agent_id", None) if metadata else None,
                capability_names=getattr(info, "capability_names", []),
                bound_pages=getattr(info, "bound_pages", []),
                tenant_id=getattr(metadata, "tenant_id", "") if metadata else "",
            ))
        return nodes

    except Exception as e:
        logger.warning(f"Failed to get agent hierarchy: {e}")
        return []


@router.get("/agents/{agent_id}")
async def get_agent_detail(
    agent_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed information about a specific agent."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_agent_system()
        info = await handle.get_agent_info(agent_id=agent_id)
        if info is None:
            return {"error": "agent not found", "agent_id": agent_id}
        # Return the full Pydantic model as dict
        if hasattr(info, "model_dump"):
            return info.model_dump()
        return {"agent_id": agent_id, "raw": str(info)}

    except Exception as e:
        return {"error": str(e), "agent_id": agent_id}


@router.get("/agents/{agent_id}/capabilities")
async def get_agent_capabilities(
    agent_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get capabilities of a specific agent."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_agent_system()
        info = await handle.get_agent_info(agent_id=agent_id)
        if info is None:
            return {"error": "agent not found", "agent_id": agent_id}
        return {
            "agent_id": agent_id,
            "capabilities": getattr(info, "capabilities", []),
        }

    except Exception as e:
        return {"error": str(e), "agent_id": agent_id}


@router.get("/agents/stats/system")
async def get_system_stats(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get agent system statistics."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    try:
        handle = colony.get_agent_system()
        return await handle.get_system_stats()
    except Exception as e:
        return {"error": str(e)}
