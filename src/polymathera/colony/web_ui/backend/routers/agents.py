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
    """List all registered agents.

    Tries the live AgentSystem first. If the Ray cluster is down,
    falls back to the spans table in Postgres for persistent history.
    """
    # Try live system first
    if colony.is_connected:
        with colony.kernel_execution_context(origin="dashboard"):
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
                logger.warning(f"Live agent system unavailable: {e}")

    # Fallback: reconstruct from persisted spans
    store = colony.get_span_query_store()
    if store is None:
        return []
    try:
        rows = await store.list_agents_from_spans()
        return [
            AgentSummary(
                agent_id=r["agent_id"],
                agent_type=r.get("agent_type", ""),
                state=r.get("state", "unknown"),
                capabilities=r.get("capabilities", []),
            )
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"Failed to list agents from spans: {e}")
        return []


@router.get("/agents/hierarchy", response_model=list[AgentHierarchyNode])
async def get_agent_hierarchy(
    colony: ColonyConnection = Depends(get_colony),
):
    """Get all agents with parent-child relationship info for hierarchy view.

    Falls back to persisted spans when the live system is unavailable.
    """
    # Try live system first
    if colony.is_connected:
        with colony.kernel_execution_context(origin="dashboard"):
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
                        colony_id=getattr(metadata, "colony_id", "") if metadata else "",
                    ))
                return nodes
            except Exception as e:
                logger.warning(f"Live agent system unavailable: {e}")

    # Fallback: reconstruct from persisted spans
    store = colony.get_span_query_store()
    if store is None:
        return []
    try:
        rows = await store.list_agents_from_spans()
        return [
            AgentHierarchyNode(
                agent_id=r["agent_id"],
                agent_type=r.get("agent_type", ""),
                state=r.get("state", "unknown"),
                parent_agent_id=r.get("parent_agent_id"),
                capability_names=r.get("capabilities", []),
                bound_pages=r.get("bound_pages", []),
            )
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"Failed to get agent hierarchy from spans: {e}")
        return []


@router.get("/agents/{agent_id}")
async def get_agent_detail(
    agent_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed information about a specific agent.

    Falls back to persisted spans when the live system is unavailable.
    """
    # Try live system first
    if colony.is_connected:
        with colony.kernel_execution_context(origin="dashboard"):
            try:
                handle = colony.get_agent_system()
                info = await handle.get_agent_info(agent_id=agent_id)
                if info is not None:
                    if hasattr(info, "model_dump"):
                        return info.model_dump()
                    return {"agent_id": agent_id, "raw": str(info)}
            except Exception as e:
                logger.warning(f"Live agent detail unavailable: {e}")

    # Fallback: reconstruct from persisted spans
    store = colony.get_span_query_store()
    if store is None:
        return {"error": "not available", "agent_id": agent_id}
    try:
        rows = await store.list_agents_from_spans()
        for r in rows:
            if r["agent_id"] == agent_id:
                return r
        return {"error": "agent not found", "agent_id": agent_id}
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

    with colony.kernel_execution_context(
        origin="dashboard",
    ):
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


@router.get("/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get aggregated agent history from the spans table.

    Returns lifecycle events, action success/fail counts, token usage,
    and last error — all persisted via the tracing pipeline.
    """
    store = colony.get_span_query_store()
    if store is None:
        return {"error": "Span query store not available"}
    try:
        return await store.get_agent_history(agent_id)
    except Exception as e:
        logger.warning("Failed to get agent history for %s: %s", agent_id, e)
        return {"error": str(e)}


@router.get("/agents/stats/system")
async def get_system_stats(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get agent system statistics."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    with colony.kernel_execution_context(
        origin="dashboard",
    ):
        try:
            handle = colony.get_agent_system()
            return await handle.get_system_stats()
        except Exception as e:
            return {"error": str(e)}
