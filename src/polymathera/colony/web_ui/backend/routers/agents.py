"""Agent system endpoints.

All endpoints require authentication. The AuthMiddleware sets the
ExecutionContext before endpoint code runs.

For agent queries that need cluster-wide visibility (agents may run on
any replica), we use kernel_execution_context. This is safe because
require_auth already verified the user's identity — the kernel context
is only for the Ray RPC, not for bypassing authorization.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..models.api_models import AgentHierarchyNode, AgentSummary
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SpawnAgentRequest(BaseModel):
    """Request to spawn an agent from a blueprint."""

    agent_type: str = Field(description="Fully qualified agent class path")
    capabilities: list[str] = Field(default_factory=list, description="Capability FQN paths")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Agent metadata parameters")
    bound_pages: list[str] = Field(default_factory=list, description="VCM pages to bind to")
    session_id: str | None = Field(default=None, description="Session to associate with")


class SpawnAgentResponse(BaseModel):
    """Response from agent spawning."""

    agent_id: str = ""
    status: str = "error"  # "spawned" or "error"
    message: str = ""

# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------

@router.get("/agents/", response_model=list[AgentSummary])
async def list_agents(
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """List all registered agents.

    Tries the live AgentSystem first. If the Ray cluster is down,
    falls back to the spans table in Postgres for persistent history.

    NOTE: Agent queries use kernel context for the Ray RPC because
    agents may run on any replica. Authorization is enforced by
    require_auth — the user is already verified.
    """
    # Try live system first
    if colony.is_connected:
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
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """Get all agents with parent-child relationship info for hierarchy view.

    Falls back to persisted spans when the live system is unavailable.
    """
    # Try live system first
    if colony.is_connected:
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
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed information about a specific agent.

    Falls back to persisted spans when the live system is unavailable.
    """
    # Try live system first
    if colony.is_connected:
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
    user: dict = Depends(require_auth),
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


@router.get("/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    user: dict = Depends(require_auth),
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
    user: dict = Depends(require_auth),
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


# ---------------------------------------------------------------------------
# Agent lifecycle — write endpoints
# ---------------------------------------------------------------------------

@router.post("/agents/spawn", response_model=SpawnAgentResponse)
async def spawn_agent(
    request: SpawnAgentRequest,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SpawnAgentResponse:
    """Spawn an agent from a blueprint.

    Creates a new agent instance with the specified type, capabilities,
    and metadata. The agent begins its action loop immediately.
    """
    if not colony.is_connected:
        return SpawnAgentResponse(status="error", message="Not connected to cluster")

    try:
        from polymathera.colony.agents import AgentMetadata, AgentHandle
        from polymathera.colony.cli.polymath import _resolve_class

        # Build capability blueprints
        cap_blueprints = []
        for cap_path in request.capabilities:
            cap_cls = _resolve_class(cap_path)
            cap_blueprints.append(cap_cls.bind())

        # Build metadata
        metadata = AgentMetadata(
            session_id=request.session_id,
            parameters=request.metadata,
        )

        # Build and spawn
        agent_cls = _resolve_class(request.agent_type)
        bp = agent_cls.bind(
            agent_type=request.agent_type,
            metadata=metadata,
            bound_pages=request.bound_pages,
            capability_blueprints=cap_blueprints,
        )

        handle = await AgentHandle.from_blueprint(
            agent_blueprint=bp,
            app_name=colony.app_name,
        )

        return SpawnAgentResponse(
            agent_id=handle.agent_id,
            status="spawned",
            message=f"Agent {handle.agent_id} spawned ({request.agent_type.rsplit('.', 1)[-1]})",
        )

    except Exception as e:
        logger.error("Failed to spawn agent: %s", e)
        return SpawnAgentResponse(status="error", message=str(e))


class InterruptAgentRequest(BaseModel):
    """Request to interrupt a running agent."""

    action: str = Field(default="cancel", description="cancel or suspend")
    reason: str = Field(default="user requested via dashboard", description="Reason for interruption")


class InterruptAgentResponse(BaseModel):
    """Response from agent interruption."""

    agent_id: str
    action: str
    success: bool
    message: str = ""


@router.post("/agents/{agent_id}/interrupt", response_model=InterruptAgentResponse)
async def interrupt_agent(
    agent_id: str,
    request: InterruptAgentRequest | None = None,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> InterruptAgentResponse:
    """Interrupt a running agent.

    Actions:
    - cancel: Stop the agent immediately (default)
    - suspend: Suspend the agent (saves state, future)
    """
    if not colony.is_connected:
        return InterruptAgentResponse(
            agent_id=agent_id, action="cancel", success=False, message="Not connected",
        )

    request = request or InterruptAgentRequest()

    try:
        if request.action == "cancel":
            handle = colony.get_agent_system()
            await handle.stop_agent(agent_id=agent_id, reason=request.reason)
            return InterruptAgentResponse(
                agent_id=agent_id, action="cancel", success=True,
                message=f"Agent {agent_id} stop requested",
            )
        else:
            return InterruptAgentResponse(
                agent_id=agent_id, action=request.action, success=False,
                message=f"Action '{request.action}' not yet supported. Use 'cancel'.",
            )
    except Exception as e:
        logger.error("Failed to interrupt agent %s: %s", agent_id, e)
        return InterruptAgentResponse(
            agent_id=agent_id, action=request.action, success=False, message=str(e),
        )
