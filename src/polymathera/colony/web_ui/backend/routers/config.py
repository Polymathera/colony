"""Colony configuration endpoints.

Provides read access to cluster configuration and read/write access
to tenant quotas. Full dynamic configuration (Redis-backed config
manager with change notifications) is planned for a future phase.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TenantQuotaRequest(BaseModel):
    """Tenant resource quota configuration."""

    max_concurrent_sessions: int = Field(default=10, ge=1)
    max_concurrent_agents: int = Field(default=100, ge=1)
    max_total_cpu_cores: float = Field(default=10.0, gt=0)
    max_total_memory_mb: int = Field(default=51200, gt=0)
    max_total_gpu_cores: float = Field(default=2.0, ge=0)
    max_total_gpu_memory_mb: int = Field(default=16384, ge=0)


class TenantQuotaResponse(BaseModel):
    """Response with quota and current usage."""

    tenant_id: str
    quota: dict[str, Any]
    usage: dict[str, Any]


# ---------------------------------------------------------------------------
# Colony-level config (read-only)
# ---------------------------------------------------------------------------

@router.get("/config/colony")
async def get_colony_config(
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get current colony cluster configuration.

    Returns deployment info, VCM config, and agent system config
    as read from the running cluster. This is read-only — changing
    cluster topology requires a restart.
    """
    if not colony.is_connected:
        return {"status": "disconnected"}

    result: dict[str, Any] = {"status": "connected"}

    # Cluster deployments
    try:
        from polymathera.colony.distributed.ray_utils import serving
        apps = serving.list_applications()
        result["applications"] = [
            {
                "app_name": app.app_name,
                "deployments": [
                    {"name": d.deployment_name, "proxy": d.proxy_actor_name}
                    for d in app.deployments
                ],
            }
            for app in apps
        ]
    except Exception as e:
        result["applications_error"] = str(e)

    # VCM stats
    try:
        vcm_stats = await colony.get_vcm().get_stats()
        result["vcm"] = vcm_stats
    except Exception as e:
        result["vcm_error"] = str(e)

    # Session stats
    try:
        session_stats = await colony.get_session_manager().get_stats()
        result["sessions"] = session_stats
    except Exception as e:
        result["sessions_error"] = str(e)

    # Agent system stats
    try:
        agent_stats = await colony.get_agent_system().get_system_stats()
        result["agents"] = agent_stats
    except Exception as e:
        result["agents_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Tenant quota management
# ---------------------------------------------------------------------------

@router.get("/config/tenants/{tenant_id}/quota", response_model=TenantQuotaResponse)
async def get_tenant_quota(
    tenant_id: str,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> TenantQuotaResponse:
    """Get tenant resource quota and current usage."""
    if not colony.is_connected:
        return TenantQuotaResponse(tenant_id=tenant_id, quota={}, usage={})

    try:
        sm = colony.get_session_manager()
        quota = await sm.get_tenant_quota(tenant_id=tenant_id)
        usage = await sm.get_tenant_resource_usage(tenant_id=tenant_id)

        quota_dict = quota.model_dump() if hasattr(quota, "model_dump") else (quota if isinstance(quota, dict) else {})
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else {})

        return TenantQuotaResponse(
            tenant_id=tenant_id,
            quota=quota_dict,
            usage=usage_dict,
        )
    except Exception as e:
        logger.error("Failed to get tenant quota: %s", e)
        return TenantQuotaResponse(tenant_id=tenant_id, quota={}, usage={"error": str(e)})


@router.put("/config/tenants/{tenant_id}/quota")
async def set_tenant_quota(
    tenant_id: str,
    quota: TenantQuotaRequest,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Update tenant resource quotas.

    Changes take effect immediately for new session/agent creation.
    Existing sessions are not affected until they try to scale.
    """
    if not colony.is_connected:
        return {"success": False, "message": "Not connected"}

    try:
        from polymathera.colony.agents.sessions.models import TenantQuota

        sm = colony.get_session_manager()
        tenant_quota = TenantQuota(**quota.model_dump())
        await sm.set_tenant_quota(tenant_id=tenant_id, quota=tenant_quota)

        return {
            "success": True,
            "tenant_id": tenant_id,
            "message": "Quota updated",
        }
    except Exception as e:
        logger.error("Failed to set tenant quota: %s", e)
        return {"success": False, "message": str(e)}


@router.get("/config/tenants/")
async def list_tenant_quotas(
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[TenantQuotaResponse]:
    """List all tenants with their quotas and usage.

    Reads from the session system state to find all known tenants.
    """
    if not colony.is_connected:
        return []

    try:
        sm = colony.get_session_manager()
        stats = await sm.get_stats()

        if not isinstance(stats, dict):
            return []

        tenant_ids = stats.get("tenant_ids", [])
        results = []

        for tid in tenant_ids:
            try:
                quota = await sm.get_tenant_quota(tenant_id=tid)
                usage = await sm.get_tenant_resource_usage(tenant_id=tid)
                quota_dict = quota.model_dump() if hasattr(quota, "model_dump") else (quota if isinstance(quota, dict) else {})
                usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else {})
                results.append(TenantQuotaResponse(tenant_id=tid, quota=quota_dict, usage=usage_dict))
            except Exception:
                results.append(TenantQuotaResponse(tenant_id=tid, quota={}, usage={}))

        return results
    except Exception as e:
        logger.warning("Failed to list tenants: %s", e)
        return []
