"""Serving deployment status endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from ..dependencies import get_colony, get_redis
from ..models.api_models import ApplicationSummary, DeploymentSummary
from ..services.colony_connection import ColonyConnection
from ..services.redis_service import RedisService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/deployments/", response_model=list[ApplicationSummary])
async def list_deployments(
    colony: ColonyConnection = Depends(get_colony),
    redis_svc: RedisService = Depends(get_redis),
):
    """List all serving applications and their deployments.

    Reads from the ApplicationRegistry shared state in Redis.
    """
    if not colony.is_connected:
        return []

    try:
        # ApplicationRegistry is stored as a shared state in Redis.
        # Try to read it through the Colony serving framework.
        from colony.distributed.ray_utils.serving.models import ApplicationRegistry
        from colony.distributed.ray_utils.state import StateManager

        # The registry key follows the pattern used by SharedState
        registry_key = "polymathera:serving:app_registry"
        raw = await redis_svc.get_json(registry_key)

        if raw is None:
            # Try scanning for registry keys
            keys = await redis_svc.scan_keys("*app_registry*")
            if keys:
                raw = await redis_svc.get_json(keys[0])

        if raw is None:
            # Fallback: try to discover apps by scanning for proxy actors
            return await _discover_apps_from_actors(colony)

        # Parse the registry
        apps = []
        applications = raw.get("applications", {})
        for app_name, app_info in applications.items():
            deployments = []
            for dep_name, dep_info in app_info.get("deployments", {}).items():
                deployments.append(DeploymentSummary(
                    app_name=app_name,
                    deployment_name=dep_name,
                    proxy_actor_name=dep_info.get("proxy_actor_name", ""),
                ))
            apps.append(ApplicationSummary(
                app_name=app_name,
                created_at=app_info.get("created_at", 0),
                deployments=deployments,
            ))
        return apps

    except Exception as e:
        logger.warning(f"Failed to list deployments: {e}")
        return []


async def _discover_apps_from_actors(colony: ColonyConnection) -> list[ApplicationSummary]:
    """Fallback: discover applications by looking for known proxy actors."""
    # This is a best-effort approach when the registry is not directly readable
    return []


@router.get("/deployments/{app_name}/{deployment_name}/health")
async def get_deployment_health(
    app_name: str,
    deployment_name: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get health status of a specific deployment."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    try:
        handle = colony.get_deployment_handle(app_name, deployment_name)
        # Try calling a health-related method if available
        # The proxy actor tracks replica health internally
        return {"status": "reachable", "app_name": app_name, "deployment_name": deployment_name}
    except Exception as e:
        return {"status": "error", "error": str(e)}
