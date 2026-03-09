"""Serving deployment status endpoints.

Reads application registry via Colony deployment handle.
No direct Redis access.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from ..dependencies import get_colony
from ..models.api_models import ApplicationSummary, DeploymentSummary
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/deployments/", response_model=list[ApplicationSummary])
async def list_deployments(
    colony: ColonyConnection = Depends(get_colony),
):
    """List all serving applications and their deployments."""
    if not colony.is_connected:
        return []

    try:
        handle = colony.get_agent_system()
        infra = await handle.get_infrastructure_status()

        apps = []
        for app_data in infra.get("applications", []):
            deployments = [
                DeploymentSummary(
                    app_name=app_data["app_name"],
                    deployment_name=d["deployment_name"],
                    proxy_actor_name=d.get("proxy_actor_name", ""),
                )
                for d in app_data.get("deployments", [])
            ]
            apps.append(ApplicationSummary(
                app_name=app_data["app_name"],
                created_at=app_data.get("created_at", 0),
                deployments=deployments,
            ))
        return apps

    except Exception as e:
        logger.warning("Failed to list deployments: %s", e)
        return []


@router.get("/deployments/{app_name}/{deployment_name}/health")
async def get_deployment_health(
    app_name: str,
    deployment_name: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get health status of a specific deployment via its proxy's get_stats()."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    try:
        handle = colony.get_deployment_handle(app_name, deployment_name)
        stats = await handle.get_stats()
        healthy = stats.get("healthy_replicas", 0)
        total = stats.get("total_replicas", 0)
        return {
            "status": "healthy" if healthy == total and total > 0 else "degraded",
            "app_name": app_name,
            "deployment_name": deployment_name,
            "total_replicas": total,
            "healthy_replicas": healthy,
            "total_queue_length": stats.get("total_queue_length", 0),
            "total_in_flight": stats.get("total_in_flight", 0),
            "autoscaling_config": stats.get("autoscaling_config", {}),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
