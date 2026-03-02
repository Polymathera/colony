"""Infrastructure health check endpoints.

All data flows through Colony deployment handles or the Ray Dashboard HTTP API.
No direct backend access.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_colony
from ..models.api_models import HealthStatus, RedisInfo
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_APP_NAME = "polymathera"


@router.get("/infra/status", response_model=HealthStatus)
async def get_status(
    colony: ColonyConnection = Depends(get_colony),
):
    """Get overall cluster health status."""
    # Ray status via Ray Dashboard HTTP API (a Ray service, not a Colony backend)
    ray_status = await colony.get_ray_cluster_status()
    ray_nodes = await colony.get_ray_nodes()

    redis_connected = False
    if colony.is_connected:
        try:
            handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "agent_system")
            infra = await handle.get_infrastructure_status()
            redis_connected = infra.get("redis_connected", False)
        except Exception as e:
            logger.warning("Failed to get infra status from deployment: %s", e)

    return HealthStatus(
        ray_connected=colony.is_connected,
        redis_connected=redis_connected,
        ray_cluster_status=ray_status.get("result", {}).get("status", "unknown") if isinstance(ray_status.get("result"), dict) else str(ray_status.get("status", "unknown")),
        node_count=len(ray_nodes),
    )


@router.get("/infra/redis", response_model=RedisInfo)
async def get_redis_info(
    colony: ColonyConnection = Depends(get_colony),
):
    """Get Redis server information via Colony deployment handle."""
    if not colony.is_connected:
        return RedisInfo()

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "agent_system")
        infra = await handle.get_infrastructure_status()
        info = infra.get("redis_info", {})
        return RedisInfo(
            connected_clients=info.get("connected_clients", 0),
            used_memory_human=info.get("used_memory_human", ""),
            total_commands_processed=info.get("total_commands_processed", 0),
            keyspace_hits=info.get("keyspace_hits", 0),
            keyspace_misses=info.get("keyspace_misses", 0),
            uptime_in_seconds=info.get("uptime_in_seconds", 0),
        )
    except Exception as e:
        logger.warning("Failed to get Redis info: %s", e)
        return RedisInfo()
