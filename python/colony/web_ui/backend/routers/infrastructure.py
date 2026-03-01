"""Infrastructure health check endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_colony, get_redis, get_config
from ..models.api_models import HealthStatus, RedisInfo
from ..services.colony_connection import ColonyConnection
from ..services.redis_service import RedisService
from ..config import DashboardConfig

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/infra/status", response_model=HealthStatus)
async def get_status(
    colony: ColonyConnection = Depends(get_colony),
    redis_svc: RedisService = Depends(get_redis),
):
    """Get overall cluster health status."""
    redis_ok = await redis_svc.ping()

    # Ray status
    ray_status = await colony.get_ray_cluster_status()
    ray_nodes = await colony.get_ray_nodes()

    return HealthStatus(
        ray_connected=colony.is_connected,
        redis_connected=redis_ok,
        postgres_connected=False,  # TODO: Add asyncpg health check
        ray_cluster_status=ray_status.get("result", {}).get("status", "unknown") if isinstance(ray_status.get("result"), dict) else str(ray_status.get("status", "unknown")),
        node_count=len(ray_nodes),
    )


@router.get("/infra/redis", response_model=RedisInfo)
async def get_redis_info(
    redis_svc: RedisService = Depends(get_redis),
):
    """Get Redis server information."""
    info = await redis_svc.info()
    return RedisInfo(
        connected_clients=info.get("connected_clients", 0),
        used_memory_human=info.get("used_memory_human", ""),
        total_commands_processed=info.get("total_commands_processed", 0),
        keyspace_hits=info.get("keyspace_hits", 0),
        keyspace_misses=info.get("keyspace_misses", 0),
        uptime_in_seconds=info.get("uptime_in_seconds", 0),
    )
