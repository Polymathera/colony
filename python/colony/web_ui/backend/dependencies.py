"""FastAPI dependency injection for the dashboard backend.

Provides shared service instances (Colony connection, Redis, DB) to routers
via FastAPI's Depends() mechanism. Services are initialized at app startup
and torn down at shutdown via the lifespan context.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Request

from .config import DashboardConfig
from .services.colony_connection import ColonyConnection
from .services.redis_service import RedisService

logger = logging.getLogger(__name__)


def get_config(request: Request) -> DashboardConfig:
    """Get dashboard configuration from app state."""
    return request.app.state.config


def get_colony(request: Request) -> ColonyConnection:
    """Get the Colony connection from app state."""
    return request.app.state.colony


def get_redis(request: Request) -> RedisService:
    """Get the Redis service from app state."""
    return request.app.state.redis_service
