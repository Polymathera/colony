"""FastAPI dependency injection for the dashboard backend.

Provides shared service instances to routers via FastAPI's Depends() mechanism.
All Colony data access goes through deployment handles — no direct backend access.
"""

from __future__ import annotations

from fastapi import Request

from .config import DashboardConfig
from .services.colony_connection import ColonyConnection


def get_config(request: Request) -> DashboardConfig:
    """Get dashboard configuration from app state."""
    return request.app.state.config


def get_colony(request: Request) -> ColonyConnection:
    """Get the Colony connection from app state."""
    return request.app.state.colony
