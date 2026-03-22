"""Blackboard scope and entry endpoints.

All data flows through AgentSystemDeployment endpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/blackboard/scopes")
async def list_blackboard_scopes(
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Discover active blackboard scopes and their statistics."""
    if not colony.is_connected:
        return []

    with colony.kernel_execution_context(
        origin="dashboard",
    ):
        try:
            handle = colony.get_agent_system()
            return await handle.get_blackboard_scopes()
        except Exception as e:
            logger.warning("Failed to list blackboard scopes: %s", e)
            return []


@router.get("/blackboard/scopes/{scope}/{scope_id}/entries")
async def get_blackboard_entries(
    scope: str,
    scope_id: str,
    limit: int = Query(default=100, le=1000),
    backend_type: str = Query(default=""),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """List entries in a specific blackboard scope."""
    if not colony.is_connected:
        return []

    with colony.kernel_execution_context(
        origin="dashboard",
    ):
        try:
            handle = colony.get_agent_system()
            return await handle.get_blackboard_entries(
                scope=scope, scope_id=scope_id, limit=limit,
                backend_type=backend_type,
            )
        except Exception as e:
            logger.warning("Failed to get blackboard entries: %s", e)
            return []
