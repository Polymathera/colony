"""Colony (workspace) management endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..auth import service as auth_service
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateColonyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class ColonyInfo(BaseModel):
    colony_id: str
    name: str
    tenant_id: str
    description: str = ""
    is_default: bool = False
    created_at: str | None = None


def _get_db_pool(colony: ColonyConnection):
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


@router.get("/colonies/", response_model=list[ColonyInfo])
async def list_colonies(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[ColonyInfo]:
    """List all colonies for the current user."""
    db = _get_db_pool(colony)
    colonies = await auth_service.list_colonies(db, user["tenant_id"])
    return [ColonyInfo(**c) for c in colonies]


@router.post("/colonies/", response_model=ColonyInfo)
async def create_colony(
    request: CreateColonyRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> ColonyInfo:
    """Create a new colony (workspace)."""
    db = _get_db_pool(colony)
    result = await auth_service.create_colony(
        db, user["tenant_id"], request.name, request.description,
    )
    return ColonyInfo(**result)
