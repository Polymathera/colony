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


# ---------------------------------------------------------------------------
# Per-colony design-monorepo URL — single source of truth for the
# SessionAgent's per-agent clones, the materialiser's ``Map Repo``
# trigger, and the dashboard's "Design Monorepo" tab. Auth (PAT,
# credential helper, ssh-agent, …) is operator-managed via the
# colony's git environment; we surface clone errors verbatim.
# ---------------------------------------------------------------------------


class DesignMonorepoConfig(BaseModel):
    origin_url: str | None = Field(
        default=None,
        description="Git URL. ``None`` when the colony has no design monorepo configured.",
    )
    branch: str = "main"
    commit: str = "HEAD"


class SetDesignMonorepoRequest(BaseModel):
    origin_url: str = Field(min_length=1)
    branch: str = "main"
    commit: str = "HEAD"


@router.get(
    "/colonies/{colony_id}/design-monorepo",
    response_model=DesignMonorepoConfig,
)
async def get_colony_design_monorepo(
    colony_id: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> DesignMonorepoConfig:
    """Return the colony's design-monorepo URL (if configured)."""
    db = _get_db_pool(colony)
    row = await auth_service.get_design_monorepo(
        db, colony_id=colony_id, tenant_id=user["tenant_id"],
    )
    if row is None:
        return DesignMonorepoConfig()
    return DesignMonorepoConfig(**row)


@router.put(
    "/colonies/{colony_id}/design-monorepo",
    response_model=DesignMonorepoConfig,
)
async def set_colony_design_monorepo(
    colony_id: str,
    request: SetDesignMonorepoRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> DesignMonorepoConfig:
    """Persist the colony's design-monorepo URL.

    The dashboard's "Design Monorepo" tab calls this. The SessionAgent
    can call it from chat via the ``set_design_monorepo`` action on
    :class:`DesignMonorepoBootstrap` — both end up here.
    """

    db = _get_db_pool(colony)
    try:
        row = await auth_service.set_design_monorepo(
            db,
            colony_id=colony_id,
            tenant_id=user["tenant_id"],
            origin_url=request.origin_url,
            branch=request.branch,
            commit=request.commit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DesignMonorepoConfig(**row)
