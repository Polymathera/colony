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


# ---------------------------------------------------------------------------
# Per-colony source selection — the operator's checkbox state from the
# "Design Monorepo" tab. Two independent sets, both read by the
# materialiser at action time so chat-driven ingestion (which has no
# checkboxes) honours the same selection.
# ---------------------------------------------------------------------------


class SourceSelection(BaseModel):
    """List of row names the operator has ticked. ``None`` (the
    default until the operator changes anything) means "all rows
    enabled" — same convention as
    ``materialize_*_sources(enabled_sources=...)``."""

    enabled: list[str] | None = None


@router.get(
    "/colonies/{colony_id}/enabled-vcm-sources",
    response_model=SourceSelection,
)
async def get_colony_enabled_vcm_sources(
    colony_id: str,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        list_enabled_vcm_sources,
    )
    return SourceSelection(enabled=await list_enabled_vcm_sources(colony_id))


@router.put(
    "/colonies/{colony_id}/enabled-vcm-sources",
    response_model=SourceSelection,
)
async def set_colony_enabled_vcm_sources(
    colony_id: str,
    request: SourceSelection,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        set_enabled_vcm_sources,
    )
    await set_enabled_vcm_sources(colony_id, request.enabled)
    return request


@router.get(
    "/colonies/{colony_id}/enabled-knowledge-sources",
    response_model=SourceSelection,
)
async def get_colony_enabled_knowledge_sources(
    colony_id: str,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        list_enabled_knowledge_sources,
    )
    return SourceSelection(
        enabled=await list_enabled_knowledge_sources(colony_id),
    )


@router.put(
    "/colonies/{colony_id}/enabled-knowledge-sources",
    response_model=SourceSelection,
)
async def set_colony_enabled_knowledge_sources(
    colony_id: str,
    request: SourceSelection,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        set_enabled_knowledge_sources,
    )
    await set_enabled_knowledge_sources(colony_id, request.enabled)
    return request


# ---------------------------------------------------------------------------
# Per-colony git-commit attribution (principal + optional co-author).
# Default: principal=colony, co_author=user — the persistent
# collective identity does the work on behalf of the human who
# started the session. Operator can swap to per-agent or any
# free-form agent-type label via the landing-page UI.
# ---------------------------------------------------------------------------


class GitAttributionConfig(BaseModel):
    git_user_name: str | None = Field(
        default=None,
        description=(
            "Display name used when ``commit_principal`` or "
            "``commit_co_author`` is ``\"user\"``. Required in those "
            "cases; ignored otherwise."
        ),
    )
    git_user_email: str | None = Field(
        default=None,
        description="Email paired with ``git_user_name``.",
    )
    commit_principal: str = Field(
        default="colony",
        description=(
            "Free-form identity string. Well-known: ``user`` / "
            "``colony`` / ``agent``. Anything else is treated as an "
            "agent-type label (e.g. ``session_agent``)."
        ),
    )
    commit_co_author: str | None = Field(
        default="user",
        description=(
            "Optional second identity, rendered as a "
            "``Co-Authored-By:`` trailer. Same value space as "
            "``commit_principal``. ``null`` disables the trailer."
        ),
    )


class SetGitAttributionRequest(BaseModel):
    git_user_name: str | None = None
    git_user_email: str | None = None
    commit_principal: str = "colony"
    commit_co_author: str | None = "user"


@router.get(
    "/colonies/{colony_id}/git-attribution",
    response_model=GitAttributionConfig,
)
async def get_colony_git_attribution(
    colony_id: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> GitAttributionConfig:
    """Return the colony's per-commit attribution settings."""
    db = _get_db_pool(colony)
    row = await auth_service.get_git_attribution(
        db, colony_id=colony_id, tenant_id=user["tenant_id"],
    )
    if row is None:
        return GitAttributionConfig()
    return GitAttributionConfig(**row)


@router.put(
    "/colonies/{colony_id}/git-attribution",
    response_model=GitAttributionConfig,
)
async def set_colony_git_attribution(
    colony_id: str,
    request: SetGitAttributionRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> GitAttributionConfig:
    """Persist the colony's per-commit attribution settings."""

    db = _get_db_pool(colony)
    try:
        row = await auth_service.set_git_attribution(
            db,
            colony_id=colony_id,
            tenant_id=user["tenant_id"],
            git_user_name=request.git_user_name,
            git_user_email=request.git_user_email,
            commit_principal=request.commit_principal,
            commit_co_author=request.commit_co_author,
        )
    except ValueError as exc:
        # Operator picked a 'user' principal/co_author without
        # configuring name/email — surface as 400 so the UI can show
        # an inline validation message instead of a generic 5xx.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GitAttributionConfig(**row)
