"""Tenant management endpoints.

Currently exposes the per-tenant GitHub App installation id. ``users``
table still has a 1:1 with ``tenants`` in v1, so "me" resolves to the
authenticated user's own tenant. v2 (multi-user-per-tenant) will gate
``PUT`` behind a tenant-admin check; for now any tenant member can set
their tenant's installation id.

See ``colony/github_identity_fix_plan.md`` §2 (per-tenant credential
layer) for the wider picture.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import service as auth_service
from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


class DiscoverableRepo(BaseModel):
    vcs_repo_id: str
    vcs_repo_full_name: str
    default_branch: str
    user_permission: str
    has_colony_marker: bool
    # Pre-rendered clone URL via the tenant's VCS provider. Lets the
    # UI populate ``colonies.design_monorepo_url`` without knowing
    # the provider's URL template. ``None`` when the tenant's
    # provider isn't currently registered (operator missing OAuth
    # creds) — the UI falls back to ``vcs_repo_full_name`` display.
    clone_url: str | None = None


class TenantGitHubInstallation(BaseModel):
    installation_id: str | None = Field(
        default=None,
        description=(
            "The GitHub App installation id Colony uses to mint REST "
            "tokens scoped to this tenant's repos. ``null`` until a "
            "tenant admin pastes it in from the GitHub installation "
            "URL (``https://github.com/settings/installations/<id>``)."
        ),
    )


class SetTenantGitHubInstallationRequest(BaseModel):
    installation_id: str | None = None


def _get_db_pool(colony: ColonyConnection):
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


@router.get(
    "/tenants/me/discoverable-repos",
    response_model=list[DiscoverableRepo],
)
async def list_discoverable_repos(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[DiscoverableRepo]:
    """Return every repo the sign-in walker cached for the caller's
    tenant, with ``clone_url`` pre-rendered by the tenant's VCS
    provider. Powers the "+ New colony" form's repo dropdown + the
    per-colony "Design monorepo" picker. ``has_colony_marker`` lets
    the UI badge repos that already opt in to Colony."""
    db = _get_db_pool(colony)
    rows = await auth_service.list_tenant_repos_for_tenant(
        db, tenant_id=user["tenant_id"],
    )

    # Resolve the provider once + render clone URLs in-process.
    # Provider not registered ⇒ clone_url stays None on every row;
    # UI falls back to the bare full_name.
    from polymathera.colony.vcs import get_provider
    from polymathera.colony.vcs.provider import VcsRepoRef

    async with db.acquire() as conn:
        tenant_row = await conn.fetchrow(
            "SELECT vcs_provider FROM tenants WHERE id = $1",
            user["tenant_id"],
        )
    provider = None
    provider_id = (tenant_row or {}).get("vcs_provider") or "github"
    try:
        provider = get_provider(provider_id)
    except KeyError:
        provider = None

    out: list[DiscoverableRepo] = []
    for r in rows:
        clone_url: str | None = None
        if provider is not None:
            try:
                clone_url = provider.repo_clone_url(VcsRepoRef(
                    vcs_repo_id=r["vcs_repo_id"],
                    full_name=r["vcs_repo_full_name"],
                    default_branch=r["default_branch"],
                    user_permission=r["user_permission"],
                ))
            except Exception:  # noqa: BLE001 — defensive
                clone_url = None
        out.append(DiscoverableRepo(**r, clone_url=clone_url))
    return out


@router.get(
    "/tenants/me/github-installation",
    response_model=TenantGitHubInstallation,
)
async def get_tenant_github_installation(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> TenantGitHubInstallation:
    """Return the installation id for the caller's tenant. ``None`` when
    the tenant hasn't installed the Colony GitHub App yet."""

    db = _get_db_pool(colony)
    row = await auth_service.get_tenant_github_installation(
        db, tenant_id=user["tenant_id"],
    )
    if row is None:
        # Should never happen — the tenant row is created at signup
        # (P1 fixup in auth_service.create_user). Treat as a 404 so
        # the UI can surface a clear error.
        raise HTTPException(
            status_code=404,
            detail=f"tenant {user['tenant_id']!r} not found",
        )
    return TenantGitHubInstallation(**row)


@router.put(
    "/tenants/me/github-installation",
    response_model=TenantGitHubInstallation,
)
async def set_tenant_github_installation(
    request: SetTenantGitHubInstallationRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> TenantGitHubInstallation:
    """Set (or clear) the installation id for the caller's tenant.

    The installation id is the numeric value visible on the App
    installation URL on GitHub. v1 lets any tenant member set it;
    v2's tenant-admin gate is a separate change.
    """

    db = _get_db_pool(colony)
    # Light input shape check: strip whitespace, treat empty as None.
    cleaned = (request.installation_id or "").strip() or None
    try:
        row = await auth_service.set_tenant_github_installation(
            db, tenant_id=user["tenant_id"], installation_id=cleaned,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info(
        "tenants: %s set github_installation_id=%s",
        user["tenant_id"], cleaned,
    )
    return TenantGitHubInstallation(**row)
