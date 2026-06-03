"""Colony auto-discovery walker.

For each tenant a user belongs to, list the tenant's repos, probe
each one for a ``.colony/`` directory, and provision a colony for
every match the tenant doesn't already have.

Discipline (per plan §3.4 + §4.3):

- **Many-to-one colony↔repo is allowed.** No SQL UNIQUE on
  ``(tenant_id, vcs_repo_id)``. The walker uses
  ``any_colony_exists_for_repo`` as an *application-level* "first
  auto-discovery only" gate — once a tenant has any colony for a
  repo, subsequent walker passes skip that repo. The user can still
  create additional colonies on the same repo explicitly via the UI.
- **Best-effort per repo.** A single repo's probe failing logs a
  warning and continues with the rest — discovery shouldn't
  abandon the whole sign-in over one flaky repo.
- **Goes through :func:`provision_colony`.** This is the single
  colony-create entry point — colony-discovery doesn't bypass it,
  so every auto-discovered colony gets its system-session bootstrap
  for free.
"""

from __future__ import annotations

import logging
from typing import Callable

from polymathera.colony.vcs.provider import (
    OAuthExchangeError,
    VcsProvider,
    VcsTenantRef,
)

from ..auth import service as auth_service
from .colony_connection import ColonyConnection
from .colony_lifecycle import provision_colony

logger = logging.getLogger(__name__)


# Repo-root path that marks a repo as a Colony project. Operators
# create ``.colony/`` in their repo to opt in to auto-discovery.
_COLONY_MARKER_PATH = ".colony"


ProgressCallback = Callable[[str], None]


async def discover_colonies_for_tenant(
    colony: ColonyConnection,
    *,
    tenant_id: str,
    tenant_ref: VcsTenantRef,
    provider: VcsProvider,
    access_token: str,
    on_progress: ProgressCallback | None = None,
) -> list[str]:
    """Walk the tenant's repos, auto-create a colony for each one
    with ``.colony/`` that the tenant doesn't already have one for.

    Returns the list of newly-provisioned ``colony_id`` values (empty
    when nothing new was discovered). Idempotent — subsequent calls
    for the same tenant skip repos that already have a colony.

    Failures are logged + swallowed at the per-repo granularity so a
    single bad repo doesn't block the rest of the discovery for this
    tenant. Failures of the top-level ``list_tenant_repos`` call DO
    propagate ``OAuthExchangeError`` so the sign-in handler can surface
    a clean 4xx to the user (revoked token, scope missing).
    """

    def _emit(msg: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(msg)
        except Exception:  # noqa: BLE001
            logger.warning(
                "discover_colonies_for_tenant: progress callback raised",
                exc_info=True,
            )

    db_pool = colony._db_pool
    if db_pool is None:
        logger.warning(
            "discover_colonies_for_tenant: db_pool unavailable; "
            "skipping discovery for tenant=%s",
            tenant_id,
        )
        return []

    try:
        repos = await provider.list_tenant_repos(
            tenant=tenant_ref, access_token=access_token,
        )
    except OAuthExchangeError:
        # Let the sign-in handler turn this into a 4xx.
        raise
    except Exception:  # noqa: BLE001
        logger.exception(
            "discover_colonies_for_tenant: list_tenant_repos failed "
            "for tenant=%s vcs_org=%s; skipping discovery",
            tenant_id, tenant_ref.vcs_org_login,
        )
        return []

    _emit(
        f"  Scanning {len(repos)} repo(s) in "
        f"{tenant_ref.vcs_org_login} for .colony/ markers…",
    )

    new_colony_ids: list[str] = []
    for repo in repos:
        try:
            # Probe for ``.colony/`` first so the upsert below records
            # the right ``has_colony_marker`` value. Errors here are
            # caught by the outer except — the repo is silently
            # excluded from the cache rather than recorded with a
            # stale/false marker bit.
            has_marker = await provider.repo_path_exists(
                tenant=tenant_ref,
                repo=repo,
                path=_COLONY_MARKER_PATH,
                access_token=access_token,
            )

            # Cache the repo for the dashboard's discoverable-repos
            # dropdown so the "+ New colony" form + per-colony
            # "Design monorepo" picker can list it without needing
            # the user's OAuth token (which we discard after the
            # callback finishes).
            await auth_service.upsert_tenant_repo(
                db_pool,
                tenant_id=tenant_id,
                vcs_repo_id=repo.vcs_repo_id,
                vcs_repo_full_name=repo.full_name,
                default_branch=repo.default_branch,
                user_permission=repo.user_permission,
                has_colony_marker=has_marker,
            )

            if not has_marker:
                continue

            # Application-level "already discovered" gate — see
            # plan §4.3 for why this is NOT a SQL UNIQUE constraint.
            # Runs AFTER the cache upsert so the repo still shows up
            # in the dropdown even when it already has a colony.
            if await auth_service.any_colony_exists_for_repo(
                db_pool,
                tenant_id=tenant_id,
                vcs_repo_id=repo.vcs_repo_id,
            ):
                continue

            display_name = repo.full_name.split("/")[-1]
            _emit(f"    + Provisioning colony for {repo.full_name}…")
            result = await provision_colony(
                colony,
                tenant_id=tenant_id,
                name=display_name,
                description=(
                    f"Auto-discovered from {provider.display_name} "
                    f"repo {repo.full_name}"
                ),
                vcs_repo_id=repo.vcs_repo_id,
                vcs_repo_full_name=repo.full_name,
                default_branch=repo.default_branch,
            )
            # ``provision_colony`` derives + persists
            # ``design_monorepo_url`` from the tenant's provider when
            # the repo trio is supplied — no extra call here.
            new_colony_ids.append(result["colony_id"])
            logger.info(
                "discover_colonies_for_tenant: provisioned "
                "colony=%s for tenant=%s repo=%s",
                result["colony_id"], tenant_id, repo.full_name,
            )
        except Exception:  # noqa: BLE001 — best-effort per repo
            logger.exception(
                "discover_colonies_for_tenant: failed for "
                "tenant=%s repo=%s; continuing with next repo",
                tenant_id, repo.full_name,
            )

    return new_colony_ids


__all__ = ("discover_colonies_for_tenant",)
