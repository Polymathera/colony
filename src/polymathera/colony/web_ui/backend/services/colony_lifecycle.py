"""Single entry point for colony creation.

**Discipline**: every code path that lands a row in the ``colonies``
table MUST go through :func:`provision_colony`. Going around it via a
direct ``auth_service.create_colony(...)`` (or a raw ``INSERT``)
skips the per-colony ``SessionAgent`` bootstrap that hosts the
colony-singleton capabilities (P8 ``GitHubInboundCapability``,
``InteractionLogCapability``; P9 webhook receiver fan-out; P10
``MentionRoutingCapability``). The result is a colony whose
capabilities never run — visible to operators as: nothing shows up
in the Traces / Agents / Overview tabs except per-user sessions.

This was the regression that triggered this centralization: the
signup handler was creating a "Default" colony inside
``auth_service.create_user``'s transaction without bootstrapping a
system session, and the Traces tab silently lost half its rows.

If you find yourself adding ``INSERT INTO colonies`` somewhere, OR
adding a second helper that creates colonies, STOP and call
:func:`provision_colony` instead. The whole point of this module is
that it is the ONLY way to provision a colony.
"""

from __future__ import annotations

import asyncio
import logging

from ..auth import service as auth_service
from ..chat.system_session import ensure_system_session_for_colony
from .colony_connection import ColonyConnection


logger = logging.getLogger(__name__)

# Bounded wait for the session_manager deployment's ``@on_app_ready``
# hook to finish (populates vcm_handle). Without this, every endpoint
# raises ``RuntimeError("VCM handle not yet available")`` until the
# framework finishes wiring siblings — typically a few seconds after
# Ray cluster reachability.
_SESSION_MANAGER_READY_TIMEOUT_S = 60.0
_SESSION_MANAGER_READY_POLL_S = 1.0


async def _wait_for_session_manager_ready(colony: ColonyConnection) -> bool:
    """Poll until the session_manager deployment is registered AND
    its ``@on_app_ready`` hook has finished. Returns True on ready,
    False on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SESSION_MANAGER_READY_TIMEOUT_S
    while True:
        try:
            sm = await colony.get_session_manager()
            if await sm.is_ready():
                return True
        except Exception:  # noqa: BLE001 — pre-registration / pre-init
            pass
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_SESSION_MANAGER_READY_POLL_S)


async def provision_colony(
    colony: ColonyConnection,
    *,
    tenant_id: str,
    name: str,
    description: str = "",
    vcs_repo_id: str | None = None,
    vcs_repo_full_name: str | None = None,
    default_branch: str | None = None,
    commit_principal: str | None = None,
    commit_co_author: str | None = None,
) -> dict[str, str]:
    """Land a colony row + bootstrap its always-on system SessionAgent.

    Composes the steps every colony-creation path MUST run together:

    1. ``auth_service.create_colony`` — atomic SQL insert into the
       ``colonies`` table.
    2. If repo binding fields are supplied: derive the clone URL via
       the tenant's provider's ``repo_clone_url`` + persist via
       ``set_design_monorepo``. Best-effort — the colony row already
       exists, so a derivation failure logs and the operator can
       still type the URL in the UI.
    3. If commit attribution is supplied: persist via
       ``set_git_attribution`` (also best-effort).
    4. ``ensure_system_session_for_colony`` — best-effort Ray-side
       bootstrap of the system ``SessionAgent`` that hosts
       colony-singleton capabilities.

    Args:
        colony: The dashboard's ``ColonyConnection``.
        tenant_id: The tenant the new colony belongs to.
        name: Human-facing colony name.
        description: Free-text description (optional).
        vcs_repo_id / vcs_repo_full_name / default_branch: Bind this
            colony to a specific VCS repo. Auto-discovery walker (PR 4)
            populates all three; the "+ New colony" UI form populates
            them when the user picks from the discoverable-repos
            dropdown.
        commit_principal / commit_co_author: Per-commit attribution
            preferences. ``None`` leaves the schema defaults in place
            (``colony`` / ``user``).
    """

    db_pool = colony._db_pool
    if db_pool is None:
        raise RuntimeError(
            "provision_colony: no db_pool on ColonyConnection — "
            "cannot insert colony row.",
        )

    result = await auth_service.create_colony(
        db_pool,
        tenant_id=tenant_id,
        name=name,
        description=description,
        vcs_repo_id=vcs_repo_id,
        vcs_repo_full_name=vcs_repo_full_name,
        default_branch=default_branch,
    )

    # Derive + persist the design_monorepo_url from the tenant's
    # provider's clone-URL template. Today's design-monorepo readers
    # (UI textbox, RepoStateProvider, materialize_design_context) all
    # read this column.
    if vcs_repo_full_name and default_branch and vcs_repo_id:
        try:
            await _persist_design_monorepo_url(
                db_pool,
                colony_id=result["colony_id"],
                tenant_id=tenant_id,
                vcs_repo_id=vcs_repo_id,
                vcs_repo_full_name=vcs_repo_full_name,
                default_branch=default_branch,
            )
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "provision_colony: failed to derive design_monorepo_url "
                "for colony %s; operator can set it manually via UI.",
                result["colony_id"],
            )

    if commit_principal is not None or commit_co_author is not None:
        try:
            await auth_service.set_git_attribution(
                db_pool,
                colony_id=result["colony_id"],
                tenant_id=tenant_id,
                # Fall back to the schema defaults when the caller
                # only supplied one of the two.
                commit_principal=commit_principal or "colony",
                commit_co_author=commit_co_author,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "provision_colony: failed to persist git_attribution "
                "for colony %s; operator can set it via UI.",
                result["colony_id"],
            )

    # Wait for the session_manager's ``@on_app_ready`` to finish
    # (populates vcm_handle) before invoking endpoints that depend on
    # it. The lifespan walker is the safety net for any colony that
    # times out here — re-runs bootstrap on next restart.
    if not await _wait_for_session_manager_ready(colony):
        logger.error(
            "provision_colony: session_manager not ready within %.0fs; "
            "skipping bootstrap for colony %s — lifespan walker will "
            "retry next restart.",
            _SESSION_MANAGER_READY_TIMEOUT_S, result["colony_id"],
        )
        return result

    try:
        await ensure_system_session_for_colony(
            colony,
            tenant_id=result["tenant_id"],
            colony_id=result["colony_id"],
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "provision_colony: system-session bootstrap failed for "
            "colony %s (tenant=%s); colony-singleton capabilities "
            "will not run until the next dashboard restart.",
            result["colony_id"], result["tenant_id"],
        )

    return result


async def _persist_design_monorepo_url(
    db_pool,
    *,
    colony_id: str,
    tenant_id: str,
    vcs_repo_id: str,
    vcs_repo_full_name: str,
    default_branch: str,
) -> None:
    """Look up the tenant's VCS provider in the registry, render the
    clone URL, and persist it on the colony row.

    Lazy import of the registry: this module is imported from many
    paths and we don't want a circular at module-load. The provider
    is registered at dashboard startup (``main._register_vcs_providers``);
    if not registered (operator deployed without that provider's
    OAuth creds), we log and skip — the colony row stays with
    ``design_monorepo_url = NULL`` and the operator can set it manually.
    """
    from polymathera.colony.vcs import get_provider
    from polymathera.colony.vcs.provider import VcsRepoRef

    # Read the tenant's provider id to know which provider to ask.
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT vcs_provider FROM tenants WHERE id = $1",
            tenant_id,
        )
    if row is None:
        logger.warning(
            "_persist_design_monorepo_url: tenant %s not found; "
            "leaving design_monorepo_url NULL",
            tenant_id,
        )
        return
    provider_id = row["vcs_provider"] or "github"
    try:
        provider = get_provider(provider_id)
    except KeyError:
        logger.warning(
            "_persist_design_monorepo_url: provider %r not registered "
            "(operator may not have configured its OAuth creds); "
            "leaving design_monorepo_url NULL for colony %s.",
            provider_id, colony_id,
        )
        return
    clone_url = provider.repo_clone_url(VcsRepoRef(
        vcs_repo_id=vcs_repo_id,
        full_name=vcs_repo_full_name,
        default_branch=default_branch,
        # Permission is irrelevant for URL rendering — the
        # ``repo_clone_url`` method is pure formatting.
        user_permission="read",
    ))
    await auth_service.set_design_monorepo(
        db_pool,
        colony_id=colony_id,
        tenant_id=tenant_id,
        origin_url=clone_url,
        branch=default_branch,
    )


__all__ = ("provision_colony",)
