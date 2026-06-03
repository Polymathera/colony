"""First-login walker: sync a user's tenants + colonies from their VCS.

Called by ``routers/auth.py`` after every OAuth sign-in. The walker is
idempotent — running it on a repeat sign-in refreshes tenant rows +
membership ``last_verified_at`` timestamps + discovers any newly-added
``.colony/``-marked repos without creating duplicates.

See ``colony/vcs_native_tenancy_plan.md §4.2`` for the design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from polymathera.colony.vcs.provider import VcsProvider, VcsUserIdentity

from ..auth import service as auth_service
from ..auth.license_service import seed_dev_licenses
from .colony_connection import ColonyConnection
from .colony_discovery import discover_colonies_for_tenant


ProgressCallback = Callable[[str], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignInResult:
    """Walker outcome — what the calling sign-in route needs to mint
    the JWT cookie + decide the post-sign-in redirect."""

    user_id: str
    is_new_user: bool
    tenant_ids: tuple[str, ...]
    active_colony_id: str | None
    discovered_colony_ids: tuple[str, ...] = ()


async def sync_user_after_signin(
    colony: ColonyConnection,
    *,
    provider: VcsProvider,
    identity: VcsUserIdentity,
    access_token: str,
    license_env_value: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> SignInResult:
    """Upsert the user, their tenants + memberships, auto-discover
    colonies for each tenant.

    ``on_progress`` receives short user-facing status strings at each
    macro-step (user upsert, tenant listing, per-tenant discovery,
    license seeding, active-colony resolution). Wired to the
    signin_progress in-memory dict by the OAuth callback so the
    HTML loading page can poll + display them.

    Returns enough info for the sign-in handler to mint a JWT + decide
    the post-sign-in redirect.
    """

    db_pool = colony._db_pool

    def _emit(msg: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(msg)
        except Exception:  # noqa: BLE001 — progress is best-effort
            logger.warning("sync_user_after_signin: progress callback raised", exc_info=True)

    # Step 1 — user upsert.
    _emit(f"Welcome, {identity.login}. Setting up your account…")
    user_result = await auth_service.upsert_user_from_vcs(
        db_pool,
        vcs_provider=provider.provider_id,
        vcs_user_id=identity.vcs_user_id,
        vcs_login=identity.login,
        vcs_email=identity.primary_email,
        name=identity.name,
    )
    user_id = user_result["user_id"]
    is_new_user = user_result["is_new"]

    # Step 2 — list tenants. Provider errors propagate; the sign-in
    # route catches OAuthExchangeError and surfaces a clean 4xx.
    _emit("Discovering your VCS organizations…")
    tenants = await provider.list_user_tenants(access_token=access_token)
    if tenants:
        _emit(f"Found {len(tenants)} organization(s) with Colony installed.")
    else:
        _emit("No organizations with Colony installed.")

    # Step 3 — upsert each tenant + its membership + discover colonies.
    tenant_ids: list[str] = []
    discovered_colony_ids: list[str] = []
    for tenant_ref in tenants:
        _emit(f"Syncing organization {tenant_ref.vcs_org_login}…")
        tenant_result = await auth_service.upsert_tenant_from_vcs(
            db_pool,
            vcs_provider=provider.provider_id,
            vcs_org_id=tenant_ref.vcs_org_id,
            vcs_org_login=tenant_ref.vcs_org_login,
            name=tenant_ref.display_name,
            github_installation_id=tenant_ref.installation_id,
        )
        await auth_service.upsert_user_tenant(
            db_pool,
            user_id=user_id,
            tenant_id=tenant_result["tenant_id"],
            role=tenant_ref.role_hint,
        )
        tenant_ids.append(tenant_result["tenant_id"])

        # Discover colonies. Failures inside the discovery walker
        # log + skip per-repo; only an OAuth-level error (revoked
        # token, missing scope) propagates and is caught here so a
        # single failing tenant doesn't abort sign-in.
        try:
            new_colony_ids = await discover_colonies_for_tenant(
                colony,
                tenant_id=tenant_result["tenant_id"],
                tenant_ref=tenant_ref,
                provider=provider,
                access_token=access_token,
                on_progress=on_progress,
            )
            discovered_colony_ids.extend(new_colony_ids)
            if new_colony_ids:
                _emit(
                    f"  ✓ Provisioned {len(new_colony_ids)} new "
                    f"colony in {tenant_ref.vcs_org_login}.",
                )
        except Exception:  # noqa: BLE001 — best-effort per tenant
            logger.exception(
                "sync_user_after_signin: colony discovery failed "
                "for tenant=%s vcs_org=%s; continuing sign-in",
                tenant_result["tenant_id"], tenant_ref.vcs_org_login,
            )
            _emit(
                f"  ⚠ Colony discovery failed for "
                f"{tenant_ref.vcs_org_login} — continuing.",
            )

    # Step 4 — re-seed dev licenses. Tenants that landed in step 3
    # without their license row get one here (idempotent on tenants
    # whose row already exists at a higher source).
    if tenant_ids:
        _emit("Refreshing license seeding…")
        try:
            await seed_dev_licenses(db_pool, license_env_value)
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "sync_user_after_signin: dev license seeding failed; "
                "user_id=%s; quotas may default to 'free' until next "
                "dashboard restart.", user_id,
            )

    # Pick the active colony — prefers an existing valid pointer,
    # otherwise the first colony in the first tenant (which is
    # likely one we just auto-discovered above).
    _emit("Resolving your active workspace…")
    active_colony_id = await _resolve_active_colony(
        db_pool, user_id=user_id, tenant_ids=tenant_ids,
    )
    _emit("Done — signing you in…")

    logger.info(
        "sync_user_after_signin: user=%s vcs=%s login=%s tenants=%d "
        "discovered_colonies=%d is_new=%s active_colony=%s",
        user_id, provider.provider_id, identity.login, len(tenant_ids),
        len(discovered_colony_ids), is_new_user, active_colony_id,
    )
    return SignInResult(
        user_id=user_id,
        is_new_user=is_new_user,
        tenant_ids=tuple(tenant_ids),
        active_colony_id=active_colony_id,
        discovered_colony_ids=tuple(discovered_colony_ids),
    )


async def _resolve_active_colony(
    db_pool, *, user_id: str, tenant_ids: list[str],
) -> str | None:
    """Pick the user's active colony after sign-in. Prefers an
    existing ``users.active_colony_id`` if it still points at a
    colony the user can see (membership-checked); otherwise picks the
    first colony in the first tenant; otherwise ``None``.

    Updates ``users.active_colony_id`` to the chosen value so the
    JWT-minting caller has the freshest row to read."""

    existing = await auth_service.get_user_by_id(db_pool, user_id)
    existing_colony_id = (existing or {}).get("active_colony_id")
    if existing_colony_id and tenant_ids:
        # Verify the previously-active colony still belongs to a
        # tenant the user can see — otherwise an admin demoting them
        # would leave a dangling active_colony_id.
        for tenant_id in tenant_ids:
            colonies = await auth_service.list_colonies(db_pool, tenant_id)
            if any(c["colony_id"] == existing_colony_id for c in colonies):
                return existing_colony_id
        # Stale pointer — fall through to "pick first" below.

    for tenant_id in tenant_ids:
        colonies = await auth_service.list_colonies(db_pool, tenant_id)
        if colonies:
            new_active = colonies[0]["colony_id"]
            await auth_service.set_active_colony(
                db_pool, user_id=user_id, colony_id=new_active,
            )
            return new_active

    # No colonies in any tenant the user can see. The sign-in
    # response carries None; the UI tells the user "no colonies yet".
    await auth_service.set_active_colony(
        db_pool, user_id=user_id, colony_id=None,
    )
    return None


__all__ = (
    "SignInResult",
    "sync_user_after_signin",
)
