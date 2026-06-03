"""SQL helpers for the ``licenses`` table.

The non-trivial logic here is the **source-precedence ladder** on
upsert: a higher-priority source can overwrite a lower-priority one,
but never vice versa. The ladder is:

    admin > license_jwt > marketplace > stripe > env_bootstrap > default

i.e. a tenant whose plan was set via the (future) admin UI is NOT
silently downgraded by the env-bootstrap walker on the next dashboard
restart. A marketplace event from GitHub can overwrite an
env_bootstrap seed (paid customer → managed by Marketplace), but a
re-run of the env walker can't undo a paid plan.

See plan §9 for the full design.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .license_plans import PLAN_DEFAULTS, resolve_entitlements

logger = logging.getLogger(__name__)


# Higher index = higher precedence. Order MUST match the CHECK
# constraint values in ``schema.LICENSES_TABLE_SQL``'s ``source``
# column — adding a new source means adding here AND there.
_SOURCE_PRECEDENCE: tuple[str, ...] = (
    "default",
    "env_bootstrap",
    "stripe",
    "marketplace",
    "license_jwt",
    "admin",
)


def _precedence(source: str) -> int:
    """Numeric rank for ``source``. Unknown sources rank ``-1`` so
    they can't overwrite anything — fail closed."""
    try:
        return _SOURCE_PRECEDENCE.index(source)
    except ValueError:
        return -1


async def upsert_license(
    db_pool,
    *,
    tenant_id: str,
    plan: str,
    source: str,
    entitlements: dict[str, Any] | None = None,
    valid_until: Any = None,
) -> dict[str, Any]:
    """Insert or overwrite a license row, respecting source precedence.

    Returns ``{"applied": bool, "row": <current-row>}``. ``applied`` is
    False when the existing row's source outranks ``source`` — the
    write was skipped, but no error is raised (env_bootstrap walker
    re-runs are idempotent and must not log-spam when a real plan is
    already in place).

    Raises :class:`ValueError` for unknown plan or unknown source.
    """
    if plan not in PLAN_DEFAULTS:
        raise ValueError(f"Unknown plan {plan!r}")
    if source not in _SOURCE_PRECEDENCE:
        raise ValueError(f"Unknown source {source!r}")

    new_rank = _precedence(source)
    entitlements_json = json.dumps(entitlements or {})

    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT plan, source, entitlements, valid_until, updated_at "
            "FROM licenses WHERE tenant_id = $1",
            tenant_id,
        )
        if existing is not None:
            existing_rank = _precedence(existing["source"])
            if existing_rank > new_rank:
                logger.info(
                    "upsert_license: skipping %s write for tenant=%s "
                    "(existing source=%s outranks)",
                    source, tenant_id, existing["source"],
                )
                return {"applied": False, "row": _row_to_dict(existing)}

        await conn.execute(
            "INSERT INTO licenses ("
            "  tenant_id, plan, entitlements, source, valid_until, updated_at"
            ") VALUES ($1, $2, $3::jsonb, $4, $5, NOW()) "
            "ON CONFLICT (tenant_id) DO UPDATE SET "
            "  plan = EXCLUDED.plan, "
            "  entitlements = EXCLUDED.entitlements, "
            "  source = EXCLUDED.source, "
            "  valid_until = EXCLUDED.valid_until, "
            "  updated_at = NOW()",
            tenant_id, plan, entitlements_json, source, valid_until,
        )
        fresh = await conn.fetchrow(
            "SELECT plan, source, entitlements, valid_until, updated_at "
            "FROM licenses WHERE tenant_id = $1",
            tenant_id,
        )
        # ``fresh`` is non-None here — the row was just upserted.
        # The assert is a correctness check, not an error path.
        assert fresh is not None
    return {"applied": True, "row": _row_to_dict(fresh)}


async def get_license(
    db_pool, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Return the tenant's license row + resolved entitlements.

    Returns ``None`` when no license row exists (the caller decides
    whether to lazily upsert a ``default``-source ``free`` row). The
    returned dict has the row's plan + source + raw entitlements
    overrides AND a top-level ``effective_entitlements`` key that is
    the merge of plan defaults + overrides — the shape callers
    actually want to enforce against.
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT plan, source, entitlements, valid_until, updated_at "
            "FROM licenses WHERE tenant_id = $1",
            tenant_id,
        )
    if row is None:
        return None
    base = _row_to_dict(row)
    base["effective_entitlements"] = resolve_entitlements(
        plan=base["plan"], overrides=base["entitlements"],
    )
    return base


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an asyncpg ``Record`` (or anything that supports
    ``__getitem__`` by column name) into a plain dict. Parses the
    ``entitlements`` JSONB column into a Python dict."""
    raw_entitlements = row["entitlements"]
    if isinstance(raw_entitlements, str):
        parsed: dict[str, Any] = json.loads(raw_entitlements) or {}
    elif isinstance(raw_entitlements, dict):
        parsed = raw_entitlements
    else:
        parsed = {}
    return {
        "plan": row["plan"],
        "source": row["source"],
        "entitlements": parsed,
        "valid_until": row["valid_until"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Dev-license seeding from .env
# ---------------------------------------------------------------------------

async def seed_dev_licenses(
    db_pool,
    raw_env: str | None,
) -> int:
    """Parse ``COLONY_DEV_LICENSED_INSTALLATIONS`` and upsert one
    license row per entry. Returns the number of rows successfully
    written (excludes skips: malformed entries, unknown plans,
    installations whose tenant row doesn't exist yet).

    Idempotent — designed to re-run on every dashboard startup AND
    every successful user sign-in (the user-tenant-sync walker calls
    it after landing a new tenant from the user's
    ``/user/installations`` payload).
    """
    if not raw_env:
        return 0

    written = 0
    for entry in (e.strip() for e in raw_env.split(",")):
        if not entry:
            continue
        installation_id, sep, plan = entry.partition(":")
        installation_id = installation_id.strip()
        plan = (plan.strip() or "dev") if sep else "dev"
        if not installation_id:
            logger.warning(
                "seed_dev_licenses: empty installation id in entry %r; "
                "skipping",
                entry,
            )
            continue
        if plan not in PLAN_DEFAULTS:
            logger.warning(
                "seed_dev_licenses: unknown plan %r for installation "
                "%s; skipping",
                plan, installation_id,
            )
            continue

        # Resolve installation_id → tenant_id. Tenant may not exist
        # yet (first sign-in lands it via the user-tenant-sync
        # walker). When it lands, this same seeder runs again and
        # writes the row — idempotent ON CONFLICT semantics.
        async with db_pool.acquire() as conn:
            tenant_row = await conn.fetchrow(
                "SELECT id FROM tenants WHERE github_installation_id = $1",
                installation_id,
            )
        if tenant_row is None:
            logger.info(
                "seed_dev_licenses: no tenant yet for installation=%s; "
                "skipping (will retry on next signup walker pass)",
                installation_id,
            )
            continue

        result = await upsert_license(
            db_pool,
            tenant_id=tenant_row["id"],
            plan=plan,
            source="env_bootstrap",
        )
        if result["applied"]:
            written += 1
            logger.info(
                "seed_dev_licenses: tenant=%s installation=%s plan=%s",
                tenant_row["id"], installation_id, plan,
            )

    return written


__all__ = (
    "get_license",
    "seed_dev_licenses",
    "upsert_license",
)
