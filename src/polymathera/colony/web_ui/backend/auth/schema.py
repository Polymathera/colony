"""Auth database schema and migrations.

Tables:
- users: user accounts with bcrypt password hashes
- colonies: user-created workspaces (maps to colony_id in ExecutionContext)

Each user is assigned a unique tenant_id (= user_id for v1).
Colonies are owned by a tenant and provide isolated VCM scopes,
sessions, and agent namespaces.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    tenant_id   TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

COLONIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS colonies (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    tenant_id   TEXT NOT NULL REFERENCES users(tenant_id) ON DELETE CASCADE,
    description TEXT NOT NULL DEFAULT '',
    is_default  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, name)
);
"""

COLONIES_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_colonies_tenant ON colonies(tenant_id);"

# Per-tenant table. A tenant is the GitHub-installation owner and (in
# v2) hosts multiple users + colonies. ``github_installation_id`` is
# the per-tenant GitHub App installation id; ``GitHubCapability`` will
# read it from agent metadata (P4/P5 of
# ``colony/github_identity_fix_plan.md``) to mint REST tokens scoped
# to this tenant's repos.
TENANTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    id                       TEXT PRIMARY KEY,
    name                     TEXT NOT NULL,
    github_installation_id   TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Backfill ``tenants`` from existing ``users.tenant_id`` so legacy DBs
# (where tenant_id = user_id, one tenant per user) gain a real
# tenants row per existing tenant_id. Fresh DBs with no users are a
# no-op. Idempotent (``ON CONFLICT DO NOTHING``).
TENANTS_BACKFILL_SQL = """
INSERT INTO tenants (id, name)
SELECT u.tenant_id, COALESCE(u.username, u.tenant_id)
FROM users u
ON CONFLICT (id) DO NOTHING;
"""

# Rewire ``users.tenant_id`` and ``colonies.tenant_id`` FKs to point at
# ``tenants(id)``. Drops the legacy UNIQUE on users.tenant_id (v1's
# ``tenant_id = user_id`` shortcut blocks v2 multi-user-per-tenant).
# Each ``DROP CONSTRAINT IF EXISTS`` + named ``ADD CONSTRAINT`` pair is
# idempotent across re-runs (DROP IF EXISTS handles both inline-FK and
# previously-migrated cases; the matching DROP before ADD makes ADD
# safe to repeat).
TENANTS_REWIRE_MIGRATIONS = (
    # Order matters: drop ``colonies_tenant_id_fkey`` BEFORE
    # ``users_tenant_id_key``. The v1 schema declared
    # ``colonies.tenant_id REFERENCES users(tenant_id)`` and
    # ``users.tenant_id ... UNIQUE`` — Postgres auto-creates
    # ``users_tenant_id_key`` (the UNIQUE index) and the colonies FK
    # depends on it. Dropping the UNIQUE first raises
    # ``DependentObjectsStillExistError``.
    "ALTER TABLE colonies DROP CONSTRAINT IF EXISTS colonies_tenant_id_fkey;",
    "ALTER TABLE users DROP CONSTRAINT IF EXISTS users_tenant_id_key;",
    "ALTER TABLE users DROP CONSTRAINT IF EXISTS users_tenant_id_fk;",
    "ALTER TABLE users ADD CONSTRAINT users_tenant_id_fk "
    "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE;",
    "ALTER TABLE colonies ADD CONSTRAINT colonies_tenant_id_fkey "
    "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE;",
)

# Per-colony design-monorepo configuration. ``ALTER TABLE ... ADD
# COLUMN IF NOT EXISTS`` migrates pre-existing colony rows in place
# without dropping data. Branch / commit have defaults; url is
# nullable because a colony can exist before the user has wired up
# its design monorepo.
COLONIES_DESIGN_MONOREPO_MIGRATIONS = (
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS design_monorepo_url TEXT;",
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS design_monorepo_branch TEXT NOT NULL DEFAULT 'main';",
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS design_monorepo_commit TEXT NOT NULL DEFAULT 'HEAD';",
)

# Per-colony git-commit attribution preferences. ``commit_principal``
# is the free-form identity string (well-known: ``user`` / ``colony`` /
# ``agent``; anything else is treated as an agent-type label).
# ``commit_co_author`` is the optional second identity rendered in
# the ``Co-Authored-By:`` trailer; NULL disables the trailer.
# Defaults match the framework's recommended shape: colony as
# principal, user as co-author — i.e., the colony (the persistent
# collective) did the work on behalf of the user (the human who
# started the session). Per-user ``git_user_name`` / ``git_user_email``
# live on ``users`` now (see USERS_GITHUB_IDENTITY_MIGRATIONS); the
# old per-colony copies are dropped by COLONIES_DROP_USER_IDENTITY_MIGRATIONS.
COLONIES_GIT_ATTRIBUTION_MIGRATIONS = (
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS commit_principal TEXT NOT NULL DEFAULT 'colony';",
    # ``DEFAULT 'user'`` backfills existing rows to ``'user'`` on the
    # first run of this migration AND applies to new rows thereafter.
    # Operators who want no co-author trailer set this to NULL via
    # the landing-page UI (``UPDATE`` in :func:`set_git_attribution`).
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS commit_co_author TEXT DEFAULT 'user';",
)

# Drop the legacy per-colony identity columns. Per-user identity (now
# OAuth-verified, see USERS_GITHUB_IDENTITY_MIGRATIONS) is the only
# source of git_user_name + git_user_email. No fallback, no compat
# (per github_identity_fix_plan.md §2 "What disappears").
COLONIES_DROP_USER_IDENTITY_MIGRATIONS = (
    "ALTER TABLE colonies DROP COLUMN IF EXISTS git_user_name;",
    "ALTER TABLE colonies DROP COLUMN IF EXISTS git_user_email;",
)

# Per-user GitHub identity. Populated only by the OAuth callback (the
# user clicks "Connect GitHub" on their profile and Colony pulls
# verified ``(login, github_user_id, email)`` from ``GET /user`` +
# ``GET /user/emails``). Free-form typing is not accepted — typed
# emails would let users impersonate any GitHub account in commit
# attribution. Backend routes land in P2 of
# ``colony/github_identity_fix_plan.md``; UI in P3; metadata threading
# in P4-P5.
USERS_GITHUB_IDENTITY_MIGRATIONS = (
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_login              TEXT;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_user_id            BIGINT;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_email              TEXT;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS git_user_name             TEXT;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_connected_at       TIMESTAMPTZ;",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_last_verified_at   TIMESTAMPTZ;",
)


async def ensure_auth_schema(db_pool) -> None:
    """Create auth tables if they don't exist.

    Called during dashboard startup alongside the observability schema.
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(USERS_TABLE_SQL)
            await conn.execute(COLONIES_TABLE_SQL)
            await conn.execute(COLONIES_INDEX_SQL)
            # Tenants table + FK rewire: must come after users +
            # colonies exist (so the backfill SELECT runs and the FK
            # ADDs find the tables) and before any column-level
            # migration that touches per-user identity (so the new
            # ``users.*`` columns land on a schema that's already
            # rewired to ``tenants``).
            await conn.execute(TENANTS_TABLE_SQL)
            await conn.execute(TENANTS_BACKFILL_SQL)
            for stmt in TENANTS_REWIRE_MIGRATIONS:
                await conn.execute(stmt)
            for stmt in COLONIES_DESIGN_MONOREPO_MIGRATIONS:
                await conn.execute(stmt)
            for stmt in COLONIES_GIT_ATTRIBUTION_MIGRATIONS:
                await conn.execute(stmt)
            for stmt in COLONIES_DROP_USER_IDENTITY_MIGRATIONS:
                await conn.execute(stmt)
            for stmt in USERS_GITHUB_IDENTITY_MIGRATIONS:
                await conn.execute(stmt)
        logger.info(
            "Auth schema ensured (users + tenants + colonies + "
            "design-monorepo + git-attribution prefs + per-user "
            "GitHub identity)"
        )
    except Exception:
        logger.error("Failed to create auth schema", exc_info=True)
        raise
