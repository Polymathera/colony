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

# Per-colony git-commit attribution. ``commit_principal`` is the
# free-form identity string (well-known: ``user`` / ``colony`` /
# ``agent``; anything else is treated as an agent-type label).
# ``commit_co_author`` is the optional second identity rendered in
# the ``Co-Authored-By:`` trailer; NULL disables the trailer.
# ``git_user_name`` / ``git_user_email`` are required when either
# field equals ``user``. Defaults match the framework's recommended
# shape: colony as principal, user as co-author — i.e., the colony
# (the persistent collective) did the work on behalf of the user
# (the human who started the session).
COLONIES_GIT_ATTRIBUTION_MIGRATIONS = (
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS git_user_name TEXT;",
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS git_user_email TEXT;",
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS commit_principal TEXT NOT NULL DEFAULT 'colony';",
    # ``DEFAULT 'user'`` backfills existing rows to ``'user'`` on the
    # first run of this migration AND applies to new rows thereafter.
    # Operators who want no co-author trailer set this to NULL via
    # the landing-page UI (``UPDATE`` in :func:`set_git_attribution`).
    "ALTER TABLE colonies ADD COLUMN IF NOT EXISTS commit_co_author TEXT DEFAULT 'user';",
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
            for stmt in COLONIES_DESIGN_MONOREPO_MIGRATIONS:
                await conn.execute(stmt)
            for stmt in COLONIES_GIT_ATTRIBUTION_MIGRATIONS:
                await conn.execute(stmt)
        logger.info(
            "Auth schema ensured (users + colonies + design-monorepo "
            "+ git-attribution columns)"
        )
    except Exception:
        logger.error("Failed to create auth schema", exc_info=True)
        raise
