"""Postgres schema for the GitHub inbound poller's cursor state.

One row per ``(tenant_id, colony_id, repo, channel)``. The PK is
intentionally multi-tenant — two tenants polling the same upstream
GitHub repo each get their own cursor so neither tenant's poll
position bleeds into the other's.

``channel`` is `'issues'` in P8a (v1); future channels (`'pr_reviews'`,
`'discussions'`, …) get their own rows under the same PK.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


GITHUB_POLL_CURSORS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS github_poll_cursors (
    tenant_id    TEXT NOT NULL,
    colony_id    TEXT NOT NULL,
    repo         TEXT NOT NULL,
    channel      TEXT NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL,
    last_seen_id TEXT,
    PRIMARY KEY (tenant_id, colony_id, repo, channel)
);
"""

GITHUB_POLL_CURSORS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_gpc_lookup
    ON github_poll_cursors (tenant_id, colony_id);
"""


async def ensure_github_inbound_schema(db_pool) -> None:
    """Create the ``github_poll_cursors`` table if missing.

    Idempotent — safe to call on every dashboard startup. No ALTERs
    yet (v1 schema); future column additions follow the
    ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` pattern used by
    ``ensure_auth_schema`` / ``ensure_chat_schema``.
    """

    try:
        async with db_pool.acquire() as conn:
            await conn.execute(GITHUB_POLL_CURSORS_TABLE_SQL)
            await conn.execute(GITHUB_POLL_CURSORS_INDEX_SQL)
        logger.info("GitHub inbound schema ensured")
    except Exception as e:
        logger.error("Failed to ensure github_inbound schema: %s", e)
        raise
