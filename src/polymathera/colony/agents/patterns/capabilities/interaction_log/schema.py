"""Postgres schema for the cross-channel `interaction_log`.

JSONB ``payload`` + ``refs`` per design doc §14 — the set of event
kinds is open-ended (each blackboard protocol's value shape is
captured as-is) so a normalised per-kind table set would be
impossible to keep stable. Indexes on the structured columns
(``session_id`` / ``user_login`` / ``event_kind``) + a GIN index on
``refs`` keep the common queries fast:

- "what did user alice do recently?" → `(tenant_id, user_login)` index
- "what's the full history of issue X?" → GIN on `refs`
- "what happened in session S?" → `(session_id)` index

**Multi-tenant addition** over the design doc §14 schema: `tenant_id`
+ `colony_id` columns are NOT in §14 (the doc was sketched single-
tenant). The InteractionLog is one row per event across every
tenant's colonies, so without these columns a `WHERE user_login='alice'`
query would return rows from any tenant that happens to have a user
named alice. Added them as `NOT NULL` columns + a composite
`(tenant_id, colony_id, ts DESC)` index for the common
"timeline-for-this-colony" query.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


INTERACTION_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interaction_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    tenant_id     TEXT NOT NULL,
    colony_id     TEXT NOT NULL,
    session_id    TEXT,
    run_id        TEXT,
    user_login    TEXT,
    channel       TEXT NOT NULL,
    channel_ref   TEXT,
    event_kind    TEXT NOT NULL,
    payload       JSONB NOT NULL,
    refs          JSONB NOT NULL DEFAULT '[]'::jsonb,
    CHECK (channel IN ('chat', 'github', 'scheduled', 'internal'))
);
"""

INTERACTION_LOG_INDEXES_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_il_tenant_colony_ts "
    "ON interaction_log (tenant_id, colony_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_il_session "
    "ON interaction_log (session_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_il_user "
    "ON interaction_log (tenant_id, user_login, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_il_kind "
    "ON interaction_log (event_kind, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_il_refs "
    "ON interaction_log USING gin (refs jsonb_path_ops);",
)


async def ensure_interaction_log_schema(db_pool) -> None:
    """Create the ``interaction_log`` table + indexes if missing.

    Idempotent — every statement is ``CREATE IF NOT EXISTS``. Safe to
    call on every dashboard startup."""

    try:
        async with db_pool.acquire() as conn:
            await conn.execute(INTERACTION_LOG_TABLE_SQL)
            for stmt in INTERACTION_LOG_INDEXES_SQL:
                await conn.execute(stmt)
        logger.info("interaction_log schema ensured")
    except Exception as e:
        logger.error("Failed to ensure interaction_log schema: %s", e)
        raise
