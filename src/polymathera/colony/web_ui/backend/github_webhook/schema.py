"""Postgres dedup table for GitHub webhook deliveries.

GitHub retries with exponential backoff if our endpoint returns
non-2xx (up to 5 times). Combined with this PK-on-delivery_id dedup
table, that gives at-least-once → exactly-once-effective semantics:
- Every delivery has a unique ``X-GitHub-Delivery`` UUID.
- The receiver INSERTs that id with ``ON CONFLICT DO NOTHING``.
- ``rowcount==0`` → seen before → respond 200 with ``{"status":
  "duplicate"}`` + skip republish.

Retention is a follow-up: rows older than 30 days can be cleaned by
a periodic mission (or a plain SQL cron). For v1 the table grows
unbounded; on a busy deployment that's still ≤ tens of millions of
rows per year, well within Postgres single-table limits.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


GITHUB_WEBHOOK_DELIVERIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS github_webhook_deliveries (
    delivery_id TEXT PRIMARY KEY,
    event_type  TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

GITHUB_WEBHOOK_DELIVERIES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_gwd_received_at
    ON github_webhook_deliveries (received_at);
"""


async def ensure_github_webhook_schema(db_pool) -> None:
    """Create the dedup table + retention index if missing.

    Idempotent — safe to call on every dashboard startup."""

    try:
        async with db_pool.acquire() as conn:
            await conn.execute(GITHUB_WEBHOOK_DELIVERIES_TABLE_SQL)
            await conn.execute(GITHUB_WEBHOOK_DELIVERIES_INDEX_SQL)
        logger.info("github_webhook_deliveries schema ensured")
    except Exception as e:
        logger.error("Failed to ensure github_webhook schema: %s", e)
        raise


async def record_delivery(
    db_pool, *, delivery_id: str, event_type: str,
) -> bool:
    """Insert one delivery row. Returns ``True`` if newly inserted,
    ``False`` if the delivery id was already present (GitHub retry).

    Driver of the at-least-once → exactly-once-effective contract.
    """

    async with db_pool.acquire() as conn:
        # ``ON CONFLICT DO NOTHING`` + ``RETURNING xmax`` is the
        # idiomatic shape for "did this insert actually happen?".
        # When the row already existed, the INSERT yields no rows;
        # when it's new, exactly one row comes back.
        row = await conn.fetchrow(
            "INSERT INTO github_webhook_deliveries "
            "(delivery_id, event_type) "
            "VALUES ($1, $2) "
            "ON CONFLICT (delivery_id) DO NOTHING "
            "RETURNING delivery_id",
            delivery_id, event_type,
        )
    return row is not None
