"""Async wrappers around the ``github_poll_cursors`` table.

One row per ``(tenant_id, colony_id, repo, channel)``. The poller
reads the row at the top of each tick to compute the GraphQL
``since:`` filter, then bumps it after a successful pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cursor:
    """In-memory view of one row of ``github_poll_cursors``."""

    tenant_id: str
    colony_id: str
    repo: str
    channel: str
    last_updated: datetime
    last_seen_id: str | None


# Default cursor when the table has no row yet. Far enough back to
# include "any reasonable" recent activity on first tick without
# returning the full repo history.
_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)


async def get_cursor(
    db_pool,
    *,
    tenant_id: str,
    colony_id: str,
    repo: str,
    channel: str = "issues",
) -> Cursor:
    """Read the cursor row, returning a default if absent.

    The default ``last_updated`` is 2020-01-01 — a fixed "far past"
    sentinel that bounds first-tick load on a busy repo without
    hard-coding ``epoch=0`` (which made the GraphQL ``since`` filter
    return every issue ever in pre-P8a manual tests).
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_updated, last_seen_id "
            "FROM github_poll_cursors "
            "WHERE tenant_id = $1 AND colony_id = $2 "
            "AND repo = $3 AND channel = $4",
            tenant_id, colony_id, repo, channel,
        )

    if row is None:
        return Cursor(
            tenant_id=tenant_id, colony_id=colony_id, repo=repo,
            channel=channel, last_updated=_EPOCH, last_seen_id=None,
        )
    return Cursor(
        tenant_id=tenant_id, colony_id=colony_id, repo=repo,
        channel=channel,
        last_updated=row["last_updated"],
        last_seen_id=row["last_seen_id"],
    )


async def bump_cursor(
    db_pool,
    *,
    tenant_id: str,
    colony_id: str,
    repo: str,
    channel: str,
    last_updated: datetime,
    last_seen_id: str | None,
) -> None:
    """Upsert the cursor. Bumped only after a tick succeeds.

    Idempotent — re-running the same upsert with the same values is a
    no-op. The poller calls this at the END of a successful tick;
    partial failures roll back implicitly (we just don't call it).
    """

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO github_poll_cursors "
            "(tenant_id, colony_id, repo, channel, last_updated, "
            "last_seen_id) "
            "VALUES ($1, $2, $3, $4, $5, $6) "
            "ON CONFLICT (tenant_id, colony_id, repo, channel) "
            "DO UPDATE SET last_updated = EXCLUDED.last_updated, "
            "              last_seen_id = EXCLUDED.last_seen_id",
            tenant_id, colony_id, repo, channel,
            last_updated, last_seen_id,
        )
