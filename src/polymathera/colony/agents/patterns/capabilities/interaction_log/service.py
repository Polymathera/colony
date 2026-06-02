"""Pure SQL helpers for the ``interaction_log`` table.

Separated from :mod:`.capability` so the write path + read APIs are
testable against a fake asyncpg pool without standing up the
full ``AgentCapability`` machinery. The capability's
``@event_handler`` methods classify a blackboard event into the
right shape + call :func:`insert_event`; dashboard routes (P11)
will call the ``fetch_*`` helpers directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


# JSON-encoder for the payload + refs columns. asyncpg's default
# JSONB codec serializes via Python's stdlib ``json``; we apply it
# explicitly so a non-serialisable value (e.g. a ``datetime``)
# surfaces here with a clear traceback instead of an opaque codec
# error at the connection layer.
def _to_jsonb(value: Any) -> str:
    return json.dumps(value, default=str)


async def insert_event(
    db_pool,
    *,
    tenant_id: str,
    colony_id: str,
    channel: str,
    event_kind: str,
    payload: dict[str, Any],
    refs: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    user_login: str | None = None,
    channel_ref: str | None = None,
) -> int:
    """Insert one ``interaction_log`` row, return the new ``id``.

    ``tenant_id`` + ``colony_id`` are required (multi-tenant
    correctness — see :mod:`.schema` for the rationale). Everything
    else is nullable so callers don't have to invent placeholders for
    events that genuinely don't have e.g. a session_id (a GitHub
    inbound event has no session).
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO interaction_log "
            "(tenant_id, colony_id, session_id, run_id, user_login, "
            " channel, channel_ref, event_kind, payload, refs) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb) "
            "RETURNING id",
            tenant_id, colony_id, session_id, run_id, user_login,
            channel, channel_ref, event_kind,
            _to_jsonb(payload),
            _to_jsonb(refs or []),
        )
    return int(row["id"])


async def fetch_recent_activity(
    db_pool,
    *,
    tenant_id: str,
    colony_id: str,
    user_login: str | None = None,
    since: Any | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return the most recent N rows for the given tenant + colony,
    most-recent first.

    Filters:
    - ``user_login``: when set, only rows whose ``user_login`` matches.
    - ``since``: when set (a ``datetime``), only rows with ``ts >= since``.

    Tenant + colony are NOT optional — cross-tenant leak prevention.
    """

    clauses = ["tenant_id = $1", "colony_id = $2"]
    args: list[Any] = [tenant_id, colony_id]
    if user_login is not None:
        args.append(user_login)
        clauses.append(f"user_login = ${len(args)}")
    if since is not None:
        args.append(since)
        clauses.append(f"ts >= ${len(args)}")
    args.append(int(limit))
    sql = (
        "SELECT id, ts, tenant_id, colony_id, session_id, run_id, "
        "user_login, channel, channel_ref, event_kind, payload, refs "
        f"FROM interaction_log WHERE {' AND '.join(clauses)} "
        f"ORDER BY ts DESC LIMIT ${len(args)}"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [_row_to_dict(r) for r in rows]


async def fetch_by_ref(
    db_pool,
    *,
    tenant_id: str,
    colony_id: str,
    ref_kind: str,
    ref_value: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return rows whose ``refs`` JSONB array contains
    ``{"kind": ref_kind, "value": ref_value}``.

    Uses the ``idx_il_refs`` GIN index with ``jsonb_path_ops`` —
    cheap even on large logs.

    Tenant + colony are NOT optional.
    """

    needle = [{"kind": ref_kind, "value": ref_value}]
    sql = (
        "SELECT id, ts, tenant_id, colony_id, session_id, run_id, "
        "user_login, channel, channel_ref, event_kind, payload, refs "
        "FROM interaction_log "
        "WHERE tenant_id = $1 AND colony_id = $2 "
        "AND refs @> $3::jsonb "
        "ORDER BY ts DESC LIMIT $4"
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            sql, tenant_id, colony_id, _to_jsonb(needle), int(limit),
        )
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict[str, Any]:
    """Convert an asyncpg ``Record`` to a plain dict with JSONB
    columns decoded back to Python objects.

    asyncpg returns JSONB as strings unless a custom codec is set; we
    decode here so callers get ``dict``/``list`` directly."""

    out = dict(row)
    for col in ("payload", "refs"):
        if isinstance(out.get(col), str):
            try:
                out[col] = json.loads(out[col])
            except (ValueError, TypeError):
                pass  # leave as string; surface the corruption
    return out
