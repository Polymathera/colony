"""Log query store — reads logs from PostgreSQL for the dashboard API.

Provides filtered, paginated queries over persisted log records.
Mirrors the SpanQueryStore pattern for consistency.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class LogQueryStore:
    """Read-side store: query logs from PostgreSQL for the dashboard."""

    def __init__(self, db_pool: Any):
        self._db_pool = db_pool

    async def query_logs(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        actor_class: str | None = None,
        level: str | None = None,
        search: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query logs with flexible filters.

        Args:
            session_id: Filter by session
            run_id: Filter by run
            trace_id: Filter by trace
            actor_class: Filter by deployment/actor class name
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            search: Full-text search in message (ILIKE)
            since: Unix timestamp — logs after this time
            until: Unix timestamp — logs before this time
            limit: Max records to return
            offset: Pagination offset

        Returns:
            List of log record dicts, newest first.
        """
        conditions = []
        params: list[Any] = []
        idx = 1

        if session_id:
            conditions.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1

        if run_id:
            conditions.append(f"run_id = ${idx}")
            params.append(run_id)
            idx += 1

        if trace_id:
            conditions.append(f"trace_id = ${idx}")
            params.append(trace_id)
            idx += 1

        if actor_class:
            conditions.append(f"actor_class = ${idx}")
            params.append(actor_class)
            idx += 1

        if level:
            conditions.append(f"level = ${idx}")
            params.append(level.upper())
            idx += 1

        if search:
            conditions.append(f"message ILIKE ${idx}")
            params.append(f"%{search}%")
            idx += 1

        if since:
            conditions.append(f"timestamp >= ${idx}")
            params.append(datetime.fromtimestamp(since, tz=timezone.utc))
            idx += 1

        if until:
            conditions.append(f"timestamp <= ${idx}")
            params.append(datetime.fromtimestamp(until, tz=timezone.utc))
            idx += 1

        where = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT * FROM logs
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        params.extend([limit, offset])

        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_dict(row) for row in rows]

    async def get_log_stats(
        self,
        session_id: str | None = None,
        since: float | None = None,
    ) -> dict[str, Any]:
        """Get aggregate log statistics."""
        conditions = []
        params: list[Any] = []
        idx = 1

        if session_id:
            conditions.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1

        if since:
            conditions.append(f"timestamp >= ${idx}")
            params.append(datetime.fromtimestamp(since, tz=timezone.utc))
            idx += 1

        where = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE level = 'ERROR') AS errors,
                COUNT(*) FILTER (WHERE level = 'WARNING') AS warnings,
                COUNT(DISTINCT session_id) FILTER (WHERE session_id IS NOT NULL) AS sessions,
                COUNT(DISTINCT actor_class) FILTER (WHERE actor_class != '') AS actors,
                MIN(timestamp) AS earliest,
                MAX(timestamp) AS latest
            FROM logs
            WHERE {where}
        """
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            if not row:
                return {}
            d = dict(row)
            for key in ("earliest", "latest"):
                if d.get(key) is not None:
                    d[key] = d[key].timestamp()
            return d

    async def list_actor_classes(self) -> list[dict[str, Any]]:
        """List distinct actor classes with log counts."""
        query = """
            SELECT
                actor_class,
                COUNT(*) AS log_count,
                MAX(timestamp) AS latest
            FROM logs
            WHERE actor_class != '' AND actor_class IS NOT NULL
            GROUP BY actor_class
            ORDER BY MAX(timestamp) DESC
        """
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query)
            result = []
            for row in rows:
                d = dict(row)
                if d.get("latest"):
                    d["latest"] = d["latest"].timestamp()
                result.append(d)
            return result

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert an asyncpg Record to a JSON-serializable dict."""
        d = dict(row)
        if d.get("timestamp") is not None:
            d["timestamp"] = d["timestamp"].timestamp()
        return d
