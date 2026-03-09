"""Span query store — reads spans from PostgreSQL for the dashboard API."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SpanQueryStore:
    """Read-side store: query spans from PostgreSQL for the dashboard."""

    def __init__(self, db_pool: Any):
        self._db_pool = db_pool

    async def get_spans(
        self,
        trace_id: str,
        run_id: str | None = None,
        kind: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Query spans for a trace with optional filters."""
        query = "SELECT * FROM spans WHERE trace_id = $1"
        params: list[Any] = [trace_id]
        idx = 2

        if run_id is not None:
            query += f" AND run_id = ${idx}"
            params.append(run_id)
            idx += 1

        if kind is not None:
            query += f" AND kind = ${idx}"
            params.append(kind)
            idx += 1

        query += f" ORDER BY start_wall ASC LIMIT ${idx}"
        params.append(limit)

        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_dict(row) for row in rows]

    async def list_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all traces (distinct trace_ids with aggregated stats)."""
        query = """
            SELECT
                trace_id,
                MIN(agent_id) AS agent_id,
                MAX(status) AS status,
                MIN(start_wall) AS start_time,
                COUNT(*) AS span_count,
                COUNT(DISTINCT run_id) FILTER (WHERE run_id IS NOT NULL) AS run_count,
                COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) AS total_tokens
            FROM spans
            GROUP BY trace_id
            ORDER BY MIN(start_wall) DESC
            LIMIT $1
        """
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query, limit)
            return [self._summary_to_dict(row) for row in rows]

    async def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """Aggregate stats for a single trace."""
        query = """
            SELECT
                trace_id,
                MIN(agent_id) AS agent_id,
                MAX(status) AS status,
                MIN(start_wall) AS start_time,
                COUNT(*) AS span_count,
                COUNT(DISTINCT run_id) FILTER (WHERE run_id IS NOT NULL) AS run_count,
                COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) AS total_tokens,
                COUNT(*) FILTER (WHERE status = 'error') AS error_count,
                MAX(end_wall) AS end_time
            FROM spans
            WHERE trace_id = $1
        """
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(query, trace_id)
            if row is None:
                return {}
            return self._summary_to_dict(row)

    @staticmethod
    def _summary_to_dict(row: Any) -> dict[str, Any]:
        """Convert a summary/aggregate row to a JSON-serializable dict."""
        d = dict(row)
        for key in ("start_time", "end_time"):
            if d.get(key) is not None:
                d[key] = d[key].timestamp()
        return d

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert an asyncpg Record to a JSON-serializable dict."""
        d = dict(row)
        # Convert timestamps to floats for JSON
        for key in ("start_wall", "end_wall"):
            if d.get(key) is not None:
                d[key] = d[key].timestamp()
        # Parse JSONB strings back to dicts
        for key in ("input_summary", "output_summary", "metadata"):
            val = d.get(key)
            if isinstance(val, str):
                d[key] = json.loads(val)
        return d
