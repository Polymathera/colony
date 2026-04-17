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
        ring: str | None = None,
        service_name: str | None = None,
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

        if ring is not None:
            query += f" AND ring = ${idx}"
            params.append(ring)
            idx += 1

        if service_name is not None:
            query += f" AND service_name = ${idx}"
            params.append(service_name)
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

    async def list_agents_from_spans(self) -> list[dict[str, Any]]:
        """Reconstruct the agent list from persisted AGENT spans.

        Used when the Ray cluster is down and the live AgentSystem is
        unavailable. Each AGENT span (kind='agent') stores agent_type,
        capabilities, parent_agent_id, and bound_pages in input_summary,
        and stop_reason in output_summary.
        """
        query = """
            SELECT
                agent_id,
                status,
                input_summary,
                output_summary,
                start_wall,
                end_wall,
                trace_id
            FROM spans
            WHERE kind = 'agent'
            ORDER BY start_wall DESC
        """
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query)

        # Deduplicate by agent_id (keep the latest span per agent)
        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            d = dict(row)
            aid = d["agent_id"]
            if aid in seen:
                continue
            for key in ("start_wall", "end_wall"):
                if d.get(key) is not None:
                    d[key] = d[key].timestamp()
            for key in ("input_summary", "output_summary"):
                val = d.get(key)
                if isinstance(val, str):
                    d[key] = json.loads(val)
            inp = d.get("input_summary") or {}
            out = d.get("output_summary") or {}
            seen[aid] = {
                "agent_id": aid,
                "agent_type": inp.get("agent_type", ""),
                "state": "stopped" if d["status"] == "ok" else d["status"],
                "capabilities": inp.get("capability_names", []),
                "parent_agent_id": inp.get("parent_agent_id"),
                "bound_pages": inp.get("bound_pages", []),
                "stop_reason": out.get("stop_reason"),
                "start_wall": d.get("start_wall"),
                "end_wall": d.get("end_wall"),
                "trace_id": d.get("trace_id"),
            }

        return list(seen.values())

    async def get_agent_history(self, agent_id: str) -> dict[str, Any]:
        """Aggregate span data for one agent: lifecycle events, action stats, token usage.

        Returns a summary suitable for the Agents tab detail panel.
        Uses the existing ``idx_spans_agent`` index.
        """
        query = """
            SELECT
                COUNT(*) FILTER (WHERE kind = 'action') AS action_count,
                COUNT(*) FILTER (WHERE kind = 'action' AND status = 'ok') AS action_successes,
                COUNT(*) FILTER (WHERE kind = 'action' AND status = 'error') AS action_failures,
                COUNT(*) FILTER (WHERE kind = 'infer') AS infer_count,
                COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS total_cache_tokens,
                MIN(start_wall) AS first_seen,
                MAX(COALESCE(end_wall, start_wall)) AS last_seen,
                COUNT(*) AS total_spans
            FROM spans
            WHERE agent_id = $1
        """
        lifecycle_query = """
            SELECT name, status, error, start_wall, output_summary
            FROM spans
            WHERE agent_id = $1 AND kind = 'lifecycle'
            ORDER BY start_wall ASC
        """
        last_error_query = """
            SELECT error, start_wall, name
            FROM spans
            WHERE agent_id = $1 AND status = 'error' AND error IS NOT NULL
            ORDER BY start_wall DESC
            LIMIT 1
        """
        async with self._db_pool.acquire() as conn:
            stats_row = await conn.fetchrow(query, agent_id)
            lifecycle_rows = await conn.fetch(lifecycle_query, agent_id)
            last_error_row = await conn.fetchrow(last_error_query, agent_id)

        stats = dict(stats_row) if stats_row else {}
        for key in ("first_seen", "last_seen"):
            if stats.get(key) is not None:
                stats[key] = stats[key].timestamp()

        lifecycle_events = []
        for row in lifecycle_rows:
            d = dict(row)
            if d.get("start_wall") is not None:
                d["start_wall"] = d["start_wall"].timestamp()
            output = d.get("output_summary")
            if isinstance(output, str):
                d["output_summary"] = json.loads(output)
            lifecycle_events.append(d)

        last_error = None
        if last_error_row:
            last_error = {
                "error": last_error_row["error"],
                "timestamp": last_error_row["start_wall"].timestamp() if last_error_row["start_wall"] else None,
                "span_name": last_error_row["name"],
            }

        return {
            **stats,
            "lifecycle_events": lifecycle_events,
            "last_error": last_error,
        }

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
