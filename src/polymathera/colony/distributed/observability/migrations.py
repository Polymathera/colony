"""PostgreSQL schema for observability tables (spans + logs).

Auto-creates tables and indexes on first connection.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SPANS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    run_id TEXT,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_wall TIMESTAMPTZ NOT NULL,
    end_wall TIMESTAMPTZ,
    duration_ms DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    input_summary JSONB DEFAULT '{}',
    output_summary JSONB DEFAULT '{}',
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    model_name TEXT,
    context_page_ids TEXT[],
    ring TEXT,
    service_name TEXT,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);
"""

LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS logs (
    log_id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    level TEXT NOT NULL,
    logger_name TEXT NOT NULL,
    message TEXT NOT NULL,
    module TEXT,
    func_name TEXT,
    line_no INTEGER,
    pid INTEGER,
    thread_name TEXT,
    actor_class TEXT,
    node_id TEXT,
    tenant_id TEXT,
    colony_id TEXT,
    session_id TEXT,
    run_id TEXT,
    trace_id TEXT,
    exc_info TEXT
);
"""

SPAN_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans (trace_id, start_wall);",
    "CREATE INDEX IF NOT EXISTS idx_spans_run ON spans (run_id);",
    "CREATE INDEX IF NOT EXISTS idx_spans_agent ON spans (agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans (kind);",
]

LOG_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_logs_session_time ON logs (session_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_logs_run_time ON logs (run_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_logs_actor_time ON logs (actor_class, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_logs_level_time ON logs (level, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_logs_trace ON logs (trace_id, timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs (timestamp DESC);",
]


async def ensure_schema(db_pool: Any) -> None:
    """Create all observability tables and indexes if they don't exist."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(SPANS_TABLE_SQL)
            for idx_sql in SPAN_INDEXES_SQL:
                await conn.execute(idx_sql)
            await conn.execute(LOGS_TABLE_SQL)
            for idx_sql in LOG_INDEXES_SQL:
                await conn.execute(idx_sql)
        logger.info("Observability schema ensured (spans + logs)")
    except Exception:
        logger.error("Failed to create observability schema", exc_info=True)
        raise
