"""PostgreSQL schema for the spans table.

Auto-creates the table and indexes on first connection.
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
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);
"""

INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans (trace_id, start_wall);",
    "CREATE INDEX IF NOT EXISTS idx_spans_run ON spans (run_id);",
    "CREATE INDEX IF NOT EXISTS idx_spans_agent ON spans (agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans (kind);",
]


async def ensure_schema(db_pool: Any) -> None:
    """Create the spans table and indexes if they don't exist."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(SPANS_TABLE_SQL)
            for idx_sql in INDEXES_SQL:
                await conn.execute(idx_sql)
        logger.info("Observability schema ensured")
    except Exception:
        logger.error("Failed to create observability schema", exc_info=True)
        raise
