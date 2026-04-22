"""Chat message persistence schema.

Table:
- chat_messages: stores all chat messages for all sessions, linked by session_id.
  Supports user messages, agent messages, system messages, and agent questions.
  Ordered by timestamp for chronological history retrieval.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CHAT_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    run_id          TEXT,
    role            TEXT NOT NULL,           -- 'user', 'agent', 'system'
    agent_id        TEXT,
    agent_type      TEXT,
    user_id         TEXT,
    username        TEXT,
    content         TEXT NOT NULL,
    timestamp       DOUBLE PRECISION NOT NULL,
    -- Agent-to-user questions
    request_id      TEXT,                    -- for routing user replies back
    response_options JSONB,                  -- multiple-choice options
    awaiting_reply  BOOLEAN DEFAULT FALSE,
    -- Run lifecycle
    run_status      TEXT,                    -- 'submitted', 'running', 'completed', 'failed'
    -- Controls sent with the message (JSON blob)
    controls        JSONB
);
"""

CHAT_MESSAGES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, timestamp);
"""


async def ensure_chat_schema(db_pool) -> None:
    """Create chat tables if they don't exist."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(CHAT_MESSAGES_TABLE_SQL)
            await conn.execute(CHAT_MESSAGES_INDEX_SQL)
        logger.info("Chat schema ensured")
    except Exception as e:
        logger.error("Failed to ensure chat schema: %s", e)
