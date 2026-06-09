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
    -- Short action name for typed approvals (e.g. "create_decomposition").
    -- Drives the 3-choice button labels (Reject / Approve once /
    -- Approve all <action_type> this session). NULL for untyped
    -- legacy approve/reject requests.
    action_type     TEXT,
    -- Run lifecycle
    run_status      TEXT,                    -- 'submitted', 'running', 'completed', 'failed'
    -- Controls sent with the message (JSON blob)
    controls        JSONB,
    -- Structured attachments emitted by ``respond_to_user`` /
    -- ``respond_to_user_with_table`` / ``respond_to_user_with_diff``. Each element is
    -- a typed dict (kind=code/table/diff/...) the chat UI renders
    -- alongside the markdown content. JSONB so future kinds plug in
    -- without a schema migration.
    attachments     JSONB
);
"""

CHAT_MESSAGES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, timestamp);
"""

# Idempotent ALTERs for existing deployments that pre-date a column.
# ``ADD COLUMN IF NOT EXISTS`` is a no-op when the column already
# exists; safe to run on every startup.
CHAT_MESSAGES_MIGRATIONS_SQL = (
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS attachments JSONB;",
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS action_type TEXT;",
)


async def ensure_chat_schema(db_pool) -> None:
    """Create chat tables if they don't exist."""
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(CHAT_MESSAGES_TABLE_SQL)
            await conn.execute(CHAT_MESSAGES_INDEX_SQL)
            for stmt in CHAT_MESSAGES_MIGRATIONS_SQL:
                await conn.execute(stmt)
        logger.info("Chat schema ensured")
    except Exception as e:
        logger.error("Failed to ensure chat schema: %s", e)
