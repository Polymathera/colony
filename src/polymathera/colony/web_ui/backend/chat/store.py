"""Chat message store — CRUD operations for persistent chat history."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChatMessageStore:
    """PostgreSQL-backed store for chat messages."""

    def __init__(self, db_pool):
        self._pool = db_pool

    async def save_message(self, message: dict[str, Any]) -> None:
        """Persist a chat message."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_messages (
                    id, session_id, run_id, role, agent_id, agent_type,
                    user_id, username, content, timestamp,
                    request_id, response_options, awaiting_reply,
                    run_status, controls
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (id) DO NOTHING
                """,
                message.get("id"),
                message.get("session_id"),
                message.get("run_id"),
                message.get("role"),
                message.get("agent_id"),
                message.get("agent_type"),
                message.get("user_id"),
                message.get("username"),
                message.get("content", ""),
                message.get("timestamp", 0.0),
                message.get("request_id"),
                json.dumps(message["response_options"]) if message.get("response_options") else None,
                message.get("awaiting_reply", False),
                message.get("run_status"),
                json.dumps(message["controls"]) if message.get("controls") else None,
            )

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get chat history for a session, ordered newest-first.

        Args:
            session_id: Session to fetch messages for.
            limit: Max messages to return.
            before_timestamp: Only return messages before this timestamp (for pagination).

        Returns:
            List of message dicts, newest first.
        """
        async with self._pool.acquire() as conn:
            if before_timestamp:
                rows = await conn.fetch(
                    """
                    SELECT * FROM chat_messages
                    WHERE session_id = $1 AND timestamp < $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                    """,
                    session_id, before_timestamp, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM chat_messages
                    WHERE session_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                    """,
                    session_id, limit,
                )

        # Convert to dicts and reverse to chronological order
        messages = []
        for row in reversed(rows):
            msg = dict(row)
            # Parse JSON fields
            if msg.get("response_options"):
                msg["response_options"] = json.loads(msg["response_options"])
            if msg.get("controls"):
                msg["controls"] = json.loads(msg["controls"])
            messages.append(msg)

        return messages

    async def mark_reply_received(self, request_id: str) -> None:
        """Mark an agent question as answered (awaiting_reply = False)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chat_messages SET awaiting_reply = FALSE WHERE request_id = $1",
                request_id,
            )
