"""Session chat protocol for user <-> session agent communication.

Defines the blackboard key format for chat messages within a session.
Operates at SESSION scope — all participants in a session share
the same blackboard partition.

Key formats:

- ``chat:user:{message_id}`` — User message to session agent
- ``chat:agent:{agent_id}:{message_id}`` — Agent message to user
- ``chat:reply:{request_id}:{message_id}`` — User reply to agent question
- ``chat:event:{run_id}:{event_name}`` — Run lifecycle event

Usage::

    from polymathera.colony.web_ui.backend.routers.chat import SessionChatProtocol

    # User sends a message
    key = SessionChatProtocol.user_message_key("msg_abc")
    await blackboard.write(key, {"content": "analyze the auth module", ...})

    # Agent sends a response
    key = SessionChatProtocol.agent_message_key("agent_xyz", "msg_def")
    await blackboard.write(key, {"content": "Starting analysis...", ...})

    # Agent asks a question (user reply expected)
    key = SessionChatProtocol.agent_message_key("agent_xyz", "msg_ghi")
    await blackboard.write(key, {
        "content": "Which auth strategy should I use?",
        "request_id": "req_123",
        "response_options": ["JWT", "Session cookies"],
        "awaiting_reply": True,
    })

    # User replies to agent question
    key = SessionChatProtocol.reply_key("req_123", "msg_jkl")
    await blackboard.write(key, {"content": "JWT", ...})

    # Session agent subscribes to all user messages
    pattern = SessionChatProtocol.user_message_pattern()
    # -> "chat:user:*"
"""

from __future__ import annotations

from typing import ClassVar

from polymathera.colony.agents.blackboard.protocol import BlackboardProtocol
from polymathera.colony.agents.scopes import BlackboardScope


class SessionChatProtocol(BlackboardProtocol):
    """Protocol for session chat — user <-> session agent <-> child agents.

    Operates at session scope. Multiple agents post to the same partition
    so agent_id is included in agent message keys.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.SESSION

    # --- Key construction ---

    @staticmethod
    def user_message_key(message_id: str) -> str:
        """Key for a user message to the session agent."""
        return f"chat:user:{message_id}"

    @staticmethod
    def agent_message_key(agent_id: str, message_id: str) -> str:
        """Key for an agent message to the user."""
        return f"chat:agent:{agent_id}:{message_id}"

    @staticmethod
    def reply_key(request_id: str, message_id: str) -> str:
        """Key for a user reply to an agent question."""
        return f"chat:reply:{request_id}:{message_id}"

    @staticmethod
    def event_key(run_id: str, event_name: str) -> str:
        """Key for a run lifecycle event (spawning, progress, completed, etc.)."""
        return f"chat:event:{run_id}:{event_name}"

    @staticmethod
    def control_message_key(message_id: str) -> str:
        """Key for a high-priority control command from the user.

        Slash commands like ``/status``, ``/whatdoing``, ``/help``,
        ``/abort``, ``/cancel`` get classified by the chat router and
        written to this key shape instead of ``chat:user:*``. The
        session orchestrator subscribes with
        ``@event_handler(pattern=control_message_pattern(), priority="high")``
        so the command is processed on the policy's concurrent
        high-priority lane and is NOT blocked by long-running actions
        in the main planning loop.
        """
        return f"chat:control:{message_id}"

    @staticmethod
    def action_status_key(agent_id: str, action_id: str) -> str:
        """Key for a per-action status update.

        Written by ``CodeGenerationActionPolicy`` whenever it dispatches
        an action (twice: once on entry with ``status="running"``, once on
        exit with ``status="complete"`` or ``"failed"``). The chat
        WebSocket relays these to the frontend so the user sees a small
        "currently running …" badge while a long-running action is in
        flight.
        """
        return f"chat:action_status:{agent_id}:{action_id}"

    # --- Pattern construction ---

    @staticmethod
    def user_message_pattern() -> str:
        """Pattern matching all user messages."""
        return "chat:user:*"

    @staticmethod
    def control_message_pattern() -> str:
        """Pattern matching all high-priority control commands."""
        return "chat:control:*"

    @staticmethod
    def agent_message_pattern() -> str:
        """Pattern matching all agent messages."""
        return "chat:agent:*"

    @staticmethod
    def reply_pattern(request_id: str | None = None) -> str:
        """Pattern matching user replies, optionally for a specific request."""
        if request_id:
            return f"chat:reply:{request_id}:*"
        return "chat:reply:*"

    @staticmethod
    def event_pattern(run_id: str | None = None) -> str:
        """Pattern matching run events, optionally for a specific run."""
        if run_id:
            return f"chat:event:{run_id}:*"
        return "chat:event:*"

    @staticmethod
    def action_status_pattern(agent_id: str | None = None) -> str:
        """Pattern matching action-status updates, optionally for one
        agent."""
        if agent_id:
            return f"chat:action_status:{agent_id}:*"
        return "chat:action_status:*"

    @staticmethod
    def all_chat_pattern() -> str:
        """Pattern matching all chat activity (messages, replies, events)."""
        return "chat:*"

    # --- Key parsing ---

    @staticmethod
    def parse_user_message_key(key: str) -> str:
        """Extract message_id from a user message key.

        Args:
            key: Key like ``"chat:user:msg_abc"``

        Returns:
            The message_id.
        """
        prefix = "chat:user:"
        if not key.startswith(prefix):
            raise ValueError(f"Not a SessionChatProtocol user message key: {key!r}")
        return key[len(prefix):]

    @staticmethod
    def parse_agent_message_key(key: str) -> tuple[str, str]:
        """Extract (agent_id, message_id) from an agent message key.

        Args:
            key: Key like ``"chat:agent:agent_xyz:msg_def"``

        Returns:
            Tuple of (agent_id, message_id).
        """
        prefix = "chat:agent:"
        if not key.startswith(prefix):
            raise ValueError(f"Not a SessionChatProtocol agent message key: {key!r}")
        rest = key[len(prefix):]
        parts = rest.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed agent message key (expected agent_id:message_id): {key!r}")
        return parts[0], parts[1]

    @staticmethod
    def parse_reply_key(key: str) -> tuple[str, str]:
        """Extract (request_id, message_id) from a reply key.

        Args:
            key: Key like ``"chat:reply:req_123:msg_jkl"``

        Returns:
            Tuple of (request_id, message_id).
        """
        prefix = "chat:reply:"
        if not key.startswith(prefix):
            raise ValueError(f"Not a SessionChatProtocol reply key: {key!r}")
        rest = key[len(prefix):]
        parts = rest.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed reply key (expected request_id:message_id): {key!r}")
        return parts[0], parts[1]

    @staticmethod
    def parse_event_key(key: str) -> tuple[str, str]:
        """Extract (run_id, event_name) from an event key.

        Args:
            key: Key like ``"chat:event:run_abc:progress"``

        Returns:
            Tuple of (run_id, event_name).
        """
        prefix = "chat:event:"
        if not key.startswith(prefix):
            raise ValueError(f"Not a SessionChatProtocol event key: {key!r}")
        rest = key[len(prefix):]
        parts = rest.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed event key (expected run_id:event_name): {key!r}")
        return parts[0], parts[1]
