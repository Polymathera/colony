"""Event history formatters for planning prompts.

Controls how event handler results (EventProcessingResult.context) appear
in the LLM planning prompt. Different agents need different presentations:

- Analysis coordinators see events as structured summaries
- Session agents see events as a conversation thread
- Game agents see events as game state transitions

Users of the library create custom formatters by subclassing
``EventHistoryFormatter`` and passing them via ``action_policy_config``.

Usage::

    from polymathera.colony.agents.patterns.planning.formatters import (
        ConversationFormatter,
    )

    agent_metadata = AgentMetadata(
        ...,
        action_policy_config={
            "event_history_formatter": ConversationFormatter(),
        },
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ...blueprint import Blueprint


class EventHistoryFormatter(ABC):
    """Base class for formatting event history into planning prompt sections.

    Subclass this to control how events appear in the LLM planning prompt.
    The formatter receives a list of event history entries (each containing
    an iteration number, timestamp, and the accumulated context dicts from
    all event handlers that fired in that iteration).
    """

    @classmethod
    def bind(cls, **kwargs: Any) -> Blueprint:
        """Create a serializable blueprint for this formatter.

        Blueprints store the class + kwargs and are serialized via cloudpickle
        across Ray boundaries. Call ``blueprint.local_instance()`` on the
        target node to construct the formatter.
        """
        bp = Blueprint(cls, kwargs)
        bp.validate_serializable()
        return bp

    @abstractmethod
    def format(self, event_history: list[dict[str, Any]]) -> str:
        """Format event history entries into a prompt section.

        Args:
            event_history: List of entries, each with:
                - ``iteration``: int — policy iteration number
                - ``timestamp``: float — wall clock time
                - ``contexts``: dict[str, dict] — accumulated handler contexts
                  keyed by ``EventProcessingResult.context_key``

        Returns:
            Formatted string to insert into the planning prompt.
            Return empty string to suppress the section entirely.
        """
        ...


class DefaultEventHistoryFormatter(EventHistoryFormatter):
    """Renders events as a compact bullet list.

    Suitable for analysis agents where events are status updates,
    worker results, or game state transitions.
    """

    def __init__(self, max_value_chars: int = 200):
        self._max_value_chars = max_value_chars

    def format(self, event_history: list[dict[str, Any]]) -> str:
        if not event_history:
            return ""

        lines = ["## Recent Events", ""]
        for entry in event_history:
            contexts = entry.get("contexts", {})
            for key, ctx in contexts.items():
                value_str = json.dumps(ctx, default=str) if isinstance(ctx, dict) else str(ctx)
                if len(value_str) > self._max_value_chars:
                    value_str = value_str[:self._max_value_chars] + "..."
                lines.append(f"- **{key}**: {value_str}")
        return "\n".join(lines)


class ConversationFormatter(EventHistoryFormatter):
    """Renders events as a chat conversation thread.

    Designed for session agents where events are user messages.
    Presents a clear back-and-forth that the LLM can reason about
    and respond to naturally.
    """

    def __init__(
        self,
        user_context_key: str = "user_chat_message",
        user_content_field: str = "user_message",
        max_message_chars: int = 500,
    ):
        """
        Args:
            user_context_key: The ``context_key`` used by the user message
                event handler (matches ``EventProcessingResult.context_key``).
            user_content_field: Field name within the context dict that
                contains the user's message text.
            max_message_chars: Maximum characters per message in the prompt.
        """
        self._user_context_key = user_context_key
        self._user_content_field = user_content_field
        self._max_message_chars = max_message_chars

    def format(self, event_history: list[dict[str, Any]]) -> str:
        if not event_history:
            return ""

        lines = ["## Conversation", ""]
        for entry in event_history:
            contexts = entry.get("contexts", {})
            for key, ctx in contexts.items():
                if key == self._user_context_key and isinstance(ctx, dict):
                    message = ctx.get(self._user_content_field, "")
                    if len(message) > self._max_message_chars:
                        message = message[:self._max_message_chars] + "..."
                    lines.append(f"**User**: {message}")
                else:
                    # Non-user events (agent responses, system events)
                    value_str = json.dumps(ctx, default=str) if isinstance(ctx, dict) else str(ctx)
                    if len(value_str) > self._max_message_chars:
                        value_str = value_str[:self._max_message_chars] + "..."
                    lines.append(f"**Event** ({key}): {value_str}")
        return "\n".join(lines)
