"""Consciousness streams — filtered views of an agent's experience.

A ``ConsciousnessStream`` captures a slice of what an agent experiences
(events received, actions taken) and renders it into the LLM planning
prompt. An agent's action policy can maintain multiple streams; each
stream decides independently what to capture and how to present it.

Why streams and not a single event history?
- A conversational agent's experience is user messages + agent responses
  rendered as a chat thread.
- An analysis coordinator's experience includes worker result events,
  game moves, synthesis actions — each with different presentation needs.
- A monitoring agent might stream telemetry events but no actions.

Each stream is fully defined by:

- **event filter**: which events (accumulated context from event handlers)
  should be recorded
- **action filter**: which action calls (from the action dispatcher's
  call trace) should be recorded
- **formatter**: how recorded entries render into a prompt section

All three are pluggable. Users of the library compose streams by binding
existing pieces or implementing their own.

Example — session agent's conversational stream::

    from polymathera.colony.agents.patterns.planning.streams import (
        ConsciousnessStream,
        ConversationFormatter,
        EventContextKeyFilter,
        ActionKeySubstringFilter,
    )

    conversation = ConsciousnessStream.bind(
        name="conversation",
        event_filter=EventContextKeyFilter("user_chat_message"),
        action_filter=ActionKeySubstringFilter("respond_to_user"),
        formatter=ConversationFormatter.bind(),
    )

    agent_metadata.action_policy_blueprints = {
        "consciousness_streams": [conversation],
    }
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from ...blueprint import Blueprint


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
# Filters are callables with specific signatures. Provided implementations
# are classes (picklable). Users can also pass top-level functions, or
# compose filters with AnyOf / AllOf.

class EventContextKeyFilter:
    """Event filter — accept events whose accumulated context contains any
    of the given ``context_key`` values.

    ``accumulated_context`` is a dict keyed by ``EventProcessingResult.context_key``.
    Matches if any of ``keys`` appears as a top-level key.
    """

    def __init__(self, *keys: str):
        self._keys = set(keys)

    def __call__(self, contexts: dict[str, Any]) -> bool:
        return any(k in contexts for k in self._keys)


class ActionKeySubstringFilter:
    """Action filter — accept action calls whose ``action_key`` contains
    any of the given substrings.

    Matches against the ``action_key`` field of a call-trace entry.
    """

    def __init__(self, *substrings: str):
        self._substrings = substrings

    def __call__(self, call: dict[str, Any]) -> bool:
        action_key = call.get("action_key", "")
        return any(s in action_key for s in self._substrings)


class SuccessfulActionFilter:
    """Action filter — wraps another filter and additionally requires
    ``call.success`` to be truthy. Use to exclude failed action calls.
    """

    def __init__(self, inner: Callable[[dict[str, Any]], bool]):
        self._inner = inner

    def __call__(self, call: dict[str, Any]) -> bool:
        return bool(call.get("success")) and self._inner(call)


# ---------------------------------------------------------------------------
# Formatter protocol
# ---------------------------------------------------------------------------

class ConsciousnessStreamFormatter(ABC):
    """Renders a stream's captured entries into a planning prompt section.

    Subclasses implement ``format`` to produce a markdown section.
    """

    @classmethod
    def bind(cls, **kwargs: Any) -> Blueprint:
        """Create a serializable blueprint for this formatter."""
        bp = Blueprint(cls, kwargs)
        bp.validate_serializable()
        return bp

    @abstractmethod
    def format(self, entries: list[dict[str, Any]]) -> str:
        """Render entries into a prompt section.

        Args:
            entries: List of entries captured by the stream. Each is a dict with:
                - ``kind``: ``"event"`` or ``"action"``
                - ``timestamp``: float
                - For ``event``: ``contexts`` (dict of context_key -> context dict)
                - For ``action``: ``call`` (dict with action_key, output_preview, ...)

        Returns:
            Formatted markdown string. Return empty to suppress the section.
        """
        ...


# ---------------------------------------------------------------------------
# ConsciousnessStream
# ---------------------------------------------------------------------------

class ConsciousnessStream:
    """A filtered, ordered record of events and action calls.

    The action policy feeds events and actions to every registered stream.
    Each stream decides independently what to keep (via its filters) and
    how to render it (via its formatter). A rolling window bounds memory.
    """

    @classmethod
    def bind(cls, **kwargs: Any) -> Blueprint:
        """Create a serializable blueprint for this stream."""
        bp = Blueprint(cls, kwargs)
        bp.validate_serializable()
        return bp

    def __init__(
        self,
        name: str,
        formatter: ConsciousnessStreamFormatter | Blueprint,
        event_filter: Callable[[dict[str, Any]], bool] | None = None,
        action_filter: Callable[[dict[str, Any]], bool] | None = None,
        max_entries: int = 20,
    ):
        """
        Args:
            name: Unique identifier for this stream. Used for logging.
            formatter: Renders captured entries into a prompt section. May be
                a Blueprint — resolved here so bind()-constructed streams can
                nest a formatter blueprint without manual unwrapping upstream.
            event_filter: Predicate on accumulated event context. Accepted
                events are recorded. ``None`` means accept no events.
            action_filter: Predicate on action-dispatcher call trace entries.
                Accepted actions are recorded. ``None`` means accept no actions.
            max_entries: Rolling window size before old entries are dropped.
        """
        self.name = name
        self.formatter = formatter.local_instance() if isinstance(formatter, Blueprint) else formatter
        self._event_filter = event_filter
        self._action_filter = action_filter
        self._max_entries = max_entries
        self._entries: list[dict[str, Any]] = []

    def consider_event(self, contexts: dict[str, Any]) -> None:
        """Invoked by the action policy after event handlers run.

        If the event filter accepts the accumulated contexts, an entry
        is recorded.
        """
        if not contexts or self._event_filter is None:
            return
        if self._event_filter(contexts):
            self._append({
                "kind": "event",
                "timestamp": time.time(),
                "contexts": contexts,
            })

    def consider_action(self, call: dict[str, Any]) -> None:
        """Invoked by the action policy after each action call.

        If the action filter accepts the call, an entry is recorded.
        """
        if self._action_filter is None:
            return
        if self._action_filter(call):
            self._append({
                "kind": "action",
                "timestamp": time.time(),
                "call": call,
            })

    def render(self) -> str:
        """Render this stream's entries into a prompt section."""
        return self.formatter.format(list(self._entries))

    def _append(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]


# ---------------------------------------------------------------------------
# Stock formatters
# ---------------------------------------------------------------------------

class ConversationFormatter(ConsciousnessStreamFormatter):
    """Renders a stream as a chat conversation: user messages + agent responses.

    Event entries with the configured ``user_context_key`` render as
    ``**User**: <message>``. Action entries render as
    ``**You (Agent)**: <message>``, preferring the action's input parameter
    named ``agent_content_field`` (e.g., ``respond_to_user(content=...)``)
    over the action's output — agent-reply actions typically return only a
    receipt (message_id, timestamp) while the actual prose is in the call's
    arguments.
    """

    def __init__(
        self,
        user_context_key: str = "user_chat_message",
        user_content_field: str = "user_message",
        agent_content_field: str = "content",
        user_label: str = "User",
        agent_label: str = "You (Agent)",
        section_title: str = "## Conversation",
        max_message_chars: int = 500,
    ):
        self._user_context_key = user_context_key
        self._user_content_field = user_content_field
        self._agent_content_field = agent_content_field
        self._user_label = user_label
        self._agent_label = agent_label
        self._section_title = section_title
        self._max_message_chars = max_message_chars

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_message_chars:
            return text[:self._max_message_chars] + "..."
        return text

    def format(self, entries: list[dict[str, Any]]) -> str:
        if not entries:
            return ""

        lines = [self._section_title, ""]
        for entry in entries:
            if entry["kind"] == "event":
                ctx = entry["contexts"].get(self._user_context_key)
                if isinstance(ctx, dict):
                    message = ctx.get(self._user_content_field, "")
                    lines.append(f"**{self._user_label}**: {self._truncate(message)}")
            elif entry["kind"] == "action":
                call = entry["call"]
                params = call.get("parameters") or {}
                message = params.get(self._agent_content_field)
                if not message:
                    # Fall back to the action's output (may be a receipt dict)
                    message = call.get("output_preview", "")
                lines.append(f"**{self._agent_label}**: {self._truncate(str(message))}")
        return "\n".join(lines)


class JSONStreamFormatter(ConsciousnessStreamFormatter):
    """Renders a stream as a flat JSON-ish bullet list. Good default when
    no domain-specific formatter is provided.
    """

    def __init__(
        self,
        section_title: str,
        max_value_chars: int = 200,
    ):
        self._section_title = section_title
        self._max_value_chars = max_value_chars

    def format(self, entries: list[dict[str, Any]]) -> str:
        if not entries:
            return ""

        lines = [self._section_title, ""]
        for entry in entries:
            if entry["kind"] == "event":
                for key, ctx in entry["contexts"].items():
                    value_str = json.dumps(ctx, default=str) if isinstance(ctx, dict) else str(ctx)
                    if len(value_str) > self._max_value_chars:
                        value_str = value_str[:self._max_value_chars] + "..."
                    lines.append(f"- **event** [{key}]: {value_str}")
            elif entry["kind"] == "action":
                call = entry["call"]
                action_key = call.get("action_key", "?")
                output = call.get("output_preview", "")
                if len(output) > self._max_value_chars:
                    output = output[:self._max_value_chars] + "..."
                lines.append(f"- **action** [{action_key}]: {output}")
        return "\n".join(lines)
