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

    # Canonical entry kinds the stream knows how to record. New
    # kinds land alongside their ``consider_*`` method below; the
    # set is also used by ``record_kind`` to validate.
    KINDS: tuple[str, ...] = (
        "event",
        "action",
        "tool_output",
        "vcm_update",
        "monorepo_commit",
        "domain_state",
    )

    def __init__(
        self,
        name: str,
        formatter: ConsciousnessStreamFormatter | Blueprint,
        event_filter: Callable[[dict[str, Any]], bool] | None = None,
        action_filter: Callable[[dict[str, Any]], bool] | None = None,
        tool_output_filter: Callable[[dict[str, Any]], bool] | None = None,
        vcm_update_filter: Callable[[dict[str, Any]], bool] | None = None,
        monorepo_commit_filter: Callable[[dict[str, Any]], bool] | None = None,
        domain_state_filter: Callable[[dict[str, Any]], bool] | None = None,
        filters: dict[str, Callable[[dict[str, Any]], bool]] | None = None,
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
            tool_output_filter: Predicate on a ``ToolResult``-bearing dict
                produced when an action's return value is a ``ToolResult``.
                ``None`` means accept no tool outputs.
            vcm_update_filter: Predicate on a VCM page-graph mutation
                payload (page added / evicted / refreshed). ``None`` =
                accept none.
            monorepo_commit_filter: Predicate on a design-monorepo commit
                payload (sha + branch + commit msg + path summary).
                ``None`` = accept none.
            domain_state_filter: Predicate on a domain state-machine
                transition payload (``state_machine_name`` / ``transition`` /
                ``payload``). ``None`` = accept none.
            filters: Unified per-kind filter dict. Overrides any
                ``<kind>_filter`` kwarg with the same key. Use for
                programmatic configuration (e.g. when subclassing a
                shared stream helper that wants to selectively override
                one kind without re-specifying the others).
            max_entries: Rolling window size before old entries are dropped.
        """
        self.name = name
        self.formatter = formatter.local_instance() if isinstance(formatter, Blueprint) else formatter
        # Per-kind filter map — the source of truth. Per-kind kwargs
        # above are sugar that lifts into this dict; ``filters=`` takes
        # precedence per-key when both are supplied.
        kwarg_filters: dict[str, Callable[[dict[str, Any]], bool]] = {}
        for kind, fn in (
            ("event", event_filter),
            ("action", action_filter),
            ("tool_output", tool_output_filter),
            ("vcm_update", vcm_update_filter),
            ("monorepo_commit", monorepo_commit_filter),
            ("domain_state", domain_state_filter),
        ):
            if fn is not None:
                kwarg_filters[kind] = fn
        self._filters: dict[str, Callable[[dict[str, Any]], bool]] = {
            **kwarg_filters, **(filters or {}),
        }
        self._max_entries = max_entries
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Backward-compat properties: existing code reads ``_event_filter``
    # / ``_action_filter`` directly (tests + a few capabilities). Keep
    # them readable so the migration to per-kind filters is non-breaking.
    # ------------------------------------------------------------------

    @property
    def _event_filter(self) -> Callable[[dict[str, Any]], bool] | None:
        return self._filters.get("event")

    @property
    def _action_filter(self) -> Callable[[dict[str, Any]], bool] | None:
        return self._filters.get("action")

    def consider_event(self, contexts: dict[str, Any]) -> None:
        """Invoked by the action policy after event handlers run.

        If the event filter accepts the accumulated contexts, an entry
        is recorded.
        """
        if not contexts:
            return
        f = self._filters.get("event")
        if f is None:
            return
        if f(contexts):
            self._append({
                "kind": "event",
                "timestamp": time.time(),
                "contexts": contexts,
            })

    def consider_action(self, call: dict[str, Any]) -> None:
        """Invoked by the action policy after each action call.

        If the action filter accepts the call, an entry is recorded.
        """
        f = self._filters.get("action")
        if f is None:
            return
        if f(call):
            self._append({
                "kind": "action",
                "timestamp": time.time(),
                "call": call,
            })

    def consider_tool_output(self, payload: dict[str, Any]) -> None:
        """Invoked by the action dispatcher's post-action path when an
        action's return value is a typed :class:`ToolResult`.

        Payload shape (built by
        :class:`ToolResultSource`): ``{"action_key": str,
        "tool_result": dict_form_of_ToolResult, "success": bool,
        "agent_id": str}``.
        """
        f = self._filters.get("tool_output")
        if f is None:
            return
        if f(payload):
            self._append({
                "kind": "tool_output",
                "timestamp": time.time(),
                "payload": payload,
            })

    def consider_vcm_update(self, payload: dict[str, Any]) -> None:
        """Invoked by :class:`VCMPageEventSource` on a VCM page-graph
        mutation visible to the agent's bound scope.

        Payload shape: ``{"kind": "added"|"evicted", "page_source": str,
        "scope_id": str, "page_id": str}``. The source subscribes
        ``VCMPageEventProtocol`` on the colony scope, so mutations from
        any VCM replica reach the agent regardless of process.
        """
        f = self._filters.get("vcm_update")
        if f is None:
            return
        if f(payload):
            self._append({
                "kind": "vcm_update",
                "timestamp": time.time(),
                "payload": payload,
            })

    def consider_monorepo_commit(self, payload: dict[str, Any]) -> None:
        """Invoked by :class:`MonorepoCommitEventSource` after a tier-2
        ``BranchScopedCapabilityBase`` commit succeeds on a shared
        ``(scope, branch)``.

        Payload shape: ``{"sha": str, "branch": str, "message": str,
        "paths": list[str], "capability_fqn": str}``. The source
        subscribes ``MonorepoCommitProtocol`` on the colony scope, so a
        peer agent's commit reaches this agent even from another
        process / replica.
        """
        f = self._filters.get("monorepo_commit")
        if f is None:
            return
        if f(payload):
            self._append({
                "kind": "monorepo_commit",
                "timestamp": time.time(),
                "payload": payload,
            })

    def consider_domain_state(self, payload: dict[str, Any]) -> None:
        """Invoked by capability-specific adapters (hypothesis-game
        phase, experiment lifecycle, mission progress, budget violations)
        on a state-machine transition.

        Payload shape: ``{"state_machine": str, "transition": str,
        "from_state": str, "to_state": str, "data": dict}``. Adapters
        per state machine land alongside their respective
        per-capability rollout PRs.
        """
        f = self._filters.get("domain_state")
        if f is None:
            return
        if f(payload):
            self._append({
                "kind": "domain_state",
                "timestamp": time.time(),
                "payload": payload,
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


def _truncate(text: str, max_chars: int) -> str:
    """Shared helper — truncate ``text`` to ``max_chars`` with an
    ellipsis suffix when it overflows."""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


class EventLogFormatter(ConsciousnessStreamFormatter):
    """Chronological generic formatter — renders entries of any kind
    as a single bullet list in insertion order.

    Use as the default when a stream feeds multiple kinds and the
    rendering doesn't need to differentiate them. Each line shows the
    entry's ``kind`` + a short payload summary (uses
    ``json.dumps(default=str)`` so non-JSON-clean values still render).

    Parameters
    ----------
    section_title : str
        Markdown header for the rendered section.
    max_value_chars : int
        Per-line value truncation budget.
    max_entries_shown : int | None
        Cap on entries rendered (most recent kept). ``None`` (the
        default) renders all entries the stream's rolling window
        retains.
    """

    def __init__(
        self,
        section_title: str,
        max_value_chars: int = 200,
        max_entries_shown: int | None = None,
    ):
        self._section_title = section_title
        self._max_value_chars = max_value_chars
        self._max_entries_shown = max_entries_shown

    def format(self, entries: list[dict[str, Any]]) -> str:
        if not entries:
            return ""
        rendered = entries
        if (
            self._max_entries_shown is not None
            and len(rendered) > self._max_entries_shown
        ):
            rendered = rendered[-self._max_entries_shown:]
        lines: list[str] = [self._section_title, ""]
        for entry in rendered:
            kind = entry.get("kind", "?")
            # Pick the "interesting" payload field per kind. Falls back
            # to ``json.dumps`` of the whole entry for unknown kinds.
            if kind == "event":
                body = entry.get("contexts", {})
            elif kind == "action":
                body = entry.get("call", {})
            else:
                body = entry.get("payload", entry)
            value_str = json.dumps(body, default=str, ensure_ascii=False)
            lines.append(f"- **{kind}**: {_truncate(value_str, self._max_value_chars)}")
        return "\n".join(lines)


class ToolResultFormatter(ConsciousnessStreamFormatter):
    """Formats ``tool_output`` entries by extracting the
    :class:`ToolResult`-shape fields the agent's planner cares about
    (payload + units + provenance.tool_name).

    Non-``tool_output`` entries are silently skipped — pair this
    formatter with a stream that filters to ``tool_output`` only, or
    let the skip be a no-op for mixed streams.
    """

    def __init__(
        self,
        section_title: str = "## Recent tool outputs",
        max_payload_chars: int = 300,
    ):
        self._section_title = section_title
        self._max_payload_chars = max_payload_chars

    def format(self, entries: list[dict[str, Any]]) -> str:
        tool_entries = [e for e in entries if e.get("kind") == "tool_output"]
        if not tool_entries:
            return ""
        lines: list[str] = [self._section_title, ""]
        for entry in tool_entries:
            payload = entry.get("payload", {})
            action_key = payload.get("action_key", "?")
            tool_result = payload.get("tool_result", {}) or {}
            provenance = tool_result.get("provenance", {}) or {}
            tool_name = (
                provenance.get("tool_name")
                or provenance.get("capability_fqn")
                or "?"
            )
            success_marker = "✓" if payload.get("success") else "✗"
            tr_payload = tool_result.get("payload", {})
            units = tool_result.get("units", {})
            payload_str = json.dumps(tr_payload, default=str, ensure_ascii=False)
            units_str = (
                json.dumps(units, default=str, ensure_ascii=False)
                if units else ""
            )
            line = (
                f"- {success_marker} **{action_key}** (tool: `{tool_name}`): "
                f"{_truncate(payload_str, self._max_payload_chars)}"
            )
            if units_str and units_str != "{}":
                line += f" — units: {units_str}"
            lines.append(line)
        return "\n".join(lines)


class VCMUpdateFormatter(ConsciousnessStreamFormatter):
    """Formats ``vcm_update`` entries — VCM page-graph mutations
    visible to the agent's bound scope.

    Each line shows the mutation kind (``added`` / ``evicted`` /
    ``refreshed``) + the page source + the page identifier. Non-VCM
    entries skipped.
    """

    def __init__(
        self,
        section_title: str = "## Recent VCM page-graph updates",
        max_entries_shown: int | None = None,
    ):
        self._section_title = section_title
        self._max_entries_shown = max_entries_shown

    def format(self, entries: list[dict[str, Any]]) -> str:
        vcm_entries = [e for e in entries if e.get("kind") == "vcm_update"]
        if not vcm_entries:
            return ""
        if (
            self._max_entries_shown is not None
            and len(vcm_entries) > self._max_entries_shown
        ):
            vcm_entries = vcm_entries[-self._max_entries_shown:]
        lines: list[str] = [self._section_title, ""]
        for entry in vcm_entries:
            payload = entry.get("payload", {})
            mutation_kind = payload.get("kind", "?")
            page_source = payload.get("page_source", "?")
            page_id = payload.get("page_id", "?")
            scope_id = payload.get("scope_id", "")
            scope_suffix = f" [scope={scope_id}]" if scope_id else ""
            lines.append(
                f"- **{mutation_kind}** page `{page_id}` "
                f"(source: `{page_source}`){scope_suffix}"
            )
        return "\n".join(lines)


class MonorepoCommitFormatter(ConsciousnessStreamFormatter):
    """Formats ``monorepo_commit`` entries — tier-2 commits the
    agent's ``BranchScopedCapabilityBase`` (or a peer's in the same
    ``(scope, branch)``) just landed on the design monorepo.

    Each line shows the short SHA, the branch, the commit-message
    prefix (the ``L2 / G-1 / G-2 / F-3 ...`` convention), and a path
    summary truncated to the configured budget. Non-commit entries
    skipped.
    """

    def __init__(
        self,
        section_title: str = "## Recent design-monorepo commits",
        max_message_chars: int = 120,
        max_paths_shown: int = 5,
    ):
        self._section_title = section_title
        self._max_message_chars = max_message_chars
        self._max_paths_shown = max_paths_shown

    def format(self, entries: list[dict[str, Any]]) -> str:
        commit_entries = [
            e for e in entries if e.get("kind") == "monorepo_commit"
        ]
        if not commit_entries:
            return ""
        lines: list[str] = [self._section_title, ""]
        for entry in commit_entries:
            payload = entry.get("payload", {})
            sha = (payload.get("sha") or "")[:8] or "?"
            branch = payload.get("branch", "?")
            message = _truncate(
                payload.get("message", ""), self._max_message_chars,
            )
            paths = payload.get("paths", []) or []
            paths_str = ", ".join(paths[:self._max_paths_shown])
            if len(paths) > self._max_paths_shown:
                paths_str += f" (+{len(paths) - self._max_paths_shown} more)"
            line = f"- `{sha}` on `{branch}`: {message}"
            if paths_str:
                line += f"\n    paths: {paths_str}"
            lines.append(line)
        return "\n".join(lines)


class DomainStateFormatter(ConsciousnessStreamFormatter):
    """Formats ``domain_state`` entries — state-machine transitions
    from one specific machine (hypothesis-game phase, experiment
    lifecycle, mission progress, budget violations).

    When ``state_machine_name`` is supplied, only transitions from
    that machine render; otherwise all domain-state transitions
    render (with the machine name on each line). Non-domain-state
    entries skipped.
    """

    def __init__(
        self,
        section_title: str,
        state_machine_name: str | None = None,
        max_entries_shown: int | None = None,
    ):
        self._section_title = section_title
        self._state_machine_name = state_machine_name
        self._max_entries_shown = max_entries_shown

    def format(self, entries: list[dict[str, Any]]) -> str:
        domain_entries = [
            e for e in entries if e.get("kind") == "domain_state"
        ]
        if self._state_machine_name is not None:
            domain_entries = [
                e for e in domain_entries
                if (e.get("payload") or {}).get("state_machine")
                == self._state_machine_name
            ]
        if not domain_entries:
            return ""
        if (
            self._max_entries_shown is not None
            and len(domain_entries) > self._max_entries_shown
        ):
            domain_entries = domain_entries[-self._max_entries_shown:]
        lines: list[str] = [self._section_title, ""]
        for entry in domain_entries:
            payload = entry.get("payload", {})
            transition = payload.get("transition", "?")
            from_state = payload.get("from_state", "")
            to_state = payload.get("to_state", "")
            machine_prefix = ""
            if self._state_machine_name is None:
                machine_prefix = (
                    f"[{payload.get('state_machine', '?')}] "
                )
            arrow = (
                f"`{from_state}` → `{to_state}`"
                if from_state and to_state
                else f"`{transition}`"
            )
            lines.append(f"- {machine_prefix}{arrow}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filter combinators
# ---------------------------------------------------------------------------


class AnyOf:
    """Filter combinator — accepts an entry iff ANY of the wrapped
    filters accepts it.

    Use to compose otherwise-orthogonal predicates: e.g. accept
    actions tagged ``"design"`` OR actions whose action_key starts
    with ``"plan_"``.
    """

    def __init__(self, *filters: Callable[[dict[str, Any]], bool]):
        if not filters:
            raise ValueError("AnyOf requires at least one filter")
        self._filters = filters

    def __call__(self, payload: dict[str, Any]) -> bool:
        return any(f(payload) for f in self._filters)


class AllOf:
    """Filter combinator — accepts an entry iff EVERY wrapped filter
    accepts it.

    Equivalent to chaining through ``SuccessfulActionFilter``-style
    wrappers but more readable for non-success-specific composition.
    """

    def __init__(self, *filters: Callable[[dict[str, Any]], bool]):
        if not filters:
            raise ValueError("AllOf requires at least one filter")
        self._filters = filters

    def __call__(self, payload: dict[str, Any]) -> bool:
        return all(f(payload) for f in self._filters)


class Not:
    """Filter combinator — accepts an entry iff the wrapped filter
    rejects it. Use to exclude a specific subset that an upstream
    filter would otherwise accept."""

    def __init__(self, inner: Callable[[dict[str, Any]], bool]):
        self._inner = inner

    def __call__(self, payload: dict[str, Any]) -> bool:
        return not self._inner(payload)


# ---------------------------------------------------------------------------
# Colony-side bind() helpers (Colony cannot import the per-domain helpers
# in ``polymathera.cps.agents.streams`` without inverting the architectural
# layering — so the bare-bones agent-experience helpers live here.)
# ---------------------------------------------------------------------------


def _accept_all_payload(_payload: dict[str, Any]) -> bool:
    return True


def colony_basic_stream(
    *,
    name: str = "agent_experience",
    section_title: str = "## Recent agent experience",
    max_entries: int = 30,
) -> "Blueprint":
    """Generic catch-all stream for any Colony agent that wants to
    surface its recent events + action calls + tool outputs +
    monorepo / VCM updates to the LLM planner.

    Pairs with the three universally-available sources
    (:class:`AccumulatedContextSource`, :class:`ActionCallSource`,
    :class:`ToolResultSource`) — set them up via
    ``policy.attach_source(...)`` from the agent's ``initialize``.

    Use as a starting point; agents that need role-specific
    rendering should compose targeted streams (e.g. CPS's
    :func:`polymathera.cps.agents.streams.design_reasoning_stream`).
    """
    return ConsciousnessStream.bind(
        name=name,
        formatter=EventLogFormatter.bind(
            section_title=section_title,
            max_entries_shown=max_entries,
        ),
        filters={
            "event": _accept_all_payload,
            "action": _accept_all_payload,
            "tool_output": _accept_all_payload,
            "vcm_update": _accept_all_payload,
            "monorepo_commit": _accept_all_payload,
            "domain_state": _accept_all_payload,
        },
        max_entries=max_entries,
    )


async def attach_colony_standard_sources(policy: Any) -> None:
    """Attach the three Colony-universal sources (Accumulated event
    context, action calls, tool outputs) to an agent's action policy
    + call ``attach_pending_sources``.

    Use from a sample / Colony-internal agent's ``initialize`` AFTER
    ``super().initialize()``. CPS agents use the richer
    :func:`polymathera.cps.agents.streams.attach_cps_standard_sources`
    helper which also wires the MonorepoCommit + (optionally) VCMPage
    sources.
    """
    # Late imports to avoid a planning ↔ patterns circular dependency.
    from .sources import (  # noqa: PLC0415
        AccumulatedContextSource,
        ActionCallSource,
        ToolResultSource,
    )
    policy.attach_source(AccumulatedContextSource())
    policy.attach_source(ActionCallSource())
    policy.attach_source(ToolResultSource())
    await policy.attach_pending_sources()
