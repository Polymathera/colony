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
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from ...blueprint import Blueprint

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ....cluster.tokenization import TokenizerProtocol
    from .compaction import CompactionPolicy, SpillArchive, StreamCompactor
    from .stream_log import StreamLogStore


logger = logging.getLogger(__name__)


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
        compaction_budget_tokens: int | None = None,
        compaction_keep_recent: int = 12,
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
                Only used in *legacy* mode (``compaction_budget_tokens``
                is ``None``). With compaction enabled, the durable log
                bounds the prompt instead and entries are never dropped.
            compaction_budget_tokens: Enable compaction/spillover. When
                set, the rendered view is kept under this many tokens by
                compacting + spilling the oldest entries (the durable
                log preserves originals losslessly). ``None`` (default)
                keeps the legacy rolling-window behavior.
            compaction_keep_recent: Number of most-recent raw entries the
                auto-compaction policy never compacts (kept verbatim in
                the view). Only meaningful when compaction is enabled.
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

        # --- Compaction / spillover (opt-in via compaction_budget_tokens) ---
        # When enabled, entries carry a monotonic ``seq``, are never
        # dropped from memory by ``_append`` (the durable log bounds the
        # prompt instead), and ``_entries`` becomes the hot *view* (raw
        # entries + synthesized ``compaction_summary`` stand-ins) rather
        # than the storage. Runtime collaborators are injected by the
        # owning policy via :meth:`bind_log` (they need the live agent),
        # so they stay out of the serialized blueprint.
        self._compaction_budget_tokens = compaction_budget_tokens
        self._compaction_keep_recent = compaction_keep_recent
        self._compaction_enabled = compaction_budget_tokens is not None
        self._next_seq = 0
        self._last_flushed_seq = -1
        self._compactions: list[Any] = []        # list[CompactionDescriptor]
        self._store: "StreamLogStore | None" = None
        self._compactor: "StreamCompactor | None" = None
        self._archive: "SpillArchive | None" = None
        self._policy: "CompactionPolicy | None" = None
        self._estimator: "TokenizerProtocol | None" = None

    _MAX_COMPACT_PASSES = 4
    """Safety cap on compactions per :meth:`maintain` call — bounds the
    LLM-call count per iteration even if a stream is wildly over budget.
    ``KeepRecentCompactionPolicy`` converges in one pass; the cap guards
    against a pathological policy."""

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
        """Render this stream's entries into a prompt section.

        In compaction mode the hot view mixes raw entries with
        synthesized ``compaction_summary`` stand-ins, so it is rendered
        in logical order (summaries sort by the span they cover; raw
        entries by their seq) — keeping correct interleaving even after
        an arbitrary :meth:`expand_span`.
        """
        entries = (
            sorted(self._entries, key=self._view_sort_key)
            if self._compaction_enabled
            else list(self._entries)
        )
        return self.formatter.format(entries)

    def _append(self, entry: dict[str, Any]) -> None:
        if self._compaction_enabled:
            # Stamp a monotonic seq; never drop (the durable log bounds
            # the prompt via maintain(), losslessly).
            entry["seq"] = self._next_seq
            self._next_seq += 1
            self._entries.append(entry)
            return
        # Legacy rolling window — unchanged behavior.
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    @staticmethod
    def _view_sort_key(entry: dict[str, Any]) -> int:
        if entry.get("kind") == "compaction_summary":
            covers = entry.get("payload", {}).get("covers") or [0, 0]
            return int(covers[0])
        return int(entry.get("seq", 0))

    @property
    def compaction_enabled(self) -> bool:
        """Whether this stream runs in compaction/spillover mode (i.e.
        ``compaction_budget_tokens`` was set). Public so the owning
        policy can decide whether to wire a durable log without reaching
        into stream internals."""
        return self._compaction_enabled

    @property
    def compaction_keep_recent(self) -> int:
        """Number of most-recent raw entries the auto-compaction policy
        keeps verbatim. Public read accessor for the policy's wiring."""
        return self._compaction_keep_recent

    # ------------------------------------------------------------------
    # Compaction / spillover lifecycle (no-ops unless compaction enabled
    # AND a log is bound). All durable I/O happens here, at async policy
    # boundaries — never inside the sync render() / consider_* path.
    # ------------------------------------------------------------------

    async def bind_log(
        self,
        *,
        store: "StreamLogStore",
        compactor: "StreamCompactor",
        archive: "SpillArchive",
        policy: "CompactionPolicy",
        estimator: "TokenizerProtocol",
    ) -> None:
        """Inject the runtime collaborators (built by the owning policy
        from the live agent). Idempotent; call before :meth:`rehydrate`."""
        self._store = store
        self._compactor = compactor
        self._archive = archive
        self._policy = policy
        self._estimator = estimator

    async def rehydrate(self) -> None:
        """Restore the view from the durable log (suspend/resume +
        restart). Loads every active summary plus a bounded tail of the
        most-recent uncovered raw entries; older uncovered entries stay
        in the log (retrievable via :meth:`expand_span`)."""
        if self._store is None:
            return
        index = await self._store.read_index()
        self._next_seq = index.next_seq
        self._last_flushed_seq = index.next_seq - 1
        self._compactions = list(index.compactions)
        summaries = [self._summary_entry(c) for c in self._compactions]
        # Uncovered raw seqs = [0, next_seq) minus the covered ranges.
        uncovered = self._complement(
            index.covered_ranges(), 0, index.next_seq - 1,
        )
        # Load only a bounded recent tail to keep rehydration cheap.
        tail_budget = max(self._compaction_keep_recent * 2, self._compaction_keep_recent)
        raw: list[dict[str, Any]] = []
        for start, end in reversed(uncovered):
            span = await self._store.read_span(start, end)
            raw = span + raw
            if len(raw) >= tail_budget:
                raw = raw[-tail_budget:]
                break
        self._entries = sorted(raw + summaries, key=self._view_sort_key)

    async def flush(self) -> None:
        """Persist newly-recorded raw entries to the durable log + sync
        the index. No-op unless compaction is enabled with a bound log."""
        if self._store is None or not self._compaction_enabled:
            return
        new = sorted(
            (
                e for e in self._entries
                if e.get("kind") != "compaction_summary"
                and int(e.get("seq", -1)) > self._last_flushed_seq
            ),
            key=lambda e: int(e["seq"]),
        )
        for entry in new:
            await self._store.append(int(entry["seq"]), entry)
            self._last_flushed_seq = int(entry["seq"])
        if new:
            await self._store.write_index(self._index())

    async def maintain(self) -> None:
        """Auto safety-net: while the rendered view exceeds the token
        budget, compact the oldest span the policy selects. Bounded by
        :attr:`_MAX_COMPACT_PASSES`. No-op unless fully wired."""
        if not (
            self._compaction_enabled
            and self._store is not None
            and self._compactor is not None
            and self._policy is not None
            and self._estimator is not None
            and self._compaction_budget_tokens is not None
        ):
            return
        for _ in range(self._MAX_COMPACT_PASSES):
            if self._estimator.count_tokens(self.render()) <= self._compaction_budget_tokens:
                return
            raw_window = sorted(
                (e for e in self._entries if e.get("kind") != "compaction_summary"),
                key=lambda e: int(e.get("seq", 0)),
            )
            span = self._policy.select_span(raw_window=raw_window)
            if span is None:
                return
            start, end = span
            victims = [
                e for e in raw_window if start <= int(e.get("seq", -1)) <= end
            ]
            if not victims:
                return
            await self._do_compact(start, end, victims, produced_by="auto")

    async def compact_span(
        self, start_seq: int, end_seq: int, *, produced_by: str = "agent",
    ) -> dict[str, Any] | None:
        """Agent-driven compaction of an explicit raw span currently in
        the view. Returns the resulting descriptor as a dict, or ``None``
        if the span holds no raw entries."""
        if self._store is None or self._compactor is None:
            return None
        victims = sorted(
            (
                e for e in self._entries
                if e.get("kind") != "compaction_summary"
                and start_seq <= int(e.get("seq", -1)) <= end_seq
            ),
            key=lambda e: int(e["seq"]),
        )
        if not victims:
            return None
        desc = await self._do_compact(
            int(victims[0]["seq"]), int(victims[-1]["seq"]), victims,
            produced_by=produced_by,
        )
        return desc.to_dict()

    async def compact_now(self, *, keep_recent: int | None = None) -> dict[str, Any] | None:
        """Agent-driven: compact the oldest raw entries beyond
        ``keep_recent`` (default: the stream's configured value) right
        now, independent of the token budget. Returns the descriptor or
        ``None`` if there's nothing eligible to compact."""
        if self._store is None or self._compactor is None:
            return None
        from .compaction import KeepRecentCompactionPolicy
        keep = self._compaction_keep_recent if keep_recent is None else keep_recent
        raw_window = sorted(
            (e for e in self._entries if e.get("kind") != "compaction_summary"),
            key=lambda e: int(e.get("seq", 0)),
        )
        span = KeepRecentCompactionPolicy(keep_recent=keep).select_span(
            raw_window=raw_window,
        )
        if span is None:
            return None
        start, end = span
        victims = [e for e in raw_window if start <= int(e.get("seq", -1)) <= end]
        if not victims:
            return None
        desc = await self._do_compact(start, end, victims, produced_by="agent")
        return desc.to_dict()

    async def expand_span(
        self, start_seq: int, end_seq: int, *, reattach_to_context: bool = False,
    ) -> dict[str, Any]:
        """Reverse compaction: pull the originals of every compacted span
        intersecting ``[start_seq, end_seq]`` back into the view from the
        durable log. With ``reattach_to_context``, also page the span back
        into the agent's real LLM context window via the spill archive."""
        if self._store is None:
            return {"expanded": 0, "reattached_pages": []}
        matched = [
            c for c in self._compactions
            if not (c.end_seq < start_seq or c.start_seq > end_seq)
        ]
        expanded = 0
        reattached: list[str] = []
        for desc in matched:
            originals = await self._store.read_span(desc.start_seq, desc.end_seq)
            self._entries = [
                e for e in self._entries
                if not (
                    e.get("kind") == "compaction_summary"
                    and (e.get("payload", {}).get("covers") or [None, None])
                    == [desc.start_seq, desc.end_seq]
                )
            ]
            self._entries.extend(originals)
            self._compactions.remove(desc)
            expanded += len(originals)
            if reattach_to_context and self._archive is not None:
                reattached += await self._archive.reattach(
                    stream_name=self.name,
                    start_seq=desc.start_seq,
                    end_seq=desc.end_seq,
                    entries=originals,
                )
        if matched:
            self._entries.sort(key=self._view_sort_key)
            await self._store.write_index(self._index())
        return {"expanded": expanded, "reattached_pages": reattached}

    def history_summary(self) -> list[dict[str, Any]]:
        """Planner-facing introspection: the active compacted spans (so
        the agent knows what it can expand) + the current view extent."""
        return [
            {
                "start_seq": c.start_seq,
                "end_seq": c.end_seq,
                "entry_count": c.entry_count,
                "kinds": c.kinds,
                "produced_by": c.produced_by,
            }
            for c in sorted(self._compactions, key=lambda c: c.start_seq)
        ]

    async def _do_compact(
        self, start_seq: int, end_seq: int,
        victims: list[dict[str, Any]], *, produced_by: str,
    ) -> Any:
        from .stream_log import CompactionDescriptor
        payload = await self._compactor.compact(victims, self.formatter)
        desc = CompactionDescriptor(
            start_seq=start_seq,
            end_seq=end_seq,
            summary=payload["summary"],
            kinds=payload.get("kinds", {}),
            entry_count=payload.get("entry_count", len(victims)),
            archive_ref=None,
            produced_by=produced_by,
            timestamp=time.time(),
        )
        self._compactions.append(desc)
        self._entries = [
            e for e in self._entries
            if not (
                e.get("kind") != "compaction_summary"
                and start_seq <= int(e.get("seq", -1)) <= end_seq
            )
        ]
        self._entries.append(self._summary_entry(desc))
        self._entries.sort(key=self._view_sort_key)
        await self._store.write_index(self._index())
        return desc

    def _summary_entry(self, desc: Any) -> dict[str, Any]:
        return {
            "kind": "compaction_summary",
            "timestamp": desc.timestamp,
            "payload": {
                "covers": [desc.start_seq, desc.end_seq],
                "summary": desc.summary,
                "kinds": desc.kinds,
                "entry_count": desc.entry_count,
                "produced_by": desc.produced_by,
                "archive_ref": desc.archive_ref,
            },
        }

    def _index(self) -> Any:
        from .stream_log import StreamLogIndex
        return StreamLogIndex(
            next_seq=self._next_seq, compactions=list(self._compactions),
        )

    @staticmethod
    def _complement(
        covered: list[tuple[int, int]], lo: int, hi: int,
    ) -> list[tuple[int, int]]:
        """Return the sub-ranges of ``[lo, hi]`` not covered by any span
        in ``covered`` (sorted, possibly overlapping), in ascending order."""
        if hi < lo:
            return []
        result: list[tuple[int, int]] = []
        cursor = lo
        for start, end in sorted(covered):
            if end < cursor:
                continue
            if start > cursor:
                result.append((cursor, min(start - 1, hi)))
            cursor = max(cursor, end + 1)
            if cursor > hi:
                break
        if cursor <= hi:
            result.append((cursor, hi))
        return result


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
            if entry["kind"] == "compaction_summary":
                lines.append(render_compaction_summary(entry))
            elif entry["kind"] == "event":
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
            if entry["kind"] == "compaction_summary":
                lines.append(render_compaction_summary(entry))
            elif entry["kind"] == "event":
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


def render_compaction_summary(entry: dict[str, Any], max_chars: int = 500) -> str:
    """Render a ``compaction_summary`` view-entry as one markdown line.

    Shared by every stock formatter so a compacted span stays visible
    (and its seq span discoverable, so the agent knows what it can
    ``expand_stream_span``). Custom formatters should call this for
    ``entry["kind"] == "compaction_summary"`` entries.
    """
    payload = entry.get("payload", {}) or {}
    covers = payload.get("covers") or [None, None]
    count = payload.get("entry_count", 0)
    summary = _truncate(str(payload.get("summary", "")), max_chars)
    return (
        f"- ▸ *condensed history* (seq {covers[0]}–{covers[1]}, "
        f"{count} entries — expandable): {summary}"
    )


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
            if kind == "compaction_summary":
                lines.append(render_compaction_summary(entry))
                continue
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
        kept = [
            e for e in entries
            if e.get("kind") in ("tool_output", "compaction_summary")
        ]
        if not kept:
            return ""
        lines: list[str] = [self._section_title, ""]
        for entry in kept:
            if entry.get("kind") == "compaction_summary":
                lines.append(render_compaction_summary(entry))
                continue
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
        vcm_entries = [
            e for e in entries
            if e.get("kind") in ("vcm_update", "compaction_summary")
        ]
        if not vcm_entries:
            return ""
        if (
            self._max_entries_shown is not None
            and len(vcm_entries) > self._max_entries_shown
        ):
            vcm_entries = vcm_entries[-self._max_entries_shown:]
        lines: list[str] = [self._section_title, ""]
        for entry in vcm_entries:
            if entry.get("kind") == "compaction_summary":
                lines.append(render_compaction_summary(entry))
                continue
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
            e for e in entries
            if e.get("kind") in ("monorepo_commit", "compaction_summary")
        ]
        if not commit_entries:
            return ""
        lines: list[str] = [self._section_title, ""]
        for entry in commit_entries:
            if entry.get("kind") == "compaction_summary":
                lines.append(render_compaction_summary(entry))
                continue
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
        domain_entries: list[dict[str, Any]] = []
        for e in entries:
            kind = e.get("kind")
            if kind == "compaction_summary":
                domain_entries.append(e)
            elif kind == "domain_state":
                if (
                    self._state_machine_name is None
                    or (e.get("payload") or {}).get("state_machine")
                    == self._state_machine_name
                ):
                    domain_entries.append(e)
        if not domain_entries:
            return ""
        if (
            self._max_entries_shown is not None
            and len(domain_entries) > self._max_entries_shown
        ):
            domain_entries = domain_entries[-self._max_entries_shown:]
        lines: list[str] = [self._section_title, ""]
        for entry in domain_entries:
            if entry.get("kind") == "compaction_summary":
                lines.append(render_compaction_summary(entry))
                continue
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
