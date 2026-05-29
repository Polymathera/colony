"""Compaction + spillover strategy objects for consciousness streams.

Every design alternative here sits behind an ABC so implementations
swap without touching the stream:

- :class:`CompactionPolicy` — *when/what* to compact. Default
  :class:`KeepRecentCompactionPolicy` (compact the oldest raw entries
  beyond a recent window). A relevance-ranked or time-based policy is
  a drop-in.
- :class:`StreamCompactor` — *how* to condense a span. Default
  :class:`LLMStreamCompactor` (calls ``agent.infer``).
  :class:`ExtractiveStreamCompactor` is the no-LLM alternative
  (head/tail digest) — this is the "keep only the most relevant
  entries" arm of the spec, vs. the "summarize" arm.
- :class:`SpillArchive` — *deep archive + re-attention* of a span as
  real VCM context pages. Default :class:`VcmSpillArchive`;
  :class:`NoopSpillArchive` when no VCM is available.

Token estimation reuses the cluster's existing
:class:`~polymathera.colony.cluster.tokenization.TokenizerProtocol`
(``count_tokens``) — :func:`default_token_estimator` returns a
``TiktokenTokenizer`` (cl100k_base, no model load), falling back to a
char heuristic only if tiktoken is unavailable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ...base import Agent
    from .streams import ConsciousnessStreamFormatter
    from ....cluster.tokenization import TokenizerProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation — reuse the cluster tokenizer
# ---------------------------------------------------------------------------


class _HeuristicTokenizer:
    """Zero-dependency fallback implementing ``TokenizerProtocol.count_tokens``
    when tiktoken is unavailable. ~4 chars/token is good enough for a
    budget safety-net (which never needs exactness)."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def default_token_estimator() -> "TokenizerProtocol":
    """Reuse the cluster's ``TiktokenTokenizer`` (cl100k_base, fast,
    no model download). Falls back to a char heuristic only if tiktoken
    can't be imported. A model-exact ``HuggingFaceTokenizer`` is a
    drop-in via the same protocol."""
    try:
        from ....cluster.tokenization import TiktokenTokenizer
        return TiktokenTokenizer()
    except Exception:  # noqa: BLE001 — tiktoken missing / load failure
        logger.debug("tiktoken unavailable; using heuristic token estimator")
        return _HeuristicTokenizer()


# ---------------------------------------------------------------------------
# Compaction policy — when/what to compact
# ---------------------------------------------------------------------------


class CompactionPolicy(ABC):
    """Decides which raw span to compact. Pure logic (no I/O)."""

    @abstractmethod
    def select_span(self, *, raw_window: list[dict[str, Any]]) -> tuple[int, int] | None:
        """Given the raw (non-summary) entries currently in the view, in
        seq order, return the ``(start_seq, end_seq)`` span to compact,
        or ``None`` to compact nothing.

        ``raw_window`` is the eligible set (entries at/above the index
        ``window_floor_seq``). Called only after the stream has already
        determined it is over its token budget.
        """


class KeepRecentCompactionPolicy(CompactionPolicy):
    """Compact everything except the most recent ``keep_recent`` raw
    entries. Simple, predictable, and prefers the oldest (coldest)
    span — least likely to be needed again."""

    def __init__(self, keep_recent: int = 12):
        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        self._keep_recent = keep_recent

    def select_span(self, *, raw_window: list[dict[str, Any]]) -> tuple[int, int] | None:
        if len(raw_window) <= self._keep_recent:
            return None
        victims = raw_window[: len(raw_window) - self._keep_recent]
        return victims[0]["seq"], victims[-1]["seq"]


# ---------------------------------------------------------------------------
# Compactor — how to condense a span
# ---------------------------------------------------------------------------


def _kind_histogram(entries: list[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for e in entries:
        k = e.get("kind", "?")
        hist[k] = hist.get(k, 0) + 1
    return hist


class StreamCompactor(ABC):
    """Condenses a span of raw entries into a summary payload."""

    @abstractmethod
    async def compact(
        self,
        entries: list[dict[str, Any]],
        formatter: "ConsciousnessStreamFormatter",
    ) -> dict[str, Any]:
        """Return ``{"summary": str, "kinds": dict, "entry_count": int}``.

        ``formatter`` is the stream's own formatter so the condensation
        works from the same representation the LLM normally sees.
        """


DEFAULT_COMPACTION_PROMPT = (
    "You are condensing part of an AI agent's working memory so it stays "
    "concise without losing decision-relevant facts. Summarize the entries "
    "below into a short, faithful digest: keep concrete values, decisions, "
    "outcomes, identifiers, and unresolved issues; drop boilerplate and "
    "repetition. Do not invent anything."
)


class LLMStreamCompactor(StreamCompactor):
    """Summarize a span with ``agent.infer``. On any inference failure
    it degrades to an extractive fallback so compaction never blocks
    the agent."""

    def __init__(
        self,
        agent: "Agent",
        prompt: str = DEFAULT_COMPACTION_PROMPT,
        max_summary_tokens: int = 400,
    ):
        self._agent = agent
        self._prompt = prompt
        self._max_summary_tokens = max_summary_tokens

    async def compact(
        self,
        entries: list[dict[str, Any]],
        formatter: "ConsciousnessStreamFormatter",
    ) -> dict[str, Any]:
        kinds = _kind_histogram(entries)
        rendered = formatter.format(entries)
        summary: str
        try:
            resp = await self._agent.infer(
                prompt=f"{self._prompt}\n\n{rendered}\n\nDigest:",
                max_tokens=self._max_summary_tokens,
            )
            summary = (getattr(resp, "generated_text", "") or "").strip()
        except Exception:  # noqa: BLE001 — LLM unavailable / errored
            logger.exception("LLMStreamCompactor: infer failed; using extractive fallback")
            summary = ""
        if not summary:
            summary = _extractive_digest(entries, kinds)
        return {"summary": summary, "kinds": kinds, "entry_count": len(entries)}


def _extractive_digest(entries: list[dict[str, Any]], kinds: dict[str, int]) -> str:
    parts = ", ".join(f"{n}×{k}" for k, n in sorted(kinds.items()))
    return f"[{len(entries)} earlier entries condensed: {parts}]"


class ExtractiveStreamCompactor(StreamCompactor):
    """No-LLM alternative: a deterministic histogram digest. Cheap and
    side-effect-free — use when LLM cost/latency isn't warranted or in
    tests. The originals remain in the log + expandable, so the lossy
    digest is always recoverable."""

    async def compact(
        self,
        entries: list[dict[str, Any]],
        formatter: "ConsciousnessStreamFormatter",
    ) -> dict[str, Any]:
        kinds = _kind_histogram(entries)
        return {
            "summary": _extractive_digest(entries, kinds),
            "kinds": kinds,
            "entry_count": len(entries),
        }


# ---------------------------------------------------------------------------
# Spill archive — deep durable archive + re-attention as VCM context pages
# ---------------------------------------------------------------------------


class SpillArchive(ABC):
    """Re-attaches a historical span into the agent's *real* LLM context
    window (not just the prompt text). On-demand: invoked only when the
    agent expands a span with re-attention requested — never on the spill
    path (the raw log is the authoritative recall)."""

    @abstractmethod
    async def reattach(
        self,
        *,
        stream_name: str,
        start_seq: int,
        end_seq: int,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        """Materialize the span as VCM pages + page-fault them in.
        Returns the page_ids attended (empty if unavailable)."""


class NoopSpillArchive(SpillArchive):
    """Used when no VCM is available. Expansion still works (originals
    come back from the raw log as prompt text); only the page-fault-into-
    real-context enhancement is absent."""

    async def reattach(self, **_: Any) -> list[str]:
        return []


class VcmSpillArchive(SpillArchive):
    """Maps a spilled span into a dedicated VCM scope and page-faults its
    pages into the agent's context window. Best-effort: any failure
    degrades to ``[]`` (the agent still sees the originals as prompt
    text via the raw-log expand path)."""

    def __init__(self, agent: "Agent"):
        self._agent = agent

    async def reattach(
        self,
        *,
        stream_name: str,
        start_seq: int,
        end_seq: int,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        try:
            from ...scopes import ScopeUtils
            # Resolve the VCM handle lazily via the public accessor — it
            # is wired in the manager's @on_app_ready discover_handles()
            # (NOT at policy init), and reattach only runs at action time
            # when the app is fully ready. ``_handles`` is the no-cycle
            # accessor module the rest of the codebase uses.
            from ...._handles import get_vcm
            vcm = await get_vcm()
            agent_scope = ScopeUtils.get_agent_level_scope(self._agent)
            archive_scope = (
                f"{agent_scope}:cstream:{stream_name}:archive:{start_seq}-{end_seq}"
            )
            # Stage the span into a blackboard scope, then map it into VCM.
            bb = await self._agent.get_blackboard(
                scope_id=archive_scope, enable_events=False,
            )
            for e in entries:
                await bb.write(
                    key=f"entry:{int(e.get('seq', 0)):020d}",
                    value=e,
                    created_by=f"cstream_archive:{stream_name}",
                )
            await vcm.mmap_application_scope(
                scope_id=archive_scope, source_type="blackboard",
            )
            pages = await vcm.get_pages_for_scope(archive_scope)
            page_ids = [p["page_id"] for p in pages if isinstance(p, dict) and "page_id" in p]
            for page_id in page_ids:
                await vcm.request_page_load(
                    page_id=page_id,
                    agent_id=self._agent.agent_id,
                    priority=10,
                )
            return page_ids
        except Exception:  # noqa: BLE001
            logger.exception(
                "VcmSpillArchive: re-attach failed for %s[%d-%d]; "
                "originals remain available as prompt text via the log.",
                stream_name, start_seq, end_seq,
            )
            return []


__all__ = (
    "default_token_estimator",
    "CompactionPolicy",
    "KeepRecentCompactionPolicy",
    "StreamCompactor",
    "LLMStreamCompactor",
    "ExtractiveStreamCompactor",
    "DEFAULT_COMPACTION_PROMPT",
    "SpillArchive",
    "NoopSpillArchive",
    "VcmSpillArchive",
)
