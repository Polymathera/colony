"""StreamMaintenanceCapability — planner-facing control over a stream's
compaction / spillover.

The auto safety-net (in :class:`BaseActionPolicy.execute_iteration`)
keeps each compaction-enabled stream's rendered view under its token
budget without any agent action. This capability layers the
*reasoning-aware* control the spec calls for: the agent can condense a
stream itself, bring a condensed span back when relevance shifts, and
inspect what's currently condensed.

All three actions are legitimately planner-facing (no internal-only
actions here): the LLM reasons about its own working memory and acts on
it. They operate on the streams mounted on the agent's own action
policy — found by name via ``action_policy.get_consciousness_streams``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...base import Agent, AgentCapability
from ...models import AgentSuspensionState
from ..actions import action_executor

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ..planning.streams import ConsciousnessStream


class StreamMaintenanceCapability(AgentCapability):
    """Planner-facing compaction/spillover control for the agent's own
    consciousness streams.

    Mount on any agent that has compaction-enabled streams and wants the
    LLM to manage its working memory deliberately (in addition to the
    automatic budget safety-net).
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        # Pure action surface — no event subscriptions.
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"meta", "memory"})

    def _find_stream(self, name: str) -> "ConsciousnessStream | None":
        policy = getattr(self.agent, "action_policy", None)
        if policy is None or not hasattr(policy, "get_consciousness_streams"):
            return None
        for stream in policy.get_consciousness_streams():
            if getattr(stream, "name", None) == name:
                return stream
        return None

    @action_executor(
        planning_summary=(
            "Condense older entries of one of your consciousness streams "
            "to free up context. Lossless: the originals stay retrievable "
            "via expand_stream_span. Pass start_seq+end_seq to condense a "
            "specific span, or just the stream name to condense the oldest "
            "entries beyond the recent window."
        ),
    )
    async def compact_stream(
        self,
        *,
        stream: str,
        start_seq: int | None = None,
        end_seq: int | None = None,
        keep_recent: int | None = None,
    ) -> dict[str, Any]:
        """Condense a span (or the oldest eligible entries) of ``stream``."""
        s = self._find_stream(stream)
        if s is None:
            return {"status": "error", "message": f"No stream named {stream!r}."}
        if start_seq is not None and end_seq is not None:
            desc = await s.compact_span(start_seq, end_seq, produced_by="agent")
        else:
            desc = await s.compact_now(keep_recent=keep_recent)
        if desc is None:
            return {"status": "noop", "message": "Nothing eligible to condense."}
        return {"status": "compacted", "descriptor": desc}

    @action_executor(
        planning_summary=(
            "Bring a previously-condensed span of a consciousness stream "
            "back into view verbatim (reverses a compaction). Set "
            "reattach_to_context=true to also page the original span back "
            "into your real context window via the VCM."
        ),
    )
    async def expand_stream_span(
        self,
        *,
        stream: str,
        start_seq: int,
        end_seq: int,
        reattach_to_context: bool = False,
    ) -> dict[str, Any]:
        """Re-expand a condensed span of ``stream`` from the durable log."""
        s = self._find_stream(stream)
        if s is None:
            return {"status": "error", "message": f"No stream named {stream!r}."}
        result = await s.expand_span(
            start_seq, end_seq, reattach_to_context=reattach_to_context,
        )
        return {"status": "expanded", **result}

    @action_executor(
        planning_summary=(
            "List the condensed spans of a consciousness stream (their seq "
            "ranges + entry counts) so you know what you can expand back."
        ),
    )
    async def list_stream_history(self, *, stream: str) -> dict[str, Any]:
        """Report the active condensed spans of ``stream``."""
        s = self._find_stream(stream)
        if s is None:
            return {"status": "error", "message": f"No stream named {stream!r}."}
        return {"status": "ok", "stream": stream, "condensed_spans": s.history_summary()}

    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        return state

    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        return None


__all__ = ("StreamMaintenanceCapability",)
