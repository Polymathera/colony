"""Typed self-reflection primitives for consciousness streams.

A :class:`~polymathera.colony.agents.patterns.planning.streams.ConsciousnessStream`
is a stateful, typed observer over agent experience. Each stream owns a
:class:`StreamReflector` that runs at well-defined moments and returns a
:class:`StreamReflection` — the typed bag the substrate aggregates into
the next planner prompt and the cross-agent diagnostic blackboard.

Three reflection moments:

- ``"entry"`` — after every accepted observation (per-entry rendering;
  the default for display streams that show a rolling view of experience).
- ``"iteration_boundary"`` — after each codegen iteration ends, with the
  full :class:`IterationObservation` in scope (the moment detector-shaped
  reflectors look at a coherent batch of actions / blocks / errors).
- ``"planning_step"`` — just before the next LLM prompt is assembled
  (for reflectors that integrate over the entire history).

Three optional surfaces on every reflection:

- ``section_markdown`` — rendered into the stream's own prompt section.
- ``advisories`` — typed :class:`AdvisoryEntry` entries the substrate
  rolls up under a single ``## Last-Iteration Advisories`` section,
  preserving stream registration order + per-stream advisory order.
- ``diagnostics`` — typed :class:`Diagnostic` events the substrate emits
  on the cross-agent ``AgentDiagnosticProtocol`` blackboard, routed by
  the diagnostic's ``kind``.

This module also hosts :class:`IterationObservation` — the immutable
per-iteration snapshot reflectors read from — plus the helpers
(:func:`build_iteration_observation`,
:func:`outer_error_from_action_result`) that build it from a policy's
in-memory state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ...models import ActionResult, ErrorContext
from .models import (
    BlockedDispatch,
    CallRecord,
    IterationObservation,
    ReflectMoment,
    StreamReflection,
)



def build_iteration_observation(
    *,
    iter_index: int,
    actions_called: "list[CallRecord]",
    guardrail_blocks: "list[BlockedDispatch]",
    outer_action_error: ErrorContext | None = None,
) -> IterationObservation:
    """Construct an observation from the slices the policy maintains.

    Each slice is derived by the caller from its ``_call_history`` and
    ``_last_blocked_dispatches`` using the per-iteration boundary marker."""

    return IterationObservation(
        iter_index=iter_index,
        actions_called=list(actions_called),
        guardrail_blocks=list(guardrail_blocks),
        outer_action_error=outer_action_error,
    )


def outer_error_from_action_result(
    result: ActionResult | None,
    action_id: str,
) -> ErrorContext | None:
    """Return a typed :class:`ErrorContext` when the outer
    ``execute_code`` action recorded a failure; ``None`` otherwise.

    Centralised so call sites surfacing outer errors into the observation
    don't each re-implement the field mapping."""

    if result is None or result.success:
        return None
    return ErrorContext(
        error_type="execute_code_error",
        error_details={"message": result.error or "(no error message)"},
        action_context={"action_id": action_id},
    )


class StreamReflector(ABC):
    """Pluggable producer of a :class:`StreamReflection`.

    Reflectors hold the stream's private state across calls. The
    substrate persists :meth:`serialize_state` / :meth:`deserialize_state`
    via the same :class:`~polymathera.colony.agents.patterns.planning.stream_log.StreamLogStore`
    that already persists raw entries, so suspend/resume is durable.

    Subclasses declare which :data:`ReflectMoment`s they care about via
    :attr:`REFLECT_AT`. The substrate skips a reflector at moments it
    didn't declare.
    """

    name: str = "stream_reflector"

    REFLECT_AT: ClassVar[frozenset[ReflectMoment]] = frozenset({"planning_step"})
    """When the substrate should call :meth:`reflect`. Defaults to
    ``"planning_step"`` (the moment display-style reflectors render at).
    Inference reflectors override to ``{"iteration_boundary"}`` (or to
    include ``"entry"`` if they want per-entry granularity)."""

    @abstractmethod
    def reflect(
        self,
        *,
        entries: list[dict[str, Any]],
        observation: IterationObservation | None,
        moment: ReflectMoment,
    ) -> StreamReflection:
        """Produce a reflection for this stream at ``moment``.

        Args:
            entries: The stream's recorded entries (rolling window in
                legacy mode; raw + compaction-summary view-entries in
                compaction mode), ordered as the stream's view sees them.
            observation: The per-iteration snapshot, present only when
                ``moment == "iteration_boundary"``. ``None`` otherwise.
            moment: The reflection moment the substrate is dispatching.
        """

    def serialize_state(self) -> dict[str, Any]:
        """Reflector's private state, JSON-shaped. Default no-op for
        stateless reflectors. Override when the reflector holds
        cross-iteration state (prior-terminal maps, streak counters,
        pending-request sets, …)."""
        return {}

    def deserialize_state(self, state: dict[str, Any]) -> None:
        """Restore from the JSON shape :meth:`serialize_state` produced.
        Default no-op. Idempotent — called once during ``rehydrate``."""
        return None


# ---------------------------------------------------------------------------
# Adapter for legacy display-only formatters
# ---------------------------------------------------------------------------


class RenderOnlyReflector(StreamReflector):
    """Adapter that lifts a legacy
    :class:`~polymathera.colony.agents.patterns.planning.streams.ConsciousnessStreamFormatter`
    into the reflector contract by returning its ``format()`` output as
    :attr:`StreamReflection.section_markdown`.

    Display-only streams (Conversation, ToolResult, MonorepoCommit,
    VCMUpdate, DomainState, EventLog) reach the unified contract via
    this adapter without per-call-site migration — the stream auto-wraps
    a ``formatter=`` Blueprint at construction.
    """

    REFLECT_AT: ClassVar[frozenset[ReflectMoment]] = frozenset({"planning_step"})

    def __init__(self, formatter: Any, name: str = "render_only") -> None:
        self._formatter = formatter
        self.name = name

    def reflect(
        self,
        *,
        entries: list[dict[str, Any]],
        observation: IterationObservation | None,  # noqa: ARG002
        moment: ReflectMoment,  # noqa: ARG002
    ) -> StreamReflection:
        return StreamReflection(section_markdown=self._formatter.format(entries))


__all__ = (
    "StreamReflector",
    "RenderOnlyReflector",
    "build_iteration_observation",
    "outer_error_from_action_result",
)
