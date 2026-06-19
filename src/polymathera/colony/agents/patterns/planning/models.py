
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from ...models import ErrorContext

# ============================================================================
# Call history record — the shared data shape every RuntimeGuardrail reads.
# ============================================================================


@dataclass
class CallRecord:
    """One entry in ``CodeGenerationActionPolicy._call_history``.

    Replaces the previous ``list[str]`` shape so guardrails can make
    args-aware decisions (e.g. "did get_agent_status get called for
    THIS agent_id?") rather than only matching on action-key
    substrings.

    Status semantics:

    - ``"pending"`` — record was created at admit time but the action
      hasn't returned yet. Rare for guardrail reads since check()
      runs BEFORE dispatch; the field exists so the policy can
      pre-create the record at admit time without lying about status.
    - ``"ok"`` / ``"error"`` — terminal states populated when the
      dispatch returns.
    - ``"blocked"`` — the guardrail itself refused the call. Lets
      downstream guardrails distinguish "the LLM tried X but a gate
      stopped it" from "the LLM never tried X".

    ``result`` carries a small snapshot of the action's return value
    so guardrails can inspect the OUTCOME of prior calls — e.g.
    ``ApprovalRequiredGuardrail`` keys off
    ``HumanApprovalCapability.get_response`` returning
    ``{"choice": "Approve"}``. Capped at
    :data:`CALL_RECORD_RESULT_PREVIEW_BYTES` so a single big LLM
    payload can't blow the per-iteration history budget; guardrails
    that need the full payload should query the span store, not
    ``call_history``.
    """

    action_key: str
    params: dict[str, Any]
    action_id: str = ""
    """The dispatched action's id — keys into
    :attr:`PlanExecutionContext.action_results`. Empty for the
    ``"blocked"`` status (no dispatch happened)."""
    start_wall: float = field(default_factory=time.time)
    end_wall: float | None = None
    status: Literal["pending", "ok", "error", "blocked"] = "pending"
    error: str | None = None
    result: Any = None


CALL_RECORD_RESULT_PREVIEW_BYTES = 4096
"""Per-call truncation cap on ``CallRecord.result``. 4 KiB easily fits
typed envelopes like ``{request_id, choice, decided_by, decided_at}``
that guardrails inspect, while keeping the policy's in-memory
history bounded across long-running coordinators."""


@dataclass
class BlockedDispatch:
    """One entry in
    :attr:`CodeGenerationActionPolicy._last_blocked_dispatches`.

    Captured at the moment the runtime guardrail refuses a ``run()``
    call inside the REPL. Survives exactly one iteration — cleared at
    the start of the next code-generation cycle, same lifecycle as
    :attr:`CodeGenerationActionPolicy._call_history`. Rendered into
    the planner prompt under "## Blocked Dispatches (last iteration)"
    so the LLM sees the block AND the guardrail's suggestion BEFORE
    proposing its next cell — instead of having to infer the block
    after the fact from a ``result.success=False`` in the cell's own
    code.

    The ``params_preview`` is a JSON-truncated snapshot of the
    proposed call's params capped at
    :data:`BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES`; full payloads
    stay in the span store, not in this in-memory list.
    """

    action_key: str
    params_preview: Any
    reason: str
    suggestion: str
    wall_time: float = field(default_factory=time.time)


BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES = 1024
"""Per-block truncation cap on ``BlockedDispatch.params_preview``.
Smaller than ``CALL_RECORD_RESULT_PREVIEW_BYTES`` because blocked
dispatches usually carry the LLM's proposed args verbatim and
typically a few short fields (action_key + content + a few kwargs)
are enough to recover; long-tail payloads (entire decomposition
proposals) get truncated."""


_FROZEN = ConfigDict(frozen=True, arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Iteration observation (built once per iteration boundary)
# ---------------------------------------------------------------------------


class IterationObservation(BaseModel):
    """Immutable per-iteration snapshot reflectors read.

    Reflectors MUST treat the lists as read-only — they are shared
    across the substrate's reflector dispatch.
    """

    model_config = _FROZEN

    iter_index: int
    """Iteration number this observation covers (1-indexed)."""

    actions_called: list[CallRecord]
    """``CallRecord`` entries appended during the iteration. Includes
    ``ok`` / ``error`` / ``blocked`` statuses."""

    guardrail_blocks: list[BlockedDispatch]
    """``BlockedDispatch`` entries — pre-dispatch guardrail rejections."""

    outer_action_error: ErrorContext | None = None
    """When the iteration's ``execute_code`` action itself failed (the
    LLM-generated code raised outside any inner ``run()``), this carries
    the typed error. Distinct from per-CallRecord errors in
    :attr:`actions_called` because the failure is at the wrapping-action
    layer; without surfacing it here a reflector would only see the
    surviving inner ``run()`` calls."""



# ---------------------------------------------------------------------------
# Typed reflection outputs
# ---------------------------------------------------------------------------


class AdvisoryEntry(BaseModel):
    """Typed next-iteration guidance produced by a reflector.

    ``body`` is the prose the LLM reads; ``next_action_code`` is an
    optional literal code snippet the LLM may copy verbatim (the
    actionability discipline — see memory rule
    ``llm-prompts-must-be-actionable``).
    """

    model_config = _FROZEN

    source: str
    """Reflector name. Used for deduplication, ordering, and debugging."""

    kind: str
    body: str
    next_action_code: str | None = None


class Diagnostic(BaseModel):
    """Typed cross-agent event a reflector emits on the diagnostic
    blackboard. Routed by :attr:`kind`; subscribers match on the kind,
    not on the producing stream's identity.
    """

    model_config = _FROZEN

    kind: str
    severity: Literal["info", "warning", "alert"] = "info"
    payload: dict[str, Any] = Field(default_factory=dict)


class StreamReflection(BaseModel):
    """One stream's typed output for one ``reflect()`` call.

    Three optional surfaces — every reflection populates ONE OR MORE;
    the empty reflection is valid and represents 'nothing to surface'.

    The substrate aggregates across streams:
    - ``section_markdown`` from every stream is rendered as that
      stream's own prompt section, in stream registration order.
    - ``advisories`` from every stream are flattened, sorted by
      ``(stream_registration_order, AdvisoryEntry.source)``, and
      rendered under ONE ``## Last-Iteration Advisories`` section.
    - ``diagnostics`` from every stream are each emitted as a
      blackboard event keyed by ``Diagnostic.kind``.
    """

    model_config = _FROZEN

    section_markdown: str = ""
    advisories: list[AdvisoryEntry] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.section_markdown
            and not self.advisories
            and not self.diagnostics
        )


# ---------------------------------------------------------------------------
# Reflector contract
# ---------------------------------------------------------------------------


ReflectMoment = Literal["entry", "iteration_boundary", "planning_step"]
"""When the substrate calls ``StreamReflector.reflect``."""



__all__ = (
    "AdvisoryEntry",
    "Diagnostic",
    "IterationObservation",
    "ReflectMoment",
    "StreamReflection",
)

