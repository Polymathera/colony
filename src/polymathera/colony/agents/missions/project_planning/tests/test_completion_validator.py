"""Tests for :class:`DecomposeCompletionValidator` + sibling tracker.

The validator's contract: for decompose mode, allow ``signal_completion``
only when the in-scope issue set has been drained (decomposed,
classified non-decomposable, or explicitly early-stopped). For other
modes, delegate to a fallback (default :class:`LLMCompletionValidator`).
The decision is structural — no LLM judgment involved — so these
tests construct the inputs (scope params, run_call_trace, action
results, blackboard early-stop signal) directly and assert the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from unittest.mock import MagicMock

from polymathera.colony.agents.missions.project_planning.completion_validator import (
    DecomposeBacklogTracker,
    DecomposeCompletionValidator,
)
from polymathera.colony.agents.missions.project_planning.coordinator import (
    ProjectPlanningCoordinator,
)
from polymathera.colony.agents.missions.project_planning.mission_control import (
    DecomposeEarlyStopProtocol,
)
from polymathera.colony.agents.patterns.actions.code_constraints import (
    CompletionValidationResult,
    CompletionValidator,
)
from polymathera.colony.agents.models import PlanExecutionContext
from polymathera.colony.design_monorepo.process import DesignProcessCapability


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeActionResult:
    success: bool
    output: Any


class _StubBlackboard:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.writes: list[tuple[str, dict[str, Any]]] = []

    async def read(self, key: str) -> Any:
        if key not in self.store:
            raise KeyError(key)
        return self.store[key]

    async def write(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: Any = None,
        metadata: Any = None,
    ) -> None:
        self.store[key] = value
        self.writes.append((key, value))


class _StubAgent:
    def __init__(
        self,
        params: dict[str, Any],
        agent_id: str = "agent-coord-99",
    ) -> None:
        self.agent_id = agent_id
        self.metadata = MagicMock()
        self.metadata.parameters = params
        self.metadata.goals = []
        self.blackboard = _StubBlackboard()

    async def get_blackboard(self, *, scope_id: str | None = None) -> _StubBlackboard:
        return self.blackboard


_SNAPSHOT_ACTION_ID = "act-snapshot-1"


def _snapshot_entry() -> dict[str, Any]:
    """A run_call_trace entry for a successful
    ``snapshot_open_roadmap_issues`` call."""

    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )
    action_name = (
        GitHubCapability.snapshot_open_roadmap_issues._action_key
    )
    return {
        "call_index": 0,
        "action_key": f"GitHubCapability.GitHubCapability.{action_name}",
        "parameters": {},
        "success": True,
        "error": None,
        "output_preview": "",
        "blocked": False,
    }


def _exec_ctx(
    *,
    classify_action_id: str = "act-classify-1",
    create_action_id: str = "act-create-1",
    trace_entries: list[dict[str, Any]] | None = None,
    action_results: dict[str, _FakeActionResult] | None = None,
    scope_numbers: list[int] | None = None,
) -> PlanExecutionContext:
    """Build a ``PlanExecutionContext`` carrying a typed
    :class:`CodegenStepSummary`. ``scope_numbers``, when set, prepends
    a successful snapshot whose action_result is that list."""

    from polymathera.colony.agents.models import CodegenStepSummary

    ctx = PlanExecutionContext()
    combined_entries: list[dict[str, Any]] = []
    if scope_numbers is not None:
        combined_entries.append(_snapshot_entry())
    if trace_entries:
        offset = 1 if scope_numbers is not None else 0
        for i, e in enumerate(trace_entries):
            shifted = dict(e)
            shifted["call_index"] = i + offset
            combined_entries.append(shifted)
    if combined_entries:
        ctx.codegen_step_summaries["step-1"] = CodegenStepSummary(
            actions_called=[e["action_key"] for e in combined_entries],
            action_ids=[
                _SNAPSHOT_ACTION_ID
                if "snapshot_open_roadmap_issues" in e["action_key"]
                else (
                    classify_action_id if "classify" in e["action_key"]
                    else create_action_id
                )
                for e in combined_entries
            ],
            run_call_trace=combined_entries,
        )
    if scope_numbers is not None:
        ctx.action_results[_SNAPSHOT_ACTION_ID] = _FakeActionResult(
            success=True, output=list(scope_numbers),
        )
    if action_results:
        for aid, result in action_results.items():
            ctx.action_results[aid] = result
    return ctx


def _create_entry(
    parent_issue_number: int,
    *,
    dry_run: bool = False,
    success: bool = True,
    blocked: bool = False,
    call_index: int = 0,
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "action_key": (
            f"DesignProcessCapability.DesignProcessCapability."
            f"{DesignProcessCapability.CREATE_DECOMPOSITION_ACTION_KEY}"
        ),
        "parameters": {
            "parent_issue_number": parent_issue_number,
            "children": [],
            "dry_run": dry_run,
        },
        "success": success,
        "error": None if success else "Boom",
        "output_preview": "",
        "blocked": blocked,
    }


def _classify_entry(call_index: int = 0) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "action_key": (
            f"DesignProcessCapability.DesignProcessCapability."
            f"{DesignProcessCapability.CLASSIFY_ISSUES_DECOMPOSABILITY_ACTION_KEY}"
        ),
        "parameters": {"issue_numbers": [44, 45, 46]},
        "success": True,
        "error": None,
        "output_preview": "",
        "blocked": False,
    }


# ---------------------------------------------------------------------------
# Drain predicate (DecomposeBacklogTracker)
# ---------------------------------------------------------------------------


def test_tracker_remaining_subtracts_applied_and_non_decomposable() -> None:
    t = DecomposeBacklogTracker(
        in_scope=frozenset({44, 45, 46}),
        applied=frozenset({44}),
        classified_non_decomposable=frozenset({46}),
        early_stopped=False,
        max_parents_per_run=None,
    )
    assert t.remaining() == frozenset({45})
    assert t.is_drained() is False


def test_tracker_is_drained_when_remaining_empty() -> None:
    t = DecomposeBacklogTracker(
        in_scope=frozenset({44, 45}),
        applied=frozenset({44, 45}),
        classified_non_decomposable=frozenset(),
        early_stopped=False,
        max_parents_per_run=None,
    )
    assert t.is_drained() is True


def test_tracker_is_drained_when_cap_reached() -> None:
    """Cap interaction: when ``max_parents_per_run`` is set and the
    apply count reaches it, the run is drained for completion purposes
    even if ``remaining`` is non-empty (deferred-out-of-scope semantics)."""

    t = DecomposeBacklogTracker(
        in_scope=frozenset({44, 45, 46}),
        applied=frozenset({44}),
        classified_non_decomposable=frozenset(),
        early_stopped=False,
        max_parents_per_run=1,
    )
    assert t.is_drained() is True


def test_tracker_is_drained_when_early_stopped() -> None:
    t = DecomposeBacklogTracker(
        in_scope=frozenset({44, 45}),
        applied=frozenset(),
        classified_non_decomposable=frozenset(),
        early_stopped=True,
        max_parents_per_run=None,
    )
    assert t.is_drained() is True
    assert t.remaining() == frozenset()


# ---------------------------------------------------------------------------
# Validator — happy paths and rejection
# ---------------------------------------------------------------------------


async def test_allows_completion_when_scope_drained() -> None:
    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(
        scope_numbers=[44, 45],
        trace_entries=[
            _create_entry(44, call_index=0),
            _create_entry(45, call_index=1),
        ],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is True
    assert "drained" in verdict.reason


async def test_rejects_completion_with_remaining_set_in_message() -> None:
    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(
        scope_numbers=[44, 45, 46],
        trace_entries=[_create_entry(44, call_index=0)],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is False
    assert "[45, 46]" in verdict.reason
    # FACT only — no procedural prose
    for forbidden in ("then call", "next, call", "->", "→", "Step 1"):
        assert forbidden not in verdict.reason, verdict.reason


async def test_dry_run_create_decomposition_does_not_count_as_applied() -> None:
    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(
        scope_numbers=[44, 45],
        trace_entries=[
            _create_entry(44, dry_run=True, call_index=0),
            _create_entry(45, dry_run=False, call_index=1),
        ],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is False  # only #45 actually applied
    assert "Applied (decomposed): [45]" in verdict.reason


async def test_blocked_create_does_not_count_as_applied() -> None:
    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(
        scope_numbers=[44],
        trace_entries=[
            _create_entry(44, blocked=True, success=False, call_index=0),
        ],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is False


async def test_classified_non_decomposable_counts_toward_drain() -> None:
    """A successful ``classify_issues_decomposability`` whose output
    marks an in-scope issue as ``decomposable=False`` removes it from
    the drain set without requiring ``create_decomposition``."""

    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    classify_result = _FakeActionResult(
        success=True,
        output={"classifications": [
            {"number": 44, "decomposable": True, "reason": "..."},
            {"number": 45, "decomposable": False, "reason": "atomic"},
            {"number": 46, "decomposable": False, "reason": "atomic"},
        ]},
    )
    ctx = _exec_ctx(
        scope_numbers=[44, 45, 46],
        classify_action_id="act-classify-1",
        create_action_id="act-create-1",
        trace_entries=[
            _classify_entry(call_index=0),
            _create_entry(44, call_index=1),
        ],
        action_results={
            "act-classify-1": classify_result,
            "act-create-1": _FakeActionResult(success=True, output={"ok": True}),
        },
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is True


async def test_max_parents_per_run_caps_required_applies() -> None:
    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
        ProjectPlanningCoordinator.MAX_PARENTS_PER_RUN_PARAM_NAME: 2,
    })
    ctx = _exec_ctx(
        scope_numbers=[44, 45, 46, 47, 48],
        trace_entries=[
            _create_entry(44, call_index=0),
            _create_entry(45, call_index=1),
        ],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is True
    assert "cap reached" in verdict.reason


async def test_early_stop_state_allows_completion() -> None:
    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    agent.blackboard.store[DecomposeEarlyStopProtocol.signal_key(agent.agent_id)] = {
        "agent_id": agent.agent_id,
        "user_acknowledgement_quote": "stop after the first one",
        "remaining_in_scope": [45, 46],
    }
    ctx = _exec_ctx(
        scope_numbers=[44, 45, 46],
        trace_entries=[_create_entry(44, call_index=0)],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is True
    assert "early-stop recorded" in verdict.reason


async def test_no_recorded_early_stop_blocks_completion() -> None:
    """Without a real early-stop blackboard entry, the LLM CANNOT
    self-certify the stop by, e.g., adding a wishful entry to the
    trace. The validator only reads the typed framework state."""

    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(
        scope_numbers=[44, 45],
        trace_entries=[_create_entry(44, call_index=0)],
    )
    validator = DecomposeCompletionValidator()
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is False


# ---------------------------------------------------------------------------
# Fallback delegation
# ---------------------------------------------------------------------------


class _RecordingFallback(CompletionValidator):
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def validate(
        self,
        agent: Any,
        goals: list[str],
        results: dict[str, Any],
        execution_context: PlanExecutionContext,
    ) -> CompletionValidationResult:
        self.calls.append(agent.metadata.parameters.get(ProjectPlanningCoordinator.MODE_PARAM_NAME))
        return CompletionValidationResult(
            allowed=True, reason="fallback says yes",
        )


@pytest.mark.parametrize("mode", ["bootstrap", "refresh", "assignments", None])
async def test_non_decompose_mode_delegates_to_fallback(mode: Any) -> None:
    agent = _StubAgent(params={ProjectPlanningCoordinator.MODE_PARAM_NAME: mode} if mode else {})
    ctx = _exec_ctx()
    fallback = _RecordingFallback()
    validator = DecomposeCompletionValidator(fallback=fallback)
    verdict = await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    assert verdict.allowed is True
    assert verdict.reason == "fallback says yes"
    assert fallback.calls == [mode]


# ---------------------------------------------------------------------------
# Drain-state publication
# ---------------------------------------------------------------------------


async def test_rejection_publishes_drain_state_to_blackboard() -> None:
    """Every reject snapshots the drain state. Postmortems can replay
    what the validator believed at each rejection point."""

    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(
        scope_numbers=[44, 45, 46],
        trace_entries=[_create_entry(44, call_index=0)],
    )
    validator = DecomposeCompletionValidator()
    await validator.validate(
        agent=agent, goals=[], results={}, execution_context=ctx,
    )
    drain_writes = [
        (k, v) for (k, v) in agent.blackboard.writes
        if "drain_state" in k
    ]
    assert len(drain_writes) == 1
    _, payload = drain_writes[0]
    assert payload["remaining"] == [45, 46]
    assert payload["applied"] == [44]


# ---------------------------------------------------------------------------
# Action-key constant pin (Change 4 coupling)
# ---------------------------------------------------------------------------


def test_validator_resolves_action_keys_via_decorator_attribute() -> None:
    """The validator MUST recover action keys from the canonical
    ``@action_executor``-decorated methods themselves via the
    decorator-attached ``_action_key`` attribute — NOT via bare
    string literals (would silently rot on a rename) and NOT via a
    parallel ClassVar constant (duplicates the method name spelling).
    A rename of the method surfaces here at import time because the
    attribute access targets the named method directly."""

    import polymathera.colony.agents.missions.project_planning.completion_validator as mod
    source = (mod.__file__ or "").rstrip("c")
    with open(source, "r", encoding="utf-8") as fh:
        text = fh.read()
    # The validator must reach the action keys via the decorator
    # attribute on the canonical method, not a duplicated string.
    assert ".create_decomposition._action_key" in text
    assert ".classify_issues_decomposability._action_key" in text
    assert ".snapshot_open_roadmap_issues._action_key" in text
    # And NO bare string literal of any of the action names.
    for forbidden in (
        '"create_decomposition"',
        '"classify_issues_decomposability"',
        '"snapshot_open_roadmap_issues"',
    ):
        assert forbidden not in text, forbidden


# ---------------------------------------------------------------------------
# _resolve_scope: trace-read snapshot contract (Change 3)
# ---------------------------------------------------------------------------
#
# The validator reads the in-scope set from the FIRST successful
# ``snapshot_open_roadmap_issues`` call recorded in the run trace's
# action_results — same shape it uses for classify results. The
# coordinator's goal block teaches the LLM to call snapshot as the
# FIRST action; the validator's rejection on a missing snapshot
# tells the LLM what to do next.


async def test_validate_raises_when_snapshot_not_recorded() -> None:
    """Without a snapshot in the trace the in-scope set is
    unestablished — the validator raises with an action-key-named
    rejection so the LLM's next iteration can call the snapshot and
    recover. Regression for the pre-Change-3 band-aid that returned
    ``frozenset()`` here and let ``is_drained()`` trivially go True."""

    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    ctx = _exec_ctx(trace_entries=[])
    validator = DecomposeCompletionValidator()
    with pytest.raises(RuntimeError, match=r"snapshot_open_roadmap_issues"):
        await validator.validate(
            agent=agent, goals=[], results={}, execution_context=ctx,
        )


async def test_validate_uses_first_snapshot_result_as_scope() -> None:
    """The validator pins on the FIRST successful snapshot — a
    subsequent re-snapshot does NOT silently extend or shrink the
    in-scope set mid-run. This is the canonical contract the
    coordinator's goal block teaches."""

    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    # Seed: first snapshot returns [44, 45] (that is the canonical
    # scope). Then synthesise a SECOND snapshot at a later step
    # returning [44, 45, 99] (a new issue filed mid-run) — the
    # validator must ignore the re-snapshot.
    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )
    from polymathera.colony.agents.models import CodegenStepSummary

    snapshot_action_key = (
        f"GitHubCapability.GitHubCapability."
        f"{GitHubCapability.snapshot_open_roadmap_issues._action_key}"
    )
    ctx = _exec_ctx(scope_numbers=[44, 45])
    # Append a second snapshot step.
    ctx.codegen_step_summaries["step-2"] = CodegenStepSummary(
        actions_called=[snapshot_action_key],
        action_ids=["act-snapshot-2"],
        run_call_trace=[{
            "call_index": 1,
            "action_key": snapshot_action_key,
            "parameters": {},
            "success": True,
            "error": None,
            "output_preview": "",
            "blocked": False,
        }],
    )
    ctx.action_results["act-snapshot-2"] = _FakeActionResult(
        success=True, output=[44, 45, 99],
    )

    validator = DecomposeCompletionValidator()
    in_scope = await validator._resolve_scope(agent, ctx)
    assert in_scope == frozenset({44, 45})  # first snapshot wins; #99 ignored


async def test_validate_ignores_failed_snapshot_attempts() -> None:
    """A snapshot call whose ``success`` is False is skipped — the
    validator picks the first SUCCESSFUL call. A transport failure
    that left a trace entry must not silently establish an empty
    scope."""

    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )
    from polymathera.colony.agents.models import CodegenStepSummary

    agent = _StubAgent(params={
        ProjectPlanningCoordinator.MODE_PARAM_NAME: ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE,
    })
    # First call failed; second succeeded.
    ctx = PlanExecutionContext()
    snapshot_key = (
        f"GitHubCapability.GitHubCapability."
        f"{GitHubCapability.snapshot_open_roadmap_issues._action_key}"
    )
    ctx.codegen_step_summaries["step-1"] = CodegenStepSummary(
        actions_called=[snapshot_key, snapshot_key],
        action_ids=["act-snap-failed", "act-snap-ok"],
        run_call_trace=[
            {
                "call_index": 0, "action_key": snapshot_key,
                "parameters": {}, "success": False,
                "error": "503", "output_preview": "", "blocked": False,
            },
            {
                "call_index": 1, "action_key": snapshot_key,
                "parameters": {}, "success": True,
                "error": None, "output_preview": "", "blocked": False,
            },
        ],
    )
    ctx.action_results["act-snap-failed"] = _FakeActionResult(
        success=False, output=None,
    )
    ctx.action_results["act-snap-ok"] = _FakeActionResult(
        success=True, output=[44, 45],
    )

    validator = DecomposeCompletionValidator()
    in_scope = await validator._resolve_scope(agent, ctx)
    assert in_scope == frozenset({44, 45})


def test_resolve_scope_does_not_return_empty_frozenset_bandaid() -> None:
    """Regression pin: the pre-Change-3 ``return frozenset()`` fallback
    inside ``_resolve_scope`` was the band-aid that defeated the
    validator's drained-set semantics (with ``in_scope == frozenset()``,
    ``is_drained()`` was True on the first ``signal_completion()`` call).
    It must not come back to THIS METHOD.

    Other ``return frozenset()`` sites in the file (e.g.,
    ``DecomposeBacklogTracker.remaining()``'s early-stop
    short-circuit at line 111) are legitimate — they describe a
    deliberately-empty remainder, not a missing scope."""

    import inspect

    source = inspect.getsource(
        DecomposeCompletionValidator._resolve_scope,
    )
    assert "return frozenset()" not in source, (
        "DecomposeCompletionValidator._resolve_scope contains "
        "``return frozenset()`` — the pre-Change-3 band-aid was "
        "restored. Delete it; raise instead per the method's "
        "docstring."
    )
    # And the inline confession the band-aid carried must also not
    # return (per [[dont-ship-with-inline-todos]]).
    for marker in (
        "Until the spawn-time snapshot is wired",
        "fall back to an EMPTY scope",
    ):
        assert marker not in source, (
            f"DecomposeCompletionValidator._resolve_scope contains "
            f"stale band-aid prose ({marker!r}); the snapshot IS "
            f"wired (Change 3)."
        )
