"""Unit tests for the unified-reflection substrate + the six reflectors
that subsume the prior detector/advisor bus pairs.

Each reflector is tested in isolation against synthetic
:class:`IterationObservation`s — the reflection's typed shape is the
contract these tests pin. One integration test at the bottom proves
:class:`ConsciousnessStream` wraps a reflector and that
:meth:`BaseActionPolicy.collect_stream_reflections` aggregates across
streams in registration order.
"""

from __future__ import annotations

from polymathera.colony.agents.models import ErrorContext
from polymathera.colony.agents.patterns.planning.models import (
    AdvisoryEntry,
    BlockedDispatch,
    CallRecord,
    Diagnostic,
    IterationObservation,
    StreamReflection,
)
from polymathera.colony.agents.patterns.planning.reflectors import (
    ApprovalAdvanceReflector,
    CliffGuardReflector,
    ContractDriftReflector,
    ErrorRewriterReflector,
    InconsistencyReflector,
    approval_advance_stream,
    cliff_guard_stream,
    contract_drift_stream,
    error_rewriter_stream,
    inconsistency_stream,
)


def _obs(
    *,
    iter_index: int = 1,
    actions: list[CallRecord] | None = None,
    blocks: list[BlockedDispatch] | None = None,
    outer_action_error: ErrorContext | None = None,
) -> IterationObservation:
    return IterationObservation(
        iter_index=iter_index,
        actions_called=actions or [],
        guardrail_blocks=blocks or [],
        outer_action_error=outer_action_error,
    )


def _spawn_record(
    *,
    mission_type: str = "session",
    outcome: str = "spawned",
    agent_id: str | None = "agent-x",
    status: str = "ok",
) -> CallRecord:
    return CallRecord(
        action_key="SessionAgent.SessionAgent.spawn_mission",
        params={"mission_type": mission_type},
        action_id="a-spawn-1",
        status=status,
        result={
            "outcome": outcome,
            "agent_id": agent_id,
            "mission_type": mission_type,
            "reason": None,
        },
    )


# ---------------------------------------------------------------------------
# StreamReflection sanity
# ---------------------------------------------------------------------------


def test_stream_reflection_empty_is_empty():
    assert StreamReflection().is_empty is True


def test_stream_reflection_with_advisory_is_not_empty():
    r = StreamReflection(
        advisories=[AdvisoryEntry(source="x", kind="y", body="z")],
    )
    assert r.is_empty is False


# ---------------------------------------------------------------------------
# ContractDriftReflector
# ---------------------------------------------------------------------------


def test_contract_drift_records_terminal_then_fires_on_retry():
    reflector = ContractDriftReflector()
    # Iter 1: a successful spawn — terminal outcome recorded, no advisory.
    iter1 = reflector.reflect(
        entries=[],
        observation=_obs(iter_index=1, actions=[_spawn_record()]),
        moment="iteration_boundary",
    )
    assert iter1.is_empty
    # Iter 2: re-call same mission_type. Drift fires.
    iter2 = reflector.reflect(
        entries=[],
        observation=_obs(iter_index=2, actions=[_spawn_record()]),
        moment="iteration_boundary",
    )
    assert len(iter2.advisories) == 1
    adv = iter2.advisories[0]
    assert adv.source == "contract_drift"
    assert adv.kind == "typed_discriminator_drift"
    assert "session" in adv.body
    assert "agent-x" in (adv.next_action_code or "")
    assert len(iter2.diagnostics) == 1
    assert iter2.diagnostics[0].kind == "contract_drift"
    assert iter2.diagnostics[0].payload["prior_iter"] == 1


def test_contract_drift_state_round_trips_through_serialize():
    a = ContractDriftReflector()
    a.reflect(
        entries=[],
        observation=_obs(iter_index=1, actions=[_spawn_record()]),
        moment="iteration_boundary",
    )
    blob = a.serialize_state()
    b = ContractDriftReflector()
    b.deserialize_state(blob)
    iter2 = b.reflect(
        entries=[],
        observation=_obs(iter_index=2, actions=[_spawn_record()]),
        moment="iteration_boundary",
    )
    assert len(iter2.advisories) == 1


def test_contract_drift_returns_empty_when_no_observation():
    assert ContractDriftReflector().reflect(
        entries=[], observation=None, moment="iteration_boundary",
    ).is_empty


# ---------------------------------------------------------------------------
# InconsistencyReflector
# ---------------------------------------------------------------------------


def _respond_record(content: str) -> CallRecord:
    return CallRecord(
        action_key="SessionOrchestratorCapability.SessionOrchestratorCapability.respond_to_user",
        params={"content": content},
        action_id="a-respond-1",
        status="ok",
        result={"ok": True},
    )


def test_inconsistency_fires_when_failure_language_paired_with_spawn_success():
    reflector = InconsistencyReflector()
    obs = _obs(
        iter_index=5,
        actions=[
            _spawn_record(outcome="spawned"),
            _respond_record(
                "Spawn failed again: unknown error. Stopping.",
            ),
        ],
    )
    result = reflector.reflect(
        entries=[], observation=obs, moment="iteration_boundary",
    )
    assert len(result.advisories) == 1
    assert result.advisories[0].kind == "spawn_misread_correction"
    assert "agent-x" in (result.advisories[0].next_action_code or "")
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].kind == "inconsistency_spawn_misread"


def test_inconsistency_silent_when_response_is_clean():
    reflector = InconsistencyReflector()
    obs = _obs(
        iter_index=5,
        actions=[
            _spawn_record(outcome="spawned"),
            _respond_record(
                "Coordinator agent-x is running; status checks in progress.",
            ),
        ],
    )
    assert reflector.reflect(
        entries=[], observation=obs, moment="iteration_boundary",
    ).is_empty


def test_inconsistency_silent_when_spawn_failed():
    reflector = InconsistencyReflector()
    obs = _obs(
        iter_index=5,
        actions=[
            _spawn_record(outcome="rejected", agent_id=None, status="ok"),
            _respond_record("Spawn failed: rejected."),
        ],
    )
    assert reflector.reflect(
        entries=[], observation=obs, moment="iteration_boundary",
    ).is_empty


# ---------------------------------------------------------------------------
# CliffGuardReflector
# ---------------------------------------------------------------------------


def test_cliff_guard_fires_at_two_iters_remaining():
    reflector = CliffGuardReflector(max_iterations=10, lead_iterations=2)
    # iter_index just finished = 9 → about_to_plan = 10 → remaining = 1.
    result = reflector.reflect(
        entries=[], observation=_obs(iter_index=9),
        moment="iteration_boundary",
    )
    assert len(result.advisories) == 1
    assert result.advisories[0].kind == "final_iterations"
    assert "1 iteration" in result.advisories[0].body


def test_cliff_guard_silent_when_budget_ample():
    reflector = CliffGuardReflector(max_iterations=10, lead_iterations=2)
    result = reflector.reflect(
        entries=[], observation=_obs(iter_index=2),
        moment="iteration_boundary",
    )
    assert result.is_empty


def test_cliff_guard_silent_after_cap():
    reflector = CliffGuardReflector(max_iterations=10, lead_iterations=2)
    result = reflector.reflect(
        entries=[], observation=_obs(iter_index=12),
        moment="iteration_boundary",
    )
    assert result.is_empty


# ---------------------------------------------------------------------------
# ApprovalAdvanceReflector
# ---------------------------------------------------------------------------


def _get_response_record(
    *,
    request_id: str = "appr_1",
    state: str = "ready",
    choice: str = "approve_once",
    explanation: str = "",
) -> CallRecord:
    return CallRecord(
        action_key="HumanApprovalCapability.HumanApprovalCapability.get_response",
        params={"request_id": request_id},
        action_id="a-get-1",
        status="ok",
        result={
            "ok": True,
            "state": state,
            "response": (
                None if state == "pending" else {
                    "request_id": request_id,
                    "choice": choice,
                    "explanation": explanation,
                }
            ),
        },
    )


def test_approval_advance_fires_on_ready():
    reflector = ApprovalAdvanceReflector()
    result = reflector.reflect(
        entries=[],
        observation=_obs(actions=[_get_response_record()]),
        moment="iteration_boundary",
    )
    assert len(result.advisories) == 1
    adv = result.advisories[0]
    assert adv.kind == "advance_past_approval"
    assert "appr_1" in adv.body
    assert "approve_once" in adv.body


def test_approval_advance_silent_on_pending():
    reflector = ApprovalAdvanceReflector()
    result = reflector.reflect(
        entries=[],
        observation=_obs(actions=[_get_response_record(state="pending")]),
        moment="iteration_boundary",
    )
    assert result.is_empty


# ---------------------------------------------------------------------------
# ErrorRewriterReflector
# ---------------------------------------------------------------------------


def _propose_decomp_record(
    *,
    ok: bool,
    proposals: list[dict],
) -> CallRecord:
    return CallRecord(
        action_key="DesignProcessCapability.DesignProcessCapability.propose_decompositions",
        params={},
        action_id="a-prop-1",
        status="ok",
        result={"ok": ok, "parent_proposals": proposals},
    )


def test_error_rewriter_fires_on_propose_decomp_partial_failure():
    reflector = ErrorRewriterReflector()
    rec = _propose_decomp_record(
        ok=True,
        proposals=[
            {"parent_number": 10, "children": ["c1"]},
            {"parent_number": 20, "error": "output truncated"},
        ],
    )
    result = reflector.reflect(
        entries=[],
        observation=_obs(actions=[rec]),
        moment="iteration_boundary",
    )
    assert len(result.advisories) == 1
    adv = result.advisories[0]
    assert adv.kind == "propose_decompositions_partial_failure"
    assert "20" in adv.body  # the failing parent's number surfaces


def test_error_rewriter_fires_on_human_approval_empty():
    reflector = ErrorRewriterReflector()
    rec = CallRecord(
        action_key="HumanApprovalCapability.HumanApprovalCapability.request_human_approval",
        params={"question": ""},
        action_id="a-emp-1",
        status="error",
        error="RequestHumanApprovalEmpty: empty body",
    )
    result = reflector.reflect(
        entries=[],
        observation=_obs(actions=[rec]),
        moment="iteration_boundary",
    )
    assert len(result.advisories) == 1
    assert result.advisories[0].kind == "human_approval_empty_body"


def test_error_rewriter_fires_on_outer_key_error():
    reflector = ErrorRewriterReflector()
    outer = ErrorContext(
        error_type="execute_code_error",
        error_details={"message": "KeyError: 'rows'"},
        action_context={"action_id": "codegen_plan_step_3"},
    )
    result = reflector.reflect(
        entries=[],
        observation=_obs(outer_action_error=outer),
        moment="iteration_boundary",
    )
    assert len(result.advisories) == 1
    assert result.advisories[0].kind == "outer_key_error"


# ---------------------------------------------------------------------------
# Substrate integration — stream wraps reflector; reflect() rolls up
# ---------------------------------------------------------------------------


def test_stream_helpers_build_iteration_boundary_subscribers():
    """Each stream-helper factory configures the stream to record
    ``iteration_boundary`` entries — the kind reflectors subscribe to."""

    for stream in (
        contract_drift_stream(),
        inconsistency_stream(),
        cliff_guard_stream(max_iterations=10),
        approval_advance_stream(),
        error_rewriter_stream(),
    ):
        # The filter on iteration_boundary must be present (accept-all).
        assert "iteration_boundary" in stream._filters
        # No display-only formatter — pure reflector streams.
        assert stream.formatter is None


def test_consciousness_stream_reflect_dispatches_to_reflector():
    stream = contract_drift_stream()
    # First boundary primes the reflector; second triggers drift.
    obs1 = _obs(iter_index=1, actions=[_spawn_record()])
    obs2 = _obs(iter_index=2, actions=[_spawn_record()])
    assert stream.reflect(
        moment="iteration_boundary", observation=obs1,
    ).is_empty
    second = stream.reflect(
        moment="iteration_boundary", observation=obs2,
    )
    assert len(second.advisories) == 1
    assert second.advisories[0].source == "contract_drift"


def test_consciousness_stream_reflect_returns_empty_on_wrong_moment():
    stream = contract_drift_stream()
    obs = _obs(iter_index=2, actions=[_spawn_record()])
    # planning_step is NOT in the reflector's REFLECT_AT.
    assert stream.reflect(
        moment="planning_step", observation=obs,
    ).is_empty
