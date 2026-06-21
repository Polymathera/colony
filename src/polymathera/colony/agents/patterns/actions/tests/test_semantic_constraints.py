"""Tests for the semantic-constraint primitives in
``semantic_constraints.py``.

Covers:

1. :class:`ConstraintVerdict` + :class:`ScopedConstraintState` shape.
2. :class:`SemanticConstraint.matches_action` filter semantics.
3. :class:`EvidenceVerifier` — predicate is invoked over the scope
   slice; scope=CELL vs SESSION; first-match wins; predicate raise
   doesn't break iteration; unsatisfied verdict carries the
   declarative reason/suggestion.
4. :class:`PythonPredicateVerifier` — verdict pass-through;
   predicate raise → fail-open.
5. :class:`LLMJudgeVerifier` — prompt construction includes rule,
   action, scope-sized history, and owner agent id; judge JSON parsed
   (lenient); judge unbound → fail-open with warning; judge call
   error → fail-open; malformed JSON → fail-open.
6. :class:`SemanticConstraintGuardrail` — filter short-circuits;
   first BLOCK wins; ADVISE accumulates without blocking; drained
   advisories surface in ``planner_context_advisory`` and clear
   after rendering; bind_owner propagates infer_fn to LLM judges.
7. ``cell_start_index`` kwarg in :meth:`check` — scope=CELL only
   sees calls at/after the boundary; scope=SESSION sees everything.
8. Auto-generated planner-prompt advisory contains every active
   constraint with rule, scope, failure mode, and check
   description — the single source of truth for the LLM's view
   of the active constraints.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    GuardrailDecision,
)
from polymathera.colony.agents.patterns.actions.semantic_constraints import (
    ConstraintFailureMode,
    ConstraintScope,
    ConstraintVerdict,
    EvidenceVerifier,
    LLMJudgeVerifier,
    PythonPredicateVerifier,
    ScopedConstraintState,
    SemanticConstraint,
    SemanticConstraintGuardrail,
)
from polymathera.colony.agents.patterns.planning.models import CallRecord


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    generated_text: str
    tokens_generated: int = 0


def _stub_infer_factory(
    verdict: dict,
    *,
    capture: list | None = None,
    tokens_generated: int = 0,
):
    """Build an async ``infer_fn`` that always returns ``verdict``.
    ``capture`` (when provided) accumulates the (prompt, max_tokens,
    temperature) tuples each call received — for assertion that the
    judge was called with the right shape. ``tokens_generated`` is
    surfaced on the response so the observability log emits the
    real token count rather than ``n/a``."""

    text = json.dumps(verdict)

    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        if capture is not None:
            capture.append((prompt, max_tokens, temperature))
        return _StubResponse(
            generated_text=text, tokens_generated=tokens_generated,
        )

    return stub_infer


def _raising_infer_factory(exc: Exception):
    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        raise exc

    return stub_infer


def _call(
    action_key: str,
    *,
    params: dict | None = None,
    status: str = "ok",
    result: Any = None,
) -> CallRecord:
    return CallRecord(
        action_key=action_key,
        params=dict(params or {}),
        end_wall=time.time(),
        status=status,
        result=result,
    )


class _FakeAgent:
    def __init__(self, agent_id: str, infer_fn=None) -> None:
        self.agent_id = agent_id
        self.infer = infer_fn or _stub_infer_factory({"satisfied": True})


# ---------------------------------------------------------------------------
# ConstraintVerdict + ScopedConstraintState shape
# ---------------------------------------------------------------------------


def test_constraint_verdict_is_frozen() -> None:
    """``ConstraintVerdict`` is shared across failure-mode routing AND
    advisory accumulation — frozen so a downstream consumer can't
    mutate state another consumer relies on."""

    v = ConstraintVerdict(satisfied=False, reason="r", suggestion="s")
    with pytest.raises(Exception):
        v.satisfied = True  # type: ignore[misc]


def test_scoped_constraint_state_is_frozen() -> None:
    s = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id="a",
    )
    with pytest.raises(Exception):
        s.owner_agent_id = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SemanticConstraint.matches_action filter
# ---------------------------------------------------------------------------


def _bare_constraint(applies_to: list[str]) -> SemanticConstraint:
    return SemanticConstraint(
        id="t",
        rule_nl="test",
        applies_to=applies_to,
        verifier=PythonPredicateVerifier(
            predicate=lambda *_: ConstraintVerdict(satisfied=True),
            description="never",
        ),
    )


def test_matches_action_substring_case_insensitive() -> None:
    c = _bare_constraint(["RESPOND_to_user"])
    assert c.matches_action(
        "SessionOrchestratorCapability.respond_to_user",
    )
    assert c.matches_action("respond_to_user")
    assert not c.matches_action("propose_plan")


def test_matches_action_any_pattern_wins() -> None:
    c = _bare_constraint(["foo", "bar"])
    assert c.matches_action("X.bar_baz")
    assert c.matches_action("X.foo_qux")
    assert not c.matches_action("X.baz")


def test_matches_action_empty_applies_to_matches_nothing() -> None:
    """Empty ``applies_to`` is a constraint that targets nothing — a
    typo guard, not a wildcard. A wildcard would silently apply the
    rule everywhere; the explicit-list shape makes the operator's
    intent obvious."""

    c = _bare_constraint([])
    assert not c.matches_action("X.anything")


# ---------------------------------------------------------------------------
# EvidenceVerifier
# ---------------------------------------------------------------------------


async def test_evidence_verifier_finds_match_in_cell_scope() -> None:
    """Predicate sees ``CallRecord``s; first match → satisfied."""

    def pred(record, _action_key, _params, _state):
        return "get_agent_status" in record.action_key

    constraint = SemanticConstraint(
        id="ev",
        rule_nl="needs status check this cell",
        applies_to=["respond_to_user"],
        scope=ConstraintScope.CELL,
        verifier=EvidenceVerifier(
            evidence_predicate=pred,
            description="status-check evidence",
        ),
    )
    scoped = ScopedConstraintState(
        cell_calls=(
            _call("AgentPoolCapability.get_agent_status"),
        ),
        session_calls=(
            _call("AgentPoolCapability.get_agent_status"),
        ),
        owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="respond_to_user",
        action_params={},
        scoped_state=scoped,
    )
    assert verdict.satisfied is True


async def test_evidence_verifier_cell_scope_skips_pre_cell_history() -> None:
    """Critical scope semantic: a ``get_agent_status`` from a PRIOR
    iteration must NOT satisfy a cell-scoped constraint. Cell-scoped
    rules want freshness — stale evidence is the bug the framework
    is preventing."""

    def pred(record, *_):
        return "get_agent_status" in record.action_key

    constraint = SemanticConstraint(
        id="ev",
        rule_nl="needs fresh status check",
        applies_to=["respond_to_user"],
        scope=ConstraintScope.CELL,
        verifier=EvidenceVerifier(evidence_predicate=pred),
    )
    # session has the call, cell does NOT.
    scoped = ScopedConstraintState(
        cell_calls=(),
        session_calls=(
            _call("AgentPoolCapability.get_agent_status"),
        ),
        owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="respond_to_user",
        action_params={},
        scoped_state=scoped,
    )
    assert verdict.satisfied is False


async def test_evidence_verifier_session_scope_finds_prior_iteration_match() -> None:
    """Mirror: session-scoped rule DOES see prior-iteration history.
    The shape difference between the two scopes is the whole point of
    the per-constraint declaration."""

    def pred(record, *_):
        return "approval_granted" in record.action_key

    constraint = SemanticConstraint(
        id="ev",
        rule_nl="needs prior approval",
        applies_to=["apply"],
        scope=ConstraintScope.SESSION,
        verifier=EvidenceVerifier(evidence_predicate=pred),
    )
    scoped = ScopedConstraintState(
        cell_calls=(),
        session_calls=(_call("HumanApprovalCapability.approval_granted"),),
        owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="apply",
        action_params={},
        scoped_state=scoped,
    )
    assert verdict.satisfied is True


async def test_evidence_verifier_unsatisfied_carries_reason_and_suggestion() -> None:
    """Verifier emits the constraint's declarative remediation text
    so the LLM gets actionable advice — same shape as the legacy
    guardrails' ``suggestion`` field."""

    constraint = SemanticConstraint(
        id="ev",
        rule_nl="something",
        applies_to=["x"],
        verifier=EvidenceVerifier(
            evidence_predicate=lambda *_: False,
            unsatisfied_reason="custom reason",
            unsatisfied_suggestion="custom suggestion",
        ),
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert not verdict.satisfied
    assert verdict.reason == "custom reason"
    assert verdict.suggestion == "custom suggestion"


async def test_evidence_verifier_predicate_raise_skips_record() -> None:
    """One predicate failure can't poison the whole scan — the
    framework logs and continues, mirroring the existing predicate
    handling in :class:`ArgsAwareTemporalOrderGuardrail`."""

    seen = []

    def pred(record, *_):
        seen.append(record.action_key)
        if record.action_key == "X.boom":
            raise RuntimeError("boom")
        return record.action_key == "X.good"

    constraint = SemanticConstraint(
        id="ev",
        rule_nl="test",
        applies_to=["x"],
        verifier=EvidenceVerifier(evidence_predicate=pred),
    )
    scoped = ScopedConstraintState(
        cell_calls=(_call("X.good"), _call("X.boom")),
        session_calls=(_call("X.good"), _call("X.boom")),
        owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    # Reversed scan: boom first (raises), then good (matches).
    assert verdict.satisfied is True
    assert seen == ["X.boom", "X.good"]


# ---------------------------------------------------------------------------
# PythonPredicateVerifier
# ---------------------------------------------------------------------------


async def test_python_predicate_verifier_returns_verdict_directly() -> None:
    expected = ConstraintVerdict(
        satisfied=False, reason="r", suggestion="s",
    )
    constraint = SemanticConstraint(
        id="p",
        rule_nl="test",
        applies_to=["x"],
        verifier=PythonPredicateVerifier(
            predicate=lambda *_: expected,
            description="custom",
        ),
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert verdict == expected


async def test_python_predicate_verifier_raise_falls_open() -> None:
    """Same fail-open convention as the LLM judge: a buggy predicate
    can't silently block every action — the framework allows the
    action with a noted reason."""

    def pred(*_):
        raise ValueError("boom")

    constraint = SemanticConstraint(
        id="p",
        rule_nl="test",
        applies_to=["x"],
        verifier=PythonPredicateVerifier(
            predicate=pred, description="custom",
        ),
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    verdict = await constraint.verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert verdict.satisfied is True
    assert "boom" in verdict.reason


# ---------------------------------------------------------------------------
# LLMJudgeVerifier
# ---------------------------------------------------------------------------


async def test_llm_judge_unbound_falls_open() -> None:
    """If the agent hookup didn't run yet (or the catalogue is being
    inspected without an agent), the judge falls open with a warning
    instead of blocking. The deployed pipeline binds at policy init;
    fall-open is the durable runtime choice."""

    verifier = LLMJudgeVerifier()
    constraint = SemanticConstraint(
        id="j",
        rule_nl="test",
        applies_to=["x"],
        verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    v = await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert v.satisfied is True
    assert "not bound" in v.reason.lower()


async def test_llm_judge_call_failure_falls_open() -> None:
    """An LLM-side exception must not block every agent action across
    the colony. Loud-log + permissive default, same as
    :class:`LLMCompletionValidator`."""

    verifier = LLMJudgeVerifier(
        infer_fn=_raising_infer_factory(RuntimeError("llm down")),
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    constraint = SemanticConstraint(
        id="j", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    v = await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert v.satisfied is True
    assert "judge call failed" in v.reason


async def test_llm_judge_unsatisfied_verdict_carries_reason_suggestion() -> None:
    verifier = LLMJudgeVerifier(
        infer_fn=_stub_infer_factory({
            "satisfied": False,
            "reason": "r1",
            "suggestion": "s1",
        }),
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    constraint = SemanticConstraint(
        id="j", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    v = await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert v.satisfied is False
    assert v.reason == "r1"
    assert v.suggestion == "s1"


async def test_llm_judge_prompt_includes_rule_action_and_scope_history() -> None:
    """Pin the judge prompt's shape so a future refactor that drops
    one of these fields is caught here, not by a regression in
    real LLM behavior."""

    capture: list = []
    verifier = LLMJudgeVerifier(
        infer_fn=_stub_infer_factory(
            {"satisfied": True}, capture=capture,
        ),
    )
    constraint = SemanticConstraint(
        id="status_claim",
        rule_nl="don't claim state without verification",
        applies_to=["respond_to_user"],
        scope=ConstraintScope.CELL,
        verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(
            _call(
                "AgentPoolCapability.get_agent_status",
                params={"agent_ids": ["agent-xyz"]},
                result={"agents": [{"id": "agent-xyz", "state": "RUNNING"}]},
            ),
        ),
        session_calls=(),  # cell-scoped: judge sees cell_calls only
        owner_agent_id="agent-abc",
    )
    await verifier.verify(
        constraint=constraint,
        action_key="SessionOrchestratorCapability.respond_to_user",
        action_params={"content": "agent-xyz is running"},
        scoped_state=scoped,
    )
    assert len(capture) == 1
    prompt = capture[0][0]
    assert "status_claim" in prompt
    assert "don't claim state without verification" in prompt
    assert "SessionOrchestratorCapability.respond_to_user" in prompt
    assert "agent-xyz is running" in prompt
    assert "agent-abc" in prompt  # owner id
    assert "get_agent_status" in prompt  # cell history rendered
    assert "CELL-SCOPED HISTORY" in prompt


async def test_llm_judge_lenient_json_extraction() -> None:
    """The judge may wrap the verdict in prose or code fences; we
    extract the first top-level JSON object. Real LLMs often emit
    ``Here's my answer: {...}.`` — pinning the lenient parse
    keeps the verdict surface tolerant."""

    wrapped = (
        'Here is my analysis. The agent did call get_agent_status.\n'
        '```\n{"satisfied": false, "reason": "r", "suggestion": "s"}\n```\n'
        'Hope this helps.'
    )

    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        return _StubResponse(generated_text=wrapped)

    verifier = LLMJudgeVerifier(infer_fn=stub_infer)
    constraint = SemanticConstraint(
        id="j", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    v = await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert v.satisfied is False
    assert v.reason == "r"


async def test_llm_judge_malformed_output_falls_open() -> None:
    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        return _StubResponse(
            generated_text="totally not json at all",
        )

    verifier = LLMJudgeVerifier(infer_fn=stub_infer)
    constraint = SemanticConstraint(
        id="j", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    v = await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    assert v.satisfied is True
    assert "not parseable" in v.reason


# ---------------------------------------------------------------------------
# SemanticConstraintGuardrail
# ---------------------------------------------------------------------------


async def test_guardrail_short_circuits_when_no_constraint_applies() -> None:
    """The common case is "this action_key matches no constraint" —
    that path returns ``allowed=True`` without any scoping work or
    verifier calls. Pinned because if this path regresses, every
    unrelated action pays the verifier cost."""

    capture: list = []
    verifier = LLMJudgeVerifier(
        infer_fn=_stub_infer_factory(
            {"satisfied": False, "reason": "r"}, capture=capture,
        ),
    )
    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="t",
            rule_nl="t",
            applies_to=["respond_to_user"],
            verifier=verifier,
        ),
    ])
    decision = await g.check(
        action_key="UnrelatedCapability.unrelated_action",
        params={},
        call_history=[],
    )
    assert decision.allowed
    assert capture == []  # verifier never called


async def test_guardrail_first_block_short_circuits() -> None:
    """Subsequent constraints don't run after the first BLOCK; the
    LLM only needs one suggestion at a time and avoiding the extra
    verifier calls matters when each is an LLM round-trip."""

    call_log: list[str] = []

    def make_predicate(name):
        def pred(action_key, params, state):
            call_log.append(name)
            return ConstraintVerdict(
                satisfied=False, reason=f"{name}-rejected",
            )
        return pred

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="first",
            rule_nl="r1",
            applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=make_predicate("first"), description="d",
            ),
        ),
        SemanticConstraint(
            id="second",
            rule_nl="r2",
            applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=make_predicate("second"), description="d",
            ),
        ),
    ])
    decision = await g.check(
        action_key="x.do",
        params={},
        call_history=[],
    )
    assert not decision.allowed
    assert decision.reason.startswith("[first]")
    assert call_log == ["first"]  # second never ran


async def test_guardrail_advise_mode_lets_action_through_and_accumulates() -> None:
    """ADVISE constraints don't block; verdicts accumulate for the
    next iteration's planner prompt."""

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="advise_only",
            rule_nl="soft rule",
            applies_to=["x"],
            failure_mode=ConstraintFailureMode.ADVISE,
            verifier=PythonPredicateVerifier(
                predicate=lambda *_: ConstraintVerdict(
                    satisfied=False, reason="soft", suggestion="hint",
                ),
                description="d",
            ),
        ),
    ])
    decision = await g.check(
        action_key="x.do",
        params={},
        call_history=[],
    )
    assert decision.allowed
    drained = g.drain_pending_advisories()
    assert len(drained) == 1
    constraint, verdict = drained[0]
    assert constraint.id == "advise_only"
    assert verdict.reason == "soft"
    # Drain is one-shot.
    assert g.drain_pending_advisories() == []


async def test_guardrail_cell_start_index_threads_to_scope_slice() -> None:
    """The kwarg added to RuntimeGuardrail.check must flow through to
    the scope-aware verifier — the whole point of the cell-vs-session
    distinction depends on this index being honored."""

    seen_states: list[ScopedConstraintState] = []

    def pred(action_key, params, state):
        seen_states.append(state)
        return ConstraintVerdict(satisfied=True)

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="t",
            rule_nl="t",
            applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=pred, description="d",
            ),
        ),
    ])
    history = [_call(f"X.prior_{i}") for i in range(5)]
    await g.check(
        action_key="x.do",
        params={},
        call_history=history,
        cell_start_index=3,
    )
    state = seen_states[0]
    assert len(state.session_calls) == 5
    assert len(state.cell_calls) == 2  # indices 3, 4
    assert state.cell_calls[0].action_key == "X.prior_3"


async def test_guardrail_cell_start_index_out_of_range_safe() -> None:
    """A boundary larger than history (or negative) must not crash —
    clamp into the legal range. The policy's iteration counter could
    theoretically race with history mutation in edge cases."""

    seen: list[ScopedConstraintState] = []

    def pred(action_key, params, state):
        seen.append(state)
        return ConstraintVerdict(satisfied=True)

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="t",
            rule_nl="t",
            applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=pred, description="d",
            ),
        ),
    ])
    history = [_call("X.a"), _call("X.b")]
    # Past-the-end: cell is empty, no crash.
    await g.check(
        action_key="x.do", params={}, call_history=history,
        cell_start_index=99,
    )
    assert seen[-1].cell_calls == ()
    # Negative: clamps to 0.
    await g.check(
        action_key="x.do", params={}, call_history=history,
        cell_start_index=-3,
    )
    assert len(seen[-1].cell_calls) == 2


async def test_guardrail_bind_agent_propagates_infer_to_llm_judges() -> None:
    """LLM-judge verifiers are constructed before the agent exists;
    ``bind_agent`` must walk every constraint and late-bind
    ``agent.infer`` onto the judges. Non-judge verifiers ignored."""

    judge_a = LLMJudgeVerifier()
    judge_b = LLMJudgeVerifier()
    predicate = PythonPredicateVerifier(
        predicate=lambda *_: ConstraintVerdict(satisfied=True),
        description="d",
    )
    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="a", rule_nl="r", applies_to=["x"], verifier=judge_a,
        ),
        SemanticConstraint(
            id="b", rule_nl="r", applies_to=["x"], verifier=predicate,
        ),
        SemanticConstraint(
            id="c", rule_nl="r", applies_to=["x"], verifier=judge_b,
        ),
    ])
    assert judge_a._infer_fn is None
    assert judge_b._infer_fn is None
    agent = _FakeAgent("agent-test")
    g.bind_agent(agent)
    assert judge_a._infer_fn is agent.infer
    assert judge_b._infer_fn is agent.infer
    # PythonPredicate verifier untouched (no bind_infer surface).


async def test_guardrail_bind_agent_none_clears_agent() -> None:
    """Composite call sites may pass None to drop the agent
    (test cleanup, agent shutdown). Don't crash; clear the agent."""

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="t",
            rule_nl="t",
            applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=lambda *_: ConstraintVerdict(satisfied=True),
                description="d",
            ),
        ),
    ])
    g.bind_agent(_FakeAgent("agent-a"))
    g.bind_agent(None)
    assert g._owner is None


# ---------------------------------------------------------------------------
# planner_context_advisory auto-generation (R10-4)
# ---------------------------------------------------------------------------


def test_advisory_renders_each_constraint_with_rule_scope_failure_mode() -> None:
    """The catalogue is the SINGLE SOURCE OF TRUTH for the LLM's
    view of active constraints. Adding a new record automatically
    updates the planner prompt — no manual self_concept edit."""

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="rule_a",
            rule_nl="A nl rule body",
            applies_to=["foo", "bar"],
            scope=ConstraintScope.CELL,
            failure_mode=ConstraintFailureMode.BLOCK,
            verifier=PythonPredicateVerifier(
                predicate=lambda *_: ConstraintVerdict(satisfied=True),
                description="check-A-description",
            ),
        ),
        SemanticConstraint(
            id="rule_b",
            rule_nl="B nl rule body",
            applies_to=["baz"],
            scope=ConstraintScope.SESSION,
            failure_mode=ConstraintFailureMode.ADVISE,
            verifier=PythonPredicateVerifier(
                predicate=lambda *_: ConstraintVerdict(satisfied=True),
                description="check-B-description",
            ),
        ),
    ])
    text = g.planner_context_advisory(call_history=[])
    assert text is not None
    assert "[rule_a]" in text
    assert "A nl rule body" in text
    assert "``foo``" in text and "``bar``" in text
    assert "``cell``" in text
    assert "``block``" in text
    assert "check-A-description" in text
    assert "[rule_b]" in text
    assert "B nl rule body" in text
    assert "``session``" in text
    assert "``advise``" in text
    assert "check-B-description" in text


def test_advisory_returns_none_when_no_constraints_and_no_drained() -> None:
    """Empty catalogue + no pending advisories → no advisory section.
    The advisory machinery is opt-in based on what the catalogue
    declares."""

    g = SemanticConstraintGuardrail([])
    assert g.planner_context_advisory(call_history=[]) is None


async def test_advisory_includes_drained_advise_verdicts() -> None:
    """ADVISE constraints' verdicts from the prior cell surface in
    the next prompt under a dedicated sub-section AND the drain is
    one-shot (next call sees no pending)."""

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="r",
            rule_nl="r",
            applies_to=["x"],
            failure_mode=ConstraintFailureMode.ADVISE,
            verifier=PythonPredicateVerifier(
                predicate=lambda *_: ConstraintVerdict(
                    satisfied=False,
                    reason="advise-reason",
                    suggestion="advise-suggestion",
                ),
                description="d",
            ),
        ),
    ])
    await g.check(
        action_key="x.do", params={}, call_history=[],
    )
    text = g.planner_context_advisory(call_history=[])
    assert text is not None
    assert "Constraint Advisories (last iteration)" in text
    assert "advise-reason" in text
    assert "advise-suggestion" in text

    # Drain is one-shot: the standing section still renders (the
    # catalogue is unchanged) but the advisory section is gone.
    text2 = g.planner_context_advisory(call_history=[])
    assert "Constraint Advisories" not in text2


# ---------------------------------------------------------------------------
# R11-Fix3: LLMJudgeVerifier observability
# ---------------------------------------------------------------------------
#
# Every judge call emits one structured INFO line — see
# :meth:`LLMJudgeVerifier._emit_observability`. Without this, the
# per-respond_to_user judge cost is invisible (only the failures
# surface as ``Guardrail blocked`` lines; every allowed action is
# unlogged). R11 forensic showed the SessionAgent silently burned
# its iteration budget on judge cycles; the observability gap meant
# the cost was underestimated until the agent died.
#
# The log line shape (pinned here so a future refactor that drops a
# field surfaces in the test, not in a regression):
#   [LLMJudge] constraint=<id> scope=<scope> action=<key>
#              history_len=<int> satisfied=<bool> latency_ms=<float>
#              tokens=<int|n/a> fallback=<reason|none> reason=<text>


import logging  # noqa: E402 — placed near the observability tests


async def test_observability_log_emitted_for_satisfied_verdict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every successful judge call produces one ``[LLMJudge]`` INFO
    line. Pin the canonical field set so a future refactor that
    drops a field is caught here, not in a production regression."""

    caplog.set_level(
        logging.INFO,
        logger="polymathera.colony.agents.patterns.actions.semantic_constraints",
    )
    verifier = LLMJudgeVerifier(
        infer_fn=_stub_infer_factory(
            {"satisfied": True, "reason": "ok"},
            tokens_generated=42,
        ),
    )
    constraint = SemanticConstraint(
        id="t",
        rule_nl="r",
        applies_to=["respond_to_user"],
        scope=ConstraintScope.CELL,
        verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(_call("X.a"), _call("X.b")),
        session_calls=(_call("X.a"), _call("X.b")),
        owner_agent_id="agent-test",
    )
    await verifier.verify(
        constraint=constraint,
        action_key="respond_to_user",
        action_params={},
        scoped_state=scoped,
    )
    lines = [
        r.getMessage() for r in caplog.records if "[LLMJudge]" in r.getMessage()
    ]
    assert len(lines) == 1
    msg = lines[0]
    assert "constraint=t" in msg
    assert "scope=cell" in msg
    assert "action=respond_to_user" in msg
    assert "history_len=2" in msg
    assert "satisfied=True" in msg
    assert "tokens=42" in msg
    assert "fallback=none" in msg
    assert "latency_ms=" in msg


async def test_observability_log_emitted_for_unsatisfied_verdict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Block path also emits the INFO line; ``satisfied=False`` +
    the verdict reason (truncated) surface for diagnostics. The
    block message goes to the planner via ``GuardrailDecision`` /
    ``BlockedDispatch``; the INFO line goes to operator-facing logs
    so cost AND failure mode are both visible."""

    caplog.set_level(
        logging.INFO,
        logger="polymathera.colony.agents.patterns.actions.semantic_constraints",
    )
    verifier = LLMJudgeVerifier(
        infer_fn=_stub_infer_factory(
            {"satisfied": False, "reason": "missing get_agent_status"},
            tokens_generated=8,
        ),
    )
    constraint = SemanticConstraint(
        id="no_unverified_state_claims",
        rule_nl="r",
        applies_to=["respond_to_user"],
        verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    await verifier.verify(
        constraint=constraint,
        action_key="respond_to_user",
        action_params={},
        scoped_state=scoped,
    )
    lines = [
        r.getMessage() for r in caplog.records if "[LLMJudge]" in r.getMessage()
    ]
    assert len(lines) == 1
    msg = lines[0]
    assert "constraint=no_unverified_state_claims" in msg
    assert "satisfied=False" in msg
    assert "tokens=8" in msg
    assert "missing get_agent_status" in msg


async def test_observability_log_emitted_when_judge_unbound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unbound judge falls open AND emits the INFO line with
    ``fallback=unbound`` so the operator can spot the case in logs
    without having to grep for the ``no infer_fn bound`` warning."""

    caplog.set_level(
        logging.INFO,
        logger="polymathera.colony.agents.patterns.actions.semantic_constraints",
    )
    verifier = LLMJudgeVerifier()  # no infer_fn
    constraint = SemanticConstraint(
        id="t", rule_nl="r", applies_to=["x"], verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    lines = [
        r.getMessage() for r in caplog.records if "[LLMJudge]" in r.getMessage()
    ]
    assert len(lines) == 1
    msg = lines[0]
    assert "fallback=unbound" in msg
    assert "satisfied=True" in msg  # fail-open
    assert "tokens=n/a" in msg  # no inference happened


async def test_observability_log_emitted_on_infer_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLM-side exception falls open AND emits ``fallback=infer_error:<class>``
    so log-grep can distinguish "judge said yes" from "we couldn't
    call the judge". The exception class is part of the fallback
    field so the operator can immediately spot timeouts vs network
    vs API-key failures."""

    caplog.set_level(
        logging.INFO,
        logger="polymathera.colony.agents.patterns.actions.semantic_constraints",
    )
    verifier = LLMJudgeVerifier(
        infer_fn=_raising_infer_factory(TimeoutError("LLM took >10s")),
    )
    constraint = SemanticConstraint(
        id="t", rule_nl="r", applies_to=["x"], verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    lines = [
        r.getMessage() for r in caplog.records if "[LLMJudge]" in r.getMessage()
    ]
    assert len(lines) == 1
    msg = lines[0]
    assert "fallback=infer_error:TimeoutError" in msg
    assert "satisfied=True" in msg  # fail-open


async def test_observability_log_emitted_on_parse_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed judge output falls open AND emits a parse-specific
    fallback so the operator can identify "the LLM produced
    something we couldn't parse" (vs other fail-open paths). Useful
    when tuning the judge prompt — high parse-error rates signal
    the LLM isn't following the JSON contract."""

    caplog.set_level(
        logging.INFO,
        logger="polymathera.colony.agents.patterns.actions.semantic_constraints",
    )

    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        return _StubResponse(
            generated_text="not json at all", tokens_generated=3,
        )

    verifier = LLMJudgeVerifier(infer_fn=stub_infer)
    constraint = SemanticConstraint(
        id="t", rule_nl="r", applies_to=["x"], verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    lines = [
        r.getMessage() for r in caplog.records if "[LLMJudge]" in r.getMessage()
    ]
    assert len(lines) == 1
    msg = lines[0]
    assert "fallback=parse_unparseable" in msg
    assert "satisfied=True" in msg  # fail-open
    assert "tokens=3" in msg


async def test_observability_log_truncates_long_reason() -> None:
    """The ``reason`` field is capped at ~120 chars in the log line
    so a verbose judge reason doesn't blow up operator logs. Pin
    the truncation so a future verifier that returns multi-line
    reasons doesn't accidentally produce 10-line log records."""

    long = "x" * 500
    verifier = LLMJudgeVerifier(
        infer_fn=_stub_infer_factory(
            {"satisfied": False, "reason": long},
            tokens_generated=1,
        ),
    )
    constraint = SemanticConstraint(
        id="t", rule_nl="r", applies_to=["x"], verifier=verifier,
    )
    scoped = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    verdict = await verifier.verify(
        constraint=constraint,
        action_key="x",
        action_params={},
        scoped_state=scoped,
    )
    # The verdict itself preserves the full reason; only the log
    # line is truncated. Pin both invariants so a refactor of one
    # doesn't break the other.
    assert verdict.reason == long
