"""Tests for PR3 (R12-D + B2) — judge robustness.

Three primitives wired into the existing semantic-constraint surface:

1. **Verdict cache** on :class:`LLMJudgeVerifier` — per-(rule,
   action, content, history-fingerprint) LRU. Run12 made 46 judge
   calls on near-identical drafts; the cache collapses repeats to 1.
2. **Streak escalation** on :class:`SemanticConstraintGuardrail` —
   after N consecutive BLOCKs on the SAME (rule, content), the rule
   is downgraded to ADVISE for the rest of the session so the LLM
   stops retrying the same blocked draft forever.
3. **Precondition pre-filter** on :class:`SemanticConstraint` —
   cheap structural check runs BEFORE the verifier; False short-
   circuits to ``satisfied=True`` without the LLM call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest

from polymathera.colony.agents.patterns.actions.semantic_constraints import (
    ConstraintFailureMode,
    ConstraintScope,
    ConstraintVerdict,
    LLMJudgeVerifier,
    PythonPredicateVerifier,
    ScopedConstraintState,
    SemanticConstraint,
    SemanticConstraintGuardrail,
    _content_fingerprint,
    _judge_cache_key,
)
from polymathera.colony.agents.patterns.planning.models import CallRecord


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    generated_text: str
    tokens_generated: int = 0


def _make_infer_factory(verdict: dict, *, capture: list | None = None):
    text = json.dumps(verdict)

    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        if capture is not None:
            capture.append(prompt)
        return _StubResponse(generated_text=text, tokens_generated=1)

    return stub_infer


def _bare_scope() -> ScopedConstraintState:
    return ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id="agent-owner",
    )


# ---------------------------------------------------------------------------
# 1. Verdict cache
# ---------------------------------------------------------------------------


async def test_judge_cache_collapses_identical_calls() -> None:
    """Same (rule, action, content, history) → one LLM call;
    the second verify hits the cache."""

    capture: list = []
    verifier = LLMJudgeVerifier(
        infer_fn=_make_infer_factory(
            {"satisfied": False, "reason": "no", "suggestion": "do X"},
            capture=capture,
        ),
    )
    constraint = SemanticConstraint(
        id="r", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    for _ in range(5):
        v = await verifier.verify(
            constraint=constraint,
            action_key="x",
            action_params={"content": "same"},
            scoped_state=_bare_scope(),
        )
        assert v.satisfied is False
    assert len(capture) == 1  # 4 cache hits, 1 fresh call


async def test_judge_cache_misses_when_history_changes() -> None:
    """Adding a call to the scoped history flips the cache key so
    a fresh judge call fires. Without this, a get_agent_status that
    NOW satisfies the rule would still return the cached unsatisfied
    verdict."""

    capture: list = []
    verifier = LLMJudgeVerifier(
        infer_fn=_make_infer_factory(
            {"satisfied": True, "reason": "ok"}, capture=capture,
        ),
    )
    constraint = SemanticConstraint(
        id="r", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    s1 = ScopedConstraintState(
        cell_calls=(), session_calls=(), owner_agent_id=None,
    )
    s2 = ScopedConstraintState(
        cell_calls=(
            CallRecord(
                action_key="get_agent_status", params={},
                end_wall=time.time(), status="ok",
            ),
        ),
        session_calls=(
            CallRecord(
                action_key="get_agent_status", params={},
                end_wall=time.time(), status="ok",
            ),
        ),
        owner_agent_id=None,
    )
    await verifier.verify(
        constraint=constraint, action_key="x",
        action_params={"content": "same"}, scoped_state=s1,
    )
    await verifier.verify(
        constraint=constraint, action_key="x",
        action_params={"content": "same"}, scoped_state=s2,
    )
    assert len(capture) == 2  # history differs → different keys


async def test_judge_cache_does_not_cache_parse_failures() -> None:
    """Fail-open parse errors are transient — don't pin a bad
    verdict; let the next call get a fresh shot at the judge."""

    capture: list = []

    async def stub_infer(*, prompt, max_tokens=None, temperature=None):
        capture.append(prompt)
        return _StubResponse(
            generated_text="not json", tokens_generated=1,
        )

    verifier = LLMJudgeVerifier(infer_fn=stub_infer)
    constraint = SemanticConstraint(
        id="r", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    for _ in range(3):
        await verifier.verify(
            constraint=constraint, action_key="x",
            action_params={"content": "same"}, scoped_state=_bare_scope(),
        )
    assert len(capture) == 3  # no caching of parse_unparseable


async def test_judge_cache_emits_cache_hit_in_observability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cache hits surface in the [LLMJudge] observability line with
    ``fallback=cache_hit`` so cost analyses can distinguish."""

    import logging
    caplog.set_level(
        logging.INFO,
        logger="polymathera.colony.agents.patterns.actions.semantic_constraints",
    )
    verifier = LLMJudgeVerifier(
        infer_fn=_make_infer_factory({"satisfied": True, "reason": "ok"}),
    )
    constraint = SemanticConstraint(
        id="r", rule_nl="t", applies_to=["x"], verifier=verifier,
    )
    await verifier.verify(
        constraint=constraint, action_key="x",
        action_params={"content": "same"}, scoped_state=_bare_scope(),
    )
    caplog.clear()
    await verifier.verify(
        constraint=constraint, action_key="x",
        action_params={"content": "same"}, scoped_state=_bare_scope(),
    )
    msgs = [r.getMessage() for r in caplog.records if "[LLMJudge]" in r.getMessage()]
    assert len(msgs) == 1
    assert "fallback=cache_hit" in msgs[0]


# ---------------------------------------------------------------------------
# 2. Streak escalation
# ---------------------------------------------------------------------------


async def test_streak_escalation_downgrades_after_threshold() -> None:
    """N consecutive blocks on same (rule, content) → downgrade
    BLOCK→ADVISE for the rest of the session. The third call returns
    allowed=True with the verdict accumulating as advisory."""

    g = SemanticConstraintGuardrail(
        [
            SemanticConstraint(
                id="r",
                rule_nl="t",
                applies_to=["x"],
                verifier=PythonPredicateVerifier(
                    predicate=lambda *_: ConstraintVerdict(
                        satisfied=False, reason="no",
                    ),
                    description="d",
                ),
            )
        ],
        escalation_threshold=2,
    )
    d1 = await g.check(
        action_key="x.go", params={"content": "same"}, call_history=[],
    )
    assert not d1.allowed  # block 1
    d2 = await g.check(
        action_key="x.go", params={"content": "same"}, call_history=[],
    )
    assert not d2.allowed  # block 2 — triggers downgrade
    d3 = await g.check(
        action_key="x.go", params={"content": "same"}, call_history=[],
    )
    assert d3.allowed  # downgraded → allowed + advisory
    drained = g.drain_pending_advisories()
    assert len(drained) == 1
    assert drained[0][0].id == "r"


async def test_streak_resets_on_satisfied_verdict() -> None:
    """A satisfied verdict for (rule, content) resets the streak —
    the LLM course-corrected and the same content might be blocked
    again later legitimately (not stuck)."""

    flip = {"satisfied": False}

    def pred(action_key, params, state):
        return ConstraintVerdict(satisfied=flip["satisfied"], reason="x")

    g = SemanticConstraintGuardrail(
        [
            SemanticConstraint(
                id="r", rule_nl="t", applies_to=["x"],
                verifier=PythonPredicateVerifier(
                    predicate=pred, description="d",
                ),
            )
        ],
        escalation_threshold=2,
    )
    await g.check(action_key="x", params={"c": "v"}, call_history=[])
    flip["satisfied"] = True
    await g.check(action_key="x", params={"c": "v"}, call_history=[])
    flip["satisfied"] = False
    # Streak reset; next block is the first of a new streak, not the second.
    d = await g.check(action_key="x", params={"c": "v"}, call_history=[])
    assert not d.allowed
    assert "r" not in g._downgraded


# ---------------------------------------------------------------------------
# 3. Precondition pre-filter
# ---------------------------------------------------------------------------


async def test_precondition_false_skips_verifier() -> None:
    """When ``precondition(...)`` returns False, the verifier is
    never invoked — the constraint passes silently. Eliminates LLM
    calls for actions that obviously can't trigger the rule."""

    invoked = {"count": 0}

    def pred(action_key, params, state):
        invoked["count"] += 1
        return ConstraintVerdict(satisfied=False, reason="x")

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="r", rule_nl="t", applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=pred, description="d",
            ),
            precondition=lambda *_: False,
        )
    ])
    decision = await g.check(
        action_key="x.go", params={"content": "anything"}, call_history=[],
    )
    assert decision.allowed
    assert invoked["count"] == 0


async def test_precondition_true_runs_verifier() -> None:
    """``precondition=True`` falls through to the verifier."""

    invoked = {"count": 0}

    def pred(action_key, params, state):
        invoked["count"] += 1
        return ConstraintVerdict(satisfied=True)

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="r", rule_nl="t", applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=pred, description="d",
            ),
            precondition=lambda *_: True,
        )
    ])
    await g.check(
        action_key="x.go", params={"content": "anything"}, call_history=[],
    )
    assert invoked["count"] == 1


async def test_precondition_exception_falls_through_to_verifier() -> None:
    """A buggy precondition (raises) must not crash the check — fall
    through to the verifier so the rule still runs (defense-in-depth
    matches the existing fail-open conventions)."""

    invoked = {"count": 0}

    def pred(action_key, params, state):
        invoked["count"] += 1
        return ConstraintVerdict(satisfied=True)

    def bad_pre(*_):
        raise RuntimeError("boom")

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="r", rule_nl="t", applies_to=["x"],
            verifier=PythonPredicateVerifier(
                predicate=pred, description="d",
            ),
            precondition=bad_pre,
        )
    ])
    decision = await g.check(
        action_key="x.go", params={"content": "x"}, call_history=[],
    )
    assert decision.allowed
    assert invoked["count"] == 1


# ---------------------------------------------------------------------------
# 4. Hash helpers (pure)
# ---------------------------------------------------------------------------


def test_judge_cache_key_stable_for_same_inputs() -> None:
    k1 = _judge_cache_key(
        constraint_id="r", action_key="x",
        action_params={"a": 1, "b": [1, 2]}, history=[],
    )
    k2 = _judge_cache_key(
        constraint_id="r", action_key="x",
        action_params={"b": [1, 2], "a": 1}, history=[],
    )
    assert k1 == k2  # key order doesn't matter


def test_judge_cache_key_differs_on_any_input_change() -> None:
    base = dict(
        constraint_id="r", action_key="x",
        action_params={"a": 1}, history=[],
    )
    k0 = _judge_cache_key(**base)
    k1 = _judge_cache_key(**{**base, "constraint_id": "r2"})
    k2 = _judge_cache_key(**{**base, "action_key": "y"})
    k3 = _judge_cache_key(**{**base, "action_params": {"a": 2}})
    k4 = _judge_cache_key(**{
        **base,
        "history": [CallRecord(
            action_key="z", params={}, end_wall=time.time(), status="ok",
        )],
    })
    assert len({k0, k1, k2, k3, k4}) == 5


def test_content_fingerprint_stable_and_differs() -> None:
    assert _content_fingerprint({"a": 1}) == _content_fingerprint({"a": 1})
    assert _content_fingerprint({"a": 1}) != _content_fingerprint({"a": 2})


# ---------------------------------------------------------------------------
# 5. SessionAgent rule precondition (regression pin)
# ---------------------------------------------------------------------------


def test_session_agent_no_unverified_rule_wires_precondition() -> None:
    """The R12 fix MUST wire the regex pre-filter onto the
    no_unverified_agent_state_claims rule. Without it, every
    respond_to_user (even "Hi!") fires the LLM judge."""

    from polymathera.colony.web_ui.backend.chat.session_agent_guardrails import (
        session_agent_semantic_constraints,
    )
    catalogue = session_agent_semantic_constraints()
    rule = next(
        c for c in catalogue
        if c.id == "no_unverified_agent_state_claims"
    )
    assert rule.precondition is not None
    # Owner id must be hex-shaped to match the regex.
    scope = ScopedConstraintState(
        cell_calls=(), session_calls=(),
        owner_agent_id="agent-0123abcd",
    )
    # Plain content with no agent_id mention → skip
    assert (
        rule.precondition("respond_to_user", {"content": "Hi!"}, scope)
        is False
    )
    # Non-self agent_id mention → run judge
    assert (
        rule.precondition(
            "respond_to_user",
            {"content": "the coordinator agent-abc123ef is running"},
            scope,
        )
        is True
    )
    # Speaker's OWN id is exempt → skip
    assert (
        rule.precondition(
            "respond_to_user",
            {"content": "I (agent-0123abcd) am responding."},
            scope,
        )
        is False
    )
