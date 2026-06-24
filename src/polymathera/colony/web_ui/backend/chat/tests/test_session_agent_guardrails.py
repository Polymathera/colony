"""Tests for the SessionAgent's runtime guardrail composition after
the R10-3 migration to declarative :class:`SemanticConstraint` records.

Pre-R10-3 the SessionAgent's status-claim rule lived as a syntactic
predicate (regex on ``agent-<hex>`` in respond_to_user content) inside
an :class:`ArgsAwareTemporalOrderGuardrail`. After R10-3 it's one
record in :func:`session_agent_semantic_constraints` checked by an
:class:`LLMJudgeVerifier` — the judge decides whether the proposed
content actually makes a STATE CLAIM about another agent.

What these tests pin:

1. The factory produces a :class:`CompositeGuardrail` of
   :class:`SemanticConstraintGuardrail` + :class:`ApprovalRequiredGuardrail`
   in that order (status-claim cheap-first front, approval gate second).
2. The catalogue ships the ``no_unverified_agent_state_claims`` rule
   with the right scope (CELL), failure mode (BLOCK), applies_to
   filter, and verifier shape.
3. The composite's ``bind_agent`` propagates the agent's ``infer``
   into the LLM-judge verifier.
4. Verdict routing: stub-judge ``satisfied=true`` → ``allowed=True``;
   ``satisfied=false`` → ``allowed=False`` with the verdict's
   reason + suggestion + the rule id prefix.
5. The constraint's ``applies_to=[respond_to_user]`` short-circuits
   unrelated action keys without invoking the judge (cost matters).

The LLM judge's actual semantic decisions (does *this* content make
a state claim about *that* agent?) are NOT pinned here — that's the
judge's responsibility under non-deterministic semantics. Stub judges
let us verify FRAMEWORK PLUMBING; the prompt-engineering of the
``rule_nl`` is evaluated against real LLMs at the staging layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    ApprovalRequiredGuardrail,
    CompositeGuardrail,
)
from polymathera.colony.agents.patterns.actions.semantic_constraints import (
    ConstraintFailureMode,
    ConstraintScope,
    LLMJudgeVerifier,
    SemanticConstraintGuardrail,
)
from polymathera.colony.web_ui.backend.chat.session_agent_guardrails import (
    build_session_agent_runtime_guardrail,
    session_agent_semantic_constraints,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    generated_text: str


def _stub_infer(verdict: dict, *, capture: list | None = None):
    """Build an async ``infer_fn`` that returns ``verdict`` for every
    call. ``capture`` (when supplied) accumulates the prompts so
    tests can assert the judge saw the right inputs."""

    text = json.dumps(verdict)

    async def _f(*, prompt, max_tokens=None, temperature=None):
        if capture is not None:
            capture.append(prompt)
        return _StubResponse(generated_text=text)

    return _f


class _FakeAgent:
    def __init__(
        self, *, agent_id: str, infer_fn,
    ) -> None:
        self.agent_id = agent_id
        self.infer = infer_fn


RESPOND_TO_USER_KEY = (
    "SessionOrchestratorCapability."
    "SessionOrchestratorCapability.respond_to_user"
)


# ---------------------------------------------------------------------------
# 1. Factory composition shape
# ---------------------------------------------------------------------------


async def test_factory_builds_composite_of_semantic_and_approval() -> None:
    """The composite has TWO inner guardrails in the canonical order:
    semantic constraints first (cheaper check, more frequent path),
    approval gate second (narrower mutating-action set)."""

    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    assert isinstance(g, CompositeGuardrail)
    inner = g._inner
    assert len(inner) == 2
    assert isinstance(inner[0], SemanticConstraintGuardrail)
    assert isinstance(inner[1], ApprovalRequiredGuardrail)


# ---------------------------------------------------------------------------
# 2. Catalogue contents pinned
# ---------------------------------------------------------------------------


def test_catalogue_includes_no_unverified_state_claims_rule() -> None:
    """The rule id is the stable handle the LLM and operator refer
    to in block messages, advisories, and code review — pin it so
    a typo in the migration doesn't silently change the handle."""

    catalogue = session_agent_semantic_constraints()
    ids = {c.id for c in catalogue}
    assert "no_unverified_agent_state_claims" in ids


def test_no_unverified_state_claims_rule_has_correct_shape() -> None:
    """The rule's scope, failure mode, applies_to filter, and verifier
    type are pinned. Cell scope is required (fresh evidence only —
    stale status checks from prior iterations don't satisfy the
    rule). Block mode is required (state-claim fabrication is
    irreversible from the user's perspective once shipped)."""

    catalogue = session_agent_semantic_constraints()
    rule = next(
        c for c in catalogue
        if c.id == "no_unverified_agent_state_claims"
    )
    assert rule.scope == ConstraintScope.CELL
    assert rule.failure_mode == ConstraintFailureMode.BLOCK
    assert rule.applies_to == ["respond_to_user"]
    assert isinstance(rule.verifier, LLMJudgeVerifier)
    # The natural-language rule must reference the THREE invariants
    # the judge enforces — pin so a rewrite doesn't drop a key piece.
    rule_text = rule.rule_nl.lower()
    assert "lifecycle state" in rule_text or "state" in rule_text
    assert "get_agent_status" in rule_text
    assert "owner" in rule_text  # self-exclusion invariant
    # D4 (2026-06-23): the rule MUST teach the judge that
    # ``agent_ids=[...]`` (the action's actual parameter name) is
    # the form of evidence to look for. Without this, the judge
    # rejects valid calls because the action's signature uses the
    # plural form while the rule's prose used to say "for that
    # exact agent_id" — causing the 2026-06-23 retry storm where
    # the SessionAgent's LLM made the right call but the judge
    # didn't recognise it.
    assert "agent_ids" in rule_text, (
        "D4: rule must mention the plural ``agent_ids`` form so the "
        "LLM judge recognises the action's actual parameter shape."
    )


# ---------------------------------------------------------------------------
# 3. Agent propagation through the composite
# ---------------------------------------------------------------------------


async def test_composite_bind_agent_propagates_to_llm_judge() -> None:
    """The composite walks every inner guardrail; the semantic
    constraint guardrail then walks every LLM-judge verifier and
    late-binds ``agent.infer``. Without this chain, the judge stays
    unbound and falls open (allowing every action) — a real
    regression that would defeat the whole rule."""

    capture: list = []
    agent = _FakeAgent(
        agent_id="agent-aaaaaaaa",
        infer_fn=_stub_infer(
            {"satisfied": True, "reason": ""},
            capture=capture,
        ),
    )
    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    g.bind_agent(agent)

    semantic = g._inner[0]
    assert isinstance(semantic, SemanticConstraintGuardrail)
    rule = next(
        c for c in semantic._constraints
        if c.id == "no_unverified_agent_state_claims"
    )
    assert isinstance(rule.verifier, LLMJudgeVerifier)
    assert rule.verifier._infer_fn is agent.infer


# ---------------------------------------------------------------------------
# 4. Verdict routing
# ---------------------------------------------------------------------------


async def test_judge_satisfied_passes_respond_to_user() -> None:
    """The framework's job is to call the judge with the right
    inputs and route the verdict. When the judge says satisfied,
    the action passes — regardless of content. (The judge's
    semantic decision-making is its own concern.)"""

    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    agent = _FakeAgent(
        agent_id="agent-owner",
        infer_fn=_stub_infer({"satisfied": True}),
    )
    g.bind_agent(agent)
    decision = await g.check(
        action_key=RESPOND_TO_USER_KEY,
        params={"content": "The agent-coord is running."},
        call_history=[],
    )
    assert decision.allowed


async def test_judge_unsatisfied_blocks_with_rule_id_prefix() -> None:
    """Block message carries the rule id prefix so the LLM /
    operator can attribute it. Reason + suggestion flow from the
    judge into the GuardrailDecision unchanged."""

    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    agent = _FakeAgent(
        agent_id="agent-owner",
        infer_fn=_stub_infer({
            "satisfied": False,
            "reason": "claims agent-coord is running without a status check this cell",
            "suggestion": "call get_agent_status before respond_to_user",
        }),
    )
    g.bind_agent(agent)
    decision = await g.check(
        action_key=RESPOND_TO_USER_KEY,
        params={"content": "The agent-coord is running."},
        call_history=[],
    )
    assert not decision.allowed
    assert "[no_unverified_agent_state_claims]" in decision.reason
    assert "without a status check" in decision.reason
    assert "get_agent_status" in decision.suggestion


# ---------------------------------------------------------------------------
# 5. applies_to filter short-circuits
# ---------------------------------------------------------------------------


async def test_unrelated_action_does_not_invoke_judge() -> None:
    """The constraint applies only to ``respond_to_user``; an
    unrelated action key (e.g. ``get_agent_status`` itself) must
    NOT trigger the judge. Otherwise every action would pay the
    LLM round-trip cost."""

    capture: list = []
    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    agent = _FakeAgent(
        agent_id="agent-owner",
        infer_fn=_stub_infer(
            {"satisfied": True}, capture=capture,
        ),
    )
    g.bind_agent(agent)
    decision = await g.check(
        action_key="AgentPoolCapability.get_agent_status",
        params={"agent_ids": ["agent-coord"]},
        call_history=[],
    )
    assert decision.allowed
    assert capture == []  # judge never called


# ---------------------------------------------------------------------------
# 6. Judge prompt carries the cell-scoped slice
# ---------------------------------------------------------------------------


async def test_judge_sees_cell_history_via_cell_start_index() -> None:
    """When the code-generation policy passes the iteration boundary,
    the judge prompt should include the calls AT/AFTER that index
    (cell scope) and exclude earlier ones. Pins the wiring of
    ``cell_start_index`` from policy → composite → semantic guardrail
    → judge prompt."""

    from polymathera.colony.agents.patterns.planning.models import CallRecord
    import time as _t

    capture: list = []
    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    agent = _FakeAgent(
        agent_id="agent-owner",
        infer_fn=_stub_infer(
            {"satisfied": True}, capture=capture,
        ),
    )
    g.bind_agent(agent)
    history = [
        CallRecord(
            action_key="PRIOR.iter_call_a",
            params={},
            end_wall=_t.time(),
            status="ok",
        ),
        CallRecord(
            action_key="PRIOR.iter_call_b",
            params={},
            end_wall=_t.time(),
            status="ok",
        ),
        CallRecord(
            action_key="CELL.get_agent_status",
            params={"agent_ids": ["agent-coord"]},
            end_wall=_t.time(),
            status="ok",
            result={"agents": [{"id": "agent-coord", "state": "RUNNING"}]},
        ),
    ]
    await g.check(
        action_key=RESPOND_TO_USER_KEY,
        params={"content": "agent-coord status update"},
        call_history=history,
        cell_start_index=2,  # cell starts at the get_agent_status call
    )
    assert len(capture) == 1
    prompt = capture[0]
    assert "CELL.get_agent_status" in prompt
    assert "PRIOR.iter_call_a" not in prompt
    assert "PRIOR.iter_call_b" not in prompt


# ---------------------------------------------------------------------------
# 7. Auto-generated advisory contains the rule (R10-4)
# ---------------------------------------------------------------------------


def test_advisory_auto_generates_no_unverified_state_claims_section() -> None:
    """The advisory the SessionAgent's planner prompt picks up from
    the composite must include the catalogue's rule — auto-generated,
    NOT hand-written into self_concept. Adding a new rule to the
    catalogue should automatically update the prompt."""

    g = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    advisory = g.planner_context_advisory(call_history=[])
    assert advisory is not None
    assert "no_unverified_agent_state_claims" in advisory
    assert "respond_to_user" in advisory
    assert "cell" in advisory.lower()
    assert "block" in advisory.lower()
    # The advisory must reference the natural-language rule body
    # (not just the id), since the LLM reads the rule to know what
    # to do — id alone is opaque.
    assert "lifecycle state" in advisory or "state" in advisory
