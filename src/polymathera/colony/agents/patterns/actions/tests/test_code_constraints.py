"""Tests for ``IterationShapeValidator`` — path-aware ``run()``
counting and the discovery-abuse / size limits.

The path-aware counter is the load-bearing change: a textual walker
would conflate ``if/else`` branches and reject the natural
"acknowledge → action → branched-response" pattern, which trips the
SessionAgent's REPL loop on routine commands. These tests pin the
intended semantics (max across mutually-exclusive branches, sum
elsewhere) so a future refactor can't silently regress to textual
counting.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    ApprovalRequiredGuardrail,
    ArgsAwareOrderingRule,
    ArgsAwareTemporalOrderGuardrail,
    CallRecord,
    CompositeGuardrail,
    IterationShapeValidator,
    NoGuardrail,
)


@pytest.mark.asyncio
async def test_branched_response_counts_as_one_path() -> None:
    """ack + action + (success-branch | failure-branch) must validate
    at the default ``max_actions=3`` because only ONE response branch
    runs per execution. This is the canonical SessionAgent shape."""

    code = (
        'await run("ack", content="working on it")\n'
        'r = await run("the_action")\n'
        "if r.success:\n"
        '    await run("respond_to_user", content="ok")\n'
        "else:\n"
        '    await run("respond_to_user", content="failed")\n'
    )
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert result.valid, result.errors


@pytest.mark.asyncio
async def test_linear_calls_still_sum() -> None:
    """No branching → every ``run()`` is on the single path → the
    validator must add them up. Four straight-line calls fails at
    ``max_actions=3``."""

    code = (
        'await run("a")\n'
        'await run("b")\n'
        'await run("c")\n'
        'await run("d")\n'
    )
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert not result.valid
    assert any("Too many actions" in e for e in result.errors)


@pytest.mark.asyncio
async def test_try_except_takes_max_of_body_and_handlers() -> None:
    """Either the ``try`` body completes (body+else) or an exception
    routes through one handler — so the worst-case path is
    ``body + max(handlers, else)``. ``finally`` always runs and adds
    on top.
    """

    code = (
        "try:\n"
        '    await run("a")\n'
        '    await run("b")\n'
        "except Exception:\n"
        '    await run("recover_a")\n'
        '    await run("recover_b")\n'
        "finally:\n"
        '    await run("cleanup")\n'
    )
    # Any path: body (2) + max(handlers=2, else=0) + finally (1) = 5.
    # That exceeds ``max_actions=3``, so this must fail.
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert not result.valid

    code_ok = (
        "try:\n"
        '    await run("the_action")\n'
        "except Exception:\n"
        '    await run("respond_err")\n'
        "else:\n"
        '    await run("respond_ok")\n'
    )
    # body (1) + max(handlers=1, else=1) = 2. Validates.
    result = await IterationShapeValidator().validate(code_ok, agent=MagicMock())
    assert result.valid, result.errors


@pytest.mark.asyncio
async def test_nested_if_else_takes_max_at_each_level() -> None:
    """Nested branches collapse path-by-path. The deepest path here
    runs at most ``ack + outer-then(inner-then) = 1 + 1 + 1 = 3``."""

    code = (
        'await run("ack")\n'
        "if outer:\n"
        "    if inner:\n"
        '        await run("path_aa")\n'
        "    else:\n"
        '        await run("path_ab")\n'
        "else:\n"
        '    await run("path_b")\n'
    )
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert result.valid, result.errors


@pytest.mark.asyncio
async def test_browse_count_is_textual_not_path_aware() -> None:
    """browse() is a discovery anti-pattern — multiple browses are
    excessive regardless of which branch they sit in. Text-counted."""

    code = (
        "if x:\n"
        '    browse("a")\n'
        "else:\n"
        '    browse("b")\n'
    )
    # Two textual browse() calls — fails at default ``max_browse=1``
    # even though only one runs per path.
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert not result.valid
    assert any("Too many browse" in e for e in result.errors)


# ---------------------------------------------------------------------------
# planner_context_advisory — items 3 + 5 of
# project_planning_workflow_fixes_plan.md
# ---------------------------------------------------------------------------


def _approve_call(action_key: str = "HumanApprovalCapability.get_response") -> CallRecord:
    """Synthesise a successful approval-poll CallRecord matching
    ``_default_is_approval_granted``."""

    return CallRecord(
        action_key=action_key,
        params={"request_id": "r"},
        start_wall=1.0,
        end_wall=1.1,
        status="ok",
        result={
            "ok": True,
            "state": "ready",
            "response": {"request_id": "r", "choice": "approve"},
        },
    )


def test_noguardrail_has_no_advisory() -> None:
    assert NoGuardrail().planner_context_advisory([]) is None


def test_approval_advisory_present_when_gate_unsatisfied() -> None:
    """The propose → approve → apply sequence must surface in
    planner context whenever an approval-gated prefix is configured
    and no ``approve`` choice has landed yet."""

    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=(
            "DesignProcessCapability.sync_roadmap_with_github",
        ),
    )
    advisory = g.planner_context_advisory([])
    assert advisory is not None
    assert "request_human_approval" in advisory
    assert "get_response" in advisory
    assert "dry_run=True" in advisory
    assert "dry_run=False" in advisory


def test_approval_advisory_silent_after_approval_landed() -> None:
    """Once a positive ``approve`` choice is in call_history, the
    gate would let the apply through — the advisory becomes noise
    and must be ``None``."""

    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=(
            "DesignProcessCapability.sync_roadmap_with_github",
        ),
    )
    assert g.planner_context_advisory([_approve_call()]) is None


def test_approval_advisory_silent_with_empty_prefix_list() -> None:
    """An approval guardrail mounted with no gated prefixes is a
    no-op; the advisory should reflect that."""

    g = ApprovalRequiredGuardrail(approval_required_action_prefixes=())
    assert g.planner_context_advisory([]) is None


def test_args_aware_ordering_advisory_lists_rule_suggestions() -> None:
    """Each rule's ``suggestion`` becomes a bullet so the planner
    sees the constraint as standing context, not just as a
    post-hoc block message."""

    rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=lambda params: True,
        required_prior="get_agent_status",
        max_age_calls=20,
        suggestion=(
            "Call AgentPoolCapability.get_agent_status with the "
            "agent_id(s) you're about to reference."
        ),
    )
    g = ArgsAwareTemporalOrderGuardrail(rules=[rule])
    advisory = g.planner_context_advisory([])
    assert advisory is not None
    assert "respond_to_user" in advisory
    assert "get_agent_status" in advisory


def test_args_aware_ordering_advisory_skips_rules_without_suggestion() -> None:
    """Rules with no ``suggestion`` field have no useful prompt
    text — they fall back to the auto-generated ``check`` message
    only, so the advisory should be ``None`` when no rule has a
    suggestion."""

    rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=None,
        required_prior="get_agent_status",
        max_age_calls=20,
        suggestion=None,
    )
    g = ArgsAwareTemporalOrderGuardrail(rules=[rule])
    assert g.planner_context_advisory([]) is None


def test_composite_advisory_joins_inner_advisories() -> None:
    """The composite must surface every inner guardrail's advisory
    so a SessionAgent mounting both an approval gate and a
    status-claim gate sees both rules in the prompt."""

    ordering_rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=None,
        required_prior="get_agent_status",
        max_age_calls=20,
        suggestion="Call get_agent_status first.",
    )
    composite = CompositeGuardrail(
        ArgsAwareTemporalOrderGuardrail(rules=[ordering_rule]),
        ApprovalRequiredGuardrail(
            approval_required_action_prefixes=(
                "DesignProcessCapability.sync_roadmap_with_github",
            ),
        ),
    )
    advisory = composite.planner_context_advisory([])
    assert advisory is not None
    assert "get_agent_status" in advisory
    assert "request_human_approval" in advisory


def test_composite_advisory_returns_none_when_all_inner_silent() -> None:
    """When every inner guardrail's advisory is None, the composite
    advisory is None — keeps the prompt small."""

    composite = CompositeGuardrail(NoGuardrail(), NoGuardrail())
    assert composite.planner_context_advisory([]) is None


def test_composite_advisory_silent_after_approval_landed() -> None:
    """End-to-end: a composite that only carries an approval gate
    goes silent once approval has landed. Important so the planner
    isn't told "you need to approve" AFTER having already done so."""

    composite = CompositeGuardrail(
        ApprovalRequiredGuardrail(
            approval_required_action_prefixes=(
                "DesignProcessCapability.sync_roadmap_with_github",
            ),
        ),
    )
    assert composite.planner_context_advisory([_approve_call()]) is None
