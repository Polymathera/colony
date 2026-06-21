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
    CompositeGuardrail,
    IterationShapeValidator,
    NoGuardrail,
)
from polymathera.colony.agents.patterns.planning.models import CallRecord


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


def test_noguardrail_has_no_advisory() -> None:
    assert NoGuardrail().planner_context_advisory([]) is None


def test_approval_advisory_describes_the_full_flow() -> None:
    """The advisory is standing context — it always surfaces when
    there's a gated prefix, naming each primitive in isolation per
    [[primitives-not-pipelines]]. The LLM composes the strategy; the
    advisory describes the capability surface (request +
    wait_for_next_event + get_response) and the four response
    choices (approve_once / approve_all / reject / abort)."""

    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=(
            "DesignProcessCapability.sync_roadmap_with_github",
        ),
    )
    advisory = g.planner_context_advisory([])
    assert advisory is not None
    # Capability surface is named in full.
    assert "request_human_approval" in advisory
    assert "wait_for_next_event" in advisory
    assert "get_response" in advisory
    # action_type-scoped four-way choice.
    assert "action_type" in advisory
    assert "approve_once" in advisory
    assert "approve_all" in advisory
    assert "reject" in advisory
    assert "abort" in advisory
    # Explanation surface is named so the planner expects it on
    # reject / abort context bindings.
    assert "explanation" in advisory
    # No "poll" instruction — the prompt must not teach the failure
    # mode it caused on run 3.
    lowered = advisory.lower()
    assert "poll get_response" not in lowered
    assert "keep polling" not in lowered
    # The advisory references the ClassVar prefix, not an inline
    # ``human_approval_response:`` literal — guards against a future
    # rename silently rotting the prompt.
    from polymathera.colony.agents.patterns.capabilities.human_approval import (
        HumanApprovalCapability,
    )
    assert HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX in advisory


def test_approval_block_suggestion_names_event_queue_as_wake_surface() -> None:
    """The block-message suggestion fires when the LLM proposes a
    mutating call without prior approval. It should teach the same
    capability surface as the advisory — primitives in isolation,
    no poll-then-apply recipe — so the LLM converges on the same
    pattern from either path."""

    import asyncio
    from unittest.mock import MagicMock
    from polymathera.colony.agents.patterns.capabilities.human_approval import (
        HumanApprovalCapability,
    )

    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=(
            "DesignProcessCapability.create_decomposition",
        ),
    )
    # No HumanApprovalCapability mounted → the suggestion fires.
    g.bind_agent(MagicMock(get_capability_by_type=lambda _: None))
    decision = asyncio.run(
        g.check(
            "DesignProcessCapability.create_decomposition",
            params={"dry_run": False},
            call_history=[],
        )
    )
    assert decision.allowed is False
    suggestion = decision.suggestion
    assert "request_human_approval" in suggestion
    assert "wait_for_next_event" in suggestion
    assert "get_response" in suggestion
    assert HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX in suggestion
    lowered = suggestion.lower()
    assert "poll get_response" not in lowered
    assert "keep polling" not in lowered
    assert "until the operator answers" not in lowered


def test_approval_advisory_silent_with_empty_prefix_list() -> None:
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


def test_composite_advisory_surfaces_inner_approval_advisory() -> None:
    """A composite that carries an approval gate must propagate the
    advisory from its inner guardrail. The advisory is standing
    context — it does not depend on call_history state."""

    composite = CompositeGuardrail(
        ApprovalRequiredGuardrail(
            approval_required_action_prefixes=(
                "DesignProcessCapability.sync_roadmap_with_github",
            ),
        ),
    )
    advisory = composite.planner_context_advisory([])
    assert advisory is not None
    assert "request_human_approval" in advisory
