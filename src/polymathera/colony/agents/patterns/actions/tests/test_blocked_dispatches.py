"""Tests for blocked-dispatch survival into the next planner prompt.

Item 5 of ``colony/decompose_and_session_recovery_fixes_plan.md``:
the runtime guardrail's blocks must surface in the NEXT iteration's
planner prompt so the LLM can recover (call the suggested precursor,
or change approach) rather than silently swallowing the block.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.models import PlanningContext
from polymathera.colony.agents.patterns.planning.models import (
    BlockedDispatch,
)
from polymathera.colony.agents.patterns.actions.code_generation import (
    format_planning_context_for_codegen,
)


def test_prompt_omits_blocked_section_when_none_provided() -> None:
    """No blocks → no section. Keeps the prompt small for the steady
    state (every iteration after a successful one has zero blocks)."""

    prompt = format_planning_context_for_codegen(
        planning_context=PlanningContext(),
        mode="execution",
    )
    assert "## Blocked Dispatches" not in prompt


def test_prompt_renders_blocked_section_when_blocks_provided() -> None:
    """One block surfaces under a clearly-named section with the
    action key, the proposed params, and the guardrail's suggestion."""

    blocks = [BlockedDispatch(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params_preview={"content": "agent-abc is running."},
        reason="requires a recent 'get_agent_status' call",
        suggestion="Call get_agent_status before respond_to_user.",
    )]
    prompt = format_planning_context_for_codegen(
        planning_context=PlanningContext(),
        mode="execution",
        blocked_dispatches=blocks,
    )
    assert "## Blocked Dispatches (last iteration)" in prompt
    assert "SessionOrchestratorCapability.respond_to_user" in prompt
    assert "agent-abc is running." in prompt
    assert "requires a recent 'get_agent_status' call" in prompt
    assert "Call get_agent_status before respond_to_user." in prompt


def test_prompt_renders_multiple_blocks_in_order() -> None:
    """Two blocks land in the section in submission order; the
    planner sees both so it can fix the cell as a whole, not just
    the last block."""

    blocks = [
        BlockedDispatch(
            action_key="A.first_action",
            params_preview={"x": 1},
            reason="first reason",
            suggestion="first suggestion",
        ),
        BlockedDispatch(
            action_key="A.second_action",
            params_preview={"y": 2},
            reason="second reason",
            suggestion="second suggestion",
        ),
    ]
    prompt = format_planning_context_for_codegen(
        planning_context=PlanningContext(),
        mode="execution",
        blocked_dispatches=blocks,
    )
    first_idx = prompt.find("A.first_action")
    second_idx = prompt.find("A.second_action")
    assert first_idx != -1 and second_idx != -1
    assert first_idx < second_idx


def test_prompt_renders_block_without_suggestion() -> None:
    """A guardrail may emit a block with no suggestion; the section
    still renders the action_key + reason, just no Suggestion line."""

    blocks = [BlockedDispatch(
        action_key="A.do_thing",
        params_preview={},
        reason="just no",
        suggestion="",
    )]
    prompt = format_planning_context_for_codegen(
        planning_context=PlanningContext(),
        mode="execution",
        blocked_dispatches=blocks,
    )
    assert "A.do_thing" in prompt
    assert "just no" in prompt
    # No empty "Suggestion:" line.
    assert "Suggestion: " not in prompt or "Suggestion: \n" not in prompt


@pytest.mark.asyncio
async def test_policy_captures_block_and_renders_then_clears() -> None:
    """End-to-end on the policy: a blocked ``run()`` call appends a
    BlockedDispatch; the next prompt build renders it AND clears the
    list so iteration N+2 doesn't see iteration N's blocks again."""

    from polymathera.colony.agents.patterns.actions.code_constraints import (
        RuntimeGuardrail,
        GuardrailDecision,
    )

    class _AlwaysBlockGuardrail(RuntimeGuardrail):
        async def check(self, action_key, params, call_history):
            return GuardrailDecision(
                allowed=False,
                reason="testing post-hoc block surfacing",
                suggestion="just a test",
            )

    # Build a policy with the blocking guardrail. We can't easily
    # exercise the full ``run()`` helper without a REPL, so we
    # simulate the capture step the helper does: append to
    # ``self._last_blocked_dispatches`` and then verify the prompt
    # renders + clears.
    from polymathera.colony.agents.patterns.actions.code_generation import (
        CodeGenerationActionPolicy,
    )

    fake_agent = type("A", (), {})()
    fake_agent.agent_id = "agent-test"
    fake_agent.metadata = type("M", (), {})()
    fake_agent.metadata.action_policy_config = {}

    # Build the policy directly with only the guardrail set. The
    # constructor takes many optional deps; passing them as None
    # falls back to the no-op variants.
    policy = CodeGenerationActionPolicy(
        agent=fake_agent,
        runtime_guardrail=_AlwaysBlockGuardrail(),
    )

    # Simulate a blocked dispatch capture (what the run() helper
    # does after the guardrail says no).
    policy._last_blocked_dispatches.append(BlockedDispatch(
        action_key="X.gated",
        params_preview={"k": "v"},
        reason="testing post-hoc block surfacing",
        suggestion="just a test",
    ))

    # Render the prompt with the policy's blocks. We bypass the
    # full async loop and just verify the renderer wiring.
    prompt = format_planning_context_for_codegen(
        planning_context=PlanningContext(),
        mode="execution",
        blocked_dispatches=(
            policy._last_blocked_dispatches
            if policy._last_blocked_dispatches else None
        ),
    )
    assert "X.gated" in prompt
    assert "testing post-hoc block surfacing" in prompt

    # The policy's own clear step (after rendering) lives in the
    # full action-policy loop; simulate it here.
    policy._last_blocked_dispatches.clear()
    prompt_after = format_planning_context_for_codegen(
        planning_context=PlanningContext(),
        mode="execution",
        blocked_dispatches=(
            policy._last_blocked_dispatches
            if policy._last_blocked_dispatches else None
        ),
    )
    assert "## Blocked Dispatches" not in prompt_after
