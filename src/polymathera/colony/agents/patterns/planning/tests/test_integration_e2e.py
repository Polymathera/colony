"""Integration tests for planning framework components.

Tests planning models and their interactions without requiring
a full Agent instance, Ray, or external services.
"""

import pytest

from ....models import (
    Action,
    ActionPlan,
    ActionResult,
    ActionStatus,
    ActionType,
    CacheContext,
    PlanStatus,
)

AGENT_ID = "test-agent"


def test_plan_creation_and_lifecycle():
    """Test creating a plan and moving through lifecycle states."""
    plan = ActionPlan(
        plan_id="test-plan-001",
        agent_id=AGENT_ID,
        goals=["Analyze repository structure", "Identify dependencies"],
    )

    assert plan.status is None
    assert len(plan.goals) == 2
    assert len(plan.actions) == 0

    # Add actions
    plan.actions = [
        Action(
            action_id="a1",
            agent_id=AGENT_ID,
            action_type=ActionType.ANALYZE_PAGE,
            description="Analyze main module",
            status=ActionStatus.PENDING,
        ),
        Action(
            action_id="a2",
            agent_id=AGENT_ID,
            action_type=ActionType.SYNTHESIZE,
            description="Synthesize findings",
            status=ActionStatus.PENDING,
        ),
    ]

    assert plan.has_remaining_actions()
    assert len(plan.get_pending_actions()) == 2

    # Simulate execution
    plan.status = PlanStatus.ACTIVE
    plan.actions[0].status = ActionStatus.COMPLETED
    plan.current_action_index = 1

    assert not plan.is_complete()
    assert plan.has_remaining_actions()
    assert len(plan.get_pending_actions()) == 1

    # Complete all actions
    plan.actions[1].status = ActionStatus.COMPLETED
    plan.current_action_index = 2
    plan.status = PlanStatus.COMPLETED

    assert plan.is_complete()
    assert not plan.has_remaining_actions()


def test_plan_execution_context_tracking():
    """Test that execution context tracks completed actions and findings."""
    plan = ActionPlan(
        plan_id="ctx-plan",
        agent_id=AGENT_ID,
        goals=["Test context tracking"],
    )

    # Add spawned agents
    plan.add_spawned_agent("child-1", "Analyze module A", "child-plan-1")
    plan.add_spawned_agent("child-2", "Analyze module B")

    assert len(plan.execution_context.spawned_children) == 2
    assert plan.execution_context.spawned_children[0].agent_id == "child-1"
    assert plan.execution_context.spawned_children[0].plan_id == "child-plan-1"
    assert plan.execution_context.spawned_children[1].plan_id is None


def test_cache_context_creation():
    """Test cache context with working set and priorities."""
    cache_ctx = CacheContext(
        working_set=["page-1", "page-2", "page-3"],
        working_set_priority={"page-1": 0.9, "page-2": 0.7, "page-3": 0.3},
        min_cache_size=2,
        ideal_cache_size=5,
    )

    assert len(cache_ctx.working_set) == 3
    assert cache_ctx.working_set_priority["page-1"] == 0.9
    assert cache_ctx.min_cache_size == 2
    assert cache_ctx.ideal_cache_size == 5


def test_plan_with_overlapping_cache_contexts():
    """Test plans with overlapping working sets for conflict analysis."""
    plan1 = ActionPlan(
        plan_id="plan-1",
        agent_id="agent-1",
        goals=["Goal 1"],
        cache_context=CacheContext(
            working_set=["page-1", "page-2", "page-3", "page-4", "page-5", "page-6"]
        ),
    )

    plan2 = ActionPlan(
        plan_id="plan-2",
        agent_id="agent-2",
        goals=["Goal 2"],
        cache_context=CacheContext(
            working_set=["page-4", "page-5", "page-6", "page-7", "page-8", "page-9"]
        ),
    )

    # Compute overlap manually (what a coordination policy would do)
    overlap = set(plan1.cache_context.working_set) & set(plan2.cache_context.working_set)
    assert overlap == {"page-4", "page-5", "page-6"}


def test_plan_with_action_results():
    """Test plan with action results tracking."""
    plan = ActionPlan(
        plan_id="result-plan",
        agent_id=AGENT_ID,
        goals=["Test results"],
        actions=[
            Action(
                action_id="a1",
                agent_id=AGENT_ID,
                action_type=ActionType.ANALYZE_PAGE,
                description="Analyze page",
                status=ActionStatus.COMPLETED,
                result=ActionResult(
                    success=True,
                    output={"findings": ["auth module found"]},
                    metrics={"time_ms": 150},
                ),
            ),
            Action(
                action_id="a2",
                agent_id=AGENT_ID,
                action_type=ActionType.ANALYZE_PAGE,
                description="Analyze page 2",
                status=ActionStatus.FAILED,
                result=ActionResult(
                    success=False,
                    error="Page not found",
                ),
            ),
        ],
    )

    completed = [a for a in plan.actions if a.status == ActionStatus.COMPLETED]
    failed = [a for a in plan.actions if a.status == ActionStatus.FAILED]

    assert len(completed) == 1
    assert len(failed) == 1
    assert completed[0].result.success
    assert not failed[0].result.success
    assert failed[0].result.error == "Page not found"
