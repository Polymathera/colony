"""Tests for planning data models."""

import pytest

from ....models import (
    Action,
    ActionPlan,
    ActionPolicyExecutionState,
    ActionResult,
    ActionStatus,
    ActionType,
    CacheContext,
    PlanExecutionContext,
    PlanningParameters,
    PlanningStrategy,
    PlanStatus,
    SpawnedChildInfo,
)
from ..capabilities.replanning import PeriodicReplanningPolicy

AGENT_ID = "agent_1"


def test_plan_creation():
    """Test creating a Plan."""
    plan = ActionPlan(
        plan_id="test_plan",
        agent_id=AGENT_ID,
        goals=["Analyze repository", "Generate report"],
    )

    assert plan.plan_id == "test_plan"
    assert plan.agent_id == AGENT_ID
    assert len(plan.goals) == 2
    assert plan.status is None  # No status until explicitly set
    assert plan.generation_method == "manual"


@pytest.mark.asyncio
async def test_plan_methods():
    """Test Plan helper methods."""
    plan = ActionPlan(
        plan_id="test_plan",
        agent_id=AGENT_ID,
        goals=["Test goal"],
        replan_every_n_steps=3,
    )

    # Test is_complete
    plan.status = PlanStatus.ACTIVE
    assert not plan.is_complete()
    plan.status = PlanStatus.COMPLETED
    assert plan.is_complete()

    # Test replanning
    plan.status = PlanStatus.ACTIVE
    policy = PeriodicReplanningPolicy(replan_every_n_steps=3, replan_on_failure=True)
    plan.actions = [
        Action(
            action_id="a1",
            agent_id=AGENT_ID,
            action_type=ActionType.ANALYZE_PAGE,
            description="Analyze page 1",
            status=ActionStatus.COMPLETED,
        ),
        Action(
            action_id="a2",
            agent_id=AGENT_ID,
            action_type=ActionType.ANALYZE_PAGE,
            description="Analyze page 2",
            status=ActionStatus.COMPLETED,
        ),
        Action(
            action_id="a3",
            agent_id=AGENT_ID,
            action_type=ActionType.ANALYZE_PAGE,
            description="Analyze page 3",
            status=ActionStatus.COMPLETED,
        ),
    ]
    state = ActionPolicyExecutionState(current_plan=plan)
    decision = await policy.evaluate_replanning_need(state, None)
    assert decision.should_replan

    # Test get_pending_actions
    plan.actions.append(
        Action(
            action_id="a4",
            agent_id=AGENT_ID,
            action_type=ActionType.SYNTHESIZE,
            description="Synthesize results",
            status=ActionStatus.PENDING,
        )
    )
    pending = plan.get_pending_actions()
    assert len(pending) == 1
    assert pending[0].action_id == "a4"


def test_action_creation():
    """Test creating an Action."""
    action = Action(
        action_id="action_1",
        agent_id=AGENT_ID,
        action_type=ActionType.ANALYZE_PAGE,
        description="Analyze main page",
        parameters={"page_id": "page_001"},
        reasoning="Need to understand the main structure",
    )

    assert action.action_id == "action_1"
    assert action.action_type == ActionType.ANALYZE_PAGE
    assert action.status == ActionStatus.PENDING
    assert action.parameters["page_id"] == "page_001"


def test_action_result():
    """Test ActionResult."""
    result = ActionResult(
        success=True, output={"analyzed": True}, metrics={"time_ms": 150}
    )

    assert result.success
    assert result.output["analyzed"]
    assert result.metrics["time_ms"] == 150
    assert not result.blocked


def test_planning_parameters():
    """Test PlanningParameters."""
    params = PlanningParameters(
        strategy=PlanningStrategy.MPC, planning_horizon=7, max_actions=100
    )

    assert params.strategy == PlanningStrategy.MPC
    assert params.planning_horizon == 7
    assert params.max_actions == 100


def test_cache_context():
    """Test CacheContext."""
    cache_ctx = CacheContext(
        working_set=["page_1", "page_2"],
        working_set_priority={"page_1": 0.9, "page_2": 0.5},
        min_cache_size=10,
    )

    assert len(cache_ctx.working_set) == 2
    assert cache_ctx.working_set_priority["page_1"] == 0.9
    assert cache_ctx.min_cache_size == 10


def test_plan_execution_context():
    """Test PlanExecutionContext."""
    ctx = PlanExecutionContext(
        completed_action_ids=["a1", "a2"],
        findings={"discovery_1": "Found AuthManager"},
    )

    assert len(ctx.completed_action_ids) == 2
    assert "discovery_1" in ctx.findings
    assert len(ctx.spawned_children) == 0


def test_spawned_child_info():
    """Test SpawnedChildInfo."""
    child = SpawnedChildInfo(
        agent_id="child_agent_1", purpose="Analyze module A", plan_id="child_plan_1"
    )

    assert child.agent_id == "child_agent_1"
    assert child.purpose == "Analyze module A"
    assert child.plan_id == "child_plan_1"
    assert child.spawned_at > 0


def test_plan_add_spawned_agent():
    """Test Plan.add_spawned_agent method."""
    plan = ActionPlan(plan_id="parent_plan", agent_id="parent_agent", goals=["Main goal"])

    plan.add_spawned_agent("child_1", "Analyze module A", "child_plan_1")
    plan.add_spawned_agent("child_2", "Analyze module B")

    assert len(plan.execution_context.spawned_children) == 2
    assert plan.execution_context.spawned_children[0].agent_id == "child_1"
    assert plan.execution_context.spawned_children[1].agent_id == "child_2"
    assert plan.execution_context.spawned_children[1].plan_id is None
