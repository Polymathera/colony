"""End-to-end integration test for planning framework.

Tests complete planning workflow from agent initialization through plan execution
to learning feedback.
"""

import pytest
from polymathera.colony.agents.models import AgentState, PlanStatus, ActionStatus

from agents.base import Agent


@pytest.mark.asyncio
async def test_planning_agent_complete_workflow():
    """Test complete planning agent workflow end-to-end.

    This test verifies:
    1. Agent initialization with automatic policy creation
    2. Plan creation with learned patterns and cache context
    3. Plan execution with action progression
    4. Learning feedback when plan completes
    5. Access control integration
    """

    # === Step 1: Initialize Agent ===
    agent = Agent(
        agent_id="test-agent-001",
        metadata={
            "goals": ["Analyze repository structure", "Identify dependencies"],
            "planning_params": {
                "planning_horizon": 5,
                "ideal_cache_size": 10,
            },
        },
    )

    await agent.initialize()

    # Verify policies were created
    assert agent.planner is not None
    assert agent.planner.cache_policy is not None
    assert agent.planner.learning_policy is not None
    assert agent.planner.coordination_policy is not None

    # Verify planner is initialized with policies
    assert agent.planner.cache_policy.cache_capacity == 10
    assert agent.planner.learning_policy.history_store is not None

    # === Step 2: Verify Plan Creation ===
    # Initial plan should be created during initialization
    assert agent.current_plan is not None
    initial_plan = agent.current_plan

    # Plan should have goals
    assert len(initial_plan.goals) == 2
    assert "Analyze repository structure" in initial_plan.goals

    # Plan should have actions (created by LLM)
    assert len(initial_plan.actions) > 0

    # Plan should have cache context (from cache policy)
    assert initial_plan.cache_context is not None

    # === Step 3: Execute Plan ===
    execution_steps = 0
    max_steps = 20  # Prevent infinite loop

    while not initial_plan.is_complete() and execution_steps < max_steps:
        await agent.run_step()
        execution_steps += 1

        # Refresh plan from blackboard
        initial_plan = await agent.plan_blackboard.get_plan(agent.agent_id)

        # Verify progress
        assert initial_plan.current_action_index <= len(initial_plan.actions)

    # Verify plan completed or at least made progress
    assert execution_steps > 0

    # === Step 4: Verify Learning Feedback ===
    if initial_plan.is_complete():
        # Learning should have been triggered
        history_store = agent.planner.learning_policy.history_store

        # Check that execution was recorded
        stats = await history_store.get_statistics()
        assert stats.get("total_executions", 0) > 0

        # Verify plan status
        assert initial_plan.status in [PlanStatus.COMPLETED, PlanStatus.FAILED]

        # Verify completed actions
        completed_actions = [
            a for a in initial_plan.actions if a.status == ActionStatus.COMPLETED
        ]
        assert len(completed_actions) > 0

    # === Step 5: Verify Access Control ===
    # Agent should be able to read own plan
    blackboard = agent.plan_blackboard
    access_policy = blackboard.plan_access_policy

    if access_policy:
        assert access_policy.can_read_plan(agent.agent_id, initial_plan)
        assert access_policy.can_update_plan(agent.agent_id, initial_plan)

        # Other agents should not be able to update this plan
        assert not access_policy.can_update_plan("other-agent", initial_plan)


@pytest.mark.asyncio
async def test_planning_agent_with_custom_policies():
    """Test planning agent with custom policy configuration."""

    from polymathera.colony.agents.patterns.planning.policies import (
        CacheAwarePlanningPolicy,
        LearningPlanningPolicy,
        CoordinationPlanningPolicy,
    )

    # Create custom policies
    cache_policy = CacheAwarePlanningPolicy(
        agent=agent,
        cache_capacity=20,
        query_vcm_state=True
    )
    await cache_policy.initialize()

    # Note: LearningPlanningPolicy needs blackboard, which is created in agent.initialize()
    # So we'll create it after agent initialization

    coordination_policy = CoordinationPlanningPolicy(cache_capacity=20)
    await coordination_policy.initialize()

    # Create agent with custom policies
    agent = Agent(
        agent_id="test-agent-002",
        metadata={
            "goals": ["Custom goal"],
            "cache_policy": cache_policy,
            "coordination_policy": coordination_policy,
        },
    )

    await agent.initialize()

    # Verify custom policies were used
    assert agent.planner.cache_policy is cache_policy
    assert agent.planner.cache_policy.cache_capacity == 20
    assert agent.planner.cache_policy.query_vcm_state is True

    assert agent.planner.coordination_policy is coordination_policy


@pytest.mark.asyncio
async def test_planning_agent_hierarchy_and_teams():
    """Test planning agent with hierarchy and team structure."""

    # Create parent agent
    parent_agent = Agent(
        agent_id="parent-agent",
        metadata={
            "goals": ["Coordinate child agents"],
            "team": "team-alpha",
        },
    )
    await parent_agent.initialize()

    # Create child agent
    child_agent = Agent(
        agent_id="child-agent",
        metadata={
            "goals": ["Execute subtask"],
            "parent_agent_id": "parent-agent",
            "team": "team-alpha",
        },
    )
    await child_agent.initialize()

    # Verify hierarchy was discovered
    blackboard = parent_agent.plan_blackboard
    access_policy = blackboard.plan_access_policy

    if access_policy and hasattr(access_policy, "agent_hierarchy"):
        # Child should be in hierarchy
        assert "child-agent" in access_policy.agent_hierarchy
        assert access_policy.agent_hierarchy["child-agent"] == "parent-agent"

        # Parent should be able to read child plan
        child_plan = child_agent.current_plan
        assert access_policy.can_read_plan("parent-agent", child_plan)
        assert access_policy.can_approve_plan("parent-agent", child_plan)

    # Verify team structure
    if access_policy and hasattr(access_policy, "team_structure"):
        # Both agents should be in team-alpha
        assert "team-alpha" in access_policy.team_structure
        team_members = access_policy.team_structure["team-alpha"]
        assert "parent-agent" in team_members
        assert "child-agent" in team_members


@pytest.mark.asyncio
async def test_cache_aware_planning():
    """Test cache-aware planning with VCM integration."""

    agent = Agent(
        agent_id="cache-test-agent",
        metadata={
            "goals": ["Test cache-aware planning"],
            "planning_params": {
                "ideal_cache_size": 5,
            },
        },
    )

    await agent.initialize()

    # Verify cache policy was created
    cache_policy = agent.planner.cache_policy
    assert cache_policy is not None
    assert cache_policy.cache_capacity == 5

    # Get plan
    plan = agent.current_plan

    # Plan should have cache context
    assert plan.cache_context is not None
    assert plan.cache_context.ideal_cache_size <= 5

    # If query_vcm_state is enabled, cache context should reflect VCM state
    # (This would require VCM to be running, which it might not be in tests)


@pytest.mark.asyncio
async def test_learning_from_execution():
    """Test learning feedback loop after plan execution."""

    agent = Agent(
        agent_id="learning-test-agent",
        metadata={
            "goals": ["Test learning"],
            "planning_params": {
                "planning_horizon": 2,  # Small plan for quick test
            },
        },
    )

    await agent.initialize()

    # Execute plan to completion
    plan = agent.current_plan
    initial_action_count = len(plan.actions)

    for _ in range(initial_action_count + 5):  # Extra iterations in case of issues
        if plan.is_complete():
            break
        await agent.run_step()
        plan = await agent.plan_blackboard.get_plan(agent.agent_id)

    # If plan completed, check learning was triggered
    if plan.status == PlanStatus.COMPLETED:
        history_store = agent.planner.learning_policy.history_store

        # Execution should be recorded
        stats = await history_store.get_statistics()
        assert stats.get("total_executions", 0) >= 1

        # Can query for similar plans
        similar = await agent.planner.learning_policy.get_similar_plans(
            goals=["Test learning"],
            context={},
            limit=5,
        )
        assert isinstance(similar, list)


@pytest.mark.asyncio
async def test_conflict_detection_and_resolution():
    """Test multi-agent conflict detection and resolution."""

    from polymathera.colony.agents.models import ActionPlan, CacheContext

    # Create coordination policy
    from polymathera.colony.agents.patterns.planning.policies import CoordinationPlanningPolicy

    coord_policy = CoordinationPlanningPolicy(cache_capacity=10)
    await coord_policy.initialize()

    # Create two plans with overlapping cache requirements
    plan1 = ActionPlan(
        plan_id="plan-1",
        agent_id="agent-1",
        goals=["Goal 1"],
        actions=[],
        cache_context=CacheContext(
            working_set=["page-1", "page-2", "page-3", "page-4", "page-5", "page-6"]
        ),
    )

    plan2 = ActionPlan(
        plan_id="plan-2",
        agent_id="agent-2",
        goals=["Goal 2"],
        actions=[],
        cache_context=CacheContext(
            working_set=["page-4", "page-5", "page-6", "page-7", "page-8", "page-9"]
        ),
    )

    # Detect conflicts
    conflicts = await coord_policy.check_conflicts(plan1, [plan2])

    # Should detect cache contention (combined working set > capacity)
    assert len(conflicts) > 0

    # Resolve conflict
    if conflicts:
        resolved_plan = await coord_policy.resolve_conflict(plan1, conflicts[0])
        assert resolved_plan is not None

        # Plan should have resolution metadata
        assert "delayed_by_priority" in resolved_plan.metadata or \
               "execution_delay_s" in resolved_plan.metadata or \
               resolved_plan.status == PlanStatus.SUSPENDED

