"""Planner class for creating and executing plans.

This module provides the core Planner class that:
- Creates plans (LLM-generated or manual)
- Executes plans incrementally (model-predictive control)
- Handles replanning when needed
"""

import logging
import time
from abc import ABC, abstractmethod
from overrides import override

from ...base import Agent
from ...models import (
    Action,
    ActionPlan,
    ActionStatus,
    CacheContext,
    PlanningContext,
    PlanningParameters,
    PlanningStrategy,
    PlanStatus,
)
from .strategies import (
    ActionPlanningStrategy,
    get_default_planning_strategy,
)
from .cache import ActionPlanningCachePolicy
from .coordination import ActionPlanCoordinationPolicy
from .learning import ActionPlanLearningPolicy


logger = logging.getLogger(__name__)



class ActionPlanner(ABC):
    """Policy for creating and revising agent action plans.

    Implementations:
    - SequentialPlanner: Manually-specified linear sequence
    - CacheAwareActionPlanner: LLM-driven planning via pluggable ActionPlanningStrategy
    """

    @abstractmethod
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Create initial plan to achieve goal.

        Args:
            planning_context: Current context (goals, constraints, available resources, state, custom data, etc.)
        Returns:
            ActionPlan with ordered actions
        """
        ...

    @abstractmethod
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext
    ) -> ActionPlan:
        """Revise plan based on critique and new information.

        Args:
            current_plan: Current plan
            planning_context: Current context (goals, constraints, available resources, state, custom data, etc.)
            critique: Critique suggesting revision

        Returns:
            Revised plan
        """
        # TODO: Remove critique? It should be part of planning_context?
        ...

    @abstractmethod
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        """Learn from completed plan execution.

        Args:
            plan: Completed plan to learn from
        """
        pass



# ============================================================================
# Default Implementations
# ============================================================================


class SequentialPlanner(ActionPlanner):
    """Simple manually-specified sequential planner for straightforward tasks.

    Creates linear sequence of actions.
    """

    def __init__(
        self,
        agent: Agent,
        planning_params: PlanningParameters,
    ):
        """Initialize with optional action templates.

        Args:
            agent: Agent instance
            planning_params: Planning parameters
            action_templates: Predefined action sequence
        """
        self.agent = agent
        self.planning_params = planning_params

    @override
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Create manually-specified sequential plan."""
        if not planning_context.manual_plan:
            raise ValueError("manual_plan must be provided in context for manual plan creation")

        return ActionPlan(
            plan_id=f"plan_{self.agent.agent_id}_{int(time.time() * 1000)}",
            agent_id=self.agent.agent_id,
            goals=planning_context.goals,
            constraints=planning_context.constraints,
            generation_method="manual",
            strategy="sequential",
            actions=[Action(**a) for a in planning_context.manual_plan.actions],
            planning_horizon=self.planning_params.planning_horizon,
            replan_every_n_steps=self.planning_params.replan_every_n_steps,
            parent_plan_id=planning_context.parent_plan_id,
        )

    @override
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext
    ) -> ActionPlan:
        """Return the current plan without modification."""
        return current_plan

    @override
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        raise NotImplementedError("SequentialPlanner does not support learning from execution.")



class CacheAwareActionPlanner(ActionPlanner):
    """ONE planner class, customized via pluggable policies."""

    def __init__(
        self,
        agent: Agent,
        planning_strategy: ActionPlanningStrategy,
        planning_params: PlanningParameters,
        cache_policy: ActionPlanningCachePolicy | None = None,
        learning_policy: ActionPlanLearningPolicy | None = None,
        coordination_policy: ActionPlanCoordinationPolicy | None = None, # FIXME: Not used yet
    ):
        self.agent = agent
        self.agent_id = agent.agent_id
        self.planning_strategy = planning_strategy

        # Ensure strategy has agent reference
        if not planning_strategy.agent:
            planning_strategy.set_agent(agent)

        self.planning_params = planning_params
        self.cache_policy = cache_policy
        self.learning_policy = learning_policy
        self.coordination_policy = coordination_policy  # TODO: This is not used yet.

    async def initialize(self) -> None:
        """Initialize planner and its policies."""
        if self.cache_policy:
            await self.cache_policy.initialize()

        if self.learning_policy:
            await self.learning_policy.initialize()

        if self.coordination_policy:
            await self.coordination_policy.initialize()

    @override
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Create plan (LLM-generated or manual).

        Args:
            goals: List of goals to achieve
            planning_context: Planning context (goals, constraints, resources, etc.)
        """
        # Apply learning policy (get similar successful plans)
        learned_patterns = None
        if self.learning_policy:
            learned_patterns = await self.learning_policy.get_applicable_patterns(planning_context)

        # Apply cache policy (analyze cache requirements)
        cache_context = CacheContext()
        if self.cache_policy:
            cache_context = await self.cache_policy.analyze_cache_requirements(planning_context)

        # Generate plan via strategy
        logger.warning(
            f"\n"
            f"          ╔══════════════════════════════════════╗\n"
            f"          ║  🎯 PLANNER: calling strategy        ║\n"
            f"          ║  {self.planning_strategy.__class__.__name__:<36}║\n"
            f"          ╚══════════════════════════════════════╝"
        )
        plan: ActionPlan = await self.planning_strategy.generate_plan(
            planning_context=planning_context,
            params=self.planning_params,
            learned_patterns=learned_patterns,
            cache_context=cache_context,
        )
        logger.warning(
            f"          🎯 PLANNER: strategy returned plan with "
            f"{len(plan.actions)} actions, status={plan.status}"
        )

        # Set parent relationship
        plan.agent_id = self.agent_id
        plan.parent_plan_id = planning_context.parent_plan_id
        plan.cache_context = cache_context

        return plan

    @override
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext
    ) -> ActionPlan:
        """Replan next N steps (MPC)."""
        # Get learned patterns and cache context if policies are available
        learned_patterns = None
        if self.learning_policy:
            learned_patterns = await self.learning_policy.get_applicable_patterns(
                planning_context
            )

        cache_context = current_plan.cache_context if hasattr(current_plan, "cache_context") else None
        if not cache_context and self.cache_policy:
            cache_context = await self.cache_policy.analyze_cache_requirements(planning_context)
        else:
            cache_context = CacheContext()

        # TODO: Get critique here or from memory and add to the planning context for replanning
        ### critique: Critique = 
        ### # Simple strategy: add actions based on critique suggestions
        ### corrective_actions = []
        ### for suggestion in critique.suggestions:
        ###     # Convert suggestion to action (simple heuristic)
        ###     corrective_actions.append(
        ###         Action(
        ###             type=ActionType.CUSTOM,  # TODO: How are these actions handled by executor?
        ###             parameters={"suggestion": suggestion},
        ###             reasoning=f"Addressing critique: {suggestion}",
        ###         )
        ###     )

        ### # Prepend corrective actions (handle issues first)
        ### for action in reversed(corrective_actions):
        ###     current_plan.prepend_action(action)

        new_actions = await self.planning_strategy.replan_horizon(
            plan=current_plan,
            planning_context=planning_context,
            params=self.planning_params,
            learned_patterns=learned_patterns,
            cache_context=cache_context,
        )

        # Replace future actions
        current_plan.actions = (
            current_plan.actions[: current_plan.current_action_index] + new_actions
        )
        return current_plan

    @override
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        """Learn from completed plan execution.

        Args:
            plan: Completed plan to learn from
        """
        # Calculate outcome metrics
        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]
        failed_actions = [a for a in plan.actions if a.status == ActionStatus.FAILED]

        success_rate = (
            len(completed_actions) / len(plan.actions) if plan.actions else 0.0
        )

        # Determine overall outcome
        if plan.status == PlanStatus.COMPLETED:
            outcome_status = "success"
        elif plan.status == PlanStatus.FAILED:
            outcome_status = "failed"
        else:
            outcome_status = "partial"

        # Calculate duration
        duration_s = (
            plan.completed_at - plan.created_at if plan.completed_at else 0.0
        )

        # Calculate quality score (simple heuristic)
        quality_score = success_rate
        if plan.execution_context.findings:
            # Bonus for gathering information
            quality_score = min(1.0, quality_score + 0.1)

        # Build outcome dictionary
        # TODO: Convert this dict to a pydantic model for better structure and validation
        outcome = {
            "status": outcome_status,
            "duration_s": duration_s,
            "success_rate": success_rate,
            "actions_completed": len(completed_actions),
            "actions_failed": len(failed_actions),
            "quality_score": quality_score,
            "actual_cost": {
                "pages_loaded": len(plan.cache_context.working_set),
                "actions_executed": len(completed_actions),
                "children_spawned": len(plan.execution_context.spawned_children),
            },
        }

        # Let learning policy record and learn
        logger.info(
            f"Learning from plan {plan.plan_id}: {outcome_status} with {success_rate:.1%} success rate"
        )
        if self.learning_policy:
            await self.learning_policy.learn_from_execution(plan, outcome)





async def create_cache_aware_planner(
    agent: Agent,
    max_iterations: int = 5,
    quality_threshold: float = 0.8,
    planning_horizon: int = 5,
    ideal_cache_size: int = 10,
) -> CacheAwareActionPlanner:
    """Create sophisticated planner with cache-awareness and learning.

    Returns:
        CacheAwareActionPlanner wrapping planning.Planner
    """

    # Create planning parameters
    planning_params = PlanningParameters(
        strategy=PlanningStrategy.TOP_DOWN,  # TopDown works well for code analysis
        planning_horizon=planning_horizon,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        ideal_cache_size=ideal_cache_size,
        ### **agent.metadata.parameters.get("planning_params", {})  # FIXME: Get the planning parameters properly
    )

    # Create planning strategy
    planning_strategy: ActionPlanningStrategy = get_default_planning_strategy(
        planning_params, agent=agent
    )

    # Create default policies if not provided in metadata
    # Analyze cache needs and update working set
    cache_policy = ActionPlanningCachePolicy(
        agent=agent,
        cache_capacity=planning_params.ideal_cache_size
    )
    await cache_policy.initialize()
    logger.info(
        f"Created default ActionPlanningCachePolicy for agent {agent.agent_id}"
    )

    # Learn from execution outcomes
    learning_policy = ActionPlanLearningPolicy(agent=agent)
    await learning_policy.initialize()
    logger.info(
        f"Created default ActionPlanLearningPolicy for agent {agent.agent_id}"
    )

    coordination_policy = ActionPlanCoordinationPolicy(
        cache_capacity=planning_params.ideal_cache_size
    )
    await coordination_policy.initialize()
    logger.info(
        f"Created default ActionPlanCoordinationPolicy for agent {agent.agent_id}"
    )


    # Create sophisticated planner with policies
    cache_aware_planner = CacheAwareActionPlanner(
        agent=agent,
        planning_strategy=planning_strategy,
        planning_params=planning_params,
        learning_policy=learning_policy,
        cache_policy=cache_policy,
        coordination_policy=coordination_policy,
        # Policies will be created automatically in Agent.initialize()
        # when metadata doesn't provide them
    )
    await cache_aware_planner.initialize()

    return cache_aware_planner






