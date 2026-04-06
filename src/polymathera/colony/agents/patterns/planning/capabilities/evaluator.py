"""Plan evaluation and selection for cost-benefit-risk analysis.

This module provides plan evaluation and selection capabilities:
- PlanEvaluator: Evaluates plans based on cost, benefit, and risk
- PlanSelector: Selects best plan from candidates
- PlanEvaluationCapability: Agent capability for evaluating plans with both programmatic and LLM APIs

PlanEvaluationCapability provides cost-benefit-risk analysis for action plans. Used by agents
that need to assess plan quality before committing or to compare
alternative approaches.

Dual interface:
- **Programmatic API**: ``evaluate()`` — used by ``CacheAwareActionPlanner``
  and ``AdaptiveReplanningPolicy``.
- **LLM API**: ``@action_executor`` methods — used by ``MinimalActionPolicy``.
"""

from __future__ import annotations

import logging
from typing import Any
from overrides import override

from ....base import Agent, AgentCapability
from ....models import (
    ActionPlan,
    AgentSuspensionState,
    PlanEvaluation,
    PlanningContext,
    BenefitModel,
    CostModel,
    RiskModel,
)
from ...actions.dispatcher import action_executor
from ....scopes import BlackboardScope, get_scope_prefix


logger = logging.getLogger(__name__)


class PlanEvaluator:
    """Evaluate plans based on cost, benefit, and risk.

    Provides cost-benefit-risk analysis for plan selection and revision decisions.
    """

    def __init__(self, use_llm_evaluation: bool = False):
        """Initialize plan evaluator.

        Args:
            use_llm_evaluation: If True, use LLM for cost/benefit/risk estimation (slower but more accurate)
        """
        self.use_llm_evaluation = use_llm_evaluation

    async def evaluate(self, plan: ActionPlan, planning_context: PlanningContext) -> PlanEvaluation:
        """Evaluate plan and calculate utility score.

        Args:
            plan: Plan to evaluate
            planning_context: Execution context

        Returns:
            Complete plan evaluation with utility score
        """
        logger.info(f"Evaluating plan {plan.plan_id}")

        # Estimate cost, benefit, and risk
        cost = await self.estimate_cost(plan, planning_context)
        benefit = await self.estimate_benefit(plan, planning_context)
        risk = await self.estimate_risk(plan, planning_context)

        # Create evaluation
        evaluation = PlanEvaluation(
            plan_id=plan.plan_id,
            cost=cost,
            benefit=benefit,
            risk=risk,
        )

        # Calculate utility
        evaluation.calculate_utility()

        logger.info(
            f"Evaluated plan {plan.plan_id}: utility={evaluation.utility_score:.3f}, "
            f"cost={cost.total_tokens:.0f} tokens, benefit={benefit.expected_quality:.2f}"
        )

        return evaluation

    async def estimate_cost(self, plan: ActionPlan, planning_context: PlanningContext) -> CostModel:
        """Estimate execution cost for plan.

        Args:
            plan: Plan to estimate
            planning_context: Execution context

        Returns:
            Cost model with estimates
        """
        #==========================================================
        # TODO: Cost modeling needs to also use a reasoning LLM.
        # TODO: Make cost estimator a policy component for customization.
        #==========================================================

        cost = CostModel()

        # Count actions to estimate LLM calls
        cost.llm_calls = len(plan.actions)

        # Estimate tokens based on actions and cache context
        # Assume average of 2000 tokens per action (prompt + completion)
        cost.prompt_tokens = len(plan.actions) * 1500.0
        cost.completion_tokens = len(plan.actions) * 500.0
        cost.total_tokens = cost.prompt_tokens + cost.completion_tokens

        # Estimate duration (assume 2 seconds per LLM call)
        cost.estimated_duration_seconds = len(plan.actions) * 2.0

        # Pages to load from cache context
        cost.pages_to_load = len(plan.cache_context.working_set)

        # Estimate memory (pages * page size)
        cost.memory_mb = len(plan.cache_context.working_set) * 40  # 40 MB per page estimate

        # Estimate compute cost (rough: $0.01 per 1M tokens)
        cost.compute_cost_usd = (cost.total_tokens / 1_000_000) * 0.01

        return cost

    async def estimate_benefit(self, plan: ActionPlan, planning_context: PlanningContext) -> BenefitModel:
        """Estimate benefit of plan execution.

        Args:
            plan: Plan to estimate
            planning_context: Execution context

        Returns:
            Benefit model with estimates
        """
        #==========================================================
        # TODO: Benefit modeling needs to also use a reasoning LLM.
        # TODO: Make benefit estimator a policy component for customization.
        #==========================================================

        benefit = BenefitModel()

        # Estimate quality based on plan complexity
        # More comprehensive plans (more actions) may have higher quality
        action_count = len(plan.actions)
        if action_count > 10:
            benefit.expected_quality = 0.9
        elif action_count > 5:
            benefit.expected_quality = 0.75
        else:
            benefit.expected_quality = 0.6

        # Information gain based on working set size
        # More pages analyzed = more information
        benefit.information_gain = min(len(plan.cache_context.working_set) / 20.0, 1.0)

        # Goal progress (assume each goal contributes equally)
        benefit.goal_progress = min(len(plan.goals) * 0.3, 1.0)

        # Learning value (hierarchical plans have higher learning value)
        if plan.get_depth() > 0:
            benefit.learning_value = 0.8
        else:
            benefit.learning_value = 0.4

        # Reusability (plans with sub-plans are more reusable)
        composite_actions = sum(1 for a in plan.actions if a.is_composite())
        if composite_actions > 0:
            benefit.reusability = 0.7
        else:
            benefit.reusability = 0.3

        return benefit

    async def estimate_risk(self, plan: ActionPlan, planning_context: PlanningContext) -> RiskModel:
        """Estimate execution risk for plan.

        Args:
            plan: Plan to estimate
            planning_context: Execution context

        Returns:
            Risk model with estimates
        """
        #==========================================================
        # TODO: Risk modeling needs to also use a reasoning LLM.
        # TODO: Make risk estimator a policy component for customization.
        #==========================================================

        risk = RiskModel()

        # Failure probability based on plan complexity
        # More complex plans = higher risk
        action_count = len(plan.actions)
        if action_count > 20:
            risk.failure_probability = 0.3
        elif action_count > 10:
            risk.failure_probability = 0.15
        else:
            risk.failure_probability = 0.05

        # Partial completion risk (hierarchical plans have lower risk)
        if plan.get_depth() > 0:
            risk.partial_completion_risk = 0.2  # Can fall back to partial results
        else:
            risk.partial_completion_risk = 0.4

        # Resource exhaustion risk based on cache requirements
        cache_ratio = len(plan.cache_context.working_set) / max(plan.cache_context.ideal_cache_size, 1)
        if cache_ratio > 1.0:
            risk.resource_exhaustion_risk = 0.7  # Working set exceeds capacity
        elif cache_ratio > 0.8:
            risk.resource_exhaustion_risk = 0.4
        else:
            risk.resource_exhaustion_risk = 0.1

        # Coordination conflict risk (if plan depends on others)
        if len(plan.depends_on) > 0:
            risk.coordination_conflict_risk = 0.3
        else:
            risk.coordination_conflict_risk = 0.1

        # Max loss if failed (in tokens wasted)
        estimated_cost = await self.estimate_cost(plan, planning_context)
        risk.max_loss_if_failed = estimated_cost.total_tokens

        return risk

    def calculate_utility(self, evaluation: PlanEvaluation) -> float:
        """Calculate utility score from evaluation.

        Args:
            evaluation: Plan evaluation

        Returns:
            Utility score (higher is better)
        """
        return evaluation.calculate_utility()


class PlanSelector:
    """Select best plan from candidates based on utility.

    Uses PlanEvaluator to evaluate and rank candidate plans.
    """

    def __init__(self, evaluator: PlanEvaluator | None = None):
        """Initialize plan selector.

        Args:
            evaluator: Plan evaluator (creates default if None)
        """
        self.evaluator = evaluator or PlanEvaluator()

    async def select_best_plan(
        self, candidate_plans: list[ActionPlan], planning_context: PlanningContext
    ) -> ActionPlan | None:
        """Select best plan from candidates.

        Args:
            candidate_plans: List of candidate plans
            context: Execution context

        Returns:
            Best plan, or None if no candidates
        """
        if not candidate_plans:
            return None

        logger.info(f"Selecting best plan from {len(candidate_plans)} candidates")

        # Evaluate all candidates
        evaluations: list[PlanEvaluation] = []
        for plan in candidate_plans:
            evaluation = await self.evaluator.evaluate(plan, planning_context)
            evaluations.append(evaluation)

        # Rank plans by utility
        rankings = self.rank_plans(evaluations)

        # Select plan with highest utility
        best_plan_id, best_utility = rankings[0]
        best_plan = next(p for p in candidate_plans if p.plan_id == best_plan_id)

        logger.info(
            f"Selected plan {best_plan_id} with utility {best_utility:.3f} "
            f"from {len(candidate_plans)} candidates"
        )

        return best_plan

    def rank_plans(self, evaluations: list[PlanEvaluation]) -> list[tuple[str, float]]:
        """Rank plans by utility score.

        Args:
            evaluations: List of plan evaluations

        Returns:
            List of (plan_id, utility_score) tuples, sorted by utility (descending)
        """
        rankings = [(e.plan_id, e.utility_score) for e in evaluations]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class PlanEvaluationCapability(AgentCapability):
    """Cost-benefit-risk evaluation for action plans.

    Estimates execution cost (tokens, time, pages), benefit (quality,
    information gain), and risk (failure probability, resource exhaustion).
    Computes a utility score that balances all three.

    Usage::

        # Programmatic API
        evaluation = await cap.evaluate(plan, planning_context)
        print(f"Utility: {evaluation.utility}")

        # LLM API
        # evaluate_plan — returns utility, cost, benefit, risk breakdown
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "plan_evaluation",
        input_patterns: list[str] | None = None,
        capability_key: str = "plan_evaluation",
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=input_patterns or [],
            capability_key=capability_key,
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"planning"})

    def get_action_group_description(self) -> str:
        return (
            "Plan Evaluation — estimates cost, benefit, and risk of action plans. "
            "Computes a utility score to help decide whether to commit to a plan "
            "or revise it."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    # =========================================================================
    # Programmatic API
    # =========================================================================

    async def evaluate(
        self, plan: ActionPlan, context: PlanningContext | None = None
    ) -> PlanEvaluation:
        """Evaluate a plan's cost, benefit, and risk.

        Args:
            plan: The plan to evaluate.
            context: Planning context for additional signals.

        Returns:
            PlanEvaluation with cost, benefit, risk, and utility score.
        """
        evaluator = PlanEvaluator()
        return evaluator.evaluate(plan, context or PlanningContext(goals=plan.goals))

    # =========================================================================
    # LLM API (@action_executor)
    # =========================================================================

    @action_executor(
        planning_summary=(
            "Evaluate the quality of the current plan. Returns utility score, "
            "cost estimate, benefit estimate, and risk factors."
        ),
    )
    async def evaluate_plan(self) -> dict[str, Any]:
        """Evaluate the current plan's cost-benefit-risk tradeoff.

        Retrieves the agent's current plan from the plan blackboard and
        evaluates it.

        Returns:
            Dict with utility, cost, benefit, risk breakdowns, and recommendation.
        """
        from ..blackboard import PlanBlackboard

        try:
            plan_bb = PlanBlackboard(
                agent=self.agent,
                scope=BlackboardScope.COLONY,
                namespace="action_plans",
            )
            plan = await plan_bb.get_plan(self.agent.agent_id)
        except Exception:
            plan = None

        if not plan:
            return {
                "utility": 0.0,
                "note": "No active plan to evaluate",
            }

        evaluation = await self.evaluate(plan)
        return {
            "utility": evaluation.utility,
            "cost": evaluation.cost.model_dump() if evaluation.cost else {},
            "benefit": evaluation.benefit.model_dump() if evaluation.benefit else {},
            "risk": evaluation.risk.model_dump() if evaluation.risk else {},
            "recommendation": (
                "proceed" if evaluation.utility > 0.5 else "consider revising"
            ),
        }

