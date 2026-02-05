"""Replanning policies for adaptive plan execution.

This module provides the ReplanningPolicy abstraction that decides WHEN to replan
and WHAT revision strategy to use. This separates the replanning decision from
the replanning execution (which remains in ActionPlanner.revise_plan).

Core pattern:
- ReplanningPolicy: Protocol for replanning decisions (WHEN + WHAT strategy)
- ActionPlanner.revise_plan(): Executes the replanning (HOW)
- CacheAwareActionPolicy: Orchestrates the flow (calls policy, then planner)

Concrete policies:
- PeriodicReplanningPolicy: Replan every N steps (default, current behavior)
- AdaptiveReplanningPolicy: Trigger-based replanning (failure, blocked, quality, resources)
- CompositeReplanningPolicy: Combines multiple policies (triggers if ANY says yes)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ...models import (
    ActionPlan,
    ActionResult,
    ActionStatus,
    RevisionStrategy,
    RevisionTrigger,
)
from .evaluator import PlanEvaluator

logger = logging.getLogger(__name__)


class ReplanningDecision(BaseModel):
    """Decision from a replanning policy.

    Communicates whether replanning is needed, what triggered it,
    and what revision strategy to use.
    """

    should_replan: bool = False
    triggers: list[RevisionTrigger] = Field(default_factory=list)
    strategy: RevisionStrategy = RevisionStrategy.INCREMENTAL_REPAIR
    reason: str = ""


class ReplanningPolicy(ABC):
    """Policy that decides WHEN to replan and WHAT revision strategy to use.

    Separation of concerns:
    - ReplanningPolicy decides WHEN to replan and what strategy (this class)
    - ActionPlanner.revise_plan() decides HOW to replan (existing abstraction)
    - CacheAwareActionPolicy orchestrates the flow

    The policy receives the current plan and the result of the last executed
    action. It does NOT receive PlanningContext (which is expensive to build
    due to memory gathering). PlanningContext is only built inside
    _replan_horizon() when replanning is actually triggered.
    """

    @abstractmethod
    async def evaluate_replanning_need(
        self,
        plan: ActionPlan,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate whether the current plan needs replanning.

        Called after each action execution in plan_step().

        Args:
            plan: Current plan with execution state
            last_result: Result of last executed action (None on first call
                or when no previous action result is available)

        Returns:
            ReplanningDecision with should_replan, triggers, strategy, reason
        """
        ...


class PeriodicReplanningPolicy(ReplanningPolicy):
    """Replan every N completed steps. This is the default behavior.

    Also supports replanning on action failure (controlled by replan_on_failure).
    """

    def __init__(
        self,
        replan_every_n_steps: int = 3,
        replan_on_failure: bool = True,
    ):
        """Initialize periodic replanning policy.

        Args:
            replan_every_n_steps: Replan after every N completed actions
            replan_on_failure: Whether to trigger replanning on action failure
        """
        self.replan_every_n_steps = replan_every_n_steps
        self.replan_on_failure = replan_on_failure

    async def evaluate_replanning_need(
        self,
        plan: ActionPlan,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate based on periodic schedule and optional failure detection."""
        # Check failure trigger (if enabled)
        if self.replan_on_failure and last_result and not last_result.success:
            return ReplanningDecision(
                should_replan=True,
                triggers=[RevisionTrigger.FAILURE],
                strategy=RevisionStrategy.INCREMENTAL_REPAIR,
                reason=f"Action failed: {last_result.error}",
            )

        # Check periodic trigger
        completed = sum(
            1 for a in plan.actions if a.status == ActionStatus.COMPLETED
        )
        if (
            completed > 0
            and self.replan_every_n_steps > 0
            and completed % self.replan_every_n_steps == 0
        ):
            return ReplanningDecision(
                should_replan=True,
                triggers=[RevisionTrigger.PERIODIC],
                strategy=RevisionStrategy.INCREMENTAL_REPAIR,
                reason=f"Periodic replanning after {completed} completed actions",
            )

        return ReplanningDecision(should_replan=False)


class AdaptiveReplanningPolicy(ReplanningPolicy):
    """Adaptive replanning based on trigger detection and strategy selection.

    Absorbs the useful logic from AdaptivePlanExecutor:
    - Failure detection with resource vs. generic failure distinction
    - Blocked action detection
    - Quality threshold checking via PlanEvaluator
    - Resource exhaustion detection (cache pressure)
    - Strategy selection based on trigger priority

    Falls back to periodic replanning when no event-driven triggers fire.
    """

    def __init__(
        self,
        evaluator: PlanEvaluator | None = None,
        quality_threshold: float = 0.5,
        replan_every_n_steps: int = 3,
    ):
        """Initialize adaptive replanning policy.

        Args:
            evaluator: PlanEvaluator for quality threshold checks.
                Created with defaults if not provided.
            quality_threshold: Quality score below which replanning triggers
            replan_every_n_steps: Periodic fallback (0 to disable)
        """
        # Lazy import to avoid circular dependency at module level
        from .evaluator import PlanEvaluator as _PlanEvaluator

        self.evaluator = evaluator or _PlanEvaluator()
        self.quality_threshold = quality_threshold
        self.replan_every_n_steps = replan_every_n_steps

    async def evaluate_replanning_need(
        self,
        plan: ActionPlan,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate using multi-signal trigger detection + strategy selection."""
        triggers = await self._detect_triggers(plan, last_result)

        if not triggers:
            # Also check periodic (as fallback)
            if self.replan_every_n_steps > 0:
                completed = sum(
                    1 for a in plan.actions if a.status == ActionStatus.COMPLETED
                )
                if completed > 0 and completed % self.replan_every_n_steps == 0:
                    triggers.append(RevisionTrigger.PERIODIC)

        if not triggers:
            return ReplanningDecision(should_replan=False)

        strategy = self._select_strategy(triggers, plan)
        return ReplanningDecision(
            should_replan=True,
            triggers=triggers,
            strategy=strategy,
            reason=f"Triggers detected: {[t.value for t in triggers]}",
        )

    async def _detect_triggers(
        self,
        plan: ActionPlan,
        last_result: ActionResult | None,
    ) -> list[RevisionTrigger]:
        """Detect revision triggers from plan state and last action result.

        Absorbed from AdaptivePlanExecutor._detect_revision_triggers().
        """
        triggers: list[RevisionTrigger] = []

        if last_result:
            # Check for action failure (distinguish resource vs. generic)
            if not last_result.success:
                if last_result.error and "resource" in str(last_result.error).lower():
                    triggers.append(RevisionTrigger.RESOURCE_EXHAUSTION)
                else:
                    triggers.append(RevisionTrigger.FAILURE)

            # Check for blocked action
            if last_result.blocked:
                triggers.append(RevisionTrigger.BLOCKED)

        # Check quality threshold via evaluator
        # PlanEvaluator.evaluate() only reads plan fields (actions, cache_context,
        # goals, depth, depends_on) — it does not use PlanningContext.
        try:
            from ...models import PlanningContext as _PC

            evaluation = await self.evaluator.evaluate(plan, _PC())
            if evaluation.benefit.expected_quality < self.quality_threshold:
                triggers.append(RevisionTrigger.QUALITY_THRESHOLD)
        except Exception as e:
            logger.warning(f"Plan evaluation failed during trigger detection: {e}")

        # Check resource exhaustion (cache pressure)
        if (
            plan.cache_context.ideal_cache_size > 0
            and len(plan.cache_context.working_set)
            > plan.cache_context.ideal_cache_size
        ):
            if RevisionTrigger.RESOURCE_EXHAUSTION not in triggers:
                triggers.append(RevisionTrigger.RESOURCE_EXHAUSTION)

        return triggers

    def _select_strategy(
        self,
        triggers: list[RevisionTrigger],
        plan: ActionPlan,
    ) -> RevisionStrategy:
        """Select revision strategy based on trigger priority.

        Absorbed from AdaptivePlanExecutor._select_revision_strategy().
        Priority order (highest first):
        1. RESOURCE_EXHAUSTION → REPLAN_FROM_SCRATCH
        2. QUALITY_THRESHOLD → INCREMENTAL_REPAIR
        3. BLOCKED → REORDER_ACTIONS
        4. FAILURE → INCREMENTAL_REPAIR (first time), BACKTRACK (if already revised)
        """
        if RevisionTrigger.RESOURCE_EXHAUSTION in triggers:
            return RevisionStrategy.REPLAN_FROM_SCRATCH

        if RevisionTrigger.QUALITY_THRESHOLD in triggers:
            return RevisionStrategy.INCREMENTAL_REPAIR

        if RevisionTrigger.BLOCKED in triggers:
            return RevisionStrategy.REORDER_ACTIONS

        if RevisionTrigger.FAILURE in triggers:
            if plan.version > 1:
                # Already revised at least once, escalate to backtrack
                return RevisionStrategy.BACKTRACK
            else:
                # First failure, try incremental repair
                return RevisionStrategy.INCREMENTAL_REPAIR

        # Default for periodic or other triggers
        return RevisionStrategy.INCREMENTAL_REPAIR


class CompositeReplanningPolicy(ReplanningPolicy):
    """Runs multiple replanning policies. Triggers if ANY sub-policy says yes.

    When multiple policies trigger simultaneously, merges all triggers and
    selects the highest-priority strategy.
    """

    # Strategy priority (higher number = more aggressive)
    STRATEGY_PRIORITY: dict[RevisionStrategy, int] = {
        RevisionStrategy.REPLAN_FROM_SCRATCH: 6,
        RevisionStrategy.BACKTRACK: 5,
        RevisionStrategy.SUBSTITUTE_ACTIONS: 4,
        RevisionStrategy.ADD_ACTIONS: 3,
        RevisionStrategy.REMOVE_ACTIONS: 3,
        RevisionStrategy.REORDER_ACTIONS: 2,
        RevisionStrategy.INCREMENTAL_REPAIR: 1,
    }

    def __init__(self, policies: list[ReplanningPolicy]):
        """Initialize composite replanning policy.

        Args:
            policies: List of replanning policies to evaluate
        """
        if not policies:
            raise ValueError("CompositeReplanningPolicy requires at least one policy")
        self.policies = policies

    async def evaluate_replanning_need(
        self,
        plan: ActionPlan,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate all sub-policies. Trigger if ANY says yes."""
        decisions: list[ReplanningDecision] = []
        for policy in self.policies:
            decision = await policy.evaluate_replanning_need(plan, last_result)
            if decision.should_replan:
                decisions.append(decision)

        if not decisions:
            return ReplanningDecision(should_replan=False)

        # Merge: collect all triggers, use highest-priority strategy
        all_triggers: list[RevisionTrigger] = []
        all_reasons: list[str] = []
        best_strategy = RevisionStrategy.INCREMENTAL_REPAIR
        best_priority = 0

        for d in decisions:
            all_triggers.extend(d.triggers)
            if d.reason:
                all_reasons.append(d.reason)
            priority = self.STRATEGY_PRIORITY.get(d.strategy, 0)
            if priority > best_priority:
                best_priority = priority
                best_strategy = d.strategy

        # Deduplicate triggers while preserving order
        unique_triggers: list[RevisionTrigger] = list(dict.fromkeys(all_triggers))

        return ReplanningDecision(
            should_replan=True,
            triggers=unique_triggers,
            strategy=best_strategy,
            reason="; ".join(all_reasons),
        )
