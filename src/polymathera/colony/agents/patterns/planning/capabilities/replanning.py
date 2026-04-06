"""Replanning policies and capability for adaptive plan execution.

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

ReplanningCapability provides replanning trigger evaluation so agents can check whether
their current plan should be revised. Wraps the composable
``ReplanningPolicy`` system.

Dual interface:
- **Programmatic API**: ``evaluate_replanning_need()`` — used by
  ``CacheAwareActionPolicy`` in its pre-programmed loop.
- **LLM API**: ``@action_executor`` methods — used by ``MinimalActionPolicy``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from overrides import override
from pydantic import BaseModel, Field
from typing import Any, TYPE_CHECKING

from ....base import Agent, AgentCapability
from ...actions.dispatcher import action_executor
from ....scopes import BlackboardScope, get_scope_prefix
from ....models import (
    ActionPlan,
    ActionResult,
    ActionStatus,
    RevisionStrategy,
    RevisionTrigger,
    ActionPolicyExecutionState,
    PlanExhaustionBehavior,
    AgentSuspensionState,
)
if TYPE_CHECKING:
    from .evaluator import PlanEvaluator

logger = logging.getLogger(__name__)


class ReplanningDecision(BaseModel):
    """Decision from a replanning policy.

    Communicates whether replanning is needed, what triggered it,
    and what revision strategy to use.
    """

    should_replan: bool = False
    plan_exhausted: bool = False
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
        state: ActionPolicyExecutionState,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate whether the current plan needs replanning.

        Called after each action execution in plan_step().

        Args:
            state: Current execution state of the action policy
            last_result: Result of last executed action (None on first call
                or when no previous action result is available)

        Returns:
            ReplanningDecision with should_replan, triggers, strategy, reason
        """
        ...

    @abstractmethod
    def reset_state(self, state: ActionPolicyExecutionState) -> None:
        """Reset any internal state of the replanning policy.

        Called when the current plan is abandoned or completed but the agent
        is running in continuous mode. This allows policies to clear
        counters, flags, or other state that should not persist across plans.
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

    @override
    def reset_state(self, state: ActionPolicyExecutionState) -> None:
        """No internal state to reset for periodic replanning."""
        pass

    @override
    async def evaluate_replanning_need(
        self,
        state: ActionPolicyExecutionState,
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
            1 for a in state.current_plan.actions if a.status == ActionStatus.COMPLETED
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

    @override
    def reset_state(self, state: ActionPolicyExecutionState) -> None:
        """No internal state to reset for periodic replanning."""
        pass

    @override
    async def evaluate_replanning_need(
        self,
        state: ActionPolicyExecutionState,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate using multi-signal trigger detection + strategy selection."""
        triggers = await self._detect_triggers(state.current_plan, last_result)

        if not triggers:
            # Also check periodic (as fallback)
            if self.replan_every_n_steps > 0:
                completed = sum(
                    1 for a in state.current_plan.actions if a.status == ActionStatus.COMPLETED
                )
                if completed > 0 and completed % self.replan_every_n_steps == 0:
                    triggers.append(RevisionTrigger.PERIODIC)

        if not triggers:
            return ReplanningDecision(should_replan=False)

        strategy = self._select_strategy(triggers, state.current_plan)
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
            from ....models import PlanningContext as _PC

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


class PlanExhaustionReplanningPolicy(ReplanningPolicy):
    """Triggers replanning when plan is exhausted (no pending actions) but goals not met."""

    def __init__(
        self,
        plan_exhaustion_behavior: PlanExhaustionBehavior = PlanExhaustionBehavior.REPLAN,
        max_replan_cycles: int = 5,
    ):
        """Initialize plan exhaustion replanning policy.

        Args:
            plan_exhaustion_behavior: Behavior when plan is exhausted
            max_replan_cycles: Maximum number of replan cycles
        """
        self._plan_exhaustion_behavior = plan_exhaustion_behavior
        self._max_replan_cycles = max_replan_cycles

    @override
    def reset_state(self, state: ActionPolicyExecutionState) -> None:
        """Reset internal state for periodic replanning."""
        state.custom["_replan_cycle_count"] = 0

    @override
    async def evaluate_replanning_need(
        self,
        state: ActionPolicyExecutionState,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate if replanning is needed due to plan exhaustion."""
        if state.current_plan.current_action_index < len(state.current_plan.actions):
            return ReplanningDecision(
                should_replan=False,
                plan_exhausted=False,
            )

        # Plan horizon exhausted — all current actions executed
        replan_count = state.custom.get("_replan_cycle_count", 0)

        if (
            self._plan_exhaustion_behavior == PlanExhaustionBehavior.REPLAN
            and replan_count < self._max_replan_cycles
        ):
            # Try to extend the plan via replanning (MPC continuation).
            # The planner sees the full execution_context (completed actions + results)
            # and decides if more work is needed or if the goal is satisfied.
            state.custom["_replan_cycle_count"] = replan_count + 1
            logger.warning(
                f"      📋 PLAN_STEP: plan horizon exhausted, "
                f"attempting replan cycle {replan_count + 1}/{self._max_replan_cycles}"
            )
            return ReplanningDecision(
                should_replan=True,
                plan_exhausted=True,
                triggers=[RevisionTrigger.PLAN_EXHAUSTED],
                strategy=RevisionStrategy.ADD_ACTIONS,
                reason=(
                    f"Plan horizon exhausted after {len(state.current_plan.actions)} actions "
                    f"(replan cycle {replan_count + 1}/{self._max_replan_cycles})"
                ),
            )

        # True completion: plan exhausted AND (no replan OR planner says done OR budget exceeded)
        return ReplanningDecision(
            should_replan=False,
            plan_exhausted=True,
            reason=(
                f"Plan horizon exhausted after {len(state.current_plan.actions)} actions "
                f"(replan cycle {replan_count + 1}/{self._max_replan_cycles})"
            ),
        )


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

    @override
    def reset_state(self, state: ActionPolicyExecutionState) -> None:
        """Reset the state for all sub-policies."""
        for policy in self.policies:
            policy.reset_state(state)

    @override
    async def evaluate_replanning_need(
        self,
        state: ActionPolicyExecutionState,
        last_result: ActionResult | None,
    ) -> ReplanningDecision:
        """Evaluate all sub-policies. Trigger if ANY says yes."""
        replan_decisions: list[ReplanningDecision] = []
        plan_exhausted = False
        exhaustion_reason = ""
        for policy in self.policies:
            decision = await policy.evaluate_replanning_need(state, last_result)
            if decision.plan_exhausted:
                plan_exhausted = True
                if decision.reason:
                    exhaustion_reason = decision.reason
            if decision.should_replan:
                replan_decisions.append(decision)

        if not replan_decisions:
            return ReplanningDecision(
                should_replan=False,
                plan_exhausted=plan_exhausted,
                reason=exhaustion_reason,
            )

        # Merge: collect all triggers, use highest-priority strategy
        all_triggers: list[RevisionTrigger] = []
        all_reasons: list[str] = []
        best_strategy = RevisionStrategy.INCREMENTAL_REPAIR
        best_priority = 0

        for d in replan_decisions:
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
            plan_exhausted=plan_exhausted,
            triggers=unique_triggers,
            strategy=best_strategy,
            reason="; ".join(all_reasons),
        )



class ReplanningCapability(AgentCapability):
    """Replanning trigger evaluation.

    Evaluates whether the current plan should be revised based on
    execution progress, failures, resource exhaustion, or quality
    threshold breaches. Does NOT execute replanning — only decides
    WHEN to replan and WHAT strategy to use.

     Defaults to CompositeReplanningPolicy with PeriodicReplanningPolicy + PlanExhaustionReplanningPolicy.

    Usage::

        # Programmatic API (used by CacheAwareActionPolicy)
        decision = await cap.evaluate_replanning_need(state, last_result)
        if decision.should_replan:
            await planner.revise_plan(plan, context)

        # LLM API
        # should_replan — returns whether to revise and why
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "replanning",
        replan_every_n_steps: int = 3,
        replan_on_failure: bool = True,
        max_replan_cycles: int = 5,
        input_patterns: list[str] | None = None,
        capability_key: str = "replanning",
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=input_patterns or [],
            capability_key=capability_key,
        )
        self._replan_every_n_steps = replan_every_n_steps
        self._replan_on_failure = replan_on_failure
        self._max_replan_cycles = max_replan_cycles
        self._composite_policy = None

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"planning"})

    def get_action_group_description(self) -> str:
        return (
            "Replanning — evaluates whether the current plan should be revised. "
            "Checks for failures, plan exhaustion, periodic triggers, and "
            "resource exhaustion. Returns recommended revision strategy."
        )

    async def _ensure_initialized(self) -> None:
        """Lazily initialize the composite replanning policy."""
        if self._composite_policy is not None:
            return

        self._composite_policy = CompositeReplanningPolicy(policies=[
            PeriodicReplanningPolicy(
                replan_every_n_steps=self._replan_every_n_steps,
                replan_on_failure=self._replan_on_failure,
            ),
            PlanExhaustionReplanningPolicy(
                max_replan_cycles=self._max_replan_cycles,
            ),
        ])

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    # =========================================================================
    # Programmatic API
    # =========================================================================

    async def evaluate_replanning_need(
        self,
        state: ActionPolicyExecutionState,
        last_result: ActionResult | None = None,
    ) -> Any:
        """Evaluate whether the current plan should be revised.

        Args:
            state: Current execution state with plan, iteration count, etc.
            last_result: Result of the most recent action (if any).

        Returns:
            ReplanningDecision with should_replan, triggers, strategy, reason.
        """
        await self._ensure_initialized()
        return await self._composite_policy.evaluate_replanning_need(state, last_result)

    def reset_state(self, state: ActionPolicyExecutionState) -> None:
        """Reset replanning state counters (e.g., replan cycle count)."""
        if self._composite_policy:
            self._composite_policy.reset_state(state)

    # =========================================================================
    # LLM API (@action_executor)
    # =========================================================================

    @action_executor(
        planning_summary=(
            "Check if the current plan should be revised based on progress, "
            "failures, or resource constraints. Returns whether to replan and why."
        ),
    )
    async def should_replan(self) -> dict[str, Any]:
        """Evaluate whether the current plan needs revision.

        Uses the agent's execution state to check for replanning triggers:
        failures, plan exhaustion, periodic threshold, resource pressure.

        Returns:
            Dict with 'should_replan', 'triggers', 'strategy', and 'reason'.
        """
        await self._ensure_initialized()

        # Build a minimal state from agent context
        # NOTE: In the pre-programmed path (CacheAwareActionPolicy),
        # the full state is passed. Here we construct what we can.
        state = ActionPolicyExecutionState(
            iteration_count=0,
            custom={},
        )

        decision = await self._composite_policy.evaluate_replanning_need(state, None)

        return {
            "should_replan": decision.should_replan,
            "plan_exhausted": decision.plan_exhausted,
            "triggers": [t.value if hasattr(t, 'value') else str(t) for t in decision.triggers],
            "strategy": decision.strategy.value if hasattr(decision.strategy, 'value') else str(decision.strategy),
            "reason": decision.reason,
        }



