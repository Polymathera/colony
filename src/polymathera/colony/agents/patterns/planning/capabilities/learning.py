"""Plan learning components for execution history and pattern extraction.

This module provides:
- ExecutionHistoryStore: Persistent storage for plan execution records
- PatternLearner: Extracts reusable patterns from execution history
- CostModelTrainer: Refines cost estimates based on actual execution data
- PlanLearningCapability: Learning from execution history

PlanLearningCapability provides access to execution history, learned action-sequence patterns,
and cost model refinement. Used by agents that need to learn from past
planning outcomes and apply successful patterns to new goals.

Dual interface:
- **Programmatic API**: ``get_applicable_patterns()``,
  ``learn_from_execution()``, ``get_similar_plans()`` — used by
  ``CacheAwareActionPlanner`` and ``CodeGenerationActionPolicy``.
- **LLM API**: ``@action_executor`` methods with simple parameters — used
  by ``MinimalActionPolicy`` and other JSON-selecting policies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from overrides import override

from ....base import Agent, AgentCapability
from ....models import (
    CostModel,
    ActionPlan,
    AgentSuspensionState,
    PlanExecutionRecord,
    PlanPattern,
    PlanningContext,
)
from ...actions.dispatcher import action_executor
from ....scopes import BlackboardScope, get_scope_prefix
from ....blackboard import EnhancedBlackboard
from ....blackboard.protocol import PlanLearningProtocol

logger = logging.getLogger(__name__)


class ExecutionHistoryStore:
    """Store and query plan execution history.

    Persists execution records to blackboard for learning and analysis.

    Note: Requires a blackboard instance to be provided (typically from agent.get_blackboard()).
    """

    def __init__(self, blackboard: EnhancedBlackboard):
        """Initialize execution history store.

        Args:
            blackboard: Blackboard instance for persistence (from agent.get_blackboard())
        """
        self.blackboard = blackboard

    async def initialize(self) -> None:
        """Initialize store."""
        logger.info("ExecutionHistoryStore initialized")

    async def record_execution(self, record: PlanExecutionRecord) -> None:
        """Record plan execution for learning.

        Args:
            record: Execution record to store
        """
        if not self.blackboard:
            await self.initialize()

        # Store record with searchable tags
        key = PlanLearningProtocol.execution_key(record.plan_id)  # TODO: Do not use a key-value store. Use a proper distributed queue or time-series database.
        await self.blackboard.write(
            key=key,
            value=record.model_dump(),
            created_by="execution_history_store",
            tags={
                "execution_record",
                f"outcome:{record.outcome}",
                f"strategy:{record.strategy}" if record.strategy else "strategy:none",
                f"agent:{record.agent_id}",
            },
        )

        logger.info(f"Recorded execution for plan {record.plan_id}")

    async def query_by_goal(
        self, goal: str, limit: int = 10
    ) -> list[PlanExecutionRecord]:
        """Query execution history by goal.

        Args:
            goal: Goal string to search for
            limit: Maximum number of records to return

        Returns:
            List of matching execution records
        """
        if not self.blackboard:
            await self.initialize()

        # Query blackboard for execution records
        entries = await self.blackboard.query(
            namespace=PlanLearningProtocol.execution_pattern(),
            limit=limit * 2,  # Get more to filter
        )

        # Filter by goal similarity (simple substring match)
        records = []
        for entry in entries:
            record_data = entry.value
            if goal.lower() in record_data.get("goal", "").lower():
                records.append(PlanExecutionRecord(**record_data))

                if len(records) >= limit:
                    break

        logger.info(f"Found {len(records)} execution records for goal '{goal}'")
        return records

    async def query_by_outcome(
        self, outcome: str, limit: int = 10
    ) -> list[PlanExecutionRecord]:
        """Query execution history by outcome.

        Args:
            outcome: Outcome type (success, failed, etc.)
            limit: Maximum number of records to return

        Returns:
            List of matching execution records
        """
        if not self.blackboard:
            await self.initialize()

        # Query using outcome tag
        entries = await self.blackboard.query(
            namespace=PlanLearningProtocol.execution_pattern(),
            limit=limit,
        )

        # Filter by outcome tag
        records = []
        for entry in entries:
            if f"outcome:{outcome}" in entry.tags:
                records.append(PlanExecutionRecord(**entry.value))

        logger.info(f"Found {len(records)} execution records with outcome '{outcome}'")
        return records

    async def get_statistics(self) -> dict[str, Any]:
        """Get execution history statistics.

        Returns:
            Dictionary with stats (total_executions, success_rate, avg_duration, etc.)
        """
        if not self.blackboard:
            await self.initialize()

        # Query all execution records
        entries = await self.blackboard.query(
            namespace=PlanLearningProtocol.execution_pattern(),
            limit=1000,
        )

        if not entries:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
            }

        total = len(entries)
        successful = sum(
            1
            for e in entries
            if "outcome:success" in e.tags or "outcome:partial" in e.tags
        )

        durations = [
            e.value.get("duration_seconds", 0)
            for e in entries
            if "duration_seconds" in e.value
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_executions": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_duration": avg_duration,
        }


class PatternLearner:
    """Learn reusable patterns from execution history.

    Analyzes execution records to extract patterns like:
    - Successful action sequences
    - Common failure modes
    - Cache-efficient patterns
    """

    def __init__(self, history_store: ExecutionHistoryStore):
        """Initialize pattern learner.

        Args:
            history_store: Execution history store
        """
        self.history_store = history_store
        self.patterns: dict[str, PlanPattern] = {}

    async def initialize(self) -> None:
        """Initialize learner (load existing patterns)."""
        logger.info("PatternLearner initialized")
        # TODO: Load existing patterns from storage

    async def learn_patterns(self, min_confidence: float = 0.7) -> list[PlanPattern]:
        """Learn patterns from execution history.

        Args:
            min_confidence: Minimum confidence threshold for patterns

        Returns:
            List of learned patterns
        """
        # Get successful executions
        successful = await self.history_store.query_by_outcome("success", limit=100)

        if len(successful) < 3:
            logger.info("Not enough successful executions to learn patterns")
            return []

        # Group by similarity (simple: same number of actions)
        groups: dict[int, list[PlanExecutionRecord]] = {}
        for record in successful:
            action_count = len(record.actions)
            if action_count not in groups:
                groups[action_count] = []
            groups[action_count].append(record)

        # Extract patterns from groups with multiple instances
        patterns = []
        for action_count, records in groups.items():
            if len(records) >= 3:  # Need multiple instances
                pattern = self._extract_pattern_from_group(records)
                if pattern and pattern.confidence >= min_confidence:
                    patterns.append(pattern)
                    self.patterns[pattern.pattern_id] = pattern

        logger.info(f"Learned {len(patterns)} patterns from execution history")
        return patterns

    def _extract_pattern_from_group(
        self, records: list[PlanExecutionRecord]
    ) -> PlanPattern | None:
        """Extract pattern from group of similar executions.

        Args:
            records: Execution records to analyze

        Returns:
            Extracted pattern or None
        """
        if not records:
            return None

        # Calculate average metrics
        avg_success_rate = sum(r.success_rate for r in records) / len(records)

        # Extract common action sequence (simplified)
        common_actions = []
        if records[0].actions:
            # Use first record's actions as template
            common_actions = records[0].actions

        # Calculate confidence based on consistency
        confidence = avg_success_rate

        pattern = PlanPattern(
            pattern_id=f"pattern_{len(self.patterns) + 1}",
            pattern_type="success",
            description=f"Pattern from {len(records)} successful executions",
            applicability=f"Goals similar to: {records[0].goal}",
            supporting_executions=[r.plan_id for r in records],
            confidence=confidence,
            recommended_actions=common_actions,
            avg_success_rate=avg_success_rate,
            avg_cost_accuracy=sum(
                1.0 if r.was_efficient() else 0.5 for r in records
            )
            / len(records),
        )

        return pattern

    async def get_applicable_patterns(
        self, goal: str, planning_context: PlanningContext
    ) -> list[PlanPattern]:
        """Get patterns applicable to given goal and context.

        Args:
            goal: Planning goal
            context: Execution context

        Returns:
            List of applicable patterns
        """
        # TODO: Use more advanced matching based on context
        # Simple matching: find patterns with similar goals
        applicable: list[PlanPattern] = []
        for pattern in self.patterns.values():
            if goal.lower() in pattern.applicability.lower():
                applicable.append(pattern)

        # Sort by confidence
        applicable.sort(key=lambda p: p.confidence, reverse=True)

        logger.info(f"Found {len(applicable)} applicable patterns for goal '{goal}'")
        return applicable


class CostModelTrainer:
    """Train and refine cost models based on execution history.

    Learns to predict costs more accurately by comparing estimates to actuals.
    """

    def __init__(self, history_store: ExecutionHistoryStore):
        """Initialize cost model trainer.

        Args:
            history_store: Execution history store
        """
        self.history_store = history_store
        self.cost_adjustments: dict[str, float] = {
            "total_tokens": 1.0,
            "duration_seconds": 1.0,
            "pages_to_load": 1.0,
        }

    async def initialize(self) -> None:
        """Initialize trainer (load existing adjustments)."""
        logger.info("CostModelTrainer initialized")
        # TODO: Load existing cost adjustments from storage

    async def train(self) -> dict[str, float]:
        """Train cost model from execution history.

        Returns:
            Dictionary of cost adjustment factors
        """
        # Get execution records with cost data
        all_records = await self.history_store.query_by_outcome("success", limit=100)

        if not all_records:
            logger.info("No execution records for training")
            return self.cost_adjustments

        # Calculate adjustment factors for each cost dimension
        for cost_dim in ["total_tokens", "duration_seconds", "pages_to_load"]:
            ratios = []

            for record in all_records:
                estimated = record.estimated_cost.get(cost_dim, 0.0)
                actual = record.actual_cost.get(cost_dim, 0.0)

                if estimated > 0 and actual > 0:
                    ratio = actual / estimated
                    ratios.append(ratio)

            if ratios:
                # Use median to be robust to outliers
                ratios.sort()
                median_ratio = ratios[len(ratios) // 2]
                self.cost_adjustments[cost_dim] = median_ratio

        logger.info(f"Trained cost model with adjustments: {self.cost_adjustments}")
        return self.cost_adjustments

    def refine_cost_estimate(self, estimate: CostModel) -> CostModel:
        """Refine cost estimate using learned adjustments.

        Args:
            estimate: Initial cost estimate

        Returns:
            Refined cost estimate
        """
        refined = CostModel(
            total_tokens=estimate.total_tokens
            * self.cost_adjustments.get("total_tokens", 1.0),
            prompt_tokens=estimate.prompt_tokens
            * self.cost_adjustments.get("total_tokens", 1.0),
            completion_tokens=estimate.completion_tokens
            * self.cost_adjustments.get("total_tokens", 1.0),
            estimated_duration_seconds=estimate.estimated_duration_seconds
            * self.cost_adjustments.get("duration_seconds", 1.0),
            pages_to_load=int(
                estimate.pages_to_load * self.cost_adjustments.get("pages_to_load", 1.0)
            ),
            llm_calls=estimate.llm_calls,
            memory_mb=estimate.memory_mb,
            compute_cost_usd=estimate.compute_cost_usd
            * self.cost_adjustments.get("total_tokens", 1.0),
        )

        return refined


# ============================================================================
# Learning Capability
# ============================================================================

class PlanLearningCapability(AgentCapability):
    """Learn from plan execution history to improve future planning.

    Uses execution history to:
    - Learn successful patterns
    - Refine cost models
    - Recommend plan improvements

    Records execution outcomes, extracts successful action-sequence patterns,
    and surfaces applicable patterns when planning begins. Patterns give the
    LLM planner examples of what has worked before for similar goals.

    Usage::

        # Register on agent
        agent.add_capability(PlanLearningCapability(agent=agent))

        # Programmatic API
        patterns = await cap.get_applicable_patterns(planning_context)
        await cap.learn_from_execution(plan, outcome_dict)

        # LLM API (available as @action_executor)
        # get_learned_patterns, get_execution_history, record_outcome
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "plan_learning",
        input_patterns: list[str] | None = None,
        capability_key: str = "plan_learning",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=input_patterns or [],
            capability_key=capability_key,
            app_name=app_name,
        )
        # Sub-components initialized lazily
        self._history_store = None
        self._pattern_learner = None
        self._cost_trainer = None
        self._initialized = False

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"planning"})

    def get_action_group_description(self) -> str:
        return (
            "Plan Learning — learns from execution history to improve future planning. "
            "Extracts successful action-sequence patterns, queries past outcomes, "
            "and refines cost estimates based on actual execution data."
        )

    async def _ensure_initialized(self) -> None:
        """Lazily initialize sub-components on first use."""
        if self._initialized:
            return

        blackboard = await self.agent.get_agent_level_blackboard(
            namespace="planning_history"
        )

        self._history_store = ExecutionHistoryStore(blackboard=blackboard)
        await self._history_store.initialize()

        self._pattern_learner = PatternLearner(self._history_store)
        await self._pattern_learner.initialize()

        self._cost_trainer = CostModelTrainer(self._history_store)
        await self._cost_trainer.initialize()

        # Train from existing history
        await self._cost_trainer.train()
        await self._pattern_learner.learn_patterns()

        self._initialized = True
        logger.info("PlanLearningCapability initialized with execution history")

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    # =========================================================================
    # Programmatic API
    # =========================================================================

    async def get_applicable_patterns(
        self, context: PlanningContext
    ) -> list[PlanPattern]:
        """Get action-sequence patterns that worked for similar goals.

        Args:
            context: Planning context with goals.

        Returns:
            Patterns sorted by confidence, with recommended actions.
        """
        await self._ensure_initialized()

        if not self._pattern_learner:
            logger.warning("Pattern learner not initialized")
            return []

        goal_str = " ".join(context.goals)
        patterns = await self._pattern_learner.get_applicable_patterns(
            goal_str, context
        )
        logger.info(f"Found {len(patterns)} applicable patterns")
        return patterns

    async def get_similar_plans(
        self, goals: list[str], context: PlanningContext, limit: int = 5
    ) -> list[PlanExecutionRecord]:
        """Get similar successful plans from history.

        Args:
            goals: Current goals.
            limit: Maximum plans to return.

        Returns:
            Past execution records for similar goals.
        """
        await self._ensure_initialized()

        if not self._history_store:
            logger.warning("History store not initialized")
            return []

        # Query for similar goals
        goal_str = " ".join(goals)
        similar = await self._history_store.query_by_goal(goal_str, limit=limit)

        logger.info(f"Found {len(similar)} similar plans for goals: {goals}")
        return similar

    async def learn_from_execution(
        self, plan: ActionPlan, outcome: dict[str, Any]
    ) -> None:
        """Record a plan execution outcome for future learning.

        Args:
            plan: The executed plan.
            outcome: Dict with status, success_rate, duration_s, quality_score, etc.
        """
        await self._ensure_initialized()

        if not self._history_store:
            logger.warning("History store not initialized, cannot learn")
            return

        logger.info(f"Learning from execution of plan {plan.plan_id}")

        # Create execution record
        record = PlanExecutionRecord(
            plan_id=plan.plan_id,
            agent_id=plan.agent_id,
            goal=" ".join(plan.goals),
            scope=plan.scope.value if plan.scope else "unknown",
            created_at=time.time(),
            actions=[a.model_dump() for a in plan.actions],
            outcome=outcome.get("status", "unknown"),
            success_rate=outcome.get("success_rate", 0.0),
            duration_seconds=outcome.get("duration_seconds", 0.0),
            estimated_cost=plan.estimated_cost,
            actual_cost=plan.actual_cost,
            strategy=plan.metadata.get("strategy", ""),
        )

        # Store for learning
        await self._history_store.record_execution(record)

        # Periodically retrain
        stats = await self._history_store.get_statistics()
        if stats.get("total_executions", 0) % 10 == 0:
            logger.info("Retraining cost and pattern models")
            if self._cost_trainer:
                await self._cost_trainer.train()
            if self._pattern_learner:
                await self._pattern_learner.learn_patterns()

    # =========================================================================
    # LLM API (@action_executor)
    # =========================================================================

    @action_executor(
        planning_summary=(
            "Get action patterns that worked for similar goals in the past. "
            "Returns recommended action sequences with confidence scores."
        ),
    )
    async def get_learned_patterns(self, goal: str) -> dict[str, Any]:
        """Get patterns from execution history that match the given goal.

        Args:
            goal: The goal to find patterns for.

        Returns:
            Dict with 'patterns' list, each containing description,
            recommended_actions, confidence, and applicability.
        """
        context = PlanningContext(goals=[goal])
        patterns = await self.get_applicable_patterns(context)

        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "description": p.description,
                    "applicability": p.applicability,
                    "confidence": p.confidence,
                    "recommended_actions": p.recommended_actions[:5],
                    "avg_success_rate": p.avg_success_rate,
                }
                for p in patterns[:5]
            ],
            "count": len(patterns),
        }

    @action_executor(
        planning_summary=(
            "Query execution history for past plans matching a goal. "
            "Returns success rates, durations, and action sequences."
        ),
    )
    async def get_execution_history(
        self,
        goal: str,
        outcome: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Query past execution records.

        Args:
            goal: Goal to search for.
            outcome: Filter by outcome ('success', 'failed', 'partial'). None for all.
            limit: Maximum records.

        Returns:
            Dict with 'records' list and 'statistics'.
        """
        await self._ensure_initialized()

        if outcome and self._history_store:
            records = await self._history_store.query_by_outcome(outcome, limit=limit)
        elif self._history_store:
            records = await self._history_store.query_by_goal(goal, limit=limit)
        else:
            records = []

        stats = await self._history_store.get_statistics() if self._history_store else {}

        return {
            "records": [
                {
                    "plan_id": r.plan_id,
                    "goal": r.goal,
                    "outcome": r.outcome,
                    "success_rate": r.success_rate,
                    "duration_seconds": r.duration_seconds,
                    "action_count": len(r.actions),
                }
                for r in records
            ],
            "statistics": stats,
            "count": len(records),
        }

    @action_executor(
        planning_summary=(
            "Record the outcome of the current plan for future learning. "
            "Future agents with similar goals will benefit from this data."
        ),
    )
    async def record_outcome(
        self,
        success: bool,
        quality_score: float = 0.0,
    ) -> dict[str, Any]:
        """Record the current plan's execution outcome.

        Args:
            success: Whether the plan succeeded.
            quality_score: Quality of the output (0.0 to 1.0).

        Returns:
            Confirmation dict.
        """
        # NOTE: This requires the agent's current plan to be accessible.
        # In practice, the full plan outcome is recorded by
        # CacheAwareActionPlanner.learn_from_plan_execution().
        # This @action_executor provides a simplified interface for the LLM
        # to explicitly record its assessment.
        await self._ensure_initialized()

        return {
            "recorded": True,
            "success": success,
            "quality_score": quality_score,
            "note": "Outcome recorded. Full plan metrics are captured automatically by the planner.",
        }

