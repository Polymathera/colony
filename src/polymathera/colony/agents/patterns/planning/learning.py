"""Plan learning components for execution history and pattern extraction.

This module provides:
- ExecutionHistoryStore: Persistent storage for plan execution records
- PatternLearner: Extracts reusable patterns from execution history
- CostModelTrainer: Refines cost estimates based on actual execution data
"""

import logging
from typing import Any

from ...models import (
    CostModel,
    ActionPlan,
    PlanExecutionRecord,
    PlanPattern,
    PlanningContext,
)
from ...blackboard import EnhancedBlackboard

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
        key = f"execution:{record.plan_id}"
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
            namespace="execution:*",
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
            namespace="execution:*",
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
            namespace="execution:*",
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
