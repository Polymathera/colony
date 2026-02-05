"""Refinement policies for iterative improvement of analysis results.

This module implements refinement strategies that allow analysis results to be
improved incrementally as more context becomes available.

Key patterns:
- RefinementPolicy: Base interface for refinement strategies
- LLMRefinementPolicy: LLM-based refinement with new context
- IncrementalRefiner: Tracks refinement history and dependencies
- ProgressiveRefinementTracker: Monitors refinement progress
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from .scope import ScopeAwareResult
from ..base import Agent, AgentCapability
from .actions import action_executor


T = TypeVar('T')


class RefinementContext(BaseModel):
    """Context for refinement operations.

    Provides information to guide refinement:
    - New evidence or context
    - Refinement goals
    - Constraints
    """

    # TODO: Add critique field to capture specific feedback for refinement
    # and allow this refinement capability to be driven by the critique capability.

    new_evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="New evidence or context for refinement"
    )

    refinement_goal: str | None = Field(
        default=None,
        description="What aspect to refine (completeness, accuracy, etc.)"
    )

    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints on refinement (time, size, etc.)"
    )

    requesting_agent_id: str | None = Field(
        default=None,
        description="Agent requesting refinement"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional refinement context"
    )


class RefinementStep(BaseModel):
    """Record of a single refinement step.

    Tracks what changed and why during refinement.
    """

    step_number: int = Field(
        description="Step number in refinement sequence"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When refinement occurred"
    )

    previous_confidence: float = Field(
        description="Confidence before refinement"
    )

    new_confidence: float = Field(
        description="Confidence after refinement"
    )

    new_evidence_sources: list[str] = Field(
        default_factory=list,
        description="Sources of new evidence used in refinement"
    )

    changes_made: list[str] = Field(
        default_factory=list,
        description="Description of changes made"
    )

    refinement_reason: str | None = Field(
        default=None,
        description="Why this refinement was performed"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional refinement metadata"
    )


class RefinementPolicy(ABC, Generic[T]):
    """Abstract base class for refinement policies.

    Refinement policies define how to improve results when new context arrives.
    Different strategies:
    - LLM-based refinement (merge new context into existing)
    - Rule-based refinement (apply transformation rules)
    - Hybrid refinement (LLM + rules)
    """

    @abstractmethod
    async def should_refine(
        self,
        result: ScopeAwareResult[T],
        new_context: RefinementContext
    ) -> bool:
        """Decide if result should be refined with new context.

        Args:
            result: Existing result
            new_context: New context for refinement

        Returns:
            True if refinement would be beneficial
        """
        pass

    @abstractmethod
    async def refine(
        self,
        original: ScopeAwareResult[T],
        new_context: RefinementContext
    ) -> ScopeAwareResult[T]:
        """Refine result with new context.

        Args:
            original: Original result to refine
            new_context: New context for refinement

        Returns:
            Refined result
        """
        pass


class LLMRefinementPolicy(RefinementPolicy[T]):
    """LLM-based refinement that merges new context into existing result.

    Uses LLM to intelligently incorporate new evidence while preserving
    valid insights from the original result.
    """

    def __init__(self, agent: Agent, confidence_threshold: float = 0.7):
        """Initialize with LLM client.

        Args:
            agent: Agent for refinement
            confidence_threshold: Minimum confidence to skip refinement
        """
        self.agent = agent
        self.confidence_threshold = confidence_threshold

    async def should_refine(
        self,
        result: ScopeAwareResult[T],
        new_context: RefinementContext
    ) -> bool:
        """Check if new context is relevant for refinement.

        Args:
            result: Existing result
            new_context: New context

        Returns:
            True if new context addresses missing context or low confidence
        """
        # Refine if:
        # 1. Result is incomplete and we have new evidence
        # 2. Result has low confidence and new evidence might help
        # 3. Explicit refinement goal matches result needs

        if new_context.new_evidence:
            if not result.scope.is_complete or result.scope.confidence < self.confidence_threshold:
                return True

        if new_context.refinement_goal:
            # Check if new evidence addresses missing context
            for missing_item in result.scope.missing_context:
                if missing_item in str(new_context.new_evidence):
                    return True

        return False

    async def refine(
        self,
        original: ScopeAwareResult[T],
        new_context: RefinementContext
    ) -> ScopeAwareResult[T]:
        """Refine result using LLM.

        Args:
            original: Original result
            new_context: New context for refinement

        Returns:
            Refined result
        """
        # Build refinement prompt
        prompt = f"""Refine this analysis result with new context:

Original Result:
{original.content}

Original Scope:
- Complete: {original.scope.is_complete}
- Missing Context: {original.scope.missing_context}
- Confidence: {original.scope.confidence}

New Evidence:
{new_context.new_evidence}

Instructions:
1. Incorporate new evidence into the original result
2. Address any missing context items if evidence provides them
3. Preserve valid insights from the original
4. Update confidence based on new evidence
5. Mark as complete if all missing context is now addressed

Provide the refined result in the same format as the original.
"""

        # Get LLM response (placeholder - implementation depends on agent)
        response = await self.agent.infer(
            context_page_ids=self.agent.bound_pages,  # TODO: How to select pages?
            prompt=prompt,
            temperature=0.1,  # TODO: Make configurable - low temp for refinement
            max_tokens=500,   # TODO: Make configurable - short responses for refinement
            json_schema=T.model_json_schema()  # Structured output
        )
        refined_content = T.model_validate_json(response.generated_text)

        # Create refined result
        refined = original.clone_for_refinement()
        refined.content = refined_content  # TODO: Fix this type mismatch.

        # Update scope based on refinement
        if new_context.new_evidence:
            # Check if missing context was addressed
            refined.scope.updated_at = time.time()

        return refined


class RefinementGraph(BaseModel):
    """Graph tracking refinement dependencies between results.

    Tracks which results depend on which for refinement propagation.

    TODO: Use nx.DiGraph or similar for graph representation and complex graph operations.
    """

    nodes: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Result nodes in graph"
    )

    edges: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Refinement dependencies (result_id -> depends_on)"
    )

    def add_node(self, result_id: str, metadata: dict[str, Any]) -> None:
        """Add result node to graph.

        Args:
            result_id: Result ID
            metadata: Node metadata
        """
        self.nodes[result_id] = metadata
        if result_id not in self.edges:
            self.edges[result_id] = []

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Add refinement dependency.

        Args:
            dependent: Result that depends on dependency
            dependency: Result that dependent relies on
        """
        if dependent not in self.edges:
            self.edges[dependent] = []
        if dependency not in self.edges[dependent]:
            self.edges[dependent].append(dependency)

    def get_dependents(self, result_id: str) -> list[str]:
        """Get results that depend on this result.

        Args:
            result_id: Result ID

        Returns:
            List of dependent result IDs
        """
        dependents = []
        for dep_id, deps in self.edges.items():
            if result_id in deps:
                dependents.append(dep_id)
        return dependents

    def get_dependencies(self, result_id: str) -> list[str]:
        """Get results this result depends on.

        Args:
            result_id: Result ID

        Returns:
            List of dependency result IDs
        """
        return self.edges.get(result_id, [])

    def topological_sort(self) -> list[str]:
        """Get topological ordering of results.

        Returns:
            Ordered list of result IDs (dependencies before dependents)
        """
        # Simple topological sort using DFS
        visited = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            # Visit dependencies first
            for dep in self.edges.get(node_id, []):
                visit(dep)

            order.append(node_id)

        # Visit all nodes
        for node_id in self.nodes:
            visit(node_id)

        return order


class IncrementalRefiner(Generic[T]):
    """Tracks incremental refinement of results with feedback loops.

    Maintains:
    - History of refinements
    - Dependencies between results
    - Refinement graph

    Enables:
    - Progressive improvement
    - Feedback propagation
    - Refinement tracking
    """

    def __init__(
        self,
        refinement_policy: RefinementPolicy[T]
    ):
        """Initialize refiner.

        Args:
            refinement_policy: Policy for performing refinements
        """
        self.refinement_policy = refinement_policy
        self.partial_results: dict[str, ScopeAwareResult[T]] = {}
        self.refinement_history: list[RefinementStep] = []
        self.refinement_graph: RefinementGraph = RefinementGraph()

    async def add_result(
        self,
        result_id: str,
        result: ScopeAwareResult[T]
    ) -> list[tuple[str, ScopeAwareResult[T]]]:
        """Add result and trigger refinements of dependent results.

        Args:
            result_id: Unique ID for this result
            result: Result to add

        Returns:
            List of (result_id, refined_result) tuples for results that were refined
        """
        self.partial_results[result_id] = result

        # Find which existing results need refinement based on this new result
        refinements = []

        for existing_id, existing_result in self.partial_results.items():
            if existing_id == result_id:
                continue

            # Check if new result provides context for existing result
            if self._provides_context(result, existing_result):
                # Refine existing result with new result as context
                refinement_context = RefinementContext(
                    new_evidence={"related_result": result.to_blackboard_entry()},
                    refinement_goal="incorporate_related_findings"
                )

                if await self.refinement_policy.should_refine(existing_result, refinement_context):
                    refined = await self.refinement_policy.refine(existing_result, refinement_context)

                    # Track refinement
                    step = RefinementStep(
                        step_number=len(self.refinement_history) + 1,
                        previous_confidence=existing_result.scope.confidence,
                        new_confidence=refined.scope.confidence,
                        new_evidence_sources=[result_id],
                        changes_made=[f"Refined with context from {result_id}"],
                        refinement_reason="Related result provided missing context"
                    )
                    self.refinement_history.append(step)

                    # Update stored result
                    self.partial_results[existing_id] = refined
                    refinements.append((existing_id, refined))

                    # Track dependency
                    if existing_id not in self.refinement_graph:
                        self.refinement_graph[existing_id] = []
                    self.refinement_graph[existing_id].append(result_id)

        return refinements

    def _provides_context(
        self,
        new_result: ScopeAwareResult[T],
        existing_result: ScopeAwareResult[T]
    ) -> bool:
        """Check if new result provides context for existing result.

        Args:
            new_result: New result
            existing_result: Existing result

        Returns:
            True if new result might help refine existing result
        """
        # Check if new result is in existing result's related shards
        if existing_result.result_id in new_result.scope.related_shards:
            return True

        # Check if new result addresses missing context
        for missing_item in existing_result.scope.missing_context:
            if missing_item in str(new_result.content):
                return True

        # Check if results share related shards
        shared_shards = set(new_result.scope.related_shards) & set(existing_result.scope.related_shards)
        if shared_shards:
            return True

        return False

    def get_refinement_chain(self, result_id: str) -> list[str]:
        """Get chain of results that contributed to refining this result.

        Args:
            result_id: Result ID

        Returns:
            List of result IDs that contributed to refinements
        """
        return self.refinement_graph.get(result_id, [])

    def get_all_results(self) -> dict[str, ScopeAwareResult[T]]:
        """Get all current results.

        Returns:
            Dictionary of result_id -> result
        """
        return self.partial_results.copy()


class ProgressiveRefinementTracker:
    """Tracks progressive refinement across multiple results.

    Monitors:
    - Overall confidence trend
    - Completeness trend
    - Refinement velocity
    - Convergence detection
    """

    def __init__(self):
        """Initialize tracker."""
        self.refinement_history: list[RefinementStep] = []
        self.confidence_history: list[tuple[float, float]] = []  # (timestamp, avg_confidence)
        self.completeness_history: list[tuple[float, float]] = []  # (timestamp, completion_ratio)

    def record_refinement(self, step: RefinementStep) -> None:
        """Record a refinement step.

        Args:
            step: Refinement step to record
        """
        self.refinement_history.append(step)

    def update_metrics(
        self,
        results: dict[str, ScopeAwareResult]
    ) -> None:
        """Update tracking metrics.

        Args:
            results: Current results to analyze
        """
        if not results:
            return

        # Calculate average confidence
        confidences = [r.scope.confidence for r in results.values()]
        avg_confidence = sum(confidences) / len(confidences)
        self.confidence_history.append((time.time(), avg_confidence))

        # Calculate completion ratio
        complete_count = sum(1 for r in results.values() if r.scope.is_complete)
        completion_ratio = complete_count / len(results)
        self.completeness_history.append((time.time(), completion_ratio))

    def is_converging(self, window_size: int = 5) -> bool:
        """Check if refinement is converging (confidence stabilizing).

        Args:
            window_size: Number of recent steps to check

        Returns:
            True if confidence is stabilizing
        """
        if len(self.confidence_history) < window_size + 1:
            return False

        recent = self.confidence_history[-window_size:]
        confidences = [c for _, c in recent]

        # Check if variance is low (converging)
        if len(confidences) > 1:
            import statistics
            variance = statistics.variance(confidences)
            return variance < 0.01  # Confidence varying by less than 0.1

        return False

    def get_refinement_velocity(self) -> float:
        """Get rate of confidence improvement.

        Returns:
            Average confidence increase per refinement step
        """
        if len(self.refinement_history) < 2:
            return 0.0

        total_gain = sum(
            step.new_confidence - step.previous_confidence
            for step in self.refinement_history
        )

        return total_gain / len(self.refinement_history)

    def should_continue_refining(
        self,
        target_confidence: float = 0.9,
        max_steps: int = 10
    ) -> bool:
        """Decide if more refinement is needed.

        Args:
            target_confidence: Target confidence level
            max_steps: Maximum refinement steps

        Returns:
            True if should continue refining
        """
        # Stop if max steps reached
        if len(self.refinement_history) >= max_steps:
            return False

        # Stop if converged
        if self.is_converging():
            return False

        # Stop if target confidence reached
        if self.confidence_history and self.confidence_history[-1][1] >= target_confidence:
            return False

        # Continue if making progress
        velocity = self.get_refinement_velocity()
        return velocity > 0.01  # Still improving meaningfully


class RefinementCapability(AgentCapability):
    """Capability for refining analysis results with new context.

    Wraps a RefinementPolicy and exposes @action_executor methods for:
    - should_refine: Check if refinement is beneficial
    - refine_result: Refine a result with new context

    Uses dynamic lookup via self.agent.get_capability_by_type() for
    coordination with other capabilities.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None
    ):
        super().__init__(agent=agent, scope_id=scope_id or agent.agent_id)
        self.refinement_policy: RefinementPolicy | None = None
        self._refiner: IncrementalRefiner | None = None

    def set_policy(self, policy: RefinementPolicy) -> None:
        """Configure the refinement policy after instantiation.

        Args:
            policy: RefinementPolicy implementation to use
        """
        self.refinement_policy = policy
        self._refiner = IncrementalRefiner(policy)

    def _require_policy(self) -> RefinementPolicy:
        """Get policy, raising if not configured."""
        if self.refinement_policy is None:
            raise RuntimeError(
                f"RefinementCapability.set_policy() must be called before use. "
                f"Configure with a RefinementPolicy implementation."
            )
        return self.refinement_policy

    @action_executor()
    async def should_refine(
        self,
        result: ScopeAwareResult,
        new_context: RefinementContext
    ) -> bool:
        """Check if result should be refined with new context.

        Args:
            result: Existing result to potentially refine
            new_context: New context that may trigger refinement

        Returns:
            True if refinement would be beneficial
        """
        policy = self._require_policy()
        return await policy.should_refine(result, new_context)

    @action_executor()
    async def refine_result(
        self,
        original: ScopeAwareResult,
        new_context: RefinementContext
    ) -> ScopeAwareResult:
        """Refine result with new context (plannable by LLM).

        Args:
            original: Original result to refine
            new_context: New context for refinement

        Returns:
            Refined result with updated content and scope
        """
        policy = self._require_policy()
        return await policy.refine(original, new_context)

    @action_executor()
    async def add_result_and_refine(
        self,
        result_id: str,
        result: ScopeAwareResult
    ) -> list[tuple[str, ScopeAwareResult]]:
        """Add result and trigger refinements of dependent results.

        Uses IncrementalRefiner to track dependencies and propagate
        refinements to related results.

        Args:
            result_id: Unique ID for this result
            result: Result to add

        Returns:
            List of (result_id, refined_result) tuples for refined results
        """
        self._require_policy()
        if self._refiner is None:
            raise RuntimeError("Refiner not initialized")
        return await self._refiner.add_result(result_id, result)


# Utility functions

async def refine_result_with_evidence(
    result: ScopeAwareResult[T],
    evidence: dict[str, Any],
    policy: RefinementPolicy[T]
) -> ScopeAwareResult[T]:
    """Convenience function to refine result with new evidence.

    Args:
        result: Result to refine
        evidence: New evidence
        policy: Refinement policy

    Returns:
        Refined result
    """
    context = RefinementContext(new_evidence=evidence)

    if await policy.should_refine(result, context):
        return await policy.refine(result, context)
    else:
        return result


def should_trigger_refinement(
    result: ScopeAwareResult[T],
    confidence_threshold: float = 0.8
) -> bool:
    """Check if result should trigger refinement.

    Args:
        result: Result to check
        confidence_threshold: Minimum confidence threshold

    Returns:
        True if refinement should be triggered
    """
    return (
        not result.scope.is_complete
        or result.scope.confidence < confidence_threshold
        or len(result.scope.missing_context) > 0
    )
