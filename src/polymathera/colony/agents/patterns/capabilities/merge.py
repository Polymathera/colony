"""Merge policies for combining analysis results from multiple sources.

This module implements hierarchical merge strategies for different data types:
- SemanticMergePolicy: LLM-based merging for semantic content
- StatisticalMergePolicy: Statistical aggregation for numerical metrics
- GraphMergePolicy: Graph union/merge for dependency graphs
- RelationshipMergePolicy: Relationship deduplication and consolidation

Each merge policy handles scope merging and validates merged results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import statistics
from typing import Any, Generic, TypeVar
from overrides import override
from pydantic import BaseModel, Field

from ..scope import AnalysisScope, ScopeAwareResult, merge_scopes
from ...base import Agent, AgentCapability
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...models import AgentSuspensionState
from ....utils import setup_logger
from ..actions import action_executor

logger = setup_logger(__name__)


T = TypeVar('T')


class MergeContext(BaseModel):
    """Context for merge operations.

    Provides information to guide merge decisions:
    - Strategy hints (prefer newer vs higher confidence)
    - Constraints (max size, max time)
    - Metadata (merge reason, requesting agent)
    """

    # Merge strategy hints
    prefer_higher_confidence: bool = Field(
        default=True,
        description="Prefer results with higher confidence when merging"
    )

    prefer_newer: bool = Field(
        default=False,
        description="Prefer more recent results"
    )

    prefer_complete: bool = Field(
        default=True,
        description="Prefer complete results over incomplete"
    )

    # Constraints
    max_merge_size: int | None = Field(
        default=None,
        description="Maximum size of merged result (implementation-specific)"
    )

    timeout_seconds: float | None = Field(
        default=None,
        description="Timeout for merge operation"
    )

    # Metadata
    merge_reason: str | None = Field(
        default=None,
        description="Why are we merging these results?"
    )

    requesting_agent_id: str | None = Field(
        default=None,
        description="Agent requesting the merge"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional merge context"
    )


class MergeValidationResult(BaseModel):
    """Result of validating a merged result.

    Checks if merge is valid and preserves important information.
    """

    is_valid: bool = Field(
        description="Whether merge is valid"
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Issues found in merge"
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about merge quality"
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in merge validity"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation metadata"
    )


class MergePolicy(ABC, Generic[T]):
    """Abstract base class for merge policies.

    Different merge strategies for different data types:
    - Semantic content → LLM-based merging
    - Numerical metrics → Statistical aggregation
    - Graphs → Graph union/intersection
    - Relationships → Deduplication

    All merge policies should:
    1. Merge content using domain-appropriate strategy
    2. Merge scopes conservatively (union of concerns)
    3. Validate merged result
    4. Handle merge failures gracefully
    """

    @abstractmethod
    async def merge(
        self,
        results: list[ScopeAwareResult[T]],
        context: MergeContext
    ) -> ScopeAwareResult[T]:
        """Merge multiple results into one.

        Args:
            results: List of results to merge
            context: Merge context with hints and constraints

        Returns:
            Merged result with combined content and scope

        Raises:
            MergeError: If merge fails
        """
        pass

    @abstractmethod
    async def validate(
        self,
        original: list[ScopeAwareResult[T]],
        merged: ScopeAwareResult[T]
    ) -> MergeValidationResult:
        """Validate merged result against originals.

        Args:
            original: Original results that were merged
            merged: Merged result to validate

        Returns:
            Validation result
        """
        pass

    def _merge_scopes(
        self,
        results: list[ScopeAwareResult[T]]
    ) -> AnalysisScope:
        """Merge scopes from multiple results.

        Args:
            results: Results whose scopes to merge

        Returns:
            Merged scope
        """
        return merge_scopes([r.scope for r in results])


class SemanticMergePolicy(MergePolicy[T]):
    """LLM-based semantic merging for text content.

    Uses LLM to merge semantic content while preserving meaning and
    eliminating redundancy. Falls back to simple concatenation if LLM fails.

    Use for:
    - Documentation summaries
    - Code explanations
    - Analysis findings (text)
    - Intent descriptions
    """

    def __init__(self, agent: Agent | None = None):
        """Initialize with an optional agent.

        Args:
            agent: Agent instance for semantic merging (if None, fallback only)
        """
        self.agent = agent

    async def merge(
        self,
        results: list[ScopeAwareResult[T]],
        context: MergeContext
    ) -> ScopeAwareResult[T]:
        """Merge results using LLM semantic understanding.

        Args:
            results: Results to merge
            context: Merge context

        Returns:
            Merged result
        """
        if not results:
            raise ValueError("Cannot merge empty list of results")

        if len(results) == 1:
            return results[0]

        # Try LLM merge first
        if self.agent is not None:
            try:
                merged_content = await self._llm_merge(results, context)
            except Exception as e:
                # Fall back to simple merge
                merged_content = await self._fallback_merge(results)
        else:
            merged_content = await self._fallback_merge(results)

        # Merge scopes
        merged_scope = self._merge_scopes(results)

        return ScopeAwareResult(
            content=merged_content,
            scope=merged_scope,
            result_type=results[0].result_type,
            refinement_count=max(r.refinement_count for r in results) + 1
        )

    async def _llm_merge(
        self,
        results: list[ScopeAwareResult[T]],
        context: MergeContext
    ) -> T:
        """Merge using LLM.

        Args:
            results: Results to merge
            context: Merge context

        Returns:
            Merged content
        """
        # Build merge prompt
        contents_str = "\n\n".join([
            f"Result {i+1} (confidence: {r.scope.confidence:.2f}):\n{r.content}"
            for i, r in enumerate(results)
        ])

        prompt = f"""Merge these analysis results into a unified summary:

{contents_str}

Instructions:
1. Preserve all unique insights and findings
2. Eliminate redundancy and duplication
3. Resolve any contradictions (explain how)
4. Maintain coherence and clarity
5. Highlight key findings

Provide the merged result in the same format as the originals.
"""

        # Get LLM response
        response = await self.agent.infer(
            context_page_ids=self.agent.bound_pages,  # TODO: How to select pages?
            prompt=prompt,
            temperature=0.1,  # TODO: Make configurable - low temp for refinement
            max_tokens=500,   # TODO: Make configurable - short responses for refinement
            json_schema=T.model_json_schema()  # Structured output
        )
        content = T.model_validate_json(response.generated_text)

        return content

    async def _fallback_merge(
        self,
        results: list[ScopeAwareResult[T]]
    ) -> T:
        """Simple fallback merge (concatenation or first result).

        Args:
            results: Results to merge

        Returns:
            Merged content (conservative)
        """
        # Simple fallback: return highest confidence result
        return max(results, key=lambda r: r.scope.confidence).content

    async def validate(
        self,
        original: list[ScopeAwareResult[T]],
        merged: ScopeAwareResult[T]
    ) -> MergeValidationResult:
        """Validate semantic merge preserves information.

        Args:
            original: Original results
            merged: Merged result

        Returns:
            Validation result
        """
        # Simple validation: check merged confidence
        original_confidences = [r.scope.confidence for r in original]
        avg_confidence = statistics.mean(original_confidences)

        issues = []
        if merged.scope.confidence < avg_confidence * 0.8:
            issues.append(f"Merged confidence ({merged.scope.confidence:.2f}) much lower than average ({avg_confidence:.2f})")

        return MergeValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=0.8 if len(issues) == 0 else 0.5
        )


class StatisticalMergePolicy(MergePolicy[dict[str, float]]):
    """Statistical aggregation for numerical metrics.

    Computes mean, median, std dev, min, max, percentiles for metric values.

    Use for:
    - Performance metrics
    - Code complexity metrics
    - Test coverage metrics
    - Security scores
    """

    async def merge(
        self,
        results: list[ScopeAwareResult[dict[str, float]]],
        context: MergeContext
    ) -> ScopeAwareResult[dict[str, float]]:
        """Merge metrics using statistical aggregation.

        Args:
            results: Results with metric dictionaries
            context: Merge context

        Returns:
            Merged result with aggregated statistics
        """
        if not results:
            raise ValueError("Cannot merge empty list of results")

        if len(results) == 1:
            return results[0]

        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.content.keys())

        # Aggregate each metric
        aggregated = {}
        for metric_name in all_metrics:
            values = [
                r.content[metric_name]
                for r in results
                if metric_name in r.content
            ]

            if not values:
                continue

            # Compute statistics
            aggregated[metric_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "values": values  # Keep original values for reference
            }

        # Merge scopes
        merged_scope = self._merge_scopes(results)

        return ScopeAwareResult(
            content=aggregated,
            scope=merged_scope,
            result_type="statistical_metrics",
            refinement_count=max(r.refinement_count for r in results) + 1
        )

    async def validate(
        self,
        original: list[ScopeAwareResult[dict[str, float]]],
        merged: ScopeAwareResult[dict[str, float]]
    ) -> MergeValidationResult:
        """Validate statistical merge.

        Args:
            original: Original results
            merged: Merged result

        Returns:
            Validation result
        """
        issues = []

        # Check all original metrics are included
        for result in original:
            for metric_name in result.content.keys():
                if metric_name not in merged.content:
                    issues.append(f"Metric {metric_name} missing in merged result")

        return MergeValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if len(issues) == 0 else 0.7
        )


class GraphMergePolicy(MergePolicy[Any]):
    """Graph union/merge policy for dependency graphs, call graphs, etc.

    Merges graphs by:
    1. Unioning nodes (deduplicate by ID)
    2. Unioning edges (preserve all relationships)
    3. Merging node attributes (take max confidence)
    4. Validating graph properties (no cycles if DAG, etc.)

    Use for:
    - Dependency graphs
    - Call graphs
    - Data flow graphs
    - Relationship graphs
    """

    def __init__(self, graph_type: str = "directed"):
        """Initialize graph merge policy.

        Args:
            graph_type: Type of graph ("directed", "undirected", "dag")
        """
        self.graph_type = graph_type

    async def merge(
        self,
        results: list[ScopeAwareResult[Any]],
        context: MergeContext
    ) -> ScopeAwareResult[Any]:
        """Merge graphs by union of nodes and edges.

        Args:
            results: Results containing graph data
            context: Merge context

        Returns:
            Merged graph result
        """
        if not results:
            raise ValueError("Cannot merge empty list of results")

        if len(results) == 1:
            return results[0]

        # Merge graphs (implementation depends on graph representation)
        # This is a simplified version - real implementation needs to handle
        # specific graph types (NetworkX, custom Graph class, etc.)

        merged_graph = self._merge_graphs([r.content for r in results])
        merged_scope = self._merge_scopes(results)

        return ScopeAwareResult(
            content=merged_graph,
            scope=merged_scope,
            result_type="graph",
            refinement_count=max(r.refinement_count for r in results) + 1
        )

    def _merge_graphs(self, graphs: list[Any]) -> Any:
        """Merge multiple graphs.

        Args:
            graphs: List of graphs to merge

        Returns:
            Merged graph

        Note: Implementation depends on graph representation.
        This is a placeholder that should be specialized for each graph type.
        """
        # Placeholder: return first graph
        # Real implementation would merge nodes and edges
        return graphs[0]

    async def validate(
        self,
        original: list[ScopeAwareResult[Any]],
        merged: ScopeAwareResult[Any]
    ) -> MergeValidationResult:
        """Validate graph merge.

        Args:
            original: Original graph results
            merged: Merged graph result

        Returns:
            Validation result
        """
        # Placeholder validation
        # Real implementation would check:
        # - All nodes from originals are in merged
        # - All edges from originals are in merged
        # - Graph properties maintained (DAG, etc.)

        return MergeValidationResult(
            is_valid=True,
            confidence=0.9
        )


class ListMergePolicy(MergePolicy[list]):
    """Simple list merge policy.

    Merges lists by concatenation with optional deduplication.

    Use for:
    - Lists of findings
    - Lists of issues
    - Lists of recommendations
    """

    def __init__(self, deduplicate: bool = True):
        """Initialize list merge policy.

        Args:
            deduplicate: Whether to remove duplicates
        """
        self.deduplicate = deduplicate

    async def merge(
        self,
        results: list[ScopeAwareResult[list]],
        context: MergeContext
    ) -> ScopeAwareResult[list]:
        """Merge lists by concatenation.

        Args:
            results: Results containing lists
            context: Merge context

        Returns:
            Merged list result
        """
        if not results:
            raise ValueError("Cannot merge empty list of results")

        if len(results) == 1:
            return results[0]

        # Concatenate all lists
        merged_list = []
        for result in results:
            merged_list.extend(result.content)

        # Deduplicate if requested
        if self.deduplicate:
            # For simple types, use set
            # For complex types, need custom deduplication
            try:
                merged_list = list(dict.fromkeys(merged_list))
            except TypeError:
                # Not hashable, keep duplicates
                pass

        merged_scope = self._merge_scopes(results)

        return ScopeAwareResult(
            content=merged_list,
            scope=merged_scope,
            result_type="list",
            refinement_count=max(r.refinement_count for r in results) + 1
        )

    async def validate(
        self,
        original: list[ScopeAwareResult[list]],
        merged: ScopeAwareResult[list]
    ) -> MergeValidationResult:
        """Validate list merge.

        Args:
            original: Original list results
            merged: Merged list result

        Returns:
            Validation result
        """
        # Check all items from originals are in merged
        total_items = sum(len(r.content) for r in original)
        merged_items = len(merged.content)

        issues = []
        if self.deduplicate:
            if merged_items > total_items:
                issues.append(f"Merged has more items ({merged_items}) than originals ({total_items})")
        else:
            if merged_items != total_items:
                issues.append(f"Merged has {merged_items} items but originals have {total_items}")

        return MergeValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if len(issues) == 0 else 0.8
        )


class DictMergePolicy(MergePolicy[dict]):
    """Dictionary merge policy with conflict resolution.

    Merges dictionaries using specified conflict resolution strategy.

    Use for:
    - Configuration merging
    - Metadata merging
    - Attribute merging
    """

    def __init__(
        self,
        conflict_strategy: str = "prefer_higher_confidence"
    ):
        """Initialize dict merge policy.

        Args:
            conflict_strategy: How to resolve key conflicts
                - "prefer_higher_confidence": Use value from higher confidence result
                - "prefer_newer": Use value from newer result
                - "merge_recursive": Recursively merge nested dicts
                - "keep_all": Keep all values as list
        """
        self.conflict_strategy = conflict_strategy

    async def merge(
        self,
        results: list[ScopeAwareResult[dict]],
        context: MergeContext
    ) -> ScopeAwareResult[dict]:
        """Merge dictionaries with conflict resolution.

        Args:
            results: Results containing dictionaries
            context: Merge context

        Returns:
            Merged dictionary result
        """
        if not results:
            raise ValueError("Cannot merge empty list of results")

        if len(results) == 1:
            return results[0]

        merged_dict = {}
        key_sources = {}  # Track which result each key came from

        for result in results:
            for key, value in result.content.items():
                if key not in merged_dict:
                    # New key, add it
                    merged_dict[key] = value
                    key_sources[key] = result
                else:
                    # Key conflict, resolve
                    existing_source = key_sources[key]

                    if self.conflict_strategy == "prefer_higher_confidence":
                        if result.scope.confidence > existing_source.scope.confidence:
                            merged_dict[key] = value
                            key_sources[key] = result
                    elif self.conflict_strategy == "prefer_newer":
                        if result.created_at > existing_source.created_at:
                            merged_dict[key] = value
                            key_sources[key] = result
                    elif self.conflict_strategy == "keep_all":
                        # Keep both values as list
                        if not isinstance(merged_dict[key], list):
                            merged_dict[key] = [merged_dict[key]]
                        merged_dict[key].append(value)
                    # merge_recursive would go here

        merged_scope = self._merge_scopes(results)

        return ScopeAwareResult(
            content=merged_dict,
            scope=merged_scope,
            result_type="dictionary",
            refinement_count=max(r.refinement_count for r in results) + 1
        )

    async def validate(
        self,
        original: list[ScopeAwareResult[dict]],
        merged: ScopeAwareResult[dict]
    ) -> MergeValidationResult:
        """Validate dictionary merge.

        Args:
            original: Original dict results
            merged: Merged dict result

        Returns:
            Validation result
        """
        # Collect all keys from originals
        all_keys = set()
        for result in original:
            all_keys.update(result.content.keys())

        # Check all keys present in merged
        merged_keys = set(merged.content.keys())
        missing_keys = all_keys - merged_keys

        issues = []
        if missing_keys:
            issues.append(f"Missing keys in merged result: {missing_keys}")

        return MergeValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if len(issues) == 0 else 0.7
        )


class HierarchicalMerger(Generic[T]):
    """Hierarchical merging with multiple strategies.

    Merges results in a tree structure:
    1. Group results by some criterion (type, page cluster, etc.)
    2. Merge each group using appropriate policy
    3. Recursively merge group results

    This scales better than flat merging for large numbers of results.
    """

    def __init__(
        self,
        strategy_selector: callable[[type], MergePolicy] | None = None
    ):
        """Initialize hierarchical merger.

        Args:
            strategy_selector: Function to select merge policy based on content type
        """
        self.strategy_selector = strategy_selector or self._default_strategy_selector

    def _default_strategy_selector(self, content_type: type) -> MergePolicy:
        """Default strategy selection.

        Args:
            content_type: Type of content

        Returns:
            Appropriate merge policy
        """
        if content_type == dict:
            return DictMergePolicy()
        elif content_type == list:
            return ListMergePolicy()
        else:
            # Fallback to semantic merge (needs LLM)
            return SemanticMergePolicy()

    async def merge_hierarchical(
        self,
        results: list[ScopeAwareResult[T]],
        group_by: callable[[ScopeAwareResult[T]], str],
        context: MergeContext
    ) -> ScopeAwareResult[T]:
        """Merge results hierarchically.

        Args:
            results: Results to merge
            group_by: Function to group results
            context: Merge context

        Returns:
            Merged result
        """
        if not results:
            raise ValueError("Cannot merge empty list of results")

        if len(results) == 1:
            return results[0]

        # Group results
        groups: dict[str, list[ScopeAwareResult[T]]] = {}
        for result in results:
            group_key = group_by(result)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(result)

        # Merge each group
        group_results = []
        for group_key, group_items in groups.items():
            # Select merge strategy based on content type
            content_type = type(group_items[0].content)
            strategy = self.strategy_selector(content_type)

            # Merge group
            merged_group = await strategy.merge(group_items, context)
            group_results.append(merged_group)

        # Recursively merge groups if needed
        if len(group_results) > 1:
            # Merge the group results
            content_type = type(group_results[0].content)
            strategy = self.strategy_selector(content_type)
            return await strategy.merge(group_results, context)
        else:
            return group_results[0]


class MergeCapability(AgentCapability):
    """Capability for merging analysis results from multiple sources.

    Wraps a MergePolicy and exposes @action_executor methods for:
    - merge_results: Merge multiple ScopeAwareResults
    - validate_merge: Validate a merged result

    Uses dynamic lookup via self.agent.get_capability_by_type() for
    coordination with other capabilities.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "merge",
        capability_key: str = "merge",
        merge_policy: MergePolicy | None = None,
    ):
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=None, capability_key=capability_key)
        self.merge_policy: MergePolicy | None = merge_policy

    def get_action_group_description(self) -> str:
        return (
            "Result Merging — combines multiple ScopeAwareResults via configurable merge policy. "
            "Requires set_policy() before use. Supports hierarchical merging with group_by, "
            "and automatic strategy selection based on content type (dict, list, semantic). "
            "Always validate_merge after merging to check for information loss."
        )

    def set_policy(self, policy: MergePolicy) -> None:
        """Configure the merge policy after instantiation.

        Args:
            policy: MergePolicy implementation to use
        """
        self.merge_policy = policy

    def _require_policy(self) -> MergePolicy:
        """Get policy, raising if not configured."""
        if self.merge_policy is None:
            raise RuntimeError(
                "MergeCapability.set_policy() must be called before use. "
                "Configure with a MergePolicy implementation."
            )
        return self.merge_policy

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for MMergeCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for MergeCapability")
        pass

    @action_executor()
    async def merge_results(
        self,
        results: list[ScopeAwareResult],
        context: MergeContext | None = None
    ) -> ScopeAwareResult:
        """Merge multiple results into one (plannable by LLM).

        Args:
            results: List of results to merge
            context: Optional merge context with hints and constraints

        Returns:
            Merged result with combined content and scope
        """
        policy = self._require_policy()
        if context is None:
            context = MergeContext()
        return await policy.merge(results, context)

    @action_executor()
    async def validate_merge(
        self,
        original: list[ScopeAwareResult],
        merged: ScopeAwareResult
    ) -> MergeValidationResult:
        """Validate merged result against originals.

        Args:
            original: Original results that were merged
            merged: Merged result to validate

        Returns:
            Validation result with issues and confidence
        """
        policy = self._require_policy()
        return await policy.validate(original, merged)


# Utility functions

async def merge_results(
    results: list[ScopeAwareResult[T]],
    policy: MergePolicy[T],
    context: MergeContext | None = None
) -> ScopeAwareResult[T]:
    """Convenience function to merge results with a policy.

    Args:
        results: Results to merge
        policy: Merge policy to use
        context: Optional merge context

    Returns:
        Merged result
    """
    if context is None:
        context = MergeContext()

    merged = await policy.merge(results, context)

    # Validate merge
    validation = await policy.validate(results, merged)
    if not validation.is_valid:
        # Log issues but don't fail
        # In production, might want to raise or retry
        pass

    return merged

