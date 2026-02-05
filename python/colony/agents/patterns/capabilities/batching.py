"""Policy hierarchies for multi-agent execution framework.

Provides pluggable policies for:
- Batching: How to group work items for processing
- Prefetching: How to predict and preload pages
- Coordination: How to assign work to agents

These policies are used by capabilities to implement various
distributed multi-agent execution strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# =============================================================================
# Batching Policies
# =============================================================================

class BatchingPolicy(ABC):
    """Abstract base for work batching strategies.

    Determines how to group work items (pages, tasks, etc.) for processing.
    Different strategies trade off between:
    - Cache locality (clustering by page relationships)
    - Throughput (continuous batching as slots free up)
    - Latency (smaller batches, faster feedback)

    LLM-controllable options:
    - locality_weight: Trade-off between relevance and cache locality
    """

    @abstractmethod
    async def create_batch(
        self,
        candidates: list[str],
        working_set: set[str],
        context: dict[str, Any],
        locality_weight: float = 0.5,
    ) -> list[str]:
        """Select items for the next batch.

        LLM-controllable cache-awareness:
        - locality_weight: Controls relevance vs cache locality trade-off

        Args:
            candidates: Available items to batch
            working_set: Currently loaded pages
            context: Additional context (active_count, page_graph, etc.)
            locality_weight: Trade-off between relevance and cache locality
                (LLM decides based on current strategy):
                - 0.0: Pure relevance (ignore cache completely)
                - 0.5: Balanced (default - consider both equally)
                - 1.0: Pure locality (maximize cache hits)

        Returns:
            List of items for the batch
        """
        pass

    @abstractmethod
    async def should_execute_batch(
        self,
        current_batch: list[str],
        completed: set[str],
        context: dict[str, Any],
    ) -> bool:
        """Decide if batch should execute now.

        Args:
            current_batch: Items in current batch
            completed: Items that have completed
            context: Additional context

        Returns:
            True if batch should execute
        """
        pass


class ClusteringBatchPolicy(BatchingPolicy):
    """Batch by clustering with working set overlap.

    Groups items that share pages with the current working set,
    maximizing KV cache reuse.

    Use when:
    - You have clear page clusters
    - Cache locality is critical
    - You can wait for full batches
    """

    def __init__(
        self,
        min_overlap: float = 0.3,
        max_batch_size: int = 5,
        min_batch_size: int = 1,
    ):
        """Initialize clustering batch policy.

        Args:
            min_overlap: Minimum overlap ratio with working set
            max_batch_size: Maximum items per batch
            min_batch_size: Minimum items to form a batch
        """
        self.min_overlap = min_overlap
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

    async def create_batch(
        self,
        candidates: list[str],
        working_set: set[str],
        context: dict[str, Any],
        locality_weight: float = 0.5,
    ) -> list[str]:
        """Select candidates with high working set overlap.

        Args:
            candidates: Available items to batch
            working_set: Currently loaded pages
            context: Additional context (candidate_pages, relevance_scores)
            locality_weight: Trade-off between relevance (0.0) and cache locality (1.0)
        """
        # Get page associations and relevance scores for candidates
        candidate_pages = context.get("candidate_pages", {})
        relevance_scores = context.get("relevance_scores", {})  # Optional relevance scores

        # Score by combined relevance and overlap
        scored = []
        for cand in candidates:
            pages = set(candidate_pages.get(cand, []))

            # Locality score: overlap with working set
            if pages:
                locality_score = len(pages & working_set) / len(pages)
            else:
                locality_score = 0.0

            # Relevance score: provided or default to 0.5
            relevance_score = relevance_scores.get(cand, 0.5)

            # Combined score using LLM-controlled locality_weight
            # combined = (1 - locality_weight) * relevance + locality_weight * locality
            combined_score = (
                (1.0 - locality_weight) * relevance_score +
                locality_weight * locality_score
            )

            # Apply minimum overlap filter (adjusted by locality_weight)
            effective_min_overlap = self.min_overlap * locality_weight
            if locality_score >= effective_min_overlap or locality_weight < 0.1:
                scored.append((cand, combined_score, locality_score))

        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top candidates up to max_batch_size
        return [c for c, _, _ in scored[:self.max_batch_size]]

    async def should_execute_batch(
        self,
        current_batch: list[str],
        completed: set[str],
        context: dict[str, Any],
    ) -> bool:
        """Execute when batch reaches minimum size."""
        return len(current_batch) >= self.min_batch_size


class ContinuousBatchPolicy(BatchingPolicy):
    """Continuous batching - add work as slots free up.

    Assigns work immediately when capacity is available,
    maximizing throughput.

    Use when:
    - You want maximum throughput
    - Work items are independent
    - Cache locality is less critical
    """

    def __init__(self, max_concurrent: int = 10):
        """Initialize continuous batch policy.

        Args:
            max_concurrent: Maximum concurrent work items
        """
        self.max_concurrent = max_concurrent

    async def create_batch(
        self,
        candidates: list[str],
        working_set: set[str],
        context: dict[str, Any],
        locality_weight: float = 0.5,
    ) -> list[str]:
        """Fill available capacity with candidates.

        When locality_weight > 0, prioritizes candidates in working set.

        Args:
            candidates: Available items to batch
            working_set: Currently loaded pages
            context: Additional context (active_count, candidate_pages)
            locality_weight: If > 0, sort cached candidates first
        """
        active = context.get("active_count", 0)
        available = self.max_concurrent - active

        if available <= 0:
            return []

        # If locality_weight is significant, prioritize cached candidates
        if locality_weight > 0.1:
            candidate_pages = context.get("candidate_pages", {})

            # Score candidates by cache overlap
            scored = []
            for cand in candidates:
                pages = set(candidate_pages.get(cand, []))
                if pages:
                    cache_score = len(pages & working_set) / len(pages)
                else:
                    cache_score = 0.0
                scored.append((cand, cache_score))

            # Sort by cache score, weighted by locality_weight
            # Higher locality_weight = stronger preference for cached
            scored.sort(key=lambda x: x[1], reverse=True)
            sorted_candidates = [c for c, _ in scored]
            return sorted_candidates[:available]

        # Pure throughput mode - take in order
        return candidates[:available]

    async def should_execute_batch(
        self,
        current_batch: list[str],
        completed: set[str],
        context: dict[str, Any],
    ) -> bool:
        """Always execute if we have work and capacity."""
        return len(current_batch) > 0


class HybridBatchPolicy(BatchingPolicy):
    """Hybrid batching combining clustering and continuous strategies.

    Uses clustering when cache overlap is high, falls back to
    continuous batching when overlap is low.

    Use when:
    - You want best of both worlds
    - Work patterns vary (sometimes clustered, sometimes scattered)
    """

    def __init__(
        self,
        clustering_policy: ClusteringBatchPolicy | None = None,
        continuous_policy: ContinuousBatchPolicy | None = None,
        overlap_threshold: float = 0.5,
    ):
        """Initialize hybrid batch policy.

        Args:
            clustering_policy: Policy for clustered batching
            continuous_policy: Policy for continuous batching
            overlap_threshold: Threshold to switch between strategies
        """
        self.clustering_policy = clustering_policy or ClusteringBatchPolicy()
        self.continuous_policy = continuous_policy or ContinuousBatchPolicy()
        self.overlap_threshold = overlap_threshold

    async def create_batch(
        self,
        candidates: list[str],
        working_set: set[str],
        context: dict[str, Any],
        locality_weight: float = 0.5,
    ) -> list[str]:
        """Use clustering if high overlap, else continuous.

        locality_weight affects both strategies:
        - Higher: Favor clustering, require higher overlap
        - Lower: Fall back to continuous more readily

        Args:
            candidates: Available items to batch
            working_set: Currently loaded pages
            context: Additional context
            locality_weight: Trade-off between relevance and cache locality
        """
        # Adjust overlap threshold based on locality_weight
        # Higher locality_weight = higher threshold (more strict clustering)
        effective_threshold = self.overlap_threshold * locality_weight

        # Try clustering first
        cluster_batch = await self.clustering_policy.create_batch(
            candidates, working_set, context, locality_weight
        )

        if cluster_batch:
            # Check if we got good overlap
            candidate_pages = context.get("candidate_pages", {})
            total_overlap = 0.0
            for cand in cluster_batch:
                pages = set(candidate_pages.get(cand, []))
                if pages:
                    total_overlap += len(pages & working_set) / len(pages)

            avg_overlap = total_overlap / len(cluster_batch) if cluster_batch else 0
            if avg_overlap >= effective_threshold:
                return cluster_batch

        # Fall back to continuous (pass through locality_weight)
        return await self.continuous_policy.create_batch(
            candidates, working_set, context, locality_weight
        )

    async def should_execute_batch(
        self,
        current_batch: list[str],
        completed: set[str],
        context: dict[str, Any],
    ) -> bool:
        """Execute if either strategy would execute."""
        return (
            await self.clustering_policy.should_execute_batch(
                current_batch, completed, context
            ) or
            await self.continuous_policy.should_execute_batch(
                current_batch, completed, context
            )
        )

