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

import networkx as nx


# =============================================================================
# Prefetch Policies
# =============================================================================

class PrefetchPolicy(ABC):
    """Abstract base for page prefetching strategies.

    Predicts which pages will be needed soon and preloads them
    to maximize cache hit rate.
    """

    @abstractmethod
    async def predict_needed_pages(
        self,
        current_context: dict[str, Any],
        page_graph: nx.DiGraph,
    ) -> list[str]:
        """Predict pages that will be needed.

        Args:
            current_context: Current execution context
            page_graph: Page relationship graph

        Returns:
            List of page IDs to prefetch
        """
        pass


class GraphPrefetchPolicy(PrefetchPolicy):
    """Prefetch graph neighbors of active pages.

    Assumes pages related in the graph are likely to be
    needed together.

    Use when:
    - Page graph captures true access patterns
    - Analysis follows graph structure
    """

    def __init__(
        self,
        hop_distance: int = 1,
        max_prefetch: int = 10,
        min_edge_weight: float = 0.3,
    ):
        """Initialize graph prefetch policy.

        Args:
            hop_distance: How many hops to look ahead
            max_prefetch: Maximum pages to prefetch
            min_edge_weight: Minimum edge weight to follow
        """
        self.hop_distance = hop_distance
        self.max_prefetch = max_prefetch
        self.min_edge_weight = min_edge_weight

    async def predict_needed_pages(
        self,
        current_context: dict[str, Any],
        page_graph: nx.DiGraph,
    ) -> list[str]:
        """Prefetch neighbors of active pages."""
        active_pages = current_context.get("active_pages", [])
        working_set = set(current_context.get("working_set", []))

        candidates = []
        for page in active_pages:
            if page not in page_graph:
                continue

            # Get neighbors up to hop_distance
            visited = {page}
            frontier = [(page, 0)]

            while frontier:
                current, depth = frontier.pop(0)
                if depth >= self.hop_distance:
                    continue

                for neighbor in list(page_graph.successors(current)) + list(page_graph.predecessors(current)):
                    if neighbor in visited:
                        continue

                    # Check edge weight
                    edge_data = page_graph.get_edge_data(current, neighbor) or {}
                    weight = edge_data.get("weight", 1.0)
                    if weight < self.min_edge_weight:
                        continue

                    visited.add(neighbor)

                    # Only add if not already in working set
                    if neighbor not in working_set:
                        candidates.append((neighbor, weight, depth + 1))

                    frontier.append((neighbor, depth + 1))

        # Sort by weight (descending) and depth (ascending)
        candidates.sort(key=lambda x: (-x[1], x[2]))

        return [c[0] for c in candidates[:self.max_prefetch]]


class QueryPrefetchPolicy(PrefetchPolicy):
    """Prefetch based on query patterns.

    Analyzes recent queries to predict which pages will be needed.

    Use when:
    - Query patterns are predictable
    - Historical query data is available
    """

    def __init__(
        self,
        max_prefetch: int = 10,
        lookback_queries: int = 10,
    ):
        """Initialize query prefetch policy.

        Args:
            max_prefetch: Maximum pages to prefetch
            lookback_queries: Number of recent queries to analyze
        """
        self.max_prefetch = max_prefetch
        self.lookback_queries = lookback_queries

    async def predict_needed_pages(
        self,
        current_context: dict[str, Any],
        page_graph: nx.DiGraph,
    ) -> list[str]:
        """Prefetch based on query history patterns."""
        working_set = set(current_context.get("working_set", []))
        query_history = current_context.get("query_history", [])[-self.lookback_queries:]

        # Count page frequencies in recent query results
        page_counts: dict[str, int] = {}
        for query_item in query_history:
            for page_id in query_item.get("relevant_pages", []):
                if page_id not in working_set:
                    page_counts[page_id] = page_counts.get(page_id, 0) + 1

        # Sort by frequency
        sorted_pages = sorted(
            page_counts.keys(),
            key=lambda p: page_counts[p],
            reverse=True
        )

        return sorted_pages[:self.max_prefetch]


class FeedbackPrefetchPolicy(PrefetchPolicy):
    """Prefetch for anticipated feedback loops.

    Predicts pages needed for critique, grounding, hypothesis games
    based on current analysis.

    Use when:
    - Running multi-agent critique/grounding flows
    - Need to preload evidence pages
    """

    def __init__(
        self,
        max_prefetch: int = 10,
    ):
        """Initialize feedback prefetch policy.

        Args:
            max_prefetch: Maximum pages to prefetch
        """
        self.max_prefetch = max_prefetch

    async def predict_needed_pages(
        self,
        current_context: dict[str, Any],
        page_graph: nx.DiGraph,
    ) -> list[str]:
        """Prefetch pages needed for feedback loops."""
        working_set = set(current_context.get("working_set", []))
        current_analysis = current_context.get("current_analysis", {})

        candidates = []

        # Look for pages mentioned in claims/hypotheses
        claims = current_analysis.get("claims", [])
        for claim in claims:
            evidence_pages = claim.get("evidence_pages", [])
            for page_id in evidence_pages:
                if page_id not in working_set:
                    candidates.append(page_id)

        # Look for pages in dependencies
        dependencies = current_analysis.get("dependencies", [])
        for dep in dependencies:
            target_page = dep.get("target_page")
            if target_page and target_page not in working_set:
                candidates.append(target_page)

        # Deduplicate and limit
        seen = set()
        result = []
        for page_id in candidates:
            if page_id not in seen:
                seen.add(page_id)
                result.append(page_id)
                if len(result) >= self.max_prefetch:
                    break

        return result


class CompositePrefetchPolicy(PrefetchPolicy):
    """Composite prefetch combining multiple strategies.

    Combines predictions from multiple policies with weighting.
    """

    def __init__(
        self,
        policies: list[PrefetchPolicy],
        max_prefetch: int = 10,
    ):
        """Initialize composite prefetch policy.

        Args:
            policies: List of prefetch policies to combine
            max_prefetch: Maximum total pages to prefetch
        """
        self.policies = policies
        self.max_prefetch = max_prefetch

    async def predict_needed_pages(
        self,
        current_context: dict[str, Any],
        page_graph: nx.DiGraph,
    ) -> list[str]:
        """Combine predictions from all policies."""
        all_candidates = []

        for policy in self.policies:
            candidates = await policy.predict_needed_pages(
                current_context, page_graph
            )
            all_candidates.extend(candidates)

        # Count occurrences (pages predicted by multiple policies rank higher)
        page_counts: dict[str, int] = {}
        for page_id in all_candidates:
            page_counts[page_id] = page_counts.get(page_id, 0) + 1

        # Sort by count
        sorted_pages = sorted(
            page_counts.keys(),
            key=lambda p: page_counts[p],
            reverse=True
        )

        return sorted_pages[:self.max_prefetch]
