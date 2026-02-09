"""Cache-aware coordination for minimizing VCM page faults.

This module provides working set management and dynamic page graph updates
to maximize temporal locality and minimize expensive page loads.
"""
from __future__ import annotations

import asyncio
import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Callable
from overrides import override

import networkx as nx

from .base import Agent
from .models import RunContext
from .patterns.attention import (
    PageQueryRoutingPolicy,
    create_page_query_router1,
    create_page_query_router2
)


logger = logging.getLogger(__name__)


class PageScorer(ABC):
    """Abstract base for page scoring strategies."""

    @abstractmethod
    async def score_page(self, page_id: str) -> float:
        """Score candidate page for inclusion in the working set.

        Args:
            page_id: ID of the page to score

        Returns:
            Relevance score for the page
        """
        pass


class SimplePageScorer(PageScorer):
    """Simple page scorer based on query history frequency."""
    def __init__(self, query_history: list[dict]):
        """Initialize with query history.
        Args:
            query_history: Recent query routing results
        """
        self.query_history = query_history
        # Prioritize by query history (pages that were queried recently)
        self.query_counts: dict[str, int] = {}
        for query_item in query_history[-100:]:  # Last 100 queries  # TODO: Make this configurable
            for page_id in query_item.get("relevant_pages", []):
                self.query_counts[page_id] = self.query_counts.get(page_id, 0) + 1

    @override
    async def score_page(self, page_id: str) -> float:
        """Score page based on query frequency."""
        return self.query_counts.get(page_id, 0)


class EdgePageScorer(PageScorer):
    """Simple page scorer based on edge weights in the page graph."""
    def __init__(self, page_graph: nx.DiGraph, completed_pages: set[str]):
        """Initialize with page graph.
        Args:
            page_graph: Graph of page relationships
        """
        self.page_graph = page_graph
        self.completed_pages = completed_pages

    @override
    async def score_page(self, page_id: str) -> float:
        """Score page based on query frequency."""

        return sum(
            self.page_graph[comp][page_id].get("weight", 1.0)
            for comp in self.completed_pages
            if self.page_graph.has_edge(comp, page_id)
        )


class CompositePageScorer(PageScorer):
    """Composite page scorer combining multiple strategies."""
    def __init__(self, scorers: list[PageScorer], weights: list[float]):
        """Initialize with multiple scorers and their weights.
        Args:
            scorers: List of PageScorer instances
            weights: Corresponding weights for each scorer
        """
        self.scorers = scorers
        self.weights = weights

    @override
    async def score_page(self, page_id: str) -> float:
        """Score page by combining scores from all scorers."""
        total_score = 0.0
        for scorer, weight in zip(self.scorers, self.weights):
            score = await scorer.score_page(page_id)
            total_score += score * weight
        return total_score


class CacheAwareCoordinationPolicy(ABC):
    """Abstract base for cache-aware coordination strategies.

    Broader than PageQueryRoutingPolicy - manages entire working set lifecycle:
    - Initial working set selection
    - Query routing (delegates to PageQueryRoutingPolicy)
    - Working set updates (eviction/loading)
    - Page graph learning from query results
    """

    @abstractmethod
    async def select_initial_working_set(
        self,
        page_graph: nx.DiGraph,
        available_pages: list[str],
        working_set_size: int,
        run_context: RunContext
    ) -> list[str]:
        """Select initial working set of pages to load.

        Args:
            page_graph: Full page graph
            available_pages: All available page IDs
            working_set_size: Maximum pages in working set (job quota)
            run_context: Analysis goal and context

        Returns:
            List of page IDs for initial working set
        """
        pass

    @abstractmethod
    async def update_working_set(
        self,
        current_working_set: set[str],
        completed_pages: set[str],
        page_graph: nx.DiGraph,
        working_set_size: int,
        page_scoring_function: PageScorer,
    ) -> set[str]:
        """Update working set based on completed pages and query patterns.

        Args:
            current_working_set: Currently loaded pages
            completed_pages: Pages that finished analysis
            page_graph: Current page graph
            working_set_size: Maximum pages in working set
            page_scoring_function: Function to score candidate pages

        Returns:
            New working set (may evict and load pages)
        """
        pass

    @abstractmethod
    async def update_page_graph(
        self,
        page_graph: nx.DiGraph,
        query_source: str,
        query_target: str,
        resolution_success: bool,
        metadata: dict[str, Any]
    ) -> None:
        """Update page graph with discovered relationships.

        Args:
            page_graph: Page graph to update
            query_source: Page that generated query
            query_target: Page that answered query
            resolution_success: Whether query found relevant content
            metadata: Additional info (relevance score, query text, etc.)
        """
        # TODO: Where is this method called and where will these updates be persisted?
        pass


class PageGraphCoordinationPolicy(CacheAwareCoordinationPolicy):
    """Coordination policy using page graph for working set management.

    Integrates PageQueryRoutingPolicy for query routing while managing the
    broader working set lifecycle (initial selection, updates, graph learning).
    """

    def __init__(
        self,
        query_routing_policy: PageQueryRoutingPolicy,
        max_hops_initial: int = 2,
        prefer_high_degree: bool = True
    ):
        """Initialize page graph coordination policy.

        Args:
            query_routing_policy: Policy for routing queries (delegates to existing abstraction)
            max_hops_initial: Max hops for initial working set selection
            prefer_high_degree: Prefer high-degree nodes for initial working set
        """
        self.query_routing_policy = query_routing_policy # TODO: Not used yet
        self.max_hops_initial = max_hops_initial
        self.prefer_high_degree = prefer_high_degree

    @override
    async def select_initial_working_set(
        self,
        page_graph: nx.DiGraph,
        available_pages: list[str],
        working_set_size: int,
        run_context: RunContext
    ) -> list[str]:
        """Select initial working set using graph centrality.

        Strategy:
        1. Find high-degree nodes (likely to be important)
        2. Expand via BFS to get connected components
        3. Take top-k by PageRank or degree centrality
        """
        if not page_graph.nodes():
            # No graph, just take first N pages
            return available_pages[:working_set_size]

        # Compute centrality metrics
        if self.prefer_high_degree:
            centrality = dict(page_graph.degree())
        else:
            # Use PageRank for more sophisticated measure
            centrality = nx.pagerank(page_graph)

        # Sort by centrality
        sorted_pages = sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top-k
        initial_pages = [page_id for page_id, _ in sorted_pages[:working_set_size]]

        logger.info(
            f"Selected initial working set: {len(initial_pages)} pages "
            f"(strategy: {'degree' if self.prefer_high_degree else 'pagerank'})"
        )

        return initial_pages

    @override
    async def update_working_set(
        self,
        current_working_set: set[str],
        completed_pages: set[str],
        page_graph: nx.DiGraph,
        working_set_size: int,
        page_scoring_function: PageScorer,
    ) -> set[str]:
        """Update working set by expanding from completed pages.

        Strategy:
        1. Keep pages that haven't been analyzed yet
        2. Remove completed pages
        3. Add neighbors of completed pages (BFS expansion)
        4. Prioritize by page score
        """
        # Start with current working set
        new_working_set = current_working_set - completed_pages

        # Find candidates from neighbors of completed pages
        candidates = set()
        for page_id in completed_pages:
            if page_id in page_graph:
                # Get neighbors (both in and out edges)
                neighbors = set(page_graph.successors(page_id)) | set(page_graph.predecessors(page_id))
                candidates.update(neighbors)

        # Remove pages already in working set or completed
        candidates = candidates - new_working_set - completed_pages

        # Sort candidates by query count + edge weight
        scored_candidates = []
        for page_id in candidates:
            # Also consider edge weights from completed pages
            page_score = page_scoring_function.score_page(page_id)
            scored_candidates.append((page_id, page_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Add top candidates until working set is full
        for page_id, _ in scored_candidates:
            if len(new_working_set) >= working_set_size:
                break
            new_working_set.add(page_id)

        return new_working_set

    @override
    async def update_page_graph(
        self,
        page_graph: nx.DiGraph,
        query_source: str,
        query_target: str,
        resolution_success: bool,
        metadata: dict[str, Any]
    ) -> None:
        """Update page graph with discovered relationship.

        Updates edge weight using exponential moving average.
        """
        # TODO: Where is this method called and where will these updates be persisted?
        relevance_score = metadata.get("relevance_score", 1.0)

        if not resolution_success or relevance_score < 0.3:
            return

        if page_graph.has_edge(query_source, query_target):
            # Update existing edge
            edge_data = page_graph[query_source][query_target]
            edge_data["weight"] = (
                edge_data.get("weight", 0) * 0.9 +
                relevance_score * 0.1
            )
            edge_data["query_count"] = edge_data.get("query_count", 0) + 1
        else:
            # Add new discovered edge
            page_graph.add_edge(
                query_source,
                query_target,
                weight=relevance_score,
                relationship_types=["discovered_dependency"],
                query_count=1,
                discovered_at=time.time()
            )



async def create_default_cache_aware_coordination_policy(
    agent: Agent,
    top_k_clusters: int = 5,
    top_n_pages_per_cluster: int = 3,
    top_n_pages_overall: int = 10,
    prefer_high_degree: bool = True,
) -> CacheAwareCoordinationPolicy:
    """
     Max pages in working set is determined by job quota
    """
    # Create query routing policy
    query_router: PageQueryRoutingPolicy = await create_page_query_router1(
        agent=agent,
        top_k_clusters=top_k_clusters,
        top_n_pages_per_cluster=top_n_pages_per_cluster,
        top_n_pages_overall=top_n_pages_overall,
    )

    # Create coordination policy
    return PageGraphCoordinationPolicy(
        query_routing_policy=query_router, # TODO: Not used yet.
        prefer_high_degree=prefer_high_degree
    )

