"""Unified query routing abstraction.

This module provides a clean, policy-based interface for routing queries to relevant pages.
It unifies the previously fragmented implementations:
- attention.score_attention() (direct, local)
- attention_policy.find_relevant_pages() (hierarchical, global)
- Future: cache-aware, graph-based routing

All query routing now goes through PageQueryRoutingPolicy with multiple implementations.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

from .attention import PageQuery, AttentionScore, AttentionScoringMechanism, PageKey
from .attention_policy import AttentionPolicy
from ...models import AttentionContext
from ...base import Agent

logger = logging.getLogger(__name__)


class PageQueryRoutingPolicy(ABC):
    """Abstract base for query routing strategies.

    Provides unified interface for routing queries to relevant pages.
    Different implementations offer different trade-offs in:
    - Scope (local, hierarchical, global)
    - Efficiency (number of comparisons, LLM calls)
    - Cache awareness (consider working set)
    - Graph awareness (use page relationships)
    """

    def __init__(self, agent: Agent):
        """Initialize query routing policy.

        Args:
            agent: Agent context for routing decisions
        """
        self.agent = agent

    @abstractmethod
    async def route_query(
        self,
        query: PageQuery,
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route query to relevant pages.

        Args:
            query: Query to route
            available_pages: Optional list of page IDs to consider (None = all pages in GlobalPageKeyRegistry)
            context: Optional context for routing decisions

        Returns:
            List of attention scores for relevant pages, sorted by relevance
        """
        pass


class DirectAttentionRouting(PageQueryRoutingPolicy):
    """Direct attention routing using provided keys.

    This is the simplest routing strategy - computes attention directly
    against a provided set of page keys. No hierarchy, no registry lookup.

    Used when:
    - You have a small, fixed set of keys to compare against
    - You want local, controlled routing (e.g., within a cluster)
    - You don't need global discovery
    """

    def __init__(
        self,
        agent: Agent,
        attention_mechanism: AttentionScoringMechanism,
        page_keys: dict[str, PageKey]
    ):
        """Initialize direct attention routing.

        Args:
            attention_mechanism: Attention mechanism for computing Q·K
            page_keys: Dict mapping page_id → PageKey
        """
        super().__init__(agent)
        self.attention_mechanism = attention_mechanism
        self.page_keys = page_keys

    async def route_query(
        self,
        query: PageQuery,
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route query using direct attention.

        Args:
            query: Query to route
            available_pages: If provided, only consider these pages
            context: Unused for direct routing

        Returns:
            List of attention scores for relevant pages
        """
        # Filter keys if available_pages specified
        if available_pages is not None:
            keys = [self.page_keys[page_id] for page_id in available_pages if page_id in self.page_keys]
        else:
            keys = list(self.page_keys.values())

        if not keys:
            logger.warning("No keys available for direct attention routing")
            return []

        # Compute attention
        relevant_pages = await self.attention_mechanism.score_attention(
            query=query,
            keys=keys
        )

        logger.debug(
            f"Direct attention routing: found {len(relevant_pages)} relevant pages "
            f"for query '{query.query_text[:50]}...'"
        )

        return relevant_pages


class HierarchicalAttentionRouting(PageQueryRoutingPolicy):
    """Hierarchical attention routing via attention policy.

    Wraps existing AttentionPolicy implementations (HierarchicalAttentionPolicy,
    GlobalAttentionPolicy, LocalAttentionPolicy) to provide unified interface.

    Used when:
    - You need global page discovery across clusters
    - You want efficient two-level search (cluster → page)
    - You need to respect scope constraints
    """

    def __init__(self, attention_policy: AttentionPolicy):
        """Initialize hierarchical routing.

        Args:
            attention_policy: Existing attention policy to wrap
        """
        super().__init__(attention_policy.agent)
        self.attention_policy = attention_policy

    async def route_query(
        self,
        query: PageQuery,
        available_pages: list[str] | None = None,  # Ignored (policy handles page discovery)
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route query using hierarchical attention policy.

        Args:
            `query`: Query to route
            `available_pages`: Ignored (policy handles page discovery)
            `context`: Passed to attention policy

        Returns:
            List of attention scores for relevant pages
        """
        context = context or AttentionContext()

        # Delegate to attention policy
        relevant_pages = await self.attention_policy.find_relevant_pages(
            query=query,
            context=context
        )

        logger.debug(
            f"Hierarchical routing: found {len(relevant_pages)} relevant pages "
            f"for query '{query.query_text[:50]}...'"
        )

        return relevant_pages


class CacheAwareRouting(PageQueryRoutingPolicy):
    """Cache-aware routing that prioritizes pages in working set.

    Considers which pages are currently loaded in VCM (working set) and
    boosts their relevance scores to minimize page faults.

    Used when:
    - You want to maximize temporal locality
    - Working set management is critical
    - Page load cost is high
    """

    def __init__(
        self,
        base_routing: PageQueryRoutingPolicy,
        working_set: set[str],  # TODO: Get the working set dynamically from the VCM
        cache_boost_factor: float = 1.5
    ):
        """Initialize cache-aware routing.

        Args:
            base_routing: Base routing policy to augment
            working_set: Set of currently loaded page IDs
            cache_boost_factor: Multiplier for pages in working set (>1.0)
        """
        super().__init__(base_routing.agent)
        self.base_routing = base_routing
        self.working_set = working_set
        self.cache_boost_factor = cache_boost_factor

    async def route_query(
        self,
        query: PageQuery,
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route query with cache awareness.

        Args:
            query: Query to route
            available_pages: Optional page filter
            context: Passed to base routing

        Returns:
            List of attention scores, with working set pages boosted
        """
        # Get base routing results
        relevant_pages = await self.base_routing.route_query(
            query=query,
            available_pages=available_pages,
            context=context
        )

        # Boost scores for pages in working set
        boosted_pages = []
        for score in relevant_pages:
            if score.page_id in self.working_set:
                # Boost score for cached pages
                boosted_score = AttentionScore(
                    page_id=score.page_id,
                    score=score.score * self.cache_boost_factor,
                    explanation=f"[CACHED] {score.explanation}"
                )
                boosted_pages.append(boosted_score)
            else:
                boosted_pages.append(score)

        # Re-sort by boosted scores
        boosted_pages.sort(key=lambda x: x.score, reverse=True)

        cached_count = sum(1 for score in relevant_pages if score.page_id in self.working_set)
        logger.debug(
            f"Cache-aware routing: {cached_count}/{len(relevant_pages)} pages in working set, "
            f"boosted by {self.cache_boost_factor}x"
        )

        return boosted_pages


class GraphBasedRouting(PageQueryRoutingPolicy):
    """Graph-based routing using page relationships.

    Uses page graph structure to route queries based on:
    - Direct relationships (imports, dependencies)
    - Graph distance (BFS with weighted edges)
    - Relationship types (import, call, reference)

    Used when:
    - You have a page graph with relationships
    - Structural relationships predict relevance
    - You want to explore related pages
    """

    def __init__(
        self,
        agent: Agent,
        page_graph: nx.DiGraph,  # TODO: Get the graph from the PageStorage
        max_hops: int = 2,
        edge_weight_threshold: float = 0.5
    ):
        """Initialize graph-based routing.

        Args:
            page_graph: NetworkX DiGraph of page relationships
            max_hops: Maximum graph distance to explore
            edge_weight_threshold: Minimum edge weight to follow
        """
        super().__init__(agent)
        self.page_graph = page_graph
        self.max_hops = max_hops
        self.edge_weight_threshold = edge_weight_threshold

    async def route_query(
        self,
        query: PageQuery,  # TODO: Use query content to guide graph traversal?
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route query using graph structure.

        Args:
            query: Query to route
            available_pages: Optional page filter
            context: Must contain 'source_page_id' to start BFS

        Returns:
            List of attention scores based on graph distance and weights
        """
        context = context or AttentionContext()
        source_page_id = context.source_page_id

        if not source_page_id:
            logger.warning("Graph-based routing requires source_page_id in context")
            return []

        if source_page_id not in self.page_graph:
            logger.warning(f"Source page {source_page_id} not in graph")
            return []

        # BFS from source page
        candidates = []
        visited = {source_page_id}
        queue = [(source_page_id, 0, 1.0)]  # (page_id, hops, weight)

        while queue:
            current_page, hops, weight = queue.pop(0)

            if hops > self.max_hops:
                continue

            # Get neighbors
            for neighbor in self.page_graph.neighbors(current_page):
                if neighbor in visited:
                    continue

                # Filter by available_pages if specified
                if available_pages is not None and neighbor not in available_pages:
                    continue

                visited.add(neighbor)

                # Get edge weight
                edge_data = self.page_graph[current_page][neighbor]
                edge_weight = edge_data.get("weight", 1.0)

                if edge_weight < self.edge_weight_threshold:
                    continue

                # Propagate weight (decay with distance)
                neighbor_weight = weight * edge_weight * (0.8 ** hops)

                candidates.append(
                    AttentionScore(
                        page_id=neighbor,
                        score=neighbor_weight,
                        explanation=f"Graph distance: {hops+1} hops, edge weight: {edge_weight:.2f}"
                    )
                )

                queue.append((neighbor, hops + 1, neighbor_weight))

        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            f"Graph-based routing: found {len(candidates)} pages within {self.max_hops} hops "
            f"from {source_page_id}"
        )

        return candidates


class HybridQueryRouting(PageQueryRoutingPolicy):
    """Hybrid routing combining multiple strategies.

    Combines results from multiple routing policies with configurable weights.
    For example:
    - 60% attention-based (semantic similarity)
    - 40% graph-based (structural relationships)

    Used when:
    - You want to balance multiple relevance signals
    - Different strategies capture different aspects
    - You need robust routing
    """

    def __init__(
        self,
        strategies: list[tuple[PageQueryRoutingPolicy, float]]
    ):
        """Initialize hybrid routing.

        Args:
            strategies: List of (policy, weight) tuples
        """
        super().__init__(strategies[0][0].agent if strategies else None)
        self.strategies = strategies

        # Normalize weights
        total_weight = sum(weight for _, weight in strategies)
        self.strategies = [(policy, weight / total_weight) for policy, weight in strategies]

    async def route_query(
        self,
        query: PageQuery,
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route query using hybrid approach.

        Args:
            query: Query to route
            available_pages: Optional page filter
            context: Passed to all strategies

        Returns:
            List of attention scores, combined from all strategies
        """
        # Collect results from all strategies
        all_results = []
        for policy, weight in self.strategies:
            results = await policy.route_query(
                query=query,
                available_pages=available_pages,
                context=context
            )
            all_results.append((results, weight))

        # Combine scores
        combined_scores: dict[str, float] = {}
        for results, weight in all_results:
            for score in results:
                combined_scores[score.page_id] = combined_scores.get(score.page_id, 0.0) + (score.score * weight)

        # Convert back to AttentionScore list
        final_scores = [
            AttentionScore(page_id=page_id, score=score)
            for page_id, score in combined_scores.items()
        ]

        # Sort by combined score
        final_scores.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            f"Hybrid routing: combined {len(self.strategies)} strategies, "
            f"found {len(final_scores)} pages"
        )

        return final_scores


class BatchedQueryRouting(PageQueryRoutingPolicy):
    """Batched routing for multiple queries.

    Batches multiple queries together to improve efficiency when routing
    many queries at once. Useful for reducing overhead and parallelizing.

    Used when:
    - You have multiple queries to route at once
    - Batching provides efficiency gains
    - You want to amortize setup costs
    """

    def __init__(self, base_routing: PageQueryRoutingPolicy):
        """Initialize batched routing.

        Args:
            base_routing: Base routing policy to batch
        """
        super().__init__(base_routing.agent)
        self.base_routing = base_routing

    async def route_query(
        self,
        query: PageQuery,
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> list[AttentionScore]:
        """Route single query (delegates to base).

        For batching, use route_queries_batch() instead.
        """
        return await self.base_routing.route_query(
            query=query,
            available_pages=available_pages,
            context=context
        )

    async def route_queries_batch(
        self,
        queries: list[PageQuery],
        available_pages: list[str] | None = None,
        context: AttentionContext | None = None
    ) -> dict[str, list[AttentionScore]]:
        """Route multiple queries in batch.

        Args:
            queries: List of queries to route
            available_pages: Optional page filter
            context: Passed to base routing

        Returns:
            Dict mapping query text → relevant pages
        """
        # TODO: Maximize overlap between pages of different queries.
        # Perhaps cluster queries first? Or boost score for pages relevant to multiple queries?

        import asyncio

        # Route all queries in parallel
        tasks = [
            self.base_routing.route_query(query, available_pages, context)
            for query in queries
        ]

        results = await asyncio.gather(*tasks)

        # Map query text to results
        query_results = {
            query.query_text: result
            for query, result in zip(queries, results)
        }

        logger.info(f"Batched routing: routed {len(queries)} queries")

        return query_results




async def create_page_query_router1(
    agent: Agent,
    top_k_clusters: int = 5,
    top_n_pages_per_cluster: int = 3,
    top_n_pages_overall: int = 10,
) -> PageQueryRoutingPolicy:
    """
     Max pages in working set is determined by job quota
    """
    # TODO: All this setup is for the working set manager. Hide it.
    # Initialize coordination policy
    from .attention_policy import HierarchicalAttentionPolicy
    from .attention import EmbeddingBasedAttention

    # Create attention components (needed for query routing)
    attention_mechanism = EmbeddingBasedAttention()

    # Create attention policy
    attention_policy = HierarchicalAttentionPolicy(
        agent=agent,
        attention_mechanism=attention_mechanism,
        top_k_clusters=top_k_clusters,
        top_n_pages_per_cluster=top_n_pages_per_cluster,
        top_n_pages_overall=top_n_pages_overall
    )
    await attention_policy.initialize()

    # Create query routing policy
    return HierarchicalAttentionRouting(attention_policy)


async def create_page_query_router2(
    agent: Agent,
    attention_policy_type: str = "hierarchical",
    top_k_clusters: int = 5,
    top_n_pages_per_cluster: int = 3,
    top_n_pages_overall: int = 10,
    top_n_pages: int = 10,
    cluster_id: str | None = None,
    router_type: str = "hierarchical",
    page_keys: dict[str, Any] | None = None,
    working_set: set[str] | None = None,
    cache_boost_factor: float = 1.5
) -> PageQueryRoutingPolicy | None:
    from .attention_policy import (
        HierarchicalAttentionPolicy,
        GlobalAttentionPolicy,
        LocalAttentionPolicy
    )
    from .attention import EmbeddingBasedAttention

    # Attention mechanism
    attention_mechanism = EmbeddingBasedAttention()

    # Initialize attention policy (injectable, defaults to hierarchical)
    if attention_policy_type == "hierarchical":
        attention_policy = HierarchicalAttentionPolicy(
            agent=agent,
            attention_mechanism=attention_mechanism,
            top_k_clusters=top_k_clusters,
            top_n_pages_per_cluster=top_n_pages_per_cluster,
            top_n_pages_overall=top_n_pages_overall
        )
    elif attention_policy_type == "global":
        attention_policy = GlobalAttentionPolicy(
            agent=agent,
            attention_mechanism=attention_mechanism,
            top_n_pages=top_n_pages
        )
    elif attention_policy_type == "local":
        attention_policy = LocalAttentionPolicy(
            agent=agent,
            attention_mechanism=attention_mechanism,
            scope_clusters=[cluster_id] if cluster_id else None,
            top_n_pages=top_n_pages
        )
    else:
        raise ValueError(f"Unknown attention_policy_type: {attention_policy_type}")

    await attention_policy.initialize()

    # Initialize query routing policy (unified interface)
    # Wrap attention_policy in HierarchicalAttentionRouting for backwards compatibility

    if router_type == "hierarchical":
        # Use hierarchical routing (wraps attention_policy)
        query_router = HierarchicalAttentionRouting(attention_policy)
    elif router_type == "direct":
        # Use direct attention with local keys only
        query_router = DirectAttentionRouting(
            attention_mechanism=attention_mechanism,
            page_keys=page_keys
        )
    elif router_type == "cache_aware":
        # Use cache-aware routing (requires working_set)
        base_router = HierarchicalAttentionRouting(attention_policy)
        query_router = CacheAwareRouting(
            base_routing=base_router,
            working_set=working_set,
            cache_boost_factor=cache_boost_factor
        )
    else:
        # Default to hierarchical
        query_router = HierarchicalAttentionRouting(attention_policy)

    return query_router




