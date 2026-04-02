"""Pluggable policies for the planning system.

This module provides pluggable policies that customize planning behavior:
- ActionPlanningCachePolicy: Cache-aware planning and optimization

Per Architecture Principle #2: ONE Planner class customized via pluggable policies.

Design: Policies are stateless and receive context (page graphs, etc.) when called.
The Planner that uses these policies is responsible for providing the necessary context.
"""

from typing import Any
import logging

import networkx as nx

from ...models import (
    Action,
    CacheContext,
    PlanningContext,
)
from ...base import Agent

logger = logging.getLogger(__name__)



# ============================================================================
# Cache-Aware Planning Policy
# ============================================================================


class ActionPlanningCachePolicy:
    """Cache-aware planning policy.

    Plugs into existing Planner to provide cache-aware planning:
    - Analyzes cache requirements (working set, min/ideal cache sizes, page priorities) for goals
    - Estimates working sets from page graphs
    - Optimizes action sequences for cache efficiency

    Per Architecture Principle #2: This is a pluggable policy, NOT a separate planner.

    Design: Policy is stateless. Page graph and other context are passed in when calling methods.
    """
    def __init__(self, agent: Agent | None = None, cache_capacity: int = 10, query_vcm_state: bool = False):
        """Initialize cache-aware planning policy.

        Args:
            cache_capacity: Maximum number of pages that can be cached
            query_vcm_state: If True, query VCM for current cache state (requires VCM access)
        """
        self.agent = agent
        self.cache_capacity = cache_capacity
        self.query_vcm_state = query_vcm_state

    async def initialize(self) -> None:
        """Initialize policy."""
        logger.info("Initializing ActionPlanningCachePolicy")

    async def analyze_cache_requirements(
        self,
        context: PlanningContext,
        page_graph: nx.DiGraph | None = None,
    ) -> CacheContext:
        """Analyze cache requirements (working set, min/ideal cache sizes, page priorities) for given goals.

        Args:
            context: Planning context with hints about required pages
            page_graph: Page dependency graph (networkx.DiGraph)

        Returns:
            CacheContext with estimated requirements
        """
        # TODO: Actually analyze goals and context to estimate working set
        logger.info(f"Analyzing cache requirements for {len(context.goals)} goals")

        cache_context = CacheContext()

        # Extract bound pages from context (if agent has page affinity)
        page_ids = context.page_ids
        cache_context.working_set.extend(page_ids)

        # Analyze page relationships from graph
        if page_graph:
            # Add related pages based on graph structure
            related_pages = self._find_related_pages_in_graph(
                cache_context.working_set, page_graph
            )
            cache_context.working_set.extend(related_pages)

            # Build summary of page relationships
            cache_context.page_graph_summary = self._summarize_page_graph(
                cache_context.working_set, page_graph
            )

            # Calculate spatial locality from graph
            cache_context.spatial_locality = self._calculate_spatial_locality(
                cache_context.working_set, page_graph
            )

        # Set cache sizing
        cache_context.min_cache_size = len(cache_context.working_set)
        cache_context.ideal_cache_size = min(
            len(cache_context.working_set) + 5,  # Add buffer for prefetching - TODO: Make configurable
            self.cache_capacity,
        )

        # Assign priorities based on graph centrality and VCM state
        cache_context.working_set_priority = await self._calculate_page_priorities(
            cache_context, page_graph
        )

        logger.info(
            f"Cache requirements: {len(cache_context.working_set)} pages, "
            f"min={cache_context.min_cache_size}, ideal={cache_context.ideal_cache_size}"
        )

        return cache_context

    async def estimate_working_set(
        self,
        goals: list[str],
        actions: list[Action],
        page_graph: nx.DiGraph | None = None,  # networkx.DiGraph
    ) -> list[str]:
        """Estimate working set of pages for plan execution.

        Args:
            goals: Planning goals
            actions: Planned actions
            page_graph: Page dependency graph (networkx.DiGraph)

        Returns:
            List of page IDs in working set
        """
        working_set = []

        # Extract pages from actions
        for action in actions:
            page_id = action.parameters.get("page_id")
            if page_id:
                working_set.append(page_id)

            # For ANALYZE_PAGE actions, add dependencies from graph
            if action.action_type.value == "analyze_page" and page_graph and page_id:
                if page_graph.has_node(page_id):
                    # Add direct predecessors (dependencies)
                    predecessors = list(page_graph.predecessors(page_id))
                    working_set.extend(predecessors)

        # Remove duplicates while preserving order
        seen = set()
        unique_working_set = []
        for page_id in working_set:
            if page_id not in seen:
                seen.add(page_id)
                unique_working_set.append(page_id)

        return unique_working_set

    async def optimize_action_sequence(
        self,
        actions: list[Action],
        cache_context: CacheContext,
    ) -> list[Action]:
        """Optimize action sequence for cache efficiency.

        Reorders actions to maximize cache hits by:
        - Grouping actions on same pages
        - Respecting spatial locality
        - Minimizing cache thrashing

        Args:
            actions: Original action sequence
            cache_context: Cache context with locality information

        Returns:
            Optimized action sequence
        """
        logger.info(f"Optimizing {len(actions)} actions for cache efficiency")

        # Separate actions that access pages from those that don't
        page_actions = []
        non_page_actions = []

        for action in actions:
            if "page_id" in action.parameters:
                page_actions.append(action)
            else:
                non_page_actions.append(action)

        # Group page actions by page_id
        page_groups: dict[str, list[Action]] = {}
        for action in page_actions:
            page_id = action.parameters["page_id"]
            if page_id not in page_groups:
                page_groups[page_id] = []
            page_groups[page_id].append(action)

        # Build optimized sequence using spatial locality
        optimized = []
        if cache_context.spatial_locality:
            processed_pages = set()
            for page_id in cache_context.working_set:
                if page_id in page_groups and page_id not in processed_pages:
                    optimized.extend(page_groups[page_id])
                    processed_pages.add(page_id)

                    # Add spatially related pages immediately after
                    related = cache_context.spatial_locality.get(page_id, [])
                    for related_page in related:
                        if (
                            related_page in page_groups
                            and related_page not in processed_pages
                        ):
                            optimized.extend(page_groups[related_page])
                            processed_pages.add(related_page)
        else:
            # No locality info, just concatenate groups
            for actions_group in page_groups.values():
                optimized.extend(actions_group)

        # Append non-page actions at end
        optimized.extend(non_page_actions)

        logger.info(
            f"Optimized action sequence: {len(actions)} -> {len(optimized)} actions"
        )
        return optimized

    def _find_related_pages_in_graph(
        self, pages: list[str], page_graph: nx.DiGraph
    ) -> list[str]:
        """Find pages related to given pages via graph edges.

        Args:
            pages: Starting set of pages
            page_graph: networkx.DiGraph

        Returns:
            Related pages (dependencies, dependents)
        """
        related = set()

        for page_id in pages:
            if not page_graph.has_node(page_id):
                continue

            # Add direct predecessors (dependencies)
            related.update(page_graph.predecessors(page_id))

            # Add direct successors (dependents)
            related.update(page_graph.successors(page_id))

        # Remove pages already in input
        related -= set(pages)
        return list(related)

    def _summarize_page_graph(
        self, working_set: list[str], page_graph: nx.DiGraph
    ) -> dict[str, Any]:
        """Summarize page graph for working set.

        Args:
            working_set: Pages in working set
            page_graph: networkx.DiGraph

        Returns:
            Summary with clusters, dependencies, etc.
        """
        summary: dict[str, Any] = {
            "clusters": [],
            "dependencies": {},
            "central_pages": [],
        }

        # Extract dependencies for working set
        for page_id in working_set:
            if page_graph.has_node(page_id):
                deps = list(page_graph.predecessors(page_id))
                if deps:
                    summary["dependencies"][page_id] = deps

        # Find central pages (high degree)
        degrees = [(node, page_graph.degree(node)) for node in working_set if page_graph.has_node(node)]
        degrees.sort(key=lambda x: x[1], reverse=True)
        summary["central_pages"] = [node for node, _ in degrees[:5]]

        return summary

    def _calculate_spatial_locality(
        self, working_set: list[str], page_graph: nx.DiGraph
    ) -> dict[str, list[str]]:
        """Calculate spatial locality from graph structure.

        Args:
            working_set: Pages in working set
            page_graph: networkx.DiGraph

        Returns:
            Mapping of page_id -> list of related pages
        """
        locality: dict[str, list[str]] = {}

        for page_id in working_set:
            if not page_graph.has_node(page_id):
                continue

            related = set()

            # Pages with shared dependencies are likely accessed together
            my_deps = set(page_graph.predecessors(page_id))

            for other_page in working_set:
                if other_page == page_id or not page_graph.has_node(other_page):
                    continue

                other_deps = set(page_graph.predecessors(other_page))

                # Check for shared dependencies
                if my_deps & other_deps:
                    related.add(other_page)

            locality[page_id] = list(related)

        return locality

    async def _calculate_page_priorities(
        self,
        cache_context: CacheContext,
        page_graph: nx.DiGraph | None,
    ) -> dict[str, float]:
        """Calculate page priorities based on graph centrality and VCM cache state.

        Args:
            cache_context: CacheContext with working set
            page_graph: networkx.DiGraph

        Returns:
            Mapping of page_id -> priority (0.0 to 2.0, higher = more important)
        """
        working_set: list[str] = cache_context.working_set

        priorities: dict[str, float] = {}

        # Query VCM for current cache state if enabled
        currently_cached_pages = []
        if self.query_vcm_state:
            currently_cached_pages = await self._query_vcm_cached_pages()
            logger.info(f"VCM reports {len(currently_cached_pages)} pages currently cached")

        if page_graph is None:
            # Default priorities, prioritize already cached pages
            for page_id in working_set:
                # Higher priority for already cached pages (cache hit > cache miss)
                if page_id in currently_cached_pages:
                    priorities[page_id] = 1.5
                else:
                    priorities[page_id] = 1.0
            return priorities

        cached_set = set(currently_cached_pages) if currently_cached_pages else set()

        # Calculate degree centrality for working set nodes
        max_degree = 0
        degrees: dict[str, int] = {}

        for page_id in working_set:
            if page_graph.has_node(page_id):
                degree = page_graph.degree(page_id)
                degrees[page_id] = degree
                max_degree = max(max_degree, degree)

        # Normalize to 0.0-1.0 range and boost cached pages
        for page_id in working_set:
            if page_id in degrees and max_degree > 0:
                base_priority = degrees[page_id] / max_degree
            else:
                base_priority = 0.5  # Default for nodes not in graph

            # Boost priority for already cached pages (cache hit > cache miss)
            if page_id in cached_set:
                priorities[page_id] = min(base_priority + 0.5, 2.0)
            else:
                priorities[page_id] = base_priority

        return priorities

    async def _query_vcm_cached_pages(self) -> list[str]:
        """Query VCM for currently cached pages.

        Returns:
            List of page IDs currently loaded in VCM
        """
        try:
            from ...system import get_vcm

            # Get VCM handle
            vcm_handle = get_vcm()

            # Query VCM for all loaded pages
            cached_pages = await vcm_handle.get_all_loaded_pages()

            logger.debug(f"VCM query returned {len(cached_pages)} cached pages")
            return cached_pages

        except Exception as e:
            logger.warning(f"Failed to query VCM for cached pages: {e}")
            return []

