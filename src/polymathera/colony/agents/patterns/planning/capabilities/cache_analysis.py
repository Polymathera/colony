"""Cache-aware planning capability.

Provides cache analysis, working set estimation, and action sequence
optimization for VCM-paged context. Used by agents that need to reason
about which pages to load and in what order.

Dual interface:
- **Programmatic API**: ``analyze_cache_requirements()``,
  ``optimize_action_sequence()``, ``estimate_working_set()`` — used by
  ``CacheAwareActionPlanner`` (pre-programmed) and
  ``CodeGenerationActionPolicy`` (generated code).
- **LLM API**: ``@action_executor`` methods with simple parameters — used by
  ``MinimalActionPolicy`` and other JSON-selecting policies where the LLM
  decides whether and when to invoke cache analysis.
"""

from __future__ import annotations

from typing import Any
import logging

import networkx as nx
from overrides import override

from ....base import Agent, AgentCapability
from ....models import (
    Action,
    AgentSuspensionState,
    CacheContext,
    PlanningContext,
)
from ...actions.dispatcher import action_executor
from ....scopes import BlackboardScope, get_scope_prefix

logger = logging.getLogger(__name__)


class CacheAnalysisCapability(AgentCapability):
    """Cache-aware planning analysis for VCM-paged context.

    - Analyzes cache requirements (working set, min/ideal cache sizes, page priorities) for goals. Analyzes which pages an agent should prioritize.
    - Estimates working sets from page graphs
    - Calculates spatial locality from page dependency graphs
    - Optimizes action sequences for cache efficiency

    This is a **cognitive capability** — it augments an agent's reasoning
    about its own cache usage. It does NOT directly load or evict pages;
    it produces analysis that informs planning decisions.

    Usage::

        # Register on agent (CacheAwareActionPlanner does this automatically)
        agent.add_capability(CacheAnalysisCapability(
            agent=agent,
            cache_capacity=20,
        ))

        # Programmatic API (CacheAwareActionPlanner, CodeGenerationActionPolicy)
        cache_ctx = await cap.analyze_cache_requirements(planning_context)
        optimized = await cap.optimize_action_sequence(actions, cache_ctx)

        # LLM API (MinimalActionPolicy — LLM decides to call this)
        # Available as: analyze_cache, get_cache_optimal_batches, get_page_dependencies
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "cache_analysis",
        cache_capacity: int = 10,
        query_vcm_state: bool = False,
        input_patterns: list[str] | None = None,
        capability_key: str = "cache_analysis",
        app_name: str | None = None,
    ):
        """Initialize cache analysis capability.

        Args:
            agent: Owning agent.
            scope: Blackboard scope for this capability.
            namespace: Namespace for blackboard entries.
            cache_capacity: Maximum number of pages that can be cached.
            query_vcm_state: If True, query live VCM for currently cached pages.
            input_patterns: Event patterns (empty — this is action-executor only).
            capability_key: Capability registration key.
            app_name: The `serving.Application` name where the agent system resides.
                    Required when creating detached handles from outside any `serving.deployment`.
        """
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=input_patterns or [],
            capability_key=capability_key,
            app_name=app_name,
        )
        self.cache_capacity = cache_capacity
        self.query_vcm_state = query_vcm_state

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"planning"})

    def get_action_group_description(self) -> str:
        return (
            "Cache Analysis — analyzes VCM cache requirements for agent action planning. "
            "Estimates working sets, calculates page priorities based on graph "
            "centrality, optimizes action sequences for cache locality, and "
            "splits large working sets into cache-friendly batches."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    # =========================================================================
    # Programmatic API
    # =========================================================================
    # Used by CacheAwareActionPlanner (pre-programmed sequence) and
    # CodeGenerationActionPolicy (generated Python code).
    # These accept complex system objects (PlanningContext, CacheContext, etc.)
    # that the LLM planner cannot construct from JSON.

    async def analyze_cache_requirements(
        self,
        context: PlanningContext,
        page_graph: nx.DiGraph | None = None,
    ) -> CacheContext:
        """Analyze cache requirements (working set, min/ideal cache sizes, page priorities) for the given planning context.

        Examines the agent's bound pages, expands the working set using
        the page dependency graph, calculates spatial locality and page
        priorities, and returns a fully populated ``CacheContext``.

        Args:
            context: Planning context with goals and page hints.
            page_graph: Page dependency graph. If None, uses agent's
                page graph if available.

        Returns:
            CacheContext with working set, priorities, locality, and sizing.
        """
        # TODO: Actually analyze goals and context to estimate working set
        logger.info(f"Analyzing cache requirements for {len(context.goals)} goals")

        # Try to load page graph from agent if not provided
        if page_graph is None:
            try:
                page_graph = await self.agent.load_page_graph()
            except Exception:
                pass

        cache_context = CacheContext()

        # Extract bound pages from context (if agent has page affinity)
        page_ids = context.page_ids
        cache_context.working_set.extend(page_ids)

        # Analyze page relationships from graph
        # Expand working set using graph structure
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
        page_graph: nx.DiGraph | None = None,
    ) -> list[str]:
        """Estimate the working set of pages needed for plan execution.

        Args:
            goals: Planning goals
            actions: Planned actions (inspected for page_id parameters)
            page_graph: Page dependency graph for expanding dependencies

        Returns:
            Deduplicated list of page IDs in the estimated working set.
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

        # Deduplicate while preserving order
        seen = set()
        return [p for p in working_set if p not in seen and not seen.add(p)]

    async def optimize_action_sequence(
        self,
        actions: list[Action],
        cache_context: CacheContext,
    ) -> list[Action]:
        """Reorder actions for cache efficiency.

        Groups actions on the same pages and respects spatial locality
        to minimize cache thrashing.

        Args:
            actions: Original action sequence.
            cache_context: Cache context with locality information.

        Returns:
            Reordered action sequence optimized for cache hits.
        """
        logger.info(f"Optimizing {len(actions)} actions for cache efficiency")

        page_actions = []
        non_page_actions = []

        for action in actions:
            if "page_id" in action.parameters:
                page_actions.append(action)
            else:
                non_page_actions.append(action)

        # Group by page_id
        page_groups: dict[str, list[Action]] = {}
        for action in page_actions:
            page_id = action.parameters["page_id"]
            page_groups.setdefault(page_id, []).append(action)

        # Order by spatial locality
        optimized: list[Action] = []
        if cache_context.spatial_locality:
            processed_pages = set()
            for page_id in cache_context.working_set:
                if page_id in page_groups and page_id not in processed_pages:
                    optimized.extend(page_groups[page_id])
                    processed_pages.add(page_id)

                    # Add spatially related pages immediately after
                    for related_page in cache_context.spatial_locality.get(page_id, []):
                        if related_page in page_groups and related_page not in processed_pages:
                            optimized.extend(page_groups[related_page])
                            processed_pages.add(related_page)
        else:
            # No locality info, just concatenate groups
            for group in page_groups.values():
                optimized.extend(group)

        # Append non-page actions at end
        optimized.extend(non_page_actions)

        logger.info(
            f"Optimized action sequence: {len(actions)} -> {len(optimized)} actions"
        )
        return optimized

    # =========================================================================
    # LLM API (@action_executor)
    # =========================================================================
    # Used by MinimalActionPolicy and other JSON-selecting policies.
    # These accept simple, LLM-producible parameters and delegate to the
    # programmatic API internally.

    @action_executor(
        planning_summary=(
            "Analyze cache requirements and return working set priorities "
            "for the agent's bound pages or specified page IDs."
        ),
    )
    async def analyze_cache(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze which pages to prioritize for cache-optimal execution.

        Args:
            page_ids: Pages to analyze. If None, uses the agent's bound pages.

        Returns:
            Dict with working_set, priorities, spatial_locality, and sizing.
        """
        # Build a minimal PlanningContext from simple params
        context = PlanningContext(
            goals=self.agent.metadata.goals or [],
            page_ids=page_ids or list(self.agent.bound_pages),
        )
        cache_ctx = await self.analyze_cache_requirements(context)
        return cache_ctx.model_dump()

    @action_executor(
        planning_summary=(
            "Split pages into cache-friendly batches for sequential processing. "
            "Use when the working set is too large for the cache."
        ),
    )
    async def get_cache_optimal_batches(
        self,
        page_ids: list[str],
        batch_size: int = 10,
    ) -> dict[str, Any]:
        """Split pages into cache-friendly batches.

        Groups pages by spatial locality so each batch maximizes cache hits.

        Args:
            page_ids: Pages to batch.
            batch_size: Maximum pages per batch.

        Returns:
            Dict with 'batches' (list of page_id lists) and 'batch_count'.
        """
        # Build cache context for locality analysis
        context = PlanningContext(
            goals=self.agent.metadata.goals or [],
            page_ids=page_ids,
        )
        cache_ctx = await self.analyze_cache_requirements(context)

        # Use spatial locality to group pages into batches
        batches: list[list[str]] = []
        current_batch: list[str] = []
        processed = set()

        for page_id in cache_ctx.working_set:
            if page_id in processed:
                continue

            current_batch.append(page_id)
            processed.add(page_id)

            # Add spatially related pages to the same batch
            for related in cache_ctx.spatial_locality.get(page_id, []):
                if related not in processed and len(current_batch) < batch_size:
                    current_batch.append(related)
                    processed.add(related)

            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        return {
            "batches": batches,
            "batch_count": len(batches),
            "total_pages": sum(len(b) for b in batches),
        }

    @action_executor(
        planning_summary=(
            "Get dependency information for a specific page from the page graph."
        ),
    )
    async def get_page_dependencies(
        self,
        page_id: str,
    ) -> dict[str, Any]:
        """Get page graph information for a specific page.

        Args:
            page_id: The page to inspect.

        Returns:
            Dict with predecessors, successors, degree, and centrality info.
        """
        page_graph = None
        try:
            page_graph = await self.agent.load_page_graph()
        except Exception:
            pass

        if not page_graph or not page_graph.has_node(page_id):
            return {
                "page_id": page_id,
                "in_graph": False,
                "predecessors": [],
                "successors": [],
                "degree": 0,
            }

        return {
            "page_id": page_id,
            "in_graph": True,
            "predecessors": list(page_graph.predecessors(page_id)),
            "successors": list(page_graph.successors(page_id)),
            "degree": page_graph.degree(page_id),
        }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _find_related_pages_in_graph(
        self, pages: list[str], page_graph: nx.DiGraph
    ) -> list[str]:
        """Find pages related to given pages via graph edges (predecessors + successors).

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
        degrees = [
            (node, page_graph.degree(node))
            for node in working_set
            if page_graph.has_node(node)
        ]
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
                priorities[page_id] = 1.5 if page_id in currently_cached_pages else 1.0
            return priorities

        cached_set = set(currently_cached_pages)

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
            from .....system import get_vcm

            # Get VCM handle
            vcm_handle = get_vcm()

            # Query VCM for all loaded pages
            cached_pages = await vcm_handle.get_all_loaded_pages()

            logger.debug(f"VCM query returned {len(cached_pages)} cached pages")
            return cached_pages

        except Exception as e:
            logger.warning(f"Failed to query VCM for cached pages: {e}")
            return []

