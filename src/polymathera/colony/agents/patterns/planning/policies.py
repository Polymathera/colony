"""Pluggable policies for the planning system.

This module provides pluggable policies that customize planning behavior:
- CacheAwarePlanningPolicy: Cache-aware planning and optimization
- LearningPlanningPolicy: Learning from execution history
- CoordinationPlanningPolicy: Multi-agent coordination

Per Architecture Principle #2: ONE Planner class customized via pluggable policies.

Design: Policies are stateless and receive context (page graphs, etc.) when called.
The Planner that uses these policies is responsible for providing the necessary context.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging
import time

import networkx as nx

from ...models import (
    Action,
    CacheContext,
    ActionPlan,
    PlanStatus,
    ConflictType,
    ConflictSeverity,
    ActionPlanConflict,
    ConflictResolutionStrategy,
    PlanExecutionRecord,
    PlanningContext,
    PlanPattern,
)
from ...base import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# Policy Protocols
# ============================================================================


class PlanningPolicy(ABC):
    """Base protocol for planning policies."""
    def __init__(self, agent: Agent | None = None):
        self.agent = agent

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize policy (load models, connect to services, etc.)."""
        ...


# ============================================================================
# Cache-Aware Planning Policy
# ============================================================================


class CacheAwarePlanningPolicy(PlanningPolicy):
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
        super().__init__(agent)
        self.cache_capacity = cache_capacity
        self.query_vcm_state = query_vcm_state

    async def initialize(self) -> None:
        """Initialize policy."""
        logger.info("Initializing CacheAwarePlanningPolicy")

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


# ============================================================================
# Learning Policy
# ============================================================================


class LearningPlanningPolicy(PlanningPolicy):
    """Learning policy for plan improvement.

    Uses execution history to:
    - Learn successful patterns
    - Refine cost models
    - Recommend plan improvements

    Per Architecture Principle #2: This is a pluggable policy, NOT a separate planner.
    """

    def __init__(self, agent: Agent | None = None):
        """Initialize learning policy.

        Args:
            agent: Agent instance for accessing blackboard and other capabilities
        """
        super().__init__(agent)
        self.history_store = None
        self.pattern_learner = None
        self.cost_trainer = None

    async def initialize(self) -> None:
        """Initialize policy (load learned patterns, cost models)."""
        logger.info("Initializing LearningPlanningPolicy")

        # Initialize learning components
        from .learning import (
            ExecutionHistoryStore,
            PatternLearner,
            CostModelTrainer,
        )

        self.history_store = ExecutionHistoryStore(
            blackboard=await self.agent.get_blackboard(
                scope_id=f"{self.agent.agent_id}:planning_history"
            )
        )
        await self.history_store.initialize()

        self.pattern_learner = PatternLearner(self.history_store)
        await self.pattern_learner.initialize()

        self.cost_trainer = CostModelTrainer(self.history_store)
        await self.cost_trainer.initialize()

        # Train models from existing history
        await self.cost_trainer.train()
        await self.pattern_learner.learn_patterns()

        logger.info("Learning policy initialized with execution history")

    async def get_applicable_patterns(
        self,
        context: PlanningContext
    ) -> list[PlanPattern]:
        """Get patterns applicable to given goals.

        Args:
            context: Planning context

        Returns:
            List of applicable patterns
        """
        if not self.pattern_learner:
            logger.warning("Pattern learner not initialized")
            return []

        # Query patterns for goals
        goal_str = " ".join(context.goals)
        patterns = await self.pattern_learner.get_applicable_patterns(goal_str, context)

        logger.info(f"Found {len(patterns)} applicable patterns for goals")
        return patterns

    async def get_similar_plans(
        self, goals: list[str], context: PlanningContext, limit: int = 5
    ) -> list[PlanExecutionRecord]:
        """Get similar successful plans from history.

        Args:
            goals: Current goals
            context: Planning context
            limit: Maximum number of plans to return

        Returns:
            List of similar plan records
        """
        if not self.history_store:
            logger.warning("History store not initialized")
            return []

        # Query for similar goals
        goal_str = " ".join(goals)
        similar = await self.history_store.query_by_goal(goal_str, limit=limit)

        logger.info(f"Found {len(similar)} similar plans for goals: {goals}")
        return similar

    async def learn_from_execution(
        self, plan: ActionPlan, outcome: dict[str, Any]
    ) -> None:
        """Learn from plan execution outcome.

        Args:
            plan: Executed plan
            outcome: Execution outcome with metrics
        """
        if not self.history_store:
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
        await self.history_store.record_execution(record)

        # Periodically retrain models (every 10 executions)
        stats = await self.history_store.get_statistics()
        if stats.get("total_executions", 0) % 10 == 0:
            logger.info("Retraining cost and pattern models")
            await self.cost_trainer.train()
            await self.pattern_learner.learn_patterns()


# ============================================================================
# Cache-Aware Conflict Detection and Resolution
# ============================================================================


class CacheAwareConflictDetector:
    """Detect conflicts between plans based on cache requirements.

    Identifies conflicts such as:
    - Resource exhaustion (total working set exceeds cache capacity)
    - Cache contention (multiple plans need same pages)
    - Sequential dependencies (plans must execute in order)
    """

    def __init__(self, cache_capacity: int, page_size: int = 40000):
        """Initialize conflict detector.

        Args:
            cache_capacity: Maximum number of pages that can be cached
            page_size: Average page size in tokens
        """
        self.cache_capacity = cache_capacity
        self.page_size = page_size

    async def detect_conflicts(self, plans: list[ActionPlan]) -> list[ActionPlanConflict]:
        """Detect conflicts between plans.

        Args:
            plans: List of plans to check for conflicts

        Returns:
            List of action plan conflicts
        """

        conflicts = []

        # Check for resource exhaustion
        all_pages = set()
        for plan in plans:
            all_pages.update(plan.cache_context.working_set)

        resource_conflict = self._check_resource_exhaustion(all_pages)
        if resource_conflict:
            conflicts.append(resource_conflict)

        # Check pairwise cache contention
        for i, plan_a in enumerate(plans):
            for plan_b in plans[i + 1 :]:
                contention = self._check_cache_contention(plan_a, plan_b)
                if contention:
                    conflicts.append(contention)

        # Calculate severity for each conflict
        for conflict in conflicts:
            conflict.severity = self._calculate_conflict_severity(conflict)

        logger.info(f"Detected {len(conflicts)} conflicts between {len(plans)} plans")
        return conflicts

    def _check_resource_exhaustion(self, total_working_set: set[str]) -> ActionPlanConflict | None:
        """Check if total working set exceeds cache capacity.

        Args:
            total_working_set: Combined working set of all plans

        Returns:
            Conflict description if exhaustion detected, None otherwise
        """

        if len(total_working_set) > self.cache_capacity:
            return ActionPlanConflict(
                type=ConflictType.RESOURCE_CONTENTION,
                description=f"Total working set ({len(total_working_set)} pages) exceeds cache capacity ({self.cache_capacity} pages)",
                severity=ConflictSeverity.HIGH,
                details={
                    "total_pages": len(total_working_set),
                    "capacity": self.cache_capacity,
                    "overflow": len(total_working_set) - self.cache_capacity,
                },
                recommended_strategy=ConflictResolutionStrategy.STAGGER_EXECUTION,
            )
        return None

    def _check_cache_contention(self, plan_a: ActionPlan, plan_b: ActionPlan) -> ActionPlanConflict | None:
        """Check for cache contention between two plans.

        Args:
            plan_a: First plan
            plan_b: Second plan

        Returns:
            Conflict description if contention detected, None otherwise
        """

        # Find overlapping pages
        pages_a = set(plan_a.cache_context.working_set)
        pages_b = set(plan_b.cache_context.working_set)
        shared_pages = pages_a & pages_b

        # Check if combined working set exceeds capacity
        combined_size = len(pages_a | pages_b)
        if combined_size > self.cache_capacity:
            return ActionPlanConflict(
                type=ConflictType.CACHE_CONTENTION,
                description=f"Plans {plan_a.plan_id} and {plan_b.plan_id} have combined working set ({combined_size} pages) exceeding cache capacity",
                plan_a_id=plan_a.plan_id,
                plan_b_id=plan_b.plan_id,
                details={
                    "shared_pages": list(shared_pages),
                    "combined_size": combined_size,
                    "capacity": self.cache_capacity,
                    "overflow": combined_size - self.cache_capacity,
                },
                recommended_strategy=ConflictResolutionStrategy.STAGGER_EXECUTION,
            )
        return None

    def _calculate_conflict_severity(self, conflict: ActionPlanConflict) -> ConflictSeverity:
        """Calculate conflict severity.

        Args:
            conflict: Conflict description

        Returns:
            Severity level: "low", "medium", "high", "critical"
        """

        conflict_type = conflict.type

        if conflict_type == ConflictType.RESOURCE_CONTENTION:
            overflow = conflict.details.get("overflow", 0)
            overflow_ratio = overflow / self.cache_capacity

            if overflow_ratio > 0.5:
                return ConflictSeverity.CRITICAL
            elif overflow_ratio > 0.25:
                return ConflictSeverity.HIGH
            elif overflow_ratio > 0.1:
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW

        elif conflict_type == ConflictType.CACHE_CONTENTION:
            overflow = conflict.details.get("overflow", 0)
            overflow_ratio = overflow / self.cache_capacity

            if overflow_ratio > 0.3:
                return ConflictSeverity.HIGH
            elif overflow_ratio > 0.15:
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW

        return ConflictSeverity.MEDIUM


class CacheAwareConflictResolver:
    """Resolve conflicts between plans by modifying them.

    Resolution strategies:
    - Stagger execution (delay some plans)
    - Partition working sets (split plans)
    - Coordinate shared pages (optimize for sharing)
    """

    async def resolve_cache_contention(
        self, plan_a: ActionPlan, plan_b: ActionPlan, conflict: ActionPlanConflict
    ) -> tuple[ActionPlan, ActionPlan]:
        """Resolve cache contention between two plans.

        Args:
            plan_a: First plan
            plan_b: Second plan
            conflict: Conflict description

        Returns:
            Modified (plan_a, plan_b) tuple
        """
        logger.info(
            f"Resolving cache contention between {plan_a.plan_id} and {plan_b.plan_id}"
        )

        # Strategy: Stagger execution by adding delays
        modified_a, modified_b = await self._stagger_execution([plan_a, plan_b])

        return modified_a, modified_b

    async def _stagger_execution(self, plans: list[ActionPlan]) -> list[ActionPlan]:
        """Stagger plan execution to avoid simultaneous cache usage.

        Args:
            plans: Plans to stagger

        Returns:
            Modified plans with execution delays
        """
        # TODO: Is this even a robust strategy?

        # Add execution delay to all but first plan
        for i, plan in enumerate(plans):
            if i > 0:
                # Add metadata to indicate this plan should be delayed
                plan.metadata["execution_delay_s"] = i * 10.0  # 10 second stagger
                plan.metadata["staggered"] = True

        logger.info(f"Staggered execution of {len(plans)} plans")
        return plans

    async def _partition_working_sets(self, plans: list[ActionPlan]) -> list[ActionPlan]:
        """Partition working sets to reduce overlap.

        Args:
            plans: Plans to partition

        Returns:
            Modified plans with partitioned working sets
        """
        # Strategy: Identify shared pages and assign exclusivity
        all_pages: dict[str, list[str]] = {}  # page_id -> [plan_ids using it]

        for plan in plans:
            for page_id in plan.cache_context.working_set:
                if page_id not in all_pages:
                    all_pages[page_id] = []
                all_pages[page_id].append(plan.plan_id)

        # Mark pages as exclusive or shareable
        for plan in plans:
            exclusive = []
            shareable = []

            for page_id in plan.cache_context.working_set:
                if len(all_pages[page_id]) == 1:
                    # Only this plan uses this page
                    exclusive.append(page_id)
                else:
                    # Multiple plans use this page
                    shareable.append(page_id)

            plan.cache_context.exclusive_pages = exclusive
            plan.cache_context.shareable_pages = shareable

        logger.info(f"Partitioned working sets for {len(plans)} plans")
        return plans

    async def _coordinate_shared_pages(self, plans: list[ActionPlan]) -> list[ActionPlan]:
        """Coordinate shared page access between plans.

        Args:
            plans: Plans to coordinate

        Returns:
            Modified plans with coordinated access
        """
        # Identify frequently shared pages
        page_usage: dict[str, int] = {}
        for plan in plans:
            for page_id in plan.cache_context.working_set:
                page_usage[page_id] = page_usage.get(page_id, 0) + 1

        # Mark highly shared pages for keeping in cache
        high_usage_threshold = len(plans) // 2
        for plan in plans:
            for page_id in plan.cache_context.working_set:
                if page_usage[page_id] >= high_usage_threshold:
                    # Mark as shareable and high priority
                    if page_id not in plan.cache_context.shareable_pages:
                        plan.cache_context.shareable_pages.append(page_id)
                    plan.cache_context.working_set_priority[page_id] = 2.0  # High priority

        logger.info(f"Coordinated shared pages for {len(plans)} plans")
        return plans


# ============================================================================
# Generic Conflict Resolution
# ============================================================================


class ConflictResolver:
    """Generic conflict resolver for plans.

    Resolves conflicts between plans using various strategies:
    - Priority-based resolution/negotiation
    - Temporal staggering
    - Resource efficiency optimization
    - Negotiation between agents
    - Escalation to higher authority
    - Partition working sets
    """

    def __init__(self, default_strategy: str = ConflictResolutionStrategy.PRIORITY):
        """Initialize conflict resolver.

        Args:
            default_strategy: Default resolution strategy to use
        """

        self.default_strategy = default_strategy
        self.cache_resolver = CacheAwareConflictResolver()

    async def resolve_conflict(
        self,
        conflict: ActionPlanConflict,
        plans: list[ActionPlan],
        strategy: str | None = None,
    ) -> list[ActionPlan]:
        """Resolve a conflict between plans.

        Args:
            conflict: Conflict description from detector
            plans: Plans involved in conflict
            strategy: Resolution strategy (uses default if None)

        Returns:
            Modified plans after resolution
        """

        strategy = strategy or self.default_strategy
        conflict_type = conflict.type

        logger.info(
            f"Resolving {conflict_type} conflict using {strategy} strategy"
        )

        if strategy == ConflictResolutionStrategy.PRIORITY:
            return await self._resolve_by_priority(conflict, plans)

        elif strategy == ConflictResolutionStrategy.TEMPORAL:
            return await self._resolve_by_temporal_staggering(conflict, plans)

        elif strategy == ConflictResolutionStrategy.RESOURCE_EFFICIENCY:
            return await self._resolve_by_efficiency(conflict, plans)

        elif strategy == ConflictResolutionStrategy.NEGOTIATION:
            return await self._resolve_by_negotiation(conflict, plans)

        elif strategy == ConflictResolutionStrategy.ESCALATION:
            return await self._resolve_by_escalation(conflict, plans)

        elif strategy == ConflictResolutionStrategy.REPLAN:
            return await self._resolve_by_replanning(conflict, plans)

        else:
            logger.warning(f"Unknown resolution strategy {strategy}, using priority")
            return await self._resolve_by_priority(conflict, plans)

    async def _resolve_by_priority(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by priority ordering.

        Higher priority plans execute first, lower priority plans delayed or cancelled.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans
        """
        # Sort plans by priority (metadata.priority field)
        sorted_plans = sorted(
            plans,
            key=lambda p: p.metadata.get("priority", 0),
            reverse=True,  # Higher priority first
        )

        # Highest priority plan proceeds unchanged
        # Lower priority plans are delayed or suspended
        for i, plan in enumerate(sorted_plans):
            if i == 0:
                # Highest priority - no changes
                continue
            else:
                # Lower priority - add delay or suspend
                plan.metadata["delayed_by_priority"] = True
                plan.metadata["delay_reason"] = f"Conflict with higher priority plan {sorted_plans[0].plan_id}"

                # Suspend if severe conflict
                severity = conflict.severity
                if severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
                    plan.status = PlanStatus.SUSPENDED
                    plan.blocked_reason = f"Suspended due to {severity} conflict with higher priority plan"

        logger.info(f"Resolved conflict by priority: {len(plans)} plans affected")
        return sorted_plans

    async def _resolve_by_temporal_staggering(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by staggering execution over time.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans with execution delays
        """
        # Use cache resolver's staggering logic
        return await self.cache_resolver._stagger_execution(plans)

    async def _resolve_by_efficiency(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by optimizing resource usage.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans optimized for efficiency
        """

        conflict_type = conflict.type

        if conflict_type == ConflictType.CACHE_CONTENTION:
            # Use cache-specific resolution
            return await self.cache_resolver._partition_working_sets(plans)

        elif conflict_type == ConflictType.RESOURCE_CONTENTION:
            # Generic resource optimization - share resources when possible
            for plan in plans:
                # Mark resources as shareable
                plan.metadata["resource_sharing_enabled"] = True

        logger.info("Resolved conflict by efficiency optimization")
        return plans

    async def _resolve_by_negotiation(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict through negotiation between agents.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans after negotiation
        """
        # TODO: Implement negotiation protocol
        # For now, fall back to priority
        logger.warning("Negotiation not implemented, falling back to priority")
        return await self._resolve_by_priority(conflict, plans)

    async def _resolve_by_escalation(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Escalate conflict to higher authority.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Plans marked for escalation
        """
        # Mark all plans as needing approval
        for plan in plans:
            plan.approval_required = True
            plan.metadata["escalated_conflict"] = True
            plan.metadata["conflict_details"] = conflict
            plan.status = PlanStatus.PROPOSED  # Wait for approval

        logger.info(f"Escalated conflict involving {len(plans)} plans")
        return plans

    async def _resolve_by_replanning(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by requesting replanning.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Plans marked for replanning
        """
        # Mark lower priority plans for replanning
        sorted_plans = sorted(
            plans,
            key=lambda p: p.metadata.get("priority", 0),
            reverse=True,
        )

        for i, plan in enumerate(sorted_plans):
            if i > 0:  # Skip highest priority plan
                plan.metadata["requires_replanning"] = True
                plan.metadata["replan_reason"] = f"Conflict: {conflict.description}"

        logger.info(f"Marked {len(plans) - 1} plans for replanning")
        return sorted_plans


# ============================================================================
# Coordination Policy
# ============================================================================


class CoordinationPlanningPolicy:
    """Multi-agent coordination policy.

    Handles coordination between multiple agent plans:
    - Conflict detection
    - Conflict resolution
    - Resource allocation

    Per Architecture Principle #2: This is a pluggable policy, NOT a separate planner.
    """

    def __init__(self, cache_capacity: int = 100):
        """Initialize coordination policy.

        Args:
            cache_capacity: Maximum number of pages that can be cached
        """
        self.detector = CacheAwareConflictDetector(cache_capacity)
        self.resolver = ConflictResolver(default_strategy="priority")

    async def initialize(self) -> None:
        """Initialize policy."""
        logger.info("Initializing CoordinationPlanningPolicy")

    async def check_conflicts(
        self, plan: ActionPlan, other_plans: list[ActionPlan]
    ) -> list[ActionPlanConflict]:
        """Check for conflicts with other plans.

        Args:
            plan: ActionPlan to check
            other_plans: Other active plans

        Returns:
            List of conflicts (empty if no conflicts)
        """
        logger.info(f"Checking conflicts for plan {plan.plan_id}")

        # Use existing conflict detector
        all_plans = [plan] + other_plans
        conflicts = await self.detector.detect_conflicts(all_plans)

        logger.info(f"Found {len(conflicts)} conflicts for plan {plan.plan_id}")
        return conflicts

    async def resolve_conflict(
        self, plan: ActionPlan, conflict: ActionPlanConflict
    ) -> ActionPlan | None:
        """Resolve conflict for a plan.

        Args:
            plan: ActionPlan with conflict
            conflict: Conflict details

        Returns:
            Modified plan, or None if conflict cannot be resolved
        """
        conflict_type = conflict.type
        logger.info(f"Resolving {conflict_type} conflict for plan {plan.plan_id}")

        # Use existing conflict resolver
        strategy = conflict.recommended_strategy
        resolved_plans = await self.resolver.resolve_conflict(
            conflict=conflict,
            plans=[plan],
            strategy=strategy,
        )

        if resolved_plans:
            resolved_plan = resolved_plans[0]
            logger.info(f"Resolved conflict for plan {plan.plan_id} using {strategy} strategy")
            return resolved_plan

        logger.warning(f"Could not resolve conflict for plan {plan.plan_id}")
        return None


# ============================================================================
# Access Control Policy
# ============================================================================


class PlanAccessPolicy(ABC):
    """Policy protocol for plan access control.

    Controls which agents can create, read, update, and approve plans based on
    agent hierarchy, team structure, and plan scope.
    """

    @abstractmethod
    def can_create_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can create this plan.

        Args:
            agent_id: ID of agent attempting to create plan
            plan: ActionPlan to be created

        Returns:
            True if agent can create plan
        """
        ...

    @abstractmethod
    def can_read_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can read this plan.

        Args:
            agent_id: ID of agent attempting to read plan
            plan: ActionPlan to be read

        Returns:
            True if agent can read plan
        """
        ...

    @abstractmethod
    def can_update_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can update this plan.

        Args:
            agent_id: ID of agent attempting to update plan
            plan: ActionPlan to be updated

        Returns:
            True if agent can update plan
        """
        ...

    @abstractmethod
    def can_approve_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can approve this plan.

        Args:
            agent_id: ID of agent attempting to approve plan
            plan: ActionPlan to be approved

        Returns:
            True if agent can approve plan
        """
        ...


class HierarchicalAccessPolicy(PlanAccessPolicy):
    """Hierarchical access control for plans.

    Access rules:
    - Agents can always manage (create/read/update) their own plans
    - Parent agents can read/approve child plans
    - Agents in same team can read each other's plans
    - Only parent agents can approve plans requiring approval
    - System-scope plans require higher-level approval
    """

    def __init__(
        self,
        agent_hierarchy: dict[str, str] | None = None,
        team_structure: dict[str, set[str]] | None = None,
    ):
        """Initialize hierarchical access policy.

        Args:
            agent_hierarchy: Mapping of agent_id -> parent_agent_id
            team_structure: Mapping of team_name -> set of agent_ids
        """
        self.agent_hierarchy = agent_hierarchy or {}
        self.team_structure = team_structure or {}

    def can_create_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can create this plan.

        Rules:
        - Agents can create plans for themselves
        - Plans with parent_plan_id require approval from parent plan's agent

        Args:
            agent_id: ID of agent attempting to create plan
            plan: ActionPlan to be created

        Returns:
            True if agent can create plan
        """
        # Agents can always create plans for themselves
        if plan.agent_id == agent_id:
            return True

        # Cannot create plans for other agents
        return False

    def can_read_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can read this plan.

        Rules:
        - Agents can read their own plans
        - Parent agents can read child plans
        - Team members can read each other's plans
        - GLOBAL visibility plans can be read by anyone

        Args:
            agent_id: ID of agent attempting to read plan
            plan: ActionPlan to be read

        Returns:
            True if agent can read plan
        """
        # Global plans are readable by all
        if plan.visibility == "global":
            return True

        # Agents can read their own plans
        if plan.agent_id == agent_id:
            return True

        # Parent agents can read child plans
        if self._is_parent(agent_id, plan.agent_id):
            return True

        # Team members can read each other's plans
        if self._is_in_team(agent_id, plan.agent_id):
            return True

        # Agents in subscriber list can read
        if agent_id in plan.subscribers:
            return True

        return False

    def can_update_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can update this plan.

        Rules:
        - Agents can update their own plans
        - Parent agents can update child plans (for coordination)

        Args:
            agent_id: ID of agent attempting to update plan
            plan: ActionPlan to be updated

        Returns:
            True if agent can update plan
        """
        # Agents can update their own plans
        if plan.agent_id == agent_id:
            return True

        # Parent agents can update child plans
        if self._is_parent(agent_id, plan.agent_id):
            return True

        return False

    def can_approve_plan(self, agent_id: str, plan: ActionPlan) -> bool:
        """Check if agent can approve this plan.

        Rules:
        - Only parent agents can approve plans
        - System-scope plans require approval from system-level agents

        Args:
            agent_id: ID of agent attempting to approve plan
            plan: ActionPlan to be approved

        Returns:
            True if agent can approve plan
        """
        # Only parent can approve child's plan
        if self._is_parent(agent_id, plan.agent_id):
            # System-scope plans need higher-level approval
            if plan.scope == "system":
                # Check if approver is at system level (no parent)
                return agent_id not in self.agent_hierarchy
            return True

        return False

    def _is_parent(self, potential_parent: str, agent_id: str) -> bool:
        """Check if potential_parent is parent of agent_id.

        Args:
            potential_parent: Agent ID of potential parent
            agent_id: Agent ID to check

        Returns:
            True if potential_parent is direct parent of agent_id
        """
        return self.agent_hierarchy.get(agent_id) == potential_parent

    def _is_in_team(self, agent1: str, agent2: str) -> bool:
        """Check if two agents are in same team.

        Args:
            agent1: First agent ID
            agent2: Second agent ID

        Returns:
            True if agents are in same team
        """
        for team_members in self.team_structure.values():
            if agent1 in team_members and agent2 in team_members:
                return True
        return False

    def _is_in_hierarchy(self, agent1: str, agent2: str) -> bool:
        """Check if two agents are in same hierarchy chain.

        Args:
            agent1: First agent ID
            agent2: Second agent ID

        Returns:
            True if agents share a hierarchy chain
        """
        # Check if agent1 is ancestor of agent2
        current = agent2
        while current in self.agent_hierarchy:
            if self.agent_hierarchy[current] == agent1:
                return True
            current = self.agent_hierarchy[current]

        # Check if agent2 is ancestor of agent1
        current = agent1
        while current in self.agent_hierarchy:
            if self.agent_hierarchy[current] == agent2:
                return True
            current = self.agent_hierarchy[current]

        return False

