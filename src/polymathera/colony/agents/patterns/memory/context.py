"""Agent Context Engine - unified interface for memory operations.

The `AgentContextEngine` provides:
- Dynamic discovery of all `MemoryCapability` instances
- Cross-level querying (gather context from all memory levels)
- Coordinated consolidation (trigger all consolidations)
- Integration with `PlanningContext` for LLM planning
- Memory introspection (inspect, navigate, and reason about memory structure)

This is the primary interface for action policies to interact with memory.
The context engine discovers capabilities dynamically - it doesn't need
explicit references to STM, LTM, etc.

Example:
    ```python
    # Setup
    ctx_engine = AgentContextEngine(agent)
    await ctx_engine.initialize()
    agent.add_capability(ctx_engine)

    # In action policy
    context = await ctx_engine.gather_context(
        query=MemoryQuery(query="authentication flow", max_results=20)
    )

    # Add to planning context
    planning_ctx.recalled_memories = context
    ```
"""

from __future__ import annotations

import uuid
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal
from overrides import override

from ...base import AgentCapability, CapabilityResultFuture
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...models import AgentSuspensionState
from ...blackboard.types import BlackboardEntry, BlackboardEvent
from ..actions.policies import action_executor
from .types import (
    MemoryQuery,
    MemoryScopeInfo,
    MemoryMap,
    ScopeInspectionResult,
    MemoryStatistics,
    MemorySearchResult,
    MemoryValidationIssue,
)
from .capability import MemoryCapability

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class AgentContextEngine(AgentCapability):
    """Unified interface for agent memory operations.

    The context engine is the agent's gateway to its memory system. It:

    1. **Discovers** all `MemoryCapability` instances on the agent
    2. **Queries** across all memory capabilities to gather relevant context
    3. **Coordinates** memory maintenance and consolidation
    4. **Integrates** with `PlanningContext` for LLM-based action planning

    Unlike individual MemoryCapability instances, the context engine
    provides a holistic view across all memory levels. The LLM can ask
    for relevant context without knowing the memory architecture.

    Design principles:
    - Dynamic discovery: No hardcoded references to specific memory levels
    - Unified interface: One place to gather all relevant memories
    - Composable: Works with any memory hierarchy (2 levels or 10)
    """

    def __init__(
        self,
        agent: "Agent",
        scope: BlackboardScope = BlackboardScope.AGENT,
    ):
        """Initialize context engine.

        Args:
            agent: Agent that owns this capability
            scope: Scope of the context engine (defaults to AGENT)
        """
        super().__init__(
            agent=agent,
            scope_id=f"{get_scope_prefix(scope, agent)}:context:{uuid.uuid4()}"
        )

        # Discovered capabilities (populated during initialize)
        self._memory_capabilities: list[MemoryCapability] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the context engine.

        Discovers all memory capabilities on the agent and validates
        the memory system configuration. Call this after all
        MemoryCapability instances have been added to the agent.
        """
        if self._initialized:
            return

        # Discover memory capabilities
        self._memory_capabilities = []

        for name in self.agent.get_capability_names():
            capability = self.agent.get_capability(name)
            # Unified MemoryCapability (includes WorkingMemoryCapability)
            if isinstance(capability, MemoryCapability):
                self._memory_capabilities.append(capability)
                logger.debug(f"Discovered memory capability: {capability.scope_id}")

        # Validate memory system configuration
        issues = await self.validate_memory_system()
        error_count = sum(1 for i in issues if i.severity == "error")

        self._initialized = True
        scope_types = [self._get_scope_type(c.scope_id) for c in self._memory_capabilities]
        logger.info(
            f"AgentContextEngine initialized: "
            f"{len(self._memory_capabilities)} capabilities "
            f"({', '.join(scope_types)})"
            f"{f', {error_count} validation errors' if error_count else ''}"
        )

    async def refresh_capabilities(self) -> None:
        """Re-discover memory capabilities.

        Call this if memory capabilities are added after initialization.
        """
        self._initialized = False
        await self.initialize()

    # -------------------------------------------------------------------------
    # AgentCapability Abstract Methods
    # -------------------------------------------------------------------------

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for AgentContextEngine")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for AgentContextEngine")
        pass

    @override
    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent]
    ) -> None:
        """Stream memory events from all memory capabilities to the given queue.

        Aggregates events from all discovered memory capabilities.
        """
        for cap in self._memory_capabilities:
            await cap.stream_events_to_queue(event_queue)

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for context engine result.

        The context engine is a persistent service without a single result.
        """
        raise NotImplementedError(
            "AgentContextEngine is a persistent service without a single result. "
            "Use stream_events_to_queue() to monitor memory changes."
        )

    # -------------------------------------------------------------------------
    # LLM-Plannable Actions
    # -------------------------------------------------------------------------

    @action_executor(action_key="gather_context", planning_summary="Gather relevant memories across all memory scopes. Supports semantic search (query text), logical filters (tag_filter with all_of/any_of/none_of), or hybrid. Use list_all_tags first to discover available tags.")
    async def gather_context(
        self,
        query: MemoryQuery | str | None = None,
        scopes: list[str] | None = None,
    ) -> list[BlackboardEntry]:
        """Gather relevant context from memory scopes.

        This is the primary action for LLM-driven memory retrieval.
        The LLM can request relevant memories without knowing the
        underlying memory architecture.

        Args:
            query: Query string or MemoryQuery object for filtering/ranking memories.
                LLM planners typically pass a plain string which is auto-wrapped.
            scopes: Optional list of specific scope IDs to query.
                   If None, queries all scopes of all discovered memory capabilities.

        Returns:
            List of relevant BlackboardEntry objects from all queried scopes.
            Entries are interleaved and sorted by relevance/recency.
        """
        if isinstance(query, str):
            query = MemoryQuery(query=query)
        elif isinstance(query, dict):
            query = MemoryQuery(**query)
        query = query or MemoryQuery()
        all_entries: list[BlackboardEntry] = []

        # Determine which scopes of memory capabilities to query.
        # Fall back to all scopes if none of the requested scopes match
        # (LLM planners may guess scope names that don't exist).
        caps_to_query = self.get_capabilities_by_scopes(scopes)
        if not caps_to_query and scopes:
            logger.info(
                f"None of the requested scopes {scopes} exist, "
                f"falling back to all {len(self._memory_capabilities)} scopes"
            )
            caps_to_query = list(self._memory_capabilities)

        # Query unified MemoryCapability instances
        results = await asyncio.gather(*[cap.recall(query) for cap in caps_to_query], return_exceptions=True)
        for result, cap in zip(results, caps_to_query):
            if isinstance(result, Exception):
                logger.error(f"Error querying memory capability {cap.scope_id}: {result}")
            else:
                all_entries.extend(result)

        # Sort by recency (most recent first)
        all_entries.sort(key=lambda e: e.created_at, reverse=True)

        # Limit total results
        return all_entries[:query.max_results]

    @action_executor(action_key="ingest_pending", planning_summary="Process pending ingestion entries across memory scopes.")
    async def ingest_pending(
        self,
        scopes: list[str] | None = None,
    ) -> dict[str, int]:
        """Trigger ingestion of pending entries across all capabilities.

        This action allows the LLM to explicitly ingest pending memory
        entries from lower-level memory capabilities when appropriate
        as decided by the action policy (e.g., after receiving new data,
        before making decisions that need fresh context, etc.).

        Args:
            scopes: Optional list of specific scope IDs to ingest.
                   If None, triggers ingestion on all capabilities.

        Returns:
            Dict mapping scope IDs to number of entries ingested.
        """
        results: dict[str, int] = {}

        # Determine which capabilities to process
        caps_to_ingest = self.get_capabilities_by_scopes(scopes)

        # Trigger ingestion on each capability
        for cap in caps_to_ingest:
            try:
                count = await cap.ingest_now()
                results[cap.scope_id] = count
            except Exception as e:
                logger.error(f"Error ingesting {cap.scope_id}: {e}")
                results[cap.scope_id] = -1  # Error indicator

        total = sum(c for c in results.values() if c >= 0)
        logger.info(f"Ingestion complete: {total} total entries ingested")
        return results

    @action_executor(
        action_key="list_all_tags",
        planning_summary=(
            "List all tags across all memory scopes with counts. "
            "Use this to discover available tags for constructing "
            "tag-based queries with gather_context or recall."
        ),
    )
    async def list_all_tags(self) -> dict[str, dict[str, int]]:
        """List all tags across all memory scopes.

        Returns:
            Dict mapping scope_id to {tag: count} dicts.
        """
        result: dict[str, dict[str, int]] = {}
        for cap in self._memory_capabilities:
            try:
                tags = await cap.list_tags()
                if tags:
                    result[cap.scope_id] = tags
            except Exception as e:
                logger.error(f"Error listing tags for {cap.scope_id}: {e}")
        return result

    @action_executor(action_key="maintain_memories", planning_summary="Run maintenance (TTL, capacity, decay) on memory scopes.")
    async def maintain(
        self,
        scopes: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Trigger maintenance on all memory capabilities.

        This action allows the LLM to explicitly run maintenance when
        appropriate (e.g., during idle time, before heavy processing).

        Maintenance includes:
        - Decay: Reduce relevance of old memories
        - Pruning: Remove low-value memories
        - Deduplication: Merge similar memories

        Args:
            scopes: Optional list of specific scope IDs to maintain.
                   If None, maintains all capabilities.

        Returns:
            Dict mapping scope IDs to maintenance results.
        """
        results: dict[str, dict[str, int]] = {}

        # Determine which capabilities to maintain
        caps_to_maintain = self.get_capabilities_by_scopes(scopes)

        # Maintain each capability
        for cap in caps_to_maintain:
            try:
                # Unified MemoryCapability uses deduplicate action
                results[cap.scope_id] = {
                    "decayed": await cap.decay(),
                    "pruned": await cap.prune(),
                    "deduplicated": await cap.deduplicate(),
                }
            except Exception as e:
                logger.error(f"Error maintaining capability {cap.scope_id}: {e}")
                results[cap.scope_id] = {"error": -1}

        logger.info(f"Maintenance complete for {len(results)} capabilities")
        return results

    # -------------------------------------------------------------------------
    # Read-Only Properties
    # -------------------------------------------------------------------------

    @property
    def memory_capabilities(self) -> list[MemoryCapability]:
        """Get all discovered unified memory capabilities.

        Useful for introspection and debugging.
        """
        return list(self._memory_capabilities)

    def get_capability_by_scope(self, scope_id: str) -> MemoryCapability | None:
        """Get a specific memory capability by its scope ID.

        Args:
            scope_id: The scope ID of the capability

        Returns:
            The MemoryCapability or None if not found
        """
        for cap in self._memory_capabilities:
            if cap.scope_id == scope_id:
                return cap
        return None

    def get_capabilities_by_scopes(self, scopes: list[str] | None) -> list[MemoryCapability]:
        """Determine which memory capabilities to query for the given scopes.

        Args:
            scopes: List of scope IDs to query, or None to query all

        Returns:
            List of matching MemoryCapability instances
        """
        if scopes is None:
            return list(self._memory_capabilities)

        caps: list[MemoryCapability] = []
        for scope in scopes:
            cap = self.get_capability_by_scope(scope)
            if cap:
                caps.append(cap)
            else:
                logger.warning(f"Scope {scope} not found in any memory capability")
        return caps

    def get_repl_context(self) -> dict[str, Any] | None:
        """Get REPL context from REPLCapability if available.

        Returns:
            Dict with REPL variables, functions, and recent code,
            or None if no REPLCapability found
        """
        from ..actions.repl import REPLCapability

        for name in self.agent.get_capability_names():
            capability = self.agent.get_capability(name)
            if isinstance(capability, REPLCapability):
                return capability.export_for_context()

        return None

    @action_executor(action_key="gather_full_context", planning_summary="Gather full memory context with statistics and validation.")
    async def gather_full_context(
        self,
        query: MemoryQuery | str | None = None,
        scopes: list[str] | None = None,
        include_repl: bool = True,
    ) -> dict[str, Any]:
        """Gather comprehensive context including memories and REPL state.

        This method provides a unified view of agent context for planning:
        - Memory entries from specified scopes
        - REPL variables and code history (if REPLCapability exists)

        Args:
            query: Query string or MemoryQuery object for memory retrieval.
                LLM planners typically pass a plain string which is auto-wrapped.
            scopes: Optional list of specific scope IDs to query
            include_repl: Whether to include REPL context (default: True)

        Returns:
            Dict with:
                - memories: List of BlackboardEntry from memory capabilities
                - repl: Dict with REPL variables/functions/recent_code (or None)
        """
        # Gather memory context (gather_context handles str→MemoryQuery coercion)
        memories = await self.gather_context(query=query, scopes=scopes)

        result: dict[str, Any] = {
            "memories": memories,
        }

        # Add REPL context if requested and available
        if include_repl:
            result["repl"] = self.get_repl_context()

        return result

    # -------------------------------------------------------------------------
    # Memory Introspection Actions
    # -------------------------------------------------------------------------

    @action_executor(action_key="inspect_memory_map", planning_summary="Get overview of all memory scopes and their relationships.")
    async def inspect_memory_map(self) -> MemoryMap:
        """Get a complete map of the agent's memory layout.

        Provides an overview of all memory scopes, their relationships,
        and current state. Use this to understand:
        - What memory levels exist (working, STM, LTM, etc.)
        - How data flows between levels (subscriptions)
        - Current capacity and health of each level
        - What data might be relevant for your task

        Returns:
            MemoryMap with all scope information and dataflow edges.

        Example use cases:
        - Before complex reasoning: Check what memories are available
        - After failures: Understand if memory might have relevant context
        - Debugging: See where data is stored and how it flows
        """
        return await self._build_memory_map()

    @action_executor(action_key="inspect_scope", planning_summary="Inspect a specific memory scope's entries, stats, and health.")
    async def inspect_scope(
        self,
        scope_id: str,
        include_sample_entries: bool = False,
        sample_limit: int = 5,
    ) -> ScopeInspectionResult:
        """Get detailed information about a specific memory scope.

        Provides deeper insight into a single scope including:
        - Configuration (TTL, capacity limits, maintenance policies)
        - Current statistics (entry count, ages)
        - Optional sample entries to preview contents
        - Subscription and producer configurations

        Args:
            scope_id: The scope to inspect (e.g., 'agent:abc123:stm')
            include_sample_entries: Whether to include sample entries
            sample_limit: Maximum sample entries to return (1-20)

        Returns:
            ScopeInspectionResult with detailed scope information.

        Raises:
            ValueError: If scope_id is not found in any memory capability.

        Use when you need to understand:
        - What kind of data is in a specific scope
        - Whether a scope might have relevant information
        - The health and configuration of a scope
        """
        cap = self.get_capability_by_scope(scope_id)
        if cap is None:
            raise ValueError(f"Scope not found: {scope_id}")

        scope_info = await self._build_scope_info(cap)
        result = ScopeInspectionResult(
            scope_info=scope_info,
            retrieval_strategy=cap.retrieval.__class__.__name__,
            maintenance_policies=[p.__class__.__name__ for p in cap.maintenance_policies],
            lens_names=list(cap.lenses.keys()),
        )

        # Add sample entries if requested
        if include_sample_entries:
            sample_limit = max(1, min(sample_limit, 20))
            try:
                entries = await cap.recall(MemoryQuery(max_results=sample_limit))
                result.sample_entries = [
                    {
                        "key": e.key,
                        "tags": list(e.tags) if e.tags else [],
                        "created_at": e.created_at,
                        "value_type": type(e.value).__name__ if e.value else "None",
                        "value_preview": str(e.value)[:200] if e.value else "",
                    }
                    for e in entries
                ]
            except Exception as e:
                logger.warning(f"Failed to get sample entries for {scope_id}: {e}")

        return result

    @action_executor(action_key="search_memory", planning_summary="Search across memory scopes by text query with optional tag/recency filters.")
    async def search_memory(
        self,
        query: str,
        scopes: list[str] | None = None,
        tags: list[str] | None = None,
        max_results: int = 10,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for memories across all scopes using semantic similarity.

        More powerful than recall() because it:
        - Searches across ALL memory levels (not just one)
        - Returns results ranked by relevance
        - Can filter by tags
        - Includes source scope information

        Args:
            query: Natural language search query
            scopes: Optional list of scope IDs to search (None = all)
            tags: Filter to entries with these tags
            max_results: Maximum results to return (1-100)
            min_relevance: Minimum relevance score (0.0-1.0)

        Returns:
            List of dicts with entry info, scope, and relevance score.

        Use when you need to:
        - Find relevant context for a task across all memory
        - Locate where specific information is stored
        - Understand what you know about a topic
        """
        from .types import TagFilter
        max_results = max(1, min(max_results, 100))
        mem_query = MemoryQuery(
            query=query,
            tag_filter=TagFilter(all_of=set(tags)) if tags else TagFilter(),
            max_results=max_results,
            min_relevance=min_relevance,
        )

        caps_to_search = self.get_capabilities_by_scopes(scopes)
        all_results: list[MemorySearchResult] = []

        # Query all scopes in parallel
        tasks = [cap.recall(mem_query) for cap in caps_to_search]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entries, cap in zip(results, caps_to_search):
            if isinstance(entries, Exception):
                logger.warning(f"Error searching {cap.scope_id}: {entries}")
                continue
            for entry in entries:
                relevance = entry.metadata.get("relevance", 0.5) if entry.metadata else 0.5
                all_results.append(MemorySearchResult(
                    entry=entry,
                    scope_id=cap.scope_id,
                    relevance_score=relevance,
                ))

        # Sort by relevance (highest first), then recency
        all_results.sort(key=lambda r: (r.relevance_score, r.entry.created_at), reverse=True)
        all_results = all_results[:max_results]

        return [
            {
                "key": r.entry.key,
                "scope_id": r.scope_id,
                "scope_type": self._get_scope_type(r.scope_id),
                "relevance_score": r.relevance_score,
                "tags": list(r.entry.tags) if r.entry.tags else [],
                "created_at": r.entry.created_at,
                "value_type": type(r.entry.value).__name__ if r.entry.value else "None",
                "value_preview": str(r.entry.value)[:300] if r.entry.value else "",
            }
            for r in all_results
        ]

    @action_executor(action_key="get_memory_statistics", planning_summary="Get entry counts, token usage, and health stats for memory scopes.")
    async def get_memory_statistics(
        self,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get health and usage statistics for the memory system.

        Provides metrics about:
        - Capacity utilization per scope
        - Entry counts and ages
        - Pending operations
        - Maintenance policy status

        Args:
            scopes: Specific scopes to analyze (None = all)

        Returns:
            Dict with health and usage metrics per scope.

        Use for:
        - Monitoring memory health
        - Deciding when to trigger maintenance or ingestion
        - Understanding memory pressure
        """
        caps = self.get_capabilities_by_scopes(scopes)
        stats = MemoryStatistics()
        now = time.time()

        for cap in caps:
            try:
                count = await cap.storage.count()
                stats.total_entries += count

                pending = len(cap._pending_entries)
                stats.total_pending_ingestion += pending

                capacity_pct = (count / cap.max_entries * 100) if cap.max_entries else None

                stats.scope_stats[cap.scope_id] = {
                    "scope_type": self._get_scope_type(cap.scope_id),
                    "entry_count": count,
                    "max_entries": cap.max_entries,
                    "capacity_pct": round(capacity_pct, 1) if capacity_pct is not None else None,
                    "ttl_seconds": cap.ttl_seconds,
                    "pending_ingestion": pending,
                    "producer_count": len(cap.producers),
                    "maintenance_policy_count": len(cap.maintenance_policies),
                    "subscription_count": len(cap._ingestion_policy.subscriptions),
                }
            except Exception as e:
                logger.warning(f"Failed to get stats for {cap.scope_id}: {e}")
                stats.scope_stats[cap.scope_id] = {"error": str(e)}

        return {
            "total_entries": stats.total_entries,
            "total_pending_ingestion": stats.total_pending_ingestion,
            "scope_count": len(caps),
            "scopes": stats.scope_stats,
            "generated_at": now,
        }

    # -------------------------------------------------------------------------
    # Memory System Validation
    # -------------------------------------------------------------------------

    async def validate_memory_system(self) -> list[MemoryValidationIssue]:
        """Validate memory system configuration.

        Checks:
        1. All capability memory requirements are satisfied
        2. Dataflow graph has no circular subscriptions
        3. All subscribed scopes exist as capabilities
        4. Scope IDs follow naming conventions

        Returns:
            List of validation issues (empty if valid).
        """
        issues: list[MemoryValidationIssue] = []
        known_scopes = {cap.scope_id for cap in self._memory_capabilities}

        # 1. Check capability memory requirements
        for name in self.agent.get_capability_names():
            cap = self.agent.get_capability(name)
            if hasattr(cap, 'get_memory_requirements'):
                reqs = cap.get_memory_requirements()
                if reqs is not None:
                    for req_scope_type in reqs.required_scopes:
                        # Check if any scope matches the required type
                        found = any(
                            self._get_scope_type(s) == req_scope_type
                            for s in known_scopes
                        )
                        if not found:
                            issues.append(MemoryValidationIssue(
                                severity="warning",
                                message=(
                                    f"Capability {cap.__class__.__name__} requires "
                                    f"scope type '{req_scope_type}' but none found"
                                ),
                                capability=cap.__class__.__name__,
                            ))

        # 2. Check for cycles in subscription graph
        # Build adjacency: target_scope -> [source_scopes]
        edges: list[tuple[str, str]] = []
        for cap in self._memory_capabilities:
            for sub in cap._ingestion_policy.subscriptions:
                edges.append((sub.source_scope_id, cap.scope_id))

        if self._has_cycle(edges):
            issues.append(MemoryValidationIssue(
                severity="error",
                message="Circular subscription detected in memory dataflow graph",
            ))

        # 3. Check all subscriptions point to existing scopes
        for cap in self._memory_capabilities:
            for sub in cap._ingestion_policy.subscriptions:
                if sub.source_scope_id not in known_scopes:
                    issues.append(MemoryValidationIssue(
                        severity="warning",
                        scope=sub.source_scope_id,
                        message=(
                            f"Scope '{cap.scope_id}' subscribes to "
                            f"'{sub.source_scope_id}' which has no capability"
                        ),
                    ))

        if issues:
            for issue in issues:
                log_fn = logger.error if issue.severity == "error" else logger.warning
                log_fn(f"Memory validation [{issue.severity}]: {issue.message}")
        else:
            logger.info("Memory system validation passed: no issues found")

        return issues

    # -------------------------------------------------------------------------
    # Memory Architecture Guidance
    # -------------------------------------------------------------------------

    async def get_memory_architecture_guidance(self) -> str:
        """Generate memory architecture guidance for ActionPolicy prompts.

        Builds a MemoryMap and formats it as guidance text that describes
        the agent's memory system for LLM-based reasoning.

        Returns:
            Formatted guidance string for inclusion in planning prompts.
        """
        from .prompts import get_memory_architecture_guidance
        memory_map = await self._build_memory_map()
        return get_memory_architecture_guidance(memory_map)

    # -------------------------------------------------------------------------
    # Action Group Description
    # -------------------------------------------------------------------------

    def get_action_group_description(self) -> str:
        """Describe this capability's actions for the LLM planner.

        Used by ActionDispatcher to group memory introspection actions
        with a meaningful description.
        """
        scope_count = len(self._memory_capabilities)
        scope_types = [self._get_scope_type(c.scope_id) for c in self._memory_capabilities]
        return (
            f"Memory Introspection & Context Engine ({scope_count} scopes: "
            f"{', '.join(scope_types)}). "
            f"Use these actions to inspect, search, and reason about your memory system. "
            "Unified cross-scope interface for the entire memory system. "
            "gather_context queries across all scopes using semantic search (text query), "
            "logical filters (tag_filter with all_of/any_of/none_of), or hybrid, without needing to know the architecture. "
            "inspect_memory_map reveals scope structure, relationships, and dataflow. "
            "list_all_tags discovers available tags across all scopes — use before constructing tag queries. "
        )

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    async def _build_memory_map(self) -> MemoryMap:
        """Build a complete memory map from discovered capabilities.

        Returns:
            MemoryMap with all scope information and dataflow edges.
        """
        memory_map = MemoryMap(agent_id=self.agent.agent_id)

        # Build reverse subscription index: scope_id -> list of subscriber scope_ids
        reverse_subs: dict[str, list[str]] = {}
        for cap in self._memory_capabilities:
            for sub in cap._ingestion_policy.subscriptions:
                reverse_subs.setdefault(sub.source_scope_id, []).append(cap.scope_id)

        # Build scope info for each capability
        for cap in self._memory_capabilities:
            scope_info = await self._build_scope_info(cap, reverse_subs)
            memory_map.scopes[cap.scope_id] = scope_info
            memory_map.total_entries += scope_info.entry_count
            memory_map.total_pending_ingestion += scope_info.pending_ingestion_count

            # Add dataflow edges
            for sub in cap._ingestion_policy.subscriptions:
                memory_map.dataflow_edges.append((sub.source_scope_id, cap.scope_id))

        return memory_map

    async def _build_scope_info(
        self,
        cap: MemoryCapability,
        reverse_subs: dict[str, list[str]] | None = None,
    ) -> MemoryScopeInfo:
        """Build MemoryScopeInfo for a single capability.

        Args:
            cap: The memory capability to inspect
            reverse_subs: Pre-computed reverse subscription index

        Returns:
            MemoryScopeInfo with current state.
        """
        scope_type = self._get_scope_type(cap.scope_id)

        # Get entry count from storage
        try:
            entry_count = await cap.storage.count()
        except Exception:
            entry_count = 0

        # Get subscription sources
        subscribes_to = [sub.source_scope_id for sub in cap._ingestion_policy.subscriptions]

        # Get subscribers (who pulls from this scope)
        subscribers = []
        if reverse_subs is not None:
            subscribers = reverse_subs.get(cap.scope_id, [])
        else:
            for other_cap in self._memory_capabilities:
                for sub in other_cap._ingestion_policy.subscriptions:
                    if sub.source_scope_id == cap.scope_id:
                        subscribers.append(other_cap.scope_id)

        return MemoryScopeInfo(
            scope_id=cap.scope_id,
            scope_type=scope_type,
            purpose=self._get_scope_purpose(scope_type),
            ttl_seconds=cap.ttl_seconds,
            max_entries=cap.max_entries,
            entry_count=entry_count,
            subscribes_to=subscribes_to,
            subscribers=subscribers,
            producer_count=len(cap.producers),
            pending_ingestion_count=len(cap._pending_entries),
            maintenance_policy_count=len(cap.maintenance_policies),
        )

    @staticmethod
    def _get_scope_type(scope_id: str) -> str:
        """Extract human-readable scope type from a scope ID.

        Examples:
            "agent:abc123:working" -> "working"
            "agent:abc123:ltm:episodic" -> "ltm:episodic"
            "agent_type:coder:collective:semantic" -> "collective:semantic"
            "game:g1:state" -> "game:state"
        """
        from ...scopes import MemoryScope
        parsed = MemoryScope.parse_agent_scope(scope_id)
        if parsed:
            return parsed[1]  # level (e.g., "working", "stm", "ltm:episodic")

        parsed_collective = MemoryScope.parse_collective_scope(scope_id)
        if parsed_collective:
            return f"collective:{parsed_collective[1]}"

        # Fallback: use last segment(s) after first colon-separated part
        parts = scope_id.split(":")
        if len(parts) >= 3:
            return ":".join(parts[2:])
        return scope_id

    @staticmethod
    def _get_scope_purpose(scope_type: str) -> str:
        """Get human-readable purpose for a scope type."""
        purposes = {
            "sensory": "Raw observations with very short retention (seconds). Captures events before filtering.",
            "working": "Current task context. Stores actions, plans, and filtered observations for immediate use.",
            "stm": "Short-term memory. Consolidated recent experiences with moderate retention (minutes to hours).",
            "ltm:episodic": "Long-term episodic memory. Stores experiences, events, and outcomes for pattern recognition.",
            "ltm:semantic": "Long-term semantic memory. Distilled knowledge, facts, patterns, and learned concepts.",
            "ltm:procedural": "Long-term procedural memory. Skills, strategies, and reusable action patterns.",
            "collective:episodic": "Shared episodic memory across agents of the same type.",
            "collective:semantic": "Shared knowledge base across agents of the same type.",
            "collective:procedural": "Shared skills and strategies across agents of the same type.",
        }
        return purposes.get(scope_type, f"Memory scope of type '{scope_type}'.")

    @staticmethod
    def _has_cycle(edges: list[tuple[str, str]]) -> bool:
        """Check if a directed graph has cycles.

        Args:
            edges: List of (source, target) edges.

        Returns:
            True if cycle detected.
        """
        # Build adjacency list
        adj: dict[str, list[str]] = {}
        nodes: set[str] = set()
        for src, tgt in edges:
            adj.setdefault(src, []).append(tgt)
            nodes.add(src)
            nodes.add(tgt)

        # DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color[neighbor] == GRAY:
                    return True  # Back edge -> cycle
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        return any(dfs(n) for n in nodes if color[n] == WHITE)



