"""Abstract VCM Analysis Capability providing composable primitives.

This module provides the `VCMAnalysisCapability` abstract base class that offers
atomic, composable primitives for distributed VCM (Virtual Context Memory) analysis.

Design Philosophy:
- Does NOT prescribe a workflow
- Provides atomic operations that the LLM planner can compose into arbitrary
  distributed reasoning strategies
- The LLM decides IF/WHEN/HOW to batch, cluster, merge, revisit, etc.
- Cache-awareness is an EMERGENT property from LLM planner composition, not
  hardcoded into primitives

Primitive Categories:
- Worker Lifecycle: spawn_worker, spawn_workers, terminate_worker, get_idle_workers
- Work Assignment: assign_work, prioritize_work, get_pending_work
- Results: get_result, get_results, merge_results, detect_contradictions
- State Queries: get_analyzed_pages, get_unanalyzed_pages, get_pages_with_issues
- Iteration/Revisit: mark_for_revisit, revisit_page, clear_result

Domain-specific subclasses implement abstract hooks:
- get_worker_capability_class(): Worker capability type
- get_worker_agent_type(): Worker agent fully qualified name
- get_domain_merge_policy(): How to merge domain results

Example Usage by LLM Planner:
    # Strategy A: Cluster-Based Analysis
    1. Action: page_graph.get_clusters() → result_var: clusters
    2. Action: working_set.request_pages(Ref(clusters[0]))
    3. Action: vcm_analysis.spawn_workers(Ref(clusters[0]), cache_affine=True)
    4. [wait for completion events]
    5. Action: vcm_analysis.merge_results(Ref(clusters[0]))

    # Strategy B: Query-Driven Continuous Analysis
    1. Action: vcm_analysis.get_outstanding_queries() → queries
    2. Action: vcm_analysis.spawn_worker(top_query_target)
    3. [on completion]: Action: vcm_analysis.detect_contradictions(...)
    4. [loop based on discoveries]
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Type, TYPE_CHECKING

from ...base import AgentCapability
from ..actions.policies import action_executor
from ..scope import ScopeAwareResult, AnalysisScope

if TYPE_CHECKING:
    from ...base import Agent
    from .merge import MergePolicy

logger = logging.getLogger(__name__)


class VCMAnalysisCapability(AgentCapability, ABC):
    """Abstract capability providing composable primitives for VCM analysis.

    Does NOT prescribe a workflow. Provides atomic operations that the
    LLM planner can compose into arbitrary distributed reasoning strategies.

    Subclasses provide domain-specific:
    - Worker capability class and agent type
    - Merge policy
    - Domain-specific analysis parameters

    State Tracking:
    - Workers: Managed via AgentPoolCapability (obtained dynamically)
    - Results: Stored in blackboard for cluster-wide visibility
    - Revisit queue: Pages marked for re-analysis
    - Outstanding queries: Queries generated but not yet answered
    """

    # Blackboard key patterns
    RESULT_KEY = "vcm_analysis:{tenant_id}:{scope_id}:result:{page_id}"
    REVISIT_QUEUE_KEY = "vcm_analysis:{tenant_id}:{scope_id}:revisit_queue"
    OUTSTANDING_QUERIES_KEY = "vcm_analysis:{tenant_id}:{scope_id}:outstanding_queries"
    ANALYSIS_STATE_KEY = "vcm_analysis:{tenant_id}:{scope_id}:state"

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None,
    ):
        """Initialize VCM analysis capability.

        Args:
            agent: Owning agent (coordinator)
            scope_id: Blackboard scope (defaults to agent_id)
        """
        super().__init__(agent=agent, scope_id=scope_id or agent.agent_id)

        # Internal tracking (backed by blackboard for persistence)
        self._worker_ids: dict[str, str] = {}  # page_id -> worker_agent_id
        self._worker_pages: dict[str, str] = {}  # worker_agent_id -> page_id
        self._pending_work: list[dict[str, Any]] = []  # Work queue

    async def initialize(self) -> None:
        """Initialize capability and restore state from blackboard."""
        await super().initialize()

        # Restore state from blackboard if available
        blackboard = await self.get_blackboard()
        state_key = self._get_state_key()
        state = await blackboard.read(state_key)
        if state:
            self._worker_ids = state.get("worker_ids", {})
            self._worker_pages = state.get("worker_pages", {})
            self._pending_work = state.get("pending_work", [])
            logger.debug(f"VCMAnalysisCapability: restored state with {len(self._worker_ids)} workers")

    # =========================================================================
    # ABSTRACT HOOKS - Domain-specific subclasses implement
    # =========================================================================

    @abstractmethod
    def get_worker_capability_class(self) -> Type[AgentCapability]:
        """Return the capability class for worker agents.

        Returns:
            Type of capability that workers should have.
        """

    @abstractmethod
    def get_worker_agent_type(self) -> str:
        """Return the fully qualified type string for worker agents.

        Returns:
            Fully qualified class path for worker agent type.
            E.g., "polymathera.colony.samples.code_analysis.intent.IntentInferenceAgent"
        """

    @abstractmethod
    def get_domain_merge_policy(self) -> MergePolicy:
        """Return the merge policy for combining domain results.

        Returns:
            MergePolicy implementation appropriate for this domain.
        """

    def get_analysis_parameters(self, **kwargs) -> dict[str, Any]:
        """Return domain-specific parameters for worker analysis.

        Override in subclasses to provide domain-specific defaults.

        Args:
            **kwargs: Parameters passed from action call

        Returns:
            Dictionary of parameters to pass to workers.
        """
        return kwargs

    def post_process_result(self, result: ScopeAwareResult) -> ScopeAwareResult:
        """Optional hook for domain-specific post-processing.

        Override in subclasses if results need post-processing.

        Args:
            result: Raw result from worker

        Returns:
            Post-processed result.
        """
        return result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_result_key(self, page_id: str) -> str:
        """Get blackboard key for a page result."""
        return self.RESULT_KEY.format(
            tenant_id=self.agent.tenant_id,
            scope_id=self.scope_id,
            page_id=page_id,
        )

    def _get_revisit_queue_key(self) -> str:
        """Get blackboard key for revisit queue."""
        return self.REVISIT_QUEUE_KEY.format(
            tenant_id=self.agent.tenant_id,
            scope_id=self.scope_id,
        )

    def _get_outstanding_queries_key(self) -> str:
        """Get blackboard key for outstanding queries."""
        return self.OUTSTANDING_QUERIES_KEY.format(
            tenant_id=self.agent.tenant_id,
            scope_id=self.scope_id,
        )

    def _get_state_key(self) -> str:
        """Get blackboard key for capability state."""
        return self.ANALYSIS_STATE_KEY.format(
            tenant_id=self.agent.tenant_id,
            scope_id=self.scope_id,
        )

    async def _persist_state(self) -> None:
        """Persist internal state to blackboard."""
        blackboard = await self.get_blackboard()
        await blackboard.write(
            self._get_state_key(),
            {
                "worker_ids": self._worker_ids,
                "worker_pages": self._worker_pages,
                "pending_work": self._pending_work,
                "updated_at": time.time(),
            },
            created_by=self.agent.agent_id,
        )

    async def _get_agent_pool_cap(self):
        """Get AgentPoolCapability, creating if needed."""
        from .agent_pool import AgentPoolCapability

        pool_cap = self.agent.get_capability_by_type(AgentPoolCapability)
        if not pool_cap:
            pool_cap = AgentPoolCapability(agent=self.agent, scope_id=self.scope_id)
            await pool_cap.initialize()
            self.agent.add_capability(pool_cap)
        return pool_cap

    async def _get_result_cap(self):
        """Get ResultCapability, creating if needed."""
        from .result import ResultCapability

        result_cap = self.agent.get_capability_by_type(ResultCapability)
        if not result_cap:
            result_cap = ResultCapability(agent=self.agent, scope_id=self.scope_id)
            await result_cap.initialize()
            self.agent.add_capability(result_cap)
        return result_cap

    async def _get_merge_cap(self):
        """Get MergeCapability, creating if needed."""
        from .merge import MergeCapability

        merge_cap = self.agent.get_capability_by_type(MergeCapability)
        if not merge_cap:
            merge_cap = MergeCapability(agent=self.agent, scope_id=self.scope_id)
            merge_cap.set_policy(self.get_domain_merge_policy())
            await merge_cap.initialize()
            self.agent.add_capability(merge_cap)
        return merge_cap

    # =========================================================================
    # WORKER LIFECYCLE PRIMITIVES
    # =========================================================================

    @action_executor(action_key="spawn_worker")
    async def spawn_worker(
        self,
        page_id: str,
        cache_affine: bool = False,
        **domain_params,
    ) -> dict[str, Any]:
        """Spawn a single worker agent for a page.

        LLM decides when to spawn, for which page, with what affinity.

        Args:
            page_id: Page to analyze
            cache_affine: If True, route to replica with page cached
            **domain_params: Domain-specific analysis parameters

        Returns:
            Dict with:
            - worker_id: Created worker's ID
            - page_id: Page being analyzed
            - spawned: Whether spawn succeeded
        """
        pool_cap = await self._get_agent_pool_cap()

        # Prepare analysis parameters
        analysis_params = self.get_analysis_parameters(**domain_params)

        # Determine bound_pages for cache affinity
        bound_pages = [page_id] if cache_affine else None

        result = await pool_cap.create_agent(
            agent_type=self.get_worker_agent_type(),
            capabilities=[self.get_worker_capability_class().__name__],
            bound_pages=bound_pages,
            metadata={
                "page_id": page_id,
                "analysis_params": analysis_params,
                "coordinator_id": self.agent.agent_id,
                "scope_id": self.scope_id,
            },
            role=f"worker_{page_id}",
        )

        if result.get("created"):
            worker_id = result["agent_id"]
            self._worker_ids[page_id] = worker_id
            self._worker_pages[worker_id] = page_id
            await self._persist_state()

            logger.info(
                f"VCMAnalysisCapability: spawned worker {worker_id} for page {page_id} "
                f"(cache_affine={cache_affine})"
            )

            return {
                "worker_id": worker_id,
                "page_id": page_id,
                "spawned": True,
            }

        logger.error(f"VCMAnalysisCapability: failed to spawn worker for {page_id}: {result.get('error')}")
        return {
            "worker_id": None,
            "page_id": page_id,
            "spawned": False,
            "error": result.get("error"),
        }

    @action_executor(action_key="spawn_workers")
    async def spawn_workers(
        self,
        page_ids: list[str],
        cache_affine: bool = False,
        max_parallel: int = 0,
        **domain_params,
    ) -> dict[str, Any]:
        """Spawn multiple worker agents.

        LLM decides batch size, which pages, parallelism level.

        Args:
            page_ids: Pages to analyze
            cache_affine: If True, route workers to replicas with pages cached
            max_parallel: Maximum concurrent spawns (0 = no limit)
            **domain_params: Domain-specific analysis parameters

        Returns:
            Dict with:
            - spawned: List of successfully spawned {worker_id, page_id}
            - failed: List of failed page_ids
            - total_spawned: Count of spawned workers
        """
        spawned = []
        failed = []

        # Limit parallelism if requested
        pages_to_spawn = page_ids
        if max_parallel > 0:
            pages_to_spawn = page_ids[:max_parallel]

        for page_id in pages_to_spawn:
            result = await self.spawn_worker(
                page_id=page_id,
                cache_affine=cache_affine,
                **domain_params,
            )

            if result.get("spawned"):
                spawned.append({
                    "worker_id": result["worker_id"],
                    "page_id": page_id,
                })
            else:
                failed.append(page_id)

        logger.info(
            f"VCMAnalysisCapability: spawned {len(spawned)}/{len(pages_to_spawn)} workers "
            f"(cache_affine={cache_affine})"
        )

        return {
            "spawned": spawned,
            "failed": failed,
            "total_spawned": len(spawned),
            "remaining": page_ids[max_parallel:] if max_parallel > 0 else [],
        }

    @action_executor(action_key="get_idle_workers")
    async def get_idle_workers(self) -> dict[str, Any]:
        """Get workers available for new work.

        Returns:
            Dict with:
            - workers: List of idle worker IDs
            - count: Number of idle workers
        """
        pool_cap = await self._get_agent_pool_cap()
        result = await pool_cap.list_available_agents(filter_workload="idle")

        # Filter to our workers
        our_workers = [
            w for w in result.get("agents", [])
            if w["agent_id"] in self._worker_pages
        ]

        return {
            "workers": [w["agent_id"] for w in our_workers],
            "count": len(our_workers),
        }

    @action_executor(action_key="get_busy_workers")
    async def get_busy_workers(self) -> dict[str, Any]:
        """Get workers currently processing work.

        Returns:
            Dict with:
            - workers: List of {worker_id, page_id} for busy workers
            - count: Number of busy workers
        """
        pool_cap = await self._get_agent_pool_cap()
        result = await pool_cap.get_agent_status(agent_ids=list(self._worker_pages.keys()))

        busy = []
        for agent_info in result.get("agents", []):
            if agent_info.get("current_work"):
                worker_id = agent_info["agent_id"]
                busy.append({
                    "worker_id": worker_id,
                    "page_id": self._worker_pages.get(worker_id),
                    "current_work": agent_info.get("current_work"),
                })

        return {
            "workers": busy,
            "count": len(busy),
        }

    @action_executor(action_key="get_worker_status")
    async def get_worker_status(self, worker_id: str) -> dict[str, Any]:
        """Get status of a specific worker.

        Args:
            worker_id: Worker to query

        Returns:
            Dict with worker status info.
        """
        pool_cap = await self._get_agent_pool_cap()
        result = await pool_cap.get_agent_status(agent_ids=[worker_id])

        agents = result.get("agents", [])
        if agents:
            agent_info = agents[0]
            return {
                "worker_id": worker_id,
                "page_id": self._worker_pages.get(worker_id),
                "state": agent_info.get("state"),
                "current_work": agent_info.get("current_work"),
                "found": True,
            }

        return {
            "worker_id": worker_id,
            "found": False,
        }

    @action_executor(action_key="terminate_worker")
    async def terminate_worker(
        self,
        worker_id: str,
        collect_results: bool = True,
    ) -> dict[str, Any]:
        """Terminate a specific worker.

        Args:
            worker_id: Worker to terminate
            collect_results: Whether to collect results before terminating

        Returns:
            Dict with:
            - terminated: Whether termination succeeded
            - final_result: Collected result (if collect_results=True)
        """
        pool_cap = await self._get_agent_pool_cap()

        result = await pool_cap.terminate_agent(
            agent_id=worker_id,
            collect_results=collect_results,
        )

        if result.get("terminated"):
            # Remove from tracking
            page_id = self._worker_pages.pop(worker_id, None)
            if page_id:
                self._worker_ids.pop(page_id, None)
            await self._persist_state()

            logger.info(f"VCMAnalysisCapability: terminated worker {worker_id}")

        return {
            "terminated": result.get("terminated", False),
            "worker_id": worker_id,
            "page_id": self._worker_pages.get(worker_id),
            "final_results": result.get("final_results", []),
        }

    # =========================================================================
    # WORK ASSIGNMENT PRIMITIVES
    # =========================================================================

    @action_executor(action_key="assign_work")
    async def assign_work(
        self,
        worker_id: str,
        page_id: str,
        priority: int = 5,
        **params,
    ) -> dict[str, Any]:
        """Assign work to a specific worker.

        LLM controls work distribution strategy.

        Args:
            worker_id: Worker to assign to
            page_id: Page to analyze
            priority: Work priority (higher = more urgent)
            **params: Additional parameters for the work

        Returns:
            Dict with:
            - assigned: Whether work was assigned
        """
        pool_cap = await self._get_agent_pool_cap()

        work_unit = {
            "page_id": page_id,
            "analysis_params": self.get_analysis_parameters(**params),
            "priority": priority,
        }

        result = await pool_cap.assign_work(
            agent_id=worker_id,
            work_unit=work_unit,
            priority=priority,
        )

        if result.get("assigned"):
            # Track assignment
            self._worker_ids[page_id] = worker_id
            self._worker_pages[worker_id] = page_id
            await self._persist_state()

        return {
            "assigned": result.get("assigned", False),
            "worker_id": worker_id,
            "page_id": page_id,
        }

    @action_executor(action_key="prioritize_work")
    async def prioritize_work(
        self,
        page_ids: list[str],
        priority: int,
    ) -> dict[str, Any]:
        """Change priority of pending work items.

        LLM can reprioritize based on discoveries.

        Args:
            page_ids: Pages to reprioritize
            priority: New priority value

        Returns:
            Dict with:
            - updated: Number of items updated
        """
        updated = 0
        for item in self._pending_work:
            if item.get("page_id") in page_ids:
                item["priority"] = priority
                updated += 1

        # Re-sort pending work by priority (descending)
        self._pending_work.sort(key=lambda x: x.get("priority", 0), reverse=True)
        await self._persist_state()

        return {
            "updated": updated,
            "page_ids": page_ids,
            "new_priority": priority,
        }

    @action_executor(action_key="get_pending_work")
    async def get_pending_work(self) -> dict[str, Any]:
        """Get work items waiting to be assigned.

        Returns:
            Dict with:
            - items: List of pending work items
            - count: Number of pending items
        """
        return {
            "items": self._pending_work,
            "count": len(self._pending_work),
        }

    @action_executor(action_key="add_pending_work")
    async def add_pending_work(
        self,
        page_ids: list[str],
        priority: int = 5,
        **params,
    ) -> dict[str, Any]:
        """Add pages to pending work queue.

        Args:
            page_ids: Pages to add
            priority: Priority for these items
            **params: Additional parameters

        Returns:
            Dict with count of added items.
        """
        added = 0
        for page_id in page_ids:
            # Don't add duplicates
            if not any(item.get("page_id") == page_id for item in self._pending_work):
                self._pending_work.append({
                    "page_id": page_id,
                    "priority": priority,
                    "queued_at": time.time(),
                    "params": params,
                })
                added += 1

        # Sort by priority
        self._pending_work.sort(key=lambda x: x.get("priority", 0), reverse=True)
        await self._persist_state()

        return {
            "added": added,
            "total_pending": len(self._pending_work),
        }

    # =========================================================================
    # RESULT PRIMITIVES
    # =========================================================================

    @action_executor(action_key="get_result")
    async def get_result(self, page_id: str) -> dict[str, Any]:
        """Get result for a specific page.

        Args:
            page_id: Page to get result for

        Returns:
            Dict with result data or None if not found.
        """
        blackboard = await self.get_blackboard()
        key = self._get_result_key(page_id)
        result = await blackboard.read(key)

        if result:
            return {
                "page_id": page_id,
                "found": True,
                "result": result,
            }

        return {
            "page_id": page_id,
            "found": False,
            "result": None,
        }

    @action_executor(action_key="get_results")
    async def get_results(self, page_ids: list[str]) -> dict[str, Any]:
        """Get results for multiple pages.

        Args:
            page_ids: Pages to get results for

        Returns:
            Dict with results mapping.
        """
        blackboard = await self.get_blackboard()
        results = {}
        found = []
        not_found = []

        for page_id in page_ids:
            key = self._get_result_key(page_id)
            result = await blackboard.read(key)
            if result:
                results[page_id] = result
                found.append(page_id)
            else:
                not_found.append(page_id)

        return {
            "results": results,
            "found": found,
            "not_found": not_found,
            "total_found": len(found),
        }

    @action_executor(action_key="get_all_results")
    async def get_all_results(self) -> dict[str, Any]:
        """Get all results for analyzed pages.

        Returns:
            Dict with all results.
        """
        analyzed = await self.get_analyzed_pages()
        return await self.get_results(analyzed.get("pages", []))

    @action_executor(action_key="store_result")
    async def store_result(
        self,
        page_id: str,
        result: dict[str, Any] | ScopeAwareResult,
        source_agent: str | None = None,
    ) -> dict[str, Any]:
        """Store a result for a page.

        Args:
            page_id: Page the result is for
            result: Result data
            source_agent: Agent that produced the result

        Returns:
            Dict with storage confirmation.
        """
        blackboard = await self.get_blackboard()

        # Convert ScopeAwareResult to dict if needed
        if isinstance(result, ScopeAwareResult):
            result_data = result.to_blackboard_entry()
        else:
            result_data = result

        # Add metadata
        entry = {
            "page_id": page_id,
            "result": result_data,
            "source_agent": source_agent or self.agent.agent_id,
            "stored_at": time.time(),
            "scope_id": self.scope_id,
        }

        key = self._get_result_key(page_id)
        await blackboard.write(key, entry, created_by=self.agent.agent_id)

        # Also store via ResultCapability for cluster-wide access
        result_cap = await self._get_result_cap()
        await result_cap.store_partial(
            result_id=f"{self.scope_id}:{page_id}",
            result=result_data,
            source_agent=source_agent or self.agent.agent_id,
            source_pages=[page_id],
            result_type=self.__class__.__name__,
        )

        logger.debug(f"VCMAnalysisCapability: stored result for page {page_id}")

        return {
            "stored": True,
            "page_id": page_id,
        }

    @action_executor(action_key="merge_results")
    async def merge_results(
        self,
        page_ids: list[str],
        detect_conflicts: bool = True,
        resolve_conflicts: bool = False,
    ) -> dict[str, Any]:
        """Merge results from specified pages.

        LLM decides when to merge, which pages, conflict handling.

        Args:
            page_ids: Pages whose results to merge
            detect_conflicts: Whether to detect conflicts
            resolve_conflicts: Whether to attempt automatic resolution

        Returns:
            Dict with merged result.
        """
        # Get results
        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        if not results:
            return {
                "merged": None,
                "page_ids": page_ids,
                "error": "no_results_found",
            }

        # Convert to ScopeAwareResults for merge
        scope_results = []
        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            if isinstance(result_data, dict) and "content" in result_data:
                scope_results.append(ScopeAwareResult(
                    content=result_data.get("content"),
                    scope=AnalysisScope(**result_data.get("scope", {})) if result_data.get("scope") else AnalysisScope(),
                    producer_agent_id=entry.get("source_agent"),
                ))
            else:
                scope_results.append(ScopeAwareResult(
                    content=result_data,
                    scope=AnalysisScope(related_shards=[page_id]),
                    producer_agent_id=entry.get("source_agent"),
                ))

        # Use MergeCapability
        merge_cap = await self._get_merge_cap()
        from .merge import MergeContext

        context = MergeContext(
            prefer_higher_confidence=True,
            merge_reason=f"Merging results for {len(page_ids)} pages",
        )

        try:
            merged = await merge_cap.merge_results(scope_results, context)

            # Detect conflicts if requested
            conflicts = []
            if detect_conflicts and len(scope_results) > 1:
                result_cap = await self._get_result_cap()
                conflict_result = await result_cap.detect_contradictions(
                    [f"{self.scope_id}:{pid}" for pid in page_ids]
                )
                conflicts = conflict_result.get("contradictions", [])

            return {
                "merged": merged.content if hasattr(merged, 'content') else merged,
                "page_ids": page_ids,
                "merge_method": "merge_capability",
                "conflicts_detected": len(conflicts),
                "conflicts": conflicts if detect_conflicts else [],
            }

        except Exception as e:
            logger.error(f"VCMAnalysisCapability: merge failed: {e}")
            return {
                "merged": None,
                "page_ids": page_ids,
                "error": str(e),
            }

    @action_executor(action_key="detect_contradictions")
    async def detect_contradictions(
        self,
        page_ids: list[str],
    ) -> dict[str, Any]:
        """Detect contradictions between results.

        LLM can use this to decide revisits.

        Args:
            page_ids: Pages whose results to check

        Returns:
            Dict with contradictions found.
        """
        result_cap = await self._get_result_cap()
        result_ids = [f"{self.scope_id}:{pid}" for pid in page_ids]

        result = await result_cap.detect_contradictions(result_ids)

        return {
            "page_ids": page_ids,
            "contradictions": result.get("contradictions", []),
            "count": result.get("count", 0),
        }

    @action_executor(action_key="synthesize_results")
    async def synthesize_results(
        self,
        page_ids: list[str],
        synthesis_goal: str | None = None,
    ) -> dict[str, Any]:
        """Synthesize final output from results.

        Args:
            page_ids: Pages whose results to synthesize
            synthesis_goal: Goal for synthesis

        Returns:
            Dict with synthesis.
        """
        result_cap = await self._get_result_cap()
        result_ids = [f"{self.scope_id}:{pid}" for pid in page_ids]

        result = await result_cap.synthesize(
            result_ids=result_ids,
            synthesis_goal=synthesis_goal,
        )

        return {
            "synthesis": result.get("synthesis"),
            "page_ids": page_ids,
            "synthesis_method": result.get("synthesis_method"),
        }

    # =========================================================================
    # STATE QUERY PRIMITIVES
    # =========================================================================

    @action_executor(action_key="get_analyzed_pages")
    async def get_analyzed_pages(self) -> dict[str, Any]:
        """Get pages that have been analyzed.

        Returns:
            Dict with list of analyzed page IDs.
        """
        blackboard = await self.get_blackboard()

        # Find all result keys for our scope
        pattern = self.RESULT_KEY.format(
            tenant_id=self.agent.tenant_id,
            scope_id=self.scope_id,
            page_id="*",
        )

        # Get keys matching pattern
        analyzed = []
        result_cap = await self._get_result_cap()
        partials = await result_cap.get_partials(filter_type=self.__class__.__name__)

        for partial in partials.get("results", []):
            # Extract page_id from result_id
            result_id = partial.get("result_id", "")
            if ":" in result_id:
                parts = result_id.split(":")
                if len(parts) >= 2:
                    analyzed.append(parts[-1])  # page_id is last part

        return {
            "pages": analyzed,
            "count": len(analyzed),
        }

    @action_executor(action_key="get_unanalyzed_pages")
    async def get_unanalyzed_pages(
        self,
        from_set: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get pages not yet analyzed.

        Args:
            from_set: Set of pages to check (if None, uses all known pages)

        Returns:
            Dict with list of unanalyzed page IDs.
        """
        analyzed = await self.get_analyzed_pages()
        analyzed_set = set(analyzed.get("pages", []))

        if from_set is None:
            # Get all pages from working set or pending work
            from_set = list(self._worker_ids.keys())
            from_set.extend([item["page_id"] for item in self._pending_work])

        unanalyzed = [p for p in from_set if p not in analyzed_set]

        return {
            "pages": unanalyzed,
            "count": len(unanalyzed),
            "from_total": len(from_set),
        }

    @action_executor(action_key="get_pages_with_issues")
    async def get_pages_with_issues(
        self,
        issue_types: list[str] | None = None,
        min_severity: str = "low",
    ) -> dict[str, Any]:
        """Get pages with unresolved issues.

        LLM can query to decide what needs attention.

        Args:
            issue_types: Types of issues to look for
            min_severity: Minimum severity to include

        Returns:
            Dict with pages having issues.
        """
        results_data = await self.get_all_results()
        results = results_data.get("results", {})

        pages_with_issues = []
        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        min_sev = severity_order.get(min_severity, 1)

        for page_id, entry in results.items():
            result = entry.get("result", {})

            # Check scope for issues
            scope_data = result.get("scope", {})
            if not scope_data.get("is_complete", True):
                pages_with_issues.append({
                    "page_id": page_id,
                    "issue_type": "incomplete",
                    "severity": "medium",
                    "missing_context": scope_data.get("missing_context", []),
                })

            # Check confidence
            confidence = scope_data.get("confidence", 1.0)
            if confidence < 0.7:
                pages_with_issues.append({
                    "page_id": page_id,
                    "issue_type": "low_confidence",
                    "severity": "high" if confidence < 0.5 else "medium",
                    "confidence": confidence,
                })

        # Filter by issue types if specified
        if issue_types:
            pages_with_issues = [
                p for p in pages_with_issues
                if p.get("issue_type") in issue_types
            ]

        # Filter by severity
        pages_with_issues = [
            p for p in pages_with_issues
            if severity_order.get(p.get("severity", "low"), 1) >= min_sev
        ]

        return {
            "pages": pages_with_issues,
            "count": len(pages_with_issues),
        }

    @action_executor(action_key="get_outstanding_queries")
    async def get_outstanding_queries(self) -> dict[str, Any]:
        """Get queries that haven't been fully answered.

        For query-driven strategies where analysis follows queries.

        Returns:
            Dict with outstanding queries.
        """
        blackboard = await self.get_blackboard()
        key = self._get_outstanding_queries_key()
        queries = await blackboard.read(key) or {"queries": []}

        return {
            "queries": queries.get("queries", []),
            "count": len(queries.get("queries", [])),
        }

    @action_executor(action_key="add_outstanding_query")
    async def add_outstanding_query(
        self,
        query: str,
        target_pages: list[str],
        priority: int = 5,
    ) -> dict[str, Any]:
        """Add a query that needs to be answered.

        Args:
            query: The query text
            target_pages: Pages that might answer this query
            priority: Query priority

        Returns:
            Dict with confirmation.
        """
        blackboard = await self.get_blackboard()
        key = self._get_outstanding_queries_key()
        queries = await blackboard.read(key) or {"queries": []}

        queries["queries"].append({
            "query": query,
            "target_pages": target_pages,
            "priority": priority,
            "created_at": time.time(),
        })

        # Sort by priority
        queries["queries"].sort(key=lambda x: x.get("priority", 0), reverse=True)

        await blackboard.write(key, queries, created_by=self.agent.agent_id)

        return {
            "added": True,
            "query": query,
            "total_queries": len(queries["queries"]),
        }

    @action_executor(action_key="resolve_query")
    async def resolve_query(
        self,
        query: str,
        answer_page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mark a query as resolved.

        Args:
            query: The query to resolve
            answer_page_ids: Pages that answered the query

        Returns:
            Dict with confirmation.
        """
        blackboard = await self.get_blackboard()
        key = self._get_outstanding_queries_key()
        queries = await blackboard.read(key) or {"queries": []}

        # Remove matching query
        queries["queries"] = [
            q for q in queries["queries"]
            if q.get("query") != query
        ]

        await blackboard.write(key, queries, created_by=self.agent.agent_id)

        return {
            "resolved": True,
            "query": query,
            "answer_pages": answer_page_ids,
        }

    @action_executor(action_key="get_analysis_coverage")
    async def get_analysis_coverage(self) -> dict[str, Any]:
        """Get overall analysis progress/coverage.

        Returns:
            Dict with coverage statistics.
        """
        analyzed = await self.get_analyzed_pages()
        pending = await self.get_pending_work()
        issues = await self.get_pages_with_issues()

        analyzed_count = analyzed.get("count", 0)
        pending_count = pending.get("count", 0)
        total = analyzed_count + pending_count

        return {
            "analyzed": analyzed_count,
            "pending": pending_count,
            "total": total,
            "coverage_pct": (analyzed_count / total * 100) if total > 0 else 0,
            "pages_with_issues": issues.get("count", 0),
            "workers_active": len(self._worker_ids),
        }

    # =========================================================================
    # ITERATION/REVISIT PRIMITIVES
    # =========================================================================

    @action_executor(action_key="mark_for_revisit")
    async def mark_for_revisit(
        self,
        page_id: str,
        reason: str,
        priority: int = 5,
        new_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark a page for later revisit.

        LLM decides what needs revisiting based on findings.

        Args:
            page_id: Page to revisit
            reason: Why revisiting is needed
            priority: Revisit priority
            new_context: New context to use on revisit

        Returns:
            Dict with confirmation.
        """
        blackboard = await self.get_blackboard()
        key = self._get_revisit_queue_key()
        queue = await blackboard.read(key) or {"items": []}

        queue["items"].append({
            "page_id": page_id,
            "reason": reason,
            "priority": priority,
            "new_context": new_context,
            "marked_at": time.time(),
        })

        # Sort by priority
        queue["items"].sort(key=lambda x: x.get("priority", 0), reverse=True)

        await blackboard.write(key, queue, created_by=self.agent.agent_id)

        logger.debug(f"VCMAnalysisCapability: marked {page_id} for revisit (reason: {reason})")

        return {
            "marked": True,
            "page_id": page_id,
            "reason": reason,
        }

    @action_executor(action_key="get_pages_needing_revisit")
    async def get_pages_needing_revisit(self) -> dict[str, Any]:
        """Get pages marked for revisit.

        Returns:
            Dict with revisit queue.
        """
        blackboard = await self.get_blackboard()
        key = self._get_revisit_queue_key()
        queue = await blackboard.read(key) or {"items": []}

        return {
            "pages": queue.get("items", []),
            "count": len(queue.get("items", [])),
        }

    @action_executor(action_key="revisit_page")
    async def revisit_page(
        self,
        page_id: str,
        new_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-analyze a page with updated context.

        LLM decides when to revisit and with what context.

        Args:
            page_id: Page to re-analyze
            new_context: New context for re-analysis

        Returns:
            Dict with revisit initiation status.
        """
        # Clear existing result
        await self.clear_result(page_id)

        # Remove from revisit queue
        blackboard = await self.get_blackboard()
        queue_key = self._get_revisit_queue_key()
        queue = await blackboard.read(queue_key) or {"items": []}
        queue["items"] = [
            item for item in queue.get("items", [])
            if item.get("page_id") != page_id
        ]
        await blackboard.write(queue_key, queue, created_by=self.agent.agent_id)

        # Spawn new worker for revisit
        params = new_context or {}
        params["is_revisit"] = True

        result = await self.spawn_worker(
            page_id=page_id,
            cache_affine=True,  # Prefer cached page on revisit
            **params,
        )

        return {
            "revisit_started": result.get("spawned", False),
            "page_id": page_id,
            "worker_id": result.get("worker_id"),
            "new_context": new_context,
        }

    @action_executor(action_key="clear_result")
    async def clear_result(self, page_id: str) -> dict[str, Any]:
        """Clear a page's result to allow re-analysis.

        Args:
            page_id: Page to clear

        Returns:
            Dict with confirmation.
        """
        blackboard = await self.get_blackboard()
        key = self._get_result_key(page_id)

        try:
            await blackboard.delete(key)
            logger.debug(f"VCMAnalysisCapability: cleared result for {page_id}")
            return {
                "cleared": True,
                "page_id": page_id,
            }
        except Exception as e:
            return {
                "cleared": False,
                "page_id": page_id,
                "error": str(e),
            }

    @action_executor(action_key="reset_analysis")
    async def reset_analysis(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Reset analysis for specified pages (or all).

        Args:
            page_ids: Pages to reset (None = all)

        Returns:
            Dict with reset count.
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        cleared = 0
        for page_id in page_ids:
            result = await self.clear_result(page_id)
            if result.get("cleared"):
                cleared += 1

        logger.info(f"VCMAnalysisCapability: reset {cleared} page results")

        return {
            "reset": cleared,
            "page_ids": page_ids,
        }
