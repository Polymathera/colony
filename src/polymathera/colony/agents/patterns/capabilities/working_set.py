"""Working set capability for cluster-wide page management.

Provides @action_executor methods for managing the working set of VCM pages.
The working set is stored in the blackboard for cluster-wide visibility -
all agents in the same tenant share the same working set state.

WorkingSetCapability uses cluster-wide state via blackboard rather than
agent-local state (self.working_set: set[str])

Usage:
    # Add capability to agent
    working_set_cap = WorkingSetCapability(
        agent=self,
        eviction_policy=LRUEvictionPolicy(),
        working_set_size=50,
    )
    self.add_capability(working_set_cap)

    # ActionPolicy can now use these actions:
    # - get_working_set()
    # - request_pages(page_ids, priority)
    # - release_pages(page_ids)
    # - score_pages(page_ids, scorer)
    # - record_page_access(page_id, access_type)
    # - identify_eviction_candidates(count)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from overrides import override

import networkx as nx

from ...base import AgentCapability
from ...models import AgentSuspensionState, RunContext
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...cache_coordination import (
    CacheAwareCoordinationPolicy,
    PageScorer,
    SimplePageScorer,
    EdgePageScorer,
    CompositePageScorer,
)
from ..actions.policies import action_executor
from ...blackboard.protocol import WorkingSetStateProtocol

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


# === Eviction Policies ===

class PageEvictionPolicy(ABC):
    """Abstract base for page eviction strategies."""

    @abstractmethod
    async def score_for_eviction(
        self,
        page_id: str,
        page_status: dict[str, Any],
    ) -> float:
        """Score page for eviction (higher = more likely to evict).

        Args:
            page_id: Page to score
            page_status: Status dict with access_count, last_access, etc.

        Returns:
            Eviction score (higher = evict first)
        """
        pass


class LRUEvictionPolicy(PageEvictionPolicy):
    """Least Recently Used eviction policy."""

    async def score_for_eviction(
        self,
        page_id: str,
        page_status: dict[str, Any],
    ) -> float:
        """Score by inverse of last access time (older = higher score)."""
        last_access = page_status.get("last_access", 0)
        if last_access == 0:
            return float("inf")  # Never accessed, evict first
        return time.time() - last_access


class ReferenceCountEvictionPolicy(PageEvictionPolicy):
    """Reference count based eviction policy."""

    async def score_for_eviction(
        self,
        page_id: str,
        page_status: dict[str, Any],
    ) -> float:
        """Score by inverse of access count (fewer accesses = higher score)."""
        access_count = page_status.get("access_count", 0)
        if access_count == 0:
            return float("inf")
        return 1.0 / access_count


class CompositeEvictionPolicy(PageEvictionPolicy):
    """Composite eviction policy combining multiple strategies."""

    def __init__(self, policies: list[PageEvictionPolicy], weights: list[float]):
        """Initialize with policies and weights.

        Args:
            policies: List of eviction policies
            weights: Corresponding weights
        """
        self.policies = policies
        self.weights = weights

    async def score_for_eviction(
        self,
        page_id: str,
        page_status: dict[str, Any],
    ) -> float:
        """Combine scores from all policies."""
        total = 0.0
        for policy, weight in zip(self.policies, self.weights):
            score = await policy.score_for_eviction(page_id, page_status)
            total += score * weight
        return total


# === Working Set Capability ===

class WorkingSetCapability(AgentCapability):
    """Manages the CLUSTER-WIDE working set of loaded VCM pages.

    Stores working set state in blackboard for cluster-wide visibility.
    All agents in the same tenant share the same working set state since
    VCM pages are shared across sessions/runs.

    Key: "{colony_level_scope}:vcm:working_set" in blackboard

    This capability provides working-set manager functionality but adds:
    - Cluster-wide state via blackboard
    - @action_executor methods for ActionPolicy composition
    - Pluggable eviction policies

    The ActionPolicy decides when to load/evict pages. This capability
    provides the primitives without assuming batching, clustering, or
    any specific cache strategy.
    """

    protocols = [WorkingSetStateProtocol]
    input_patterns = [WorkingSetStateProtocol.state_pattern(namespace="working_set")]

    def __init__(
        self,
        agent: Agent,
        eviction_policy: PageEvictionPolicy | None = None,
        coordination_policy: CacheAwareCoordinationPolicy | None = None,
        working_set_size: int = 50,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "working_set",
    ):
        """Initialize working set capability.

        Args:
            agent: Owning agent
            eviction_policy: Policy for selecting pages to evict
            coordination_policy: Policy for working set selection (optional)
            working_set_size: Maximum pages in working set (job quota)
            scope: Blackboard scope (defaults to COLONY)
            namespace: Namespace for the working set (defaults to "working_set")
        """
        super().__init__(agent=agent, scope_id=f"{get_scope_prefix(scope, agent)}:{namespace}")
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self.coordination_policy = coordination_policy
        self.working_set_size: int = working_set_size

        # Local cache for performance (synced from blackboard)
        self._local_cache: dict[str, Any] | None = None
        self._cache_version: int = 0

    def get_action_group_description(self) -> str:
        return (
            f"Working Set — manages cluster-wide VCM page cache (max {self.working_set_size} pages). "
            "request_pages is the main cost driver (loads pages into KV cache). "
            "release_pages frees slots. score_pages ranks by relevance for eviction decisions. "
            "State is cluster-wide via blackboard — all agents in same tenant share the working set. "
            "record_page_access before processing for accurate eviction scoring."
        )

    def _get_working_set_key(self) -> str:
        """Get blackboard key for working set state."""
        return ScopeUtils.format_key(vcm="working_set")

    def _get_page_status_key(self) -> str:
        """Get blackboard key for page status."""
        return ScopeUtils.format_key(vcm="page_status")

    async def _read_cluster_state(self) -> dict[str, Any]:
        """Read working set state from blackboard (cluster-wide)."""
        blackboard = await self.get_blackboard()
        key = self._get_working_set_key()
        data = await blackboard.read(key)
        return data or {
            "pages": [],
            "page_status": {},
            "version": 0,
            "last_updated_by": None,
            "last_updated_at": 0,
        }

    async def _write_cluster_state(self, state: dict[str, Any]) -> None:
        """Write working set state to blackboard (broadcasts to all agents)."""
        blackboard = await self.get_blackboard()
        key = self._get_working_set_key()
        state["version"] = state.get("version", 0) + 1
        state["last_updated_by"] = self.agent.agent_id
        state["last_updated_at"] = time.time()
        await blackboard.write(key, state, created_by=self.agent.agent_id)
        self._local_cache = state
        self._cache_version = state["version"]

    # === Action Executors ===

    @action_executor()
    async def get_working_set(self) -> dict[str, Any]:
        """Get current cluster-wide working set.

        All agents in the tenant see the same working set since VCM pages
        are shared across sessions/runs.

        Returns:
            Dict with:
            - pages: List of page IDs currently in working set
            - size: Number of pages in working set
            - capacity: Maximum working set size (job quota)
            - version: State version for optimistic concurrency
        """
        state = await self._read_cluster_state()
        return {
            "pages": state.get("pages", []),
            "size": len(state.get("pages", [])),
            "capacity": self.working_set_size,
            "version": state.get("version", 0),
        }

    @action_executor()
    async def request_pages(
        self,
        page_ids: list[str],
        priority: int = 0,
    ) -> dict[str, Any]:
        """Request pages to be loaded into VCM KV cache.

        Updates cluster-wide working set state and triggers actual page
        loading via Agent.request_page() which delegates to VCM.

        Args:
            page_ids: Pages to load
            priority: Load priority (higher = sooner, passed to VCM)

        Returns:
            Dict with:
            - requested: Pages that were requested for loading
            - already_loaded: Pages already in working set
            - new_size: New working set size
        """
        state = await self._read_cluster_state()
        current_pages = set(state.get("pages", []))
        already_loaded = [p for p in page_ids if p in current_pages]
        to_load = [p for p in page_ids if p not in current_pages]

        # Request loading via Agent (delegates to VCM)
        # TODO: Batch requests or parallelize?
        for page_id in to_load:
            await self.agent.request_page(page_id, priority=priority)

        # Update cluster state
        page_status = state.get("page_status", {})
        for page_id in to_load:
            page_status[page_id] = {
                "access_count": 0,
                "last_access": time.time(),
                "loaded_at": time.time(),
                "loaded_by": self.agent.agent_id,
            }

        new_pages = list(current_pages | set(to_load))
        state["pages"] = new_pages
        state["page_status"] = page_status
        await self._write_cluster_state(state)

        logger.info(
            f"WorkingSetCapability: requested {len(to_load)} pages, "
            f"{len(already_loaded)} already loaded, new size={len(new_pages)}"
        )

        return {
            "requested": to_load,
            "already_loaded": already_loaded,
            "new_size": len(new_pages),
        }

    @action_executor()
    async def release_pages(
        self,
        page_ids: list[str] | None = None,
        release_unused: bool = False,
        unused_threshold_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Release pages from working set.

        Removed pages may be evicted from VCM based on VCM's own policy.

        Args:
            page_ids: Specific pages to release (None = none unless release_unused)
            release_unused: If True, release pages with no recent access
            unused_threshold_seconds: Time threshold for "unused" (default 5 min)

        Returns:
            Dict with:
            - released: Pages that were released
            - still_loaded: Pages still in working set
        """
        state = await self._read_cluster_state()
        current_pages = set(state.get("pages", []))
        page_status = state.get("page_status", {})

        to_release = set()

        # Add specific pages if provided
        if page_ids:
            to_release.update(p for p in page_ids if p in current_pages)

        # Add unused pages if requested
        if release_unused:
            now = time.time()
            for page_id in current_pages:
                status = page_status.get(page_id, {})
                last_access = status.get("last_access", 0)
                if now - last_access > unused_threshold_seconds:
                    to_release.add(page_id)

        # Update cluster state
        new_pages = current_pages - to_release
        for page_id in to_release:
            page_status.pop(page_id, None)

        state["pages"] = list(new_pages)
        state["page_status"] = page_status
        await self._write_cluster_state(state)

        logger.info(
            f"WorkingSetCapability: released {len(to_release)} pages, "
            f"{len(new_pages)} remaining"
        )

        return {
            "released": list(to_release),
            "still_loaded": list(new_pages),
        }

    @action_executor()
    async def get_page_status(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get status of pages in working set.

        Args:
            page_ids: Specific pages to query (None = all in working set)

        Returns:
            Dict with:
            - pages: List of page status dicts with page_id, access_count,
                     last_access, loaded_at, loaded_by
        """
        state = await self._read_cluster_state()
        current_pages = set(state.get("pages", []))
        page_status = state.get("page_status", {})

        if page_ids is None:
            page_ids = list(current_pages)

        result = []
        for page_id in page_ids:
            if page_id in current_pages:
                status = page_status.get(page_id, {})
                result.append({
                    "page_id": page_id,
                    "access_count": status.get("access_count", 0),
                    "last_access": status.get("last_access", 0),
                    "loaded_at": status.get("loaded_at", 0),
                    "loaded_by": status.get("loaded_by"),
                    "in_working_set": True,
                })
            else:
                result.append({
                    "page_id": page_id,
                    "in_working_set": False,
                })

        return {"pages": result}

    @action_executor()
    async def score_pages(
        self,
        page_ids: list[str],
        scorer: str = "simple",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score pages for prioritization using PageScorer hierarchy.

        Uses existing PageScorer implementations from cache_coordination.py.

        Args:
            page_ids: Pages to score
            scorer: Scoring strategy ("simple", "edge", "composite")
            context: Context for scoring (query_history, page_graph, completed_pages)

        Returns:
            Dict with:
            - scores: Dict mapping page_id -> score
            - sorted_pages: Page IDs sorted by score (highest first)
        """
        context = context or {}

        # Create appropriate scorer
        if scorer == "simple":
            query_history = context.get("query_history", [])
            page_scorer = SimplePageScorer(query_history)
        elif scorer == "edge":
            completed_pages = context.get("completed_pages", set())
            # Load from page storage
            page_graph = await self.agent.load_page_graph()
            page_scorer = EdgePageScorer(page_graph, completed_pages)
        elif scorer == "composite":
            query_history = context.get("query_history", [])
            completed_pages = context.get("completed_pages", set())
            page_graph = await self.agent.load_page_graph()
            page_scorer = CompositePageScorer(
                [SimplePageScorer(query_history), EdgePageScorer(page_graph, completed_pages)],
                [0.5, 0.5]
            )
        else:
            raise ValueError(f"Unknown scorer: {scorer}. Use 'simple', 'edge', or 'composite'.")

        # Score each page
        scores = {}
        for page_id in page_ids:
            scores[page_id] = await page_scorer.score_page(page_id)

        # Sort by score
        sorted_pages = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

        return {
            "scores": scores,
            "sorted_pages": sorted_pages,
        }

    @action_executor()
    async def record_page_access(
        self,
        page_id: str,
        access_type: str = "read",
    ) -> dict[str, Any]:
        """Record page access for working set tracking.

        Should be called when an agent accesses a page during inference.

        Args:
            page_id: Page that was accessed
            access_type: Type of access ("read", "write", "query")

        Returns:
            Dict with:
            - recorded: Whether access was recorded
            - total_accesses: Total access count for this page
        """
        state = await self._read_cluster_state()
        current_pages = set(state.get("pages", []))
        page_status = state.get("page_status", {})

        if page_id not in current_pages:
            return {"recorded": False, "total_accesses": 0, "reason": "not_in_working_set"}

        status = page_status.get(page_id, {})
        status["access_count"] = status.get("access_count", 0) + 1
        status["last_access"] = time.time()
        status["last_access_type"] = access_type
        status["last_accessed_by"] = self.agent.agent_id
        page_status[page_id] = status

        state["page_status"] = page_status
        await self._write_cluster_state(state)

        return {
            "recorded": True,
            "total_accesses": status["access_count"],
        }

    @action_executor()
    async def identify_eviction_candidates(
        self,
        count: int = 5,
    ) -> dict[str, Any]:
        """Identify pages that could be evicted using configured eviction policy.

        Args:
            count: Number of candidates to return

        Returns:
            Dict with:
            - candidates: List of dicts with page_id, eviction_score, last_access, access_count
        """
        state = await self._read_cluster_state()
        current_pages = set(state.get("pages", []))
        page_status = state.get("page_status", {})

        # Score each page for eviction
        scored = []
        for page_id in current_pages:
            status = page_status.get(page_id, {})
            score = await self.eviction_policy.score_for_eviction(page_id, status)
            scored.append({
                "page_id": page_id,
                "eviction_score": score,
                "last_access": status.get("last_access", 0),
                "access_count": status.get("access_count", 0),
            })

        # Sort by eviction score (highest = evict first)
        scored.sort(key=lambda x: x["eviction_score"], reverse=True)

        return {
            "candidates": scored[:count],
        }

    @action_executor()
    async def initialize_from_policy(
        self,
        available_pages: list[str] | None = None,
        run_context: RunContext | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Initialize cluster-wide working set using coordination policy.

        Args:
            available_pages: Available page IDs (loads from page storage if None)
            run_context: Analysis goal and context

        Returns:
            Dict with:
            - initialized: Whether initialization succeeded
            - pages_loaded: Number of pages loaded
        """
        if self.coordination_policy is None:
            from ...cache_coordination import create_default_cache_aware_coordination_policy
            self.coordination_policy = await create_default_cache_aware_coordination_policy(
                agent=self.agent
            )

        # Load page graph
        page_graph: nx.DiGraph | None = await self.agent.load_page_graph()

        # Get available pages if not provided
        if available_pages is None:
            available_pages = list(page_graph.nodes()) if page_graph else []

        # Create RunContext if needed
        if run_context is None:
            run_context = RunContext(analysis_goal="")
        elif isinstance(run_context, dict):
            run_context = RunContext(**run_context)

        # Use policy to select initial pages
        initial_pages = await self.coordination_policy.select_initial_working_set(
            page_graph=page_graph,
            available_pages=available_pages,
            working_set_size=self.working_set_size,
            run_context=run_context,
        )

        # Request the selected pages
        result = await self.request_pages(initial_pages, priority=10)

        logger.info(
            f"WorkingSetCapability: initialized from policy, "
            f"loaded {len(initial_pages)} pages"
        )

        return {
            "initialized": True,
            "pages_loaded": len(initial_pages),
            "pages": initial_pages,
        }

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize WorkingSetCapability-specific state.

        Overrides base implementation to add capability-specific state.
        Calls super() first to get base state, then adds subclass state.

        Returns:
            AgentSuspensionState with all agent state serialized
        """
        # Get base state from parent
        state = await super().serialize_suspension_state(state)

        state.custom_data["_working_set_capability"] = dict(
            eviction_policy = self.eviction_policy,
            coordination_policy = self.coordination_policy,
            working_set_size = self.working_set_size
        )

        return state

    @override
    async def deserialize_suspension_state(
        self,
        state: AgentSuspensionState
    ) -> None:
        """Restore WorkingSetCapability-specific state from suspension.

        Overrides base implementation to restore capability-specific state.
        Calls super() first to restore base state, then restores subclass state.

        Args:
            state: AgentSuspensionState to restore from
        """
        # Restore base state first
        await super().deserialize_suspension_state(state)

        self.eviction_policy = state.custom_data["_working_set_capability"].get("eviction_policy", LRUEvictionPolicy())
        self.coordination_policy = state.custom_data["_working_set_capability"].get("coordination_policy", None)
        self.working_set_size = state.custom_data["_working_set_capability"].get("working_set_size", 50)

        logger.info(
            f"Restored cache state: {len(state.working_set_pages)} pages in working set"
        )

