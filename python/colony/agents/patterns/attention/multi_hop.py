"""Multi-hop search capability using query routing for transitive exploration.

Provides actions for multi-hop search that follows relationships
across pages to find relevant context. The planner decides hop count dynamically.

Use cases:
- Dependency chains: Find all transitive dependencies
- Data flow tracing: Follow data flow across pages
- Pattern discovery: Find pattern instances across codebase
- Deductive reasoning: Chain inferences across knowledge
"""

from __future__ import annotations

import asyncio
from typing import Any
from overrides import override
from pydantic import BaseModel, Field

from ...blackboard.types import BlackboardEvent, KeyPatternFilter
from ...models import QueryContext, ActionPolicyIO
from ...base import Agent, AgentCapability
from ..actions.policies import CacheAwareActionPolicy, action_executor
from .attention import PageQuery, AttentionScore
from .query_routing import PageQueryRoutingPolicy

class MultiHopSearchResult(BaseModel):
    """Result of multi-hop search."""

    relevant_pages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All relevant pages found with scores"
    )

    hops_completed: int = Field(
        default=0,
        description="Number of hops completed"
    )

    visited_pages: list[str] = Field(
        default_factory=list,
        description="All pages visited during search"
    )


# TODO: Implement bidirectional search capability.
# Useful for dependency analysis where you want both:
# - What this page depends on (backward)
# - What depends on this page (forward)
class MultiHopSearchCapability(AgentCapability):
    """Capability for multi-hop search.
    Routes queries across multiple hops to find relevant context.

    Provides actions that a planner can use to:
    1. Start with initial pages
    2. Route from current frontier (unvisited pages) to find relevant context.
    3. Expand frontier with results: Use top results as starting points for next hop.
    4. Get search summary: all discovered relevant pages sorted by score

    The planner decides when to do another hop or stop.

    This is especially useful for:
    - Following dependency chains
    - Tracing data flow
    - Discovering pattern instances across codebase
    - Deductive reasoning: Chain inferences across knowledge
    """

    def __init__(
        self,
        agent: Agent,
        base_router: PageQueryRoutingPolicy,  # Base query routing policy
        max_hops: int = 2
    ):
        """Initialize multi-hop search capability.

        Args:
            agent: Agent using this capability
            base_router: Base query routing policy (single-hop)
            max_hops: Maximum number of hops to explore
        """
        super().__init__(agent)
        self.base_router = base_router
        self.max_hops = max_hops
        self._visited: set[str] = set()
        self._relevant_pages: dict[str, AttentionScore] = {}
        self._frontier: list[str] = []
        self._hops: int = 0

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream capability-specific events to the given queue.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        # TODO: We can get either explicit multi-hop search requests or we can snoop
        # on published analysis results to convert them into multi-hop search requests in the action policy.
        # TODO: Stream code analysis result events? Use `AnalysisResult.get_key_pattern()` when available.
        # TODO: Code analyzers even better separate their output results into different categories (e.g.,
        # tentative findings vs. confirmed findings, partial findings vs. rejected findings) so that
        # multi-hop search requests can focus on specific categories.
        # TODO: Make scope configurable because agents that send multi-hop search requests need not know the agent_id of the multi-hop search agent (decoupling).
        blackboard = await self.agent.get_blackboard(scope="shared", scope_id=self.agent.agent_id)
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=MultiHopSearchRequest.get_key_pattern()
            )
        )

    @action_executor()
    async def initialize_search(
        self,
        query: PageQuery,
        initial_pages: list[str],
    ) -> dict[str, Any]:
        """Initialize search from starting pages.

        Args:
            query: Query to start search from
            initial_pages: Pages to start search from

        Returns:
            Initialization status
        """
        self._visited = set()
        self._relevant_pages = {}
        # Starting pages
        if initial_pages:
            self._frontier = list(initial_pages)
        elif query.source_page_ids:
            self._frontier = list(query.source_page_ids)
        else:
            self._frontier = []

        self._hops = 0
        return {"initialized": True, "frontier_size": len(self._frontier)}

    @action_executor(writes=["hop_results"])
    async def execute_hop(
        self,
        query: PageQuery,
        top_k: int = 5,
        context: QueryContext | None = None  # FIXME: Wrong type. Should be AttentionContext?
    ) -> dict[str, Any]:
        """Execute one hop from current frontier.

        Args:
            query: Query to route
            top_k: Number of top results to use for next frontier
            context: Additional routing context

        Returns:
            Results from this hop
        """
        if not self.base_router or not self._frontier:
            return {"hop_results": [], "new_pages_found": 0}

        # Filter to unvisited
        unvisited = [p for p in self._frontier if p not in self._visited]
        if not unvisited:
            return {"hop_results": [], "new_pages_found": 0}

        page_query = PageQuery(
            query_text=query,
            source_page_ids=unvisited,
        )
        context = context or QueryContext()

        attention_scores = await self.base_router.route_query(
            query=page_query,
            available_pages=unvisited,
            context=context,
        )

        # Process results
        new_pages = []
        for score in attention_scores:
            if score.page_id not in self._visited:
                self._relevant_pages[score.page_id] = score
                self._visited.add(score.page_id)
                new_pages.append(score.page_id)
            elif score.score > self._relevant_pages.get(score.page_id, AttentionScore(page_id=score.page_id, score=0)).score:
                self._relevant_pages[score.page_id] = score

        # Update frontier for next hop
        # Use top results as starting points for next hop
        top_results = sorted(attention_scores, key=lambda x: x.score, reverse=True)[:top_k]
        self._frontier = [s.page_id for s in top_results]
        self._hops += 1

        return {
            "hop_results": [s.model_dump() for s in attention_scores],
            "new_pages_found": len(new_pages),
            "hops_completed": self._hops,
        }

    @action_executor(writes=["search_result"])
    async def get_search_summary(self) -> dict[str, Any]:
        """Get current search summary.

        Returns:
            MultiHopSearchResult with all findings
        """
        sorted_pages = sorted(
            self._relevant_pages.values(),
            key=lambda x: x.score,
            reverse=True
        )

        result = MultiHopSearchResult(
            relevant_pages=[s.model_dump() for s in sorted_pages],
            hops_completed=self._hops,
            visited_pages=list(self._visited),
        )
        return {"search_result": result.model_dump()}

    @action_executor()
    async def has_more_frontier(self) -> dict[str, Any]:
        """Check if there are more pages to explore.

        Returns:
            Whether frontier has unvisited pages
        """
        unvisited = [p for p in self._frontier if p not in self._visited]
        return {
            "has_more": len(unvisited) > 0,
            "frontier_size": len(unvisited),
        }


async def create_multi_hop_search_policy(
    agent: Agent,
    base_router: PageQueryRoutingPolicy | None = None,
    **kwargs,
) -> CacheAwareActionPolicy:
    """Create policy for multi-hop search.

    The planner decides when to do another hop or stop based on
    results and goals.

    Args:
        agent: Agent using this policy
        base_router: Base routing policy for single hops
        **kwargs: Additional arguments for CacheAwareActionPolicy

    Returns:
        CacheAwareActionPolicy with multi-hop search capability
    """
    cap_name = MultiHopSearchCapability.get_capability_name()

    if not agent.has_capability(cap_name):
        capability = MultiHopSearchCapability(
            agent=agent,
            base_router=base_router,
        )
        await capability.initialize()
        agent.add_capability(capability)

    # capability = agent.get_capability(cap_name)

    policy = CacheAwareActionPolicy(
        agent=agent,
        action_providers=[], # [capability],
        io=ActionPolicyIO(
            inputs={"query": str, "initial_pages": list},
            outputs={"search_result": MultiHopSearchResult},
        ),
        **kwargs,
    )
    policy.use_agent_capabilities([cap_name])
    await policy.initialize()
    return policy


