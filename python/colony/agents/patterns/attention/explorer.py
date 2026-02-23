"""Generic query-driven exploration capability.

This module implements the query-driven discovery meta-pattern that
applies to any domain where agents need to explore by generating queries
from their findings.
It provides actions for iterative exploration by generating queries
from findings. The planner decides iteration flow dynamically.

This is the information-seeking pattern where agents identify gaps and
generate queries to fill them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generic, TypeVar
from overrides import override
from pydantic import BaseModel, Field

from ...blackboard.types import BlackboardEvent, KeyPatternFilter
from ...models import AgentSuspensionState, QueryContext, ActionPolicyIO
from ...base import Agent, AgentCapability
from ....utils import setup_logger
from ..actions.policies import CacheAwareActionPolicy, action_executor
from .incremental import PageQuery
from .query_routing import PageQueryRoutingPolicy
from .attention import QueryGenerator

logger = setup_logger(__name__)

FindingType = TypeVar('FindingType')


class ExplorationResult(BaseModel):
    """Result of query-driven exploration."""

    findings: list[Any] = Field(
        default_factory=list,
        description="All findings discovered"
    )

    explored_contexts: list[str] = Field(
        default_factory=list,
        description="Context IDs that were explored"
    )


    converged: bool = Field(
        default=False,
        description="Whether exploration converged (no new findings)"
    )

    queries_generated: int = Field(
        default=0,
        description="Total queries generated"
    )

class QueryDrivenExplorationCapability(AgentCapability):
    """Capability for query-driven exploration.

    Provides actions that a planner can use to:
    1. Generate queries from findings
    2. Route queries to contexts
    3. Analyze contexts for new findings

    This pattern applies to any domain where:
    - Initial findings reveal gaps in knowledge
    - Queries can be generated to fill gaps
    - Query results lead to more findings
    - Process repeats until satisfactory coverage

    Explores by:
    1. Analyzing initial context
    2. Generating queries from findings
    3. Routing queries to relevant content
    4. Analyzing query results
    5. Generating new queries (iterative)

    Generalizes to:
    - Research: Following citations
    - Investigation: Following leads
    - Learning: Asking questions
    - Medical diagnosis: Ordering tests

    The planner decides when to call each and when to stop.
    """

    def __init__(
        self,
        agent: Agent,
        query_generator: QueryGenerator,  # Generates queries from findings
        query_router: PageQueryRoutingPolicy,     # Routes queries to contexts
    ):
        """Initialize explorer.

        Args:
            agent: Agent using this capability
            query_generator: Component that generates queries from findings
            query_router: Component that routes queries to relevant contexts
        """
        super().__init__(agent)
        self.query_generator = query_generator
        self.query_router = query_router
        self._all_findings: list[Any] = []
        self._explored_contexts: set[str] = set()
        self._total_queries: int = 0
        self._iteration_history: list[dict[str, Any]] = []

    def get_action_group_description(self) -> str:
        return (
            "Query-Driven Exploration — iteratively explores by following knowledge gaps. "
            "Loop: generate queries from findings → route to contexts → analyze for new findings → repeat. "
            "Each context is analyzed at most once (tracked). Planner decides when to stop. "
            "Generalizes to: research (following citations), investigation (following leads), "
            "diagnosis (ordering tests). get_exploration_summary returns all findings sorted by relevance."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for QueryDrivenExplorationCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for QueryDrivenExplorationCapability")
        pass

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream capability-specific events to the given queue.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        # TODO: We can get either explicit exploration requests or we can snoop
        # on published analysis results to convert them into exploration requests in the action policy.
        # TODO: Stream code analysis result events? Use `AnalysisResult.get_key_pattern()` when available.
        # TODO: Code analyzers even better separate their output results into different categories (e.g.,
        # tentative findings vs. confirmed findings, partial findings vs. rejected findings) so that
        # exploration requests can focus on specific categories.
        # TODO: Make scope configurable because agents that send exploration requests need not know the agent_id of the exploration agent (decoupling).
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=ExplorationRequest.get_key_pattern()
            )
        )

    @action_executor(writes=["queries"])
    async def generate_queries_from_findings(
        self,
        findings: list[Any],
        query_context: QueryContext | None = None,
    ) -> dict[str, Any]:
        """Generate queries from current findings.

        Args:
            findings: Current findings to generate queries from. The initial findings are provided by the action policy caller and referenced by the planner.
            query_context: Additional exploration context. The initial query context is provided by the action policy caller and referenced by the planner.

        Returns:
            List of generated queries
        """
        if not self.query_generator:
            return {"queries": []}

        query_context = query_context or QueryContext()
        queries = await self.query_generator.generate_queries(
            context=query_context,
            findings=findings,
        )

        self._total_queries += len(queries)
        self._iteration_history.append({
            "iteration": len(self._iteration_history) + 1,
            "queries_generated": len(queries),
        })
        return {"queries": queries}  # This is equivalent to `return queries` when stored in PolicyREPL.

    @action_executor(writes=["new_contexts"])
    async def route_queries_to_contexts(
        self,
        queries: list[PageQuery],
    ) -> dict[str, Any]:
        """Route queries to find relevant contexts.

        Args:
            queries: Queries to route

        Returns:
            List of new context IDs
        """
        if not self.query_router:
            return {"new_contexts": []} # This is equivalent to `return []` when stored in PolicyREPL.

        # TODO: Parallelize this if needed
        new_contexts: list[str] = []
        for query in queries:
            attention_scores = await self.query_router.route_query(query)
            for score in attention_scores:
                if score.page_id not in self._explored_contexts:
                    new_contexts.append(score.page_id)

        self._iteration_history[-1]["contexts_found"] = len(new_contexts)

        return {"new_contexts": list(set(new_contexts))} # This is equivalent to `return new_contexts` when stored in PolicyREPL.

    @action_executor(writes=["new_findings"])
    async def analyze_contexts(
        self,
        context_ids: list[str],
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze contexts to extract findings.

        Args:
            context_ids: Context IDs to analyze
            analysis_context: Additional context for analysis

        Returns:
            New findings from contexts
        """
        # Mark as explored
        self._explored_contexts.update(context_ids)

        # TODO: Implement actual context analysis
        new_findings: list[Any] = []

        self._all_findings.extend(new_findings)
        self._iteration_history[-1]["findings_added"] = len(new_findings)
        return {"new_findings": new_findings}

    @action_executor(writes=["exploration_result"])
    async def get_exploration_summary(self) -> dict[str, Any]:
        """Get current exploration summary.

        Returns:
            ExplorationResult with all findings and stats
        """
        result = ExplorationResult(
            findings=self._all_findings,
            explored_contexts=list(self._explored_contexts),
            queries_generated=self._total_queries,
            converged=False,  # TODO: Planner determines this
            iterations=len(self._iteration_history) + 1,
            final_confidence=0.0,  # TODO: Planner determines this
            iteration_history=self._iteration_history
        )
        return {"exploration_result": result.model_dump()}


async def create_exploration_policy(
    agent: Agent,
    query_generator: QueryGenerator | None = None,
    query_router: PageQueryRoutingPolicy | None = None,
    **kwargs,
) -> CacheAwareActionPolicy:
    """Create policy for query-driven exploration.

    The planner decides when to generate queries, route them,
    analyze contexts, or stop exploration.

    Args:
        agent: Agent using this policy
        query_generator: Component to generate queries from findings
        query_router: Component to route queries to contexts
        **kwargs: Additional arguments for CacheAwareActionPolicy

    Returns:
        CacheAwareActionPolicy with exploration capability
    """
    policy = CacheAwareActionPolicy(
        agent=agent,
        action_providers=[], # [capability],
        io=ActionPolicyIO(
            inputs={"initial_findings": list, "context": dict},
            outputs={"exploration_result": ExplorationResult},
        ),
        **kwargs,
    )

    await policy.use_agent_capability_types([
        QueryDrivenExplorationCapability
    ])
    await policy.initialize()


    return policy


