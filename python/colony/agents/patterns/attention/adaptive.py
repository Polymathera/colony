"""Adaptive query strategy that learns from past query results to improve future queries."""

from __future__ import annotations

import time
from typing import Any, TypeVar
import asyncio
from overrides import override

from ...blackboard.types import BlackboardEvent, KeyPatternFilter
from ..scope import ScopeAwareResult
from ...models import QueryContext, ActionPolicyIO
from ...base import Agent, AgentCapability
from .attention import PageQuery
from ..actions.policies import (
    CacheAwareActionPolicy,
    action_executor,
)


T = TypeVar("T")

#--------------------------------------------------------------------------
# This adaptive query strategy should not be an ActionPolicy.
# Instead, it should be an agent capability/component used as
# a QueryGenerator because it outlives individual action executions.
#--------------------------------------------------------------------------

async def create_adaptive_query_policy(
    agent: Agent,
    **kwargs,
) -> CacheAwareActionPolicy:
    """Create a cache-aware action policy with adaptive query generation.

    This is the recommended way to use `AdaptiveQueryGenerator` with
    `CacheAwareActionPolicy`.

    Args:
        `agent`: Agent using this policy
        `**kwargs`: Additional arguments passed to `CacheAwareActionPolicy`

    Returns:
        `CacheAwareActionPolicy` with query generator as action provider
    """
    if not agent.has_capability(AdaptiveQueryGenerator.get_capability_name()):
        capability = AdaptiveQueryGenerator(agent)
        await capability.initialize()
        agent.add_capability(capability)

    # query_generator = agent.get_capability(AdaptiveQueryGenerator.get_capability_name())

    policy = CacheAwareActionPolicy(
        agent=agent,
        action_providers=[], # [query_generator],
        io=ActionPolicyIO(
            inputs={"context": QueryContext, "queries": list},
            outputs={"analysis": ScopeAwareResult, "next_queries": list},
        ),
        **kwargs,
    )

    await policy.use_agent_capability_types([
        AdaptiveQueryGenerator
    ])
    await policy.initialize()

    return policy


class AdaptiveQueryGenerator(AgentCapability):
    """Adapts query strategy based on what has been learned.

    Tracks query history and learns which query patterns work best,
    then adapts strategy accordingly.

    This is an AgentCapability that provides action executors for:
    - Running queries and analyzing results
    - Generating next queries based on history
    """

    def __init__(self, agent: Agent):
        """Initialize strategy."""
        super().__init__(agent)
        self.query_history: list[dict[str, Any]] = []
        self.successful_patterns: list[dict[str, Any]] = []
        self.failed_patterns: list[dict[str, Any]] = []

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream capability-specific events to the given queue.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        # TODO: We can get either explicit adaptive query requests or we can snoop
        # on published analysis results to convert them into adaptive queries in the action policy.
        # TODO: Stream code analysis result events? Use `AnalysisResult.get_key_pattern()` when available.
        # TODO: Code analyzers even better separate their output results into different categories (e.g.,
        # tentative findings vs. confirmed findings, partial findings vs. rejected findings) so that
        # adaptive queries can focus on specific categories.
        # TODO: Make scope configurable because agents that send adaptive queries need not know the agent_id of the adaptive query agents (decoupling).
        blackboard = await self.agent.get_blackboard(scope="shared", scope_id=self.agent.agent_id)
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=AdapativeQueryRequest.get_key_pattern()
            )
        )

    # TODO: This method should be removed in favor of adding `writes=["query_history"]` to
    # the `@action_executor` decorators.
    def record_query_result(
        self,
        query: str,
        result_confidence: float,
        result_useful: bool
    ) -> None:
        """Record query result for learning.

        Args:
            query: Query that was executed
            result_confidence: Confidence of result
            result_useful: Whether result was useful
        """
        query_record = {
            "query": query,
            "confidence": result_confidence,
            "useful": result_useful,
            "timestamp": time.time()
        }

        self.query_history.append(query_record)

        if result_useful and result_confidence > 0.7:
            self.successful_patterns.append(query_record)
        elif not result_useful or result_confidence < 0.5:
            self.failed_patterns.append(query_record)

    @action_executor()
    async def run_queries_and_analyze(
        self,
        context: QueryContext,
        queries: list[PageQuery] = [],
    ) -> ScopeAwareResult[T]:
        """Run queries and analyze results adaptively.

        Args:
            queries: Queries to execute
            context: Additional context for analysis

        Returns:
            Analysis summary
        """
        # TODO: Execute queries via agent's query router and get results
        return None  # Placeholder

    @action_executor()
    async def generate_next_queries(
        self,
        context: QueryContext,
        current_analysis: ScopeAwareResult[T],
    ) -> list[PageQuery]:
        """Generate queries adapted to what we've learned.

        Args:
            current_analysis: Current analysis state
            context: Additional context

        Returns:
            List of adapted queries
        """
        analysis = self._analyze_query_history()

        # Adapt strategy based on success rate
        if analysis["success_rate"] > 0.7:
            # Continue with similar queries
            queries = await self._generate_similar_queries(
                analysis["successful_queries"],
                context
            )
        else:
            # Try different approach
            queries = await self._generate_different_queries(
                analysis["failed_queries"],
                context
            )

        return queries

    def _analyze_query_history(self) -> dict[str, Any]:
        """Analyze query history to identify patterns.

        Returns:
            Analysis summary
        """
        if not self.query_history:
            return {
                "success_rate": 0.0,
                "successful_queries": [],
                "failed_queries": []
            }

        successful = [q for q in self.query_history if q.get("useful", False)]
        failed = [q for q in self.query_history if not q.get("useful", True)]

        return {
            "success_rate": len(successful) / len(self.query_history),
            "successful_queries": successful,
            "failed_queries": failed
        }

    async def _generate_similar_queries(
        self,
        successful_queries: list[dict[str, Any]],
        context: QueryContext
    ) -> list[PageQuery]:
        """Generate queries similar to successful ones.

        Args:
            successful_queries: Previous successful queries
            context: Current context

        Returns:
            New queries
        """
        # TODO: Actually use LLM to generate similar queries
        return []

    async def _generate_different_queries(
        self,
        failed_queries: list[dict[str, Any]],
        context: QueryContext
    ) -> list[PageQuery]:
        """Generate queries different from failed ones.

        Args:
            failed_queries: Previous failed queries
            context: Current context

        Returns:
            New queries
        """
        # TODO: Actually use LLM to generate different approach
        return []

