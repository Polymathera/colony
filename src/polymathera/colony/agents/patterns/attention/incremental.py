"""Incremental query processing with feedback loops as an agent capability.

This module implements incremental query processing that:
- Starts with initial pages
- Gets answer with current context
- Checks if answer is satisfactory
- Routes to additional pages if needed
- Repeats until planner decides to stop (TODO: Limit iterations: confidence threshold met or max iterations reached)

This reduces unnecessary page loads by stopping when answer is good enough.
The planner decides termination dynamically, not a fixed iteration count.
"""

from __future__ import annotations

import asyncio
from typing import Any
from overrides import override
from pydantic import BaseModel, Field

from ..scope import ScopeAwareResult, AnalysisScope
from .attention import PageQuery, AttentionScore
from .query_routing import PageQueryRoutingPolicy
from ...models import AgentSuspensionState, QueryContext, ActionPolicyIO
from ..models import QueryAnswer
from ...base import Agent, AgentCapability
from ...blackboard.protocol import IncrementalQueryProtocol
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions.policies import CacheAwareActionPolicy, action_executor
from ....utils import setup_logger

logger = setup_logger(__name__)


class IncrementalQueryCapability(AgentCapability):
    """Capability for incremental query processing.

    Instead of loading all potentially relevant pages upfront,
    processes queries incrementally. Provides actions that a planner can use to:
    1. Get answer from current pages
    2. Route to additional pages
    3. Refine queries

    Strategy (planner-driven):
    1. Start with initial pages
    2. Get partial answer from current pages
    3. Check if answer is satisfactory (high confidence, no additional pages needed)
    4. If not satisfactory:
       - Route to additional pages if requested
       - Or refine query if low confidence
    5. Planner decides when to stop (TODO: Iterate until satisfactory or max iterations reached)

    This is more efficient than loading all potentially relevant pages upfront.

    Example:
        ```python
        policy = create_incremental_query_policy(
            agent=agent,
            answer_generator=my_answer_gen,
            query_router=my_router,
        )

        state = ActionPolicyExecutionState()
        state.scope.set("query", query)
        state.scope.set("initial_pages", ["page_001", "page_042"])

        while not result.policy_completed:
            result = await policy.execute_iteration(state)

        answer = state.scope.get("answer")
        # answer.confidence might be 0.85
        # answer.pages_used might be ["page_001", "page_042", "page_089"]
        ```
    The planner decides when to call each action and when to stop.
    """

    input_patterns = [IncrementalQueryProtocol.request_pattern(namespace="incremental_query")]

    def __init__(
        self,
        agent: Agent,
        answer_generator: Any = None,  # Component that generates answers from pages
        query_router: PageQueryRoutingPolicy | None = None,
        answer_evaluator: Any | None = None,
        confidence_threshold: float = 0.8,
        attention_threshold: float = 0.5,
        attention_top_k: int = 5,
        max_iterations: int = 3,
        scope: BlackboardScope = BlackboardScope.COLONY
    ):
        super().__init__(agent, scope_id=get_scope_prefix(scope, agent))
        self.answer_generator = answer_generator
        self.query_router = query_router
        self.confidence_threshold = confidence_threshold
        self.attention_threshold = attention_threshold
        self.attention_top_k = attention_top_k
        self._pages_used: set[str] = set()

    def get_action_group_description(self) -> str:
        return (
            "Incremental Query — refines answers by progressively loading more pages. "
            "Loop: get_answer → check confidence → if low, route_for_pages → refine_query → repeat. "
            f"Stops when confidence >= {self.confidence_threshold} or max_iterations ({self.max_iterations}) reached. "
            "Each page processed at most once. route_for_pages filters by attention threshold and top-k."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for IncrementalQueryCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for IncrementalQueryCapability")
        pass

    @action_executor(writes=["answer"])
    async def get_answer(
        self,
        query: PageQuery,
        pages: list[str],  # On first call, the planner should map this input_schema to the initial_pages input
    ) -> QueryAnswer:
        """Get answer from specified pages.

        Args:
            query: Query to answer
            pages: Page IDs to use for answering (on first call, the planner should map this input_schema to the initial_pages input)
            context: Context for the query

        Returns:
            QueryAnswer with confidence and scope
        """
        self._pages_used.update(pages)

        # TODO: Use answer_generator to get actual answer
        answer = QueryAnswer(
            answer="Placeholder - implement with answer_generator",
            confidence=0.5,
            additional_pages_needed=[],
            pages_used=list(self._pages_used),
            scope=AnalysisScope(is_complete=False, confidence=0.5),
        )

        # In case of one element in `writes=["answer"]`, this is equivalent to `return answer.model_dump()`
        # Returning a dict is only required if there are multiple elements in `writes=[...]` in the decorator.
        return {"answer": answer.model_dump()}

    # This `additional_pages` variable written in the scope can then be referenced by the planner
    @action_executor(writes=["additional_pages"])
    async def route_for_pages(
        self,
        query: PageQuery,  # Original query or refined query created by the planner
        available_pages: list[str] | None = None,  # Will be mapped to the `available_pages` input by the planner
    ) -> list[str]:
        """Route to find additional relevant pages.

        Args:
            query: Original query or refined query containing descriptions of pages needed (from answer)

        Returns:
            List of new page IDs
        """
        if not self.query_router:
            # In case of one element in `writes=["additional_pages"]`,
            # `return {"additional_pages": []}` is equivalent to `return []`
            # Returning a dict is only required if there are multiple elements
            # in `writes=[...]` in the decorator.
            return []

        query.source_page_ids = list(self._pages_used)

        attention_scores: list[AttentionScore] = await self.query_router.route_query(
            query=query,
            available_pages=available_pages,
            context=None,
        )

        # Filter by threshold and already used
        new_pages = [
            s.page_id for s in attention_scores
            if s.score > self.attention_threshold
            and s.page_id not in self._pages_used
        ][:self.attention_top_k]

        # In case of one element in `writes=["additional_pages"]`, this is equivalent to `{"additional_pages": new_pages}`
        # Returning a dict is only required if there are multiple elements in `writes=[...]` in the decorator.
        return new_pages

    @action_executor(writes=["refined_query"])
    async def refine_query(
        self,
        original_query: str,
        low_confidence_answer: QueryAnswer,  # Will be mapped to `answer` by the planner
    ) -> PageQuery:
        """Refine query based on low-confidence answer.

        Args:
            original_query: Original query or a previously refined query
            low_confidence_answer: Previous answer with low confidence

        Returns:
            Refined query
        """
        # TODO: Use LLM to refine query based on what was unclear
        refined = f"{original_query} (more specific)"
        return {"refined_query": refined}


async def create_incremental_query_policy(
    agent: Agent,
    answer_generator: Any = None,
    query_router: PageQueryRoutingPolicy | None = None,
    **kwargs,
) -> CacheAwareActionPolicy:
    """Create policy for incremental query processing.

    The planner decides when to get answers, route for more pages,
    or refine queries. No hardcoded iteration logic.

    Args:
        agent: Agent using this policy
        answer_generator: Component to generate answers from pages
        query_router: Component to route queries to pages
        **kwargs: Additional arguments for CacheAwareActionPolicy

    Returns:
        CacheAwareActionPolicy with incremental query capability
    """
    cap_name = IncrementalQueryCapability.get_capability_name()

    if not agent.has_capability(cap_name):
        capability = IncrementalQueryCapability(
            agent=agent,
            answer_generator=answer_generator,
            query_router=query_router,
        )
        await capability.initialize()
        agent.add_capability(capability)

    # capability = agent.get_capability(cap_name)

    policy = CacheAwareActionPolicy(
        agent=agent,
        action_providers=[], # [capability],
        io=ActionPolicyIO(
            inputs={
                "query": PageQuery,  # str or PageQuery?
                "initial_pages": list[str],
                "available_pages": list[str] | None,
            },
            outputs={"answer": QueryAnswer},
        ),
        **kwargs,
    )
    policy.use_agent_capabilities([cap_name])
    await policy.initialize()
    return policy


