"""Query attention capability for query generation and routing.

Provides @action_executor methods for generating queries and routing them
to relevant pages. Wraps PageQueryRoutingPolicy and QueryGenerator.

This capability provides the core query primitives that ActionPolicies can
compose. It does NOT replace existing capabilities like IncrementalQueryCapability
or QueryDrivenExplorationCapability - those provide higher-level patterns that
can be used alongside or on top of this capability.

Capabilities can call each other via agent.get_capability().

Usage:
    # Add capability to agent
    query_cap = QueryAttentionCapability(
        agent=self,
        query_generator=my_generator,
        routing_policy=my_router,
    )
    self.add_capability(query_cap)

    # ActionPolicy can now use these actions:
    # - generate_queries(context, max_queries)
    # - route_query(query, available_pages, max_results)
    # - score_relevance(query, page_ids)
    # - get_answer(query, page_ids)
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from overrides import override

from ...base import AgentCapability
from ..attention.attention import (
    PageQuery,
    AttentionScore,
    QueryGenerator,
    AttentionScoringMechanism,
)
from ..attention.query_routing import PageQueryRoutingPolicy
from ...models import AttentionContext, AgentSuspensionState
from ..actions.policies import action_executor

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class QueryAttentionCapability(AgentCapability):
    """Core primitives for query generation and routing.

    Wraps PageQueryRoutingPolicy and QueryGenerator as @action_executors.
    Does NOT replace IncrementalQueryCapability or QueryDrivenExplorationCapability -
    those provide higher-level patterns that can use this capability's primitives.

    Capabilities can call each other via agent.get_capability():
        incremental = self.agent.get_capability(IncrementalQueryCapability)
        await incremental.get_answer(query, pages)

    Primitives:
    - generate_queries: Generate queries from analysis context
    - route_query: Route query to find relevant pages
    - score_relevance: Score relevance of query to specific pages
    """

    def __init__(
        self,
        agent: Agent,
        query_generator: QueryGenerator | None = None,
        routing_policy: PageQueryRoutingPolicy | None = None,
        attention_mechanism: AttentionScoringMechanism | None = None,
        scope_id: str | None = None,
    ):
        """Initialize query attention capability.

        Args:
            agent: Owning agent
            query_generator: Generator for creating queries from context
            routing_policy: Policy for routing queries to pages
            attention_mechanism: Mechanism for scoring query-key attention
            scope_id: Blackboard scope (defaults to agent_id)
        """
        super().__init__(agent=agent, scope_id=scope_id)
        self.query_generator = query_generator
        self.routing_policy = routing_policy
        self.attention_mechanism = attention_mechanism

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for QueryAttentionCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for QueryAttentionCapability")
        pass

    # === Action Executors ===

    @action_executor()
    async def generate_queries(
        self,
        findings: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        max_queries: int = 5,
    ) -> dict[str, Any]:
        """Generate queries from analysis context and findings.

        Use when you need to find additional relevant pages based on
        current analysis findings.

        Wraps: QueryGenerator.generate_queries()

        Args:
            findings: Current analysis findings to generate queries from
            context: Additional context (goal, constraints, etc.)
            max_queries: Maximum number of queries to generate

        Returns:
            Dict with:
            - queries: List of PageQuery objects
            - generated_count: Number of queries generated
        """
        if self.query_generator is None:
            logger.warning("QueryAttentionCapability: no query_generator configured")
            return {"queries": [], "generated_count": 0, "error": "no_query_generator"}

        # Build AttentionContext from provided context
        attention_context = AttentionContext(
            goal=context.get("goal", "") if context else "",
            current_findings=findings,
            constraints=context.get("constraints", []) if context else [],
            metadata=context or {},
        )

        try:
            queries = await self.query_generator.generate_queries(
                context=attention_context,
                findings=findings,
                metadata=context,
            )

            # Limit to max_queries
            queries = queries[:max_queries]

            logger.debug(
                f"QueryAttentionCapability: generated {len(queries)} queries "
                f"from {len(findings)} findings"
            )

            return {
                "queries": [q.model_dump() for q in queries],
                "generated_count": len(queries),
            }

        except Exception as e:
            logger.error(f"QueryAttentionCapability: query generation failed: {e}")
            return {"queries": [], "generated_count": 0, "error": str(e)}

    @action_executor()
    async def route_query(
        self,
        query: PageQuery | dict[str, Any],
        available_pages: list[str] | None = None,
        max_results: int = 10,
        min_relevance: float = 0.5,
        boost_ws: bool = False,
        prefer_locality: str = "none",
    ) -> dict[str, Any]:
        """Route a query to find relevant pages.

        LLM-controllable cache-awareness options:
        - boost_ws: Boost scores for pages already in working set (cache-aware)
        - prefer_locality: Prefer pages with spatial/temporal locality

        Wraps: PageQueryRoutingPolicy.route_query()

        Args:
            query: Query to route (PageQuery or dict)
            available_pages: Pages to consider (None = all available)
            max_results: Maximum pages to return
            min_relevance: Minimum relevance score threshold
            boost_ws: If True, boost pages in working set by 20% (LLM decides when useful)
            prefer_locality: Locality preference (LLM decides based on data patterns):
                - "none": No locality preference (default)
                - "spatial": Prefer pages near recently accessed (graph neighbors)
                - "temporal": Prefer recently accessed pages
                - "both": Combine spatial and temporal locality

        Returns:
            Dict with:
            - pages: List of dicts with page_id, score, explanation, cache_hit
            - total_found: Total pages found above threshold
            - cache_hits: Number of pages in working set (if boost_ws=True)
        """
        if self.routing_policy is None:
            logger.warning("QueryAttentionCapability: no routing_policy configured")
            return {"pages": [], "total_found": 0, "error": "no_routing_policy"}

        # Convert dict to PageQuery if needed
        if isinstance(query, dict):
            query = PageQuery(**query)

        # Override query settings
        query.max_results = max_results
        query.min_relevance = min_relevance

        # Get working set if cache-aware options are enabled
        working_set_pages: set[str] = set()
        recently_accessed: list[str] = []
        if boost_ws or prefer_locality in ("temporal", "both"):
            from .working_set import WorkingSetCapability
            ws_cap = self.agent.get_capability_by_type(WorkingSetCapability)
            if ws_cap:
                try:
                    ws_result = await ws_cap.get_working_set()
                    working_set_pages = set(ws_result.get("pages", []))
                    recently_accessed = ws_result.get("access_order", [])  # Most recent first
                except Exception as e:
                    logger.debug(f"Could not get working set for cache-aware routing: {e}")

        # Get graph neighbors if spatial locality requested
        spatial_neighbors: set[str] = set()
        if prefer_locality in ("spatial", "both") and recently_accessed:
            from .page_graph import PageGraphCapability
            pg_cap = self.agent.get_capability_by_type(PageGraphCapability)
            if pg_cap:
                try:
                    # Get neighbors of recently accessed pages
                    for page_id in recently_accessed[:5]:  # Top 5 recently accessed
                        neighbors_result = await pg_cap.get_neighbors(
                            page_id=page_id,
                            direction="both",
                        )
                        spatial_neighbors.update(neighbors_result.get("neighbors", []))
                except Exception as e:
                    logger.debug(f"Could not get spatial neighbors: {e}")

        try:
            attention_scores: list[AttentionScore] = await self.routing_policy.route_query(
                query=query,
                available_pages=available_pages,
                context=None,
            )

            # Apply LLM-controllable boosts
            for s in attention_scores:
                original_score = s.score
                boost_reasons = []

                # Working set boost (20% if enabled)
                if boost_ws and s.page_id in working_set_pages:
                    s.score = min(1.0, s.score * 1.2)
                    boost_reasons.append(f"ws_boost:{original_score:.2f}->{s.score:.2f}")

                # Temporal locality boost (10% for recently accessed)
                if prefer_locality in ("temporal", "both") and s.page_id in recently_accessed[:10]:
                    recency_idx = recently_accessed.index(s.page_id)
                    recency_boost = 1.1 - (recency_idx * 0.01)  # 10% for most recent, decreasing
                    s.score = min(1.0, s.score * recency_boost)
                    boost_reasons.append(f"temporal:{recency_boost:.2f}")

                # Spatial locality boost (15% for graph neighbors)
                if prefer_locality in ("spatial", "both") and s.page_id in spatial_neighbors:
                    s.score = min(1.0, s.score * 1.15)
                    boost_reasons.append("spatial:1.15")

                if boost_reasons:
                    s.explanation = f"{s.explanation or ''} [{','.join(boost_reasons)}]"

            # Re-sort after boosting
            attention_scores.sort(key=lambda x: x.score, reverse=True)

            # Filter by min_relevance and limit
            filtered = [
                s for s in attention_scores
                if s.score >= min_relevance
            ][:max_results]

            # Build result with cache_hit info
            cache_hits = 0
            pages = []
            for s in filtered:
                is_cache_hit = s.page_id in working_set_pages
                if is_cache_hit:
                    cache_hits += 1
                pages.append({
                    "page_id": s.page_id,
                    "score": s.score,
                    "explanation": s.explanation,
                    "matched_features": s.matched_features,
                    "cache_hit": is_cache_hit,
                })

            logger.debug(
                f"QueryAttentionCapability: routed query to {len(pages)} pages "
                f"(threshold={min_relevance}, boost_ws={boost_ws}, "
                f"prefer_locality={prefer_locality}, cache_hits={cache_hits})"
            )

            return {
                "pages": pages,
                "total_found": len(filtered),
                "cache_hits": cache_hits,
                "options_used": {
                    "boost_ws": boost_ws,
                    "prefer_locality": prefer_locality,
                },
            }

        except Exception as e:
            logger.error(f"QueryAttentionCapability: query routing failed: {e}")
            return {"pages": [], "total_found": 0, "error": str(e)}

    @action_executor()
    async def score_relevance(
        self,
        query: PageQuery | dict[str, Any],
        page_ids: list[str],
    ) -> dict[str, Any]:
        """Score relevance of a query to specific pages.

        Use for evaluating candidate pages without full routing.

        Args:
            query: Query to score against
            page_ids: Specific pages to score

        Returns:
            Dict with:
            - scores: Dict mapping page_id -> relevance score
            - sorted_pages: Page IDs sorted by relevance (highest first)
        """
        if self.routing_policy is None:
            logger.warning("QueryAttentionCapability: no routing_policy configured")
            return {"scores": {}, "sorted_pages": [], "error": "no_routing_policy"}

        # Convert dict to PageQuery if needed
        if isinstance(query, dict):
            query = PageQuery(**query)

        try:
            # Route with specific pages
            attention_scores = await self.routing_policy.route_query(
                query=query,
                available_pages=page_ids,
                context=None,
            )

            # Build scores dict
            scores = {s.page_id: s.score for s in attention_scores}

            # Ensure all requested pages have a score (0 if not found)
            for page_id in page_ids:
                if page_id not in scores:
                    scores[page_id] = 0.0

            # Sort by score
            sorted_pages = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

            return {
                "scores": scores,
                "sorted_pages": sorted_pages,
            }

        except Exception as e:
            logger.error(f"QueryAttentionCapability: scoring failed: {e}")
            return {"scores": {}, "sorted_pages": [], "error": str(e)}

    @action_executor()
    async def create_query(
        self,
        query_text: str,
        query_type: str = "semantic",
        source_page_ids: list[str] | None = None,
        max_results: int = 10,
        min_relevance: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a PageQuery object from parameters.

        Helper action for ActionPolicy to create queries programmatically.

        Args:
            query_text: Natural language query
            query_type: Query type ("semantic", "keyword", "structural", "hybrid")
            source_page_ids: Pages generating this query
            max_results: Maximum results to return
            min_relevance: Minimum relevance threshold
            filters: Optional filters

        Returns:
            Dict with:
            - query: PageQuery as dict
        """
        query = PageQuery(
            query_text=query_text,
            query_type=query_type,
            source_page_ids=source_page_ids or [],
            max_results=max_results,
            min_relevance=min_relevance,
            filters=filters or {},
        )

        return {"query": query.model_dump()}

    @action_executor()
    async def batch_route_queries(
        self,
        queries: list[PageQuery | dict[str, Any]],
        available_pages: list[str] | None = None,
        max_results_per_query: int = 5,
        min_relevance: float = 0.5,
        deduplicate: bool = True,
        boost_ws: bool = False,
        prefer_locality: str = "none",
    ) -> dict[str, Any]:
        """Route multiple queries in batch.

        More efficient than routing queries one by one when you have
        multiple queries to route.

        Args:
            queries: List of queries to route
            available_pages: Pages to consider (None = all available)
            max_results_per_query: Maximum pages per query
            min_relevance: Minimum relevance threshold
            deduplicate: Remove duplicate pages across queries
            boost_ws: If True, boost pages in working set (passed to route_query)
            prefer_locality: Locality preference (passed to route_query)

        Returns:
            Dict with:
            - results: List of routing results (one per query)
            - all_pages: Deduplicated list of all relevant pages
            - total_queries: Number of queries processed
            - total_cache_hits: Total cache hits across all queries
        """
        results = []
        all_pages_set = set()
        total_cache_hits = 0

        for query in queries:
            result = await self.route_query(
                query=query,
                available_pages=available_pages,
                max_results=max_results_per_query,
                min_relevance=min_relevance,
                boost_ws=boost_ws,
                prefer_locality=prefer_locality,
            )
            results.append(result)
            total_cache_hits += result.get("cache_hits", 0)

            # Collect all pages
            for page_info in result.get("pages", []):
                all_pages_set.add(page_info["page_id"])

        # Deduplicate or just list all
        all_pages = list(all_pages_set) if deduplicate else [
            page_info["page_id"]
            for result in results
            for page_info in result.get("pages", [])
        ]

        logger.debug(
            f"QueryAttentionCapability: batch routed {len(queries)} queries, "
            f"found {len(all_pages)} unique pages (cache_hits={total_cache_hits})"
        )

        return {
            "results": results,
            "all_pages": all_pages,
            "total_queries": len(queries),
            "total_cache_hits": total_cache_hits,
            "options_used": {
                "boost_ws": boost_ws,
                "prefer_locality": prefer_locality,
            },
        }
