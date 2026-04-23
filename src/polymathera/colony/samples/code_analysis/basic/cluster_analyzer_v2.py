"""ClusterAnalyzer V2: Iterative reasoning-based multi-shard inference.

This is the CORRECT design: replaces FSM with iterative reasoning loop.

Key differences from V1:
- V1: Hardcoded phases (FSM): key_generation → local_analysis → query_processing → synthesis
- V2: LLM-driven reasoning loop: PLAN → ACT → REFLECT → CRITIQUE → ADAPT

The agent REASONS about what to do next instead of following a predetermined sequence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from overrides import override

from polymathera.colony.agents.scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from polymathera.colony.agents.blackboard.protocol import BasicAnalysisProtocol
from polymathera.colony.agents.base import Agent, AgentCapability
from polymathera.colony.agents.models import (
    ActionType,
    ActionResult,
    AgentSuspensionState,
    AttentionContext,
)
from polymathera.colony.cluster.models import InferenceResponse
from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.capabilities.reflection import ReflectionCapability
from polymathera.colony.agents.patterns.capabilities.critique import CriticCapability
from polymathera.colony.agents.patterns.capabilities.result import ResultCapability
from polymathera.colony.agents.patterns.capabilities.query_attention import QueryAttentionCapability
from polymathera.colony.agents.patterns.models import Critique
from polymathera.colony.agents.patterns.attention import (
    PageKey,
    QueryGenerator,
    DependencyQueryGenerator,
    HybridKeyGenerator,
)
from polymathera.colony.agents.patterns.attention.key_registry import GlobalPageKeyRegistry
from polymathera.colony.agents.patterns.attention.query_routing import PageQueryRoutingPolicy, create_page_query_router2
from polymathera.colony.vcm.sources import PageCluster

from .config import ClusterAnalyzerConfig

logger = logging.getLogger(__name__)



class ClusterAnalyzerCapabilityV2(AgentCapability):
    """Capability providing cluster analysis action executors.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "cluster_analyzer_v2",
        capability_key: str = "cluster_analyzer_v2_capability",
        app_name: str | None = None,
    ):
        super().__init__(agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=[], capability_key=capability_key, app_name=app_name)  # TODO: Replace some of the actions below with event handlers and input patterns BasicAnalysisProtocol
        self.blackboard_scope = scope
        self.key_registry: GlobalPageKeyRegistry | None = None
        self.key_generator: HybridKeyGenerator | None = None
        self.query_generator: QueryGenerator | None = None
        self.query_router: PageQueryRoutingPolicy | None = None

    def get_action_group_description(self) -> str:
        return (
            "Cluster Analysis V2 (Iterative reasoning loop) — replaces FSM with LLM-driven planning. "
            "PLAN → ACT → REFLECT → CRITIQUE → ADAPT cycle. "
            "Agent reasons about what to analyze next, generates cross-page queries, "
            "routes to relevant pages, and iterates until quality threshold is met. "
            "Uses game protocols (hypothesis, negotiation) for advanced validation."
        )

    async def initialize(self) -> None:
        """Initialize ClusterAnalyzerCapabilityV2."""
        await super().initialize()

        parameters = self.agent.metadata.parameters or {}

        # Get cluster — needed by query router below
        cluster_data = parameters.get("cluster")
        if not cluster_data:
            raise ValueError("Missing cluster in metadata")
        self.cluster = PageCluster(**cluster_data)

        # Analysis state — initialized before query router which references page_keys
        self.page_keys: dict[str, PageKey] = {}
        self.local_analyses: dict[str, dict] = {}
        self.query_results: list[dict] = []
        self.cluster_summary: dict | None = None

        # Key generator and registry to be used by action executors
        self.key_registry = GlobalPageKeyRegistry(self.agent)
        await self.key_registry.initialize()
        self.key_generator = HybridKeyGenerator(self)

        # Query generator to be used by action executors
        self.query_generator = DependencyQueryGenerator(max_queries=10)

        # Query router — depends on self.cluster and self.page_keys
        self.query_router = await create_page_query_router2(
            agent=self,
            attention_policy_type=parameters.get("attention_policy_type", "hierarchical"),
            top_k_clusters=parameters.get("top_k_clusters", 5),
            top_n_pages_per_cluster=parameters.get("top_n_pages_per_cluster", 3),
            top_n_pages_overall=parameters.get("top_n_pages_overall", 10),
            top_n_pages=parameters.get("top_n_pages", 10),
            cluster_id=self.cluster.cluster_id,
            router_type=parameters.get("query_router_type", "hierarchical"),
            page_keys=self.page_keys or None,
            working_set=parameters.get("working_set", set()),
            cache_boost_factor=parameters.get("cache_boost_factor", 1.5)
        )

        # Load configuration from metadata
        config_data = parameters.get("config", {})
        self.config = ClusterAnalyzerConfig(**config_data)

        # Get or create ResultCapability for cluster-wide result visibility
        self.result_cap: ResultCapability | None = self.agent.get_capability_by_type(ResultCapability)
        if not self.result_cap:
            self.result_cap = ResultCapability(agent=self.agent, scope=self.blackboard_scope)
            await self.result_cap.initialize()
            self.agent.add_capability(self.result_cap)

        # Get or create QueryAttentionCapability for standardized query routing
        self.query_attention_cap: QueryAttentionCapability | None = self.agent.get_capability_by_type(QueryAttentionCapability)
        if not self.query_attention_cap:
            self.query_attention_cap = QueryAttentionCapability(
                agent=self.agent,
                scope=self.blackboard_scope,
                query_generator=self.query_generator,
                routing_policy=self.query_router,
            )
            await self.query_attention_cap.initialize()
            self.agent.add_capability(self.query_attention_cap)

        logger.info(
            f"ClusterAnalyzerCapabilityV2 initialized: cluster={self.cluster.cluster_id}, "
            f"pages={len(self.cluster.page_ids)}"
        )

    @action_executor()
    async def analysis_done(self, critique: Critique) -> ActionResult:
        # Check if done
        if critique.quality_score >= self.config.quality_threshold:
            logger.info("Analysis complete (quality threshold reached)")
            return ActionResult(success=True, output={"done": True})
        else:
            logger.info("Analysis not yet complete (quality threshold not met)")
            return ActionResult(success=True, output={"done": False})

    @action_executor(action_key=ActionType.ANALYZE_PAGE)
    async def analyze_page(self, page_id: str) -> ActionResult:
        """Analyze a single page: generate key + local analysis.

        Args:
            page_id: VCM page ID to analyze

        Returns:
            ActionResult with page analysis output
        """
        # TODO: Replace all calls to self.agent methods with calls to tools/policies where applicable
        if not page_id:
            return ActionResult(success=False, error="Missing page_id parameter")

        if page_id not in self.cluster.page_ids:
            return ActionResult(success=False, error=f"Page {page_id} not in assigned cluster")

        try:
            # Request page load
            await self.agent.request_page(page_id, priority=10)

            # Wait for load
            if not await self._wait_for_page_load(page_id, max_wait=30):
                return ActionResult(
                    success=False, error=f"Page {page_id} failed to load"
                )

            # Generate key if not cached
            if page_id not in self.page_keys:
                key = await self.generate_page_key(page_id)
                self.page_keys[page_id] = key

                # Publish to global registry
                await self.key_registry.publish_page_key(
                    page_id=page_id,
                    key=key,
                    cluster_id=self.cluster.cluster_id,
                    metadata={"agent_id": self.agent.agent_id}
                )

            # Perform local analysis
            # TODO: Use PageAnalyzer
            analysis = await self._analyze_page_local(
                page_id, self.page_keys[page_id]
            )
            self.local_analyses[page_id] = analysis

            # Compute coverage for critique
            total_pages = len(self.cluster.page_ids) if hasattr(self, 'cluster') else 1
            coverage = len(self.local_analyses) / total_pages if total_pages > 0 else 1.0

            return ActionResult(
                success=True,
                output={
                    "page_id": page_id,
                    "analysis": analysis,
                    "_reflection_learnings": {  # TODO: Devise a better way to inject reflection learnings for ReflectionCapability to pick up
                        f"analyzed:{page_id}": True,
                        "pages_analyzed_count": len(self.local_analyses),
                    },
                    "_critique_learnings": {  # TODO: Devise a better way to inject critique learnings for CriticCapability to pick up
                        "coverage": coverage,
                        "pages_analyzed": len(self.local_analyses),
                        "total_pages": total_pages,
                        "confidence": 0.8 if analysis else 0.3,
                    },
                },
                metrics={"pages_analyzed": len(self.local_analyses)},
            )

        except Exception as e:
            logger.error(f"Failed to analyze page {page_id}: {e}", exc_info=True)
            return ActionResult(success=False, error=str(e))

    @action_executor(action_key=ActionType.GENERATE_QUERIES)
    async def generate_queries(self, page_ids: list[str]) -> ActionResult:
        """Generate queries from analysis findings.

        Args:
            page_ids: List of page IDs whose analyses to use for query generation

        Returns:
            ActionResult with generated queries
        """
        if not page_ids:
            return ActionResult(success=False, error="Missing page_ids parameter")

        try:
            # Get findings for specified pages
            findings = [
                self.local_analyses.get(page_id)
                for page_id in page_ids
                if page_id in self.local_analyses
            ]

            if not findings:
                return ActionResult(
                    success=False,
                    error=f"No analyses found for pages: {page_ids}"
                )

            # Generate queries using QueryGenerator policy
            if not self.query_generator:
                return ActionResult(
                    success=True,
                    output={"queries": []},
                    metrics={"queries_generated": 0}
                )

            queries = await self.query_generator.generate_queries(
                context={
                    "current_pages": list(self.local_analyses.keys()),
                    "cluster_id": self.cluster.cluster_id,
                    "source_page_ids": page_ids
                },
                findings=findings
            )

            logger.info(f"Generated {len(queries)} queries from {len(findings)} findings")

            return ActionResult(
                success=True,
                output={
                    "queries": queries,
                    "_reflection_learnings": {  # TODO: Devise a better way to inject reflection learnings for ReflectionCapability to pick up
                        "pending_queries": list(queries),
                        "queries_generated_count": len(queries),
                    },
                },
                metrics={"queries_generated": len(queries)}
            )

        except Exception as e:
            logger.error(f"Query generation failed: {e}", exc_info=True)
            return ActionResult(success=False, error=str(e))

    @action_executor(action_key=ActionType.ROUTE_QUERY)
    async def route_query(
        self,
        query: Any,
        available_pages: list[str] | None = None
    ) -> ActionResult:
        """Route a single query to relevant pages using unified query routing.

        Args:
            query: Query object to route
            available_pages: Optional list of page IDs to consider

        Returns:
            ActionResult with routing results
        """
        if not query:
            return ActionResult(success=False, error="Missing query parameter")

        try:
            # Use unified query routing policy
            relevant_pages = await self.query_router.route_query(
                query=query,
                available_pages=available_pages,
                context=AttentionContext(
                    source_agent=self.agent.agent_id,
                    source_cluster=self.cluster.cluster_id,
                )
            )

            # Store query results (just the routing, not the answer yet)
            self.query_results.append(
                {
                    "query": query,
                    "results": [
                        {"page_id": score.page_id, "score": score.score}
                        for score in relevant_pages
                    ],
                }
            )

            relevant_page_ids = [score.page_id for score in relevant_pages]
            scores = [score.score for score in relevant_pages]

            return ActionResult(
                success=True,
                output={
                    "query": query,
                    "relevant_pages": relevant_page_ids,
                    "scores": scores,
                    "_reflection_learnings": {  # TODO: Devise a better way to inject reflection learnings for ReflectionCapability to pick up
                        "last_query_routing": {
                            "query": query,
                            "relevant_pages": relevant_page_ids,
                            "attention_scores": dict(zip(relevant_page_ids, scores)),
                            "top_pages": relevant_page_ids[:5] if len(relevant_page_ids) > 5 else relevant_page_ids,
                            "total_candidates": len(relevant_page_ids),
                        },
                        "query_results_count": len(self.query_results),
                    },
                },
                metrics={"pages_found": len(relevant_pages)},
            )

        except Exception as e:
            logger.error(f"Query routing failed: {e}", exc_info=True)
            return ActionResult(success=False, error=str(e))

    @action_executor(action_key=ActionType.PROCESS_QUERY)
    async def process_query(
        self,
        query: Any,
        page_ids: list[str],
        max_wait: int = 30,
        max_tokens: int = 1000
    ) -> ActionResult:
        """Process a query by loading relevant pages and asking LLM for answer.

        Args:
            query: Query to process
            page_ids: Pages identified by ROUTE_QUERY
            max_wait: Max seconds to wait for page load
            max_tokens: Max tokens for LLM response

        Returns:
            ActionResult with query answer
        """

        if not query:
            return ActionResult(success=False, error="Missing query parameter")
        if not page_ids:
            return ActionResult(success=False, error="Missing page_ids parameter")

        try:
            # Load relevant pages
            for page_id in page_ids:
                await self.agent.request_page(page_id, priority=5)

            # Wait for pages to load
            for page_id in page_ids:
                loaded = await self._wait_for_page_load(page_id, max_wait=max_wait)
                if not loaded:
                    logger.warning(f"Page {page_id} failed to load for query processing")

            # Ask LLM to answer query using loaded pages as context
            query_text = query.query_text if hasattr(query, 'query_text') else str(query)
            prompt = f"""Answer this question using the provided code pages:

Question: {query_text}

Analyze the code in the loaded pages and provide a concise answer.
Focus on facts from the code, not speculation.

Output format (JSON):
{{
    "answer": "Direct answer to the question",
    "evidence": [{{"page_id": "...", "excerpt": "relevant code snippet", "explanation": "why this is relevant"}}],
    "confidence": "high|medium|low",
    "additional_pages_needed": ["page_id_1", "page_id_2"] or []
}}
"""
            # TODO: Almost all `infer` calls can be agent calls, especially
            # if the inference call involves multiple pages.
            response: InferenceResponse = await self.agent.infer(
                context_page_ids=page_ids,
                prompt=prompt,
                max_tokens=max_tokens,
            )

            # Parse answer
            try:
                answer = json.loads(response.generated_text)
            except json.JSONDecodeError:
                answer = { # TODO: Convert this to a pydantic model
                    "answer": response.generated_text,
                    "evidence": [],
                    "confidence": "low",
                    "additional_pages_needed": []
                }

            # Determine if answer is satisfactory
            is_satisfactory = False
            if isinstance(answer, dict):
                is_satisfactory = (
                    answer.get("confidence") in ["high", "medium"] and
                    not answer.get("additional_pages_needed")
                )

            return ActionResult(
                success=True,
                output={
                    "query": query,
                    "answer": answer,
                    "pages_used": page_ids,
                    "_reflection_learnings": {  # TODO: Devise a better way to inject reflection learnings for ReflectionCapability to pick up
                        "last_query_answer": {
                            "query": query,
                            "answer": answer,
                            "pages_used": page_ids,
                            "pages_used_count": len(page_ids),
                        },
                        "answer_satisfactory": is_satisfactory,
                    },
                },
                metrics={"pages_loaded": len(page_ids)},
            )

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return ActionResult(success=False, error=str(e))

    @action_executor(action_key=ActionType.SYNTHESIZE)
    async def summarize_cluster(self) -> ActionResult:
        """Synthesize cluster-level summary from local analyses.

        Returns:
            ActionResult with cluster summary
        """
        import json
        from polymathera.colony.cluster.models import InferenceResponse

        try:
            # Build synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt()

            # Use LLM to synthesize (NO pages loaded - just reasoning)
            response: InferenceResponse = await self.agent.infer(
                context_page_ids=[],
                prompt=synthesis_prompt,
                max_tokens=2000,
            )

            # Parse result
            try:
                summary = json.loads(response.generated_text)
            except json.JSONDecodeError:
                summary = {"summary": response.generated_text}

            self.cluster_summary = summary

            # Create representative cluster key (centroid of page keys)
            representative_key = await self._create_representative_cluster_key()

            # Publish cluster summary to global registry for hierarchical attention
            await self.key_registry.publish_cluster_summary(
                cluster_id=self.cluster.cluster_id,
                summary=summary,
                representative_key=representative_key,
                page_ids=self.cluster.page_ids,
                metadata={
                    "agent_id": self.agent.agent_id,
                    "pages_analyzed": len(self.local_analyses)
                }
            )

            # Write completion to parent's blackboard (existing pattern)
            parent_id = self.agent.metadata.parent_agent_id
            if parent_id:
                parent_blackboard = await self.agent.get_blackboard(
                    scope_id=ScopeUtils.get_agent_level_scope(parent_id)
                )
                await parent_blackboard.write(
                    BasicAnalysisProtocol.cluster_analysis_complete_key(self.agent.agent_id),
                    {
                        "summary": summary,
                        "cluster_id": self.cluster.cluster_id,
                        "representative_key": representative_key.model_dump() if hasattr(representative_key, 'model_dump') else str(representative_key)
                    }
                )

            # Store result via ResultCapability for cluster-wide visibility
            if self.result_cap:
                await self.result_cap.store_partial(
                    result_id=f"cluster:{self.cluster.cluster_id}",
                    result={
                        "summary": summary,
                        "cluster_id": self.cluster.cluster_id,
                        "representative_key": representative_key.model_dump() if hasattr(representative_key, 'model_dump') else str(representative_key),
                        "pages_analyzed": len(self.local_analyses),
                        "local_analyses": self.local_analyses,
                    },
                    source_agent=self.agent.agent_id,
                    source_pages=self.cluster.page_ids,
                    result_type="cluster_analysis",
                )

            # Compute quality metrics for critique
            total_pages = len(self.cluster.page_ids) if hasattr(self, 'cluster') else 1
            analyzed_pages = len(self.local_analyses)
            coverage = analyzed_pages / total_pages if total_pages > 0 else 1.0
            summary_quality = 0.9 if summary and len(str(summary)) > 100 else 0.5

            return ActionResult(
                success=True,
                output={
                    "summary": summary,
                    "representative_key": representative_key.model_dump() if hasattr(representative_key, 'model_dump') else str(representative_key),
                    "_reflection_learnings": {  # TODO: Devise a better way to inject reflection learnings for ReflectionCapability to pick up
                        "synthesis_complete": True,
                        "cluster_summary": summary,
                    },
                    "_critique_learnings": {  # TODO: Devise a better way to inject critique learnings for CriticCapability to pick up
                        "coverage": coverage,
                        "pages_analyzed": analyzed_pages,
                        "total_pages": total_pages,
                        "summary_quality": summary_quality,
                        "confidence": coverage * summary_quality,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return ActionResult(success=False, error=str(e))

    @action_executor(action_key=ActionType.WRITE_BLACKBOARD)
    async def write_blackboard(self, key: str, value: Any) -> ActionResult:
        """Write result to blackboard.

        Args:
            key: Blackboard key
            value: Value to write

        Returns:
            ActionResult indicating success
        """
        if not key:
            return ActionResult(success=False, error="Missing key parameter")

        try:
            parent_id = self.agent.metadata.parent_agent_id
            if parent_id:
                parent_blackboard = await self.agent.get_blackboard(
                    scope_id=ScopeUtils.get_agent_level_scope(parent_id)
                )
                await parent_blackboard.write(key, value)

            return ActionResult(success=True, output={"key": key})

        except Exception as e:
            logger.error(f"Blackboard write failed: {e}", exc_info=True)
            return ActionResult(success=False, error=str(e))

    @action_executor(action_key="escalate_error")
    async def escalate_error(
        self,
        error_message: str,
        action_type: str | None = None,
        retry_count: int = 0
    ) -> ActionResult:
        """Escalate error to parent agent.

        Args:
            error_message: Error message to escalate
            action_type: Type of action that failed
            retry_count: Number of retries attempted

        Returns:
            ActionResult indicating escalation success
        """
        try:
            # TODO: Remove the parameters of this @action_executor.
            # TODO: Get the last action from the action_policy instead.
            # TODO: This executor assumes the action has already been executed and failed.
            # If action failed, escalate to parent for retry decision
            ### # Get retry count from action metadata
            ### retry_count = action.parameters.get("retry_count", 0)
            # TODO: Escalate to parent blackboard or to team channel or to human operator
            return ActionResult(success=True, output={
                "escalated": True,
                "retry_count": retry_count, # action.parameters.get("retry_count", 0)
                "action_type": action_type or "unknown",
                "parameters": {}, # TODO: action.parameters if action else {}
            })
        except Exception as escalation_error:
            # Don't let escalation failures prevent returning the result
            logger.error(
                f"Failed to escalate error to parent: {escalation_error}",
                exc_info=True
            )
            return ActionResult(success=False, error=str(escalation_error))

    async def _wait_for_page_load(
        self, page_id: str, max_wait: int = 30, period: float = 0.5
    ) -> bool:
        """Wait for page to be loaded."""
        waited = 0
        while not await self.agent.is_page_loaded(page_id) and waited < max_wait:
            await asyncio.sleep(period)
            waited += period
        return await self.agent.is_page_loaded(page_id)

    async def generate_page_key(self, page_id: str) -> Any:
        """Generate key for page using key generator.

        Uses VCM architecture: key generator accesses content via context_page_ids,
        not by passing raw content strings.
        """
        # Check cache first
        cached_key = await self.key_registry.get_page_key(page_id)
        if cached_key:
            return cached_key

        if page_id not in self.cluster.page_ids:
            raise ValueError(f"Page {page_id} not in assigned cluster")

        # Ensure page is loaded into VCM (if not already)
        await self.agent.request_page(page_id, priority=10)
        await self._wait_for_page_load(page_id, max_wait=30)

        # Generate key (VCM loads content automatically via context_page_ids)
        metadata = {"page_id": page_id}
        key = await self.key_generator.generate_key(page_id, metadata)

        # Cache it
        await self.key_registry.publish_page_key(
            page_id,
            key=key,
            cluster_id=self.cluster.cluster_id,
        )

        return key

    async def _analyze_page_local(self, page_id: str, page_key: Any) -> dict:
        """Perform local analysis of single page."""
        prompt = self._build_local_analysis_prompt(page_id, str(page_key))

        response: InferenceResponse = await self.agent.infer(
            context_page_ids=[page_id],  # Only this page
            prompt=prompt,
            max_tokens=2000, # TODO: Make configurable
        )

        # Parse response
        try:
            return json.loads(response.generated_text)
        except json.JSONDecodeError:
            return {"structure": response.generated_text, "queries": []}

    def _build_local_analysis_prompt(self, page_id: str, page_key_summary: str) -> str:
        """Build prompt for local analysis."""
        return f"""Analyze this code page and extract its structure.

Page ID: {page_id}

Key Summary:
{page_key_summary}

Extract (JSON format):
- classes: [list of classes]
- functions: [list of functions]
- dependencies: [list of imports/dependencies]
- queries: [list of questions about other pages]

Output ONLY valid JSON."""

    def _build_synthesis_prompt(self) -> str:
        """Build prompt for synthesis."""
        analyses_summary = json.dumps(self.local_analyses, indent=2)

        return f"""Synthesize a cluster-level summary from these local analyses.

Local Analyses:
{analyses_summary}

Create a coherent summary (JSON format) with:
- overall_structure: High-level architecture
- key_components: Main components and their roles
- dependencies: Cross-file dependencies
- insights: Key insights about the cluster

Output ONLY valid JSON."""

    async def _create_representative_cluster_key(self) -> Any:
        """Create representative key for this cluster (centroid of page keys).

        The representative key is used for hierarchical attention - when other agents
        query at cluster level, this key represents the entire cluster.

        Strategy: Average semantic embeddings + combine structural features.
        """

        if not self.page_keys:
            # No keys yet, create minimal representative
            return PageKey(
                page_id=f"cluster:{self.cluster.cluster_id}",
                key_type="hybrid",
                summary=self.cluster_summary.get("summary", "Cluster summary") if self.cluster_summary else "Unknown cluster"
            )

        # Average semantic embeddings
        embeddings = [
            key.semantic_embedding
            for key in self.page_keys.values()
            if hasattr(key, 'semantic_embedding') and key.semantic_embedding is not None
        ]

        if embeddings:
            import numpy as np
            avg_embedding = np.mean(embeddings, axis=0).tolist()
        else:
            avg_embedding = None

        # Combine structural features (union of all features)
        combined_structural = {
            "classes": [],
            "functions": [],
            "imports": []
        }

        # TODO: Allow this merging strategy to be configurable (policy-based)
        for key in self.page_keys.values():
            if hasattr(key, 'structural_features') and key.structural_features:
                combined_structural["classes"].extend(key.structural_features.get("classes", []))
                combined_structural["functions"].extend(key.structural_features.get("functions", []))
                combined_structural["imports"].extend(key.structural_features.get("imports", []))

        # Deduplicate
        for feature_type in combined_structural:
            combined_structural[feature_type] = list(set(combined_structural[feature_type]))

        # Create representative key
        summary = self.cluster_summary.get("summary", "") if self.cluster_summary else ""

        return PageKey(
            page_id=f"cluster:{self.cluster.cluster_id}",
            key_type="hybrid",
            structural_features=combined_structural,
            semantic_embedding=avg_embedding,
            summary=summary,
            metadata={
                "cluster_id": self.cluster.cluster_id,
                "page_count": len(self.page_keys),
                "is_cluster_key": True
            }
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize ClusterAnalyzerV2-specific state.

        Overrides base implementation to add ClusterAnalyzerV2-specific state.
        Calls super() first to get base state, then adds subclass state.

        Returns:
            AgentSuspensionState with all agent state serialized
        """
        # Get base state from parent
        state = await super().serialize_suspension_state(state)

        # Store ClusterAnalyzerV2-specific state in custom_data with special key
        # Note: Some complex objects (like PageKey, attention_policy) are not serialized
        # as they can be reconstructed from metadata/config
        state.custom_data["_cluster_analyzer_v2_state"] = {
            "page_keys": {
                k: v.model_dump() if hasattr(v, 'model_dump') else str(v)
                for k, v in self.page_keys.items()
            },
            "local_analyses": self.local_analyses,
            "query_results": self.query_results,
            "cluster_summary": self.cluster_summary,
            "cluster": self.cluster.model_dump() if hasattr(self.cluster, 'model_dump') else {
                "cluster_id": getattr(self.cluster, 'cluster_id', None),
                "page_ids": getattr(self.cluster, 'page_ids', []),
            },
        }

        return state

    @override
    async def deserialize_suspension_state(
        self,
        state: AgentSuspensionState
    ) -> None:
        """Restore ClusterAnalyzerV2-specific state from suspension.

        Overrides base implementation to restore ClusterAnalyzerV2-specific state.
        Calls super() first to restore base state, then restores subclass state.

        Args:
            state: AgentSuspensionState to restore from
        """
        # Restore base state first
        await super().deserialize_suspension_state(state)

        # Restore ClusterAnalyzerV2-specific state
        custom_state = state.custom_data.get("_cluster_analyzer_v2_state", {})
        if custom_state:
            # Restore page_keys (may need reconstruction from serialized data)
            page_keys_data = custom_state.get("page_keys", {})
            self.page_keys = {}
            for k, v in page_keys_data.items():
                # Try to reconstruct PageKey if possible, otherwise store as-is
                if isinstance(v, dict) and 'page_id' in v:
                    try:
                        from polymathera.colony.agents.patterns.attention import PageKey
                        self.page_keys[k] = PageKey(**v)
                    except Exception:
                        self.page_keys[k] = v
                else:
                    self.page_keys[k] = v

            self.local_analyses = custom_state.get("local_analyses", {})
            self.query_results = custom_state.get("query_results", [])
            self.cluster_summary = custom_state.get("cluster_summary")

            # Restore cluster from metadata (should already be restored in initialize)
            # But if it's in custom_state, use it
            cluster_data = custom_state.get("cluster")
            if cluster_data and not hasattr(self, 'cluster'):
                self.cluster = PageCluster(**cluster_data)

            logger.info(
                f"Restored ClusterAnalyzerV2 state: "
                f"page_keys={len(self.page_keys)}, "
                f"local_analyses={len(self.local_analyses)}, "
                f"query_results={len(self.query_results)}"
            )




class ClusterAnalyzerV2(Agent):
    """ClusterAnalyzer using iterative reasoning loop (NOT FSM).

    Key improvements over V1:
    - LLM-driven control flow (not hardcoded phases)
    - Iterative refinement based on critique
    - Adaptive planning (can adjust strategy mid-execution)
    - Self-reflection and quality assessment

    Example reasoning loop:
    1. PLAN: "Analyze all pages, then synthesize"
    2. ACT: Analyze page_042
    3. REFLECT: "Learned about AuthManager, need to find usage"
    4. CRITIQUE: "Good start, but need more pages"
    5. ADAPT: Keep original plan, continue
    (repeat...)
    """

    async def initialize(self) -> None:
        """Initialize cluster analyzer."""
        self.add_capability_blueprints([
            ReflectionCapability.bind(),
            CriticCapability.bind(),
            ClusterAnalyzerCapabilityV2.bind(),
        ])
        await super().initialize()
        logger.info(f"ClusterAnalyzerV2 {self.agent_id} initialized")

