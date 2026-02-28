"""ClusterAnalyzer: Multi-shard inference over page clusters.

This agent demonstrates the correct pattern for analyzing multiple pages
when they don't all fit in LLM context:
1. Key generation: Create summaries for all pages
2. Local analysis: Analyze each page independently (ONE at a time)
3. Query processing: Use key-query matching to find relevant pages
4. Synthesis: Combine insights (no pages loaded)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from overrides import override

from colony.agents.base import Agent, AgentState, AgentCapability
from colony.agents.models import (
    AgentSuspensionState,
    AttentionContext,
)
from colony.agents.patterns.attention import PageQuery
from colony.cluster import InferenceResponse
from colony.vcm.sources import PageCluster
from colony.agents.patterns.attention.query_routing import PageQueryRoutingPolicy, create_page_query_router2

from colony.agents.patterns.scope import ScopeAwareResult
from colony.agents.patterns.actions.policies import (
    action_executor,
)

logger = logging.getLogger(__name__)


class ClusterAnalyzerCapability(AgentCapability):
    """Capability providing cluster analysis action executors.
    """

    def __init__(self, agent = None, scope_id = None, *, blackboard = None):
        super().__init__(agent, scope_id, blackboard=blackboard)
        self.query_router: PageQueryRoutingPolicy | None = None

    def get_action_group_description(self) -> str:
        return (
            "Cluster Analysis (V1, FSM-based) — multi-page inference over a page cluster. "
            "Fixed phase progression: key generation → local analysis (one page at a time) → "
            "query processing (cross-page via routing) → synthesis (no pages loaded). "
            "Pages don't all fit in LLM context, so analysis is done one page at a time."
        )

    async def initialize(self) -> None:
        """Initialize cluster analyzer."""
        await super().initialize()

        parameters = self.agent.metadata.parameters or {}

        # Get cluster info
        cluster_data = parameters.get("cluster")
        if not cluster_data:
            raise ValueError("Missing cluster in metadata")

        self.cluster = PageCluster(**cluster_data)

        # Analysis state
        self.page_keys: dict[str, str] = {}  # page_id → summary
        self.local_analyses: dict[str, dict] = {}  # page_id → analysis
        self.queries: list[dict] = []  # Cross-page queries
        self.cluster_summary: dict | None = None

        # State machine
        self.pages_analyzed = 0


        self.query_router = await create_page_query_router2(
            agent=self,
            attention_policy_type=parameters.get("attention_policy_type", "hierarchical"),
            top_k_clusters=parameters.get("top_k_clusters", 5),
            top_n_pages_per_cluster=parameters.get("top_n_pages_per_cluster", 3),
            top_n_pages_overall=parameters.get("top_n_pages_overall", 10),
            top_n_pages=parameters.get("top_n_pages", 10),
            cluster_id=self.cluster.cluster_id if self.cluster else None,
            router_type=parameters.get("query_router_type", "hierarchical"),
            page_keys=self.page_keys if self.page_keys else None,
            working_set=parameters.get("working_set", set()),
            cache_boost_factor=parameters.get("cache_boost_factor", 1.5)
        )

        logger.info(
            f"ClusterAnalyzer initialized: cluster={self.cluster.cluster_id}, "
            f"pages={len(self.cluster.page_ids)}"
        )

    async def _wait_for_page_load(self, page_id: str, max_wait: int = 30, period: float = 0.5) -> bool:
        """Wait for a page to be loaded, up to max_wait seconds."""
        waited = 0
        while not await self.agent.is_page_loaded(page_id) and waited < max_wait:
            await asyncio.sleep(period)
            waited += period
        return await self.agent.is_page_loaded(page_id)

    @action_executor()
    async def generate_cluster_page_keys(self) -> ScopeAwareResult[dict[str, str]]:
        """Generate keys (summaries) for all pages in cluster."""
        try:
            for page_id in self.cluster.page_ids:
                # Request page load
                await self.agent.request_page(page_id, priority=10)

                # Wait for page to be loaded
                if not await self._wait_for_page_load(page_id, max_wait=30):
                    logger.warning(f"Page {page_id} not loaded after 30s, skipping")
                    continue

                # Generate structural summary using LLM
                response: InferenceResponse = await self.agent.infer(
                    context_page_ids=[page_id],  # ONLY this page
                    prompt=self._build_key_generation_prompt(),
                    max_tokens=500
                )

                self.page_keys[page_id] = response.generated_text

        except Exception as e:
            logger.error(f"Key generation failed: {e}", exc_info=True)
            self.state = AgentState.FAILED
            raise

    @action_executor()
    async def analyze_cluster_pages(self) -> None:
        """Analyze each page independently (ONE at a time)."""
        for page_id in self.cluster.page_ids:
            if page_id not in self.page_keys:
                logger.warning(f"No key for page {page_id}, skipping")
                continue

            try:
                # Ensure page is loaded
                await self.agent.request_page(page_id, priority=10)

                # Wait for load
                if not await self._wait_for_page_load(page_id, max_wait=30):
                    logger.warning(f"Page {page_id} not loaded after 30s, skipping")
                    continue

                # Analyze THIS PAGE ONLY
                # TODO: Pass JSON schema for stricter generation
                response: InferenceResponse = await self.agent.infer(
                    context_page_ids=[page_id],  # CRITICAL: Only one page
                    prompt=self._build_local_analysis_prompt(
                        page_id,
                        self.page_keys[page_id]
                    ),
                    max_tokens=2000
                )

                # Parse response (expecting JSON)
                # TODO: Use JSON schema validation for stricter parsing
                try:
                    local_result = json.loads(response.generated_text)
                except json.JSONDecodeError:
                    # LLM didn't return valid JSON, extract what we can
                    local_result = {
                        "structure": response.generated_text,
                        "queries": []
                    }

                # Store local analysis
                self.local_analyses[page_id] = local_result.get("structure", {})

                # Collect queries for cross-referencing
                for query_dict in local_result.get("queries", []):
                    # Create PageQuery
                    page_query = PageQuery(
                        query_text=query_dict.get("question", ""),
                        source_page_ids=[page_id],
                        max_results=3,
                        min_relevance=0.6,
                        query_type="keyword"
                    )

                    # Find relevant pages using attention mechanism
                    # Use unified query routing policy
                    relevant_pages = await self.query_router.route_query(
                        query=page_query,
                        context=AttentionContext(
                            source_agent=self.agent.agent_id,
                            source_cluster=self.cluster.cluster_id,
                        )
                    )

                    self.queries.append({
                        "source_page": page_id,
                        "query": page_query,
                        "relevant_pages": [p.page_id for p in relevant_pages]
                    })

                self.pages_analyzed += 1

            except Exception as e:
                logger.error(f"Local analysis failed for {page_id}: {e}", exc_info=True)
                # Continue with other pages

    @action_executor()
    async def process_queries(self) -> None:
        """Process cross-page queries by loading relevant pages."""
        #--------------------------------------------------------------------------
        # TODO: Move this to a query processing policy/capability?
        #--------------------------------------------------------------------------

        # Group queries by target page for efficiency
        queries_by_target = defaultdict(list)

        for query_info in self.queries:
            for target_page_id in query_info["relevant_pages"]:
                queries_by_target[target_page_id].append(query_info)

        for target_page_id, query_infos in queries_by_target.items():
            try:
                # Load target page
                await self.agent.request_page(target_page_id, priority=10)

                # Wait for load
                if not await self._wait_for_page_load(target_page_id, max_wait=30):
                    logger.warning(f"Target page {target_page_id} not loaded, skipping queries")
                    continue

                # Submit query batch to LLM
                response: InferenceResponse = await self.agent.infer(
                    context_page_ids=[target_page_id],  # Only target page
                    prompt=self._build_query_prompt(query_infos),
                    max_tokens=1500
                )

                # Parse responses
                try:
                    query_responses = json.loads(response.generated_text)
                except json.JSONDecodeError:
                    query_responses = {"responses": []}

                # Update local analyses with query responses
                for resp in query_responses.get("responses", []):
                    source_page = resp.get("source_page")
                    if source_page and source_page in self.local_analyses:
                        if "query_responses" not in self.local_analyses[source_page]:
                            self.local_analyses[source_page]["query_responses"] = []
                        self.local_analyses[source_page]["query_responses"].append(resp)

            except Exception as e:
                logger.error(f"Query processing failed for {target_page_id}: {e}", exc_info=True)
                # Continue with other pages

    @action_executor()
    async def synthesize_cluster_summary(self) -> None:
        """Synthesize cluster-level summary from local analyses.

        NO PAGES LOADED - just reasoning over summaries.
        """
        try:
            # Build synthesis prompt with all local analyses
            synthesis_prompt = self._build_synthesis_prompt()

            # Use LLM to synthesize (NO page context - just reasoning)
            response: InferenceResponse = await self.agent.infer(
                context_page_ids=[],  # No pages loaded
                prompt=synthesis_prompt,
                max_tokens=2000
            )

            # Parse cluster summary
            try:
                self.cluster_summary = json.loads(response.generated_text)
            except json.JSONDecodeError:
                self.cluster_summary = {"summary": response.generated_text}

            # Write to blackboard for parent to read
            parent_id = self.agent.metadata.parent_agent_id
            if parent_id:
                await self._write_to_blackboard(
                    "cluster_analysis_complete",
                    self.cluster_summary
                )

            # Mark work complete
            self.state = AgentState.STOPPED

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            self.state = AgentState.FAILED
            raise

    # === Helper Methods ===

    async def _write_to_blackboard(self, key: str, value: dict) -> None:
        """Write result to blackboard for parent to poll."""
        parent_id = self.agent.metadata.parent_agent_id
        if not parent_id:
            logger.warning("No parent_id in metadata, cannot write to blackboard")
            return

        # Write to blackboard via manager
        # TODO: Need to implement blackboard access methods in AgentManagerBase
        logger.info(f"Writing {key} to blackboard for parent {parent_id}")

    def _build_key_generation_prompt(self) -> str:
        """Build prompt for generating page key (summary)."""
        return """Analyze the code in this page and generate a concise structural summary.

List:
1. Main classes and their purposes
2. Key functions and what they do
3. Important constants or configurations

Keep it under 100 words. Focus on what someone would need to know to decide if this page is relevant to their analysis.

Format as plain text, not JSON."""

    def _build_local_analysis_prompt(self, page_id: str, page_key: str) -> str:
        """Build prompt for local analysis of a single page."""
        return f"""You are analyzing a SHARD of a git repository. This shard contains a portion of the codebase.

**Page ID**: {page_id}

**Structural Summary**:
{page_key}

**Your Task**: Analyze the code structure within THIS SHARD ONLY.

Extract:
1. Files present in this shard
2. Classes and their relationships (within this shard)
3. Functions and their purposes
4. Any external references (code not in this shard)

For any external references, generate QUERIES to other shards.

**Output Format** (JSON):
{{
    "structure": {{
        "files": ["list of files"],
        "classes": [
            {{
                "name": "ClassName",
                "location": "file:line",
                "methods": ["method names"],
                "dependencies": ["what this class depends on"]
            }}
        ],
        "functions": ["function signatures"],
        "external_references": ["things referenced but not defined here"]
    }},
    "queries": [
        {{
            "question": "What does ClassX do?",
            "reason": "ClassX is imported but not defined here"
        }}
    ]
}}

Analyze carefully. Be thorough but concise."""

    def _build_query_prompt(self, query_infos: list[dict]) -> str:
        """Build prompt for answering queries about a page."""
        queries_text = "\n".join([
            f"Q{i+1}: {qi['query'].query_text} (from page {qi['source_page']})"
            for i, qi in enumerate(query_infos)
        ])

        return f"""You are examining code to answer specific questions from other parts of the analysis.

**Queries**:
{queries_text}

For each query, provide:
- Answer: The information requested
- Confidence: high/medium/low
- Evidence: Code snippets or references

**Output Format** (JSON):
{{
    "responses": [
        {{
            "source_page": "page_id",
            "query_id": 0,
            "answer": "...",
            "confidence": "high",
            "evidence": ["code snippet"]
        }}
    ]
}}"""

    def _build_synthesis_prompt(self) -> str:
        """Build prompt for synthesizing cluster-level summary."""
        # Include summaries of all local analyses
        local_summaries = "\n\n".join([
            f"**Page {page_id}**:\n{json.dumps(analysis, indent=2)}"
            for page_id, analysis in self.local_analyses.items()
        ])

        return f"""You are synthesizing a cluster-level summary from multiple local analyses of code pages.

**Local Analyses**:
{local_summaries}

**Your Task**: Build a coherent summary of this cluster.

1. Merge overlapping information
2. Identify the main components and their relationships
3. Note any architectural patterns
4. List key findings

**Output Format** (JSON):
{{
    "cluster_summary": {{
        "main_components": ["list of main components"],
        "architecture_pattern": "description",
        "key_findings": ["list of important findings"],
        "relationships": ["how components relate"]
    }}
}}"""

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize ClusterAnalyzer-specific state.

        Overrides base implementation to add ClusterAnalyzer-specific state.
        Calls super() first to get base state, then adds subclass state.

        Returns:
            AgentSuspensionState with all agent state serialized
        """
        # Get base state from parent
        state = await super().serialize_suspension_state(state)

        # Store ClusterAnalyzer-specific state in custom_data with special key
        state.custom_data["_cluster_analyzer_state"] = {
            "page_keys": self.page_keys,
            "local_analyses": self.local_analyses,
            "queries": self.queries,
            "cluster_summary": self.cluster_summary,
            "pages_analyzed": self.pages_analyzed,
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
        """Restore ClusterAnalyzer-specific state from suspension.

        Overrides base implementation to restore ClusterAnalyzer-specific state.
        Calls super() first to restore base state, then restores subclass state.

        Args:
            state: AgentSuspensionState to restore from
        """
        # Restore base state first
        await super().deserialize_suspension_state(state)

        # Restore ClusterAnalyzer-specific state
        custom_state = state.custom_data.get("_cluster_analyzer_state", {})
        if custom_state:
            self.page_keys = custom_state.get("page_keys", {})
            self.local_analyses = custom_state.get("local_analyses", {})
            self.queries = custom_state.get("queries", [])
            self.cluster_summary = custom_state.get("cluster_summary")
            self.pages_analyzed = custom_state.get("pages_analyzed", 0)

            # Restore cluster from metadata (should already be restored in initialize)
            # But if it's in custom_state, use it
            cluster_data = custom_state.get("cluster")
            if cluster_data and not hasattr(self, 'cluster'):
                self.cluster = PageCluster(**cluster_data)

            logger.info(
                f"Restored ClusterAnalyzer state: "
                f"pages_analyzed={self.pages_analyzed}, "
                f"page_keys={len(self.page_keys)}, "
                f"local_analyses={len(self.local_analyses)}"
            )


class ClusterAnalyzer(Agent):
    """Analyzes a cluster of pages with multi-shard inference protocol.

    Inference phases:
    1. Key generation: Generate structural summaries for all pages
    2. Local analysis: Analyze each page independently
    3. Query processing: Cross-reference findings between pages
    4. Synthesis: Build cluster-level summary

    CRITICAL: Only loads ONE page at a time into LLM context.
    """

    async def initialize(self) -> None:
        """Initialize cluster analyzer."""
        self.add_capability_blueprints([
            ClusterAnalyzerCapability.bind(),
        ])
        await super().initialize()

        logger.info(f"ClusterAnalyzer {self.agent_id} initialized")

