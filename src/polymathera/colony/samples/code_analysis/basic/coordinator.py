"""CodeAnalysisCoordinator: Root agent for distributed code analysis.

This agent demonstrates distributed orchestration:
1. Spawns ClusterAnalyzer agents for each page cluster
2. Monitors children via EVENT-DRIVEN blackboard (no polling!)
3. Synthesizes global report
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import logging
from abc import ABC, abstractmethod
from overrides import override
from pydantic import BaseModel, Field
import time

from colony.agents.base import Agent, AgentState, AgentCapability
from colony.agents.models import (
    Action,
    ActionResult,
    AgentMetadata,
    AgentResourceRequirements,
    AgentSuspensionState,
    RunContext,
    PolicyREPL,
)
from colony.system import get_agent_system
from colony.vcm.sources import PageCluster
from colony.agents.patterns.capabilities.critique import (
    CriticCapability,
    CritiqueContext,
    OutputRelationship,
)
from colony.agents.blackboard import KeyPatternFilter, BlackboardEvent
from colony.agents.patterns.attention import HierarchicalAttentionRouting
from colony.agents.patterns.actions.policies import action_executor
from colony.agents.patterns.events import event_handler, EventProcessingResult
from colony.agents.patterns.capabilities.working_set import WorkingSetCapability
from colony.agents.patterns.capabilities.agent_pool import AgentPoolCapability
from colony.agents.patterns.capabilities.result import ResultCapability
from colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from colony.agents.patterns.capabilities.batching import (
    BatchingPolicy,
    ClusteringBatchPolicy,
    HybridBatchPolicy,
    ContinuousBatchPolicy,
)


logger = logging.getLogger(__name__)


class BaseCodeAnalysisCoordinatorCapability(AgentCapability, ABC):
    """Capability implementing core coordinator logic.

    Keeps Agent subclasses thin by providing @action_executor methods for
    spawning, monitoring, and synthesis.
    """

    def __init__(self, agent: Agent):
        super().__init__(agent=agent)

    def get_action_group_description(self) -> str:
        return (
            "Code Analysis Coordination — orchestrates distributed analysis across page clusters. "
            "Spawns ClusterAnalyzer agents as children, monitors via EVENT-DRIVEN blackboard (no polling). "
            "Batches by working set overlap for cache efficiency. "
            "Synthesizes global report from cluster results with optional peer critique."
        )

    async def initialize(self) -> None:
        """Initialize coordinator."""
        await super().initialize()

        # Track child agents
        self.child_agents: dict[str, str] = {}  # role → agent_id
        self.cluster_analyses: dict[str, dict] = {}  # role → analysis result
        self.clusters_spawned = 0

    # **TODO**: When Coordinator Reasoning Loop Is Implemented:
    # - Adaptive orchestration (adjust strategy based on results)
    # - Quality-driven iterative refinement at global level (iterate until threshold met)
    # - Dynamic strategy (e.g., spawn more agents vs synthesize)
    # - Complex multi-agent synthesis patterns
    # - Hierarchical planning for very large codebases (millions of files)

    @abstractmethod
    async def spawn_cluster_analyzers(self) -> None:
        """Spawn ClusterAnalyzer agents (strategy-specific)."""
        raise NotImplementedError

    @event_handler(pattern="*:cluster_analysis_complete")  # TODO: Use a more specific pattern to avoid conflicts (e.g., include scope_id or use a structured event type)
    async def on_child_complete(self, event: BlackboardEvent, repl: PolicyREPL) -> EventProcessingResult | None:
        agent_id = event.key.split(":")[0]
        role = None
        for r, aid in self.child_agents.items():
            if aid == agent_id:
                role = r
                break

        if not role:
            return

        logger.info(f"Child {role} ({agent_id}) completed via event")
        result = event.value

        # Critique child work using CriticCapability
        critique = await self.agent.critic_capability.critique_output( # TODO: Remove. Critiquing is now handled inside CriticCapability
            output=result,
            context=CritiqueContext(
                producer_id=agent_id,
                relationship=OutputRelationship.CHILD,
                goal="analyze page cluster and produce findings",
                premises=[],
                evidence=result.get("evidence", {}),
                metadata=result.get("metadata", {})
            )
        )

        # Write critique to blackboard for child to see
        blackboard = await self.get_blackboard()
        await blackboard.write(
            f"{agent_id}:critique",
            critique.model_dump(),
            created_by=self.agent.agent_id
        )

        if critique.requires_revision:
            await blackboard.write(
                f"{agent_id}:revision_request",
                {
                    "critique": critique.model_dump(),
                    "timestamp": time.time()
                },
                created_by=self.agent.agent_id
            )
            logger.info(
                f"Requested revision from child {agent_id}: "
                f"{critique.invalid_conclusions + critique.unsupported_claims}"
            )
            # Don't remove from child_agents yet - wait for revised result
        else:
            # Accept result
            self.cluster_analyses[role] = result
            completed_pages = result.get("analyzed_pages", set())

            # Hook for subclass-specific post-completion logic
            await self.on_cluster_complete(role, agent_id, completed_pages)

            del self.child_agents[role]
            logger.info(
                f"Child {role} ({agent_id}) completed with quality "
                f"{critique.quality_score:.2f}"
            )

        if not self.child_agents:
            await self.synthesize_global_report()
            return EventProcessingResult(
                immediate_action=Action(
                    action_type="synthesize_global_report",
                )
            )
        return None

    @event_handler(pattern="error:*") # TODO: Use a more specific pattern to avoid conflicts (e.g., include scope_id or use a structured event type)
    async def on_child_error(self, event: BlackboardEvent, repl: PolicyREPL) -> EventProcessingResult | None:
        agent_id = event.key.split(":")[1]
        role = None
        for r, aid in self.child_agents.items():
            if aid == agent_id:
                role = r
                break

        if not role:
            return None

        error_data = event.value
        logger.warning(
            f"Child {agent_id} ({role}) escalated error: "
            f"{error_data.get('error_type', 'Unknown')}: "
            f"{error_data.get('error', 'No details')}"
        )

        # TODO: Use LLM to decide whether to retry, skip, or try alternative
        # For now, simple policy: retry once, then skip

        retry_count = error_data.get("context", {}).get("retry_count", 0)
        if retry_count < 1:
            logger.info(f"Retrying child {agent_id}")
            blackboard = await self.get_blackboard()
            await blackboard.delete(f"error:{agent_id}")

            # TODO: In future, use LLM inference to decide best retry strategy
            # For now, just log and let the child continue
            # The child agent is still running and will retry its failed action
        else:
            # Max retries exceeded
            logger.error(
                f"Child {agent_id} ({role}) failed after {retry_count} retries, skipping"
            )
            del self.child_agents[role]
            self.cluster_analyses[role] = {
                "error": error_data.get("error"),
                "error_type": error_data.get("error_type"),
                "failed": True,
            }

            if not self.child_agents:
                await self.synthesize_global_report()
                return EventProcessingResult(
                    immediate_action=Action(
                        action_type="synthesize_global_report",
                    )
                )
        return None

    async def on_cluster_complete(self, role: str, agent_id: str, completed_pages: set[str]) -> None:
        """Hook called when a cluster completes.

        Subclasses can override to add custom logic (e.g., working set updates).

        Args:
            role: Cluster role (e.g., "cluster_0")
            agent_id: Agent ID
            completed_pages: Set of page IDs that were analyzed
        """
        pass  # Base implementation does nothing

    @action_executor()
    async def request_critique_from_peer(self, peer_id: str, my_output: dict, goal: str) -> dict | None:
        """Request critique from a peer agent using event-driven response.

        Delegates to CriticCapability.request_critique_from_peer() for handling.
        """
        critic_capability = self.agent.get_capability_by_type(CriticCapability)
        if critic_capability is None:
            raise RuntimeError(
                "CodeAnalysisCoordinatorCapability requires CriticCapability on the agent. "
            )
        return await critic_capability.request_critique_from_peer(
            peer_id=peer_id,
            my_output=my_output,
            goal=goal,
            timeout=30.0
        )

    @action_executor(writes=["global_report"])
    async def synthesize_global_report(self) -> None:
        """Synthesize global report from cluster analyses.

        TODO: **Multi-Agent Synthesis**
        Synthesis can itself be complex multi-agent orchestration rather than a single LLM call.

        **Impact When Implemented**:

        - Better quality for complex synthesis
        - Hierarchical synthesis for large codebases
        - Collaborative refinement of results
        - Coordinator spawns multiple agents to:
          - Aggregate insights
          - Resolve contradictions
          - Prioritize findings
          - Critique claims
        """
        try:
            # For now, simple LLM synthesis
            # TODO: In production, this could spawn multiple synthesis agents
            synthesis = await self.single_llm_synthesis()

            # TODO: Write to blackboard or return to user
            logger.info(f"Global synthesis complete: {synthesis}")

            return {"global_report": synthesis}

        except Exception as e:
            logger.error(f"Global synthesis failed: {e}", exc_info=True)
            self.agent.state = AgentState.FAILED
            raise

    async def single_llm_synthesis(self) -> dict:
        """Simple single-LLM synthesis (can be extended to multi-agent).

        Attempts to get results from ResultCapability first for cluster-wide visibility,
        falls back to self.cluster_analyses if no results found.
        """
        import json

        # Try to get results from ResultCapability for cluster-wide visibility
        analyses_data = {}
        if hasattr(self, 'result_cap') and self.result_cap:
            partials_result = await self.result_cap.get_partials(filter_type="cluster_analysis")
            partials = partials_result.get("results", [])
            for partial in partials:
                result_id = partial.get("result_id", "unknown")
                analyses_data[result_id] = partial.get("result", {})

        # Fall back to self.cluster_analyses if no results from ResultCapability
        if not analyses_data:
            analyses_data = self.cluster_analyses

        if not analyses_data:
            return {"summary": "No cluster analyses available"}

        # Build synthesis prompt
        analyses_text = "\n\n".join([
            f"**{role}**:\n{json.dumps(analysis, indent=2)}"
            for role, analysis in analyses_data.items()
        ])

        prompt = f"""You are synthesizing a global code analysis report from multiple cluster analyses.

**Cluster Analyses**:
{analyses_text}

**Your Task**: Create a comprehensive report covering:
1. Overall architecture and design patterns
2. Key components and their relationships
3. Potential issues or concerns
4. Recommendations for improvement

**Output Format** (JSON):
{{
    "architecture": {{
        "pattern": "description",
        "main_components": ["list"],
        "relationships": ["how they relate"]
    }},
    "findings": {{
        "strengths": ["list"],
        "issues": ["list"],
        "recommendations": ["list"]
    }}
}}"""

        response = await self.agent.infer(
            context_page_ids=[],  # No pages, just reasoning
            prompt=prompt,
            max_tokens=3000
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"summary": response.text}

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize BaseCodeAnalysisCoordinatorCapability-specific state.

        Overrides base implementation to add coordinator-specific state.
        Calls super() first to get base state, then adds subclass state.

        Returns:
            AgentSuspensionState with all agent state serialized
        """
        # Get base state from parent
        state = await super().serialize_suspension_state(state)

        # Store coordinator-specific state in custom_data with special key
        state.custom_data["_coordinator_state"] = {
            "child_agents": self.child_agents,
            "cluster_analyses": self.cluster_analyses,
            "clusters_spawned": self.clusters_spawned,
        }

        return state

    @override
    async def deserialize_suspension_state(
        self,
        state: AgentSuspensionState
    ) -> None:
        """Restore BaseCodeAnalysisCoordinator-specific state from suspension.

        Overrides base implementation to restore coordinator-specific state.
        Calls super() first to restore base state, then restores subclass state.

        Args:
            state: AgentSuspensionState to restore from
        """
        # Restore base state first
        await super().deserialize_suspension_state(state)

        # Restore coordinator-specific state
        custom_state = state.custom_data.get("_coordinator_state", {})
        if custom_state:
            self.child_agents = custom_state.get("child_agents", {})
            self.cluster_analyses = custom_state.get("cluster_analyses", {})
            self.clusters_spawned = custom_state.get("clusters_spawned", 0)

            logger.info(
                f"Restored BaseCodeAnalysisCoordinator state: "
                f"child_agents={len(self.child_agents)}, "
                f"cluster_analyses={len(self.cluster_analyses)}, "
                f"clusters_spawned={self.clusters_spawned}"
            )


class CodeAnalysisCoordinatorCapability(BaseCodeAnalysisCoordinatorCapability):
    """Capability that coordinates distributed code analysis (cache-oblivious).
    Spawning strategy: Spawns ALL clusters at once (cache-oblivious).
    Use CodeAnalysisCoordinatorV2 for cache-aware coordination.
    """

    @action_executor()
    async def spawn_cluster_analyzers(self) -> None:
        """Spawn ClusterAnalyzer agents for each page cluster."""
        try:
            if not self.agent._manager:
                raise RuntimeError("No manager attached")

            agent_system = get_agent_system()

            from colony.samples.code_analysis.basic.cluster_analyzer import ClusterAnalyzer
            from colony.cluster import LLMClientRequirements

            cluster_count = 0
            blueprints = []
            cluster_metadata_list = []

            page_storage = await self.agent.get_page_storage()
            async for cluster in page_storage.get_all_clusters(
                tenant_id=self.agent.tenant_id,
                group_id=self.agent.group_id,
                max_cluster_size=10,  # TODO: Make configurable
                min_cluster_size=2    # TODO: Make configurable
            ):
                llm_requirements = LLMClientRequirements(
                    num_tokens_context=8192,
                    num_tokens_generation=2000
                ),
                # Create blueprint for this cluster
                bp = ClusterAnalyzer.bind(
                    metadata=AgentMetadata(
                        session_id=self.agent.metadata.session_id,
                        run_id=self.agent.metadata.run_id,
                        parent_agent_id=self.agent.agent_id,
                        parameters={"cluster": cluster.model_dump()},
                    ),
                    bound_pages=cluster.page_ids,
                )
                blueprints.append(bp)
                cluster_metadata_list.append(cluster.model_dump())
                cluster_count += 1

            if not blueprints:
                logger.warning("No clusters found to analyze")
                self.agent.state = AgentState.STOPPED
                return

            # Spawn all cluster analyzers
            # TODO: Pass LLM requirements to spawn_agents
            agent_ids = await agent_system.spawn_agents(
                blueprints=blueprints,
                requirements=llm_requirements,
                soft_affinity=True,
                suspend_agents=False
            )

            # Track children
            for i, agent_id in enumerate(agent_ids):
                role = f"cluster_{cluster_metadata_list[i]['cluster_id']}"
                self.child_agents[role] = agent_id

            self.clusters_spawned = cluster_count

            logger.info(
                f"Spawned {cluster_count} ClusterAnalyzer agents"
            )

        except Exception as e:
            logger.error(f"Failed to spawn cluster analyzers: {e}", exc_info=True)
            self.agent.state = AgentState.FAILED
            raise


class CodeAnalysisCoordinatorV2Capability(BaseCodeAnalysisCoordinatorCapability):
    """Capability for cache-aware coordination with explicit working set management.

    Key differences from V1:
    - Loads page graph and manages global working set
    - Spawns clusters incrementally based on working set overlap
    - Updates working set as clusters complete
    - Records query resolutions for graph learning

    This implements Option 3 from CACHE_COORDINATION_INTEGRATION.md.
    """

    async def initialize(self) -> None:
        """Initialize cache-aware coordinator."""
        await super().initialize()

        # Load page graph from storage
        # TODO: The page graph is dynamic. Should be loaded when needed.
        # await self.agent.load_page_graph()

        job_quota = self.agent.metadata.parameters.get("job_quota", 50)  # Max pages in working set

        # Initialize working set manager
        self.working_set_cap: WorkingSetCapability = await self.agent.get_capability_by_type(WorkingSetCapability)

        # Initialize agent pool capability for lifecycle management
        self.agent_pool_cap: AgentPoolCapability | None = self.agent.get_capability_by_type(AgentPoolCapability)
        if not self.agent_pool_cap:
            self.agent_pool_cap = AgentPoolCapability(agent=self.agent, scope_id=self.scope_id)
            await self.agent_pool_cap.initialize()
            self.agent.add_capability(self.agent_pool_cap)

        # Initialize batching policy for cache-aware batch selection
        # Can be overridden via agent metadata or by subclasses
        batching_policy_config = self.agent.metadata.parameters.get("batching_policy", {})
        self.batching_policy: BatchingPolicy = self._create_batching_policy(batching_policy_config)

        # Initialize result capability for cluster-wide result visibility
        self.result_cap: ResultCapability | None = self.agent.get_capability_by_type(ResultCapability)
        if not self.result_cap:
            self.result_cap = ResultCapability(agent=self.agent, scope_id=self.scope_id)
            await self.result_cap.initialize()
            self.agent.add_capability(self.result_cap)

        # Initialize page graph capability for standardized graph operations
        self.page_graph_cap: PageGraphCapability | None = self.agent.get_capability_by_type(PageGraphCapability)
        if not self.page_graph_cap:
            self.page_graph_cap = PageGraphCapability(agent=self.agent, scope_id=self.scope_id)
            await self.page_graph_cap.initialize()
            self.agent.add_capability(self.page_graph_cap)

        # Track query history for working set updates
        self.query_history: list[dict] = []

        # Track pending cluster specs (not yet spawned)
        self.pending_clusters: list[PageCluster] = []

        # Track spawned clusters by role
        self.spawned_cluster_pages: dict[str, set[str]] = {}  # role → page_ids

        logger.info(f"CodeAnalysisCoordinatorV2 initialized with job_quota={job_quota}")

    def _create_batching_policy(self, config: dict) -> BatchingPolicy:
        """Create batching policy from configuration.

        Args:
            config: Policy configuration dict with keys:
                - type: "hybrid" | "clustering" | "continuous"
                - overlap_threshold: float (for clustering)
                - batch_size: int (max batch size)
                - max_concurrent: int (for continuous)

        Returns:
            Configured BatchingPolicy instance
        """
        policy_type = config.get("type", "hybrid")
        overlap_threshold = config.get("overlap_threshold", self.agent.metadata.parameters.get("overlap_threshold", 0.3))
        batch_size = config.get("batch_size", self.agent.metadata.parameters.get("batch_size", 5))
        max_concurrent = config.get("max_concurrent", 10)

        if policy_type == "clustering":
            return ClusteringBatchPolicy(
                min_overlap=overlap_threshold,
                max_batch_size=batch_size,
            )
        elif policy_type == "continuous":
            return ContinuousBatchPolicy(max_concurrent=max_concurrent)
        else:  # hybrid (default)
            return HybridBatchPolicy(
                clustering_policy=ClusteringBatchPolicy(
                    min_overlap=overlap_threshold,
                    max_batch_size=batch_size,
                ),
                continuous_policy=ContinuousBatchPolicy(max_concurrent=max_concurrent),
                overlap_threshold=overlap_threshold,
            )

    @action_executor()
    async def spawn_cluster_analyzers(self) -> None:
        """Spawn cluster analyzers incrementally based on working set.

        Strategy:
        1. Initialize working set from page graph
        2. Collect all clusters from context page source
        3. Spawn clusters with high working set overlap first
        4. As clusters complete, update working set and spawn next batch
        """
        try:
            # Collect all clusters (but don't spawn yet)
            all_clusters: list[PageCluster] = []
            page_storage = await self.agent.get_page_storage()
            async for cluster in page_storage.get_all_clusters(
                self.agent.tenant_id,
                self.agent.group_id,
                max_cluster_size=self.agent.metadata.parameters.get("max_cluster_size", 10),
                min_cluster_size=self.agent.metadata.parameters.get("min_cluster_size", 2)
            ):
                all_clusters.append(cluster)

            if not all_clusters:
                logger.warning("No clusters found to analyze")
                self.agent.state = AgentState.STOPPED
                return

            logger.info(f"Collected {len(all_clusters)} clusters for analysis")

            # Initialize working set based on page graph
            # The page graph may be empty if not yet built
            page_graph = await self.agent.load_page_graph()
            all_page_ids = list(page_graph.nodes()) if page_graph.number_of_nodes() > 0 else []
            if not all_page_ids:
                # No graph, use pages from first cluster
                all_page_ids = all_clusters[0].page_ids

            await self.working_set_cap.initialize_from_policy(
                page_graph=page_graph,
                available_pages=all_page_ids,
                run_context=RunContext(
                    analysis_goal=self.agent.metadata.parameters.get("goal", "code analysis"),
                    run_id=self.agent.metadata.run_id,
                )
            )

            # Store pending clusters
            self.pending_clusters = all_clusters

            # Spawn first batch based on working set overlap
            await self.spawn_next_batch()

        except Exception as e:
            logger.error(f"Failed to spawn cache-aware cluster analyzers: {e}", exc_info=True)
            self.agent.state = AgentState.FAILED
            raise

    @action_executor()
    async def spawn_next_batch(self) -> None:
        """Spawn next batch of clusters based on working set overlap.

        Uses BatchingPolicy for batch selection and AgentPoolCapability for agent
        lifecycle management while maintaining backward-compatible tracking.
        """
        if not self.pending_clusters:
            logger.info("No more pending clusters to spawn")
            return

        # Get current working set
        working_set_result = await self.working_set_cap.get_working_set()
        working_set = set(working_set_result.get("pages", []))

        # Build context for batching policy
        candidate_pages = {
            cluster.cluster_id: cluster.page_ids
            for cluster in self.pending_clusters
        }

        context = {
            "candidate_pages": candidate_pages,
            "active_count": len(self.child_agents),
        }

        # Use batching policy to select clusters for this batch
        batch_cluster_ids = await self.batching_policy.create_batch(
            candidates=[c.cluster_id for c in self.pending_clusters],
            working_set=working_set,
            context=context,
        )

        # If policy returns empty batch and we have pending clusters, spawn at least one
        if not batch_cluster_ids and self.pending_clusters:
            batch_cluster_ids = [self.pending_clusters[0].cluster_id]

        # Get cluster objects for batch
        cluster_map = {c.cluster_id: c for c in self.pending_clusters}
        to_spawn = [cluster_map[cid] for cid in batch_cluster_ids if cid in cluster_map]

        if not to_spawn:
            logger.warning("No clusters to spawn in this batch")
            return

        # Spawn agents via AgentPoolCapability
        for cluster in to_spawn:
            role = f"cluster_{cluster.cluster_id}"

            result = await self.agent_pool_cap.create_agent(
                agent_type="polymathera.colony.samples.code_analysis.ClusterAnalyzerV2",
                capabilities=["ClusterAnalyzerCapabilityV2"],
                bound_pages=cluster.page_ids,
                metadata=AgentMetadata(
                    session_id=self.agent.metadata.session_id,
                    run_id=self.agent.metadata.run_id,
                    parent_agent_id=self.agent.agent_id,
                    parameters={
                        "cluster": cluster.model_dump(),
                        "query_router_type": "cache_aware",
                        "cache_boost_factor": self.agent.metadata.parameters.get("cache_boost_factor", 1.5),
                    }
                ),
                role=role,
            )

            if result.get("created"):
                agent_id = result["agent_id"]
                # Maintain backward-compatible tracking
                self.child_agents[role] = agent_id
                self.spawned_cluster_pages[role] = set(cluster.page_ids)
                self.clusters_spawned += 1
            else:
                logger.warning(f"Failed to create agent for cluster {cluster.cluster_id}: {result.get('error')}")

        # Remove spawned clusters from pending
        spawned_ids = {c.cluster_id for c in to_spawn}
        self.pending_clusters = [c for c in self.pending_clusters if c.cluster_id not in spawned_ids]

        logger.info(
            f"Spawned batch of {len(to_spawn)} clusters "
            f"(pending: {len(self.pending_clusters)}, "
            f"working set: {len(working_set)} pages)"
        )

    async def on_cluster_complete(self, role: str, agent_id: str, completed_pages: set[str]) -> None:
        """Handle cluster completion with working set updates.

        Args:
            role: Cluster role
            agent_id: Agent ID
            completed_pages: Pages analyzed by this cluster
        """
        # Update working set
        await self.working_set_cap.release_pages(
            page_ids=completed_pages,
        )

        # Spawn next batch if there are pending clusters
        if self.pending_clusters:
            await self.spawn_next_batch()

        logger.info(
            f"Cluster {role} complete. "
            f"Working set updated: {len(self.working_set_cap.working_set)} pages. "
            f"Pending clusters: {len(self.pending_clusters)}"
        )


class BaseCodeAnalysisCoordinator(Agent):
    """Base class for code analysis coordinators.

    Extracts common functionality:
    - Context page source initialization
    - Critique policy management
    - Child agent tracking
    - Event-driven monitoring
    - Global synthesis
    """
    critic_capability: CriticCapability | None = Field(default=None)
    coordinator_capability: BaseCodeAnalysisCoordinatorCapability | None = Field(default=None)

    async def initialize(self) -> None:
        """Initialize coordinator."""
        # Add CriticCapability for critique handling
        # CriticCapability initializes policies from agent metadata
        self.add_capability_blueprints([
            CriticCapability.bind(),
        ])

        await super().initialize()

        self.critic_capability: CriticCapability = self.get_capability(CriticCapability.get_capability_name())



class CodeAnalysisCoordinator(BaseCodeAnalysisCoordinator):
    """Root agent that coordinates distributed code analysis (cache-oblivious).

    Spawning protocol:
    1. Use AgentSystemDeployment to spawn child agents
    2. Communicate via blackboard (EVENT-DRIVEN, not polling!)
    3. Handle agent failures and retries

    Critique protocol:
    - Separate critique policies for each relationship type (self/child/peer/parent)
    - Critique requests/results exchanged via blackboard

    Spawning strategy: Spawns ALL clusters at once (cache-oblivious).
    Use CodeAnalysisCoordinatorV2 for cache-aware coordination.
    """

    async def initialize(self) -> None:
        """Initialize coordinator and attach capability."""
        self.add_capability_blueprints([
            CodeAnalysisCoordinatorCapability.bind(),
        ])
        await super().initialize()

        self.coordinator_capability = self.get_capability(CodeAnalysisCoordinatorCapability.get_capability_name())

    # NOTE: request_critique_from_peer
    # are inherited from BaseCodeAnalysisCoordinator.
    # They use CriticCapability for critique handling.


class CodeAnalysisCoordinatorV2(BaseCodeAnalysisCoordinator):
    """Cache-aware coordinator with explicit working set management.

    Key differences from V1:
    - Loads page graph and manages global working set
    - Spawns clusters incrementally based on working set overlap
    - Updates working set as clusters complete
    - Records query resolutions for graph learning

    This implements Option 3 from CACHE_COORDINATION_INTEGRATION.md.
    """

    async def initialize(self) -> None:
        """Initialize cache-aware coordinator."""
        self.add_capability_blueprints([
            CodeAnalysisCoordinatorV2Capability.bind(),
        ])
        await super().initialize()

        self.coordinator_capability = self.get_capability(CodeAnalysisCoordinatorV2Capability.get_capability_name())



