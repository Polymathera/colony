"""Extremely Long Context Inference Management.

This module provides the PolymatheraCluster class for managing a cluster of Polymathera deployments.
"""

from __future__ import annotations

import logging
from typing import Any, Type, TYPE_CHECKING

from pydantic import ConfigDict, Field

from .distributed import get_polymathera
from .distributed.config import ConfigComponent
from .distributed.state_management import SharedState, StateManager
from .distributed.ray_utils import serving
from .cluster.config import ClusterConfig
from .vcm.config import VCMConfig
from .agents.config import AgentSystemConfig
from .knowledge.cluster_config import KnowledgeConfig
from .agents.blueprint import AgentBlueprint
from . import get_deployment_names

if TYPE_CHECKING:
    from .cluster import LLMClientRequirements


logger = logging.getLogger(__name__)


class PolymatheraClusterConfig(ConfigComponent):
    """Runtime-built configuration for the complete Polymathera system stack.

    NOT a registered config-registry component — operator YAML drives this
    indirectly via the ``cluster.*`` block (legacy ``LLMClusterYAMLConfig``);
    ``polymath.cli.polymath._build_cluster_config`` constructs an instance
    from ``TestConfig.cluster`` at deploy time. The class stays a
    :class:`ConfigComponent` subclass for the Pydantic / validator
    plumbing but does not claim its own YAML path — that avoids the
    historical confusion where ``cluster.app_name`` (the consumed key) and
    ``polymathera_cluster.app_name`` (a phantom registered key that nothing
    read) could disagree.

    ``app_name`` and ``llm_cluster_config`` are ``Optional`` so callers
    can default-instantiate; :meth:`assert_ready_for_deploy` enforces both
    are set before deploy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str | None = None
    llm_cluster_config: ClusterConfig | None = None
    vcm_config: VCMConfig = Field(default_factory=VCMConfig)
    agent_system_config: AgentSystemConfig = Field(default_factory=AgentSystemConfig)
    knowledge_config: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    cleanup_on_init: bool = False

    def assert_ready_for_deploy(self) -> None:
        """Fail fast if any deploy-required field is unset.

        Not named ``validate_*`` so the ``ConfigComponent`` auto-validator does
        not run it at construction time — defaults would always trip it.
        """
        if self.app_name is None:
            raise ValueError("PolymatheraClusterConfig.app_name is required for deploy")
        if self.llm_cluster_config is None:
            raise ValueError(
                "PolymatheraClusterConfig.llm_cluster_config is required for deploy"
            )
        self.llm_cluster_config.validate_config()

    def add_deployments_to_app(self, app: serving.Application, top_level: bool) -> None:
        """Add all Polymathera components to the application.

        This recursively adds:
        - PolymatheraCluster deployment (if not top_level)
        - LLMCluster + VLLMDeployments (recursive via ClusterConfig)
        - VirtualContextManager
        - PhysicalContextManager
        - Agent system (if configured)

        Args:
            app: The serving Application to add deployments to
            top_level: If True, don't deploy PolymatheraCluster itself (it's on the driver)
        """
        # Skip deploying ourselves if we're the top-level
        if not top_level:
            app.add_deployment(
                PolymatheraCluster.bind(config=self, top_level=False),
                name="polymathera_cluster"
            )

        # LLM cluster (recursive - adds LLMCluster + VLLMDeployments)
        self.llm_cluster_config.add_deployments_to_app(app, top_level=False)

        # VCM with config
        self.vcm_config.add_deployments_to_app(app, top_level=False)

        # Knowledge layer — for self-hosted PDF extractor backends
        # this brings up the matching *ExtractorDeployment. Hosted
        # backends are no-ops at deploy time; every worker resolves
        # the active reader from KnowledgeConfig directly via the
        # global ConfigurationManager.
        self.knowledge_config.add_deployments_to_app(app, top_level=False)

        # Agent system deployments
        self.agent_system_config.add_deployments_to_app(app, top_level=False)

        logger.info(f"Added all Polymathera deployments to app '{app.name}' (LLMCluster, VCM, Agent System)")



class PolymatheraClusterState(SharedState):
    """Distributed state for PolymatheraCluster."""

    cluster_status: str = "INITIALIZING"
    additional_info: dict[str, Any] = {}

    @staticmethod
    def get_state_key(app_name: str) -> str:
        """Get the state key for this cluster state."""
        return f"polymathera:serving:{app_name}:cluster:state"



class PolymatheraCluster:
    """Manager for a cluster of Polymathera deployments.

    The PolymatheraCluster provides high-level management of Polymathera components deployed
    using colony.distributed.ray_utils.serving. It handles:

    1. **Cluster Deployment**: Deploy vLLM instances with custom configurations
    2. **Context Management**: Track which pages are loaded in which clients
    3. **Intelligent Routing**: Route requests based on page locality
    4. **Health Monitoring**: Monitor client health and performance
    5. **Statistics**: Collect and report cluster-wide metrics

    Architecture:
    - Uses serving.Application to deploy VLLMDeployment instances
    - Uses ContextAwareRouter for intelligent request routing
    - Uses distributed StateManager for cluster-wide state
    - Provides high-level APIs for inference and page management

    This layer sits between:
    - **Below**: colony.distributed.ray_utils.serving (deployment infrastructure)
    - **Above**: Virtual Context Manager (VCM) - to be implemented

    Example:
        ```python
        from polymathera.colony.system import (
            PolymatheraCluster,
            PolymatheraClusterConfig,
            get_agent_system,
        )

        llm_cluster_config = ClusterConfig()
        vcm_config = VCMConfig()
        agent_system_config = AgentSystemConfig()

        polyconfig = PolymatheraClusterConfig(
            app_name="my_polymathera_cluster",
            llm_cluster_config=llm_cluster_config,
            vcm_config=vcm_config,
            agent_system_config=agent_system_config,
        )

        # Create and deploy cluster
        polycluster = PolymatheraCluster(
            config=polyconfig,
            top_level=True,
        )
        await polycluster.deploy()

        agent_system = await get_agent_system()
        # Use agent_system handle to interact with agents...
        agent_ids = await agent_system.spawn_agent(
            role="research_assistant",
            goals=["Conduct research on topic X", "Summarize findings"],
            capabilities=["web_search", "document_analysis"],
        )
        ```
    """

    def __init__(self, config: PolymatheraClusterConfig, top_level: bool = True) -> None:
        """Initialize LLM cluster.

        Args:
            config: Complete cluster configuration

        Example:
            ```python
            from polymathera.colony.cluster import ClusterConfig, LLMDeploymentConfig

            # Create deployment configs from registry
            llama_8b = LLMDeploymentConfig.from_model_registry(
                model_name="meta-llama/Llama-3.1-8B",
                tensor_parallel_size=2,
                num_replicas=4,
            )
            llama_70b = LLMDeploymentConfig.from_model_registry(
                model_name="meta-llama/Llama-3.1-70B",
                tensor_parallel_size=4,
                num_replicas=2,
            )

            # Create cluster config
            config = ClusterConfig(
                app_name="llm-cluster",
                vllm_deployments=[llama_8b, llama_70b],
            )

            cluster = LLMCluster(config=config, top_level=True)
            ```
        """
        self.config: PolymatheraClusterConfig = config
        self.top_level = top_level

        # Validate all deployment configurations
        self.config.assert_ready_for_deploy()

        # Store app name
        self.app_name = self.config.app_name

        # Will be set during deployment
        self.app: serving.Application | None = None
        self.state_manager: StateManager | None = None

        # Deployment handles (set after deploy())
        self.llm_cluster_handle: serving.DeploymentHandle | None = None
        self.vcm_handle: serving.DeploymentHandle | None = None

    async def cleanup_state_managers(self) -> None:
        """Cleanup all state managers associated with this cluster."""
        # Cleanup state managers for cluster and all deployments
        # Cluster-level state
        if self.state_manager:
            cluster_state_key = PolymatheraClusterState.get_state_key(self.app_name)
            logger.info(f"Cleaning up cluster state: {cluster_state_key}")
            try:
                await self.state_manager.cleanup()
                logger.info(f"Cleaned up cluster state: {cluster_state_key}")
            except Exception as e:
                logger.warning(f"Failed to cleanup cluster state {cluster_state_key}: {e}")

        # TODO: Cleanup LLMCluster and VCM state managers via their deployment handles

    async def deploy(self) -> None:
        """Deploy the LLM cluster.

        This creates a serving.Application with multiple VLLMDeployment instances
        (one per configured model) and starts them with their configured routing policies.
        All deployment parameters are derived from the cluster configuration.
        """
        if not self.top_level:
            raise RuntimeError("System.deploy() can only be called on top-level clusters")

        logger.info(f"Deploying Polymathera cluster '{self.app_name}'")

        # Initialize distributed state managers
        polymathera = get_polymathera()

        # Cluster-level state manager
        if self.state_manager is None:
            cluster_state_key = PolymatheraClusterState.get_state_key(self.app_name)
            self.state_manager = await polymathera.get_state_manager(
                state_type=PolymatheraClusterState,
                state_key=cluster_state_key,
            )

        # Cleanup existing deployments and states if requested
        if self.config.cleanup_on_init:
            logger.info(f"Cleanup requested: removing existing application '{self.app_name}' and its states")

            await serving.Application.cleanup(self.app_name)

            await self.cleanup_state_managers()

            logger.info(f"Cleanup complete for application '{self.app_name}'")

        # Create application
        self.app = serving.Application(name=self.app_name)

        # Use config to recursively add all deployments (VLLMDeployments, LLMCluster, VCM, etc.)
        self.config.add_deployments_to_app(self.app, top_level=True)

        # Start application (triggers all @serving.initialize_deployment methods)
        await self.app.start()

        # Get handles to deployed components for convenience
        self.llm_cluster_handle = await get_llm_cluster(self.app_name)
        self.vcm_handle = await get_vcm(self.app_name)

        logger.info(f"Polymathera cluster '{self.app_name}' deployed successfully")

    async def shutdown(self) -> None:
        """Shutdown the Polymathera cluster.

        This stops the serving application and cleans up resources.
        """
        logger.info(f"Shutting down Polymathera cluster '{self.app_name}'")
        if self.app and self.top_level:
            await self.app.stop()
            self.app = None
            if self.llm_cluster_handle:
                await self.llm_cluster_handle.shutdown()
            if self.vcm_handle:
                await self.vcm_handle.shutdown()

            self.llm_cluster_handle = None
            self.vcm_handle = None
        logger.info(f"Polymathera cluster '{self.app_name}' shut down successfully")




# Re-exported from ``colony._handles``. ALL deployment-handle
# accessors live there now, to break a real circular import cycle
# with ``agents/base.py``'s ``discover_handles`` — see
# ``colony/_handles.py`` header for the full incident write-up.
# Existing callers of ``from colony.system import get_*`` continue
# to work unchanged.
from ._handles import (  # noqa: E402, F401
    _get_deployment_by_name,
    fetch_agent_info,
    get_agent_system,
    get_embedding_deployment,
    get_llm_cluster,
    get_remote_llm_deployment,
    get_session_manager,
    get_standalone_agents,
    get_vcm,
    get_vllm_deployment,
)



async def spawn_agents(
    blueprints: list[AgentBlueprint],
    *,
    requirements: "LLMClientRequirements" | None = None,
    soft_affinity: bool = True,
    suspend_agents: bool = False,
    app_name: str | None = None
) -> list[str]:
    """Spawn agents by calling the agent system.
    This is a convenience wrapper around the agent system's spawn_agents method.
    You do not need an owner agent to spawn new agents with this function. So, it
    can be called from anywhere in the cluster to spawn the root agents.
    But it can also be called by an agent to spawn child agents by specifying
    parent_agent_id in the blueprint's metadata.

    Each blueprint carries its own metadata (session_id, run_id,
    max_iterations, etc.) — no separate threading of these values.

    Args:
        blueprints: List of AgentBlueprint defining agents to spawn
        requirements: Optional LLMClientRequirements to apply to all agents
        soft_affinity: Whether to use soft affinity for agent placement
        suspend_agents: Whether to suspend existing agents to make room
        app_name: The `serving.Application` name where the agent system resides.
            This is required when `spawn_agents` is called from outside any
            `serving.deployment`.

    Returns:
        List of spawned agent IDs
    """
    agent_system = await get_agent_system(app_name)
    agent_ids = await agent_system.spawn_agents(
        blueprints=blueprints,
        requirements=requirements,
        soft_affinity=soft_affinity,
        suspend_agents=suspend_agents,
    )
    return agent_ids


