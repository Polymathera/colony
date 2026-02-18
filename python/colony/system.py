"""Extremely Long Context Inference Management.

This module provides the PolymatheraCluster class for managing a cluster of Polymathera deployments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .distributed import get_polymathera
from .distributed.state_management import SharedState, StateManager
from .distributed.ray_utils import serving
from .cluster.config import ClusterConfig
from .vcm.config import VCMConfig
from .agents.config import AgentSystemConfig
from .agents.models import AgentSpawnSpec
from . import get_deployment_names


logger = logging.getLogger(__name__)


@dataclass
class PolymatheraClusterConfig:
    """Configuration for the complete Polymathera system stack.

    Includes LLM cluster, VCM, and optional agent system.
    """

    app_name: str
    llm_cluster_config: ClusterConfig
    vcm_config: VCMConfig = field(default_factory=VCMConfig)
    agent_system_config: AgentSystemConfig = field(default_factory=AgentSystemConfig)
    cleanup_on_init: bool = False

    def validate_config(self) -> None:
        """Validate all sub-configurations."""
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

        agent_system = get_agent_system()
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
        self.config.validate_config()

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
        self.llm_cluster_handle = get_llm_cluster(self.app_name)
        self.vcm_handle = get_vcm(self.app_name)

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




def _get_deployment_by_name(name_attr: str, app_name: str | None = None) -> serving.DeploymentHandle:
    """Get agent system deployment via serving."""
    try:
        names = get_deployment_names()
        handle = serving.get_deployment(
            app_name or serving.get_my_app_name(),
            getattr(names, name_attr)
        )
        logger.info(f"Connected to {name_attr} deployment: {getattr(names, name_attr)}")
        return handle
    except Exception as e:
        logger.error(f"{name_attr} deployment '{getattr(names, name_attr)}' not found: {e}")
        raise e


def get_agent_system(app_name: str | None = None) -> serving.DeploymentHandle:
    """Get agent system deployment via serving."""
    return _get_deployment_by_name("agent_system", app_name)


def get_llm_cluster(app_name: str | None = None) -> serving.DeploymentHandle:
    """Get LLM deployment via serving."""
    return _get_deployment_by_name("llm_cluster", app_name)


def get_tool_manager(app_name: str | None = None) -> serving.DeploymentHandle:
    """Get tool manager deployment via serving."""
    return _get_deployment_by_name("tool_manager", app_name)


def get_vcm(app_name: str | None = None) -> serving.DeploymentHandle:
    """Get VCM deployment via serving."""
    return _get_deployment_by_name("vcm", app_name)


def get_standalone_agents(app_name: str | None = None) -> serving.DeploymentHandle:
    """Get standalone agents deployment via serving."""
    return _get_deployment_by_name("standalone_agents", app_name)


def get_session_manager(app_name: str | None = None) -> serving.DeploymentHandle:
    """Get session manager deployment via serving."""
    return _get_deployment_by_name("session_manager", app_name)


def get_vllm_deployment(deployment_name: str, app_name: str | None = None) -> serving.DeploymentHandle:
    """Get specific VLLM deployment via serving."""
    try:
        handle = serving.get_deployment(
            app_name or serving.get_my_app_name(),
            deployment_name
        )
        logger.info(f"Connected to VLLM deployment: {deployment_name}")
        return handle
    except Exception as e:
        logger.error(f"VLLM deployment '{deployment_name}' not found: {e}")
        raise e


def get_embedding_deployment(app_name: str | None = None) -> serving.DeploymentHandle:
    return _get_deployment_by_name("embedding", app_name)



async def spawn_agents(
    agent_specs: list[AgentSpawnSpec],
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    soft_affinity: bool = True,
    suspend_agents: bool = False,
    app_name: str | None = None
) -> list[str]:
    """Spawn agents by calling the agent system.
    This is a convenience wrapper around the agent system's spawn_agents method.
    You do not need an owner agent to spawn new agents with this function. So, it
    can be called from anywhere in the cluster to spawn the root agents.
    But it can also be called by an agent to spawn child agents by specifying
    parent_agent_id in the AgentSpawnSpec.metadata.

    Args:
        agent_specs: List of AgentSpawnSpec defining agents to spawn
        session_id: Optional session ID to associate with spawned agents
        run_id: Optional run ID to associate with spawned agents
        soft_affinity: Whether to use soft affinity for agent placement
        suspend_agents: Whether to start agents in suspended state
        app_name: The `serving.Application` name where the agent system resides.
            This is required when `spawn_agents` is called from outside any
            `serving.deployment`.

    Returns:
        List of spawned agent IDs
    """
    agent_system = get_agent_system(app_name)
    agent_ids = await agent_system.spawn_agents(
        agent_specs=agent_specs,
        session_id=session_id,
        run_id=run_id,
        soft_affinity=soft_affinity,
        suspend_agents=suspend_agents,
    )
    return agent_ids


