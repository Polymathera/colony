"""Agent System Deployment for global agent management.

The AgentSystemDeployment provides:
- Global view of all agents across all replicas
- Agent discovery and routing
- Agent lifecycle coordination with VCM
- Monitoring and statistics

NOTE: This deployment does NOT handle message passing.
Messages are handled via direct messaging (communication.py) and blackboard.

This deployment maintains distributed state via StateManager.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING
import uuid

from pydantic import Field

from ..distributed import get_polymathera
from ..distributed.state_management import SharedState, StateManager
from ..distributed.ray_utils import serving
from .base import Agent, AgentState, ResourceExhausted
from .blueprint import AgentBlueprint
from .models import AgentRegistrationInfo
from .models import (
    AgentMetadata,
    AgentResourceRequirements,
    AgentSuspensionState,
    ResourceExhaustedConfig,
    ResourceExhaustedStrategy,
)
if TYPE_CHECKING:
    from ..cluster import LLMClientRequirements


logger = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    """Raised when tenant quota is exceeded."""
    pass


class AgentSystemState(SharedState):
    """Global agent registry - READ-MOSTLY, rarely written.

    This state is for:
    - Agent discovery (where is agent X?)
    - Page-to-agent mapping (which agents are bound to page Y?)
    - Capability-based agent discovery
    - Monitoring and statistics

    This state is NOT for:
    - Message passing (use blackboard or direct channels)
    - Frequent updates (use local state)

    Attributes:
        agents: All agents in the system (agent_id -> AgentRegistrationInfo)
        agent_locations: Agent to deployment mapping (agent_id -> deployment_id)
        deployment_names: Deployment name mapping (deployment_id -> deployment_name)
        page_agents: Page-bound agents (page_id -> list of agent_ids)
        agents_by_type: Index for finding agents by type
        agents_by_capability: Index for finding agents by capability
        stats: System-wide statistics
    """

    # Agent registry (updated only on spawn/terminate)
    agents: dict[str, AgentRegistrationInfo] = Field(default_factory=dict)
    agent_locations: dict[str, str] = Field(default_factory=dict)  # agent_id -> deployment_replica_id
    deployment_names: dict[str, str] = Field(default_factory=dict)  # deployment_replica_id -> deployment_name

    # Page bindings (updated only on bind/unbind)
    page_agents: dict[str, list[str]] = Field(default_factory=dict)  # page_id -> agent_ids

    # Capability indices (for discovery)
    agents_by_type: dict[str, list[str]] = Field(default_factory=dict)  # type -> agent_ids
    agents_by_capability: dict[str, list[str]] = Field(
        default_factory=dict
    )  # capability -> agent_ids

    # Statistics (updated periodically, not per-message!)
    stats: dict[str, Any] = Field(default_factory=dict)

    # Blackboard scope registry: scope_id -> {backend_type, registered_at, ...}
    # Populated by EnhancedBlackboard.initialize() via register_blackboard_scope()
    blackboard_scopes: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def get_state_key(cls, app_name: str) -> str:
        """Generate state key for this agent system."""
        return f"polymathera:serving:{app_name}:agents:system"


@serving.deployment
class AgentSystemDeployment:
    """Global agent manager - tracks all agents across all replicas.

    Provides:
    - Global view of all agents
    - Agent discovery and routing
    - Agent lifecycle coordination with VCM
    - Monitoring and statistics

    Does NOT provide:
    - Message passing (use communication layer)
    """

    def __init__(
        self,
        max_retries: int = 3,
        resource_exhausted_config: ResourceExhaustedConfig | None = None,
        blackboard_backend_type: str = "redis",
    ):
        """Initialize agent system deployment.

        Args:
            max_retries: Maximum retries for general operations (deprecated - use resource_exhausted_config)
            resource_exhausted_config: Configuration for handling ResourceExhausted errors
            blackboard_backend_type: Default backend for blackboards ("distributed", "redis", "memory")
        """
        # Initialized in initialize()
        self.max_retries = max_retries  # Legacy parameter, kept for backwards compatibility
        self.resource_exhausted_config = resource_exhausted_config or ResourceExhaustedConfig()
        self.blackboard_backend_type = blackboard_backend_type
        self.app_name: str | None = None
        self.state_manager: StateManager[AgentSystemState] | None = None
        self.vcm_handle = None
        self.session_manager_handle = None

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize self-contained state (state managers, background tasks).

        Cross-deployment handle discovery (VCM, SessionManager) is deferred
        to on_ready() via @on_app_ready.
        """
        # Get app name from environment
        self.app_name = serving.get_my_app_name()
        logger.info(f"Initializing AgentSystemDeployment for app {self.app_name}")

        # Get Polymathera for state management
        polymathera = get_polymathera()

        # Initialize StateManager
        self.state_manager = await polymathera.get_state_manager(
            state_type=AgentSystemState,
            state_key=AgentSystemState.get_state_key(self.app_name),
        )

        logger.info("AgentSystemDeployment initialized (awaiting app ready for handle discovery)")

    @serving.on_app_ready
    async def on_ready(self):
        """Discover sibling deployment handles after all deployments are started."""
        from ..system import (
            get_vcm,
            get_session_manager,
        )

        self.vcm_handle = get_vcm()
        try:
            self.session_manager_handle = get_session_manager()
        except Exception as e:
            logger.warning(f"SessionManager deployment not found: {e}")
            # This is optional - don't fail initialization

        # Start background task for resource-driven resumption
        self._resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        logger.info("Started resource monitor loop for agent resumption")

        logger.info("AgentSystemDeployment handle discovery complete")

    # === Agent Discovery ===

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def list_all_agents(self) -> list[str]:
        """List all agent IDs in the system.

        Returns:
            List of agent IDs
        """
        async for state in self.state_manager.read_transaction():
            return list(state.agents.keys())

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def get_agent_info(self, agent_id: str) -> AgentRegistrationInfo | None:
        """Get agent information.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent if exists, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            return state.agents.get(agent_id)

    @serving.endpoint
    async def get_agent_location(self, agent_id: str) -> str | None:
        """Get deployment location of an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Deployment ID (format: "app_name.deployment_name") or None
        """
        async for state in self.state_manager.read_transaction():
            return state.agent_locations.get(agent_id)

    @serving.endpoint
    async def find_agents_by_page(self, page_id: str) -> list[str]:
        """Find all agents bound to a specific page.

        Args:
            page_id: Page identifier

        Returns:
            List of agent IDs
        """
        async for state in self.state_manager.read_transaction():
            return state.page_agents.get(page_id, []).copy()

    @serving.endpoint
    async def find_agents_by_type(self, agent_type: str) -> list[str]:
        """Find all agents of a specific type.

        Args:
            agent_type: Agent type

        Returns:
            List of agent IDs
        """
        async for state in self.state_manager.read_transaction():
            return state.agents_by_type.get(agent_type, []).copy()

    @serving.endpoint
    async def find_agents_by_capability(self, capability: str) -> list[str]:
        """Find all agents with a specific capability.

        Args:
            capability: Capability name

        Returns:
            List of agent IDs
        """
        async for state in self.state_manager.read_transaction():
            return state.agents_by_capability.get(capability, []).copy()

    # === Agent Spawning (Entry Point) ===
    async def _start_agent(
        self,
        deployment_handle: serving.DeploymentHandle,
        blueprint: AgentBlueprint,
        suspend_agents: bool,
        expected_agent_id: str | None = None
    ) -> str:
        """Helper to start an agent on a specific deployment handle."""
        try:
            spawned_id = await deployment_handle.start_agent(
                blueprint,
                suspend_agents=suspend_agents,
            )
            if expected_agent_id is not None and spawned_id != expected_agent_id:
                raise ValueError(f"Spawned agent ID {spawned_id} does not match expected ID {expected_agent_id}")

            logger.info(f"Spawned agent {spawned_id} on deployment {deployment_handle.deployment_name}")
            return spawned_id
        except ResourceExhausted as e:
            logger.warning(f"ResourceExhausted when spawning agent with blueprint {blueprint}: {e}")
            raise

    @serving.endpoint
    async def spawn_from_blueprint(
        self,
        blueprint: AgentBlueprint,
        *,
        requirements: LLMClientRequirements | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
    ) -> str:
        """Spawn a single agent from a blueprint with deployment routing.

        The blueprint is the single source of truth for the agent's identity:
        agent_id, session_id, run_id, max_iterations all live in the blueprint.
        This method only handles true deployment concerns: quota checking
        and routing to the correct deployment.

        If agent_id is not in the blueprint, one is generated.

        Args:
            blueprint: AgentBlueprint with class, constructor args, and metadata
            requirements: LLM deployment requirements for routing (LLMClientRequirements)
            soft_affinity: If True, allows spawning on replicas without all pages
            suspend_agents: If True, replica may suspend existing agents to make room

        Returns:
            Spawned agent ID
        """
        from ..system import (
            get_llm_cluster,
            get_standalone_agents,
            get_vllm_deployment
        )

        # Ensure agent_id is in the blueprint
        if "agent_id" not in blueprint.kwargs:
            blueprint.kwargs["agent_id"] = f"agent-{uuid.uuid4().hex[:8]}"
        agent_id = blueprint.agent_id

        # Quota check — metadata and resource_requirements are in the blueprint
        if blueprint.metadata.tenant_id:
            try:
                await self._check_tenant_quota(
                    blueprint.metadata.tenant_id, blueprint.resource_requirements
                )
            except QuotaExceeded as e:
                logger.error(f"Quota exceeded for tenant {blueprint.metadata.tenant_id}: {e}")
                raise

        # Route based on deployment affinity
        has_affinity = blueprint.has_deployment_affinity() or (requirements is not None)

        if has_affinity:
            llm_cluster_handle = get_llm_cluster()
            # Use LLMCluster to select deployment based on requirements
            try:
                vllm_deployment_name = await llm_cluster_handle.select_deployment(
                    requirements=requirements
                )

                vllm_handle = get_vllm_deployment(vllm_deployment_name)
                spawned_id = await self._start_agent(
                    vllm_handle,
                    blueprint,
                    suspend_agents=suspend_agents,
                    expected_agent_id=agent_id,
                )
                logger.info(
                    f"Spawned page-affinity agent {spawned_id} on {vllm_deployment_name} "
                    f"(pages={len(blueprint.bound_pages)}, requirements={'set' if requirements else 'none'})"
                )
                return spawned_id

            except ResourceExhausted as e:
                # Handle resource exhaustion with strategy loop
                logger.warning(
                    f"ResourceExhausted for VLLM agent {agent_id}: {e}"
                )

                spawned_id = await self._handle_resource_exhausted_vllm(
                    blueprint=blueprint,
                    vllm_handle=vllm_handle,
                    soft_affinity=soft_affinity,
                    suspend_agents=suspend_agents,
                )
                logger.info(
                    f"Spawned VLLM agent {spawned_id} after handling ResourceExhausted"
                )
                return spawned_id

            except Exception as e:
                logger.error(
                    f"Failed to spawn agent {agent_id} on VLLM deployment: {e}"
                )
                raise

        else:
            standalone_handle = get_standalone_agents()
            try:
                spawned_id = await self._start_agent(
                    standalone_handle,
                    blueprint,
                    suspend_agents=suspend_agents,
                    expected_agent_id=agent_id,
                )
                logger.info(f"Spawned standalone agent {spawned_id}")
                return spawned_id

            except ResourceExhausted as e:
                if not self.resource_exhausted_config.enable_auto_scaling:
                    logger.error(f"ResourceExhausted for agent {agent_id} but auto-scaling disabled")
                    raise

                logger.warning(f"ResourceExhausted for agent {agent_id}: {e}")
                spawned_id = await self._handle_resource_exhausted_standalone(
                    blueprint=blueprint,
                    standalone_handle=standalone_handle,
                    suspend_agents=suspend_agents,
                )
                logger.info(f"Spawned standalone agent {spawned_id} after handling ResourceExhausted")
                return spawned_id

            except Exception as e:
                logger.error(f"Failed to spawn agent {agent_id} on standalone deployment: {e}")
                raise

    @serving.endpoint
    async def spawn_agents(
        self,
        blueprints: list[AgentBlueprint],
        *,
        requirements: LLMClientRequirements | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
    ) -> list[str]:
        """Spawn multiple agents from blueprints.

        Convenience method that calls spawn_from_blueprint for each blueprint.
        Each blueprint carries its own metadata (session_id, run_id, etc.).

        Args:
            blueprints: List of AgentBlueprint defining agents to spawn
            requirements: Optional LLMClientRequirements to apply to all agents
            soft_affinity: If True, allows spawning on replicas without all pages
            suspend_agents: If True, replica may suspend existing agents to make room

        Returns:
            List of spawned agent IDs
        """
        agent_ids = []
        for i, bp in enumerate(blueprints):
            spawned_id = await self.spawn_from_blueprint(
                blueprint=bp,
                requirements=requirements,
                soft_affinity=soft_affinity,
                suspend_agents=suspend_agents,
            )
            agent_ids.append(spawned_id)

        logger.info(f"Spawned {len(agent_ids)} agents")
        return agent_ids

    @serving.endpoint
    async def stop_agent(self, agent_id: str) -> None:
        """Stop an agent.

        Args:
            agent_id: Agent identifier
        """
        replica_id = await self.get_agent_location(agent_id)
        if not replica_id:
            logger.warning(f"Agent {agent_id} not found")
            return
        deployment_name = self.deployment_names.get(replica_id)
        if not deployment_name:
            logger.warning(f"Deployment name not found for {replica_id}")
            return
        handle = serving.get_deployment(self.app_name, deployment_name)
        await handle.stop_agent(agent_id)
        logger.info(f"Stopped agent {agent_id} on {deployment_name}")

    # === Agent Registration ===

    @serving.endpoint
    async def register_agent(self, agent_info: AgentRegistrationInfo, deployment_replica_id: str, deployment_name: str) -> None:
        """Register a new agent.

        Args:
            agent_info: Lightweight agent registration info (serializable)
            deployment_replica_id: Deployment replica hosting this agent (format: "app_name.deployment_name#replica_id")
            deployment_name: Deployment name
        """
        async for state in self.state_manager.write_transaction():
            state.agents[agent_info.agent_id] = agent_info
            state.agent_locations[agent_info.agent_id] = deployment_replica_id
            if deployment_replica_id in state.deployment_names:
                if state.deployment_names[deployment_replica_id] != deployment_name:
                    logger.warning(f"Deployment name mismatch for {deployment_replica_id}: {state.deployment_names[deployment_replica_id]} != {deployment_name}")
            state.deployment_names[deployment_replica_id] = deployment_name
            # Register page bindings
            for page_id in agent_info.bound_pages:
                if page_id not in state.page_agents:
                    state.page_agents[page_id] = []
                if agent_info.agent_id not in state.page_agents[page_id]:
                    state.page_agents[page_id].append(agent_info.agent_id)

            # Register in type index
            if agent_info.agent_type not in state.agents_by_type:
                state.agents_by_type[agent_info.agent_type] = []
            if agent_info.agent_id not in state.agents_by_type[agent_info.agent_type]:
                state.agents_by_type[agent_info.agent_type].append(agent_info.agent_id)

            # Register capabilities for discovery
            for capability in agent_info.capability_names:
                if capability not in state.agents_by_capability:
                    state.agents_by_capability[capability] = []
                if agent_info.agent_id not in state.agents_by_capability[capability]:
                    state.agents_by_capability[capability].append(agent_info.agent_id)

            logger.info(f"Registered agent {agent_info.agent_id} on deployment replica {deployment_replica_id}")

        # Update tenant resource usage
        if self.session_manager_handle:
            try:
                # Increment resource usage (update via SessionManager)
                await self.session_manager_handle.increment_tenant_resources(
                    cpu_cores=agent_info.resource_requirements.cpu_cores,
                    memory_mb=agent_info.resource_requirements.memory_mb,
                    gpu_cores=agent_info.resource_requirements.gpu_cores,
                    gpu_memory_mb=agent_info.resource_requirements.gpu_memory_mb,
                )
                logger.debug(f"Updated tenant {agent_info.tenant_id} resource usage after registering agent {agent_info.agent_id}")
            except Exception as e:
                logger.warning(f"Failed to update tenant resource usage for {agent_info.tenant_id}: {e}")

    @serving.endpoint
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.

        Args:
            agent_id: Agent identifier
        """
        # Save agent info before removing from state
        agent = None
        async for state in self.state_manager.write_transaction():
            if agent_id in state.agents:
                agent = state.agents[agent_id]

                # Remove page bindings
                for page_id in agent.bound_pages:
                    if page_id in state.page_agents:
                        if agent_id in state.page_agents[page_id]:
                            state.page_agents[page_id].remove(agent_id)
                        if not state.page_agents[page_id]:
                            del state.page_agents[page_id]

                # Remove from type index
                if agent.agent_type in state.agents_by_type:
                    if agent_id in state.agents_by_type[agent.agent_type]:
                        state.agents_by_type[agent.agent_type].remove(agent_id)
                    if not state.agents_by_type[agent.agent_type]:
                        del state.agents_by_type[agent.agent_type]

                # Remove from capability indices
                for capability in agent.capability_names:
                    if capability in state.agents_by_capability:
                        if agent_id in state.agents_by_capability[capability]:
                            state.agents_by_capability[capability].remove(agent_id)
                        if not state.agents_by_capability[capability]:
                            del state.agents_by_capability[capability]

                # Remove from main registries
                del state.agents[agent_id]
                state.agent_locations.pop(agent_id, None)

                logger.info(f"Unregistered agent {agent_id}")

        # Update tenant resource usage if agent had tenant_id
        if agent and agent.tenant_id and self.session_manager_handle:
            syscontext = serving.require_execution_context()
            if syscontext.tenant_id != agent.tenant_id:
                raise ValueError(
                    f"Tenant ID mismatch when unregistering agent {agent_id}: "
                    f"syscontext tenant {syscontext.tenant_id} vs agent tenant {agent.tenant_id}"
                )
            try:
                # Decrement resource usage (update via SessionManager)
                await self.session_manager_handle.decrement_tenant_resources(
                    cpu_cores=agent.resource_requirements.cpu_cores,
                    memory_mb=agent.resource_requirements.memory_mb,
                    gpu_cores=agent.resource_requirements.gpu_cores,
                    gpu_memory_mb=agent.resource_requirements.gpu_memory_mb,
                )
                logger.debug(f"Updated tenant {agent.tenant_id} resource usage after unregistering agent {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to update tenant resource usage for {agent.tenant_id}: {e}")

    @serving.endpoint
    async def update_agent_state(self, agent_id: str, new_state: AgentState) -> None:
        """Update agent state.

        When state transitions to SUSPENDED or STOPPED, agent location is removed
        because suspended agents are deleted from replicas.

        Args:
            agent_id: Agent identifier
            new_state: New state
        """
        async for state in self.state_manager.write_transaction():
            if agent_id in state.agents:
                state.agents[agent_id].state = new_state
                logger.debug(f"Updated agent {agent_id} state to {new_state}")

                # Remove location when agent is suspended or stopped
                # Suspended agents are deleted from replicas, so no longer "owned"
                if new_state in (AgentState.SUSPENDED, AgentState.STOPPED):
                    if agent_id in state.agent_locations:
                        old_location = state.agent_locations[agent_id]
                        del state.agent_locations[agent_id]
                        logger.info(
                            f"Removed agent {agent_id} from location tracking "
                            f"(was on {old_location}, now {new_state})"
                        )

    # === Agent Resumption ===

    @serving.endpoint
    async def resume_agent(self, agent_id: str) -> str:
        """Resume a suspended agent.

        This method:
        1. Loads suspension state from StateManager
        2. Builds AgentBlueprint from suspension state
        3. Spawns agent via spawn_agents() with soft_affinity=True (uses SoftPageAffinityRouter)
        4. Returns the new agent ID (may differ from suspended_agent_id if collision)

        Args:
            agent_id: ID of suspended agent to resume

        Returns:
            New agent ID (agent is spawned fresh, not restarted in place)

        Raises:
            ValueError: If suspension state not found or invalid
            ResourceExhausted: If no replica has capacity even after suspension

        Example:
            ```python
            # Resume suspended agent (routes to replica with best page affinity)
            new_agent_id = await agent_system.resume_agent("agent-abc123")
            ```
        """
        # 1. Load suspension state from StateManager
        polymathera = get_polymathera()
        state_key = AgentSuspensionState.get_state_key(self.app_name, agent_id)

        try:
            state_manager = await polymathera.get_state_manager(
                state_type=AgentSuspensionState,
                state_key=state_key,
            )

            suspension_state: AgentSuspensionState | None = None
            async for state in state_manager.read_transaction():
                suspension_state = state
                break

            if not suspension_state:
                raise ValueError(
                    f"No suspension state found for agent {agent_id}. "
                    f"Agent may not be suspended or state was lost."
                )

            if not suspension_state.agent_id:
                raise ValueError(f"Invalid suspension state for {agent_id}: missing agent_id")

        except Exception as e:
            logger.error(f"Failed to load suspension state for {agent_id}: {e}")
            raise

        # 2. Build AgentBlueprint from suspension state
        # NOTE: agent_type will be added to AgentSuspensionState in Phase 2
        # For now, we'll need to get it from the agent registry or use a placeholder
        agent_type = getattr(suspension_state, 'agent_type', None)
        if not agent_type:
            # Fallback: try to get from agent registry
            async for system_state in self.state_manager.read_transaction():
                if agent_id in system_state.agents:
                    agent_type = system_state.agents[agent_id].agent_type
                    break

        if not agent_type:
            raise ValueError(
                f"Cannot resume agent {agent_id}: agent_type not found in suspension state or registry. "
                f"This may indicate that Phase 2 (adding agent_type to AgentSuspensionState) is incomplete."
            )

        # Build metadata for resumed agent
        metadata = AgentMetadata(
            parameters={
                "resuming_from_suspension": True,
                "suspended_agent_id": agent_id,
                "suspension_reason": suspension_state.suspension_reason,
                "suspension_count": suspension_state.suspension_count,
            }
        )

        # Add parent/role info if present (for child agents)
        if suspension_state.parent_agent_id:
            metadata.parent_agent_id = suspension_state.parent_agent_id
        if suspension_state.role:
            metadata.role = suspension_state.role

        # Build resource requirements
        resource_requirements = AgentResourceRequirements(
            cpu_cores=suspension_state.allocated_cpu_cores,
            memory_mb=suspension_state.allocated_memory_mb,
            gpu_cores=suspension_state.allocated_gpu_cores,
            gpu_memory_mb=suspension_state.allocated_gpu_memory_mb,
        )

        # Resolve agent class from type string, then build blueprint
        agent_class = self._resolve_agent_class(agent_type)
        bp = agent_class.bind(
            agent_type=agent_type,
            bound_pages=suspension_state.bound_pages,
            resource_requirements=resource_requirements,
            metadata=metadata,
        )

        # 3. Spawn via spawn_from_blueprint() with soft_affinity=True
        # This will route to replica with best page affinity via SoftPageAffinityRouter
        try:
            new_agent_id = await self.spawn_from_blueprint(
                blueprint=bp,
                soft_affinity=True,  # Enable soft page affinity routing
                suspend_agents=True,  # Allow suspension of other agents if needed
            )

            logger.info(
                f"Resumed agent {agent_id} as {new_agent_id} "
                f"(reason: {suspension_state.suspension_reason}, "
                f"bound_pages: {len(suspension_state.bound_pages)}, "
                f"working_set: {len(suspension_state.working_set_pages)})"
            )

            # 4. Clean up suspension state (agent successfully resumed)
            # NOTE: The agent's initialize() will clean it up after successful restoration
            # via Agent._restore_from_suspension() which calls deserialize_suspension_state()

            return new_agent_id

        except Exception as e:
            logger.error(
                f"Failed to resume agent {agent_id}: {e}",
                exc_info=True
            )
            raise

    # === Integration with VCM ===

    @serving.endpoint
    async def on_page_loaded(self, page_id: str, replica_id: str) -> None:
        """Callback when VCM loads a page - notify bound agents.

        ATOMIC: Single write transaction, no ABA problem.

        Args:
            page_id: Page identifier
            replica_id: Replica where page was loaded
        """
        async for state in self.state_manager.write_transaction():
            # Get agents bound to this page
            agent_ids = state.page_agents.get(page_id, [])

            # Update all in one transaction
            updated_count = 0
            for agent_id in agent_ids:
                if agent_id in state.agents:
                    state.agents[agent_id].state = AgentState.LOADED
                    updated_count += 1

            logger.info(
                f"Page {page_id} loaded on {replica_id} - updated {updated_count} agents to LOADED"
            )

    @serving.endpoint
    async def on_page_unloaded(self, page_id: str) -> None:
        """Callback when VCM unloads a page - update bound agents.

        ATOMIC: Single write transaction, no ABA problem.

        Args:
            page_id: Page identifier
        """
        async for state in self.state_manager.write_transaction():
            # Get agents bound to this page
            agent_ids = state.page_agents.get(page_id, [])

            # Update all in one transaction
            updated_count = 0
            for agent_id in agent_ids:
                if agent_id in state.agents:
                    state.agents[agent_id].state = AgentState.UNLOADED
                    updated_count += 1

            logger.info(f"Page {page_id} unloaded - updated {updated_count} agents to UNLOADED")

    # === Resource-Aware Replica Selection (P1) ===

    @serving.endpoint
    async def get_all_replica_resources(
        self,
        deployment_name: str
    ) -> dict[str, dict[str, Any]]:
        """Query all replicas of a deployment for their resource usage.

        Args:
            deployment_name: Name of the deployment to query

        Returns:
            Dictionary mapping replica_id to resource usage dict
        """
        try:
            # Get deployment handle through proper abstraction
            handle = serving.get_deployment(self.app_name, deployment_name)

            # Query resource_usage property from all replicas
            replica_resources = await handle.get_all_replica_property("resource_usage")

            return replica_resources
        except Exception as e:
            logger.error(f"Failed to query replicas of {deployment_name}: {e}")
            return {}

    @serving.endpoint
    async def select_replica_for_agent(
        self,
        deployment_name: str,
        resource_requirements: AgentResourceRequirements
    ) -> str | None:
        """Select best replica for spawning an agent (resource-aware).

        Strategy:
        1. Query all replicas for resource usage
        2. Filter replicas with sufficient resources
        3. Select replica with most available resources (best-fit)
        4. Return None if no replica has capacity (triggers scale-up)

        Args:
            deployment_name: Deployment to select replica from
            resource_requirements: Resource requirements for the agent

        Returns:
            replica_id if capacity found, None if all replicas full
        """
        # Get resource usage from all replicas
        replica_resources = await self.get_all_replica_resources(deployment_name)

        if not replica_resources:
            logger.warning(f"No resource information available for {deployment_name}")
            return None

        # Find replicas with sufficient resources
        viable_replicas = []
        for replica_id, usage in replica_resources.items():
            # Check if replica has capacity
            agents_available = usage["max_agents"] - usage["agents"]
            cpu_available = usage.get("cpu_cores_available", 0)
            memory_available = usage.get("memory_mb_available", 0)

            if (agents_available >= 1 and
                cpu_available >= resource_requirements.cpu_cores and
                memory_available >= resource_requirements.memory_mb):

                # Calculate "fitness" score (higher = more available resources)
                fitness = (
                    agents_available * 100 +  # Prioritize agent count
                    cpu_available * 10 +
                    memory_available / 100
                )

                viable_replicas.append((replica_id, fitness, usage))

        if not viable_replicas:
            logger.info(
                f"No replica of {deployment_name} has sufficient resources for agent "
                f"(needs: {resource_requirements.cpu_cores} CPU, {resource_requirements.memory_mb}MB)"
            )
            return None

        # Select replica with highest fitness (most available resources)
        viable_replicas.sort(key=lambda x: x[1], reverse=True)
        best_replica_id, _, best_usage = viable_replicas[0]

        logger.info(
            f"Selected replica {best_replica_id} of {deployment_name} "
            f"(agents: {best_usage['agents']}/{best_usage['max_agents']}, "
            f"cpu: {best_usage.get('cpu_cores_used', 0):.2f}/{best_usage.get('cpu_cores_max', 0):.2f})"
        )

        return best_replica_id

    # === Class Resolution ===

    @staticmethod
    def _resolve_agent_class(agent_type: str) -> type[Agent]:
        """Resolve an Agent subclass from a type string (for resume path).

        Supports fully qualified names ('mypackage.agents.CodeAnalyzer')
        and simple names searched in standard packages.
        """
        import importlib

        agent_type = agent_type.strip()

        if "." in agent_type:
            module_path, class_name = agent_type.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import agent class '{agent_type}': {e}") from e
        else:
            cls = None
            for package_path in ("polymathera.colony.agents", "polymathera.agents"):
                try:
                    module = importlib.import_module(package_path)
                    cls = getattr(module, agent_type)
                    break
                except (ImportError, AttributeError):
                    continue
            if cls is None:
                raise ValueError(
                    f"Agent class '{agent_type}' not found. "
                    f"Use a fully qualified name (e.g., 'mypackage.agents.{agent_type}')."
                )

        if not (isinstance(cls, type) and issubclass(cls, Agent)):
            raise ValueError(f"'{agent_type}' is not an Agent subclass")
        return cls

    # === Resource Exhausted Handling ===

    async def _handle_resource_exhausted_standalone(
        self,
        blueprint: AgentBlueprint,
        standalone_handle: serving.DeploymentHandle,
        suspend_agents: bool,
    ) -> str:
        """Handle ResourceExhausted error for standalone agent spawning.

        Implements SCALE_DEPLOYMENT strategy: scale up the standalone deployment
        by adding one replica, then retry the spawn.

        Args:
            blueprint: Agent blueprint (agent_id already in kwargs)
            standalone_handle: Handle to StandaloneAgentDeployment
            suspend_agents: Whether to suspend agents during scaling

        Returns:
            Spawned agent ID if successful

        Raises:
            ResourceExhausted: If all retries exhausted or scaling disabled
        """
        agent_id = blueprint.agent_id

        if not self.resource_exhausted_config.enable_auto_scaling:
            logger.error(
                f"ResourceExhausted for agent {agent_id}, but auto-scaling is disabled"
            )
            raise

        if self.resource_exhausted_config.standalone_strategy != ResourceExhaustedStrategy.SCALE_DEPLOYMENT:
            logger.error(
                f"Unsupported standalone strategy: {self.resource_exhausted_config.standalone_strategy}"
            )
            raise

        for attempt in range(self.resource_exhausted_config.max_retries):
            logger.info(
                f"ResourceExhausted for agent {agent_id}, attempt {attempt + 1}/"
                f"{self.resource_exhausted_config.max_retries}: Scaling up standalone deployment"
            )

            try:
                # Get current replica count and scale up by 1
                current_count = await standalone_handle.get_replica_count()
                success = await standalone_handle.scale_deployment(current_count + 1)

                if not success:
                    logger.warning(
                        f"Failed to scale standalone deployment to {current_count + 1} replicas"
                    )
                    if attempt < self.resource_exhausted_config.max_retries - 1:
                        await asyncio.sleep(self.resource_exhausted_config.retry_delay_s)
                        continue
                    else:
                        raise ResourceExhausted(
                            f"Failed to scale standalone deployment after {attempt + 1} attempts"
                        )

                # Wait for new replica to come online
                logger.info(
                    f"Scaled standalone deployment to {current_count + 1} replicas, "
                    f"waiting {self.resource_exhausted_config.retry_delay_s}s for new replica"
                )
                await asyncio.sleep(self.resource_exhausted_config.retry_delay_s)

                # Retry spawn on scaled deployment
                spawned_id = await self._start_agent(
                    standalone_handle,
                    blueprint,
                    suspend_agents=suspend_agents,
                    expected_agent_id=agent_id,
                )

                logger.info(
                    f"Successfully spawned agent {spawned_id} after scaling standalone deployment"
                )
                return spawned_id

            except ResourceExhausted as e:
                if attempt < self.resource_exhausted_config.max_retries - 1:
                    logger.warning(
                        f"Still ResourceExhausted after scaling, retrying in "
                        f"{self.resource_exhausted_config.retry_delay_s}s: {e}"
                    )
                    await asyncio.sleep(self.resource_exhausted_config.retry_delay_s)
                else:
                    logger.error(
                        f"ResourceExhausted persists after {attempt + 1} retries and scaling standalone deployment"
                    )
                    raise

        # Should not reach here, but for safety
        raise ResourceExhausted(
            f"Failed to spawn agent {agent_id} after {self.resource_exhausted_config.max_retries} retries"
        )

    async def _handle_resource_exhausted_vllm(
        self,
        blueprint: AgentBlueprint,
        vllm_handle: serving.DeploymentHandle,
        soft_affinity: bool,
        suspend_agents: bool,
    ) -> str:
        """Handle ResourceExhausted error for VLLM agent spawning.

        Tries strategies in configured order:
        - SOFT_CONSTRAINT: Allow spawning without all pages (will page fault)
        - SUSPEND_AGENTS: Enable agent suspension to free resources
        - RETRY_LATER: Wait and retry

        Args:
            blueprint: Agent blueprint (agent_id already in kwargs)
            requirements: LLM deployment requirements
            vllm_handle: Handle to VLLMDeployment
            vllm_deployment_name: Name of VLLM deployment
            soft_affinity: Current soft_affinity setting
            suspend_agents: Current suspend_agents setting

        Returns:
            Spawned agent ID if successful

        Raises:
            ResourceExhausted: If all strategies exhausted
        """
        agent_id = blueprint.agent_id

        if not self.resource_exhausted_config.enable_auto_scaling:
            logger.error(
                f"ResourceExhausted for agent {agent_id}, but auto-scaling is disabled"
            )
            raise

        strategies = self.resource_exhausted_config.vllm_strategy_order
        logger.info(
            f"ResourceExhausted for VLLM agent {agent_id}, trying {len(strategies)} strategies: "
            f"{[s.value for s in strategies]}"
        )

        for strategy_idx, strategy in enumerate(strategies):
            logger.info(
                f"Trying strategy {strategy_idx + 1}/{len(strategies)} for agent {agent_id}: {strategy.value}"
            )

            try:
                if strategy == ResourceExhaustedStrategy.SOFT_CONSTRAINT:
                    # Already tried with current soft_affinity setting
                    if soft_affinity:
                        logger.info("SOFT_CONSTRAINT already enabled, skipping")
                        continue

                    logger.info(
                        f"SOFT_CONSTRAINT strategy: Retrying agent {agent_id} with soft_affinity=True"
                    )
                    spawned_id = await self._start_agent(
                        vllm_handle,
                        blueprint,
                        suspend_agents=suspend_agents,
                    )
                    logger.info(
                        f"Successfully spawned agent {spawned_id} with SOFT_CONSTRAINT strategy"
                    )
                    return spawned_id

                elif strategy == ResourceExhaustedStrategy.SUSPEND_AGENTS:
                    # Already tried with current suspend_agents setting
                    if suspend_agents:
                        logger.info("SUSPEND_AGENTS already enabled, skipping")
                        continue

                    logger.info(
                        f"SUSPEND_AGENTS strategy: Retrying agent {agent_id} with suspend_agents=True"
                    )
                    spawned_id = await self._start_agent(
                        vllm_handle,
                        blueprint,
                        suspend_agents=True,  # Enable suspension
                        expected_agent_id=agent_id,
                    )
                    logger.info(
                        f"Successfully spawned agent {spawned_id} with SUSPEND_AGENTS strategy"
                    )
                    return spawned_id

                elif strategy == ResourceExhaustedStrategy.REPLICATE_PAGES:
                    logger.warning(
                        "REPLICATE_PAGES strategy is no longer supported (simplified design)"
                    )
                    continue

                elif strategy == ResourceExhaustedStrategy.SCALE_DEPLOYMENT:
                    logger.warning(
                        "SCALE_DEPLOYMENT strategy is not applicable for VLLM agents "
                        "(page affinity prevents simple scaling)"
                    )
                    continue

                else:
                    logger.warning(f"Unknown strategy: {strategy}")
                    continue

            except ResourceExhausted as e:
                logger.warning(
                    f"Strategy {strategy.value} failed for agent {agent_id}: {e}"
                )
                # Try next strategy
                continue

            except Exception as e:
                logger.error(
                    f"Strategy {strategy.value} raised unexpected error for agent {agent_id}: {e}",
                    exc_info=True
                )
                # Try next strategy
                continue

        # All strategies failed
        error_msg = f"All {len(strategies)} strategies failed for VLLM agent {agent_id}"
        logger.error(error_msg)
        raise ResourceExhausted(error_msg)

    # === Tenant Quota Management ===

    async def _check_tenant_quota(
        self,
        tenant_id: str,
        resource_requirements: AgentResourceRequirements
    ) -> None:
        """Check if tenant has quota for these resources (internal).

        Args:
            tenant_id: Tenant identifier
            resource_requirements: Resource requirements for new agent

        Raises:
            QuotaExceeded: If tenant quota would be exceeded
        """
        if not self.session_manager_handle:
            # No session manager, skip quota check
            logger.debug(f"No session manager available, skipping quota check for tenant {tenant_id}")
            return

        # Get tenant quota from session manager
        quota = await self.session_manager_handle.get_tenant_quota(tenant_id)
        if not quota:
            # No quota set for this tenant
            logger.debug(f"No quota set for tenant {tenant_id}, allowing agent")
            return

        # Get current tenant resource usage from session manager
        usage = await self.session_manager_handle.get_tenant_resource_usage(tenant_id)
        if not usage:
            # Should not happen if quota exists, but handle gracefully
            logger.warning(f"Quota exists but no usage tracking for tenant {tenant_id}")
            return

        # Check agent count quota
        new_agent_count = usage.active_agents + 1
        if new_agent_count > quota.max_concurrent_agents:
            raise QuotaExceeded(
                f"Tenant {tenant_id} agent quota exceeded: "
                f"{new_agent_count}/{quota.max_concurrent_agents} agents"
            )

        # Check CPU quota
        new_cpu = usage.total_cpu_cores + resource_requirements.cpu_cores
        if new_cpu > quota.max_total_cpu_cores:
            raise QuotaExceeded(
                f"Tenant {tenant_id} CPU quota exceeded: "
                f"{new_cpu:.2f}/{quota.max_total_cpu_cores} cores"
            )

        # Check memory quota
        new_memory = usage.total_memory_mb + resource_requirements.memory_mb
        if new_memory > quota.max_total_memory_mb:
            raise QuotaExceeded(
                f"Tenant {tenant_id} memory quota exceeded: "
                f"{new_memory}/{quota.max_total_memory_mb} MB"
            )

        # Check GPU cores quota
        new_gpu = usage.total_gpu_cores + resource_requirements.gpu_cores
        if new_gpu > quota.max_total_gpu_cores:
            raise QuotaExceeded(
                f"Tenant {tenant_id} GPU quota exceeded: "
                f"{new_gpu:.2f}/{quota.max_total_gpu_cores} GPU cores"
            )

        # Check GPU memory quota
        new_gpu_memory = usage.total_gpu_memory_mb + resource_requirements.gpu_memory_mb
        if new_gpu_memory > quota.max_total_gpu_memory_mb:
            raise QuotaExceeded(
                f"Tenant {tenant_id} GPU memory quota exceeded: "
                f"{new_gpu_memory}/{quota.max_total_gpu_memory_mb} MB"
            )

        logger.debug(
            f"Tenant {tenant_id} quota check passed: "
            f"agents={new_agent_count}/{quota.max_concurrent_agents}, "
            f"cpu={new_cpu:.2f}/{quota.max_total_cpu_cores}, "
            f"mem={new_memory}/{quota.max_total_memory_mb}MB"
        )

    # === Monitoring ===

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def get_system_stats(self) -> dict[str, Any]:
        """Get agent system statistics.

        Returns:
            Dictionary with stats
        """
        async for state in self.state_manager.read_transaction():
            # Count agents by state
            state_counts = {}
            for agent in state.agents.values():
                state_counts[agent.state.value] = state_counts.get(agent.state.value, 0) + 1

            # Count agents by type
            type_counts = {
                agent_type: len(agent_ids)
                for agent_type, agent_ids in state.agents_by_type.items()
            }

            return {
                "total_agents": len(state.agents),
                "agents_by_state": state_counts,
                "agents_by_type": type_counts,
                "page_bound_agents": len(state.page_agents),
                "total_pages_with_agents": len(state.page_agents),
            }

    # === Infrastructure Status (for dashboard) ===

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def get_infrastructure_status(self) -> dict[str, Any]:
        """Get cluster infrastructure status: Redis health/info, app registry.

        Returns a single consolidated view so the dashboard needs only one RPC.
        """
        polymathera = get_polymathera()
        result: dict[str, Any] = {
            "redis_connected": False,
            "redis_info": {},
            "applications": [],
        }

        # Redis health + info
        try:
            redis_client = await polymathera.get_redis_client()
            result["redis_connected"] = await redis_client.is_healthy()
            info_data = await redis_client.execute_with_semaphore(
                lambda r: r.info("server", "clients", "memory")
            )
            result["redis_info"] = {
                "connected_clients": info_data.get("connected_clients", 0),
                "used_memory_human": info_data.get("used_memory_human", ""),
                "total_commands_processed": info_data.get("total_commands_processed", 0),
                "keyspace_hits": info_data.get("keyspace_hits", 0),
                "keyspace_misses": info_data.get("keyspace_misses", 0),
                "uptime_in_seconds": info_data.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.warning("Failed to get Redis info: %s", e)

        # Application registry
        try:
            from ..distributed.ray_utils.serving.models import ApplicationRegistry
            registry_sm = await polymathera.get_state_manager(
                state_type=ApplicationRegistry,
                state_key="polymathera.colony.distributed.ray_utils.serving.apps",
            )
            async for registry in registry_sm.read_transaction():
                for app_info in registry.list_apps():
                    result["applications"].append({
                        "app_name": app_info.app_name,
                        "created_at": app_info.created_at,
                        "deployments": [
                            {
                                "deployment_name": dep_name,
                                "proxy_actor_name": dep_info.proxy_actor_name,
                            }
                            for dep_name, dep_info in app_info.deployments.items()
                        ],
                    })
        except Exception as e:
            logger.warning("Failed to read application registry: %s", e)

        return result

    # === Blackboard Configuration & Registry ===

    @serving.endpoint
    async def get_blackboard_backend_type(self) -> str:
        """Return the configured default blackboard backend type."""
        return self.blackboard_backend_type

    @serving.endpoint
    async def register_blackboard_scope(
        self,
        scope_id: str,
        backend_type: str,
    ) -> None:
        """Register an active blackboard scope.

        Called by EnhancedBlackboard.initialize() so the dashboard can
        discover scopes without scanning Redis keys.
        """
        import time
        async for state in self.state_manager.write_transaction():
            state.blackboard_scopes[scope_id] = {
                "scope_id": scope_id,
                "backend_type": backend_type,
                "registered_at": time.time(),
            }

    # === Blackboard Observer (for dashboard) ===

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def get_blackboard_scopes(self) -> list[dict[str, Any]]:
        """List all registered blackboard scopes with live statistics.

        Reads the scope registry (populated by EnhancedBlackboard.initialize())
        and queries each scope's backend for current stats.
        """
        from ..agents.blackboard import EnhancedBlackboard

        # Read registered scopes from shared state
        async for state in self.state_manager.read_transaction():
            registered = dict(state.blackboard_scopes)

        scopes: list[dict[str, Any]] = []
        for _key, info in registered.items():
            scope_id = info["scope_id"]
            backend_type = info.get("backend_type", self.blackboard_backend_type)

            try:
                bb = EnhancedBlackboard(
                    app_name=self.app_name,
                    scope_id=scope_id,
                    enable_events=False,
                    backend_type=backend_type,
                )
                stats = await bb.get_statistics()
                scopes.append({
                    "scope_id": scope_id,
                    "entry_count": stats.get("entry_count", 0),
                    "oldest_entry_age": stats.get("oldest_entry_age"),
                    "newest_entry_age": stats.get("newest_entry_age"),
                    "backend_type": stats.get("backend_type", backend_type),
                    "subscriber_count": stats.get("subscriber_count", 0),
                })
            except Exception as e:
                logger.debug(f"Failed to get stats for {scope_id}: {e}")
                scopes.append({
                    "scope_id": scope_id,
                    "entry_count": 0,
                    "backend_type": backend_type,
                    "error": str(e),
                })

        return scopes

    @serving.endpoint(ring=serving.Ring.KERNEL)
    async def get_blackboard_entries(
        self,
        scope_id: str,
        limit: int = 100,
        backend_type: str = "",
    ) -> list[dict[str, Any]]:
        """List entries in a specific blackboard scope."""
        try:
            from ..agents.blackboard import EnhancedBlackboard
            # Map backend class name to backend_type string for EnhancedBlackboard
            bt = None
            if backend_type == "RedisBackend":
                bt = "redis"
            elif backend_type == "DistributedBackend":
                bt = "distributed"
            elif backend_type == "InMemoryBackend":
                bt = "memory"
            bb = EnhancedBlackboard(
                app_name=self.app_name,
                scope_id=scope_id,
                enable_events=False,
                backend_type=bt,
            )
            entries = await bb.query(limit=limit)
            return [
                {
                    "key": e.key,
                    "value": e.value,
                    "version": e.version,
                    "created_by": e.created_by,
                    "updated_by": e.updated_by,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                    "tags": list(e.tags) if e.tags else [],
                }
                for e in entries
            ]
        except Exception as e:
            logger.warning(f"Failed to get blackboard entries for {scope_id}: {e}")
            return []

    # === Resource-Driven Resumption ===

    async def _resource_monitor_loop(self):
        """Periodically check if resources freed up and resume suspended agents.

        This runs in AgentSystemDeployment (global coordinator) because:
        1. Suspended agents are DELETED from AgentManagerBase replicas
        2. AgentSystemDeployment tracks all suspension states via StateManager
        3. AgentSystemDeployment can query cluster-wide resource availability
        4. Resume routing happens via AgentSystemDeployment.resume_agent() anyway

        Runs in Ring.KERNEL context — this is a cross-tenant infrastructure
        task that scans all suspended agents across all tenants.
        """
        from polymathera.colony.distributed.ray_utils.serving.context import Ring, execution_context

        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                with execution_context(ring=Ring.KERNEL, origin="agent_resource_monitor"):
                    # Query all suspended agents from StateManager
                    suspended_agents = await self._list_all_suspended_agents()

                    if not suspended_agents:
                        continue

                    # Sort by resumption priority (fairness)
                    sorted_agents = sorted(
                        suspended_agents,
                        key=lambda state: (
                            state.resumption_priority,
                            -state.suspension_count,  # Penalize frequently suspended
                            state.suspended_at,  # FIFO among same priority
                        ),
                        reverse=True
                    )

                    # Try to resume agents one at a time
                    for suspension_state in sorted_agents:
                        agent_id = suspension_state.agent_id
                        try:
                            # Attempt to resume the agent
                            # This will call spawn_agents() with suspend_agents=True
                            # which allows suspension of other agents to make room
                            new_agent_id = await self.resume_agent(agent_id)
                            logger.info(
                                f"Resource monitor: Resumed agent {agent_id} as {new_agent_id} "
                                f"(priority={suspension_state.resumption_priority}, "
                                f"suspensions={suspension_state.suspension_count})"
                            )
                        except ResourceExhausted as e:
                            # Not enough resources yet, try next agent
                            logger.debug(
                                f"Resource monitor: Cannot resume {agent_id} yet (ResourceExhausted): {e}"
                            )
                            continue
                        except Exception as e:
                            # Log error but continue monitoring
                            logger.error(
                                f"Resource monitor: Failed to resume {agent_id}: {e}",
                                exc_info=True
                            )
                            continue

            except Exception as e:
                # Don't crash the monitor loop on errors
                logger.error(
                    f"Error in resource monitor loop: {e}",
                    exc_info=True
                )
                await asyncio.sleep(10)

    async def _list_all_suspended_agents(self) -> list[AgentSuspensionState]:
        """List all suspended agents from StateManager.

        Returns:
            List of AgentSuspensionState objects for all suspended agents
        """
        suspended_agents = []
        polymathera = get_polymathera()

        try:
            # Query all agents from the agent registry
            async for state in self.state_manager.read_transaction():
                for agent_id, agent in state.agents.items():
                    # Check if agent is suspended
                    if agent.state != AgentState.SUSPENDED:
                        continue

                    # Load suspension state for this agent
                    state_key = AgentSuspensionState.get_state_key(self.app_name, agent_id)
                    try:
                        state_manager = await polymathera.get_state_manager(
                            state_type=AgentSuspensionState,
                            state_key=state_key,
                        )

                        async for suspension_state in state_manager.read_transaction():
                            if suspension_state.agent_id:
                                suspended_agents.append(suspension_state)
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to load suspension state for {agent_id}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Failed to list suspended agents: {e}", exc_info=True)

        return suspended_agents

