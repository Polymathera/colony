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

import asyncio
import logging
from typing import Any
import uuid

from pydantic import Field

from ..distributed import get_polymathera
from ..distributed.state_management import SharedState, StateManager
from ..distributed.ray_utils import serving
from ..deployment_names import get_deployment_names
from .base import Agent, AgentState, ResourceExhausted
from .models import (
    AgentResourceRequirements,
    AgentSpawnSpec,
    AgentSuspensionState,
    ResourceExhaustedConfig,
    ResourceExhaustedStrategy,
)

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
        agents: All agents in the system (agent_id -> Agent)
        agent_locations: Agent to deployment mapping (agent_id -> deployment_id)
        deployment_names: Deployment name mapping (deployment_id -> deployment_name)
        page_agents: Page-bound agents (page_id -> list of agent_ids)
        agents_by_type: Index for finding agents by type
        agents_by_capability: Index for finding agents by capability
        stats: System-wide statistics
    """

    # Agent registry (updated only on spawn/terminate)
    agents: dict[str, Agent] = Field(default_factory=dict)
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
    ):
        """Initialize agent system deployment.

        Args:
            max_retries: Maximum retries for general operations (deprecated - use resource_exhausted_config)
            resource_exhausted_config: Configuration for handling ResourceExhausted errors
        """
        # Initialized in initialize()
        self.max_retries = max_retries  # Legacy parameter, kept for backwards compatibility
        self.resource_exhausted_config = resource_exhausted_config or ResourceExhaustedConfig()
        self.app_name: str | None = None
        self.state_manager: StateManager[AgentSystemState] | None = None
        self.vcm_handle = None
        self.session_manager_handle = None

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize state manager and handles."""
        # Get app name from environment
        self.app_name = serving.get_my_app_name()
        logger.info(f"Initializing AgentSystemDeployment for app {self.app_name}")

        # Get deployment names configuration
        names = get_deployment_names()

        # Get Polymathera for state management
        polymathera = get_polymathera()

        # Initialize StateManager
        self.state_manager = await polymathera.get_state_manager(
            state_type=AgentSystemState,
            state_key=AgentSystemState.get_state_key(self.app_name),
        )

        # Try to discover VCM deployment in same app
        try:
            self.vcm_handle = serving.get_deployment(
                app_name=self.app_name,
                deployment_name=names.vcm,
            )
            logger.info(f"Connected to VCM deployment: {names.vcm}")
        except Exception as e:
            logger.error(f"VCM deployment '{names.vcm}' not found: {e}")
            raise e

        # Start background task for resource-driven resumption
        import asyncio
        self._resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        logger.info("Started resource monitor loop for agent resumption")

        # Try to discover SessionManager deployment in same app
        try:
            self.session_manager_handle = serving.get_deployment(
                app_name=self.app_name,
                deployment_name=names.session_manager,
            )
            logger.info(f"Connected to SessionManagerDeployment: {names.session_manager}")
        except Exception as e:
            logger.warning(f"SessionManager deployment '{names.session_manager}' not found: {e}")
            # This is optional - don't fail initialization

        logger.info("AgentSystemDeployment initialized")

    # === Agent Discovery ===

    @serving.endpoint
    async def list_all_agents(self) -> list[str]:
        """List all agent IDs in the system.

        Returns:
            List of agent IDs
        """
        async for state in self.state_manager.read_transaction():
            return list(state.agents.keys())

    @serving.endpoint
    async def get_agent_info(self, agent_id: str) -> Agent | None:
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

    @serving.endpoint
    async def spawn_agents(
        self,
        agent_specs: list[AgentSpawnSpec],
        session_id: str | None = None,
        run_id: str | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
    ) -> list[str]:
        """Spawn multiple agents with requirement-based deployment routing.

        This is the entry point for spawning agents. It decides WHERE to spawn
        each agent based on:
        - Page affinity (bound_pages) or LLM requirements → VLLMDeployment
          - Uses LLMCluster.select_deployment() with RequirementBasedRouter
        - No affinity → StandaloneAgentDeployment

        Args:
            agent_specs: List of AgentSpawnSpec defining agents to spawn
            session_id: Optional session ID for tracking which session spawned these agents
            run_id: Optional run ID for tracking which AgentRun spawned these agents
            soft_affinity: If True, allows spawning on replicas without all pages (will page fault)
            suspend_agents: If True, replica may suspend existing agents to make room

        Returns:
            List of spawned agent IDs

        Example:
            ```python
            from polymathera.colony.agents.models import AgentSpawnSpec
            from polymathera.colony.cluster.models import LLMClientRequirements

            specs = [
                AgentSpawnSpec(
                    agent_type="code_analyzer",
                    bound_pages=["repo-123-context"],
                    requirements=LLMClientRequirements(
                        model_family="llama",
                        min_context_window=32000,
                    ),
                    metadata={"repo_id": "123"},
                ),
                AgentSpawnSpec(
                    agent_type="supervisor",  # No affinity
                    metadata={"role": "coordinator"},
                ),
            ]
            agent_ids = await agent_system.spawn_agents(specs, session_id="sess-123")
            ```
        """
        agent_ids = []
        app_name = serving.get_my_app_name()
        names = get_deployment_names()

        # Get LLMCluster handle for requirement-based routing
        llm_cluster_handle = None
        if any(spec.has_deployment_affinity() for spec in agent_specs):
            try:
                llm_cluster_handle = serving.get_deployment(
                    app_name,
                    names.llm_cluster,
                    #deployment_class=LLMCluster,
                )
            except Exception as e:
                logger.error(f"Failed to get LLMCluster handle for routing: {e}")
                raise

        for spec in agent_specs:
            # Generate agent ID if not provided
            agent_id = spec.agent_id or f"agent-{uuid.uuid4().hex[:8]}"

            # Add session_id and run_id to metadata for tracking
            metadata = spec.metadata.copy()
            if session_id:
                metadata["session_id"] = session_id
            if run_id:
                metadata["run_id"] = run_id

            # Check tenant quota if tenant_id is present
            tenant_id = metadata.get("tenant_id")
            if tenant_id:
                try:
                    await self._check_tenant_quota(tenant_id, spec.resource_requirements)
                except QuotaExceeded as e:
                    logger.error(f"Quota exceeded for tenant {tenant_id}: {e}")
                    raise

            # Decide where to spawn based on deployment affinity
            if spec.has_deployment_affinity():
                # Agent needs specific VLLM deployment (has pages or requirements)
                # Use LLMCluster to select deployment based on requirements
                try:
                    vllm_deployment_name = await llm_cluster_handle.select_deployment(
                        requirements=spec.requirements
                    )

                    vllm_handle = serving.get_deployment(app_name, vllm_deployment_name)
                    spawned_id = await vllm_handle.start_agent(
                        agent_class_id=spec.agent_type,
                        agent_id=agent_id,
                        bound_pages=spec.bound_pages,
                        soft_affinity=soft_affinity,
                        suspend_agents=suspend_agents,
                        metadata=metadata,
                    )
                    agent_ids.append(spawned_id)
                    logger.info(
                        f"Spawned page-affinity agent {spawned_id} on {vllm_deployment_name} "
                        f"(pages={len(spec.bound_pages)}, requirements={'set' if spec.requirements else 'none'})"
                    )

                except ResourceExhausted as e:
                    # Handle resource exhaustion with strategy loop
                    logger.warning(
                        f"ResourceExhausted for VLLM agent {agent_id}: {e}"
                    )

                    spawned_id = await self._handle_resource_exhausted_vllm(
                        spec=spec,
                        agent_id=agent_id,
                        metadata=metadata,
                        vllm_handle=vllm_handle,
                        vllm_deployment_name=vllm_deployment_name,
                        soft_affinity=soft_affinity,
                        suspend_agents=suspend_agents,
                    )

                    agent_ids.append(spawned_id)
                    logger.info(
                        f"Spawned VLLM agent {spawned_id} after handling ResourceExhausted"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to spawn agent {agent_id} on VLLM deployment: {e}"
                    )
                    raise

            else:
                # Standalone agent → spawn on StandaloneAgentDeployment
                retry_count = 0
                spawned = False

                while not spawned and retry_count <= self.resource_exhausted_config.max_retries:
                    try:
                        standalone_handle = serving.get_deployment(
                            app_name, names.standalone_agents
                        )
                        spawned_id = await standalone_handle.start_agent(
                            agent_class_id=spec.agent_type,
                            agent_id=agent_id,
                            bound_pages=[],
                            metadata=metadata,
                        )
                        agent_ids.append(spawned_id)
                        logger.info(f"Spawned standalone agent {spawned_id}")
                        spawned = True

                    except ResourceExhausted as e:
                        if not self.resource_exhausted_config.enable_auto_scaling:
                            logger.error(
                                f"ResourceExhausted for agent {agent_id} but auto-scaling disabled"
                            )
                            raise

                        if retry_count >= self.resource_exhausted_config.max_retries:
                            logger.error(
                                f"Failed to spawn agent {agent_id} after {retry_count} retries: {e}"
                            )
                            raise

                        # Handle resource exhaustion (scale deployment and retry)
                        logger.warning(
                            f"ResourceExhausted for agent {agent_id}, attempt {retry_count + 1}/{self.resource_exhausted_config.max_retries + 1}: {e}"
                        )

                        # The handler will scale and retry internally
                        spawned_id = await self._handle_resource_exhausted_standalone(
                            spec=spec,
                            agent_id=agent_id,
                            metadata=metadata,
                            standalone_handle=standalone_handle,
                            deployment_name=names.standalone_agents,
                        )

                        agent_ids.append(spawned_id)
                        logger.info(f"Spawned standalone agent {spawned_id} after handling ResourceExhausted")
                        spawned = True
                        retry_count += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to spawn agent {agent_id} on standalone deployment '{names.standalone_agents}': {e}"
                        )
                        raise

        context_info = []
        if session_id:
            context_info.append(f"session={session_id}")
        if run_id:
            context_info.append(f"run={run_id}")
        context_str = f" ({', '.join(context_info)})" if context_info else ""
        logger.info(f"Spawned {len(agent_ids)} agents{context_str}")

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
    async def register_agent(self, agent: Agent, deployment_replica_id: str, deployment_name: str) -> None:
        """Register a new agent.

        Args:
            agent: Agent instance
            deployment_replica_id: Deployment replica hosting this agent (format: "app_name.deployment_name#replica_id")
            deployment_name: Deployment name
        """
        async for state in self.state_manager.write_transaction():
            state.agents[agent.agent_id] = agent
            state.agent_locations[agent.agent_id] = deployment_replica_id
            if deployment_replica_id in state.deployment_names:
                if state.deployment_names[deployment_replica_id] != deployment_name:
                    logger.warning(f"Deployment name mismatch for {deployment_replica_id}: {state.deployment_names[deployment_replica_id]} != {deployment_name}")
            state.deployment_names[deployment_replica_id] = deployment_name
            # Register page bindings
            for page_id in agent.bound_pages:
                if page_id not in state.page_agents:
                    state.page_agents[page_id] = []
                if agent.agent_id not in state.page_agents[page_id]:
                    state.page_agents[page_id].append(agent.agent_id)

            # Register in type index
            if agent.agent_type not in state.agents_by_type:
                state.agents_by_type[agent.agent_type] = []
            if agent.agent_id not in state.agents_by_type[agent.agent_type]:
                state.agents_by_type[agent.agent_type].append(agent.agent_id)

            # Register capabilities if present
            capabilities = agent.metadata.get("capabilities", [])
            for capability in capabilities:
                if capability not in state.agents_by_capability:
                    state.agents_by_capability[capability] = []
                if agent.agent_id not in state.agents_by_capability[capability]:
                    state.agents_by_capability[capability].append(agent.agent_id)

            logger.info(f"Registered agent {agent.agent_id} on deployment replica {deployment_replica_id}")

        # Update tenant resource usage if agent has tenant_id
        tenant_id = agent.metadata.get("tenant_id")
        if tenant_id and self.session_manager_handle:
            try:
                # Increment resource usage (update via SessionManager)
                await self.session_manager_handle.increment_tenant_resources(
                    tenant_id=tenant_id,
                    cpu_cores=agent.resource_requirements.cpu_cores,
                    memory_mb=agent.resource_requirements.memory_mb,
                    gpu_cores=agent.resource_requirements.gpu_cores,
                    gpu_memory_mb=agent.resource_requirements.gpu_memory_mb,
                )
                logger.debug(f"Updated tenant {tenant_id} resource usage after registering agent {agent.agent_id}")
            except Exception as e:
                logger.warning(f"Failed to update tenant resource usage for {tenant_id}: {e}")

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
                capabilities = agent.metadata.get("capabilities", [])
                for capability in capabilities:
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
        if agent:
            tenant_id = agent.metadata.get("tenant_id")
            if tenant_id and self.session_manager_handle:
                try:
                    # Decrement resource usage (update via SessionManager)
                    await self.session_manager_handle.decrement_tenant_resources(
                        tenant_id=tenant_id,
                        cpu_cores=agent.resource_requirements.cpu_cores,
                        memory_mb=agent.resource_requirements.memory_mb,
                        gpu_cores=agent.resource_requirements.gpu_cores,
                        gpu_memory_mb=agent.resource_requirements.gpu_memory_mb,
                    )
                    logger.debug(f"Updated tenant {tenant_id} resource usage after unregistering agent {agent_id}")
                except Exception as e:
                    logger.warning(f"Failed to update tenant resource usage for {tenant_id}: {e}")

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
        2. Builds AgentSpawnSpec from suspension state
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

            suspension_state = None
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

        # 2. Build AgentSpawnSpec from suspension state
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
        metadata = {
            "resuming_from_suspension": True,
            "suspended_agent_id": agent_id,
            "suspension_reason": suspension_state.suspension_reason,
            "suspension_count": suspension_state.suspension_count,
        }

        # Add parent/role info if present (for child agents)
        if suspension_state.parent_agent_id:
            metadata["parent_agent_id"] = suspension_state.parent_agent_id
        if suspension_state.role:
            metadata["role"] = suspension_state.role

        # Build resource requirements
        resource_requirements = AgentResourceRequirements(
            cpu_cores=suspension_state.allocated_cpu_cores,
            memory_mb=suspension_state.allocated_memory_mb,
            gpu_cores=suspension_state.allocated_gpu_cores,
            gpu_memory_mb=suspension_state.allocated_gpu_memory_mb,
        )

        # Build spawn spec with bound_pages for cache-aware routing
        spec = AgentSpawnSpec(
            agent_type=agent_type,
            agent_id=None,  # Let spawn_agents generate new ID to avoid collisions
            bound_pages=suspension_state.bound_pages,
            resource_requirements=resource_requirements,
            metadata=metadata,
        )

        # 3. Spawn via spawn_agents() with soft_affinity=True
        # This will route to replica with best page affinity via SoftPageAffinityRouter
        try:
            agent_ids = await self.spawn_agents(
                agent_specs=[spec],
                soft_affinity=True,  # Enable soft page affinity routing
                suspend_agents=True,  # Allow suspension of other agents if needed
            )

            if not agent_ids:
                raise ValueError(f"spawn_agents returned empty list for resumed agent {agent_id}")

            new_agent_id = agent_ids[0]

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

    # === Resource Exhausted Handling ===

    async def _handle_resource_exhausted_standalone(
        self,
        spec: AgentSpawnSpec,
        agent_id: str,
        metadata: dict[str, Any],
        standalone_handle: serving.DeploymentHandle,
        deployment_name: str,
    ) -> str:
        """Handle ResourceExhausted error for standalone agent spawning.

        Implements SCALE_DEPLOYMENT strategy: scale up the standalone deployment
        by adding one replica, then retry the spawn.

        Args:
            spec: Agent spawn specification
            agent_id: Generated agent ID
            metadata: Agent metadata
            standalone_handle: Handle to StandaloneAgentDeployment
            deployment_name: Name of standalone deployment

        Returns:
            Spawned agent ID if successful

        Raises:
            ResourceExhausted: If all retries exhausted or scaling disabled
        """
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
                f"{self.resource_exhausted_config.max_retries}: Scaling up {deployment_name}"
            )

            try:
                # Get current replica count and scale up by 1
                current_count = await standalone_handle.get_replica_count()
                success = await standalone_handle.scale_deployment(current_count + 1)

                if not success:
                    logger.warning(
                        f"Failed to scale {deployment_name} to {current_count + 1} replicas"
                    )
                    if attempt < self.resource_exhausted_config.max_retries - 1:
                        await asyncio.sleep(self.resource_exhausted_config.retry_delay_s)
                        continue
                    else:
                        raise ResourceExhausted(
                            f"Failed to scale {deployment_name} after {attempt + 1} attempts"
                        )

                # Wait for new replica to come online
                logger.info(
                    f"Scaled {deployment_name} to {current_count + 1} replicas, "
                    f"waiting {self.resource_exhausted_config.retry_delay_s}s for new replica"
                )
                await asyncio.sleep(self.resource_exhausted_config.retry_delay_s)

                # Retry spawn on scaled deployment
                spawned_id = await standalone_handle.start_agent(
                    agent_class_id=spec.agent_type,
                    agent_id=agent_id,
                    bound_pages=[],
                    metadata=metadata,
                )

                logger.info(
                    f"Successfully spawned agent {spawned_id} after scaling {deployment_name}"
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
                        f"ResourceExhausted persists after {attempt + 1} retries and scaling"
                    )
                    raise

        # Should not reach here, but for safety
        raise ResourceExhausted(
            f"Failed to spawn agent {agent_id} after {self.resource_exhausted_config.max_retries} retries"
        )

    async def _handle_resource_exhausted_vllm(
        self,
        spec: AgentSpawnSpec,
        agent_id: str,
        metadata: dict[str, Any],
        vllm_handle: serving.DeploymentHandle,
        vllm_deployment_name: str,
        soft_affinity: bool,
        suspend_agents: bool,
    ) -> str:
        """Handle ResourceExhausted error for VLLM agent spawning.

        Tries strategies in configured order:
        - SOFT_CONSTRAINT: Allow spawning without all pages (will page fault)
        - SUSPEND_AGENTS: Enable agent suspension to free resources
        - RETRY_LATER: Wait and retry

        Args:
            spec: Agent spawn specification
            agent_id: Generated agent ID
            metadata: Agent metadata
            vllm_handle: Handle to VLLMDeployment
            vllm_deployment_name: Name of VLLM deployment
            soft_affinity: Current soft_affinity setting
            suspend_agents: Current suspend_agents setting

        Returns:
            Spawned agent ID if successful

        Raises:
            ResourceExhausted: If all strategies exhausted
        """
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
                    spawned_id = await vllm_handle.start_agent(
                        agent_class_id=spec.agent_type,
                        agent_id=agent_id,
                        bound_pages=spec.bound_pages,
                        soft_affinity=True,  # Enable soft affinity
                        suspend_agents=suspend_agents,
                        metadata=metadata,
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
                    spawned_id = await vllm_handle.start_agent(
                        agent_class_id=spec.agent_type,
                        agent_id=agent_id,
                        bound_pages=spec.bound_pages,
                        soft_affinity=True,  # Use soft affinity with suspension
                        suspend_agents=True,  # Enable suspension
                        metadata=metadata,
                    )
                    logger.info(
                        f"Successfully spawned agent {spawned_id} with SUSPEND_AGENTS strategy"
                    )
                    return spawned_id

                elif strategy == ResourceExhaustedStrategy.REPLICATE_PAGES:
                    logger.warning(
                        f"REPLICATE_PAGES strategy is no longer supported (simplified design)"
                    )
                    continue

                elif strategy == ResourceExhaustedStrategy.SCALE_DEPLOYMENT:
                    logger.warning(
                        f"SCALE_DEPLOYMENT strategy is not applicable for VLLM agents "
                        f"(page affinity prevents simple scaling)"
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

    @serving.endpoint
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

    # === Resource-Driven Resumption ===

    async def _resource_monitor_loop(self):
        """Periodically check if resources freed up and resume suspended agents.

        This runs in AgentSystemDeployment (global coordinator) because:
        1. Suspended agents are DELETED from AgentManagerBase replicas
        2. AgentSystemDeployment tracks all suspension states via StateManager
        3. AgentSystemDeployment can query cluster-wide resource availability
        4. Resume routing happens via AgentSystemDeployment.resume_agent() anyway
        """
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

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

