"""Consciousness integration for the agent system.

Provides:
- SystemDocumentation for system-level documentation storage
- ConsciousnessCapability for agent self-awareness
- AgentSelfConcept integration
- Performance monitoring

This module focuses on self-awareness (self-concept, system docs, metrics).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from overrides import override
from pydantic import Field

from ....distributed import get_polymathera
from ....distributed.ray_utils import serving
from ....distributed.state_management import SharedState, StateManager
from ...self_concept import AgentSelfConcept
from ...base import AgentCapability, CapabilityResultFuture
from ...blackboard.protocol import ConsciousnessProtocol
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...models import AgentSuspensionState
from ..actions.policies import action_executor

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class SystemDocumentation(SharedState):
    """System documentation for consciousness.

    Stored per tenant/VMR/session in authenticated stores.
    NOT mapped to VCM - agents access directly via storage.

    Contains:
    - System architecture description
    - Agent roles and responsibilities
    - Available tools and usage patterns
    - Guidelines and constraints
    - Runtime costs and performance characteristics
    """

    # Core documentation
    architecture: str = Field(
        default="",
        description="Overall system design, virtual vs physical context, distributed architecture",
    )

    agent_roles: dict[str, str] = Field(
        default_factory=dict,
        description="Agent type -> role description mapping",
    )

    tools_documentation: dict[str, str] = Field(
        default_factory=dict,
        description="Tool ID -> documentation mapping",
    )

    guidelines: list[str] = Field(
        default_factory=list,
        description="Best practices, constraints, policies",
    )

    cost_model: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime costs: LLM inference, tool calls, memory operations",
    )

    deployment_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Cloud environment, resource limits, quotas",
    )

    # Version and metadata
    version: str = Field(default="1.0.0")
    last_updated: float = Field(default_factory=lambda: time.time())


class ConsciousnessCapability(AgentCapability):
    """Agent capability for self-awareness and consciousness.

    Provides:
    - System documentation storage and retrieval
    - Self-concept management
    - Consciousness stream tracking
    - Performance monitoring

    This capability focuses on self-awareness context.

    This capability directly manages:
    - AgentSelfConcept via upload/download methods
    - Consciousness stream via upload/download methods
    - SystemDocumentation via StateManager

    Example:
        ```python
        # Add consciousness capability to agent
        consciousness = ConsciousnessCapability(agent=self)
        await consciousness.initialize()
        agent.add_capability(consciousness)

        # Use via action executors
        await consciousness.update_performance_metrics({"latency_ms": 150})
        self_concept = await consciousness.get_self_concept()
        ```
    """

    input_patterns = [ConsciousnessProtocol.state_pattern(namespace="consciousness")]

    def __init__(
        self,
        agent: "Agent | None" = None,
        scope: BlackboardScope = BlackboardScope.COLONY,
    ):
        """Initialize consciousness capability.

        Args:
            agent: Owning agent (None for detached mode)
            scope: Blackboard scope (defaults to COLONY)
        """
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent))

        # Cached self-concept (loaded lazily)
        self._self_concept: AgentSelfConcept | None = None

        # StateManager for system documentation (created lazily)
        self._state_managers: dict[str, StateManager[SystemDocumentation]] = {}

    async def initialize(self) -> None:
        """Initialize the consciousness capability.

        Loads self-concept from storage if available, otherwise creates a
        default one from agent metadata so the agent always has identity
        context after initialization.
        """
        # Try to load existing self-concept
        try:
            self._self_concept = await self._load_self_concept()
            if self._self_concept:
                logger.info(f"Loaded self-concept for agent {self.agent.agent_id}")
        except Exception as e:
            logger.debug(f"No existing self-concept for agent {self.agent.agent_id}: {e}")

        # Create default self-concept if none was loaded
        if self._self_concept is None and self.agent is not None:
            agent_type = getattr(self.agent, "agent_type", "general")
            metadata: dict[str, Any] = {}
            if hasattr(self.agent, "get_capability_names"):
                metadata["capabilities"] = self.agent.get_capability_names()
            if hasattr(self.agent, "metadata"):
                metadata["goals"] = getattr(self.agent.metadata, "goals", [])
            self._self_concept = await self._create_default_self_concept(
                agent_id=self.agent.agent_id,
                agent_type=agent_type,
                metadata=metadata,
            )

    def get_action_group_description(self) -> str:
        return (
            "Self-Awareness — provides access to system documentation (architecture, guidelines) "
            "and agent self-concept (identity, role, capabilities). Self-concept is lazily loaded "
            "from persistent storage on first access. Use to orient yourself before planning."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ConsciousnessCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ConsciousnessCapability")
        pass

    # =========================================================================
    # Internal Helpers (Direct Storage Access)
    # =========================================================================

    async def _get_state_manager(self) -> StateManager[SystemDocumentation]:
        """Get or create StateManager for system documentation."""
        state_key = f"{ScopeUtils.get_colony_level_scope()}:system_docs"

        if state_key not in self._state_managers:
            polymathera = get_polymathera()
            self._state_managers[state_key] = await polymathera.get_state_manager(
                state_type=SystemDocumentation,
                state_key=state_key,
            )

        return self._state_managers[state_key]

    async def _load_self_concept(self) -> AgentSelfConcept | None:
        """Load agent's self-concept from storage."""
        try:
            storage = await get_polymathera().get_storage()
            d = await storage.load_json(
                metadata={
                    "type": "self_concept",
                    "agent_id": self.agent.agent_id
                }
            )
            d.update({"agent_id": self.agent.agent_id})
            return AgentSelfConcept.from_dict(d)
        except Exception as e:
            logger.warning(f"Could not load self-concept for agent {self.agent.agent_id}: {e}")
            return None

    async def _save_self_concept(self, self_concept: AgentSelfConcept) -> None:
        """Save agent's self-concept to storage."""
        storage = await get_polymathera().get_storage()
        await storage.save_json(
            self_concept.to_dict(),
            metadata={
                "type": "self_concept",
                "agent_id": self.agent.agent_id
            },
        )
        logger.info(f"Saved self-concept for agent {self_concept.agent_id}")

    async def _create_default_self_concept(
        self, agent_id: str, agent_type: str, metadata: dict[str, Any]
    ) -> AgentSelfConcept:
        """Create a default self-concept for a new agent.

        Identity is assembled from three sources (highest priority wins):
        1. ``metadata.parameters["self_concept"]`` — explicit overrides from the
           spawner (e.g., analysis-specific goals/constraints from ANALYSIS_REGISTRY)
        2. Agent instance attributes — class name, docstring, metadata.role
        3. Generic defaults — agent_id prefix, type-based role description

        Args:
            agent_id: Agent identifier
            agent_type: Agent type
            metadata: Agent metadata (capabilities, etc.)

        Returns:
            Created AgentSelfConcept
        """
        # Layer 1: generic defaults
        name = f"Agent {agent_id[:8]}"
        role = self._get_default_role_for_type(agent_type)
        description = f"A {agent_type} agent"

        # Layer 2: agent instance attributes
        if self.agent is not None:
            name = self.agent.__class__.__name__
            if self.agent.metadata.role:
                role = self.agent.metadata.role
            doc = self.agent.__class__.__doc__
            if doc:
                description = doc.strip().split('\n\n')[0].strip()

        # Layer 3: explicit self_concept config from spawner
        sc_config: dict[str, Any] = {}
        if self.agent is not None:
            sc_config = self.agent.metadata.self_concept

        self_concept = AgentSelfConcept(
            agent_id=agent_id,
            name=name,
            role=role,
            description=sc_config.description if sc_config else description,
            capabilities=metadata.get("capabilities", []),
            goals=sc_config.goals if sc_config else metadata.get("goals", []),
            constraints=sc_config.constraints if sc_config else [],
            limitations=sc_config.limitations if sc_config else [],
            world_model=sc_config.world_model if sc_config else "",
        )

        await self._save_self_concept(self_concept)
        logger.info(f"Created default self-concept for agent {agent_id}")
        return self_concept

    def _get_default_role_for_type(self, agent_type: str) -> str:
        """Get default role description for agent type.

        Args:
            agent_type: Agent type

        Returns:
            Role description
        """
        # TODO: Expand roles as needed
        _DEFAULT_AGENT_ROLES = {
            "general": "a general-purpose reasoning agent capable of diverse tasks",
            "specialized": "a specialized agent focused on specific domain tasks",
            "service": "a service agent providing specific capabilities to other agents",
            "supervisor": "a supervisor agent coordinating and managing other agents",
        }
        return _DEFAULT_AGENT_ROLES.get(agent_type, "an autonomous computational agent")

    # =========================================================================
    # AgentCapability Abstract Method Implementations
    # =========================================================================

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Consciousness capability is a persistent service without a single result.

        Raises:
            NotImplementedError: Always, use stream_events_to_queue() instead
        """
        raise NotImplementedError(
            "ConsciousnessCapability is a persistent service without a single result. "
            "Use stream_events_to_queue() to monitor consciousness events."
        )

    # =========================================================================
    # Action Executors (LLM-Plannable)
    # =========================================================================

    @action_executor(action_key="consciousness_get_system_docs")
    async def get_system_documentation(self) -> SystemDocumentation:
        """Get system documentation for self-awareness.

        Returns:
            System documentation
        """
        state_manager = await self._get_state_manager()
        async for state in state_manager.read_transaction():
            return state

        # Should not reach here, but return empty docs if somehow it does
        return SystemDocumentation()

    @action_executor(action_key="consciousness_update_system_docs")
    async def update_system_documentation(self, updates: dict[str, Any]) -> None:
        """Update system documentation.

        Args:
            updates: Fields to update
        """
        state_manager = await self._get_state_manager()
        async for state in state_manager.write_transaction():
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.last_updated = time.time()

        logger.info(f"Updated system documentation for agent {self.agent.agent_id}")

    @action_executor(action_key="consciousness_update_metrics")
    async def update_performance_metrics(
        self,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Update performance metrics for self-awareness.

        Args:
            metrics: Performance metrics to record

        Returns:
            Updated metrics with timestamp
        """
        # Store metrics in agent metadata if attached
        if self._agent is not None:
            self._agent.metadata.performance_metrics.update(metrics)
            self._agent.metadata.performance_last_updated = time.time()

        # Write metrics to blackboard for event streaming
        metrics_entry = {
            "metrics": metrics,
            "timestamp": time.time(),
            "agent_id": self.scope_id,
        }

        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=ConsciousnessProtocol.state_key(f"metrics:{int(time.time() * 1000)}", namespace="consciousness"),
            value=metrics_entry,
            created_by=self.scope_id,
            tags={"consciousness", "metrics"},
        )

        return metrics_entry

    @action_executor(action_key="consciousness_get_self_concept")
    async def get_self_concept(self) -> AgentSelfConcept | None:
        """Get the agent's self-concept.

        Returns:
            AgentSelfConcept if exists, None otherwise
        """
        if self._self_concept is None:
            self._self_concept = await self._load_self_concept()
        return self._self_concept

    @action_executor(action_key="consciousness_update_self_concept")
    async def update_self_concept(
        self,
        updates: dict[str, Any],
    ) -> AgentSelfConcept:
        """Update the agent's self-concept.

        Args:
            updates: Fields to update on the self-concept

        Returns:
            Updated AgentSelfConcept
        """
        # Load current self-concept or create default
        if self._self_concept is None:
            self._self_concept = await self._load_self_concept()

        if self._self_concept is None:
            # Create default self-concept
            agent_type = "general"
            metadata: dict[str, Any] = {}
            if self._agent is not None:
                agent_type = self._agent.agent_type
                metadata = {
                    "capabilities": self._agent.get_capability_names(),
                    "goals": self._agent.metadata.goals,
                }

            self._self_concept = await self._create_default_self_concept(
                agent_id=self.scope_id,
                agent_type=agent_type,
                metadata=metadata,
            )

        # Apply updates
        for key, value in updates.items():
            if hasattr(self._self_concept, key):
                setattr(self._self_concept, key, value)

        # Save updated self-concept
        await self._save_self_concept(self._self_concept)

        # Write update event to blackboard
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=ConsciousnessProtocol.state_key(f"self_concept_update:{int(time.time() * 1000)}", namespace="consciousness"),
            value={"updated_fields": list(updates.keys()), "timestamp": time.time()},
            created_by=self.scope_id,
            tags={"consciousness", "self_concept"},
        )

        return self._self_concept

