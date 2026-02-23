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
from ....distributed.state_management import SharedState, StateManager
from .self_concept import AgentSelfConcept
from ...base import AgentCapability, CapabilityResultFuture
from ...models import AgentSuspensionState
from ...blackboard.types import BlackboardEvent, KeyPatternFilter
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

    @classmethod
    def get_state_key(cls, tenant_id: str, session_id: str | None = None) -> str:
        """Generate state key for system documentation.

        Args:
            tenant_id: Tenant identifier
            session_id: Optional session identifier (if session-specific)

        Returns:
            State key
        """
        key = f"polymathera:serving:agents:system_docs:{tenant_id}"
        if session_id:
            key += f":session:{session_id}"
        return key


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

    def __init__(
        self,
        agent: "Agent | None" = None,
        scope_id: str | None = None,
        *,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize consciousness capability.

        Args:
            agent: Owning agent (None for detached mode)
            scope_id: Blackboard scope ID (defaults to agent.agent_id)
            tenant_id: Tenant ID for system documentation (defaults from agent metadata)
            session_id: Session ID for system documentation (defaults from agent metadata)
        """
        super().__init__(agent=agent, scope_id=scope_id)

        # Tenant/session for system documentation access
        self._tenant_id = tenant_id
        self._session_id = session_id

        # Cached self-concept (loaded lazily)
        self._self_concept: AgentSelfConcept | None = None

        # StateManager for system documentation (created lazily)
        self._state_managers: dict[str, StateManager[SystemDocumentation]] = {}

    async def initialize(self) -> None:
        """Initialize the consciousness capability.

        Loads self-concept from storage if available.
        """
        # Try to load existing self-concept
        try:
            self._self_concept = await self._load_self_concept()
            if self._self_concept:
                logger.info(f"Loaded self-concept for agent {self.agent.agent_id}")
        except Exception as e:
            logger.debug(f"No existing self-concept for agent {self.agent.agent_id}: {e}")

    @property
    def tenant_id(self) -> str:
        """Get tenant ID for system documentation."""
        if self._tenant_id:
            return self._tenant_id
        if self.agent is not None:
            return self.agent.metadata.tenant_id
        return "default"

    @property
    def session_id(self) -> str | None:
        """Get session ID for system documentation."""
        if self._session_id:
            return self._session_id
        if self.agent is not None:
            return self.agent.metadata.session_id
        return None

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

    async def _get_state_manager(
        self, tenant_id: str, session_id: str | None = None
    ) -> StateManager[SystemDocumentation]:
        """Get or create StateManager for system documentation."""
        state_key = SystemDocumentation.get_state_key(tenant_id, session_id)

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

        Args:
            agent_id: Agent identifier
            agent_type: Agent type
            metadata: Agent metadata (capabilities, etc.)

        Returns:
            Created AgentSelfConcept
        """
        self_concept = AgentSelfConcept(
            agent_id=agent_id,
            name=f"Agent {agent_id[:8]}",
            role=self._get_default_role_for_type(agent_type),
            description=f"A {agent_type} agent",
            capabilities=metadata.get("capabilities", []),
            goals=metadata.get("goals", []),
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
    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
    ) -> None:
        """Stream consciousness-related events to the given queue.

        Streams events from the agent's consciousness scope (self-concept updates, etc.).

        Args:
            event_queue: Queue to stream events to
        """
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=f"{self.scope_id}:consciousness:*"),
        )

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
    async def get_system_documentation(
        self,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> SystemDocumentation:
        """Get system documentation for self-awareness.

        Args:
            tenant_id: Tenant identifier (uses capability default if None)
            session_id: Session identifier (uses capability default if None)

        Returns:
            System documentation
        """
        effective_tenant = tenant_id or self.tenant_id
        effective_session = session_id or self.session_id

        state_manager = await self._get_state_manager(effective_tenant, effective_session)
        async for state in state_manager.read_transaction():
            return state

        # Should not reach here, but return empty docs if somehow it does
        return SystemDocumentation()

    @action_executor(action_key="consciousness_update_system_docs")
    async def update_system_documentation(
        self,
        updates: dict[str, Any],
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Update system documentation.

        Args:
            updates: Fields to update
            tenant_id: Tenant identifier (uses capability default if None)
            session_id: Session identifier (uses capability default if None)
        """
        effective_tenant = tenant_id or self.tenant_id
        effective_session = session_id or self.session_id

        state_manager = await self._get_state_manager(effective_tenant, effective_session)
        async for state in state_manager.write_transaction():
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.last_updated = time.time()

        logger.info(f"Updated system documentation for tenant {effective_tenant}")

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
            key=f"{self.scope_id}:consciousness:metrics:{int(time.time() * 1000)}",
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
            key=f"{self.scope_id}:consciousness:self_concept_update:{int(time.time() * 1000)}",
            value={"updated_fields": list(updates.keys()), "timestamp": time.time()},
            created_by=self.scope_id,
            tags={"consciousness", "self_concept"},
        )

        return self._self_concept

