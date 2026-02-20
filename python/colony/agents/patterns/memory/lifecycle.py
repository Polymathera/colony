"""Memory Lifecycle Hooks - handles memory operations during agent lifecycle.

This module provides lifecycle hooks that:
1. Drain working memory to STM on task completion
2. Flush all memories before agent shutdown
3. Emit termination events for MemoryManagementAgent to recycle

These hooks integrate the memory system with agent lifecycle events using
the hook system defined in `polymathera.colony.agents.patterns.hooks`.

Example:
    ```python
    from polymathera.colony.agents.patterns.memory.lifecycle import MemoryLifecycleHooks

    # Add to agent after memory capabilities are set up
    lifecycle_hooks = MemoryLifecycleHooks(
        agent=agent,
        scope_id=f"{agent.agent_id}:memory_lifecycle",
    )
    await lifecycle_hooks.initialize()
    agent.add_capability(lifecycle_hooks)
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from overrides import override
from pydantic import BaseModel, Field

from ...base import AgentCapability, CapabilityResultFuture, Agent
from ...models import AgentSuspensionState
from ...blackboard.types import BlackboardEvent, KeyPatternFilter
from ..hooks.types import HookContext, HookType
from ..hooks.pointcuts import Pointcut
from ..hooks.decorator import register_hook
from .scopes import MemoryScope
from .context import AgentContextEngine
from .working import WorkingMemoryCapability

logger = logging.getLogger(__name__)


class MemoryLifecycleHooks(AgentCapability):
    """Hooks for memory operations during agent lifecycle events.

    This capability registers hooks that:
    1. Drain working memory to STM when tasks complete
    2. Consolidate all memories before agent shutdown
    3. Emit termination events for MemoryManagementAgent

    The hooks use the declarative `@register_hook` decorator which
    auto-registers during `initialize()`.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None,
        stm_scope_id: str | None = None,
    ):
        """Initialize memory lifecycle hooks.

        Args:
            agent: Agent that owns this capability
            scope_id: Scope ID for this capability
            stm_scope_id: Target STM scope (defaults to agent's STM)
        """
        super().__init__(
            agent=agent,
            scope_id=scope_id or f"{agent.agent_id}:memory_lifecycle",
        )
        self._stm_scope_id = stm_scope_id or MemoryScope.agent_stm(agent.agent_id)

    async def initialize(self) -> None:
        """Initialize lifecycle hooks.

        This auto-registers all @register_hook decorated methods.
        """
        await super().initialize()
        logger.info(f"MemoryLifecycleHooks initialized for {self.agent.agent_id}")

    @override
    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent]
    ) -> None:
        """Stream lifecycle events to queue.

        Streams termination events emitted by this capability.
        """
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=f"{self.scope_id}:*")
        )

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Lifecycle hooks don't have a single result."""
        raise NotImplementedError(
            "MemoryLifecycleHooks is a persistent service without a single result."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for MemoryLifecycleHooks")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for MemoryLifecycleHooks")
        pass

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    @register_hook(
        pointcut=Pointcut.pattern("Agent.start"),
        hook_type=HookType.AFTER,
        priority=100,  # Run after agent starts
    )
    async def emit_creation_event(
        self,
        ctx: HookContext,
        result: Any,
    ) -> Any:
        """Emit creation event after agent starts.

        MemoryManagementAgent listens for these events to initialize
        the new agent's LTM from collective memory.
        """
        try:
            await self._emit_creation_event()
        except Exception as e:
            logger.error(f"Failed to emit creation event: {e}")

        return result

    @register_hook(
        pointcut=Pointcut.pattern("*.task_complete"),
        hook_type=HookType.AFTER,
        priority=50,
    )
    async def drain_working_to_stm(
        self,
        ctx: HookContext,
        result: Any,
    ) -> Any:
        """Drain working memory to STM when task completes.

        Triggered after any method named `task_complete`.
        """
        working = self.agent.get_working_memory()
        if working:
            try:
                count = await working.drain_to_stm(self._stm_scope_id)
                logger.debug(f"Drained {count} entries from working to STM on task complete")
            except Exception as e:
                logger.error(f"Failed to drain working memory: {e}")

        return result

    @register_hook(
        pointcut=Pointcut.pattern("Agent.stop"),
        hook_type=HookType.BEFORE,
        priority=100,  # Run early before agent stops
    )
    async def flush_before_shutdown(self, ctx: HookContext) -> HookContext:
        """Flush all memories before agent shuts down.

        1. Drain working → STM
        2. Trigger all consolidation transfers
        3. Emit termination event for MemoryManagementAgent
        """
        logger.info(f"Flushing memories before shutdown for {self.agent.agent_id}")

        # 1. Drain working memory to STM
        working = self.agent.get_working_memory()
        if working:
            try:
                await working.drain_to_stm(self._stm_scope_id)
            except Exception as e:
                logger.error(f"Failed to drain working memory on shutdown: {e}")

        # 2. Trigger all memory transfers (consolidation)
        ctx_engine = self.agent.get_context_engine()
        if ctx_engine:
            try:
                await ctx_engine.consolidate()
            except Exception as e:
                logger.error(f"Failed to consolidate memories on shutdown: {e}")

        # 3. Emit termination event for MemoryManagementAgent
        try:
            await self._emit_termination_event()
        except Exception as e:
            logger.error(f"Failed to emit termination event: {e}")

        return ctx

    # -------------------------------------------------------------------------
    # Lifecycle Events
    # -------------------------------------------------------------------------

    async def _emit_creation_event(self) -> None:
        """Emit event to notify MemoryManagementAgent of agent creation.

        The MemoryManagementAgent listens to these events and:
        1. Reads collective memory for this agent_type
        2. Initializes the new agent's LTM with relevant entries
        """
        lifecycle_scope = MemoryScope.control_plane(
            self.agent.tenant_id,
            "lifecycle",
        )

        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=lifecycle_scope,
        )

        agent_id = self.agent.agent_id

        creation_data = AgentCreationEvent(
            agent_id=agent_id,
            agent_type=self.agent.agent_type,
            tenant_id=self.agent.tenant_id,
            timestamp=time.time(),
        )

        await blackboard.write(
            key=creation_data.get_blackboard_key(lifecycle_scope),
            value=creation_data.model_dump(),
            agent_id=agent_id,
            tags={"agent_created", "memory_init_pending"},
            metadata={
                "agent_type": self.agent.agent_type,
            },
        )

        logger.info(
            f"Emitted creation event for {agent_id} "
            f"(type={self.agent.agent_type})"
        )

    async def _emit_termination_event(self) -> None:
        """Emit event to notify MemoryManagementAgent of termination.

        The MemoryManagementAgent listens to these events and:
        1. Reads the agent's private LTM scopes
        2. Merges relevant memories into collective memory
        3. Cleans up the agent's private scopes
        """
        lifecycle_scope = MemoryScope.control_plane(
            self.agent.tenant_id,
            "lifecycle",
        )

        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=lifecycle_scope,
        )

        # List all memory scopes that should be recycled
        agent_id = self.agent.agent_id
        memory_scopes = [
            MemoryScope.agent_stm(agent_id),
            MemoryScope.agent_ltm_episodic(agent_id),
            MemoryScope.agent_ltm_semantic(agent_id),
            MemoryScope.agent_ltm_procedural(agent_id),
        ]

        termination_data = AgentTerminationEvent(
            agent_id=agent_id,
            agent_type=self.agent.agent_type,
            tenant_id=self.agent.tenant_id,
            timestamp=time.time(),
            memory_scopes=memory_scopes,
        )

        await blackboard.write(
            key=termination_data.get_blackboard_key(lifecycle_scope),
            value=termination_data.model_dump(),
            agent_id=agent_id,
            tags={"agent_terminated", "memory_recycle_pending"},
            metadata={
                "agent_type": self.agent.agent_type,
                "scope_count": len(memory_scopes),
            },
        )

        logger.info(
            f"Emitted termination event for {agent_id} "
            f"with {len(memory_scopes)} memory scopes"
        )


# =============================================================================
# Data Types
# =============================================================================


class AgentTerminationEvent(BaseModel):
    """Event emitted when an agent terminates.

    MemoryManagementAgent listens for these events to recycle
    the terminated agent's memories into collective memory.
    """

    agent_id: str = Field(description="ID of the terminated agent")
    agent_type: str = Field(description="Type of the agent for collective memory routing")
    tenant_id: str = Field(description="Tenant the agent belonged to")
    timestamp: float = Field(description="When termination occurred")
    memory_scopes: list[str] = Field(
        description="List of memory scope IDs to recycle"
    )

    def get_blackboard_key(self, scope_id: str) -> str:
        """Generate blackboard key for this event."""
        return f"{scope_id}:agent:{self.agent_id}:terminated"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching termination events."""
        return f"{scope_id}:agent:*:terminated"


class AgentCreationEvent(BaseModel):
    """Event emitted when a new agent is created.

    MemoryManagementAgent listens for these events to initialize
    the new agent's LTM from collective memory.
    """

    agent_id: str = Field(description="ID of the new agent")
    agent_type: str = Field(description="Type of the agent")
    tenant_id: str = Field(description="Tenant the agent belongs to")
    timestamp: float = Field(description="When creation occurred")

    def get_blackboard_key(self, scope_id: str) -> str:
        """Generate blackboard key for this event."""
        return f"{scope_id}:agent:{self.agent_id}:created"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching creation events."""
        return f"{scope_id}:agent:*:created"

