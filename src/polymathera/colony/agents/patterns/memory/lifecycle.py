"""Memory Lifecycle Hooks - handles memory operations during agent lifecycle.

This module provides lifecycle hooks that:
1. Drain working memory to STM on task completion
2. Flush all memories before agent shutdown
3. Emit termination events for MemoryManagementAgent to recycle

These hooks integrate the memory system with agent lifecycle events using
the hook system defined in `polymathera.colony.distributed.hooks`.

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
from ...scopes import MemoryScope, BlackboardScope, get_scope_prefix
from ...blackboard.types import BlackboardEvent, KeyPatternFilter
from ....distributed.ray_utils import serving
from ....distributed.hooks import (
    HookContext,
    HookType,
    Pointcut,
    hook_handler,
)
from .context import AgentContextEngine
from .working import WorkingMemoryCapability
from ...blackboard.protocol import LifecycleSignalProtocol

logger = logging.getLogger(__name__)


class MemoryLifecycleHooks(AgentCapability):
    """Hooks for memory operations during agent lifecycle events.

    This capability registers hooks that:
    1. Drain working memory to STM when tasks complete
    2. Consolidate all memories before agent shutdown
    3. Emit termination events for MemoryManagementAgent

    The hooks use the declarative `@hook_handler` decorator which
    auto-registers during `initialize()`.
    """


    def __init__(
        self,
        agent: Agent,
        stm_scope_id: str | None = None,
        capability_key: str = "memory_lifecycle_hooks",
        app_name: str | None = None,
    ):
        """Initialize memory lifecycle hooks.

        Args:
            agent: Agent that owns this capability
            stm_scope_id: Target STM scope (defaults to agent's STM)
            capability_key: Override for the capability dict key (allows multiple
                instances of MemoryLifecycleHooks with distinct keys).
            app_name: The `serving.Application` name where the agent system resides.
                    Required when creating detached handles from outside any `serving.deployment`.
        """
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(BlackboardScope.AGENT, agent, namespace="memory_lifecycle"),
            input_patterns=[], # This capability to the colony control plane lifecycle scope to receive agent creation/termination events.
            capability_key=capability_key,
            app_name=app_name,
        )
        self._stm_scope_id = stm_scope_id or MemoryScope.agent_stm(agent)

    def get_action_group_description(self) -> str:
        return (
            "Memory Lifecycle Hooks — automatic memory operations at agent lifecycle events. "
            "Drains working memory to STM on task completion, consolidates before shutdown, "
            "emits termination events for MemoryManagementAgent. "
            "Purely hook-based (no plannable actions) — fires automatically."
        )

    async def initialize(self) -> None:
        """Initialize lifecycle hooks.

        This auto-registers all @hook_handler decorated methods.
        """
        await super().initialize()
        logger.info(f"MemoryLifecycleHooks initialized for {self.agent.agent_id}")

    @override
    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent]
    ) -> None:
        """Stream lifecycle events from colony control plane.

        Lifecycle events (creation, termination) are emitted to the colony
        control plane scope, not the agent-local scope. Subscribe there
        to actually receive them.
        """
        lifecycle_bb = await self.agent.get_blackboard(
            scope_id=MemoryScope.colony_control_plane("lifecycle")
        )
        lifecycle_bb.stream_events_to_queue(
            event_queue,
            pattern=LifecycleSignalProtocol.created_pattern(),
            event_types={"write"},
        )
        lifecycle_bb.stream_events_to_queue(
            event_queue,
            pattern=LifecycleSignalProtocol.terminated_pattern(),
            event_types={"write"},
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

    @hook_handler(
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
            await self._emit_creation_event(ctx)
        except Exception as e:
            logger.error(f"Failed to emit creation event: {e}")

        return result

    @hook_handler(
        pointcut=Pointcut.pattern("*.task_complete"),  # TODO: Make sure this pointcut exists.
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

    @hook_handler(
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
                await ctx_engine.ingest_pending()
            except Exception as e:
                logger.error(f"Failed to consolidate memories on shutdown: {e}")

        # 3. Emit termination event for MemoryManagementAgent
        try:
            await self._emit_termination_event(ctx)
        except Exception as e:
            logger.error(f"Failed to emit termination event: {e}")

        return ctx

    # -------------------------------------------------------------------------
    # Lifecycle Events
    # -------------------------------------------------------------------------

    async def _emit_creation_event(self, ctx: HookContext) -> None:
        """Emit event to notify MemoryManagementAgent of agent creation.

        The MemoryManagementAgent listens to these events and:
        1. Reads collective memory for this agent_type
        2. Initializes the new agent's LTM with relevant entries
        """
        lifecycle_scope = MemoryScope.colony_control_plane("lifecycle")

        blackboard = await self.agent.get_blackboard(
            scope_id=lifecycle_scope,
        )

        agent: Agent = ctx.instance
        agent_id = agent.agent_id

        creation_data = AgentCreationEvent(
            agent_id=agent_id,
            agent_type=agent.agent_type,
            timestamp=time.time(),
        )

        await blackboard.write(
            key=LifecycleSignalProtocol.created_key(agent_id),
            value=creation_data.model_dump(),
            created_by=agent_id,
            tags={"agent_created", "memory_init_pending"},
            metadata={
                "agent_type": agent.agent_type,
            },
        )

        logger.info(
            f"Emitted creation event for {agent_id} "
            f"(type={agent.agent_type})"
        )

    async def _emit_termination_event(self, ctx: HookContext) -> None:
        """Emit event to notify MemoryManagementAgent of termination.

        The MemoryManagementAgent listens to these events and:
        1. Reads the agent's private LTM scopes
        2. Merges relevant memories into collective memory
        3. Cleans up the agent's private scopes
        """
        lifecycle_scope = MemoryScope.colony_control_plane("lifecycle")

        blackboard = await self.agent.get_blackboard(
            scope_id=lifecycle_scope,
        )

        # List all memory scopes that should be recycled
        agent: Agent = ctx.instance
        agent_id = agent.agent_id
        memory_scopes = [
            MemoryScope.agent_stm(agent),
            MemoryScope.agent_ltm_episodic(agent),
            MemoryScope.agent_ltm_semantic(agent),
            MemoryScope.agent_ltm_procedural(agent),
        ]

        termination_data = AgentTerminationEvent(
            agent_id=agent_id,
            agent_type=agent.agent_type,
            timestamp=time.time(),
            memory_scopes=memory_scopes,
        )

        await blackboard.write(
            key=LifecycleSignalProtocol.terminated_key(agent_id),
            value=termination_data.model_dump(),
            created_by=agent_id,
            tags={"agent_terminated", "memory_recycle_pending"},
            metadata={
                "agent_type": agent.agent_type,
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
    syscontext: serving.ExecutionContext = Field(
        default_factory=serving.require_execution_context,
        description="System context the agent belongs to",
    )
    timestamp: float = Field(description="When termination occurred")
    memory_scopes: list[str] = Field(
        description="List of memory scope IDs to recycle"
    )


class AgentCreationEvent(BaseModel):
    """Event emitted when a new agent is created.

    MemoryManagementAgent listens for these events to initialize
    the new agent's LTM from collective memory.
    """

    agent_id: str = Field(description="ID of the new agent")
    agent_type: str = Field(description="Type of the agent")
    syscontext: serving.ExecutionContext = Field(
        default_factory=serving.require_execution_context,
        description="System context the agent belongs to",
    )
    timestamp: float = Field(description="When creation occurred")

