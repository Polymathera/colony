"""Memory Management Agent - system administrator for memory lifecycle.

The MemoryManagementAgent is a specialized system administrator agent that:
1. Monitors agent lifecycle events (creation, termination)
2. Recycles memories of terminated agents to collective memories
3. Initializes new agents' LTM from collective memory
4. Maintains tenant-level and system-level memories

There is typically one MemoryManagementAgent per tenant, plus one system-level
agent for global memories and cross-tenant coordination.

Example:
    ```python
    # Create tenant-level memory management agent
    memory_agent = MemoryManagementAgent(
        agent_id="memory_mgmt_tenant_acme",
        tenant_id="acme_corp",
    )
    await memory_agent.initialize()
    await memory_agent.start()

    # Or create system-level agent (tenant_id=None)
    system_memory_agent = MemoryManagementAgent(
        agent_id="memory_mgmt_system",
        tenant_id=None,  # System-level
    )
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from pydantic import PrivateAttr
from overrides import override

from ...base import Agent, AgentCapability, AgentState, CapabilityResultFuture
from ...models import AgentSuspensionState
from ...blackboard.types import BlackboardEntry, BlackboardEvent, KeyPatternFilter
from ..actions.policies import action_executor
from .scopes import MemoryScope
from .lifecycle import AgentTerminationEvent, AgentCreationEvent
from ..hooks.decorator import hookable

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Recycling Capability
# =============================================================================


class AgentMemoryRecycler(AgentCapability):
    """Capability for recycling terminated agents' memories to collective.

    When an agent terminates, this capability:
    1. Reads the agent's private LTM (episodic, semantic, procedural)
    2. Transforms and merges into collective memory for that agent_type
    3. Deletes the agent's private memory scopes

    The recycling is triggered by AgentTerminationEvent on the control plane.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str,
    ):
        """Initialize memory recycler.

        Args:
            agent: The MemoryManagementAgent that owns this capability
            scope_id: Scope ID for this capability
        """
        super().__init__(agent=agent, scope_id=scope_id)
        self._monitor_task: asyncio.Task | None = None
        self._initialized = False

    def get_action_group_description(self) -> str:
        return (
            "Agent Memory Recycling — transfers terminated agents' memories to collective. "
            "Triggered by AgentTerminationEvent on lifecycle scope. "
            "Reads agent's private LTM scopes, transforms/merges into collective, deletes private scopes."
        )

    async def initialize(self) -> None:
        """Initialize the recycler and start monitoring lifecycle events."""
        if self._initialized:
            return

        # Start monitoring task - it will handle subscription via async for
        self._monitor_task = asyncio.create_task(self._monitor_lifecycle_events())
        self._initialized = True

        logger.info(
            f"AgentMemoryRecycler initialized for tenant={self.agent.tenant_id}"
        )

    async def shutdown(self) -> None:
        """Shutdown the recycler."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        self._initialized = False

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
    ) -> None:
        """Stream recycling events to queue."""
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=f"{self.scope_id}:*"),
        )

    async def get_result_future(self) -> CapabilityResultFuture:
        """Recycler is a persistent service without single result."""
        raise NotImplementedError(
            "AgentMemoryRecycler is a persistent service."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for AgentMemoryRecycler")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for AgentMemoryRecycler")
        pass

    # -------------------------------------------------------------------------
    # Lifecycle Event Monitoring
    # -------------------------------------------------------------------------

    async def _monitor_lifecycle_events(self) -> None:
        """Background task to monitor and process lifecycle events.
        
        Uses async for to stream events from the lifecycle scope. The iterator
        automatically handles subscription/unsubscription lifecycle.
        """
        lifecycle_scope = MemoryScope.control_plane(
            self.agent.tenant_id,
            "lifecycle",
        )

        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=lifecycle_scope,
        )

        # Stream termination events using async for
        pattern = AgentTerminationEvent.get_key_pattern(lifecycle_scope)
        try:
            async for event in blackboard.stream_events(
                filter=KeyPatternFilter(pattern=pattern)
            ):
                if event.event_type != "write":
                    continue

                # Parse termination event
                if ":terminated" in event.key:
                    await self._handle_termination(event)

        except asyncio.CancelledError:
            logger.debug("Lifecycle event monitoring cancelled")
        except Exception as e:
            logger.error(f"Error processing lifecycle event: {e}", exc_info=True)

    async def _handle_termination(self, event: BlackboardEvent) -> None:
        """Handle agent termination event."""
        try:
            termination = AgentTerminationEvent(**event.value)
            logger.info(
                f"Processing termination for agent {termination.agent_id} "
                f"(type={termination.agent_type})"
            )

            # Recycle memories
            result = await self.recycle_agent_memories(
                terminated_agent_id=termination.agent_id,
                agent_type=termination.agent_type,
                memory_scopes=termination.memory_scopes,
            )

            logger.info(
                f"Recycled memories for {termination.agent_id}: {result}"
            )

            # Mark event as processed
            await self._mark_event_processed(event.key, result)

        except Exception as e:
            logger.error(f"Failed to recycle memories for event {event.key}: {e}")

    # -------------------------------------------------------------------------
    # Memory Recycling
    # -------------------------------------------------------------------------

    @action_executor(action_key="recycle_agent_memories")
    async def recycle_agent_memories(
        self,
        terminated_agent_id: str,
        agent_type: str,
        memory_scopes: list[str] | None = None,
    ) -> dict[str, int]:
        """Recycle a terminated agent's memories to collective memory.

        Args:
            terminated_agent_id: ID of the terminated agent
            agent_type: Type of the agent (for collective scope routing)
            memory_scopes: List of scope IDs to recycle (auto-detected if None)

        Returns:
            Dict mapping memory type to count of recycled entries
        """
        results: dict[str, int] = {}

        # Default scopes if not provided
        if memory_scopes is None:
            memory_scopes = [
                MemoryScope.agent_ltm_episodic(terminated_agent_id),
                MemoryScope.agent_ltm_semantic(terminated_agent_id),
                MemoryScope.agent_ltm_procedural(terminated_agent_id),
            ]

        # Process each LTM scope
        for scope_id in memory_scopes:
            try:
                # Determine memory type from scope
                if ":ltm:episodic" in scope_id:
                    target_scope = MemoryScope.collective_episodic(agent_type)
                    memory_type = "episodic"
                elif ":ltm:semantic" in scope_id:
                    target_scope = MemoryScope.collective_semantic(agent_type)
                    memory_type = "semantic"
                elif ":ltm:procedural" in scope_id:
                    target_scope = MemoryScope.collective_procedural(agent_type)
                    memory_type = "procedural"
                else:
                    # Skip non-LTM scopes (STM, working, etc.)
                    logger.debug(f"Skipping non-LTM scope: {scope_id}")
                    continue

                # Read entries from agent's private scope
                entries = await self._read_scope(scope_id)

                if not entries:
                    results[memory_type] = 0
                    continue

                # Merge into collective
                count = await self._merge_to_collective(
                    entries=entries,
                    target_scope=target_scope,
                    source_agent_id=terminated_agent_id,
                    agent_type=agent_type,
                )

                results[memory_type] = count

                # Clean up agent's private scope
                await self._cleanup_scope(scope_id)

            except Exception as e:
                logger.error(f"Failed to recycle scope {scope_id}: {e}")
                results[scope_id] = -1  # Error indicator

        return results

    async def _read_scope(self, scope_id: str) -> list[BlackboardEntry]:
        """Read all entries from a scope."""
        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=scope_id,
        )

        entries = await blackboard.query(
            namespace=f"{scope_id}:*",
            limit=10000,  # High limit for memory recycling
        )

        return entries

    async def _merge_to_collective(
        self,
        entries: list[BlackboardEntry],
        target_scope: str,
        source_agent_id: str,
        agent_type: str,
    ) -> int:
        """Merge entries into collective memory.

        TODO: Add intelligent merging with deduplication and relevance filtering.
        Current implementation does simple copy with attribution metadata.
        """
        if not entries:
            return 0

        target_blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=target_scope,
        )

        count = 0
        now = time.time()

        for entry in entries:
            try:
                # Generate new key for collective scope
                # Preserve data ID but change scope prefix
                original_key = entry.key
                data_suffix = original_key.split(":")[-1]  # Get data ID
                new_key = f"{target_scope}:{data_suffix}"

                # Add attribution metadata
                new_metadata = {
                    **entry.metadata,
                    "source_agent_id": source_agent_id,
                    "source_agent_type": agent_type,
                    "recycled_at": now,
                    "original_key": original_key,
                }

                # Write to collective scope
                await target_blackboard.write(
                    key=new_key,
                    value=entry.value,
                    agent_id=self.agent.agent_id,
                    tags=entry.tags | {"recycled", f"from_agent:{source_agent_id}"},
                    metadata=new_metadata,
                    ttl_seconds=None,  # Collective memories don't expire
                )

                count += 1

            except Exception as e:
                logger.error(f"Failed to merge entry {entry.key}: {e}")

        return count

    async def _cleanup_scope(self, scope_id: str) -> None:
        """Clean up a scope after recycling."""
        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=scope_id,
        )

        # Delete all entries in scope
        entries = await blackboard.query(
            namespace=f"{scope_id}:*",
            limit=10000,
        )

        for entry in entries:
            try:
                await blackboard.delete(entry.key, agent_id=self.agent.agent_id)
            except Exception as e:
                logger.warning(f"Failed to delete {entry.key}: {e}")

        logger.debug(f"Cleaned up scope {scope_id}: {len(entries)} entries deleted")

    async def _mark_event_processed(
        self,
        event_key: str,
        result: dict[str, int],
    ) -> None:
        """Mark a termination event as processed."""
        # Update the event with processing result
        lifecycle_scope = MemoryScope.control_plane(
            self.agent.tenant_id,
            "lifecycle",
        )

        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=lifecycle_scope,
        )

        # Read current event
        entry = await blackboard.read_entry(event_key)
        if entry:
            # Update metadata
            entry.metadata["processed_at"] = time.time()
            entry.metadata["recycle_result"] = result
            entry.tags.add("processed")
            entry.tags.discard("memory_recycle_pending")

            await blackboard.write(
                key=event_key,
                value=entry.value,
                agent_id=self.agent.agent_id,
                tags=entry.tags,
                metadata=entry.metadata,
                expected_version=entry.version,
            )


# =============================================================================
# Collective Memory Initializer
# =============================================================================


class CollectiveMemoryInitializer(AgentCapability):
    """Capability for initializing new agents' LTM from collective memory.

    When a new agent is created, this capability:
    1. Reads relevant entries from collective memory for that agent_type
    2. Copies entries to the new agent's LTM (primarily procedural)
    3. Optionally samples episodic/semantic memories

    This enables transfer learning from terminated agents to new ones.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str,
    ):
        """Initialize collective memory initializer.

        Args:
            agent: The MemoryManagementAgent that owns this capability
            scope_id: Scope ID for this capability
        """
        super().__init__(agent=agent, scope_id=scope_id)
        self._monitor_task: asyncio.Task | None = None
        self._initialized = False

    def get_action_group_description(self) -> str:
        return (
            "Collective Memory Initialization — seeds new agents' LTM from collective memory. "
            "Triggered by AgentCreationEvent. Enables transfer learning: "
            "terminated agents' knowledge flows to new agents of the same type."
        )

    async def initialize(self) -> None:
        """Initialize and start monitoring for new agent events."""
        if self._initialized:
            return

        # Start monitoring task - it will handle subscription via async for
        self._monitor_task = asyncio.create_task(self._monitor_creation_events())
        self._initialized = True

        logger.info(
            f"CollectiveMemoryInitializer initialized for tenant={self.agent.tenant_id}"
        )

    async def shutdown(self) -> None:
        """Shutdown the initializer."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        self._initialized = False

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
    ) -> None:
        """Stream initialization events to queue."""
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=f"{self.scope_id}:*"),
        )

    async def get_result_future(self) -> CapabilityResultFuture:
        """Initializer is a persistent service without single result."""
        raise NotImplementedError(
            "CollectiveMemoryInitializer is a persistent service."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for CollectiveMemoryInitializer")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for CollectiveMemoryInitializer")
        pass

    # -------------------------------------------------------------------------
    # Creation Event Monitoring
    # -------------------------------------------------------------------------

    async def _monitor_creation_events(self) -> None:
        """Background task to monitor and process creation events.
        
        Uses async for to stream events from the lifecycle scope. The iterator
        automatically handles subscription/unsubscription lifecycle.
        """
        lifecycle_scope = MemoryScope.control_plane(
            self.agent.tenant_id,
            "lifecycle",
        )

        blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=lifecycle_scope,
        )

        # Stream creation events using async for
        pattern = AgentCreationEvent.get_key_pattern(lifecycle_scope)
        try:
            async for event in blackboard.stream_events(
                filter=KeyPatternFilter(pattern=pattern)
            ):
                if event.event_type != "write":
                    continue

                # Parse creation event
                if ":created" in event.key:
                    await self._handle_creation(event)

        except asyncio.CancelledError:
            logger.debug("Creation event monitoring cancelled")
        except Exception as e:
            logger.error(f"Error processing creation event: {e}", exc_info=True)

    async def _handle_creation(self, event: BlackboardEvent) -> None:
        """Handle agent creation event."""
        try:
            creation = AgentCreationEvent(**event.value)
            logger.info(
                f"Processing creation for agent {creation.agent_id} "
                f"(type={creation.agent_type})"
            )

            # Initialize memories from collective
            count = await self.initialize_agent_memory(
                new_agent_id=creation.agent_id,
                agent_type=creation.agent_type,
            )

            logger.info(
                f"Initialized {count} memories for new agent {creation.agent_id}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize memories for event {event.key}: {e}")

    # -------------------------------------------------------------------------
    # Memory Initialization
    # -------------------------------------------------------------------------

    @action_executor(action_key="initialize_agent_memory")
    async def initialize_agent_memory(
        self,
        new_agent_id: str,
        agent_type: str,
        include_episodic: bool = False,
        include_semantic: bool = False,
        max_entries_per_type: int = 100,
    ) -> int:
        """Initialize a new agent's LTM from collective memory.

        By default, only procedural memory is copied (prompts, skills).
        Episodic and semantic can be optionally included.

        Args:
            new_agent_id: ID of the new agent
            agent_type: Type of the agent
            include_episodic: Also copy episodic memories
            include_semantic: Also copy semantic memories
            max_entries_per_type: Max entries to copy per memory type

        Returns:
            Total number of memories initialized
        """
        count = 0

        # Always initialize procedural memory (prompts, skills, self-concept)
        procedural_count = await self._copy_collective_to_agent(
            source_scope=MemoryScope.collective_procedural(agent_type),
            target_scope=MemoryScope.agent_ltm_procedural(new_agent_id),
            new_agent_id=new_agent_id,
            max_entries=max_entries_per_type,
        )
        count += procedural_count

        # Optionally copy episodic
        if include_episodic:
            episodic_count = await self._copy_collective_to_agent(
                source_scope=MemoryScope.collective_episodic(agent_type),
                target_scope=MemoryScope.agent_ltm_episodic(new_agent_id),
                new_agent_id=new_agent_id,
                max_entries=max_entries_per_type,
            )
            count += episodic_count

        # Optionally copy semantic
        if include_semantic:
            semantic_count = await self._copy_collective_to_agent(
                source_scope=MemoryScope.collective_semantic(agent_type),
                target_scope=MemoryScope.agent_ltm_semantic(new_agent_id),
                new_agent_id=new_agent_id,
                max_entries=max_entries_per_type,
            )
            count += semantic_count

        return count

    async def _copy_collective_to_agent(
        self,
        source_scope: str,
        target_scope: str,
        new_agent_id: str,
        max_entries: int,
    ) -> int:
        """Copy entries from collective to new agent's scope."""
        # Read from collective
        source_blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=source_scope,
        )

        entries = await source_blackboard.query(
            namespace=f"{source_scope}:*",
            limit=max_entries,
        )

        if not entries:
            return 0

        # Write to agent's scope
        target_blackboard = await self.agent.get_blackboard(
            scope="shared",
            scope_id=target_scope,
        )

        now = time.time()
        count = 0

        for entry in entries:
            try:
                # Generate new key for agent's scope
                data_suffix = entry.key.split(":")[-1]
                new_key = f"{target_scope}:{data_suffix}"

                # Update metadata
                new_metadata = {
                    **entry.metadata,
                    "initialized_from_collective": True,
                    "collective_source": source_scope,
                    "initialized_at": now,
                }

                # Remove recycling metadata
                new_metadata.pop("recycled_at", None)
                new_metadata.pop("source_agent_id", None)

                # Copy entry
                await target_blackboard.write(
                    key=new_key,
                    value=entry.value,
                    agent_id=new_agent_id,  # Owned by new agent
                    tags=entry.tags - {"recycled"} | {"from_collective"},
                    metadata=new_metadata,
                )

                count += 1

            except Exception as e:
                logger.error(f"Failed to copy entry {entry.key}: {e}")

        return count


# =============================================================================
# Collective Memory Maintainer
# =============================================================================


class CollectiveMemoryMaintainer(AgentCapability):
    """Capability for maintaining collective memory health.

    Periodic maintenance tasks:
    - Deduplicate similar entries
    - Prune low-value entries
    - Promote high-value entries to system-level
    - Consolidate and summarize

    TODO: Implement maintenance tasks
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str,
    ):
        super().__init__(agent=agent, scope_id=scope_id)

    def get_action_group_description(self) -> str:
        return (
            "Collective Memory Maintenance — periodic health maintenance of collective memory. "
            "Deduplicates similar entries, prunes low-value ones, promotes high-value to system-level. "
            "Can filter by agent_types for targeted maintenance."
        )

    async def initialize(self) -> None:
        """Initialize maintainer."""
        # TODO: Start periodic maintenance task
        logger.info(
            f"CollectiveMemoryMaintainer initialized for tenant={self.agent.tenant_id}"
        )

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
    ) -> None:
        """Stream maintenance events."""
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=f"{self.scope_id}:*"),
        )

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Maintainer is a persistent service."""
        raise NotImplementedError(
            "CollectiveMemoryMaintainer is a persistent service."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for CollectiveMemoryMaintainer")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for CollectiveMemoryMaintainer")
        pass

    @action_executor(action_key="maintain_collective_memory")
    async def maintain(
        self,
        agent_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run maintenance on collective memories.

        Args:
            agent_types: Specific agent types to maintain (all if None)

        Returns:
            Maintenance statistics
        """
        # TODO: Implement actual maintenance logic
        # - Query collective scopes for each agent_type
        # - Run deduplication
        # - Run pruning based on relevance/access
        # - Promote high-value entries to system scope

        logger.info("Collective memory maintenance requested")
        return {"status": "not_implemented"}


# =============================================================================
# Memory Management Agent
# =============================================================================


class MemoryManagementAgent(Agent):
    """System administrator agent for memory lifecycle management.

    The MemoryManagementAgent is a specialized service agent that:
    1. Monitors agent lifecycle events (creation, termination)
    2. Recycles memories of terminated agents to collective memories
    3. Initializes new agents' LTM from collective memory
    4. Maintains tenant-level and system-level collective memories

    Deployment modes:
    - **Tenant-level**: One per tenant, manages that tenant's agents
    - **System-level**: One global (tenant_id=None), manages cross-tenant coordination

    The agent runs persistently (not tied to specific tasks) and reacts to
    lifecycle events on the control plane blackboard scope.

    Architecture:
        ```
        Control Plane (lifecycle events)
              │
              ▼
        MemoryManagementAgent
        ├── CollectiveMemoryInitializer  → Initialize new agents' LTM
        ├── AgentMemoryRecycler          → Recycle terminated agents' memories
        └── CollectiveMemoryMaintainer   → Periodic maintenance
              │
              ▼
        Collective Memory Scopes
        (agent_type:{type}:collective:*)

        ┌─────────────────────────────────────────────────┐
        │                        CONTROL PLANE            │
        │   control_plane:tenant:{tenant_id}:lifecycle    │
        │   ┌─────────────────┐    ┌──────────────────┐   │
        │   │ AgentCreation   │    │ AgentTermination │   │
        │   │ Events          │    │ Events           │   │
        │   └────────┬────────┘    └────────┬─────────┘   │
        └────────────┼──────────────────────┼─────────────┘
                     │                      │
                     ▼                      ▼
            ┌────────────────────────────────────────┐
            │       MemoryManagementAgent            │
            │  ┌──────────────────────────────┐      │
            │  │ CollectiveMemoryInitializer  │──────┼───▶ Initialize new agent's LTM
            │  └──────────────────────────────┘      │     from collective
            │  ┌──────────────────────────────┐      │
            │  │ AgentMemoryRecycler          │──────┼───▶ Recycle terminated agent's
            │  └──────────────────────────────┘      │     LTM to collective
            │  ┌──────────────────────────────┐      │
            │  │ CollectiveMemoryMaintainer   │──────┼───▶ Periodic maintenance
            │  └──────────────────────────────┘      │
            └────────────────────────────────────────┘
                              │
                              ▼
            ┌────────────────────────────────────────┐
            │         COLLECTIVE MEMORY              │
            │  agent_type:{type}:collective:*        │
            └────────────────────────────────────────┘
        ```

    Example:
        ```python
        from polymathera.colony.agents import get_agent_system
        from polymathera.colony.agents.models import AgentSpawnSpec

        # Spawn via AgentSystemDeployment (standard pattern)
        # Can be called from anywhere in the Ray cluster
        agent_system = get_agent_system()

        # Create tenant-level memory management agent
        agent_ids = await agent_system.spawn_agents([
            AgentSpawnSpec(
                agent_type="polymathera.colony.agents.patterns.memory.management.MemoryManagementAgent",
                agent_id="memory_mgmt_tenant_acme",
                tenant_id="acme_corp",
                metadata={},
            ),
        ])

        # System-level agent (no tenant)
        agent_ids = await agent_system.spawn_agents([
            AgentSpawnSpec(
                agent_type="polymathera.colony.agents.patterns.memory.management.MemoryManagementAgent",
                agent_id="memory_mgmt_system",
                tenant_id=None,  # System-level
                metadata={},  # System-level
            ),
        ])
        ```

    Integration:
        Regular agents emit lifecycle events via MemoryLifecycleHooks already
        part of the default memory hierarchy (create_default_memory_hierarchy):
        - AgentCreationEvent: Emitted when agent.initialize() completes
        - AgentTerminationEvent: Emitted when agent.stop() is called

        The MemoryManagementAgent listens to these events and:
        - On creation: Initializes new agent's LTM from collective
        - On termination: Recycles agent's LTM to collective, cleans up
    """

    # Capabilities - created during initialize
    _recycler: AgentMemoryRecycler | None = PrivateAttr(default=None)
    _initializer: CollectiveMemoryInitializer | None = PrivateAttr(default=None)
    _maintainer: CollectiveMemoryMaintainer | None = PrivateAttr(default=None)

    # Running state
    _stopped: bool = PrivateAttr(default=False)

    async def initialize(self) -> None:
        """Initialize the memory management agent and its capabilities.

        Creates and initializes:
        - AgentMemoryRecycler: Handles terminated agent memories
        - CollectiveMemoryInitializer: Initializes new agents from collective
        - CollectiveMemoryMaintainer: Periodic maintenance
        """
        # Note: We don't call super().initialize() because MemoryManagementAgent
        # doesn't use the standard action policy / context page source pattern.
        # It's a simpler service agent with event-driven capabilities.

        self.state = AgentState.INITIALIZED

        # Create capabilities
        base_scope = f"memory_mgmt:{self.tenant_id}"

        self._recycler = AgentMemoryRecycler(
            agent=self,
            scope_id=f"{base_scope}:recycler",
        )

        self._initializer = CollectiveMemoryInitializer(
            agent=self,
            scope_id=f"{base_scope}:initializer",
        )

        self._maintainer = CollectiveMemoryMaintainer(
            agent=self,
            scope_id=f"{base_scope}:maintainer",
        )

        # Add capabilities to agent
        self.add_capability(self._recycler)
        self.add_capability(self._initializer)
        self.add_capability(self._maintainer)

        # Initialize capabilities (starts background monitoring tasks)
        await self._recycler.initialize()
        await self._initializer.initialize()
        await self._maintainer.initialize()

        logger.info(
            f"MemoryManagementAgent initialized: {self.agent_id} "
            f"(tenant={self.tenant_id})"
        )

    @hookable
    @override
    async def start(self) -> None:
        """Start the memory management agent.

        The agent runs until stop() is called. Background tasks in
        capabilities handle event processing.
        """
        self.state = AgentState.RUNNING
        self._running = True
        self._stopped = False

        logger.info(f"MemoryManagementAgent started: {self.agent_id}")

        # The agent doesn't have a main loop - capabilities run their own
        # background tasks via asyncio.create_task in initialize().
        # We just need to stay alive until stop() is called.

    @hookable
    @override
    async def stop(self) -> None:
        """Stop the memory management agent and shutdown capabilities."""
        self._stopped = True
        self._stop_requested = True

        # Shutdown capabilities (cancels background tasks)
        if self._recycler:
            await self._recycler.shutdown()
        if self._initializer:
            await self._initializer.shutdown()

        self.state = AgentState.STOPPED
        self._running = False

        logger.info(f"MemoryManagementAgent stopped: {self.agent_id}")

    @hookable
    @override
    async def run_step(self) -> None:
        """Execute one step of memory management agent logic.

        For MemoryManagementAgent, the capabilities handle their own
        event-driven loops via background tasks created during initialize().
        This run_step() is a no-op since there's no action policy driving
        behavior - instead, the capabilities react to lifecycle events.

        The agent manager will call this repeatedly, but the actual work
        happens in the capability background tasks.
        """
        # Capabilities have their own background tasks processing events.
        # No action policy iteration needed here.
        await asyncio.sleep(0.1)

    # -------------------------------------------------------------------------
    # Direct API (for programmatic access)
    # -------------------------------------------------------------------------

    async def recycle_agent_memories(
        self,
        terminated_agent_id: str,
        agent_type: str,
        memory_scopes: list[str] | None = None,
    ) -> dict[str, int]:
        """Manually trigger memory recycling for a terminated agent.

        This can be called directly if automatic event-driven recycling
        is not desired.

        Args:
            terminated_agent_id: ID of the terminated agent
            agent_type: Type of the agent
            memory_scopes: Specific scopes to recycle (auto-detected if None)

        Returns:
            Dict mapping memory type to count of recycled entries
        """
        if not self._recycler:
            raise RuntimeError("Agent not initialized")
        return await self._recycler.recycle_agent_memories(
            terminated_agent_id=terminated_agent_id,
            agent_type=agent_type,
            memory_scopes=memory_scopes,
        )

    async def initialize_agent_memory(
        self,
        new_agent_id: str,
        agent_type: str,
        include_episodic: bool = False,
        include_semantic: bool = False,
        max_entries_per_type: int = 100,
    ) -> int:
        """Manually initialize a new agent's LTM from collective memory.

        This can be called directly if automatic event-driven initialization
        is not desired.

        Args:
            new_agent_id: ID of the new agent
            agent_type: Type of the agent
            include_episodic: Also copy episodic memories
            include_semantic: Also copy semantic memories
            max_entries_per_type: Max entries per memory type

        Returns:
            Total number of memories initialized
        """
        if not self._initializer:
            raise RuntimeError("Agent not initialized")
        return await self._initializer.initialize_agent_memory(
            new_agent_id=new_agent_id,
            agent_type=agent_type,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            max_entries_per_type=max_entries_per_type,
        )

    async def run_maintenance(
        self,
        agent_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Manually trigger maintenance on collective memories.

        Args:
            agent_types: Specific agent types to maintain (all if None)

        Returns:
            Maintenance statistics
        """
        if not self._maintainer:
            raise RuntimeError("Agent not initialized")
        return await self._maintainer.maintain(agent_types=agent_types)

