"""Agent base classes for the multi-agent system.

This module defines the core agent abstractions:
- AgentState: Enum for agent lifecycle states
- Agent: Base agent class with lifecycle and actions
- AgentManagerBase: Mixin for managing agents on deployments

Agents are autonomous computational entities that can:
- Read from and write to working memory (blackboard)
- Access virtual context pages via VCM
- Submit inference requests
- Communicate with other agents
- Use tools and perform actions

The guiding principle: Agent control flow should be driven by reasoning LLMs
given sufficient context, not hardcoded.
"""
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Any, Callable, AsyncIterator, TYPE_CHECKING
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import networkx as nx

from ..cluster.models import InferenceRequest, InferenceResponse
from .models import (
    ActionStatus,
    AgentState,
    ActionPolicyIterationResult,
    ActionPolicyExecutionState,
    AgentResourceRequirements,
    AgentSuspensionState,
    AgentSpawnSpec,
    AgentMetadata
)
from .blackboard import EnhancedBlackboard, BlackboardScope, BlackboardEvent
from .blackboard.types import KeyPatternFilter, EventFilter
from .patterns.hooks.decorator import hookable
from .sessions.models import AgentRun, AgentRunConfig, AgentRunEvent, RunStatus, RunResourceUsage
from ..distributed import get_polymathera
from ..distributed.state_management import StateManager
from ..distributed.ray_utils import serving
from ..system import (
    get_agent_system,
    get_llm_cluster,
    get_tool_manager,
    get_vcm,
    spawn_agents
)
from .routing import AgentAffinityRouter, SoftPageAffinityRouter
from ..vcm.page_storage import PageStorage, PageStorageConfig
from ..cluster.config import DeploymentConfig
from .patterns.hooks import AgentHookRegistry, Pointcut, HookType, ErrorMode, auto_register_hooks

if TYPE_CHECKING:
    from .patterns.memory.types import CapabilityMemoryRequirements


logger = logging.getLogger(__name__)


class ResourceExhausted(Exception):
    """Raised when replica has insufficient resources for new agent."""
    pass


class ActionPolicy(ABC):
    """Abstract base class for action policies.

    Args:
        ABC (type): Abstract base class type.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self._used_agent_capabilities: list[str] = []

    async def initialize(self) -> None:
        """Initialize action policy (override in subclasses if needed)."""
        pass

    @abstractmethod
    async def execute_iteration(self, state: ActionPolicyExecutionState) -> ActionPolicyIterationResult:
        ...

    @abstractmethod
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize policy-specific state.

        Args:
            state: AgentSuspensionState to populate with serialized state
        """
        ...

    @abstractmethod
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore policy-specific state from suspension.

        Args:
            state: AgentSuspensionState to restore from
        """
        ...

    # TODO: Move to Agent class
    def use_agent_capabilities(self, capabilities: list[str]) -> None:
        """Add agent capabilities as action providers.

        Args:
            capabilities: List of capability names to use. Extends existing list.
        """
        for capability in capabilities:
            if capability not in self._used_agent_capabilities:
                self._used_agent_capabilities.append(capability)

    async def use_agent_capability_types(self, capabilities: list[type[AgentCapability]]) -> None:
        """Instantiate and add agent capabilities to the agent and to the action policy as action providers.

        Args:
            capabilities: List of capability classes to use. Extends existing list.
        """
        for capability_cls in capabilities:
            # Get or create consistency capability
            if not self.agent.has_capability(capability_cls.get_capability_name()):
                # TODO: How to allow passing params to capability constructor?
                capability_instance = capability_cls(self.agent)
                await capability_instance.initialize()
                # TODO: How to allow passing more add_capability params?
                self.agent.add_capability(
                    capability_instance,
                    include_actions=[],
                    exclude_actions=[],
                    events_only=False,
                )

        self.use_agent_capabilities([
            capability_cls.get_capability_name()
            for capability_cls in capabilities
        ])

    def disable_agent_capabilities(self, capabilities: list[str]) -> None:
        """Remove agent capabilities from action providers.

        Args:
            capabilities: List of capability names to disable.
        """
        for capability in capabilities:
            if capability in self._used_agent_capabilities:
                self._used_agent_capabilities.remove(capability)

    def get_used_capabilities(self) -> list[AgentCapability]:
        capability_providers = [
            self.agent.get_capability(name)
            for name in self._used_agent_capabilities
            if self.agent.get_capability(name) is not None
        ]
        return capability_providers


class AgentCapability(ABC):
    """Base class for agent capabilities that agents furnish to their own action policies and to other agents.

    Capabilities encapsulate a specific aspect or protocol of agent functionality.
    The `scope_id` parameter ensures that different capability instances share
    the same blackboard namespace for communication.

    They can be used in four modes:

    1. **Local mode** (default): Capability runs within the owning agent
       ```python
       capability = MyCapability(agent=self)  # scope_id defaults to agent.agent_id
       ```

    2. **Remote mode**: Parent uses capability to communicate with child agent
       ```python
       # Parent creates capability pointing to child's scope
       capability = MyCapability(agent=parent, scope_id=child_agent_id)
       await capability.stream_events_to_queue(self.get_event_queue())
       future = await capability.get_result_future()
       result = await asyncio.wait_for(future, timeout=30.0)
       ```

    3. **Shared scope mode**: Multiple agents share the same scope (e.g., game participants)
       ```python
       # All game participants use the same game_id scope
       capability = NegotiationGameProtocol(agent=self, scope_id=game_id)
       # All agents can now see each other's events in the shared namespace
       ```

    4. **Detached mode** (NEW): Capability operates standalone without agent context
       In this mode, an AgentCapability provides limited functionality (mostly
       communication with other agents, but no inference or tool use or page graph access).
       This is useful for external (non-agentic) system components to interact with agents.
       ```python
       # Create capability without agent - useful for external systems
       capability = MyCapability(agent=None, scope_id=target_agent_id)
       # All operations work via blackboard communication
       ```

    The `scope_id` enables flexible communication patterns:
    - `agent.agent_id` (default): Agent-local scope
    - `child_agent_id`: Parent-to-child communication
    - `game_id` or `task_id`: Shared scope for group coordination

    Subclasses can override:
    - `stream_events_to_queue()`: Stream capability events to action policy (default streams {scope_id}:* writes)
    - `get_result_future()`: Get future for capability's task result (default uses {scope_id}:result:{request_id})
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        blackboard: EnhancedBlackboard | None = None,
    ):
        """Initialize capability.

        Args:
            agent: Agent using this capability (None for detached mode)
            scope_id: Blackboard scope ID. Defaults to agent.agent_id.
                Required if agent is None (detached mode).
                Can be set to:
                - child_agent_id: For parent-child communication
                - game_id: For game participants sharing a namespace
                - task_id: For agents collaborating on a shared task
            blackboard: Pre-configured blackboard (for detached mode)

        Raises:
            ValueError: If both agent and scope_id are None
        """
        self._agent = agent
        self._blackboard: EnhancedBlackboard | None = blackboard
        self._pending_request_id: str | None = None

        # In attached mode, derive scope from agent
        # In detached mode, scope_id must be provided
        if agent is not None:
            self.scope_id = scope_id or agent.agent_id
        elif scope_id is not None:
            self.scope_id = scope_id
        else:
            raise ValueError(
                "Either 'agent' or 'scope_id' must be provided. "
                "For detached mode, provide scope_id explicitly."
            )

    @property
    def agent(self) -> Agent | None:
        """Get owning agent (None in detached mode)."""
        if self._agent is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is operating in detached mode and has no owning agent or "
                "perhaps the agent reference was not passed during initialization."
            )
        return self._agent

    @property
    def is_detached(self) -> bool:
        """Check if capability is operating in detached mode."""
        return self._agent is None

    async def get_blackboard(
        self,
        backend_type: str = "redis",
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for this capability's scope.

        In attached mode: uses agent's blackboard
        In detached mode: creates/uses standalone blackboard

        Returns:
            Blackboard scoped to this capability
        """
        if self._blackboard is not None:
            return self._blackboard

        if self._agent is not None:
            # Attached mode: use agent's blackboard
            self._blackboard = await self._agent.get_blackboard(
                scope="shared",
                scope_id=self.scope_id,
                backend_type=backend_type,
                enable_events=enable_events,
            )
        else:
            # Detached mode: create standalone blackboard
            app_name = serving.get_my_app_name()

            self._blackboard = EnhancedBlackboard(
                app_name=app_name,
                scope=BlackboardScope.SHARED,
                scope_id=self.scope_id,
                backend_type=backend_type,
                enable_events=enable_events,
            )
            await self._blackboard.initialize()

        return self._blackboard

    async def publish(
        self,
        record: BaseModel,
        *,
        tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        """Write a record to the capability's scoped blackboard.

        Replaces the manual pattern of::

            bb = await self.get_blackboard()
            key = f"{self.scope_id}:analysis:result:{result.result_id}"
            await bb.write(key=key, value=result.model_dump(), created_by=...)

        With::

            await self.publish(result, tags={"analysis"})

        The key is resolved automatically from the record (via
        ``BlackboardPublishable`` protocol or legacy ``get_blackboard_key``).

        If this capability's scope is VCM-mapped (via ``mmap_application_scope()``), the
        write event is automatically picked up by the ``BlackboardContextPageSource``
        running inside the VCM. The record will eventually appear in a VCM page
        and become discoverable via ``QueryAttentionCapability``. The producer
        never needs to opt in per-record.

        Args:
            record: Any BaseModel. Must implement ``BlackboardPublishable``
                (``key_schema`` + ``key_parts``) or have ``get_blackboard_key(scope_id)``.
            tags: Tags for categorization and filtering.
            ttl_seconds: Optional TTL.

        Returns:
            The blackboard key that was written.
        """
        key = self._resolve_key(record)

        blackboard = await self.get_blackboard()
        value = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        agent_id = self._agent.agent_id if self._agent else "detached"
        await blackboard.write(
            key=key,
            value=value,
            created_by=agent_id,
            tags=tags,
            metadata=metadata,
            ttl_seconds=ttl_seconds,
        )

        return key

    def _resolve_key(self, record: BaseModel) -> str:
        """Resolve the blackboard key for a record.

        Resolution order:
        1. ``BlackboardPublishable`` protocol (``key_schema`` + ``key_parts``)
        2. Legacy ``get_blackboard_key(scope_id)`` method
        3. ``ValueError`` if neither is available

        Args:
            record: The record to resolve a key for.

        Returns:
            Resolved blackboard key string.

        Raises:
            ValueError: If no key resolution method is available.
        """
        # Try BlackboardPublishable protocol first
        from .blackboard.keys import BlackboardPublishable
        if isinstance(record, BlackboardPublishable):
            schema = record.key_schema()
            parts = record.key_parts()
            return schema.format(scope_id=self.scope_id, **parts)

        # Fall back to legacy get_blackboard_key
        if hasattr(record, "get_blackboard_key"):
            return record.get_blackboard_key(self.scope_id)

        raise ValueError(
            f"Cannot resolve blackboard key for {type(record).__name__}. "
            f"Implement BlackboardPublishable (key_schema + key_parts) or "
            f"add a get_blackboard_key(scope_id) method."
        )

    @classmethod
    def get_capability_name(cls) -> str:
        """Get the name of the capability.

        Returns:
            Name of the capability
        """
        return cls.__name__

    def get_memory_requirements(self) -> CapabilityMemoryRequirements | None:
        """Declare this capability's memory requirements.

        Capabilities that depend on specific memory scopes, produce data with
        certain tags, or consume data with certain tags should override this
        method to return a ``CapabilityMemoryRequirements`` describing those
        dependencies.

        The ``AgentContextEngine`` uses these declarations during
        ``validate_memory_system()`` to verify that the agent's memory
        hierarchy satisfies all capability requirements (e.g., required
        scopes exist, tag producers/consumers are matched).

        Returns:
            Memory requirements, or None if this capability has no memory
            dependencies (default).
        """
        return None

    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream capability-specific events to the given queue.

        Default implementation streams all "write" events matching {scope_id}:*.
        This includes all events within the capability's blackboard scope (e.g.,
        requests, results from child agents, status updates, etc.).
        Subclasses can override to customize the event filter.

        Other event patterns/filters can be encapsulated within the capability itself.
        Subclasses can define what events are relevant to their protocol.

        Event handlers can use more specific event patterns if needed.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            pattern=f"{self.scope_id}:*",
            event_types={"write"},
        )

    async def stream_events(
        self,
        *,
        filter: EventFilter | None = None,
        pattern: str = "*",
        event_types: set[str] | None = None,
        queue_maxsize: int = 0,
        until: Callable[[], bool] | None = None,
        timeout: float | None = 10.0,
    ) -> AsyncIterator[BlackboardEvent]:
        """Yield blackboard events as an async iterator.

        This packages the common pattern:
        - create an asyncio.Queue
        - subscribe events into it
        - `async for` pull events from the queue forever
        - unsubscribe automatically on cancellation / exit

        This is useful for long-running background tasks:

            async def monitor():
                async for event in blackboard.stream_events(
                    filter=KeyPatternFilter("tenant:...:scope:*"),
                    until=lambda: self._stopped,
                ):
                    ...

        Args:
            filter: Optional event filter
            pattern: Key pattern to match events (glob-style). Used if `filter` is None.
            event_types: Optional set of event types to filter (e.g., {"write"}).
            queue_maxsize: If > 0, bound the internal queue and drop events when full.
            until: Optional callable that returns True to exit the stream.
            timeout: Optional timeout for waiting on events.

        Yields:
            BlackboardEvent objects as they arrive.
        """
        event_queue: asyncio.Queue[BlackboardEvent]
        if queue_maxsize > 0:
            event_queue = asyncio.Queue(maxsize=queue_maxsize)
        else:
            event_queue = asyncio.Queue()

        # Reuse stream_events_to_queue for subscription logic
        await self.stream_events_to_queue(
            event_queue=event_queue,
            filter=filter,
            pattern=pattern,
            event_types=event_types,
        )

        if until is None:
            until = lambda: False

        try:
            while not until():
                try:
                    yield await asyncio.wait_for(event_queue.get(), timeout=timeout)

                except asyncio.TimeoutError:
                    continue
        finally:
            # Best-effort cleanup: remove the callback that was added by stream_events_to_queue
            if self._subscribed_callbacks:
                callback = self._subscribed_callbacks.pop()
                try:
                    self.unsubscribe(callback)
                except Exception:
                    pass

    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for this capability's task result.

        Default implementation returns a future that resolves when
        {scope_id}:result:{request_id} is written to the blackboard.
        Subclasses can override to customize the result key pattern.

        Each capability instance represents ONE task (which can be hierarchical
        and long-running). This returns a future that resolves when that task
        completes.

        Returns:
            Future that resolves with the task result
        """
        blackboard = await self.get_blackboard()
        # TODO: Should we listen to all request_ids?
        result_key = f"{self.scope_id}:result:{self._pending_request_id or 'default'}"
        return CapabilityResultFuture(
            result_key=result_key,
            blackboard=blackboard,
        )

    async def send_request(
        self,
        request_type: str,
        request_data: dict[str, Any],
        request_id: str | None = None,
    ) -> str:
        """Send a request to trigger/control the capability.

        Generic method for sending requests via blackboard. Subclasses may
        provide more specific methods (e.g., `abort()`, `ground_claim()`).

        Works in both attached and detached modes.

        Args:
            request_type: Type of request (e.g., "abort", "ground_claim")
            request_data: Request payload
            request_id: Optional request ID (generated if None)

        Returns:
            Request ID for tracking
        """
        blackboard = await self.get_blackboard()
        request_id = request_id or f"req_{uuid.uuid4().hex[:8]}"

        # Determine sender ID based on mode
        sender_id = self._agent.agent_id if self._agent else "detached"

        # session_id auto-added by blackboard.write from context
        key = f"{sender_id}:request:{request_type}:{request_id}"  # TODO: standardize key format
        await blackboard.write(
            key=key,
            value={
                "request_type": request_type,
                "request_id": request_id,
                "sender": sender_id,
                "target": self.scope_id,
                **request_data,
            },
            created_by=sender_id,
        )
        return request_id

    def register_hook(
        self,
        pointcut: Pointcut,
        handler: Any,
        hook_type: HookType = HookType.AFTER,
        priority: int = 0,
        on_error: ErrorMode = ErrorMode.FAIL_FAST,
    ) -> str:
        """Register a hook on the owning agent.

        Convenience method for capabilities to register hooks. The hook
        is automatically removed when this capability is removed from
        the agent (via `agent.remove_capability()`).

        Note: This method is not available in detached mode since there
        is no agent context to register hooks on.

        Args:
            pointcut: Determines which methods/instances match
            handler: The hook function to execute
            hook_type: BEFORE, AFTER, or AROUND
            priority: Higher values run first
            on_error: How to handle errors during execution

        Returns:
            Hook ID for later removal

        Raises:
            RuntimeError: If called in detached mode

        Example:
            ```python
            async def initialize(self):
                self.register_hook(
                    pointcut=Pointcut.pattern("*.infer"),
                    handler=self._track_tokens,
                    hook_type=HookType.AFTER,
                )
            ```
        """
        if self._agent is None:
            raise RuntimeError(
                "Cannot register hooks in detached mode. "
                "Hooks require an agent context."
            )

        return self._agent.hooks.register(
            pointcut=pointcut,
            handler=handler,
            hook_type=hook_type,
            priority=priority,
            on_error=on_error,
            owner=self,
        )

    async def initialize(self) -> None:
        """Initialize the capability.

        Base implementation auto-registers any methods decorated with
        `@register_hook`. Subclasses should call `await super().initialize()`
        to enable declarative hook registration.

        Example:
            ```python
            class MyCapability(AgentCapability):
                @register_hook(
                    pointcut=Pointcut.pattern("*.process"),
                    hook_type=HookType.AFTER,
                )
                async def log_result(self, ctx: HookContext, result: Any) -> Any:
                    logger.info(f"Result: {result}")
                    return result

                async def initialize(self):
                    await super().initialize()  # Auto-registers log_result hook
                    # ... additional initialization
            ```
        """
        self._auto_register_hooks()

    def _auto_register_hooks(self) -> list[str]:
        """Auto-register all @register_hook decorated methods.

        Called by `initialize()`. Can also be called manually if needed.

        In detached mode, returns empty list (no hooks can be registered).

        Returns:
            List of registered hook IDs
        """
        if self._agent is None:
            # No hooks in detached mode
            return []
        return auto_register_hooks(self, owner=self)


    @abstractmethod
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize capability-specific state.

        Args:
            state: AgentSuspensionState to populate with serialized state
        """
        ...

    @abstractmethod
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore capability-specific state from suspension.

        Args:
            state: AgentSuspensionState to restore from
        """
        ...


class AgentHandle:
    """Handle for interacting with agents from anywhere in the cluster.

    Provides access to the agent's capabilities for communication.
    The key insight is that the same `AgentCapability` class works
    in both local and remote modes via the `scope_id` parameter.

    Two modes:

    1. **Owned mode** (existing): Parent agent creates handle to communicate with child
       ```python
       handle = await owner.spawn_child_agents(...)[0]
       grounding = handle.get_capability(GroundingCapability)
       ```

    2. **Detached mode** (NEW): Any code creates handle to interact with agent by ID
       ```python
       handle = await AgentHandle.from_agent_id("agent_123")
       result = await handle.run({"query": "analyze code"})
       ```

    Detached mode enables:
    - External systems to interact with agents
    - System administrator agents with fixed IDs
    - Session-based agent interaction without parent relationship

    Communication patterns:

    1. **Wait for completion** - Get capability's result future and await it:
       ```python
       handle = await owner.spawn_child_agents(...)[0]
       grounding = handle.get_capability(GroundingCapability)
       future = await grounding.get_result_future()
       result = await asyncio.wait_for(future, timeout=30.0)
       ```

    2. **Stream events** - Receive async events from child's capabilities:
       ```python
       handle = await owner.spawn_child_agents(...)[0]
       grounding = handle.get_capability(GroundingCapability)
       await grounding.stream_events_to_queue(self.get_event_queue())
       # Events from child now flow to parent's action policy
       ```

    3. **Send requests** - Use capability methods to control child:
       ```python
       handle = await owner.spawn_child_agents(...)[0]
       grounding = handle.get_capability(GroundingCapability)
       await grounding.send_request("ground_claim", {"claim": "..."})
       ```

    4. **Run task** (NEW) - Execute a task and wait for result:
       ```python
       handle = await AgentHandle.from_agent_id("agent_123")
       result = await handle.run({"query": "analyze code"}, timeout=60)
       ```

    5. **Stream task** (NEW) - Execute with streaming intermediate results:
       ```python
       handle = await AgentHandle.from_agent_id("agent_123")
       async for event in handle.run_streamed({"query": "analyze code"}):
           print(event)
       ```
    """

    def __init__(
        self,
        child_agent_id: str,
        owner: Agent | None = None,
        capability_types: list[type[AgentCapability]] | None = None,
        *,
        default_capability_type: type[AgentCapability] | None = None,
    ):
        """Initialize agent handle.

        Args:
            child_agent_id: ID of the target agent
            owner: Parent agent (None for detached mode)
            capability_types: Expected capability types (for validation)
            default_capability_type: Default capability for run/stream methods
        """
        self.child_agent_id = child_agent_id
        self._owner = owner
        self._capability_types = capability_types or []
        self._capabilities: dict[str, AgentCapability] = {}
        self._default_capability_type = default_capability_type

        # For detached mode
        self._agent_metadata: dict[str, Any] | None = None
        self._blackboard: EnhancedBlackboard | None = None

    @property
    def agent_id(self) -> str:
        """Get the target agent's ID."""
        return self.child_agent_id

    @property
    def is_detached(self) -> bool:
        """Check if handle is operating in detached mode."""
        return self._owner is None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    async def from_agent_id(
        cls,
        agent_id: str,
        default_capability_type: type[AgentCapability] | None = None,
    ) -> "AgentHandle":
        """Create a detached AgentHandle from an agent ID.

        This is the primary way to interact with agents from outside
        the agent system (e.g., from API handlers, session managers).

        Args:
            agent_id: Target agent ID
            default_capability_type: Default capability for run/stream

        Returns:
            AgentHandle in detached mode

        Raises:
            ValueError: If agent not found

        Example:
            ```python
            # Get handle to system administrator agent
            handle = await AgentHandle.from_agent_id("system_admin_agent")
            result = await handle.run({"task": "check_health"})
            ```
        """
        handle = cls(
            child_agent_id=agent_id,
            owner=None,
            default_capability_type=default_capability_type,
        )

        # Load agent metadata from AgentSystem
        await handle._load_agent_metadata()

        return handle

    @classmethod
    async def from_agent_type(
        cls,
        agent_spec: AgentSpawnSpec,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        default_capability_type: type[AgentCapability] | None = None,
    ) -> "AgentHandle":
        """Spawn an agent by type and return handle.

        Convenience method that spawns a new agent and returns a handle.

        Args:
            agent_spec: Specification of the agent to spawn
            session_id: Session ID for context
            run_id: Run ID for context
            soft_affinity: Whether to use soft affinity routing
            suspend_agents: Whether to suspend other agents if needed
            default_capability_type: Default capability for run/stream

        Returns:
            AgentHandle for the spawned agent

        Raises:
            ValueError: If failed to spawn agent
        """
        child_ids: list[str] = spawn_agents(
            agent_specs=[agent_spec],
            session_id=session_id or agent_spec.metadata.session_id,
            run_id=run_id or agent_spec.metadata.run_id,
            soft_affinity=soft_affinity,
            suspend_agents=suspend_agents,
        )
        agent_id = child_ids[0]

        return await cls.from_agent_id(
            agent_id=agent_id,
            default_capability_type=default_capability_type,
        )

    async def _load_agent_metadata(self) -> None:
        """Load agent metadata from AgentSystemDeployment."""
        agent_system = get_agent_system()
        agent_info = await agent_system.get_agent_info(self.child_agent_id)

        if agent_info is None:
            raise ValueError(f"Agent {self.child_agent_id} not found")

        self._agent_metadata = {
            "agent_type": agent_info.agent_type,
            "capabilities": agent_info.metadata.get("capabilities", []),
            "default_capability": agent_info.metadata.get("default_capability"),
        }

    async def _get_blackboard(
        self,
        backend_type: str = "redis",
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for detached mode communication."""
        if self._blackboard is not None:
            return self._blackboard

        if self._owner is not None:
            return await self._owner.get_blackboard(
                scope="shared",
                scope_id=self.child_agent_id,
            )

        # Detached mode: create blackboard
        app_name = serving.get_my_app_name()

        self._blackboard = EnhancedBlackboard(
            app_name=app_name,
            scope=BlackboardScope.SHARED,
            scope_id=self.child_agent_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )
        return self._blackboard

    # =========================================================================
    # Capability Access
    # =========================================================================

    def get_capability(
        self,
        capability_type: type[AgentCapability]
    ) -> AgentCapability:
        """Get capability instance for communicating with agent.

        Creates a capability instance with `scope_id=child_agent_id`,
        enabling communication via the same capability interface.

        In detached mode, creates capability in detached mode using
        blackboard directly.

        Args:
            capability_type: Type of capability to instantiate

        Returns:
            Capability instance scoped to target agent

        Example:
            ```python
            grounding = handle.get_capability(GroundingCapability)
            await grounding.stream_events_to_queue(self.get_event_queue())
            future = await grounding.get_result_future()
            result = await asyncio.wait_for(future, timeout=30.0)
            ```
        """
        cap_name = capability_type.get_capability_name()
        if cap_name not in self._capabilities:
            if self.is_detached:
                # Detached mode: create capability with scope_id only
                self._capabilities[cap_name] = capability_type(
                    agent=None,
                    scope_id=self.child_agent_id,
                )
            else:
                # Owned mode: existing behavior
                self._capabilities[cap_name] = capability_type(
                    agent=self._owner,
                    scope_id=self.child_agent_id,
                )
        return self._capabilities[cap_name]

    def get_default_capability(self) -> AgentCapability | None:
        """Get the agent's default interaction capability.

        Returns:
            Default capability or None if not defined
        """
        if self._default_capability_type:
            return self.get_capability(self._default_capability_type)
        return None

    # =========================================================================
    # Runner Functionality
    # =========================================================================

    async def run(
        self,
        input_data: dict[str, Any],
        timeout: float = 30.0,
        session_id: str | None = None,
        config: "AgentRunConfig | None" = None,
        track_events: bool = False,
    ) -> "AgentRun":
        """Run a task on the agent and wait for result.

        Sends input to agent via blackboard and waits for result.
        Works in both owned and detached modes. Creates an AgentRun
        that is tracked in the current session.

        Args:
            input_data: Task input data
            timeout: Timeout in seconds
            session_id: Optional session ID for context
            config: Run configuration (uses session defaults if None)
            track_events: Whether to record intermediate events

        Returns:
            AgentRun with status, output_data, and resource_usage

        Example:
            ```python
            async with session.context():
                handle = await AgentHandle.from_agent_id("agent_123")
                run = await handle.run({"query": "analyze code"}, timeout=60)
                print(run.output_data)
            ```
        """
        blackboard = await self._get_blackboard()
        await blackboard.initialize()

        # Get session context if available
        from .sessions.context import get_current_session_id
        from ..system import get_session_manager

        # Get session context
        effective_session_id = session_id or get_current_session_id()

        # Create AgentRun via SessionManager if we have a session
        run: AgentRun | None = None
        session_manager_handle = None

        if session_id:
            try:
                session_manager_handle = get_session_manager()

                run = await session_manager_handle.create_run(
                    session_id=effective_session_id,
                    agent_id=self.child_agent_id,
                    input_data=input_data,
                    config=config,
                    timeout=timeout,
                    track_events=track_events,
                )

                # Mark as running
                await session_manager_handle.update_run_status(
                    run_id=run.run_id,
                    status=RunStatus.RUNNING,
                )
            except Exception as e:
                logger.warning(f"Failed to create tracked run: {e}")
                # Continue without tracking

        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Determine sender ID
        sender_id = self._owner.agent_id if self._owner else "detached_handle"

        # Result tracking
        result_key = f"{self.child_agent_id}:result:{request_id}"
        result_value: dict[str, Any] | None = None

        # Send request (session_id auto-added by blackboard.write from context)
        request_key = f"{self.child_agent_id}:request:{request_id}"
        await blackboard.write(
            key=request_key,
            value={
                "input": input_data,
                "request_id": request_id,
                "session_id": effective_session_id,  # Include in value for receiver to extract
                "sender": sender_id,
                "run_id": run.run_id if run else None,
            },
            created_by=sender_id,
        )

        # Check if result already exists (race condition handling)
        existing = await blackboard.read(result_key)
        if existing is not None:
            if run and session_manager_handle:
                await session_manager_handle.update_run_status(
                    run_id=run.run_id,
                    status=RunStatus.COMPLETED,
                    output_data=existing,
                )
                run.status = RunStatus.COMPLETED
                run.output_data = existing
            else:
                # Create ephemeral run for return value
                run = AgentRun(
                    session_id=session_id or "",
                    tenant_id="",
                    agent_id=self.child_agent_id,
                    status=RunStatus.COMPLETED,
                    input_data=input_data,
                    output_data=existing,
                )
            return run

        # Wait for result using stream_events
        final_status = RunStatus.COMPLETED
        error_msg: str | None = None

        try:
            async for event in blackboard.stream_events(
                pattern=result_key,
                event_types={"write"},
                timeout=timeout,
            ):
                if event.key == result_key and event.value is not None:
                    result_value = event.value
                    break
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for result from {self.child_agent_id}")
            final_status = RunStatus.TIMEOUT
            error_msg = f"Timeout after {timeout}s"

        # Update run status
        if run and session_manager_handle:
            await session_manager_handle.update_run_status(
                run_id=run.run_id,
                status=final_status,
                output_data=result_value,
                error=error_msg,
            )
            run.status = final_status
            run.output_data = result_value
            run.error = error_msg
        else:
            # Create ephemeral run for return value
            run = AgentRun(
                session_id=session_id or "",
                tenant_id="",
                agent_id=self.child_agent_id,
                status=final_status,
                input_data=input_data,
                output_data=result_value,
                error=error_msg,
            )

        return run

    async def run_streamed(
        self,
        input_data: dict[str, Any],
        config: AgentRunConfig | None = None,
        timeout: float = 300.0,
        session_id: str | None = None,
        event_types: set[str] | None = None,
        track_events: bool = True,
    ) -> AsyncIterator[AgentRunEvent]:
        """Run a task with streaming events.

        Creates an AgentRun and yields events as they occur.
        Events are also persisted to the run if track_events=True.

        Args:
            input_data: Task input data
            config: Run configuration (uses session defaults if None)
            timeout: Maximum execution time
            session_id: Optional session ID
            event_types: Filter for specific event types (if None, all write events)
            track_events: If True, events are persisted to the AgentRun

        Yields:
            AgentRunEvent for each event during execution

        Example:
            ```python
            async with session.context():
                handle = await AgentHandle.from_agent_id("agent_123")
                async for event in handle.run_streamed({"query": "analyze"}):
                    print(f"{event.event_type}: {event.data}")
                    if event.event_type == "completed":
                        break
            ```
        """
        from .sessions.context import get_current_session_id
        from ..system import get_session_manager

        blackboard = await self._get_blackboard()
        await blackboard.initialize()

        # Get session context if available
        from .sessions.context import get_current_session_id
        effective_session_id = session_id or get_current_session_id()

        # Create AgentRun via SessionManager if we have a session
        run: AgentRun | None = None
        session_manager_handle = None

        if session_id:
            try:
                session_manager_handle = get_session_manager()

                run = await session_manager_handle.create_run(
                    session_id=session_id,
                    agent_id=self.child_agent_id,
                    input_data=input_data,
                    config=config,
                    timeout=timeout,
                    track_events=track_events,
                )

                # Mark as running
                await session_manager_handle.update_run_status(
                    run_id=run.run_id,
                    status=RunStatus.RUNNING,
                )
            except Exception as e:
                logger.warning(f"Failed to create tracked run for streaming: {e}")
                # Continue without tracking

        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Determine sender ID
        sender_id = self._owner.agent_id if self._owner else "detached_handle"

        # Send request with streaming flag (session_id auto-added by blackboard.write)
        request_key = f"{self.child_agent_id}:request:{request_id}"
        await blackboard.write(
            key=request_key,
            value={
                "input": input_data,
                "request_id": request_id,
                "session_id": effective_session_id,  # Include in value for receiver to extract
                "sender": sender_id,
                "streaming": True,
                "run_id": run.run_id if run else None,
            },
            created_by=sender_id,
        )

        # Stream events
        start_time = time.time()
        event_pattern = f"{self.child_agent_id}:event:{request_id}:*"
        is_complete = False
        final_status = RunStatus.COMPLETED
        error_msg: str | None = None
        result_data: dict[str, Any] | None = None
        event_counter = 0

        try:
            async for bb_event in blackboard.stream_events(
                pattern=event_pattern,
                event_types=event_types or {"write"},
                timeout=min(10.0, timeout),
                until=lambda: is_complete,
            ):
                # Check overall timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    final_status = RunStatus.TIMEOUT
                    error_msg = f"Timeout after {timeout}s"

                    timeout_event = AgentRunEvent(
                        event_id=f"evt_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        event_type="timeout",
                        data={"error": error_msg},
                    )

                    if run and session_manager_handle and track_events:
                        try:
                            await session_manager_handle.add_run_event(
                                run_id=run.run_id,
                                event_type="timeout",
                                data={"error": error_msg},
                            )
                        except Exception:
                            pass

                    yield timeout_event
                    break

                # Determine event type from key
                is_final = bb_event.key.endswith(":complete") or bb_event.key.endswith(":result")
                event_type_str = "completed" if is_final else "progress"

                if bb_event.key.endswith(":error"):
                    event_type_str = "error"
                    final_status = RunStatus.FAILED
                    error_msg = bb_event.value.get("error") if isinstance(bb_event.value, dict) else str(bb_event.value)

                if is_final and isinstance(bb_event.value, dict):
                    result_data = bb_event.value

                # Create AgentRunEvent
                event_counter += 1
                agent_event = AgentRunEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    event_type=event_type_str,
                    data={
                        "key": bb_event.key,
                        "value": bb_event.value,
                        "is_final": is_final,
                    },
                )

                # Persist event if tracking
                if run and session_manager_handle and track_events:
                    try:
                        await session_manager_handle.add_run_event(
                            run_id=run.run_id,
                            event_type=event_type_str,
                            data=agent_event.data,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to persist event: {e}")

                yield agent_event

                # Check for completion
                if is_final:
                    is_complete = True
                    break

        except asyncio.TimeoutError:
            final_status = RunStatus.TIMEOUT
            error_msg = f"Timeout after {timeout}s"

            timeout_event = AgentRunEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                event_type="timeout",
                data={"error": error_msg},
            )
            yield timeout_event

        except Exception as e:
            final_status = RunStatus.FAILED
            error_msg = str(e)

            error_event = AgentRunEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                event_type="error",
                data={"error": error_msg},
            )
            yield error_event

        # Update final run status
        if run and session_manager_handle:
            try:
                await session_manager_handle.update_run_status(
                    run_id=run.run_id,
                    status=final_status,
                    output_data=result_data,
                    error=error_msg,
                )
            except Exception as e:
                logger.warning(f"Failed to update run status: {e}")

    # =========================================================================
    # Agent Control
    # =========================================================================

    async def stop(self) -> None:
        """Request the target agent to stop."""
        agent_system = get_agent_system()
        await agent_system.stop_agent(self.child_agent_id)


class CapabilityResultFuture:
    """Future for waiting on capability results.

    Each capability instance represents ONE task (which can be hierarchical
    and long-running). This future resolves when that task completes.

    The future is directly awaitable and handles the race condition where
    the result may already be written before `wait()` is called.

    Example:
        ```python
        # Get capability (local or remote)
        grounding = handle.get_capability(GroundingCapability)

        # Get result future
        future = await grounding.get_result_future()

        # Option 1: Direct await (preferred)
        result = await future

        # Option 2: Await with timeout
        result = await asyncio.wait_for(future, timeout=30.0)

        # Option 3: Callback
        future.on_complete(lambda r: print(f"Got result: {r}"))
        ```
    """

    def __init__(
        self,
        result_key: str,
        blackboard: EnhancedBlackboard,
    ):
        """Initialize result future.

        Args:
            result_key: Blackboard key for result. It is used to both read the result and monitor for events.
            blackboard: Blackboard for monitoring
        """
        self._result_key = result_key
        self._blackboard = blackboard
        self._result: dict[str, Any] | None = None
        self._is_complete = False
        self._callbacks: list[callable] = []
        self._future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._monitor_task: asyncio.Task | None = None
        self._checked_existing = False

    @property
    def is_complete(self) -> bool:
        """Check if result is available."""
        return self._is_complete

    @property
    def result(self) -> dict[str, Any] | None:
        """Get result if complete."""
        return self._result

    def on_complete(self, callback: callable) -> CapabilityResultFuture:
        """Register completion callback (chainable).

        Args:
            callback: Function called with result when available

        Returns:
            Self for chaining
        """
        if self._is_complete:
            callback(self._result)
        else:
            self._callbacks.append(callback)
        return self

    def __await__(self):
        """Make future directly awaitable.

        Example:
            result = await future  # Instead of await future.wait()
        """
        return self.wait().__await__()

    async def wait(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Wait for result.

        First checks if result already exists (handles race condition where
        result is written before wait is called), then monitors for events.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Result dict or None on timeout
        """
        if self._is_complete:
            return self._result

        # Check if result already exists (handles race condition)
        if not self._checked_existing:
            self._checked_existing = True
            # TODO: Should we separate the result key used to read the result from
            # the result key pattern used to monitor for result write events?
            existing = await self._blackboard.read(self._result_key)
            if existing is not None:
                self._complete(existing)
                return self._result

        # Start monitoring if not already
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(
                self._monitor_result(timeout=timeout)
            )

        try:
            return await asyncio.wait_for(self._future, timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def _complete(self, result: dict[str, Any]) -> None:
        """Mark future as complete with result."""
        if self._is_complete:
            return

        self._result = result
        self._is_complete = True

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(self._result)
            except Exception as e:
                logging.error(f"Callback error: {e}")

        if not self._future.done():
            self._future.set_result(self._result)

    async def _monitor_result(self, timeout: float | None = None) -> None:
        """Monitor blackboard for result."""
        # TODO: Should we separate the result key used to read the result from
        # the result key pattern used to monitor for result write events?
        try:
            async for event in self._blackboard.stream_events(
                pattern=self._result_key,
                event_types={"write"},
                until=lambda: self._is_complete,
                timeout=timeout,
            ):
                if event.value:
                    self._complete(event.value)

        except Exception as e:
            logging.error(f"Error monitoring result: {e}")
            if not self._future.done():
                self._future.set_exception(e)


class Agent(BaseModel):
    """Base agent class representing an autonomous computational entity.

    All agents have:
    - Unique ID and type
    - Lifecycle state
    - Optional page binding
    - Access to system services (VCM, LLM, blackboard)

    Control flow should be driven by a reasoning LLM given sufficient context,
    not hardcoded. Agents adapt behavior based on current context.

    Attributes:
        agent_id: Unique identifier
        agent_type: Type of agent ("specialized", "general", "service", "supervisor")
        state: Current lifecycle state
        bound_pages: Page IDs this agent is bound to (empty for unbound agents)
        created_at: Creation timestamp
        metadata: Arbitrary metadata
    """

    agent_id: str
    agent_type: str = "general"  # "specialized", "general", "service", "supervisor"
    state: AgentState = Field(default=AgentState.INITIALIZED)

    tenant_id: str = "default"  # Tenant/namespace for multi-tenant deployments

    # Optional page binding
    bound_pages: list[str] = Field(default_factory=list)

    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

    page_storage: PageStorage | None = Field(default=None)

    # Resource requirements (for scheduling and capacity planning)
    resource_requirements: AgentResourceRequirements = Field(
        default_factory=AgentResourceRequirements,
        description="CPU/memory/GPU requirements for this agent"
    )

    # Graceful shutdown configuration
    stop_timeout_s: float = Field(
        default=30.0,
        ge=0.0,
        description="Timeout for graceful shutdown (seconds)"
    )

    # Metadata and capabilities
    created_at: float = Field(default_factory=time.time)

    capability_classes: list[type[AgentCapability]] = Field(default_factory=list)
    _capabilities: dict[str, AgentCapability] = Field(default_factory=dict)

    # Memory configuration
    enable_memory_hierarchy: bool = Field(
        default=False,
        description="Whether to auto-initialize the default memory hierarchy"
    )
    memory_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for memory hierarchy (passed to create_default_memory_hierarchy)"
    )

    action_policy_class: type[ActionPolicy] | None = None
    action_policy_state: ActionPolicyExecutionState | None = Field(default=None)
    action_policy: ActionPolicy | None = None

    child_agents: dict[str, str] = Field(default_factory=dict)  # role -> agent_id

    # Runtime state (not serialized)
    _running: bool = False
    _stop_requested: bool = False
    _suspend_requested: bool = False
    _suspend_reason: str | None = None
    _manager: AgentManagerBase | None = None
    _hook_registry: AgentHookRegistry | None = None

    class Config:
        arbitrary_types_allowed = True
        # Don't serialize runtime state
        fields = {
            "_running": {"exclude": True},
            "_stop_requested": {"exclude": True},
            "_suspend_requested": {"exclude": True},
            "_suspend_reason": {"exclude": True},
            "_manager": {"exclude": True},
            "_hook_registry": {"exclude": True},
        }

    def set_manager(self, manager: AgentManagerBase) -> None:
        """Attach agent manager reference for delegation.

        Args:
            manager: AgentManagerBase instance
        """
        self._manager = manager

    @property
    def hooks(self) -> AgentHookRegistry:
        """Get the hook registry for this agent.

        The registry is lazily created on first access.

        Returns:
            AgentHookRegistry for registering hooks
        """
        if self._hook_registry is None:
            self._hook_registry = AgentHookRegistry(self)
        return self._hook_registry

    def add_capability(
        self,
        capability: AgentCapability,
        *,
        include_actions: list[str] | None = None,
        exclude_actions: list[str] | None = None,
        events_only: bool = False,
    ) -> None:
        """Add a capability to the agent.

        Args:
            capability: AgentCapability instance
            include_actions: If provided, only these action keys are exposed to the
                ActionPolicy for planning. Other actions can still be invoked directly.
                If None, all actions are included (unless exclude_actions is set).
            exclude_actions: Action keys to exclude from the ActionPolicy.
                If None, no actions are excluded.
            events_only: If True, no actions are exposed (equivalent to include_actions=[]).
                The capability's event handlers are still registered. Use this when
                adding a capability solely to subscribe to its events.

        The exposed actions are computed as:
            exposed = (include_actions or ALL) - (exclude_actions or {})

        Event handlers (methods with @event_handler) are always registered regardless
        of action filtering. Direct method calls also always work.

        Example:
            # Only expose detect_contradictions, not resolve_contradiction
            self.add_capability(validation_cap, include_actions=["detect_contradictions"])

            # Expose all actions except resolve_contradiction
            self.add_capability(validation_cap, exclude_actions=["resolve_contradiction"])

            # Only receive events, no actions exposed
            self.add_capability(validation_cap, events_only=True)
        """
        # Store action filter parameters for ActionDispatcher to use
        # _action_include_filter: set of action keys to include, or None for all
        # _action_exclude_filter: set of action keys to exclude
        if events_only:
            capability._action_include_filter = frozenset()  # Empty set = no actions
            capability._action_exclude_filter = frozenset()
        else:
            capability._action_include_filter = frozenset(include_actions) if include_actions is not None else None
            capability._action_exclude_filter = frozenset(exclude_actions) if exclude_actions else frozenset()

        self._capabilities[capability.get_capability_name()] = capability

    def remove_capability(self, capability_name: str) -> AgentCapability | None:
        """Remove a capability from the agent.

        Also removes any hooks registered by the capability.

        Args:
            capability_name: Name of the capability to remove

        Returns:
            The removed capability, or None if not found
        """
        capability = self._capabilities.pop(capability_name, None)
        if capability is not None and self._hook_registry is not None:
            self._hook_registry.remove_hooks_by_owner(capability)
        return capability

    def get_capability(self, capability_name: str) -> AgentCapability | None:
        """Get a specific capability by name.

        Args:
            capability_name: Name of the capability
        Returns:
            AgentCapability instance or None if not found
        """
        return self._capabilities.get(capability_name)

    def get_capability_by_type(self, capability_type: type) -> AgentCapability | None:
        """Get a capability by its type.

        This provides type-safe capability access compared to `get_capability(name)`.

        Args:
            capability_type: The capability class type to look for

        Returns:
            The capability instance if found, None otherwise

        Example:
            ```python
            from polymathera.colony.agents.patterns.memory import AgentContextEngine

            ctx_engine = agent.get_capability_by_type(AgentContextEngine)
            if ctx_engine:
                memories = await ctx_engine.gather_context()
            ```
        """
        for cap in self._capabilities.values():
            if isinstance(cap, capability_type):
                return cap
        return None

    def get_capability_names(self) -> list[str]:
        """Get names of capabilities this agent has.

        Returns:
            List of capability names
        """
        return list(self._capabilities.keys())

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability.

        Args:
            capability_name: Name of the capability
        Returns:
            True if the agent has the capability, False otherwise
        """
        return capability_name in self._capabilities

    def get_capabilities(self) -> list[str]:
        """Get agent's capabilities for discovery.

        Returns:
            List of capability strings
        """
        return list(self._capabilities.keys())

    # === Tool Usage (Delegated to manager) ===

    async def discover_tools(self, category: str | None = None) -> list[str]:
        """Discover available tools.

        Args:
            category: Optional category filter

        Returns:
            List of tool IDs
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        try:
            # Delegate to manager's tool manager handle
            tool_manager = get_tool_manager()
            if category:
                tool_ids = await tool_manager.find_tools_by_category(category)
            else:
                tool_ids = await tool_manager.list_all_tools()

            return tool_ids

        except Exception as e:
            raise

    async def use_tool(
        self, tool_id: str, parameters: dict[str, Any], auth_token: str | None = None
    ) -> Any:
        """Execute a tool.

        Args:
            tool_id: Tool identifier
            parameters: Tool parameters
            auth_token: Optional authentication token

        Returns:
            Tool result

        Raises:
            RuntimeError: If tool execution fails
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        try:
            # Create tool call
            from .models import ToolCall

            tool_call = ToolCall(
                call_id=f"{self.agent_id}-{tool_id}-{time.time()}",
                agent_id=self.agent_id,
                tool_id=tool_id,
                parameters=parameters,
                auth_token=auth_token,
            )

            # Delegate to manager's tool manager handle
            tool_manager = get_tool_manager()
            completed_call = await tool_manager.execute_tool(tool_call)

            # Check if successful
            if completed_call.status == ActionStatus.COMPLETED:
                return completed_call.result
            else:
                error_msg = completed_call.error or "Unknown error"
                raise RuntimeError(f"Tool {tool_id} failed: {error_msg}")

        except Exception as e:
            raise

    # === Lifecycle Methods (Override in subclasses) ===

    async def initialize(self) -> None:
        """Initialize agent (called after creation).

        Override in subclasses to perform initialization logic.
        """
        # Check if this is a resumed agent
        if self.metadata.resuming_from_suspension:
            await self._restore_from_suspension()
        else:
            self.state = AgentState.INITIALIZED

            # Track children - for event-driven coordination
            # TODO: How to restore this on resumption from suspension?
            self.child_agents: dict[str, str] = {}  # role -> agent_id

        # Reconstruct or create PageStorage
        vcm_handle = get_vcm()
        config: PageStorageConfig | None = vcm_handle.get_page_storage_config()
        if not config:
            raise ValueError("Missing PageStorageConfig in VCM")

        self.page_storage = PageStorage(
            group_id = self.metadata.group_id,
            tenant_id = self.metadata.tenant_id,
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            s3_bucket=config.s3_bucket,
        )
        await self.page_storage.initialize()

        # Initialize memory hierarchy if enabled
        if self.enable_memory_hierarchy:
            await self.initialize_memory_hierarchy(**self.memory_config)

        await self._create_action_policy()

    async def _create_action_policy(self) -> None:
        if self.action_policy:
            return  # Already set
        elif not self.action_policy_class:
            raise ValueError("Agent must have action_policy or action_policy_class defined")

        for cap_class in self.capability_classes:
            if not self.has_capability(cap_class.get_capability_name()):
                capability = cap_class(self)
                await capability.initialize()
                # TODO: How to allow passing more add_capability params?
                self.add_capability(
                    capability,
                    include_actions=[],
                    exclude_actions=[],
                    events_only=False,
                )

        self.action_policy = self.action_policy_class(
            agent=self,
            action_providers=list(self._capabilities.values()),
            # TODO: Allow configuring IO schemas
            # io=ActionPolicyIO(
            #     inputs={"context": QueryContext, "queries": list},
            #     outputs={"analysis": ScopeAwareResult, "next_queries": list},
            # ),
            **self.metadata.action_policy_config
        )
        self.action_policy.use_agent_capabilities([cap.get_capability_name() for cap in self.capability_classes])
        await self.action_policy.initialize()

    async def initialize_memory_hierarchy(
        self,
        include_sensory: bool = False,
        stm_ttl: float = 3600,
        stm_max_entries: int = 100,
        ltm_ttl: float | None = None,
        working_max_tokens: int = 8000,
    ) -> dict[str, AgentCapability]:
        """Initialize the default memory hierarchy for this agent.

        Creates and initializes all memory capabilities:
        - Working memory (with context compaction)
        - Short-term memory (with decay)
        - Long-term memory (episodic, semantic, procedural)
        - Memory transfers between levels
        - AgentContextEngine for unified access
        - MemoryLifecycleHooks for task completion and shutdown

        This can be called explicitly or automatically if `enable_memory_hierarchy=True`.

        Args:
            include_sensory: Whether to include sensory memory level
            stm_ttl: TTL for STM entries (seconds)
            stm_max_entries: Max entries in STM
            ltm_ttl: TTL for LTM entries (None = no expiration)
            working_max_tokens: Token budget for working memory

        Returns:
            Dict mapping capability names to capability instances

        Example:
            ```python
            # Explicit initialization
            agent = Agent(...)
            await agent.initialize()
            memories = await agent.initialize_memory_hierarchy(
                stm_ttl=7200,
                working_max_tokens=16000,
            )

            # Or via configuration
            agent = Agent(
                enable_memory_hierarchy=True,
                memory_config={"stm_ttl": 7200},
                ...
            )
            await agent.initialize()  # Memory hierarchy auto-created
            ```
        """
        from .patterns.memory.defaults import create_default_memory_hierarchy

        return await create_default_memory_hierarchy(
            agent=self,
            include_sensory=include_sensory,
            stm_ttl=stm_ttl,
            stm_max_entries=stm_max_entries,
            ltm_ttl=ltm_ttl,
            working_max_tokens=working_max_tokens,
            auto_add_to_agent=True,  # Always add to agent
        )

    async def get_page_storage(self) -> PageStorage | None:
        """Get page storage handle from context page source."""
        return self.page_storage

    async def load_page_graph(self) -> nx.DiGraph:
        """Load page graph dynamically from PageStorage.

        Uses page_storage to access PageStorage.
        This allows the agent to load the page graph when needed, rather than
        passing the entire graph in metadata.
        """
        if not self.page_storage:
            return nx.DiGraph()
        return await self.page_storage.load_page_graph()

    async def _restore_from_suspension(self) -> None:
        """Restore base agent state from suspension.

        Loads suspension state from StateManager and restores
        agent state (RUNNING, WAITING, etc.)

        Subclasses should override restore_subclass_state_from_suspension() to restore
        subclass-specific state (execution state, communication state, cache state).
        """
        suspended_agent_id = self.metadata.suspended_agent_id
        if not suspended_agent_id:
            logger.warning(
                f"Agent {self.agent_id} marked as resuming_from_suspension but no suspended_agent_id provided"
            )
            self.state = AgentState.INITIALIZED
            return

        try:
            # Load suspension state from StateManager
            app_name = serving.get_my_app_name()
            polymathera = get_polymathera()
            state_key = AgentSuspensionState.get_state_key(app_name, suspended_agent_id)

            state_manager: StateManager = await polymathera.get_state_manager(
                state_type=AgentSuspensionState,
                state_key=state_key,
            )

            suspension_state = None
            async for state in state_manager.read_transaction():
                suspension_state = state
                break

            if not suspension_state:
                logger.error(
                    f"No suspension state found for {suspended_agent_id}, "
                    f"initializing as new agent"
                )
                self.state = AgentState.INITIALIZED
                return

            # Delegate to agent to restore its own state
            # ✅ GOOD: Agent is responsible for its own state restoration
            await self.deserialize_suspension_state(suspension_state)

            logger.info(
                f"Restored agent {self.agent_id} from suspension state of {suspended_agent_id} "
                f"(suspension_count: {suspension_state.suspension_count}, "
                f"reason: {suspension_state.suspension_reason})"
            )

            # 5. Clean up suspension state (agent successfully resumed)
            try:
                await state_manager.cleanup()
                logger.debug(f"Cleaned up suspension state for {suspended_agent_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up suspension state for {suspended_agent_id}: {e}")

        except Exception as e:
            logger.error(
                f"Failed to restore agent {self.agent_id} from suspension: {e}",
                exc_info=True
            )
            # Initialize as new agent on restoration failure
            self.state = AgentState.INITIALIZED

    async def serialize_suspension_state(self) -> AgentSuspensionState:
        """Serialize agent state for suspension.

        Base implementation handles common state (agent_id, agent_type, 
        resource_requirements, bound_pages, metadata).

        Subclasses should override this method and call super() first:

        ```python
        async def serialize_suspension_state(self) -> AgentSuspensionState:
            state = await super().serialize_suspension_state()
            # Add subclass-specific state
            state.plan_id = self.current_plan_id
            return state
        ```

        Returns:
            AgentSuspensionState with all agent state serialized
        """
        # Base implementation - common state
        state = AgentSuspensionState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            suspended_at=time.time(),
            suspension_reason=self._suspend_reason or "",

            action_policy_state=self.action_policy_state,

            # Resource state (always available)
            allocated_cpu_cores=self.resource_requirements.cpu_cores,
            allocated_memory_mb=self.resource_requirements.memory_mb,
            allocated_gpu_cores=self.resource_requirements.gpu_cores,
            allocated_gpu_memory_mb=self.resource_requirements.gpu_memory_mb,

            # Bound pages (always available)
            bound_pages=self.bound_pages.copy(),

            # Metadata (always available)
            parent_agent_id=self.metadata.parent_agent_id,
            role=self.metadata.role,
            resumption_priority=self.metadata.resumption_priority,
            max_suspension_duration=self.metadata.max_suspension_duration,

            # Agent state (always available)
            agent_state=self.state.value if self.state else None,
        )

        # Communication state
        state.child_agents = self.child_agents.copy()

        # TODO: Store additional communication state in custom_data with special keys

        # Cache state
        state = await self.action_policy.serialize_suspension_state(state)

        # TODO: Call serialize_suspension_state() on capabilities that need it

        return state

    async def deserialize_suspension_state(
        self,
        state: AgentSuspensionState
    ) -> None:
        """Restore agent state from suspension.

        Base implementation restores common state.
        Subclasses should override this method and call super() first:

        ```python
        async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
            await super().deserialize_suspension_state(state)
            # Restore subclass-specific state
            self.current_plan_id = state.plan_id
        ```

        Restores:
        1. Child tracking (child_agents)
        2. Execution state (current_plan_id, current_action_index)
        3. Cache state (working set, page access patterns)
        4. Blackboard subscriptions (must re-subscribe callbacks)

        Args:
            state: AgentSuspensionState to restore from
        """
        # Restore base state
        if state.agent_state:
            self.state = AgentState(state.agent_state)
        else:
            self.state = AgentState.INITIALIZED

        self.action_policy_state = state.action_policy_state

        # Restore metadata
        if state.parent_agent_id:
            self.metadata.parent_agent_id = state.parent_agent_id
        if state.role:
            self.metadata.role = state.role
        if state.bound_pages:
            self.bound_pages = state.bound_pages.copy()

        self.metadata.last_resumed_at = time.time()
        self.metadata.suspension_count = state.suspension_count

        # Restore child tracking
        self.child_agents = state.child_agents.copy()

        # Restore action policy state
        self.action_policy.deserialize_suspension_state(state)

        # TODO: Call deserialize_suspension_state() on capabilities that need it

        logger.info(
            f"Agent {self.agent_id} successfully restored from suspension "
            f"(children: {len(self.child_agents)}, working set: {len(state.working_set_pages)} pages)"
        )

    async def restore_subclass_state_from_suspension(self, suspension_state: AgentSuspensionState) -> None:
        """DEPRECATED: Use deserialize_suspension_state() instead.

        This method is kept for backward compatibility but delegates to
        deserialize_suspension_state().

        Args:
            suspension_state: The suspension state loaded from StateManager
        """
        await self.deserialize_suspension_state(suspension_state)

    # === Execution Loop (Lifecycle Management) Methods ===

    @hookable
    async def start(self) -> None:
        """Start agent execution.

        This method is @hookable, so capabilities can register BEFORE hooks
        to initialize capabilities, AFTER hooks for cleanup, etc.

        Override in subclasses to perform startup logic.
        """
        self.state = AgentState.RUNNING
        self._running = True
        self._stop_requested = False

    @hookable
    async def stop(self) -> None:
        """Stop agent execution.

        This method is @hookable, so capabilities can register BEFORE hooks
        to flush memories, AFTER hooks for cleanup, etc.

        Override in subclasses to perform cleanup logic.
        """
        self._stop_requested = True
        self.state = AgentState.STOPPED
        self._running = False
        self.action_policy_state = None
        # TODO: Should notify parent agent if any?

    @hookable
    async def suspend(self, reason: str = "") -> None:
        """Request suspension by setting _suspend_requested flag.

        The manager loop will detect this flag and call AgentManagerBase.suspend_agent()
        which does the actual suspension (cancel task, free resources, persist state, delete agent).

        An agent can call this method on itself.

        This method ONLY sets the flag - it does NOT persist state or cancel execution.
        State persistence is done by AgentManagerBase.suspend_agent().

        This method is @hookable, so capabilities can register BEFORE hooks
        to save additional state (e.g., domain-specific checkpoints),
        AFTER hooks to run additional cleanup logic or integrity checks, etc.

        Args:
            reason: Reason for suspension (e.g., "Blocked: waiting for children",
                   "resource_exhaustion", "cache_pressure")

        Override in subclasses to save additional state (e.g., domain-specific
        checkpoints), but ALWAYS call super().suspend(reason) first.

        Example:
            ```python
            class MyAgent(Agent):
                async def suspend(self, reason: str = ""):
                    # Save agent state first
                    await super().suspend(reason)

                    # Then save domain-specific state
                    await self.save_custom_checkpoint()
            ```

        """
        # Set flag for manager to detect
        self._suspend_requested = True
        self._suspend_reason = reason

        logger.info(f"Agent {self.agent_id} requested suspension: {reason}")

    @hookable
    async def run_step(self) -> None:
        """Execute one step of agent logic.

        This is the core method that defines agent behavior. Override in
        subclasses to implement specific agent logic.

        This method should be idempotent and handle its own errors.

        This method is @hookable, so capabilities can register BEFORE hooks
        to prepare for the step,
        AFTER hooks to post-process the step,
        AROUND hooks to wrap the step execution entirely,
        and to handle errors during execution.

        NOTE: We use the repeated `run_step` approach instead of a single
        long-running `run` method to facilitate easier suspension and state
        management. This is necessary for distributed agents that may be
        suspended and resumed across different replicas.
        """
        if self.state != AgentState.RUNNING:
            logger.warning(
                f"Agent {self.agent_id} in state {self.state}, cannot run step"
            )
            return

        # Set up session_id context for the ENTIRE run_step
        # Session_id is propagated from requests via action_policy_state.custom["current_session_id"]
        # This ensures ALL operations (control messages, hooks, capabilities) have session_id
        from .sessions.context import session_id_context
        current_session_id = None
        if self.action_policy_state is not None:
            current_session_id = self.action_policy_state.custom.get("current_session_id")

        with session_id_context(current_session_id):
            # Process control messages from parent/children (mailbox)
            await self._process_control_messages()

            # Check child health (periodic health monitoring)
            await self._check_child_health()

            if self.action_policy_state is None:
                self.action_policy_state = ActionPolicyExecutionState()

            iteration_result = await self.action_policy.execute_iteration(self.action_policy_state)
            self.state = AgentState.STOPPED if iteration_result.policy_completed else self.state
            if self.state == AgentState.SUSPENDED:
                logger.info(f"Agent {self.agent_id} has entered SUSPENDED state")
                # Suspend agent until dependencies resolved
                await self.suspend(reason=f"Blocked: {iteration_result.blocked_reason}")

            if self.state == AgentState.STOPPED:
                logger.info(f"Agent {self.agent_id} has entered STOPPED state")
                # Stop agent gracefully
                await self.stop()

            if iteration_result.error_context:
                logger.error(
                    f"Agent {self.agent_id} encountered error in run_step: "
                    f"{iteration_result.error_context.error_details}"
                )
                # Optionally, could set state to FAILED here
                # self.state = AgentState.FAILED

            await asyncio.sleep(0.1)

    # === Context Access (Delegated to manager) ===

    @hookable
    async def request_page(self, page_id: str, priority: int = 0, tenant_id: str | None = None) -> None:
        """Request a virtual page to be loaded.

        Delegates to the agent manager that hosts this agent.

        This method is @hookable, so capabilities can register BEFORE hooks
        to prepare for the page request,
        AFTER hooks to post-process the page request,
        AROUND hooks to wrap the page request execution entirely,
        and to handle errors during execution.

        Args:
            page_id: Page identifier
            priority: Load priority
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        await self._manager.agent_request_page(
            agent_id=self.agent_id,
            page_id=page_id,
            priority=priority,
            tenant_id=tenant_id or self.tenant_id
        )

    async def is_page_loaded(self, page_id: str) -> bool:
        """Check if a page is loaded.

        Delegates to the agent manager that hosts this agent.

        Args:
            page_id: Page identifier

        Returns:
            True if loaded
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        return await self._manager.agent_is_page_loaded(
            agent_id=self.agent_id,
            page_id=page_id
        )

    # === Inference API (Delegated to manager) ===

    @hookable
    async def infer(
        self,
        prompt: str | None = None,
        context_page_ids: list[str] | None = None,
        **kwargs,
    ) -> InferenceResponse:
        """Submit inference request.

        Delegates to LLMCluster via the agent manager.

        Args:
            prompt: Prompt text
            context_page_ids: Optional list of virtual page IDs for context
            **kwargs: Additional inference parameters (model, max_tokens, temperature, etc.)

        Returns:
            InferenceResponse
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        return await self._manager.agent_infer(
            agent_id=self.agent_id,
            context_page_ids=context_page_ids,
            prompt=prompt,
            **kwargs
        )

    # === Blackboard (Delegated to manager) ===

    async def get_blackboard(
        self,
        scope: str = "shared",
        scope_id: str | None = None,
        backend_type: str = "redis",
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for reading/writing shared state.

        Args:
            `scope`: Blackboard scope (`"local"`, `"shared"`, `"global"`)
            `scope_id`: Scope identifier (defaults to `agent_id` for shared scope)
            `backend_type`: Backend type for the blackboard (e.g., "redis")
            `enable_events`: Whether to enable events on the blackboard

        Returns:
            Blackboard instance

        Example:
            ```python
            # Get shared blackboard with parent
            parent_id = self.metadata.parent_id
            board = await self.get_blackboard(scope="shared", scope_id=parent_id)

            # Write results
            await board.write("analysis_complete", my_results)

            # Parent reads
            result = await board.read("analysis_complete")
            ```
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        # Default scope_id to agent_id for shared scope
        if scope == "shared" and scope_id is None:
            scope_id = self.agent_id

        return await self._manager.get_blackboard(
            scope=scope,
            scope_id=scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    # === Metrics ===

    async def update_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Update performance metrics for self-awareness.

        Args:
            metrics: Performance metrics to record
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        try:
            # Store metrics in metadata
            if "performance_metrics" not in self.metadata:
                self.metadata["performance_metrics"] = {}

            self.metadata["performance_metrics"].update(metrics)
            self.metadata["performance_last_updated"] = time.time()

        except Exception as e:
            raise

    # === Agent Hierarchy Management ===

    async def stop_child_agent(self, agent_id: str) -> None:
        """Stop a child agent.

        Args:
            agent_id: Agent identifier
        """
        agent_system_handle = get_agent_system()
        await agent_system_handle.stop_agent(agent_id)

    async def spawn_child_agents(
        self,
        agent_specs: list[AgentSpawnSpec],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        roles: list[str] = [],
        capability_types: list[list[type[AgentCapability]] | None] = None,
        return_handles: bool = False,
    ) -> list[str] | list["AgentHandle"]:
        """Spawn agents via agent system. This is intended to be called by this agent
        to spawn other agents. This method can be overridden by subclasses to add custom
        logic before/after spawning.

        Setting return_handles=True is the preferred method for spawning child agents when you need
        to interact with them via their capabilities.

        Args:
            agent_specs: List of AgentSpawnSpec defining agents to spawn
            session_id: Optional session ID for tracking which session spawned these agents
            run_id: Optional run ID for tracking which AgentRun spawned these agents
            soft_affinity: If True, allows spawning on replicas without all pages (will page fault)
            suspend_agents: If True, replica may suspend existing agents to make room
            roles: Role names for these children (e.g., "analyzer", "synthesizer")
            capability_types: List of lists of capability types for each child agent. Ignored if return_handles is False.
            return_handles: If True, return AgentHandle objects instead of agent IDs for communication.

        Returns:
            List of spawned agent IDs or AgentHandles

        Example:
            ```python
            # Spawn grounding agent with handle
            handle = await self.spawn_child_agents(
                agent_specs=[AgentSpawnSpec(agent_type="...GroundingAgent")],
                capability_types=[[GroundingCapability]],
            )[0]

            # Get capability proxy and stream events
            grounding = handle.get_capability_by_type(GroundingCapability)
            await grounding.stream_events_to_queue(self.get_event_queue())

            # Send request and wait for result
            req_id = await grounding.send_request(
                request_type="ground_claim",
                request_data={"claim": "...", "context": {...}}
            )
            future = await grounding.get_result_future(task_id=req_id)
            result = await asyncio.wait_for(future, timeout=30.0)
            ```
        """

        if len(roles) != len(agent_specs):
            raise ValueError("Number of roles must match number of agent_specs")

        child_ids: list[str] = spawn_agents(
            agent_specs=agent_specs,  # TODO: Should we include parent_agent_id=self.agent_id here?
            session_id=session_id,
            run_id=run_id,
            soft_affinity=soft_affinity,
            suspend_agents=suspend_agents,
        )

        for role, child_id in zip(roles, child_ids):
            # Track child
            self.child_agents[role] = child_id

            logger.info(f"Agent {self.agent_id} spawned child {role} ({child_id}) with subscriptions")

        if not return_handles:
            return child_ids

        if not capability_types or len(capability_types) != len(agent_specs):
            raise ValueError("Number of roles must match number of capability_types lists")

        if not capability_types or len(capability_types) != len(agent_specs):
            raise ValueError("Number of roles must match number of capability_types lists")

        # Create AgentHandles for each child
        handles: list[AgentHandle] = []
        for i, child_id in enumerate(child_ids):
            handle = AgentHandle(
                child_agent_id=child_id,
                owner=self,
                capability_types=capability_types[i],
            )
            handles.append(handle)
        return handles

    async def get_agent_hierarchy(self) -> dict[str, str]:
        """Get agent hierarchy mapping (agent_id -> parent_agent_id).

        Queries blackboard to discover full agent hierarchy across all plans.

        Returns:
            Dictionary mapping agent IDs to their parent agent IDs
        """
        # TODO: This agent hierarchy needs to be kept in sync with the
        # agent spawning mechanism in AgentSystem. The HierarchicalAccessPolicy
        # should ideally get this info from a central source too because new agents
        # can be spawned dynamically.
        hierarchy = {}

        # Add self if we have a parent
        parent_id = self.metadata.parent_agent_id
        if parent_id:
            hierarchy[self.agent_id] = parent_id

        # Query blackboard for all plans to discover hierarchy
        try:
            # Get blackboard if available
            if hasattr(self, "_plan_blackboard_cached"):
                blackboard = self._plan_blackboard_cached
            else:
                # Can't query yet during initialization, return minimal hierarchy
                return hierarchy

            # Query all plans
            # TODO: Make this mediocre query implementation more efficient later.
            # TODO: Use indexing or maintain a separate hierarchy store.
            # TODO: Use tenant_id to isolate multi-tenant data.
            all_plan_entries = await blackboard.get_all_plans(
                limit=1000,  # Reasonable limit for plan discovery
            )

            # Extract parent relationships from all plans
            for entry in all_plan_entries:
                plan_data = entry.value
                agent_id = plan_data.get("agent_id")
                parent_agent_id = plan_data.get("parent_agent_id")

                if agent_id and parent_agent_id:
                    # Only add if not already present (prefer metadata over discovered)
                    if agent_id not in hierarchy:
                        hierarchy[agent_id] = parent_agent_id

            logger.info(f"Discovered agent hierarchy with {len(hierarchy)} relationships")

        except Exception as e:
            logger.warning(f"Failed to discover agent hierarchy from blackboard: {e}")
            # Fall back to minimal hierarchy from metadata

        return hierarchy

    async def get_team_structure(self) -> dict[str, set[str]]:
        """Get team structure mapping (team_name -> set of agent_ids).

        Queries blackboard to discover team members across all plans.

        Returns:
            Dictionary mapping team names to sets of agent IDs
        """
        # TODO: This agent hierarchy needs to be kept in sync with the
        # agent spawning mechanism in AgentSystem. The HierarchicalAccessPolicy
        # should ideally get this info from a central source too because new agents
        # can be spawned dynamically.
        teams: dict[str, set[str]] = {}

        # Add self to team if specified
        team_name = self.metadata.team_id
        if team_name:
            teams[team_name] = {self.agent_id}

        # Query blackboard for all plans to discover team structure
        try:
            # Get blackboard if available
            if hasattr(self, "_plan_blackboard_cached"):
                blackboard = self._plan_blackboard_cached
            else:
                # Can't query yet during initialization, return minimal structure
                return teams

            # Query all plans
            # TODO: Make this mediocre query implementation more efficient later.
            # TODO: Use indexing or maintain a separate agent team store.
            # TODO: Use tenant_id to isolate multi-tenant data.
            all_plan_entries = await blackboard.get_all_plans(
                limit=1000,  # Reasonable limit for plan discovery
            )

            # Extract team information from plan metadata
            for entry in all_plan_entries:
                plan_data = entry.value
                agent_id = plan_data.get("agent_id")
                # Team info might be in plan metadata
                plan_metadata = plan_data.get("metadata", {})
                agent_team = plan_metadata.get("team")

                if agent_id and agent_team:
                    if agent_team not in teams:
                        teams[agent_team] = set()
                    teams[agent_team].add(agent_id)

            logger.info(f"Discovered {len(teams)} teams with {sum(len(members) for members in teams.values())} total members")

        except Exception as e:
            logger.warning(f"Failed to discover team structure from blackboard: {e}")
            # Fall back to minimal structure from metadata

        return teams

    @property
    def root_agent_id(self) -> str:
        """Returns the ID of the root agent of the agent tree."""
        # TODO: Implement

    def find_agent(self, agent_id: str) -> Agent | None:
        """Finds an agent by ID in the entire agent tree."""
        # TODO: Implement

    # ========================================================================
    # Memory and Context Capabilities
    # ========================================================================

    def get_context_engine(self):
        """Get the agent's context engine if available."""
        from .patterns.memory import AgentContextEngine

        return self.get_capability_by_type(AgentContextEngine)

    def get_stm(self):
        """Get the STM capability for an agent.

        Returns:
            STM capability
        """
        from .patterns.memory import MemoryCapability, MemoryScope

        # Find STM capability
        stm_scope = MemoryScope.agent_stm(self.agent_id)

        for cap_name in self.get_capability_names():
            cap = self.get_capability(cap_name)
            if isinstance(cap, MemoryCapability):
                if cap.scope_id == stm_scope:
                    return cap

        return None

    def get_working_memory(self):
        """Get the working memory capability for an agent.

        Returns:
            Working memory capability
        """
        from .patterns.memory import WorkingMemoryCapability

        ### for name in self.get_capability_names():
        ###     cap = self.get_capability(name)
        ###     if isinstance(cap, WorkingMemoryCapability):
        ###         return cap
        ### return None

        return self.get_capability_by_type(WorkingMemoryCapability)

    def get_sensory_memory(self):
        """Get the sensory memory capability for an agent.

        Returns:
            Sensory memory capability
        """
        from .patterns.memory import SensoryMemoryCapability
        return self.get_capability_by_type(SensoryMemoryCapability)

    def get_episodic_memory(self):
        """Get the episodic memory capability for an agent.

        Returns:
            Episodic memory capability
        """
        from .patterns.memory import EpisodicMemoryCapability
        return self.get_capability_by_type(EpisodicMemoryCapability)

        ### # Find episodic memory capability
        ### from .patterns.memory import MemoryCapability, MemoryScope
        ### episodic_scope = MemoryScope.agent_ltm_episodic(self.agent_id)

        ### for cap_name in self.get_capability_names():
        ###     cap = self.get_capability(cap_name)
        ###     if isinstance(cap, MemoryCapability):
        ###         if cap.scope_id == episodic_scope:
        ###             return cap
        ### return None

    def get_semantic_memory(self):
        """Get the semantic memory capability for an agent.

        Returns:
            Semantic memory capability
        """
        from .patterns.memory import SemanticMemoryCapability
        return self.get_capability_by_type(SemanticMemoryCapability)

    def get_long_term_memory(self):
        """Get the long-term memory capability for an agent.

        Returns:
            Long-term memory capability
        """
        from .patterns.memory import LongTermMemoryCapability
        return self.get_capability_by_type(LongTermMemoryCapability)

    # ========================================================================
    # Process Control Messages
    # ========================================================================

    async def _process_control_messages(self) -> None:
        """Process control messages.

        Called at the beginning of each run_step() to handle:
        - Dynamic parameter/metadata updates
        - Status requests/responses
        - Termination requests
        - Error escalations
        """
        pass  # TODO: Implement processing of control messages

    async def _check_child_health(self) -> None:
        """Check health of child agents.

        Called periodically in run_step() to monitor child agent status.
        """
        pass  # TODO: Implement child health monitoring




class AgentManagerBase:
    """Mixin for managing agents on a deployment.

    Provides start_agent, stop_agent, list_agents API.
    Used by both VLLMDeployment and StandaloneAgentDeployment.

    This mixin:
    - Maintains local registry of agents
    - Runs agent execution loops
    - Registers agents with AgentSystemDeployment
    - Injects runtime handles into agents
    - Provides communication infrastructure (via EnhancedBlackboard)
    """

    def __init__(self, deployment_config: DeploymentConfig | None = None):
        """Initialize agent manager.

        Note: This is a mixin, so it should be called from the deployment's __init__.

        Args:
            deployment_config: Optional deployment configuration with resource limits
        """
        # Agent management
        self._agents: dict[str, Agent] = {}
        self._agent_tasks: dict[str, asyncio.Task] = {}
        self._agent_lock = asyncio.Lock()

        # Resource limits (from DeploymentConfig)
        if deployment_config:
            self.max_agents = deployment_config.max_agents_per_replica
            self.max_cpu_cores = deployment_config.max_cpu_cores_per_replica
            self.max_memory_mb = deployment_config.max_memory_mb_per_replica
            self.max_gpu_cores = deployment_config.max_gpu_cores_per_replica
            self.max_gpu_memory_mb = deployment_config.max_gpu_memory_mb_per_replica
        else:
            # Defaults if no config provided
            self.max_agents = 100
            self.max_cpu_cores = 8.0
            self.max_memory_mb = 16384  # 16GB
            self.max_gpu_cores = 0.0
            self.max_gpu_memory_mb = 0

        # Current resource usage (tracked at agent start/stop)
        self._used_cpu_cores: float = 0.0
        self._used_memory_mb: int = 0
        self._used_gpu_cores: float = 0.0
        self._used_gpu_memory_mb: int = 0

        # Set by deployment initialize
        self._vcm_handle: serving.DeploymentHandle | None = None
        self._llm_cluster_handle: serving.DeploymentHandle | None = None
        self._agent_system_handle: serving.DeploymentHandle | None = None
        self._tool_manager_handle: serving.DeploymentHandle | None = None
        self._blackboard = None

    async def initialize(self) -> None:
        self._agent_system_handle = get_agent_system()
        self._tool_manager_handle = get_tool_manager()
        self._llm_cluster_handle = get_llm_cluster()
        self._vcm_handle = get_vcm()

    @serving.endpoint(
        router_class=SoftPageAffinityRouter,
        router_kwargs={"strip_routing_params": ["soft_affinity"]}
    )
    async def start_agent(
        self,
        agent_class_id: str,
        agent_id: str | None = None,
        capabilities: list[str] | None = None,
        action_policy_id: str | None = None,
        bound_pages: list[str] | None = None,
        suspend_agents: bool = False,
        metadata: dict[str, Any] | None = None,
        resource_requirements: AgentResourceRequirements | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """Start a new agent.

        Args:
            agent_class_id: Type of agent to create ("general", "specialized", etc.)
            agent_id: Optional ID (generated if None)
            capabilities: Optional list of capabilities for the agent. The agent role is inferred from capabilities.
            action_policy_id: Optional action policy ID to use
            bound_pages: Optional page binding
            suspend_agents: If True and ResourceExhausted, suspend existing agents to make room
            metadata: Agent metadata
            resource_requirements: Optional resource requirements (uses defaults if None)
            max_iterations: Optional maximum number of iterations for the agent's action policcy

        Returns:
            agent_id

        Raises:
            ResourceExhausted: If replica has insufficient resources and suspend_agents=False
        """
        async with self._agent_lock:
            if agent_id is None:
                agent_id = f"agent-{uuid.uuid4().hex[:8]}"

            # Use provided resource requirements or defaults
            if resource_requirements is None:
                resource_requirements = AgentResourceRequirements()

            # CHECK AGENT COUNT LIMIT
            if len(self._agents) >= self.max_agents:
                if suspend_agents and self._agents:
                    # Suspend an existing agent to make room
                    agent_to_suspend = await self._select_agent_for_suspension(
                        required_resources=resource_requirements,
                        preferred_bound_pages=bound_pages
                    )
                    if agent_to_suspend:
                        logger.info(
                            f"Agent capacity reached, suspending {agent_to_suspend} to make room for {agent_id}"
                        )
                        await self.suspend_agent(
                            agent_to_suspend,
                            reason=f"Agent capacity reached, making room for {agent_id}"
                        )
                    else:
                        raise ResourceExhausted(
                            f"Agent capacity reached: {len(self._agents)}/{self.max_agents} agents, "
                            f"no agents available to suspend"
                        )
                else:
                    raise ResourceExhausted(
                        f"Agent capacity reached: {len(self._agents)}/{self.max_agents} agents"
                    )

            # CHECK RESOURCE LIMITS
            new_cpu = self._used_cpu_cores + resource_requirements.cpu_cores
            new_memory = self._used_memory_mb + resource_requirements.memory_mb
            new_gpu = self._used_gpu_cores + resource_requirements.gpu_cores
            new_gpu_memory = self._used_gpu_memory_mb + resource_requirements.gpu_memory_mb

            needed_cpu = 0
            needed_memory = 0
            needed_gpu = 0
            needed_gpu_memory = 0

            if new_cpu > self.max_cpu_cores:
                if suspend_agents and self._agents:
                    needed_cpu += resource_requirements.cpu_cores
                else:
                    raise ResourceExhausted(
                        f"CPU capacity exceeded: {new_cpu:.2f}/{self.max_cpu_cores} cores "
                        f"(agent needs {resource_requirements.cpu_cores:.2f})"
                    )

            if new_memory > self.max_memory_mb:
                if suspend_agents and self._agents:
                    needed_memory += resource_requirements.memory_mb
                else:
                    raise ResourceExhausted(
                        f"Memory capacity exceeded: {new_memory}/{self.max_memory_mb} MB "
                        f"(agent needs {resource_requirements.memory_mb})"
                    )

            if new_gpu > self.max_gpu_cores:
                if suspend_agents and self._agents:
                    needed_gpu = resource_requirements.gpu_cores
                else:
                    raise ResourceExhausted(
                        f"GPU capacity exceeded: {new_gpu:.2f}/{self.max_gpu_cores} cores "
                        f"(agent needs {resource_requirements.gpu_cores:.2f})"
                    )

            if new_gpu_memory > self.max_gpu_memory_mb:
                if suspend_agents and self._agents:
                    needed_gpu_memory += resource_requirements.gpu_memory_mb
                else:
                    raise ResourceExhausted(
                        f"GPU memory capacity exceeded: {new_gpu_memory}/{self.max_gpu_memory_mb} MB "
                        f"(agent needs {resource_requirements.gpu_memory_mb})"
                    )

            if any(needed_cpu > 0, needed_memory > 0, needed_gpu > 0, needed_gpu_memory > 0):
                # Suspend agents until we have enough CPU/memory
                await self._suspend_agents_for_resources(
                    needed_cpu=needed_cpu,
                    needed_memory=needed_memory,
                    needed_gpu=needed_gpu,
                    needed_gpu_memory=needed_gpu_memory,
                    requester_id=agent_id,
                )
                # Recalculate after suspension
                new_cpu = self._used_cpu_cores + resource_requirements.cpu_cores
                new_memory = self._used_memory_mb + resource_requirements.memory_mb
                new_gpu = self._used_gpu_cores + resource_requirements.gpu_cores
                new_gpu_memory = self._used_gpu_memory_mb + resource_requirements.gpu_memory_mb
                if any(
                    new_cpu > self.max_cpu_cores,
                    new_memory > self.max_memory_mb,
                    new_gpu > self.max_gpu_cores,
                    new_gpu_memory > self.max_gpu_memory_mb,
                ):
                    raise ResourceExhausted(
                        f"CPU/Memory/GPU/GPU memory capacity exceeded even after suspension:\n"
                        f"{new_cpu:.2f}/{self.max_cpu_cores} cores\n"
                        f"{new_memory}/{self.max_memory_mb} MB\n"
                        f"{new_gpu:.2f}/{self.max_gpu_cores} cores\n"
                        f"{new_gpu_memory}/{self.max_gpu_memory_mb} MB"
                    )

            # Create agent instance (supports custom agent classes)
            agent: Agent = self._create_agent_instance(
                agent_class_id=agent_class_id,
                agent_id=agent_id,
                bound_pages=bound_pages or [],
                metadata=metadata or {},
                resource_requirements=resource_requirements,
                capabilities=capabilities or [],
                action_policy_id=action_policy_id,
            )

            # Attach manager reference for delegation
            agent.set_manager(self)

            # Initialize agent
            await agent.initialize()

            # Initialize mailbox for agent
            await self._ensure_mailbox(agent_id)

            # Store agent
            self._agents[agent_id] = agent

            # TRACK RESOURCE USAGE
            self._used_cpu_cores += resource_requirements.cpu_cores
            self._used_memory_mb += resource_requirements.memory_mb
            self._used_gpu_cores += resource_requirements.gpu_cores
            self._used_gpu_memory_mb += resource_requirements.gpu_memory_mb

            # Start agent loop
            # TODO: Move this to the agent.initialize()?
            task = asyncio.create_task(
                self._run_agent_loop(agent, max_iterations=max_iterations)
            )
            self._agent_tasks[agent_id] = task

            # Register with agent system
            if self._agent_system_handle:
                try:
                    deployment_replica_id = self._get_deployment_replica_id()
                    await self._agent_system_handle.register_agent(
                        agent=agent,
                        deployment_replica_id=deployment_replica_id,
                        deployment_name=serving.get_my_deployment_name(),
                    )
                except Exception as e:
                    logger.error(f"Failed to register agent {agent_id} with system: {e}")

            logger.info(
                f"Started agent {agent_id} (type={agent_class_id}, bound_pages={bound_pages}, "
                f"cpu={resource_requirements.cpu_cores}, mem={resource_requirements.memory_mb}MB)"
            )

            return agent_id

    @serving.endpoint(router_class=AgentAffinityRouter)
    async def stop_agent(self, agent_id: str, graceful: bool = True) -> None:
        """Stop an agent.

        Args:
            agent_id: Agent identifier
            graceful: If True, wait for graceful shutdown with timeout
        """
        async with self._agent_lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not found")

            agent = self._agents[agent_id]

            if graceful:
                # Signal agent to stop gracefully
                agent._stop_requested = True

                # Wait for graceful shutdown with timeout
                task = self._agent_tasks.get(agent_id)
                if task:
                    try:
                        await asyncio.wait_for(task, timeout=agent.stop_timeout_s)
                        logger.info(f"Agent {agent_id} stopped gracefully")
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Agent {agent_id} did not stop within {agent.stop_timeout_s}s, forcing"
                        )
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    finally:
                        if agent_id in self._agent_tasks:
                            del self._agent_tasks[agent_id]
            else:
                # Force stop immediately
                await agent.stop()
                task = self._agent_tasks.get(agent_id)
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    del self._agent_tasks[agent_id]

            # RELEASE RESOURCES
            self._used_cpu_cores -= agent.resource_requirements.cpu_cores
            self._used_memory_mb -= agent.resource_requirements.memory_mb
            self._used_gpu_cores -= agent.resource_requirements.gpu_cores
            self._used_gpu_memory_mb -= agent.resource_requirements.gpu_memory_mb

            # Unregister from agent system
            if self._agent_system_handle:
                try:
                    await self._agent_system_handle.unregister_agent(agent_id)
                except Exception as e:
                    logger.error(f"Failed to unregister agent {agent_id}: {e}")

            # Remove agent
            del self._agents[agent_id]

            logger.info(
                f"Stopped agent {agent_id} (released cpu={agent.resource_requirements.cpu_cores}, "
                f"mem={agent.resource_requirements.memory_mb}MB)"
            )

            # Try to resume suspended agents now that resources are available
            await self._try_resume_suspended_agents()

    @serving.endpoint(router_class=AgentAffinityRouter)
    async def suspend_agent(self, agent_id: str, reason: str = "") -> bool:
        """Actually suspend agent: cancel task, persist state, free resources, delete agent.

        This persists execution, communication, and cache state to StateManager
        (Redis/etcd) so the agent can resume seamlessly when resources become
        available.

        This is the real suspension implementation that:
        1. Cancels agent task
        2. Persists suspension state to StateManager (Redis/etcd)
        3. Frees resources (CPU, memory, GPU)
        4. Updates state to SUSPENDED
        5. DELETES agent from self._agents (agent no longer "owned")
        6. Updates AgentSystem (removes location via update_agent_state)

        State persisted includes:
        1. Execution state: plan ID, current action index, agent state
        2. Communication state: child tracking, parent info, message sequences
        3. Cache state: working set, page access patterns, page graph
        4. Resource state: allocated CPU/memory/GPU for reclamation

        Args:
            agent_id: Agent to suspend
            reason: Why agent is being suspended

        Returns:
            True if suspended successfully, False if agent not found
        """
        async with self._agent_lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return False

            # 1. Cancel agent task
            if agent_id in self._agent_tasks:
                task = self._agent_tasks[agent_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self._agent_tasks[agent_id]

            # 2. Persist suspension state to StateManager
            app_name = serving.get_my_app_name()
            polymathera = get_polymathera()
            state_key = AgentSuspensionState.get_state_key(app_name, agent_id)
            state_manager = await polymathera.get_state_manager(
                state_type=AgentSuspensionState,
                state_key=state_key,
            )

            suspension_count = agent.metadata.suspension_count + 1

            try:
                # Let agent serialize its own state
                # ✅ GOOD: Agent is responsible for its own state
                agent._suspend_reason = reason
                agent.metadata.suspension_count = suspension_count

                # Delegate to agent - it knows its own state structure
                suspension_state = await agent.serialize_suspension_state()
                suspension_state.suspension_count = suspension_count

                # Persist to StateManager
                async for state in state_manager.write_transaction():
                    # Copy suspension_state into persisted state
                    for field_name, field_value in suspension_state.model_dump().items():
                        setattr(state, field_name, field_value)

                logger.info(
                    f"Persisted suspension state for agent {agent_id} "
                    f"(suspension_count: {suspension_count})"
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist suspension state for {agent_id}: {e}",
                    exc_info=True
                )
                # Continue with suspension even if persistence fails

            # 3. Update agent state
            agent.state = AgentState.SUSPENDED
            agent._running = False

            # 4. Free resources
            self._used_cpu_cores -= agent.resource_requirements.cpu_cores
            self._used_memory_mb -= agent.resource_requirements.memory_mb
            self._used_gpu_cores -= agent.resource_requirements.gpu_cores
            self._used_gpu_memory_mb -= agent.resource_requirements.gpu_memory_mb

            # 5. DELETE agent from tracking (agent no longer "owned")
            del self._agents[agent_id]

            # 6. Update AgentSystem (this will remove location via update_agent_state)
            if self._agent_system_handle:
                try:
                    await self._agent_system_handle.update_agent_state(
                        agent_id, AgentState.SUSPENDED
                    )
                except Exception as e:
                    logger.error(f"Failed to update agent {agent_id} state in system: {e}")

            logger.info(
                f"Suspended and deleted agent {agent_id} "
                f"(freed cpu={agent.resource_requirements.cpu_cores}, "
                f"mem={agent.resource_requirements.memory_mb}MB). Reason: {reason}"
            )

        # 7. Setup event-driven resumption if blocked on dependency (outside lock)
        if reason.startswith("Blocked:"):
            await self._setup_resumption_trigger(agent_id, reason)

        return True

    async def _setup_resumption_trigger(self, agent_id: str, reason: str) -> None:
        """Setup event listener to resume agent when dependency is satisfied.

        Args:
            agent_id: Suspended agent ID
            reason: Suspension reason (contains dependency info)
        """
        try:
            # Parse reason to extract child agent IDs
            # Format: "Blocked: Waiting for children: child1,child2,child3"
            # TODO: Use structured metadata instead of parsing text
            child_agent_ids = []
            if "Waiting for children:" in reason:
                children_part = reason.split("Waiting for children:")[-1].strip()
                child_agent_ids = [cid.strip() for cid in children_part.split(",") if cid.strip()]

            if not child_agent_ids:
                logger.warning(
                    f"Could not parse child agent IDs from reason: {reason}. "
                    f"Agent {agent_id} will not auto-resume."
                )
                return

            # Get shared blackboard for plan coordination
            blackboard = await self.get_blackboard(
                scope="shared",
                scope_id="plan_coordination"
            )

            # Subscribe to specific child plan completions
            from .blackboard.types import KeyPatternFilter

            # Build pattern for child agent plan keys
            # Plan keys are like: "plan:<agent_id>"
            for child_id in child_agent_ids:
                plan_key_pattern = f"plan:{child_id}"

                async def on_child_completed(event):
                    """Resume parent when child completes."""
                    # Check if child plan is actually completed
                    if event.value and isinstance(event.value, dict):
                        status = event.value.get("status")
                        if status in ["completed", "failed"]:
                            logger.info(
                                f"Child {event.key} completed with status {status}, "
                                f"attempting to resume parent {agent_id}"
                            )
                            try:
                                # Resume via AgentSystem (not local manager)
                                await self._agent_system_handle.resume_agent(agent_id)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to resume agent {agent_id}: {e}"
                                )

                blackboard.subscribe(
                    on_child_completed,
                    filter=KeyPatternFilter(plan_key_pattern)
                )

            logger.info(
                f"Setup resumption triggers for agent {agent_id} "
                f"watching {len(child_agent_ids)} children: {child_agent_ids}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to setup resumption trigger for agent {agent_id}: {e}"
            )

    async def _select_agent_for_suspension(
        self,
        required_resources: AgentResourceRequirements,
        preferred_bound_pages: list[str] | None = None,
    ) -> str | None:
        """Select best agent to suspend to free resources.

        Selection heuristics (in priority order):
        1. Prefer agents with low recent activity (last_action_time)
        2. Prefer agents with fewer bound pages (less cache pressure)
        3. Prefer agents not in critical workflows (no parent waiting)
        4. Prefer agents with sufficient resources to free

        Args:
            required_resources: Resource requirements we need to free
            preferred_bound_pages: If provided, prefer suspending agents that DON'T use these pages

        Returns:
            Agent ID to suspend, or None if no suitable candidate found
        """
        candidates = []

        async with self._agent_lock:
            for agent_id, agent in self._agents.items():
                # Skip agents that don't free enough resources
                if (agent.resource_requirements.cpu_cores < required_resources.cpu_cores or
                    agent.resource_requirements.memory_mb < required_resources.memory_mb):
                    continue

                # Calculate score (higher = better candidate for suspension)
                score = 0.0

                # 1. Low recent activity (higher score = less recent)
                last_action_time = agent.metadata.last_action_time
                time_since_last_action = time.time() - last_action_time
                activity_score = min(time_since_last_action / 60.0, 10.0)  # Cap at 10 minutes
                score += activity_score * 100  # Weight: 100 points per minute idle (max 1000)

                # 2. Fewer bound pages (less cache pressure)
                bound_pages = agent.bound_pages  # Direct attribute, not metadata
                num_bound_pages = len(bound_pages)
                cache_score = max(0, 100 - num_bound_pages)  # Weight: -1 point per page
                score += cache_score

                # 3. Not in critical workflow (no parent waiting)
                has_parent = agent.metadata.parent_agent_id is not None
                if not has_parent:
                    score += 500  # Weight: 500 points for no parent dependency

                # 4. Page overlap penalty (avoid suspending agents using preferred pages)
                if preferred_bound_pages:
                    overlap = len(set(bound_pages) & set(preferred_bound_pages))
                    score -= overlap * 50  # Weight: -50 points per overlapping page

                # 5. Suspension fairness (penalize frequently suspended agents)
                suspension_count = agent.metadata.suspension_count
                score -= suspension_count * 20  # Weight: -20 points per previous suspension

                candidates.append({
                    'agent_id': agent_id,
                    'score': score,
                    'last_action_time': last_action_time,
                    'num_bound_pages': num_bound_pages,
                    'has_parent': has_parent,
                    'suspension_count': suspension_count,
                })

        if not candidates:
            logger.warning(
                f"No suitable agents to suspend for resources "
                f"(cpu={required_resources.cpu_cores}, mem={required_resources.memory_mb}MB)"
            )
            return None

        # Sort by score (highest first = best candidate)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]

        logger.info(
            f"Selected agent {best['agent_id']} for suspension "
            f"(score={best['score']:.1f}, idle={time.time() - best['last_action_time']:.1f}s, "
            f"pages={best['num_bound_pages']}, has_parent={best['has_parent']}, "
            f"suspensions={best['suspension_count']})"
        )

        return best['agent_id']

    @serving.endpoint
    async def list_agents(self) -> list[str]:
        """List all agent IDs on this deployment.

        Returns:
            List of agent IDs
        """
        return list(self._agents.keys())

    @serving.endpoint(router_class=AgentAffinityRouter)
    async def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Get agent's current state.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentState if agent exists, None otherwise
        """
        agent = self._agents.get(agent_id)
        return agent.state if agent else None

    @serving.replica_property("resource_usage")
    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage for this replica.

        Returns:
            Dictionary with resource usage statistics
        """
        return {
            "agents": len(self._agents),
            "max_agents": self.max_agents,
            "cpu_cores_used": self._used_cpu_cores,
            "cpu_cores_max": self.max_cpu_cores,
            "cpu_cores_available": self.max_cpu_cores - self._used_cpu_cores,
            "memory_mb_used": self._used_memory_mb,
            "memory_mb_max": self.max_memory_mb,
            "memory_mb_available": self.max_memory_mb - self._used_memory_mb,
            "gpu_cores_used": self._used_gpu_cores,
            "gpu_cores_max": self.max_gpu_cores,
            "gpu_cores_available": self.max_gpu_cores - self._used_gpu_cores,
            "gpu_memory_mb_used": self._used_gpu_memory_mb,
            "gpu_memory_mb_max": self.max_gpu_memory_mb,
            "gpu_memory_mb_available": self.max_gpu_memory_mb - self._used_gpu_memory_mb,
        }

    async def _run_agent_loop(self, agent: Agent, max_iterations: int | None = None) -> None:
        """Run agent's execution loop (internal).

        Args:
            agent: Agent instance
            max_iterations: Optional maximum number of iterations for the agent's action policy
        """
        try:
            await agent.start()

            iteration = 0
            while not agent._stop_requested and (max_iterations is None or iteration < max_iterations):
                # Check if agent requested suspension
                if agent._suspend_requested:
                    logger.info(
                        f"Agent {agent.agent_id} requested suspension, handling it"
                    )
                    await self.suspend_agent(
                        agent.agent_id,
                        reason=agent._suspend_reason or "Agent requested suspension"
                    )
                    # suspend_agent() cancels our task, so we won't reach here
                    return

                try:
                    logger.debug(f"Agent {agent.agent_id} running iteration {iteration}")
                    await agent.run_step()
                    iteration += 1
                except Exception as e:
                    logger.error(f"Error in agent {agent.agent_id} step: {e}")
                    agent.state = AgentState.FAILED
                    break

        except asyncio.CancelledError:
            logger.info(f"Agent {agent.agent_id} loop cancelled")
        except Exception as e:
            logger.error(f"Error in agent {agent.agent_id} loop: {e}")
            agent.state = AgentState.FAILED
        finally:
            agent._running = False

    def _get_deployment_replica_id(self) -> str:
        """Get deployment replica ID for this agent manager."""
        return serving.get_my_replica_id()

    async def _suspend_agents_for_resources(
        self,
        needed_cpu: float,
        needed_memory: int,
        needed_gpu: float,
        needed_gpu_memory: int,
        requester_id: str,
    ) -> None:
        """Suspend agents until enough resources are available.

        Args:
            needed_cpu: CPU cores needed
            needed_memory: Memory MB needed
            needed_gpu: GPU cores needed
            needed_gpu_memory: GPU memory MB needed
            requester_id: ID of agent requesting resources

        Raises:
            ResourceExhausted: If cannot free enough resources
        """
        freed_cpu = 0.0
        freed_memory = 0
        freed_gpu = 0.0
        freed_gpu_memory = 0

        # Sort agents by creation time (oldest first - FIFO suspension policy)
        agents_to_consider = sorted(
            self._agents.items(),
            key=lambda item: item[1].created_at
        )

        suspended_agents = []

        for agent_id, agent in agents_to_consider:
            # Check if we have enough resources
            if (freed_cpu >= needed_cpu and
                freed_memory >= needed_memory and
                freed_gpu >= needed_gpu and
                freed_gpu_memory >= needed_gpu_memory):
                break

            # Suspend this agent
            logger.info(
                f"Suspending agent {agent_id} to free resources for {requester_id} "
                f"(cpu={agent.resource_requirements.cpu_cores}, "
                f"mem={agent.resource_requirements.memory_mb}MB)"
            )

            await self.suspend_agent(
                agent_id,
                reason=f"Freeing resources for {requester_id}"
            )

            freed_cpu += agent.resource_requirements.cpu_cores
            freed_memory += agent.resource_requirements.memory_mb
            freed_gpu += agent.resource_requirements.gpu_cores
            freed_gpu_memory += agent.resource_requirements.gpu_memory_mb
            suspended_agents.append(agent_id)

        logger.info(
            f"Suspended {len(suspended_agents)} agents for {requester_id}, "
            f"freed: cpu={freed_cpu:.2f}, mem={freed_memory}MB, "
            f"gpu={freed_gpu:.2f}, gpu_mem={freed_gpu_memory}MB"
        )

    async def _try_resume_suspended_agents(self) -> None:
        """Try to resume suspended agents when resources become available.

        NOTE: This is a placeholder. Resource-driven resumption should be implemented
        in AgentSystemDeployment which has global visibility of all suspended agents
        via StateManager. Suspended agents are DELETED from replica's self._agents,
        so this method cannot find them locally.

        TODO: Move resource-driven resumption logic to AgentSystemDeployment
        as described in AGENT_SUSPENSION_ARCHITECTURE.md
        """
        # Suspended agents are deleted from self._agents, so we can't find them here
        # Resource-driven resumption should be handled by AgentSystem
        pass

    def _create_agent_instance(
        self,
        agent_class_id: str,
        agent_id: str,
        bound_pages: list[str],
        metadata: dict[str, Any],
        resource_requirements: AgentResourceRequirements,
        capability_class_ids: list[str] | None = None,
        action_policy_class_id: str | None = None,
    ) -> Agent:
        """Create an agent instance, supporting custom agent classes.

        Args:
            agent_class_id: Agent class identifier - can be:
                - Simple name like "agent" or "CodeAnalyzer" (uses base Agent or looks in default packages)
                - Fully qualified like "mypackage.agents.CodeAnalyzer"
            agent_id: Agent identifier
            bound_pages: Page bindings
            metadata: Agent metadata
            resource_requirements: Resource requirements
            capability_class_ids: Optional list of capability class IDs
            action_policy_class_id: Optional action policy ID

        Returns:
            Agent instance

        Raises:
            ValueError: If agent class cannot be found or instantiated
        """
        # Try to get agent class
        agent_class: type[Agent] | None = self._resolve_class_from_identifier(
            class_id=agent_class_id,
            base_class=Agent,
            search_packages=[
                f"polymathera.colony.agents",  # e.g., agents.CodeAnalyzer
                f"polymathera.agents",  # Legacy location
            ],
        )

        # Resolve action policy if provided
        action_policy_class: type[ActionPolicy] | None = self._resolve_class_from_identifier(
            class_id=action_policy_class_id,
            base_class=ActionPolicy,
            search_packages=[
                f"polymathera.colony.agents",  # e.g., agents.SimpleActionPolicy
                f"polymathera.agents",  # Legacy location
            ],
        )
        # Resolve capability classes if provided
        capability_classes: list[type[AgentCapability]] = []
        for cap_class_id in (capability_class_ids or []):
            cap_class = self._resolve_class_from_identifier(
                class_id=cap_class_id,
                base_class=AgentCapability,
                search_packages=[
                    f"polymathera.colony.agents",  # e.g., agents.WebBrowsingCapability
                    f"polymathera.agents",  # Legacy location
                ],
            )
            if cap_class:
                capability_classes.append(cap_class)

        # Create instance
        try:
            agent = agent_class(
                agent_id=agent_id,
                agent_type=agent_class_id,
                capability_classes=capability_classes,
                action_policy_class=action_policy_class,
                bound_pages=bound_pages,
                metadata=metadata,
                resource_requirements=resource_requirements,
            )
            return agent
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate agent class '{agent_class_id}': {e}"
            ) from e

    def _resolve_class_from_identifier(
        self,
        class_id: str,
        base_class: type | None,
        search_packages: list[str],
    ) -> type | None:
        """Resolve a class from its identifier, supporting fully qualified names.

        Args:
            class_id: Class identifier (simple or fully qualified)
            base_class: Expected base class type
            search_packages: List of packages to search for simple names

        Returns:
            Resolved class type

        Raises:
            ValueError: If class cannot be found or is not a subclass of base_class
        """
        import importlib

        # Try to get agent class
        cls = None
        if class_id is None:
            return cls

        class_id = class_id.strip()

        # Case 2: Fully qualified name like "mypackage.agents.CodeAnalyzer"
        if "." in class_id:
            try:
                module_path, class_name = class_id.rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                logger.debug(f"Loaded class from fully qualified name: {class_id}")
            except (ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import class '{class_id}': {e}"
                ) from e

        # Case 3: Simple name - try to find in common packages
        else:
            # Try common agent packages
            for package_path in search_packages:
                try:
                    module = importlib.import_module(package_path)
                    cls = getattr(module, class_id)
                    logger.debug(f"Found agent class in: {package_path}")
                    break
                except (ImportError, AttributeError):
                    continue

            if cls is None:
                raise ValueError(
                    f"Class '{class_id}' not found. "
                    f"Use fully qualified name (e.g., 'mypackage.agents.{class_id}') "
                    f"or ensure class is in common packages."
                )

        # Verify it's actually a subclass of base_class
        if base_class is not None and not issubclass(cls, base_class):
            raise ValueError(
                f"Class '{class_id}' is not a subclass of {base_class.__name__}"
            )

        return cls

    # === Agent Delegation Methods (Called by Agent stubs) ===

    async def agent_infer(
        self,
        agent_id: str,
        context_page_ids: list[str] | None = None,
        prompt: str | None = None,
        **kwargs
    ) -> InferenceResponse:
        """Delegate agent inference to LLMCluster.

        Args:
            agent_id: Agent identifier
            context_page_ids: Optional list of virtual page IDs for context
            prompt: Prompt text
            **kwargs: Additional inference parameters

        Returns:
            InferenceResponse from LLM cluster

        Raises:
            RuntimeError: If LLM cluster handle not initialized
        """
        if not self._llm_cluster_handle:
            raise RuntimeError("LLM cluster handle not initialized")

        # Create inference request
        request = InferenceRequest(
            request_id=f"agent-{agent_id}-{uuid.uuid4().hex[:8]}",
            prompt=prompt or "",
            context_page_ids=context_page_ids or [],
            **kwargs
        )

        logger.debug(
            f"Agent {agent_id}: Submitting inference request with "
            f"{len(context_page_ids or [])} context pages"
        )

        # Delegate to LLM cluster
        response = await self._llm_cluster_handle.infer(request)

        return response

    async def agent_request_page(
        self,
        agent_id: str,
        page_id: str,
        priority: int = 0,
        tenant_id: str = "default",
    ) -> None:
        """Delegate page loading request to VCM.

        Args:
            agent_id: Agent identifier
            page_id: Virtual page identifier
            priority: Load priority (0-100)

        Raises:
            RuntimeError: If VCM handle not initialized
        """
        if not self._vcm_handle:
            raise RuntimeError("VCM handle not initialized")

        logger.debug(f"Agent {agent_id}: Requesting page {page_id} (priority={priority})")

        # Delegate to VCM
        await self._vcm_handle.request_page_load(
            page_id=page_id,
            priority=priority,
            agent_id=agent_id,
            tenant_id=tenant_id
        )

    async def agent_is_page_loaded(
        self,
        agent_id: str,
        page_id: str
    ) -> bool:
        """Check if page is loaded in VCM.

        Args:
            agent_id: Agent identifier
            page_id: Virtual page identifier

        Returns:
            True if page is loaded

        Raises:
            RuntimeError: If VCM handle not initialized
        """
        if not self._vcm_handle:
            raise RuntimeError("VCM handle not initialized")

        # Delegate to VCM
        is_loaded = await self._vcm_handle.is_page_loaded(page_id)

        logger.debug(f"Agent {agent_id}: Page {page_id} loaded={is_loaded}")

        return is_loaded

    async def get_blackboard(
        self,
        scope: str = "shared",
        scope_id: str = "default",
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for shared state access.

        Args:
            scope: Blackboard scope ("local", "shared", "global")
            scope_id: Scope identifier
            backend_type: Backend type ("memory", "distributed", "redis"). Auto-selected if None.
            enable_events: Enable event system for reactive updates (default: True)

        Returns:
            EnhancedBlackboard instance with:
            - Efficient backend-specific queries
            - Event-driven notifications (TRUE pub-sub via Redis)
            - Optimistic locking for transactions
            - Rich metadata (TTL, tags, versioning)

        Example:
            ```python
            # Get shared blackboard with events enabled
            board = await self.get_blackboard(scope="shared", scope_id="group-1")

            # Subscribe to events for reactive coordination
            async def on_task_complete(event):
                logger.info(f"Task {event.key} completed: {event.value}")

            board.subscribe(on_task_complete, filter=KeyPatternFilter("task:*:complete"))

            # Write with metadata
            await board.write("task:123:result", result, created_by=self.agent_id, tags={"final"})

            # Efficient batch operations
            results = await board.read_batch(["task:1:result", "task:2:result"])
            ```
        """
        app_name = serving.get_my_app_name()

        # Create and initialize enhanced blackboard
        blackboard = EnhancedBlackboard(
            app_name=app_name,
            scope=BlackboardScope(scope),
            scope_id=scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )

        await blackboard.initialize()

        return blackboard

