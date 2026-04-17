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

import os
import asyncio
import logging
import time
import uuid
import functools
from typing import Any, Callable, AsyncIterator, ClassVar, TYPE_CHECKING
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr
import networkx as nx

from ..cluster.models import InferenceRequest, InferenceResponse
from .models import (
    ActionStatus,
    AgentState,
    ActionPolicyIterationResult,
    ActionPolicyExecutionState,
    AgentRegistrationInfo,
    AgentResourceRequirements,
    AgentSuspensionState,
    AgentMetadata,
    ResumptionCondition,
    ResumptionConditionType,
)
from .blueprint import (
    AgentCapabilityBlueprint,
    ActionPolicyBlueprint,
    AgentBlueprint,
)
from .blackboard import EnhancedBlackboard, BlackboardEvent
from .blackboard.types import KeyPatternFilter, EventFilter
from .sessions.models import AgentRun, AgentRunConfig, AgentRunEvent, RunStatus, RunResourceUsage
from ..distributed import get_polymathera
from ..distributed.state_management import StateManager
from ..distributed.ray_utils import serving
from .routing import AgentAffinityRouter, SoftPageAffinityRouter
from ..vcm.page_storage import PageStorage, PageStorageConfig
from ..cluster.config import LLMDeploymentConfig
from ..distributed.hooks import (
    tracing,
    hookable,
    install_hook_handlers,
    uninstall_hook_handlers,
    get_hook_registry,
    HookRegistry,
)
from ..distributed.observability import TracingFacility, TracingConfig


if TYPE_CHECKING:
    from .patterns.memory.types import CapabilityMemoryRequirements
    from ..cluster import LLMClientRequirements
    from .blackboard.protocol import BlackboardProtocol
    from .scopes import BlackboardScope



logger = logging.getLogger(__name__)


class ResourceExhausted(Exception):
    """Raised when replica has insufficient resources for new agent."""
    pass


@tracing(
    publish_key=lambda self: self.agent.agent_id,
    subscribe_key=lambda self: self.agent.agent_id
)
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

    async def use_capability_blueprints(self, blueprints: list[AgentCapabilityBlueprint]) -> None:
        """Instantiate capability blueprints and add them as action providers.

        Args:
            blueprints: List of capability blueprints to instantiate and use.
        """
        await self.agent._instantiate_capability_blueprints(blueprints)

        # Mark ALL capabilities as "used" by the action policy.
        self.use_agent_capabilities([bp.key for bp in blueprints])

    @classmethod
    def bind(cls, **kwargs) -> ActionPolicyBlueprint:
        """Create an ActionPolicyBlueprint from this class and constructor kwargs.

        The agent and action_providers arguments are injected at
        local_instance() time — never bound here.
        """
        bp = ActionPolicyBlueprint(cls, kwargs)
        bp.validate_serializable()
        return bp

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


@tracing(
    publish_key=lambda self: self.agent.agent_id,
    subscribe_key=lambda self: self.agent.agent_id
)
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
    - `stream_events_to_queue()`: Stream capability events to action policy (default uses ``input_patterns``)
    - `get_result_future()`: Get future for capability's task result
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        input_patterns: list[str] | None = None,
        blackboard: EnhancedBlackboard | None = None,
        capability_key: str | None = None,
    ):
        """Initialize capability.

        ``input_patterns`` is used by the default ``stream_events_to_queue()`` to subscribe only to
        relevant events instead of ``"*"``. If empty, falls back to ``"*"``
        (legacy behavior, logs a warning).

        If ``input_patterns`` is not explicitly passed, it is inferred from methods
        decorated with ``@event_handler`` on this class (including inherited methods).
        Deduplicates across the MRO. Capabilities that subscribe to a different
        blackboard than ``self.get_blackboard()`` (e.g., control plane lifecycle scope)
        must explicitly pass ``input_patterns=[]`` to opt out of inference.

        Args:
            agent: Agent using this capability (None for detached mode)
            scope_id: Blackboard scope ID. Defaults to {ScopeUtils.get_agent_level_scope(agent)}. Required if agent is None (detached mode).
            input_patterns: Glob patterns this capability monitors for incoming events (fed into ``stream_events_to_queue()`` which, among other things, is called by the agent's action policy).
            blackboard: Pre-configured blackboard (for detached mode)
            capability_key: Instance-level key for the agent's ``_capabilities`` dict.
                Defaults to ``f"{cls.__name__}:{self.scope_id}"`` via ``get_capability_name()``.

        Raises:
            ValueError: If both agent and scope_id are None
        """
        self._agent = agent
        self._input_patterns = input_patterns
        self._blackboard: EnhancedBlackboard | None = blackboard
        self._pending_request_id: str | None = None
        self._capability_key: str | None = capability_key

        # In attached mode, derive scope from agent
        # In detached mode, scope_id must be provided
        from .scopes import ScopeUtils
        if agent is not None:
            self.scope_id = scope_id or ScopeUtils.get_agent_level_scope(agent)
        elif scope_id is not None:
            self.scope_id = scope_id
        else:
            raise ValueError(
                "Either 'agent' or 'scope_id' must be provided. "
                "For detached mode, provide scope_id explicitly."
            )

    @property
    def input_patterns(self) -> list[str]:
        """Get input patterns this capability monitors.

        If ``input_patterns`` was not explicitly passed to ``__init__``,
        infers patterns from ``@event_handler``-decorated methods on this
        class (including inherited methods). Deduplicates across the MRO.

        Capabilities that subscribe to a different blackboard than
        ``self.get_blackboard()`` (e.g., control plane lifecycle scope)
        must explicitly pass ``input_patterns=[]`` to opt out of inference.
        """
        if self._input_patterns is not None:
            return list(self._input_patterns)

        # Auto-infer from @event_handler methods by inspecting the class
        # hierarchy (MRO), not the instance — avoids infinite recursion from
        # getattr triggering this property again.
        patterns: list[str] = []
        seen: set[str] = set()
        for cls in type(self).__mro__:
            for name, attr in vars(cls).items():
                if callable(attr) and getattr(attr, '_is_event_handler', False):
                    pattern = getattr(attr, '_event_pattern', None)
                    if pattern is not None and isinstance(pattern, str) and pattern not in seen:
                        seen.add(pattern)
                        patterns.append(pattern)
        return patterns

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
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for this capability's scope.

        In attached mode: uses agent's blackboard
        In detached mode: creates/uses standalone blackboard

        Args:
            backend_type: Backend type override. None reads from cluster config.
            enable_events: Enable event system for reactive updates.

        Returns:
            Blackboard scoped to this capability
        """
        if self._blackboard is not None:
            return self._blackboard

        if self._agent is not None:
            # Attached mode: use agent's blackboard
            self._blackboard = await self._agent.get_blackboard(
                scope_id=self.scope_id,
                backend_type=backend_type,
                enable_events=enable_events,
            )
        else:
            # Detached mode: create standalone blackboard
            app_name = serving.get_my_app_name()

            self._blackboard = EnhancedBlackboard(
                app_name=app_name,
                scope_id=self.scope_id,
                backend_type=backend_type,
                enable_events=enable_events,
            )
            await self._blackboard.initialize()

        return self._blackboard

    @classmethod
    def get_capability_name(cls) -> str:
        """Get the name of the capability.

        Returns:
            Name of the capability
        """
        return cls.__name__

    @classmethod
    def get_capability_tags(cls) -> frozenset[str]:
        """Return domain/modality tags for hierarchical action scoping.

        Override in subclasses to provide tags used for filtering action groups
        during scope selection. Tags are free-form strings:
        "memory", "analysis", "coordination", "synthesis", "expensive", etc.

        Returns:
            frozenset of tag strings (default: empty)
        """
        return frozenset()

    @property
    def capability_key(self) -> str:
        """Unique key for this instance within an agent's _capabilities dict.

        Set by AgentCapabilityBlueprint or directly in constructor.
        Falls back to get_capability_name() (cls.__name__) for single-instance capabilities.
        """
        return self._capability_key or f"{self.get_capability_name()}:{self.scope_id}"

    @classmethod
    def bind(cls, **kwargs) -> AgentCapabilityBlueprint:
        """Create an AgentCapabilityBlueprint from this class and constructor kwargs.

        The agent positional arg is injected at local_instance() time — never bound.
        Use .with_composition(key=, include_actions=, ...) to set composition metadata.
        """
        bp = AgentCapabilityBlueprint(cls, kwargs)
        bp.validate_serializable()
        return bp

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

        Default implementation subscribes to all patterns declared in
        ``input_patterns``. If ``input_patterns`` is empty, falls back to
        ``"*"`` (all writes within the capability's blackboard scope) with
        a deprecation warning.

        Subclasses can override to customize the event filter or subscribe
        to multiple blackboards.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        blackboard = await self.get_blackboard()
        patterns = self.input_patterns
        if not patterns:
            logger.warning(
                f"{self.__class__.__name__} has no input_patterns declared — "
                f"streaming all events (\"*\"). Declare input_patterns to "
                f"filter events and reduce noise."
            )
            patterns = ["*"]
        for pattern in patterns:
            blackboard.stream_events_to_queue(
                event_queue,
                pattern=pattern,
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
        ``result:run:{request_id}`` is written to the blackboard
        (using ``AgentRunProtocol``). Subclasses can override to use
        a different protocol.

        Returns:
            Future that resolves with the task result
        """
        from .blackboard.protocol import AgentRunProtocol
        blackboard = await self.get_blackboard()
        request_id = self._pending_request_id or "default"
        result_key = AgentRunProtocol.result_key(request_id)
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

        Generic method for sending requests via blackboard. Uses
        ``AgentRunProtocol`` key format.

        Subclasses may provide more specific methods (e.g., ``abort()``,
        ``ground_claim()``) that use capability-specific protocols.

        Works in both attached and detached modes.

        Args:
            request_type: Type of request (e.g., "abort", "ground_claim")
            request_data: Request payload
            request_id: Optional request ID (generated if None)

        Returns:
            Request ID for tracking
        """
        from .blackboard.protocol import AgentRunProtocol
        blackboard = await self.get_blackboard()
        request_id = request_id or f"req_{uuid.uuid4().hex[:8]}"

        # Determine sender ID based on mode
        sender_id = self._agent.agent_id if self._agent else "detached"

        # Key is scope-relative — no scope_id prefix
        key = AgentRunProtocol.request_key(request_id)
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

    async def initialize(self) -> None:
        """Initialize the capability.

        Base implementation auto-registers any methods decorated with
        `@hook_handler`. Subclasses should call `await super().initialize()`
        to enable declarative hook registration.

        Example:
            ```python
            class MyCapability(AgentCapability):
                @hook_handler(
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
        self.install_hook_handlers()

    def install_hook_handlers(self) -> list[str]:
        """Install all @hook_handler decorated methods.

        Called by `initialize()`. Can also be called manually if needed.

        In detached mode, returns empty list (no hooks can be registered).

        Returns:
            List of registered hook IDs
        """
        if self._agent is None:
            # No hooks in detached mode
            return []
        return install_hook_handlers(listener=self)

    def uninstall_hooks(self) -> None:
        """Uninstall all hooks.

        In detached mode, does nothing (no hooks can be registered).
        """
        if self._agent is None:
            # No hooks in detached mode
            return
        uninstall_hook_handlers(listener=self)

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

    async def stop(self) -> None:
        """Stop the capability and perform any necessary cleanup.

        This is called when the owning agent is stopping. Subclasses can
        override to clean up resources, cancel tasks, etc.

        Note: This does not remove the capability from the agent's
        `_capabilities` dict. To fully remove a capability, call
        `agent.remove_capability(capability_key)`.
        """
        pass



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
       result = await handle.run({"query": "analyze code"}, protocol=AgentRunProtocol, namespace="analysis")
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
       result = await handle.run({"query": "analyze code"}, protocol=AgentRunProtocol, namespace="analysis", timeout=60)
       ```

    5. **Stream task** (NEW) - Execute with streaming intermediate results:
       ```python
       handle = await AgentHandle.from_agent_id("agent_123")
       async for event in handle.run_streamed({"query": "analyze code"}, protocol=AgentRunProtocol, namespace="analysis"):
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
        app_name: str | None = None
    ):
        """Initialize agent handle.

        Args:
            child_agent_id: ID of the target agent
            owner: Parent agent (None for detached mode)
            capability_types: Expected capability types (for validation)
            default_capability_type: Default capability for run/stream methods
            app_name: The `serving.Application` name where the agent system resides.
                    Required when creating detached handles from outside any `serving.deployment`.
        """
        self.child_agent_id = child_agent_id
        self._owner = owner
        self._capability_types = capability_types or []
        self._capabilities: dict[str, AgentCapability] = {}
        self._default_capability_type = default_capability_type
        self._app_name = app_name

        # For detached mode
        self._agent_info: AgentRegistrationInfo | None = None
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
        app_name: str | None = None
    ) -> "AgentHandle":
        """Create a detached AgentHandle from an agent ID.

        This is the primary way to interact with agents from outside
        the agent system (e.g., from API handlers, session managers).

        Args:
            agent_id: Target agent ID
            default_capability_type: Default capability for run/stream
            app_name: The `serving.Application` name where the agent system resides.

        Returns:
            AgentHandle in detached mode

        Raises:
            ValueError: If agent not found

        Example:
            ```python
            # Get handle to system administrator agent
            handle = await AgentHandle.from_agent_id("system_admin_agent")
            result = await handle.run({"task": "check_health"}, protocol=AgentRunProtocol, namespace="admin")
            ```
        """
        handle = cls(
            child_agent_id=agent_id,
            owner=None,
            default_capability_type=default_capability_type,
            app_name=app_name
        )

        # Load agent metadata from AgentSystem
        await handle._load_agent_metadata()

        return handle

    @classmethod
    async def from_blueprint(
        cls,
        agent_blueprint: AgentBlueprint,
        *,
        requirements: "LLMClientRequirements" | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        default_capability_type: type[AgentCapability] | None = None,
        app_name: str | None = None
    ) -> "AgentHandle":
        """Spawn an agent from a blueprint and return handle.

        The blueprint is the single source of truth — session_id, run_id,
        max_iterations all live in blueprint.metadata.

        Args:
            agent_blueprint: AgentBlueprint defining the agent to spawn
            requirements: LLM deployment requirements for routing (LLMClientRequirements)
            soft_affinity: Whether to use soft affinity routing
            suspend_agents: Whether to suspend other agents if needed
            default_capability_type: Default capability for run/stream (handle config)
            app_name: The `serving.Application` name where the agent system resides.
                This is required when `from_blueprint` is called from outside any
                `serving.deployment`.

        Returns:
            AgentHandle for the spawned agent

        Raises:
            ValueError: If failed to spawn agent
        """
        from ..system import spawn_agents

        child_ids: list[str] = await spawn_agents(
            blueprints=[agent_blueprint],
            requirements=requirements,
            soft_affinity=soft_affinity,
            suspend_agents=suspend_agents,
            app_name=app_name
        )
        agent_id = child_ids[0]

        return await cls.from_agent_id(
            agent_id=agent_id,
            default_capability_type=default_capability_type,
            app_name=app_name
        )

    async def _load_agent_metadata(self) -> None:
        """Load agent registration info from AgentSystemDeployment."""
        from ..system import get_agent_system

        agent_system = get_agent_system(app_name=self._app_name)
        agent_info = await agent_system.get_agent_info(self.child_agent_id)

        if agent_info is None:
            raise ValueError(f"Agent {self.child_agent_id} not found")

        self._agent_info = agent_info

    @property
    def agent_info(self) -> AgentRegistrationInfo | None:
        """Get the agent's registration info.

        Available after creating handle via from_agent_id() or from_blueprint().
        Contains agent_type, capability_names, state, bound_pages, etc.
        """
        return self._agent_info

    async def _get_child_blackboard(
        self,
        scope_id: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for communicating with the child agent.

        Args:
            scope_id: Override the blackboard scope. If None, uses the child's
                agent-level scope. Pass a colony/session scope when the target
                capability operates at a shared scope level.
            backend_type: Backend type override.
            enable_events: Whether to enable event notifications.
        """
        # Only cache when using default scope (agent-level)
        if scope_id is None and self._blackboard is not None:
            return self._blackboard

        from .scopes import ScopeUtils
        effective_scope_id = scope_id or ScopeUtils.get_agent_level_scope(self.child_agent_id)

        if self._owner is not None:
            return await self._owner.get_blackboard(
                scope_id=effective_scope_id,
                backend_type=backend_type,
                enable_events=enable_events,
            )

        # Detached mode: create blackboard
        app_name = self._app_name or serving.get_my_app_name()

        bb = EnhancedBlackboard(
            app_name=app_name,
            scope_id=effective_scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )
        await bb.initialize()

        # Cache only the default (agent-level) blackboard
        if scope_id is None:
            self._blackboard = bb

        return bb

    # =========================================================================
    # Capability Access
    # =========================================================================

    def get_capability(
        self,
        capability_type: type[AgentCapability],
        capability_key: str | None = None,
        **kwargs,
    ) -> AgentCapability:
        """Get capability instance for communicating with agent.

        Creates a capability instance with `**kwargs` passed to the constructor,
        enabling communication via the same capability interface.

        In detached mode, creates capability in detached mode using
        blackboard directly.

        Args:
            capability_type: Type of capability to instantiate
            capability_key: Explicit cache key matching the key the target agent
                registered this capability under.  When ``None``, falls back to
                ``capability_type.get_capability_name()`` (the class name).
            **kwargs: Additional kwargs to pass to the capability constructor

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
        cache_key = capability_key or capability_type.get_capability_name()
        if cache_key not in self._capabilities:
            cap_kwargs = dict(kwargs)
            if capability_key is not None:
                cap_kwargs["capability_key"] = capability_key
            if self.is_detached:
                # Detached mode: create capability with scope_id only
                self._capabilities[cache_key] = capability_type(
                    agent=None,
                    **cap_kwargs,
                )
            else:
                # Owned mode: existing behavior
                self._capabilities[cache_key] = capability_type(
                    agent=self._owner,
                    **cap_kwargs,
                )
        return self._capabilities[cache_key]

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
        run_id: str | None = None,
        protocol: "type[BlackboardProtocol] | None" = None,
        scope: BlackboardScope | None = None,
        namespace: str = "",
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
            run_id: Explicit run ID (matches agent's metadata.run_id to avoid mismatch)
            protocol: BlackboardProtocol subclass defining key format for
                request/result exchange. Defaults to AgentRunProtocol.
            scope: Optional BlackboardScope to determine communication scope. If None, defaults to ``protocol.scope`` and then ``BlackboardScope.AGENT``.
            namespace: Protocol namespace for disambiguation when the child
                agent has multiple capabilities using the same scope.
                Must match the namespace the target capability receives in
                its ``__init__``.

        Returns:
            AgentRun with status, output_data, and resource_usage

        Example:
            ```python
            async with session.context():
                handle = await AgentHandle.from_agent_id("agent_123")
                run = await handle.run({"query": "analyze code"}, namespace="analysis", timeout=60)
                print(run.output_data)
            ```
        """
        from .blackboard.protocol import AgentRunProtocol
        from .scopes import BlackboardScope, get_scope_prefix
        proto = protocol or AgentRunProtocol

        # Determine the blackboard scope from the protocol.
        # Agent-scoped protocols write to the child's agent-level scope.
        # Colony/session-scoped protocols write to the shared scope.
        scope = scope or proto.scope if hasattr(proto, 'scope') else BlackboardScope.AGENT
        scope_id = get_scope_prefix(scope, self.child_agent_id, namespace=namespace)

        blackboard = await self._get_child_blackboard(scope_id=scope_id)

        # Get session context if available
        from .sessions.context import get_current_session_id
        from ..system import get_session_manager

        # Get session context
        effective_session_id = session_id or get_current_session_id()

        # Create AgentRun via SessionManager if we have a session
        run: AgentRun | None = None
        session_manager_handle = None

        run_id: str | None = None
        if session_id:
            try:
                session_manager_handle = get_session_manager(app_name=self._app_name)

                run_result = await session_manager_handle.create_run(
                    session_id=effective_session_id,
                    agent_id=self.child_agent_id,
                    input_data=input_data,
                    config=config,
                    timeout=timeout,
                    track_events=track_events,
                    run_id=run_id,
                )

                # Handle both Pydantic model and dict from DeploymentHandle
                if isinstance(run_result, dict):
                    run = None
                    run_id = run_result.get("run_id")
                else:
                    run = run_result
                    run_id = run.run_id

                if run_id:
                    # Mark as running
                    await session_manager_handle.update_run_status(
                        run_id=run_id,
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
        result_key = proto.result_key(request_id)
        result_value: dict[str, Any] | None = None

        # Send request (session_id auto-added by blackboard.write from context)
        request_key = proto.request_key(request_id)
        await blackboard.write(
            key=request_key,
            value={
                "input": input_data,
                "request_id": request_id,
                "session_id": effective_session_id,  # Include in value for receiver to extract
                "sender": sender_id,
                "run_id": run_id,
            },
            created_by=sender_id,
        )

        # Check if result already exists (race condition handling)
        existing = await blackboard.read(result_key)
        if existing is not None:
            if run_id and session_manager_handle:
                await session_manager_handle.update_run_status(
                    run_id=run_id,
                    status=RunStatus.COMPLETED,
                    output_data=existing,
                )
            if run:
                run.status = RunStatus.COMPLETED
                run.output_data = existing
            else:
                # Create ephemeral run for return value
                run = AgentRun(
                    session_id=effective_session_id,
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
        if run_id and session_manager_handle:
            await session_manager_handle.update_run_status(
                run_id=run_id,
                status=final_status,
                output_data=result_value,
                error=error_msg,
            )
        if run:
            run.status = final_status
            run.output_data = result_value
            run.error = error_msg
        else:
            # Create ephemeral run for return value
            run = AgentRun(
                session_id=effective_session_id,
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
        run_id: str | None = None,
        protocol: type[BlackboardProtocol] | None = None,
        scope: BlackboardScope | None = None,
        namespace: str = "",
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
            run_id: Explicit run ID (matches agent's metadata.run_id to avoid mismatch)
            protocol: BlackboardProtocol subclass defining key format. Defaults to AgentRunProtocol.
            scope: Optional BlackboardScope to determine communication scope. If None, defaults to ``protocol.scope`` and then ``BlackboardScope.AGENT``.
            namespace: Protocol namespace for disambiguation within the same
                blackboard scope. Must match the namespace the target capability
                declares in its ``input_patterns``.

        Yields:
            AgentRunEvent for each event during execution

        Example:
            ```python
            async with session.context():
                handle = await AgentHandle.from_agent_id("agent_123")
                async for event in handle.run_streamed({"query": "analyze"}, namespace="analysis"):
                    print(f"{event.event_type}: {event.data}")
                    if event.event_type == "completed":
                        break
            ```
        """
        from .sessions.context import get_current_session_id
        from ..system import get_session_manager
        from .blackboard.protocol import AgentRunProtocol
        from .scopes import BlackboardScope, get_scope_prefix

        proto = protocol or AgentRunProtocol

        # Determine the blackboard scope from the protocol.
        scope = scope or proto.scope if hasattr(proto, 'scope') else BlackboardScope.AGENT
        scope_id = get_scope_prefix(scope, self.child_agent_id, namespace=namespace)

        blackboard = await self._get_child_blackboard(scope_id=scope_id)

        # Get session context if available
        effective_session_id = session_id or get_current_session_id()

        # Create AgentRun via SessionManager if we have a session
        run: AgentRun | None = None
        session_manager_handle = None

        if session_id:
            try:
                session_manager_handle = get_session_manager(app_name=self._app_name)

                if run_id:
                    # Caller pre-created the run — just mark it as running
                    await session_manager_handle.update_run_status(
                        run_id=run_id,
                        status=RunStatus.RUNNING,
                    )
                else:
                    # Create a new run
                    run_result = await session_manager_handle.create_run(
                        session_id=session_id,
                        agent_id=self.child_agent_id,
                        input_data=input_data,
                        config=config,
                        timeout=timeout,
                        track_events=track_events,
                        run_id=None,
                    )

                    # Handle both Pydantic model and dict from DeploymentHandle
                    if isinstance(run_result, dict):
                        run = None
                        run_id = run_result.get("run_id")
                    else:
                        run = run_result
                        run_id = run.run_id

                    if run_id:
                        # Mark as running
                        await session_manager_handle.update_run_status(
                            run_id=run_id,
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
        request_key = proto.request_key(request_id)
        await blackboard.write(
            key=request_key,
            value={
                "input": input_data,
                "request_id": request_id,
                "session_id": effective_session_id,  # Include in value for receiver to extract
                "sender": sender_id,
                "streaming": True,
                "run_id": run_id,
            },
            created_by=sender_id,
        )

        # Stream events
        start_time = time.time()
        event_pattern = proto.event_pattern(request_id)
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

                    if run_id and session_manager_handle and track_events:
                        try:
                            await session_manager_handle.add_run_event(
                                run_id=run_id,
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
                if run_id and session_manager_handle and track_events:
                    try:
                        await session_manager_handle.add_run_event(
                            run_id=run_id,
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
        if run_id and session_manager_handle:
            try:
                await session_manager_handle.update_run_status(
                    run_id=run_id,
                    status=final_status,
                    output_data=result_data,
                    error=error_msg,
                )
            except Exception as e:
                logger.warning(f"Failed to update run status: {e}")

    # =========================================================================
    # Agent Control
    # =========================================================================

    async def stop(self, reason: str = "stop_requested") -> None:
        """Request the target agent to stop."""
        from ..system import get_agent_system

        agent_system = get_agent_system(app_name=self._app_name)
        await agent_system.stop_agent(self.child_agent_id, reason=reason)


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



def check_isolation(method):
    """Decorator to check that the execution context matches the agent's tenant/colony identity."""
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        ctx = serving.require_execution_context()
        if ctx.tenant_id != self.syscontext.tenant_id or ctx.colony_id != self.syscontext.colony_id:
            raise RuntimeError(
                f"Isolation context mismatch: agent={self.syscontext.tenant_id}/{self.syscontext.colony_id}, "
                f"context={ctx.tenant_id}/{ctx.colony_id}"
            )
        return await method(self, *args, **kwargs)
    return wrapper


@tracing(
    publish_key=lambda self: self.agent_id,
    subscribe_key=lambda self: self.agent_id
)
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
    agent_type: str = Field(default="general", description="Type of agent: specialized, general, service, supervisor, service.memory_management")
    state: AgentState = Field(default=AgentState.INITIALIZED)

    syscontext: serving.ExecutionContext = Field(
        default_factory=serving.require_execution_context,
        description="Execution context for tenant and colony isolation and user/kernel protection"
    )

    # Optional page binding
    bound_pages: list[str] = Field(default_factory=list)

    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

    page_storage: PageStorage | None = Field(default=None)

    # TODO: Move resource_requirements to AgentManagerBase.
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

    capability_blueprints: list[AgentCapabilityBlueprint] = Field(default_factory=list)

    # Memory configuration
    enable_memory_hierarchy: bool = Field(
        default=True,
        description="Whether to auto-initialize the default memory hierarchy"
    )
    memory_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for memory hierarchy (passed to create_default_memory_hierarchy)"
    )

    action_policy_blueprint: ActionPolicyBlueprint | None = None
    action_policy_state: ActionPolicyExecutionState | None = Field(default=None)
    action_policy: ActionPolicy | None = None

    child_agents: dict[str, str] = Field(default_factory=dict)  # role -> agent_id

    # Private attributes (not serialized, use PrivateAttr for Pydantic v2)
    _capabilities: dict[str, AgentCapability] = PrivateAttr(default_factory=dict)
    _running: bool = PrivateAttr(default=False)
    _stop_requested: bool = PrivateAttr(default=False)
    _stop_reason: str = PrivateAttr(default="")
    _suspend_requested: bool = PrivateAttr(default=False)
    _suspend_reason: str | None = PrivateAttr(default=None)
    _resumption_condition: ResumptionCondition | None = PrivateAttr(default=None)
    _manager: AgentManagerBase | None = PrivateAttr(default=None)
    _tracing_facility: TracingFacility | None = PrivateAttr(default=None)

    # Tracing config (set externally, e.g., by AgentSystemConfig)
    _tracing_config: TracingConfig | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def tenant_id(self) -> str:
        return self.syscontext.tenant_id

    @property
    def colony_id(self) -> str:
        return self.syscontext.colony_id

    @classmethod
    def bind(cls, **kwargs) -> AgentBlueprint:
        """Create an AgentBlueprint from this class and constructor kwargs.

        Validates kwargs against model_fields at bind time.
        Use .remote_instance() to spawn on a remote deployment.
        """
        bp = AgentBlueprint(cls, kwargs)
        bp.validate_serializable()
        return bp

    def get_action_group_description(self) -> str:
        return (
            f"Agent-level actions for {self.__class__.__name__} ({self.agent_type}). "
            "Direct agent actions outside of capabilities."
        )

    def to_registration_info(self) -> AgentRegistrationInfo:
        """Create a lightweight, serializable registration info for the agent system.

        This extracts only the fields needed for agent discovery and resource
        tracking, avoiding non-serializable internals (capabilities, action
        policies, weakrefs).
        """
        return AgentRegistrationInfo(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            state=self.state,
            tenant_id=self.syscontext.tenant_id,
            colony_id=self.syscontext.colony_id,
            bound_pages=self.bound_pages,
            metadata=self.metadata,
            capability_names=self.get_capability_names(),
            resource_requirements=self.resource_requirements,
        )

    def set_manager(self, manager: AgentManagerBase) -> None:
        """Attach agent manager reference for delegation.

        Args:
            manager: AgentManagerBase instance
        """
        self._manager = manager

    @property
    def hooks(self) -> HookRegistry:
        """Get the hook registry for this agent.

        The registry is lazily created on first access.

        Returns:
            HookRegistry for registering hooks
        """
        return get_hook_registry(self.agent_id)

    def add_capability_blueprints(
        self,
        blueprints: list[AgentCapabilityBlueprint],
    ) -> None:
        """Add capability blueprints to be instantiated during `_create_action_policy`.

        Typically called in an agent subclass `initialize()` method
        before calling `super().initialize()`. If called after initialization,
        the new blueprints will be added but won't be materialized into agent capabilities. You can call `agent.action_policy.use_capability_blueprints()` to materialize blueprints dynamically after agent initialization.
        No need to call `action_policy.use_agent_capabilities()` if the blueprints are added before initialization, as they will be used automatically during the initial action policy creation.

        Args:
            blueprints: The capability blueprints to add
        """
        for bp in blueprints:
            if not any(b.key == bp.key for b in self.capability_blueprints):
                self.capability_blueprints.append(bp)

    def add_capability(
        self,
        capability: AgentCapability,
        *,
        include_actions: list[str] | None = None,
        exclude_actions: list[str] | None = None,
        events_only: bool = False,
    ) -> None:
        """Add a capability to the agent.
        This capability must already be
        initialized with the agent as its owner. This capability must still
        be passed to `agent.action_policy.use_agent_capabilities()` to be included in the action policy for planning.

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

        self._capabilities[capability.capability_key] = capability

    def remove_capability(self, capability_name: str) -> AgentCapability | None:
        """Remove a capability from the agent.

        Also removes any hooks registered by the capability.

        Args:
            capability_name: Name of the capability to remove

        Returns:
            The removed capability, or None if not found
        """
        capability = self._capabilities.pop(capability_name, None)
        if capability is not None:
            capability.uninstall_hooks()
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

    def get_capabilities_by_class(self, capability_class: type[AgentCapability]) -> list[AgentCapability]:
        """Get all capabilities that are instances of the given class.

        Unlike get_capability_by_type() which returns the first match,
        this returns all matching instances — useful for multi-instance
        capabilities like MemoryCapability with different capability_keys.
        """
        return [cap for cap in self._capabilities.values() if isinstance(cap, capability_class)]

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

    def get_capabilities(self) -> list[AgentCapability]:
        """Get agent's capabilities for discovery.

        Returns:
            List of capabilities
        """
        return list(self._capabilities.values())

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
            from ..system import get_tool_manager

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
            from ..system import get_tool_manager

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

    async def _instantiate_capability_blueprints(self, blueprints: list[AgentCapabilityBlueprint]) -> None:
        """Instantiate capabilities from blueprints and add to agent."""
        # Phase 1: Instantiate capability blueprints and add to agent.
        # Do not initialize yet — capabilities may discover each other during
        # initialization and we want them all present first.
        capability_instances = []
        for bp in blueprints:
            if not self.has_capability(bp.key):
                # local_instance creates a new instance of the capability for this agent, allowing the same blueprint to be reused across multiple agents with different internal state. It also passes other capability-specific parameters defined in the blueprint kwargs.
                capability_instance = bp.local_instance(self)
                capability_instance._capability_key = bp.key
                self.add_capability(
                    capability_instance,
                    include_actions=bp.include_actions,
                    exclude_actions=bp.exclude_actions,
                    events_only=bp.events_only,
                )
                capability_instances.append(capability_instance)
            else:
                logger.warning(f"Agent {self.agent_id} already has capability {bp.key}, skipping blueprint instantiation.")

        # Phase 2: Initialize all capabilities.
        # Note that a capability's initialize() may add pre-initialized
        # sub-capabilities (e.g., AgentPoolCapability, PageGraphCapability).
        # So, do not use self._capabilities.values() directly in the loop, as it may mutate during iteration. Instead, snapshot the values into a list first.
        for capability_instance in capability_instances:
            await capability_instance.initialize()

    async def initialize(self) -> None:
        """Initialize agent (called after creation).

        Override in subclasses to perform initialization logic.
        """
        from ..system import get_vcm

        # Check if this is a resumed agent
        if self.metadata.resuming_from_suspension:
            await self._restore_from_suspension()
        else:
            self.state = AgentState.INITIALIZED

            # Track children - for event-driven coordination
            # TODO: How to restore this on resumption from suspension?
            self.child_agents: dict[str, str] = {}  # role -> agent_id

        # Initialize tracing FIRST so that _current_span is set to AGENT span
        # before any deployment calls (page storage, memory hierarchy, etc.).
        # DeploymentHandle.call_method() reads _current_span to propagate
        # parent_span_id to remote hooks.
        self._init_tracing_config()

        # Add AgentTracingFacility if tracing is enabled
        if self._tracing_config and self._tracing_config.enabled:
            try:
                from .observability import AgentTracingFacility
                self._tracing_facility = AgentTracingFacility(agent=self, config=self._tracing_config)
                await self._tracing_facility.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize tracing for agent {self.agent_id}: {e}")

        # Reconstruct or create PageStorage
        vcm_handle = get_vcm()
        config: PageStorageConfig | None = await vcm_handle.get_page_storage_config()
        if not config:
            raise ValueError("Missing PageStorageConfig in VCM")

        self.page_storage = PageStorage(
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            s3_bucket=config.s3_bucket,
        )
        await self.page_storage.initialize()

        # Initialize memory hierarchy if enabled
        if self.enable_memory_hierarchy:
            await self.initialize_memory_hierarchy(**self.memory_config)

        await self._create_action_policy()

    def _init_tracing_config(self) -> None:
        """Initialize tracing config from environment variables."""
        tracing_enabled = os.environ.get("TRACING_ENABLED", "").lower() in ("true", "1", "yes")
        if tracing_enabled:
            self._tracing_config = TracingConfig(
                enabled=True,
                kafka_bootstrap=os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"),
                kafka_topic=os.environ.get("KAFKA_SPANS_TOPIC", "colony.spans"),
            )

    async def emit_lifecycle_stop_event(self, stop_reason: str, error_msg: str | None = None, iteration: int | None = None) -> None:
        """Emit a lifecycle stop event for this agent."""
        # Finalize tracing spans with actual agent state
        if self._tracing_facility:
            from .observability import AgentTracingFacility
            tracing_facility: AgentTracingFacility = self._tracing_facility
            tracing_facility.emit_lifecycle_event(stop_reason, {
                "error": error_msg,
                "iterations": iteration,
                "agent_state": str(self.state),
            })
            await tracing_facility.shutdown()  # Shutdown because agent is stopped. Ensure spans are flushed

    async def _create_action_policy(self) -> None:
        if self.action_policy:
            return  # Already set

        # Phase 1: Instantiate capability blueprints and add to agent.
        await self._instantiate_capability_blueprints(self.capability_blueprints)

        # Phase 1b: Auto-create REPLCapability if enabled and not already present.
        if self.metadata.enable_repl:
            from .patterns.actions.repl import REPLCapability
            if not any(
                isinstance(cap, REPLCapability) for cap in self._capabilities.values()
            ):
                repl_cap = REPLCapability(
                    agent=self,
                    capability_key="repl",
                    allowed_imports=None,     # TODO: Make configurable
                    restrict_builtins=True,   # TODO: Make configurable
                    max_execution_time=self.metadata.repl_max_execution_time,
                )
                await repl_cap.initialize()
                self.add_capability(
                    repl_cap,
                    events_only=False,
                )

        # Phase 2: Create action policy
        # NOTE: Capabilities are NOT passed as action_providers here.
        # They are registered via use_agent_capabilities() in Phase 4,
        # which is the single authoritative path for capability → dispatcher.
        # action_providers is reserved for non-capability action sources
        # (standalone functions, external objects).
        if not self.action_policy_blueprint:
            logger.warning("Agent does not have action_policy or action_policy_blueprint defined. We will create a default ActionPolicy, but consider defining a custom one for better performance and capabilities.")
            from .patterns.actions import create_default_action_policy
            self.action_policy = await create_default_action_policy(
                agent=self,
                action_map={},  # Action executors are discovered from capabilities
                max_iterations=self.metadata.max_iterations,
                # TODO: Allow configuring IO schemas
                # io=ActionPolicyIO(
                #     inputs={"context": QueryContext, "queries": list},
                #     outputs={"analysis": ScopeAwareResult, "next_queries": list},
                # ),
                **self.metadata.action_policy_config,
            )
        else:
            self.action_policy = self.action_policy_blueprint.local_instance(
                self,
                # TODO: Allow configuring IO schemas
                # io=ActionPolicyIO(
                #     inputs={"context": QueryContext, "queries": list},
                #     outputs={"analysis": ScopeAwareResult, "next_queries": list},
                # ),
                **self.metadata.action_policy_config,
            )

        # Phase 3: Mark ALL capabilities as "used" by the action policy.
        # This includes both blueprint-instantiated capabilities AND
        # capabilities added directly via add_capability() during initialize().
        self.action_policy.use_agent_capabilities(list(self._capabilities.keys()))

        logger.info(f"________ Created action policy {self.action_policy.__class__.__name__} for agent {self.agent_id} with capabilities: {self.get_capability_names()}")

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

    @check_isolation
    async def load_page_graph(self, cached: bool = True) -> nx.DiGraph:
        """Load page graph dynamically from PageStorage.

        Uses page_storage to access PageStorage.
        This allows the agent to load the page graph when needed, rather than
        passing the entire graph in metadata.
        """
        if not self.page_storage:
            return nx.DiGraph()
        return await self.page_storage.load_page_graph(cached)

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
    async def stop(self, reason: str = "completed") -> None:
        """Stop agent execution.

        This method is @hookable, so capabilities can register BEFORE hooks
        to flush memories, AFTER hooks for cleanup, etc.

        Args:
            reason: Why the agent is stopping. Stored in ``_stop_reason``
                for tracing and run status updates. Common values:
                ``"policy_completed"``, ``"idle_timeout"``,
                ``"max_iterations"``, ``"error"``, ``"cancelled"``,
                ``"stop_requested"``.
        """
        self._stop_requested = True
        self._stop_reason = reason
        self.state = AgentState.STOPPED
        self._running = False
        self.action_policy_state = None
        # TODO: Should notify parent agent if any?

    @hookable
    async def suspend(
        self,
        reason: str = "",
        resumption_condition: ResumptionCondition | None = None,
    ) -> None:
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
            reason: Human-readable reason for suspension.
            resumption_condition: Structured condition that must be met before
                the agent can resume. If None, defaults to IMMEDIATE (resume ASAP).
                The system's resource monitor loop evaluates this condition
                before attempting resumption.

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
        from .models import ResumptionCondition, ResumptionConditionType
        # Set flag for manager to detect
        self._suspend_requested = True
        self._suspend_reason = reason
        self._resumption_condition = resumption_condition or ResumptionCondition(
            condition_type=ResumptionConditionType.IMMEDIATE
        )

        logger.info(f"Agent {self.agent_id} requested suspension: {reason}")

    def _model_to_str(self, model: BaseModel | None, trunc: int = 1000) -> tuple[str, str]:
        """Helper to convert a Pydantic model to a pretty string for logging."""
        if model is None:
            return "None", "full"
        result_str = model.model_dump_json(indent=3)
        return (result_str[:trunc], "truncated") if len(result_str) > trunc else (result_str, "full" )

    @hookable
    @check_isolation
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
        if self.state not in (AgentState.RUNNING, AgentState.IDLE):
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

            logger.warning(
                f"\n"
                f"  ┌──────────────────────────────────────────────┐\n"
                f"  │  ▶ RUN_STEP: calling execute_iteration       │\n"
                f"  │  agent={self.agent_id:<40}│\n"
                f"  └──────────────────────────────────────────────┘"
            )
            iteration_result = await self.action_policy.execute_iteration(self.action_policy_state)
            action_str, trunc = self._model_to_str(iteration_result.action_executed)
            logger.warning(
                f"  ◀ RUN_STEP returned: success={iteration_result.success} "
                f"completed={iteration_result.policy_completed} "
                f"idle={iteration_result.idle} "
                f"action={action_str} ({trunc})"
            )

            # Policy-driven state transitions
            if iteration_result.policy_completed:
                self.state = AgentState.STOPPED
            elif iteration_result.idle:
                if self.state != AgentState.IDLE:
                    self._idle_since = time.time()
                self.state = AgentState.IDLE
            elif self.state == AgentState.IDLE:
                # Policy returned work while we were IDLE — wake up
                self.state = AgentState.RUNNING
                self._idle_since = None

            if self.state == AgentState.STOPPED:
                logger.info(f"Agent {self.agent_id} has entered STOPPED state")
                # Stop agent gracefully
                await self.stop(reason="policy_completed")

            if self.state == AgentState.IDLE:
                timeout = self.metadata.idle_timeout
                if timeout is not None and hasattr(self, '_idle_since'):
                    if (time.time() - self._idle_since) > timeout:
                        logger.info(
                            f"Agent {self.agent_id} idle timeout ({timeout}s) reached, stopping"
                        )
                        await self.stop(reason="idle_timeout")
                        return
                await self.on_idle()

            if iteration_result.error_context:
                logger.error(
                    f"Agent {self.agent_id} encountered error in run_step: "
                    f"{iteration_result.error_context.error_details}"
                )
                # Optionally, could set state to FAILED here
                # self.state = AgentState.FAILED

            # Sleep interval: longer when IDLE to avoid busy-looping
            sleep_interval = (
                self.metadata.idle_sleep_interval
                if self.state == AgentState.IDLE
                else 0.1
            )
            await asyncio.sleep(sleep_interval)

    @hookable
    async def on_idle(self) -> None:
        """Called each idle cycle. Hook for background work.

        Override or register AROUND/AFTER hooks for memory consolidation,
        intrinsic goals, health checks, or other idle-time activities.
        """
        pass

    # === Context Access (Delegated to manager) ===

    @hookable
    @check_isolation
    async def request_page(self, page_id: str, priority: int = 0) -> None:
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
            page_id=page_id,
            agent_id=self.agent_id,
            priority=priority,
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

        logger.warning(
            f"\n"
            f"              ╔══════════════════════════════════════╗\n"
            f"              ║  📡 agent.infer({self.agent_id}) → manager ║\n"
            f"              ║  prompt_len={len(prompt or ''):<24} ║\n"
            f"              ║  pages={str(context_page_ids)[:30]:<30}║\n"
            f"              ╚══════════════════════════════════════╝"
        )
        result = await self._manager.agent_infer(
            agent_id=self.agent_id,
            context_page_ids=context_page_ids,
            prompt=prompt,
            **kwargs
        )
        resp_text = result.generated_text if hasattr(result, "generated_text") else ""
        logger.warning(
            f"              📡 agent.infer() returned — len={len(resp_text)}"
        )
        logger.debug(f"__________ Agent {self.agent_id} inference request:\n\tPrompt:{prompt}\n\tResult:{resp_text}")

        # Report token usage to session manager
        await self._report_llm_usage(result)

        return result

    async def _report_llm_usage(self, response: InferenceResponse) -> None:
        """Report LLM token usage from an inference response to the session manager."""
        # run_id comes from action_policy_state (propagated from blackboard request)
        # or falls back to metadata.run_id (set at spawn time)
        run_id = None
        if self.action_policy_state is not None:
            run_id = self.action_policy_state.custom.get("current_run_id")
        if not run_id or run_id == "default":
            run_id = self.metadata.run_id
        if not run_id or run_id == "default":
            logger.warning("_report_llm_usage: no run_id available, skipping")
            return

        # Handle both Pydantic model and dict (Ray RPC may return dicts)
        if isinstance(response, dict):
            meta = response.get("metadata") or {}
        else:
            meta = getattr(response, "metadata", None) or {}
        if isinstance(meta, str):
            logger.warning("_report_llm_usage: metadata is a string, skipping")
            return

        input_tokens = meta.get("input_tokens", 0)
        output_tokens = meta.get("output_tokens", 0)
        logger.warning(
            "_report_llm_usage: run_id=%s input=%d output=%d cost=%.4f",
            run_id, input_tokens, output_tokens, meta.get("cost_usd", 0.0),
        )
        if input_tokens == 0 and output_tokens == 0:
            return

        try:
            from ..system import get_session_manager
            handle = get_session_manager(app_name=serving.get_my_app_name())
            result = await handle.update_run_resources(
                run_id=run_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_calls=1,
                cost_usd=meta.get("cost_usd", 0.0),
                cache_read_tokens=meta.get("cache_read_tokens", 0),
                cache_write_tokens=meta.get("cache_write_tokens", 0),
                agent_id=self.agent_id,
            )
            logger.warning(
                "_report_llm_usage: update_run_resources returned %s for run %s",
                result, run_id,
            )
        except Exception as e:
            logger.warning("Failed to report LLM usage for run %s: %s", run_id, e)

    # === Blackboard (Delegated to manager) ===

    async def get_agent_level_blackboard(
        self,
        *,
        namespace: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard scoped to this agent for private state."""
        return await self._manager.get_agent_level_blackboard(
            self,
            namespace=namespace,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    async def get_session_level_blackboard(
        self,
        *,
        namespace: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard scoped to this session for state shared across agents in the same session."""
        return await self._manager.get_session_level_blackboard(
            namespace=namespace,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    async def get_colony_level_blackboard(
        self,
        *,
        namespace: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard scoped to this colony for state shared across agents in the same colony.
        
        Args:
            namespace: Optional namespace for the blackboard within the colony scope (e.g., "coordination", "resources"). If None, defaults to the colony-level scope.
            backend_type: Backend type for the blackboard (e.g., "redis")
            enable_events: Whether to enable events on the blackboard

        Returns:
            Blackboard instance
        """
        return await self._manager.get_colony_level_blackboard(
            namespace=namespace,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    async def get_blackboard(
        self,
        scope_id: str | None = None,
        backend_type: str = "redis",
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for reading/writing shared state.

        Args:
            scope_id: Scope identifier (defaults to `agent_id` for shared scope)
            backend_type: Backend type for the blackboard (e.g., "redis")
            enable_events: Whether to enable events on the blackboard

        Returns:
            Blackboard instance

        Example:
            ```python
            # Get shared blackboard with parent
            parent_id = self.metadata.parent_id
            board = await self.get_blackboard(scope_id=parent_id)

            # Write results
            await board.write("analysis_complete", my_results)

            # Parent reads
            result = await board.read("analysis_complete")
            ```
        """
        if not self._manager:
            raise RuntimeError(f"Agent {self.agent_id} not attached to manager")

        # Default scope_id to agent_id for shared scope
        if scope_id is None:
            from .scopes import ScopeUtils
            scope_id = ScopeUtils.get_agent_level_scope(self)

        return await self._manager.get_blackboard(
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
            self.metadata.performance_metrics.update(metrics)
            self.metadata.performance_last_updated = time.time()

        except Exception as e:
            raise

    # === Agent Hierarchy Management ===

    async def stop_child_agent(self, agent_id: str, reason: str = "stop_requested") -> None:
        """Stop a child agent.

        Args:
            agent_id: Agent identifier
            reason: Reason for stopping the agent
        """
        from ..system import get_agent_system

        agent_system_handle = get_agent_system()
        await agent_system_handle.stop_agent(agent_id, reason=reason)

    async def spawn_child_agents(
        self,
        blueprints: list[AgentBlueprint],
        *,
        requirements: "LLMClientRequirements" | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        return_handles: bool = False,
    ) -> list[str] | list["AgentHandle"]:
        """Spawn agents via agent system. This is intended to be called by this agent
        to spawn other agents. This method can be overridden by subclasses to add custom
        logic before/after spawning.

        Setting return_handles=True is the preferred method for spawning child agents when you need
        to interact with them via their capabilities.

        Args:
            blueprints: List of AgentBlueprint defining agents to spawn.
                Each blueprint carries its own metadata (session_id, run_id,
                max_iterations, role) and capability_blueprints.
                If metadata.role is set, the child is tracked in self.child_agents[role].
                capability_blueprints are used to infer capability types for AgentHandle
                when return_handles=True.
            requirements: Optional LLMClientRequirements to apply to all agents
            soft_affinity: If True, allows spawning on replicas without all pages (will page fault)
            suspend_agents: If True, replica may suspend existing agents to make room
            return_handles: If True, return AgentHandle objects instead of agent IDs.

        Returns:
            List of spawned agent IDs or AgentHandles

        Example:
            ```python
            handle = (await self.spawn_child_agents(
                blueprints=[GroundingAgent.bind(
                    metadata=AgentMetadata(session_id="s1", role="grounding"),
                    capability_blueprints=[GroundingCapability.bind()],
                )],
                return_handles=True,
            ))[0]

            grounding = handle.get_capability_by_type(GroundingCapability)
            await grounding.stream_events_to_queue(self.get_event_queue())
            ```
        """
        from ..system import spawn_agents

        child_ids: list[str] = await spawn_agents(
            blueprints=blueprints,
            requirements=requirements,
            soft_affinity=soft_affinity,
            suspend_agents=suspend_agents,
        )

        # Track children in self.child_agents using role from blueprint metadata
        for i, child_id in enumerate(child_ids):
            role = blueprints[i].metadata.role
            if role:
                if role in self.child_agents:
                    raise ValueError(f"Duplicate child role '{role}' in spawn_child_agents - roles must be unique for tracking")
                self.child_agents[role] = child_id
                logger.info(f"Agent {self.agent_id} spawned child {role} ({child_id})")

        if not return_handles:
            return child_ids

        # Create AgentHandles — infer capability types from each blueprint's
        # capability_blueprints so callers don't need to repeat them.
        handles: list[AgentHandle] = []
        for i, child_id in enumerate(child_ids):
            cap_types = [cb.cls for cb in blueprints[i].capability_blueprints] or None
            handle = AgentHandle(
                child_agent_id=child_id,
                owner=self,
                capability_types=cap_types,
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
        from .patterns.memory import MemoryCapability
        from .scopes import MemoryScope

        # Find STM capability
        stm_scope = MemoryScope.agent_stm(self)

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
        ### from .patterns.memory import MemoryCapability
        ### from .scopes import MemoryScope
        ### episodic_scope = MemoryScope.agent_ltm_episodic(self)

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

    def __init__(self, deployment_config: LLMDeploymentConfig | None = None):
        """Initialize agent manager.

        Note: This is a mixin, so it should be called from the deployment's __init__.

        Args:
            deployment_config: Optional deployment configuration with resource limits
        """
        # Agent management
        self._agents: dict[str, Agent] = {}
        self._agent_tasks: dict[str, asyncio.Task] = {}
        self._agent_lock = asyncio.Lock()

        # Resource limits (from LLMDeploymentConfig)
        if deployment_config:
            self.max_agents = deployment_config.max_agents_per_replica
            self.max_cpu_cores = deployment_config.max_cpu_cores_per_replica
            self.max_memory_mb = deployment_config.max_memory_mb_per_replica
            self.max_gpu_cores = deployment_config.max_gpu_cores_per_replica
            self.max_gpu_memory_mb = deployment_config.max_gpu_memory_mb_per_replica
        else:
            # Defaults if no config provided
            self.max_agents = 20
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
        """Initialize self-contained state. Override in subclasses if needed."""
        pass

    async def discover_handles(self) -> None:
        """Discover sibling deployment handles. Call from @on_app_ready."""
        from ..system import (
            get_agent_system,
            get_llm_cluster,
            get_tool_manager,
            get_vcm,
        )
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
        agent_blueprint: AgentBlueprint,
        *,
        suspend_agents: bool = False,
    ) -> str:
        """Start a new agent from a blueprint.

        The blueprint must contain agent_id (set by spawn_from_blueprint).

        Args:
            agent_blueprint: AgentBlueprint with class and all constructor args.
                max_iterations is read from the agent's metadata after construction.
            suspend_agents: If True and ResourceExhausted, suspend existing agents to make room

        Returns:
            agent_id

        Raises:
            ResourceExhausted: If replica has insufficient resources and suspend_agents=False
        """
        async with self._agent_lock:
            agent_id = agent_blueprint.agent_id
            resource_requirements: "LLMClientRequirements" | None = agent_blueprint.resource_requirements
            bound_pages = agent_blueprint.bound_pages or None

            if agent_id is None:
                agent_id = f"agent-{uuid.uuid4().hex[:8]}"

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

            if any([needed_cpu > 0, needed_memory > 0, needed_gpu > 0, needed_gpu_memory > 0]):
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
                if any([
                    new_cpu > self.max_cpu_cores,
                    new_memory > self.max_memory_mb,
                    new_gpu > self.max_gpu_cores,
                    new_gpu_memory > self.max_gpu_memory_mb,
                ]):
                    raise ResourceExhausted(
                        f"CPU/Memory/GPU/GPU memory capacity exceeded even after suspension:\n"
                        f"{new_cpu:.2f}/{self.max_cpu_cores} cores\n"
                        f"{new_memory}/{self.max_memory_mb} MB\n"
                        f"{new_gpu:.2f}/{self.max_gpu_cores} cores\n"
                        f"{new_gpu_memory}/{self.max_gpu_memory_mb} MB"
                    )

            # Create agent instance from blueprint
            agent: Agent = agent_blueprint.local_instance()

            # Attach manager reference for delegation
            agent.set_manager(self)

            # Initialize agent
            await agent.initialize()

            # Store agent
            if agent_id in self._agents:
                raise ValueError(f"Agent ID collision: {agent_id} already exists")
            self._agents[agent_id] = agent

            # TRACK RESOURCE USAGE
            self._used_cpu_cores += resource_requirements.cpu_cores
            self._used_memory_mb += resource_requirements.memory_mb
            self._used_gpu_cores += resource_requirements.gpu_cores
            self._used_gpu_memory_mb += resource_requirements.gpu_memory_mb

            # Start agent loop — max_iterations comes from the agent's metadata,
            # which was set by spawn_from_blueprint before reaching here.
            task = asyncio.create_task(
                self._run_agent_loop(agent, max_iterations=agent.metadata.max_iterations)
            )
            self._agent_tasks[agent_id] = task

            # Register with agent system
            if self._agent_system_handle:
                try:
                    deployment_replica_id = self._get_deployment_replica_id()
                    await self._agent_system_handle.register_agent(
                        agent_info=agent.to_registration_info(),
                        deployment_replica_id=deployment_replica_id,
                        deployment_name=serving.get_my_deployment_name(),
                    )
                except Exception as e:
                    logger.error(f"Failed to register agent {agent_id} with system: {e}")

            logger.info(
                f"Started agent {agent_id} (type={agent_blueprint.cls.__name__}, bound_pages={bound_pages}, "
                f"cpu={resource_requirements.cpu_cores}, mem={resource_requirements.memory_mb}MB)"
            )

            return agent_id

    @serving.endpoint(router_class=AgentAffinityRouter)
    async def stop_agent(self, agent_id: str, graceful: bool = True, reason: str = "stop_requested") -> None:
        """Stop an agent.

        Args:
            agent_id: Agent identifier
            graceful: If True, wait for graceful shutdown with timeout
            reason: Reason for stopping the agent
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
                await agent.stop(reason=reason)
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

                # Set resumption condition: use agent's own condition (from
                # self-suspension) or derive from reason (manager-initiated)
                if agent._resumption_condition is not None:
                    suspension_state.resumption_condition = agent._resumption_condition
                else:
                    # Manager-initiated suspension (resource pressure, capacity)
                    suspension_state.resumption_condition = ResumptionCondition(
                        condition_type=ResumptionConditionType.RESOURCE_AVAILABLE,
                        min_cpu_cores=agent.resource_requirements.cpu_cores,
                        min_memory_mb=agent.resource_requirements.memory_mb,
                        min_gpu_cores=agent.resource_requirements.gpu_cores,
                        min_gpu_memory_mb=agent.resource_requirements.gpu_memory_mb,
                    )

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

        # Resumption is handled by AgentSystemDeployment._resource_monitor_loop()
        # which evaluates the ResumptionCondition stored in the suspension state.

        return True

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

        1. `start_agent` is a `@serving.endpoint`
        2. When called, `__handle_request__` wraps it in with `execution_context(...)`:
        3. Inside that context, `start_agent` does `asyncio.create_task(self._run_agent_loop(...))`
        4. `create_task` snapshots the current `contextvars` — so the agent loop task does inherit `execution_context` values from the request that started the agent.

        So, the agent loop gets the right context automatically because Python's `asyncio.create_task` copies the active `contextvars.Context` at creation time.

        The snapshot is a copy, not a reference — so even after `__handle_request__` exits and its `execution_context` resets the `contextvars`, the agent loop's copy is unaffected. The agent loop retains the values for its entire lifetime.

        No explicit setting needed in `_run_agent_loop`.

        Args:
            agent: Agent instance
            max_iterations: Optional maximum number of iterations for the agent's action policy
        """
        iteration = 0
        error_msg: str | None = None
        try:
            await agent.start()

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
                    logger.warning(
                        f"\n"
                        f"╔══════════════════════════════════════════════════════════╗\n"
                        f"║  🔄 AGENT LOOP  iter={iteration:<4}                          ║\n"
                        f"║  agent={agent.agent_id:<48}║\n"
                        f"║  state={str(agent.state):<20} stop_req={agent._stop_requested!s:<14}║\n"
                        f"╚══════════════════════════════════════════════════════════╝"
                    )
                    await agent.run_step()
                    iteration += 1
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    logger.error(f"Error in agent {agent.agent_id} step: {error_msg}")
                    agent.state = AgentState.FAILED
                    break

        except asyncio.CancelledError:
            logger.info(f"Agent {agent.agent_id} loop cancelled")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Error in agent {agent.agent_id} loop: {error_msg}")
            agent.state = AgentState.FAILED
        finally:
            agent._running = False

            # Determine stop reason from agent state
            if agent.state == AgentState.FAILED:
                stop_reason = "error"
            elif agent._stop_reason:
                stop_reason = agent._stop_reason
            elif agent._stop_requested:
                stop_reason = "stop_requested"
            elif max_iterations is not None and iteration >= max_iterations:
                stop_reason = "max_iterations"
            else:
                stop_reason = "unknown"

            # Ensure stop() is called if the agent hasn't already stopped
            if agent.state not in (AgentState.STOPPED, AgentState.SUSPENDED):
                await agent.stop(reason=stop_reason)

            logger.warning(
                f"\n"
                f"╔══════════════════════════════════════════════════════════╗\n"
                f"║  🔄 AGENT LOOP  FINISHED iter={iteration:<4}                    ║\n"
                f"║  agent={agent.agent_id:<48}║\n"
                f"║  state={str(agent.state):<20} reason={stop_reason:<14}║\n"
                f"╚══════════════════════════════════════════════════════════╝"
            )

            # Finalize tracing spans with actual agent state
            await agent.emit_lifecycle_stop_event(stop_reason, error_msg, iteration)

            # Update AgentRun status in session system
            await self._finalize_agent_run(agent, stop_reason, error_msg)

    async def _finalize_agent_run(
        self,
        agent: Agent,
        stop_reason: str,
        error_msg: str | None = None,
    ) -> None:
        """Update the AgentRun status when the agent loop exits.

        Bridges the agent loop exit to the session system so that
        ``AgentRun.status`` reflects the actual outcome.
        """
        run_id = getattr(agent.metadata, "run_id", None)
        if not run_id or run_id == "default":
            return

        from .sessions.models import RunStatus
        status_map = {
            "completed": RunStatus.COMPLETED,
            "policy_completed": RunStatus.COMPLETED,
            "stop_requested": RunStatus.COMPLETED,
            "idle_timeout": RunStatus.COMPLETED,
            "max_iterations": RunStatus.TIMEOUT,
            "error": RunStatus.FAILED,
            "cancelled": RunStatus.CANCELLED,
        }
        status = status_map.get(stop_reason, RunStatus.FAILED)

        try:
            from ..system import get_session_manager
            session_mgr = get_session_manager()
            await session_mgr.update_run_status.remote(
                run_id=run_id,
                status=status,
                error=error_msg,
            )
        except Exception as e:
            logger.warning(
                f"Failed to finalize run {run_id} for agent {agent.agent_id}: {e}"
            )

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

    # These helper methods are unused since we moved to using AgentBlueprints with local_instance().

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
                "polymathera.colony.agents",  # e.g., agents.CodeAnalyzer
                "polymathera.agents",  # Legacy location
            ],
        )

        # Resolve action policy if provided
        action_policy_class: type[ActionPolicy] | None = self._resolve_class_from_identifier(
            class_id=action_policy_class_id,
            base_class=ActionPolicy,
            search_packages=[
                "polymathera.colony.agents",  # e.g., agents.SimpleActionPolicy
                "polymathera.agents",  # Legacy location
            ],
        )
        # Resolve capability classes if provided
        capability_classes: list[type[AgentCapability]] = []
        for cap_class_id in (capability_class_ids or []):
            cap_class = self._resolve_class_from_identifier(
                class_id=cap_class_id,
                base_class=AgentCapability,
                search_packages=[
                    "polymathera.colony.agents",  # e.g., agents.WebBrowsingCapability
                    "polymathera.agents",  # Legacy location
                ],
            )
            if cap_class:
                capability_classes.append(cap_class)

        # Create instance
        try:
            # Propagate syscontext from metadata to Agent fields
            # so page graph lookups use the correct keys.
            extra_fields: dict[str, Any] = {}
            ### if isinstance(metadata, AgentMetadata):
            ###     extra_fields["tenant_id"] = metadata.syscontext.tenant_id
            ###     if metadata.syscontext.colony_id:
            ###         extra_fields["colony_id"] = metadata.syscontext.colony_id
            agent = agent_class(
                agent_id=agent_id,
                agent_type=agent_class_id,
                capability_classes=capability_classes,
                action_policy_class=action_policy_class,
                bound_pages=bound_pages,
                metadata=metadata,
                resource_requirements=resource_requirements,
                **extra_fields,
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
            syscontext=self._agents[agent_id].syscontext,
            context_page_ids=context_page_ids or [],
            **kwargs
        )

        logger.warning(
            f"              📡 agent_infer({agent_id}): submitting to LLM cluster — "
            f"request_id={request.request_id}, pages={len(context_page_ids or [])}"
        )

        # Delegate to LLM cluster
        response = await self._llm_cluster_handle.infer(request)

        logger.warning(
            f"              📡 agent_infer({agent_id}): LLM cluster responded — "
            f"len={len(response.generated_text) if hasattr(response, 'generated_text') else '?'}"
        )
        return response

    async def agent_request_page(
        self,
        *,
        page_id: str,
        agent_id: str,
        priority: int = 0,
    ) -> None:
        """Delegate page loading request to VCM.

        Args:
            page_id: Virtual page identifier
            agent_id: Agent identifier
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
            agent_id=agent_id,
            priority=priority,
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
        agent = self._agents.get(agent_id)
        if not agent:
            logger.warning(f"agent_is_page_loaded: Agent {agent_id} not found")
            return False
        is_loaded = await self._vcm_handle.is_page_loaded(page_id)

        logger.debug(f"Agent {agent_id}: Page {page_id} loaded={is_loaded}")

        return is_loaded

    async def get_agent_level_blackboard(
        self,
        agent: Agent,
        *,
        namespace: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard scoped to this agent for private state."""
        from .scopes import BlackboardScope, get_scope_prefix
        scope_id = get_scope_prefix(BlackboardScope.AGENT, agent, namespace=namespace)

        return await self.get_blackboard(
            scope_id=scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    async def get_session_level_blackboard(
        self,
        *,
        namespace: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard scoped to this session for state shared across agents in the same session."""
        from .scopes import BlackboardScope, get_scope_prefix
        scope_id = get_scope_prefix(BlackboardScope.SESSION, namespace=namespace)

        return await self.get_blackboard(
            scope_id=scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    async def get_colony_level_blackboard(
        self,
        *,
        namespace: str | None = None,
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard scoped to this colony for state shared across agents in the same colony."""
        from .scopes import BlackboardScope, get_scope_prefix
        scope_id = get_scope_prefix(BlackboardScope.COLONY, namespace=namespace)

        return await self.get_blackboard(
            scope_id=scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )

    async def get_blackboard(
        self,
        scope_id: str = "default",
        backend_type: str | None = None,
        enable_events: bool = True,
    ) -> EnhancedBlackboard:
        """Get blackboard for shared state access.

        Args:
            scope_id: Scope identifier
            backend_type: Backend type ("memory", "distributed", "redis"). If None, the globally configured backend will be used.
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
            board = await self.get_blackboard(scope_id="group-1")

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
            scope_id=scope_id,
            backend_type=backend_type,
            enable_events=enable_events,
        )

        await blackboard.initialize()

        return blackboard

