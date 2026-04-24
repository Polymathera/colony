"""Iterative Reasoning Loop for Agents.

This module implements the planning → action → reflection → critique cycle
that allows agents to iteratively refine their approach instead of using
finite state machines with single-shot inference.

Philosophy:
- Agents should reason iteratively, not execute predetermined sequences
- Each iteration: PLAN → ACT → REFLECT → CRITIQUE → ADAPT
- LLM-driven: Control flow determined by reasoning, not hardcoded
- Composable: Planner, Dispatcher, Critic are independent policies
- Observable: Emit events at each stage for debugging/monitoring

This replaces FSM-based agent design with true iterative reasoning.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, TYPE_CHECKING
from overrides import override
from pydantic import BaseModel

from ....utils import setup_logger
from ...models import (
    Action,
    ActionResult,
    ActionGroupDescription,
    ActionPlan,
    ActionStatus,
    PlanExecutionContext,
    AgentSuspensionState,
    ActionPolicyExecutionState,
    ActionPolicyIO,
    PlanStatus,
    LifecycleMode,
    ResumptionCondition,
    ResumptionConditionType,
)
from ...base import Agent, ActionPolicy, ActionPolicyIterationResult, AgentCapability
from ...blackboard import BlackboardEvent
from ...blackboard.protocol import ActionPolicyProtocol
from ...scopes import BlackboardScope
from ....distributed.hooks import hookable
from .dispatcher import ActionDispatcher, ActionGroup, SchemaDetail, pydantic_model_to_str
from ..planning import (
    ActionPlanner,
    PlanBlackboard,
    HierarchicalAccessPolicy,
)
from ..planning.planner import create_cache_aware_planner
from ..planning.context import PlanningContextBuilder
from ..planning.streams import ConsciousnessStream
if TYPE_CHECKING:
    from ..planning.capabilities import ReplanningDecision


logger = setup_logger(__name__)



class BaseActionPolicy(ActionPolicy):
    """Base class for action policies with dataflow and nested policy support.

    Provides:
    - Automatic action dispatcher creation
    - Integration with agent capabilities
    - Nested policy execution with scope inheritance
    - Dispatch with automatic Ref resolution

    Subclasses implement `plan_step` to produce the next action or child policy.
    The base `execute_iteration` handles:
    - Delegating to active child policies
    - Executing actions returned by `plan_step`
    - Setting up child policies returned by `plan_step`

    TODO: For example, we can orchestrate iterative reasoning to follow the pattern
    (PLAN → ACT → REFLECT → CRITIQUE → ADAPT) by adding AgentCapabilities that
    implement each step as an action executor, and then implementing `plan_step`
    to select the next action based on the current state. This can be enforced by:
    - Restricting available actions in the action dispatcher depending on the
      last completed step, or
    - Using an ActionPolicy subclass that implements the iterative pattern by
      overriding `execute_iteration` to enforce the sequence of steps, and
      only calling `plan_step` to get parameters for each step, or
    - Prompting the LLM planner with this workflow.

    Example:
        ```python
        class MyPolicy(BaseActionPolicy):
            io = ActionPolicyIO(
                inputs={"query": str},
                outputs={"result": dict}
            )

            async def plan_step(self, state) -> Action | None:
                # Return None when policy is complete
                if state.custom.get("done"):
                    return None

                # Return an Action to execute
                return Action(
                    action_id="analyze_001",
                    agent_id=self.agent.agent_id,
                    action_type="analyze",
                    parameters={"query": state.scope.get("query")}
                )

                # Or return an ActionPolicy for nested execution
                # return ChildPolicy(self.agent)
        ```
    """

    def __init__(
        self,
        agent: Agent,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
        io: ActionPolicyIO | None = None, # Declare I/O contract (override in subclasses)
    ):
        super().__init__(agent)
        self._action_map = action_map
        self._action_providers = action_providers
        self._action_dispatcher: ActionDispatcher | None = None
        self.io: ActionPolicyIO = io or ActionPolicyIO()

    @override
    def use_agent_capabilities(self, capabilities: list[str]) -> None:
        """Add agent capabilities as action providers.

        Args:
            capabilities: List of capability names to use. Extends existing list.
        """
        super().use_agent_capabilities(capabilities)
        # Force recreation of dispatcher and action map
        self._action_dispatcher = None

    @override
    def disable_agent_capabilities(self, capabilities: list[str]) -> None:
        """Remove agent capabilities from action providers.

        Args:
            capabilities: List of capability names to disable.
        """
        super().disable_agent_capabilities(capabilities)
        # Force recreation of dispatcher and action map
        self._action_dispatcher = None

    @override
    async def initialize(self) -> None:
        """Initialize action policy."""
        await super().initialize()

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize policy-specific state.

        Subclasses should call super() and add their own state.
        """
        # Base implementation stores scope bindings
        state = await super().serialize_suspension_state(state)
        state.action_policy_state["scope_bindings"] = {}
        state.action_policy_state["scope_results"] = {}
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore policy-specific state from suspension.

        Subclasses should call super() and restore their own state.
        """
        # Base implementation - subclasses restore from state.policy_state
        pass

    def get_consciousness_streams(self) -> list[ConsciousnessStream]:
        """Return the policy's consciousness streams, in render order.

        Returns empty by default. Overridden by policies that maintain
        streams of recorded experience (events + actions) that should
        surface in the planning prompt.
        """
        return []

    async def get_action_descriptions(
        self,
        selected_groups: list[str] | None = None,
        schema_detail: SchemaDetail = SchemaDetail.SELECTIVE,
        include_tags: frozenset[str] | None = None,
        exclude_tags: frozenset[str] | None = None,
    ) -> list[ActionGroupDescription]:
        """Get descriptions of available actions.

        Args:
            selected_groups: If provided, only return descriptions for these group keys.
            schema_detail: How to render parameter schemas. See ``SchemaDetail``.
            include_tags: If provided, only include groups with at least one of these tags.
                Used for mode filtering (e.g., ``frozenset({"planning"})`` for planning mode).
            exclude_tags: If provided, exclude groups with any of these tags.
        """
        await self._create_action_dispatcher()
        return await self._action_dispatcher.get_action_descriptions(
            selected_groups=selected_groups,
            schema_detail=schema_detail,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

    async def get_action_group_summaries(self) -> list[ActionGroupDescription]:
        """Get lightweight group summaries for scope selection."""
        await self._create_action_dispatcher()
        return self._action_dispatcher.get_action_group_summaries()

    async def dispatch(
        self,
        action: Action,
    ) -> ActionResult:
        """Dispatch an action with dataflow support.

        Uses PolicyPythonREPL (via ActionDispatcher) for variable and result storage.

        Args:
            action: Action to execute

        Returns:
            ActionResult with execution outcome
        """
        await self._create_action_dispatcher()
        result = await self._action_dispatcher.dispatch(action)
        # Do not clear shared data dependencies here because a data
        # dependency may span multiple actions in multiple policy
        # iterations and must only be handled by the appropriate
        # AgentCapability implementing a multi-agent protocol.
        # scope._clear_shared_data_dependencies()
        return result

    async def _create_action_dispatcher(self) -> None:
        """Create action dispatcher with capability providers."""
        if self._action_dispatcher:
            return

        # Collect capability providers from _used_agent_capabilities (the
        # authoritative filter for which agent capabilities are exposed).
        capability_providers = self.get_used_capabilities()

        # self._action_providers may contain AgentCapability instances that
        # overlap with capability_providers (e.g. passed at construction from
        # agent._capabilities AND also registered via use_agent_capabilities).
        # Deduplicate by identity to avoid duplicate action groups in the prompt.
        seen = set(id(p) for p in capability_providers)
        extra_providers = [p for p in self._action_providers if id(p) not in seen]

        self._action_dispatcher = ActionDispatcher(
            agent=self.agent,
            action_policy=self,
            action_map=self._action_map,
            action_providers=capability_providers + extra_providers,
        )

        await self._action_dispatcher.initialize()

    @hookable
    @override
    async def execute_iteration(
        self,
        state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one iteration of this policy.

        This method is @hookable so memory capabilities can observe iterations.

        Calls `plan_step` to get next action, then dispatches it.

        For hierarchical composition (nested policies), spawn child agents
        instead of nesting policies. Use `self.agent.spawn_child_agents()`.

        Args:
            state: Execution state for this policy (all mutable state lives here)

        Returns:
            Iteration result
        """

        # TODO: Add iteration time limit check here
        ### if len(self.iteration_history) >= self.max_iterations:
        ###     logger.info("Analysis complete (max iterations reached)")
        ###     return ActionPolicyIterationResult(
        ###         success=True,
        ###         policy_completed=True
        ###     )

        # Set up session_id context for the ENTIRE iteration
        # Session_id may be set by a previous iteration's plan_step() from event metadata
        # This ensures all memory operations, hooks, and capabilities have session_id
        from ...sessions.context import session_id_context
        current_session_id = state.custom.get("current_session_id")

        with session_id_context(current_session_id):
            # Ensure dispatcher is initialized
            await self._create_action_dispatcher()

            # Get next action from subclass (plan_step may update current_session_id)
            logger.info(
                f"\n"
                f"    ┌────────────────────────────────────────────┐\n"
                f"    │  ⚙ EXEC_ITER: calling plan_step            │\n"
                f"    │  agent={self.agent.agent_id:<38}│\n"
                f"    └────────────────────────────────────────────┘"
            )
            next_action = await self.plan_step(state)
            action_str, trunc = pydantic_model_to_str(next_action)
            logger.info(f"    ⚙ EXEC_ITER: plan_step returned → {type(next_action).__name__}: {action_str} ({trunc})")

            # Re-check session_id in case plan_step updated it from a new event
            updated_session_id = state.custom.get("current_session_id")
            if updated_session_id != current_session_id:
                # Session changed mid-iteration, update context for dispatch
                # This handles the case where plan_step processes a new event with different session_id
                from ...sessions.context import set_current_session_id
                set_current_session_id(updated_session_id)

            if next_action is None:
                # Check if policy signaled completion
                if state.custom.get("policy_complete"):
                    logger.warning("    ⚙ EXEC_ITER: policy_complete=True → TERMINATING")
                    return ActionPolicyIterationResult(
                        success=True,
                        policy_completed=True
                    )

                # Check if policy signaled idle (no work, but not completed)
                if state.custom.get("idle"):
                    logger.warning("    ⚙ EXEC_ITER: idle=True → IDLE")
                    return ActionPolicyIterationResult(
                        success=True,
                        policy_completed=False,
                        idle=True,
                    )

                # Otherwise just skip this iteration (policy continues)
                logger.warning("    ⚙ EXEC_ITER: next_action=None → skipping iteration")
                return ActionPolicyIterationResult(
                    success=True,
                    policy_completed=False
                )

            # dispatch is @hookable, memory captures action there
            logger.warning(
                f"\n"
                f"    ╔════════════════════════════════════════════╗\n"
                f"    ║  🚀 DISPATCHING ACTION                    ║\n"
                f"    ║  id={next_action.action_id:<40}║\n"
                f"    ║  type={next_action.action_type:<38}║\n"
                f"    ╚════════════════════════════════════════════╝"
            )
            result = await self.dispatch(next_action)
            logger.warning(f"    🚀 DISPATCH returned: success={result.success}")

            return ActionPolicyIterationResult(
                success=result.success,
                policy_completed=False,
                action_executed=next_action,
                result=result,
            )

    async def plan_step(
        self,
        state: ActionPolicyExecutionState
    ) -> Action | None:
        """Produce the next action to execute.

        Override this method to implement policy-specific planning logic.

        For hierarchical composition, spawn child agents instead of nesting
        policies. Use `self.agent.spawn_child_agents()` with appropriate
        action policies for child agents.

        Args:
            state: Execution state for this policy

        Returns:
            - Action: Execute this action
            - None: Skip this iteration. Set `state.custom["policy_complete"] = True`
                before returning None to signal that the policy is finished.

        Example:
            ```python
            async def plan_step(self, state) -> Action | None:
                phase = state.custom.get("phase", "act")

                if phase == "act":
                    action = self._get_next_action(state)
                    if action is None:
                        state.custom["policy_complete"] = True
                        return None
                    state.custom["phase"] = "process"
                    return action

                elif phase == "process":
                    # Do some processing without dispatching an action
                    self._process_results(state)
                    state.custom["phase"] = "act"
                    return None  # Skip iteration, continue policy
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement plan_step"
        )


# ============================================================================
# Event-Driven Action Policy
# ============================================================================



class EventDrivenActionPolicy(BaseActionPolicy):
    """`ActionPolicy` that processes blackboard events via capabilities.

    Subclasses subscribe to blackboard events in `initialize()`.
    Events are queued and processed by `plan_step` on each iteration.

    This bridges the async event model with the synchronous iteration model
    without modifying `Agent.run_step()` or `BaseActionPolicy.execute_iteration()`.

    **Usage**:
        ```python
        class MyPolicy(EventDrivenActionPolicy):
            async def initialize(self) -> None:
                await super().initialize()
                capability = await self.agent.get_capability(CapabilityName)
                await capability.stream_events_to_queue(self.get_event_queue())

            @override
            async def plan_step(self, state) -> Action | None:
                event: BlackboardEvent = await self.get_next_event_nowait()
                if not event:
                    return None  # No events pending
                # Parse event.value and produce action
                return Action(action_type="process_event", agent_id=self.agent.agent_id, parameters={...})
        ```
    """

    def __init__(
        self,
        agent: Agent,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
        io: ActionPolicyIO | None = None, # Declare I/O contract (override in subclasses)
        reactive_only: bool = False,
        consciousness_streams: list[ConsciousnessStream] | None = None,
        **kwargs
    ):
        super().__init__(agent, action_map=action_map, action_providers=action_providers, io=io, **kwargs)
        self._event_queue: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
        self._subscribed_callbacks: list[Callable] = []
        self._subscribed_providers: set[int] = set()  # Track by identity to prevent duplicate subscriptions
        self._reactive_only = reactive_only
        self._consciousness_streams: list[ConsciousnessStream] = list(consciousness_streams or [])

    @override
    async def initialize(self) -> None:
        """Initialize event-driven policy."""
        await super().initialize()

        # Subscribe to blackboard events.
        #
        # We subscribe both:
        # - Agent capabilities (the default modular extension mechanism), and
        # - Explicit action_providers (for backwards compatibility / advanced composition).
        #
        # Deduplicate by object identity to avoid double-streaming the same capability.
        # Uses persistent _subscribed_providers set so capabilities added later
        # (via use_agent_capabilities) don't get subscribed again.
        for provider in list(self.agent.get_capabilities()) + list(self._action_providers):
            if id(provider) in self._subscribed_providers:
                continue
            self._subscribed_providers.add(id(provider))

            if isinstance(provider, AgentCapability) and hasattr(provider, "stream_events_to_queue"):
                logger.debug(
                    "Subscribing capability %s (scope_id=%s, input_patterns=%s) to event queue",
                    type(provider).__name__,
                    getattr(provider, "scope_id", "?"),
                    provider.input_patterns if hasattr(provider, "input_patterns") else "?",
                )
                await provider.stream_events_to_queue(self.get_event_queue())

    def get_event_queue(self) -> asyncio.Queue[BlackboardEvent]:
        """Get the local event queue.

        Returns:
            Local asyncio.Queue of BlackboardEvents
        """
        return self._event_queue

    @hookable
    async def get_next_event_nowait(self) -> BlackboardEvent | None:
        """Get the next pending event (non-blocking).

        This method is @hookable so memory capabilities can observe events.
        The returned BlackboardEvent can be captured by sensory memory hooks.

        The plan_step method can call this method to get as many
        events as needed within a single iteration.

        Returns:
            The next pending event, or None if no events are pending
        """
        try:
            return self._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @hookable
    async def get_next_event(self) -> BlackboardEvent:
        """Block until an event arrives (for reactive_only mode).

        Like get_next_event_nowait but blocking. Used when the agent should only
        act in response to events, never spontaneously. @hookable so tracing
        and memory hooks can observe events just like get_next_event_nowait.

        Returns:
            The next event (never None — blocks until one arrives).
        """
        return await self._event_queue.get()

    @override
    def get_consciousness_streams(self) -> list[ConsciousnessStream]:
        """Return the streams this policy feeds (events + actions)."""
        return list(self._consciousness_streams)

    def _get_event_handlers(self) -> list[Callable]:
        """Get all event handlers from capabilities and action providers.

        Returns:
            List of event handler methods (decorated with @event_handler)
        """
        handlers = []
        # Deduplicate by identity — agent capabilities and _action_providers
        # may overlap (same objects passed via both paths).
        seen: set[int] = set()
        for source in list(self.agent.get_capabilities()) + list(self._action_providers):
            if id(source) in seen:
                continue
            seen.add(id(source))
            handlers.extend(self._get_object_event_handlers(source))
        return handlers
    
    def _get_object_event_handlers(self, obj: Any) -> list[Callable]:
        """Get event handlers from an object.
        
        Args:
            obj: Object to inspect for @event_handler decorated methods
        
        Returns:
            List of event handler methods
        """
        handlers = []
        for name in dir(obj):
            method = getattr(obj, name, None)
            if callable(method) and hasattr(method, '_is_event_handler'):
                handlers.append(method)
        return handlers

    @override
    async def plan_step(self, state: ActionPolicyExecutionState) -> Action | None:
        """Plan next action with event-driven context enrichment.

        Flow:
        1. Get next event from queue via get_next_event_nowait() (non-blocking)
        2. Extract session_id from event metadata and set up context
        3. If event exists, broadcast to @event_handler methods in capabilities
           and action providers
        4. Accumulate context and check for immediate actions
        5. If no immediate action, invoke LLM planner with enriched context (via super().plan_step)

        NOTE: Event handlers provide context only, not transactions.
        Transaction management belongs in action executors that need it.
        Transactions are needed for states that are shared among multiple agents.
        Any action executor can use shared state (e.g., blackboard) to coordinate
        with other agents and needs to ensure the action or sequence of actions
        based on specific shared state versions are executed atomically.

        This enables modular, extensible event handling where:
        - Game capabilities handle game events
        - Memory capabilities observe and store events
        - Custom capabilities can participate in context enrichment

        Returns:
            Action to execute, or None
        """
        # 1. Get next event
        if self._reactive_only:
            # Block until an event arrives — no LLM calls when idle.
            # This makes the agent purely event-driven: it only acts
            # when something happens (user message, child agent event, etc.)
            event = await self.get_next_event()
        else:
            event = await self.get_next_event_nowait()

        if event is not None:
            logger.debug(
                "plan_step received event: key=%s, value_type=%s, value_preview=%s, metadata=%s",
                event.key,
                type(event.value).__name__,
                str(event.value)[:200] if event.value else "None",
                event.metadata,
            )
        else:
            logger.debug("plan_step: no event")

        # 2. Extract session_id and run_id from event and store in state for distributed traceability.
        # In distributed Ray systems, context variables don't cross node boundaries,
        # so we extract these from the event metadata and propagate them explicitly.
        # Store in state so execute_iteration() can set up context around dispatch()
        from ...sessions.context import session_id_context
        event_session_id = event.metadata.get("session_id") if event else None
        if event_session_id:
            state.custom["current_session_id"] = event_session_id
        # run_id is in the event value (set by AgentHandle.run/run_streamed)
        event_value = event.value if event else None
        if isinstance(event_value, dict) and event_value.get("run_id"):
            state.custom["current_run_id"] = event_value["run_id"]
            # Also update agent metadata so child agents inherit the run_id
            self.agent.metadata.run_id = event_value["run_id"]

        # 3. Broadcast to event handlers (within session_id context)
        immediate_actions = []
        accumulated_context: dict[str, dict[str, Any]] = {}

        if event is not None:
            with session_id_context(event_session_id):
                handlers = self._get_event_handlers()
                logger.debug(
                    "Broadcasting event key=%s to %d handlers: %s",
                    event.key, len(handlers),
                    [h.__name__ for h in handlers],
                )
                for handler in handlers:
                    try:
                        result = await handler(event, self._action_dispatcher.repl)

                        if result is None:
                            continue  # Event not relevant to this handler

                        logger.debug(
                            "Event handler %s returned: context_key=%s, has_context=%s, has_immediate=%s",
                            handler.__name__, result.context_key,
                            result.context is not None, result.immediate_action is not None,
                        )

                        # Accumulate context from all handlers
                        # This context is available to action executors via scope
                        context = result.context
                        if context:
                            if isinstance(result.context, BaseModel):
                                context = result.context.model_dump()
                            # TODO: Handle key conflicts (namespacing?)
                            # Event handlers can also store context in agent's
                            # working memory if needed
                            accumulated_context[result.context_key] = context

                        # Collect immediate actions (rule-based decision)
                        # (Don't return yet - process all handlers)
                        if result.immediate_action:
                            immediate_actions.append(result.immediate_action)

                        # Check for terminal state
                        if result.done:
                            state.custom["policy_complete"] = True
                            return None

                    except Exception as e:
                        logger.warning(
                            f"Event handler {handler.__name__} failed: {e}",
                            exc_info=True
                        )
                        continue

                # Store accumulated context in REPL for action executors
                # Action executors can access via repl.get("event_context")
                # Event handlers can also store context in agent's working memory
                if accumulated_context and self._action_dispatcher and self._action_dispatcher.repl:
                    self._action_dispatcher.repl.set("event_context", accumulated_context)
                    await self._store_event_context(
                        accumulated_context,
                        state,
                        namespace="event_context"
                    )

                    # Feed the accumulated context to every consciousness
                    # stream. Each stream decides independently (via its
                    # event_filter) whether to record this event.
                    for stream in self._consciousness_streams:
                        stream.consider_event(accumulated_context)

                # If any handler provided immediate action, return the first one
                # (others are ignored)
                # TODO: Could be made configurable (e.g., priority-based selection or
                #       let the LLM planner choose among them)
                if len(immediate_actions) == 1:
                    return immediate_actions[0]
                elif len(immediate_actions) > 1:
                    logger.info(
                        "Multiple immediate actions from event handlers; passing them to the LLM planner to decide among them."
                    )
                    # TODO: Ensure that these actions are not identical.
                    # Pass all to LLM planner via context
                    await self._store_event_context(
                        {
                            "immediate_actions": [action.model_dump() for action in immediate_actions],
                            "description": "Multiple immediate actions from event handlers"
                        },
                        state,
                        namespace="conflicting_immediate_actions"
                    )
                    # Let LLM planner decide among them

        # 4. Invoke LLM planner with enriched context
        # Subclasses should override this or use CacheAwareActionPolicy
        # This allows LLM to plan actions from other capabilities based on the data
        # previously injected into the policy scope and working memory by past events or
        # previous reasoning steps.
        return None

    async def _store_event_context(
        self,
        context: dict[str, Any],
        state: ActionPolicyExecutionState,
        namespace: str = "event_context",
    ) -> None:
        """Store event context in working memory for persistence.
        
        This is an optional method that can be called to persist event
        context beyond the current iteration. Most use cases should
        just use scope.bindings["event_context"] which is set automatically
        in plan_step.

        Event handlers can also store context directly in any memory level
        (e.g., STM, LTM episodic/semantic/procedural) as needed.

        Args:
            context: Context dict from event handler
            state: Policy execution state
            namespace: Namespace prefix for storage key
        """
        try:
            working_memory = self.agent.get_working_memory()
            if working_memory:
                await working_memory.store(
                    key=ActionPolicyProtocol.iteration_key(namespace, state.iteration_num),
                    value=context,
                    tags={namespace, "planning_context"},
                    ttl_seconds=3600,  # 1 hour - TODO: Make configurable
                )
        except Exception as e:
            logger.warning(f"Failed to store event context in memory: {e}", exc_info=True)


class CacheAwareActionPolicy(EventDrivenActionPolicy):

    """Action policy class for agents that use multi-step planning.

    This agent:
    - Creates plans using configurable strategies (MPC, top-down, bottom-up)
    - Executes plans incrementally via Agent.run_step
    - Handles replanning when needed
    - Coordinates with child agents event-driven (no polling)

    Attributes:
        action_history: History of actions (for debugging/logging)
    """
    def __init__(
        self,
        agent: Agent,
        planner: ActionPlanner,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
        io: ActionPolicyIO | None = None,
        context_builder: PlanningContextBuilder | None = None,
    ):
        """Initialize planning agent.

        Args:
            agent: Agent using this policy
            planner: Action planner
            action_map: List of action groups
            action_providers: Additional action providers
            io: Policy I/O contract (inputs/outputs)
        """
        super().__init__(
            agent=agent,
            action_map=action_map,
            action_providers=action_providers,
            io=io,
        )
        self.planner = planner  # TODO: Unify planner with planning strategy.
        self.plan_blackboard: PlanBlackboard | None = None

        # replanning_capability: Capability that decides WHEN to replan and what strategy to use.
        self.replanning_capability = None

        # Stream of consciousness: actions and planning
        self.action_history: list[Action] = [] # TODO: Currently unused
        self.current_plan: ActionPlan | None = None
        self.current_plan_id: str | None = None
        self.current_action_index: int | None = None
        self.context_builder = context_builder or PlanningContextBuilder(agent)

    def get_action_group_description(self) -> str:
        return (
            "Planning & Execution Control — manages the agent's plan lifecycle. "
            "Handles plan creation, replanning on failure or periodic triggers, "
            "and plan-level coordination with child agents via blackboard events."
        )

    async def initialize(self) -> None:
        """Initialize planning agent."""

        await super().initialize()

        # Get plan blackboard
        self.plan_blackboard = await self._get_plan_blackboard()

        if self.planner is None:
            self.planner = create_cache_aware_planner(agent=self.agent)

        from ..planning.capabilities import ReplanningCapability
        # Create default replanning capability if none provided
        if not self.agent.get_capability_by_type(ReplanningCapability):
            replan_every_n = 3
            replan_on_failure = True
            if hasattr(self.planner, 'planning_params'):
                replan_every_n = self.planner.planning_params.replan_every_n_steps
                replan_on_failure = self.planner.planning_params.replan_on_failure
            self.replanning_capability = ReplanningCapability(
                agent=self.agent,
                replan_every_n_steps=replan_every_n,
                replan_on_failure=replan_on_failure,
            )
            self.agent.add_capability(self.replanning_capability)
            logger.info(f"Added default ReplanningCapability to agent {self.agent.agent_id}")

        # Get current plan (if resuming from a previous session)
        # NOTE: Initial plan creation is NOT done here — it happens in
        # plan_step() on the first call, so that the LLM call falls inside
        # the STEP → PLAN span hierarchy for proper observability tracing.
        # TODO: Add a new tracing span on Agent.initialize()
        self.current_plan = await self.plan_blackboard.get_plan(self.agent.agent_id)
        ### if not self.current_plan:
        ###     await self._create_initial_plan()

        # Sync plan ID and action index
        if self.current_plan:
            self.current_plan_id = self.current_plan.plan_id
            self.current_action_index = self.current_plan.current_action_index

    @hookable
    async def _replan_horizon(self, decision: ReplanningDecision | None = None) -> ActionPlan:
        """Replan the remaining horizon of the current plan.

        This method is @hookable so memory capabilities can observe plan revisions.
        Returns the revised plan for hook-based capture.

        Args:
            decision: Optional replanning decision with triggers and strategy info.
                Passed through to the planner via planning_context.custom_data
                so the planning strategy can adapt its revision approach.
        """
        if not self.current_plan:
            raise RuntimeError("No current plan to replan.")

        planning_context = await self.context_builder.get_replanning_context(
            execution_context=self.current_plan.execution_context,
            decision=decision
        )

        # Generate plan via strategy
        self.current_plan = await self.planner.revise_plan(
            current_plan=self.current_plan,
            planning_context=planning_context,
        )

        triggers_str = (
            [t.value for t in decision.triggers] if decision else []
        )
        strategy_str = decision.strategy.value if decision else "default"
        logger.info(
            f"Replanned horizon for agent {self.agent.agent_id}, "
            f"triggers={triggers_str}, strategy={strategy_str}, "
            f"new plan has {len(self.current_plan.actions)} actions."
        )
        await self.plan_blackboard.update_plan(self.current_plan)

        return self.current_plan

    async def plan_step(
        self,
        state: ActionPolicyExecutionState
    ) -> Action | None:
        """Produce next action using model-predictive control planning.

        Flow:
        1. Process events via event handlers (super().plan_step)
        2. Process result of previous action (if any)
        3. Get/create plan from blackboard
        4. Check if plan complete
        5. Check if replanning needed (MPC)
        6. Return next action
        """
        # Process events first (calls event handlers, enriches context)
        logger.warning(f"      📋 PLAN_STEP: checking events  agent={self.agent.agent_id}")
        event_action = await super().plan_step(state)
        if event_action:
            logger.warning(f"      📋 PLAN_STEP: event produced immediate action → {event_action}")
            return event_action
        if state.custom.get("policy_complete"):
            logger.warning("      📋 PLAN_STEP: policy_complete set by event handler")
            return None

        # Get plan from blackboard
        logger.warning(f"      📋 PLAN_STEP: fetching plan from blackboard  agent_id={self.agent.agent_id}")
        state.current_plan = await self.plan_blackboard.get_plan(self.agent.agent_id)
        if state.current_plan:
            logger.warning(
                f"      📋 PLAN_STEP: got plan → {len(state.current_plan.actions)} actions, "
                f"idx={state.current_plan.current_action_index}, status={state.current_plan.status}"
            )
        else:
            logger.warning("      📋 PLAN_STEP: got plan → None")
            # No plan - create initial plan
            logger.warning(
                f"\n"
                f"      ╔════════════════════════════════════════════╗\n"
                f"      ║  🧠 CREATING INITIAL PLAN (LLM call)      ║\n"
                f"      ║  agent={self.agent.agent_id:<38}║\n"
                f"      ╚════════════════════════════════════════════╝"
            )
            await self._create_initial_plan()
            state.current_plan = self.current_plan
            logger.warning(
                f"      🧠 PLAN CREATED: id={self.current_plan.plan_id if self.current_plan else 'NONE!'} "
                f"actions={len(self.current_plan.actions) if self.current_plan else 0}"
            )
            return None  # Plan created, will get action on next call

        # Process result of previous action (from previous iteration)
        last_action_id = state.custom.get("last_action_id")
        repl = self._action_dispatcher.repl if self._action_dispatcher else None
        last_result: ActionResult | None = None  # Hoisted for replanning policy
        if last_action_id and repl:
            last_result = repl.get_result(last_action_id)
            if last_result:
                # Update plan execution context
                state.current_plan.execution_context.completed_action_ids.append(last_action_id)
                state.current_plan.execution_context.action_results[last_action_id] = last_result

                # Update the action's status in the plan so it survives
                # blackboard round-trips (get_plan re-deserializes the plan,
                # losing in-memory status changes made by dispatch())
                last_action = state.current_plan.get_action_by_id(last_action_id)
                if last_action:
                    last_action.status = ActionStatus.COMPLETED if last_result.success else ActionStatus.FAILED
                    last_action.result = last_result
                else:
                    logger.warning(
                        f"Could not find last action {last_action_id} in current plan for agent {self.agent.agent_id}"
                    )

                # Handle failure
                if not last_result.success:
                    logger.warning(
                        f"Agent {self.agent.agent_id} failed to execute action {last_action_id}"
                    )
                    # Check if blocked
                    if last_result.blocked_reason:
                        logger.info(f"Agent {self.agent.agent_id} blocked: {last_result.blocked_reason}")
                        await self.plan_blackboard.update_plan(state.current_plan)

                        # Build structured resumption condition from action result
                        if last_result.blocking_agent_ids:
                            condition = ResumptionCondition(
                                condition_type=ResumptionConditionType.CHILDREN_COMPLETED,
                                blocking_agent_ids=last_result.blocking_agent_ids,
                            )
                        else:
                            condition = ResumptionCondition(
                                condition_type=ResumptionConditionType.IMMEDIATE,
                            )
                        await self.agent.suspend(
                            reason=f"Blocked: {last_result.blocked_reason}",
                            resumption_condition=condition,
                        )
                        # Return None but don't set complete - we're suspended
                        return None

                # Persist plan state (results, completed_action_ids).
                # NOTE: current_action_index is persisted AFTER increment below.
                await self.plan_blackboard.update_plan(state.current_plan)

                # Sync local state
                self.current_plan_id = state.current_plan.plan_id
                self.current_action_index = state.current_plan.current_action_index

            state.custom["last_action_id"] = None

        # MPC: Evaluate replanning need via policy (handles both periodic and plan exhaustion)
        logger.warning(
            f"      📋 PLAN_STEP: idx={state.current_plan.current_action_index} / "
            f"{len(state.current_plan.actions)} actions"
        )
        decision = await self.replanning_capability.evaluate_replanning_need(
            state=state,
            last_result=last_result,
        )

        if decision.should_replan:
            logger.warning(f"      📋 PLAN_STEP: !!! REPLANNING triggered: {decision.reason}")
            # Try to extend the plan via replanning (MPC continuation).
            # The planner sees the full execution_context (completed actions + results)
            # and decides if more work is needed or if the goal is satisfied.
            await self._replan_horizon(decision)
            # _replan_horizon calls revise_plan which keeps completed actions
            # and appends new ones. Refresh from self.current_plan.
            state.current_plan = self.current_plan

            if decision.plan_exhausted and state.current_plan.has_remaining_actions():
                # Plan exhaustion replan produced continuation actions — keep going
                logger.warning(
                    f"      📋 PLAN_STEP: replan produced "
                    f"{len(state.current_plan.actions) - state.current_plan.current_action_index} "
                    f"new actions, continuing"
                )
                return None  # Will get next action on next iteration

        if decision.plan_exhausted and not state.current_plan.has_remaining_actions():
            # True completion: plan exhausted AND (planner says done OR budget exceeded)
            logger.warning("      📋 PLAN_STEP: ★★★ PLAN COMPLETE ★★★")
            state.current_plan.status = PlanStatus.COMPLETED
            state.current_plan.completed_at = time.time()
            await self.plan_blackboard.update_plan(state.current_plan)

            # Learn from execution
            await self.planner.learn_from_plan_execution(state.current_plan)

            logger.info(f"Agent {self.agent.agent_id} completed plan {state.current_plan.plan_id}")

            if self.agent.metadata.lifecycle_mode == LifecycleMode.CONTINUOUS:
                # Signal IDLE via state.custom → execute_iteration reads it → returns idle=True
                # Agent.run_step() reads idle=True and transitions agent state
                state.custom["idle"] = True
                self.replanning_capability.reset_state(state)  # Reset for next work cycle
                return None
            else:
                # ONE_SHOT: signal completion → execute_iteration returns policy_completed=True
                state.custom["policy_complete"] = True
                return None

        # Get next action and advance index
        next_action = state.current_plan.actions[state.current_plan.current_action_index]
        state.current_plan.current_action_index += 1
        state.custom["last_action_id"] = next_action.action_id

        # Persist the incremented index so re-fetch from blackboard sees it
        # Always persist plan state (index, results) — even on failure.
        # Without this, the re-fetch from blackboard resets the index.
        await self.plan_blackboard.update_plan(state.current_plan)
        self.current_plan_id = state.current_plan.plan_id
        self.current_action_index = state.current_plan.current_action_index

        logger.info(
            f"      📋 PLAN_STEP: returning action → id={next_action.action_id} type={next_action.action_type}"
        )
        return next_action

    async def _get_plan_blackboard(self) -> PlanBlackboard:
        """Get or create plan blackboard with access control."""
        # Create access policy with agent hierarchy
        # TODO: FIXME: These two methods are incorrectly implemented.
        agent_hierarchy = await self.agent.get_agent_hierarchy()
        team_structure = await self.agent.get_team_structure()

        # TODO: The agent hierarchy and team structure are dynamic. FIXME
        access_policy = HierarchicalAccessPolicy(
            agent_hierarchy=agent_hierarchy,
            team_structure=team_structure,
        )

        plan_blackboard = PlanBlackboard(
            agent=self.agent,
            plan_access_policy=access_policy,
            scope=BlackboardScope.COLONY,
        )
        await plan_blackboard.initialize()
        return plan_blackboard

    @hookable
    async def _create_initial_plan(self) -> ActionPlan:
        """Create initial plan.

        This method is @hookable so memory capabilities can observe plan creation.
        Returns the created plan for hook-based capture.
        """

        logger.warning("        🧠 _create_initial_plan: building planning context...")
        planning_context = await self.context_builder.get_planning_context(
            execution_context=PlanExecutionContext()
        )
        logger.warning(
            f"        🧠 _create_initial_plan: context ready — "
            f"goals={planning_context.goals}, "
            f"pages={len(planning_context.page_ids)}, "
            f"actions={len(planning_context.action_descriptions)}"
        )

        logger.warning(
            "\n"
            "        ╔════════════════════════════════════════╗\n"
            "        ║  🔮 CALLING planner.create_plan()      ║\n"
            "        ║  (THIS IS THE LLM INFERENCE CALL)      ║\n"
            "        ╚════════════════════════════════════════╝"
        )
        plan: ActionPlan = await self.planner.create_plan(planning_context)
        logger.warning(
            f"        🔮 planner.create_plan() returned: "
            f"plan_id={plan.plan_id}, actions={len(plan.actions)}, status={plan.status}"
        )
        plan.agent_id = self.agent.agent_id

        approved, msg = await self.plan_blackboard.propose_plan(plan, self.agent.agent_id)
        if approved:
            self.current_plan = plan
            # Sync plan ID and action index
            self.current_plan_id = plan.plan_id
            self.current_action_index = plan.current_action_index
            logger.info(f"Plan approved for agent {self.agent.agent_id}: {plan.plan_id}")
        else:
            # TODO: Handle plan rejection (e.g., modify and resubmit)
            # TODO: Handle pending approval properly
            logger.info(f"Plan pending approval for agent {self.agent.agent_id}: {msg}")
            self.current_plan = plan
            # Sync plan ID and action index
            self.current_plan_id = plan.plan_id
            self.current_action_index = plan.current_action_index

        return plan

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize policy-specific state.

        Args:
            state: AgentSuspensionState with all agent state serialized
        """
        state = await super().serialize_suspension_state(state)
        # Add ActionPolicy-specific state
        # Execution state
        if self.current_plan:
            state.plan_id = self.current_plan.plan_id
            state.current_action_index = self.current_plan.current_action_index
        else:
            state.plan_id = self.current_plan_id
            state.current_action_index = self.current_action_index
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore policy-specific state from suspension.

        Args:
            state: AgentSuspensionState to restore from
        """
        self.current_action_index = state.current_action_index
        self.current_plan_id = state.plan_id

    async def get_status_snapshot(self) -> dict[str, Any]:
        """Get snapshot of current status for debugging/monitoring.

        Returns:
            Status dictionary
        """
        status = {
            "current_plan_id": self.current_plan_id,
            "current_action_index": self.current_action_index,
            "total_actions_executed": len(self.action_history),
        }
        if self.current_plan:
            status.update({
                "plan_status": str(self.current_plan.status),
                "plan_actions_total": len(self.current_plan.actions),
                "plan_actions_completed": self.current_plan.current_action_index,
            })
        return status




async def create_cache_aware_action_policy(
    agent: Agent,
    action_map: list[ActionGroup] | None = None,
    action_providers: list[Any] = [],
    io: ActionPolicyIO | None = None,
    max_iterations: int = 50,
    quality_threshold: float = 0.9,
    planning_horizon: int = 5,
    ideal_cache_size: int = 10,
) -> CacheAwareActionPolicy:
    """Create sophisticated action policy with cache-awareness and learning.

    Returns:
        CacheAwareActionPolicy
    """
    from ..planning.planner import create_cache_aware_planner

    planner = await create_cache_aware_planner(
        agent=agent,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        planning_horizon=planning_horizon,
        ideal_cache_size=ideal_cache_size,
    )

    action_policy = CacheAwareActionPolicy(
        agent=agent,
        planner=planner,
        action_map=action_map,
        action_providers=action_providers,
        io=io
    )
    await action_policy.initialize()

    return action_policy

