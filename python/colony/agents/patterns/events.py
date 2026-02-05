"""Event processing infrastructure for capabilities.

This module provides the `@event_handler` decorator and `EventProcessingResult`
for capabilities to process events and influence action planning.

Architecture:
- Event handlers provide CONTEXT for LLM planning, not transactions
- Transactions are managed by action executors that modify shared state
- This ensures transaction lifetime matches the action that needs it
"""

from __future__ import annotations

import fnmatch
import functools
from pydantic import BaseModel, Field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..blackboard.types import BlackboardEvent
    from ..models import Action, PolicyREPL


class EventProcessingResult(BaseModel):
    """Result of processing an event by a capability.
    
    Capabilities return this to:
    1. Add context for LLM planning (stored in scope for action executors)
    2. Execute rule-based decisions immediately (skip LLM planning)
    3. Signal terminal states (stop processing)
    
    NOTE: Event handlers do NOT manage transactions. Transaction management
    belongs in action executors that modify shared state (e.g., submit_move).
    This ensures transaction lifetime matches the action that needs it.
    
    Example:
        ```python
        @event_handler
        async def handle_game_event(
            self,
            event: BlackboardEvent,
            repl: PolicyREPL,
        ) -> EventProcessingResult | None:
            game_event = self._parse_event(event)
            if game_event is None:
                return None  # Not relevant to this capability
            
            # Check for terminal state
            if game_event.game_state.is_terminal():
                return EventProcessingResult(done=True)
            
            # Check for rule-based response
            if action := self._rule_based_response(game_event):
                return EventProcessingResult(immediate_action=action)
            
            # Provide context for LLM planning (no transactions here!)
            return EventProcessingResult(
                context={
                    "game_state": game_event.game_state.model_dump(),
                    "game_state_version": game_event.version,  # For action executor validation
                    "phase": game_event.game_state.phase.value,
                    "my_role": self.role,
                },
            )
        ```
    """
    context_key: str = Field(
        default="default",
        description=(
            "Key to distinguish between context from different event handlers when "
            'storing contexts in agent memory / scope (e.g. "game_context").'
        ),
    )

    # Additional context for LLM planner (merged into scope.bindings["event_context"])
    # Action executors can read this to validate state versions before modifying
    context: BaseModel | dict[str, Any] | None = Field(default=None)
    
    # If set, skip LLM planning and execute this action immediately
    # Use for rule-based responses that don't need LLM reasoning
    immediate_action: Action | None = Field(default=None)
    
    # If True, stop processing (terminal state, game over, task complete)
    done: bool = Field(default=False)
    
    model_config = {"arbitrary_types_allowed": True}


# Singleton for "I processed this but have nothing to contribute"
PROCESSED = EventProcessingResult(context_key="processed")


def _resolve_pattern(pattern: str, capability: Any) -> str:
    """Resolve pattern template variables from capability instance.

    Supports:
    - {scope_id} - replaced with capability.scope_id
    - {agent_id} - replaced with capability.agent.agent_id

    Args:
        pattern: Pattern template string
        capability: Capability instance (self)

    Returns:
        Resolved pattern with variables substituted
    """
    resolved = pattern

    # Resolve {scope_id}
    if "{scope_id}" in resolved:
        scope_id = getattr(capability, "scope_id", None)
        if scope_id:
            resolved = resolved.replace("{scope_id}", scope_id)

    # Resolve {agent_id}
    if "{agent_id}" in resolved:
        agent = getattr(capability, "agent", None)
        if agent:
            agent_id = getattr(agent, "agent_id", None)
            if agent_id:
                resolved = resolved.replace("{agent_id}", agent_id)

    return resolved


def event_handler(
    func: Callable | None = None,
    *,
    pattern: str | None = None,
):
    """Decorator to mark a capability method as an event handler.

    Event handlers are called by `EventDrivenActionPolicy.plan_step` when
    events are available. They can:
    - Process events and update internal state
    - Enrich planning context for LLM (context dict)
    - Provide rule-based immediate actions
    - Signal terminal states

    IMPORTANT: Event handlers should NOT manage transactions. They provide
    context only. Transaction management belongs in action executors.

    Args:
        pattern: Optional event key pattern to filter events before calling handler.
            Supports glob-style wildcards (*) and template variables:
            - {scope_id}: Resolved to self.scope_id
            - {agent_id}: Resolved to self.agent.agent_id

            Example patterns:
            - "{scope_id}:request:*" - matches any request for this capability
            - "{agent_id}:result:*" - matches any result for this agent
            - "game:*:move" - matches any game move event

            If None, handler receives all events (must filter manually).

    The handler signature must be:
        async def handler(
            self,
            event: BlackboardEvent,
            repl: PolicyREPL,
        ) -> EventProcessingResult | None

    Returns:
        - EventProcessingResult: If you processed the event
        - None: If this event is not relevant to you (skip)

    Example:
        ```python
        class GameProtocolCapability(AgentCapability):
            # Without pattern - handler must filter manually
            @event_handler
            async def handle_game_event(
                self,
                event: BlackboardEvent,
                repl: PolicyREPL,
            ) -> EventProcessingResult | None:
                if not event.key.startswith(...):
                    return None
                ...

            # With pattern - automatic filtering
            @event_handler(pattern="{scope_id}:request:*")
            async def handle_analysis_request(
                self,
                event: BlackboardEvent,
                repl: PolicyREPL,
            ) -> EventProcessingResult | None:
                # No need to check pattern - decorator handles it
                ...
        ```
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self, event, scope):
            # If pattern specified, check before calling handler
            if pattern is not None:
                resolved_pattern = _resolve_pattern(pattern, self)
                if not fnmatch.fnmatch(event.key, resolved_pattern):
                    return None
            return await fn(self, event, scope)

        wrapper._is_event_handler = True
        wrapper._event_pattern = pattern
        return wrapper

    # Support both @event_handler and @event_handler(pattern="...")
    if func is not None:
        # Called without parentheses: @event_handler
        func._is_event_handler = True
        func._event_pattern = None
        return func
    else:
        # Called with parentheses: @event_handler(pattern="...")
        return decorator
