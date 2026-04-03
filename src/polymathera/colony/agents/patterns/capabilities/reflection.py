
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from overrides import override
from pydantic import BaseModel, Field

from ...blackboard.types import BlackboardEvent, KeyPatternFilter
from .consciousness import SystemDocumentation
from ..models import Reflection
from ...models import Action, ActionResult, ActionPlan, AgentSuspensionState
from ...base import Agent, AgentCapability
from ...blackboard.protocol import ReflectionProtocol
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ....distributed import get_polymathera
from ..actions import action_executor

if TYPE_CHECKING:
    from ..memory.capability import MemoryCapability

logger = logging.getLogger(__name__)


class ReflectionRequest(BaseModel):
    """Request for reflection from another agent.

    Published to blackboard when an agent requests reflection assistance.
    """

    request_id: str = Field(
        default_factory=lambda: f"refl_req_{int(time.time() * 1000)}",
        description="Unique request identifier"
    )
    requester_id: str = Field(..., description="ID of agent requesting reflection")
    focus: str = Field(..., description="What to reflect on (e.g., 'performance', 'goals', 'capabilities')")
    context: ReflectionContext = Field(
        default_factory=lambda: ReflectionContext(focus="general"),
        description="Additional context for reflection (e.g., system architecture, recent actions, challenges, current plan)"
    )
    created_at: float = Field(default_factory=time.time)


class ReflectionContext(BaseModel):
    """Context for reflection actions.

    Attributes:
        focus: What to reflect on (e.g., "performance", "goals", "capabilities")
        system_architecture: Overview of system architecture
        role: Agent's role in the system
        capabilities: List of agent capabilities
        goals: List of agent goals
        recent_actions: Recent actions taken by the agent
        current_plan: Current action plan
    """

    focus: str = Field(
        ...,
        description="What to reflect on (e.g., 'performance', 'goals', 'capabilities')"
    )
    system_architecture: str = Field(
        default="",
        description="Overview of system architecture"
    )
    role: str = Field(default="unknown", description="Agent's role in the system")
    goals: list[str] = Field(
        default_factory=list,
        description="List of agent goals"
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of agent capabilities"
    )
    recent_actions: list[tuple[Action, ActionResult]] = Field(
        default_factory=list,
        description="Recent actions taken by the agent"
    )
    # TODO: The plan needs to include branches abandoned by replanning/MPC
    current_plan: ActionPlan | None = Field(
        default=None,
        description="Current action plan"
    )

    @staticmethod
    def from_action_params(action: Action) -> ReflectionContext:
        """Create ReflectionContext from action execution data."""
        # Only require 'focus' - other fields have defaults
        if "focus" not in action.parameters:
            raise ValueError("Reflection action requires 'focus' in parameters")

        return ReflectionContext(**{
            field: action.parameters.get(field, ReflectionContext.model_fields[field].default)
            for field in ReflectionContext.model_fields.keys()
            if field in action.parameters or ReflectionContext.model_fields[field].default is not None
        })


class ReflectionCapability(AgentCapability):
    """Capability for agent self-reflection and learning.

    Reflects on action results to:
    - Identify what was learned
    - Detect violated assumptions
    - Determine if more context is needed

    This capability queries memory for recent action results with reflection
    context (learnings written by action executors), then uses LLM to generate
    insights.

    The key design principle: action executors write reflection-relevant context
    to their results via `result.output["_reflection_learnings"]`, which gets
    captured by memory hooks. This capability then queries memory to gather
    that context for LLM-based reflection.

    Allowing the action policy to select self-reflection as an action type allows
    recursive reflection (reflecting on reflections), which may be powerful.

    Example:
        ```python
        # Add reflection capability to agent
        reflector = ReflectionCapability(agent=self)
        await reflector.initialize()
        agent.add_capability(reflector)

        # Trigger reflection via action
        result = await action_policy.dispatch(Action(
            action_type="reflect",
            parameters={"focus": "performance"}
        ))
        ```
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "reflection",
        input_patterns: list[str] = [ReflectionProtocol.request_pattern()],
        capability_key: str = "reflection",
    ):
        """Initialize reflection capability.

        Args:
            agent: The owning agent
            scope: Blackboard scope. Defaults to BlackboardScope.COLONY.
            namespace: Namespace for the capability within the scope (default "reflection")
            input_patterns: List of input patterns for the capability (default listens for reflection requests)
            capability_key: Unique key for this capability (default "reflection")
        """
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=input_patterns, capability_key=capability_key)
        self._state_managers: dict[str, Any] = {}

    def get_action_group_description(self) -> str:
        return (
            "Self-Reflection — reflects on recent action results to identify learnings, "
            "detect violated assumptions, and determine if more context is needed. "
            "Queries memory for reflection-relevant context written by action executors "
            "(via result.output['_reflection_learnings']). "
            "Can request reflection from a peer agent for external perspective."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ReflectionCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ReflectionCapability")
        pass

    async def get_system_documentation(self) -> SystemDocumentation:
        """Get system documentation for self-awareness.

        Returns:
            System documentation
        """
        state_key = f"{ScopeUtils.get_agent_level_scope(self.agent.agent_id)}:system_documentation"

        if state_key not in self._state_managers:
            polymathera = get_polymathera()
            self._state_managers[state_key] = await polymathera.get_state_manager(
                state_type=SystemDocumentation,
                state_key=state_key,
            )

        state_manager = self._state_managers[state_key]
        async for state in state_manager.read_transaction():
            return state

        return SystemDocumentation()

    async def _gather_reflection_context_from_memory(
        self, focus: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Query memory for recent actions with reflection context.

        Action executors write reflection-relevant learnings to their results
        via `result.output["_reflection_learnings"]`. This method queries
        memory to gather that context.

        Args:
            focus: What to focus the reflection on
            max_results: Maximum number of recent actions to retrieve

        Returns:
            List of action contexts with their learnings
        """
        from ..memory.capability import MemoryCapability
        from ..memory.types import MemoryQuery, TagFilter

        # Try to get memory capability
        try:
            memory: MemoryCapability = self.agent.get_capability(
                MemoryCapability.get_capability_name()
            )
        except (KeyError, AttributeError):
            logger.debug("No memory capability available for reflection context")
            return []

        # Query for recent actions with reflection context
        try:
            entries = await memory.recall(
                MemoryQuery(
                    tag_filter=TagFilter(all_of={"action"}),
                    max_results=max_results,
                    max_age_seconds=3600,  # Last hour
                )
            )

            contexts = []
            for entry in entries:
                if hasattr(entry, "value") and isinstance(entry.value, dict):
                    action_data = entry.value
                    contexts.append({
                        "action_type": action_data.get("action_type"),
                        "success": action_data.get("success"),
                        "learnings": action_data.get("_reflection_learnings", {}),
                        "timestamp": entry.metadata.get("created_at"),
                    })
                elif hasattr(entry, "value") and hasattr(entry.value, "action_type"):
                    # It's an Action object
                    action = entry.value
                    learnings = {}
                    if action.result and action.result.output:
                        learnings = action.result.output.get("_reflection_learnings", {})
                    contexts.append({
                        "action_type": str(action.action_type),
                        "success": action.result.success if action.result else None,
                        "learnings": learnings,
                        "timestamp": action.created_at,
                    })

            return contexts

        except Exception as e:
            logger.warning(f"Failed to query memory for reflection context: {e}")
            return []

    def _build_reflection_prompt(
        self,
        focus: str,
        action_contexts: list[dict[str, Any]],
        system_docs: SystemDocumentation,
    ) -> str:
        """Build LLM prompt for reflection.

        Args:
            focus: What to reflect on
            action_contexts: Recent action contexts from memory
            system_docs: System documentation

        Returns:
            Formatted prompt for LLM
        """
        # Format action history
        action_history = []
        for ctx in action_contexts:
            entry = f"- {ctx.get('action_type', 'unknown')}: "
            if ctx.get("success"):
                entry += "SUCCESS"
            else:
                entry += "FAILED"
            if ctx.get("learnings"):
                entry += f" | Learnings: {ctx['learnings']}"
            action_history.append(entry)

        action_text = "\n".join(action_history) if action_history else "No recent actions"

        # TODO: Should reflection prompt depend on the actions being reflected on and goals?
        prompt = f"""Reflect on recent agent behavior with focus: {focus}

System Architecture:
{system_docs.architecture or "Not specified"}

Recent Actions:
{action_text}

Based on this context, provide reflection on:
1. What was learned from recent actions?
2. Were any assumptions violated?
3. Is more information needed?
4. How confident are you in these conclusions?

Respond with a structured analysis."""

        return prompt

    @action_executor(action_key="reflect")
    async def reflect(self, context: ReflectionContext | dict) -> Reflection:
        """Perform self-reflection using memory-gathered context.

        Queries memory for recent action results with reflection learnings,
        then uses LLM to generate insights.

        Args:
            context: Reflection context containing focus and optional context

        Returns:
            Reflection with learned insights
        """

        # TODO: Do not ask the LLM planner to provide a ReflectionContext.
        # Most of its fields should be gathered from the agent's state directly.

        if isinstance(context, dict):
            context = ReflectionContext(**context)

        focus = context.focus

        # Get system documentation for LLM-based reflection (TODO: use in prompt)
        sys_docs = await self.get_system_documentation()

        # Load self-concept directly from storage
        # TODO: This should be cached in the ConsciousnessCapability which
        # should be called by the ActionPolicy to get self-concept info into context.
        try:
            from ...self_concept import AgentSelfConcept
            from .consciousness import ConsciousnessCapability
            consciousness: ConsciousnessCapability | None = (
                self.agent.get_capability(ConsciousnessCapability.get_capability_name())
            )
            self_concept: AgentSelfConcept | None = (
                await consciousness.get_self_concept() if consciousness else None
            )
        except Exception:
            self_concept = None

        # Prepare reflection context
        action_policy = self.agent.action_policy
        reflection_context = {
            "focus": context.focus,
            "system_architecture": sys_docs.architecture,
            "role": self_concept.role if self_concept else "unknown",
            "capabilities": self_concept.capabilities if self_concept else [],
            "goals": self_concept.goals if self_concept else [],
            "recent_actions": [
                {
                    "type": a.action_type.value,
                    "status": a.status.value,
                    "success": a.result.success if a.result else None,
                }
                for a in action_policy.action_history[-10:]  # TODO: make configurable
            ],
            "current_plan": action_policy.current_plan.model_dump() if action_policy.current_plan else None,
            **(context or {}),
        }

        # Gather context from memory
        action_contexts = await self._gather_reflection_context_from_memory(focus)

        # If we have recent_actions in context, use those as fallback
        if not action_contexts and context.recent_actions:
            for action, result in context.recent_actions[-10:]:
                learnings = {}
                if result and result.output:
                    learnings = result.output.get("_reflection_learnings", {})
                action_contexts.append({
                    "action_type": str(action.action_type),
                    "success": result.success if result else None,
                    "learnings": learnings,
                    "timestamp": action.created_at,
                })

        # TODO: Action failure is a major reflection opportunity.
        # Check for any recent failures
        recent_failures = [
            ctx for ctx in action_contexts
            if ctx.get("success") is False
        ]

        if recent_failures:
            # TODO: More nuanced reflection on failures. This is just STUPID.
            # Action failed - we learned nothing, need to retry or adapt
            # If there are failures, note them in reflection
            failed_actions = [ctx.get("action_type", "unknown") for ctx in recent_failures]
            return Reflection(
                success=False,
                learned={},
                assumptions_violated=[f"Actions failed: {failed_actions}"],
                needs_more_info=True,
                confidence=0.3,
                reasoning=f"Recent actions failed: {failed_actions}. Need to investigate and adapt.",
            )

        # Aggregate learnings from all action contexts
        all_learnings: dict[str, Any] = {}
        for ctx in action_contexts:
            learnings = ctx.get("learnings", {})
            if learnings:
                all_learnings.update(learnings)

        # Build prompt and use LLM for deeper reflection
        # TODO: Use LLM inference to actually perform reflection
        # For now, aggregate learnings and return structured reflection

        # Determine if more info is needed based on learnings
        needs_more_info = all_learnings.get("needs_more_info", False)

        return Reflection(
            success=True,
            learned=all_learnings,
            assumptions_violated=[],
            needs_more_info=needs_more_info,
            confidence=0.8,
            reasoning=f"Reflected on {len(action_contexts)} recent actions with focus: {focus}",
        )

    @action_executor(action_key="request_reflection")
    async def request_reflection_from_peer(
        self,
        peer_id: str,
        focus: str,
        context: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> Reflection | None:
        """Request reflection assistance from a peer agent.

        Posts a reflection request to the peer's blackboard and waits for response.

        Args:
            peer_id: ID of peer agent to request from
            focus: What to reflect on
            context: Additional context
            timeout: How long to wait for response

        Returns:
            Reflection from peer, or None if timeout
        """
        request = ReflectionRequest(
            requester_id=self.agent.agent_id,
            focus=focus,
            context=context or {},
        )

        # Post request to peer's blackboard
        peer_blackboard = await self.agent.get_blackboard(
            scope_id=ScopeUtils.get_agent_level_scope(peer_id)
        )
        await peer_blackboard.write(
            key=ReflectionProtocol.request_key(request.request_id),
            value=request.model_dump(),
            created_by=self.agent.agent_id,
            tags={"reflection_request", f"from:{self.agent.agent_id}"},
        )

        # Set up event for response
        response_event = asyncio.Event()
        response_data: dict[str, Any] = {}

        response_key = ReflectionProtocol.response_key(request.request_id)

        async def on_response(event: BlackboardEvent) -> None:
            if event.key == response_key:
                response_data["value"] = event.value
                response_event.set()

        # Subscribe to our own blackboard for response
        my_blackboard = await self.agent.get_agent_level_blackboard()
        my_blackboard.subscribe(on_response, filter=KeyPatternFilter(response_key))

        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            if "value" in response_data:
                return Reflection(**response_data["value"])
            return None
        except asyncio.TimeoutError:
            logger.warning(
                f"No reflection response from {peer_id} after {timeout}s"
            )
            return None
        finally:
            my_blackboard.unsubscribe(on_response)
