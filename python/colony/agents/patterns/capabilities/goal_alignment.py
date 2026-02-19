"""ObjectiveGuardAgent: Prevents goal drift.

The ObjectiveGuardAgent is a meta-agent that:
- Monitors outputs to ensure alignment with original goals
- Detects goal drift (agents doing something other than requested)
- Challenges off-topic or irrelevant work
- Triggers re-planning when drift detected

This addresses the LLM failure mode of:
"Deviating from stated objectives" - agents doing clever but unhelpful things.

Integration:
- Maintains explicit JointGoal for each task
- Compares intermediate results against goals
- Emits challenge messages when drift detected

Programming Model (AgentHandle Pattern):
---------------------------------------
```python
# Spawn objective guard agent with handle
handle = await owner.spawn_child_agents(
    agent_specs=[AgentSpawnSpec(agent_type="...ObjectiveGuardAgent")],
    capability_types=[ObjectiveGuardCapability],
)[0]

# Get capability and communicate
guard = handle.get_capability(ObjectiveGuardCapability)
await guard.stream_events_to_queue(self.get_event_queue())

# Or send request and wait
await guard.check_alignment(goal_id="...", output={...})
future = await guard.get_result_future()
result = await future.wait(timeout=30.0)
```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4
import time

from pydantic import BaseModel, Field
from overrides import override

from ...base import (
    Agent,
    AgentCapability,
    AgentHandle,
    CapabilityResultFuture,
)
from ..actions.policies import action_executor
from ... import KeyPatternFilter, BlackboardEvent
from ...models import Action, PolicyREPL
from ..events import event_handler, EventProcessingResult


logger = logging.getLogger(__name__)


class GoalAlignmentRequest(BaseModel):
    """Request to check goal alignment of output."""

    requesting_agent_id: str = Field(
        description="ID of agent requesting the alignment check"
    )
    goal_id: str = Field(
        description="ID of the goal being checked"
    )
    output: Any = Field(
        description="Output to check for alignment"
    )

    @staticmethod
    def get_blackboard_key(scope_id: str, requesting_agent_id: str, goal_id: str) -> str:
        """Get blackboard key for goal alignment request.

        Args:
            scope_id: Blackboard scope ID
            requesting_agent_id: ID of agent requesting the check
            goal_id: ID of the goal being checked
        Returns:
            Blackboard key
        """
        return f"{scope_id}:goal_alignment:request:{requesting_agent_id}:{goal_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Get blackboard key pattern for goal alignment requests.

        Args:
            scope_id: Blackboard scope ID

        Returns:
            Blackboard key pattern
        """
        return f"{scope_id}:goal_alignment:request:*:*"


class JointGoalRegistration(BaseModel):
    """Registration request of a joint goal to agents."""

    goal: JointGoal = Field(
        description="Joint goal to register"
    )

    @staticmethod
    def get_blackboard_key(scope_id: str, goal_id: str) -> str:
        """Get blackboard key for joint goal registration.

        Args:
            scope_id: Blackboard scope ID
            goal_id: ID of the goal being registered

        Returns:
            Blackboard key
        """
        return f"{scope_id}:goal_alignment:joint_goal:{goal_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Get blackboard key pattern for joint goal registrations.

        Args:
            scope_id: Blackboard scope ID

        Returns:
            Blackboard key pattern
        """
        return f"{scope_id}:goal_alignment:joint_goal:*"



class GoalAlignment(BaseModel):
    """Assessment of how well output aligns with goal."""

    requesting_agent_id: str = Field(
        description="ID of agent requesting the alignment check"
    )
    goal_id: str = Field(
        description="ID of the goal being checked"
    )
    is_aligned: bool = Field(
        description="Whether output aligns with goal"
    )

    alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Degree of alignment (0.0 = completely off-topic, 1.0 = perfect)"
    )

    goal: str = Field(
        description="Original goal"
    )

    output_summary: str = Field(
        description="Summary of output being checked"
    )

    drift_detected: bool = Field(
        default=False,
        description="Whether goal drift was detected"
    )

    drift_description: str | None = Field(
        default=None,
        description="Description of drift if detected"
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions to realign with goal"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in alignment judgment"
    )

    @staticmethod
    def get_blackboard_key(scope_id: str, requesting_agent_id: str, goal_id: str) -> str:
        """Get blackboard key for storing goal alignment result.

        Args:
            scope_id: Blackboard scope ID
            requesting_agent_id: ID of agent requesting the check
            goal_id: ID of the goal being checked

        Returns:
            Blackboard key
        """
        return f"{scope_id}:goal_alignment:result:{requesting_agent_id}:{goal_id}"


class JointGoal(BaseModel):
    """Shared goal for multi-agent collaboration.

    Represents what the user actually requested, to prevent drift.
    """

    goal_id: str = Field(
        description="Goal identifier"
    )

    description: str = Field(
        description="What needs to be achieved"
    )

    success_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria for successful completion"
    )

    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints on solution"
    )

    participating_agents: list[str] = Field(
        default_factory=list,
        description="Agents committed to this goal"
    )

    created_at: float = Field(
        default_factory=lambda: time.time(),
        description="When goal was created"
    )

    def generate_goal_key(self) -> str:
        """Generate a unique goal key for blackboard storage."""
        return f"joint_goal:{self.goal_id}"


# ============================================================================
# New Pattern: ObjectiveGuardCapability + Agent + CacheAwareActionPolicy
# ============================================================================


class ObjectiveGuardCapability(AgentCapability):
    """Capability for checking goal alignment.

    Works in two modes via the `scope_id` parameter:

    1. **Local mode** (in ObjectiveGuardAgent): Processes alignment requests
       ```python
       capability = ObjectiveGuardCapability(agent=self)
       ```

    2. **Remote mode** (in parent agent): Communicates with child guard agent
       ```python
       handle = await parent.spawn_child_agents(...)[0]
       guard = handle.get_capability(ObjectiveGuardCapability)
       await guard.check_alignment(goal_id="...", output={...})
       future = await guard.get_result_future()
       result = await future.wait(timeout=30.0)
       ```

    Provides @action_executor methods for:
    - check_goal_alignment: Check if output aligns with goal
    - register_goal: Register a new goal to monitor
    """

    def __init__(self, agent: Agent, scope_id: str | None = None):
        """Initialize objective guard capability.

        Args:
            agent: Agent using this capability
            scope_id: Blackboard scope ID. Defaults to agent.agent_id.
        """
        super().__init__(agent, scope_id)
        self.active_goals: dict[str, JointGoal] = {}

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream goal alignment events to the given queue.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        # TODO: We can get either explicit goal alignment check requests or we can snoop
        # on published analysis results to convert them into goal alignment check requests in the action policy.
        # TODO: Stream code analysis result events? Use `AnalysisResult.get_key_pattern()` when available.
        # TODO: Code analyzers even better separate their output results into different categories (e.g.,
        # tentative findings vs. confirmed findings, partial findings vs. rejected findings) so that
        # consistency check can focus on specific categories.
        # TODO: Make scope configurable because agents that register goals need not know the agent_id of the objective guard (decoupling).
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=GoalAlignmentRequest.get_key_pattern(self.scope_id)
            )
        )
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=JointGoalRegistration.get_key_pattern(self.scope_id)
            )
        )

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for goal alignment result.

        Returns:
            Future that resolves with GoalAlignment result
        """
        blackboard = await self.get_blackboard()
        return CapabilityResultFuture(
            result_key=GoalAlignment.get_key_pattern(self.scope_id),
            blackboard=blackboard,
        )

    # -------------------------------------------------------------------------
    # Capability-Specific Request Methods
    # -------------------------------------------------------------------------

    @event_handler(
        pattern=lambda self: f"{self.scope_id}:{GoalAlignmentRequest.get_key_pattern(self.scope_id)}"
    )
    async def handle_goal_alignment_request(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle a new draft result event."""
        request = GoalAlignmentRequest.model_validate(event.value)
        return EventProcessingResult(
            immediate_action=Action(
                action_type="check_goal_alignment",
                parameters={
                    "requesting_agent_id": request.requesting_agent_id,
                    "goal_id": request.goal_id,
                    "output": request.output
                },
            )
        )

    @event_handler(
        pattern=lambda self: f"{self.scope_id}:{JointGoalRegistration.get_key_pattern(self.scope_id)}"
    )
    async def handle_joint_goal_registration(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        registration = JointGoalRegistration.model_validate(event.value)
        return EventProcessingResult(
            immediate_action=Action(
                action_type="register_goal",
                parameters={
                    "goal": registration.goal
                },
            )
        )

    async def check_alignment(
        self,
        goal_id: str,
        output: Any,
    ) -> str:
        """Send a goal alignment check request.

        Args:
            goal_id: ID of goal to check against
            output: Output to check for alignment

        Returns:
            Request ID
        """
        request = GoalAlignmentRequest(
            requesting_agent_id=self.agent.agent_id,
            goal_id=goal_id,
            output=output,
        )
        return await self.send_request(
            request_type="check_alignment",
            request_data=request.model_dump(),
        )

    @action_executor()
    async def register_goal(self, goal: JointGoal) -> None:
        """Register a joint goal to monitor.

        Args:
            goal: Joint goal to track
        """
        self.active_goals[goal.goal_id] = goal

        # Write goal to this capability's scope
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=JointGoalRegistration.get_blackboard_key(self.scope_id, goal.goal_id),
            value=goal.model_dump(),
            agent_id=self.agent.agent_id,
        )

    @action_executor()
    async def check_goal_alignment(
        self,
        requesting_agent_id: str,
        goal_id: str,
        output: Any
    ) -> GoalAlignment:
        """Check if output aligns with goal.

        Args:
            requesting_agent_id: ID of agent requesting the check
            goal_id: ID of goal to check against
            output: Output to check

        Returns:
            Goal alignment assessment
        """
        goal = self.active_goals.get(goal_id)
        if not goal:
            return GoalAlignment(
                requesting_agent_id=requesting_agent_id,
                goal_id=goal_id,
                is_aligned=False,
                alignment_score=0.0,
                goal="Unknown goal",
                output_summary=str(output)[:200],
                drift_detected=True,
                drift_description="Goal not found",
                confidence=1.0
            )

        prompt = self._get_goal_alignment_prompt(goal, output)
        response = self.agent.infer(
            prompt,
            context_page_ids=self.agent.bound_pages,
            max_tokens=500,   # Limit tokens for efficiency - TODO: Make configurable
            temperature=0.2,  # Low temp for consistency - TODO: Make configurable
            json_schema=GoalAlignment.model_json_schema()  # Structured output
        )

        return GoalAlignment.model_validate_json(response.generated_text)

    @action_executor()
    async def publish_alignment_result(
        self,
        result: GoalAlignment
    ) -> None:
        """Publish goal alignment result.

        Writes result to the capability's result key, which resolves
        any CapabilityResultFuture waiting on this capability.

        Args:
            result: Goal alignment check result to publish
        """
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=GoalAlignment.get_blackboard_key(self.scope_id, result.requesting_agent_id, result.goal_id),
            value=result.model_dump(),
            agent_id=self.agent.agent_id,
        )

    def _get_goal_alignment_prompt(
        self,
        goal: JointGoal,
        output: Any
    ) -> str:
        """Construct prompt to check goal alignment."""
        # Use LLM to check alignment
        return f"""Check if this output aligns with the stated goal:

Goal:
{goal.description}

Success Criteria:
{goal.success_criteria}

Output:
{output}

Questions:
1. Does the output address the goal?
2. Is the output relevant to the goal?
3. Are the success criteria being met?
4. Is there goal drift (doing something different)?

Provide:
- alignment_score (0.0-1.0)
- is_aligned (bool)
- drift_detected (bool)
- drift_description (if drift detected)
- suggestions (how to realign)

Return as JSON.
"""


