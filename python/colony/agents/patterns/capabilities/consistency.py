"""ConsistencyAgent: Detects cross-hypothesis contradictions.

The ConsistencyAgent is a meta-agent that:
- Monitors all hypotheses and claims
- Detects contradictions between results
- Uses ContradictionResolver to resolve conflicts
- Ensures logical consistency across the knowledge base

Role:
- Subscribes to hypothesis and finding events
- Compares new items against existing knowledge
- Flags contradictions for resolution
- Maintains consistency invariants

Programming Model (AgentHandle Pattern):
---------------------------------------
```python
# Spawn consistency agent with handle
handle = await owner.spawn_child_agents(
    agent_specs=[AgentSpawnSpec(agent_type="...ConsistencyAgent")],
    capability_types=[ConsistencyCapability],
)[0]

# Get capability and communicate
consistency = handle.get_capability(ConsistencyCapability)
await consistency.stream_events_to_queue(self.get_event_queue())

# Wait for result
future = await consistency.get_result_future()
result = await future
```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import time

from pydantic import BaseModel, Field
from overrides import override

from ...base import (
    Agent,
    AgentCapability,
    AgentHandle,
    CapabilityResultFuture,
)
from .validation import ValidationCapability, Contradiction
from ..scope import ScopeAwareResult, AnalysisScope
from ..actions.policies import (
    action_executor,
)
from ...models import Action, PolicyREPL, AgentSuspensionState
from ... import KeyPatternFilter, BlackboardEvent
from ..games.epistemic import EpistemicLayer
from ..events import event_handler, EventProcessingResult

logger = logging.getLogger(__name__)


class ConsistencyCheckRequest(BaseModel):
    """Request to check consistency of a new result."""

    requesting_agent_id: str = Field(
        description="ID of agent requesting the consistency check"
    )

    new_result: ScopeAwareResult[Any] = Field(
        description="New result to check for consistency"
    )

    @staticmethod
    def get_blackboard_key(scope_id: str, requesting_agent_id: str, result_id: str) -> str:
        """Get blackboard key for storing consistency check request.

        Args:
            scope_id: Blackboard scope ID
            requesting_agent_id: ID of agent requesting the check
            result_id: ID of the result being checked
        Returns:
            Blackboard key
        """
        return f"{scope_id}:consistency_check:request:{requesting_agent_id}:{result_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Get blackboard key pattern for consistency check requests.

        Args:
            scope_id: Blackboard scope ID

        Returns:
            Key pattern
        """
        return f"{scope_id}:consistency_check:request:*:*"



class ConsistencyEvent(BaseModel):
    """Event containing request to check consistency of a new result."""

    request: ConsistencyCheckRequest
    agent_id: str = Field(
        description="ID of agent that created the event"
    )
    version: int = Field(
        description="Version of the event"
    )
    tags: set[str] = Field(
        default_factory=set,
        description="Tags associated with the event"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the event"
    )

    @classmethod
    def from_blackboard_event(cls, event: BlackboardEvent) -> ConsistencyEvent:
        request = ConsistencyCheckRequest(event.value) if event.value else None
        return cls(
            agent_id=event.agent_id,
            request=request,
            version=event.version,
            tags=event.tags,
            metadata=event.metadata
        )


class ConsistencyCheck(BaseModel):
    """Result of consistency checking."""

    is_consistent: bool = Field(
        description="Whether results are consistent"
    )

    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Contradictions found"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in consistency judgment"
    )

    checked_results: list[str] = Field(
        default_factory=list,
        description="Result IDs that were checked"
    )

    timestamp: float = Field(
        default_factory=lambda: time.time(),
        description="When check was performed"
    )

    @staticmethod
    def get_blackboard_key(scope_id: str, requesting_agent_id: str, result_id: str) -> str:
        """Get blackboard key for storing consistency check result.

        Args:
            scope_id: Blackboard scope ID
            requesting_agent_id: ID of agent requesting the check
            result_id: ID of the result being checked

        Returns:
            Blackboard key
        """
        return f"{scope_id}:consistency_check:result:{requesting_agent_id}:{result_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Get blackboard key pattern for consistency check results.

        Args:
            scope_id: Blackboard scope ID

        Returns:
            Key pattern
        """
        return f"{scope_id}:consistency_check:result:*:*"



class ConsistencyCapability(AgentCapability):
    """Capability for checking consistency across results.

    Works in two modes via the `scope_id` parameter:

    1. **Local mode** (in ConsistencyAgent): Processes consistency check requests
       ```python
       capability = ConsistencyCapability(agent=self)  # scope_id = agent.agent_id
       ```

    2. **Remote mode** (in parent agent): Communicates with child ConsistencyAgent
       ```python
       handles = await parent.spawn_child_agents(...)
       consistency = handles[0].get_capability(ConsistencyCapability)
       await consistency.stream_events_to_queue(self.get_event_queue())
       future = await consistency.get_result_future()
       result = await future
       ```

    Provides @action_executor methods for:
    - check_consistency: Check new result against existing results
    - resolve_contradictions: Attempt to resolve detected contradictions
    """

    def __init__(self, agent: Agent, scope_id: str | None = None):
        """Initialize consistency capability.

        Args:
            agent: Agent using this capability
            scope_id: Blackboard scope ID. Defaults to agent.agent_id.
        """
        super().__init__(agent, scope_id)
        self.epistemic_layer = EpistemicLayer(self.agent)  # TODO: Currently unused
        self.checked_results: dict[str, ScopeAwareResult] = {}

    def _get_validation_capability(self) -> ValidationCapability | None:
        """Get ValidationCapability from agent (optional).

        Returns:
            ValidationCapability or None if not configured
        """
        return self.agent.get_capability_by_type(ValidationCapability)

    def _get_event_pattern(self) -> str:
        """Get pattern for consistency events."""
        return f"{self.scope_id}:consistency:*"

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ConsistencyCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ConsistencyCapability")
        pass

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream consistency events to the given queue.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        # TODO: We can get either explicit consistency check requests or we can snoop
        # on published analysis results to convert them into consistency check requests in the action policy.
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=ConsistencyCheckRequest.get_key_pattern(self.scope_id)
            )
        )

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for consistency check result.

        Returns:
            Future that resolves with ConsistencyCheck result
        """
        blackboard = await self.get_blackboard()
        return CapabilityResultFuture(
            result_key=ConsistencyCheck.get_key_pattern(self.scope_id),
            blackboard=blackboard,
        )

    # -------------------------------------------------------------------------
    # Capability-Specific Request Methods
    # -------------------------------------------------------------------------

    async def check_result_consistency(
        self,
        new_result: ScopeAwareResult[Any],
    ) -> str:
        """Send a consistency check request.

        Args:
            new_result: New result to check for consistency

        Returns:
            Request ID
        """
        request = ConsistencyCheckRequest(
            requesting_agent_id=self.agent.agent_id,
            new_result=new_result,
        )
        return await self.send_request(
            request_type="check_consistency",
            request_data=request.model_dump(),
        )

    @action_executor()
    async def check_consistency(
        self,
        requesting_agent_id: str,
        new_result: ScopeAwareResult[Any]
    ) -> ConsistencyCheck:
        """Check consistency of new result against existing results.

        Args:
            requesting_agent_id: ID of agent requesting the check
            new_result: New result to check

        Returns:
            Consistency check result
        """
        # Get relevant existing results (those sharing related shards)
        relevant_results = self._get_relevant_results(new_result)

        # Detect contradictions
        all_results = [new_result] + list(relevant_results.values())

        contradictions = []
        validation_cap = self._get_validation_capability()
        if validation_cap:
            contradictions = await validation_cap.detect_contradictions(all_results)

        is_consistent = len(contradictions) == 0

        # Add to checked results
        self.checked_results[new_result.result_id] = new_result

        return ConsistencyCheck(
            is_consistent=is_consistent,
            contradictions=contradictions,
            confidence=0.8 if is_consistent else 0.6,
            checked_results=[r.result_id for r in all_results]
        )

    @action_executor()
    async def publish_consistency_result(
        self,
        result: ConsistencyCheck
    ) -> None:
        """Publish consistency result.

        Writes result to the capability's result key, which resolves
        any CapabilityResultFuture waiting on this capability.

        Args:
            result: Consistency check result to publish
        """
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=ConsistencyCheck.get_blackboard_key(self.scope_id, result.requesting_agent_id, result.result_id),
            value=result.model_dump(),
            agent_id=self.agent.agent_id,
        )

    def _get_relevant_results(
        self,
        new_result: ScopeAwareResult[Any]
    ) -> dict[str, ScopeAwareResult]:
        """Get results relevant for consistency checking.

        Args:
            new_result: New result

        Returns:
            Dictionary of relevant results
        """
        relevant_results = {}

        # Results that share related shards
        for result_id, result in self.checked_results.items():
            # Check if they share related shards
            shared_shards = set(new_result.scope.related_shards) & set(result.scope.related_shards)
            if shared_shards:
                relevant_results[result_id] = result

        return relevant_results

    @event_handler(pattern="{scope_id}:*")
    async def handle_consistency_check_event(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Process consistency check events and enrich planning context.

        This event handler:
        1. Parses blackboard events to consistency check events
        2. Verifies the event is a consistency check request
        3. Optionally Enriches context with consistency check state (including version for OCC)
        4. Optionally returns rule-based immediate actions

        Args:
            event: Blackboard event to process
            scope: Policy scope to enrich

        Returns:
            EventProcessingResult if processed, None if not relevant
        """
        try:
            # Parse event
            consistency_event = ConsistencyEvent.from_blackboard_event(event)
            if consistency_event is None:
                return None  # Not a consistency check event

            request = consistency_event.request

            return EventProcessingResult(
                immediate_action=Action(
                    action_type="check_consistency",
                    parameters={
                        "requesting_agent_id": request.requesting_agent_id,
                        "new_result": request.new_result
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to handle consistency event: {e}")
            return None


