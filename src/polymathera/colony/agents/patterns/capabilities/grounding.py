"""GroundingAgent: Validates claims against evidence.

The GroundingAgent is a meta-agent that:
- Checks claims against external or retrieved context
- Uses PageQueryRoutingPolicy and IncrementalQueryProcessor to fetch evidence - TODO: Not implemented
- Validates that INFORM messages include proper evidence
- Challenges unsupported claims

Role in games:
- Hypothesis Game: Provides evidence during GROUND phase
- General validation: Checks any claim for grounding

Integration:
- Subscribes to INFORM messages in blackboard
- Uses existing query infrastructure for evidence retrieval
- Publishes validation results

Programming Model (AgentHandle Pattern):
---------------------------------------
The GroundingAgent demonstrates the `AgentHandle` pattern for parent-child
agent communication. When a parent spawns a GroundingAgent, it receives an
`AgentHandle` that provides three communication modes:

1. **Wait for result** - Block until grounding completes:
   ```python
   handle = (await owner.spawn_child_agents(
       blueprints=[GroundingAgent.bind(
           capability_blueprints=[GroundingCapability.bind()],
       )],
       return_handles=True,
   ))[0]
   grounding = handle.get_capability(GroundingCapability)
   future = await grounding.get_result_future()
   result = await future.wait(timeout=30.0)
   ```

2. **Stream events** - Receive async updates in event loop:
   ```python
   grounding = handle.get_capability(GroundingCapability)
   await grounding.stream_events_to_queue(self.get_event_queue())
   # Events now flow to parent's action policy
   ```

3. **Send requests** - Call capability methods:
   ```python
   await grounding.send_request(
       request_type="ground_claim",
       request_data={"claim": "...", "context": {...}}
   )
   ```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import uuid

from pydantic import BaseModel, Field
from overrides import override

from ...base import (
    Agent,
    AgentCapability,
    CapabilityResultFuture,
)
from ...scopes import BlackboardScope, get_scope_prefix
from ..attention import PageQuery
from .validation import ValidationResult, ValidationIssue
from ..actions.policies import action_executor
from ...models import Action, PolicyREPL, AgentSuspensionState
from ... import BlackboardEvent
from ..events import event_handler, EventProcessingResult
from ...blackboard.protocol import GroundingProtocol


logger = logging.getLogger(__name__)


class GroundingRequest(BaseModel):
    """Request to ground a claim."""

    claim_id: str = Field(
        description="Unique identifier for the claim"
    )

    claim: str = Field(
        description="Claim to ground"
    )

    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Context for grounding"
    )

    initial_pages: list[str] = Field(
        default_factory=list,
        description="Initial pages to consider for evidence. Search may expand beyond these."
    )

    evidence_provided: list[str] = Field(
        default_factory=list,
        description="Evidence already provided"
    )

    requesting_agent_id: str | None = Field(
        default=None,
        description="Agent requesting grounding"
    )


class GroundingResult(BaseModel):
    """Result of grounding check."""

    claim: str = Field(
        description="Claim that was grounded"
    )

    is_grounded: bool = Field(
        description="Whether claim is grounded in evidence"
    )

    evidence_found: list[str] = Field(
        default_factory=list,
        description="Evidence found supporting claim"
    )

    evidence_against: list[str] = Field(
        default_factory=list,
        description="Evidence contradicting claim"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in grounding judgment"
    )

    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="Issues with claim grounding"
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for better grounding"
    )


# ============================================================================
# New Pattern: GroundingCapability + Agent + CacheAwareActionPolicy
# ============================================================================


class GroundingCapability(AgentCapability):
    """Agentic capability for validating/grounding claims against evidence.

    Works in two modes via the `scope_id` parameter:

    1. **Local mode** (in GroundingAgent): Processes grounding requests
       ```python
       capability = GroundingCapability(agent=self)  # scope_id = agent.agent_id
       await capability.stream_events_to_queue(policy.get_event_queue())
       ```

    2. **Remote mode** (in parent agent): Communicates with child GroundingAgent
       ```python
       handle = await parent.spawn_child_agent_with_handle(...)
       grounding = handle.get_capability(GroundingCapability)
       # grounding.scope_id = child_agent_id

       await grounding.stream_events_to_queue(self.get_event_queue())
       future = await grounding.get_result_future()
       result = await future.wait(timeout=30.0)
       ```

    Provides @action_executor methods for:
    - generate_grounding_query: Validate a claim against evidence by generating a query
    - publish_grounding_result: Publish grounding result to blackboard

    Use with EventDrivenActionPolicy or CacheAwareActionPolicy.

    Responsibilities:
    - Check all INFORM messages for evidence
    - Fetch additional evidence when needed
    - Challenge unsupported claims
    - Provide evidence for claims in hypothesis games

    Uses:
    - PageQueryRoutingPolicy for finding evidence - TODO: Not implemented
    - IncrementalQueryCapability for iterative evidence gathering - TODO: Not implemented
    - ValidationPolicy for grounding checks
    """

    protocols = [GroundingProtocol]
    input_patterns = [GroundingProtocol.request_pattern(namespace="grounding")]

    def __init__(self, agent: Agent, scope: BlackboardScope = BlackboardScope.COLONY):
        """Initialize grounding capability.

        Args:
            agent: Agent using this capability
            scope: Blackboard scope. Defaults to BlackboardScope.COLONY.
                Parent agents set this to child_agent_id to communicate
                with the child's grounding capability.
        """
        super().__init__(agent, scope_id=get_scope_prefix(scope, agent))

    def get_action_group_description(self) -> str:
        return (
            "Claim Grounding — validates claims against evidence via query generation. "
            "Workflow: generate a grounding query for a claim, then publish the result. "
            "Triggered by INFORM messages containing claims that need evidence. "
            "Challenges unsupported claims and fetches additional evidence as needed."
        )

    def _get_result_key(self) -> str:
        """Get blackboard key for this capability's result."""
        return GroundingProtocol.result_key("default", namespace="grounding")

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for MultiHopSearchCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for MultiHopSearchCapability")
        pass

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for grounding result.

        Each GroundingCapability instance represents ONE grounding task.
        This returns a future that resolves when grounding completes.

        Returns:
            Future that resolves with GroundingResult
        """
        blackboard = await self.get_blackboard()
        return CapabilityResultFuture(
            result_key=self._get_result_key(),
            blackboard=blackboard,
        )

    # -------------------------------------------------------------------------
    # Capability-Specific Request Methods
    # -------------------------------------------------------------------------

    @event_handler(pattern=GroundingProtocol.request_pattern(namespace="grounding"))
    async def handle_grounding_request(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle a grounding request event.

        Parses the event, queues the request, and returns an action
        to process it.
        """
        # Parse request from event
        requesting_agent_id = event.key.split(":")[0]
        request_data = event.value
        request_data["requesting_agent_id"] = requesting_agent_id
        request = GroundingRequest(**request_data)

        # Return action to process the request
        # The dispatcher will invoke GroundingCapability.generate_grounding_query(request=action.parameters["request"])
        return EventProcessingResult(
            immediate_action=Action(
                action_type="generate_grounding_query",
                parameters={
                    "request": request
                },
                metadata={}
            )
        )

    async def ground_claim(
        self,
        claim: str,
        context: dict[str, Any] | None = None,
        initial_pages: list[str] | None = None,
    ) -> str:
        """Send a grounding request to the capability.

        This is a convenience method that wraps `send_request()`.

        Args:
            claim: Claim to ground
            context: Optional context for grounding
            initial_pages: Optional initial pages for evidence search

        Returns:
            Request ID
        """
        request = GroundingRequest(
            claim_id=f"claim_{self.scope_id}",
            claim=claim,
            context=context or {},
            initial_pages=initial_pages or [],
            requesting_agent_id=self.agent.agent_id,
        )
        return await self.send_request(
            request_type="ground",
            request_data=request.model_dump(),
        )

    @action_executor(writes=["initial_pages", "query"])
    async def generate_grounding_query(self, request: GroundingRequest) -> dict[str, Any]:
        """Ground a claim by finding evidence. Generate a query that can be
        used to fetch evidence by incrementally searching and traversing the page graph.

        Args:
            request: Grounding request

        Returns:
            Grounding result
        """
        # The ActionPolicy will use these outputs to drive the IncrementalQueryCapability.get_answer action
        # TODO: This is a placeholder implementation. Real implementation would generate queries based on claim and context.
        # TODO: Integrate with IncrementalQueryCapability to fetch evidence iteratively.
        return {
            "initial_pages": request.initial_pages,
            "query": PageQuery(query_text=f"Find evidence for claim: {request.claim}")
        }

    @action_executor()
    async def publish_grounding_result(
        self,
        request: GroundingRequest,
        result: GroundingResult
    ) -> None:
        """Publish grounding result.

        Writes result to the capability's result key, which resolves
        any CapabilityResultFuture waiting on this capability.

        Args:
            request: Grounding request
            result: Grounding result to publish
        """
        # Determine if grounded
        is_grounded = len(result.evidence_found) >= 1 and len(result.evidence_against) == 0

        if not is_grounded:
            if len(result.evidence_found) == 0:
                result.issues.append(ValidationIssue(
                    severity="high",
                    issue_type="no_evidence",
                    description=f"No evidence found for claim: {request.claim}",
                    suggestion="Provide explicit evidence references"
                ))
            if result.evidence_against:
                result.issues.append(ValidationIssue(
                    severity="critical",
                    issue_type="contradictory_evidence",
                    description="Found evidence contradicting claim",
                    suggestion="Revise claim or explain contradiction"
                ))

            result.suggestions.extend([
                "Add code location references",
                "Cite specific files/lines"
            ])

        # Write result to capability's result key
        # This resolves any CapabilityResultFuture.wait() calls
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=self._get_result_key(),
            value=result.model_dump(),
            agent_id=self.agent.agent_id,
        )

