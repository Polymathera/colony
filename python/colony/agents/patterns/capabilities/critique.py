"""Critique policies for evaluating reasoning outputs.

Critique is a reasoning process that validates conclusions against premises.
It is independent of who produced the output (self, child, peer, parent).

The key insight: critique evaluates logical soundness, not social relationships.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field
from overrides import override
import json
import time
import logging
import asyncio

from ...models import Action, ActionResult, PolicyREPL, AgentSuspensionState
from ..models import Reflection, Critique
from ...base import Agent, AgentCapability
from ..actions import action_executor
from ..events import event_handler, EventProcessingResult, PROCESSED
from ...blackboard import KeyPatternFilter, BlackboardEvent


logger = logging.getLogger(__name__)


class OutputRelationship(str, Enum):
    """Relationship between critiquer and producer of output."""
    SELF = "self"  # Critiquing own work
    CHILD = "child"  # Critiquing subordinate's work
    PEER = "peer"  # Critiquing colleague's work
    PARENT = "parent"  # Critiquing supervisor's work (rare but possible)


@dataclass
class CritiqueContext:
    """Context for critique evaluation.

    Captures what is needed to evaluate logical soundness:
    - Who produced it (for communication)
    - What was the goal
    - What premises/assumptions were used
    - What evidence supports it
    """
    producer_id: str
    relationship: OutputRelationship
    goal: str
    premises: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class CritiqueRequest(BaseModel):
    """Request for critique from another agent.

    Published to blackboard when an agent requests critique assistance.
    """

    request_id: str = Field(
        default_factory=lambda: f"crit_req_{int(time.time() * 1000)}",
        description="Unique request identifier"
    )
    requester_id: str = Field(..., description="ID of agent requesting critique")
    output_to_critique: dict[str, Any] = Field(
        ...,
        description="Output to be critiqued"
    )
    goal: str = Field(..., description="What the output was trying to achieve")
    relationship: OutputRelationship = Field(
        default=OutputRelationship.PEER,
        description="Relationship between requester and responder"
    )
    premises: list[str] = Field(
        default_factory=list,
        description="Premises/assumptions used"
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence supporting the output"
    )
    created_at: float = Field(default_factory=time.time)

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching critique requests in a scope."""
        return f"{scope_id}:critique_request:*"

    def get_blackboard_key(self, scope_id: str) -> str:
        """Get blackboard key for this request."""
        return f"{scope_id}:critique_request:{self.request_id}"


class CritiquePolicy(ABC):
    """Policy for critiquing quality of reasoning and planning outputs.

    Different implementations support different critique strategies:
    - Single LLM evaluation: Use LLM to evaluate quality
    - Multi-LLM debate
    - Iterative refinement with producer
    - Formal verification
    - Metric-based validation: Use quality metrics

    Different implementations:
    - SelfCritic: Agent critiques own work
    - PeerCritic: Another agent provides critique
    """

    @abstractmethod
    async def critique_output(
        self,
        output: Any,
        context: CritiqueContext,
    ) -> Critique:
        """Critique output for logical soundness.

        Args:
            output: The output to critique (any format)
            context: Context about producer, goal, premises

        Returns:
            Critique with validation results
        """
        ...

    @abstractmethod
    async def critique_action_result(
        self, action: Action, result: ActionResult, reflection: Reflection
    ) -> Critique:
        """Critique the quality of action result.

        Args:
            action: Action that was executed
            result: Result of execution
            reflection: Reflection on result

        Returns:
            Critique with quality assessment
        """
        ...


class LLMCritiquePolicy(CritiquePolicy):
    """Critique using single LLM evaluation.

    Simple, fast policy: use LLM to evaluate output against context.
    """

    def __init__(self, llm_cluster_handle: Any, max_tokens=1000):
        """Initialize with LLM cluster handle.

        Args:
            llm_cluster_handle: Handle to LLMCluster deployment
        """
        self.llm_cluster = llm_cluster_handle
        self.max_tokens = max_tokens

    async def critique_output(
        self,
        output: Any,
        context: CritiqueContext,
    ) -> Critique:
        """Critique using LLM evaluation."""
        # Build critique prompt
        output_text = json.dumps(output, indent=2) if not isinstance(output, str) else output

        premises_text = "\n".join(f"- {p}" for p in context.premises) if context.premises else "None specified"
        evidence_text = json.dumps(context.evidence, indent=2) if context.evidence else "None provided"

        prompt = f"""Evaluate the logical soundness of this output:

**Goal**: {context.goal}

**Premises/Assumptions**:
{premises_text}

**Evidence**:
{evidence_text}

**Output to Critique**:
{output_text}

**Your Task**: Evaluate whether conclusions follow from premises and evidence.

As a peer reviewer, evaluate:
1. **Completeness**: Is the analysis complete? Any gaps or missing aspects?
2. **Accuracy**: Are the findings accurate?
3. **Quality**: Is the quality acceptable?
4. **Issues**: Any problems or concerns?
5. **Correctness**: Are the findings correct?
6. **Consistency**: Is it consistent with your own analysis?
7. **Suggestions**: How could it be improved?

**Output Format** (JSON):
{{
    "valid_conclusions": ["conclusions that are logically sound"],
    "invalid_conclusions": ["conclusions that don't follow from premises"],
    "missing_premises": ["premises that should have been considered"],
    "unsupported_claims": ["claims without sufficient evidence"],
    "quality_score": 0.0-1.0,
    "requires_revision": true/false,
    "suggestions": ["specific improvements"],
    "reasoning": "explanation of critique"
}}"""

        # Use LLM to critique
        from ....cluster.models import InferenceRequest

        request = InferenceRequest(
            request_id=f"critique-{context.producer_id}",
            prompt=prompt,
            context_page_ids=[],
            max_tokens=self.max_tokens,
        )

        response = await self.llm_cluster.infer(request)

        # Parse response
        try:
            critique_data = json.loads(response.text)
            return Critique(**critique_data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: low quality if can't parse
            return Critique(
                quality_score=0.3,
                requires_revision=True,
                suggestions=["Could not parse LLM critique response"],
                reasoning=f"LLM response: {response.text[:200]}"
            )


class MetricBasedCritiquePolicy(CritiquePolicy):
    """Critique using quantitative metrics.

    Fast, deterministic policy: check metrics against thresholds.
    Use when you have clear quality criteria.
    """

    def __init__(
        self,
        min_coverage: float = 0.7,
        min_evidence_count: int = 2,
        max_claim_ratio: float = 2.0,
    ):
        """Initialize with metric thresholds.

        Args:
            min_coverage: Minimum coverage of goal (0.0-1.0)
            min_evidence_count: Minimum evidence items required
            max_claim_ratio: Maximum claims per evidence item
        """
        self.min_coverage = min_coverage
        self.min_evidence_count = min_evidence_count
        self.max_claim_ratio = max_claim_ratio

    async def critique_output(
        self,
        output: Any,
        context: CritiqueContext,
    ) -> Critique:
        """Critique using metric validation."""
        issues = []
        suggestions = []

        # Check evidence count
        evidence_count = len(context.evidence) if context.evidence else 0
        if evidence_count < self.min_evidence_count:
            issues.append(f"Insufficient evidence: {evidence_count} < {self.min_evidence_count}")
            suggestions.append(f"Provide at least {self.min_evidence_count} evidence items")

        # Check claim-to-evidence ratio
        # (Simple heuristic: count conclusions in output)
        if isinstance(output, dict):
            conclusions = output.get("conclusions", []) or output.get("findings", [])
            claim_count = len(conclusions) if isinstance(conclusions, list) else 0

            if evidence_count > 0:
                claim_ratio = claim_count / evidence_count
                if claim_ratio > self.max_claim_ratio:
                    issues.append(
                        f"Too many claims per evidence: {claim_ratio:.1f} > {self.max_claim_ratio}"
                    )
                    suggestions.append("Reduce claims or provide more evidence")

        # Check coverage (if metrics provided)
        coverage = context.metadata.get("coverage", 0.0)
        if coverage < self.min_coverage:
            issues.append(f"Low coverage: {coverage:.1%} < {self.min_coverage:.1%}")
            suggestions.append(f"Increase coverage to at least {self.min_coverage:.1%}")

        # Compute quality score
        quality_score = 1.0
        if issues:
            quality_score = max(0.0, 1.0 - (0.2 * len(issues)))

        return Critique(
            quality_score=quality_score,
            requires_revision=len(issues) > 0,
            suggestions=suggestions,
            unsupported_claims=issues,  # Use issues as unsupported claims
            reasoning=f"Metric-based critique: {len(issues)} issues found"
        )


class CriticCapability(AgentCapability):
    """AgentCapability subclass for handling critique requests and requesting critiques.

    Provides:
    - Event-driven handling of incoming CritiqueRequest events via @event_handler
    - Plannable actions for requesting critique from peers/parents via @action_executor
    """
    def __init__(self, agent: Agent):
        super().__init__(agent=agent, scope_id=agent.agent_id)
        # Injected critique policies for different relationships
        self.critique_policy_self: CritiquePolicy | None = self.agent.metadata.parameters.get("critique_policy_self")  # FIXME: Get the policy instances properly
        self.critique_policy_child: CritiquePolicy | None = self.agent.metadata.parameters.get("critique_policy_child")  # FIXME: Get the policy instances properly
        self.critique_policy_peer: CritiquePolicy | None = self.agent.metadata.parameters.get("critique_policy_peer")  # FIXME: Get the policy instances properly
        self.critique_policy_parent: CritiquePolicy | None = self.agent.metadata.parameters.get("critique_policy_parent")  # FIXME: Get the policy instances properly

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for CriticCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for CriticCapability")
        pass

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream CritiqueRequest events to the ActionPolicy queue.

        When a CritiqueRequest arrives, EventDrivenActionPolicy will call
        handle_critique_request() to process it reactively.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        # TODO: We can also snoop on published analysis results to convert them into
        # reputation updates in the action policy.
        # TODO: Make scope configurable because agents that request critiques need not
        # know the agent_id of the critic agent (decoupling).
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(
                pattern=CritiqueRequest.get_key_pattern(scope_id=self.agent.agent_id)
            )
        )

    def _get_request_from_peer_key(self, requester_id: str) -> str:
        """Get blackboard key for peer critique request."""
        return f"{requester_id}:critique_request_from_peer"

    def _get_request_from_parent_key(self, child_id: str) -> str:
        """Get blackboard key for parent critique request."""
        return f"{child_id}:critique_request_from_parent"

    def _get_request_from_child_key(self, parent_id: str) -> str:
        """Get blackboard key for child critique request."""
        return f"{parent_id}:critique_request_from_child"

    def _get_response_from_agent_key(self, responder_id: str, requester_id: str) -> str:
        """Get blackboard key for critique response."""
        return f"{requester_id}:critique_response_from_{responder_id}"

    async def _send_critique_request(
        self,
        from_agent: str,
        to_agent: str,
        relation: Literal["peer2peer", "child2parent", "parent2child"],
        request: dict[str, Any]
    ) -> None:
        """Send critique request to another agent's blackboard."""
        request_key = ""
        if relation == "peer2peer":
            request_key = self._get_request_from_peer_key(to_agent)
        elif relation == "child2parent":
            request_key = self._get_request_from_child_key(to_agent)
        elif relation == "parent2child":
            request_key = self._get_request_from_parent_key(to_agent)

        to_blackboard = await self.agent.get_blackboard(scope="shared", scope_id=to_agent)
        await to_blackboard.write(
            request_key,
            request,
            created_by=from_agent
        )

    async def _send_critique_response(self, responder_id: str, requester_id: str, critique: Critique) -> None:
        # Write response back to requester's blackboard
        requester_blackboard = await self.agent.get_blackboard(scope="shared", scope_id=requester_id)
        await requester_blackboard.write(
            self._get_response_from_agent_key(responder_id, requester_id),
            critique.model_dump()
        )

    def _get_policy_for_relationship(self, relationship: OutputRelationship) -> CritiquePolicy | None:
        """Get the appropriate critique policy for a relationship type."""
        policy_map = {
            OutputRelationship.SELF: self.critique_policy_self,
            OutputRelationship.CHILD: self.critique_policy_child,
            OutputRelationship.PEER: self.critique_policy_peer,
            OutputRelationship.PARENT: self.critique_policy_parent,
        }
        if relationship not in policy_map or policy_map[relationship] is None:
            logger.debug(f"No critique policy configured for relationship {relationship}")
            return None
        return policy_map.get(relationship)

    @event_handler(pattern="{scope_id}:critique_request:*")
    async def handle_critique_request(
        self, event: BlackboardEvent, repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle incoming CritiqueRequest event (reactive/background process).

        Called by EventDrivenActionPolicy.plan_step() when CritiqueRequest events
        arrive on the event queue. The event queue is populated by stream_events_to_queue().

        Flow:
        1. EventDrivenActionPolicy.initialize() calls capability.stream_events_to_queue()
        2. stream_events_to_queue() subscribes to CritiqueRequest events on blackboard
        3. When event arrives, it's queued to EventDrivenActionPolicy._event_queue
        4. plan_step() calls get_next_event(), then broadcasts to @event_handler methods
        5. This method processes the request and sends response back to requester

        Args:
            event: BlackboardEvent containing CritiqueRequest
            repl: PolicyREPL from ActionPolicy (unused but required by decorator)

        Returns:
            EventProcessingResult with PROCESSED status, or None if event not relevant
        """
        try:
            request = CritiqueRequest.model_validate(event.value)
        except Exception as e:
            logger.warning(f"Invalid CritiqueRequest event: {e}")
            return None

        # Get appropriate policy for the relationship
        critique_policy = self._get_policy_for_relationship(request.relationship)

        if not critique_policy:
            logger.debug(f"No critique policy for relationship {request.relationship}")
            return PROCESSED  # Event was for us but we can't handle it

        # Perform critique
        critique = await critique_policy.critique_output(
            output=request.output_to_critique,
            context=CritiqueContext(
                producer_id=request.requester_id,
                relationship=request.relationship,
                goal=request.goal,
                premises=request.premises,
                evidence=request.evidence,
            )
        )

        # Send response back to requester's blackboard
        await self._send_critique_response(
            responder_id=self.agent.agent_id,
            requester_id=request.requester_id,
            critique=critique
        )

        logger.info(
            f"Handled critique request from {request.requester_id}: "
            f"quality={critique.quality_score:.2f}"
        )

        # Return PROCESSED to indicate we handled the event
        # No immediate_action needed - we already sent the response
        return PROCESSED

    @action_executor()
    async def request_critique_from_peer(
        self, peer_id: str, my_output: dict, goal: str, timeout: float = 30.0
    ) -> dict | None:
        """Request critique from a peer agent using event-driven response.

        Posts output to peer's blackboard, waits for event notification.

        This pattern will be abstracted to Agent base class.

        Args:
            peer_id: ID of peer agent to request critique from
            my_output: My output to be critiqued
            goal: What the output was trying to achieve
            timeout: How long to wait for response

        Returns:
            Critique dict from peer, or None if timeout
        """

        return await self._request_critique(
            from_agent=self.agent.agent_id,
            to_agent=peer_id,
            relation="peer2peer",
            my_output=my_output,
            goal=goal,
            timeout=timeout
        )

    @action_executor()
    async def request_critique_from_parent(
        self, my_output: dict, goal: str, timeout: float = 30.0
    ) -> dict | None:
        """Request critique from parent agent using event-driven response.

        Posts output to parent's blackboard, waits for event notification.

        This pattern will be abstracted to Agent base class.

        Args:
            my_output: My output to be critiqued
            goal: What the output was trying to achieve
            timeout: How long to wait for response

        Returns:
            Critique dict from parent, or None if timeout
        """
        parent_id = self.agent.metadata.parent_agent_id
        if not parent_id:
            logger.warning("No parent_id in metadata, cannot request parent critique")
            return None

        return await self._request_critique(
            from_agent=self.agent.agent_id,
            to_agent=parent_id,
            relation="child2parent",
            my_output=my_output,
            goal=goal,
            timeout=timeout
        )

    async def _request_critique(
        self,
        from_agent: str,
        to_agent: str,
        relation: Literal["child2parent", "parent2child", "peer2peer"],
        my_output: dict,
        goal: str,
        timeout: float = 30.0
    ) -> dict | None:
        """Request critique from another agent using event-driven response.

        Posts output to another agent's blackboard, waits for event notification.

        Args:
            from_agent: ID of requesting agent
            to_agent: ID of target agent to request critique from
            relation: Relationship type for critique
            my_output: My output to be critiqued
            goal: What the output was trying to achieve
            timeout: How long to wait for response

        Returns:
            Critique dict from another agent, or None if timeout
        """
        # Setup event for response
        response_event = asyncio.Event()
        critique_data = {}

        response_blackboard = await self.agent.get_blackboard(scope="shared", scope_id=from_agent)

        response_key = self._get_response_from_agent_key(to_agent, from_agent)

        # Subscribe to response event
        async def on_critique_response(event: BlackboardEvent):
            if event.key == response_key:
                critique_data["value"] = event.value
                response_event.set()

        response_blackboard.subscribe(
            on_critique_response,
            filter=KeyPatternFilter(response_key)
        )

        # Write request to target agent's blackboard
        await self._send_critique_request(
            from_agent=from_agent,
            to_agent=to_agent,
            relation=relation,
            request={
                "requester_id": from_agent,
                "output": my_output,
                "goal": goal,
                "premises": [],
                "evidence": my_output.get("evidence", {}),
                "timestamp": time.time()
            },
        )

        try:
            # Wait for response with timeout (event-driven!)
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            logger.info(
                f"Received critique from {to_agent}: "
                f"quality={critique_data['value'].get('quality_score', 0):.2f}"
            )
            return critique_data["value"]
        except asyncio.TimeoutError:
            logger.warning(f"No critique received from {to_agent} after {timeout}s")
            return None
        finally:
            # Unsubscribe after getting response
            response_blackboard.unsubscribe(on_critique_response)

    @action_executor()
    async def critique_self_output(self, my_output: dict, goal: str) -> Critique:
        """Critique own output using self-critique policy.

        This is a convenience @action_executor that delegates to critique_output()
        with SELF relationship. Enables LLM planner to invoke self-critique.

        Args:
            my_output: My output to be critiqued
            goal: What the output was trying to achieve

        Returns:
            Critique of the output
        """
        return await self.critique_output(
            output=my_output,
            context=CritiqueContext(
                producer_id=self.agent.agent_id,
                relationship=OutputRelationship.SELF,
                goal=goal,
                premises=[],
                evidence=my_output.get("evidence", {}),
                metadata=my_output.get("metadata", {})
            )
        )

    @action_executor()
    async def critique_output(
        self,
        output: Any,
        context: CritiqueContext,
    ) -> Critique:
        """Critique output for logical soundness.

        Delegates to configured critique policy or uses default output critique.

        Args:
            output: The output to critique
            context: Context about producer, goal, premises

        Returns:
            Critique with validation results
        """
        # Get appropriate policy based on relationship
        policy = self._get_policy_for_relationship(context.relationship)

        if policy:
            return await policy.critique_output(output, context)

        # No policy configured - use default output critique
        return self._default_output_critique(output, context)

    def _default_output_critique(self, output: Any, context: CritiqueContext) -> Critique:
        """Default critique for outputs when no policy is configured.

        Used by critique_output() when no relationship-specific policy exists.
        Performs basic quality checks on the output and context.

        Args:
            output: The output to critique
            context: CritiqueContext with goal, evidence, premises

        Returns:
            Critique with basic quality assessment
        """
        issues: list[str] = []
        suggestions: list[str] = []
        quality_score = 0.8

        # Check if output is empty
        if not output:
            issues.append("Output is empty")
            quality_score = 0.3
            suggestions.append("Produce meaningful output")

        # Check evidence sufficiency
        if context.evidence:
            evidence_count = len(context.evidence)
            if evidence_count < 2:
                issues.append(f"Insufficient evidence: {evidence_count} items")
                quality_score *= 0.8
                suggestions.append("Gather more evidence to support claims")
        else:
            issues.append("No evidence provided")
            quality_score *= 0.7
            suggestions.append("Provide evidence to support conclusions")

        # Check premises
        if not context.premises:
            quality_score *= 0.95  # Minor penalty for missing premises

        # Check metadata for coverage info (if available)
        coverage = context.metadata.get("coverage", 1.0) if context.metadata else 1.0
        if coverage < 0.5:
            issues.append(f"Low coverage: {coverage:.1%}")
            quality_score *= 0.8
            suggestions.append("Analyze more items for better coverage")

        requires_revision = quality_score < 0.7 or len(issues) >= 2

        return Critique(
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            requires_revision=requires_revision,
            confidence=0.7,
            reasoning=f"Default output critique: {len(issues)} issues found",
        )

    async def _gather_critique_context_from_memory(
        self, action: Action, result: ActionResult, max_results: int = 10, context: CritiqueContext | None = None
    ) -> dict[str, Any]:
        """Query memory for critique-relevant context.

        Queries the agent's memory for:
        1. Recent actions with their outcomes and learnings
        2. Prior critiques for pattern detection
        3. Reflection learnings that may inform critique

        Also includes the immediate result's learnings as fallback.

        Args:
            action: Action that was executed
            result: Result of execution
            max_results: Maximum number of memory entries to retrieve
            context: Optional CritiqueContext to include in the query

        Returns:
            Dict of gathered context for critique including memory-sourced data
        """
        # =========================================================================
        # TODO: Use AgentContextEngine to gather context instead of direct memory queries.
        # TODO: Use context.relationship to filter relevant memory entries (e.g.,
        # results of child agents).
        # =========================================================================
        from ..memory.capability import MemoryCapability
        from ..memory.types import MemoryQuery

        context: dict[str, Any] = {
            "action_type": str(action.action_type),
            "success": result.success,
            "error": result.error,
            "recent_actions": [],
            "prior_critiques": [],
        }

        # Extract immediate learnings from result (fallback if memory unavailable)
        if result.output:
            learnings = result.output.get("_critique_learnings", {})
            if learnings:
                context["learnings"] = learnings

            reflection_learnings = result.output.get("_reflection_learnings", {})
            if reflection_learnings:
                context["reflection_learnings"] = reflection_learnings

        # Try to get memory capability for richer context
        try:
            memory: MemoryCapability = self.agent.get_capability(
                MemoryCapability.get_capability_name()
            )
        except (KeyError, AttributeError):
            logger.debug("No memory capability available for critique context")
            return context

        # Query for recent actions with critique context
        try:
            action_entries = await memory.recall(
                MemoryQuery(
                    tags={"action"},
                    max_results=max_results,
                    max_age_seconds=3600,  # Last hour
                )
            )

            for entry in action_entries:
                if hasattr(entry, "value") and isinstance(entry.value, dict):
                    action_data = entry.value
                    context["recent_actions"].append({
                        "action_type": action_data.get("action_type"),
                        "success": action_data.get("success"),
                        "critique_learnings": action_data.get("_critique_learnings", {}),
                        "reflection_learnings": action_data.get("_reflection_learnings", {}),
                        "timestamp": entry.metadata.get("created_at"),
                    })
                elif hasattr(entry, "value") and hasattr(entry.value, "action_type"):
                    # It's an Action object
                    mem_action = entry.value
                    critique_learnings = {}
                    reflection_learnings = {}
                    if mem_action.result and mem_action.result.output:
                        critique_learnings = mem_action.result.output.get("_critique_learnings", {})
                        reflection_learnings = mem_action.result.output.get("_reflection_learnings", {})
                    context["recent_actions"].append({
                        "action_type": str(mem_action.action_type),
                        "success": mem_action.result.success if mem_action.result else None,
                        "critique_learnings": critique_learnings,
                        "reflection_learnings": reflection_learnings,
                        "timestamp": mem_action.created_at,
                    })

        except Exception as e:
            logger.warning(f"Failed to query memory for action context: {e}")

        # Query for prior critiques to detect patterns
        try:
            critique_entries = await memory.recall(
                MemoryQuery(
                    tags={"critique"},
                    max_results=5,
                    max_age_seconds=7200,  # Last 2 hours
                )
            )

            for entry in critique_entries:
                if hasattr(entry, "value"):
                    critique_data = entry.value
                    if isinstance(critique_data, dict):
                        context["prior_critiques"].append({
                            "quality_score": critique_data.get("quality_score"),
                            "issues": critique_data.get("issues", []),
                            "requires_replanning": critique_data.get("requires_replanning"),
                            "timestamp": entry.metadata.get("created_at"),
                        })
                    elif hasattr(critique_data, "quality_score"):
                        # It's a Critique object
                        context["prior_critiques"].append({
                            "quality_score": critique_data.quality_score,
                            "issues": critique_data.issues,
                            "requires_replanning": critique_data.requires_replanning,
                            "timestamp": getattr(critique_data, "created_at", None),
                        })

        except Exception as e:
            logger.warning(f"Failed to query memory for prior critiques: {e}")

        return context

    def _default_critique(
        self,
        action: Action,
        result: ActionResult,
        mem_context: dict[str, Any],
    ) -> Critique:
        """Generate default critique based on action result and memory context.

        Uses:
        - Immediate result learnings (_critique_learnings, _reflection_learnings)
        - Recent action history from memory for pattern detection
        - Prior critiques from memory to avoid repeated issues

        Args:
            action: Action that was executed
            result: Result of execution
            mem_context: Gathered context from memory (includes recent_actions, prior_critiques)

        Returns:
            Default Critique based on metrics and memory patterns
        """
        issues: list[str] = []
        suggestions: list[str] = []
        quality_score = 1.0

        # Check for failures
        if not result.success:
            quality_score = 0.2
            issues.append(f"Action failed: {result.error}")
            suggestions.append("Retry action or try alternative approach")

        # Check learnings for quality indicators (from immediate result)
        learnings = mem_context.get("learnings", {})
        if learnings:
            coverage = learnings.get("coverage", 1.0)
            if coverage < 0.5:
                quality_score *= 0.7
                issues.append(f"Low coverage: {coverage:.1%}")
                suggestions.append("Analyze more items for better coverage")

            confidence = learnings.get("confidence", 1.0)
            if confidence < 0.5:
                quality_score *= 0.8
                issues.append(f"Low confidence: {confidence:.1%}")
                suggestions.append("Gather more evidence")

            # Impact analysis specific learnings (migrated from ImpactAnalysisCritic)
            if "has_impacts" in learnings and not learnings["has_impacts"]:
                issues.append("No impacts identified - may be incomplete")
                suggestions.append("Verify no impacts or search broader context")

            low_confidence_count = learnings.get("low_confidence_count", 0)
            if low_confidence_count > 0:
                issues.append(f"{low_confidence_count} low-confidence impacts")
                suggestions.append("Gather more evidence for uncertain impacts")

            # Synthesis quality specific (migrated from ClusterAnalyzerCritic)
            summary_quality = learnings.get("summary_quality")
            if summary_quality is not None and summary_quality < 0.6:
                issues.append("Summary quality is low")
                suggestions.append("Regenerate summary with more detail")

        # Check reflection learnings
        reflection = mem_context.get("reflection_learnings", {})
        if reflection.get("needs_more_info"):
            quality_score *= 0.9
            issues.append("More information needed")
            suggestions.append("Fetch additional context")

        # Analyze recent action patterns from memory
        recent_actions = mem_context.get("recent_actions", [])
        if recent_actions:
            # Check for repeated failures
            recent_failures = [a for a in recent_actions if a.get("success") is False]
            if len(recent_failures) >= 3:
                quality_score *= 0.8
                issues.append(f"Pattern: {len(recent_failures)} recent failures detected")
                suggestions.append("Consider alternative approach or escalate")

            # Check if similar actions have had low quality
            similar_action_learnings = [
                a.get("critique_learnings", {})
                for a in recent_actions
                if a.get("action_type") == str(action.action_type)
            ]
            low_quality_similar = [
                l for l in similar_action_learnings
                if l.get("coverage", 1.0) < 0.5 or l.get("confidence", 1.0) < 0.5
            ]
            if low_quality_similar:
                quality_score *= 0.9
                issues.append(f"Pattern: {len(low_quality_similar)} similar actions had quality issues")

        # Learn from prior critiques
        prior_critiques = mem_context.get("prior_critiques", [])
        if prior_critiques:
            # Check if replanning was frequently required
            replanning_needed = [c for c in prior_critiques if c.get("requires_replanning")]
            if len(replanning_needed) >= 2:
                suggestions.append("Prior critiques suggest plan quality may need improvement")

            # Aggregate common issues
            all_prior_issues = []
            for c in prior_critiques:
                all_prior_issues.extend(c.get("issues", []))
            if all_prior_issues:
                # Note recurring patterns (simplified - real impl would use similarity)
                suggestions.append(f"Review {len(all_prior_issues)} issues from prior critiques")

        requires_replanning = quality_score < 0.7

        return Critique(
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions,
            requires_replanning=requires_replanning,
            confidence=0.8,
            reasoning=f"Default critique: {len(issues)} issues found",
        )

    @action_executor(action_key="critique_action_result")
    async def critique_action_result(
        self,
        action: Action,
        result: ActionResult,
    ) -> Critique:
        """Critique action result using memory-gathered context.

        Queries memory for critique-relevant learnings written by action executors,
        then applies critique policy (or default critique if none configured).

        This method is an @action_executor (plannable by ActionPolicy)

        Args:
            action: Action that was executed
            result: Result of execution

        Returns:
            Critique with quality assessment
        """
        context = CritiqueContext(
            producer_id=self.agent.agent_id,
            relationship=OutputRelationship.SELF,
            goal=f"Execute {action.action_type}",
            premises=[],
            evidence={},
        )

        # Gather context from memory/result
        mem_context = await self._gather_critique_context_from_memory(action, result, context=context)
        context.evidence = mem_context.get("learnings", {})

        # Apply critique policy if configured
        if self.critique_policy_self:
            critique = await self.critique_policy_self.critique_output(
                output=result.output or {},
                context=context,
            )
        else:
            # Default critique based on metrics
            critique = self._default_critique(action, result, mem_context)

        logger.info(
            f"Critiqued action {action.action_type}: "
            f"quality={critique.quality_score:.2f}, replanning={critique.requires_replanning}"
        )
        return critique


