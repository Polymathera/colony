"""Epistemic capability for knowledge and belief tracking.

Based on epistemic logic from Shoham & Leyton-Brown (Ch 13-14):
"Representing 'Who Knows What' on the Blackboard"

This module implements:
- EpistemicStatus: Tracks which agents believe what
- Common knowledge approximation
- Belief propagation and updates
- Intention tracking (BDI architecture)

The epistemic capability enables:
- Tracking which propositions are "common knowledge"
- Prioritizing validation of commonly believed but unvalidated claims
- Detecting belief conflicts between agents
- Preventing goal drift through intention monitoring

---------------------------------------------------------------
NOTE: When we refer to an "agent" regarding its epistemic state, we mean the combination of:
- The pre-trained LLM reasoning model underlying its behavior
- The specific knowledge private to that agent:
    - Blackboard data,
    - Context pages it is bound to or it considered when forming beliefs,
    - Any fine-tuning or prompt engineering that shapes its reasoning.

That is why an agent can "believe" something even if the underlying LLM does not have that knowledge inherently.
---------------------------------------------------------------
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ...base import Agent, AgentCapability
from ...models import AgentSuspensionState
from ..actions.policies import action_executor
from ...blackboard.blackboard import EnhancedBlackboard
from ...blackboard.protocol import EpistemicProtocol
from ...scopes import BlackboardScope, get_scope_prefix

class BeliefStrength(str, Enum):
    """Strength of belief in a proposition."""

    CERTAIN = "certain"  # Confidence >= 0.95
    STRONG = "strong"  # Confidence >= 0.8
    MODERATE = "moderate"  # Confidence >= 0.6
    WEAK = "weak"  # Confidence >= 0.4
    UNCERTAIN = "uncertain"  # Confidence < 0.4


class EpistemicStatus(BaseModel):
    """Tracks epistemic status of a proposition.

    For each proposition ("function f leaks tainted data"), tracks:
    - Which agents believe it (and how strongly)
    - Which agents disbelieve it
    - Whether it's common knowledge

    Examples:
        Commonly believed fact:
        ```python
        status = EpistemicStatus(
            proposition_id="prop_auth_uses_jwt",
            proposition="Authentication system uses JWT tokens",
            believers={"analyzer_001": 0.95, "security_agent": 0.90},
            disbelievers={},
            common_knowledge=True,
            evidence_count=5
        )
        ```

        Disputed claim:
        ```python
        status = EpistemicStatus(
            proposition_id="prop_no_sql_injection",
            proposition="System has no SQL injection vulnerabilities",
            believers={"analyzer_001": 0.70},
            disbelievers={"skeptic_002": 0.85},
            common_knowledge=False,
            evidence_count=2,
            conflicts=["Skeptic found unsanitized input at line 42"]
        )
        ```
    """

    proposition_id: str = Field(
        description="Unique proposition identifier"
    )

    proposition: str = Field(
        description="The proposition (claim, fact, hypothesis)"
    )

    # Belief tracking
    believers: dict[str, float] = Field(
        default_factory=dict,
        description="Agent ID -> confidence mapping for believers"
    )

    disbelievers: dict[str, float] = Field(
        default_factory=dict,
        description="Agent ID -> confidence mapping for disbelievers"
    )

    uncertain: dict[str, str] = Field(
        default_factory=dict,
        description="Agent ID -> reason for uncertainty"
    )

    # Common knowledge
    common_knowledge: bool = Field(
        default=False,
        description="Whether this is approximated common knowledge"
    )

    common_knowledge_threshold: int = Field(
        default=2,
        description="Number of independent high-reputation agents needed for common knowledge"
    )

    # Supporting data
    evidence_count: int = Field(
        default=0,
        description="Number of evidence items supporting proposition"
    )

    evidence_refs: list[str] = Field(
        default_factory=list,
        description="References to evidence (blackboard keys, code locations)"
    )

    # Conflicts
    conflicts: list[str] = Field(
        default_factory=list,
        description="Conflicts or contradictions related to this proposition"
    )

    # Metadata
    domain: str | None = Field(
        default=None,
        description="Knowledge domain (security, architecture, etc.)"
    )

    created_at: float = Field(
        default_factory=time.time,
        description="When proposition was first recorded"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="Last update"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def add_believer(self, agent_id: str, confidence: float) -> None:
        """Add agent as believer.

        Args:
            agent_id: Agent ID
            confidence: Belief confidence
        """
        self.believers[agent_id] = confidence
        # Remove from disbelievers if present
        self.disbelievers.pop(agent_id, None)
        self.uncertain.pop(agent_id, None)
        self.updated_at = time.time()

    def add_disbeliever(self, agent_id: str, confidence: float) -> None:
        """Add agent as disbeliever.

        Args:
            agent_id: Agent ID
            confidence: Disbelief confidence
        """
        self.disbelievers[agent_id] = confidence
        # Remove from believers if present
        self.believers.pop(agent_id, None)
        self.uncertain.pop(agent_id, None)
        self.updated_at = time.time()

    def check_common_knowledge(
        self,
        agent_reputations: dict[str, float]
    ) -> bool:
        """Check if proposition is common knowledge.

        Rule: Mark common = True when:
        - At least K independent agents with decent reputation believe p
        - No high-reputation agent explicitly disbelieves p

        Args:
            agent_reputations: Agent ID -> reputation score mapping

        Returns:
            True if common knowledge
        """
        # Count high-reputation believers
        high_rep_believers = sum(
            1 for aid, conf in self.believers.items()
            if conf >= 0.8 and agent_reputations.get(aid, 0.0) >= 0.7
        )

        # Check for high-reputation disbelievers
        high_rep_disbelievers = any(
            conf >= 0.7 and agent_reputations.get(aid, 0.0) >= 0.7
            for aid, conf in self.disbelievers.items()
        )

        is_common = (
            high_rep_believers >= self.common_knowledge_threshold
            and not high_rep_disbelievers
        )

        self.common_knowledge = is_common
        self.updated_at = time.time()

        return is_common


class Intention(BaseModel):
    """An agent's intention (commitment to achieve a goal).

    From BDI (Beliefs-Desires-Intentions) architecture.

    Examples:
        Active intention:
        ```python
        intention = Intention(
            owner="analyzer_001",
            goal="Complete security analysis of auth module",
            status="active",
            dependencies=["task_analyze_auth_completed"],
            plan_id="plan_sec_analysis_001"
        )
        ```
    """

    intention_id: str = Field(
        default_factory=lambda: f"intention_{int(time.time() * 1000)}",
        description="Intention identifier"
    )

    owner: str = Field(
        description="Agent that has this intention"
    )

    goal: str = Field(
        description="Goal to achieve"
    )

    status: str = Field(
        default="active",
        description="Status: 'active', 'suspended', 'fulfilled', 'abandoned'"
    )

    # Dependencies
    dependencies: list[str] = Field(
        default_factory=list,
        description="Other intentions or conditions this depends on"
    )

    # Plan
    plan_id: str | None = Field(
        default=None,
        description="Associated plan ID"
    )

    # Metadata
    created_at: float = Field(
        default_factory=time.time,
        description="When intention was formed"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="Last update"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class JointIntention(BaseModel):
    """Joint intention shared by multiple agents.

    Represents collaborative commitment: "we intend to achieve G together"

    Examples:
        ```python
        joint = JointIntention(
            members=["coord_001", "analyzer_002", "analyzer_003"],
            goal="Produce validated security report",
            commitment_condition="All members commit to thorough analysis",
            termination_condition="Report validated by arbiter OR deadline reached"
        )
        ```
    """

    intention_id: str = Field(
        default_factory=lambda: f"joint_{int(time.time() * 1000)}",
        description="Joint intention identifier"
    )

    members: list[str] = Field(
        description="Agent IDs participating"
    )

    goal: str = Field(
        description="Shared goal"
    )

    commitment_condition: str = Field(
        description="Conditions for commitment (narrative or constraints)"
    )

    termination_condition: str = Field(
        description="When joint intention is fulfilled or abandoned"
    )

    # Status
    active: bool = Field(
        default=True,
        description="Whether joint intention is active"
    )

    fulfilled: bool = Field(
        default=False,
        description="Whether goal was achieved"
    )

    # Metadata
    created_at: float = Field(
        default_factory=time.time,
        description="When formed"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class EpistemicCapability(AgentCapability):
    """Manages epistemic state (beliefs, knowledge, intentions).

    Provides:
    - Proposition tracking
    - Belief updates
    - Common knowledge detection
    - Intention management

    Colony-scoped by default — epistemic state is shared across all agents
    in the colony so they can track shared beliefs and joint intentions.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "epistemic",
        input_patterns: list[str] | None = None,
        capability_key: str = "epistemic_capability",
    ):
        """Initialize epistemic capability.

        Args:
            agent: Owning agent
            scope: Blackboard scope (defaults to COLONY)
            namespace: Namespace for this capability's blackboard entries
            input_patterns: Event patterns to subscribe to
            capability_key: Capability key
        """
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=input_patterns,
            capability_key=capability_key,
        )

    def get_action_group_description(self) -> str:
        return (
            "Epistemic Capability — manages shared beliefs, knowledge, and intentions. "
            "Records propositions, tracks which agents believe what, detects common knowledge, "
            "and manages individual and joint intentions."
        )

    def _get_proposition_key(self, proposition_id: str) -> str:
        return EpistemicProtocol.proposition_key(proposition_id)

    def _get_intention_key(self, intention_id: str) -> str:
        return EpistemicProtocol.intention_key(intention_id)

    def _get_joint_intention_key(self, intention_id: str) -> str:
        return EpistemicProtocol.joint_intention_key(intention_id)

    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    @action_executor(
        action_key="record_proposition",
        planning_summary="Record a new proposition for epistemic tracking (belief/knowledge).",
    )
    async def record_proposition(
        self,
        proposition: str,
        domain: str | None = None
    ) -> EpistemicStatus:
        """Record a new proposition.

        Args:
            proposition: Proposition text
            domain: Optional knowledge domain

        Returns:
            Created epistemic status
        """
        status = EpistemicStatus(
            proposition=proposition,
            domain=domain
        )

        await self._store_status(status)
        return status

    @action_executor(
        action_key="update_belief",
        planning_summary="Update an agent's belief about a proposition (believes/disbelieves with confidence).",
    )
    async def update_belief(
        self,
        proposition_id: str,
        agent_id: str,
        believes: bool,
        confidence: float
    ) -> EpistemicStatus:
        """Update agent's belief about proposition.

        Args:
            proposition_id: Proposition ID
            agent_id: Agent ID
            believes: Whether agent believes it
            confidence: Confidence level

        Returns:
            Updated epistemic status
        """
        status = await self.get_status(proposition_id)
        if not status:
            raise ValueError(f"Proposition {proposition_id} not found")

        if believes:
            status.add_believer(agent_id, confidence)
        else:
            status.add_disbeliever(agent_id, confidence)

        await self._store_status(status)
        return status

    @action_executor(
        action_key="get_epistemic_status",
        planning_summary="Get epistemic status of a proposition (who believes/disbelieves, common knowledge).",
    )
    async def get_status(self, proposition_id: str) -> EpistemicStatus | None:
        """Get epistemic status.

        Args:
            proposition_id: Proposition ID

        Returns:
            Epistemic status or None
        """
        key = self._get_proposition_key(proposition_id)
        blackboard = await self.get_blackboard()
        data = await blackboard.read(key)

        if data is None:
            return None

        return EpistemicStatus(**data)

    @action_executor(
        action_key="get_common_knowledge",
        planning_summary="Get all propositions that are common knowledge among agents.",
    )
    async def get_common_knowledge(self) -> list[EpistemicStatus]:
        """Get all common knowledge propositions.

        Returns:
            List of common knowledge items
        """
        # TODO: Query all propositions marked as common knowledge
        return []

    @action_executor(
        action_key="add_intention",
        planning_summary="Add an intention (goal + plan) for an agent to the shared epistemic state.",
    )
    async def add_intention(
        self,
        agent_id: str,
        goal: str,
        plan_id: str | None = None,
        dependencies: list[str] | None = None
    ) -> Intention:
        """Add intention for agent.

        Args:
            agent_id: Agent ID
            goal: Goal to achieve
            plan_id: Associated plan
            dependencies: Dependencies

        Returns:
            Created intention
        """
        intention = Intention(
            owner=agent_id,
            goal=goal,
            plan_id=plan_id,
            dependencies=dependencies or []
        )

        key = self._get_intention_key(intention.intention_id)
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=key,
            value=intention.model_dump(),
            tags={"intention", agent_id, intention.status}
        )

        return intention

    @action_executor(
        action_key="add_joint_intention",
        planning_summary="Create a joint intention shared by multiple agents with commitment/termination conditions.",
    )
    async def add_joint_intention(
        self,
        members: list[str],
        goal: str,
        commitment_condition: str,
        termination_condition: str
    ) -> JointIntention:
        """Create joint intention.

        Args:
            members: Member agent IDs
            goal: Shared goal
            commitment_condition: Commitment conditions
            termination_condition: Termination conditions

        Returns:
            Created joint intention
        """
        joint = JointIntention(
            members=members,
            goal=goal,
            commitment_condition=commitment_condition,
            termination_condition=termination_condition
        )

        key = self._get_joint_intention_key(joint.intention_id)
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=key,
            value=joint.model_dump(),
            tags={"joint_intention", *members}
        )

        return joint

    async def _store_status(self, status: EpistemicStatus) -> None:
        """Store epistemic status.

        Args:
            status: Status to store
        """
        key = self._get_proposition_key(status.proposition_id)

        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=key,
            value=status.model_dump(),
            tags={
                "epistemic_status",
                "common" if status.common_knowledge else "not_common",
                status.domain or "general"
            }
        )


# Utility functions

async def check_goal_alignment(
    agent_id: str,
    proposed_action: Any,
    agent: Agent,
) -> tuple[bool, str]:
    """Check if proposed action aligns with agent's intentions.

    Prevents goal drift by validating actions against intentions.

    Args:
        agent_id: Agent ID
        proposed_action: Action being proposed
        agent: Agent instance (EpistemicCapability is created from agent)

    Returns:
        (is_aligned, reason) tuple
    """
    epistemic = EpistemicCapability(agent)

    # Get agent's intentions
    # Placeholder - would query blackboard

    # Check if action supports some active intention
    # Placeholder

    return (True, "Action aligned with intentions")


async def update_common_knowledge(
    proposition_id: str,
    agent: Agent,
    reputation_tracker: Any
) -> bool:
    """Update common knowledge status based on current beliefs.

    Args:
        proposition_id: Proposition to check
        agent: Agent instance (EpistemicCapability is created from agent)
        reputation_tracker: Reputation tracker for agent scores

    Returns:
        True if now common knowledge
    """
    epistemic = EpistemicCapability(agent)
    status = await epistemic.get_status(proposition_id)

    if not status:
        return False

    # Get agent reputations
    agent_reputations = {}
    for agent_id in set(list(status.believers.keys()) + list(status.disbelievers.keys())):
        reputation = await reputation_tracker.get_reputation(agent_id)
        agent_reputations[agent_id] = reputation.get_overall_score()

    # Check and update common knowledge status
    is_common = status.check_common_knowledge(agent_reputations)

    if is_common != status.common_knowledge:
        await epistemic._store_status(status)

    return is_common

