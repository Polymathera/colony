"""Game roles and capabilities.

This module defines roles that agents can play in games:
- GameRole: Role definitions
- RoleCapabilities: What each role can do
- Role assignment and validation
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .acl import Performative
from .state import GameState


class GameRole(str, Enum):
    """Roles agents can play in games."""

    # Hypothesis game roles
    PROPOSER = "proposer"  # Proposes hypotheses
    SKEPTIC = "skeptic"  # Challenges proposals
    GROUNDER = "grounder"  # Provides evidence
    ARBITER = "arbiter"  # Makes final judgment

    # Contract net roles
    COORDINATOR = "coordinator"  # Announces tasks
    BIDDER = "bidder"  # Bids on tasks
    EXECUTOR = "executor"  # Executes awarded task
    VALIDATOR = "validator"  # Validates execution

    # Consensus game roles
    PROPOSER_CONSENSUS = "proposer_consensus"  # Proposes options
    VOTER = "voter"  # Votes on options
    AGGREGATOR = "aggregator"  # Aggregates votes

    # Negotiation game roles
    NEGOTIATOR = "negotiator"  # Negotiates terms
    MEDIATOR = "mediator"  # Mediates negotiation

    # General roles
    OBSERVER = "observer"  # Observes but doesn't participate
    FACILITATOR = "facilitator"  # Facilitates but doesn't decide


class RoleCapabilities(BaseModel):
    """Capabilities and constraints for a role.

    Defines what a role can and cannot do in a game.
    """

    role: GameRole = Field(
        description="The role"
    )

    # Allowed actions
    allowed_performatives: list[Performative] = Field(
        default_factory=list,
        description="Performatives this role can use"
    )

    allowed_phases: list[str] = Field(
        default_factory=list,
        description="Phases where this role can act"
    )

    # Constraints
    max_messages_per_phase: int | None = Field(
        default=None,
        description="Maximum messages per phase (None = unlimited)"
    )

    requires_capabilities: list[str] = Field(
        default_factory=list,
        description="Agent capabilities required for this role"
    )

    # Responsibilities
    responsibilities: list[str] = Field(
        default_factory=list,
        description="What this role is responsible for"
    )

    # Metadata
    description: str | None = Field(
        default=None,
        description="Role description"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional role metadata"
    )

    def can_perform(self, performative: Performative, phase: str) -> bool:
        """Check if role can perform action in phase.

        Args:
            performative: Action performative
            phase: Current phase

        Returns:
            True if allowed
        """
        # Check performative allowed
        if self.allowed_performatives and performative not in self.allowed_performatives:
            return False

        # Check phase allowed
        if self.allowed_phases and phase not in self.allowed_phases:
            return False

        return True


# Predefined role capabilities

HYPOTHESIS_GAME_ROLES = {
    GameRole.PROPOSER: RoleCapabilities(
        role=GameRole.PROPOSER,
        allowed_performatives=[Performative.PROPOSE, Performative.INFORM, Performative.CLARIFY],
        allowed_phases=["PROPOSE", "DEFEND"],
        responsibilities=[
            "Propose well-formed hypotheses",
            "Provide supporting evidence",
            "Defend against challenges",
            "Revise when needed"
        ],
        description="Proposes hypotheses and defends them with evidence"
    ),

    GameRole.SKEPTIC: RoleCapabilities(
        role=GameRole.SKEPTIC,
        allowed_performatives=[Performative.CHALLENGE, Performative.QUESTION, Performative.REFUTE],
        allowed_phases=["CHALLENGE"],
        max_messages_per_phase=3,
        responsibilities=[
            "Identify unsupported claims",
            "Challenge weak evidence",
            "Request additional evidence",
            "Point out contradictions"
        ],
        description="Challenges hypotheses and identifies weaknesses"
    ),

    GameRole.GROUNDER: RoleCapabilities(
        role=GameRole.GROUNDER,
        allowed_performatives=[Performative.INFORM, Performative.ANSWER],
        allowed_phases=["GROUND"],
        responsibilities=[
            "Fetch requested evidence",
            "Provide additional context",
            "Verify claims against data"
        ],
        description="Provides evidence and grounds claims in data"
    ),

    GameRole.ARBITER: RoleCapabilities(
        role=GameRole.ARBITER,
        allowed_performatives=[Performative.ACCEPT, Performative.REJECT, Performative.REQUEST],
        allowed_phases=["ARBITRATE"],
        responsibilities=[
            "Evaluate evidence quality",
            "Resolve conflicts",
            "Make final judgment",
            "Request more information if needed"
        ],
        description="Makes final judgment on hypotheses"
    ),
}


CONTRACT_NET_ROLES = {
    GameRole.COORDINATOR: RoleCapabilities(
        role=GameRole.COORDINATOR,
        allowed_performatives=[Performative.REQUEST, Performative.ACCEPT, Performative.REJECT],
        allowed_phases=["ANNOUNCE", "AWARD", "VALIDATE"],
        responsibilities=[
            "Announce tasks clearly",
            "Evaluate bids fairly",
            "Award contracts",
            "Validate execution"
        ],
        description="Coordinates task allocation through bidding"
    ),

    GameRole.BIDDER: RoleCapabilities(
        role=GameRole.BIDDER,
        allowed_performatives=[Performative.OFFER, Performative.COMMIT],
        allowed_phases=["BID"],
        responsibilities=[
            "Evaluate task fit",
            "Estimate costs accurately",
            "Submit competitive bids",
            "Honor commitments"
        ],
        description="Bids on tasks based on capabilities and costs"
    ),
}


# Utility functions

def get_role_capabilities(
    role: GameRole,
    game_type: str
) -> RoleCapabilities | None:
    """Get capabilities for a role in a game type.

    Args:
        role: Game role
        game_type: Type of game

    Returns:
        Role capabilities or None if not found
    """
    if game_type == "hypothesis_game":
        return HYPOTHESIS_GAME_ROLES.get(role)
    elif game_type == "contract_net":
        return CONTRACT_NET_ROLES.get(role)

    return None


def assign_role_to_agent(
    agent_id: str,
    role: GameRole,
    game_state: GameState
) -> bool:
    """Assign role to agent in game.

    Args:
        agent_id: Agent ID
        role: Role to assign
        game_state: Game state

    Returns:
        True if successfully assigned
    """
    # Check if agent not already in game
    if agent_id in game_state.roles:
        return False

    # Add to participants and roles
    if agent_id not in game_state.participants:
        game_state.participants.append(agent_id)

    game_state.roles[agent_id] = role.value

    return True

