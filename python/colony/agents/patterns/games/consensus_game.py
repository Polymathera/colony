"""Consensus Game for agreement building.

Based on social choice theory and voting mechanisms from Shoham & Leyton-Brown.

Game structure:
- Roles: Proposers, Voters, Aggregator
- Phases: NOMINATE → VOTE → COUNT → DECLARE → TERMINAL
- Purpose: Build consensus through structured voting

Voting mechanisms:
- Borda count: Ranked voting with position scores
- Approval voting: Vote for acceptable options
- Plurality: Simple majority
- Condorcet: Pairwise comparisons

The game ensures:
- Multiple options can be proposed
- All voters can express preferences
- Transparent aggregation
- Democratic decision making
"""

from __future__ import annotations

import time
from typing import Any
from enum import Enum
from overrides import override

from pydantic import BaseModel, Field

from .acl import ACLMessage, Performative
from .state import GameState, GamePhase, GameOutcome, GameProtocolCapability, GameEventType
from ...base import Agent
from ..actions.policies import action_executor

class VotingMethod(str, Enum):
    """Voting aggregation method."""

    BORDA_COUNT = "borda_count"  # Ranked voting
    APPROVAL = "approval"  # Vote for acceptable options
    PLURALITY = "plurality"  # Simple majority
    RANKED_CHOICE = "ranked_choice"  # Instant runoff
    SCORE = "score"  # Score each option


class Vote(BaseModel):
    """A vote in consensus game."""

    vote_id: str = Field(
        default_factory=lambda: f"vote_{int(time.time() * 1000)}",
        description="Vote identifier"
    )

    voter_id: str = Field(
        description="Voting agent ID"
    )

    # Voting content (depends on voting method)
    ranked_options: list[str] = Field(
        default_factory=list,
        description="Ranked list of option IDs (for ranked voting)"
    )

    approved_options: list[str] = Field(
        default_factory=list,
        description="Approved option IDs (for approval voting)"
    )

    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Option ID -> score (for score voting)"
    )

    # Justification
    reasoning: str | None = Field(
        default=None,
        description="Reasoning for vote"
    )

    submitted_at: float = Field(
        default_factory=time.time,
        description="When vote was submitted"
    )


class ConsensusResult(BaseModel):
    """Result of consensus aggregation."""

    winner: str | None = Field(
        default=None,
        description="Winning option ID"
    )

    ranking: list[str] = Field(
        default_factory=list,
        description="Full ranking of options"
    )

    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Final scores for each option"
    )

    agreement_level: float = Field(
        ge=0.0,
        le=1.0,
        description="Level of agreement (1.0 = unanimous)"
    )

    voting_method: VotingMethod = Field(
        description="Method used for aggregation"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata"
    )


class ConsensusGameRole(str, Enum):
    """Roles in consensus game."""
    # TODO: Define specific roles
    VOTER = "voter"  # Must be present
    AGGREGATOR = "aggregator"  # Must be present
    PRPOSER = "proposer"  # Optional role for nominating options
    OBSERVER = "observer"  # Must be present in every game to allow passive observation


class ConsensusGameData(BaseModel):
    """Game-specific data for consensus game."""

    question: str = Field(
        default="",
        description="Question being decided"
    )

    options: list[str] = Field(
        default_factory=list,
        description="List of proposed options"
    )

    votes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of votes (stored as dicts for serialization)"
    )

    consensus_result: dict[str, Any] | None = Field(
        default=None,
        description="Final consensus result (stored as dict for serialization)"
    )



class ConsensusGameProtocol(GameProtocolCapability[ConsensusGameData, ConsensusGameRole]):
    """Protocol for consensus building through voting.

    Phases:
    1. NOMINATE: Agents propose options
    2. VOTE: Agents vote on options
    3. COUNT: Aggregator counts votes
    4. DECLARE: Winner declared
    5. TERMINAL: Game concludes
    """

    def __init__(
        self,
        agent: Agent,
        game_id: str | None = None,
        role: str = "aggregator",
        voting_method: VotingMethod = VotingMethod.BORDA_COUNT
    ):
        """Initialize consensus game protocol.

        Args:
            agent: Owning agent
            voting_method: Voting aggregation method
        """
        super().__init__(agent, game_id=game_id, role=role, game_type="consensus_game")
        self.voting_method = voting_method

    @override
    @action_executor(exclude_from_planning=True)
    async def start_game(
        self,
        participants: dict[str, str],  # agent_id -> role
        initial_data: dict[str, Any],
        game_id: str | None = None,
        config: dict[str, Any] | None = None
    ) -> GameState:
        """Start consensus game.

        Args:
            participants: Participants with roles (voters, aggregator)
            initial_data: Must contain 'question' or 'options'
            config: Optional configuration

        Returns:
            GameState
        """
        # Validate aggregator exists
        if "aggregator" not in participants.values():
            raise ValueError("Consensus game requires an aggregator")

        # Create game state
        state = GameState(
            game_type="consensus_game",
            conversation_id=f"consensus_{int(time.time() * 1000)}",
            participants=list(participants.keys()),
            roles=participants,
            phase=GamePhase.NOMINATE,
            game_data=ConsensusGameData().model_dump(),
            config=config or {"voting_method": self.voting_method.value}
        )

        await self.save_game_state(state, GameEventType.GAME_STARTED.value, move=None)
        return state

    @override
    async def validate_move(
        self,
        agent_id: str,
        move: ACLMessage,
        state: GameState
    ) -> tuple[bool, str]:
        """Validate move.

        Args:
            agent_id: Agent making move
            move: ACLMessage
            state: Current state

        Returns:
            (is_valid, reason) tuple
        """
        if agent_id not in state.participants:
            return (False, "Not a participant")

        role = state.get_role(agent_id)

        if state.phase == GamePhase.NOMINATE:
            if move.performative != Performative.PROPOSE:
                return (False, "NOMINATE requires PROPOSE")

        elif state.phase == GamePhase.VOTE:
            if role != "voter":
                return (False, "Only voters can vote")
            if move.performative != Performative.INFORM:
                return (False, "VOTE requires INFORM with vote content")

        elif state.phase == GamePhase.COUNT:
            if role != "aggregator":
                return (False, "Only aggregator can count")

        return (True, "Valid move")

    async def apply_move(
        self,
        state: GameState,
        move: ACLMessage
    ) -> GameState:
        """Transition state.

        Args:
            state: Current state
            move: ACLMessage

        Returns:
            New state
        """
        if state.phase == GamePhase.NOMINATE:
            # Add option
            option = move.get_payload().get("option")
            if option:
                if "options" not in state.game_data:
                    state.game_data["options"] = []
                state.game_data["options"].append(option)

            # Transition when ready (would use timeout in real implementation)
            # state.phase = GamePhase.VOTE

        elif state.phase == GamePhase.VOTE:
            # Collect vote
            vote_data = move.get_payload()
            vote = Vote(**vote_data)
            state.game_data["votes"].append(vote.model_dump())

            # Check if all voters voted (transition to COUNT)
            # Placeholder

        elif state.phase == GamePhase.COUNT:
            # Aggregate votes
            votes = [Vote(**v) for v in state.game_data.get("votes", [])]
            consensus = await self._aggregate_votes(
                votes,
                state.game_data.get("options", []),
                self.voting_method
            )
            state.game_data["consensus_result"] = consensus.model_dump()

            # COUNT → DECLARE
            state.phase = GamePhase.DECLARE

        elif state.phase == GamePhase.DECLARE:
            # DECLARE → TERMINAL
            state.phase = GamePhase.TERMINAL

        return state

    @override
    async def is_terminal(self, state: GameState) -> bool:
        """Check if terminal.

        Args:
            state: Game state

        Returns:
            True if terminal
        """
        return state.phase == GamePhase.TERMINAL

    @override
    async def compute_outcome(self, state: GameState) -> GameOutcome:
        """Compute outcome.

        Args:
            state: Terminal state

        Returns:
            Game outcome
        """
        consensus = ConsensusResult(**state.game_data.get("consensus_result", {}))

        duration = state.ended_at - state.started_at if state.ended_at else None

        return GameOutcome(
            outcome_type="consensus_reached",
            success=consensus.winner is not None,
            result=consensus.model_dump(),
            participants=state.participants,
            rounds_played=len(state.history),
            messages_exchanged=len(state.history),
            duration_seconds=duration,
            consensus_level=consensus.agreement_level,
            summary=f"Consensus reached on {consensus.winner} with {consensus.agreement_level:.2f} agreement",
            metadata={"voting_method": self.voting_method.value}
        )

    async def _aggregate_votes(
        self,
        votes: list[Vote],
        options: list[str],
        method: VotingMethod
    ) -> ConsensusResult:
        """Aggregate votes using specified method.

        Args:
            votes: All votes
            options: All options
            method: Voting method

        Returns:
            Consensus result
        """
        if method == VotingMethod.BORDA_COUNT:
            return await self._borda_count(votes, options)
        elif method == VotingMethod.APPROVAL:
            return await self._approval_voting(votes, options)
        elif method == VotingMethod.PLURALITY:
            return await self._plurality(votes, options)
        else:
            # Default to simple plurality
            return await self._plurality(votes, options)

    async def _borda_count(
        self,
        votes: list[Vote],
        options: list[str]
    ) -> ConsensusResult:
        """Borda count aggregation.

        Args:
            votes: Votes with rankings
            options: All options

        Returns:
            Consensus result
        """
        scores = {opt: 0.0 for opt in options}

        for vote in votes:
            # Award points based on rank position
            # 1st place = n-1 points, 2nd = n-2, etc.
            n = len(options)
            for rank, option in enumerate(vote.ranked_options):
                if option in scores:
                    scores[option] += (n - rank - 1)

        # Sort by score
        ranking = sorted(scores.keys(), key=lambda opt: scores[opt], reverse=True)
        winner = ranking[0] if ranking else None

        # Calculate agreement level
        max_score = scores[winner] if winner else 0
        max_possible = len(votes) * (len(options) - 1)
        agreement_level = max_score / max_possible if max_possible > 0 else 0.0

        return ConsensusResult(
            winner=winner,
            ranking=ranking,
            scores=scores,
            agreement_level=agreement_level,
            voting_method=VotingMethod.BORDA_COUNT
        )

    async def _approval_voting(
        self,
        votes: list[Vote],
        options: list[str]
    ) -> ConsensusResult:
        """Approval voting aggregation.

        Args:
            votes: Votes with approved options
            options: All options

        Returns:
            Consensus result
        """
        scores = {opt: 0.0 for opt in options}

        for vote in votes:
            for approved in vote.approved_options:
                if approved in scores:
                    scores[approved] += 1.0

        ranking = sorted(scores.keys(), key=lambda opt: scores[opt], reverse=True)
        winner = ranking[0] if ranking else None

        agreement_level = scores[winner] / len(votes) if winner and votes else 0.0

        return ConsensusResult(
            winner=winner,
            ranking=ranking,
            scores=scores,
            agreement_level=agreement_level,
            voting_method=VotingMethod.APPROVAL
        )

    async def _plurality(
        self,
        votes: list[Vote],
        options: list[str]
    ) -> ConsensusResult:
        """Plurality voting (simple majority).

        Args:
            votes: Votes
            options: All options

        Returns:
            Consensus result
        """
        scores = {opt: 0.0 for opt in options}

        for vote in votes:
            # Take first choice
            if vote.ranked_options:
                first_choice = vote.ranked_options[0]
                if first_choice in scores:
                    scores[first_choice] += 1.0

        ranking = sorted(scores.keys(), key=lambda opt: scores[opt], reverse=True)
        winner = ranking[0] if ranking else None

        agreement_level = scores[winner] / len(votes) if winner and votes else 0.0

        return ConsensusResult(
            winner=winner,
            ranking=ranking,
            scores=scores,
            agreement_level=agreement_level,
            voting_method=VotingMethod.PLURALITY
        )


# Utility functions

async def build_consensus(
    question: str,
    options: list[Any],
    voters: list[str],
    aggregator_agent: Any,  # Agent instance (aggregator)
    voting_method: VotingMethod = VotingMethod.BORDA_COUNT
) -> ConsensusResult:
    """Build consensus through voting game.

    Args:
        question: Question being decided
        options: List of options to choose from
        voters: List of voter agent IDs
        aggregator_agent: Aggregator agent instance
        voting_method: Voting method to use

    Returns:
        Consensus result
    """
    protocol = ConsensusGameProtocol(aggregator_agent, voting_method)
    await protocol.initialize()

    # Setup participants
    participants = {aggregator_agent.agent_id: "aggregator"}
    for voter_id in voters:
        participants[voter_id] = "voter"

    # Start game
    game_id = await protocol.start_game(
        participants=participants,
        initial_data={"question": question, "options": options},
        config={"voting_method": voting_method.value}
    )

    # Game proceeds through voting...
    # Return result when complete
    final_state = await protocol.load_state(game_id)
    if final_state and final_state.game_data.get("consensus_result"):
        return ConsensusResult(**final_state.game_data["consensus_result"])

    # Placeholder
    return ConsensusResult(
        winner=None,
        ranking=[],
        scores={},
        agreement_level=0.0,
        voting_method=voting_method
    )

