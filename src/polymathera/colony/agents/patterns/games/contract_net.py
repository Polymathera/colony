"""Contract Net protocol for task allocation.

Based on the Contract Net protocol from multi-agent systems literature,
adapted for LLM agents with reputation-based selection.

Game structure:
- Roles: Coordinator, Bidders, Winner, Validator
- Phases: ANNOUNCE → BID → AWARD → EXECUTE → VALIDATE → TERMINAL
- Purpose: Combat laziness through competitive bidding and reputation tracking

The protocol:
1. Coordinator announces task with requirements and constraints
2. Capable agents bid (estimate cost, quality, provide rationale)
3. Coordinator awards contract based on reputation and bid quality
4. Winner executes task
5. Validator checks result quality
6. Reputation updated based on performance

This encourages:
- Honest cost estimation (lying hurts future bids)
- Thorough work (poor work hurts reputation)
- Competitive quality (better work wins more tasks)
"""

from __future__ import annotations

import time
from typing import Any
from overrides import override
import uuid
from enum import Enum
from pydantic import BaseModel, Field

from .acl import ACLMessage, Performative, MessageSchema
from .state import GameState, GamePhase, GameOutcome, GameProtocolCapability, GameEventType
from .roles import GameRole
from ..capabilities.reputation import ReputationTracker
from ...blackboard.task_graph import Task, TaskStatus
from ...base import Agent
from ..actions import action_executor


class TaskBid(BaseModel):
    """A bid on a task in contract net.

    Examples:
        ```python
        bid = TaskBid(
            bidder_id="analyzer_003",
            task_id="task_analyze_auth",
            estimated_cost_tokens=50000,
            estimated_duration_seconds=120,
            estimated_quality_gain=0.85,
            rationale="Specialized in security analysis, high reputation in auth domain",
            capabilities_match=["security_analysis", "authentication_patterns"],
            past_performance={"success_rate": 0.92, "avg_quality": 0.88}
        )
        ```
    """

    bid_id: str = Field(
        default_factory=lambda: f"bid_{int(time.time() * 1000)}",
        description="Unique bid identifier"
    )

    bidder_id: str = Field(
        description="Agent submitting bid"
    )

    task_id: str = Field(
        description="Task being bid on"
    )

    # Cost estimates
    estimated_cost_tokens: int = Field(
        description="Estimated token cost"
    )

    estimated_duration_seconds: float = Field(
        description="Estimated time to complete"
    )

    estimated_quality_gain: float = Field(
        ge=0.0,
        le=1.0,
        description="Expected quality of result"
    )

    # Justification
    rationale: str = Field(
        description="Why this agent is suitable for the task"
    )

    capabilities_match: list[str] = Field(
        default_factory=list,
        description="Agent capabilities that match task requirements"
    )

    past_performance: dict[str, float] = Field(
        default_factory=dict,
        description="Past performance metrics (success rate, avg quality, etc.)"
    )

    # Bid strength (computed by coordinator)
    bid_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall bid score (computed from reputation, cost, quality)"
    )

    # Metadata
    submitted_at: float = Field(
        default_factory=time.time,
        description="When bid was submitted"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional bid metadata"
    )


class ContractAward(BaseModel):
    """Contract award decision."""

    task_id: str = Field(
        description="Task being awarded"
    )

    winner_id: str = Field(
        description="Agent awarded the contract"
    )

    winning_bid: TaskBid = Field(
        description="The winning bid"
    )

    selection_reasoning: str = Field(
        description="Why this bid was selected"
    )

    awarded_at: float = Field(
        default_factory=time.time,
        description="When contract was awarded"
    )


class ContractNetGameRole(str, Enum):
    """Roles in contract net game."""
    # TODO: Define specific roles
    COORDINATOR = "coordinator"  # Must be present
    BIDDER = "bidder"  # Must be present
    OBSERVER = "observer"  # Must be present in every game to allow passive observation


class ContractGameData(BaseModel):
    """Game data structure for contract net."""

    task: dict[str, Any]
    bids: list[TaskBid] = []
    award: ContractAward | None = None
    execution_result: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None



class ContractNetGameCapability(GameProtocolCapability[ContractGameData, ContractNetGameRole]):
    """Protocol for contract net task allocation.

    Phases:
    1. ANNOUNCE: Coordinator announces task with spec
    2. BID: Agents submit bids with cost/quality estimates
    3. AWARD: Coordinator awards contract to best bid
    4. EXECUTE: Winner executes task
    5. VALIDATE: Validator checks result quality
    6. TERMINAL: Game concludes with outcome

    Reputation integration:
    - Bid scores weighted by agent reputation
    - Poor execution lowers reputation
    - Good execution raises reputation
    """

    def __init__(
        self,
        agent: Agent,
        reputation_tracker: ReputationTracker | None = None
    ):
        """Initialize contract net protocol.

        Args:
            agent: Owning agent
            reputation_tracker: Optional reputation tracker for bid scoring
        """
        super().__init__(agent, game_type="contract_net")
        self.reputation_tracker = reputation_tracker

    def get_action_group_description(self) -> str:
        return (
            "Contract Net — competitive task allocation via bidding. "
            "Phases: ANNOUNCE → BID → AWARD → EXECUTE → VALIDATE. "
            "Bid scoring: 50% reputation + 30% quality + 20% cost. "
            "Poor execution lowers reputation; good execution raises it. "
            "Requires coordinator (announces/awards) and at least one bidder."
        )

    @override
    @action_executor(exclude_from_planning=True)
    async def start_game(
        self,
        participants: dict[str, str],  # agent_id -> role
        initial_data: dict[str, Any],
        game_id: str | None = None,
        config: dict[str, Any] | None = None
    ) -> GameState:
        """Start contract net game.

        Args:
            participants: Must include coordinator
            initial_data: Must include 'task'
            game_id: Optional game ID
            config: Optional configuration

        Returns:
            Initial game state
        """
        # Validate coordinator exists
        if "coordinator" not in participants.values():
            raise ValueError("Contract net requires a coordinator")

        # Extract task
        task_data = initial_data.get("task")
        if not task_data:
            raise ValueError("Initial data must contain task")

        task = Task(**task_data) if isinstance(task_data, dict) else task_data

        # Create game state
        game_id = game_id or f"game_{uuid.uuid4().hex}"
        state = GameState(
            game_id=game_id,
            game_type=self.game_type,
            conversation_id=task.task_id,
            participants=list(participants.keys()),
            roles=participants,
            phase=GamePhase.ANNOUNCE,
            game_data={
                "task": task.model_dump(),
                "bids": [],
                "award": None,
                "execution_result": None,
                "validation_result": None
            },
            config=config or {
                "bid_timeout_seconds": 30,
                "execution_timeout_seconds": 300
            }
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
        """Validate move legality.

        Args:
            agent_id: Agent making the move
            state: Current game state
            move: Move to validate

        Returns:
            (is_valid, reason) tuple
        """
        assert move.sender == self.agent.agent_id, "Move sender must be this agent"
        if move.sender not in state.participants:
            return (False, "Agent not a participant")

        role = state.get_role(move.sender)
        if not role:
            return (False, "Agent has no role")

        # Phase-specific validation
        if state.phase == GamePhase.ANNOUNCE:
            if role != "coordinator":
                return (False, "Only coordinator can announce")
            if move.performative != Performative.REQUEST:
                return (False, "ANNOUNCE requires REQUEST message")

        elif state.phase == GamePhase.BID:
            if role not in ["bidder", "coordinator"]:
                return (False, "Only bidders can bid in BID phase")
            if move.performative != Performative.OFFER:
                return (False, "BID phase requires OFFER message")

        elif state.phase == GamePhase.AWARD:
            if role != "coordinator":
                return (False, "Only coordinator can award")
            if move.performative not in [Performative.ACCEPT, Performative.REJECT]:
                return (False, "AWARD requires ACCEPT or REJECT")

        elif state.phase == GamePhase.EXECUTE:
            # Winner executes
            game_data = state.game_data
            if game_data.get("award"):
                winner_id = game_data["award"].get("winner_id")
                if move.sender != winner_id:
                    return (False, "Only winner can execute")

        elif state.phase == GamePhase.VALIDATE:
            if role not in ["validator", "coordinator"]:
                return (False, "Only validator or coordinator can validate")

        return (True, "Valid move")

    @override
    async def apply_move(
        self,
        state: GameState,
        move: ACLMessage
    ) -> GameState:
        """Transition state based on message.

        Args:
            state: Current game state
            move: Move to apply

        Returns:
            New game state
        """
        if state.phase == GamePhase.ANNOUNCE:
            # ANNOUNCE → BID
            state.phase = GamePhase.BID

        elif state.phase == GamePhase.BID:
            # Collect bid
            if move.performative == Performative.OFFER:
                bid_data = move.get_payload()
                bid = TaskBid(**bid_data)

                # Score bid using reputation
                if self.reputation_tracker:
                    bid.bid_score = await self._score_bid(bid)

                state.game_data["bids"].append(bid.model_dump())

            # Check if bidding period over (would use timeout in real implementation)
            # For now, transition immediately after first bid
            # state.phase = GamePhase.AWARD

        elif state.phase == GamePhase.AWARD:
            # Award contract
            if move.performative == Performative.ACCEPT:
                accepted_bid_id = move.get_payload().get("bid_id")

                # Find winning bid
                bids = [TaskBid(**b) for b in state.game_data.get("bids", [])]
                winning_bid = next((b for b in bids if b.bid_id == accepted_bid_id), None)

                if winning_bid:
                    award = ContractAward(
                        task_id=state.game_data["task"]["task_id"],
                        winner_id=winning_bid.bidder_id,
                        winning_bid=winning_bid,
                        selection_reasoning=move.get_payload().get("reasoning", "")
                    )
                    state.game_data["award"] = award.model_dump()

                # AWARD → EXECUTE
                state.phase = GamePhase.EXECUTE

        elif state.phase == GamePhase.EXECUTE:
            # Task execution completed
            if move.performative == Performative.INFORM:
                result = move.get_payload().get("result")
                state.game_data["execution_result"] = result

                # EXECUTE → VALIDATE
                state.phase = GamePhase.VALIDATE

        elif state.phase == GamePhase.VALIDATE:
            # Validation complete
            validation = move.get_payload()
            state.game_data["validation_result"] = validation

            # VALIDATE → TERMINAL
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
        """Compute game outcome.

        Args:
            state: Terminal state

        Returns:
            Game outcome
        """
        game_data = state.game_data

        validation = game_data.get("validation_result", {})
        success = validation.get("is_valid", False)

        duration = state.ended_at - state.started_at if state.ended_at else None

        # Extract winner and award
        award_data = game_data.get("award")
        winner_id = award_data.get("winner_id") if award_data else None

        return GameOutcome(
            outcome_type="task_allocated_and_executed",
            success=success,
            result=game_data.get("execution_result"),
            participants=state.participants,
            rounds_played=len(state.history),
            messages_exchanged=len(state.history),
            duration_seconds=duration,
            summary=f"Task {'successfully' if success else 'unsuccessfully'} executed by {winner_id}",
            metadata={
                "bids_received": len(game_data.get("bids", [])),
                "winner_id": winner_id,
                "validation": validation
            }
        )

    async def _score_bid(self, bid: TaskBid) -> float:
        """Score bid using reputation and estimates.

        Args:
            bid: Bid to score

        Returns:
            Bid score (0.0-1.0, higher is better)
        """
        # Get agent reputation
        if self.reputation_tracker:
            reputation = await self.reputation_tracker.get_reputation(bid.bidder_id)
            reputation_score = reputation.accuracy_score if reputation else 0.5
        else:
            reputation_score = 0.5

        # Combine reputation with bid quality
        # Higher quality and lower cost = better bid
        quality_component = bid.estimated_quality_gain
        cost_component = 1.0 / (1.0 + bid.estimated_cost_tokens / 100000.0)  # Normalize cost

        # Weighted combination
        score = (
            0.5 * reputation_score +
            0.3 * quality_component +
            0.2 * cost_component
        )

        return min(1.0, max(0.0, score))


# Utility functions

async def allocate_task_via_contract_net(
    task: Task,
    coordinator_agent: Agent,
    potential_bidders: list[str],
    reputation_tracker: ReputationTracker | None = None
) -> tuple[str | None, GameOutcome]:
    """Allocate task using contract net protocol.

    Args:
        task: Task to allocate
        coordinator_agent: Coordinator agent instance (Agent with initialized blackboard)
        potential_bidders: List of potential bidder agent IDs
        reputation_tracker: Optional reputation tracker

    Returns:
        (winner_agent_id, outcome) tuple
    """
    protocol = ContractNetGameCapability(coordinator_agent, reputation_tracker)
    await protocol.initialize()

    # Setup participants
    participants = {coordinator_agent.agent_id: "coordinator"}
    for bidder_id in potential_bidders:
        participants[bidder_id] = "bidder"

    # Start game
    game_id = await protocol.start_game(
        participants=participants,
        initial_data={"task": task},
        config={"bid_timeout_seconds": 30}
    )

    # Game proceeds through agent actions...
    # (Agents submit bids, coordinator awards, winner executes)

    # Return outcome when complete
    final_state = await protocol.load_state(game_id)
    if final_state and final_state.outcome:
        winner_id = final_state.game_data.get("award", {}).get("winner_id")
        return (winner_id, final_state.outcome)

    return (None, GameOutcome(
        outcome_type="allocation_failed",
        success=False,
        participants=list(participants.keys()),
        rounds_played=0,
        messages_exchanged=0
    ))


def create_bid(
    agent_id: str,
    task: Task,
    capabilities: list[str],
    past_performance: dict[str, float]
) -> TaskBid:
    """Create a bid for a task.

    Args:
        agent_id: Bidding agent ID
        task: Task to bid on
        capabilities: Agent capabilities
        past_performance: Past performance metrics

    Returns:
        Task bid
    """
    # Estimate cost based on task requirements
    # Placeholder - would use more sophisticated estimation
    estimated_tokens = task.constraints.get("max_tokens", 50000)
    estimated_duration = task.constraints.get("deadline_seconds", 120)

    # Match capabilities
    required = task.required_capabilities
    matched = [cap for cap in capabilities if cap in required] if required else capabilities

    return TaskBid(
        bidder_id=agent_id,
        task_id=task.task_id,
        estimated_cost_tokens=estimated_tokens,
        estimated_duration_seconds=estimated_duration,
        estimated_quality_gain=0.8,  # Placeholder
        rationale=f"Capabilities match: {matched}",
        capabilities_match=matched,
        past_performance=past_performance
    )

