"""Negotiation Game Protocol for conflict resolution and agreement finding.

Based on game-theoretic negotiation and mediation protocols (Wooldridge,
Rosenschein & Zlotkin). Enables agents to negotiate agreements through
iterative offer/counter-offer cycles with optional mediation.

Key Features:
- Iterative negotiation with offer/counter-offer
- Multiple negotiation strategies (compromising, hardball, integrative)
- Deadlock detection and breaking
- Optional mediator role
- Utility-based evaluation
- Pareto efficiency checking

Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                    NegotiationGameProtocol                      │
│                      (AgentCapability)                          │
├─────────────────────────────────────────────────────────────────┤
│  OWNS:                                                          │
│  • Game rules (valid moves, phase transitions)                  │
│  • @action_executor methods (dispatcher invokes these)          │
│  • Blackboard I/O (load/save game state, emit events)           │
│  • Pure analysis functions (check_convergence, check_deadlock)  │
│  • Long-term state spanning game lifetime                       │
│  • @event_handler methods (context enrichment + immediate moves)│
│  • Strategy + move selection (rule-based and/or LLM-assisted)   │
│                                                                 │
│  DOES NOT:                                                      │
│  • Implement the agent's non-game goals (that remains in the    │
│    agent's chosen ActionPolicy / planner)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ provides executors
                              │ emits events
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EventDrivenActionPolicy                        │
│               (e.g. CacheAwareActionPolicy)                     │
├─────────────────────────────────────────────────────────────────┤
│  OWNS:                                                          │
│  • Pulling events (`get_next_event*`) and broadcasting to       │
│    capability `@event_handler`s                                 │
│  • Invoking an LLM planner (if configured) using enriched scope │
│                                                                 │
│  DOES NOT:                                                      │
│  • Define game rules                                            │
│  • Directly manipulate game state                               │
│  • Handle blackboard I/O                                        │
└─────────────────────────────────────────────────────────────────┘

TODO: What kind of negotiation problems do you need to support? For example:
- Categorical terms (e.g., "delivery_method": "express" | "standard")
- Multi-attribute utility with non-linear preferences
- Constraints beyond simple bounds
- Package deals / linked issues
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Callable
from overrides import override
from pydantic import BaseModel, Field
from logging import getLogger

from ..acl import ACLMessage, Performative
from ..state import (
    GamePhase,
    GameProtocolCapability,
    GameState,
    GameOutcome,
    GameEventType,
    GameEvent,
    RolePermissions,
)
from ...events import event_handler, EventProcessingResult
from ....models import (
    Action,
    PolicyREPL,
)
from ....scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...actions import action_executor
from ....blackboard import BlackboardEvent
from ....blackboard.protocol import GameStateProtocol

logger = getLogger(__name__)


class NegotiationStrategy(str, Enum):
    """Negotiation strategies for agents.

    Each strategy corresponds to different game-theoretic behaviors and can be
    used to simulate various optimization objectives:

    COMPROMISING: Gradient descent toward Nash equilibrium. Agents move toward
        middle ground, accepting offers with utility >= 0.5. Use when seeking
        stable, mutually acceptable outcomes quickly.

    HARDBALL: Maximizing individual utility (greedy). Agents only accept offers
        with utility >= 0.8. Use when agent has strong BATNA or wants to test
        opponent's reservation price. Risk: deadlock.

    INTEGRATIVE: Pareto optimization. Seeks win-win solutions that expand the
        pie rather than just dividing it. Accepts at >= 0.6 utility. Use for
        multi-issue negotiations where creative solutions are possible.

    CONCESSION: Simulated annealing / time-pressure. Acceptance threshold
        decreases over rounds (0.8 - 0.1*round). Use when deadline pressure
        exists or to avoid deadlock after prolonged negotiation.

    TFTFT (Tit-for-Tat): Evolutionary stable strategy. Mirrors opponent's
        cooperation level - cooperate if they concede, defect if they don't.
        Use in repeated negotiations to establish cooperative norms.

    Example - configuring for different objectives:
        ```python
        # Configure the protocol strategy; use any EventDrivenActionPolicy.
        protocol = NegotiationGameProtocol(
            agent,
            game_id="...",
            strategy=NegotiationStrategy.HARDBALL,
        )
        await protocol.initialize()
        agent.add_capability(protocol)
        ```
    """

    COMPROMISING = "compromising"
    HARDBALL = "hardball"
    INTEGRATIVE = "integrative"
    CONCESSION = "concession"
    TFTFT = "tit_for_tat"


class NegotiationPhase(str, Enum):
    """Conceptual phases of negotiation (for documentation/external reference).

    Note: The game state machine uses GamePhase enum values directly:
        - OFFER: Initial offer phase.
        - COUNTER_OFFER: Counter-offer exchange. Other parties make counter-offers
        - EVALUATE: Mediation phase (mediator proposes)
        - AGREE: Acceptance phase (parties accept/reject)
        - TERMINAL: Negotiation complete
    """

    SETUP = "setup"
    INITIAL_OFFER = "initial_offer"  # First party makes offer
    COUNTER_OFFER = "counter_offer"  # Other parties respond
    MEDIATION = "mediation"  # Mediator proposes solution
    ACCEPTANCE = "acceptance"  # Parties accept/reject
    COMPLETE = "complete"  # Negotiation complete


class NegotiationRole(str, Enum):
    """Roles in negotiation game."""

    COORDINATOR = "coordinator"  # Manages game lifecycle
    PARTICIPANT = "participant"  # Makes offers and responses
    MEDIATOR = "mediator"  # Proposes solutions during deadlock


class Offer(BaseModel):
    """An offer in negotiation.

    An offer consists of terms (the proposed values for negotiable items) and
    metadata about who proposed it, its utility, and any concessions made.

    Example:
        ```python
        offer = Offer(
            proposer="agent_1",
            terms={"price": 100, "quantity": 50, "delivery_days": 7},
            utility=0.75,
            concessions={"price": -10},  # Reduced price by 10 from previous offer
            justification="Meeting halfway on price while maintaining quantity"
        )
        ```
    """

    offer_id: str = Field(
        default_factory=lambda: f"offer_{uuid.uuid4().hex[:8]}",
        description="Unique offer identifier"
    )

    proposer: str = Field(
        description="Agent making this offer"
    )

    terms: dict[str, Any] = Field(
        description="Proposed values for negotiable items. Keys are term names "
        "(e.g., 'price', 'quantity'), values are proposed amounts. These correspond "
        "to the terms in NegotiationIssue.preferences where each agent specifies "
        "their weight/importance for each term."
    )

    utility: float = Field(
        description="Proposer's utility for this offer"
    )

    concessions: dict[str, Any] = Field(
        default_factory=dict,
        description="Concessions made from previous offer"
    )

    justification: str | None = Field(
        default=None,
        description="Why this offer is reasonable"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When offer was made"
    )


class NegotiationIssue(BaseModel):
    """Issue being negotiated."""

    issue_id: str = Field(
        description="Unique issue identifier"
    )

    description: str = Field(
        description="What is being negotiated"
    )

    parties: list[str] = Field(
        description="Agent IDs involved in negotiation"
    )

    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Hard constraints that must be satisfied"
    )

    preferences: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Each party's preferences mapping agent_id -> {term_name: importance_weight}. "
        "Term names must match keys in Offer.terms. Weights indicate how much the agent "
        "values each term (higher = more important). Used for utility calculation: "
        "utility = sum(weight * term_value) / sum(weights). Example: "
        "{'agent_1': {'price': 0.8, 'delivery': 0.2}} means agent_1 cares mostly about price."
    )

    batna: dict[str, float] = Field(
        default_factory=dict,
        description="Best Alternative To Negotiated Agreement per party"
    )


class NegotiationRound(BaseModel):
    """A round of negotiation."""

    round_number: int = Field(
        description="Round number (1-indexed)"
    )

    offers: list[Offer] = Field(
        default_factory=list,
        description="Offers made in this round"
    )

    responses: dict[str, str] = Field(
        default_factory=dict,
        description="Agent responses (agent_id -> response_type)"
    )

    deadlock: bool = Field(
        default=False,
        description="Whether this round ended in deadlock"
    )


class NegotiationGameData(BaseModel):
    """Data specific to negotiation game."""

    issue: NegotiationIssue = Field(
        description="Issue being negotiated"
    )

    rounds: list[NegotiationRound] = Field(
        default_factory=list,
        description="History of negotiation rounds"
    )

    current_offers: dict[str, Offer] = Field(
        default_factory=dict,
        description="Current offer from each party"
    )

    mediator: str | None = Field(
        default=None,
        description="Optional mediator agent"
    )

    strategies: dict[str, NegotiationStrategy] = Field(
        default_factory=dict,
        description="Strategy each agent is using"
    )

    deadlock_count: int = Field(
        default=0,
        description="Number of consecutive deadlocks"
    )

    final_agreement: Offer | None = Field(
        default=None,
        description="Final agreed terms"
    )


_NEGOTIATION_GAME_TYPE = "negotiation_game"


class NegotiationGameProtocol(GameProtocolCapability):
    """Protocol for negotiation game.

    Phases:
    1. SETUP - Define issue, parties, constraints, preferences
    2. INITIAL_OFFER - First party makes opening offer
    3. COUNTER_OFFER - Other parties make counter-offers
    4. MEDIATION - If deadlock, mediator proposes solution
    5. ACCEPTANCE - Parties accept or reject
    6. COMPLETE - Agreement reached or negotiation failed

    Example:
        async def initialize(self):
            await super().initialize()
            protocol = NegotiationGameProtocol(agent)
            await protocol.initialize()
            self.agent.add_capability(protocol)

        @action_executor()
        async def start_negotiation_game(self) -> GameState:
            protocol = self.agent.get_capability(NegotiationGameProtocol.get_capability_name())
            state = await protocol.start_game(
                participants={"agent1": "negotiator", "agent2": "negotiator"},
                initial_data={
                    "issue": NegotiationIssue(
                        issue_id="resource_001",
                        description="Allocation of 100 compute units",
                        parties=["agent1", "agent2", "agent3"],
                        preferences={
                            "agent1": {"units": 1.0},
                            "agent2": {"units": 0.8, "priority": 0.2},
                            "agent3": {"units": 0.6, "quality": 0.4}
                        }
                    ).model_dump(),
                },
                game_id="game_001"
            )
            return state
    """

    # Define role-based permissions for negotiation game
    role_permissions = RolePermissions({
        # Coordinator: cannot make offers, only manages game lifecycle
        ("coordinator", GamePhase.OFFER): set(),
        ("coordinator", GamePhase.COUNTER_OFFER): set(),
        ("coordinator", GamePhase.EVALUATE): set(),
        ("coordinator", GamePhase.AGREE): set(),

        # Participant: can propose, accept, reject in offer phases
        ("participant", GamePhase.OFFER): {Performative.PROPOSE},
        ("participant", GamePhase.COUNTER_OFFER): {
            Performative.PROPOSE,
            Performative.ACCEPT,
            Performative.REJECT
        },
        ("participant", GamePhase.EVALUATE): set(),  # Wait for mediator
        ("participant", GamePhase.AGREE): {Performative.ACCEPT, Performative.REJECT},

        # Mediator: can only propose during mediation (EVALUATE phase)
        ("mediator", GamePhase.OFFER): set(),
        ("mediator", GamePhase.COUNTER_OFFER): set(),
        ("mediator", GamePhase.EVALUATE): {Performative.PROPOSE},
        ("mediator", GamePhase.AGREE): set(),
    })

    def __init__(
        self,
        agent: Any,
        scope: BlackboardScope = BlackboardScope.COLONY,
        game_id: str | None = None,
        role: str | None = None,
        strategy: NegotiationStrategy = NegotiationStrategy.COMPROMISING,
        min_acceptable_utility: float = 0.3,
        utility_function: Callable[[dict[str, Any]], float] | None = None,
        use_llm_reasoning: bool = False,
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 500,
        capability_key: str = "negotiation_game_protocol"
    ):
        """Initialize negotiation game protocol.

        Args:
            agent: Owning agent
            scope: Blackboard scope for game state (default: colony-level shared blackboard)
            game_id: Game instance ID. All participants should use the same game_id
                to share the same blackboard namespace for coordination.
            role: Agent's role ("participant", "coordinator", "mediator").
                If None, determined from game state.
            strategy: Negotiation strategy to use
            min_acceptable_utility: Minimum utility threshold (BATNA)
            utility_function: Custom utility function for evaluating offers
            use_llm_reasoning: If True, use LLM for strategic decisions
            llm_temperature: Temperature for LLM inference
            llm_max_tokens: Max tokens for LLM response
            capability_key: Unique key for this capability within the agent (default: "negotiation_game_protocol")
        """
        super().__init__(
            agent=agent,
            scope=scope,
            game_type=_NEGOTIATION_GAME_TYPE,
            game_id=game_id,
            role=role,
            use_llm_reasoning=use_llm_reasoning,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            capability_key=capability_key
        )

        # Strategy configuration
        self.strategy = strategy
        self.utility_function = utility_function or self._default_utility_function
        self._min_acceptable_utility = min_acceptable_utility
        self._concession_rate: float = 0.1  # How much to concede each round
        
        # Track negotiation state for strategy
        self._my_offers: list[Offer] = []
        self._received_offers: list[Offer] = []

    def get_action_group_description(self) -> str:
        return (
            "Negotiation Game — iterative offer/counter-offer with convergence detection. "
            "Phases: OFFER → COUNTER_OFFER → [EVALUATE] → AGREE. "
            "Convergence: terms within 10% triggers AGREE phase. "
            "Deadlock: 3+ stalls without mediator → TERMINAL (failure). "
            "Mediator (optional) proposes solutions in EVALUATE phase. "
            "LLM-based decision making for accept/reject/counter."
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
        """Start negotiation game.

        Args:
            participants: Agent ID -> role mapping
            initial_data: Must contain 'issue' (NegotiationIssue data)
            game_id: Optional game ID (generated if None)
            config: Optional configuration

        Returns:
            GameState
        """
        # Check permissions using base class method
        can_start, reason = self.can_start_game(self.agent.agent_id, participants)
        if not can_start:
            raise PermissionError(reason)

        issue_data = initial_data.get("issue")
        if not issue_data:
            raise ValueError("Initial data must contain 'issue'")

        issue = NegotiationIssue(**issue_data) if isinstance(issue_data, dict) else issue_data

        game_data = NegotiationGameData(
            issue=issue,
            mediator=initial_data.get("mediator"),
            strategies={pid: NegotiationStrategy.COMPROMISING for pid in participants}
        )

        state = GameState(
            game_id=game_id,  # Pass through to GameState
            game_type=_NEGOTIATION_GAME_TYPE,
            conversation_id=issue.issue_id,
            participants=list(participants.keys()),
            roles=participants,
            phase=GamePhase.OFFER,
            game_data=game_data.model_dump(),
            config=config or {}
        )
        state.history.append({
            "phase": "setup",
            "timestamp": time.time(),
            "message": f"Negotiation initialized by {state.roles[state.participants[0]]}"
        })
        await self.save_game_state(state, GameEventType.GAME_STARTED.value, move=None)
        return state

    @override
    async def validate_move(
        self,
        agent_id: str,
        move: ACLMessage,
        state: GameState
    ) -> tuple[bool, str]:
        """Validate move legality (game-specific rules).

        Note: Role permissions are already checked by base class via
        validate_move_permissions() before this method is called.
        This method handles negotiation-specific validation only.

        Args:
            agent_id: Agent making the move
            move: The ACL message
            state: Current game state

        Returns:
            (is_valid, reason) tuple
        """
        data = NegotiationGameData(**state.game_data)

        # Game-specific validation: check offer structure if proposing
        if move.performative == Performative.PROPOSE:
            content = move.content if isinstance(move.content, dict) else {}
            if "payload" not in content or "terms" not in content.get("payload", {}):
                return False, "PROPOSE must include payload.terms"

        # Check ACCEPT/REJECT has valid in_reply_to
        if move.performative in (Performative.ACCEPT, Performative.REJECT):
            if not move.in_reply_to:
                return False, f"{move.performative.value} must reference an offer via in_reply_to"
            # Verify the referenced offer exists
            if move.in_reply_to not in data.current_offers:
                # Check if it's an offer_id
                valid_offer_ids = {o.offer_id for o in data.current_offers.values()}
                if move.in_reply_to not in valid_offer_ids:
                    return False, f"Referenced offer '{move.in_reply_to}' not found"

        return True, "Valid move"

    @override
    async def apply_move(
        self,
        state: GameState,
        move: ACLMessage
    ) -> GameState:
        """Transition state based on message.

        Args:
            move: ACL message from agent

        Returns:
            Updated game state
        """
        # TODO: Which player is making the move and applying it?
        # TODO: The validate_move and apply_move methods should depend on the role of the player.

        data = NegotiationGameData(**state.game_data)
        phase = state.phase

        # Phase transitions use GamePhase enum directly.
        # Mapping: OFFER (initial), COUNTER_OFFER, EVALUATE (mediation), AGREE (acceptance), TERMINAL

        if phase == GamePhase.OFFER:
            if move.performative == Performative.PROPOSE:
                # First party makes offer
                offer = self._extract_offer(move, data)
                data.current_offers[move.sender] = offer

                # Move to counter-offer phase
                state.phase = GamePhase.COUNTER_OFFER
                state.history.append({
                    "phase": "offer",
                    "timestamp": time.time(),
                    "offer": offer.model_dump()
                })

        elif phase == GamePhase.COUNTER_OFFER:
            if move.performative == Performative.PROPOSE:
                # Agent makes counter-offer
                offer = self._extract_offer(move, data)
                data.current_offers[move.sender] = offer

                # Check if all parties have made offers
                if len(data.current_offers) == len(data.issue.parties):
                    # Check for convergence or deadlock
                    if self.check_convergence(data):
                        # Move to acceptance phase
                        state.phase = GamePhase.AGREE  # NegotiationPhase.ACCEPTANCE
                    elif self.check_deadlock(data):
                        data.deadlock_count += 1
                        if data.mediator and data.deadlock_count >= 2:
                            # Invoke mediator
                            state.phase = GamePhase.EVALUATE  # NegotiationPhase.MEDIATION
                        elif data.deadlock_count >= 3:
                            # Negotiation failed
                            state.phase = GamePhase.TERMINAL  # NegotiationPhase.COMPLETE
                            state.outcome = GameOutcome(
                                success=False,
                                result={"status": "failed", "reason": "deadlock"}
                            )
                    # else: stay in counter-offer for another round

            elif move.performative == Performative.ACCEPT:
                # Agent accepts current offer
                data.final_agreement = data.current_offers.get(move.in_reply_to)
                state.phase = GamePhase.TERMINAL  # NegotiationPhase.COMPLETE
                state.outcome = GameOutcome(
                    success=True,
                    result={"agreement": data.final_agreement.model_dump() if data.final_agreement else None}
                )

        elif phase == GamePhase.EVALUATE:
            # Mediation phase
            if move.sender == data.mediator and move.performative == Performative.PROPOSE:
                # Mediator proposes solution
                mediated_offer = self._extract_offer(move, data)
                data.current_offers["mediator"] = mediated_offer

                # Move to acceptance
                state.phase = GamePhase.AGREE  # NegotiationPhase.ACCEPTANCE

        elif phase == GamePhase.AGREE:
            # Acceptance phase
            if move.performative == Performative.ACCEPT:
                # Check if all parties accept
                # For now, simplified: any acceptance completes
                data.final_agreement = data.current_offers.get("mediator") or list(data.current_offers.values())[0]
                state.phase = GamePhase.TERMINAL  # NegotiationPhase.COMPLETE
                state.outcome = GameOutcome(
                    success=True,
                    result={"agreement": data.final_agreement.model_dump()}
                )

            elif move.performative == Performative.REJECT:
                # Rejection, back to counter-offer
                state.phase = GamePhase.COUNTER_OFFER  # NegotiationPhase.COUNTER_OFFER
                data.deadlock_count = 0  # Reset deadlock counter

        # Update game data
        state.game_data = data.model_dump()
        state.updated_at = time.time()

        return state

    def _extract_offer(self, move: ACLMessage, data: NegotiationGameData) -> Offer:
        """Extract offer from ACL message.

        Args:
            move: ACL message
            data: Current game data

        Returns:
            Offer object
        """
        content = move.content.get("payload", {}) if isinstance(move.content, dict) else {}

        return Offer(
            offer_id=content.get("offer_id", f"offer_{move.message_id}"),
            proposer=move.sender,
            terms=content.get("terms", {}),
            utility=content.get("utility", 0.0),
            concessions=content.get("concessions", {}),
            justification=content.get("justification")
        )

    def check_convergence(self, data: NegotiationGameData) -> bool:
        """Check if offers are converging.

        Args:
            data: Current game data

        Returns:
            True if offers are close enough
        """
        if len(data.current_offers) < 2:
            return False

        # Simple convergence check: compare terms
        offers = list(data.current_offers.values())
        first_terms = offers[0].terms

        # Check if all offers have similar terms
        # Calculate similarity (simplified)

        return all(self._terms_similar(first_terms, offer.terms) for offer in offers[1:])

    def _terms_similar(self, terms1: dict[str, Any], terms2: dict[str, Any]) -> bool:
        """Check if two sets of terms are similar enough to indicate convergence.

        This is a simplified similarity check for numeric terms only. For negotiations
        with non-numeric terms (strings, booleans), override this method or use a
        custom convergence check.

        Args:
            terms1: First offer's terms (e.g., {"price": 100, "quantity": 50})
            terms2: Second offer's terms

        Returns:
            True if all numeric terms are within 10% of each other

        Limitations:
            - Only compares numeric (int/float) values
            - Non-numeric terms are ignored in similarity calculation
            - 10% threshold is hardcoded; may need adjustment for different domains
        """
        if set(terms1.keys()) != set(terms2.keys()):
            return False

        for key in terms1:
            v1, v2 = terms1[key], terms2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Check numeric similarity (within 10%)
                max_val = max(abs(v1), abs(v2), 0.001)  # Avoid division by zero
                if abs(v1 - v2) > 0.1 * max_val:
                    return False

        return True

    def check_deadlock(self, data: NegotiationGameData) -> bool:
        """Check if negotiation is deadlocked.

        Args:
            data: Current game data

        Returns:
            True if deadlocked
        """
        # Deadlock if:
        # 1. All parties have made offers
        # 2. Offers are not converging
        # 3. Consecutive rounds with no progress

        if len(data.current_offers) < len(data.issue.parties):
            return False

        if len(data.rounds) < 2:
            return False

        # Check if last two rounds had similar offers (no progress)
        # Simplified: if not converging, assume deadlock potential
        return not self.check_convergence(data)

    @override
    async def is_terminal(self, state: GameState) -> bool:
        """Check if game is complete.

        Returns:
            True if in COMPLETE phase
        """
        return state.phase == GamePhase.TERMINAL

    @override
    async def compute_outcome(self, state: GameState) -> GameOutcome:
        """Compute outcome."""
        data = NegotiationGameData(**state.game_data)
        duration = state.ended_at - state.started_at if state.ended_at else None

        if data.final_agreement:
            return GameOutcome(
                outcome_type="agreement_reached",
                success=True,
                result=data.final_agreement.model_dump(),
                participants=state.participants,
                rounds_played=len(state.history),
                messages_exchanged=len(state.history),
                duration_seconds=duration,
                summary=f"Agreement reached after {len(data.rounds)} rounds"
            )
        else:
            return GameOutcome(
                outcome_type="negotiation_failed",
                success=False,
                result={"status": "no_agreement", "deadlocks": data.deadlock_count},
                participants=state.participants,
                rounds_played=len(state.history),
                messages_exchanged=len(state.history),
                duration_seconds=duration,
                summary=f"Negotiation failed after {data.deadlock_count} deadlocks"
            )

    @event_handler(pattern=GameStateProtocol.state_pattern()) # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
    async def _populate_game_specific_scope(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Add negotiation-specific data to scope."""
        game_event = GameEvent.from_blackboard_event(event)
        if game_event is None or game_event.game_state is None:
            logger.warning(f"Received event with no game state: {event.key}")
            return None  # Not a game event

        game_state = game_event.game_state

        repl.set("my_strategy", self.strategy.value)

        # Track received offers for tit-for-tat strategy
        for agent_id, offer in game_state.game_data.current_offers.items():
            if agent_id != self.agent.agent_id and offer not in self._received_offers:
                self._received_offers.append(offer)

    # =========================================================================
    # LLM-Based Decision Making
    # =========================================================================

    async def _llm_decide_action(
        self,
        game_data: NegotiationGameData,
        decision_type: str,
    ) -> dict[str, Any] | None:
        """Use LLM to decide next action based on game state.

        Uses LLMReasoningConfig for consistent LLM interface across all games.

        Args:
            game_data: Current game data
            decision_type: Type of decision (e.g., "counter_offer", "accept_reject")

        Returns:
            Dict with decision: {"action": "...", "reasoning": "..."}
        """
        context = self._build_decision_context(game_data)
        options = ["accept", "reject", "counter"]

        decision_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": options,
                    "description": "The action to take"
                },
                "terms": {
                    "type": "object",
                    "description": "If action is 'counter', the proposed terms"
                },
                "target_offer_id": {
                    "type": "string",
                    "description": "If action is 'accept' or 'reject', which offer"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the decision"
                }
            },
            "required": ["action", "reasoning"]
        }

        try:
            decision = await self.llm_config.generate_decision(
                self.agent,
                game_type=_NEGOTIATION_GAME_TYPE,
                decision_type=decision_type,
                context=context,
                options=options,
                json_schema=decision_schema,
            )

            if decision:
                logger.info(f"LLM decision for {self.agent.agent_id}: {decision}")
            return decision

        except Exception as e:
            logger.warning(f"LLM decision failed, falling back to rules: {e}")
            return None

    async def _execute_llm_decision(
        self,
        game_data: NegotiationGameData,
        game_event: GameEvent,
        decision: dict[str, Any],
    ) -> EventProcessingResult | None:
        """Execute the LLM's decision by creating the appropriate action.

        Args:
            game_data: Current game data
            context: Decision context
            decision: LLM decision dict with action, terms, reasoning

        Returns:
            Action to execute
        """
        action_type = decision.get("action")
        reasoning = decision.get("reasoning")

        if action_type == "accept":
            # Find the offer to accept
            target_id = decision.get("target_offer_id")
            for aid, offer in game_data.current_offers.items():
                if aid != self.agent.agent_id:
                    if target_id is None or offer.offer_id == target_id:
                        return await self._create_accept_action(offer, game_event, reasoning=reasoning)

        elif action_type == "reject":
            target_id = decision.get("target_offer_id")
            for aid, offer in game_data.current_offers.items():
                if aid != self.agent.agent_id:
                    if target_id is None or offer.offer_id == target_id:
                        return await self._create_reject_action(offer, game_event, reasoning=reasoning)

        elif action_type == "counter":
            # Use LLM-provided terms if available
            llm_terms = decision.get("terms")
            if llm_terms:
                # Create offer with LLM-provided terms
                offer = Offer(
                    proposer=self.agent.agent_id,
                    terms=llm_terms,
                    utility=self._calculate_utility(
                        llm_terms,
                        game_data.issue.preferences.get(self.agent.agent_id, {})
                    ),
                    justification=reasoning or "LLM-generated counter-offer"
                )
                self._my_offers.append(offer)
            # Create action (will generate offer if we didn't already)
            return await self._create_make_offer_action(
                game_event, game_data, is_initial=False, reasoning=reasoning
            )

        return None

    def _build_decision_context(self, game_data: NegotiationGameData) -> dict[str, Any]:
        """Build context dict for LLM decision making."""
        my_preferences = game_data.issue.preferences.get(self.agent.agent_id, {})
        other_offers = {
            aid: offer.terms
            for aid, offer in game_data.current_offers.items()
            if aid != self.agent.agent_id
        }
        return {
            "Your role": self.role,
            "Your strategy": f"""{self.strategy.value}
- COMPROMISING: Seek middle ground, accept at utility >= 0.5
- HARDBALL: Hold firm, only accept at utility >= 0.8
- INTEGRATIVE: Seek win-win, accept at utility >= 0.6
- CONCESSION: Lower threshold over time
- TFTFT: Mirror opponent's cooperation level
            """,
            "The issue being negotiated": game_data.issue.description,
            "Your preferences (term -> importance weight)": my_preferences,
            "Your BATNA threshold": self._min_acceptable_utility,
            "Current offers from other parties": other_offers,
            "Your last offer": self._my_offers[-1].terms if self._my_offers else None,
            "Negotiation round": len(game_data.rounds) + 1,
            "Deadlock count": game_data.deadlock_count,
        }

    # =========================================================================
    # Event Handler Overrides
    # =========================================================================

    @event_handler(pattern=GameStateProtocol.state_pattern()) # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
    async def _get_additional_context(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Add negotiation-specific context for LLM planning."""
        game_event = GameEvent.from_blackboard_event(event)
        if game_event is None or game_event.game_state is None:
            logger.warning(f"Received event with no game state: {event.key}")
            return None  # Not a game event

        game_state = game_event.game_state
        game_data = NegotiationGameData(**game_state.game_data)

        # Track received offers for tit-for-tat strategy
        for agent_id, offer in game_data.current_offers.items():
            if agent_id != self.agent.agent_id and offer not in self._received_offers:
                self._received_offers.append(offer)

        context = {
            "my_strategy": self.strategy.value,
            "my_offers_count": len(self._my_offers),
            #"offer_history": [o.model_dump() for o in game_data.offer_history],
            "received_offers_count": len(self._received_offers),
            "min_acceptable_utility": self._min_acceptable_utility,
            "current_offers": {
                aid: offer.model_dump() for aid, offer in game_data.current_offers.items()
            },
            "suggested_actions": self._get_suggested_actions(game_state.phase), # TODO: Implement this method
        }
        return EventProcessingResult(
            context_key="negotiation_context",
            context=context,
        )

    @override
    async def _handle_game_started(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game started event.

        - Coordinator: Verify we started the game, prepare to manage
        - Participant: Prepare initial offer if we're first
        - Mediator: Wait for deadlock
        """
        game_state = game_event.game_state
        game_data = NegotiationGameData(**game_state.game_data)

        logger.info(f"Game {self.game_id} started. Role: {self.role}")

        if self.role == NegotiationRole.COORDINATOR.value:
            # Coordinator started the game - verify ownership
            if game_event.agent_id != self.agent.agent_id:
                logger.warning(
                    f"Game {self.game_id} started by different agent {game_event.agent_id}"
                )

            # Coordinator monitors but doesn't make offers
            return None

        elif self.role == NegotiationRole.PARTICIPANT.value:
            # Check if we should make the initial offer
            if self._should_make_initial_offer(game_data):
                return await self._create_make_offer_action(game_data, is_initial=True)
            return None

        elif self.role == NegotiationRole.MEDIATOR.value:
            # Mediator waits for deadlock
            logger.info(f"Mediator {self.agent.agent_id} waiting for deadlock")
            return None

        return None

    @override
    async def _handle_game_move(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game move event (game state was updated by an agent) - dispatch to phase-specific handler."""
        # TODO: The decision logic can be a combination of hardcoded rules and (LLM-based) learned strategies.
        game_state = game_event.game_state
        phase = game_state.phase

        logger.info(
            f"Game {self.game_id} move. Phase: {phase.value}, "
            f"Agent: {game_event.agent_id}, Role: {self.role}"
        )

        # Check if game is terminal
        if game_state.is_terminal():
            return await self._handle_game_complete(game_event)

        # Dispatch based on phase and role
        phase_handlers = {
            GamePhase.OFFER: self._handle_offer_phase,
            GamePhase.COUNTER_OFFER: self._handle_counter_offer_phase,
            GamePhase.EVALUATE: self._handle_evaluate_phase,
            GamePhase.AGREE: self._handle_acceptance_phase,
        }

        handler = phase_handlers.get(phase)
        if handler:
            game_data = NegotiationGameData(**game_state.game_data)
            # Get role from context or determine from state
            role_str = game_state.get_role(self.agent.agent_id)
            if not role_str:
                return None
            role = NegotiationRole(role_str)
            return await handler(game_event, game_data, role)

        logger.warning(f"No handler for phase {phase.value}")
        return None

    @override
    async def _handle_game_complete(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game completion."""
        game_state = game_event.game_state
        outcome = game_state.outcome

        if outcome and outcome.success:
            logger.info(f"Game {self.game_id} completed successfully: {outcome.result}")
        else:
            logger.info(f"Game {self.game_id} failed: {outcome.result if outcome else 'unknown'}")

        # Mark policy as complete
        return EventProcessingResult(
            done=True
        )

    # =========================================================================
    # Phase Handlers
    # =========================================================================

    async def _handle_offer_phase(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData,
        role: NegotiationRole,
    ) -> EventProcessingResult | None:
        """Handle OFFER phase (initial offers)."""
        if role != NegotiationRole.PARTICIPANT:
            return None  # Only participants make offers

        # Check if we haven't made an offer yet
        if self._should_make_offer(game_data):
            return await self._create_make_offer_action(game_data, is_initial=True)

        return None

    async def _handle_counter_offer_phase(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData,
        role: NegotiationRole,
    ) -> EventProcessingResult | None:
        """Handle COUNTER_OFFER phase."""
        if role == NegotiationRole.PARTICIPANT:
            return await self._handle_participant_counter_offer(game_event, game_data)
        elif role == NegotiationRole.MEDIATOR:
            return await self._handle_mediation_phase(game_event, game_data)
        return None

    async def _handle_mediation_phase(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData
    ) -> EventProcessingResult | None:
        """Handle MEDIATION phase."""
        if self.role != NegotiationRole.MEDIATOR.value:
            return None

        # Mediator checks for deadlock and finds a middle ground solution.
        if self.check_deadlock(game_data):
            return await self._create_mediation_action(game_data, game_event)
        return None

    async def _handle_participant_counter_offer(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData,
    ) -> EventProcessingResult | None:
        """Participant logic for counter-offer phase.

        Uses LLM-based planning if `llm_config.enabled` is True, otherwise
        falls back to rule-based strategy logic.
        """
        # LLM-based planning path
        if self.llm_config.enabled:
            decision = await self._llm_decide_action(
                game_data,
                decision_type="counter_offer_decision (accept the best offer, reject it, or make a counter-offer based on your strategy)"
            )
            if decision:
                return await self._execute_llm_decision(game_data, game_event, decision)
            # Fall through to rule-based if LLM fails

        # Rule-based planning path
        best_offer = self._find_best_offer(game_data)

        if best_offer and best_offer.proposer != self.agent.agent_id:
            # Evaluate the offer
            utility = self._evaluate_offer(best_offer, game_data)

            # Decision based on utility and strategy
            if utility >= self._min_acceptable_utility:
                # Offer is acceptable - consider accepting
                if self._should_accept(utility, game_data):
                    return await self._create_accept_action(best_offer, game_event)

            # Either utility too low or strategy says counter
            if self._should_make_counter_offer(game_data):
                return await self._create_make_offer_action(game_event, game_data, is_initial=False)

        return None

    async def _handle_evaluate_phase(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData,
        role: NegotiationRole,
    ) -> EventProcessingResult | None:
        """Handle EVALUATE phase."""
        if role != NegotiationRole.PARTICIPANT:
            return None

        # Evaluate all current offers
        best_offer = self._find_best_offer(game_data)
        if best_offer:
            utility = self._evaluate_offer(best_offer, game_data)
            if utility >= self._min_acceptable_utility:
                return await self._create_accept_action(best_offer, game_event)
            else:
                return await self._create_reject_action(best_offer, game_event)

        return None

    async def _handle_acceptance_phase(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData,
        role: NegotiationRole,
    ) -> EventProcessingResult | None:
        """Handle ACCEPTANCE phase."""
        if role != NegotiationRole.PARTICIPANT:
            return None

        # In acceptance phase, evaluate the proposed agreement
        agreement = game_data.current_offers.get("mediator") or self._find_best_offer(game_data)

        if agreement:
            utility = self._evaluate_offer(agreement, game_data)
            if utility >= self._min_acceptable_utility:
                return await self._create_accept_action(agreement, game_event)
            else:
                # Reject and potentially trigger another round
                return await self._create_reject_action(agreement, game_event)

        return None

    # =========================================================================
    # Action Creators
    # =========================================================================

    def _create_make_offer_action(
        self,
        game_event: GameEvent,
        game_data: NegotiationGameData,
        is_initial: bool,
        reasoning: str | None = None,
    ) -> EventProcessingResult:
        """Create action to make an offer."""
        # Generate offer based on strategy
        offer = self._generate_offer(game_data, is_initial)
        self._my_offers.append(offer)

        # Create ACL message
        message = ACLMessage(
            performative=Performative.PROPOSE,
            sender=self.agent.agent_id,
            receivers=game_data.issue.parties,
            content={
                "schema": "offer",
                "payload": {
                    "offer_id": offer.offer_id,
                    "terms": offer.terms,
                    "utility": offer.utility,
                    "concessions": offer.concessions,
                    "justification": offer.justification,
                }
            }
        )

        action = Action(
            action_id=f"offer_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(),  # Use current version since this is an immediate action
            },
            reasoning=reasoning or f"Making {'initial' if is_initial else 'counter'} offer: {offer.terms}"
        )
        return EventProcessingResult(immediate_action=action)

    def _create_accept_action(
        self,
        offer: Offer,
        game_event: GameEvent,
        reasoning: str | None = None,
    ) -> EventProcessingResult:
        """Create action to accept an offer."""
        message = ACLMessage(
            performative=Performative.ACCEPT,
            sender=self.agent.agent_id,
            receivers=[offer.proposer],
            content={
                "schema": "acceptance",
                "payload": {
                    "accepted_offer_id": offer.offer_id,
                }
            },
            in_reply_to=offer.offer_id,
        )

        action = Action(
            action_id=f"accept_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(), # Use current version since this is an immediate action
            },
            reasoning=reasoning or f"Accepting offer {offer.offer_id} from {offer.proposer}"
        )
        return EventProcessingResult(immediate_action=action)

    def _create_reject_action(
        self,
        offer: Offer,
        game_event: GameEvent,
        reasoning: str | None = None,
    ) -> EventProcessingResult:
        """Create action to reject an offer."""
        message = ACLMessage(
            performative=Performative.REJECT,
            sender=self.agent.agent_id,
            receivers=[offer.proposer],
            content={
                "schema": "rejection",
                "payload": {
                    "rejected_offer_id": offer.offer_id,
                    "reason": reasoning or "Below acceptable utility threshold",
                }
            },
            in_reply_to=offer.offer_id,
        )

        action = Action(
            action_id=f"reject_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(), # Use current version since this is an immediate action
            },
            reasoning=reasoning or f"Rejecting offer {offer.offer_id} from {offer.proposer}"
        )
        return EventProcessingResult(immediate_action=action)

    def _create_mediation_action(
        self,
        game_data: NegotiationGameData,
        game_event: GameEvent,
        reasoning: str | None = None,
    ) -> EventProcessingResult:
        """Create mediation proposal action (for mediator role)."""
        # Mediator finds a middle ground
        mediated_offer = self._generate_mediation_proposal(game_data)

        justification = reasoning or "Mediated compromise based on all parties' preferences"

        message = ACLMessage(
            performative=Performative.PROPOSE,
            sender=self.agent.agent_id,
            receivers=game_data.issue.parties,
            content={
                "schema": "mediation_proposal",
                "payload": {
                    "offer_id": mediated_offer.offer_id,
                    "terms": mediated_offer.terms,
                    "justification": justification,
                }
            }
        )

        action = Action(
            action_id=f"mediate_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(),
            },
            reasoning=reasoning or f"Proposing mediated solution: {mediated_offer.terms}"
        )
        return EventProcessingResult(immediate_action=action)

    # =========================================================================
    # Strategy Logic
    # =========================================================================

    # Decision helpers

    def _should_make_initial_offer(self, game_data: NegotiationGameData) -> bool:
        """Determine if this agent should make the initial offer."""
        # First party in the list makes initial offer
        if not game_data.issue.parties:
            return False
        return game_data.issue.parties[0] == self.agent.agent_id

    def _should_make_offer(self, game_data: NegotiationGameData) -> bool:
        return self.agent.agent_id not in game_data.current_offers

    def _should_make_counter_offer(self, game_data: NegotiationGameData) -> bool:
        """Determine if agent should make a counter offer."""
        # Don't counter if we already made an offer this round
        my_offer = game_data.current_offers.get(self.agent.agent_id)
        if my_offer:
            # Check if we've made offer this round
            current_round = len(game_data.rounds)
            if my_offer.timestamp > time.time() - 60:  # Recent offer
                return False
        return True

    def _should_accept(self, utility: float, game_data: NegotiationGameData) -> bool:
        """Determine if agent should accept based on strategy."""
        if self.strategy == NegotiationStrategy.HARDBALL:
            # Hardball: only accept if very good
            return utility >= 0.8
        elif self.strategy == NegotiationStrategy.COMPROMISING:
            # Compromising: accept reasonable offers
            return utility >= 0.5
        elif self.strategy == NegotiationStrategy.INTEGRATIVE:
            # Integrative: accept if creates value for all
            return utility >= 0.6
        elif self.strategy == NegotiationStrategy.CONCESSION:
            # Concession: lower threshold over time
            rounds = len(game_data.rounds)
            threshold = max(0.3, 0.8 - rounds * 0.1)
            return utility >= threshold
        elif self.strategy == NegotiationStrategy.TFTFT:
            # Tit-for-tat: match opponent's cooperation level
            return utility >= self._get_opponent_cooperation_level(game_data)

        return utility >= 0.5  # Default

    # Generation Helpers

    def _generate_offer(self, game_data: NegotiationGameData, is_initial: bool) -> Offer:
        """Generate an offer based on strategy."""
        preferences = game_data.issue.preferences.get(self.agent.agent_id, {})
        constraints = game_data.issue.constraints

        if is_initial:
            # Initial offer - based on strategy
            terms = self._generate_initial_terms(preferences, constraints)
        else:
            # Counter offer - concede from previous
            previous_offer = self._my_offers[-1] if self._my_offers else None
            terms = self._generate_counter_terms(previous_offer, game_data)

        utility = self._calculate_utility(terms, preferences)

        concessions = {}
        if not is_initial and self._my_offers:
            concessions = self._calculate_concessions(self._my_offers[-1].terms, terms)

        return Offer(
            proposer=self.agent.agent_id,
            terms=terms,
            utility=utility,
            concessions=concessions,
            justification=self._generate_justification(terms, is_initial)
        )

    def _generate_initial_terms(
        self,
        preferences: dict[str, float],
        constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate initial offer terms based on strategy."""
        terms = {}

        for term, weight in preferences.items():
            if self.strategy == NegotiationStrategy.HARDBALL:
                # Ask for maximum
                terms[term] = 1.0 * weight
            elif self.strategy == NegotiationStrategy.COMPROMISING:
                # Start at 70%
                terms[term] = 0.7 * weight
            elif self.strategy == NegotiationStrategy.INTEGRATIVE:
                # Balanced start
                terms[term] = 0.6 * weight
            else:
                terms[term] = 0.75 * weight

        # Apply constraints
        for key, value in constraints.items():
            if key in terms:
                terms[key] = max(terms.get(key, 0), value)

        return terms

    def _generate_counter_terms(
        self,
        previous_offer: Offer | None,
        game_data: NegotiationGameData
    ) -> dict[str, Any]:
        """Generate counter-offer terms."""
        if not previous_offer:
            preferences = game_data.issue.preferences.get(self.agent.agent_id, {})
            return self._generate_initial_terms(preferences, game_data.issue.constraints)

        terms = dict(previous_offer.terms)
        best_other = self._find_best_offer(game_data)

        if best_other and best_other.proposer != self.agent.agent_id:
            # Move towards best other offer
            for key in terms:
                if key in best_other.terms:
                    # Concede by concession_rate towards their position
                    my_val = terms[key]
                    their_val = best_other.terms[key]
                    terms[key] = my_val + self._concession_rate * (their_val - my_val)

        return terms

    def _generate_mediation_proposal(self, game_data: NegotiationGameData) -> Offer:
        """Generate a mediation proposal (average of all offers)."""
        all_offers = list(game_data.current_offers.values())
        if not all_offers:
            # Fallback: use preferences
            return self._generate_offer(game_data, is_initial=True)

        # Average all terms
        terms: dict[str, float] = {}
        for offer in all_offers:
            for key, value in offer.terms.items():
                if isinstance(value, (int, float)):
                    terms[key] = terms.get(key, 0) + value / len(all_offers)

        return Offer(
            proposer=self.agent.agent_id,
            terms=terms,
            utility=0.5,  # Mediator doesn't have utility function
            justification="Mediated compromise based on averaging all parties' proposals"
        )

    def _find_best_offer(self, game_data: NegotiationGameData) -> Offer | None:
        """Find the best offer for this agent."""
        best_offer = None
        best_utility = -float('inf')

        for agent_id, offer in game_data.current_offers.items():
            if agent_id == self.agent.agent_id:
                continue  # Skip our own offer

            utility = self._evaluate_offer(offer, game_data)
            if utility > best_utility:
                best_utility = utility
                best_offer = offer

        return best_offer

    def _evaluate_offer(self, offer: Offer, game_data: NegotiationGameData) -> float:
        """Evaluate an offer's utility for this agent."""
        preferences = game_data.issue.preferences.get(self.agent.agent_id, {})
        return self._calculate_utility(offer.terms, preferences)

    def _calculate_utility(
        self,
        terms: dict[str, Any],
        preferences: dict[str, float]
    ) -> float:
        """Calculate utility of terms given preferences."""
        if self.utility_function:
            return self.utility_function(terms)
        return self._default_utility_function(terms, preferences)

    def _default_utility_function(
        self,
        terms: dict[str, Any],
        preferences: dict[str, float] | None = None
    ) -> float:
        """Default utility calculation: weighted sum."""
        if not preferences:
            preferences = {}

        total_weight = sum(preferences.values()) or 1.0
        utility = 0.0

        for key, value in terms.items():
            weight = preferences.get(key, 1.0)
            if isinstance(value, (int, float)):
                utility += weight * value

        return utility / total_weight

    def _calculate_concessions(
        self,
        old_terms: dict[str, Any],
        new_terms: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate concessions made between offers."""
        concessions = {}
        for key in old_terms:
            if key in new_terms:
                old_val = old_terms[key]
                new_val = new_terms[key]
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    diff = new_val - old_val
                    if abs(diff) > 0.01:
                        concessions[key] = diff
        return concessions

    def _generate_justification(self, terms: dict[str, Any], is_initial: bool) -> str:
        """Generate justification for an offer."""
        if is_initial:
            return f"Initial proposal based on {self.strategy.value} strategy"
        return f"Counter-offer with {self._concession_rate*100:.0f}% concession toward agreement"

    def _get_opponent_cooperation_level(self, game_data: NegotiationGameData) -> float:
        """Estimate opponent's cooperation level for tit-for-tat."""
        if not self._received_offers:
            return 0.5

        # Check if opponent made concessions
        if len(self._received_offers) < 2:
            return 0.5

        # Compare last two opponent offers
        prev = self._received_offers[-2]
        curr = self._received_offers[-1]

        preferences = game_data.issue.preferences.get(self.agent.agent_id, {})
        prev_utility = self._calculate_utility(prev.terms, preferences)
        curr_utility = self._calculate_utility(curr.terms, preferences)

        if curr_utility > prev_utility:
            return 0.6  # Cooperative
        elif curr_utility < prev_utility:
            return 0.4  # Defecting
        return 0.5  # Neutral


# NOTE: NegotiationGameActionPolicy has been removed.
# All game logic is now in NegotiationGameProtocol which handles events
# via @event_handler methods. Use NegotiationGameProtocol with any
# EventDrivenActionPolicy (e.g., CacheAwareActionPolicy).
#
# Migration example:
#   OLD:
#     policy = NegotiationGameActionPolicy(game_id="...", role="participant", ...)
#   
#   NEW:
#     protocol = NegotiationGameProtocol(
#         agent, game_id="...", role="participant",
#         strategy=NegotiationStrategy.COMPROMISING,
#     )
#     await protocol.initialize()
#     agent.add_capability(protocol)
#     policy = CacheAwareActionPolicy(agent=agent, planner=planner, ...)


# ============================================================================
# Utility functions
# ============================================================================


def calculate_pareto_efficiency(
    offers: list[Offer],
    issue: NegotiationIssue
) -> dict[str, Any]:
    """Calculate Pareto efficiency of offers.

    Args:
        offers: Offers to evaluate
        issue: Negotiation issue

    Returns:
        Efficiency analysis
    """
    # Simplified Pareto check
    # An offer is Pareto optimal if no other offer makes everyone better off

    pareto_optimal = []

    for i, offer in enumerate(offers):
        is_dominated = False
        for j, other in enumerate(offers):
            if i != j:
                # Check if other dominates offer
                if _dominates(other, offer, issue):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_optimal.append(offer.offer_id)

    return {
        "pareto_optimal_offers": pareto_optimal,
        "total_offers": len(offers),
        "efficiency_ratio": len(pareto_optimal) / len(offers) if offers else 0.0
    }


def _dominates(offer1: Offer, offer2: Offer, issue: NegotiationIssue) -> bool:
    """Check if offer1 Pareto dominates offer2.

    Args:
        offer1: First offer
        offer2: Second offer
        issue: Negotiation issue

    Returns:
        True if offer1 dominates offer2
    """
    # Simplified domination check
    # Would need actual utility functions for each party
    return offer1.utility > offer2.utility

