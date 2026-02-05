"""Hypothesis Game for hallucination control.

As specified in MULTI_AGENT_GAME_ENGINE.md:
"Hypothesis game: one agent proposes a solution, others try to refute or refine."

Game structure (Extensive Form):
- Roles: Proposer, Skeptic(s), Grounder(s), Arbiter
- Phases: PROPOSE → CHALLENGE → GROUND → DEFEND → ARBITRATE → TERMINAL
- Purpose: Combat hallucination through structured challenge and evidence requirements

The game ensures:
- Every claim must have supporting evidence
- Claims can be challenged by skeptics
- Challenges must be addressed with evidence or revision
- Final acceptance requires arbiter validation

Architecture:

┌────────────────────────────────────────────────────────────────────┐
│                     HypothesisGameProtocol                         │
│                       (AgentCapability)                            │
├────────────────────────────────────────────────────────────────────┤
│  OWNS:                                                             │
│  • Game rules (valid moves, phase transitions)                     │
│  • @action_executor methods (start_game, submit_move)              │
│  • Blackboard I/O (load/save game state, emit events)              │
│  • Role-based permission validation                                │
│                                                                    │
│  • @event_handler for processing game events                       │
│  • Decision logic (rule-based or supports LLM-based action policy) │
│  • Action creation for game moves                                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │ provides executors
                              │ emits events, writes to agent memory
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     EventDrivenActionPolicy                        │
│                        (ActionPolicy)                              │
├────────────────────────────────────────────────────────────────────┤
│  OWNS:                                                             │
│  • EventDrivenActionPolicy (e.g., CacheAwareActionPolicy)          │
│  • Decision logic (when to challenge, defend, accept) if not       │
│    overridden by HypothesisGameProtocol event handler (by          │
│    returning immediate actions)                                    │
│  • The protocol's @event_handler enriches planning context         │
│    and can return immediate actions for rule-based decisions       │
│                                                                    │
│  DOES NOT:                                                         │
│  • Define game rules                                               │
│  • Directly manipulate game state                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

This implements ideas from:
- Wooldridge's agent communication and coordination
- Shoham & Leyton-Brown's game-theoretic foundations
- Epistemic logic for belief tracking
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any
from overrides import override
from pydantic import BaseModel, Field
from logging import getLogger

from ....base import Agent, CapabilityResultFuture
from ...events import event_handler, EventProcessingResult

from ..acl import ACLMessage, Performative
from ..state import (
    GameState,
    GamePhase,
    GameOutcome,
    GameProtocolCapability,
    GameEventType,
    GameEvent,
    RolePermissions,
    LLMReasoningConfig,
)
from ..roles import GameRole, HYPOTHESIS_GAME_ROLES
from .. import Hypothesis
from ....patterns.capabilities.validation import ValidationResult, ValidationCapability
from ....patterns.capabilities.agent_pool import AgentPoolCapability
from ....models import (
    Action,
    PolicyREPL,
)
from ...actions.policies import action_executor
from ....blackboard import BlackboardEvent

# Import strategy protocols and types
from .types import (
    HypothesisContext,
    Evidence,
    EvaluationResult,
    HypothesisFormationTrigger,
    HypothesisStatus,
)
from .strategies import (
    HypothesisFormationStrategy,
    EvidenceGatheringStrategy,
    HypothesisEvaluationStrategy,
)

logger = getLogger(__name__)


class ChallengeRecord(BaseModel):
    """Record of a challenge in the hypothesis game."""

    challenge_id: str = Field(
        description="Unique challenge ID"
    )

    skeptic_id: str = Field(
        description="Agent that issued challenge"
    )

    challenged_aspect: str = Field(
        description="What part of hypothesis is challenged"
    )

    reason: str = Field(
        description="Why it's being challenged"
    )

    counter_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence against the claim"
    )

    status: str = Field(
        default="open",
        description="Status: 'open', 'addressed', 'unresolved'"
    )

    response: str | None = Field(
        default=None,
        description="Proposer's response to challenge"
    )


class HypothesisGameData(BaseModel):
    """Game-specific data for hypothesis game.

    Supports multiple hypotheses per game. Single-hypothesis is a special case.
    """

    # Changed from single hypothesis to list for multi-hypothesis games
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Hypotheses being validated in this game"
    )

    # Track which hypothesis is currently being processed
    current_hypothesis_index: int = Field(
        default=0,
        description="Index of hypothesis currently being validated"
    )

    challenges: list[ChallengeRecord] = Field(
        default_factory=list,
        description="Challenges raised"
    )

    additional_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence gathered by grounders"
    )

    revision_count: int = Field(
        default=0,
        description="Number of times hypothesis was revised"
    )

    # Per-hypothesis decisions for multi-hypothesis games
    arbiter_decisions: dict[str, str] = Field(
        default_factory=dict,
        description="hypothesis_id -> decision mapping"
    )

    arbiter_reasoning: str | None = Field(
        default=None,
        description="Arbiter's reasoning for decision"
    )

    @property
    def hypothesis(self) -> Hypothesis | None:
        """Get current hypothesis (backward compat)."""
        if self.hypotheses and 0 <= self.current_hypothesis_index < len(self.hypotheses):
            return self.hypotheses[self.current_hypothesis_index]
        return None

    @property
    def arbiter_decision(self) -> str | None:
        """Get decision for current hypothesis (backward compat)."""
        if self.hypothesis:
            return self.arbiter_decisions.get(self.hypothesis.hypothesis_id)
        return None


_HYPOTHESIS_GAME_TYPE = "hypothesis_game"


class HypothesisRole(str, Enum):
    """Roles in hypothesis game."""
    PROPOSER = "proposer"
    SKEPTIC = "skeptic"
    GROUNDER = "grounder"
    ARBITER = "arbiter"
    OBSERVER = "observer"  # Must be present in every game to allow passive observation


class HypothesisGameProtocol(GameProtocolCapability[HypothesisGameData, HypothesisRole]):
    """Protocol for agents participating in hypothesis validation games.

    Extends GameProtocolCapability with hypothesis-specific:
    - Challenge and evidence tracking
    - Epistemic reasoning for claim validation
    - Phase handlers for hypothesis workflow

    Phases:
    1. PROPOSE: Proposer submits hypothesis with evidence
    2. CHALLENGE: Skeptics challenge unsupported claims
    3. GROUND: Grounders fetch additional evidence if requested
    4. DEFEND: Proposer defends or revises hypothesis
    5. ARBITRATE: Arbiter makes final judgment
    6. TERMINAL: Game concludes

    Roles:
    - proposer: Proposes hypotheses and defends them
    - skeptic: Challenges unsupported claims
    - grounder: Provides evidence
    - arbiter: Makes final judgment

    Payoffs (for learning/reputation):
    - Proposer: +1 for accepted hypothesis, -0.5 for rejected
    - Skeptic: +0.5 for valid challenge, -0.5 for invalid
    - Grounder: +0.3 for useful evidence
    - Arbiter: +0.5 for correct judgment (validated later)

    Example:
        ```python
        async def initialize(self):
            await super().initialize()
            protocol = HypothesisGameProtocol(agent)
            await protocol.initialize()
            self.agent.add_capability(protocol)

        # Start a game
        game_id = await protocol.start_game(
            participants={"agent1": "proposer", "agent2": "skeptic", "agent3": "arbiter"},
            initial_data={"hypothesis": hypothesis.model_dump()},
        )
        ```
    """

    # Define role-based permissions for hypothesis game
    role_permissions = RolePermissions({
        # Proposer: can propose in PROPOSE phase, defend/revise in DEFEND phase
        ("proposer", GamePhase.PROPOSE): {Performative.PROPOSE},
        ("proposer", GamePhase.CHALLENGE): set(),  # Wait for challenges
        ("proposer", GamePhase.GROUND): set(),  # Wait for evidence
        ("proposer", GamePhase.DEFEND): {Performative.INFORM, Performative.PROPOSE},
        ("proposer", GamePhase.ARBITRATE): set(),  # Wait for judgment

        # Skeptic: can challenge in CHALLENGE phase
        ("skeptic", GamePhase.PROPOSE): set(),  # Wait for proposal
        ("skeptic", GamePhase.CHALLENGE): {Performative.CHALLENGE, Performative.ACCEPT},
        ("skeptic", GamePhase.GROUND): set(),
        ("skeptic", GamePhase.DEFEND): set(),
        ("skeptic", GamePhase.ARBITRATE): set(),

        # Grounder: provides evidence in GROUND phase
        ("grounder", GamePhase.PROPOSE): set(),
        ("grounder", GamePhase.CHALLENGE): set(),
        ("grounder", GamePhase.GROUND): {Performative.INFORM, Performative.ANSWER},
        ("grounder", GamePhase.DEFEND): set(),
        ("grounder", GamePhase.ARBITRATE): set(),

        # Arbiter: makes judgment in ARBITRATE phase, can also accept in CHALLENGE to skip
        ("arbiter", GamePhase.PROPOSE): set(),
        ("arbiter", GamePhase.CHALLENGE): {Performative.ACCEPT},  # Can accept to skip challenges
        ("arbiter", GamePhase.GROUND): set(),
        ("arbiter", GamePhase.DEFEND): set(),
        ("arbiter", GamePhase.ARBITRATE): {Performative.ACCEPT, Performative.REJECT, Performative.REQUEST},
    })

    def __init__(
        self,
        agent: Agent,
        game_id: str | None = None,
        role: str | None = None,
        use_llm_reasoning: bool = False,
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 500,
        # Pluggable strategies (can change dynamically)
        formation_strategy: HypothesisFormationStrategy | None = None,
        evidence_strategy: EvidenceGatheringStrategy | None = None,
        evaluation_strategy: HypothesisEvaluationStrategy | None = None,
    ):
        """Initialize hypothesis game protocol.

        Args:
            agent: Owning agent
            game_id: Game instance ID. All participants should use the same game_id
                to share the same blackboard namespace for coordination.
            role: Agent's role ("proposer", "skeptic", "grounder", "arbiter").
                If None, determined from game state.
            use_llm_reasoning: If True, use LLM for strategic decisions
            llm_temperature: Temperature for LLM inference
            llm_max_tokens: Max tokens for LLM response
            formation_strategy: Strategy for hypothesis formation (optional)
            evidence_strategy: Strategy for evidence gathering (optional)
            evaluation_strategy: Strategy for hypothesis evaluation (optional)
        """
        super().__init__(
            agent,
            game_type=_HYPOTHESIS_GAME_TYPE,
            game_id=game_id,
            role=role,
            use_llm_reasoning=use_llm_reasoning,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
        )

        # Hypothesis-specific tracking
        self._challenges_made: list[ChallengeRecord] = []
        self._evidence_provided: list[str] = []

        # Pluggable strategies (can be set/changed dynamically)
        self._formation_strategy = formation_strategy
        self._evidence_strategy = evidence_strategy
        self._evaluation_strategy = evaluation_strategy

    # =========================================================================
    # Strategy Accessors (dynamic selection supported)
    # =========================================================================

    def set_formation_strategy(self, strategy: HypothesisFormationStrategy) -> None:
        """Dynamically set formation strategy."""
        self._formation_strategy = strategy

    def set_evidence_strategy(self, strategy: EvidenceGatheringStrategy) -> None:
        """Dynamically set evidence strategy."""
        self._evidence_strategy = strategy

    def set_evaluation_strategy(self, strategy: HypothesisEvaluationStrategy) -> None:
        """Dynamically set evaluation strategy."""
        self._evaluation_strategy = strategy

    # =========================================================================
    # Strategy Integration Methods
    # =========================================================================

    async def form_hypotheses_from_context(
        self,
        context: HypothesisContext,
        max_hypotheses: int = 5,
    ) -> list[Hypothesis]:
        """Form hypotheses using the configured formation strategy.

        Convenience method for coordinators to generate hypotheses before
        starting a game.

        Args:
            context: Hypothesis formation context
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            List of formed hypotheses
        """
        if not self._formation_strategy:
            raise RuntimeError(
                "No formation strategy configured. Set one via "
                "set_formation_strategy() or pass to __init__."
            )

        hypotheses = await self._formation_strategy.form_hypotheses(
            context, max_hypotheses
        )

        # Register with tracking capability if available
        tracking = self._get_tracking_capability()
        if tracking:
            for hyp in hypotheses:
                await tracking.register_hypothesis(
                    hypothesis=hyp,
                    game_id=self.game_id,
                    domain=context.domain,
                )

        return hypotheses

    async def gather_evidence_for_hypothesis(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
        challenges: list[ChallengeRecord] | None = None,
    ) -> list[Evidence]:
        """Gather evidence using the configured evidence strategy.

        Args:
            hypothesis: Hypothesis to gather evidence for
            context: Context for evidence gathering
            challenges: Challenges to address

        Returns:
            List of evidence items
        """
        if not self._evidence_strategy:
            logger.warning("No evidence strategy configured")
            return []

        return await self._evidence_strategy.gather_evidence(
            hypothesis, context, challenges
        )

    async def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
        challenges: list[ChallengeRecord] | None = None,
    ) -> EvaluationResult:
        """Evaluate hypothesis using the configured evaluation strategy.

        Args:
            hypothesis: Hypothesis to evaluate
            evidence: Evidence gathered
            challenges: Challenges raised

        Returns:
            Evaluation result
        """
        if not self._evaluation_strategy:
            raise RuntimeError(
                "No evaluation strategy configured. Set one via "
                "set_evaluation_strategy() or pass to __init__."
            )

        return await self._evaluation_strategy.evaluate(
            hypothesis, evidence, challenges
        )

    # =========================================================================
    # Capability Accessors
    # =========================================================================

    def _get_validation_capability(self) -> ValidationCapability | None:
        """Get ValidationCapability from agent (optional).

        Returns:
            ValidationCapability or None if not configured
        """
        return self.agent.get_capability_by_type(ValidationCapability)

    def _get_agent_pool(self) -> AgentPoolCapability | None:
        """Get AgentPoolCapability for spawning participants."""
        return self.agent.get_capability_by_type(AgentPoolCapability)

    def _get_tracking_capability(self):
        """Get HypothesisTrackingCapability if available."""
        from .tracking import HypothesisTrackingCapability
        return self.agent.get_capability_by_type(HypothesisTrackingCapability)

    # =========================================================================
    # Agent Spawning via AgentPoolCapability
    # =========================================================================

    async def spawn_game_participants(
        self,
        participants: dict[str, str],  # agent_id -> role
    ) -> list[str]:
        """Spawn game participants using AgentPoolCapability.

        Replaces custom spawn_game_agents() method in agents.py.

        Args:
            participants: Mapping of agent_id to role

        Returns:
            List of spawned agent IDs
        """
        pool = self._get_agent_pool()
        if not pool:
            raise RuntimeError(
                "AgentPoolCapability required for spawning participants. "
                "Add AgentPoolCapability to the coordinator agent."
            )

        spawned = []
        role_to_agent_type = {
            "proposer": "polymathera.colony.agents.patterns.games.hypothesis.agents.HypothesisProposerAgent",
            "skeptic": "polymathera.colony.agents.patterns.games.hypothesis.agents.HypothesisSkepticAgent",
            "grounder": "polymathera.colony.agents.patterns.games.hypothesis.agents.HypothesisGrounderAgent",
            "arbiter": "polymathera.colony.agents.patterns.games.hypothesis.agents.HypothesisArbiterAgent",
        }

        for agent_id, role in participants.items():
            if agent_id == self.agent.agent_id:
                continue  # Don't spawn self

            agent_type = role_to_agent_type.get(role)
            if not agent_type:
                logger.warning(f"Unknown role {role} for agent {agent_id}")
                continue

            result = await pool.create_agent(
                agent_type=agent_type,
                metadata={"game_id": self.game_id, "role": role},
                role=role,
            )
            if result.get("created"):
                spawned.append(result["agent_id"])
                logger.info(f"Spawned {role} agent: {result['agent_id']}")
            else:
                logger.error(f"Failed to spawn {role}: {result.get('error')}")

        return spawned

    @override
    @action_executor(writes=["game_id"], exclude_from_planning=True)
    async def start_game(
        self,
        participants: dict[str, str],  # agent_id -> role
        initial_data: dict[str, Any],
        game_id: str | None = None,
        config: dict[str, Any] | None = None
    ) -> str:
        """Start a new hypothesis game.

        Args:
            participants: Participants with roles
                Required: proposer, arbiter
                Optional: skeptic(s), grounder(s)
            initial_data: Must contain 'hypothesis'
            game_id: Optional game ID (generated if None)
            config: Optional configuration (timeouts, etc.)

        Returns:
            Game ID
        """
        # Check permissions using base class method
        can_start, reason = self.can_start_game(self.agent.agent_id, participants)
        if not can_start:
            raise PermissionError(reason)

        # Validate participants
        if "proposer" not in participants.values():
            raise ValueError("Hypothesis game requires a proposer")
        if "arbiter" not in participants.values():
            raise ValueError("Hypothesis game requires an arbiter")

        # Extract hypotheses (supports both single and multiple)
        hypotheses_data = initial_data.get("hypotheses") or initial_data.get("hypothesis")
        if not hypotheses_data:
            raise ValueError("Initial data must contain hypothesis or hypotheses")

        # Normalize to list
        if not isinstance(hypotheses_data, list):
            hypotheses_data = [hypotheses_data]

        hypotheses = [
            Hypothesis(**h) if isinstance(h, dict) else h
            for h in hypotheses_data
        ]

        game_data = HypothesisGameData(
            hypotheses=hypotheses,
            challenges=[],
            additional_evidence=[],
            revision_count=0,
        )

        # Create game state
        state = GameState(
            game_id=game_id,  # Pass through to GameState
            game_type=_HYPOTHESIS_GAME_TYPE,
            conversation_id=hypotheses[0].hypothesis_id if hypotheses else game_id,
            participants=list(participants.keys()),
            roles=participants,
            phase=GamePhase.PROPOSE,
            game_data=game_data.model_dump(),
            config=config or {}
        )
        state.history.append({
            "phase": "setup",
            "timestamp": time.time(),
            "message": f"Hypothesis game initialized with proposer {[aid for aid, r in participants.items() if r == 'proposer'][0]}"
        })

        # Save initial state (emits GAME_STARTED event)
        await self.save_game_state(state, GameEventType.GAME_STARTED.value, move=None)

        return state.game_id

    @override
    async def validate_move(
        self,
        agent_id: str,
        move: ACLMessage,
        state: GameState
    ) -> tuple[bool, str]:
        """Validate move legality (game-specific rules).

        Note: Role permissions are already checked by base class via
        role_permissions. This method handles hypothesis-specific validation.

        Args:
            agent_id: Agent making move
            move: The ACL message
            state: Current game state

        Returns:
            (is_valid, reason) tuple
        """
        game_data = HypothesisGameData(**state.game_data)

        # Game-specific validation: check PROPOSE has hypothesis content
        if move.performative == Performative.PROPOSE:
            content = move.content if isinstance(move.content, dict) else {}
        if state.phase == GamePhase.PROPOSE:
                # Initial proposal must include hypothesis
                if "payload" not in content:
                    return False, "PROPOSE must include hypothesis payload"
        elif state.phase == GamePhase.DEFEND:
                # Defense/revision must include reasoning
                pass  # Content validation if needed

        # Check CHALLENGE has required fields
        if move.performative == Performative.CHALLENGE:
            content = move.content if isinstance(move.content, dict) else {}
            payload = content.get("payload", {})
            if not payload.get("challenged_claim") and not payload.get("reason"):
                return False, "CHALLENGE must specify challenged_claim or reason"

        # Check ACCEPT/REJECT in ARBITRATE phase has reasoning
        if state.phase == GamePhase.ARBITRATE:
            if move.performative in (Performative.ACCEPT, Performative.REJECT):
                content = move.content if isinstance(move.content, dict) else {}
                if not content.get("payload", {}).get("reasoning"):
                    logger.warning(f"Arbiter decision without reasoning: {move.performative}")
                    # Don't fail, just warn

        return True, "Valid move"

    @override
    async def apply_move(
        self,
        state: GameState,
        message: ACLMessage
    ) -> GameState:
        """Transition game state based on message.

        Args:
            state: Current state
            message: ACLMessage causing transition

        Returns:
            New state
        """
        game_data = HypothesisGameData(**state.game_data)

        # Handle transitions based on current phase
        if state.phase == GamePhase.PROPOSE:
            # PROPOSE → CHALLENGE
            state.phase = GamePhase.CHALLENGE

        elif state.phase == GamePhase.CHALLENGE:
            if message.performative == Performative.CHALLENGE:
                # Record challenge
                challenge = ChallengeRecord(
                    challenge_id=f"challenge_{len(game_data.challenges)}",
                    skeptic_id=message.sender,
                    challenged_aspect=message.get_payload().get("challenged_claim", ""),
                    reason=message.get_payload().get("reason", ""),
                    counter_evidence=message.get_payload().get("counter_evidence", [])
                )
                game_data.challenges.append(challenge)
                state.game_data = game_data.model_dump()

                # Stay in CHALLENGE (allow multiple challenges)
                # OR move to GROUND if evidence requested
                # For now, stay in CHALLENGE

            elif message.performative == Performative.ACCEPT:
                # No challenges, move to ARBITRATE
                state.phase = GamePhase.ARBITRATE

        elif state.phase == GamePhase.GROUND:
            # Grounder provided evidence
            evidence = message.get_payload().get("evidence", [])
            game_data.additional_evidence.extend(evidence)
            state.game_data = game_data.model_dump()

            # GROUND → DEFEND
            state.phase = GamePhase.DEFEND

        elif state.phase == GamePhase.DEFEND:
            # Proposer defended or revised
            if message.performative == Performative.PROPOSE:
                # Hypothesis revised
                game_data.revision_count += 1
                revised_data = message.get_payload().get("revised_hypothesis", message.get_payload())
                revised_hypothesis = Hypothesis(**revised_data) if isinstance(revised_data, dict) else revised_data
                # Update current hypothesis in the list
                if game_data.hypotheses and 0 <= game_data.current_hypothesis_index < len(game_data.hypotheses):
                    game_data.hypotheses[game_data.current_hypothesis_index] = revised_hypothesis
                state.game_data = game_data.model_dump()

            # DEFEND → ARBITRATE
            state.phase = GamePhase.ARBITRATE

        elif state.phase == GamePhase.ARBITRATE:
            # Arbiter decision for current hypothesis
            current_hyp = game_data.hypothesis
            hyp_id = current_hyp.hypothesis_id if current_hyp else "unknown"

            if message.performative == Performative.ACCEPT:
                game_data.arbiter_decisions[hyp_id] = "accept"
                game_data.arbiter_reasoning = message.get_payload().get("reasoning", "")
                state.game_data = game_data.model_dump()

                # Check if more hypotheses to process
                if game_data.current_hypothesis_index < len(game_data.hypotheses) - 1:
                    # Move to next hypothesis
                    game_data.current_hypothesis_index += 1
                    game_data.challenges = []  # Reset challenges for next
                    state.game_data = game_data.model_dump()
                    state.phase = GamePhase.PROPOSE  # Restart for next hypothesis
                else:
                    state.phase = GamePhase.TERMINAL

            elif message.performative == Performative.REJECT:
                game_data.arbiter_decisions[hyp_id] = "reject"
                game_data.arbiter_reasoning = message.get_payload().get("reasoning", "")
                state.game_data = game_data.model_dump()

                # Check if more hypotheses to process
                if game_data.current_hypothesis_index < len(game_data.hypotheses) - 1:
                    game_data.current_hypothesis_index += 1
                    game_data.challenges = []
                    state.game_data = game_data.model_dump()
                    state.phase = GamePhase.PROPOSE
                else:
                    state.phase = GamePhase.TERMINAL

            elif message.performative == Performative.REQUEST:
                # Request more info - back to GROUND or DEFEND
                state.phase = GamePhase.GROUND

        return state

    @override
    async def is_terminal(self, state: GameState) -> bool:
        """Check if game is terminal.

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
            state: Terminal game state

        Returns:
            Game outcome
        """
        game_data = HypothesisGameData(**state.game_data)

        # For multi-hypothesis: success if any accepted (configurable via strategy)
        accepted_count = sum(1 for d in game_data.arbiter_decisions.values() if d == "accept")
        rejected_count = sum(1 for d in game_data.arbiter_decisions.values() if d == "reject")
        total_hypotheses = len(game_data.hypotheses)

        # Default: success if majority accepted
        success = accepted_count > rejected_count

        duration = state.ended_at - state.started_at if state.ended_at else None

        # Collect accepted hypotheses
        accepted_hypotheses = [
            h for h in game_data.hypotheses
            if game_data.arbiter_decisions.get(h.hypothesis_id) == "accept"
        ]

        return GameOutcome(
            outcome_type="hypothesis_validated" if success else "hypothesis_rejected",
            success=success,
            result=accepted_hypotheses if accepted_hypotheses else None,
            participants=state.participants,
            rounds_played=len(state.history),
            messages_exchanged=len(state.history),
            duration_seconds=duration,
            summary=(
                f"{accepted_count}/{total_hypotheses} hypotheses accepted "
                f"after {game_data.revision_count} revisions and {len(game_data.challenges)} challenges"
            ),
            lessons_learned=self._extract_lessons(state, game_data),
            metadata={
                "arbiter_reasoning": game_data.arbiter_reasoning,
                "arbiter_decisions": game_data.arbiter_decisions,
                "challenges_count": len(game_data.challenges),
                "revision_count": game_data.revision_count,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
            }
        )

    def _extract_lessons(
        self,
        state: GameState,
        game_data: HypothesisGameData
    ) -> list[str]:
        """Extract lessons learned from game.

        Args:
            state: Final game state
            game_data: Game data

        Returns:
            List of lessons
        """
        # TODO: More sophisticated analysis
        lessons = []

        # If many challenges, hypothesis was weak
        if len(game_data.challenges) > 3:
            lessons.append("Initial hypothesis had insufficient evidence")

        # If many revisions, hypothesis was imprecise
        if game_data.revision_count > 2:
            lessons.append("Hypothesis required significant refinement")

        # If accepted with no challenges, might be too obvious
        if game_data.arbiter_decision == "accept" and len(game_data.challenges) == 0:
            lessons.append("Hypothesis was well-supported from start")

        return lessons

    # =========================================================================
    # Event Handler Overrides
    # =========================================================================

    @event_handler(pattern="{scope_id}:" + GameState.get_key_pattern())
    async def _get_additional_context(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Add hypothesis-specific context for LLM planning."""
        game_event = GameEvent.from_blackboard_event(event)
        if game_event is None or game_event.game_state is None:
            logger.warning(f"Received event with no game state: {event.key}")
            return None  # Not a game event

        game_state = game_event.game_state
        game_data = HypothesisGameData(**game_state.game_data)

        current_hyp = game_data.hypothesis
        context = {
            "hypothesis": current_hyp.model_dump() if current_hyp and hasattr(current_hyp, 'model_dump') else str(current_hyp),
            "hypothesis_index": game_data.current_hypothesis_index,
            "total_hypotheses": len(game_data.hypotheses),
            "arbiter_decisions": game_data.arbiter_decisions,
            "challenges_count": len(game_data.challenges),
            "open_challenges": len([c for c in game_data.challenges if c.status == "open"]),
            "revision_count": game_data.revision_count,
            "challenges_made_by_me": len(self._challenges_made),
            "evidence_provided_by_me": len(self._evidence_provided),
        }
        return EventProcessingResult(
            context_key="hypothesis_context",
            context=context,
        )

    @override
    async def _handle_game_started(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game started event."""
        logger.info(f"Game {self.game_id} started. Role: {self.role.value}")

        if self.role == HypothesisRole.PROPOSER:
            # Proposer should submit initial hypothesis
            # (Usually already done by coordinator in start_game)
            return None

        # Other roles wait for their phase
        return None

    @override
    async def _handle_game_move(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Provide rule-based immediate actions for hypothesis game.
        
        Dispatches to phase-specific handlers based on current game phase.
        Returns an Action to execute immediately, or None for LLM planning.
        """
        game_state = game_event.game_state
        phase = game_state.phase
        # Get role from context or determine from state
        role_str = game_state.get_role(self.agent.agent_id)
        if not role_str:
            return None
        role = HypothesisRole(role_str)

        logger.debug(
            f"Game {game_state.game_id} processing. Phase: {phase.value}, "
            f"Agent: {self.agent.agent_id}, Role: {role.value}"
        )

        # Check if game is terminal
        if game_state.is_terminal():
            return await self._handle_game_complete(game_event)

        # Dispatch based on phase and role
        game_data = HypothesisGameData(**game_state.game_data)

        if phase == GamePhase.PROPOSE:
            return await self._handle_propose_phase(game_event, game_data, role)
        elif phase == GamePhase.CHALLENGE:
            return await self._handle_challenge_phase(game_event, game_data, role)
        elif phase == GamePhase.GROUND:
            return await self._handle_ground_phase(game_event, game_data, role)
        elif phase == GamePhase.DEFEND:
            return await self._handle_defend_phase(game_event, game_data, role)
        elif phase == GamePhase.ARBITRATE:
            return await self._handle_arbitrate_phase(game_event, game_data, role)

        return None

    @override
    async def _handle_game_complete(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game completion."""
        outcome = game_event.game_state.outcome
        if outcome and outcome.success:
            logger.info(f"Game {self.game_id} completed: hypothesis accepted")
        else:
            logger.info(f"Game {self.game_id} completed: hypothesis rejected")

        return EventProcessingResult(
            done=True
        )

    # =========================================================================
    # Phase Handlers
    # =========================================================================

    async def _handle_propose_phase(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        role: HypothesisRole,
    ) -> EventProcessingResult | None:
        """Handle PROPOSE phase."""
        if role != HypothesisRole.PROPOSER:
            return None  # Only proposer acts in this phase

        # TODO: Integrate this with:
        #       - polymathera/colony/agents/patterns/hypothesis
        #       - polymathera/colony/agents/patterns/validation.py
        #       - polymathera/colony/agents/patterns/refinement.py
        #       - polymathera/colony/agents/patterns/synthesis.py
        #       - polymathera/colony/agents/patterns/reflection.py
        # TODO: Expand the game moves to include hypothesis refinement.

        # Check if hypothesis already proposed (usually is by start_game)
        # If we need to propose, create the action
        return None  # Usually handled by start_game

    async def _handle_challenge_phase(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        role: HypothesisRole,
    ) -> EventProcessingResult | None:
        """Handle CHALLENGE phase."""
        if role == HypothesisRole.SKEPTIC:
            return await self._handle_skeptic_challenge(game_event, game_data)
        elif role == HypothesisRole.ARBITER:
            if len(game_data.challenges) == 0:
                if self._should_accept_without_challenge(game_data):
                    return await self._create_accept_action(
                        game_event,
                        "Hypothesis well-supported, no challenges needed"  # TODO: Get this from hypothesis evaluator?
                    )
        return None

    async def _handle_skeptic_challenge(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
    ) -> EventProcessingResult | None:
        """Skeptic logic for challenge phase."""
        if self.llm_config.enabled:
            decision = await self._llm_evaluate_hypothesis(
                game_data,
                decision_type="identify_weaknesses"
            )
            if decision and decision.get("action") == "challenge":
                return await self._create_challenge_action(
                    game_event,
                    game_data,
                    challenged_claim=decision.get("challenged_claim", ""),
                    reason=decision.get("reasoning", ""),
                )
            elif decision and decision.get("action") == "accept":
                return await self._create_accept_action(
                    game_event,
                    reasoning=decision.get("reasoning", "No issues found")
                )

        # Rule-based fallback
        hypothesis = game_data.hypothesis
        if self._has_weak_evidence(hypothesis):
            return await self._create_challenge_action(
                game_event,
                game_data,
                challenged_claim="Insufficient supporting evidence",  # TODO: Get this from hypothesis evaluator?
                reason="The hypothesis lacks sufficient grounding in provided evidence",  # TODO: Get this from hypothesis evaluator?
            )

        return None

    async def _handle_ground_phase(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        role: HypothesisRole,
    ) -> EventProcessingResult | None:
        """Handle GROUND phase."""
        if role != HypothesisRole.GROUNDER:
            return None

        # Grounder provides evidence for open challenges
        open_challenges = [c for c in game_data.challenges if c.status == "open"]
        if open_challenges:
            # Use LLM to gather evidence if enabled
            if self.llm_config.enabled:
                decision = await self._llm_gather_evidence(game_data, open_challenges)
                if decision and decision.get("evidence"):
                    return await self._create_evidence_action(
                        game_event,
                        evidence=decision.get("evidence", []),
                        reasoning=decision.get("reasoning", ""),
                    )

            # Rule-based: provide placeholder evidence
            return await self._create_evidence_action(
                game_event,
                evidence=["Additional context gathered"],
                reasoning="Providing supporting context for challenged claims",
            )
        return None

    async def _handle_defend_phase(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        role: HypothesisRole,
    ) -> EventProcessingResult | None:
        """Handle DEFEND phase."""
        if role != HypothesisRole.PROPOSER:
            return None

        # Proposer defends or revises hypothesis
        open_challenges = [c for c in game_data.challenges if c.status == "open"]

        if self.llm_config.enabled:
            decision = await self._llm_decide_defense(game_data, open_challenges)
            if decision:
                if decision.get("action") == "revise":
                    return await self._create_revise_action(
                        game_event,
                        game_data,
                        revised_hypothesis=decision.get("revised_hypothesis"),
                        reasoning=decision.get("reasoning", ""),
                    )
                elif decision.get("action") == "defend":
                    return await self._create_defend_action(
                        game_event,
                        game_data,
                        reasoning=decision.get("reasoning", ""),
                    )

        # Rule-based: if many challenges, revise; otherwise defend
        if len(open_challenges) > 2:
            return await self._create_revise_action(
                game_event,
                game_data,
                revised_hypothesis=None,  # Will use current with modifications
                reasoning="Revising based on multiple challenges",
            )
        else:
            return await self._create_defend_action(
                game_event,
                game_data,
                reasoning="Defending hypothesis with existing evidence",
            )

    async def _handle_arbitrate_phase(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        role: HypothesisRole,
    ) -> EventProcessingResult | None:
        """Handle ARBITRATE phase."""
        if role != HypothesisRole.ARBITER:
            return None

        # Arbiter makes final judgment
        if self.llm_config.enabled:
            decision = await self._llm_evaluate_hypothesis(
                game_data,
                decision_type="final_judgment"
            )
            if decision:
                if decision.get("action") == "accept":
                    return await self._create_accept_action(
                        game_event,
                        reasoning=decision.get("reasoning", "Hypothesis accepted"),
                    )
                elif decision.get("action") == "reject":
                    return await self._create_reject_action(
                        game_event,
                        reasoning=decision.get("reasoning", "Hypothesis rejected"),
                    )
                elif decision.get("action") == "request_more":
                    return await self._create_request_more_action(
                        game_event,
                        reasoning=decision.get("reasoning", "Need more evidence"),
                    )

        # Rule-based judgment
        unresolved = [c for c in game_data.challenges if c.status == "open"]
        if len(unresolved) == 0:
            return await self._create_accept_action(
                game_event,
                reasoning="All challenges addressed, hypothesis accepted",
            )
        elif len(unresolved) <= 1 and game_data.revision_count > 0:
            return await self._create_accept_action(
                game_event,
                reasoning="Hypothesis revised to address concerns",
            )
        else:
            return await self._create_reject_action(
                game_event,
                reasoning=f"Unresolved challenges: {len(unresolved)}",
            )

    # =========================================================================
    # LLM Helpers
    # =========================================================================

    async def _llm_evaluate_hypothesis(
        self,
        game_data: HypothesisGameData,
        decision_type: str,
    ) -> dict[str, Any] | None:
        """Use LLM to evaluate hypothesis."""
        context = {
            "hypothesis": game_data.hypothesis.model_dump() if hasattr(game_data.hypothesis, 'model_dump') else game_data.hypothesis,
            "challenges": [c.model_dump() for c in game_data.challenges],
            "additional_evidence": game_data.additional_evidence,
            "revision_count": game_data.revision_count,
        }

        options = ["accept", "reject", "challenge", "request_more"]

        try:
            return await self.llm_config.generate_decision(
                self.agent,
                game_type=_HYPOTHESIS_GAME_TYPE,
                decision_type=decision_type,
                context=context,
                options=options,
            )
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            return None

    async def _llm_gather_evidence(
        self,
        game_data: HypothesisGameData,
        challenges: list[ChallengeRecord],
    ) -> dict[str, Any] | None:
        """Use LLM to determine what evidence to gather."""
        # TODO: This would integrate with agent's capabilities to fetch evidence
        # TODO: Use either the GroundingAgent or both the IncrementalQueryCapability and GroundingCapability.
        # TODO: Define a new class AgentCapabilityGroup or AgentCapabilityPack to group capabilities together.
        # TODO: Provide guidance on when to use an AgentCapabilityPack vs an agent:
        #       - Both `Agent` and `AgentCapabilityPack` allow composition: Combining multiple capabilities that depend on each other but exposing a single interface.
        #       - But we use an `Agent` when we need:
        #           - Concurrency: Firing multiple agents to run multiple composiite actions in parallel.
        #           - Encapsulation: Hiding the individual actions from the parent action planner and exposing only the agent spawning action and event-driven interactions with the agent.
        return None

    async def _llm_decide_defense(
        self,
        game_data: HypothesisGameData,
        challenges: list[ChallengeRecord],
    ) -> dict[str, Any] | None:
        """Use LLM to decide how to defend/revise hypothesis."""
        context = {
            "hypothesis": game_data.hypothesis.model_dump() if hasattr(game_data.hypothesis, 'model_dump') else game_data.hypothesis,
            "challenges": [c.model_dump() for c in challenges],
            "additional_evidence": game_data.additional_evidence,
        }

        try:
            return await self.llm_config.generate_decision(
                self.agent,
                game_type=_HYPOTHESIS_GAME_TYPE,
                decision_type="defend_or_revise",
                context=context,
                options=["defend", "revise"],
            )
        except Exception as e:
            logger.warning(f"LLM defense decision failed: {e}")
            return None

    # =========================================================================
    # Decision Helpers
    # =========================================================================

    def _has_weak_evidence(self, hypothesis: Hypothesis) -> bool:
        """Check if hypothesis has weak evidence."""
        # TODO: Is there a capability or agent to do hypothesis evaluation?
        # Simple heuristic: check if supporting_evidence is empty or minimal
        evidence = getattr(hypothesis, 'supporting_evidence', [])
        return len(evidence) < 2

    def _should_accept_without_challenge(self, game_data: HypothesisGameData) -> bool:
        """Check if hypothesis should be accepted without challenges."""
        # TODO: Is there a capability or agent to do hypothesis evaluation?
        hypothesis = game_data.hypothesis
        # Accept if evidence is strong
        evidence = getattr(hypothesis, 'supporting_evidence', [])
        return len(evidence) >= 3

    # =========================================================================
    # Action Creators
    # =========================================================================

    async def _create_challenge_action(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        challenged_claim: str,
        reason: str,
    ) -> EventProcessingResult | None:
        """Create action to challenge hypothesis."""
        challenge = ChallengeRecord(
            challenge_id=f"challenge_{len(self._challenges_made)}",
            skeptic_id=self.agent.agent_id,
            challenged_aspect=challenged_claim,
            reason=reason,
        )
        self._challenges_made.append(challenge)

        message = ACLMessage(
            performative=Performative.CHALLENGE,
            sender=self.agent.agent_id,
            receivers=[r for r, role in game_data.hypothesis.__dict__.items() if role == "proposer"] if hasattr(game_data.hypothesis, '__dict__') else [],
            # receivers=[game_data.hypothesis.proposer] if hasattr(game_data.hypothesis, 'proposer') else [],
            content={
                "schema": "challenge",
                "payload": {
                    "challenge_id": challenge.challenge_id,
                    "challenged_claim": challenged_claim,
                    "reason": reason,
                }
            }
        )

        action = Action(
            action_id=f"challenge_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(),  # Use current version since this is an immediate action
            },
            reasoning=f"Challenging: {challenged_claim[:50]}... Reason: {reason[:50]}..."
        )
        return EventProcessingResult(immediate_action=action)

    async def _create_accept_action(
        self,
        game_event: GameEvent,
        reasoning: str,
    ) -> EventProcessingResult | None:
        """Create action to accept hypothesis."""
        message = ACLMessage(
            performative=Performative.ACCEPT,
            sender=self.agent.agent_id,
            receivers=[],
            content={
                "schema": "acceptance",
                "payload": {
                    "reasoning": reasoning,
                }
            }
        )

        action = Action(
            action_id=f"accept_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(),  # Use current version since this is an immediate action
            },
            reasoning=reasoning
        )
        return EventProcessingResult(immediate_action=action)

    async def _create_reject_action(
        self,
        game_event: GameEvent,
        reasoning: str,
    ) -> EventProcessingResult | None:
        """Create action to reject hypothesis."""
        message = ACLMessage(
            performative=Performative.REJECT,
            sender=self.agent.agent_id,
            receivers=[],
            content={
                "schema": "rejection",
                "payload": {
                    "reasoning": reasoning,
                }
            }
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
            reasoning=reasoning
        )
        return EventProcessingResult(immediate_action=action)

    async def _create_request_more_action(
        self,
        game_event: GameEvent,
        reasoning: str,
    ) -> EventProcessingResult | None:
        """Create action to request more evidence."""
        message = ACLMessage(
            performative=Performative.REQUEST,
            sender=self.agent.agent_id,
            receivers=[],
            content={
                "schema": "request_evidence",
                "payload": {
                    "reasoning": reasoning,
                }
            }
        )

        action = Action(
            action_id=f"request_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(), # Use current version since this is an immediate action
            },
            reasoning=reasoning
        )
        return EventProcessingResult(immediate_action=action)

    async def _create_evidence_action(
        self,
        game_event: GameEvent,
        evidence: list[str],
        reasoning: str,
    ) -> EventProcessingResult | None:
        """Create action to provide evidence."""
        self._evidence_provided.extend(evidence)

        message = ACLMessage(
            performative=Performative.INFORM,
            sender=self.agent.agent_id,
            receivers=[],
            content={
                "schema": "evidence",
                "payload": {
                    "evidence": evidence,
                    "reasoning": reasoning,
                }
            }
        )

        action = Action(
            action_id=f"evidence_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(), # Use current version since this is an immediate action
            },
            reasoning=reasoning
        )
        return EventProcessingResult(immediate_action=action)

    async def _create_defend_action(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        reasoning: str,
    ) -> EventProcessingResult | None:
        """Create action to defend hypothesis."""
        message = ACLMessage(
            performative=Performative.INFORM,
            sender=self.agent.agent_id,
            receivers=[],
            content={
                "schema": "defense",
                "payload": {
                    "reasoning": reasoning,
                }
            }
        )

        action = Action(
            action_id=f"defend_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(), # Use current version since this is an immediate action
            },
            reasoning=reasoning
        )
        return EventProcessingResult(immediate_action=action)

    async def _create_revise_action(
        self,
        game_event: GameEvent,
        game_data: HypothesisGameData,
        revised_hypothesis: dict[str, Any] | None,
        reasoning: str,
    ) -> EventProcessingResult | None:
        """Create action to revise hypothesis."""
        # If no revised hypothesis provided, use current with note
        payload = revised_hypothesis or (
            game_data.hypothesis.model_dump()
            if hasattr(game_data.hypothesis, 'model_dump')
            else game_data.hypothesis
        )

        message = ACLMessage(
            performative=Performative.PROPOSE,
            sender=self.agent.agent_id,
            receivers=[],
            content={
                "schema": "revised_hypothesis",
                "payload": {
                    "revised_hypothesis": payload,
                    "reasoning": reasoning,
                }
            }
        )

        action = Action(
            action_id=f"revise_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
                "expected_version": game_event.game_state.get_digest(), # Use current version since this is an immediate action
            },
            reasoning=reasoning
        )
        return EventProcessingResult(immediate_action=action)


    # ============================================================================
    # TODO: Integrate with hypothesis analysis capability. These methods we moved
    # from the change impact analyzer.
    # ============================================================================
    async def _play_skeptic_role(self, hypothesis: dict) -> dict:
        """Play skeptic role - find counter-evidence."""
        prompt = f"""You are playing the SKEPTIC role in validating an impact hypothesis.

Hypothesis: {hypothesis.get('claim', '')}

Your task: Find counter-evidence or weaknesses in this claim.
Look for:
- Cases where the impact might not occur
- Alternative explanations
- Missing evidence

Respond with:
- counter_evidence: List of counter-examples or weaknesses
- confidence: How confident are you in your counter-argument (0-1)
- recommendation: accept, reject, or needs_more_evidence
"""

        response = await self.agent.infer(
            prompt=prompt,
            context_page_ids=[self.page_id] if self.page_id else []
        )

        return {
            "role": "skeptic",
            "analysis": response.generated_text,
            "page_id": self.page_id
        }

    async def _play_proposer_role(self, hypothesis: dict) -> dict:
        """Play proposer role - provide supporting evidence."""
        prompt = f"""You are playing the PROPOSER role defending an impact hypothesis.

Hypothesis: {hypothesis.get('claim', '')}

Your task: Provide supporting evidence for this claim.
Look for:
- Direct evidence of the impact
- Similar patterns that support the claim
- Logical reasoning supporting the impact

Respond with:
- evidence: List of supporting evidence
- confidence: How confident are you (0-1)
- additional_claims: Related impacts you discovered
"""

        response = await self.agent.infer(
            prompt=prompt,
            context_page_ids=[self.page_id] if self.page_id else []
        )

        return {
            "role": "proposer",
            "analysis": response.generated_text,
            "page_id": self.page_id
        }


