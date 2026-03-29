"""Game state management for agent protocols.

This module implements game state tracking and protocol management:
- GameState: Current state of a game instance
- GamePhase: Phases in extensive-form games
- GameOutcome: Terminal outcomes
- GameProtocolCapability: Base class for game protocols
- RolePermissions: Declarative role-based move validation

Games are explicit finite state machines where:
- Phases define valid agent actions
- Transitions are triggered by messages
- Outcomes are validated and stored
- History is tracked for learning

Programming Model for Library Users:
-------------------------------------
To create a new game protocol:

1. Define roles and their permissions per phase:
   ```python
   PERMISSIONS = RolePermissions({
       ("coordinator", GamePhase.OFFER): {Performative.REQUEST},
       ("participant", GamePhase.OFFER): {Performative.PROPOSE},
       ("participant", GamePhase.COUNTER_OFFER): {Performative.PROPOSE, Performative.ACCEPT},
   })
   ```

2. Subclass GameProtocolCapability with your permissions:
   ```python
   class MyGameProtocol(GameProtocolCapability):
       role_permissions = PERMISSIONS

       async def apply_move(self, state, move):
           # Implement state transitions
           ...
   ```

3. The base class handles:
   - Role-based validation via `validate_move_permissions()`
   - Terminal state detection
   - Blackboard I/O and event emission

4. For LLM-augmented decision making, use `LLMReasoningConfig` in your policy:
   ```python
   class MyGamePolicy(CacheAwareActionPolicy):
       def __init__(self, ..., use_llm_reasoning: bool = False):
           self.llm_config = LLMReasoningConfig(enabled=use_llm_reasoning)

       async def _decide_action(self, game_data):
           # Try LLM - returns {"action": ..., "reasoning": ...} or None
           decision = await self.llm_config.generate_decision(
               self.agent, game_type="my_game", decision_type="choose_move",
               context={...}, options=["accept", "reject", "counter"]
           )
           if decision:
               return decision["action"], decision["reasoning"]
           # Fall back to rules with rule-based reasoning
           return self._rule_based_decision(game_data)
   ```
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar
from overrides import override
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import hashlib
import logging

from ...base import Agent, AgentCapability, CapabilityResultFuture
from ...blackboard import EnhancedBlackboard, BlackboardEvent, CombinationFilter
from ...blackboard.protocol import GameStateProtocol
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ..actions.policies import action_executor
from ..hooks import hookable
from ..events import event_handler, EventProcessingResult
from ...models import Action, PolicyREPL
from .acl import ACLMessage, Performative


# Type variables for generic game policy
TGameData = TypeVar("TGameData", bound=BaseModel)
TRole = TypeVar("TRole", bound=Enum)


logger = logging.getLogger(__name__)


# ============================================================================
# Role-Permission Matrix
# ============================================================================

class RolePermissions:
    """Declarative role-based permission matrix for game moves.

    Maps (role, phase) -> set of allowed performatives. Used by
    GameProtocolCapability.validate_move_permissions() for consistent
    role-based validation across all game types.

    Example:
        ```python
        # Define permissions for a negotiation game
        NEGOTIATION_PERMISSIONS = RolePermissions({
            # Coordinator can only start games
            ("coordinator", GamePhase.OFFER): set(),  # Cannot make offers

            # Participants can propose, accept, reject in various phases
            ("participant", GamePhase.OFFER): {Performative.PROPOSE},
            ("participant", GamePhase.COUNTER_OFFER): {
                Performative.PROPOSE, Performative.ACCEPT, Performative.REJECT
            },
            ("participant", GamePhase.AGREE): {
                Performative.ACCEPT, Performative.REJECT
            },

            # Mediator can only propose during mediation
            ("mediator", GamePhase.EVALUATE): {Performative.PROPOSE},
        })

        class MyProtocol(GameProtocolCapability):
            role_permissions = NEGOTIATION_PERMISSIONS
        ```

    Design Notes:
        - Missing (role, phase) entries mean NO permissions (fail-closed)
        - Use `RolePermissions.any_phase(role, perfs)` to allow in all phases
        - Permissions are checked BEFORE apply_move is called
    """

    def __init__(
        self,
        permissions: dict[tuple[str, GamePhase], set[Performative]] | None = None
    ):
        """Initialize with permission mapping.

        Args:
            permissions: Dict mapping (role, phase) -> allowed performatives
        """
        self._permissions: dict[tuple[str, GamePhase], set[Performative]] = permissions or {}

    def allows(self, role: str, phase: GamePhase, performative: Performative) -> bool:
        """Check if role can use performative in phase.

        Args:
            role: Agent's role in the game
            phase: Current game phase
            performative: The performative being attempted

        Returns:
            True if allowed, False otherwise
        """
        key = (role, phase)
        if key not in self._permissions:
            return False  # Fail-closed: no entry means no permission
        return performative in self._permissions[key]

    def allowed_performatives(self, role: str, phase: GamePhase) -> set[Performative]:
        """Get all performatives allowed for role in phase.

        Args:
            role: Agent's role
            phase: Current phase

        Returns:
            Set of allowed performatives (empty if none)
        """
        return self._permissions.get((role, phase), set())

    def add(
        self,
        role: str,
        phase: GamePhase,
        performatives: set[Performative]
    ) -> "RolePermissions":
        """Add permissions (returns self for chaining).

        Args:
            role: Role name
            phase: Game phase
            performatives: Allowed performatives

        Returns:
            self for method chaining
        """
        self._permissions[(role, phase)] = performatives
        return self

    def any_phase(self, role: str, performatives: set[Performative]) -> "RolePermissions":
        """Allow performatives for role in ALL phases.

        Args:
            role: Role name
            performatives: Allowed performatives

        Returns:
            self for method chaining
        """
        for phase in GamePhase:
            self._permissions[(role, phase)] = performatives
        return self

    @classmethod
    def permissive(cls) -> "RolePermissions":
        """Create a permissive policy that allows all performatives.

        Useful for testing or games without role restrictions.
        """
        perms = cls()
        all_perfs = set(Performative)
        # We don't know all roles, so this is a marker
        perms._allow_all = True
        return perms

    def __contains__(self, key: tuple[str, GamePhase]) -> bool:
        """Check if (role, phase) has any permissions defined."""
        if getattr(self, '_allow_all', False):
            return True
        return key in self._permissions


# ============================================================================
# LLM Reasoning Configuration
# ============================================================================

class LLMReasoningConfig:
    """Configuration for LLM-augmented decision making in game policies.

    A consistent interface for policies to optionally use LLM
    for generating action reasoning, justifications, and strategic decisions.
    Provides a single method `generate_decision` that returns both the action
    and reasoning in one LLM call. When LLM is disabled or fails, returns None
    and the caller should fall back to rule-based logic with rule-based reasoning.

    Design principle: Don't waste LLM calls on just explaining rule-based decisions.
    If using LLM, get decision + reasoning together. If using rules, use rule-based
    reasoning (no extra LLM call).

    Example:
        ```python
        class MyGamePolicy(CacheAwareActionPolicy):
            def __init__(self, agent, use_llm_reasoning: bool = False, **kwargs):
                super().__init__(agent, **kwargs)
                self.llm_config = LLMReasoningConfig(
                    use_llm_reasoning=use_llm_reasoning,
                    temperature=0.3,
                    max_tokens=500,
                )

            async def _decide_action(self, game_data):
                # Try LLM first
                decision = await self.llm_config.generate_decision(
                    self.agent,
                    game_type="my_game",
                    decision_type="choose_move",
                    context={"state": game_data},
                    options=["accept", "reject", "counter"]
                )
                if decision:
                    # LLM decided - use its action and reasoning
                    return decision["action"], decision["reasoning"]
                else:
                    # Fall back to rules with rule-based reasoning
                    return self._rule_based_decision(game_data)
        ```
    """

    def __init__(
        self,
        enabled: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        """Initialize LLM reasoning config.

        Args:
            enabled: Whether to use LLM for reasoning (False = rule-based only)
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens for LLM response
        """
        self.enabled = enabled
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_decision(
        self,
        agent: Agent,
        game_type: str,
        decision_type: str,
        context: dict[str, Any],
        options: list[str],
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Generate a structured decision with reasoning using LLM.

        Returns both the action and reasoning in one call. If LLM is disabled
        or fails, returns None and caller should fall back to rule-based logic.

        Args:
            agent: Agent to use for inference
            game_type: Type of game (for prompt context)
            decision_type: Type of decision to be made (e.g., "choose_move", "accept_reject", "counter_offer")
            context: Context dict with game state, preferences, etc.
            options: Available action options
            json_schema: Optional JSON schema for structured output

        Returns:
            Parsed decision dict with at least {"action": "...", "reasoning": "..."} or None
        """
        if not self.enabled:
            return None

        prompt = self._build_decision_prompt(game_type, decision_type, context, options)
        schema = json_schema or {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": options},
                "reasoning": {"type": "string", "description": "Brief explanation"},
            },
            "required": ["action", "reasoning"]
        }

        try:
            response = await agent.infer(
                prompt=prompt,
                json_schema=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            import json
            return json.loads(response.content)
        except Exception:
            return None

    def _build_decision_prompt(
        self,
        game_type: str,
        decision_type: str,
        context: dict[str, Any],
        options: list[str]
    ) -> str:
        """Build prompt for decision generation."""
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        options_str = ", ".join(options)
        return f"""You are a game-playing agent in a game of type {game_type}.

Decision needed: {decision_type}

Context:
{context_str}

Available actions: [{options_str}]

Choose the best action and explain why. Respond with JSON only."""


class GameEventType(str, Enum):
    """Types of game events emitted on blackboard."""

    GAME_STARTED = "game_started"
    GAME_MOVE = "game_move"


class GamePhase(str, Enum):
    """Phases in extensive-form games.

    Different games have different phase sequences.
    """

    # Hypothesis game phases
    PROPOSE = "propose"
    CHALLENGE = "challenge"
    GROUND = "ground"
    DEFEND = "defend"
    ARBITRATE = "arbitrate"

    # Contract net phases
    ANNOUNCE = "announce"
    BID = "bid"
    AWARD = "award"
    EXECUTE = "execute"
    VALIDATE = "validate"

    # Consensus game phases
    NOMINATE = "nominate"
    VOTE = "vote"
    COUNT = "count"
    DECLARE = "declare"

    # Negotiation game phases
    OFFER = "offer"
    COUNTER_OFFER = "counter_offer"
    EVALUATE = "evaluate"
    AGREE = "agree"

    # Terminal phase
    TERMINAL = "terminal"


class GameOutcome(BaseModel):
    """Terminal outcome of a game.

    Records the final result and statistics.
    """

    # TODO: Add participant payoffs, scores and rankings?

    outcome_type: str = Field(
        description="Type of outcome (specific to game type)"
    )

    success: bool = Field(
        description="Whether game succeeded (reached productive outcome)"
    )

    result: Any | None = Field(
        default=None,
        description="Game result (accepted hypothesis, awarded task, consensus, etc.)"
    )

    participants: list[str] = Field(
        default_factory=list,
        description="All participating agent IDs"
    )

    rounds_played: int = Field(
        description="Number of rounds/phases executed"
    )

    messages_exchanged: int = Field(
        description="Total messages exchanged"
    )

    duration_seconds: float | None = Field(
        default=None,
        description="Game duration"
    )

    # Quality metrics
    consensus_level: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Level of consensus reached"
    )

    conflicts_resolved: int = Field(
        default=0,
        description="Number of conflicts resolved"
    )

    # Metadata
    summary: str | None = Field(
        default=None,
        description="Summary of game outcome"
    )

    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Lessons for future games"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional outcome metadata"
    )


class GameState(BaseModel):
    """State of a running game instance.

    Tracks:
    - Current phase
    - Participants and their roles
    - ACLMessage history
    - Game-specific data
    - Outcome (when terminal)

    Examples:
        Hypothesis game state:
        ```python
        state = GameState(
            game_type="hypothesis_game",
            conversation_id="hyp_analysis_auth_flow",
            participants=["proposer_001", "skeptic_002", "grounder_003", "arbiter_004"],
            roles={
                "proposer_001": "proposer",
                "skeptic_002": "skeptic",
                "grounder_003": "grounder",
                "arbiter_004": "arbiter"
            },
            phase=GamePhase.CHALLENGE,
            game_data={
                "hypothesis_id": "hyp_123",
                "challenges": ["challenge_001"],
                "evidence_requested": ["AuthManager implementation"]
            }
        )
        ```
    """

    # Identification
    game_id: str = Field(
        default_factory=lambda: f"game_{uuid.uuid4().hex}",
        description="Unique game identifier"
    )

    game_type: str = Field(
        description="Type of game: 'hypothesis_game', 'contract_net', 'consensus_game', etc."
    )

    conversation_id: str = Field(
        description="Conversation ID (may span multiple games)"
    )

    # Participants
    participants: list[str] = Field(
        default_factory=list,
        description="All participating agent IDs"
    )

    roles: dict[str, str] = Field(
        default_factory=dict,
        description="Agent ID -> role mapping"
    )

    # State
    phase: GamePhase = Field(
        description="Current game phase"
    )

    history: list[ACLMessage] = Field(
        default_factory=list,
        description="ACLMessage history (complete protocol transcript)"
    )

    # Game-specific data
    game_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Game-specific state (hypotheses, bids, votes, etc.)"
    )

    # Outcome
    outcome: GameOutcome | None = Field(
        default=None,
        description="Terminal outcome (None if game still running)"
    )

    # Timing
    started_at: float = Field(
        default_factory=time.time,
        description="When game started"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="When game last updated"
    )

    ended_at: float | None = Field(
        default=None,
        description="When game ended"
    )

    # Configuration
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Game configuration (timeouts, thresholds, etc.)"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional game metadata"
    )

    def get_digest(self) -> str:
        """Get a digest of the game state to be used for OCC validation.

        Returns:
            Digest of the game state
        """
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()

    def add_message(self, message: ACLMessage) -> None:
        """Add message to history.

        Args:
            message: ACLMessage to add
        """
        self.history.append(message)

    def get_messages_from(self, agent_id: str) -> list[ACLMessage]:
        """Get messages from specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of messages from that agent
        """
        return [msg for msg in self.history if msg.sender == agent_id]

    def get_messages_in_phase(self, phase: GamePhase) -> list[ACLMessage]:
        """Get messages from specific phase.

        Args:
            phase: Game phase

        Returns:
            List of messages from that phase
        """
        # Would need to track phase transitions in history
        # Placeholder for now
        return []

    def is_terminal(self) -> bool:
        """Check if game has reached terminal state.

        Returns:
            True if terminal
        """
        return self.phase == GamePhase.TERMINAL or self.outcome is not None

    def get_role(self, agent_id: str) -> str | None:
        """Get role of an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Role or None
        """
        return self.roles.get(agent_id)

    def to_blackboard_entry(self) -> dict[str, Any]:
        """Convert to blackboard entry format.

        Returns:
            Dictionary for blackboard storage
        """
        return self.model_dump()

    @classmethod
    def from_blackboard_entry(cls, entry: dict[str, Any]) -> GameState:
        """Reconstruct from blackboard entry.

        Args:
            entry: Blackboard entry

        Returns:
            GameState instance
        """
        return cls(**entry)


# ============================================================================
# Game Protocol Capability (New Pattern)
# ============================================================================

class GameEvent(BaseModel):
    agent_id: str
    game_event_type: str
    game_state: GameState
    version: int = 0
    tags: set[str]
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        agent_id: str,
        game_event_type: str,
        game_state: GameState,
        move: ACLMessage | None = None
    ) -> GameEvent:
        assert move.sender == agent_id if move else True, "Move sender must be this agent"
        tags = {
            "game",
            game_state.game_type,
            game_state.phase.value,
            "terminal" if game_state.is_terminal() else "active",
        }
        metadata = {
            "last_move_by": agent_id,
            "game_event_type": game_event_type,
            "move_performative": move.performative.value if move else None
        }
        return cls(
            agent_id=agent_id,
            game_id=game_state.game_id,
            game_event_type=game_event_type,
            game_state=game_state,
            tags=tags,
            metadata=metadata
    )

    @classmethod
    def from_blackboard_event(cls, event: BlackboardEvent) -> GameEvent:
        game_state = GameState.from_blackboard_entry(event.value) if event.value else None
        return cls(
            agent_id=event.agent_id,
            game_event_type=event.metadata.get("game_event_type", "") if event else "",
            game_state=game_state,
            version=event.version,
            tags=event.tags,
            metadata=event.metadata
        )



class GameProtocolCapability(AgentCapability, ABC, Generic[TGameData, TRole]):
    """Game protocol capability for participants to interact with a specific multi-agent game
    by exposing @action_executor decorated methods of this class to their own action policies,
    by streaming events to their action policy's event queue and by waiting for the game result future.
    Subclasses only need to implement game-specific event handlers and action creators.

    All game participants share the same `scope_id` (typically the `game_id`),
    enabling them to see each other's moves and events via the shared blackboard.

    Game protocols define:
    - Valid moves (messages) in each phase via `role_permissions`
    - Phase transitions via `apply_move`. Implements role-specific handling of game phases
    - Terminal conditions via `is_terminal`
    - Outcome computation via `compute_outcome`
    - Event handling via `@event_handler` decorated methods to provide the
      agent action policy with action planning context and immediate hardoded rule-based reactions.
        - Context includes game state data and versions for OCC

    Usage modes (inherited from AgentCapability):
    1. **Local mode**: Participant in a game
       ```python
       capability = NegotiationGameProtocol(agent=self, game_id="game_001")
       # scope_id = game_id, so all participants share the namespace
       ```

    2. **Remote mode**: Parent monitoring/controlling a game
       ```python
       capability = NegotiationGameProtocol(agent=parent, game_id="game_001")
       await capability.stream_events_to_queue(self.get_event_queue())
       future = await capability.get_result_future()
       outcome = await future
       ```

    Subclasses implement game-specific validation and state transitions,
    game rules and can be used as action providers for `CacheAwareActionPolicy`
    or `EventDrivenActionPolicy` for event-driven game execution.
    Subclasses should:
    1. Define `role_permissions` class attribute with allowed (role, phase) -> performatives
    2. Override `validate_move` for game-specific validation beyond role permissions
    3. Override `apply_move` for state transitions
    4. Override `compute_outcome` for final outcome calculation

    Example:
        ```python
        class HypothesisGameProtocol(GameProtocolCapability):
            game_type = "hypothesis_game"

            role_permissions = RolePermissions({
                ("proposer", GamePhase.PROPOSE): {Performative.PROPOSE},
                ("challenger", GamePhase.CHALLENGE): {Performative.CHALLENGE},
            })

            def validate_move(self, agent_id, move, state):
                # First check role permissions (handled by base class)
                # Then add game-specific validation
                ...

            def apply_move(self, state, move):
                # Apply move and return new state
                ...

        # In policy
        if not self.agent.has_capability(HypothesisGameCapability.get_capability_name()):
            capability = HypothesisGameCapability(self.agent, game_id="game_001")
            await capability.initialize()
            self.agent.add_capability(capability)
        cap = self.agent.get_capability(HypothesisGameCapability.get_capability_name())
        state = await cap.load_game_state()
        valid, reason = cap.validate_move(state, my_move)
        ```

    Type Parameters:
        TGameData: Pydantic model for game-specific data (e.g., NegotiationGameData)
        TRole: Enum for game roles (e.g., NegotiationRole)

    Example:
        ```python
        class NegotiationGameProtocol(
            GameProtocolCapability[NegotiationGameData, NegotiationRole]
        ):

            async def _handle_game_move(self, state, game_event):
                # Game-specific phase handling
                ...
        ```

    To implement a new game protocol, subclass `GameProtocolCapability` and:
    1. Set class attributes: `game_protocol_cls`, `game_data_cls`, `role_cls`
    2. Override `_handle_game_started()` for game start logic
    3. Override `_handle_game_move()` for phase-specific handling
    4. Implement action creators for each game move type
    """

    input_patterns = [GameStateProtocol.state_pattern(namespace="game")]

    # Override in subclasses to define role-based permissions
    role_permissions: RolePermissions = RolePermissions()

    def __init__(
        self,
        *,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        game_id: str | None = None,
        game_type: str,
        role: str | None = None,
        use_llm_reasoning: bool = False,
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 500,
    ):
        """Initialize game protocol capability.

        Args:
            agent: Owning agent
            game_type: Type of game (e.g., "negotiation", "hypothesis")
            game_id: Game instance ID. All participants should use the same game_id
                to share the same blackboard namespace for coordination.
            game_type: Type of game (e.g., "negotiation", "hypothesis")
            role: Agent's role in the game (e.g., "participant", "coordinator").
                If None, will be looked up from game state when available.
            use_llm_reasoning: If True, use LLM for strategic decisions
            llm_temperature: Temperature for LLM inference
            llm_max_tokens: Max tokens for LLM response
        """
        # Use game_id as scope_id so all participants share the namespace
        self.game_id = game_id
        self.namespace = self.get_blackboard_namespace(scope, game_id)
        self.game_type = game_type
        self.role: TRole = TRole(role)
        super().__init__(agent, scope_id=self.namespace)

        # LLM reasoning configuration
        self.llm_config = LLMReasoningConfig(
            enabled=use_llm_reasoning,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )

    def get_action_group_description(self) -> str:
        return (
            f"Game Protocol ({self.game_type}) — multi-agent structured interaction. "
            f"Moves are validated against role_permissions (role+phase→allowed performatives) "
            f"and applied with optimistic concurrency control. "
            f"Game actions (start_game, submit_move, load_game_state) are excluded from planning — "
            f"the planner uses game-specific subclass actions instead."
        )

    async def initialize(self) -> None:
        """Initialize game blackboard. Call before using capability."""
        if self._blackboard is not None:
            return  # Already initialized
        self._blackboard = await self.get_blackboard()

    def get_blackboard_namespace(self, scope: BlackboardScope, game_id: str) -> str:
        # Make namespace depend on game_id to allow multiple games of the same type to coexist.
        return get_scope_prefix(scope, self.agent, namespace='games', game_type=self.game_type, game=game_id)

    def _get_result_key(self) -> str:
        """Get blackboard key for game result."""
        return GameStateProtocol.result_key(namespace="game")

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream game events to a queue.

        Streams "write" events for game state changes. The game event type is
        stored in `metadata["game_event_type"]`.

        Args:
            event_queue: Queue to stream events into. Usually the local event queue of an ActionPolicy.
        """
        # IMPORTANT:
        # - The semantic game events (GameEventType.*) are used for higher-level coordination.
        # - We stream the canonical blackboard "write" event for the game state key because
        #   all semantic game events are associated with blackboard writes. The game event
        #   type is stored in the `metadata["game_event_type"]` field of the blackboard entry which
        #   is then copied to the metadata of the emitted blackboard event.
        blackboard = await self.get_blackboard()
        blackboard.stream_events_to_queue(
            event_queue,
            CombinationFilter(
                event_types={"write"},  # All game events are write events
                pattern=GameStateProtocol.state_pattern(namespace="game"), # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
                checker=lambda event: (
                    "game" in event.tags and
                    self.game_type in event.tags and
                    event.metadata.get("game_event_type", "").startswith(f"game_{self.game_type}_")
                )
            )
        )

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for game result.

        Returns a future that resolves when the game reaches terminal state.

        Returns:
            Future that resolves with GameOutcome
        """
        blackboard = await self.get_blackboard()
        return CapabilityResultFuture(
            result_key=self._get_result_key(),
            blackboard=blackboard,
        )

    @property
    def blackboard(self) -> EnhancedBlackboard:
        """Get game blackboard (must call initialize first)."""
        if self._blackboard is None:
            raise RuntimeError("GameProtocolCapability not initialized. Call initialize() first.")
        return self._blackboard

    @event_handler(pattern=GameStateProtocol.state_pattern(namespace="game")) # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
    async def handle_game_event(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Process game events and enrich planning context.
        
        This event handler:
        1. Parses blackboard events to game events
        2. Verifies the event is for this game
        3. Enriches context with game state (including version for OCC)
        4. Checks for terminal states
        5. Optionally returns rule-based immediate actions
        
        NOTE: This handler does NOT manage transactions or register data
        dependencies for OCC (using scope.set_shared). It provides context
        including `game_state_version` which `submit_move` can use for OCC
        validation. Transaction management is internal to `submit_move`.

        Subclasses can define more @event handlers to add game-specific
        context, or to provide immediate (rule-based) actions to skip LLM planning.
        
        Args:
            event: Blackboard event to process
            scope: Policy scope to enrich
        
        Returns:
            EventProcessingResult if processed, None if not relevant
        """
        # Parse event
        game_event = GameEvent.from_blackboard_event(event)
        if game_event is None or game_event.game_state is None:
            logger.warning(f"Received event with no game state: {event.key}")
            return None  # Not a game event

        game_state = game_event.game_state

        # Verify this event is for our game
        if self.game_id and game_state.game_id != self.game_id:
            return None  # Different game

        # Get or determine role
        role = self.role
        if role is None and game_state:
            role = game_state.get_role(self.agent.agent_id)

        # Check for terminal state
        if game_state.is_terminal():
            outcome = game_state.outcome
            logger.info(f"Game {game_state.game_id} completed: {outcome}")
            return EventProcessingResult(
                context_key="game_outcome",
                context=outcome, # Must be BaseModel serializable
                done=True,
            )

        # Build context for LLM planning (includes version for OCC in submit_move)
        context = {
            "game_id": game_state.game_id,
            "game_event_type": game_event.game_event_type,
            "game_state": game_state.model_dump(),
            "game_state_version": game_state.get_digest(),  # For submit_move OCC validation
            "game_phase": game_state.phase.value,
            "game_data": game_state.game_data,
            "my_role": role,
        }

        # Populate scope with common game data
        repl.set("latest_game_event", game_event)
        repl.set("latest_game_state", game_state)
        repl.set("game_phase", game_state.phase.value)
        repl.set("game_data", game_state.game_data)
        repl.set("my_role", role)

        ###########################################################
        # TODO: Subclasses can add more context by having more @event_handler methods
        #       because all handlers are called in sequence by
        #       EventDrivenActionPolicy.plan_step and their contexts merged.
        ###########################################################
        #### Add game-specific context (subclasses override)
        ### additional_context = await self._get_additional_context(game_event, scope)
        ### context.update(additional_context)

        ###########################################################
        # TODO: Subclasses can provide rule-based immediate actions
        #       by having more @event_handler methods to return an Action.
        #       If any handler returns an Action, LLM planning is skipped.
        ###########################################################
        ### # Check for rule-based immediate action (subclasses override)
        ### immediate_action = await self._get_rule_based_action(game_event, context, scope)

        return EventProcessingResult(
            context_key="game_context",
            context=context,
            immediate_action=None, # immediate_action,
        )
    
    async def _get_additional_context(
        self,
        game_event: GameEvent,
        repl: PolicyREPL,
    ) -> dict[str, Any]:
        """Override to add game-specific context for LLM planning.
        
        Args:
            game_event: The current game event
            scope: Policy scope (read-only)
        
        Returns:
            Additional context dict to merge
        """
        return {}

    @event_handler(pattern=GameStateProtocol.state_pattern(namespace="game")) # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
    async def _get_rule_based_action(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Override to provide rule-based immediate actions.
        
        Return an Action to skip LLM planning and execute immediately.
        Return None to allow LLM planning.

        This is where game-specific strategy logic goes. For example:
        - Negotiation: accept if utility > threshold
        - Hypothesis: challenge if claim unsupported

        Args:
            event: BlackboardEvent,
            repl: PolicyREPL,
        
        Returns:
            EventProcessingResult for immediate execution, or None for LLM planning
        """
        game_event = GameEvent.from_blackboard_event(event)
        if game_event is None or game_event.game_state is None:
            logger.warning(f"Received event with no game state: {event.key}")
            return None  # Not a game event

        if game_event and game_event.game_state:
            # Dispatch based on event type
            if game_event.game_event_type == GameEventType.GAME_STARTED.value:
                return await self._handle_game_started(game_event)
            elif game_event.game_event_type == GameEventType.GAME_MOVE.value:
                return await self._handle_game_move(game_event)
            else:
                logger.debug(f"Unknown game event type: {game_event.game_event_type}")
        return None

    # -------------------------------------------------------------------------
    # Role-Based Validation (uses role_permissions)
    # -------------------------------------------------------------------------

    def validate_move_permissions(
        self,
        agent_id: str,
        move: ACLMessage,
        state: GameState
    ) -> tuple[bool, str]:
        """Validate move against role permissions.

        Checks:
        1. Agent is a participant
        2. Agent's role is allowed to use move's performative in current phase

        This is called automatically by submit_move before validate_move.
        Subclasses can also call this in their validate_move for early checks.

        Args:
            agent_id: Agent making the move
            move: The move (ACL message)
            state: Current game state

        Returns:
            (is_valid, reason) tuple
        """
        # Check agent is participant
        if agent_id not in state.participants:
            return False, f"Agent {agent_id} is not a participant in this game"

        # Get agent's role
        role = state.get_role(agent_id)
        if role is None:
            return False, f"Agent {agent_id} has no role assigned"

        # Check if permissive mode
        if getattr(self.role_permissions, '_allow_all', False):
            return True, "Permissive mode - all moves allowed"

        # Check role permissions
        if not self.role_permissions.allows(role, state.phase, move.performative):
            allowed = self.role_permissions.allowed_performatives(role, state.phase)
            allowed_str = ", ".join(p.value for p in allowed) if allowed else "none"
            return False, (
                f"Role '{role}' cannot use '{move.performative.value}' in phase '{state.phase.value}'. "
                f"Allowed: [{allowed_str}]"
            )

        return True, "Role permissions OK"

    def can_start_game(self, agent_id: str, participants: dict[str, str]) -> tuple[bool, str]:
        """Check if agent can start this game.

        Override in subclasses to add custom start validation.
        Default: only "coordinator" role can start.

        Args:
            agent_id: Agent attempting to start
            participants: Proposed participants with roles

        Returns:
            (can_start, reason) tuple
        """
        role = participants.get(agent_id)
        if role is None:
            return False, f"Agent {agent_id} is not in participants"
        if role != "coordinator":
            return False, f"Only coordinator can start game. Agent {agent_id} has role '{role}'"
        return True, "OK"

    # -------------------------------------------------------------------------
    # Pure Game Rules (no I/O) - Override in Subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    async def validate_move(
        self,
        agent_id: str,
        move: ACLMessage,
        state: GameState
    ) -> tuple[bool, str]:
        """Validate if move is legal in current state (game-specific rules).

        This is called AFTER validate_move_permissions, so role permissions
        are already checked. Implement game-specific validation here.

        This is a pure function - no I/O, no side effects.

        Args:
            agent_id: Agent making the move
            move: Move to validate
            state: Current game state

        Returns:
            (is_valid, reason) tuple
        """
        ...

    @abstractmethod
    async def apply_move(
        self,
        state: GameState,
        move: ACLMessage
    ) -> GameState:
        """Apply move (transition game state based on message) and return new state.

        This is a pure function - no I/O, no side effects.

        Args:
            state: Current game state
            move: Move to apply

        Returns:
            New game state
        """
        ...

    async def is_terminal(self, state: GameState) -> bool:
        """Check if game has reached terminal state.

        Default implementation checks phase. Override for custom logic.

        Args:
            state: Current state

        Returns:
            True if terminal
        """
        return state.phase == GamePhase.TERMINAL or state.outcome is not None

    @abstractmethod
    async def compute_outcome(self, state: GameState) -> GameOutcome:
        """Compute game outcome from terminal state.

        This is a pure function.

        Args:
            state: Terminal game state

        Returns:
            Game outcome
        """
        ...

    # -------------------------------------------------------------------------
    # Action Executors (Blackboard Interaction)
    # -------------------------------------------------------------------------

    @action_executor(exclude_from_planning=True)
    async def start_game(
        self,
        participants: dict[str, str],  # agent_id -> role
        initial_data: dict[str, Any],
        game_id: str | None = None,
        config: dict[str, Any] | None = None
    ) -> GameState:
        """Start a new game instance.

        Args:
            participants: Agent ID -> role mapping
            initial_data: Initial game data (game-specific)
            game_id: Optional game ID (generated if None)
            config: Optional game configuration

        Returns:
            Initial game state
        """
        # Check if agent can start game
        can_start, reason = self.can_start_game(self.agent.agent_id, participants)
        if not can_start:
            raise PermissionError(reason)

        # Create initial state
        game_id = game_id or f"game_{uuid.uuid4().hex}"
        state = GameState(
            game_id=game_id,
            game_type=self.game_type,
            conversation_id=initial_data.get("conversation_id", game_id),
            participants=list(participants.keys()),
            roles=participants,
            phase=self._get_initial_phase(),
            game_data=initial_data,
            config=config or {}
        )

        # Save initial state
        await self.save_game_state(state, GameEventType.GAME_STARTED.value, move=None)

        return state

    @hookable
    @action_executor(exclude_from_planning=True)
    async def submit_move(
        self,
        game_id: str,
        move: ACLMessage,
        expected_version: str | None = None,
    ) -> tuple[bool, str, GameState | None]:
        """Submit a move - validates, applies, saves, and emits event.

        This method manages its own transaction internally for atomic read-modify-write.
        If `expected_version` is provided (from event context), validates that the game
        state hasn't changed since the event was received.

        This method is @hookable so memory capabilities can observe completed games.
        The returned GameState can be captured by episodic memory hooks.

        Note: This action is excluded from LLM planning. It is invoked
        programmatically by the action policy in response to game events.

        Args:
            game_id: Game instance ID
            move: Move to submit
            expected_version: Optional[str] version from event context for OCC validation.
                If provided and state version differs, returns conflict error.

        Returns:
            (success, reason, new_state) tuple
        """
        # Validate move sender early (before transaction)
        assert move.sender == self.agent.agent_id, "Move sender must be this agent"

        # Add distributed locking because multiple agents may submit moves concurrently
        # Use internal transaction for atomic read-modify-write
        async with self.blackboard.transaction(): # Creates an ambient transaction context
            # Load current state within transaction
            state = await self.load_game_state(game_id)
            if not state:
                return False, f"No game state found for {game_id}", None

            # OCC validation: check version if expected_version provided
            if expected_version is not None:
                current_version = state.get_digest()
                if current_version is not None and current_version != expected_version:
                    return False, (
                        f"Game state version conflict: expected {expected_version}, "
                        f"got {current_version}. Replan with updated state."
                    ), state

            # Validate role permissions first (common to all games)
            valid, reason = self.validate_move_permissions(self.agent.agent_id, move, state)
            if not valid:
                return False, reason, state

            # Validate game-specific rules (implemented by subclass)
            valid, reason = await self.validate_move(self.agent.agent_id, move, state)
            if not valid:
                return False, reason, state

            # Apply move - Transition state
            new_state = await self.apply_move(state, move)

            # Add message to history
            new_state.add_message(move)

            # Check terminal
            if await self.is_terminal(new_state):
                new_state.phase = GamePhase.TERMINAL
                new_state.outcome = self.compute_outcome(new_state)
                new_state.ended_at = time.time()

            # Save updated state within transaction
            # Prefer the centralized helper so metadata/tags stay consistent
            # across all game state writes (and so event emission remains
            # coupled to transaction commit).
            await self.save_game_state(
                new_state,
                GameEventType.GAME_MOVE.value,
                move=move,
            )
            # Transaction commits on exit

        return True, "Move applied", new_state

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @action_executor(exclude_from_planning=True)
    async def load_game_state(self, game_id: str) -> GameState | None:
        """Read current game state from blackboard.

        Args:
            game_id: Game ID

        Returns:
            Current game state or None if not found
        """
        key = GameStateProtocol.state_key(game_id, namespace="game") # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
        # Prefer explicit transactional read when an ambient transaction is active.
        # Fallback to non-transactional read if not in a transaction.
        try:
            data = await self.blackboard.read_tx(key, agent_id=self.agent.agent_id)
        except RuntimeError:
            data = await self.blackboard.read(key, agent_id=self.agent.agent_id)
        return GameState.from_blackboard_entry(data) if data else None

    def _get_initial_phase(self) -> GamePhase:
        """Get initial phase for this game type. Override in subclasses."""
        return GamePhase.PROPOSE  # Default, override in subclasses

    def _get_blackboard_event_type(self, game_event: str) -> str:
        """Get blackboard event type for game event."""
        return f"game_{self.game_type}_{game_event}"

    # TODO: Add a method here to allow agents to subscribe to game state changes.
    # TODO: Emit semantic game event in addition to the canonical "write" event.

    @action_executor(exclude_from_planning=True)
    async def save_game_state(
        self,
        game_state: GameState,
        game_event_type: str,
        move: ACLMessage | None = None
    ) -> None:
        """Save game state to blackboard.

        Note: This is an internal action excluded from LLM planning.
        This also emits an event as part of the blackboard write operation when
        the current transaction (if any) is committed successfully.

        NOTE: Emitting an event using `blackboard.emit_event` here would be incorrect
        because it would be outside any transaction context and takes effect immediately
        unlike the transactional state write, which might fail.
        We need to emit the event only AFTER the write is COMMITTED successfully. This is
        the case already in the transaction commit logic.

        Args:
            `game_state`: Game state to save
            `game_event_type`: Type of game event for blackboard event emission. It is stored in the
                `metadata["game_event_type"]` field of the blackboard entry.
            `move`: Optional move that triggered this state change
        """

        game_event = GameEvent.create(
            agent_id=self.agent.agent_id,
            game_event_type=game_event_type,
            game_state=game_state,
            move=move
        )

        # Prefer explicit transactional write when an ambient transaction is active.
        # Fallback to non-transactional write if not in a transaction.
        try:
            await self.blackboard.write_tx(
                key=GameStateProtocol.state_key(game_state.game_id, namespace="game"), # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
                value=game_event.game_state.to_blackboard_entry(),
                created_by=game_event.agent_id,
                tags=game_event.tags,
                metadata=game_event.metadata,
            )
        except RuntimeError:
            await self.blackboard.write(
                key=GameStateProtocol.state_key(game_state.game_id, namespace="game"), # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
                value=game_event.game_state.to_blackboard_entry(),
                tags=game_event.tags,
                created_by=game_event.agent_id,
                metadata=game_event.metadata,
            )

    async def get_active_games(self, game_type: str | None = None) -> list[GameState]:
        """Get all active games.

        Args:
            blackboard: Blackboard instance
            game_type: Optional filter by game type

        Returns:
            List of active games
        """
        entries = await self.blackboard.query(
            namespace=GameStateProtocol.state_pattern(namespace="game"), # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
            tags={"game", "active", game_type} if game_type else {"game", "active"}
        )
        return [ GameState(**entry.value) for entry in entries ]

    async def get_game_history(self, game_id: str) -> list[ACLMessage]:
        """Get message history for a game.

        Args:
            blackboard: Blackboard instance
            game_id: Game ID

        Returns:
            List of messages in chronological order
        """
        # Load game state
        state = await self.load_game_state(game_id)

        if state is None:
            return []

        return state.history

    @asynccontextmanager
    async def game_state_transaction(self, game_id: str):
        """Context manager for game state transaction.

        Allows:
            async with obj.game_state_transaction(game_id) as state:
                ... # mutate state

        On exit, writes state back to blackboard.
        """
        # With ambient transactions, we can use blackboard.read_tx/write_tx inside the block
        # while still allowing advanced callers to use the yielded txn if needed.
        async with self.blackboard.transaction() as txn:
            key = GameStateProtocol.state_key(game_id, namespace="game") # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
            state_data = await self.blackboard.read_tx(key, agent_id=self.agent.agent_id)

            if not state_data:
                yield None
                return

            state = GameState.from_blackboard_entry(state_data)
            try:
                yield state
            finally:
                await self.blackboard.write_tx(
                    key=key,
                    value=state.to_blackboard_entry(),
                    created_by=self.agent.agent_id,
                    tags={"game", state.game_type, state.phase.value, "terminal" if state.is_terminal() else "active"},
                    metadata={"last_move_by": self.agent.agent_id},
                )

    # =========================================================================
    # Event Handlers - Subclasses should override these
    # =========================================================================

    async def _handle_game_started(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game started event.

        Override this method to implement role-specific initialization logic.
        For example:
        - Coordinator: Verify game started correctly
        - Participant: Prepare initial move if appropriate
        - Observer: Begin monitoring

        Args:
            game_event: The GAME_STARTED event

        Returns:
            EventProcessingResult containing Action to execute, or None
        """
        logger.info(f"Game {self.game_id} started. Role: {self.role.value}")
        return None

    async def _handle_game_move(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game move event.

        Override this method to implement phase-specific handling logic.
        Typically dispatches to phase handlers based on current game phase.

        Example:
            ```python
            async def _handle_game_move(self, state, game_event):
                game_state = game_event.game_state
                phase = game_state.phase

                if game_state.is_terminal():
                    return await self._handle_game_complete(state, game_event)

                game_data = self.game_data_cls(**game_state.game_data)

                if phase == GamePhase.OFFER:
                    return await self._handle_offer_phase(state, game_event, game_data)
                elif phase == GamePhase.COUNTER_OFFER:
                    return await self._handle_counter_phase(state, game_event, game_data)
                ...
            ```

        Args:
            game_event: The GAME_MOVE event

        Returns:
            Action to execute, or None
        """
        game_state = game_event.game_state

        logger.debug(
            f"Game {self.game_id} move. Phase: {game_state.phase.value}, "
            f"Agent: {game_event.agent_id}, Role: {self.role.value}"
        )

        if game_state.is_terminal():
            return await self._handle_game_complete(game_event)

        # Subclasses should override this to handle specific phases
        return None

    async def _handle_game_complete(
        self,
        game_event: GameEvent
    ) -> EventProcessingResult | None:
        """Handle game completion.

        Called when game reaches terminal state. Default implementation
        logs the outcome and marks policy as complete.

        Args:
            game_event: The terminal game event

        Returns:
            None (no action needed for completion)
        """
        game_state = game_event.game_state
        outcome = game_state.outcome

        if outcome and outcome.success:
            logger.info(f"Game {self.game_id} completed successfully")
        else:
            logger.info(f"Game {self.game_id} completed: {outcome.outcome_type if outcome else 'unknown'}")

        return None

    # =========================================================================
    # LLM Helpers
    # =========================================================================

    async def _llm_decide(
        self,
        decision_type: str,
        context: dict[str, Any],
        options: list[str],
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Use LLM to make a decision.

        Convenience wrapper around `self.llm_config.generate_decision()`.

        Args:
            decision_type: Type of decision (e.g., "counter_offer", "accept_reject")
            context: Context dict for decision making
            options: Available action options
            json_schema: Optional JSON schema for structured output

        Returns:
            Dict with decision: {"action": "...", "reasoning": "..."} or None
        """
        if not self.llm_config.enabled:
            return None

        game_type = getattr(self.game_protocol_cls, 'game_type', self.__class__.__name__)

        try:
            return await self.llm_config.generate_decision(
                self.agent,
                game_type=game_type,
                decision_type=decision_type,
                context=context,
                options=options,
                json_schema=json_schema,
            )
        except Exception as e:
            logger.warning(f"LLM decision failed: {e}")
            return None

    # =========================================================================
    # Action Creation Helpers
    # =========================================================================

    def _create_submit_move_action(
        self,
        message: ACLMessage,
        reasoning: str,
    ) -> Action:
        """Create an action to submit a game move.

        Helper method for creating submit_move actions with consistent format.

        Args:
            message: ACL message to submit
            reasoning: Reasoning for the action

        Returns:
            Action object
        """
        import uuid as _uuid
        return Action(
            action_id=f"move_{_uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type="submit_move",
            parameters={
                "game_id": self.game_id,
                "move": message.model_dump(),
            },
            reasoning=reasoning,
        )
# MultiAgentGameActionPolicy has been removed.
# Game logic is now encapsulated in GameProtocolCapability via @event_handler.
# Use CacheAwareActionPolicy directly with game capabilities as action_providers.

