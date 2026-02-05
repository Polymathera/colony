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

import uuid
from pydantic import BaseModel, Field
from logging import getLogger

from ....base import Agent, CapabilityResultFuture
from ..state import GameOutcome
from ...models import Hypothesis
from ....models import AgentSpawnSpec, AgentMetadata
from ...actions.policies import action_executor
from .capabilities import (
    HypothesisGameProtocol,
    HypothesisRole
)

logger = getLogger(__name__)


# ============================================================================
# Game Configuration
# ============================================================================


class HypothesisGameConfig(BaseModel):
    """Configuration for a hypothesis game instance.

    Used to configure agents participating in a hypothesis game.
    Serializable to pass between spawned agents.
    """

    game_id: str = Field(description="Unique game identifier")
    role: str = Field(description="Role in the game: proposer, skeptic, grounder, arbiter")
    hypothesis: Hypothesis = Field(description="The hypothesis being validated")
    use_llm_reasoning: bool = Field(
        default=False,
        description="Whether to use LLM for decision making"
    )
    participants: dict[str, str] = Field(
        default_factory=dict,
        description="Agent ID -> role mapping for all participants"
    )


# ============================================================================
# Role-Specific Agent Classes
# ============================================================================


class HypothesisGameAgent(Agent):
    """Agent configured for hypothesis game role.

    Automatically sets up HypothesisGameProtocol capability with
    CacheAwareActionPolicy. The protocol handles all game logic
    via @event_handler methods.
    """

    def __init__(self, agent_id: str, role: HypothesisRole, **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self.role = role
        self._game_config = HypothesisGameConfig(
            **self.metadata.get("game_config", {})
        )

    async def initialize(self) -> None:
        """Initialize hypothesis game agent."""
        await super().initialize()

        # Set up game protocol with role and LLM config
        capability = HypothesisGameProtocol(
            self,
            game_id=self._game_config.game_id,
            role=self.role.value,
            use_llm_reasoning=self._game_config.use_llm_reasoning,
        )
        await capability.initialize()
        self.add_capability(capability)

        # Use CacheAwareActionPolicy - the protocol's @event_handler
        # methods will handle all game logic
        from ...actions.policies import create_default_action_policy

        self.action_policy = await create_default_action_policy(agent=self)

        logger.info(
            f"HypothesisGameAgent {self.agent_id} initialized as {self.role.value} "
            f"for game {self._game_config.game_id}"
        )


class HypothesisProposerAgent(HypothesisGameAgent):
    """Agent configured for hypothesis proposer role.

    Automatically sets up HypothesisGameProtocol capability with proposer role.
    """

    def __init__(self, agent_id: str, role: HypothesisRole, **kwargs):
        super().__init__(agent_id=agent_id, role=role, **kwargs)
        self._game_config = HypothesisGameConfig(
            **self.metadata.get("game_config", {})
        )

    async def initialize(self) -> None:
        """Initialize proposer agent."""
        await super().initialize()


class HypothesisSkepticAgent(HypothesisGameAgent):
    """Agent configured for hypothesis skeptic role.

    Challenges unsupported claims in the hypothesis.
    """

    def __init__(self, agent_id: str, role: HypothesisRole, **kwargs):
        super().__init__(agent_id=agent_id, role=role, **kwargs)
        self._game_config = HypothesisGameConfig(
            **self.metadata.get("game_config", {})
        )

    async def initialize(self) -> None:
        """Initialize skeptic agent."""
        await super().initialize()


class HypothesisGrounderAgent(HypothesisGameAgent):
    """Agent configured for hypothesis grounder role.

    Provides evidence to support or refute claims.
    """

    def __init__(self, agent_id: str, role: HypothesisRole, **kwargs):
        super().__init__(agent_id=agent_id, role=role, **kwargs)
        self._game_config = HypothesisGameConfig(
            **self.metadata.get("game_config", {})
        )

    async def initialize(self) -> None:
        """Initialize grounder agent."""
        await super().initialize()


class HypothesisArbiterAgent(HypothesisGameAgent):
    """Agent configured for hypothesis arbiter role.

    Makes final judgment on hypothesis validity.
    """

    def __init__(self, agent_id: str, role: HypothesisRole, **kwargs):
        super().__init__(agent_id=agent_id, role=role, **kwargs)
        self._game_config = HypothesisGameConfig(
            **self.metadata.get("game_config", {})
        )

    async def initialize(self) -> None:
        """Initialize arbiter agent."""
        await super().initialize()


class HypothesisCoordinatorAgent(Agent):
    """Agent that coordinates hypothesis validation game.

    The coordinator:
    1. Spawns participant agents (proposer, skeptics, grounders, arbiter)
    2. Starts the game via HypothesisGameProtocol.start_game()
    3. Monitors game events
    """

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self._game_config = HypothesisGameConfig(
            **self.metadata.get("game_config", {})
        )
        self._child_agent_ids: list[str] = []

    async def initialize(self) -> None:
        """Initialize coordinator agent."""
        await super().initialize()

        capability = HypothesisGameProtocol(self, game_id=self._game_config.game_id)
        await capability.initialize()
        self.add_capability(capability)

        # Spawn other game agents
        await self.spawn_game_agents()

        # Start the game
        await capability.start_game(
            participants=self._game_config.participants,
            initial_data={"hypothesis": self._game_config.hypothesis.model_dump()},
            game_id=self._game_config.game_id,
        )

        logger.info(
            f"HypothesisCoordinatorAgent {self.agent_id} initialized "
            f"and started game {self._game_config.game_id}"
        )

    async def spawn_game_agents(self) -> list[str]:
        """Spawn participant agents for this game."""
        agent_ids = []
        participants = self._game_config.participants

        # Get role -> capabilities mapping from metadata
        role_capabilities: dict[str, list[str]] = self.metadata.get("role_capabilities", {})

        role_to_agent_type = {
            "proposer": "polymathera.colony.agents.games.hypothesis_game.HypothesisProposerAgent",
            "skeptic": "polymathera.colony.agents.games.hypothesis_game.HypothesisSkepticAgent",
            "grounder": "polymathera.colony.agents.games.hypothesis_game.HypothesisGrounderAgent",
            "arbiter": "polymathera.colony.agents.games.hypothesis_game.HypothesisArbiterAgent",
        }

        for agent_id, role in participants.items():
            if agent_id == self.agent_id:
                continue  # Don't spawn self

            agent_type = role_to_agent_type.get(role)
            if not agent_type:
                continue

            child_config = HypothesisGameConfig(
                game_id=self._game_config.game_id,
                role=role,
                hypothesis=self._game_config.hypothesis,
                use_llm_reasoning=self._game_config.use_llm_reasoning,
                participants=participants,
            )

            agent_caps = role_capabilities.get(role, [])

            spawned_id = await self.spawn_child_agents(
                agent_specs=[AgentSpawnSpec(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    capabilities=agent_caps,
                    metadata=AgentMetadata(parameters={"game_config": child_config.model_dump()}),
                )],
                return_handles=False,
            )[0]
            agent_ids.append(spawned_id)

        self._child_agent_ids = agent_ids
        logger.info(f"Spawned {len(agent_ids)} hypothesis game agents: {agent_ids}")
        return agent_ids


# ============================================================================
# Public API Functions
# ============================================================================


@action_executor(writes=["game_future"])
async def run_hypothesis_game(
    *,
    owner: Agent,
    hypothesis: Hypothesis,
    capabilities: dict[str, list[str]] | None = None,
    proposer_id: str | None = None,
    skeptic_ids: list[str] | None = None,
    grounder_ids: list[str] | None = None,
    arbiter_id: str | None = None,
    use_llm_reasoning: bool = False,
    game_id: str | None = None,
    num_skeptics: int = 2,
    num_grounders: int = 1,
) -> CapabilityResultFuture:
    """Run a complete hypothesis validation game.

    This is the primary entry point for launching hypothesis games.
    It spawns coordinator and all participant agents, starts the game,
    and returns a result handle for monitoring.

    Can be used as a standalone function or as an `action_executor` in any
    action policy, enabling composition of multi-agent patterns.

    Args:
        owner: Parent agent spawning the game
        hypothesis: Hypothesis to validate
        capabilities: Optional dict mapping role -> list of capability class paths
        proposer_id: Optional proposer agent ID
        skeptic_ids: Optional list of skeptic agent IDs
        grounder_ids: Optional list of grounder agent IDs
        arbiter_id: Optional arbiter agent ID
        use_llm_reasoning: Whether to use LLM for decision making
        game_id: Optional game ID (generated if not provided)
        num_skeptics: Number of skeptics if skeptic_ids not provided
        num_grounders: Number of grounders if grounder_ids not provided

    Returns:
        CapabilityResultFuture handle for monitoring/awaiting completion

    Example:
        ```python
        from polymathera.colony.agents.games.hypothesis_game import (
            run_hypothesis_game,
        )
        from polymathera.colony.agents.patterns import Hypothesis

        hypothesis = Hypothesis(
            hypothesis_id="hyp_001",
            statement="The function handles all edge cases correctly",
            supporting_evidence=["test_case_1", "test_case_2"],
        )

        result = await run_hypothesis_game(
            owner=my_agent,
            hypothesis=hypothesis,
            use_llm_reasoning=True,
        )

        outcome = await result.wait(timeout=60.0)
        if outcome and outcome.success:
            print("Hypothesis validated!")
        else:
            print("Hypothesis rejected")
        ```
    """
    game_id = game_id or f"hypothesis_game_{uuid.uuid4().hex[:8]}"
    capabilities = capabilities or {}

    # Build participants mapping
    participants: dict[str, str] = {}
    coordinator_id = f"{game_id}_coordinator"

    proposer_id = proposer_id or f"{game_id}_proposer"
    participants[proposer_id] = "proposer"

    if skeptic_ids is None:
        skeptic_ids = [f"{game_id}_skeptic_{i}" for i in range(num_skeptics)]
    for sid in skeptic_ids:
        participants[sid] = "skeptic"

    if grounder_ids is None:
        grounder_ids = [f"{game_id}_grounder_{i}" for i in range(num_grounders)]
    for gid in grounder_ids:
        participants[gid] = "grounder"

    arbiter_id = arbiter_id or f"{game_id}_arbiter"
    participants[arbiter_id] = "arbiter"

    # Create coordinator config
    config = HypothesisGameConfig(
        game_id=game_id,
        role="coordinator",
        hypothesis=hypothesis,
        use_llm_reasoning=use_llm_reasoning,
        participants=participants,
    )

    coordinator_caps = capabilities.get("coordinator", [])

    # Spawn coordinator agent
    await owner.spawn_child_agents(
        agent_specs=[AgentSpawnSpec(
            agent_id=coordinator_id,
            agent_type="polymathera.colony.agents.games.hypothesis_game.HypothesisCoordinatorAgent",
            capabilities=coordinator_caps,
            metadata=AgentMetadata(parameters={
                "game_config": config.model_dump(),
                "role_capabilities": capabilities,
            }),
        )],
        return_handles=False,
    )

    # Create result handle using the same game_id scope
    game_protocol = HypothesisGameProtocol(agent=owner, game_id=game_id)
    await game_protocol.initialize()
    result = await game_protocol.get_result_future()

    logger.info(
        f"Spawned hypothesis game {game_id} with {len(skeptic_ids)} skeptics, "
        f"{len(grounder_ids)} grounders"
    )

    return result


# Legacy function for backward compatibility
async def ground_hypothesis(
    hypothesis: Hypothesis,
    agent: Agent,
    use_llm: bool = False,
) -> tuple[bool, Hypothesis, GameOutcome]:
    """Validate hypothesis using hypothesis game (legacy API).

    For new code, use `run_hypothesis_game` instead.

    Args:
        hypothesis: Hypothesis to validate
        agent: Agent instance
        use_llm: Whether to use LLM reasoning

    Returns:
        (accepted, final_hypothesis, outcome) tuple
    """
    result = await run_hypothesis_game(
        owner=agent,
        hypothesis=hypothesis,
        use_llm_reasoning=use_llm,
    )

    outcome = await result.wait(timeout=120.0)

    if outcome is None:
        return (False, hypothesis, GameOutcome(
            outcome_type="timeout",
            success=False,
            participants=[],
            rounds_played=0,
            messages_exchanged=0,
        ))

    final_hypothesis = hypothesis
    if outcome.result and isinstance(outcome.result, dict):
        try:
            final_hypothesis = Hypothesis(**outcome.result)
        except Exception:
            pass

    return (outcome.success, final_hypothesis, outcome)

