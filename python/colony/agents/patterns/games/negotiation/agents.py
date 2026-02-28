
"""
=============================================================================
Role-Specific Agent Classes
=============================================================================
Each role gets its own Agent subclass with pre-configured capability and policy.
This simplifies spawning and ensures correct setup for each role.
Usage Pattern:
    result = await run_negotiation_game(
        owner=parent_agent,
        issue=NegotiationIssue(...),
        coordinator_id="coordinator1",
        participant_ids=["participant1", "participant2"],
        mediator_id="mediator1",
        strategies={"participant1": NegotiationStrategy.HARDBALL},
        use_llm_reasoning=True,
    )
    final_agreement = await result.wait()
    # Or with callback:
    result.on_complete(lambda outcome: print(f"Game finished: {outcome}"))
=============================================================================

Quick Start:
    ```python
    from polymathera.colony.agents.games.negotiation_game import (
        run_negotiation_game,
        NegotiationIssue,
        NegotiationStrategy,
    )

    # Define the issue to negotiate
    issue = NegotiationIssue(
        issue_id="resource_001",
        description="Allocate 100 compute units between teams",
        parties=["team_a", "team_b"],
        preferences={
            "team_a": {"units": 1.0, "priority": 0.5},
            "team_b": {"units": 0.8, "cost_efficiency": 0.6},
        }
    )

    # Spawn game and wait for result
    result = await run_negotiation_game(
        owner=my_agent,
        issue=issue,
        coordinator_id="coordinator1",
        participant_ids=["participant1", "participant2"],
        mediator_id="arbiter",  # Optional
    )

    outcome = await result.wait(timeout=60.0)
    if outcome and outcome.success:
        print(f"Agreement: {outcome.result['final_agreement']}")
    ```

Public API:
- `run_negotiation_game()` - High-level function to spawn a complete game
- `CapabilityResultFuture` - Handle for monitoring/awaiting game completion
- `NegotiationParticipantAgent` - Pre-configured participant agent
- `NegotiationMediatorAgent` - Pre-configured mediator agent
- `NegotiationCoordinatorAgent` - Pre-configured coordinator agent
- `NegotiationGameProtocol` - Game rules capability (for custom agents)
- `NegotiationGameActionPolicy` - Decision policy (for custom agents)

"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pydantic import BaseModel, Field, PrivateAttr

from ..state import GameOutcome
from ....base import CapabilityResultFuture
from .capabilities import (
    NegotiationIssue,
    NegotiationStrategy,
    NegotiationGameProtocol,
)
from ....base import Agent
from ....models import AgentMetadata
from ...actions.policies import action_executor, create_default_action_policy


logger = logging.getLogger(__name__)


class NegotiationGameConfig(BaseModel):
    """Configuration for a negotiation game instance.

    Used to configure agents participating in a negotiation game.
    Serializable to pass between spawned agents.
    """

    game_id: str = Field(description="Unique game identifier")
    role: str = Field(description="Role in the game: coordinator, participant, mediator")
    issue: NegotiationIssue = Field(description="The negotiation issue")
    strategy: NegotiationStrategy = Field(
        default=NegotiationStrategy.COMPROMISING,
        description="Negotiation strategy to use"
    )
    min_acceptable_utility: float = Field(
        default=0.3,
        description="Minimum utility threshold (BATNA)"
    )
    use_llm_reasoning: bool = Field(
        default=False,
        description="Whether to use LLM for decision making"
    )
    llm_temperature: float = Field(
        default=0.7,
        description="Temperature setting for LLM responses"
    )
    llm_max_tokens: int = Field(
        default=1000,
        description="Max tokens for LLM responses"
    )
    participants: dict[str, str] = Field(
        default_factory=dict,
        description="Agent ID -> role mapping for all participants"
    )


class NegotiationParticipantAgent(Agent):
    """Agent configured for negotiation participant role.

    Automatically sets up NegotiationGameProtocol capability and
    NegotiationGameActionPolicy with participant role.

    Example:
        ```python
        agent = NegotiationParticipantAgent(
            agent_id="participant_1",
            game_config=NegotiationGameConfig(
                game_id="game_001",
                role="participant",
                issue=issue,
                strategy=NegotiationStrategy.HARDBALL,
            )
        )
        await agent.initialize()
        await agent.run()  # Start processing events
        ```
    """

    _game_config: NegotiationGameConfig | None = PrivateAttr(default=None)

    async def initialize(self) -> None:
        """Initialize participant agent with negotiation capability and policy."""
        await super().initialize()

        self._game_config = NegotiationGameConfig(
            **self.metadata.parameters.get("game_config", {})  # type: ignore
        )
        # Create and register capability with game_id for shared scope
        capability = NegotiationGameProtocol(
            agent=self,
            game_id=self._game_config.game_id,
            role="participant",
            strategy=self._game_config.strategy,
            min_acceptable_utility=self._game_config.min_acceptable_utility,
            use_llm_reasoning=self._game_config.use_llm_reasoning,
            llm_temperature=self._game_config.llm_temperature,
            llm_max_tokens=self._game_config.llm_max_tokens
        )
        await capability.initialize()
        self.add_capability(capability)

        # Create action policy
        self.action_policy = await create_default_action_policy(agent=self)

        logger.info(
            f"NegotiationParticipantAgent {self.agent_id} initialized "
            f"for game {self._game_config.game_id}"
        )


class NegotiationMediatorAgent(Agent):
    """Agent configured for negotiation mediator role.

    The mediator helps resolve deadlocks by proposing compromise solutions.

    Example:
        ```python
        agent = NegotiationMediatorAgent(
            agent_id="mediator_1",
            game_config=NegotiationGameConfig(
                game_id="game_001",
                role="mediator",
                issue=issue,
            )
        )
        await agent.initialize()
        while agent.state == AgentState.RUNNING:
            await agent.run_step()
        ```
    """

    _game_config: NegotiationGameConfig | None = PrivateAttr(default=None)

    async def initialize(self) -> None:
        """Initialize mediator agent."""
        await super().initialize()

        self._game_config = NegotiationGameConfig(
            **self.metadata.parameters.get("game_config", {})  # type: ignore
        )
        capability = NegotiationGameProtocol(self, game_id=self._game_config.game_id)
        await capability.initialize()
        self.add_capability(capability)

        self.action_policy = await create_default_action_policy(agent=self)

        logger.info(
            f"NegotiationMediatorAgent {self.agent_id} initialized "
            f"for game {self._game_config.game_id}"
        )


class NegotiationCoordinatorAgent(Agent):
    """Agent configured for negotiation coordinator role.

    The coordinator agent, during initialization:
    1. Spawns participant and mediator agents
    2. Starts the game

    After initialization, the agent manager runs the coordinator's action policy
    which handles game events and coordination.

    Example:
        ```python
        # Use run_negotiation_game (recommended)
        result = await run_negotiation_game(
            owner=parent_agent,
            issue=issue,
            participant_ids=["p1", "p2"],
            mediator_id="m1",
        )
        outcome = await result.wait()
        ```
    """

    _game_config: NegotiationGameConfig | None = PrivateAttr(default=None)
    _child_agent_ids: list[str] = PrivateAttr(default_factory=list)
    _result_future: asyncio.Future[GameOutcome | None] | None = PrivateAttr(default=None)

    async def initialize(self) -> None:
        """Initialize coordinator agent.

        The coordinator:
        1. Spawns participant/mediator agents
        2. Starts the game via NegotiationGameProtocol.start_game()
        3. Then runs its action policy to handle game events
        """
        await super().initialize()

        self._game_config = NegotiationGameConfig(
            **self.metadata.parameters.get("game_config", {})  # type: ignore
        )
        capability = NegotiationGameProtocol(self, game_id=self._game_config.game_id)
        await capability.initialize()
        self.add_capability(capability)

        self.action_policy = await create_default_action_policy(agent=self)

        # Spawn other game agents
        await self.spawn_game_agents()

        # Start the game
        await capability.start_game(
            participants=self._game_config.participants,
            initial_data={"issue": self._game_config.issue.model_dump()},
            game_id=self._game_config.game_id,
        )

        logger.info(
            f"NegotiationCoordinatorAgent {self.agent_id} initialized "
            f"and started game {self._game_config.game_id}"
        )

    async def spawn_game_agents(self) -> list[str]:
        """Spawn participant and mediator agents for this game.

        Uses role_capabilities from metadata to attach additional capabilities
        to each agent beyond the built-in NegotiationGameProtocol.

        Returns:
            List of spawned agent IDs
        """
        agent_ids = []
        participants = self._game_config.participants

        # Get role -> capabilities mapping from metadata
        role_capabilities: dict[str, list[str]] = self.metadata.get("role_capabilities", {})

        role_to_agent_class = {
            "participant": NegotiationParticipantAgent,
            "mediator": NegotiationMediatorAgent,
        }

        for agent_id, role in participants.items():
            if agent_id == self.agent_id:
                assert role == "coordinator", "Only coordinator can have same ID as coordinator agent"
                continue  # Don't spawn self

            agent_cls = role_to_agent_class.get(role)
            if not agent_cls:
                continue

            # Create config for child agent
            child_config = NegotiationGameConfig(
                game_id=self._game_config.game_id,
                role=role,
                issue=self._game_config.issue,
                strategy=self._game_config.strategy,
                use_llm_reasoning=self._game_config.use_llm_reasoning,
                participants=participants,
            )

            # Get capabilities for this role
            agent_caps = role_capabilities.get(role, [])

            spawned_id = await self.spawn_child_agents(
                blueprints=[agent_cls.bind(
                    agent_id=agent_id,
                    capabilities=agent_caps,
                    metadata=AgentMetadata(parameters={"game_config": child_config.model_dump()}),
                )],
                return_handles=False,
            )[0]
            agent_ids.append(spawned_id)

        self._child_agent_ids = agent_ids
        logger.info(f"Spawned {len(agent_ids)} game agents: {agent_ids}")
        return agent_ids



# =============================================================================
# Public Utility Functions
# =============================================================================


@action_executor(writes=["game_future"])
async def run_negotiation_game(
    *,
    owner: Agent,
    issue: NegotiationIssue,
    capabilities: dict[str, list[str]] | None = None,
    coordinator_id: str | None = None,
    participant_ids: list[str] | None = None,
    mediator_id: str | None = None,
    strategies: dict[str, NegotiationStrategy] | None = None,
    use_llm_reasoning: bool = False,
    game_id: str | None = None,
    num_participants: int = 2,
) -> CapabilityResultFuture:
    """Spawn a complete negotiation game with all participants.

    This is the primary entry point for launching negotiation games.
    It spawns coordinator, participant, and optional mediator agents,
    starts the game, and returns a result handle for monitoring.

    Can be used as a standalone function or as an `action_executor` in any
    action policy, enabling composition of multi-agent patterns. For example,
    an agent can spawn a negotiation as part of a larger task (resource
    allocation, project management, etc.).

    Args:
        owner: Parent agent spawning the game
        issue: The negotiation issue to resolve
        capabilities: Optional dict mapping role -> list of capability class paths.
            These capabilities are added to agents of each role beyond the
            built-in NegotiationGameProtocol. Useful for giving negotiating
            agents access to information gathering, code analysis, etc.
            Example: {"participant": ["polymathera...CodeAnalysisCapability"]}
        coordinator_id: Optional coordinator agent ID (enables coordination)
        participant_ids: Optional list of participant agent IDs
        mediator_id: Optional mediator agent ID (enables mediation)
        strategies: Optional per-agent strategy mapping
        use_llm_reasoning: Whether to use LLM for decision making
        game_id: Optional game ID (generated if not provided)
        num_participants: Number of participants if not provided in participant_ids (default 2)

    Returns:
        CapabilityResultFuture handle for monitoring/awaiting completion

    Example - Direct call:
        ```python
        issue = NegotiationIssue(
            issue_id="resource_allocation",
            description="Divide 100 compute units",
            parties=["agent1", "agent2"],
            preferences={
                "agent1": {"units": 1.0},
                "agent2": {"units": 0.8, "priority": 0.2},
            }
        )

        result = await run_negotiation_game(
            owner=my_agent,
            coordinator_id="coordinator1",
            issue=issue,
            participant_ids=["agent1", "agent2"],
            mediator_id="mediator1",
            strategies={"participant1": NegotiationStrategy.HARDBALL},
        )

        # Wait for result
        outcome = await result.wait(timeout=60.0)
        if outcome and outcome.success:
            agreement = outcome.result.get("final_agreement")
            print(f"Agreement reached: {agreement}")
        ```

    Example - As action provider for multi-agent composition:
        ```python
        from polymathera.colony.agents.games.negotiation import run_negotiation_game

        class ProjectManagerPolicy(CacheAwareActionPolicy):
            def __init__(self, agent, ...):
                super().__init__(
                    agent=agent,
                    action_providers=[run_negotiation_game],  # Add as action
                    ...
                )

        # Now the policy can plan a "run_negotiation_game" action
        # when resource conflicts need resolution
        ```
    """
    game_id = game_id or f"negotiation_game_{uuid.uuid4().hex[:8]}"
    capabilities = capabilities or {}

    # Build participants mapping
    participants: dict[str, str] = {}
    coordinator_id = coordinator_id or f"{game_id}_coordinator"
    participants[coordinator_id] = "coordinator"

    if participant_ids is None:
        participant_ids = [f"{game_id}_participant_{i}" for i in range(num_participants)]

    for pid in participant_ids:
        participants[pid] = "participant"

    if mediator_id:
        participants[mediator_id] = "mediator"

    # Default strategies
    default_strategies = strategies or {}

    # Create coordinator config
    config = NegotiationGameConfig(
        game_id=game_id,
        role="coordinator",
        issue=issue,
        strategy=default_strategies.get(coordinator_id, NegotiationStrategy.COMPROMISING),
        use_llm_reasoning=use_llm_reasoning,
        participants=participants,
    )

    # Get capabilities for coordinator role
    coordinator_caps = capabilities.get("coordinator", [])

    # Spawn coordinator agent and initialize it (which spawns others and starts game)
    await owner.spawn_child_agents(
        blueprints=[NegotiationCoordinatorAgent.bind(
            agent_id=coordinator_id,
            capabilities=coordinator_caps,
            metadata=AgentMetadata(parameters={
                "game_config": config.model_dump(),
                "role_capabilities": capabilities,  # Pass to coordinator for spawning others
            }),
        )],
        return_handles=False,
    )

    # Create result handle using the same game_id scope
    # Parent uses NegotiationGameProtocol with game_id to monitor the game
    game_protocol = NegotiationGameProtocol(agent=owner, game_id=game_id)
    await game_protocol.initialize()
    result = await game_protocol.get_result_future()

    logger.info(
        f"Spawned negotiation game {game_id} with {len(participant_ids)} participants"
        + (f" and mediator {mediator_id}" if mediator_id else "")
    )

    return result




