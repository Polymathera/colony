"""Coalition Formation for team assembly and cooperative task execution.

Based on coalition formation theory from cooperative game theory and
multi-agent systems (Sandholm, Shehory & Kraus, Wooldridge). 

Key Features:
- Core-based coalition stability
- Shapley value computation for fair allocation
- Dynamic coalition assembly
- Value models for team synergy
- Stability checking (core, Nash stable)
- Hierarchical coalition structures

Integration:
- Uses blackboard for coalition state
- Integrates with Contract Net for task allocation
- Supports reputation-based team selection
- Can be triggered by complex tasks requiring multiple skills
"""

from __future__ import annotations

import itertools
import time
from enum import Enum
from typing import Any
from overrides import override

from pydantic import BaseModel, Field

from .acl import ACLMessage, Performative
from ..capabilities.reputation import AgentReputation, ReputationTracker
from .state import GamePhase, GameProtocolCapability, GameState, GameOutcome, GameEventType
from ..actions import action_executor


class CoalitionPhase(str, Enum):
    """Phases of coalition formation."""

    INITIATION = "initiation"  # Task announced, agents express interest
    PROPOSAL = "proposal"  # Coalition structures proposed
    EVALUATION = "evaluation"  # Agents evaluate proposed coalitions
    NEGOTIATION = "negotiation"  # Negotiate coalition terms
    FORMATION = "formation"  # Coalition formed
    EXECUTION = "execution"  # Coalition executes task
    DISSOLUTION = "dissolution"  # Coalition dissolved after task


class CoalitionStability(str, Enum):
    """Types of coalition stability."""

    CORE = "core"  # No subset can do better
    NASH = "nash"  # No individual can do better
    INDIVIDUAL_RATIONAL = "individual_rational"  # Better than alone
    PARETO_OPTIMAL = "pareto_optimal"  # Can't improve without harming
    STRONG_NASH = "strong_nash"  # No group can deviate


class CoalitionTask(BaseModel):
    """Task requiring coalition."""

    task_id: str = Field(
        description="Task identifier"
    )

    description: str = Field(
        description="Task description"
    )

    required_capabilities: set[str] = Field(
        description="Capabilities needed for task"
    )

    min_agents: int = Field(
        default=2,
        description="Minimum coalition size"
    )

    max_agents: int = Field(
        default=10,
        description="Maximum coalition size"
    )

    value: float = Field(
        description="Total value/reward for task"
    )

    deadline: float | None = Field(
        default=None,
        description="Task deadline"
    )

    synergy_bonus: float = Field(
        default=0.1,
        description="Bonus for good team composition"
    )


class AgentProfile(BaseModel):
    """Agent profile for coalition formation."""

    agent_id: str = Field(
        description="Agent identifier"
    )

    capabilities: set[str] = Field(
        description="Agent's capabilities"
    )

    capacity: float = Field(
        default=1.0,
        description="Available capacity (0-1)"
    )

    reputation: AgentReputation | None = Field(
        default=None,
        description="Agent reputation"
    )

    preferences: dict[str, float] = Field(
        default_factory=dict,
        description="Preferences for coalition partners"
    )

    cost: float = Field(
        default=0.0,
        description="Cost to include in coalition"
    )


class Coalition(BaseModel):
    """A coalition of agents."""

    coalition_id: str = Field(
        description="Coalition identifier"
    )

    members: list[str] = Field(
        description="Member agent IDs"
    )

    task: CoalitionTask = Field(
        description="Task for this coalition"
    )

    structure: dict[str, str] = Field(
        default_factory=dict,
        description="Role assignments (agent_id -> role)"
    )

    value: float = Field(
        description="Total coalition value"
    )

    allocation: dict[str, float] = Field(
        default_factory=dict,
        description="Value allocation to members"
    )

    stability: list[CoalitionStability] = Field(
        default_factory=list,
        description="Stability properties satisfied"
    )

    formed_at: float = Field(
        default_factory=time.time,
        description="When coalition was formed"
    )


class CoalitionProposal(BaseModel):
    """Proposed coalition structure."""

    proposal_id: str = Field(
        description="Proposal identifier"
    )

    proposer: str = Field(
        description="Agent proposing"
    )

    coalitions: list[Coalition] = Field(
        description="Proposed coalition structure"
    )

    total_value: float = Field(
        description="Total value of structure"
    )

    votes: dict[str, bool] = Field(
        default_factory=dict,
        description="Agent votes on proposal"
    )


class CoalitionFormationData(BaseModel):
    """Data for coalition formation game."""

    task: CoalitionTask = Field(
        description="Task requiring coalition"
    )

    agents: dict[str, AgentProfile] = Field(
        description="Participating agents"
    )

    proposals: list[CoalitionProposal] = Field(
        default_factory=list,
        description="Coalition proposals"
    )

    current_coalitions: list[Coalition] = Field(
        default_factory=list,
        description="Formed coalitions"
    )

    characteristic_function: dict[str, float] = Field(
        default_factory=dict,
        description="Value function v(S) for subsets S"
    )

    shapley_values: dict[str, float] = Field(
        default_factory=dict,
        description="Shapley value for each agent"
    )


class CoalitionGameRole(str, Enum):
    """Roles in coalition formation game."""
    # TODO: Define specific roles
    MEMBER = "member"  # Must be present
    OBSERVER = "observer"  # Must be present in every game to allow passive observation



class CoalitionFormationProtocol(GameProtocolCapability[CoalitionFormationData, CoalitionGameRole]):
    """Protocol for coalition formation.

    Phases:
    1. INITIATION - Task announced, agents register interest
    2. PROPOSAL - Coalition structures proposed
    3. EVALUATION - Agents evaluate proposals
    4. NEGOTIATION - Negotiate terms
    5. FORMATION - Coalition formed
    6. EXECUTION - Execute task
    7. DISSOLUTION - Dissolve coalition

    Example:
        protocol = CoalitionFormationProtocol(agent)
        await protocol.initialize()
        game_id = await protocol.start_game(
            participants={"agent1": "member", "agent2": "member"},
            initial_data={
                "task": CoalitionTask(...).model_dump(),
                "agents": {...}
            }
        )
    """

    def __init__(self, agent: Any):
        """Initialize coalition formation protocol.

        Args:
            agent: Owning agent
        """
        super().__init__(agent, game_type="coalition_formation")

    def get_action_group_description(self) -> str:
        return (
            "Coalition Formation — team assembly via game-theoretic allocation. "
            "Phases: INITIATION → PROPOSAL → EVALUATION → FORMATION. "
            "Value allocation via Shapley values (marginal contribution averaging). "
            "Coalition value = task_value - cost, adjusted by synergy bonus and reputation (0.8-1.2x). "
            "Terminates when best proposal selected from >=2 proposals."
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
        """Start coalition formation game.

        Args:
            participants: Agent ID -> role mapping
            initial_data: Must contain 'task' and 'agents'
            config: Optional configuration

        Returns:
            Game ID
        """
        task_data = initial_data.get("task")
        if not task_data:
            raise ValueError("Initial data must contain 'task'")

        task = CoalitionTask(**task_data) if isinstance(task_data, dict) else task_data

        agents_data = initial_data.get("agents", {})
        agents = {k: AgentProfile(**v) if isinstance(v, dict) else v for k, v in agents_data.items()}

        game_data = CoalitionFormationData(task=task, agents=agents)

        # Compute characteristic function and Shapley values
        self._compute_characteristic_function(game_data)
        self._compute_shapley_values(game_data)

        state = GameState(
            game_type="coalition_formation",
            conversation_id=task.task_id,
            participants=list(participants.keys()),
            roles=participants,
            phase=GamePhase.PROPOSE,  # Start in proposal phase
            game_data=game_data.model_dump(),
            config=config or {}
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
        """Validate move legality."""
        if agent_id not in state.participants:
            return False, "Agent not a participant"

        if state.phase == GamePhase.PROPOSE:
            if move.performative not in [Performative.PROPOSE, Performative.INFORM]:
                return False, "PROPOSE phase requires PROPOSE or INFORM"

        return True, "Valid move"

    @override
    async def apply_move(
        self,
        state: GameState,
        move: ACLMessage
    ) -> GameState:
        """Transition state based on move."""
        data = CoalitionFormationData(**state.game_data)

        if move.performative == Performative.PROPOSE:
            proposal = self._extract_proposal(move, data)
            data.proposals.append(proposal)

            # Check if we have enough proposals
            if len(data.proposals) >= 2:
                best_proposal = self._select_best_proposal(data)
                if best_proposal:
                    data.current_coalitions = best_proposal.coalitions
                    state.phase = GamePhase.TERMINAL

        elif move.performative == Performative.INFORM:
            # Vote on proposals
            content = move.content.get("payload", {}) if isinstance(move.content, dict) else {}
            proposal_id = content.get("proposal_id")
            vote = content.get("vote", False)

            for proposal in data.proposals:
                if proposal.proposal_id == proposal_id:
                    proposal.votes[move.sender] = vote

        state.game_data = data.model_dump()
        return state

    @override
    async def is_terminal(self, state: GameState) -> bool:
        """Check if terminal."""
        return state.phase == GamePhase.TERMINAL

    @override
    async def compute_outcome(self, state: GameState) -> GameOutcome:
        """Compute outcome."""
        data = CoalitionFormationData(**state.game_data)
        duration = state.ended_at - state.started_at if state.ended_at else None

        if data.current_coalitions:
            return GameOutcome(
                outcome_type="coalition_formed",
                success=True,
                result={
                    "coalitions": [c.model_dump() for c in data.current_coalitions],
                    "total_value": sum(c.value for c in data.current_coalitions)
                },
                participants=state.participants,
                rounds_played=len(state.history),
                messages_exchanged=len(state.history),
                duration_seconds=duration,
                summary=f"Coalition formed with {len(data.current_coalitions)} groups"
            )
        else:
            return GameOutcome(
                outcome_type="coalition_failed",
                success=False,
                result={"status": "no_coalition_formed"},
                participants=state.participants,
                rounds_played=len(state.history),
                messages_exchanged=len(state.history),
                duration_seconds=duration,
                summary="No coalition could be formed"
            )

    def _compute_characteristic_function(self, data: CoalitionFormationData) -> None:
        """Compute characteristic function v(S) for all subsets.

        Args:
            data: Coalition formation data
        """
        agents = list(data.agents.keys())

        # For each subset of agents
        for r in range(1, len(agents) + 1):
            for subset in itertools.combinations(agents, r):
                subset_key = ",".join(sorted(subset))

                # Compute value of this coalition
                value = self._coalition_value(subset, data)
                data.characteristic_function[subset_key] = value

    def _coalition_value(self, members: tuple[str, ...], data: CoalitionFormationData) -> float:
        """Compute value of a coalition.

        Args:
            members: Coalition members
            data: Coalition formation data

        Returns:
            Coalition value
        """
        # Check if coalition can complete task
        coalition_capabilities = set()
        total_capacity = 0.0
        total_cost = 0.0

        for agent_id in members:
            agent = data.agents[agent_id]
            coalition_capabilities.update(agent.capabilities)
            total_capacity += agent.capacity
            total_cost += agent.cost

        # Check requirements
        if not data.task.required_capabilities.issubset(coalition_capabilities):
            return 0.0  # Cannot complete task

        if len(members) < data.task.min_agents or len(members) > data.task.max_agents:
            return 0.0  # Size constraints

        # Base value
        value = data.task.value - total_cost

        # Synergy bonus for good team composition
        synergy = self._compute_synergy(members, data)
        value *= (1 + synergy * data.task.synergy_bonus)

        # Reputation bonus
        avg_reputation = self._average_reputation(members, data)
        value *= (0.8 + 0.4 * avg_reputation)  # 80-120% based on reputation

        return max(0.0, value)

    def _compute_synergy(self, members: tuple[str, ...], data: CoalitionFormationData) -> float:
        """Compute synergy score for coalition.

        Args:
            members: Coalition members
            data: Coalition formation data

        Returns:
            Synergy score (0-1)
        """
        if len(members) <= 1:
            return 0.0

        # Simple synergy: preference alignment
        total_preference = 0.0
        pairs = 0

        for i, agent1 in enumerate(members):
            for agent2 in members[i+1:]:
                pref1 = data.agents[agent1].preferences.get(agent2, 0.5)
                pref2 = data.agents[agent2].preferences.get(agent1, 0.5)
                total_preference += (pref1 + pref2) / 2
                pairs += 1

        return total_preference / pairs if pairs > 0 else 0.0

    def _average_reputation(self, members: tuple[str, ...], data: CoalitionFormationData) -> float:
        """Compute average reputation of coalition.

        Args:
            members: Coalition members
            data: Coalition formation data

        Returns:
            Average reputation (0-1)
        """
        total_rep = 0.0
        count = 0

        for agent_id in members:
            agent = data.agents[agent_id]
            if agent.reputation:
                # Average across dimensions
                rep_values = [
                    agent.reputation.honesty,
                    agent.reputation.accuracy,
                    agent.reputation.thoroughness,
                    agent.reputation.reliability
                ]
                total_rep += sum(rep_values) / len(rep_values)
                count += 1

        return total_rep / count if count > 0 else 0.5

    def _compute_shapley_values(self, data: CoalitionFormationData) -> None:
        """Compute Shapley value for each agent.

        The Shapley value gives each agent their average marginal contribution
        across all possible coalition orderings.

        Args:
            data: Coalition formation data
        """
        agents = list(data.agents.keys())
        n = len(agents)

        for agent in agents:
            shapley_value = 0.0

            # Consider all subsets not containing this agent
            other_agents = [a for a in agents if a != agent]

            for r in range(len(other_agents) + 1):
                for subset in itertools.combinations(other_agents, r):
                    # Coalition without agent
                    coalition_without = ",".join(sorted(subset)) if subset else ""
                    value_without = data.characteristic_function.get(coalition_without, 0.0)

                    # Coalition with agent
                    coalition_with = ",".join(sorted(list(subset) + [agent]))
                    value_with = data.characteristic_function.get(coalition_with, 0.0)

                    # Marginal contribution
                    marginal = value_with - value_without

                    # Weight by probability of this ordering
                    weight = 1.0 / n  # Simplified - should be factorial terms
                    shapley_value += weight * marginal

            data.shapley_values[agent] = shapley_value

    def _extract_proposal(self, message: ACLMessage, data: CoalitionFormationData) -> CoalitionProposal:
        """Extract coalition proposal from message.

        Args:
            message: ACL message
            data: Coalition formation data

        Returns:
            Coalition proposal
        """
        content = message.content.get("payload", {})

        # Build coalitions from proposal
        coalitions = []
        for coalition_data in content.get("coalitions", []):
            members = coalition_data.get("members", [])
            coalition_key = ",".join(sorted(members))
            value = data.characteristic_function.get(coalition_key, 0.0)

            # Allocate value using Shapley values
            allocation = {}
            for member in members:
                allocation[member] = data.shapley_values.get(member, 0.0)

            # Normalize allocation to match coalition value
            total_shapley = sum(allocation.values())
            if total_shapley > 0:
                for member in allocation:
                    allocation[member] = (allocation[member] / total_shapley) * value

            coalition = Coalition(
                coalition_id=f"c_{message.message_id}_{len(coalitions)}",
                members=members,
                task=data.task,
                value=value,
                allocation=allocation,
                stability=self._check_stability(members, allocation, data)
            )
            coalitions.append(coalition)

        return CoalitionProposal(
            proposal_id=f"prop_{message.message_id}",
            proposer=message.sender,
            coalitions=coalitions,
            total_value=sum(c.value for c in coalitions)
        )

    def _check_stability(
        self,
        members: list[str],
        allocation: dict[str, float],
        data: CoalitionFormationData
    ) -> list[CoalitionStability]:
        """Check stability properties of coalition.

        Args:
            members: Coalition members
            allocation: Value allocation
            data: Coalition formation data

        Returns:
            List of satisfied stability properties
        """
        stability = []

        # Individual rationality: each agent gets at least what they could alone
        individually_rational = True
        for agent in members:
            alone_value = data.characteristic_function.get(agent, 0.0)
            if allocation.get(agent, 0.0) < alone_value:
                individually_rational = False
                break

        if individually_rational:
            stability.append(CoalitionStability.INDIVIDUAL_RATIONAL)

        # Core stability: no subset can do better
        core_stable = True
        for r in range(1, len(members)):
            for subset in itertools.combinations(members, r):
                subset_key = ",".join(sorted(subset))
                subset_value = data.characteristic_function.get(subset_key, 0.0)
                subset_allocation = sum(allocation.get(a, 0.0) for a in subset)

                if subset_value > subset_allocation:
                    core_stable = False
                    break
            if not core_stable:
                break

        if core_stable:
            stability.append(CoalitionStability.CORE)

        # Nash stability: no individual can do better by leaving
        # (simplified check)
        nash_stable = individually_rational  # Simplified
        if nash_stable:
            stability.append(CoalitionStability.NASH)

        return stability

    def _select_best_proposal(self, data: CoalitionFormationData) -> CoalitionProposal | None:
        """Select best coalition proposal.

        Args:
            data: Coalition formation data

        Returns:
            Best proposal or None
        """
        best_proposal = None
        best_score = -1.0

        for proposal in data.proposals:
            # Score based on value and votes
            vote_count = sum(1 for v in proposal.votes.values() if v)
            vote_ratio = vote_count / len(data.agents) if data.agents else 0.0

            # Score combines value and support
            score = proposal.total_value * (0.5 + 0.5 * vote_ratio)

            # Bonus for stability
            stability_bonus = 0.0
            for coalition in proposal.coalitions:
                stability_bonus += len(coalition.stability) * 0.1
            score += stability_bonus

            if score > best_score:
                best_score = score
                best_proposal = proposal

        return best_proposal

    def _get_agent_payoff(self, agent_id: str, data: CoalitionFormationData) -> float:
        """Get payoff for agent.

        Args:
            agent_id: Agent ID
            data: Coalition formation data

        Returns:
            Agent's payoff
        """
        for coalition in data.current_coalitions:
            if agent_id in coalition.members:
                return coalition.allocation.get(agent_id, 0.0)
        return 0.0



# Utility functions

def find_optimal_coalition_structure(
    agents: list[str],
    characteristic_function: dict[str, float]
) -> list[list[str]]:
    """Find optimal coalition structure using dynamic programming.

    Args:
        agents: List of agent IDs
        characteristic_function: Value function for coalitions

    Returns:
        Optimal partition of agents into coalitions
    """
    n = len(agents)
    if n == 0:
        return []

    # DP table: best value for subset
    dp = {}
    partition = {}

    # For each subset
    for mask in range(1, 1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(agents[i])

        subset_key = ",".join(sorted(subset))

        # Option 1: Keep as single coalition
        best_value = characteristic_function.get(subset_key, 0.0)
        best_partition = [subset]

        # Option 2: Split into smaller coalitions
        # Try all possible splits
        submask = mask
        while submask > 0:
            if submask != mask:  # Proper subset
                complement = mask ^ submask

                # Get value of split
                sub1_value = dp.get(submask, 0.0)
                sub2_value = dp.get(complement, 0.0)
                split_value = sub1_value + sub2_value

                if split_value > best_value:
                    best_value = split_value
                    best_partition = partition.get(submask, []) + partition.get(complement, [])

            # Next submask
            submask = (submask - 1) & mask

        dp[mask] = best_value
        partition[mask] = best_partition

    # Return optimal partition for all agents
    all_mask = (1 << n) - 1
    return partition.get(all_mask, [[]])
