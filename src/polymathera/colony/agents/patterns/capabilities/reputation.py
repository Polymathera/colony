"""Reputation tracking and no-regret learning for agent adaptation.

Based on Shoham & Leyton-Brown's learning in games (Chapter 7):
- No-regret learning: Multiplicative weights algorithm
- VCG-style marginal contribution tracking
- Targeted learning for task-specific adaptation

Reputation serves multiple purposes:
- Bid scoring in Contract Net
- Agent selection for tasks
- Vote weighting in consensus
- Trust in hypothesis validation

The reputation system enables:
- Adaptation over repeated interactions
- Incentivizing truthful, thorough behavior
- Identifying reliable vs unreliable agents
- Learning optimal agent mixtures

============================================================================

ReputationAgent: Updates agent reputations based on performance.

The ReputationAgent is a meta-agent that:
- Monitors task outcomes and game results
- Updates agent reputations based on performance
- Implements VCG-style marginal contribution tracking
- Provides reputation data for agent selection

Based on Shoham & Leyton-Brown's mechanism design:
"Reward agents according to their marginal contribution to global performance"

Integration:
- Subscribes to task completion events
- Subscribes to game outcome events
- Updates reputation in blackboard
- Used by Contract Net for bid scoring

Programming Model (AgentHandle Pattern):
---------------------------------------
```python
# Spawn reputation agent with handle
handle = (await owner.spawn_child_agents(
    blueprints=[ReputationAgent.bind(
        capability_blueprints=[ReputationCapability.bind()],
    )],
    return_handles=True,
))[0]

# Get capability and communicate
reputation = handle.get_capability(ReputationCapability)
await reputation.stream_events_to_queue(self.get_event_queue())

# Or update reputation directly
await reputation.update_from_outcome(outcome=task_outcome)
future = await reputation.get_result_future()
updated_rep = await future.wait(timeout=30.0)
```
"""
from __future__ import annotations

import time
import asyncio
import logging
from typing import Any
from pydantic import BaseModel, Field
from uuid import uuid4
from overrides import override

from ...base import (
    Agent,
    AgentCapability,
    AgentHandle,
    CapabilityResultFuture,
)
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ..actions.policies import (
    action_executor,
    ActionPolicyExecutionState,
)
from ...models import Action, AgentSuspensionState, PolicyREPL
from ... import KeyPatternFilter, BlackboardEvent
from ..events import event_handler, EventProcessingResult
from ...blackboard import BlackboardEvent
from ..games.state import GameState


logger = logging.getLogger(__name__)


class AgentReputation(BaseModel):
    """Reputation profile for an agent.

    Tracks multiple dimensions:
    - Honesty: How often claims are validated
    - Accuracy: Quality of contributions
    - Thoroughness: Completeness of work
    - Reliability: Success rate on tasks

    Examples:
        High-reputation agent:
        ```python
        rep = AgentReputation(
            agent_id="analyzer_expert_001",
            honesty_score=0.95,
            accuracy_score=0.88,
            thoroughness_score=0.92,
            reliability_score=0.90,
            total_tasks=150,
            successful_tasks=135,
            failed_tasks=15
        )
        ```
    """

    agent_id: str = Field(
        description="Agent identifier"
    )

    # Core reputation dimensions
    honesty_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How often agent's claims are validated (0.0-1.0)"
    )

    accuracy_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality/correctness of agent's work (0.0-1.0)"
    )

    thoroughness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Completeness of agent's analysis (0.0-1.0)"
    )

    reliability_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Success rate on assigned tasks (0.0-1.0)"
    )

    # Task statistics
    total_tasks: int = Field(
        default=0,
        description="Total tasks attempted"
    )

    successful_tasks: int = Field(
        default=0,
        description="Successfully completed tasks"
    )

    failed_tasks: int = Field(
        default=0,
        description="Failed tasks"
    )

    # Contribution tracking (VCG-style)
    marginal_contributions: list[float] = Field(
        default_factory=list,
        description="History of marginal contributions to outcomes"
    )

    avg_marginal_contribution: float = Field(
        default=0.0,
        description="Average marginal contribution"
    )

    # Task-specific reputation
    task_type_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Reputation by task type"
    )

    # Learning data
    learning_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning rate for reputation updates"
    )

    # Timestamps
    created_at: float = Field(
        default_factory=time.time,
        description="When reputation tracking started"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="Last reputation update"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional reputation data"
    )

    def get_overall_score(self) -> float:
        """Get overall reputation score.

        Weighted combination of all dimensions.

        Returns:
            Overall score (0.0-1.0)
        """
        return (
            0.3 * self.honesty_score +
            0.3 * self.accuracy_score +
            0.2 * self.thoroughness_score +
            0.2 * self.reliability_score
        )

    def get_task_score(self, task_type: str) -> float:
        """Get reputation for specific task type.

        Args:
            task_type: Task type

        Returns:
            Task-specific score (falls back to overall if no specific data)
        """
        return self.task_type_scores.get(task_type, self.get_overall_score())


class TaskOutcome(BaseModel):
    """Outcome of a task for reputation update."""

    task_id: str = Field(
        description="Task identifier"
    )

    task_type: str = Field(
        description="Type of task"
    )

    executor_id: str = Field(
        description="Agent that executed task"
    )

    # Quality metrics
    success: bool = Field(
        description="Whether task succeeded"
    )

    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality of result (0.0-1.0)"
    )

    completeness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Completeness of result (0.0-1.0)"
    )

    # Accuracy tracking
    claims_made: int = Field(
        default=0,
        description="Number of claims made"
    )

    claims_validated: int = Field(
        default=0,
        description="Number of claims later validated"
    )

    # Cost tracking
    estimated_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Original cost estimate"
    )

    actual_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Actual cost incurred"
    )

    # Metadata
    completed_at: float = Field(
        default_factory=time.time,
        description="When task completed"
    )



class ReputationTracker:
    """Tracks and updates agent reputations.

    Implements:
    - VCG-style marginal contribution tracking
    - Multi-dimensional reputation updates
    - Task-specific reputation
    - No-regret learning preparation
    """

    def __init__(self, blackboard: Any):
        """Initialize reputation tracker.

        Args:
            blackboard: Blackboard for storing reputations
        """
        self.blackboard = blackboard
        self.namespace = "reputation"

    async def get_reputation(self, agent_id: str) -> AgentReputation:
        """Get agent's reputation.

        Args:
            agent_id: Agent ID

        Returns:
            Agent reputation (creates new if doesn't exist)
        """
        key = f"{self.namespace}:agent:{agent_id}"
        data = await self.blackboard.read(key)

        if data is None:
            # Create new reputation
            reputation = AgentReputation(agent_id=agent_id)
            await self._store_reputation(reputation)
            return reputation

        return AgentReputation(**data)

    async def update_reputation(
        self,
        agent_id: str,
        outcome: TaskOutcome
    ) -> AgentReputation:
        """Update agent reputation based on task outcome.

        Args:
            agent_id: Agent ID
            outcome: Task outcome

        Returns:
            Updated reputation
        """
        reputation = await self.get_reputation(agent_id)

        # Update task counts
        reputation.total_tasks += 1
        if outcome.success:
            reputation.successful_tasks += 1
        else:
            reputation.failed_tasks += 1

        # Update reliability
        reputation.reliability_score = reputation.successful_tasks / reputation.total_tasks

        # Update accuracy (exponential moving average)
        alpha = reputation.learning_rate
        reputation.accuracy_score = (
            (1 - alpha) * reputation.accuracy_score +
            alpha * outcome.quality_score
        )

        # Update thoroughness
        reputation.thoroughness_score = (
            (1 - alpha) * reputation.thoroughness_score +
            alpha * outcome.completeness_score
        )

        # Update honesty (if claims tracked)
        if outcome.claims_made > 0:
            honesty_in_task = outcome.claims_validated / outcome.claims_made
            reputation.honesty_score = (
                (1 - alpha) * reputation.honesty_score +
                alpha * honesty_in_task
            )

        # Update task-specific reputation
        task_type = outcome.task_type
        if task_type not in reputation.task_type_scores:
            reputation.task_type_scores[task_type] = reputation.get_overall_score()
        else:
            reputation.task_type_scores[task_type] = (
                (1 - alpha) * reputation.task_type_scores[task_type] +
                alpha * outcome.quality_score
            )

        reputation.updated_at = time.time()

        # Store updated reputation
        await self._store_reputation(reputation)

        return reputation

    async def update_marginal_contribution(
        self,
        agent_id: str,
        marginal_contribution: float
    ) -> None:
        """Update agent's marginal contribution (VCG-style).

        Args:
            agent_id: Agent ID
            marginal_contribution: Contribution value
        """
        reputation = await self.get_reputation(agent_id)

        reputation.marginal_contributions.append(marginal_contribution)

        # Update average
        reputation.avg_marginal_contribution = sum(reputation.marginal_contributions) / len(reputation.marginal_contributions)

        reputation.updated_at = time.time()
        await self._store_reputation(reputation)

    async def get_agent_ranking(
        self,
        metric: str = "overall",
        task_type: str | None = None
    ) -> list[tuple[str, float]]:
        """Get agent ranking by metric.

        Args:
            metric: Metric to rank by ("overall", "honesty", "accuracy", etc.)
            task_type: Optional task type for task-specific ranking

        Returns:
            List of (agent_id, score) tuples, sorted descending
        """
        # Query all reputations
        all_reputations = await self._get_all_reputations()

        # Score each agent
        scored = []
        for reputation in all_reputations:
            if metric == "overall":
                score = reputation.get_overall_score()
            elif metric == "honesty":
                score = reputation.honesty_score
            elif metric == "accuracy":
                score = reputation.accuracy_score
            elif metric == "thoroughness":
                score = reputation.thoroughness_score
            elif metric == "reliability":
                score = reputation.reliability_score
            elif metric == "marginal_contribution":
                score = reputation.avg_marginal_contribution
            else:
                score = reputation.get_overall_score()

            # Apply task-specific filter if requested
            if task_type:
                score = reputation.get_task_score(task_type)

            scored.append((reputation.agent_id, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    async def _store_reputation(self, reputation: AgentReputation) -> None:
        """Store reputation in blackboard.

        Args:
            reputation: Reputation to store
        """
        key = f"{self.namespace}:agent:{reputation.agent_id}"

        await self.blackboard.write(
            key=key,
            value=reputation.model_dump(),
            tags={"reputation", reputation.agent_id}
        )

    async def _get_all_reputations(self) -> list[AgentReputation]:
        """Get all agent reputations.

        Returns:
            List of reputations
        """
        # Placeholder - would query blackboard for all reputation entries
        return []


class NoRegretLearner:
    """Implements no-regret learning via multiplicative weights.

    As described in Shoham & Leyton-Brown Chapter 7:
    "At each episode, choose mixture over agents/strategies, observe quality,
    update weights to minimize regret vs best fixed policy in hindsight."

    This enables the system to learn which agents to select for which tasks,
    converging to a correlated equilibrium.
    """

    def __init__(
        self,
        agent_ids: list[str],
        eta: float = 0.1
    ):
        """Initialize no-regret learner.

        Args:
            agent_ids: List of agent IDs to track
            eta: Learning rate (0.0-1.0)
        """
        self.agent_ids = agent_ids
        self.weights = {aid: 1.0 for aid in agent_ids}
        self.eta = eta
        self.episode_history: list[dict[str, Any]] = []

    def get_mixture(self) -> dict[str, float]:
        """Get current mixture over agents.

        Returns:
            Dictionary of agent_id -> probability
        """
        total = sum(self.weights.values())
        if total == 0:
            # All weights zero, uniform distribution
            return {aid: 1.0 / len(self.agent_ids) for aid in self.agent_ids}

        return {aid: w / total for aid, w in self.weights.items()}

    def update(self, episode_rewards: dict[str, float]) -> None:
        """Update weights based on episode rewards.

        Multiplicative weights update: w_i *= (1 + eta * r_i)

        Args:
            episode_rewards: Agent ID -> reward mapping
        """
        for agent_id, reward in episode_rewards.items():
            if agent_id in self.weights:
                # Multiplicative update
                self.weights[agent_id] *= (1 + self.eta * reward)

        # Record episode
        self.episode_history.append({
            "timestamp": time.time(),
            "rewards": episode_rewards.copy(),
            "weights": self.weights.copy()
        })

    def get_regret(self) -> float:
        """Calculate cumulative regret vs best fixed agent.

        Returns:
            Regret value
        """
        if not self.episode_history:
            return 0.0

        # Calculate total reward for mixture strategy
        mixture_total = 0.0
        for episode in self.episode_history:
            mixture = self.get_mixture()
            episode_reward = sum(
                mixture.get(aid, 0.0) * episode["rewards"].get(aid, 0.0)
                for aid in self.agent_ids
            )
            mixture_total += episode_reward

        # Calculate total reward for best fixed agent
        best_fixed_total = max(
            sum(episode["rewards"].get(aid, 0.0) for episode in self.episode_history)
            for aid in self.agent_ids
        )

        return best_fixed_total - mixture_total

    def sample_agent(self) -> str:
        """Sample agent according to current mixture.

        Returns:
            Sampled agent ID
        """
        import random
        mixture = self.get_mixture()

        # Weighted random selection
        agents = list(mixture.keys())
        probabilities = [mixture[aid] for aid in agents]

        return random.choices(agents, weights=probabilities, k=1)[0]


class TargetedLearningManager:
    """Manages targeted learning for task-specific adaptation.

    As described in Shoham & Leyton-Brown:
    "Learn to be a best response against target class of opponents/tasks
    while maintaining safe defaults."

    Clusters tasks into types and learns per-cluster tuning:
    - Which agents to spawn
    - How many validation rounds
    - Which policies to use
    """

    def __init__(self, blackboard: Any):
        """Initialize targeted learning manager.

        Args:
            blackboard: Blackboard for storing learning data
        """
        self.blackboard = blackboard
        self.namespace = "learning"

        # Learners by task cluster
        self.learners: dict[str, NoRegretLearner] = {}

    async def get_or_create_learner(
        self,
        task_cluster: str,
        agent_ids: list[str]
    ) -> NoRegretLearner:
        """Get learner for task cluster.

        Args:
            task_cluster: Task cluster ID
            agent_ids: Agent IDs to track

        Returns:
            No-regret learner
        """
        if task_cluster not in self.learners:
            self.learners[task_cluster] = NoRegretLearner(agent_ids)

        return self.learners[task_cluster]

    async def update_from_outcome(
        self,
        task_cluster: str,
        agent_ids: list[str],
        quality_scores: dict[str, float]
    ) -> None:
        """Update learner based on task outcomes.

        Args:
            task_cluster: Task cluster
            agent_ids: Agents involved
            quality_scores: Quality score for each agent
        """
        learner = await self.get_or_create_learner(task_cluster, agent_ids)
        learner.update(quality_scores)

    async def select_agents(
        self,
        task_cluster: str,
        agent_pool: list[str],
        num_agents: int = 1
    ) -> list[str]:
        """Select agents for task using learned mixture.

        Args:
            task_cluster: Task cluster
            agent_pool: Available agents
            num_agents: Number of agents to select

        Returns:
            Selected agent IDs
        """
        learner = await self.get_or_create_learner(task_cluster, agent_pool)

        selected = []
        for _ in range(num_agents):
            agent_id = learner.sample_agent()
            selected.append(agent_id)

        return selected


# Utility functions

async def update_reputation_from_game(
    game_outcome: Any,  # GameOutcome
    blackboard: Any
) -> dict[str, AgentReputation]:
    """Update reputations based on game outcome.

    Args:
        game_outcome: Game outcome with participant performance
        blackboard: Blackboard instance

    Returns:
        Updated reputations by agent ID
    """
    tracker = ReputationTracker(blackboard)

    updated_reputations = {}

    # Extract performance metrics from outcome
    # This would be game-specific
    # Placeholder for now

    return updated_reputations


async def select_best_agents(
    task_type: str,
    agent_pool: list[str],
    num_agents: int,
    blackboard: Any
) -> list[str]:
    """Select best agents for task type based on reputation.

    Args:
        task_type: Type of task
        agent_pool: Available agents
        num_agents: Number to select
        blackboard: Blackboard instance

    Returns:
        Selected agent IDs
    """
    tracker = ReputationTracker(blackboard)

    # Get reputations for all agents
    agent_scores = []
    for agent_id in agent_pool:
        reputation = await tracker.get_reputation(agent_id)
        score = reputation.get_task_score(task_type)
        agent_scores.append((agent_id, score))

    # Sort by score and take top N
    agent_scores.sort(key=lambda x: x[1], reverse=True)

    return [agent_id for agent_id, _ in agent_scores[:num_agents]]





class ReputationUpdate(BaseModel):
    """Record of a reputation update."""

    agent_id: str = Field(
        description="Agent whose reputation was updated"
    )

    update_type: str = Field(
        description="Type of update: 'task_outcome', 'game_outcome', 'marginal_contribution'"
    )

    previous_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Previous reputation scores"
    )

    new_scores: dict[str, float] = Field(
        default_factory=dict,
        description="New reputation scores"
    )

    reason: str = Field(
        description="Reason for update"
    )

    timestamp: float = Field(
        default_factory=lambda: time.time(),
        description="When update occurred"
    )


class ReputationUpdateRequest(BaseModel):
    """Request for reputation update."""

    requesting_agent_id: str = Field(
        description="Agent requesting the update"
    )

    outcome: TaskOutcome = Field(
        description="Task outcome for reputation update"
    )


# ============================================================================
# New Pattern: ReputationCapability + Agent + CacheAwareActionPolicy
# ============================================================================


class ReputationCapability(AgentCapability):
    """Capability for managing agent reputations.

    Works in two modes via the `scope_id` parameter:

    1. **Local mode** (in ReputationAgent): Processes reputation update requests
       ```python
       capability = ReputationCapability(agent=self)
       ```

    2. **Remote mode** (in parent agent): Communicates with child reputation agent
       ```python
       handle = await parent.spawn_child_agents(...)[0]
       reputation = handle.get_capability(ReputationCapability)
       await reputation.update_from_outcome(outcome=task_outcome)
       future = await reputation.get_result_future()
       updated_rep = await future.wait(timeout=30.0)
       ```

    Provides @action_executor methods for:
    - update_from_task_outcome: Update reputation based on task outcome
    - update_marginal_contribution: Update VCG-style marginal contribution

    Responsibilities:
    - Monitor task and game outcomes
    - Update agent reputations
    - Track marginal contributions (VCG-style)
    - Provide reputation rankings
    - Detect reputation trends
    """

    def __init__(self, agent: Agent, scope: BlackboardScope = BlackboardScope.COLONY):
        """Initialize reputation capability.

        Args:
            agent: Agent using this capability
            scope: Blackboard scope. Defaults to BlackboardScope.COLONY.
        """
        super().__init__(agent, scope_id=get_scope_prefix(scope, agent))
        self.reputation_tracker = None
        self.update_history: list[ReputationUpdate] = []

    def get_action_group_description(self) -> str:
        return (
            "Reputation Tracking — maintains agent reputation scores based on task outcomes. "
            "Triggered by ReputationUpdateRequest events. Supports VCG-style marginal contribution "
            "updates for mechanism-design-fair scoring. Publish result to blackboard when ready. "
            "Reputation scores are consumed by ContractNet bidding and agent selection."
        )

    def _get_result_key(self) -> str:
        """Get blackboard key for this capability's result."""
        return ScopeUtils.format_key(reputation="result")

    def _get_event_pattern(self) -> str:
        """Get pattern for reputation events."""
        return ScopeUtils.pattern_key(reputation=None)

    async def ensure_tracker(self) -> ReputationTracker:
        """Ensure reputation tracker is initialized."""
        if not self.reputation_tracker:
            blackboard = await self.get_blackboard()
            self.reputation_tracker = ReputationTracker(blackboard)
        return self.reputation_tracker

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ReputationCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ReputationCapability")
        pass

    @override
    async def stream_events_to_queue(self, event_queue: asyncio.Queue[BlackboardEvent]) -> None:
        """Stream reputation events to the given queue.

        Args:
            event_queue: Queue to stream events to. Usually the local event queue of an ActionPolicy.
        """
        blackboard = await self.get_blackboard()
        ### blackboard.stream_events_to_queue(
        ###     event_queue,
        ###     KeyPatternFilter(
        ###         pattern=ScopeUtils.pattern_key(task_outcome=None)
        ###     )
        ### )
        blackboard.stream_events_to_queue(
            event_queue,
            KeyPatternFilter(pattern=self._get_event_pattern())
        )

    @event_handler(pattern=ScopeUtils.pattern_key(reputation="update", requesting_agent_id=None))
    async def handle_reputation_update_request(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle reputation update request events."""
        # Extract request ID from key
        request_id = event.key.split(":")[-1]
        request = ReputationUpdateRequest.model_validate(event.value)

        # Return immediate action to execute analysis
        return EventProcessingResult(
            immediate_action=Action(
                action_type="update_from_reputation_request",
                parameters={
                    "requesting_agent_id": request.requesting_agent_id,
                    "outcome": request.outcome
                },
            )
        )

    @event_handler(pattern=ScopeUtils.pattern_key(state=None)) # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
    async def handle_game_completion(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle game completion events."""
        # Extract request ID from key
        game_state = GameState.model_validate(event.value)
        if not game_state.is_terminal():
            return None

        # Return immediate action to execute analysis
        return EventProcessingResult(
            immediate_action=Action(
                action_type="update_from_game_outcome",
                parameters={
                    "game_type": game_state.game_type,
                    "outcome": game_state.outcome
                },
            )
        )

    @event_handler(pattern=ScopeUtils.pattern_key(task_outcome=None)) # NOTE: The scope_id already contains game_id, so this will only trigger for events in this game's context
    async def handle_task_outcome(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle task outcome events."""
        # Extract request ID from key
        task_outcome = TaskOutcome.model_validate(event.value)

        # Return immediate action to execute analysis
        return EventProcessingResult(
            immediate_action=Action(
                action_type="update_from_task_outcome",
                parameters={
                    "outcome": task_outcome.outcome
                },
            )
        )

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Get future for reputation update result.

        Returns:
            Future that resolves with AgentReputation result
        """
        blackboard = await self.get_blackboard()
        return CapabilityResultFuture(
            result_key=self._get_result_key(),
            blackboard=blackboard,
        )

    # -------------------------------------------------------------------------
    # Capability-Specific Actions
    # -------------------------------------------------------------------------

    @action_executor()
    async def update_from_task_outcome(self, outcome: TaskOutcome) -> AgentReputation:
        """Update reputation based on task outcome.

        Args:
            outcome: Task outcome

        Returns:
            Updated reputation
        """
        tracker = await self.ensure_tracker()

        # Get previous reputation
        prev_reputation = await tracker.get_reputation(outcome.executor_id)
        prev_scores = {
            "honesty": prev_reputation.honesty_score,
            "accuracy": prev_reputation.accuracy_score,
            "thoroughness": prev_reputation.thoroughness_score,
            "reliability": prev_reputation.reliability_score
        }

        # Update reputation
        new_reputation = await tracker.update_reputation(outcome.executor_id, outcome)
        new_scores = {
            "honesty": new_reputation.honesty_score,
            "accuracy": new_reputation.accuracy_score,
            "thoroughness": new_reputation.thoroughness_score,
            "reliability": new_reputation.reliability_score
        }

        # Record update
        update = ReputationUpdate(
            agent_id=outcome.executor_id,
            update_type="task_outcome",
            previous_scores=prev_scores,
            new_scores=new_scores,
            reason=f"Task {outcome.task_id} {'succeeded' if outcome.success else 'failed'} with quality {outcome.quality_score:.2f}"
        )
        self.update_history.append(update)

        logger.info(f"Updated reputation for {outcome.executor_id}: {prev_scores} -> {new_scores}")

        return new_reputation

    @action_executor()
    async def update_marginal_contribution(
        self,
        agent_id: str,
        contribution: float,
        context: str
    ) -> None:
        """Update marginal contribution (VCG-style).

        Args:
            agent_id: Agent ID
            contribution: Marginal contribution value
            context: Context for contribution
        """
        tracker = await self.ensure_tracker()
        await tracker.update_marginal_contribution(agent_id, contribution)

        # Record update
        update = ReputationUpdate(
            agent_id=agent_id,
            update_type="marginal_contribution",
            previous_scores={},
            new_scores={"marginal_contribution": contribution},
            reason=f"Marginal contribution: {context}"
        )
        self.update_history.append(update)

    @action_executor()
    async def publish_reputation_result(
        self,
        reputation: AgentReputation
    ) -> None:
        """Publish reputation update result.

        Writes result to the capability's result key, which resolves
        any CapabilityResultFuture waiting on this capability.

        Args:
            reputation: Updated reputation to publish
        """
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=self._get_result_key(),
            value=reputation.model_dump(),
            agent_id=self.agent.agent_id,
        )



