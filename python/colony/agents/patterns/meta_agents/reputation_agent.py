"""ReputationAgent: Updates agent reputations based on performance.

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

import logging
from typing import Any
from uuid import uuid4


from ...base import (
    Agent,
    AgentHandle,
)
from ...models import AgentMetadata
from ..actions.policies import (
    CacheAwareActionPolicy,
)
from ..capabilities.reputation import (
    ReputationTracker,
    TaskOutcome,
    AgentReputation,
    ReputationCapability,
)
from ..games.hypothesis.capabilities import HypothesisGameProtocol


logger = logging.getLogger(__name__)


class ReputationAgent(Agent):
    """Meta-agent for reputation management using ReputationCapability and CacheAwareActionPolicy.

    Uses ReputationCapability for reputation tracking and CacheAwareActionPolicy
    for event-driven reputation updates based on task and game outcomes.

    Example:
        agent = ReputationAgent(
            agent_id="reputation_agent_001",
            agent_type="reputation",
            tenant_id="tenant_001"
        )
        await agent.initialize()
        # Agent now monitors task_completed and game outcome events
    """

    async def initialize(self) -> None:
        """Initialize reputation agent with capability and policy."""
        self.add_capability_blueprints([ReputationCapability.bind()])

        await super().initialize()

        if not self.has_capability(HypothesisGameProtocol.get_capability_name()):
            capability = HypothesisGameProtocol(
                agent=self,
                game_id=None, # TODO: Should this be set?
                role="observer",  # Observe game outcomes
            )
            await capability.initialize()
            self.add_capability(capability, events_only=True)  # Observe game outcomes

        self.action_policy.use_agent_capabilities([ReputationCapability.get_capability_name()])

        logger.info(f"CacheAwareActionPolicy initialized for {self.agent_id}")

        # Create and initialize policy
        self.action_policy = CacheAwareActionPolicy(self)
        await self.action_policy.initialize()
        logger.info(f"ReputationAgent {self.agent_id} initialized with CacheAwareActionPolicy")




# ============================================================================
# Utility Functions
# ============================================================================


async def spawn_reputation_agent(
    owner: Agent,
    session_id: str | None = None,
    run_id: str | None = None,
) -> AgentHandle:
    """Spawn a ReputationAgent and return a handle for communication.

    Example usage:
    ```python
    handle = await spawn_reputation_agent(owner=self.agent)
    reputation = handle.get_capability(ReputationCapability)

    # Update reputation
    await reputation.update_from_outcome(outcome=task_outcome)
    future = await reputation.get_result_future()
    updated_rep = await future.wait(timeout=30.0)
    ```

    Args:
        owner: Agent spawning the reputation agent
        session_id: Optional session ID (set in blueprint metadata)
        run_id: Optional run ID (set in blueprint metadata)

    Returns:
        AgentHandle for interacting with the reputation agent
    """
    logger.info(f"Spawning ReputationAgent for {owner.agent_id}...")
    agent_id = f"reputation_agent_{owner.tenant_id}_{uuid4().hex[:8]}"

    metadata = AgentMetadata(tenant_id=owner.tenant_id)
    if session_id:
        metadata.session_id = session_id
    if run_id:
        metadata.run_id = run_id

    # TODO: Pass LLMClientRequirements and other deployment parameters to spawn_child_agents
    return await owner.spawn_child_agents(
        blueprints=[ReputationAgent.bind(
            agent_id=agent_id,
            bound_pages=[],
            capability_blueprints=[ReputationCapability.bind()],
            metadata=metadata,
        )],
        soft_affinity=True,
        suspend_agents=True,
        return_handles=True,
    )[0]


async def update_agent_reputation(
    owner: Agent,
    outcome: TaskOutcome,
    timeout: float = 30.0,
) -> AgentReputation | None:
    """Update agent reputation using the AgentHandle pattern.

    High-level utility that spawns a ReputationAgent, sends an update request,
    and waits for the result.

    Args:
        owner: Agent requesting reputation update
        outcome: Task outcome for reputation update
        timeout: Timeout in seconds

    Returns:
        Updated AgentReputation or None on timeout

    Example:
        ```python
        updated = await update_agent_reputation(
            owner=self.agent,
            outcome=task_outcome,
            timeout=30.0,
        )
        if updated:
            print(f"New reliability score: {updated.reliability_score}")
        ```
    """
    logger.info(f"Updating reputation for agent: {outcome.executor_id}...")

    handle = await spawn_reputation_agent(owner=owner)

    try:
        reputation = handle.get_capability(ReputationCapability)

        # Send reputation update request
        await reputation.update_from_outcome(outcome=outcome)

        # Wait for result
        future = await reputation.get_result_future()
        result_data = await future.wait(timeout=timeout)

        if result_data:
            return AgentReputation(**result_data)
        return None

    finally:
        await handle.stop()


async def get_agent_leaderboard(
    blackboard: Any,
    metric: str = "overall",
    top_n: int = 10
) -> list[tuple[str, float]]:
    """Get agent leaderboard by metric.

    Args:
        blackboard: Blackboard instance
        metric: Metric to rank by
        top_n: Number of top agents to return

    Returns:
        List of (agent_id, score) tuples
    """
    tracker = ReputationTracker(blackboard)
    ranking = await tracker.get_agent_ranking(metric=metric)
    return ranking[:top_n]


