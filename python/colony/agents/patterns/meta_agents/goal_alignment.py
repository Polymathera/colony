"""ObjectiveGuardAgent: Prevents goal drift.

The ObjectiveGuardAgent is a meta-agent that:
- Monitors outputs to ensure alignment with original goals
- Detects goal drift (agents doing something other than requested)
- Challenges off-topic or irrelevant work
- Triggers re-planning when drift detected

This addresses the LLM failure mode of:
"Deviating from stated objectives" - agents doing clever but unhelpful things.

Integration:
- Maintains explicit JointGoal for each task
- Compares intermediate results against goals
- Emits challenge messages when drift detected

Programming Model (AgentHandle Pattern):
---------------------------------------
```python
# Spawn objective guard agent with handle
handle = await owner.spawn_child_agents(
    agent_specs=[AgentSpawnSpec(
        agent_id=f"objective_guard_agent_{owner.tenant_id}_{uuid4().hex[:8]}",
        agent_type="...ObjectiveGuardAgent",
        tenant_id=owner.tenant_id,
        bound_pages=[],  # Objective guard agent doesn't need pages
    )],
    capability_types=[ObjectiveGuardCapability],
    return_handles=True,
)[0]

# Get capability and communicate
guard = handle.get_capability(ObjectiveGuardCapability)
await guard.stream_events_to_queue(self.get_event_queue())

# Or send request and wait
await guard.check_alignment(goal_id="...", output={...})
future = await guard.get_result_future()
result = await future.wait(timeout=30.0)
```
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4
from overrides import override

from ...base import (
    Agent,
    AgentHandle,
)
from ..capabilities.goal_alignment import (
    ObjectiveGuardCapability,
    GoalAlignment,
)

from ..actions.policies import (
    CacheAwareActionPolicy,
)
from ...models import AgentSpawnSpec, AgentMetadata

logger = logging.getLogger(__name__)


class ObjectiveGuardAgent(Agent):
    """Meta-agent that prevents goal drift.

    Responsibilities:
    - Track original goals (JointGoal)
    - Monitor intermediate results
    - Detect drift from objectives
    - Challenge off-topic work
    - Trigger re-planning when needed

    Uses:
    - LLM to compare outputs with goals
    - Challenge messages to flag drift
    - Re-planning requests when drift severe
    """

    async def initialize(self) -> None:
        """Initialize objective guard agent."""
        await super().initialize()

        self.action_policy = CacheAwareActionPolicy(self)
        await self.action_policy.initialize()

        # Get or create objective guard capability
        if not self.has_capability(ObjectiveGuardCapability.get_capability_name()):
            guard_capability = ObjectiveGuardCapability(self)
            await guard_capability.initialize()
            self.add_capability(guard_capability)

        self.action_policy.use_agent_capabilities([ObjectiveGuardCapability.get_capability_name()])

        logger.info(f"ObjectiveGuardAgent {self.agent_id} initialized")




# ============================================================================
# Utility Functions
# ============================================================================


def detect_goal_drift(
    alignments: list[GoalAlignment],
    drift_threshold: float = 0.6
) -> list[GoalAlignment]:
    """Detect which results have goal drift.

    Args:
        alignments: Alignment assessments
        drift_threshold: Threshold below which drift is detected

    Returns:
        List of drifted alignments
    """
    return [
        a for a in alignments
        if a.drift_detected or a.alignment_score < drift_threshold
    ]


async def spawn_objective_guard_agent(
    owner: Agent,
) -> AgentHandle:
    """Spawn an ObjectiveGuardAgent and return a handle for communication.

    Example usage:
    ```python
    handle = await spawn_objective_guard_agent(owner=self.agent)
    guard = handle.get_capability(ObjectiveGuardCapability)

    # Register goal and check alignment
    await guard.check_alignment(goal_id="goal_1", output=my_result)
    future = await guard.get_result_future()
    alignment = await future.wait(timeout=30.0)
    ```

    Args:
        owner: Agent spawning the objective guard agent

    Returns:
        AgentHandle for interacting with the objective guard agent
    """
    agent_id = f"objective_guard_{owner.tenant_id}_{uuid4().hex[:8]}"
    logger.info(f"Spawning ObjectiveGuardAgent {agent_id}...")

    return await owner.spawn_child_agents(
        agent_specs=[AgentSpawnSpec(
            agent_id=agent_id,
            agent_type="polymathera.colony.agents.patterns.meta_agents.goal_alignment.ObjectiveGuardAgent",
            # agent_type="polymathera.colony.agents.base.Agent",
            # action_policy="polymathera.colony.agents.patterns.meta_agents.goal_alignment.ObjectiveGuardPolicy",
            bound_pages=owner.bound_pages,
            capability_types=[ObjectiveGuardCapability],
        )],
        soft_affinity=False,
        return_handles=True,
    )[0]


async def check_goal_alignment(
    owner: Agent,
    goal_id: str,
    output: Any,
    timeout: float = 30.0,
) -> GoalAlignment | None:
    """Check goal alignment using the AgentHandle pattern.

    High-level utility that spawns an ObjectiveGuardAgent, sends a request,
    and waits for the result.

    Args:
        owner: Agent requesting alignment check
        goal_id: ID of goal to check against
        output: Output to check for alignment
        timeout: Timeout in seconds

    Returns:
        GoalAlignment or None on timeout

    Example:
        ```python
        alignment = await check_goal_alignment(
            owner=self.agent,
            goal_id="my_goal",
            output=my_result,
            timeout=30.0,
        )
        if alignment and alignment.is_aligned:
            print("Output is aligned with goal!")
        else:
            print(f"Drift detected: {alignment.drift_description}")
        ```
    """
    logger.info(f"Checking goal alignment for goal: {goal_id}...")

    handle = await spawn_objective_guard_agent(owner=owner)

    try:
        guard = handle.get_capability(ObjectiveGuardCapability)

        # Send alignment check request
        await guard.check_alignment(goal_id=goal_id, output=output)

        # Wait for result
        future = await guard.get_result_future()
        result_data = await future.wait(timeout=timeout)

        if result_data:
            return GoalAlignment(**result_data)
        return None

    finally:
        await handle.stop()




