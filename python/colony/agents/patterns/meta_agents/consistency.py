"""ConsistencyAgent: Detects cross-hypothesis contradictions.

The ConsistencyAgent is a meta-agent that:
- Monitors all hypotheses and claims
- Detects contradictions between results
- Uses ContradictionResolver to resolve conflicts
- Ensures logical consistency across the knowledge base

Role:
- Subscribes to hypothesis and finding events
- Compares new items against existing knowledge
- Flags contradictions for resolution
- Maintains consistency invariants

Programming Model (AgentHandle Pattern):
---------------------------------------
```python
# Spawn consistency agent with handle
handle = await owner.spawn_child_agents(
    agent_specs=[AgentSpawnSpec(agent_type="...ConsistencyAgent"),
    capability_types=[ConsistencyCapability],
    return_handles=True,
)[0]

# Get capability and communicate
consistency = handle.get_capability(ConsistencyCapability)
await consistency.stream_events_to_queue(self.get_event_queue())

# Wait for result
future = await consistency.get_result_future()
result = await future
```
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from ...models import AgentSpawnSpec, AgentMetadata
from ...base import (
    Agent,
    AgentHandle,
)
from ..capabilities.validation import ValidationCapability
from ..capabilities.consistency import ConsistencyCapability, ConsistencyCheck
from ..scope import ScopeAwareResult


logger = logging.getLogger(__name__)



class ConsistencyAgent(Agent):
    """Meta-agent for detecting and resolving contradictions.

    Responsibilities:
    - Monitor all analysis results and hypotheses
    - Detect contradictions using ContradictionResolver
    - Trigger resolution when conflicts found
    - Maintain epistemic consistency

    Uses:
    - ContradictionResolver for detection
    - EpistemicLayer for belief tracking
    - Blackboard events for monitoring
    """

    def __init__(self, *args, **kwargs):
        capability_classes: list[type] = kwargs.pop("capability_classes", [])
        if ConsistencyCapability not in capability_classes:
            capability_classes.append(ConsistencyCapability)
        if ValidationCapability not in capability_classes:
            capability_classes.append(ValidationCapability)
        kwargs["capability_classes"] = capability_classes
        super().__init__(*args, **kwargs)

    async def initialize(self) -> None:
        """Initialize consistency agent."""
        await super().initialize()

        logger.info(f"ConsistencyAgent {self.agent_id} initialized")


# ============================================================================
# Utility Functions
# ============================================================================


async def spawn_consistency_agent(
    owner: Agent,
    session_id: str | None = None,
    run_id: str | None = None,
) -> AgentHandle:
    """Spawn a ConsistencyAgent and return a handle for communication.

    Example usage:
    ```python
    handle = await spawn_consistency_agent(owner=self.agent)
    consistency = handle.get_capability(ConsistencyCapability)

    # Send request
    await consistency.check_result_consistency(new_result=result)

    # Wait for result
    future = await consistency.get_result_future()
    check_result = await future
    ```

    Args:
        owner: Agent spawning the consistency agent

    Returns:
        AgentHandle for interacting with the consistency agent
    """
    agent_id = f"consistency_agent_{owner.tenant_id}_{uuid4().hex[:8]}"
    logger.info(f"Spawning ConsistencyAgent {agent_id}...")

    return await owner.spawn_child_agents(
        agent_specs=[AgentSpawnSpec(
            agent_id=agent_id,
            agent_type="polymathera.colony.agents.patterns.meta_agents.consistency.ConsistencyAgent",
            # agent_type="polymathera.colony.agents.base.Agent",
            # action_policy="polymathera.colony.agents.patterns.meta_agents.consistency.ConsistencyAgentPolicy",
            tenant_id=owner.tenant_id,
            bound_pages=[],  # Consistency agent doesn't need pages
        )],
        run_id=run_id,
        session_id=session_id,
        context_page_source_config=owner.context_page_source.get_config(),
        capability_types=[ConsistencyCapability],
        soft_affinity=False,
        return_handles=True,
    )[0]


async def check_consistency(
    owner: Agent,
    new_result: ScopeAwareResult[Any],
    timeout: float = 30.0,
) -> ConsistencyCheck | None:
    """Check consistency of a result using the AgentHandle pattern.

    High-level utility that spawns a ConsistencyAgent, sends a request,
    and waits for the result.

    Args:
        owner: Agent requesting consistency check
        new_result: New result to check
        timeout: Timeout in seconds

    Returns:
        ConsistencyCheck or None on timeout

    Example:
        ```python
        result = await check_consistency(
            owner=self.agent,
            new_result=my_analysis_result,
            timeout=30.0,
        )
        if result and result.is_consistent:
            print("Result is consistent!")
        else:
            print(f"Contradictions: {result.contradictions}")
        ```
    """
    logger.info(f"Spawning ConsistencyAgent for result: {new_result.result_id}...")

    handle = await spawn_consistency_agent(owner=owner)

    try:
        consistency = handle.get_capability(ConsistencyCapability)

        # Send consistency check request
        await consistency.check_result_consistency(new_result=new_result)

        # Wait for result
        future = await consistency.get_result_future()
        result_data = await future.wait(timeout=timeout)

        if result_data:
            return ConsistencyCheck(**result_data)
        return None

    finally:
        await handle.stop()


