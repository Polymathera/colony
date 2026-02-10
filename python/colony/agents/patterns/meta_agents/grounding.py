"""GroundingAgent: Validates claims against evidence.

The GroundingAgent is a meta-agent that:
- Checks claims against external or retrieved context
- Uses PageQueryRoutingPolicy and IncrementalQueryProcessor to fetch evidence - TODO: Not implemented
- Validates that INFORM messages include proper evidence
- Challenges unsupported claims

Role in games:
- Hypothesis Game: Provides evidence during GROUND phase
- General validation: Checks any claim for grounding

Integration:
- Subscribes to INFORM messages in blackboard
- Uses existing query infrastructure for evidence retrieval
- Publishes validation results

Programming Model (AgentHandle Pattern):
---------------------------------------
The GroundingAgent demonstrates the `AgentHandle` pattern for parent-child
agent communication. When a parent spawns a GroundingAgent, it receives an
`AgentHandle` that provides three communication modes:

1. **Wait for result** - Block until grounding completes:
   ```python
   handle = await owner.spawn_child_agents(
       agent_specs=[AgentSpawnSpec(
           agent_id=f"grounding_agent_{owner.tenant_id}_{uuid.uuid4().hex[:8]}",
           agent_type="...GroundingAgent",
           bound_pages=owner.bound_pages,
           capability_types=[GroundingCapability],
       )],
       return_handles=True,
   )[0]
   grounding = handle.get_capability(GroundingCapability)
   future = await grounding.get_result_future()
   result = await future.wait(timeout=30.0)
   ```

2. **Stream events** - Receive async updates in event loop:
   ```python
   grounding = handle.get_capability(GroundingCapability)
   await grounding.stream_events_to_queue(self.get_event_queue())
   # Events now flow to parent's action policy
   ```

3. **Send requests** - Call capability methods:
   ```python
   await grounding.send_request(
       request_type="ground_claim",
       request_data={"claim": "...", "context": {...}}
   )
   ```
"""

from __future__ import annotations

import logging
from typing import Any
import uuid

from ...base import Agent, AgentHandle
from ...models import AgentSpawnSpec, AgentMetadata
from ..attention.incremental import IncrementalQueryCapability
from ..actions.policies import CacheAwareActionPolicy
from ..capabilities.grounding import (
    GroundingCapability,
    GroundingResult,
)

logger = logging.getLogger(__name__)



class GroundingAgent(Agent):

    async def initialize(self) -> None:
        """Initialize grounding agent."""
        await super().initialize()

        # Get query processor from metadata or create
        self.action_policy = CacheAwareActionPolicy(self)
        await self.action_policy.initialize()

        capability_types = [
            GroundingCapability,
            IncrementalQueryCapability,
        ]
        for capability_type in capability_types:
            if not self.has_capability(capability_type.get_capability_name()):
                capability = capability_type(self)
                await capability.initialize()
                self.add_capability(capability)
        await self.action_policy.use_agent_capability_types(capability_types)

        logger.info(f"GroundingAgent {self.agent_id} initialized")



# ============================================================================
# Utility Functions
# ============================================================================


async def spawn_grounding_agent(
    owner: Agent,
) -> AgentHandle:
    """Spawn a GroundingAgent and return a handle for communication.

    Example usage:

    1. **Stream events** to action policy:
       ```python
       handle = await spawn_grounding_agent(owner=self.agent)
       grounding = handle.get_capability(GroundingCapability)
       await grounding.stream_events_to_queue(self.get_event_queue())
       ```

    2. **Wait for result**:
       ```python
       handle = await spawn_grounding_agent(owner=self.agent)
       grounding = handle.get_capability(GroundingCapability)

       # Send request using capability method
       await grounding.ground_claim(claim="...", context={...})

       # Wait for result
       future = await grounding.get_result_future()
       result = await future.wait(timeout=30.0)
       ```

    Args:
        owner: Agent spawning the grounding agent

    Returns:
        AgentHandle for interacting with the grounding agent
    """
    # Create grounding agent
    agent_id = f"grounding_agent_{owner.tenant_id}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Spawning GroundingAgent {agent_id} for {owner.agent_id}...")

    return await owner.spawn_child_agents(
        agent_specs=[AgentSpawnSpec(
            agent_id=agent_id,
            agent_type="polymathera.colony.agents.patterns.meta_agents.grounding.GroundingAgent",
            # agent_type="polymathera.colony.agents.base.Agent",
            # action_policy="polymathera.colony.agents.patterns.meta_agents.grounding.GroundingAgentPolicy",
            tenant_id=owner.tenant_id,
            capability_types=[GroundingCapability],
            bound_pages=[],  # Grounding agent doesn't need pages
        )],
        run_id=run_id,
        session_id=session_id,
        soft_affinity=False,
        return_handles=True,
    )[0]


async def ground_claim(
    owner: Agent,
    claim: str,
    context: dict[str, Any] | None = None,
    initial_pages: list[str] | None = None,
    timeout: float = 30.0,
) -> GroundingResult | None:
    """Ground a claim using the AgentHandle pattern.

    High-level utility that spawns a GroundingAgent, sends a request,
    and waits for the result.

    Args:
        owner: Agent requesting grounding
        claim: Claim text to ground
        context: Optional context for grounding
        initial_pages: Optional initial pages for evidence search
        timeout: Timeout in seconds

    Returns:
        GroundingResult or None on timeout

    Example:
        ```python
        result = await ground_claim(
            owner=self.agent,
            claim="The function foo() is called in bar()",
            context={"file": "src/main.py"},
            timeout=30.0,
        )
        if result and result.is_grounded:
            print(f"Claim grounded with evidence: {result.evidence_found}")
        ```
    """
    logger.info(f"Spawning GroundingAgent for claim: {claim[:50]}...")

    # Spawn grounding agent with handle
    handle = await spawn_grounding_agent(owner=owner)

    try:
        # Get capability (scope_id = child_agent_id automatically)
        grounding = handle.get_capability(GroundingCapability)

        # Send grounding request using capability method
        await grounding.ground_claim(
            claim=claim,
            context=context,
            initial_pages=initial_pages,
        )

        # Wait for result
        future = await grounding.get_result_future()
        result_data = await future.wait(timeout=timeout)

        if result_data:
            return GroundingResult(**result_data)
        return None

    finally:
        # Clean up: stop the grounding agent
        await handle.stop()


