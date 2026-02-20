"""Program Slicing - Extract minimal code subset affecting a target.

Qualitative LLM-based program slicing that approximates traditional
static slicing techniques using natural language reasoning about
data and control dependencies.

Traditional Approach:
- Build PDG (Program Dependence Graph) 
- Compute backward/forward slices
- Track data and control dependencies

LLM Approach:
- Reason about variable flows qualitatively
- Identify "likely influences" on slicing criterion
- Generate approximate slices with confidence scores
"""

from __future__ import annotations

import logging

from colony.agents.patterns.capabilities.merge import MergeCapability
from colony.agents.base import Agent
from .capabilities import (
    SliceMergePolicy,
    ProgramSlicingCapability,
    SlicingAnalysisCapability,  # New VCMAnalysisCapability-based coordinator
)

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT TEAM ARCHITECTURE
# ============================================================================

class ProgramSlicingAgent(Agent):
    """Agent that computes program slices for a single VCM page.

    This agent is event-driven via AgentHandle.run(). It:
    1. Receives slicing requests via blackboard events
    2. Computes slices based on given criteria via ProgramSlicingCapability
    3. Tracks dependencies within the page
    4. Returns result via blackboard for coordinator to receive
    5. Collaborates on inter-page dependencies

    Uses capability_classes pattern - no run() method needed.
    """

    async def initialize(self) -> None:
        """Initialize agent with capabilities configured."""
        self.add_capability_classes([
            ProgramSlicingCapability,
            MergeCapability,
        ])
        await super().initialize()

        # Configure MergeCapability with SliceMergePolicy
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(SliceMergePolicy())

        logger.info(f"ProgramSlicingAgent {self.agent_id} initialized")
        # NO run() method - agent is event-driven via ProgramSlicingCapability


class ProgramSlicingCoordinator(Agent):
    """Coordinator agent that orchestrates program slicing team.

    This agent uses SlicingAnalysisCapability which provides composable primitives
    for distributed program slicing. The LLM planner decides the strategy:
    - spawn_worker/spawn_workers: Create worker agents
    - merge_results: Combine worker results
    - resolve_interprocedural: Domain-specific interprocedural resolution
    - get_external_dependencies: Query external dependencies

    The coordinator is event-driven - no run() method needed.
    """

    async def initialize(self) -> None:
        """Initialize coordinator with capabilities configured."""
        self.add_capability_classes([
            SlicingAnalysisCapability,
            MergeCapability,
        ])
        await super().initialize()

        # Configure MergeCapability
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(SliceMergePolicy())

        # SlicingAnalysisCapability configures its own merge policy via get_domain_merge_policy()
        # No need to set max_agents - LLM controls parallelism via spawn_workers(max_parallel=N)

        logger.info(f"ProgramSlicingCoordinator {self.agent_id} initialized")
        # NO run() method - coordinator is event-driven via SlicingAnalysisCapability
