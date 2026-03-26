"""Intent Inference - Understand code purpose and developer intentions.

Qualitative LLM-based intent inference that goes beyond what code does
to understand WHY it does it, extracting high-level goals, business logic,
and developer intentions.

Traditional Approach:
- Comment/documentation analysis
- Identifier name analysis  
- Pattern matching against known idioms
- Specification mining from usage

LLM Approach:
- Holistic reasoning about code purpose
- Business logic extraction
- Goal recognition from patterns
- Intent classification and explanation
- Misalignment detection (code vs intent)
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from polymathera.colony.agents.patterns.capabilities.merge import MergeCapability
from polymathera.colony.agents.base import Agent
from polymathera.colony.agents.patterns.capabilities.synthesis import SynthesisCapability
from polymathera.colony.agents.patterns.games.consensus_game import ConsensusGameProtocol

from .capabilities import (
    IntentInferenceCapability,
    IntentAnalysisCapability,  # New VCMAnalysisCapability-based coordinator
    IntentMergePolicy
)

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT TEAM ARCHITECTURE  
# ============================================================================

class IntentInferenceAgent(Agent):
    """Agent that infers intent from a single VCM page.

    This agent is event-driven via AgentHandle.run(). It:
    1. Receives analysis requests via blackboard events
    2. Analyzes code segments for intent via IntentInferenceCapability
    3. Builds intent graph for the page
    4. Returns result via blackboard for coordinator to receive

    Uses capability_classes pattern - no run() method needed.
    """
    granularity: str = Field(default="function")

    async def initialize(self) -> None:
        """Initialize agent with capabilities configured."""
        self.add_capability_blueprints([
            IntentInferenceCapability.bind(
                granularity=self.granularity,
            ),
            MergeCapability.bind(
                merge_policy=IntentMergePolicy(),
            ),
            # TODO: Participate in consensus games for intent alignment
            ConsensusGameProtocol.bind(),
        ])
        await super().initialize()

        logger.info(f"IntentInferenceAgent {self.agent_id} initialized")
        # NO run() method - agent is event-driven via IntentInferenceCapability



class IntentInferenceCoordinator(Agent):
    """Coordinator agent that orchestrates intent inference team.

    This agent uses IntentAnalysisCapability which provides composable primitives
    for distributed intent inference. The LLM planner decides the strategy:
    - spawn_worker/spawn_workers: Create worker agents
    - merge_results: Combine worker results
    - build_cross_page_hierarchies: Domain-specific hierarchy building
    - detect_misalignments: Find code-intent mismatches

    The coordinator is event-driven - no run() method needed.
    """

    async def initialize(self) -> None:
        """Initialize coordinator with capabilities configured."""
        self.add_capability_blueprints([
            IntentAnalysisCapability.bind(),
            MergeCapability.bind(
                merge_policy=IntentMergePolicy(),
            ),
            SynthesisCapability.bind(),
        ])
        await super().initialize()

        # IntentAnalysisCapability configures its own merge policy via get_domain_merge_policy()
        # No need to set max_agents - LLM controls parallelism via spawn_workers(max_parallel=N)

        logger.info(f"IntentInferenceCoordinator {self.agent_id} initialized")
        # NO run() method - coordinator is event-driven via IntentAnalysisCapability
