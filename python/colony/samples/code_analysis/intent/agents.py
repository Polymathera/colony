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

from colony.agents.patterns.capabilities.merge import (
    MergeCapability,
)
from colony.agents.base import Agent
from colony.agents.patterns.capabilities.synthesis import SynthesisCapability
from colony.agents.patterns.games.consensus_game import ConsensusGameProtocol

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
        self.add_capability_classes([
            IntentInferenceCapability,
            MergeCapability,
            # TODO: Participate in consensus games for intent alignment
            ConsensusGameProtocol,
        ])
        await super().initialize()

        # Configure MergeCapability with IntentMergePolicy
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(IntentMergePolicy())

        # Configure IntentInferenceCapability
        intent_cap = self.get_capability_by_type(IntentInferenceCapability)
        if intent_cap:
            intent_cap.granularity = self.granularity

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
        self.add_capability_classes([
            IntentAnalysisCapability,
            MergeCapability,
            SynthesisCapability,
        ])
        await super().initialize()

        # Configure MergeCapability
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(IntentMergePolicy())

        # IntentAnalysisCapability configures its own merge policy via get_domain_merge_policy()
        # No need to set max_agents - LLM controls parallelism via spawn_workers(max_parallel=N)

        logger.info(f"IntentInferenceCoordinator {self.agent_id} initialized")
        # NO run() method - coordinator is event-driven via IntentAnalysisCapability
