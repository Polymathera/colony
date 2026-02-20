"""Contract Inference - Infer function contracts and invariants.

Qualitative LLM-based contract inference that approximates formal
specification inference using natural language reasoning about
preconditions, postconditions, and invariants.

Traditional Approach:
- Symbolic execution to derive constraints
- Daikon-style dynamic invariant detection  
- Houdini/ICE learning from examples
- Interpolation-based synthesis

LLM Approach:
- Reason about function intent and requirements
- Identify implicit assumptions and guarantees
- Generate likely invariants from patterns
- Express contracts in natural language or formal spec
"""

from __future__ import annotations

import logging
import uuid
from pydantic import BaseModel, Field

from colony.agents.base import Agent
from colony.agents.patterns.capabilities.merge import MergeCapability
from colony.agents.patterns.capabilities.synthesis import SynthesisCapability
from colony.agents.patterns.games.hypothesis.capabilities import HypothesisGameProtocol

from .capabilities import (
    ContractMergePolicy,
    ContractInferenceCapability,
    ContractAnalysisCapability,  # New VCMAnalysisCapability-based coordinator
)
from .types import FormalismLevel


logger = logging.getLogger(__name__)


# ============================================================================
# AGENT TEAM ARCHITECTURE
# ============================================================================

class ContractInferenceAgent(Agent):
    """Agent that infers contracts from VCM pages.

    This agent is event-driven via AgentHandle.run(). It:
    1. Receives analysis requests via blackboard events
    2. Uses ContractInferenceCapability to analyze code
    3. Returns results via blackboard for AgentHandle.run() to receive
    4. Can participate in game protocols via HypothesisGameProtocol capability
    """
    formalism: str = Field(
        default=FormalismLevel.SEMI_FORMAL,
        description="Formalism level for contract inference"
    )

    async def initialize(self) -> None:
        """Initialize agent with capabilities configured."""
        self.add_capability_classes([
            ContractInferenceCapability,
            MergeCapability,
            HypothesisGameProtocol,
        ])
        await super().initialize()

        # Configure MergeCapability with ContractMergePolicy
        merge_cap: MergeCapability | None = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(ContractMergePolicy())

        # Configure ContractInferenceCapability
        inference_cap = self.get_capability_by_type(ContractInferenceCapability)
        if inference_cap:
            inference_cap.formalism = self.formalism

        logger.info(f"ContractInferenceAgent {self.agent_id} initialized")

    # NO run() method - agent is event-driven via blackboard
    # ContractInferenceCapability.handle_analysis_request() handles incoming requests


class ContractInferenceCoordinator(Agent):
    """Coordinator agent that orchestrates contract inference team.

    This agent uses ContractAnalysisCapability which provides composable primitives
    for distributed contract inference. The LLM planner decides the strategy:
    - spawn_worker/spawn_workers: Create worker agents
    - merge_results: Combine worker results
    - validate_critical_contracts: Domain-specific validation with hypothesis games
    - get_security_contracts: Query security-related contracts

    The coordinator is event-driven - no run() method needed.
    """

    async def initialize(self) -> None:
        """Initialize coordinator with capabilities configured."""
        self.add_capability_classes([
            ContractAnalysisCapability,
            MergeCapability,
            SynthesisCapability,
        ])
        await super().initialize()

        # Configure MergeCapability
        merge_cap: MergeCapability | None = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(ContractMergePolicy())

        # ContractAnalysisCapability configures its own merge policy via get_domain_merge_policy()
        # No need to set max_agents - LLM controls parallelism via spawn_workers(max_parallel=N)

        logger.info(f"ContractInferenceCoordinator {self.agent_id} initialized")
        # NO run() method - coordinator is event-driven via ContractAnalysisCapability

