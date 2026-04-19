"""Compliance Analysis - License, regulatory, and policy compliance checking.

Qualitative LLM-based compliance analysis that checks code against
licenses, regulations, organizational policies, and best practices.

Traditional Approach:
- License compatibility matrices
- Regex-based license detection
- Static rule checking
- Policy DSLs and validators

LLM Approach:
- Semantic understanding of license terms
- Context-aware compliance reasoning
- Policy intent interpretation
- Risk assessment and recommendations
- Cross-cutting compliance concerns
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from polymathera.colony.agents.patterns import MergeCapability
from polymathera.colony.agents.base import Agent
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from .types import ComplianceType
from .capabilities import (
    ComplianceVCMCapability,  # New VCMAnalysisCapability-based coordinator
    ComplianceAnalysisCapability,  # Worker capability for page-level analysis
    ComplianceMergePolicy,
)

logger = logging.getLogger(__name__)


# ============================================================================
# AGENT TEAM ARCHITECTURE
# ============================================================================

class ComplianceAnalysisAgent(Agent):
    """Agent that analyzes compliance for a single VCM page.

    This agent:
    1. Loads a single VCM page
    2. Analyzes compliance requirements via ComplianceAnalysisCapability
    3. Detects violations
    4. Reports findings via blackboard events
    5. Participates in consensus on ambiguous cases

    This agent is event-driven - it does NOT have a run() method.
    Work is triggered via blackboard events handled by capabilities.
    """
    page_id: str | None = Field(default=None)
    compliance_types: list[ComplianceType] | None = Field(default=[ComplianceType.LICENSE, ComplianceType.SECURITY])

    async def initialize(self) -> None:
        """Initialize agent and configure capabilities."""
        # Extract and set up capability classes
        self.add_capability_blueprints([
            ComplianceAnalysisCapability.bind(),
            MergeCapability.bind(
                merge_policy=ComplianceMergePolicy(),
            ),
        ])

        # Set bound_pages from page_id
        if self.page_id and not self.bound_pages:
            self.bound_pages = [self.page_id]

        await super().initialize()

        # Configure MergeCapability with ComplianceMergePolicy
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(ComplianceMergePolicy())

        logger.info(f"ComplianceAnalysisAgent {self.agent_id} initialized")


class ComplianceAnalysisCoordinator(Agent):
    """Coordinator for compliance analysis team.

    This agent uses ComplianceVCMCapability which provides composable primitives
    for distributed compliance analysis. The LLM planner decides the strategy:
    - spawn_worker/spawn_workers: Create worker agents
    - merge_results: Combine worker results
    - build_obligation_graph: Domain-specific obligation graph building
    - get_violations_by_severity: Query violations by severity level
    - get_risk_assessment: Get overall risk assessment
    - get_license_conflicts: Query license compatibility issues

    The coordinator is event-driven - no run() method needed.
    """

    async def initialize(self) -> None:
        """Initialize coordinator and configure capabilities."""
        self.add_capability_blueprints([
            ComplianceVCMCapability.bind(
                scope=BlackboardScope.AGENT,
                namespace="compliance_analysis_vcm",
            ),
            MergeCapability.bind(
                scope=BlackboardScope.COLONY,
                namespace="compliance_analysis_merge",
            )
        ])
        await super().initialize()

        # Configure MergeCapability with ComplianceMergePolicy
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(ComplianceMergePolicy())

        # ComplianceVCMCapability configures its own merge policy via get_domain_merge_policy()
        # No need to set max_agents - LLM controls parallelism via spawn_workers(max_parallel=N)

        logger.info(f"ComplianceAnalysisCoordinator {self.agent_id} initialized")
