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

from ....agents.patterns import MergeCapability
from ....agents.base import Agent
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

    def __init__(
        self,
        *args,
        page_id: str | None = None,
        compliance_types: list[ComplianceType] | None = None,
        **kwargs
    ):
        """Initialize compliance analysis agent.

        Args:
            *args: Passed to parent Agent
            page_id: VCM page ID to analyze
            compliance_types: Types of compliance to check
            **kwargs: Passed to parent Agent, including capability_classes
        """
        # Extract and set up capability classes
        capability_classes = kwargs.pop("capability_classes", [])
        if ComplianceAnalysisCapability not in capability_classes:
            capability_classes.append(ComplianceAnalysisCapability)
        if MergeCapability not in capability_classes:
            capability_classes.append(MergeCapability)
        kwargs["capability_classes"] = capability_classes

        # Set bound_pages from page_id
        if page_id and "bound_pages" not in kwargs:
            kwargs["bound_pages"] = [page_id]

        super().__init__(*args, **kwargs)
        self._page_id = page_id
        self._compliance_types = compliance_types or [ComplianceType.LICENSE, ComplianceType.SECURITY]

    async def initialize(self) -> None:
        """Initialize agent and configure capabilities."""
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

    def __init__(
        self,
        *args,
        max_agents: int = 10,
        **kwargs
    ):
        """Initialize coordinator.

        Args:
            *args: Passed to parent Agent
            max_agents: Maximum page agents (advisory - LLM controls actual parallelism)
            **kwargs: Passed to parent Agent, including capability_classes
        """
        # Extract and set up capability classes
        capability_classes = kwargs.pop("capability_classes", [])
        if ComplianceVCMCapability not in capability_classes:
            capability_classes.append(ComplianceVCMCapability)
        if MergeCapability not in capability_classes:
            capability_classes.append(MergeCapability)
        kwargs["capability_classes"] = capability_classes

        super().__init__(*args, **kwargs)
        # max_agents is advisory - LLM controls actual parallelism via spawn_workers(max_parallel=N)
        self._max_agents = max_agents

    async def initialize(self) -> None:
        """Initialize coordinator and configure capabilities."""
        await super().initialize()

        # Configure MergeCapability with ComplianceMergePolicy
        merge_cap = self.get_capability_by_type(MergeCapability)
        if merge_cap:
            merge_cap.set_policy(ComplianceMergePolicy())

        # ComplianceVCMCapability configures its own merge policy via get_domain_merge_policy()
        # No need to set max_agents - LLM controls parallelism via spawn_workers(max_parallel=N)

        logger.info(f"ComplianceAnalysisCoordinator {self.agent_id} initialized")
