"""Change Impact Analysis - Analyze ripple effects of code changes.

Qualitative LLM-based change impact analysis that predicts and traces
the effects of code modifications across the codebase, including
dependencies, tests, documentation, and system behavior.

Traditional Approach:
- Call graph analysis
- Dependency graph traversal  
- Test coverage mapping
- Static data flow analysis

LLM Approach:
- Semantic understanding of change intent
- Behavioral impact reasoning
- Risk assessment of changes
- Indirect impact identification
- Change categorization and prioritization

Architecture:

1. **ChangeImpactAnalysisPolicy** (Page-Level Analysis):
   - Analyzes impact for a SINGLE VCM page only
   - Used by ChangeImpactAnalysisAgent instances
   - Reports directly impacted components in the page
   - Returns local impact assessment with ScopeAwareResult

2. **ChangeImpactAnalysisAgent** (Page Agent):
   - Bound to ONE VCM page
   - Uses policy to analyze page
   - Reports results to coordinator
   - Participates in team activities

3. **ChangeImpactAnalysisCoordinator** (Team Orchestrator):
   - Spawns ONE agent PER page (up to max_agents limit)
   - Checks blackboard for dependency graphs (spawns analysis if missing)
   - Collects local impact results from all page agents
   - Performs multi-hop propagation using dependency graph
   - Builds global impact graph and unified report
   - Merges results using ImpactMergePolicy with game-theoretic conflict resolution
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from overrides import override

from polymathera.colony.agents.patterns import (
    AnalysisScope,
    ScopeAwareResult,
    RelationshipGraph,
)
from polymathera.colony.agents.patterns.capabilities.critique import CriticCapability
from polymathera.colony.agents.blackboard import EnhancedBlackboard, CausalityTimeline, BlackboardEvent
from polymathera.colony.agents.base import Agent, AgentCapability
from polymathera.colony.agents.blackboard.protocol import DependencyQueryProtocol, ImpactAnalysisProtocol
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.agents.patterns.actions.policies import action_executor
from polymathera.colony.agents.patterns.capabilities.reflection import ReflectionCapability
from polymathera.colony.agents.patterns.events import event_handler, EventProcessingResult
from polymathera.colony.agents.models import (
    ActionResult,
    ActionType,
    PolicyREPL,
    AgentSuspensionState,
)
from polymathera.colony.agents.patterns.games.hypothesis.agents import HypothesisGameAgent

from .types import (
    CodeChange,
    ChangeType,
    ImpactType,
    ImpactStep,
    ImpactSeverity,
    ChangeImpactResult,
    ChangeImpactReport,
    ImpactedComponent,
    ImpactPath
)
from .predictor import FeedbackLoopPredictor

logger = logging.getLogger(__name__)



# ============================================================================
# REASONING LOOP COMPONENTS - Executor, Reflector, Critic for page agents
# ============================================================================

# Available action executors on ChangeImpactAnalysisCapability:
# - analyze_page(): Main analysis entry point
# - execute_analyze_impact(page_id, changes, test_coverage): ActionType.ANALYZE
# - execute_query(query, context_pages): ActionType.QUERY
# - execute_custom(custom_type, page_ids): ActionType.CUSTOM


class ChangeImpactAnalysisPolicy:
    """Policy for change impact analysis using LLM reasoning.

    This policy uses an LLM to:
    1. Understand the semantic intent of changes
    2. Identify directly impacted components
    3. Trace indirect impacts through dependencies
    4. Assess risk and severity
    5. Recommend mitigation strategies
    6. Identify required test updates

    Integration:
    - Uses RelationshipGraph for dependency tracking
    - Works with program slicing for precise impact
    - Integrates with test coverage data
    - Uses CausalityTimeline for temporal impacts
    """

    def __init__(
        self,
        agent: Agent,
        blackboard: EnhancedBlackboard,
        namespace: str = "impact_analysis",
        max_depth: int = 5,
        include_tests: bool = True,
        include_docs: bool = True
    ):
        """Initialize impact analysis policy.

        Args:
            agent: Agent instance for LLM inference via VCM
            blackboard: Blackboard for storing results
            namespace: Namespace for scoping keys in the blackboard
            max_depth: Maximum depth for impact propagation
            include_tests: Whether to analyze test impact
            include_docs: Whether to analyze documentation impact
        """
        self.agent = agent
        self.blackboard = blackboard
        self.namespace = namespace
        self.max_depth = max_depth
        self.include_tests = include_tests
        self.include_docs = include_docs

        # Initialize causality timeline for temporal analysis
        self.timeline = CausalityTimeline(
            blackboard=blackboard,
            namespace=namespace
        )

    # NOTE: Multi-page analysis is handled by ChangeImpactAnalysisCoordinator
    # This policy ONLY handles single-page analysis

    async def analyze_single_page(
        self,
        page_id: str,
        changes: list[CodeChange],
        test_coverage: dict[str, Any] | None = None
    ) -> ChangeImpactResult:
        """Analyze impact for a SINGLE VCM page.

        This method performs LOCAL impact analysis only. Multi-hop propagation
        and global graph building happen at the coordinator level.

        Args:
            page_id: Single VCM page ID to analyze
            changes: Changes to analyze (relevant to this page or global changes)
            test_coverage: Optional test coverage data for this page's code

        Returns:
            Local impact analysis result for this page
        """
        # Build prompt for LOCAL impact analysis
        prompt = self._build_page_impact_prompt(changes, test_coverage)

        # Get LLM analysis for single page with structured output
        response = await self.agent.infer(
            context_page_ids=[page_id],  # Single page only
            prompt=prompt,
            temperature=0.2,
            max_tokens=3000,
            json_schema=ChangeImpactReport.model_json_schema()
        )

        # Parse structured response
        # TODO: Handle validation errors. LLMs are not perfect.
        report = ChangeImpactReport.model_validate_json(response.generated_text)

        # Set source page ID for tracking
        report.source_page_id = page_id

        # Enrich report with computed analyses (local only)
        report = await self._enrich_report(report, test_coverage)

        # Build result with scope
        scope = AnalysisScope(
            is_complete=self._check_completeness(report),
            missing_context=self._identify_missing_context(report),
            confidence=self._compute_confidence(report),
            reasoning=[self._summarize_impact(report)]
        )

        return ChangeImpactResult(
            content=report,
            scope=scope
        )

    def _build_page_impact_prompt(
        self,
        changes: list[CodeChange],
        test_coverage: dict[str, Any] | None
    ) -> str:
        """Build prompt for LOCAL page-level impact analysis.

        NOTE: This analyzes impact within the page only. Multi-hop propagation
        across pages is done by the coordinator using the global dependency graph.

        Args:
            changes: Changes to analyze
            test_coverage: Optional test coverage data for this page

        Returns:
            Formatted prompt for LOCAL impact analysis
        """
        changes_desc = "\n".join([
            f"- {c.file_path}:{c.line_range[0]}-{c.line_range[1]}: {c.description} ({c.change_type.value})"
            for c in changes
        ])

        return f"""Analyze the impact of the following code changes on the code loaded in context.

Changes:
{changes_desc}

For each change, identify:
1. DIRECTLY IMPACTED components (functions, classes, modules that directly use the changed code)
2. IMPACT TYPES for each (functional, performance, security, API, data, compatibility, test, documentation)
3. SEVERITY of each impact (critical, high, medium, low, minimal)
4. WHETHER component needs updating
5. EVIDENCE supporting the impact assessment
6. INDIRECTLY IMPACTED components (through dependencies)
7. IMPACT PATHS showing how impact propagates
8. BREAKING CHANGES that affect external interfaces
9. TEST IMPACTS - which tests need updating
10. RISKS associated with the changes
11. RECOMMENDATIONS for safe deployment

For each impacted component, provide:
COMPONENT: [name and type]
FILE: [file path if different]
IMPACT_TYPES: [comma-separated types]
SEVERITY: [severity level]
NEEDS_UPDATE: [yes/no]
REASON: [why it's impacted]
EVIDENCE: [specific code or relationship that causes impact]
Return a JSON object matching the ChangeImpactReport schema.

Focus on semantic and behavioral impacts, not just syntactic dependencies."""

    # NOTE: Multi-hop impact propagation is done by the coordinator using
    # _propagate_impact_multi_hop(), not by this policy. This policy only
    # handles single-page LOCAL impact analysis.

    async def _enrich_report(
        self,
        report: ChangeImpactReport,
        test_coverage: dict[str, Any] | None
    ) -> ChangeImpactReport:
        """Enrich LLM-generated LOCAL report with computed analyses.

        NOTE: Multi-hop propagation requires dependency graph, which is done
        at coordinator level, NOT here.

        Args:
            report: LLM-generated local impact report
            test_coverage: Optional test coverage data

        Returns:
            Enriched local impact report
        """
        # Build impact graph if not provided by LLM
        if not report.impact_graph and (report.impacted_components or report.impact_paths):
            report.impact_graph = await self._build_impact_graph(
                report.impacted_components,
                report.impact_paths
            )

        # Compute risk assessment if not provided or incomplete
        if not report.risk_assessment or "risk_score" not in report.risk_assessment:
            report.risk_assessment = self._assess_risk(
                report.impacted_components,
                report.impact_paths
            )

        # Analyze test impact if coverage data available
        if test_coverage and self.include_tests and not report.test_impact:
            report.test_impact = await self._analyze_test_impact(
                report.changes,
                test_coverage
            )

        # Identify breaking changes if not provided
        if not report.breaking_changes:
            report.breaking_changes = self._identify_breaking_changes(
                report.impacted_components
            )

        # Generate recommendations if not provided
        if not report.recommendations:
            report.recommendations = await self._generate_recommendations(
                report.changes,
                report.impacted_components,
                report.risk_assessment
            )

        # Build timeline if not provided
        if not report.timeline:
            report.timeline = await self._build_impact_timeline_for_report(
                report.changes,
                report.impacted_components
            )

        return report

    # ============================================================================
    # HELPER METHODS - Called by _enrich_report() to post-process LLM output
    # ============================================================================

    def _propagate_severity(self, severity: ImpactSeverity) -> ImpactSeverity:
        """Propagate severity through dependencies.

        Args:
            severity: Original severity

        Returns:
            Propagated severity (usually reduced)
        """
        propagation = {
            ImpactSeverity.CRITICAL: ImpactSeverity.HIGH,
            ImpactSeverity.HIGH: ImpactSeverity.MEDIUM,
            ImpactSeverity.MEDIUM: ImpactSeverity.LOW,
            ImpactSeverity.LOW: ImpactSeverity.MINIMAL,
            ImpactSeverity.MINIMAL: ImpactSeverity.MINIMAL
        }
        return propagation.get(severity, ImpactSeverity.MINIMAL)

    async def _build_impact_graph(
        self,
        impacts: list[ImpactedComponent],
        paths: list[ImpactPath]
    ) -> RelationshipGraph:
        """Build graph of impact relationships.

        Args:
            impacts: All impacts
            paths: Impact paths

        Returns:
            Impact relationship graph
        """
        graph = RelationshipGraph()

        # Add impact nodes
        for impact in impacts:
            graph.add_node(
                node_id=impact.component_id,
                node_type="component",
                metadata={
                    "severity": impact.severity.value,
                    "impact_types": [t.value for t in impact.impact_types],
                    "requires_update": impact.requires_update
                }
            )

        # Add path edges
        for path in paths:
            for step in path.steps:
                graph.add_edge(
                    source_id=step.from_component,
                    target_id=step.to_component,
                    relationship_type=step.relationship,
                    metadata={"reason": step.impact_reason}
                )

        return graph

    async def _analyze_test_impact(
        self,
        changes: list[CodeChange],
        test_coverage: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Analyze impact on tests.

        Args:
            changes: Code changes
            test_coverage: Test coverage data

        Returns:
            Tests impacted by each change
        """
        test_impact = {}

        for change in changes:
            impacted_tests = []

            # Find tests covering changed lines
            file_coverage = test_coverage.get(change.file_path, {})
            for test_name, covered_lines in file_coverage.items():
                if isinstance(covered_lines, list):
                    # Check if test covers changed lines
                    change_lines = set(range(change.line_range[0], change.line_range[1] + 1))
                    if change_lines.intersection(covered_lines):
                        impacted_tests.append(test_name)

            test_impact[change.change_id] = impacted_tests

        return test_impact

    def _assess_risk(
        self,
        impacts: list[ImpactedComponent],
        paths: list[ImpactPath]
    ) -> dict[str, Any]:
        """Assess overall risk of changes.

        Args:
            impacts: All impacts
            paths: Impact paths

        Returns:
            Risk assessment
        """
        risk_score = 0.0
        severity_weights = {
            ImpactSeverity.CRITICAL: 10.0,
            ImpactSeverity.HIGH: 5.0,
            ImpactSeverity.MEDIUM: 2.0,
            ImpactSeverity.LOW: 1.0,
            ImpactSeverity.MINIMAL: 0.2
        }

        # Score direct impacts
        for impact in impacts:
            weight = severity_weights.get(impact.severity, 0)
            risk_score += weight * impact.confidence

        # Score indirect impacts (reduced weight)
        for path in paths:
            weight = severity_weights.get(path.total_severity, 0) * 0.5
            risk_score += weight

        # Determine risk level
        risk_level = "low"
        if risk_score > 50:
            risk_level = "very_high"
        elif risk_score > 30:
            risk_level = "high"
        elif risk_score > 15:
            risk_level = "medium"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "critical_impacts": sum(1 for i in impacts if i.severity == ImpactSeverity.CRITICAL),
            "high_impacts": sum(1 for i in impacts if i.severity == ImpactSeverity.HIGH),
            "components_requiring_update": sum(1 for i in impacts if i.requires_update),
            "indirect_impact_paths": len(paths)
        }

    def _identify_breaking_changes(
        self,
        impacts: list[ImpactedComponent]
    ) -> list[str]:
        """Identify breaking changes.

        Args:
            impacts: All impacts

        Returns:
            List of breaking changes
        """
        breaking = []

        for impact in impacts:
            # API changes are often breaking
            if ImpactType.API in impact.impact_types and impact.severity in [ImpactSeverity.CRITICAL, ImpactSeverity.HIGH]:
                breaking.append(f"API change in {impact.component_id}: {impact.description}")

            # Data format changes can be breaking
            if ImpactType.DATA in impact.impact_types and impact.severity == ImpactSeverity.CRITICAL:
                breaking.append(f"Data format change in {impact.component_id}: {impact.description}")

            # Compatibility issues are breaking by definition
            if ImpactType.COMPATIBILITY in impact.impact_types:
                breaking.append(f"Compatibility issue in {impact.component_id}: {impact.description}")

        return breaking

    async def _generate_recommendations(
        self,
        changes: list[CodeChange],
        impacts: list[ImpactedComponent],
        risk_assessment: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for handling impact.

        Args:
            changes: Code changes
            impacts: All impacts
            risk_assessment: Risk analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Risk-based recommendations
        risk_level = risk_assessment.get("risk_level", "low")
        if risk_level in ["high", "very_high"]:
            recommendations.append("Consider splitting changes into smaller, incremental updates")
            recommendations.append("Increase test coverage for impacted components")
            recommendations.append("Plan phased rollout with monitoring")

        # Update recommendations
        components_to_update = [i for i in impacts if i.requires_update]
        if components_to_update:
            recommendations.append(f"Update {len(components_to_update)} impacted components")
            for comp in components_to_update[:3]:  # First 3
                recommendations.append(f"  - Update {comp.component_id}: {comp.description}")

        # Test recommendations
        if any(ImpactType.TEST in i.impact_types for i in impacts):
            recommendations.append("Update test suite to cover changed behavior")

        # Documentation recommendations
        if any(ImpactType.DOCUMENTATION in i.impact_types for i in impacts):
            recommendations.append("Update documentation to reflect changes")

        # Breaking change recommendations
        if risk_assessment.get("critical_impacts", 0) > 0:
            recommendations.append("Document breaking changes in release notes")
            recommendations.append("Consider providing migration guide")

        return recommendations

    async def _build_impact_timeline_for_report(
        self,
        changes: list[CodeChange],
        impacts: list[ImpactedComponent]
    ) -> CausalityTimeline:
        """Build timeline of cascading impacts for a specific report.

        Args:
            changes: Code changes
            impacts: All impacts

        Returns:
            Causality timeline (new instance per report)
        """
        # Create a NEW timeline for this report, not the shared one
        timeline = CausalityTimeline(
            blackboard=self.blackboard,
            namespace=f"impact_analysis_{uuid.uuid4().hex[:8]}"
        )

        # Add change events
        for change in changes:
            await timeline.add_event(
                event_id=change.change_id,
                thread_id="main",
                event_type="change",
                timestamp=change.timestamp,
                metadata={
                    "file": change.file_path,
                    "type": change.change_type.value
                }
            )

        # Add impact events
        for i, impact in enumerate(impacts):
            await timeline.add_event(
                event_id=f"impact_{i}",
                thread_id=impact.component_id,
                event_type="impact",
                timestamp=time.time() + i * 0.1,  # Simulated cascade
                metadata={
                    "severity": impact.severity.value,
                    "types": [t.value for t in impact.impact_types]
                }
            )

        # TODO: Add causal relationships between changes and impacts

        return timeline

    def _check_completeness(self, report: ChangeImpactReport) -> bool:
        """Check if impact analysis is complete.

        Args:
            report: Impact report

        Returns:
            True if complete
        """
        # Check if we analyzed all provided changes
        return len(report.changes) > 0 and len(report.impacted_components) > 0

    def _identify_missing_context(self, report: ChangeImpactReport) -> list[str]:
        """Identify missing context.

        Args:
            report: Impact report

        Returns:
            List of missing elements
        """
        missing = []

        # Check for low confidence impacts
        for impact in report.impacted_components:
            if impact.confidence < 0.5:
                missing.append(f"Low confidence impact: {impact.component_id}")

        # Check for missing test data
        if not report.test_impact:
            missing.append("No test coverage data available")

        return missing

    def _compute_confidence(self, report: ChangeImpactReport) -> float:
        """Compute overall confidence.

        Args:
            report: Impact report

        Returns:
            Confidence score
        """
        if not report.impacted_components:
            return 0.5

        total_conf = sum(i.confidence for i in report.impacted_components)
        return total_conf / len(report.impacted_components)

    def _summarize_impact(self, report: ChangeImpactReport) -> str:
        """Summarize impact analysis.

        Args:
            report: Impact report

        Returns:
            Summary
        """
        risk_level = report.risk_assessment.get("risk_level", "unknown")

        severity_counts = {}
        for impact in report.impacted_components:
            severity_counts[impact.severity.value] = severity_counts.get(impact.severity.value, 0) + 1

        return (f"Analyzed {len(report.changes)} changes. "
                f"{len(report.impacted_components)} components impacted. "
                f"Risk level: {risk_level}. "
                f"Severity breakdown: {severity_counts}. "
                f"{len(report.breaking_changes)} breaking changes detected.")



# ============================================================================
# AGENT TEAM ARCHITECTURE
# ============================================================================

class ChangeImpactAnalysisCapability(AgentCapability):
    """Capability that analyzes change impact for a SINGLE VCM page.

    Keeps Agent subclasses thin by providing @action_executor methods for
    page analysis, prefetching, and hypothesis game participation.

    Also uses FeedbackLoopPredictor for cache-aware prefetching.
    """


    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "change_impact_analysis",
        input_patterns: list[str] = [DependencyQueryProtocol.query_pattern()],
        capability_key: str = "change_impact_analysis_capability"
    ):
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=input_patterns, capability_key=capability_key)

        # Analysis state
        self.page_id: str | None = None
        self.change_description: str | None = None
        self.changes: list[CodeChange] = []  # Actual changes to analyze
        self.blackboard: EnhancedBlackboard | None = None

        # Components
        self.feedback_predictor: FeedbackLoopPredictor | None = None

    def get_action_group_description(self) -> str:
        return (
            "Change Impact Analysis (page-level) — analyzes change impact on a single bound page. "
            "Uses FeedbackLoopPredictor for cache-aware prefetching of test page dependencies. "
            "Supports cross-page queries for impact tracing and hypothesis game participation for validation."
        )

    async def initialize(self) -> None:
        """Initialize capability."""
        await super().initialize()

        self.blackboard = await self.get_blackboard()

        # Load configuration from agent metadata
        self.page_id = self.agent.metadata.parameters.get("page_id")
        self.change_description = self.agent.metadata.parameters.get("change_description")
        changes_data = self.agent.metadata.parameters.get("changes", [])
        self.changes = [CodeChange(**c) if isinstance(c, dict) else c for c in changes_data]

        # Initialize feedback predictor for cache-aware prefetching
        self.feedback_predictor = FeedbackLoopPredictor(
            agent=self.agent,
            prefetch_depth=self.agent.metadata.parameters.get("prefetch_depth", 2),
            prefetch_test_pages=self.agent.metadata.parameters.get("prefetch_test_pages", True)
        )

        logger.info(f"ChangeImpactAnalysisCapability initialized for page {self.page_id}")

    # ============================================================================
    # ACTION EXECUTORS
    # ============================================================================

    @action_executor()
    async def analyze_page(self) -> ScopeAwareResult[ChangeImpactReport]:
        """Analyze change impact for this agent's assigned page
        until quality threshold is met or max iterations reached.

        Returns:
            Local impact report for this page
        """
        if not self.page_id:
            raise ValueError("No page ID assigned to this capability")

        # Prefetch pages that will be needed during self-critique
        await self._prefetch_for_self_critique()

        if not self.changes:
            logger.warning(f"No changes provided to capability in agent {self.agent_id}, analysis may be incomplete")

        # Get test coverage from blackboard if available
        test_coverage = await self._get_test_coverage_from_blackboard()

        # Execute analysis directly
        action_result = await self.execute_analyze_impact(
            page_id=self.page_id,
            changes=[c.model_dump() for c in self.changes],
            test_coverage=test_coverage
        )

        output = action_result.output
        if isinstance(output, dict):
            # Check if it has content/scope structure (ScopeAwareResult)
            if "content" in output and "scope" in output:
                report = ChangeImpactReport(**output["content"])
                scope = AnalysisScope(**output["scope"])
                result = ChangeImpactResult(content=report, scope=scope)
            else:
                # Assume output is the report content directly
                report = ChangeImpactReport(**output)
                report.source_page_id = self.page_id
                result = ChangeImpactResult(
                    content=report,
                    scope=AnalysisScope(confidence=0.7)
                )
        else:
            result = await self._build_fallback_result(self.changes, test_coverage)

        # Store in blackboard
        if self.blackboard:
            await self.blackboard.write(
                key=ImpactAnalysisProtocol.impact_key(self.page_id),
                value=result.model_dump(),
                tags={"impact", self.agent.agent_id}
            )

        return result

    @action_executor(action_key=ActionType.ANALYZE)
    async def execute_analyze_impact(
        self,
        page_id: str,
        changes: list[dict[str, Any]] | None = None,
        test_coverage: dict[str, Any] | None = None
    ) -> ActionResult:
        """Execute analyze action - run impact analysis on page.

        Args:
            page_id: VCM page ID to analyze
            changes: List of change dictionaries to analyze
            test_coverage: Optional test coverage data

        Returns:
            ActionResult with impact analysis output
        """
        if not page_id:
            return ActionResult(success=False, error="No page_id in parameters")

        changes = changes or []

        # Get blackboard from agent
        blackboard = await self.agent.get_agent_level_blackboard() # TODO: Which blackboard scope is best for analysis results? Shared across agents or private to this agent?

        # Create policy and analyze
        policy = ChangeImpactAnalysisPolicy(
            agent=self.agent,
            blackboard=blackboard
        )

        # Parse changes if they're dicts
        parsed_changes = []
        for c in changes:
            if isinstance(c, dict):
                parsed_changes.append(CodeChange(**c))
            else:
                parsed_changes.append(c)

        result = await policy.analyze_single_page(
            page_id=page_id,
            changes=parsed_changes,
            test_coverage=test_coverage
        )

        # Compute critique learnings from impact analysis result
        impacted_components = result.impacted_components if hasattr(result, 'impacted_components') else []
        low_confidence_impacts = [
            c for c in impacted_components
            if hasattr(c, 'confidence') and c.confidence < 0.5
        ]
        risk_level = (
            result.risk_assessment.get("risk_level", "unknown")
            if hasattr(result, 'risk_assessment') and result.risk_assessment
            else "unknown"
        )
        avg_confidence = (
            sum(c.confidence for c in impacted_components if hasattr(c, 'confidence')) / len(impacted_components)
            if impacted_components else 0.5
        )

        return ActionResult(
            success=True,
            output={
                **result.model_dump(),
                "_critique_learnings": {
                    "impacted_count": len(impacted_components),
                    "low_confidence_count": len(low_confidence_impacts),
                    "has_impacts": len(impacted_components) > 0,
                    "risk_level": risk_level,
                    "avg_confidence": avg_confidence,
                    "breaking_changes_count": len(result.breaking_changes) if hasattr(result, 'breaking_changes') else 0,
                    "coverage": 1.0 if impacted_components else 0.5,
                    "confidence": avg_confidence,
                },
            }
        )

    @action_executor(action_key="execute_query")
    async def execute_query(
        self,
        query: str,
        context_pages: list[str] | None = None
    ) -> ActionResult:
        """Execute query action - get additional context.

        Args:
            query: Query string to answer
            context_pages: Optional list of page IDs to use as context

        Returns:
            ActionResult with query response
        """
        context_pages = context_pages or []

        # Use agent inference to answer query
        response = await self.agent.infer(
            prompt=query,
            context_page_ids=context_pages
        )

        return ActionResult(
            success=True,
            output={"response": response.generated_text}
        )

    @action_executor(action_key="execute_custom")
    async def execute_custom(
        self,
        custom_type: str | None = None,
        page_ids: list[str] | None = None
    ) -> ActionResult:
        """Execute custom action.

        Args:
            custom_type: Type of custom action (e.g., "prefetch")
            page_ids: Optional list of page IDs for the action

        Returns:
            ActionResult with action output
        """
        if custom_type == "prefetch":
            # Prefetch pages for upcoming operations
            page_ids = page_ids or []
            for page_id in page_ids[:5]:  # Limit prefetch
                await self.agent.request_page(page_id, priority=1)
            return ActionResult(success=True, output={"prefetched": page_ids[:5]})

        return ActionResult(success=True, output={})

    async def _prefetch_for_self_critique(self) -> None:
        """Prefetch pages that will be needed during self-critique.

        Uses FeedbackLoopPredictor to predict pages needed.
        """
        if not self.feedback_predictor or not self.page_id:
            return

        pages_to_prefetch = self.feedback_predictor.predict_self_critique_pages(self.page_id)

        # TODO: Parallelize prefetching with asyncio.gather()
        # Request prefetch with low priority (background loading)
        for page_id in pages_to_prefetch:
            try:
                await self.agent.request_page(page_id, priority=1)
            except Exception as e:
                logger.debug(f"Failed to prefetch page {page_id}: {e}")

    async def _build_fallback_result(
        self,
        changes: list[CodeChange],
        test_coverage: dict[str, Any] | None
    ) -> ChangeImpactResult:
        """Build fallback result using policy directly (no reasoning loop).

        Used when reasoning loop fails or returns no result.

        Args:
            changes: Actual code changes to analyze
            test_coverage: Optional test coverage data

        Returns:
            Impact analysis result
        """
        policy = ChangeImpactAnalysisPolicy(
            agent=self.agent,
            blackboard=self.blackboard
        )

        return await policy.analyze_single_page(
            page_id=self.page_id,
            changes=changes,
            test_coverage=test_coverage
        )

    async def _get_test_coverage_from_blackboard(self) -> dict[str, Any] | None:
        """Get test coverage data for this page from blackboard.

        Returns:
            Test coverage data if available, None otherwise
        """
        if not self.blackboard:
            return None

        try:
            # Query blackboard for test coverage results
            coverage_data = await self.blackboard.read(
                key=ImpactAnalysisProtocol.test_coverage_key(self.page_id),
                agent_id=self.agent.agent_id
            )
            return coverage_data
        except Exception:
            # No coverage data available
            return None

    # ============================================================================
    # EVENT HANDLERS
    # ============================================================================

    @event_handler(pattern=DependencyQueryProtocol.query_pattern())
    async def on_dependency_query(
        self, event: BlackboardEvent, repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle dependency query from another agent."""
        # TODO: Turn event.value into a structured query object with validation
        query_component = event.value.get("component_id")
        from_agent = event.value.get("from_agent_id")

        impact_data = await self.blackboard.read(
            key=ImpactAnalysisProtocol.impact_key(self.page_id),
            agent_id=self.agent.agent_id
        )

        if impact_data:
            dependencies = []
            for component in impact_data.get("content", {}).get("impacted_components", []):
                if component.get("component_id") == query_component:
                    dependencies.append(component)

            if from_agent:
                # Parse query_id from the event key (colony-scoped protocol)
                parsed = DependencyQueryProtocol.parse_query_key(event.key)
                query_id = parsed.get("dependency_query", "")

                colony_blackboard = await self.agent.get_colony_level_blackboard()
                await colony_blackboard.write(
                    key=DependencyQueryProtocol.result_key(from_agent, query_id),
                    value={
                        "type": "dependency_response",
                        "query_component": query_component,
                        "dependencies": dependencies
                    }
                )
        return None

    # ============================================================================
    # SERIALIZATION
    # ============================================================================

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize capability-specific state."""
        state = await super().serialize_suspension_state(state)

        state.custom_data["_agent_capability_state"] = {
            "page_id": self.page_id,
            "change_description": self.change_description,
            "changes": [c.model_dump() for c in self.changes],
        }

        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore capability-specific state from suspension."""
        await super().deserialize_suspension_state(state)

        custom_state = state.custom_data.get("_agent_capability_state", {})
        if custom_state:
            self.page_id = custom_state.get("page_id")
            self.change_description = custom_state.get("change_description")
            changes_data = custom_state.get("changes", [])
            self.changes = [CodeChange(**c) for c in changes_data]


class ChangeImpactAnalysisAgent(HypothesisGameAgent):
    """Agent that analyzes change impact for a SINGLE VCM page.

    The actual analysis logic lives in ChangeImpactAnalysisCapability.
    Actions are invoked through the action routing system via @action_executor.
    Events are handled through @event_handler decorators on capabilities.

    NOTE: This agent extends HypothesisGameAgent to participate in
    hypothesis games for collaborative impact analysis.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        page_id: str | None = None,
        change_description: str | None = None
    ):
        """Initialize impact analysis agent for a SINGLE page.

        Args:
            agent_id: Unique agent ID
            page_id: Single VCM page ID to analyze
            change_description: Description of the change to analyze
        """
        agent_id = agent_id or f"impact_agent_{uuid.uuid4().hex[:8]}"
        super().__init__(
            agent_id=agent_id,
            agent_type="impact_analysis",
            bound_pages=[page_id] if page_id else []
        )
        # Store in metadata for capability to access
        if page_id:
            self.metadata["page_id"] = page_id
        if change_description:
            self.metadata["change_description"] = change_description

        self.analysis_capability: ChangeImpactAnalysisCapability | None = None

    async def initialize(self) -> None:
        """Initialize agent and attach capabilities."""
        await super().initialize()

        await self.action_policy.use_capability_blueprints([
            ReflectionCapability.bind(),
            CriticCapability.bind(),
            ChangeImpactAnalysisCapability.bind(),
        ])

        self.analysis_capability = self.get_capability(
            ChangeImpactAnalysisCapability.get_capability_name()
        )

        logger.info(f"ChangeImpactAnalysisAgent {self.agent_id} initialized")


