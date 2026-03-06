
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any
from overrides import override
import itertools


from colony.agents.patterns import (
    AnalysisScope,
    ScopeAwareResult,
    Relationship,
    RelationshipGraph,
    Hypothesis,
)
from colony.agents.patterns.capabilities.synthesis import SynthesisCapability
from colony.agents.patterns.capabilities.merge import MergeCapability, MergePolicy, MergeContext
from colony.agents.patterns.capabilities.validation import ValidationResult
from colony.agents.patterns.capabilities.critique import CriticCapability
from colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from colony.agents.blackboard import EnhancedBlackboard, CausalityTimeline, BlackboardEvent
from colony.agents.base import Agent, AgentCapability, AgentMetadata
from colony.agents.patterns.actions.policies import action_executor
from colony.agents.patterns.planning.policies import CacheAwarePlanningPolicy
from colony.agents.patterns.planning.strategies import ModelPredictiveControlStrategy
from colony.agents.patterns.games.negotiation.capabilities import NegotiationIssue, Offer, calculate_pareto_efficiency
from colony.agents.patterns.games.coalition_formation import find_optimal_coalition_structure
from colony.agents.patterns.games.hypothesis.capabilities import HypothesisGameProtocol
from colony.agents.patterns.events import event_handler, EventProcessingResult
from colony.agents.patterns.capabilities import WorkingSetCapability, AgentPoolCapability
from colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from colony.agents.patterns.capabilities.batching import (
    BatchingPolicy,
    ClusteringBatchPolicy,
    HybridBatchPolicy,
    ContinuousBatchPolicy,
)
from colony.agents.models import (
    Action,
    ActionType,
    AgentSuspensionState,
    RunContext,
    PolicyREPL,
)
from colony.cluster.models import LLMClientRequirements
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


class ImpactMergePolicy(MergePolicy[ChangeImpactReport]):
    """Policy for merging impact analysis results."""

    async def merge(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]],
        context: MergeContext
    ) -> ScopeAwareResult[ChangeImpactReport]:
        """Merge multiple impact reports.

        Args:
            results: Impact results to merge
            context: Merge context

        Returns:
            Merged impact report
        """
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            return results[0]

        coordination_notes: list[str] = []
        agent_weights = self._compute_agent_weights(results)

        all_changes = sum([r.content.changes for r in results], [])

        impact_groups: dict[tuple[str, str], list[tuple[ImpactedComponent, str]]] = {}
        for result in results:
            agent_id = result.producer_agent_id or result.result_id
            for impact in result.content.impacted_components:
                key = (impact.component_id, impact.file_path)
                impact_groups.setdefault(key, []).append((impact, agent_id))

        merged_impacts: list[ImpactedComponent] = []
        for key, entries in impact_groups.items():
            if len(entries) == 1:
                merged_impacts.append(entries[0][0])
            else:
                merged_impacts.append(
                    self._resolve_impact_conflict(
                        key,
                        entries,
                        agent_weights,
                        coordination_notes
                    )
                )

        all_paths = sum([r.content.impact_paths for r in results], [])

        all_test_impacts = {}
        for result in results:
            all_test_impacts.update(result.content.test_impact)

        all_breaking = list(set(sum([r.content.breaking_changes for r in results], [])))
        all_recommendations = list(set(sum([r.content.recommendations for r in results], [])))

        merged_report = ChangeImpactReport(
            changes=all_changes,
            impacted_components=merged_impacts,
            impact_paths=all_paths,
            test_impact=all_test_impacts,
            breaking_changes=all_breaking,
            recommendations=all_recommendations,
            risk_assessment=self._merge_risk_assessment(results)
        )

        merged_scope = AnalysisScope(
            is_complete=all(r.scope.is_complete for r in results),
            missing_context=list(set(sum([r.scope.missing_context for r in results], []))),
            confidence=sum(r.scope.confidence for r in results) / len(results),
            reasoning=sum([r.scope.reasoning for r in results], []) + coordination_notes
        )

        return ChangeImpactResult(
            content=merged_report,
            scope=merged_scope
        )

    def _merge_risk_assessment(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> dict[str, Any]:
        """Merge risk assessments.

        Args:
            results: Results to merge

        Returns:
            Combined risk assessment
        """
        total_score = sum(r.content.risk_assessment.get("risk_score", 0) for r in results)
        total_critical = sum(r.content.risk_assessment.get("critical_impacts", 0) for r in results)
        total_high = sum(r.content.risk_assessment.get("high_impacts", 0) for r in results)

        risk_level = "low"
        if total_score > 50:
            risk_level = "very_high"
        elif total_score > 30:
            risk_level = "high"
        elif total_score > 15:
            risk_level = "medium"

        return {
            "risk_score": total_score,
            "risk_level": risk_level,
            "critical_impacts": total_critical,
            "high_impacts": total_high
        }

    async def validate(
        self,
        original: list[ScopeAwareResult[ChangeImpactReport]],
        merged: ScopeAwareResult[ChangeImpactReport]
    ) -> ValidationResult:
        """Validate merged impact report.

        Args:
            original: Original results
            merged: Merged result

        Returns:
            Validation result
        """
        issues = []

        # Check no changes were lost
        original_changes = sum(len(r.content.changes) for r in original)
        if len(merged.content.changes) < original_changes:
            issues.append("Changes lost during merge")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if not issues else 0.8
        )

    def _resolve_impact_conflict(
        self,
        key: tuple[str, str],
        entries: list[tuple[ImpactedComponent, str]],
        agent_weights: dict[str, float],
        coordination_notes: list[str]
    ) -> ImpactedComponent:
        """Resolve conflicting impact assessments via negotiation utilities."""
        component_id, file_path = key
        parties = [agent_id for _, agent_id in entries]
        issue = NegotiationIssue(
            issue_id=f"impact::{component_id}",  # TODO: Add NegotiationIssue.get_issue_id(component_id, tenant_id) method?
            description=f"Resolve impact severity for {component_id}",
            parties=parties,
            preferences={
                agent_id: {"confidence": impact.confidence}
                for impact, agent_id in entries
            }
        )

        offers = []
        for idx, (impact, agent_id) in enumerate(entries):
            utility = impact.confidence * agent_weights.get(agent_id, 1.0)
            offers.append(
                Offer(
                    offer_id=f"{issue.issue_id}::{idx}",
                    proposer=agent_id,
                    terms={
                        "severity": impact.severity.value,
                        "impact_types": [t.value for t in impact.impact_types]
                    },
                    utility=utility,
                    justification=impact.description
                )
            )

        efficiency = calculate_pareto_efficiency(offers, issue)
        pareto_ids = set(efficiency.get("pareto_optimal_offers", [])) or {offer.offer_id for offer in offers}
        winning_offer = max(
            (offer for offer in offers if offer.offer_id in pareto_ids),
            key=lambda offer: offer.utility
        )

        selected_component = next(impact for impact, agent_id in entries if agent_id == winning_offer.proposer)
        merged_component = selected_component.model_copy(deep=True)

        # Enrich metadata with contributions
        merged_component.impact_types = sorted(
            {impact_type for impact, _ in entries for impact_type in impact.impact_types},
            key=lambda impact_type: impact_type.value
        )
        merged_component.evidence = sorted(
            set(sum((impact.evidence for impact, _ in entries), []))
        )
        merged_component.confidence = min(1.0, winning_offer.utility)
        merged_component.description = merged_component.description or selected_component.description

        coordination_notes.append(
            f"Negotiation reconciled impact on {component_id} ({file_path}); "
            f"severity '{merged_component.severity.value}' selected."
        )
        return merged_component

    def _compute_agent_weights(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> dict[str, float]:
        """Compute coalition weights for impact analysis contributors."""
        agents: list[str] = []
        capabilities: dict[str, set[str]] = {}
        confidences: dict[str, float] = {}

        for result in results:
            agent_id = result.producer_agent_id or result.result_id
            agents.append(agent_id)
            capabilities[agent_id] = self._extract_capabilities(result.content)
            confidences[agent_id] = result.scope.confidence or 0.5

        if not agents:
            return {}

        characteristic_function: dict[str, float] = {}
        for r in range(1, len(agents) + 1):
            for subset in itertools.combinations(agents, r):
                subset_key = ",".join(sorted(subset))
                combined_caps = set().union(*(capabilities[a] for a in subset))
                avg_conf = sum(confidences[a] for a in subset) / len(subset)
                characteristic_function[subset_key] = len(combined_caps) + avg_conf

        optimal_structure = find_optimal_coalition_structure(agents, characteristic_function)
        weights = {agent: 0.0 for agent in agents}
        for coalition in optimal_structure:
            key = ",".join(sorted(coalition))
            value = characteristic_function.get(key, 0.0)
            if not coalition:
                continue
            for agent in coalition:
                weights[agent] += value / len(coalition)

        total = sum(weights.values()) or len(weights)
        return {agent: weights[agent] / total for agent in agents}

    def _extract_capabilities(
        self,
        report: ChangeImpactReport
    ) -> set[str]:
        """Extract capability tags (impact coverage) for coalition math."""
        caps: set[str] = set()
        for impact in report.impacted_components:
            for impact_type in impact.impact_types:
                caps.add(impact_type.value)
            if impact.requires_update:
                caps.add("update_recommendations")
        if report.breaking_changes:
            caps.add("breaking_change_detection")
        if report.test_impact:
            caps.add("test_impact_analysis")
        return caps or {"baseline_impact"}


class ChangeImpactAnalysisCoordinatorCapability(AgentCapability):
    """Capability implementing core impact analysis coordination logic.

    Keeps Agent subclasses thin by providing @action_executor methods for
    spawning, monitoring, synthesis, and multi-hop propagation.

    This capability follows the patterns from CodeAnalysisCoordinatorV2:
    1. Uses `Planner` with `CacheAwarePlanningPolicy` for cache-optimized planning
    2. Uses `spawn_next_batch()` for cache-aware agent scheduling
    3. Uses `SynthesisCapability` for progressive result synthesis
    4. Uses `HypothesisGameProtocol` for validating CRITICAL impacts
    5. Uses `FeedbackLoopPredictor` for page prefetching during feedback loops

    Key difference from naive batching: Spawns pages by WORKING SET OVERLAP,
    not arbitrary partitioning.

    This coordinator:
    1. Spawns impact agents for each page
    2. Collects and aggregates impact reports
    3. Traces cross-page dependencies
    4. Builds complete impact graph
    5. Produces unified impact report with recommendations
    """

    def __init__(self, agent: Agent):
        """Initialize coordinator.

        Args:
            agent_id: Coordinator ID
            max_agents: Maximum concurrent page agents
        """
        super().__init__(agent=agent)

        self.page_agents: dict[str, str] = {}  # page_id -> agent_id
        self.blackboard: EnhancedBlackboard | None = None

        # Cache-aware components (initialized in initialize())
        self.cache_policy: CacheAwarePlanningPolicy | None = None
        self.working_set_cap: WorkingSetCapability | None = None
        self.feedback_predictor: FeedbackLoopPredictor | None = None
        self.incremental_synthesizer: SynthesisCapability | None = None

        # Tracking state
        self.pending_pages: list[str] = []  # Pages not yet spawned
        self.completed_results: list[ScopeAwareResult[ChangeImpactReport]] = []
        self.max_depth: int = 5  # Max depth for impact propagation
        self._current_changes: list[CodeChange] = []  # Changes being analyzed
        # Get configuration from agent metadata
        self.group_id = self.agent.metadata.group_id
        self.tenant_id = self.agent.metadata.tenant_id
        self.prefetch_depth = self.agent.metadata.parameters.get("prefetch_depth", 2)
        self.prefetch_test_pages = self.agent.metadata.parameters.get("prefetch_test_pages", True)
        self.max_agents = self.agent.metadata.parameters.get("max_agents", 10)

    def get_action_group_description(self) -> str:
        pages_info = f"{len(self.pending_pages)} pending pages" if self.pending_pages else "no pending pages"
        return (
            f"Change Impact Coordination ({pages_info}, max {self.max_agents} concurrent agents) — "
            "orchestrates distributed impact analysis. "
            "analyze_change_impact handles batching, agent spawning, result aggregation, and reporting. "
            "Traces cross-page dependencies (max_depth=5). "
            "Cache-aware batching via working set overlap. "
            "Validates CRITICAL impacts through hypothesis game protocol."
        )

    async def initialize(self) -> None:
        """Initialize capability with cache-aware components."""
        await super().initialize()

        self.blackboard = await self.get_blackboard()

        # Initialize cache-aware planning policy
        self.cache_policy = CacheAwarePlanningPolicy(
            agent=self.agent,
            cache_capacity=self.agent.metadata.parameters.get("cache_capacity", 50),
            query_vcm_state=self.agent.metadata.parameters.get("query_vcm_state", False)
        )
        await self.cache_policy.initialize()

        # Get WorkingSetCapability from agent (must be added in Agent.initialize())
        self.working_set_cap = self.agent.get_capability_by_type(WorkingSetCapability)

        # Initialize agent pool capability for lifecycle management
        self.agent_pool_cap: AgentPoolCapability | None = self.agent.get_capability_by_type(AgentPoolCapability)
        if not self.agent_pool_cap:
            self.agent_pool_cap = AgentPoolCapability(agent=self.agent, scope_id=self.scope_id)
            await self.agent_pool_cap.initialize()
            self.agent.add_capability(self.agent_pool_cap)

        # Initialize batching policy
        # NOTE: Impact analysis uses custom scoring that includes centrality weighting,
        # which is not supported by the standard BatchingPolicy. The policy is provided
        # for future integration but spawn_next_batch() currently uses custom logic.
        batching_policy_config = self.agent.metadata.parameters.get("batching_policy", {})
        self.batching_policy: BatchingPolicy = self._create_batching_policy(batching_policy_config)

        # Initialize page graph capability for standardized graph operations
        self.page_graph_cap: PageGraphCapability | None = self.agent.get_capability_by_type(PageGraphCapability)
        if not self.page_graph_cap:
            self.page_graph_cap = PageGraphCapability(agent=self.agent, scope_id=self.scope_id)
            await self.page_graph_cap.initialize()
            self.agent.add_capability(self.page_graph_cap)

        # Initialize feedback predictor
        self.feedback_predictor = FeedbackLoopPredictor(
            agent=self.agent,
            group_id=self.group_id,
            tenant_id=self.tenant_id,
            prefetch_depth=self.prefetch_depth,
            prefetch_test_pages=self.prefetch_test_pages
        )

        # TODO: Switch to SynthesisCapability
        # Initialize incremental synthesizer for progressive synthesis
        self.incremental_synthesizer = self.agent.get_capability_by_type(SynthesisCapability)
        if not self.incremental_synthesizer:
            raise RuntimeError(
                "SynthesisCapability must be added to agent for incremental synthesis"
            )

        logger.info(
            f"ChangeImpactAnalysisCoordinatorCapability initialized with cache-aware planning "
            f"(max_agents={self.max_agents})"
        )

    def _create_batching_policy(self, config: dict) -> BatchingPolicy:
        """Create batching policy from configuration.

        Args:
            config: Policy configuration dict with keys:
                - type: "hybrid" | "clustering" | "continuous"
                - overlap_threshold: float (for clustering)
                - batch_size: int (max batch size)
                - max_concurrent: int (for continuous)

        Returns:
            Configured BatchingPolicy instance
        """
        policy_type = config.get("type", "continuous")  # Default to continuous for impact analysis
        overlap_threshold = config.get("overlap_threshold", 0.3)
        batch_size = config.get("batch_size", self.max_agents)
        max_concurrent = config.get("max_concurrent", self.max_agents)

        if policy_type == "clustering":
            return ClusteringBatchPolicy(
                min_overlap=overlap_threshold,
                max_batch_size=batch_size,
            )
        elif policy_type == "continuous":
            return ContinuousBatchPolicy(max_concurrent=max_concurrent)
        else:  # hybrid
            return HybridBatchPolicy(
                clustering_policy=ClusteringBatchPolicy(
                    min_overlap=overlap_threshold,
                    max_batch_size=batch_size,
                ),
                continuous_policy=ContinuousBatchPolicy(max_concurrent=max_concurrent),
                overlap_threshold=overlap_threshold,
            )

    # ============================================================================
    # SERIALIZATION
    # ============================================================================

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize capability-specific state.

        Args:
            state: AgentSuspensionState with all agent state serialized
        """
        state: AgentSuspensionState = await super().serialize_suspension_state(state)

        state.custom_data["_coordinator_capability_state"] = {
            "page_agents": self.page_agents,
            "pending_pages": self.pending_pages,
            "max_depth": self.max_depth,
            "current_changes": [c.model_dump() for c in self._current_changes],
        }

        # WorkingSetCapability handles its own state via blackboard

        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore capability-specific state from suspension.

        Args:
            state: AgentSuspensionState to restore from
        """
        await super().deserialize_suspension_state(state)

        custom_state = state.custom_data.get("_coordinator_capability_state", {})
        if custom_state:
            self.page_agents = custom_state.get("page_agents", {})
            self.pending_pages = custom_state.get("pending_pages", [])
            self.max_depth = custom_state.get("max_depth", 5)
            changes_data = custom_state.get("current_changes", [])
            self._current_changes = [CodeChange(**c) for c in changes_data]

        # WorkingSetCapability handles its own state via blackboard

    def _summarize_changes(self, changes: list[CodeChange]) -> str:
        """Create human-readable summary of changes.

        Args:
            changes: List of code changes

        Returns:
            Summary string
        """
        if not changes:
            return "No changes specified"

        summaries = []
        for change in changes[:5]:  # Limit to first 5 for brevity - TODO: Make configurable.
            summaries.append(
                f"{change.change_type.value} in {change.file_path}: {change.description}"
            )

        if len(changes) > 5:  # TODO: Make configurable.
            summaries.append(f"... and {len(changes) - 5} more changes")

        return "; ".join(summaries)

    def _propagate_severity(self, severity: ImpactSeverity) -> ImpactSeverity:
        """Propagate severity through dependencies (reduced by one level).

        Args:
            severity: Original severity

        Returns:
            Propagated severity (reduced)
        """
        # TODO: What does this mean?
        propagation = {
            ImpactSeverity.CRITICAL: ImpactSeverity.HIGH,
            ImpactSeverity.HIGH: ImpactSeverity.MEDIUM,
            ImpactSeverity.MEDIUM: ImpactSeverity.LOW,
            ImpactSeverity.LOW: ImpactSeverity.MINIMAL,
            ImpactSeverity.MINIMAL: ImpactSeverity.MINIMAL
        }
        return propagation.get(severity, ImpactSeverity.MINIMAL)

    # ============================================================================
    # EVENT HANDLERS - Replace manual blackboard subscriptions
    # ============================================================================

    @event_handler(pattern="*:analysis_complete") # TODO: Use a more specific pattern to avoid conflicts (e.g., include scope_id or use a structured event type)
    async def on_child_complete(
        self, event: BlackboardEvent, repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle child agent completion via event."""
        agent_id = event.key.split(":")[0]
        page_id = None
        for pid, aid in self.page_agents.items():
            if aid == agent_id:
                page_id = pid
                break

        if not page_id:
            return None

        logger.info(f"Child {page_id} ({agent_id}) completed via event")
        result_data = event.value

        # Parse result
        result = ScopeAwareResult[ChangeImpactReport](**result_data["result"])
        self.completed_results.append(result)

        # Incrementally synthesize
        if self.incremental_synthesizer:
            await self.incremental_synthesizer.add_result(
                result_id=page_id,
                result=result
            )

        # Prefetch pages for cross-agent refinement
        await self._prefetch_for_cross_agent_feedback(result)

        # Remove from tracking
        del self.page_agents[page_id]

        # Update working set
        completed_pages = {page_id}
        if self.working_set_cap:
            await self.working_set_cap.release_pages(page_ids=list(completed_pages))

        # Spawn next batch if there are pending pages
        max_agents = self.agent.metadata.parameters.get("max_agents", 10)
        if self.pending_pages and len(self.page_agents) < max_agents:
            await self.spawn_next_batch(self._current_changes[0].description if self._current_changes else "")

        # Check if all done
        if not self.page_agents and not self.pending_pages:
            return EventProcessingResult(
                immediate_action=Action(action_type=ActionType.CUSTOM, parameters={"custom_type": "finalize_analysis"})
            )
        return None

    @event_handler(pattern="error:*") # TODO: Use a more specific pattern to avoid conflicts (e.g., include scope_id or use a structured event type)
    async def on_child_error(
        self, event: BlackboardEvent, repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle child agent error via event."""
        agent_id = event.key.split(":")[1]
        page_id = None
        for pid, aid in self.page_agents.items():
            if aid == agent_id:
                page_id = pid
                break

        if not page_id:
            return None

        error_data = event.value
        logger.warning(
            f"Child {agent_id} ({page_id}) escalated error: "
            f"{error_data.get('error_type', 'Unknown')}: "
            f"{error_data.get('error', 'No details')}"
        )

        # Simple retry policy
        retry_count = error_data.get("context", {}).get("retry_count", 0)
        if retry_count < 1:
            logger.info(f"Retrying child {agent_id}")
            await self.blackboard.delete(f"error:{agent_id}")
        else:
            # Max retries exceeded - remove from tracking
            logger.error(f"Child {agent_id} ({page_id}) failed after {retry_count} retries, skipping")
            del self.page_agents[page_id]

            if not self.page_agents and not self.pending_pages:
                return EventProcessingResult(
                    immediate_action=Action(action_type=ActionType.CUSTOM, parameters={"custom_type": "finalize_analysis"})
                )
        return None

    # ============================================================================
    # ACTION EXECUTORS - Main analysis workflow
    # ============================================================================

    @action_executor()
    async def analyze_change_impact(
        self,
        page_ids: list[str],
        changes: list[CodeChange],
        change_description: str | None = None
    ) -> ScopeAwareResult[ChangeImpactReport]:
        """Analyze change impact using cache-aware agent scheduling.
        
        NOTE: This action executor runs the entire analysis in one call.
        It is better for the action policy to break down large analyses
        into smaller actions to allow the planner to intervene and adapt
        execution to the context.

        Handles large codebases by spawning agents in batches.
        Uses the _spawn_next_batch() pattern from CodeAnalysisCoordinatorV2:
        1. Initialize working set from page graph (high-centrality pages first)
        2. Spawn agents for pages with highest working set overlap
        3. As agents complete, update working set and spawn next batch
        4. Incrementally synthesize results as they arrive
        5. Validate CRITICAL impacts using HypothesisGameProtocol

        Args:
            page_ids: ALL VCM page IDs to analyze
            changes: Actual CodeChange objects describing what changed
            change_description: Optional human-readable description (for logging)

        Returns:
            Unified impact report with multi-hop propagation
        """
        page_graph = await self.agent.load_page_graph()
        self._current_changes = changes
        change_description = change_description or self._summarize_changes(changes)
        # TODO: Write this summary to agent memory to be used by the action policy

        # Phase 1: Initialize working set from page graph
        run_id = self.agent.metadata.run_id
        await self.working_set_cap.initialize_from_policy(
            page_graph=page_graph,
            available_pages=page_ids,
            run_context=RunContext(
                goal=f"Impact analysis: {change_description}",
                change_description=change_description,
                run_id=run_id,  # TODO: Get it from session context
            )
        )

        # Phase 2: Get or compute dependency graph
        dependency_graph = await self._get_or_compute_dependency_graph(page_ids)

        # Phase 3: Set up pending pages and spawn first batch
        self.pending_pages = list(page_ids)
        self.completed_results = []

        # Spawn first batch based on working set overlap
        await self.spawn_next_batch(change_description)

        # Phase 4: Process results as they arrive (event-driven)
        # Continue until all pages processed
        while self.pending_pages or self.page_agents:
            # Collect results from current batch
            batch_results = await self._collect_results()

            for result in batch_results:
                # Incrementally synthesize each result
                if self.incremental_synthesizer:
                    # TODO: This does not extract the synthesized result.
                    result_page_id = self._get_page_id_from_result(result)
                    await self.incremental_synthesizer.add_result(
                        result_id=result_page_id,
                        result=result
                    )
                self.completed_results.append(result)

                # Prefetch pages for cross-agent refinement
                await self._prefetch_for_cross_agent_feedback(result)

            # Stop completed agents
            completed_agents = [
                agent_id for page_id, agent_id in self.page_agents.items()
                if any(
                    self._get_page_id_from_result(r) == page_id 
                    for r in batch_results
                )
            ]
            for agent_id in completed_agents:
                await self.agent.stop_child_agent(agent_id)

            # Remove completed from tracking
            for page_id in list(self.page_agents.keys()):
                if self.page_agents[page_id] in completed_agents:
                    del self.page_agents[page_id]

            # Update working set with completed pages
            completed_page_ids = {
                self._get_page_id_from_result(r) for r in batch_results
            }
            if completed_page_ids and self.working_set_cap:
                await self.working_set_cap.release_pages(page_ids=list(completed_page_ids))

            # Spawn next batch if there are pending pages
            if self.pending_pages and len(self.page_agents) < self.max_agents:
                await self.spawn_next_batch(change_description)

            # Break if nothing more to do
            if not self.page_agents and not self.pending_pages:
                break

            # Small sleep to avoid busy-waiting
            await asyncio.sleep(0.1)

        # Phase 5: Perform multi-hop propagation using dependency graph
        propagated_results = await self._propagate_impact_multi_hop(
            self.completed_results,
            dependency_graph
        )

        # Phase 6: Validate CRITICAL impacts using HypothesisGameProtocol
        validated_results = await self._validate_critical_impacts(propagated_results)

        # Phase 7: Use SynthesisCapability for final merge
        # Reset synthesizer and add all validated results for proper synthesis
        final_synthesizer = SynthesisCapability(
            agent=self.agent,
            scope_id=f"synthesis_final:{uuid.uuid4()}"
        )

        for result in validated_results:
            result_page_id = self._get_page_id_from_result(result)
            await final_synthesizer.add_result(
                result_id=f"final:{result_page_id}",
                result=result
            )

        # Get synthesized result
        synthesized = final_synthesizer.get_current_synthesis()
        if synthesized:
            unified_report = synthesized.content
            unified_scope = synthesized.scope
        else:
            # Fallback if no synthesis (shouldn't happen with valid results)
            unified_report = await self._merge_reports(validated_results)
            unified_scope = self._compute_scope(validated_results)

        # Phase 8: Build global impact graph
        cross_page_graph = await self._build_cross_page_impact_graph(validated_results)
        unified_report.impact_graph = cross_page_graph

        return ScopeAwareResult(
            content=unified_report,
            scope=unified_scope
        )

    @action_executor()
    async def spawn_next_batch(self, change_description: str) -> None:
        """Spawn next batch of agents based on working set overlap.

        Follows CodeAnalysisCoordinatorV2 pattern:
        - Score pages by overlap with current working set
        - Spawn highest-overlap pages first
        - Uses cache-aware priorities
        """
        if not self.pending_pages:
            return

        page_graph = await self.agent.load_page_graph()
        # Get current working set
        working_set = await self.working_set_cap.get_working_set() if self.working_set_cap else {}
        working_set = set(working_set.get("pages", []))

        # Score pages by working set overlap
        scored_pages = []
        for page_id in self.pending_pages:
            # Score based on overlap with working set
            if page_id in working_set:
                overlap_score = 1.0  # Page is in working set - perfect!
            elif page_graph.has_node(page_id):
                # Check overlap with neighbors
                neighbors = set(page_graph.predecessors(page_id)) | set(page_graph.successors(page_id))
                overlap = len(neighbors & working_set)
                overlap_score = overlap / max(len(neighbors), 1)
            else:
                overlap_score = 0.0

            # Also consider page centrality
            centrality_score = 0.0
            if page_graph.has_node(page_id):
                centrality_score = page_graph.degree(page_id) / max(page_graph.number_of_nodes(), 1)

            # Combined score
            total_score = overlap_score * 0.7 + centrality_score * 0.3
            scored_pages.append((page_id, total_score))

        # Sort by score (descending)
        scored_pages.sort(key=lambda x: x[1], reverse=True)

        # Determine batch size
        available_slots = self.max_agents - len(self.page_agents)
        batch_size = min(available_slots, len(scored_pages))

        if batch_size <= 0:
            return

        # Take top pages for this batch
        batch_pages = [page_id for page_id, score in scored_pages[:batch_size]]

        logger.info(
            f"Spawning batch: {len(batch_pages)} pages "
            f"(pending: {len(self.pending_pages)}, working_set: {len(working_set)})"
        )

        # Spawn agents
        await self._spawn_page_agents(batch_pages, change_description)

        # Remove from pending
        for page_id in batch_pages:
            self.pending_pages.remove(page_id)

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    async def _prefetch_for_cross_agent_feedback(
        self,
        result: ScopeAwareResult[ChangeImpactReport]
    ) -> None:
        """Prefetch pages that might need refinement based on this result."""
        if not self.feedback_predictor:
            return

        source_page = self._get_page_id_from_result(result)
        if not source_page:
            return

        # Predict pages that might need refinement
        pages_to_prefetch = self.feedback_predictor.predict_cross_agent_pages(source_page)

        # Prefetch with low priority
        for page_id in pages_to_prefetch:
            try:
                await self.agent.request_page(page_id, priority=1)
            except Exception as e:
                logger.debug(f"Failed to prefetch page {page_id}: {e}")

    async def _validate_critical_impacts(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> list[ScopeAwareResult[ChangeImpactReport]]:
        """Validate CRITICAL impacts using HypothesisGameProtocol.

        For each CRITICAL impact, runs a hypothesis game with:
        - Proposer: Agent that found the impact
        - Skeptic: Other agents that might have counter-evidence
        - Arbiter: Coordinator synthesizes final decision

        Args:
            results: Results with potential CRITICAL impacts

        Returns:
            Results with validated/refined critical impacts
        """
        # Find CRITICAL impacts
        critical_impacts = []
        for result in results:
            for component in result.content.impacted_components:
                if component.severity == ImpactSeverity.CRITICAL:
                    critical_impacts.append((result, component))

        if not critical_impacts:
            return results  # No critical impacts to validate

        logger.info(f"Validating {len(critical_impacts)} CRITICAL impacts via HypothesisGameProtocol")

        # For each critical impact, create hypothesis and run game
        for result, impact in critical_impacts[:5]:  # Limit to top 5 to avoid delays - TODO: Make configurable.
            # TODO: Generate hypothesis using LLM reasoning.
            hypothesis = Hypothesis(
                claim=f"CRITICAL impact on {impact.component_id}: {impact.description}",
                test_queries=[
                    f"What evidence supports this critical impact on {impact.component_id}?",
                    f"What evidence contradicts this critical impact?",
                    f"Are there alternative explanations for the observed behavior?"
                ],
                confidence=impact.confidence,
                created_by=self.agent.agent_id
            )

            game_result = await self._run_impact_hypothesis_game(hypothesis, result)

            # Update impact based on game result - modify a COPY to avoid side effects
            if game_result:
                # Find the component in the result and update a copy
                for i, comp in enumerate(result.content.impacted_components):
                    if comp.component_id == impact.component_id:
                        updated_comp = comp.model_copy(deep=True)
                        if game_result.get("status") == "supported":
                            updated_comp.confidence = min(1.0, updated_comp.confidence + 0.1)
                        elif game_result.get("status") == "refuted":
                            # Downgrade from CRITICAL to HIGH
                            updated_comp.severity = ImpactSeverity.HIGH
                            updated_comp.confidence = max(0.3, updated_comp.confidence - 0.2)
                            logger.info(
                                f"Critical impact on {updated_comp.component_id} downgraded to HIGH "
                                f"(refuted by hypothesis game)"
                            )
                        # Replace in list (still mutating result, but creating new component)
                        result.content.impacted_components[i] = updated_comp
                        break

        return results

    async def _run_impact_hypothesis_game(
        self,
        hypothesis: Hypothesis,
        source_result: ScopeAwareResult[ChangeImpactReport]
    ) -> dict[str, Any] | None:
        """Run hypothesis game for validating an impact.

        NOTE: By the time this is called, page agents have already been stopped.
        So we run validation using coordinator's LLM inference on the source page,
        rather than messaging agents that no longer exist.
        # TODO: Keep the page agents running and use them to run the hypothesis game.
        """
        # TODO: Add templates for hypothesis generation and game running for
        # library users to use. The template should specify:
        # - The hypothesis generation callback function
        # - Participant agents and roles
        source_page = self._get_page_id_from_result(source_result)
        if not source_page or source_page == "unknown":
            return {"status": "uncertain", "confidence": 0.5}

        # Build validation prompt
        prompt = f"""You are validating a CRITICAL impact hypothesis.

Hypothesis: {hypothesis.claim}

Based on the code in context, evaluate:
1. What evidence SUPPORTS this claim?
2. What evidence CONTRADICTS this claim?
3. Are there alternative explanations?

Then provide your assessment:
- status: "supported" (strong evidence for), "refuted" (strong evidence against), or "uncertain"
- confidence: 0.0 to 1.0
- reasoning: brief explanation

Respond with status (supported/refuted/uncertain), confidence (0-1), and reasoning."""

        try:
            # Run LLM inference on the source page
            response = await self.agent.infer(
                prompt=prompt,
                context_page_ids=[source_page]
            )

            # Parse response (simplified - just look for key words)
            text = response.generated_text.lower()
            if "refuted" in text or "contradicts" in text or "no evidence" in text:
                return {"status": "refuted", "confidence": 0.6}
            elif "supported" in text or "confirms" in text or "strong evidence" in text:
                return {"status": "supported", "confidence": 0.8}
            else:
                return {"status": "uncertain", "confidence": 0.5}

        except Exception as e:
            logger.warning(f"Hypothesis validation failed: {e}")
            return {"status": "uncertain", "confidence": 0.5}

    async def _get_or_compute_dependency_graph(
        self,
        page_ids: list[str]
    ) -> dict[str, set[str]]:
        """Get dependency graph from blackboard or compute it from impact paths.

        Args:
            page_ids: Pages being analyzed

        Returns:
            Reverse dependency graph: component_id -> {dependent_component_ids}
        """
        if not self.blackboard:
            return {}

        try:
            # Check blackboard for pre-computed dependency graph
            dep_graph_key = f"dependency_graph:{':'.join(sorted(page_ids[:10]))}"  # TODO: Add DependencyGraphResult.get_key(page_ids, tenant_id) method?
            dep_data = await self.blackboard.read(
                key=dep_graph_key,
                namespace="analysis"  # TODO: Add DependencyGraphResult.get_namespace(tenant_id, agent_type, agent_id) method?
            )

            if dep_data and isinstance(dep_data, dict):
                # Convert lists to sets
                return {k: set(v) if isinstance(v, list) else v 
                        for k, v in dep_data.items()}
        except Exception as e:
            logger.debug(f"No cached dependency graph: {e}")

        # TODO: Spawn DependencyAnalysisCoordinator to compute full graph
        # For now, return empty - will be built from impact paths during merge
        logger.info(
            "Dependency graph not cached. Will build from impact paths during analysis. "
            "For better accuracy, implement DependencyAnalysisCoordinator."
        )
        return {}

    async def _spawn_page_agents(
        self,
        page_ids: list[str],
        change_description: str
    ) -> None:
        """Spawn ONE agent PER page with cache-aware configuration.

        Uses AgentPoolCapability for agent lifecycle management while maintaining
        backward-compatible tracking in self.page_agents.

        Args:
            page_ids: Page IDs for this batch (already prioritized by working set overlap)
            change_description: Description of changes (for logging)
        """
        # Serialize changes for passing to agents
        changes_data = [c.model_dump() for c in self._current_changes]

        # Spawn agents via AgentPoolCapability
        for page_id in page_ids:
            role = f"impact_{page_id}"

            result = await self.agent_pool_cap.create_agent(
                agent_type="polymathera.colony.samples.code_analysis.impact.ChangeImpactAnalysisAgent",
                capabilities=["ChangeImpactAnalysisCapability"],
                bound_pages=[page_id],  # Single page
                role=role,
                metadata=AgentMetadata(
                    page_id=page_id,
                    parent_agent_id=self.agent.agent_id,
                    group_id=self.agent.metadata.group_id,
                    tenant_id=self.agent.metadata.tenant_id,
                    session_id=self.agent.metadata.session_id,
                    run_id=self.agent.metadata.run_id,
                    parameters={
                        "quality_threshold": self.agent.metadata.parameters.get("quality_threshold", 0.7),
                        "max_iterations": self.agent.metadata.parameters.get("max_iterations", 3),
                        "prefetch_depth": self.agent.metadata.parameters.get("prefetch_depth", 2),
                        "prefetch_test_pages": self.agent.metadata.parameters.get("prefetch_test_pages", True),
                        # Pass actual changes, not just description
                        "changes": changes_data,
                        "change_description": change_description,
                    }
                ),
            )

            if result.get("created"):
                agent_id = result["agent_id"]
                # Maintain backward-compatible tracking
                self.page_agents[page_id] = agent_id
            else:
                logger.warning(f"Failed to create agent for page {page_id}: {result.get('error')}")

    async def _collect_results(self) -> list[ScopeAwareResult[ChangeImpactReport]]:
        """Collect local impact results from page agents.

        Returns:
            List of local impact reports (one per page)
        """

        # TODO: Replace with event-driven collection via on_child_complete()

        results = []
        timeout = 60.0
        start_time = time.time()

        while len(results) < len(self.page_agents) and time.time() - start_time < timeout:
            messages = await self.agent.receive_messages(max_messages=5)  # TODO: What the fuck is this? This is not how we receive the results.

            for message in messages:
                msg_content = message.content
                if msg_content.get("type") == "analysis_complete":
                    result = ScopeAwareResult[ChangeImpactReport](**msg_content["result"])
                    results.append(result)

            if not messages:
                await asyncio.sleep(0.1)

        return results

    async def _propagate_impact_multi_hop(
        self,
        local_results: list[ScopeAwareResult[ChangeImpactReport]],
        dependency_graph: dict[str, set[str]]
    ) -> list[ScopeAwareResult[ChangeImpactReport]]:
        """Propagate impacts through dependency chains (multi-hop).

        Uses dependency graph to identify indirect impacts:
        If component A changes and B depends on A, B is indirectly impacted.

        Args:
            local_results: Local impact reports from page agents
            dependency_graph: Reverse dependencies (component -> dependents)

        Returns:
            Results enriched with indirect impacts and propagation paths
        """
        if not dependency_graph:
            # Build dependency graph from impact paths in local results
            dependency_graph = self._extract_dependency_graph_from_results(local_results)

            if not dependency_graph:
                # Still no dependencies - return local results unchanged
                return local_results

        # Extract all directly impacted components
        directly_impacted: dict[str, tuple[ImpactedComponent, str]] = {}  # comp_id -> (component, page_id)
        for result in local_results:
            result_page_id = self._get_page_id_from_result(result)
            for component in result.content.impacted_components:
                if component.component_id not in directly_impacted:
                    directly_impacted[component.component_id] = (component, result_page_id)

        # Propagate impact through dependency chains (BFS)
        indirect_impacts: dict[str, list[tuple[ImpactedComponent, ImpactPath]]] = {}  # page_id -> impacts

        for comp_id, (component, source_page_id) in directly_impacted.items():
            if comp_id not in dependency_graph:
                continue

            # Find all dependents (multi-hop with depth limit)
            visited = {comp_id}
            frontier = [(comp_id, component, 0)]  # (comp_id, component, depth)

            while frontier and len(frontier) < 100:  # Limit total propagation
                current_id, current_comp, depth = frontier.pop(0)

                if depth >= self.max_depth:
                    continue

                # Get dependents of current component
                dependents = dependency_graph.get(current_id, set())

                for dependent_id in dependents:
                    if dependent_id in visited:
                        continue

                    visited.add(dependent_id)

                    # Create indirect impact
                    indirect_component = ImpactedComponent(
                        component_id=dependent_id,
                        component_type="component",  # Generic type  # TODO: Use a more specific type.
                        file_path="unknown",  # Will be enriched later  # TODO: Use a more specific file path.
                        impact_types=[ImpactType.FUNCTIONAL],  # TODO: Use a more specific impact type.
                        severity=self._propagate_severity(current_comp.severity),
                        description=f"Indirect impact from {current_comp.component_id}",  # TODO: Use a more specific description.
                        requires_update=False,
                        confidence=current_comp.confidence * 0.8 ** (depth + 1)  # Decay confidence  # TODO: Use a more specific confidence decay.
                    )

                    # Create impact path
                    # TODO: Why do all impact paths created here have ONE step?
                    path = ImpactPath(
                        path_id=f"path_{comp_id}_to_{dependent_id}",
                        source_change=comp_id,
                        steps=[
                            ImpactStep(
                                from_component=current_id,
                                to_component=dependent_id,
                                relationship="depends_on",
                                impact_reason=f"Component depends on {current_id}"
                            )
                        ],
                        final_impact=indirect_component,
                        path_type="indirect",
                        total_severity=indirect_component.severity
                    )

                    # Add to indirect impacts (grouped by page)
                    target_page_id = self._find_page_for_component(dependent_id, local_results)
                    if target_page_id:
                        if target_page_id not in indirect_impacts:
                            indirect_impacts[target_page_id] = []
                        indirect_impacts[target_page_id].append((indirect_component, path))

                    # Continue propagation
                    frontier.append((dependent_id, indirect_component, depth + 1))

        # Add indirect impacts and paths to results
        enriched_results = []
        for result in local_results:
            result_page_id = self._get_page_id_from_result(result)

            if result_page_id in indirect_impacts:
                # Add indirect impacts to this page's result
                enriched_report = result.content.model_copy(deep=True)
                for indirect_comp, path in indirect_impacts[result_page_id]:
                    enriched_report.impacted_components.append(indirect_comp)
                    enriched_report.impact_paths.append(path)

                enriched_results.append(ScopeAwareResult(
                    content=enriched_report,
                    scope=result.scope
                ))
            else:
                enriched_results.append(result)

        return enriched_results

    def _extract_dependency_graph_from_results(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> dict[str, set[str]]:
        """Extract dependency graph from impact paths in results.

        Args:
            results: Local impact results

        Returns:
            Reverse dependency graph built from impact paths
        """
        dep_graph: dict[str, set[str]] = {}

        for result in results:
            for path in result.content.impact_paths:
                for step in path.steps:
                    # Build reverse dependency: if A -> B, then B has dependent A
                    if step.to_component not in dep_graph:
                        dep_graph[step.to_component] = set()
                    dep_graph[step.to_component].add(step.from_component)

        return dep_graph

    def _get_page_id_from_result(self, result: ScopeAwareResult[ChangeImpactReport]) -> str:
        """Extract page ID from result.

        Args:
            result: Impact result

        Returns:
            Page ID or "unknown" if not set
        """
        # Use source_page_id which is set by the policy/agent
        if result.content.source_page_id:
            return result.content.source_page_id
        return "unknown"

    def _find_page_for_component(
        self,
        component_id: str,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> str | None:
        """Find which page contains a component.

        Args:
            component_id: Component to find
            results: All results

        Returns:
            Page ID or None
        """
        for result in results:
            for component in result.content.impacted_components:
                if component.component_id == component_id:
                    return self._get_page_id_from_result(result)
        return None

    async def _build_cross_page_impact_graph(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> RelationshipGraph:
        """Build cross-page impact graph from individual page reports.

        Also contributes cross-page edges to the persistent page graph
        via PageGraphCapability so other agents benefit from discovered
        impact relationships.

        Args:
            results: Individual page reports

        Returns:
            Unified impact graph showing cross-page dependencies
        """
        graph = RelationshipGraph()

        # Add all impacted components as nodes
        for result in results:
            report = result.content
            for component in report.impacted_components:
                graph.add_node(
                    node_id=component.component_id,
                    node_type=component.component_type,
                    metadata={
                        "file_path": component.file_path,
                        "severity": component.severity.value,
                        "impact_types": [t.value for t in component.impact_types],
                        "requires_update": component.requires_update,
                        "confidence": component.confidence
                    }
                )

        # Add edges from impact paths
        for result in results:
            report = result.content
            for path in report.impact_paths:
                for step in path.steps:
                    graph.add_edge(
                        source_id=step.from_component,
                        target_id=step.to_component,
                        relationship_type=step.relationship,
                        metadata={
                            "impact_reason": step.impact_reason,
                            "path_id": path.path_id
                        }
                    )

        # Contribute cross-page edges to the persistent page graph
        await self._contribute_impact_to_page_graph(graph, results)

        return graph

    async def _contribute_impact_to_page_graph(
        self,
        impact_graph: RelationshipGraph,
        results: list[ScopeAwareResult[ChangeImpactReport]],
    ) -> None:
        """Contribute cross-page impact relationships to the persistent page graph.

        Maps component-level edges to page-level edges: when an impact path
        crosses from a component on page A to a component on page B, that
        creates a page-level "impact_dependency" edge in the persistent graph.

        Args:
            impact_graph: Component-level impact graph
            results: Page reports (for component-to-page mapping)
        """

        page_graph_cap = self.agent.get_capability_by_type(PageGraphCapability)
        if not page_graph_cap:
            logger.debug(
                "PageGraphCapability not available, "
                "skipping page graph contribution"
            )
            return

        # Map components to their source pages
        component_to_page: dict[str, str] = {}
        for result in results:
            page_id = self._get_page_id_from_result(result)
            for component in result.content.impacted_components:
                component_to_page[component.component_id] = page_id

        # Extract page-level edges from cross-page component relationships
        page_level_relationships: list[dict] = []
        seen_page_edges: set[tuple[str, str]] = set()

        for relationship in impact_graph.edges.values():
            if not isinstance(relationship, Relationship):
                continue

            source_page = component_to_page.get(relationship.source_id)
            target_page = component_to_page.get(relationship.target_id)

            if not source_page or not target_page:
                continue
            if source_page == target_page:
                continue

            edge_key = (source_page, target_page)
            if edge_key in seen_page_edges:
                continue
            seen_page_edges.add(edge_key)

            page_level_relationships.append({
                "source_id": source_page,
                "target_id": target_page,
                "relationship_type": "impact_dependency",
                "confidence": relationship.confidence,
                "weight": relationship.weight,
                "metadata": {
                    "discovered_via": "impact_analysis",
                    "component_edge": (
                        f"{relationship.source_id} -> "
                        f"{relationship.target_id}"
                    ),
                },
            })

        if page_level_relationships:
            apply_result = await page_graph_cap.apply_relationships(
                page_level_relationships
            )
            logger.info(
                f"Contributed {apply_result.get('new_edges', 0)} new and "
                f"{apply_result.get('updated_edges', 0)} updated page-level "
                f"impact edges to persistent graph"
            )

    async def _merge_reports(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> ChangeImpactReport:
        """Merge individual page reports into unified report.

        Args:
            results: Individual page reports

        Returns:
            Unified report (without impact_graph - that's added separately)
        """
        if not results:
            return ChangeImpactReport(
                changes=[],  # Empty list of changes
                impacted_components=[],
                recommendations=[]
            )

        merge_policy = ImpactMergePolicy()
        merged_result = await merge_policy.merge(
            results,
            MergeContext(strategy="union_impacts")
        )

        return merged_result.content

    def _compute_scope(
        self,
        results: list[ScopeAwareResult[ChangeImpactReport]]
    ) -> AnalysisScope:
        """Compute unified scope.

        Args:
            results: Impact reports

        Returns:
            Unified scope
        """
        if not results:
            return AnalysisScope()

        scopes = [r.scope for r in results]

        return AnalysisScope(
            is_complete=all(s.is_complete for s in scopes),
            missing_context=list(set(sum([s.missing_context for s in scopes], []))),
            confidence=sum(s.confidence for s in scopes) / len(scopes),
            reasoning=sum([s.reasoning for s in scopes], [])
        )


class ChangeImpactAnalysisCoordinator(Agent):
    """Cache-aware coordinator for change impact analysis team.

    This coordinator:
    1. Spawns impact agents for each page
    2. Collects and aggregates impact reports
    3. Traces cross-page dependencies
    4. Builds complete impact graph
    5. Produces unified impact report with recommendations

    The actual coordination logic lives in ChangeImpactAnalysisCoordinatorCapability.
    Actions are invoked through the action routing system via @action_executor.
    Events are handled through @event_handler decorators on capabilities.
    """

    coordinator_capability: ChangeImpactAnalysisCoordinatorCapability | None = None

    async def initialize(self) -> None:
        """Initialize coordinator and attach capability."""

        job_quota = self.metadata.parameters.get("job_quota", self.metadata.parameters.get("max_agents", 10) * 5)

        self.add_capability_blueprints([
            # Add WorkingSetCapability for cache-aware coordination
            WorkingSetCapability.bind(
                working_set_size=job_quota,
            ),
            CriticCapability.bind(),
            ChangeImpactAnalysisCoordinatorCapability.bind(),
            MergeCapability.bind(
                scope_id=f"{self.agent_id}:merged_change_impact_reports",
                merge_policy=ImpactMergePolicy()
            ),
            SynthesisCapability.bind(
                scope_id=f"{self.agent_id}:change_impact_analysis"
            ),
        ])

        await super().initialize()

        self.coordinator_capability = self.get_capability(
            ChangeImpactAnalysisCoordinatorCapability.get_capability_name()
        )

        logger.info(f"ChangeImpactAnalysisCoordinator {self.agent_id} initialized")

