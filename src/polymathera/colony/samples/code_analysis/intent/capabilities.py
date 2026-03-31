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
import itertools
from typing import Any
from overrides import override

from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.agents.blackboard.protocol import AgentRunProtocol, IntentAnalysisProtocol
from polymathera.colony.agents.patterns import (
    AnalysisScope,
    ScopeAwareResult,
    ConfidenceTracker,
)
from polymathera.colony.agents.patterns.capabilities.merge import (
    MergePolicy,
    MergeContext,
    MergeCapability,
)
from polymathera.colony.agents.patterns.capabilities.validation import ValidationResult
from polymathera.colony.agents.patterns.capabilities.agent_pool import AgentPoolCapability
from polymathera.colony.agents.patterns.capabilities.result import ResultCapability
from polymathera.colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from polymathera.colony.agents.patterns.capabilities.batching import BatchingPolicy
from polymathera.colony.agents.patterns.capabilities.vcm_analysis import VCMAnalysisCapability
from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.events import event_handler, EventProcessingResult
from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.base import Agent, AgentCapability, AgentHandle, AgentMetadata
from polymathera.colony.agents.models import Action, PolicyREPL, AgentResourceRequirements, AgentSuspensionState
from polymathera.colony.agents.patterns.games.negotiation.capabilities import NegotiationIssue, Offer, calculate_pareto_efficiency
from polymathera.colony.agents.patterns.games.coalition_formation import find_optimal_coalition_structure
from polymathera.colony.cluster.models import LLMClientRequirements

from .types import (
    IntentCategory,
    IntentAlignment,
    CodeIntent,
    IntentRelationship,
    IntentGraph,
    IntentConflict,
    IntentInferenceContext,
    IntentInferenceResult,
)

logger = logging.getLogger(__name__)



class IntentInferenceCapability(AgentCapability):
    """Capability for inferring code intent using LLM reasoning.

    This capability uses an LLM to:
    1. Understand high-level purpose of code
    2. Extract business logic and goals
    3. Identify intent categories
    4. Detect misalignment between code and intent
    5. Build intent relationship graphs

    Works in two modes via the `scope_id` parameter:

    1. **Local mode** (in IntentInferenceAgent): Processes intent inference requests
       ```python
       capability = IntentInferenceCapability(agent=self)  # scope_id = agent.agent_id
       ```

    2. **Remote mode** (in parent agent): Communicates with child IntentInferenceAgent
       ```python
       handle = await parent.spawn_child_agents(...)[0]
       intent_cap = handle.get_capability(IntentInferenceCapability)
       future = await intent_cap.get_result_future()
       result = await future
       ```

    Provides @action_executor methods for:
    - infer_intent: Infer intent from code across pages
    - analyze_page: Analyze a single page for intent
    """


    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "intent_inference",
        input_patterns: list[str] = [AgentRunProtocol.request_pattern()],
        use_context: bool = True,
        detect_misalignment: bool = True,
        granularity: str = "function",
        capability_key: str = "intent_inference_capability"
    ):
        """Initialize intent inference capability.

        Args:
            agent: Agent using this capability
            scope: Blackboard scope for this capability (default: AGENT)
            namespace: Namespace for event patterns
            input_patterns: List of event patterns to subscribe to
            use_context: Whether to use surrounding context
            detect_misalignment: Whether to detect code-intent misalignment
            granularity: Analysis granularity (function/class/module)
            capability_key: Unique key for this capability within the agent
        """
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=input_patterns, capability_key=capability_key)
        self.use_context = use_context
        self.detect_misalignment = detect_misalignment
        self.granularity = granularity
        self.confidence_tracker = ConfidenceTracker()

    def get_action_group_description(self) -> str:
        return (
            "Intent Inference — infers high-level purpose and business logic from code. "
            f"Granularity: {self.granularity}. Detects code-intent misalignment. "
            "Builds intent hierarchies across pages. Confidence-tracked inferences. "
            "Supports local mode (single page) and remote mode (via AgentHandle)."
        )

    def _get_merge_capability(self) -> MergeCapability | None:
        """Get MergeCapability from agent dynamically."""
        return self.agent.get_capability_by_type(MergeCapability)

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for IntentInferenceCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for IntentInferenceCapability")
        pass

    @event_handler(pattern=AgentRunProtocol.request_pattern())
    async def handle_analysis_request(
        self,
        event: BlackboardEvent,
        _repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Handle analysis request events from AgentHandle.run().

        Args:
            event: Blackboard event to process
            _scope: Policy scope (unused)

        Returns:
            EventProcessingResult with immediate action if this is a request
        """
        # Extract request data
        request_data = event.value
        if not isinstance(request_data, dict):
            return None

        input_data = request_data.get("input", {})
        page_ids = input_data.get("page_ids", [])
        context = input_data.get("context")
        granularity = input_data.get("granularity", self.granularity)

        # Extract request_id from event key
        parts = event.key.split(":")
        request_id = parts[-1] if len(parts) >= 3 else None

        # Return immediate action to execute inference
        return EventProcessingResult(
            immediate_action=Action(
                action_type="infer_intent",
                parameters={
                    "page_ids": page_ids,
                    "context": context,
                    "granularity": granularity,
                    "request_id": request_id,
                }
            )
        )

    # TODO: Replace page_ids with more general context identifiers.
    # For example: Use a query over pages or repositories, or tags.
    # This is important because the ActionPolicy decides the parameters
    # to these actions, which can be challenging for a LLM to do, especially
    # for very long contexts spanning many pages, repositories, etc.
    @action_executor(action_key="infer_intent")
    async def infer_intent(
        self,
        page_ids: list[str],
        context: IntentInferenceContext | None = None,
        granularity: str = "function",  # function, class, module
        request_id: str | None = None,
    ) -> IntentInferenceResult:
        """Infer intent from code across multiple pages.

        This is an LLM-plannable action that aggregates intent inference
        across pages using a merge policy for consistent graph building.

        Args:
            page_ids: VCM page IDs to analyze
            context: Additional context (docs, tests, etc.)
            granularity: Analysis granularity level
            request_id: Optional request ID for blackboard response

        Returns:
            Aggregated intent graph
        """
        # TODO: Parallelize if many pages.
        # TODO: Use an iterative cache-aware strategy:
        # - Use the page graph cluster related pages together
        # - Analyze pages in a cluster in parallel and iterate to refine
        # - Move to the next cluster based on dependencies:
        #       - Analyze new pages and try to close unresolved pages from previous rounds
        # - Continue until all pages are resolved or max iterations reached
        # - Merge all results at the end, or incrementally after each cluster
        # TODO: This can be a general pattern for cross-cutting analysis.
        # Package it as a reusable abstract analysis pattern and reuse it.
        # TODO: Make pattern usage declarative
        # Process each page individually
        page_results = []
        for page_id in page_ids:
            result = await self._infer_intent_from_page(
                page_id,
                context,
                granularity
            )
            page_results.append(result)

        # Merge results using MergeCapability
        if len(page_results) == 1:
            final_result = page_results[0]
        else:
            merge_cap = self._get_merge_capability()
            if merge_cap is None:
                raise RuntimeError(
                    "IntentInferenceCapability requires MergeCapability on the agent. "
                    "Add MergeCapability configured with IntentMergePolicy."
                )
            final_result = await merge_cap.merge_results(
                page_results,
                MergeContext(strategy="graph_union")  # Union intent graphs
            )

        # Write result to blackboard for AgentHandle.run() to receive
        if request_id:
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=AgentRunProtocol.result_key(request_id),
                value=final_result.model_dump(),
                agent_id=self.agent.agent_id,
            )

        return final_result

    @action_executor(action_key="analyze_page")
    async def analyze_page(
        self,
        page_id: str,
        context: IntentInferenceContext | None = None,
        granularity: str | None = None,
        request_id: str | None = None,
    ) -> IntentInferenceResult:
        """Analyze a single page for intent - LLM-plannable action.

        Args:
            page_id: Single VCM page ID
            context: Additional context
            granularity: Analysis granularity (uses default if not specified)
            request_id: Optional request ID for blackboard response

        Returns:
            Intent graph for this page
        """
        result = await self._infer_intent_from_page(
            page_id,
            context,
            granularity or self.granularity
        )

        # Write result to blackboard for AgentHandle.run() to receive
        if request_id:
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=AgentRunProtocol.result_key(request_id),
                value=result.model_dump(),
                agent_id=self.agent.agent_id,
            )

        return result

    async def _infer_intent_from_page(
        self,
        page_id: str,
        context: IntentInferenceContext | None = None,
        granularity: str = "function"
    ) -> IntentInferenceResult:
        """Infer intent from a single VCM page.

        Args:
            page_id: Single VCM page ID
            context: Additional context
            granularity: Analysis granularity

        Returns:
            Intent graph for this page
        """
        # Build intent inference prompt
        prompt = self._build_page_intent_prompt(context, granularity)

        # Get LLM analysis for single page with structured output
        response = await self.agent.infer(
            context_page_ids=[page_id],  # Single page
            prompt=prompt,
            temperature=0.3,  # TODO: Make configurable
            max_tokens=2000,  # TODO: Make configurable
            json_schema=IntentInferenceResult.model_json_schema()
        )

        # Parse structured response
        result = IntentInferenceResult.model_validate_json(response.generated_text)  # TODO: Handle schema errors. LLMs are not perfect.

        # Detect conflicts if enabled
        if self.detect_misalignment:
            result.content.conflicts = await self._detect_conflicts(result.content)

        # Validate against context if available
        if context:
            result = self._validate_result_with_context(result, context)

        return result

    def _build_page_intent_prompt(self, context: IntentInferenceContext | None, granularity: str) -> str:
        """Build prompt for page-level intent inference.

        Args:
            context: Additional context
            granularity: Analysis granularity

        Returns:
            Formatted prompt
        """
        context_section = ""
        if context:
            if context.documentation:
                context_section += f"\nDocumentation:\n{context.documentation}\n"
            if context.tests:
                context_section += f"\nTest cases suggest:\n{context.tests}\n"
            if context.comments:
                context_section += f"\nComments:\n{context.comments}\n"

        return f"""Analyze the intent and purpose of the code loaded in context.

Analyze at {granularity} granularity level.

{context_section}

For each code segment (function/class/module), create an IntentGraph with:

1. NODES - Each significant code segment becomes a CodeIntent node with:
   - segment_id: unique identifier
   - primary_intent: main purpose in one sentence
   - secondary_intents: additional purposes
   - categories: from {', '.join([c.value for c in IntentCategory])}
   - business_goals: business objectives served
   - preconditions: what the code assumes
   - postconditions: intended outcomes
   - alignment: aligned/misaligned/partially_aligned/unclear
   - issues: implementation problems
   - evidence: supporting code patterns
   - confidence: 0.0-1.0

2. EDGES - Relationships between intents:
   - supports: one intent enables another
   - contradicts: intents conflict
   - implements: one realizes another
   - depends_on: requires another

3. HIERARCHIES - Parent-child relationships (e.g., methods within classes)

4. CONFLICTS - Detected intent conflicts with severity and suggested resolutions

Return a JSON object matching the IntentInferenceResult schema.

Focus on the WHY (intent), not just the WHAT (behavior)."""

    def _validate_result_with_context(
        self,
        result: IntentInferenceResult,
        context: IntentInferenceContext
    ) -> IntentInferenceResult:
        """Validate intent result against context.

        Args:
            result: Intent inference result
            context: Additional context

        Returns:
            Validated result
        """
        # Boost confidence if documentation aligns
        if context.documentation:
            doc = context.documentation.lower()
            for intent in result.content.nodes.values():
                if any(goal.lower() in doc for goal in intent.business_goals):
                    intent.confidence = min(1.0, intent.confidence + 0.1)

        # Update scope confidence based on validation
        confidences = [i.confidence for i in result.content.nodes.values()]
        if confidences:
            result.scope.confidence = sum(confidences) / len(confidences)

        return result


    async def _build_intent_graph(
        self,
        intents: list[CodeIntent]
    ) -> IntentGraph:
        """Build graph of intent relationships.

        Args:
            intents: Individual intents
            code: Full code for context

        Returns:
            Intent graph
        """
        graph = IntentGraph()

        # Add nodes
        for intent in intents:
            graph.nodes[intent.segment_id] = intent

        # Discover relationships
        for i, intent1 in enumerate(intents):
            for intent2 in intents[i+1:]:
                rel = await self._find_relationship(intent1, intent2)
                if rel:
                    graph.edges.append(rel)

        # Build hierarchies (simplified)
        self._build_hierarchies(graph)

        return graph

    async def _find_relationship(
        self,
        intent1: CodeIntent,
        intent2: CodeIntent
    ) -> IntentRelationship | None:
        """Find relationship between intents.

        Args:
            intent1: First intent
            intent2: Second intent

        Returns:
            Relationship if found
        """
        # Check for support relationship
        if any(pre in intent2.preconditions for pre in intent1.postconditions):
            return IntentRelationship(
                source_id=intent1.segment_id,
                target_id=intent2.segment_id,
                relationship_type="supports",
                strength=0.8
            )

        # Check for contradiction
        if intent1.primary_intent and intent2.primary_intent:
            # Simplified: check if intents seem opposed
            if ("validate" in intent1.primary_intent.lower() and 
                "skip validation" in intent2.primary_intent.lower()):
                return IntentRelationship(
                    source_id=intent1.segment_id,
                    target_id=intent2.segment_id,
                    relationship_type="contradicts",
                    strength=0.7
                )

        # Check for implementation relationship
        if intent1.line_start < intent2.line_start < intent1.line_end:
            return IntentRelationship(
                source_id=intent2.segment_id,
                target_id=intent1.segment_id,
                relationship_type="implements",
                strength=0.9
            )

        return None

    def _build_hierarchies(self, graph: IntentGraph) -> None:
        """Build intent hierarchies.

        Args:
            graph: Intent graph to update
        """
        # Simple hierarchy: functions within classes
        class_intents = [i for i in graph.nodes.values() 
                        if IntentCategory.BUSINESS_LOGIC in i.categories]

        for class_intent in class_intents:
            children = []
            for other in graph.nodes.values():
                if (other.line_start > class_intent.line_start and
                    other.line_end < class_intent.line_end and
                    other.segment_id != class_intent.segment_id):
                    children.append(other.segment_id)

            if children:
                graph.hierarchies[class_intent.segment_id] = children

    async def _detect_conflicts(self, graph: IntentGraph) -> list[IntentConflict]:
        """Detect conflicts between intents.

        Args:
            graph: Intent graph

        Returns:
            List of conflicts
        """
        conflicts = []

        # Check for contradictory relationships
        for edge in graph.edges:
            if edge.relationship_type == "contradicts":
                conflict = IntentConflict(
                    intent1_id=edge.source_id,
                    intent2_id=edge.target_id,
                    conflict_type="contradiction",
                    severity="high" if edge.strength > 0.7 else "medium"
                )
                conflicts.append(conflict)

        # Check for misaligned intents
        for intent in graph.nodes.values():
            if intent.alignment == IntentAlignment.MISALIGNED:
                conflict = IntentConflict(
                    intent1_id=intent.segment_id,
                    intent2_id=intent.segment_id,  # Self-conflict
                    conflict_type="misalignment",
                    severity="high" if intent.issues else "medium",
                    resolution="Refactor to align implementation with intent"
                )
                conflicts.append(conflict)

        return conflicts

    def _check_completeness(self, graph: IntentGraph) -> bool:
        """Check if intent inference is complete.

        Args:
            graph: Intent graph

        Returns:
            True if complete
        """
        # Complete if we have intents with clear alignment
        if not graph.nodes:
            return False

        aligned = sum(1 for i in graph.nodes.values() 
                     if i.alignment != IntentAlignment.UNCLEAR)

        return aligned / len(graph.nodes) >= 0.7 if graph.nodes else False

    def _identify_missing_context(self, graph: IntentGraph) -> list[str]:
        """Identify missing context.

        Args:
            graph: Intent graph

        Returns:
            List of missing elements
        """
        missing = []

        for intent in graph.nodes.values():
            if intent.alignment == IntentAlignment.UNCLEAR:
                missing.append(f"Unclear intent at lines {intent.line_start}-{intent.line_end}")

            if intent.confidence < 0.5:
                missing.append(f"Low confidence intent at lines {intent.line_start}-{intent.line_end}")

        return missing

    def _summarize_intents(self, graph: IntentGraph) -> str:
        """Summarize inferred intents.

        Args:
            graph: Intent graph

        Returns:
            Summary
        """
        categories = {}
        for intent in graph.nodes.values():
            for cat in intent.categories:
                categories[cat] = categories.get(cat, 0) + 1

        conflicts = len(graph.conflicts)
        aligned = sum(1 for i in graph.nodes.values() 
                     if i.alignment == IntentAlignment.ALIGNED)

        return (f"Inferred {len(graph.nodes)} intents. "
                f"Categories: {', '.join(f'{k.value}({v})' for k,v in categories.items())}. "
                f"{aligned}/{len(graph.nodes)} aligned. "
                f"{conflicts} conflicts detected.")


class IntentMergePolicy(MergePolicy[IntentGraph]):
    """Policy for merging intent inference results."""

    async def merge(
        self,
        results: list[ScopeAwareResult[IntentGraph]],
        context: MergeContext
    ) -> ScopeAwareResult[IntentGraph]:
        """Merge multiple intent graphs.

        Args:
            results: Intent results to merge
            context: Merge context

        Returns:
            Merged intent graph
        """
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            return results[0]

        agent_weights = self._compute_agent_weights(results)
        coordination_notes: list[str] = []

        merged_graph = IntentGraph()

        # Merge nodes via negotiation
        node_versions: dict[str, list[tuple[CodeIntent, str]]] = {}
        for result in results:
            agent_id = result.producer_agent_id or result.result_id
            for segment_id, intent in result.content.nodes.items():
                node_versions.setdefault(segment_id, []).append((intent, agent_id))

        for segment_id, versions in node_versions.items():
            if len(versions) == 1:
                merged_graph.nodes[segment_id] = versions[0][0]
            else:
                merged_graph.nodes[segment_id] = self._resolve_intent_conflict(
                    segment_id,
                    versions,
                    agent_weights,
                    coordination_notes
                )

        # Merge edges (relationships)
        seen_edges = set()
        for result in results:
            for edge in result.content.edges:
                edge_key = (edge.source_id, edge.target_id, edge.relationship_type)
                if edge_key not in seen_edges:
                    merged_graph.edges.append(edge)
                    seen_edges.add(edge_key)

        # Merge hierarchies
        for result in results:
            for parent, children in result.content.hierarchies.items():
                if parent not in merged_graph.hierarchies:
                    merged_graph.hierarchies[parent] = children
                else:
                    merged_graph.hierarchies[parent] = list(
                        set(merged_graph.hierarchies[parent] + children)
                    )

        merged_graph.conflicts = sum([r.content.conflicts for r in results], [])

        merged_scope = AnalysisScope(
            is_complete=all(r.scope.is_complete for r in results),
            missing_context=list(set(sum([r.scope.missing_context for r in results], []))),
            confidence=sum(r.scope.confidence for r in results) / len(results),
            reasoning=sum([r.scope.reasoning for r in results], []) + coordination_notes
        )

        return IntentInferenceResult(
            content=merged_graph,
            scope=merged_scope
        )

    def _resolve_intent_conflict(
        self,
        segment_id: str,
        versions: list[tuple[CodeIntent, str]],
        agent_weights: dict[str, float],
        coordination_notes: list[str]
    ) -> CodeIntent:
        """Resolve conflicting intents via negotiation-inspired selection."""
        parties = [agent_id for _, agent_id in versions]
        issue = NegotiationIssue(
            issue_id=f"intent_merge::{segment_id}",
            description=f"Resolve intent disagreement for {segment_id}",
            parties=parties,
            preferences={
                agent_id: {"confidence": intent.confidence}
                for intent, agent_id in versions
            }
        )

        offers = []
        for idx, (intent, agent_id) in enumerate(versions):
            utility = intent.confidence * agent_weights.get(agent_id, 1.0)
            offers.append(
                Offer(
                    offer_id=f"{issue.issue_id}::{idx}",
                    proposer=agent_id,
                    terms={
                        "primary_intent": intent.primary_intent,
                        "alignment": intent.alignment.value,
                        "categories": [c.value for c in intent.categories]
                    },
                    utility=utility,
                    justification=f"confidence={intent.confidence:.2f}"
                )
            )

        efficiency = calculate_pareto_efficiency(offers, issue)
        pareto_ids = set(efficiency.get("pareto_optimal_offers", [])) or {o.offer_id for o in offers}
        winning_offer = max(
            (offer for offer in offers if offer.offer_id in pareto_ids),
            key=lambda offer: offer.utility
        )

        winning_intent = next(intent for intent, agent_id in versions if agent_id == winning_offer.proposer)
        merged_intent = winning_intent.model_copy(deep=True)

        # Aggregate evidence and metadata from all versions
        merged_intent.secondary_intents = sorted(
            set(sum((intent.secondary_intents for intent, _ in versions), []))
        )
        merged_intent.business_goals = sorted(
            set(sum((intent.business_goals for intent, _ in versions), []))
        )
        merged_intent.categories = sorted(
            set(sum(([cat for cat in intent.categories] for intent, _ in versions), [])),
            key=lambda cat: cat.value
        )
        merged_intent.evidence = sorted(
            set(sum((intent.evidence for intent, _ in versions), []))
        )
        merged_intent.issues = sorted(
            set(sum((intent.issues for intent, _ in versions), []))
        )
        merged_intent.confidence = min(1.0, winning_offer.utility)

        coordination_notes.append(
            f"Negotiation resolved intent for {segment_id}; "
            f"selected interpretation from agent {winning_offer.proposer}."
        )

        return merged_intent

    def _compute_agent_weights(
        self,
        results: list[ScopeAwareResult[IntentGraph]]
    ) -> dict[str, float]:
        """Compute coalition-based weights for intent contributors."""
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

    def _extract_capabilities(self, graph: IntentGraph) -> set[str]:
        """Extract capability tags from an intent graph."""
        caps: set[str] = set()
        for intent in graph.nodes.values():
            for category in intent.categories:
                caps.add(category.value)
            if intent.alignment == IntentAlignment.MISALIGNED:
                caps.add("misalignment_detection")
            if intent.issues:
                caps.add("issue_detection")
        return caps or {"baseline_intent"}

    async def validate(
        self,
        original: list[ScopeAwareResult[IntentGraph]],
        merged: ScopeAwareResult[IntentGraph]
    ) -> ValidationResult:
        """Validate merged intent graph.

        Args:
            original: Original results
            merged: Merged result

        Returns:
            Validation result
        """
        issues = []

        # Check no intents were lost
        original_intents = set()
        for result in original:
            original_intents.update(result.content.nodes.keys())

        if not original_intents.issubset(merged.content.nodes.keys()):
            missing = original_intents - set(merged.content.nodes.keys())
            issues.append(f"Lost intents during merge: {missing}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if not issues else 0.7
        )


# =============================================================================
# IntentAnalysisCapability - Extends VCMAnalysisCapability
# =============================================================================
#
# This capability extends VCMAnalysisCapability to provide atomic, composable
# primitives for intent inference across multiple pages.
#
# Design Philosophy:
# - Does NOT prescribe a workflow
# - LLM decides when/how to spawn workers, merge results, build hierarchies
# - Domain-specific methods exposed as action_executors for LLM composition
#
# Example LLM-driven workflow:
#   1. LLM: spawn_workers(page_ids, cache_affine=True)
#   2. LLM: [wait for completion events]
#   3. LLM: merge_results(page_ids)
#   4. LLM: build_cross_page_hierarchies(merged_result)  # domain-specific
#   5. LLM: detect_misalignments(merged_result)  # domain-specific
# =============================================================================


class IntentAnalysisCapability(VCMAnalysisCapability):
    """Capability for distributed intent inference using VCMAnalysisCapability primitives.

    Extends VCMAnalysisCapability with intent-specific:
    - Worker type (IntentInferenceCapability)
    - Merge policy (IntentMergePolicy)
    - Domain methods (cross-page hierarchies, misalignment detection)

    The LLM planner composes the atomic primitives from the base class with
    the domain-specific methods exposed here.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "intent_analysis",
        capability_key: str = "intent_analysis_capability"
    ):
        """Initialize intent analysis capability.

        Args:
            agent: Agent using this capability (coordinator agent)
            scope: Blackboard scope
            namespace: Namespace for event patterns
            capability_key: Unique key for this capability within the agent
        """
        super().__init__(agent=agent, scope=scope, namespace=namespace, input_patterns=[], capability_key=capability_key)

    # =========================================================================
    # Abstract Hook Implementations
    # =========================================================================

    def get_worker_capability_class(self) -> type[AgentCapability]:
        """Return IntentInferenceCapability as the worker capability."""
        return IntentInferenceCapability

    def get_worker_agent_type(self) -> str:
        """Return the worker agent type string."""
        return "polymathera.colony.samples.code_analysis.intent.IntentInferenceAgent"

    def get_domain_merge_policy(self) -> MergePolicy:
        """Return IntentMergePolicy for merging intent graphs."""
        return IntentMergePolicy()

    def get_analysis_parameters(self, **kwargs) -> dict[str, Any]:
        """Return intent-specific analysis parameters.

        Args:
            **kwargs: Parameters from action call

        Returns:
            Parameters for intent inference workers
        """
        return {
            "granularity": kwargs.get("granularity", "function"),
            "use_context": kwargs.get("use_context", True),
            "detect_misalignment": kwargs.get("detect_misalignment", True),
            **kwargs,
        }

    # =========================================================================
    # Domain-Specific Action Executors
    # =========================================================================

    @action_executor(action_key="build_cross_page_hierarchies")
    async def build_cross_page_hierarchies(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build hierarchical relationships across pages.

        Call after merge_results() to add category-level hierarchies.
        LLM decides when this is appropriate (not forced on every merge).

        Args:
            page_ids: Pages to consider (if None, uses all analyzed pages)

        Returns:
            Dict with:
            - hierarchies_added: Number of hierarchies created
            - categories_found: Categories with multiple segments
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        # Get merged results
        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        # Collect all intents and build category groups
        by_category: dict[str, list[str]] = {}  # category -> segment_ids

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            nodes = content.get("nodes", {})

            for segment_id, intent_data in nodes.items():
                categories = intent_data.get("categories", [])
                for category in categories:
                    cat_value = category if isinstance(category, str) else category.get("value", str(category))
                    if cat_value not in by_category:
                        by_category[cat_value] = []
                    by_category[cat_value].append(segment_id)

        # Build hierarchies for categories with multiple segments
        hierarchies_added = 0
        categories_found = []

        for category, segment_ids in by_category.items():
            if len(segment_ids) > 1:
                categories_found.append(category)
                hierarchies_added += 1

                # Store hierarchy in blackboard for access
                blackboard = await self.get_blackboard()
                await blackboard.write(
                    key=IntentAnalysisProtocol.intent_hierarchy_key(category),
                    value={
                        "category": category,
                        "parent_id": f"category_{category}",
                        "segment_ids": segment_ids,
                        "count": len(segment_ids),
                    },
                    created_by=self.agent.agent_id,
                )

        logger.info(
            f"IntentAnalysisCapability: built {hierarchies_added} cross-page hierarchies "
            f"for categories: {categories_found}"
        )

        return {
            "hierarchies_added": hierarchies_added,
            "categories_found": categories_found,
            "by_category": {cat: len(segs) for cat, segs in by_category.items()},
        }

    @action_executor(action_key="detect_misalignments")
    async def detect_misalignments(
        self,
        page_ids: list[str] | None = None,
        min_severity: str = "medium",
    ) -> dict[str, Any]:
        """Detect code-intent misalignments across analyzed pages.

        LLM can call this to identify areas where code doesn't match intent.

        Args:
            page_ids: Pages to check (if None, uses all analyzed pages)
            min_severity: Minimum severity to report

        Returns:
            Dict with:
            - misalignments: List of misalignment records
            - count: Total misalignments found
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        misalignments = []
        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        min_sev = severity_order.get(min_severity, 2)

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            nodes = content.get("nodes", {})

            for segment_id, intent_data in nodes.items():
                alignment = intent_data.get("alignment", "ALIGNED")
                if alignment == "MISALIGNED":
                    confidence = intent_data.get("confidence", 1.0)
                    issues = intent_data.get("issues", [])

                    # Determine severity based on confidence
                    if confidence < 0.5:
                        severity = "high"
                    elif confidence < 0.7:
                        severity = "medium"
                    else:
                        severity = "low"

                    if severity_order.get(severity, 0) >= min_sev:
                        misalignments.append({
                            "segment_id": segment_id,
                            "page_id": page_id,
                            "file_path": intent_data.get("file_path"),
                            "line_start": intent_data.get("line_start"),
                            "line_end": intent_data.get("line_end"),
                            "primary_intent": intent_data.get("primary_intent"),
                            "issues": issues,
                            "confidence": confidence,
                            "severity": severity,
                        })

        if misalignments:
            logger.warning(f"IntentAnalysisCapability: detected {len(misalignments)} misalignments")

            # Store in blackboard for tracking
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=IntentAnalysisProtocol.intent_misalignments_key(),
                value={
                    "misalignments": misalignments,
                    "count": len(misalignments),
                    "min_severity": min_severity,
                },
                created_by=self.agent.agent_id,
            )

        return {
            "misalignments": misalignments,
            "count": len(misalignments),
        }

    @action_executor(action_key="get_intent_hierarchies")
    async def get_intent_hierarchies(self) -> dict[str, Any]:
        """Get all intent hierarchies built across pages.

        Returns:
            Dict with category hierarchies.
        """
        blackboard = await self.get_blackboard()

        # Find all hierarchy keys
        hierarchies = []
        # Note: In practice, would use blackboard pattern query
        # For now, return empty - the hierarchies are stored per-category

        return {
            "hierarchies": hierarchies,
            "count": len(hierarchies),
        }


# =============================================================================
# DEPRECATED: IntentCoordinatorCapability
# =============================================================================
# This class is superseded by IntentAnalysisCapability which extends
# VCMAnalysisCapability. The new design provides atomic, composable primitives
# that the LLM planner can compose into arbitrary workflows.
#
# Migration:
#   Old: IntentCoordinatorCapability with prescribed workflow
#   New: IntentAnalysisCapability with composable primitives
#
# TODO: Remove this class after migration is complete.
# =============================================================================

class IntentCoordinatorCapability(AgentCapability):
    """Capability for coordinating intent inference across multiple pages.

    Provides @action_executor methods for:
    - start_codebase_analysis: Begin analysis across pages using BatchingPolicy
    - collect_results: Gather results from worker agents via ResultCapability
    - merge_results: Combine results into unified graph

    Uses AgentPoolCapability for agent lifecycle management and PageGraphCapability
    for standardized graph operations.
    """


    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "intent_inference",
        input_patterns: list[str] = [
            AgentRunProtocol.request_pattern(),
            AgentRunProtocol.result_pattern()
        ],
        batching_policy: BatchingPolicy | None = None,
        capability_key: str = "intent_coordinator",
    ):
        """Initialize coordinator capability.

        Args:
            agent: Agent using this capability
            scope: Blackboard scope for this capability (default: COLONY)
            namespace: Namespace for this capability (default: "intent_inference")
            input_patterns: List of input patterns for this capability
            batching_policy: Policy for cache-aware batch selection
            capability_key: Unique key for this capability within the agent (default: "intent_coordinator")
        """
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=input_patterns, capability_key=capability_key)
        self.namespace = namespace
        self._worker_handles: dict[str, AgentHandle] = {}
        self._collected_results: list[ScopeAwareResult[IntentGraph]] = []
        self.max_agents: int = 10
        self._batching_policy = batching_policy

        # Layer 0/1 capability references (initialized in initialize())
        self._agent_pool_cap: AgentPoolCapability | None = None
        self._result_cap: ResultCapability | None = None
        self._page_graph_cap: PageGraphCapability | None = None

    def get_action_group_description(self) -> str:
        return (
            "Intent Coordination (DEPRECATED) — distributes intent inference across pages. "
            "Spawns agents via AgentPoolCapability, batches by cache affinity, "
            "collects and merges into unified intent graph."
        )

    async def initialize(self) -> None:
        """Initialize coordinator capability with Layer 0/1 capabilities."""
        await super().initialize()

        # AgentPoolCapability for agent lifecycle management
        self._agent_pool_cap = self.agent.get_capability_by_type(AgentPoolCapability)
        if not self._agent_pool_cap:
            self._agent_pool_cap = AgentPoolCapability(
                agent=self.agent,
                scope=BlackboardScope.COLONY,
                #max_agents=self.max_agents,
            )
            await self._agent_pool_cap.initialize()
            self.agent.add_capability(self._agent_pool_cap)

        # ResultCapability for cluster-wide result visibility
        self._result_cap = self.agent.get_capability_by_type(ResultCapability)
        if not self._result_cap:
            self._result_cap = ResultCapability(
                agent=self.agent,
                scope=BlackboardScope.COLONY,
            )
            await self._result_cap.initialize()
            self.agent.add_capability(self._result_cap)

        # PageGraphCapability for standardized graph operations
        self._page_graph_cap = self.agent.get_capability_by_type(PageGraphCapability)
        if not self._page_graph_cap:
            self._page_graph_cap = PageGraphCapability(
                agent=self.agent,
                scope=BlackboardScope.COLONY,
            )
            await self._page_graph_cap.initialize()
            self.agent.add_capability(self._page_graph_cap)

    def _get_merge_capability(self) -> MergeCapability | None:
        """Get MergeCapability from agent dynamically."""
        return self.agent.get_capability_by_type(MergeCapability)

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for IntentCoordinatorCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for IntentCoordinatorCapability")
        pass

    @event_handler(pattern=AgentRunProtocol.request_pattern())
    async def handle_analysis_request(
        self,
        event: BlackboardEvent,
        _repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Handle analysis request events from AgentHandle.run()."""
        request_data = event.value
        if not isinstance(request_data, dict):
            return None

        input_data = request_data.get("input", {})
        page_ids = input_data.get("page_ids", [])
        granularity = input_data.get("granularity", "function")

        parts = event.key.split(":")
        request_id = parts[-1] if len(parts) >= 3 else None

        return EventProcessingResult(
            immediate_action=Action(
                action_type="start_codebase_analysis",
                parameters={
                    "page_ids": page_ids,
                    "granularity": granularity,
                    "request_id": request_id,
                }
            )
        )

    @event_handler(pattern=AgentRunProtocol.result_pattern())
    async def handle_worker_result(
        self,
        event: BlackboardEvent,
        _repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Handle result from worker agent."""
        for page_id, handle in self._worker_handles.items():
            if event.key.startswith(f"{handle.agent_id}:result:"):
                result_data = event.value
                if isinstance(result_data, dict):
                    result = IntentInferenceResult(**result_data)
                    self._collected_results.append(result)
                    logger.info(f"Collected intent result for page {page_id}")

                    # Check if all results collected
                    if len(self._collected_results) >= len(self._worker_handles):
                        return EventProcessingResult(
                            immediate_action=Action(
                                action_type="finalize_analysis",
                                parameters={"request_id": self._pending_request_id}
                            )
                        )
                break

        return None

    @action_executor(action_key="start_codebase_analysis")
    async def start_codebase_analysis(
        self,
        page_ids: list[str],
        granularity: str = "function",
        request_id: str | None = None,
    ) -> str:
        """Start coordinated analysis across multiple pages.

        Uses BatchingPolicy for cache-aware batch selection and AgentPoolCapability
        for agent lifecycle management.

        Args:
            page_ids: VCM page IDs to analyze
            granularity: Analysis granularity
            request_id: Optional request ID

        Returns:
            Task ID for tracking
        """
        self._pending_request_id = request_id
        self._collected_results = []

        # Use BatchingPolicy for cache-aware batch selection if available
        if self._batching_policy and self._page_graph_cap:
            page_graph = await self._page_graph_cap.load_graph()
            batch_page_ids = await self._batching_policy.create_batch(
                candidate_pages=page_ids,
                page_graph=page_graph,
                max_batch_size=self.max_agents,
            )
        else:
            # Fallback to simple truncation
            batch_page_ids = page_ids[:self.max_agents]

        # Spawn worker agents using AgentPoolCapability
        await self._spawn_workers(batch_page_ids, granularity)

        # Run workers and collect results
        await self._run_workers_and_collect(granularity)

        # Merge and finalize
        final_result = await self._merge_and_finalize()

        # Write result to blackboard
        if request_id:
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=AgentRunProtocol.result_key(request_id),
                value=final_result.model_dump(),
                agent_id=self.agent.agent_id,
            )

        return request_id or "analysis_complete"

    async def _spawn_workers(self, page_ids: list[str], granularity: str) -> None:
        """Spawn worker agents for pages using AgentPoolCapability.

        Uses AgentPoolCapability for standardized agent lifecycle management,
        enabling cache-aware agent placement and resource optimization.
        """
        if not self._agent_pool_cap:
            raise RuntimeError("AgentPoolCapability not initialized")

        for page_id in page_ids:
            # Use AgentPoolCapability for standardized agent creation
            handle = await self._agent_pool_cap.create_agent(
                agent_type="polymathera.colony.samples.code_analysis.intent.IntentInferenceAgent",
                bound_pages=[page_id],
                capabilities=[IntentInferenceCapability],
                requirements=LLMClientRequirements(
                    model_family="llama",  # TODO: Make configurable
                    min_context_window=32000,  # TODO: Make configurable
                ),
                resource_requirements=AgentResourceRequirements(
                    cpu_cores=0.1,
                    memory_mb=512,
                    gpu_cores=0.0,
                    gpu_memory_mb=0
                ),
                metadata=AgentMetadata(
                    parameters={"page_id": page_id, "granularity": granularity}
                ),
            )
            self._worker_handles[page_id] = handle

    async def _run_workers_and_collect(self, granularity: str) -> None:
        """Run workers and collect results.

        Stores results via ResultCapability for cluster-wide visibility.
        """
        timeout = 60.0

        # TODO: This still sequentializes worker runs. For true parallelization,
        # use asyncio.gather with handle.run() calls or rely on event handlers.
        for page_id, handle in self._worker_handles.items():
            # protocol=AgentRunProtocol: worker's IntentInferenceCapability uses AgentRunProtocol
            # scope=AGENT: worker's IntentInferenceCapability uses AGENT scope
            # namespace="intent_inference": must match worker's IntentInferenceCapability namespace
            run = await handle.run(
                {"page_ids": [page_id], "granularity": granularity},
                timeout=timeout,
                protocol=AgentRunProtocol,
                scope=BlackboardScope.AGENT,
                namespace=self.namespace,  # "intent_inference" — matches worker
            )
            if run.result:
                result = IntentInferenceResult(**run.result)
                self._collected_results.append(result)

                # Store result via ResultCapability for cluster-wide visibility
                if self._result_cap:
                    await self._result_cap.store_partial(
                        result_id=f"intent:{page_id}",
                        result={
                            "page_id": page_id,
                            "intent_graph": result.content.model_dump() if result.content else {},
                            "scope": result.scope.model_dump() if result.scope else {},
                        },
                        source_agent=handle.agent_id,
                        source_pages=[page_id],
                        result_type="intent_inference",
                    )

                logger.info(f"Collected intent result for page {page_id}")

    async def _merge_and_finalize(self) -> ScopeAwareResult[IntentGraph]:
        """Merge collected results into final graph."""
        if not self._collected_results:
            return ScopeAwareResult(content=IntentGraph(), scope=AnalysisScope())

        if len(self._collected_results) == 1:
            return self._collected_results[0]

        merge_cap = self._get_merge_capability()
        if merge_cap is None:
            raise RuntimeError("IntentCoordinatorCapability requires MergeCapability")

        ### # Detect and resolve conflicts
        ### resolved_results = await self._resolve_conflicts(self._collected_results)

        merged = await merge_cap.merge_results(
            self._collected_results, # resolved_results
            MergeContext(prefer_higher_confidence=True)
        )

        ### # Detect misalignments
        ### await self._detect_misalignments(merged.content)

        # Build hierarchies across pages
        self._build_cross_page_hierarchies(merged.content)

        return merged

        ### return ScopeAwareResult(
        ###     content=merged.content,
        ###     scope=self._compute_scope(resolved_results)
        ### )

    async def _resolve_conflicts(
        self,
        results: list[ScopeAwareResult[IntentGraph]]
    ) -> list[ScopeAwareResult[IntentGraph]]:
        """Resolve conflicts between intent interpretations.

        Args:
            results: Intent graphs from agents

        Returns:
            Resolved intent graphs
        """
        # TODO: This should be part of the IntentMergePolicy.
        if len(results) <= 1:
            return results

        # Detect conflicts across graphs
        all_conflicts = []

        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                # Compare intents for same segments
                for segment_id in result1.content.nodes:
                    if segment_id in result2.content.nodes:
                        intent1 = result1.content.nodes[segment_id]
                        intent2 = result2.content.nodes[segment_id]

                        # Check for conflicts
                        agent1 = self.page_agents[list(self.page_agents.keys())[i]]
                        conflicts = await agent1.detect_conflicts(intent1, [intent2])
                        all_conflicts.extend(conflicts)

        # Run consensus game if conflicts exist
        if all_conflicts:
            logger.info(f"Resolving {len(all_conflicts)} intent conflicts")

            # Use ConsensusGameProtocol or negotiation
            # For now, use confidence-based resolution
            for conflict in all_conflicts:
                if conflict.severity == "high":
                    # High severity conflicts need consensus
                    await self._run_consensus_game(conflict, results)

        return results

    async def _run_consensus_game(
        self,
        conflict: IntentConflict,
        results: list[ScopeAwareResult[IntentGraph]]
    ) -> None:
        """Run consensus game to resolve conflict.

        Args:
            conflict: Intent conflict
            results: All intent graphs
        """
        # TODO: Move to IntentMergePolicy
        # TODO: Simplified consensus - would use ConsensusGameProtocol
        # For now, just log
        logger.info(f"Running consensus game for conflict: {conflict.conflict_type}")


    def _build_cross_page_hierarchies(self, graph: IntentGraph) -> None:
        """Build hierarchical relationships across pages.

        Args:
            graph: Intent graph to enhance
        """
        # Group intents by category
        by_category: dict[IntentCategory, list[str]] = {}

        for segment_id, intent in graph.nodes.items():
            for category in intent.categories:
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(segment_id)

        # Create category-level hierarchies
        for category, segment_ids in by_category.items():
            if len(segment_ids) > 1:
                # Create virtual parent node for category
                parent_id = f"category_{category.value}"
                graph.hierarchies[parent_id] = segment_ids

    async def _detect_misalignments(self, graph: IntentGraph) -> None:
        """Detect code-intent misalignments.

        Args:
            graph: Intent graph to analyze
        """
        # TODO: Move to IntentMergePolicy
        misalignments = []

        for segment_id, intent in graph.nodes.items():
            if intent.alignment == IntentAlignment.MISALIGNED:
                misalignments.append({
                    "segment": segment_id,
                    "file": intent.file_path,
                    "lines": f"{intent.line_start}-{intent.line_end}",
                    "issues": intent.issues,
                    "confidence": intent.confidence
                })

        if misalignments:
            logger.warning(f"Detected {len(misalignments)} code-intent misalignments")

            # Store in blackboard for further analysis
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=IntentAnalysisProtocol.intent_misalignments_key(),
                value=misalignments,
                tags={"misalignment", "intent"}
            )

    def _compute_scope(
        self,
        results: list[ScopeAwareResult[IntentGraph]]
    ) -> AnalysisScope:
        """Compute unified scope from results.

        Args:
            results: Intent analysis results

        Returns:
            Unified scope
        """
        # TODO: Move to IntentMergePolicy
        if not results:
            return AnalysisScope()

        # Merge scopes
        scopes = [r.scope for r in results]

        return AnalysisScope(
            is_complete=all(s.is_complete for s in scopes),
            missing_context=list(set(sum([s.missing_context for s in scopes], []))),
            confidence=sum(s.confidence for s in scopes) / len(scopes),
            reasoning=sum([s.reasoning for s in scopes], [])
        )

