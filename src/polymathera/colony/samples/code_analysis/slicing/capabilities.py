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
from typing import Any
from overrides import override

from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.agents.blackboard.protocol import AgentRunProtocol, SlicingAnalysisProtocol
from polymathera.colony.agents.patterns import (
    AnalysisScope,
    ScopeAwareResult,
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
from polymathera.colony.agents.base import Agent, AgentCapability, AgentRun, AgentHandle
from polymathera.colony.agents.models import Action, AgentMetadata, PolicyREPL, AgentResourceRequirements, AgentSuspensionState
from polymathera.colony.cluster.models import LLMClientRequirements
from .types import SliceType, SliceCriterion, SlicingResult, ProgramSlice, DependencyEdge


logger = logging.getLogger(__name__)



class ProgramSlicingCapability(AgentCapability):
    """Capability for computing program slices using LLM reasoning.

    This capability uses an LLM to:
    1. Identify data and control dependencies
    2. Trace variable flows
    3. Build approximate dependency graphs
    4. Extract minimal slices

    Integration:
    - Works with VCM pages for large files
    - Can handle interprocedural slicing
    - Provides confidence scores for approximate slices

    Works in two modes via the `scope_id` parameter:

    1. **Local mode** (in ProgramSlicingAgent): Processes slicing requests
       ```python
       capability = ProgramSlicingCapability(agent=self)  # scope_id = agent.agent_id
       ```

    2. **Remote mode** (in parent agent): Communicates with child ProgramSlicingAgent
       ```python
       handle = await parent.spawn_child_agents(...)[0]
       slicing_cap = handle.get_capability(ProgramSlicingCapability)
       future = await slicing_cap.get_result_future()
       result = await future
       ```

    Provides @action_executor methods for:
    - compute_slice: Compute program slice for a criterion
    - trace_dependencies: Trace dependencies for a variable
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "program_slicing",
        input_patterns: list[str] = [AgentRunProtocol.request_pattern()],
        interprocedural: bool = True,
        max_depth: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        capability_key: str = "program_slicing_capability"
    ):
        """Initialize slicing capability.

        Args:
            agent: Agent using this capability
            scope: Blackboard scope for this capability (default: AGENT)
            namespace: Namespace for event patterns (default: "program_slicing")
            input_patterns: List of event patterns to subscribe to
            interprocedural: Whether to follow function calls
            max_depth: Maximum call depth for interprocedural
            temperature: LLM temperature for inference calls
            max_tokens: Max tokens for LLM responses
            capability_key: Unique key for this capability within the agent
        """
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=input_patterns, capability_key=capability_key)
        self.interprocedural = interprocedural
        self.max_depth = max_depth
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_action_group_description(self) -> str:
        return (
            "Program Slicing — LLM-based approximation of static program slicing. "
            "Identifies data/control dependencies, traces variable flows, builds dependency graphs. "
            f"Interprocedural: {'enabled' if self.interprocedural else 'disabled'} (max_depth={self.max_depth}). "
            "Produces approximate slices with confidence scores. "
            "Supports local mode (single page) and remote mode (via AgentHandle)."
        )

    def _get_merge_capability(self) -> MergeCapability | None:
        """Get MergeCapability from agent dynamically."""
        return self.agent.get_capability_by_type(MergeCapability)

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        state = await super().serialize_suspension_state(state)
        state.custom_data["_program_slicing_state"] = {
            "interprocedural": self.interprocedural,
            "max_depth": self.max_depth,
        }
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        await super().deserialize_suspension_state(state)
        custom_state = state.custom_data.get("_program_slicing_state", {})
        if custom_state:
            self.interprocedural = custom_state.get("interprocedural", True)
            self.max_depth = custom_state.get("max_depth", 5)

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
        criterion_data = input_data.get("criterion")
        page_ids = input_data.get("page_ids", [])

        if not criterion_data:
            return None

        parts = event.key.split(":")
        request_id = parts[-1] if len(parts) >= 3 else None

        return EventProcessingResult(
            immediate_action=Action(
                action_id=f"compute_slice_{request_id}",
                agent_id=self.agent.agent_id,
                action_type="compute_slice",
                parameters={
                    "criterion": criterion_data,
                    "page_ids": page_ids,
                    "request_id": request_id,
                }
            )
        )

    @action_executor(action_key="compute_slice")
    async def compute_slice(
        self,
        criterion: SliceCriterion | dict[str, Any],
        page_ids: list[str],
        request_id: str | None = None,
    ) -> SlicingResult:
        """Compute program slice for criterion.

        This is an LLM-plannable action.

        Args:
            criterion: Slicing criterion (can be dict or SliceCriterion)
            page_ids: VCM page IDs to analyze
            request_id: Optional request ID for blackboard response

        Returns:
            Program slice with dependencies
        """
        # Parse criterion if dict
        if isinstance(criterion, dict):
            criterion = SliceCriterion(**criterion)

        # Build dependency analysis prompt (VCM will load pages)
        prompt = self._build_slicing_prompt(criterion)

        # Get LLM analysis using VCM pages with structured output
        response = await self.agent.infer(
            context_page_ids=page_ids,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_schema=ProgramSlice.model_json_schema()  # Structured output
        )

        # Parse structured response
        try:
            slice_data = ProgramSlice.model_validate_json(response.generated_text)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as ProgramSlice: {e}")
            slice_data = ProgramSlice(
                criterion=criterion,
                reasoning=f"LLM response parsing failed: {e}",
            )

        # If interprocedural, expand slice
        if self.interprocedural and slice_data.interprocedural:
            slice_data = await self._expand_interprocedural(
                slice_data,
                page_ids
            )

        # Build result with scope
        scope = AnalysisScope(
            is_complete=self._check_completeness(slice_data),
            missing_context=self._identify_missing(slice_data),
            confidence=self._compute_confidence(slice_data),
            reasoning=[slice_data.reasoning]
        )

        result = SlicingResult(
            content=slice_data,
            scope=scope
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

    def _build_slicing_prompt(self, criterion: SliceCriterion) -> str:
        """Build prompt for LLM slicing analysis.

        Args:
            criterion: Slicing criterion

        Returns:
            Prompt for LLM
        """
        slice_type_desc = {
            SliceType.BACKWARD: "all statements that could affect",
            SliceType.FORWARD: "all statements that could be affected by",
            SliceType.CHOPPING: "the flow between source and",
            SliceType.DYNAMIC: "statements executed when reaching",
            SliceType.CONDITIONED: "statements under specific conditions for"
        }

        return f"""Compute a program slice for the following criterion:

Criterion:
- Variable: {criterion.variable or "N/A"}
- Expression: {criterion.expression or "N/A"}
- Location: {criterion.file_path}:{criterion.line_number}
- Slice Type: {criterion.slice_type}

Analyze the code loaded in context and identify {slice_type_desc[criterion.slice_type]} the criterion.

Return a JSON object with:
- criterion: the slicing criterion
- included_lines: dict mapping file paths to lists of line numbers included in slice
- excluded_lines: dict mapping file paths to lists of line numbers excluded
- dependencies: list of dependency edges (from_line, to_line, dep_type, variable, condition)
- entry_points: list of entry points into the slice
- exit_points: list of exit points from the slice
- interprocedural: boolean indicating if slice crosses function boundaries
- reasoning: explanation of slicing decisions

Analyze:
1. Data dependencies (which lines define/use relevant variables)
2. Control dependencies (which conditions affect execution)
3. Call dependencies (which functions are involved)

For each line, determine if it should be INCLUDED or EXCLUDED from the slice, and provide an explanation of why these lines were selected.

Provide your analysis in this format:
DEPENDENCIES:
- Line X → Line Y: [data/control/call] dependency via [variable/condition]

INCLUDED LINES: [list of line numbers that must be in slice]
EXCLUDED LINES: [list of line numbers that can be removed]

REASONING: [explanation of why these lines were selected]

Be precise and minimize the slice while preserving correctness.
Output ONLY valid JSON matching the ProgramSlice schema."""

    async def _expand_interprocedural(
        self,
        slice_data: ProgramSlice,
        page_ids: list[str]
    ) -> ProgramSlice:
        """Expand slice across function boundaries using LLM analysis.

        For each call dependency in the slice, loads the target page and
        asks the LLM which lines of the called function affect the slice
        criterion. Merges discovered lines and dependencies into the slice.

        Args:
            slice_data: Initial slice with call dependencies
            page_ids: VCM page IDs available for expansion

        Returns:
            Expanded slice with interprocedural lines and dependencies
        """
        # Find function calls in slice
        call_deps = [d for d in slice_data.dependencies if d.dep_type == "call"]

        if not call_deps:
            return slice_data

        expanded = slice_data.model_copy(deep=True)

        # For each call, analyze the called function
        # (Simplified - would need more sophisticated call resolution)
        for dep in call_deps[:self.max_depth]:
            callee = dep.variable or f"function at line {dep.to_line}"

            prompt = f"""The program slice includes a call to '{callee}' at line {dep.from_line}.

Analyze the called function in context and identify:
1. Which lines of the called function affect the return value or have side effects relevant to the caller?
2. What data dependencies exist within the called function?

Return JSON with:
- included_lines: list of line numbers in the called function that should be in the slice
- dependencies: list of {{from_line, to_line, dep_type, variable}} within the called function"""

            try:
                response = await self.agent.infer(
                    context_page_ids=page_ids,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                import json
                expansion = json.loads(response.generated_text)

                # Merge expanded lines
                for file_path, lines in slice_data.included_lines.items():
                    new_lines = expansion.get("included_lines", [])
                    if new_lines:
                        current = set(expanded.included_lines.get(file_path, []))
                        current.update(new_lines)
                        expanded.included_lines[file_path] = sorted(current)

                # Merge expanded dependencies
                for new_dep in expansion.get("dependencies", []):
                    expanded.dependencies.append(DependencyEdge(
                        from_line=new_dep.get("from_line", 0),
                        to_line=new_dep.get("to_line", 0),
                        dep_type=new_dep.get("dep_type", "data"),
                        variable=new_dep.get("variable"),
                        confidence=0.7,
                    ))

            except Exception as e:
                logger.debug(f"Interprocedural expansion failed for {callee}: {e}")

        return expanded

    def _check_completeness(self, slice_data: ProgramSlice) -> bool:
        """Check if slice is complete.

        Args:
            slice_data: Computed slice

        Returns:
            True if complete
        """
        # Slice is complete if interprocedural analysis is not needed
        return not slice_data.interprocedural or len(slice_data.dependencies) > 0

    def _identify_missing(self, slice_data: ProgramSlice) -> list[str]:
        """Identify missing context for slice.

        Args:
            slice_data: Computed slice

        Returns:
            List of missing elements
        """
        missing = []

        # Check for unresolved function calls
        if slice_data.interprocedural:
            for dep in slice_data.dependencies:
                if dep.dep_type == "call" and dep.confidence < 0.8:
                    missing.append(f"Function definition at line {dep.to_line}")

        return missing

    def _compute_confidence(self, slice_data: ProgramSlice) -> float:
        """Compute overall confidence in slice.

        Args:
            slice_data: Computed slice

        Returns:
            Confidence score
        """
        if not slice_data.dependencies:
            return 0.5

        # Average confidence of dependencies
        total_conf = sum(d.confidence for d in slice_data.dependencies)
        return total_conf / len(slice_data.dependencies)


class SliceMergePolicy(MergePolicy[ProgramSlice]):
    """Policy for merging program slices from multiple analyses."""

    async def merge(
        self,
        results: list[ScopeAwareResult[ProgramSlice]],
        context: MergeContext
    ) -> ScopeAwareResult[ProgramSlice]:
        """Merge multiple slice results.

        Args:
            results: Slice results to merge
            context: Merge context

        Returns:
            Merged slice
        """
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            return results[0]

        # Start with first slice
        merged = results[0].content

        # Union all included lines
        all_included: dict[str, set[int]] = {}
        all_excluded: dict[str, set[int]] = {}
        all_deps: list[DependencyEdge] = []

        for result in results:
            slice_data = result.content

            # Merge included lines
            for file, lines in slice_data.included_lines.items():
                if file not in all_included:
                    all_included[file] = set()
                all_included[file].update(lines)

            # Merge excluded lines (intersection - only exclude if all agree)
            for file, lines in slice_data.excluded_lines.items():
                if file not in all_excluded:
                    all_excluded[file] = set(lines)
                else:
                    all_excluded[file].intersection_update(lines)

            # Merge dependencies
            all_deps.extend(slice_data.dependencies)

        # Deduplicate dependencies
        unique_deps = []
        seen = set()
        for dep in all_deps:
            key = (dep.from_line, dep.to_line, dep.dep_type)
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)

        # Create merged slice
        merged_slice = ProgramSlice(
            criterion=merged.criterion,
            included_lines={f: list(lines) for f, lines in all_included.items()},
            excluded_lines={f: list(lines) for f, lines in all_excluded.items()},
            dependencies=unique_deps,
            entry_points=list(set(sum([r.content.entry_points for r in results], []))),
            exit_points=list(set(sum([r.content.exit_points for r in results], []))),
            interprocedural=any(r.content.interprocedural for r in results),
            reasoning=self._merge_reasoning(results)
        )

        # Merge scopes
        merged_scope = AnalysisScope(
            is_complete=all(r.scope.is_complete for r in results),
            missing_context=list(set(sum([r.scope.missing_context for r in results], []))),
            confidence=sum(r.scope.confidence for r in results) / len(results),
            reasoning=sum([r.scope.reasoning for r in results], [])
        )

        return SlicingResult(content=merged_slice, scope=merged_scope)

    def _merge_reasoning(self, results: list[ScopeAwareResult[ProgramSlice]]) -> str:
        """Merge reasoning from multiple slices.

        Args:
            results: Results with reasoning

        Returns:
            Combined reasoning
        """
        reasonings = [r.content.reasoning for r in results if r.content.reasoning]
        if len(reasonings) == 1:
            return reasonings[0]
        return "Combined analysis: " + " | ".join(reasonings)

    async def validate(
        self,
        original: list[ScopeAwareResult[ProgramSlice]],
        merged: ScopeAwareResult[ProgramSlice]
    ) -> ValidationResult:
        """Validate merged slice.

        Args:
            original: Original slices
            merged: Merged slice

        Returns:
            Validation result
        """
        issues = []

        # Check that merged includes all originally included lines
        for orig in original:
            for file, lines in orig.content.included_lines.items():
                merged_lines = set(merged.content.included_lines.get(file, []))
                orig_lines = set(lines)
                if not orig_lines.issubset(merged_lines):
                    missing = orig_lines - merged_lines
                    issues.append(f"Merged slice missing lines {missing} from {file}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if not issues else 0.7
        )


# =============================================================================
# SlicingAnalysisCapability - Extends VCMAnalysisCapability
# =============================================================================
#
# This capability extends VCMAnalysisCapability to provide atomic, composable
# primitives for program slicing across multiple pages.
#
# Design Philosophy:
# - Does NOT prescribe a workflow
# - LLM decides when/how to spawn workers, resolve dependencies, merge results
# - Domain-specific methods exposed as action_executors for LLM composition
#
# Example LLM-driven workflow:
#   1. LLM: spawn_workers(page_ids, cache_affine=True)
#   2. LLM: [wait for completion events]
#   3. LLM: merge_results(page_ids)
#   4. LLM: resolve_interprocedural_dependencies()  # domain-specific
#   5. LLM: get_slice_for_criterion(criterion)  # domain-specific query
# =============================================================================


class SlicingAnalysisCapability(VCMAnalysisCapability):
    """Capability for distributed program slicing using VCMAnalysisCapability primitives.

    Extends VCMAnalysisCapability with slicing-specific:
    - Worker type (ProgramSlicingCapability)
    - Merge policy (SliceMergePolicy)
    - Domain methods (interprocedural resolution, slice queries)

    The LLM planner composes the atomic primitives from the base class with
    the domain-specific methods exposed here.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "slicing_analysis",
        capability_key: str = "slicing_analysis_capability"
    ):
        """Initialize slicing analysis capability.

        Args:
            agent: Agent using this capability (coordinator agent)
            scope: Blackboard scope
            namespace: Namespace for blackboard events
            capability_key: Unique key for this capability within the agent
        """
        super().__init__(agent=agent, scope=scope, namespace=namespace, capability_key=capability_key)
        self._dependency_graph: dict[str, list[str]] = {}  # location -> dependent pages

    # =========================================================================
    # Abstract Hook Implementations
    # =========================================================================

    def get_worker_capability_class(self) -> type[AgentCapability]:
        """Return ProgramSlicingCapability as the worker capability."""
        return ProgramSlicingCapability

    def get_worker_agent_type(self) -> str:
        """Return the worker agent type string."""
        return "polymathera.colony.samples.code_analysis.slicing.ProgramSlicingAgent"

    def get_domain_merge_policy(self) -> MergePolicy:
        """Return SliceMergePolicy for merging program slices."""
        return SliceMergePolicy()

    def get_analysis_parameters(self, **kwargs) -> dict[str, Any]:
        """Return slicing-specific analysis parameters.

        Args:
            **kwargs: Parameters from action call

        Returns:
            Parameters for slicing workers
        """
        return {
            "slice_type": kwargs.get("slice_type", "backward"),
            "include_control": kwargs.get("include_control", True),
            "include_data": kwargs.get("include_data", True),
            **kwargs,
        }

    # =========================================================================
    # Domain-Specific Action Executors
    # =========================================================================

    @action_executor(action_key="resolve_interprocedural")
    async def resolve_interprocedural(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Resolve interprocedural dependencies across page slices.

        Call after merge_results() to resolve cross-page dependencies.
        LLM decides when this is appropriate (e.g., when slices span multiple files).

        Args:
            page_ids: Pages to consider (if None, uses all analyzed pages)

        Returns:
            Dict with:
            - resolutions: Dict mapping location -> {defining_page, dependent_pages}
            - unresolved: List of unresolved external dependencies
            - dependency_count: Total dependencies found
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        resolutions = {}
        unresolved = []

        # Build dependency graph from results
        self._dependency_graph.clear()

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            dependencies = content.get("dependencies", [])

            for dep in dependencies:
                is_external = dep.get("is_external", False)
                location = dep.get("location") or dep.get("to_line")

                if is_external and location:
                    if location not in self._dependency_graph:
                        self._dependency_graph[location] = []
                    self._dependency_graph[location].append(page_id)

        # Resolve references between pages
        for location, dependent_pages in self._dependency_graph.items():
            defining_page = await self._find_defining_page(location, page_ids)

            if defining_page:
                resolutions[location] = {
                    "defining_page": defining_page,
                    "dependent_pages": dependent_pages,
                }
            else:
                unresolved.append({
                    "location": location,
                    "dependent_pages": dependent_pages,
                })

        logger.info(
            f"SlicingAnalysisCapability: resolved {len(resolutions)} interprocedural deps, "
            f"{len(unresolved)} unresolved"
        )

        # Store resolutions in blackboard
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=SlicingAnalysisProtocol.interprocedural_resolutions_key(),
            value={
                "resolutions": resolutions,
                "unresolved": unresolved,
            },
            created_by=self.agent.agent_id,
        )

        return {
            "resolutions": resolutions,
            "unresolved": unresolved,
            "dependency_count": len(self._dependency_graph),
        }

    async def _find_defining_page(
        self,
        location: str,
        page_ids: list[str],
    ) -> str | None:
        """Find the page containing the definition for a location.

        Args:
            location: Location string (file:line or function name)
            page_ids: Pages to search

        Returns:
            Page ID containing the definition, or None
        """
        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})

            # Check if this page contains the definition
            entry_points = content.get("entry_points", [])
            included_lines = content.get("included_lines", {})

            # Check entry points
            if location in entry_points:
                return page_id

            # Check if location is in included lines
            if ":" in location:
                file_path, line_str = location.rsplit(":", 1)
                try:
                    line_num = int(line_str)
                    if file_path in included_lines:
                        if line_num in included_lines[file_path]:
                            return page_id
                except ValueError:
                    pass

        return None

    @action_executor(action_key="get_slice_dependencies")
    async def get_slice_dependencies(
        self,
        page_id: str,
    ) -> dict[str, Any]:
        """Get all dependencies for a specific page's slice.

        Args:
            page_id: Page to get dependencies for

        Returns:
            Dict with dependency information
        """
        result_data = await self.get_result(page_id)

        if not result_data.get("found"):
            return {
                "page_id": page_id,
                "found": False,
                "dependencies": [],
            }

        result = result_data.get("result", {})
        content = result.get("result", {}).get("content", {})

        dependencies = content.get("dependencies", [])
        entry_points = content.get("entry_points", [])
        exit_points = content.get("exit_points", [])

        return {
            "page_id": page_id,
            "found": True,
            "dependencies": dependencies,
            "entry_points": entry_points,
            "exit_points": exit_points,
            "is_interprocedural": content.get("interprocedural", False),
        }

    @action_executor(action_key="get_external_dependencies")
    async def get_external_dependencies(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get all external dependencies across analyzed pages.

        LLM can use this to decide if interprocedural resolution is needed.

        Args:
            page_ids: Pages to search (if None, uses all analyzed pages)

        Returns:
            Dict with external dependencies
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        external_deps = []

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            dependencies = content.get("dependencies", [])

            for dep in dependencies:
                if dep.get("is_external", False):
                    external_deps.append({
                        "page_id": page_id,
                        "location": dep.get("location") or dep.get("to_line"),
                        "dep_type": dep.get("dep_type"),
                    })

        return {
            "external_dependencies": external_deps,
            "count": len(external_deps),
            "needs_interprocedural": len(external_deps) > 0,
        }



# SlicingCoordinatorCapability has been removed.
# Use SlicingAnalysisCapability (VCMAnalysisCapability subclass) instead.
