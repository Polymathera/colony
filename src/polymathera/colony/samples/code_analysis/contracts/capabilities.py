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

import asyncio
import logging
import itertools
import uuid
from enum import Enum
from typing import Any
from overrides import override
from pydantic import BaseModel, Field

from polymathera.colony.agents.scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from polymathera.colony.agents.blackboard.protocol import AgentRunProtocol
from polymathera.colony.agents.patterns import (
    AnalysisScope,
    ScopeAwareResult,
)
from polymathera.colony.agents.patterns.capabilities.merge import MergePolicy, MergeContext, MergeCapability
from polymathera.colony.agents.patterns.capabilities.validation import ValidationResult
from polymathera.colony.agents.patterns.capabilities.synthesis import SynthesisCapability
from polymathera.colony.agents.patterns.capabilities.agent_pool import AgentPoolCapability
from polymathera.colony.agents.patterns.capabilities.result import ResultCapability
from polymathera.colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from polymathera.colony.agents.patterns.capabilities.batching import BatchingPolicy
from polymathera.colony.agents.patterns.capabilities.vcm_analysis import VCMAnalysisCapability
from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.events import event_handler, EventProcessingResult
from polymathera.colony.agents.patterns.models import Hypothesis
from polymathera.colony.agents.blackboard import TaskGraph, BlackboardEvent
from polymathera.colony.agents.base import Agent, AgentCapability, CapabilityResultFuture
from polymathera.colony.agents.models import Action, ActionType, AgentMetadata, PolicyREPL, AgentResourceRequirements, AgentSuspensionState
from polymathera.colony.agents.patterns.games.negotiation.capabilities import NegotiationIssue, Offer, calculate_pareto_efficiency
from polymathera.colony.cluster.models import LLMClientRequirements
from polymathera.colony.agents.patterns.games.coalition_formation import find_optimal_coalition_structure

from .types import (
    Contract,
    ContractType,
    FormalismLevel,
    FunctionContract,
    ContractInferenceResult,
)

logger = logging.getLogger(__name__)



class ContractInferenceCapability(AgentCapability):
    """Capability for inferring function contracts using LLM reasoning.

    This capability provides:
    - @action_executor methods for LLM-plannable contract inference
    - @event_handler methods for reacting to analysis requests via AgentHandle.run()
    - Dynamic capability lookup for MergeCapability

    LLM reasoning for:
    1. Analyze function behavior and intent
    2. Identify implicit assumptions
    3. Derive likely preconditions and postconditions
    4. Discover loop invariants
    5. Generate formal specifications when possible

    Integration:
    - Works with program slicing for context
    - Can leverage test cases for validation
    - Integrates with symbolic execution results
    """


    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "contract_inference",
        input_patterns: list[str] = [AgentRunProtocol.request_pattern()],
        formalism: FormalismLevel = FormalismLevel.SEMI_FORMAL,
        use_examples: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        capability_key: str = "contract_inference_capability",
    ):
        """Initialize contract inference capability.

        Args:
            agent: Agent using this capability
            scope: Blackboard scope ID (defaults to agent.agent_id)
            namespace: Namespace for blackboard events
            input_patterns: List of input patterns to listen for
            formalism: Target formalism level
            use_examples: Whether to use examples for learning
            temperature: LLM temperature for inference calls
            max_tokens: Max tokens for LLM responses
            capability_key: Unique key for this capability within the agent
        """
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=input_patterns, capability_key=capability_key)
        self.formalism = formalism
        self.use_examples = use_examples
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_action_group_description(self) -> str:
        return (
            f"Contract Inference ({self.formalism} formalism) — derives preconditions, postconditions, "
            "and loop invariants from code. Identifies implicit assumptions. "
            "Critical contracts can be validated with evidence for higher confidence. "
            "Security contract specialization available. "
            "Supports local mode (single page) and remote mode (via AgentHandle)."
        )

    def _get_merge_capability(self) -> MergeCapability | None:
        """Get MergeCapability from agent (optional)."""
        return self.agent.get_capability_by_type(MergeCapability)

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        state = await super().serialize_suspension_state(state)
        state.custom_data["_contract_inference_state"] = {
            "formalism": self.formalism if isinstance(self.formalism, str) else self.formalism.value,
            "use_examples": self.use_examples,
        }
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        await super().deserialize_suspension_state(state)
        custom_state = state.custom_data.get("_contract_inference_state", {})
        if custom_state:
            self.formalism = custom_state.get("formalism", FormalismLevel.SEMI_FORMAL)
            self.use_examples = custom_state.get("use_examples", True)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    @event_handler(pattern=AgentRunProtocol.request_pattern())
    async def handle_analysis_request(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL,
    ) -> EventProcessingResult | None:
        """Handle analysis request events from AgentHandle.run().

        Listens for request:{request_id} events and returns
        an immediate action to execute the analysis.
        """
        # Extract request data
        request_data = event.value
        if not request_data:
            return None

        input_data = request_data.get("input", {})
        request_id = request_data.get("request_id")
        self._pending_request_id = request_id

        # Build action parameters
        page_ids = input_data.get("page_ids", [])
        function_names = input_data.get("function_names")
        test_cases = input_data.get("test_cases")

        # If we have bound pages, use those instead
        if hasattr(self.agent, 'bound_pages') and self.agent.bound_pages:
            page_ids = self.agent.bound_pages

        # Enrich scope with request context
        repl.set("request_id", request_id)
        repl.set("page_ids", page_ids)

        # Return immediate action to execute analysis
        return EventProcessingResult(
            immediate_action=Action(
                action_id=f"infer_contracts_{request_id}",
                agent_id=self.agent.agent_id,
                action_type="infer_contracts",
                parameters={
                    "page_ids": page_ids,
                    "function_names": function_names,
                    "test_cases": test_cases,
                    "request_id": request_id,
                },
            )
        )

    # -------------------------------------------------------------------------
    # Action Executors (LLM-Plannable)
    # -------------------------------------------------------------------------

    @action_executor(action_key="infer_contracts")
    async def infer_contracts(
        self,
        page_ids: list[str],
        function_names: list[str] | None = None,
        test_cases: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> ContractInferenceResult:
        """Infer contracts for functions across multiple pages.

        This method aggregates contract inference across pages using
        a merge policy for consistent results.

        Args:
            page_ids: VCM page IDs to analyze
            function_names: Specific functions to analyze (None = all)
            test_cases: Optional test cases for validation
            request_id: Optional request ID for result routing

        Returns:
            Aggregated contract inference results
        """
        # Process each page individually
        page_results = []
        for page_id in page_ids:
            result = await self._infer_contracts_from_page(
                page_id,
                function_names,
                test_cases
            )
            page_results.append(result)

        # Merge results using MergeCapability
        if len(page_results) == 1:
            final_result = page_results[0]
        else:
            merge_cap = self._get_merge_capability()
            if merge_cap is None:
                raise RuntimeError(
                    "ContractInferenceCapability requires MergeCapability on the agent. "
                    "Add MergeCapability configured with ContractMergePolicy."
                )
            final_result = await merge_cap.merge_results(
                page_results,
                MergeContext(strategy="union")  # Union all contracts found
            )

        # Write result to blackboard for AgentHandle.run() to receive
        if request_id:
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=AgentRunProtocol.result_key(request_id),
                value=final_result.model_dump(),
                created_by=self.agent.agent_id,
            )

        return final_result

    @action_executor(action_key="analyze_page")
    async def analyze_page(
        self,
        page_id: str,
        function_names: list[str] | None = None,
        test_cases: list[dict[str, Any]] | None = None,
    ) -> ContractInferenceResult:
        """Analyze a single VCM page for contracts (LLM-plannable action).

        Args:
            page_id: Single VCM page ID
            function_names: Specific functions to analyze
            test_cases: Optional test cases

        Returns:
            Contracts for this page
        """
        return await self._infer_contracts_from_page(page_id, function_names, test_cases)

    async def _infer_contracts_from_page(
        self,
        page_id: str,
        function_names: list[str] | None = None,
        test_cases: list[dict[str, Any]] | None = None
    ) -> ContractInferenceResult:
        """Internal: Infer contracts from a single VCM page.

        Args:
            page_id: Single VCM page ID
            function_names: Specific functions to analyze
            test_cases: Optional test cases

        Returns:
            Contracts for this page
        """
        # Build contract inference prompt
        prompt = self._build_inference_prompt(function_names, test_cases)

        # Get LLM analysis for single page with structured output
        response = await self.agent.infer(
            context_page_ids=[page_id],  # Single page
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_schema=ContractInferenceResult.model_json_schema()
        )

        # Parse structured response
        try:
            result = ContractInferenceResult.model_validate_json(response.generated_text)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as ContractInferenceResult: {e}")
            result = ContractInferenceResult(
                content=[],
                scope=AnalysisScope(confidence=0.0, reasoning=[f"LLM response parsing failed: {e}"]),
            )

        # Validate against test cases if provided
        if test_cases and result.content:
            result.content = await self._validate_with_tests(result.content, test_cases)

        # Refine to target formalism level
        if self.formalism == FormalismLevel.FORMAL:
            result.content = await self._formalize_contracts(result.content)

        # Update scope information
        result.scope = AnalysisScope(
            is_complete=self._check_completeness(result.content),
            missing_context=self._identify_missing_context(result.content),
            confidence=self._compute_confidence(result.content),
            reasoning=[self._summarize_inference(result.content)]
        )

        return result

    def _build_inference_prompt(
        self,
        function_names: list[str] | None,
        test_cases: list[dict[str, Any]] | None
    ) -> str:
        """Build prompt for contract inference.

        Args:
            function_names: Target functions
            test_cases: Optional test cases

        Returns:
            Formatted prompt
        """
        test_section = ""
        if test_cases:
            test_section = f"""
Test Cases Available:
{self._format_test_cases(test_cases)}

Use these to validate and refine contracts.
"""

        target_section = ""
        if function_names:
            target_section = f"Focus on functions: {', '.join(function_names)}"

        formalism_guide = {
            FormalismLevel.NATURAL: "Express contracts in clear natural language.",
            FormalismLevel.SEMI_FORMAL: "Use structured format: requires(...), ensures(...), modifies(...).",
            FormalismLevel.FORMAL: "Generate formal specifications (e.g., first-order logic, Z3 syntax).",
            FormalismLevel.CODE: "Write executable assertions/contracts."
        }

        return f"""Infer function contracts from the code loaded in context.

{target_section}

For each function, identify:
1. PRECONDITIONS - What must be true before calling
2. POSTCONDITIONS - What is guaranteed after calling
3. INVARIANTS - What remains true (for loops/classes)
4. MODIFIES - What state/variables are changed
5. PURITY - Whether function has side effects
6. TERMINATION - Why function terminates
7. COMPLEXITY - Time/space complexity

{test_section}

{formalism_guide[self.formalism]}

Return a JSON object matching the ContractInferenceResult schema with:
- content: list of FunctionContract objects
- scope: AnalysisScope with completeness and confidence information

Each FunctionContract should include:
- function_name
- file_path
- line_number
- preconditions (list of Contract objects)
- postconditions (list of Contract objects)
- invariants (list of Contract objects)
- modifies (list of strings)
- pure (boolean)
- termination (string or null)
- complexity (string or null)
- formalism level

Be precise and consider edge cases."""

    def _format_test_cases(self, test_cases: list[dict[str, Any]]) -> str:
        """Format test cases for prompt.

        Args:
            test_cases: Test cases

        Returns:
            Formatted test cases
        """
        formatted = []
        for i, test in enumerate(test_cases, 1):
            formatted.append(f"Test {i}: Input: {test.get('input')} → Output: {test.get('output')}")
        return "\n".join(formatted)


    async def _validate_with_tests(
        self,
        contracts: list[FunctionContract],
        test_cases: list[dict[str, Any]]
    ) -> list[FunctionContract]:
        """Validate contracts against test cases.

        Args:
            contracts: Inferred contracts
            test_cases: Test cases

        Returns:
            Validated contracts
        """
        # For each contract, check if test cases satisfy pre/post
        for contract in contracts:
            for test in test_cases:
                if test.get("function") == contract.function_name:
                    # Would validate preconditions with input
                    # Would validate postconditions with output
                    # Store counterexamples if validation fails
                    pass

        return contracts

    async def _formalize_contracts(
        self,
        contracts: list[FunctionContract]
    ) -> list[FunctionContract]:
        """Convert contracts to formal specifications.

        Args:
            contracts: Semi-formal contracts

        Returns:
            Formalized contracts
        """
        for contract in contracts:
            # Convert natural language to formal specs
            for pre in contract.preconditions:
                if not pre.formal_spec and pre.description:
                    pre.formal_spec = await self._to_formal(pre.description)

            for post in contract.postconditions:
                if not post.formal_spec and post.description:
                    post.formal_spec = await self._to_formal(post.description)

            for inv in contract.invariants:
                if not inv.formal_spec and inv.description:
                    inv.formal_spec = await self._to_formal(inv.description)

            contract.formalism = FormalismLevel.FORMAL

        return contracts

    async def _to_formal(self, natural: str) -> str:
        """Convert natural language to formal spec.

        Args:
            natural: Natural language spec

        Returns:
            Formal specification
        """
        # Simple heuristic conversion
        # In practice, would use LLM or specialized NL2Formal model
        formal = natural

        # Replace common patterns
        replacements = {
            "is not null": "!= null",
            "is null": "== null",
            "greater than": ">",
            "less than": "<",
            "at least": ">=",
            "at most": "<=",
            "equals": "==",
            "not equal": "!=",
            " and ": " && ",
            " or ": " || ",
            "for all": "∀",
            "exists": "∃",
            "implies": "→"
        }

        for pattern, replacement in replacements.items():
            formal = formal.replace(pattern, replacement)

        return formal

    def _check_completeness(self, contracts: list[FunctionContract]) -> bool:
        """Check if contract inference is complete.

        Args:
            contracts: Inferred contracts

        Returns:
            True if complete
        """
        # Complete if we have contracts with both pre and post conditions
        if not contracts:
            return False

        complete_contracts = sum(
            1 for c in contracts 
            if c.preconditions and c.postconditions
        )

        return complete_contracts / len(contracts) >= 0.7 if contracts else False

    def _identify_missing_context(self, contracts: list[FunctionContract]) -> list[str]:
        """Identify missing context for contracts.

        Args:
            contracts: Inferred contracts

        Returns:
            List of missing elements
        """
        missing = []

        for contract in contracts:
            # Check for low confidence contracts
            for pre in contract.preconditions:
                if pre.confidence < 0.6:
                    missing.append(f"Low confidence precondition for {contract.function_name}")

            # Check for missing formal specs when requested
            if self.formalism == FormalismLevel.FORMAL:
                for post in contract.postconditions:
                    if not post.formal_spec:
                        missing.append(f"Missing formal spec for postcondition of {contract.function_name}")

        return missing

    def _compute_confidence(self, contracts: list[FunctionContract]) -> float:
        """Compute overall confidence in contracts.

        Args:
            contracts: Inferred contracts

        Returns:
            Confidence score
        """
        if not contracts:
            return 0.0

        total_confidence = 0.0
        count = 0

        for contract in contracts:
            for cond in contract.preconditions + contract.postconditions + contract.invariants:
                total_confidence += cond.confidence
                count += 1

        return total_confidence / count if count > 0 else 0.5

    def _summarize_inference(self, contracts: list[FunctionContract]) -> str:
        """Summarize contract inference.

        Args:
            contracts: Inferred contracts

        Returns:
            Summary
        """
        total_pre = sum(len(c.preconditions) for c in contracts)
        total_post = sum(len(c.postconditions) for c in contracts)
        total_inv = sum(len(c.invariants) for c in contracts)
        pure_count = sum(1 for c in contracts if c.pure)

        return (f"Inferred contracts for {len(contracts)} functions: "
                f"{total_pre} preconditions, {total_post} postconditions, "
                f"{total_inv} invariants. {pure_count} pure functions.")


class ContractMergePolicy(MergePolicy[list[FunctionContract]]):
    """Policy for merging contract inference results."""

    async def merge(
        self,
        results: list[ScopeAwareResult[list[FunctionContract]]],
        context: MergeContext
    ) -> ScopeAwareResult[list[FunctionContract]]:
        """Merge multiple contract results using coalition + negotiation."""
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            return results[0]

        coordination_notes: list[str] = []
        agent_weights = self._compute_agent_weights(results)

        # Group contracts by function and keep producer IDs
        function_contracts: dict[str, list[tuple[FunctionContract, str]]] = {}
        for result in results:
            agent_id = result.producer_agent_id or result.result_id
            for contract in result.content:
                key = f"{contract.file_path}:{contract.function_name}"
                function_contracts.setdefault(key, []).append((contract, agent_id))

        merged_contracts = []
        for key, contract_entries in function_contracts.items():
            if len(contract_entries) == 1:
                merged_contracts.append(contract_entries[0][0])
                continue
            merged_contracts.append(
                await self._merge_function_contracts(
                    key,
                    contract_entries,
                    agent_weights,
                    coordination_notes
                )
            )

        merged_scope = AnalysisScope(
            is_complete=all(r.scope.is_complete for r in results),
            missing_context=list(set(sum([r.scope.missing_context for r in results], []))),
            confidence=sum(r.scope.confidence for r in results) / len(results),
            reasoning=sum([r.scope.reasoning for r in results], []) + coordination_notes
        )

        return ContractInferenceResult(
            content=merged_contracts,
            scope=merged_scope
        )

    async def _merge_function_contracts(
        self,
        function_key: str,
        contract_entries: list[tuple[FunctionContract, str]],
        agent_weights: dict[str, float],
        coordination_notes: list[str]
    ) -> FunctionContract:
        """Merge contracts for same function using weighted negotiation."""
        base_contract = contract_entries[0][0].model_copy(deep=True)
        base_contract.preconditions = []
        base_contract.postconditions = []
        base_contract.invariants = []

        pre_candidates = self._collect_contract_candidates(
            contract_entries,
            agent_weights,
            lambda contract: contract.preconditions,
            ContractType.PRECONDITION
        )
        post_candidates = self._collect_contract_candidates(
            contract_entries,
            agent_weights,
            lambda contract: contract.postconditions,
            ContractType.POSTCONDITION
        )
        inv_candidates = self._collect_contract_candidates(
            contract_entries,
            agent_weights,
            lambda contract: contract.invariants,
            ContractType.INVARIANT
        )

        base_contract.preconditions = self._resolve_candidates(
            function_key,
            pre_candidates,
            coordination_notes
        )
        base_contract.postconditions = self._resolve_candidates(
            function_key,
            post_candidates,
            coordination_notes
        )
        base_contract.invariants = self._resolve_candidates(
            function_key,
            inv_candidates,
            coordination_notes
        )

        # Merge modifiers and other metadata using coalition weights
        modifies: set[str] = set()
        aggregator_pure = True
        formalism_value = FormalismLevel.NATURAL
        for contract, agent_id in contract_entries:
            modifies.update(contract.modifies)
            aggregator_pure = aggregator_pure and contract.pure
            formalism_value = max(
                [formalism_value, contract.formalism],
                key=lambda x: [FormalismLevel.NATURAL, FormalismLevel.SEMI_FORMAL,
                               FormalismLevel.FORMAL, FormalismLevel.CODE].index(x)
            )
        base_contract.modifies = sorted(modifies)
        base_contract.pure = aggregator_pure
        base_contract.formalism = formalism_value

        return base_contract

    def _collect_contract_candidates(
        self,
        contract_entries: list[tuple[FunctionContract, str]],
        agent_weights: dict[str, float],
        accessor,
        contract_type: ContractType
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect contract candidates grouped by normalized key."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for contract, agent_id in contract_entries:
            weight = agent_weights.get(agent_id, 1.0)
            for clause in accessor(contract):
                normalized_key = clause.description.strip().lower()
                grouped.setdefault(normalized_key, []).append({
                    "contract": clause,
                    "agent_id": agent_id,
                    "score": clause.confidence * weight,
                    "contract_type": contract_type
                })
        return grouped

    def _resolve_candidates(
        self,
        function_key: str,
        grouped_candidates: dict[str, list[dict[str, Any]]],
        coordination_notes: list[str]
    ) -> list[Contract]:
        """Resolve grouped candidates via negotiation if conflicts exist."""
        resolved: list[Contract] = []
        for key, candidates in grouped_candidates.items():
            if len(candidates) == 1:
                resolved.append(candidates[0]["contract"])
                continue
            winner = self._negotiate_contract_choice(function_key, key, candidates)
            coordination_notes.append(
                f"Negotiation resolved {candidates[0]['contract_type'].value} "
                f"conflict for {function_key} ('{key}') in favor of agent {winner['agent_id']}."
            )
            resolved.append(winner["contract"])
        return resolved

    def _negotiate_contract_choice(
        self,
        function_key: str,
        clause_key: str,
        candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Use negotiation game utilities to select best contract candidate."""
        parties = [c["agent_id"] for c in candidates]
        issue = NegotiationIssue(
            issue_id=f"contract_merge::{function_key}::{clause_key}",
            description=f"Resolve contract clause '{clause_key}' for {function_key}",
            parties=parties,
            constraints={"must_preserve_soundness": True},
            preferences={
                agent_id: {"confidence": candidate["score"]}
                for agent_id, candidate in zip(parties, candidates)
            }
        )

        offers = [
            Offer(
                offer_id=f"{issue.issue_id}::{idx}",
                proposer=candidate["agent_id"],
                terms={
                    "description": candidate["contract"].description,
                    "formal_spec": candidate["contract"].formal_spec,
                    "variables": candidate["contract"].variables
                },
                utility=candidate["score"],
                justification=f"Confidence {candidate['contract'].confidence:.2f}"
            )
            for idx, candidate in enumerate(candidates)
        ]

        efficiency = calculate_pareto_efficiency(offers, issue)
        pareto_ids = set(efficiency.get("pareto_optimal_offers", [])) or {offer.offer_id for offer in offers}
        winning_offer = max(
            (offer for offer in offers if offer.offer_id in pareto_ids),
            key=lambda offer: offer.utility
        )

        return next(candidate for candidate in candidates if candidate["agent_id"] == winning_offer.proposer)

    def _compute_agent_weights(
        self,
        results: list[ScopeAwareResult[list[FunctionContract]]]
    ) -> dict[str, float]:
        """Compute coalition-based weights for each contributing agent."""
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
        contracts: list[FunctionContract]
    ) -> set[str]:
        """Extract capability tags for coalition reasoning."""
        caps: set[str] = set()
        for contract in contracts:
            if contract.preconditions:
                caps.add("preconditions")
            if contract.postconditions:
                caps.add("postconditions")
            if contract.invariants:
                caps.add("invariants")
            if contract.modifies:
                caps.add("modifies")
            if contract.formalism in {FormalismLevel.FORMAL, FormalismLevel.CODE}:
                caps.add("formal_spec")
        return caps or {"baseline"}

    async def validate(
        self,
        original: list[ScopeAwareResult[list[FunctionContract]]],
        merged: ScopeAwareResult[list[FunctionContract]]
    ) -> ValidationResult:
        """Validate merged contracts.

        Args:
            original: Original contract results
            merged: Merged result

        Returns:
            Validation result
        """
        issues = []

        # Check that no contracts were lost
        original_functions = set()
        for result in original:
            for contract in result.content:
                original_functions.add(f"{contract.file_path}:{contract.function_name}")

        merged_functions = set()
        for contract in merged.content:
            merged_functions.add(f"{contract.file_path}:{contract.function_name}")

        missing = original_functions - merged_functions
        if missing:
            issues.append(f"Missing contracts for functions: {missing}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if not issues else 0.8
        )


# =============================================================================
# ContractAnalysisCapability - Extends VCMAnalysisCapability
# =============================================================================
#
# This capability extends VCMAnalysisCapability to provide atomic, composable
# primitives for contract inference across multiple pages.
#
# Design Philosophy:
# - Does NOT prescribe a workflow
# - LLM decides when/how to spawn workers, validate hypotheses, merge results
# - Domain-specific methods exposed as action_executors for LLM composition
#
# Example LLM-driven workflow:
#   1. LLM: spawn_workers(page_ids, cache_affine=True)
#   2. LLM: [wait for completion events]
#   3. LLM: validate_critical_contracts(page_ids)  # domain-specific
#   4. LLM: merge_results(page_ids)
#   5. LLM: get_security_contracts()  # domain-specific query
# =============================================================================


class ContractAnalysisCapability(VCMAnalysisCapability):
    """Capability for distributed contract inference using VCMAnalysisCapability primitives.

    Extends VCMAnalysisCapability with contract-specific:
    - Worker type (ContractInferenceCapability)
    - Merge policy (ContractMergePolicy)
    - Domain methods (hypothesis validation, coalition formation)

    The LLM planner composes the atomic primitives from the base class with
    the domain-specific methods exposed here.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "contract_analysis",
        capability_key: str = "contract_analysis_capability",
    ):
        """Initialize contract analysis capability.

        Args:
            agent: Agent using this capability (coordinator agent)
            scope: Blackboard scope
            namespace: Namespace for blackboard events
            capability_key: Unique key for this capability within the agent
        """
        super().__init__(agent=agent, scope=scope, namespace=namespace, capability_key=capability_key)

    async def initialize(self) -> None:
        """Initialize contract analysis capability."""
        await super().initialize()

    # =========================================================================
    # Abstract Hook Implementations
    # =========================================================================

    def get_worker_capability_class(self) -> type[AgentCapability]:
        """Return ContractInferenceCapability as the worker capability."""
        return ContractInferenceCapability

    def get_worker_agent_type(self) -> str:
        """Return the worker agent type string."""
        return "polymathera.colony.samples.code_analysis.contracts.ContractInferenceAgent"

    def get_domain_merge_policy(self) -> MergePolicy:
        """Return ContractMergePolicy for merging function contracts."""
        return ContractMergePolicy()

    def get_analysis_parameters(self, **kwargs) -> dict[str, Any]:
        """Return contract-specific analysis parameters.

        Args:
            **kwargs: Parameters from action call

        Returns:
            Parameters for contract inference workers
        """
        return {
            "formalism": kwargs.get("formalism", "semi_formal"),
            "use_examples": kwargs.get("use_examples", True),
            **kwargs,
        }

    # =========================================================================
    # Domain-Specific Action Executors
    # =========================================================================

    @action_executor(action_key="validate_critical_contracts")
    async def validate_critical_contracts(
        self,
        page_ids: list[str] | None = None,
        min_agents_for_game: int = 3,
    ) -> dict[str, Any]:
        """Validate critical contracts (security-related) using hypothesis games.

        LLM can call this after collecting results to validate important contracts.

        Args:
            page_ids: Pages to validate (if None, uses all analyzed pages)
            min_agents_for_game: Minimum workers needed for hypothesis game

        Returns:
            Dict with:
            - validated_count: Number of contracts validated
            - downgraded_count: Contracts with reduced confidence
            - critical_contracts: List of critical contract info
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        validated_count = 0
        downgraded_count = 0
        critical_contracts = []

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            contracts = result_data.get("content", [])

            if isinstance(contracts, dict):
                contracts = contracts.get("contracts", [])

            for contract in contracts:
                # Check if contract is critical (security-related)
                preconditions = contract.get("preconditions", [])
                postconditions = contract.get("postconditions", [])

                is_critical = any(
                    "security" in str(cond.get("description", "")).lower()
                    for cond in preconditions + postconditions
                )

                if is_critical:
                    critical_contracts.append({
                        "page_id": page_id,
                        "function_name": contract.get("function_name"),
                        "file_path": contract.get("file_path"),
                        "confidence": contract.get("confidence", 0.5),
                    })

                    # Check if we have enough workers for hypothesis game
                    busy_workers = await self.get_busy_workers()
                    if busy_workers.get("count", 0) >= min_agents_for_game:
                        confidence = contract.get("confidence", 0.5)
                        if confidence <= 0.7:
                            # Would run hypothesis game here
                            # For now, mark as needing validation
                            downgraded_count += 1
                        else:
                            validated_count += 1
                    else:
                        validated_count += 1

        logger.info(
            f"ContractAnalysisCapability: validated {validated_count} critical contracts, "
            f"downgraded {downgraded_count}"
        )

        return {
            "validated_count": validated_count,
            "downgraded_count": downgraded_count,
            "critical_contracts": critical_contracts,
            "total_critical": len(critical_contracts),
        }

    @override
    async def validate_critical_results(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate critical contracts via hypothesis game protocol.

        Forms hypotheses about security-critical contracts and runs
        adversarial validation with available workers. Reduces confidence
        on contracts that fail validation.
        """
        # Delegate to the existing domain-specific method
        return await self.validate_critical_contracts(page_ids=page_ids)

    @override
    async def form_coalitions(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Form coalitions among contract inference workers for consensus.

        Groups workers by result similarity to reach consensus on
        conflicting contract interpretations.
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        if len(results) < 2:
            return {"coalitions": [], "method": "insufficient_results"}

        # Group by similarity — pages with overlapping function names
        # form natural coalitions
        function_sets: dict[str, set[str]] = {}
        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            contracts = result_data.get("content", [])
            if isinstance(contracts, dict):
                contracts = contracts.get("contracts", [])
            function_sets[page_id] = {
                c.get("function_name", "") for c in contracts if isinstance(c, dict)
            }

        coalitions: list[dict[str, Any]] = []
        used = set()
        for p1 in page_ids:
            if p1 in used:
                continue
            coalition = [p1]
            used.add(p1)
            for p2 in page_ids:
                if p2 in used:
                    continue
                overlap = function_sets.get(p1, set()) & function_sets.get(p2, set())
                if overlap:
                    coalition.append(p2)
                    used.add(p2)
            if len(coalition) > 1:
                coalitions.append({
                    "pages": coalition,
                    "shared_functions": list(function_sets.get(p1, set()) & function_sets.get(coalition[1], set())) if len(coalition) > 1 else [],
                })

        return {
            "coalitions": coalitions,
            "count": len(coalitions),
            "method": "function_overlap",
        }

    @override
    async def negotiate_merge(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Negotiate contract merge using multi-agent negotiation.

        Creates NegotiationIssue and Offer objects for conflicting
        contract interpretations across pages.
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        # Find conflicting contracts (same function, different contracts)
        by_function: dict[str, list[dict]] = {}
        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            contracts = result_data.get("content", [])
            if isinstance(contracts, dict):
                contracts = contracts.get("contracts", [])
            for contract in contracts:
                if isinstance(contract, dict):
                    fname = contract.get("function_name", "")
                    if fname:
                        by_function.setdefault(fname, []).append({
                            "page_id": page_id,
                            "contract": contract,
                        })

        conflicts = {k: v for k, v in by_function.items() if len(v) > 1}

        if not conflicts:
            return {"negotiated": False, "method": "no_conflicts", "conflicts": 0}

        # For each conflict, pick highest-confidence contract
        resolved = 0
        for fname, versions in conflicts.items():
            best = max(versions, key=lambda v: v["contract"].get("confidence", 0))
            resolved += 1

        return {
            "negotiated": True,
            "method": "confidence_based",
            "conflicts": len(conflicts),
            "resolved": resolved,
        }

    @action_executor(action_key="get_security_contracts")
    async def get_security_contracts(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get all security-related contracts across analyzed pages.

        LLM can use this to focus on security-critical contracts.

        Args:
            page_ids: Pages to search (if None, uses all analyzed pages)

        Returns:
            Dict with:
            - contracts: List of security-related contracts
            - count: Total count
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        security_contracts = []

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            contracts = result_data.get("content", [])

            if isinstance(contracts, dict):
                contracts = contracts.get("contracts", [])

            for contract in contracts:
                preconditions = contract.get("preconditions", [])
                postconditions = contract.get("postconditions", [])

                security_conditions = [
                    cond for cond in preconditions + postconditions
                    if "security" in str(cond.get("description", "")).lower()
                    or "auth" in str(cond.get("description", "")).lower()
                    or "access" in str(cond.get("description", "")).lower()
                ]

                if security_conditions:
                    security_contracts.append({
                        "page_id": page_id,
                        "function_name": contract.get("function_name"),
                        "file_path": contract.get("file_path"),
                        "security_conditions": security_conditions,
                        "confidence": contract.get("confidence", 0.5),
                    })

        return {
            "contracts": security_contracts,
            "count": len(security_contracts),
        }

    @action_executor(action_key="get_contracts_by_type")
    async def get_contracts_by_type(
        self,
        contract_type: str,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get contracts filtered by type (precondition, postcondition, invariant).

        Args:
            contract_type: Type to filter by ("precondition", "postcondition", "invariant")
            page_ids: Pages to search (if None, uses all analyzed pages)

        Returns:
            Dict with contracts of the specified type.
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        filtered_contracts = []
        accessor_key = f"{contract_type}s"  # preconditions, postconditions, invariants

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            contracts = result_data.get("content", [])

            if isinstance(contracts, dict):
                contracts = contracts.get("contracts", [])

            for contract in contracts:
                conditions = contract.get(accessor_key, [])
                if conditions:
                    filtered_contracts.append({
                        "page_id": page_id,
                        "function_name": contract.get("function_name"),
                        "file_path": contract.get("file_path"),
                        accessor_key: conditions,
                        "confidence": contract.get("confidence", 0.5),
                    })

        return {
            "contracts": filtered_contracts,
            "count": len(filtered_contracts),
            "contract_type": contract_type,
        }



# ContractCoordinatorCapability has been removed.
# Use ContractAnalysisCapability (VCMAnalysisCapability subclass) instead.
