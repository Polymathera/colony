"""Specification and Contract Compliance - Verify code meets specifications and contracts.

Qualitative LLM-based compliance checking that verifies code against
formal specifications, contracts (pre/post conditions), invariants,
and requirements. This is distinct from license/regulatory compliance.

Traditional Approach:
- Formal verification with theorem provers
- Model checking
- Contract checking with runtime assertions
- Requirements traceability matrices

LLM Approach:
- Semantic understanding of specifications
- Contract inference and validation
- Invariant discovery and checking
- Requirement-to-code traceability
- Property-based reasoning
"""

from __future__ import annotations

import asyncio
import logging
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
from polymathera.colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.events import event_handler, EventProcessingResult
from polymathera.colony.agents.blackboard import EnhancedBlackboard, ObligationGraph, BlackboardEvent, ComplianceRelationship
from polymathera.colony.agents.base import Agent, AgentCapability, CapabilityResultFuture
from polymathera.colony.agents.models import Action, PolicyREPL, AgentSuspensionState
from ..contracts.types import Contract, ContractType, FunctionContract

logger = logging.getLogger(__name__)


class SpecificationType(str, Enum):
    """Types of specifications."""

    FUNCTIONAL = "functional"  # What the system should do
    BEHAVIORAL = "behavioral"  # How the system should behave
    PERFORMANCE = "performance"  # Performance requirements
    INTERFACE = "interface"  # API specifications
    DATA = "data"  # Data format/schema specs
    TEMPORAL = "temporal"  # Timing constraints
    SAFETY = "safety"  # Safety properties
    LIVENESS = "liveness"  # Progress properties


class ComplianceLevel(str, Enum):
    """Level of compliance with specification."""

    SATISFIED = "satisfied"  # Fully meets spec
    PARTIALLY_SATISFIED = "partially_satisfied"  # Partially meets
    VIOLATED = "violated"  # Violates spec
    UNKNOWN = "unknown"  # Cannot determine
    NOT_APPLICABLE = "not_applicable"  # Spec doesn't apply


class Specification(BaseModel):
    """A formal or informal specification."""

    spec_id: str = Field(
        description="Unique specification identifier"
    )

    spec_type: SpecificationType = Field(
        description="Type of specification"
    )

    description: str = Field(
        description="Natural language description"
    )

    formal_spec: str | None = Field(
        default=None,
        description="Formal specification if available"
    )

    contracts: list[Contract] = Field(
        default_factory=list,
        description="Associated contracts (pre/post/invariants)"
    )

    properties: list[str] = Field(
        default_factory=list,
        description="Properties that must hold"
    )

    source: str = Field(
        default="requirements",
        description="Source document/standard"
    )

    priority: str = Field(
        default="medium",
        description="Priority: critical, high, medium, low"
    )


class ComplianceEvidence(BaseModel):
    """Evidence of compliance or violation."""

    evidence_id: str = Field(
        description="Evidence identifier"
    )

    evidence_type: str = Field(
        description="Type: code, test, documentation, analysis"
    )

    location: str = Field(
        description="Where evidence is found"
    )

    content: str = Field(
        description="Evidence content/snippet"
    )

    supports_compliance: bool = Field(
        description="Whether this supports compliance"
    )

    confidence: float = Field(
        default=0.8,
        description="Confidence in evidence"
    )


class SpecComplianceResult(BaseModel):
    """Result of checking specification compliance."""

    spec_id: str = Field(
        description="Specification being checked"
    )

    compliance_level: ComplianceLevel = Field(
        description="Level of compliance"
    )

    evidence: list[ComplianceEvidence] = Field(
        default_factory=list,
        description="Supporting or contradicting evidence"
    )

    missing_requirements: list[str] = Field(
        default_factory=list,
        description="Requirements not satisfied"
    )

    contract_violations: list[ContractViolation] = Field(
        default_factory=list,
        description="Contract violations found"
    )

    invariant_violations: list[InvariantViolation] = Field(
        default_factory=list,
        description="Invariant violations found"
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="How to achieve compliance"
    )

    risk_assessment: str | None = Field(
        default=None,
        description="Risk of non-compliance"
    )


class ContractViolation(BaseModel):
    """A contract violation."""

    contract_type: ContractType = Field(
        description="Type of contract violated"
    )

    description: str = Field(
        description="What was violated"
    )

    location: str = Field(
        description="Where violation occurs"
    )

    expected: str = Field(
        description="What was expected"
    )

    actual: str = Field(
        description="What actually happens"
    )

    severity: str = Field(
        default="medium",
        description="Severity: critical, high, medium, low"
    )


class InvariantViolation(BaseModel):
    """An invariant violation."""

    invariant: str = Field(
        description="Invariant that was violated"
    )

    location: str = Field(
        description="Where violation occurs"
    )

    counterexample: str | None = Field(
        default=None,
        description="Counterexample showing violation"
    )

    confidence: float = Field(
        default=0.7,
        description="Confidence in violation"
    )


class ComplianceThread(BaseModel):
    """Thread linking requirements to implementation."""

    requirement_id: str = Field(
        description="Requirement being traced"
    )

    supporting_evidence: list[ComplianceEvidence] = Field(
        default_factory=list,
        description="Evidence supporting compliance"
    )

    missing_controls: list[str] = Field(
        default_factory=list,
        description="Missing implementation"  # TODO: Elaborate on this
    )

    risk_level: str = Field(
        default="medium",
        description="Risk level if not compliant"
    )

    trace_path: list[str] = Field(
        default_factory=list,
        description="Path from requirement to implementation"  # TODO: Elaborate on this
    )


class SpecificationComplianceReport(BaseModel):
    """Complete specification compliance report."""

    specifications: list[Specification] = Field(
        description="Specifications checked"
    )

    compliance_results: list[SpecComplianceResult] = Field(
        default_factory=list,
        description="Results for each specification"
    )

    compliance_threads: list[ComplianceThread] = Field(
        default_factory=list,
        description="Requirement traceability threads"
    )

    inferred_contracts: list[FunctionContract] = Field(
        default_factory=list,
        description="Contracts inferred from code"
    )

    overall_compliance: ComplianceLevel = Field(
        description="Overall compliance level"
    )

    compliance_percentage: float = Field(
        description="Percentage of specs satisfied"
    )

    critical_violations: list[str] = Field(
        default_factory=list,
        description="Critical violations requiring immediate attention"  # TODO: Clarify this. Is this a list of violation descriptions?
    )

    obligation_graph: dict[str, Any] | None = Field(  # TODO: Why is this a dict not an ObligationGraph?
        default=None,
        description="Graph linking requirements to artifacts"
    )


class SpecificationComplianceResult(ScopeAwareResult[SpecificationComplianceReport]):
    """Specification compliance result with scope awareness."""

    pass


class SpecificationComplianceCapability(AgentCapability):
    """Capability for checking specification and contract compliance using LLM.

    This capability uses an LLM to:
    1. Infer contracts from code
    2. Check code against specifications
    3. Verify invariants hold
    4. Trace requirements to implementation
    5. Build obligation graphs
    6. Assess compliance risks

    This is the "main use case for LLM inference over extremely long context"
    as noted in the documentation.

    Integration:
    - Uses ObligationGraph for requirement traceability
    - Works with contract inference patterns
    - Integrates with formal verification when available
    - Works with VCM for paged code access
    - Event-driven via @event_handler and @action_executor decorators
    """

    protocols = [AgentRunProtocol]
    input_patterns = [AgentRunProtocol.request_pattern(namespace="spec_compliance")]

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        infer_contracts: bool = True,
        check_invariants: bool = True,
        trace_requirements: bool = True
    ):
        """Initialize specification compliance capability.

        Args:
            agent: Agent instance for LLM inference via VCM
            scope: Blackboard scope for this capability (default: AGENT)
            infer_contracts: Whether to infer contracts from code
            check_invariants: Whether to check invariants
            trace_requirements: Whether to trace requirements
        """
        super().__init__(agent=agent, scope_id=f"{get_scope_prefix(scope, agent)}:specification_compliance:{agent.agent_id}")
        self.infer_contracts = infer_contracts
        self.check_invariants = check_invariants
        self.trace_requirements = trace_requirements
        self.obligation_graph: ObligationGraph | None = None

        # Layer 0/1 capability reference for page graph operations
        self._page_graph_cap: PageGraphCapability | None = None

    def get_action_group_description(self) -> str:
        return (
            "Specification Compliance — checks code against specifications using VCM long-context. "
            "Single main action (check_compliance) covering: contract inference, invariant verification, "
            "requirement tracing, obligation graphs, risk assessment. "
            "Primary use case for LLM inference over extremely long context."
        )

    async def initialize(self) -> None:
        """Initialize specification compliance capability."""
        await super().initialize()

        # PageGraphCapability for page dependency graph operations
        self._page_graph_cap = self.agent.get_capability_by_type(PageGraphCapability)
        if not self._page_graph_cap:
            self._page_graph_cap = PageGraphCapability(
                agent=self.agent,
                scope=BlackboardScope.COLONY,
            )
            await self._page_graph_cap.initialize()
            self.agent.add_capability(self._page_graph_cap)

    async def _ensure_obligation_graph(self) -> ObligationGraph:
        """Ensure obligation graph is initialized."""
        if self.obligation_graph is None:
            blackboard = await self.get_blackboard()
            # Initialize obligation graph for requirement tracking
            self.obligation_graph = ObligationGraph(
                blackboard=blackboard,
            )
        return self.obligation_graph

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for SpecificationComplianceCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for SpecificationComplianceCapability")
        pass

    @event_handler(pattern=AgentRunProtocol.request_pattern(namespace="spec_compliance"))
    async def handle_compliance_request(
        self,
        event: BlackboardEvent,
        _repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle specification compliance check request events from AgentHandle.run()."""
        request_id = event.key.split(":")[-1]
        request_data = event.value

        input_data = request_data.get("input", {})
        code_page_ids = input_data.get("code_page_ids", [])
        specifications_data = input_data.get("specifications", [])
        test_page_ids = input_data.get("test_page_ids")
        doc_page_ids = input_data.get("doc_page_ids")

        # Parse specifications
        specifications = [Specification(**s) for s in specifications_data]

        return EventProcessingResult(
            immediate_action=Action(
                action_type="check_compliance",
                parameters={
                    "code_page_ids": code_page_ids,
                    "specifications": [s.model_dump() for s in specifications],
                    "test_page_ids": test_page_ids,
                    "doc_page_ids": doc_page_ids,
                    "request_id": request_id
                }
            )
        )

    @action_executor(action_key="check_compliance")
    async def check_compliance(
        self,
        code_page_ids: list[str],
        specifications: list[Specification] | list[dict[str, Any]],
        test_page_ids: list[str] | None = None,
        doc_page_ids: list[str] | None = None,
        request_id: str | None = None
    ) -> SpecificationComplianceResult:
        """Check code compliance with specifications.

        Args:
            code_page_ids: Page IDs of source code in VCM
            specifications: Specifications to verify against (can be dicts from action params)
            test_page_ids: Page IDs of test code in VCM
            doc_page_ids: Page IDs of documentation in VCM
            request_id: Optional request ID for writing result to blackboard

        Returns:
            Specification compliance report
        """
        # Parse specifications if they're dicts
        parsed_specs: list[Specification] = []
        for spec in specifications:
            if isinstance(spec, dict):
                parsed_specs.append(Specification(**spec))
            else:
                parsed_specs.append(spec)

        compliance_results = []
        compliance_threads = []
        inferred_contracts = []

        # Infer contracts from code if enabled
        if self.infer_contracts:
            for page_id in code_page_ids:
                contracts = await self._infer_contracts_from_page(page_id)
                inferred_contracts.extend(contracts)

        # Check each specification
        for spec in parsed_specs:
            result = await self._check_specification(
                spec,
                code_page_ids,
                inferred_contracts,
                test_page_ids
            )
            compliance_results.append(result)

            # Build compliance thread for traceability
            if self.trace_requirements:
                thread = await self._trace_requirement(
                    spec,
                    result,
                    code_page_ids,
                    test_page_ids,
                    doc_page_ids
                )
                compliance_threads.append(thread)

        # Build obligation graph
        obligation_graph_data = await self._build_obligation_graph(
            parsed_specs,
            compliance_results,
            compliance_threads
        )

        # Determine overall compliance
        overall_compliance = self._determine_overall_compliance(compliance_results)
        compliance_percentage = self._calculate_compliance_percentage(compliance_results)
        critical_violations = self._identify_critical_violations(compliance_results)

        # Build report
        report = SpecificationComplianceReport(
            specifications=parsed_specs,
            compliance_results=compliance_results,
            compliance_threads=compliance_threads,
            inferred_contracts=inferred_contracts,
            overall_compliance=overall_compliance,
            compliance_percentage=compliance_percentage,
            critical_violations=critical_violations,
            obligation_graph=obligation_graph_data
        )

        # Build result with scope
        scope = AnalysisScope(
            is_complete=self._check_completeness(report, code_page_ids),
            missing_context=self._identify_missing_context(report),
            confidence=self._compute_confidence(report),
            reasoning=[self._summarize_compliance(report)]
        )

        result = SpecificationComplianceResult(
            content=report,
            scope=scope
        )

        # Write result to blackboard for AgentHandle.run() to receive
        if request_id:
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=AgentRunProtocol.result_key(request_id, namespace="spec_compliance"),
                value=result.model_dump(),
            )

        return result

    async def _infer_contracts_from_page(
        self,
        page_id: str
    ) -> list[FunctionContract]:
        """Infer contracts from a VCM page using LLM.

        Args:
            page_id: Page ID in VCM

        Returns:
            Inferred contracts
        """
        prompt = """Infer formal contracts from the code in this page.

For each function, identify:
1. PRECONDITIONS - What must be true before calling
2. POSTCONDITIONS - What is guaranteed after calling
3. INVARIANTS - What remains true throughout
4. PROPERTIES - Important properties that should hold

For each function provide:
FUNCTION: [name]
PRECONDITIONS:
- [condition]: [formal spec or natural language]
POSTCONDITIONS:
- [condition]: [formal spec or natural language]
INVARIANTS:
- [invariant]: [formal spec or natural language]
PROPERTIES:
- [property]: [description]

Focus on behavioral contracts, not just type contracts.

Output ONLY the contract specifications in the format above."""

        # Use agent.infer with VCM page
        response = await self.agent.infer(
            context_page_ids=[page_id],
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000,
            json_schema={"type": "array", "items": FunctionContract.model_json_schema()}  # Structured output
        )

        # Parse structured response directly
        import json
        contracts_data = json.loads(response.generated_text)
        return [FunctionContract(**c) for c in contracts_data]

    def _parse_contracts(self, response: str, file_path: str) -> list[FunctionContract]:
        """Parse contracts from LLM response.

        Args:
            response: LLM response
            file_path: Source file

        Returns:
            List of function contracts
        """
        # Reuse contract parsing logic from contracts.py
        # This is simplified
        contracts = []

        lines = response.split("\n")
        current_contract = None

        for line in lines:
            if line.startswith("FUNCTION:"):
                if current_contract:
                    contracts.append(current_contract)
                func_name = line.split("FUNCTION:")[1].strip()
                current_contract = FunctionContract(
                    function_name=func_name,
                    file_path=file_path,
                    line_number=0
                )
            elif line.startswith("PRECONDITIONS:") and current_contract:
                # Parse preconditions
                pass
            elif line.startswith("POSTCONDITIONS:") and current_contract:
                # Parse postconditions
                pass
            elif line.startswith("INVARIANTS:") and current_contract:
                # Parse invariants
                pass

        if current_contract:
            contracts.append(current_contract)

        return contracts

    async def _check_specification(
        self,
        spec: Specification,
        code_page_ids: list[str],
        inferred_contracts: list[FunctionContract],
        test_page_ids: list[str] | None
    ) -> SpecComplianceResult:
        """Check if code complies with specification.

        Args:
            spec: Specification to check
            code_page_ids: VCM page IDs for source code
            inferred_contracts: Contracts inferred from code
            test_page_ids: VCM page IDs for test code

        Returns:
            Compliance result
        """
        # Build compliance checking prompt
        prompt = self._build_compliance_prompt(spec, inferred_contracts)

        # Get LLM analysis using VCM pages
        response = await self.agent.infer(
            context_page_ids=code_page_ids,
            prompt=prompt,
            temperature=0.2,
            max_tokens=2500,
            json_schema=SpecComplianceResult.model_json_schema()  # Structured output
        )

        # Parse structured response directly
        result = SpecComplianceResult.model_validate_json(response.generated_text)  # TODO: Handle validation errors. LLMs are not perfect.

        # Check invariants if enabled
        if self.check_invariants and spec.contracts:
            invariant_violations = await self._check_invariants(
                spec.contracts,
                code_page_ids
            )
            result.invariant_violations = invariant_violations

        # Find evidence in tests
        if test_page_ids:
            test_evidence = await self._find_test_evidence(spec, test_page_ids)
            result.evidence.extend(test_evidence)

        return result

    def _build_compliance_prompt(
        self,
        spec: Specification,
        contracts: list[FunctionContract]
    ) -> str:
        """Build prompt for compliance checking.

        Args:
            spec: Specification
            contracts: Inferred contracts

        Returns:
            Formatted prompt
        """
        contracts_desc = ""
        if contracts:
            contracts_desc = "\nInferred Contracts:\n"
            for contract in contracts[:5]:  # First 5
                contracts_desc += f"- {contract.function_name}: "
                if contract.preconditions:
                    contracts_desc += f"PRE: {contract.preconditions[0].description} "
                if contract.postconditions:
                    contracts_desc += f"POST: {contract.postconditions[0].description}\n"

        return f"""Check if the code loaded in context complies with the following specification:

SPECIFICATION:
Type: {spec.spec_type.value}
Description: {spec.description}
Formal Spec: {spec.formal_spec or "N/A"}
Properties: {', '.join(spec.properties)}
Priority: {spec.priority}

{contracts_desc}

Analyze:
1. COMPLIANCE LEVEL: Does the code satisfy/partially satisfy/violate the specification?
2. EVIDENCE: What code elements support or violate the specification?
3. MISSING: What requirements are not implemented?
4. CONTRACT VIOLATIONS: Do any functions violate their contracts relative to the spec?
5. RISKS: What are the risks of non-compliance?
6. RECOMMENDATIONS: How can compliance be achieved?

Provide analysis in this format:
COMPLIANCE_LEVEL: [satisfied/partially_satisfied/violated/unknown]
EVIDENCE:
- [Supporting/Violating]: [description] at [location]
MISSING_REQUIREMENTS:
- [requirement not satisfied]
CONTRACT_VIOLATIONS:
- [contract type]: [description] at [location]
RISKS: [risk assessment]
RECOMMENDATIONS:
- [recommendation]"""

    async def _check_invariants(
        self,
        contracts: list[Contract],
        page_ids: list[str]
    ) -> list[InvariantViolation]:
        """Check if invariants hold in code.

        Args:
            contracts: Contracts with invariants
            page_ids: VCM page IDs to check

        Returns:
            Invariant violations
        """
        violations = []

        for contract in contracts:
            if contract.contract_type == ContractType.INVARIANT:
                # Check if invariant holds across all pages
                for page_id in page_ids:
                    violation = await self._check_single_invariant(
                        contract.description,
                        page_id
                    )
                    if violation:
                        violations.append(violation)

        return violations

    async def _check_single_invariant(
        self,
        invariant: str,
        page_id: str
    ) -> InvariantViolation | None:
        """Check if a single invariant holds.

        Args:
            invariant: Invariant to check
            page_id: VCM page ID to check

        Returns:
            Violation if found
        """
        # TODO: Simplified - would use LLM with page context
        # Could integrate with symbolic execution or model checking
        return None

    async def _find_test_evidence(
        self,
        spec: Specification,
        test_page_ids: list[str]
    ) -> list[ComplianceEvidence]:
        """Find evidence in test files.

        Args:
            spec: Specification
            test_page_ids: VCM page IDs for test code

        Returns:
            Test evidence
        """
        evidence = []

        for page_id in test_page_ids:
            # TODO: Would use LLM to check if tests cover specification
            # For now, simplified
            evidence.append(ComplianceEvidence(
                evidence_id=f"test_ev_{len(evidence)}",
                evidence_type="test",
                location=page_id,
                content=f"Test coverage for {spec.spec_id}",
                supports_compliance=True,
                confidence=0.7
            ))

        return evidence

    async def _trace_requirement(
        self,
        spec: Specification,
        result: SpecComplianceResult,
        code_page_ids: list[str],
        test_page_ids: list[str] | None,
        doc_page_ids: list[str] | None
    ) -> ComplianceThread:
        """Trace requirement to implementation.

        Args:
            spec: Specification
            result: Compliance result
            code_page_ids: VCM page IDs for source code
            test_page_ids: VCM page IDs for test code
            doc_page_ids: VCM page IDs for documentation

        Returns:
            Compliance thread
        """
        thread = ComplianceThread(
            requirement_id=spec.spec_id,
            supporting_evidence=result.evidence,
            risk_level=spec.priority
        )

        # Build trace path
        trace_path = [f"Requirement: {spec.spec_id}"]

        # Find implementation
        for evidence in result.evidence:
            if evidence.supports_compliance:
                trace_path.append(f"Implementation: {evidence.location}")

        # Find tests
        if test_page_ids:
            for page_id in test_page_ids:
                trace_path.append(f"Test: {page_id}")

        # Find documentation
        if doc_page_ids:
            for page_id in doc_page_ids:
                trace_path.append(f"Documentation: {page_id}")

        thread.trace_path = trace_path

        # Identify missing controls
        thread.missing_controls = result.missing_requirements

        return thread

    async def _build_obligation_graph(
        self,
        specifications: list[Specification],
        results: list[SpecComplianceResult],
        threads: list[ComplianceThread]
    ) -> dict[str, Any]:
        """Build obligation graph linking requirements to artifacts.

        Args:
            specifications: All specifications
            results: Compliance results
            threads: Compliance threads

        Returns:
            Obligation graph data
        """
        # Add specifications as requirements
        for spec in specifications:
            await self.obligation_graph.add_requirement(
                requirement_id=spec.spec_id,
                description=spec.description,
                metadata={
                    "type": spec.spec_type.value,
                    "priority": spec.priority,
                    "source": spec.source
                }
            )

        # Add evidence as artifacts
        for result in results:
            for evidence in result.evidence:
                await self.obligation_graph.add_artifact(
                    artifact_id=evidence.evidence_id,
                    artifact_type=evidence.evidence_type,
                    metadata={
                        "location": evidence.location,
                        "supports": evidence.supports_compliance
                    }
                )

                # Link to requirement
                relationship = ComplianceRelationship.SATISFIES if evidence.supports_compliance else ComplianceRelationship.VIOLATES
                await self.obligation_graph.link(
                    requirement_id=result.spec_id,
                    artifact_id=evidence.evidence_id,
                    relationship=relationship,
                    evidence=[evidence.content],
                    confidence=evidence.confidence
                )

        # Add missing controls
        for thread in threads:
            for missing in thread.missing_controls:
                await self.obligation_graph.link(
                    requirement_id=thread.requirement_id,
                    artifact_id=f"missing_{missing}",
                    relationship=ComplianceRelationship.MISSING,
                    confidence=0.9
                )

        # Return graph summary
        compliance_status = await self.obligation_graph.get_compliance_status()
        return {
            "requirements": len(specifications),
            "artifacts": sum(len(r.evidence) for r in results),
            "satisfied": compliance_status.get("satisfied", 0),
            "partial": compliance_status.get("partial", 0),
            "missing": compliance_status.get("missing", 0)
        }

    def _determine_overall_compliance(
        self,
        results: list[SpecComplianceResult]
    ) -> ComplianceLevel:
        """Determine overall compliance level.

        Args:
            results: Individual compliance results

        Returns:
            Overall compliance level
        """
        if not results:
            return ComplianceLevel.UNKNOWN

        # If any critical spec is violated, overall is violated
        if any(r.compliance_level == ComplianceLevel.VIOLATED for r in results):
            return ComplianceLevel.VIOLATED

        # If all are satisfied, overall is satisfied
        if all(r.compliance_level == ComplianceLevel.SATISFIED for r in results):
            return ComplianceLevel.SATISFIED

        # Otherwise partially satisfied
        return ComplianceLevel.PARTIALLY_SATISFIED

    def _calculate_compliance_percentage(
        self,
        results: list[SpecComplianceResult]
    ) -> float:
        """Calculate percentage of specifications satisfied.

        Args:
            results: Compliance results

        Returns:
            Compliance percentage
        """
        if not results:
            return 0.0

        satisfied = sum(1 for r in results 
                       if r.compliance_level == ComplianceLevel.SATISFIED)
        partial = sum(0.5 for r in results 
                     if r.compliance_level == ComplianceLevel.PARTIALLY_SATISFIED)

        return ((satisfied + partial) / len(results)) * 100

    def _identify_critical_violations(
        self,
        results: list[SpecComplianceResult]
    ) -> list[str]:
        """Identify critical violations.

        Args:
            results: Compliance results

        Returns:
            List of critical violations
        """
        critical = []

        for result in results:
            if result.compliance_level == ComplianceLevel.VIOLATED:
                if result.risk_assessment and "critical" in result.risk_assessment.lower():
                    critical.append(f"Critical violation of {result.spec_id}: {result.risk_assessment}")

            for violation in result.contract_violations:
                if violation.severity == "critical":
                    critical.append(f"Critical contract violation: {violation.description}")

        return critical

    def _check_completeness(
        self,
        report: SpecificationComplianceReport,
        code_page_ids: list[str]
    ) -> bool:
        """Check if compliance checking is complete.

        Args:
            report: Compliance report
            code_page_ids: VCM page IDs analyzed

        Returns:
            True if complete
        """
        # Complete if all specifications were checked
        return len(report.compliance_results) == len(report.specifications)

    def _identify_missing_context(
        self,
        report: SpecificationComplianceReport
    ) -> list[str]:
        """Identify missing context.

        Args:
            report: Compliance report

        Returns:
            List of missing elements
        """
        missing = []

        # Check for unknown compliance
        for result in report.compliance_results:
            if result.compliance_level == ComplianceLevel.UNKNOWN:
                missing.append(f"Cannot determine compliance for {result.spec_id}")

        # Check for missing evidence
        for result in report.compliance_results:
            if not result.evidence:
                missing.append(f"No evidence found for {result.spec_id}")

        return missing

    def _compute_confidence(
        self,
        report: SpecificationComplianceReport
    ) -> float:
        """Compute overall confidence.

        Args:
            report: Compliance report

        Returns:
            Confidence score
        """
        if not report.compliance_results:
            return 0.0

        # Average confidence across evidence
        total_confidence = 0.0
        count = 0

        for result in report.compliance_results:
            for evidence in result.evidence:
                total_confidence += evidence.confidence
                count += 1

        return total_confidence / count if count > 0 else 0.5

    def _summarize_compliance(
        self,
        report: SpecificationComplianceReport
    ) -> str:
        """Summarize compliance status.

        Args:
            report: Compliance report

        Returns:
            Summary
        """
        return (f"Specification Compliance: {report.overall_compliance.value}. "
                f"{report.compliance_percentage:.1f}% of specifications satisfied. "
                f"{len(report.inferred_contracts)} contracts inferred. "
                f"{len(report.critical_violations)} critical violations found. "
                f"Obligation graph: {report.obligation_graph}")
