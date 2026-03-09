
from __future__ import annotations

import itertools
import logging
from typing import Any
from overrides import override

from colony.agents.patterns import (
    AnalysisScope,
    ScopeAwareResult,
    MergePolicy,
    MergeContext,
    MergeCapability,
    ValidationResult,
)
from colony.agents.patterns.capabilities.agent_pool import AgentPoolCapability
from colony.agents.patterns.capabilities.result import ResultCapability
from colony.agents.patterns.capabilities.page_graph import PageGraphCapability
from colony.agents.patterns.capabilities.batching import BatchingPolicy
from colony.agents.patterns.capabilities.vcm_analysis import VCMAnalysisCapability
from colony.agents.patterns.actions import action_executor
from colony.agents.patterns.events import event_handler, EventProcessingResult
from colony.agents.blackboard import EnhancedBlackboard, ObligationGraph, BlackboardEvent
from colony.agents.base import Agent, AgentCapability, CapabilityResultFuture, AgentHandle
from colony.agents.patterns.games.negotiation.capabilities import NegotiationIssue, Offer, calculate_pareto_efficiency
from colony.agents.patterns.games.coalition_formation import find_optimal_coalition_structure
from colony.agents.models import Action, AgentMetadata, PolicyREPL, AgentResourceRequirements, AgentSuspensionState
from colony.cluster.models import LLMClientRequirements

from .types import (
    ComplianceRequirement,
    ComplianceViolation,
    ComplianceStatus,
    ComplianceSeverity,
    ComplianceType,
    ComplianceResult,
    ComplianceReport,
    License,
)

logger = logging.getLogger(__name__)


class ComplianceAnalysisCapability(AgentCapability):
    """Capability for compliance analysis using LLM reasoning.

    This capability uses an LLM to:
    1. Detect licenses in code and dependencies
    2. Check license compatibility
    3. Verify regulatory compliance
    4. Validate against organizational policies
    5. Assess compliance risks
    6. Generate remediation recommendations

    Integration:
    - Uses ObligationGraph for tracking requirements
    - Works with dependency analysis
    - Can integrate with policy engines
    - Event-driven via @event_handler and @action_executor decorators
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None,
        requirements: list[ComplianceRequirement] | None = None,
        check_licenses: bool = True,
        check_security: bool = True
    ):
        """Initialize compliance analysis capability.

        Args:
            agent: Agent instance for LLM inference via VCM
            scope_id: Scope identifier for this capability
            requirements: Specific requirements to check
            check_licenses: Whether to check license compliance
            check_security: Whether to check security compliance
        """
        super().__init__(agent=agent, scope_id=scope_id or agent.agent_id)
        self.requirements = requirements or self._default_requirements()
        self.check_licenses = check_licenses
        self.check_security = check_security
        self._obligation_graph: ObligationGraph | None = None

    def get_action_group_description(self) -> str:
        return (
            "Compliance Analysis — license detection, compatibility checking, regulatory verification. "
            "Builds an ObligationGraph for requirement traceability. "
            "analyze_compliance is the main entry point; query results via severity filter or risk assessment. "
            "Configurable requirement sets (license, security, quality)."
        )

    async def _ensure_obligation_graph(self) -> ObligationGraph:
        """Ensure obligation graph is initialized."""
        if self._obligation_graph is None:
            blackboard = await self.get_blackboard()
            self._obligation_graph = ObligationGraph(
                blackboard=blackboard,
            )
        return self._obligation_graph

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ComplianceAnalysisCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ComplianceAnalysisCapability")
        pass

    @event_handler(pattern="{scope_id}:request:*")
    async def handle_analysis_request(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle compliance analysis request events from AgentHandle.run()."""
        # Extract request ID from key
        request_id = event.key.split(":")[-1]
        request_data = event.value

        # Extract parameters from request
        input_data = request_data.get("input", {})
        page_ids = input_data.get("page_ids", [])
        dependencies = input_data.get("dependencies")
        metadata = input_data.get("metadata")

        # Return immediate action to execute analysis
        return EventProcessingResult(
            immediate_action=Action(
                action_type="analyze_compliance",
                parameters={
                    "page_ids": page_ids,
                    "dependencies": dependencies,
                    "metadata": metadata,
                    "request_id": request_id
                }
            )
        )

    def _default_requirements(self) -> list[ComplianceRequirement]:
        """Get default compliance requirements.

        Returns:
            Default requirements
        """
        return [
            ComplianceRequirement(
                requirement_id="lic_001",
                type=ComplianceType.LICENSE,
                description="All dependencies must have compatible licenses",
                source="Open Source Policy",
                checks=["license detection", "compatibility check"]
            ),
            ComplianceRequirement(
                requirement_id="sec_001",
                type=ComplianceType.SECURITY,
                description="No hardcoded credentials or secrets",
                source="Security Policy",
                checks=["credential scanning", "secret detection"]
            ),
            ComplianceRequirement(
                requirement_id="sec_002",
                type=ComplianceType.SECURITY,
                description="Sensitive data must be encrypted",
                source="Data Protection Policy",
                checks=["encryption verification", "data classification"]
            ),
            ComplianceRequirement(
                requirement_id="qual_001",
                type=ComplianceType.QUALITY,
                description="Code must follow style guidelines",
                source="Development Standards",
                checks=["style checking", "naming conventions"]
            )
        ]

    # TODO: Replace page_ids with more general context identifiers.
    # For example: Use a query over pages or repositories, or tags.
    # This is important because the ActionPolicy decides the parameters
    # to these actions, which can be challenging for a LLM to do, especially
    # for very long contexts spanning many pages, repositories, etc.
    @action_executor(action_key="analyze_compliance")
    async def analyze_compliance(
        self,
        page_ids: list[str],
        dependencies: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None
    ) -> ComplianceResult:
        """Analyze compliance across multiple pages.

        This method aggregates compliance analysis across pages using
        a merge policy for comprehensive results.

        Args:
            page_ids: VCM page IDs to analyze
            dependencies: Project dependencies
            metadata: Additional project metadata
            request_id: Optional request ID for writing result to blackboard

        Returns:
            Aggregated compliance analysis report
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
        page_results = []
        # Process each page individually
        for page_id in page_ids:
            result = await self._analyze_page_compliance(
                page_id,
                dependencies,
                metadata
            )
            page_results.append(result)

        # Merge results using MergeCapability
        if len(page_results) == 1:
            merged = page_results[0]
        else:
            merge_cap = self.agent.get_capability_by_type(MergeCapability)
            if merge_cap is None:
                raise RuntimeError(
                    "ComplianceAnalysisCapability requires MergeCapability on the agent. "
                    "Add MergeCapability configured with ComplianceMergePolicy."
                )
            merged = await merge_cap.merge_results(
                page_results,
                MergeContext(strategy="union_violations")  # Union all violations
            )

        # Write result to blackboard for AgentHandle.run() to receive
        if request_id:
            blackboard = await self.get_blackboard()
            await blackboard.write(
                key=f"{self.scope_id}:result:{request_id}",  # TODO: Is scope_id needed? Blackboard already takes care of namespacing.
                value=merged.model_dump(),
            )

        return merged

    async def _analyze_page_compliance(
        self,
        page_id: str,
        dependencies: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> ComplianceResult:
        """Analyze compliance for a single VCM page.

        Args:
            page_id: Single VCM page ID
            dependencies: Project dependencies
            metadata: Additional metadata

        Returns:
            Compliance report for this page
        """
        violations = []
        licenses_found = {}

        # Check license compliance if enabled
        if self.check_licenses and dependencies:
            license_violations, licenses = await self._check_license_compliance(
                page_id,
                dependencies,
                metadata
            )
            violations.extend(license_violations)
            licenses_found = licenses

        # Check security compliance if enabled
        if self.check_security:
            security_violations = await self._check_security_compliance(
                page_id
            )
            violations.extend(security_violations)

        # Check against specific requirements
        for requirement in self.requirements:
            req_violations = await self._check_requirement(
                requirement,
                page_id,
                metadata
            )
            violations.extend(req_violations)

        # Build compliance report
        report = ComplianceReport(
            status=self._determine_status(violations),
            violations=violations,
            requirements_checked=self.requirements,
            licenses_found=licenses_found,
            license_conflicts=self._find_license_conflicts(licenses_found),
            risk_assessment=self._assess_risk(violations),
            recommendations=self._generate_recommendations(violations)
        )

        # Update obligation graph
        await self._update_obligation_graph(report)

        # Build result with scope
        scope = AnalysisScope(
            is_complete=self._check_completeness(report),
            missing_context=self._identify_missing_context(report),
            confidence=self._compute_confidence(report),
            reasoning=[self._summarize_compliance(report)]
        )

        return ComplianceResult(
            content=report,
            scope=scope
        )

    async def _check_license_compliance(
        self,
        page_id: str,
        dependencies: list[dict[str, Any]],
        metadata: dict[str, Any] | None
    ) -> tuple[list[ComplianceViolation], dict[str, License]]:
        """Check license compliance.

        Args:
            dependencies: Project dependencies
            metadata: Project metadata

        Returns:
            License violations and detected licenses
        """
        violations = []
        licenses = {}

        # Detect project license
        project_license = metadata.get("license") if metadata else None

        # Analyze each dependency
        for dep in dependencies:
            dep_name = dep.get("name", "unknown")
            dep_license = dep.get("license", "unknown")

            # Get license details
            license_info = await self._analyze_license(page_id, dep_license)
            licenses[dep_name] = license_info

            # Check compatibility
            if project_license:
                if not self._licenses_compatible(project_license, dep_license):
                    violations.append(ComplianceViolation(
                        violation_id=f"lic_violation_{dep_name}",
                        type=ComplianceType.LICENSE,
                        severity=ComplianceSeverity.HIGH,
                        description=f"License incompatibility: {dep_license} with {project_license}",
                        location=f"dependency: {dep_name}",
                        rule="License Compatibility",
                        evidence=[f"Project: {project_license}", f"Dependency: {dep_license}"],
                        remediation=f"Replace {dep_name} with compatible alternative or change project license",
                        risk="Legal liability, forced open-sourcing",
                        confidence=0.9
                    ))

        return violations, licenses

    async def _analyze_license(self, page_id: str, license_name: str) -> License:
        """Analyze a license using LLM.

        Args:
            page_id: VCM page ID for context
            license_name: License name

        Returns:
            License details
        """
        prompt = f"""Analyze the software license: {license_name}

Return a JSON object matching the License schema with:
- name: license name
- spdx_id: SPDX identifier if known
- category: permissive, copyleft, or proprietary
- permissions: list of what the license permits
- conditions: list of conditions that must be met
- limitations: list of what the license prohibits
- compatible_with: list of compatible licenses
- incompatible_with: list of incompatible licenses"""

        response = await self.agent.infer(
            context_page_ids=[page_id],  # Single page
            prompt=prompt,
            temperature=0.1,
            max_tokens=500,
            json_schema=License.model_json_schema()  # Structured output
        )

        return License.model_validate_json(response.generated_text)


    def _licenses_compatible(self, license1: str, license2: str) -> bool:
        """Check if two licenses are compatible.

        Args:
            license1: First license
            license2: Second license

        Returns:
            True if compatible
        """
        # Simplified compatibility matrix
        incompatible_pairs = [
            ("GPL", "Apache"),
            ("GPL", "MIT"),  # Only if distributed together
            ("AGPL", "proprietary"),
            ("GPL", "proprietary")
        ]

        for l1, l2 in incompatible_pairs:
            if (l1.lower() in license1.lower() and l2.lower() in license2.lower()) or \
               (l2.lower() in license1.lower() and l1.lower() in license2.lower()):
                return False

        return True

    async def _check_security_compliance(
        self,
        page_id: str
    ) -> list[ComplianceViolation]:
        """Check security compliance for a page.

        Args:
            page_id: VCM page ID

        Returns:
            Security violations
        """
        # Check for hardcoded secrets
        secrets = await self._detect_secrets(page_id)

        # Check for security patterns
        security_issues = await self._detect_security_issues(page_id)

        return secrets + security_issues

    async def _detect_secrets(self, page_id: str) -> list[ComplianceViolation]:
        """Detect hardcoded secrets in a page.

        Args:
            page_id: VCM page ID

        Returns:
            Secret violations
        """
        prompt = """Detect potential hardcoded secrets or credentials in the code loaded in context.

Look for:
1. API keys
2. Passwords
3. Database credentials
4. Private keys
5. Tokens

Return a JSON list of ComplianceViolation objects. Each violation should have:
- violation_id: unique identifier
- type: "security"
- severity: "critical", "high", "medium", "low"
- description: what was found
- location: file:line where found
- rule: "No Hardcoded Secrets"
- evidence: code snippet showing the issue
- remediation: how to fix
- risk: potential impact
- confidence: 0.0-1.0

Return empty list if no secrets found."""

        response = await self.agent.infer(
            context_page_ids=[page_id],  # Single page
            prompt=prompt,
            temperature=0.1,
            max_tokens=1000,  # TODO: Add max_tokens to constructor metadata?
            json_schema={"type": "array", "items": ComplianceViolation.model_json_schema()}  # List of violations
        )

        violations = []
        try:
            import json
            violations_data = json.loads(response.generated_text)
            for v_data in violations_data:
                violations.append(ComplianceViolation.model_validate(v_data))
        except Exception:
            # Return empty list if parsing fails
            pass

        return violations

    async def _detect_security_issues(self, page_id: str) -> list[ComplianceViolation]:
        """Detect general security issues.

        Args:
            page_id: VCM page ID

        Returns:
            Security violations
        """
        prompt = """Detect security issues in the code loaded in context.

Look for:
1. SQL injection vulnerabilities
2. Command injection risks
3. Path traversal vulnerabilities
4. Insecure deserialization
5. Cross-site scripting (XSS) risks
6. Insecure random number generation
7. Missing input validation

Return a JSON list of ComplianceViolation objects for any issues found.
Return empty list if no security issues detected."""

        response = await self.agent.infer(
            context_page_ids=[page_id],  # Single page
            prompt=prompt,
            temperature=0.1,
            max_tokens=1000,
            json_schema={"type": "array", "items": ComplianceViolation.model_json_schema()}
        )

        violations = []
        try:
            import json
            violations_data = json.loads(response.generated_text)
            for v_data in violations_data:
                violations.append(ComplianceViolation.model_validate(v_data))
        except Exception:
            pass

        return violations

    async def _check_requirement(
        self,
        requirement: ComplianceRequirement,
        page_id: str,
        metadata: dict[str, Any] | None
    ) -> list[ComplianceViolation]:
        """Check specific requirement.

        Args:
            requirement: Requirement to check
            page_id: VCM page ID
            metadata: Project metadata

        Returns:
            Violations of this requirement
        """
        # TODO: Implement requirement-specific checks
        return []

    def _find_license_conflicts(
        self,
        licenses: dict[str, License]
    ) -> list[dict[str, Any]]:
        """Find license conflicts.

        Args:
            licenses: Detected licenses

        Returns:
            List of conflicts
        """
        conflicts = []

        license_list = list(licenses.items())
        for i, (name1, lic1) in enumerate(license_list):
            for name2, lic2 in license_list[i+1:]:
                if lic1.name in lic2.incompatible_with or lic2.name in lic1.incompatible_with:
                    conflicts.append({
                        "package1": name1,
                        "license1": lic1.name,
                        "package2": name2,
                        "license2": lic2.name,
                        "conflict_type": "incompatible"
                    })

        return conflicts

    def _determine_status(self, violations: list[ComplianceViolation]) -> ComplianceStatus:
        """Determine overall compliance status.

        Args:
            violations: All violations

        Returns:
            Compliance status
        """
        if not violations:
            return ComplianceStatus.COMPLIANT

        critical = any(v.severity == ComplianceSeverity.CRITICAL for v in violations)
        high = any(v.severity == ComplianceSeverity.HIGH for v in violations)

        if critical:
            return ComplianceStatus.NON_COMPLIANT
        elif high:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.PARTIALLY_COMPLIANT

    def _assess_risk(self, violations: list[ComplianceViolation]) -> dict[str, Any]:
        """Assess overall compliance risk.

        Args:
            violations: All violations

        Returns:
            Risk assessment
        """
        risk_score = 0.0
        severity_weights = {
            ComplianceSeverity.CRITICAL: 10.0,
            ComplianceSeverity.HIGH: 5.0,
            ComplianceSeverity.MEDIUM: 2.0,
            ComplianceSeverity.LOW: 1.0,
            ComplianceSeverity.INFO: 0.1
        }

        for violation in violations:
            risk_score += severity_weights.get(violation.severity, 0) * violation.confidence

        risk_level = "low"
        if risk_score > 50:
            risk_level = "critical"
        elif risk_score > 20:
            risk_level = "high"
        elif risk_score > 10:
            risk_level = "medium"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "critical_violations": sum(1 for v in violations if v.severity == ComplianceSeverity.CRITICAL),
            "high_violations": sum(1 for v in violations if v.severity == ComplianceSeverity.HIGH)
        }

    def _generate_recommendations(self, violations: list[ComplianceViolation]) -> list[str]:
        """Generate compliance recommendations.

        Args:
            violations: All violations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Group by type
        by_type = {}
        for violation in violations:
            if violation.type not in by_type:
                by_type[violation.type] = []
            by_type[violation.type].append(violation)

        # Generate type-specific recommendations
        if ComplianceType.LICENSE in by_type:
            recommendations.append("Review and update dependency licenses for compatibility")

        if ComplianceType.SECURITY in by_type:
            recommendations.append("Implement secret scanning in CI/CD pipeline")
            recommendations.append("Use environment variables for sensitive configuration")

        # Add remediation from critical violations
        for violation in violations:
            if violation.severity == ComplianceSeverity.CRITICAL and violation.remediation:
                recommendations.append(f"CRITICAL: {violation.remediation}")

        return recommendations

    async def _update_obligation_graph(self, report: ComplianceReport) -> None:
        """Update obligation graph with compliance results.

        Args:
            report: Compliance report
        """
        if not self._obligation_graph:
            return

        # Add requirements as obligations
        for req in report.requirements_checked:
            await self._obligation_graph.add_requirement(
                requirement_id=req.requirement_id,
                description=req.description,
                metadata={"type": req.type.value, "source": req.source}
            )

        # Add evidence for violations
        for violation in report.violations:
            if violation.rule:
                await self._obligation_graph.add_artifact(
                    artifact_id=violation.violation_id,
                    artifact_type="violation",
                    metadata={
                        "severity": violation.severity.value,
                        "location": violation.location
                    }
                )
                # Link as violating the requirement
                await self._obligation_graph.add_edge(
                    requirement_id=violation.rule,
                    artifact_id=violation.violation_id,
                    label="violates",
                    confidence=violation.confidence
                )

    def _check_completeness(self, report: ComplianceReport) -> bool:
        """Check if compliance analysis is complete.

        Args:
            report: Compliance report

        Returns:
            True if complete
        """
        # Check if all requirements were checked
        return len(report.requirements_checked) == len(self.requirements)

    def _identify_missing_context(self, report: ComplianceReport) -> list[str]:
        """Identify missing context.

        Args:
            report: Compliance report

        Returns:
            List of missing elements
        """
        missing = []

        if not report.licenses_found:
            missing.append("No dependency licenses analyzed")

        for violation in report.violations:
            if violation.confidence < 0.5:
                missing.append(f"Low confidence violation: {violation.description}")

        return missing

    def _compute_confidence(self, report: ComplianceReport) -> float:
        """Compute overall confidence.

        Args:
            report: Compliance report

        Returns:
            Confidence score
        """
        if not report.violations:
            return 1.0

        total_conf = sum(v.confidence for v in report.violations)
        return total_conf / len(report.violations)

    def _summarize_compliance(self, report: ComplianceReport) -> str:
        """Summarize compliance status.

        Args:
            report: Compliance report

        Returns:
            Summary
        """
        violation_summary = {}
        for v in report.violations:
            key = f"{v.type.value}_{v.severity.value}"
            violation_summary[key] = violation_summary.get(key, 0) + 1

        return (f"Compliance Status: {report.status.value}. "
                f"{len(report.violations)} violations found. "
                f"Risk level: {report.risk_assessment.get('risk_level', 'unknown')}. "
                f"Breakdown: {violation_summary}")


class ComplianceMergePolicy(MergePolicy[ComplianceReport]):
    """Policy for merging compliance analysis results."""

    async def merge(
        self,
        results: list[ScopeAwareResult[ComplianceReport]],
        context: MergeContext
    ) -> ScopeAwareResult[ComplianceReport]:
        """Merge multiple compliance reports.

        Args:
            results: Compliance results to merge
            context: Merge context

        Returns:
            Merged compliance report
        """
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            return results[0]

        coordination_notes: list[str] = []
        agent_weights = self._compute_agent_weights(results)

        violation_groups: dict[str, list[tuple[ComplianceViolation, str]]] = {}
        for result in results:
            agent_id = result.producer_agent_id or result.result_id
            for violation in result.content.violations:
                key = violation.location or violation.violation_id
                violation_groups.setdefault(key, []).append((violation, agent_id))

        merged_violations: list[ComplianceViolation] = []
        for location, entries in violation_groups.items():
            if len(entries) == 1:
                merged_violations.append(entries[0][0])
            else:
                merged_violations.append(
                    self._resolve_violation_conflict(
                        location,
                        entries,
                        agent_weights,
                        coordination_notes
                    )
                )

        all_licenses = {}
        for result in results:
            all_licenses.update(result.content.licenses_found)

        all_requirements = []
        seen_reqs = set()
        for result in results:
            for req in result.content.requirements_checked:
                if req.requirement_id not in seen_reqs:
                    all_requirements.append(req)
                    seen_reqs.add(req.requirement_id)

        merged_report = ComplianceReport(
            status=self._merge_status(results),
            violations=merged_violations,
            requirements_checked=all_requirements,
            licenses_found=all_licenses,
            license_conflicts=sum([r.content.license_conflicts for r in results], []),
            risk_assessment=self._merge_risk_assessment(results),
            recommendations=list(set(sum([r.content.recommendations for r in results], [])))
        )

        merged_scope = AnalysisScope(
            is_complete=all(r.scope.is_complete for r in results),
            missing_context=list(set(sum([r.scope.missing_context for r in results], []))),
            confidence=sum(r.scope.confidence for r in results) / len(results),
            reasoning=sum([r.scope.reasoning for r in results], []) + coordination_notes
        )

        return ComplianceResult(
            content=merged_report,
            scope=merged_scope
        )

    def _merge_status(self, results: list[ScopeAwareResult[ComplianceReport]]) -> ComplianceStatus:
        """Merge compliance statuses.

        Args:
            results: Results to merge

        Returns:
            Worst status
        """
        status_order = [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.UNKNOWN
        ]

        worst_status = ComplianceStatus.COMPLIANT
        worst_index = 0

        for result in results:
            idx = status_order.index(result.content.status)
            if idx > worst_index:
                worst_index = idx
                worst_status = result.content.status

        return worst_status

    def _merge_risk_assessment(
        self,
        results: list[ScopeAwareResult[ComplianceReport]]
    ) -> dict[str, Any]:
        """Merge risk assessments.

        Args:
            results: Results to merge

        Returns:
            Combined risk assessment
        """
        total_score = sum(r.content.risk_assessment.get("risk_score", 0) for r in results)
        total_critical = sum(r.content.risk_assessment.get("critical_violations", 0) for r in results)
        total_high = sum(r.content.risk_assessment.get("high_violations", 0) for r in results)

        risk_level = "low"
        if total_score > 50:
            risk_level = "critical"
        elif total_score > 20:
            risk_level = "high"
        elif total_score > 10:
            risk_level = "medium"

        return {
            "risk_score": total_score,
            "risk_level": risk_level,
            "critical_violations": total_critical,
            "high_violations": total_high
        }

    async def validate(
        self,
        original: list[ScopeAwareResult[ComplianceReport]],
        merged: ScopeAwareResult[ComplianceReport]
    ) -> ValidationResult:
        """Validate merged compliance report.

        Args:
            original: Original results
            merged: Merged result

        Returns:
            Validation result
        """
        issues = []

        # Check no violations were lost
        original_violations = sum(len(r.content.violations) for r in original)
        if len(merged.content.violations) < original_violations * 0.9:
            issues.append("Too many violations lost in merge")

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=1.0 if not issues else 0.8
        )

    def _resolve_violation_conflict(
        self,
        location: str,
        entries: list[tuple[ComplianceViolation, str]],
        agent_weights: dict[str, float],
        coordination_notes: list[str]
    ) -> ComplianceViolation:
        """Resolve conflicting violation assessments via negotiation utilities."""
        parties = [agent_id for _, agent_id in entries]
        issue = NegotiationIssue(
            issue_id=f"compliance_violation::{location}",
            description=f"Harmonize compliance severity at {location}",
            parties=parties,
            preferences={
                agent_id: {"confidence": violation.confidence}
                for violation, agent_id in entries
            }
        )

        offers = []
        for idx, (violation, agent_id) in enumerate(entries):
            utility = violation.confidence * agent_weights.get(agent_id, 1.0)
            offers.append(
                Offer(
                    offer_id=f"{issue.issue_id}::{idx}",
                    proposer=agent_id,
                    terms={
                        "severity": violation.severity.value,
                        "description": violation.description
                    },
                    utility=utility,
                    justification=f"risk={violation.risk}"
                )
            )

        efficiency = calculate_pareto_efficiency(offers, issue)
        pareto_ids = set(efficiency.get("pareto_optimal_offers", [])) or {offer.offer_id for offer in offers}
        winning_offer = max(
            (offer for offer in offers if offer.offer_id in pareto_ids),
            key=lambda offer: offer.utility
        )

        selected_violation = next(violation for violation, agent_id in entries if agent_id == winning_offer.proposer)
        merged_violation = selected_violation.model_copy(deep=True)

        # Union evidence/remediation to preserve insights
        merged_violation.evidence = sorted(
            set(sum((violation.evidence for violation, _ in entries), []))
        )
        merged_violation.remediation = merged_violation.remediation or next(
            (violation.remediation for violation, _ in entries if violation.remediation),
            merged_violation.remediation
        )
        merged_violation.risk = merged_violation.risk or next(
            (violation.risk for violation, _ in entries if violation.risk),
            merged_violation.risk
        )
        merged_violation.confidence = min(1.0, winning_offer.utility)

        coordination_notes.append(
            f"Negotiation reconciled compliance assessment at {location}; "
            f"selected severity '{merged_violation.severity.value}'."
        )
        return merged_violation

    def _compute_agent_weights(
        self,
        results: list[ScopeAwareResult[ComplianceReport]]
    ) -> dict[str, float]:
        """Compute coalition-based weights for compliance agents."""
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

    def _extract_capabilities(self, report: ComplianceReport) -> set[str]:
        """Extract capability tags from a compliance report."""
        caps: set[str] = set()
        for violation in report.violations:
            caps.add(f"{violation.type.value}_{violation.severity.value}")
        if report.licenses_found:
            caps.add("license_analysis")
        if report.risk_assessment.get("risk_level"):
            caps.add(f"risk_{report.risk_assessment['risk_level']}")
        return caps or {"baseline_compliance"}


# =============================================================================
# ComplianceVCMCapability - Extends VCMAnalysisCapability
# =============================================================================
#
# This capability extends VCMAnalysisCapability to provide atomic, composable
# primitives for compliance analysis across multiple pages.
#
# Design Philosophy:
# - Does NOT prescribe a workflow
# - LLM decides when/how to spawn workers, build obligation graphs, merge results
# - Domain-specific methods exposed as action_executors for LLM composition
#
# Example LLM-driven workflow:
#   1. LLM: spawn_workers(page_ids, cache_affine=True)
#   2. LLM: [wait for completion events]
#   3. LLM: merge_results(page_ids)
#   4. LLM: build_obligation_graph()  # domain-specific
#   5. LLM: get_violations_by_severity("critical")  # domain-specific query
# =============================================================================


class ComplianceVCMCapability(VCMAnalysisCapability):
    """Capability for distributed compliance analysis using VCMAnalysisCapability primitives.

    Extends VCMAnalysisCapability with compliance-specific:
    - Worker type (ComplianceAnalysisCapability)
    - Merge policy (ComplianceMergePolicy)
    - Domain methods (obligation graph, risk assessment)

    The LLM planner composes the atomic primitives from the base class with
    the domain-specific methods exposed here.

    Note: This is named ComplianceVCMCapability (not ComplianceAnalysisCapability)
    because ComplianceAnalysisCapability is already the worker capability name.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None,
    ):
        """Initialize compliance VCM capability.

        Args:
            agent: Agent using this capability (coordinator agent)
            scope_id: Blackboard scope ID
        """
        super().__init__(agent=agent, scope_id=scope_id)
        self._obligation_graph: ObligationGraph | None = None

    async def initialize(self) -> None:
        """Initialize compliance VCM capability."""
        await super().initialize()

        # Initialize obligation graph
        blackboard = await self.get_blackboard()
        self._obligation_graph = ObligationGraph(
            blackboard=blackboard,
        )

    # =========================================================================
    # Abstract Hook Implementations
    # =========================================================================

    def get_worker_capability_class(self) -> type[AgentCapability]:
        """Return ComplianceAnalysisCapability as the worker capability."""
        return ComplianceAnalysisCapability

    def get_worker_agent_type(self) -> str:
        """Return the worker agent type string."""
        return "polymathera.colony.samples.code_analysis.compliance.ComplianceAnalysisAgent"

    def get_domain_merge_policy(self) -> MergePolicy:
        """Return ComplianceMergePolicy for merging compliance reports."""
        return ComplianceMergePolicy()

    def get_analysis_parameters(self, **kwargs) -> dict[str, Any]:
        """Return compliance-specific analysis parameters.

        Args:
            **kwargs: Parameters from action call

        Returns:
            Parameters for compliance analysis workers
        """
        return {
            "check_licenses": kwargs.get("check_licenses", True),
            "check_security": kwargs.get("check_security", True),
            "compliance_types": kwargs.get("compliance_types", []),
            **kwargs,
        }

    # =========================================================================
    # Domain-Specific Action Executors
    # =========================================================================

    @action_executor(action_key="build_obligation_graph")
    async def build_obligation_graph(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build obligation graph from compliance violations.

        Call after merge_results() to create trackable obligations.
        LLM decides when this is appropriate.

        Args:
            page_ids: Pages to consider (if None, uses all analyzed pages)

        Returns:
            Dict with:
            - obligations_created: Number of obligations added
            - by_severity: Count by severity level
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        obligations_created = 0
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            violations = content.get("violations", [])

            for violation in violations:
                severity = violation.get("severity", "medium")
                if isinstance(severity, dict):
                    severity = severity.get("value", "medium")

                by_severity[severity] = by_severity.get(severity, 0) + 1
                obligations_created += 1

                # Store obligation in blackboard
                obligation_id = f"fix_{violation.get('location', page_id)}"
                blackboard = await self.get_blackboard()
                await blackboard.write(
                    f"{self.scope_id}:obligation:{obligation_id}",
                    {
                        "obligation_id": obligation_id,
                        "description": f"Fix {violation.get('type', 'compliance')} violation: {violation.get('description', '')}",
                        "page_id": page_id,
                        "severity": severity,
                        "status": "unfulfilled",
                        "evidence": violation.get("evidence", []),
                        "remediation": violation.get("remediation"),
                    },
                    created_by=self.agent.agent_id,
                )

        logger.info(
            f"ComplianceVCMCapability: built obligation graph with {obligations_created} obligations"
        )

        return {
            "obligations_created": obligations_created,
            "by_severity": by_severity,
        }

    @action_executor(action_key="get_violations_by_severity")
    async def get_violations_by_severity(
        self,
        severity: str,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get all violations of a specific severity level.

        Args:
            severity: Severity level ("critical", "high", "medium", "low")
            page_ids: Pages to search (if None, uses all analyzed pages)

        Returns:
            Dict with violations of the specified severity
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        filtered_violations = []

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            violations = content.get("violations", [])

            for violation in violations:
                v_severity = violation.get("severity", "medium")
                if isinstance(v_severity, dict):
                    v_severity = v_severity.get("value", "medium")

                if v_severity == severity:
                    filtered_violations.append({
                        "page_id": page_id,
                        **violation,
                    })

        return {
            "violations": filtered_violations,
            "count": len(filtered_violations),
            "severity": severity,
        }

    @action_executor(action_key="get_risk_assessment")
    async def get_risk_assessment(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get aggregated risk assessment across analyzed pages.

        Args:
            page_ids: Pages to include (if None, uses all analyzed pages)

        Returns:
            Dict with risk assessment metrics
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        total_risk_score = 0
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0

        severity_scores = {"critical": 10, "high": 5, "medium": 2, "low": 1}

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})
            violations = content.get("violations", [])

            for violation in violations:
                severity = violation.get("severity", "medium")
                if isinstance(severity, dict):
                    severity = severity.get("value", "medium")

                total_risk_score += severity_scores.get(severity, 0)

                if severity == "critical":
                    critical_count += 1
                elif severity == "high":
                    high_count += 1
                elif severity == "medium":
                    medium_count += 1
                else:
                    low_count += 1

        # Determine risk level
        if total_risk_score > 50:
            risk_level = "critical"
        elif total_risk_score > 20:
            risk_level = "high"
        elif total_risk_score > 10:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_score": total_risk_score,
            "risk_level": risk_level,
            "critical_violations": critical_count,
            "high_violations": high_count,
            "medium_violations": medium_count,
            "low_violations": low_count,
            "total_violations": critical_count + high_count + medium_count + low_count,
        }

    @action_executor(action_key="get_license_conflicts")
    async def get_license_conflicts(
        self,
        page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get all license conflicts across analyzed pages.

        Args:
            page_ids: Pages to search (if None, uses all analyzed pages)

        Returns:
            Dict with license conflicts
        """
        if page_ids is None:
            analyzed = await self.get_analyzed_pages()
            page_ids = analyzed.get("pages", [])

        results_data = await self.get_results(page_ids)
        results = results_data.get("results", {})

        all_conflicts = []
        all_licenses = {}

        for page_id, entry in results.items():
            result_data = entry.get("result", {})
            content = result_data.get("content", {})

            conflicts = content.get("license_conflicts", [])
            for conflict in conflicts:
                all_conflicts.append({
                    "page_id": page_id,
                    **conflict,
                })

            licenses = content.get("licenses_found", {})
            all_licenses.update(licenses)

        return {
            "conflicts": all_conflicts,
            "conflict_count": len(all_conflicts),
            "licenses_found": all_licenses,
            "license_count": len(all_licenses),
        }


# =============================================================================
# DEPRECATED: ComplianceCoordinatorCapability
# =============================================================================
# This class is superseded by ComplianceVCMCapability which extends
# VCMAnalysisCapability. The new design provides atomic, composable primitives
# that the LLM planner can compose into arbitrary workflows.
#
# Migration:
#   Old: ComplianceCoordinatorCapability with prescribed workflow
#   New: ComplianceVCMCapability with composable primitives
#
# TODO: Remove this class after migration is complete.
# =============================================================================

class ComplianceCoordinatorCapability(AgentCapability):
    """DEPRECATED: Use ComplianceVCMCapability instead.

    Uses event-driven architecture via @event_handler and @action_executor.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None,
        max_agents: int = 10,
        batching_policy: BatchingPolicy | None = None,
    ):
        """Initialize coordinator capability.

        Args:
            agent: Agent instance
            scope_id: Scope identifier
            max_agents: Maximum page agents
            batching_policy: Policy for cache-aware batch selection
        """
        super().__init__(agent=agent, scope_id=scope_id or agent.agent_id)
        self.max_agents = max_agents
        self._page_agents: dict[str, str] = {}  # page_id -> agent_id
        self._worker_handles: dict[str, Any] = {}  # page_id -> AgentHandle
        self._pending_results: dict[str, list[ScopeAwareResult[ComplianceReport]]] = {}
        self._obligation_graph: ObligationGraph | None = None
        self._batching_policy = batching_policy

        # Layer 0/1 capability references (initialized in initialize())
        self._agent_pool_cap: AgentPoolCapability | None = None
        self._result_cap: ResultCapability | None = None
        self._page_graph_cap: PageGraphCapability | None = None

    def get_action_group_description(self) -> str:
        return (
            "Compliance Coordination (DEPRECATED — use ComplianceVCMCapability). "
            "Distributes compliance analysis across pages via AgentPoolCapability, "
            "aggregates results, maintains ObligationGraph across agents."
        )

    async def initialize(self) -> None:
        """Initialize coordinator capability with Layer 0/1 capabilities."""
        await super().initialize()

        # AgentPoolCapability for agent lifecycle management
        self._agent_pool_cap = self.agent.get_capability_by_type(AgentPoolCapability)
        if not self._agent_pool_cap:
            self._agent_pool_cap = AgentPoolCapability(
                agent=self.agent,
                scope_id=self.scope_id,
                max_agents=self.max_agents,
            )
            await self._agent_pool_cap.initialize()
            self.agent.add_capability(self._agent_pool_cap)

        # ResultCapability for cluster-wide result visibility
        self._result_cap = self.agent.get_capability_by_type(ResultCapability)
        if not self._result_cap:
            self._result_cap = ResultCapability(
                agent=self.agent,
                scope_id=self.scope_id,
            )
            await self._result_cap.initialize()
            self.agent.add_capability(self._result_cap)

        # PageGraphCapability for standardized graph operations
        self._page_graph_cap = self.agent.get_capability_by_type(PageGraphCapability)
        if not self._page_graph_cap:
            self._page_graph_cap = PageGraphCapability(
                agent=self.agent,
                scope_id=self.scope_id,
            )
            await self._page_graph_cap.initialize()
            self.agent.add_capability(self._page_graph_cap)

    async def _ensure_obligation_graph(self) -> ObligationGraph:
        """Ensure obligation graph is initialized."""
        if self._obligation_graph is None:
            blackboard = await self.get_blackboard()
            self._obligation_graph = ObligationGraph(
                blackboard=blackboard,
            )
        return self._obligation_graph

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ComplianceCoordinatorCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ComplianceCoordinatorCapability")
        pass

    @event_handler(pattern="{scope_id}:request:*")
    async def handle_analysis_request(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle analysis request events from AgentHandle.run()."""
        request_id = event.key.split(":")[-1]
        request_data = event.value

        input_data = request_data.get("input", {})
        page_ids = input_data.get("page_ids", [])  # TODO: Use a symbolic selector for pages (e.g., a query or tag)
        compliance_types_raw = input_data.get("compliance_types", [])
        compliance_types = [ComplianceType(ct) for ct in compliance_types_raw] if compliance_types_raw else None

        return EventProcessingResult(
            immediate_action=Action(
                action_type="start_codebase_analysis",
                parameters={
                    "page_ids": page_ids,
                    "compliance_types": [ct.value for ct in (compliance_types or [])],
                    "request_id": request_id
                }
            )
        )

    @event_handler(pattern="analysis:compliance:result:*")  # TODO: Use a more specific pattern or namespace for worker results (e.g., include scope_id and agent_id)
    async def handle_worker_result(
        self,
        event: BlackboardEvent,
        repl: PolicyREPL
    ) -> EventProcessingResult | None:
        """Handle results from worker agents."""
        # Parse result
        result_data = event.value
        result = ScopeAwareResult[ComplianceReport](**result_data)

        # Accumulate results for pending request
        if self._pending_request_id:
            if self._pending_request_id not in self._pending_results:
                self._pending_results[self._pending_request_id] = []
            self._pending_results[self._pending_request_id].append(result)

            # Check if all results collected
            expected_count = len(self._page_agents)
            collected_count = len(self._pending_results[self._pending_request_id])

            if collected_count >= expected_count:
                # All results collected, trigger merge
                return EventProcessingResult(
                    immediate_action=Action(
                        action_type="complete_analysis",
                        parameters={
                            "request_id": self._pending_request_id
                        }
                    )
                )

        return None

    @action_executor(action_key="start_codebase_analysis")
    async def start_codebase_analysis(
        self,
        page_ids: list[str],
        compliance_types: list[str] | None = None,
        request_id: str | None = None
    ) -> str:
        """Start coordinated compliance analysis across multiple pages.

        Uses BatchingPolicy for cache-aware batch selection and AgentPoolCapability
        for agent lifecycle management.

        Args:
            page_ids: VCM page IDs to analyze
            compliance_types: Compliance types as string values
            request_id: Request ID for tracking

        Returns:
            Task ID for this analysis
        """
        import uuid
        task_id = request_id or f"compliance_task_{uuid.uuid4().hex[:8]}"
        self._pending_request_id = task_id
        self._pending_results[task_id] = []

        # Convert compliance types back to enums
        ct_enums = [ComplianceType(ct) for ct in (compliance_types or [])] if compliance_types else None

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

        # Spawn compliance agents using AgentPoolCapability
        await self._spawn_compliance_agents(batch_page_ids, ct_enums)

        return task_id

    async def _spawn_compliance_agents(
        self,
        page_ids: list[str],
        compliance_types: list[ComplianceType] | None
    ) -> None:
        """Spawn compliance agents for pages using AgentPoolCapability.

        Uses AgentPoolCapability for standardized agent lifecycle management,
        enabling cache-aware agent placement and resource optimization.

        Args:
            page_ids: Page IDs to analyze
            compliance_types: Compliance types to check
        """
        import uuid

        if not self._agent_pool_cap:
            raise RuntimeError("AgentPoolCapability not initialized")

        for page_id in page_ids:
            agent_id = f"compliance_agent_{uuid.uuid4().hex[:8]}"

            # Use AgentPoolCapability for standardized agent creation
            handle: AgentHandle = await self._agent_pool_cap.create_agent(
                agent_type="polymathera.colony.samples.code_analysis.compliance.ComplianceAnalysisAgent",
                capabilities=["polymathera.colony.samples.code_analysis.compliance.capabilities.ComplianceAnalysisCapability"],
                bound_pages=[page_id],
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
                metadata=AgentMetadata(parameters={
                    "page_id": page_id,
                    "compliance_types": [ct.value for ct in (compliance_types or [])]
                }),
            )

            self._page_agents[page_id] = agent_id
            self._worker_handles[page_id] = handle

            # Run worker and store result via ResultCapability
            # TODO: This still sequentializes worker runs. For true parallelization,
            # use asyncio.gather with handle.run() calls or rely on event handlers.
            run = await handle.run(
                {"page_ids": [page_id], "compliance_types": [ct.value for ct in (compliance_types or [])]},
                timeout=60
            )

            # Store result via ResultCapability for cluster-wide visibility
            if run.output_data and self._result_cap:
                await self._result_cap.store_partial(
                    result_id=f"compliance:{page_id}",
                    result=run.output_data,
                    source_agent=handle.child_agent_id,
                    source_pages=[page_id],
                    result_type="compliance_analysis",
                )

    @action_executor(action_key="complete_analysis")
    async def complete_analysis(
        self,
        request_id: str
    ) -> ComplianceResult:
        """Complete the analysis by merging results and building obligation graph.

        Args:
            request_id: Request ID to complete

        Returns:
            Merged compliance result
        """
        results = self._pending_results.get(request_id, [])

        if not results:
            empty_report = ComplianceReport(
                status=ComplianceStatus.UNKNOWN,
                violations=[],
                recommendations=[]
            )
            return ComplianceResult(
                content=empty_report,
                scope=AnalysisScope()
            )

        # Build obligation graph
        await self._build_obligation_graph(results)

        # Merge reports
        unified_report = await self._merge_reports(results)
        scope = self._compute_scope(results)

        merged_result = ComplianceResult(
            content=unified_report,
            scope=scope
        )

        # Write result to blackboard
        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=f"{self.scope_id}:result:{request_id}",  # TODO: Is scope_id needed? Blackboard already takes care of namespacing.
            value=merged_result.model_dump(),
        )

        # Cleanup
        del self._pending_results[request_id]
        self._pending_request_id = None
        self._page_agents.clear()
        self._worker_handles.clear()

        return merged_result

    async def _build_obligation_graph(
        self,
        results: list[ScopeAwareResult[ComplianceReport]]
    ) -> None:
        """Build obligation graph from compliance results.

        Args:
            results: Compliance reports
        """
        if not self._obligation_graph:
            await self._ensure_obligation_graph()

        if not self._obligation_graph:
            return

        # TODO: FIXME: These classes do not exist.
        from ....agents.blackboard.obligation_graph import Obligation, ObligationStatus

        for result in results:
            report = result.content

            # Add obligations from violations
            for violation in report.violations:
                obligation = Obligation(
                    obligation_id=f"fix_{violation.location}",
                    description=f"Fix {violation.type.value} violation: {violation.description}",
                    obligated_party="development_team",
                    beneficiary="compliance",
                    status=ObligationStatus.UNFULFILLED,
                    severity=violation.severity.value,
                    deadline=None,
                    evidence=violation.evidence
                )
                # TODO: FIXME: This method does not exist.
                await self._obligation_graph.add_obligation(obligation)

    async def _merge_reports(
        self,
        results: list[ScopeAwareResult[ComplianceReport]]
    ) -> ComplianceReport:
        """Merge compliance reports using MergeCapability."""
        if not results:
            return ComplianceReport(
                status=ComplianceStatus.UNKNOWN,
                violations=[],
                recommendations=[]
            )

        merge_cap = self.agent.get_capability_by_type(MergeCapability)
        if merge_cap is None:
            raise RuntimeError(
                "ComplianceCoordinatorCapability requires MergeCapability. "
                "Ensure it's added to the agent's capability_classes."
            )
        merged_result = await merge_cap.merge_results(results, MergeContext())

        return merged_result.content

    def _compute_scope(
        self,
        results: list[ScopeAwareResult[ComplianceReport]]
    ) -> AnalysisScope:
        """Compute unified scope."""
        if not results:
            return AnalysisScope()

        scopes = [r.scope for r in results]

        return AnalysisScope(
            is_complete=all(s.is_complete for s in scopes),
            missing_context=list(set(sum([s.missing_context for s in scopes], []))),
            confidence=sum(s.confidence for s in scopes) / len(scopes),
            reasoning=sum([s.reasoning for s in scopes], [])
        )

