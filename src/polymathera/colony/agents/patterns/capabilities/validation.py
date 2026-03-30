"""Validation policies for analysis results.

This module extends the existing ValidationPolicy from blackboard with
analysis-specific validation strategies:
- CrossShardConsistencyValidator: Validates consistency across page boundaries
- EvidenceBasedValidator: Requires evidence for claims
- ConsensusValidator: Validates through multi-agent consensus
- MultiLevelValidator: Combines multiple validation strategies

Note: Base ValidationPolicy already exists in blackboard/types.py.
This module extends it for distributed analysis use cases.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from overrides import override
from pydantic import BaseModel, Field

from ..scope import ScopeAwareResult
from ...base import Agent, AgentCapability
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...models import AgentSuspensionState
from ....utils import setup_logger
from ..actions import action_executor


logger = setup_logger(__name__)

T = TypeVar('T')


class ValidationContext(BaseModel):
    """Context for validation operations."""

    validation_type: str | None = Field(
        default=None,
        description="Type of validation to perform"
    )

    strictness: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Validation strictness (0.0 = lenient, 1.0 = strict)"
    )

    require_evidence: bool = Field(
        default=True,
        description="Whether to require evidence for claims"
    )

    cross_shard_validation: bool = Field(
        default=False,
        description="Whether to validate across shard boundaries"
    )

    consensus_required: bool = Field(
        default=False,
        description="Whether to require consensus validation"
    )

    requesting_agent_id: str | None = Field(
        default=None,
        description="Agent requesting validation"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation context"
    )


class ValidationIssue(BaseModel):
    """A validation issue found in a result."""

    severity: str = Field(
        description="Severity: 'critical', 'high', 'medium', 'low'"
    )

    issue_type: str = Field(
        description="Type of issue"
    )

    description: str = Field(
        description="Description of the issue"
    )

    location: str | None = Field(
        default=None,
        description="Where issue was found"
    )

    suggestion: str | None = Field(
        default=None,
        description="How to fix the issue"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional issue metadata"
    )


class ValidationResult(BaseModel):
    """Result of validation."""

    is_valid: bool = Field(
        description="Whether result is valid"
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in validation judgment"
    )

    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="Issues found"
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings (non-critical issues)"
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions"
    )

    validated_by: str | None = Field(
        default=None,
        description="Validator agent ID"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When validation was performed"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation metadata"
    )

    def has_critical_issues(self) -> bool:
        """Check if validation found critical issues."""
        return any(issue.severity == "critical" for issue in self.issues)

    def has_high_severity_issues(self) -> bool:
        """Check if validation found high severity issues."""
        return any(issue.severity in ["critical", "high"] for issue in self.issues)


class AnalysisValidationPolicy(ABC, Generic[T]):
    """Abstract base for analysis validation policies.

    Extends blackboard ValidationPolicy for analysis-specific validation.
    """

    @abstractmethod
    async def validate(
        self,
        result: ScopeAwareResult[T],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate analysis result.

        Args:
            result: Result to validate
            context: Validation context

        Returns:
            Validation result
        """
        pass


class ConsistencyChecker(ABC, Generic[T]):
    """Abstract base for consistency checkers between results."""

    @abstractmethod
    async def check(
        self,
        result1: ScopeAwareResult[T],
        result2: ScopeAwareResult[T]
    ) -> ValidationIssue | None:
        """Check if two results are consistent.

        Args:
            result1: First result
            result2: Second result

        Returns:
            Validation issue if inconsistent, None if consistent
        """
        pass


class ResultReader(ABC, Generic[T]):
    """Reads results by page ID."""

    @abstractmethod
    async def get_result_by_page_id(
        self,
        page_id: str
    ) -> ScopeAwareResult[T] | None:
        """Get result for a given page ID.

        Args:
            page_id: Page ID to look up

        Returns:
            Result for the given page ID, or None if not found
        """
        pass


class CrossShardConsistencyValidator(AnalysisValidationPolicy[T]):
    """Validates consistency across shard boundaries.

    Checks if cross-shard references are consistent and results don't contradict.
    """

    def __init__(self, result_reader: ResultReader[T], consistency_checker: ConsistencyChecker[T]):
        """Initialize validator.

        Args:
            result_reader: Result reader for querying related results
            consistency_checker: Consistency checker for comparing results
        """
        self.result_reader = result_reader
        self.consistency_checker = consistency_checker

    async def validate(
        self,
        result: ScopeAwareResult[T],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate cross-shard consistency.

        Args:
            result: Result to validate
            context: Validation context

        Returns:
            Validation result
        """

        # Check if result has related shards
        if not result.scope.related_shards:
            # No cross-shard references, trivially consistent
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                validated_by=context.requesting_agent_id
            )

        issues = []
        warnings = []

        # Check each related shard
        for related_shard_id in result.scope.related_shards:
            # Try to get related result
            related_result = await self.result_reader.get_result_by_page_id(related_shard_id)

            if not related_result:
                warnings.append(f"Related shard {related_shard_id} not yet analyzed")
                continue

            # Check consistency
            issue = await self.consistency_checker.check(result, related_result)
            if issue:
                issues.append(issue)

        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=0.9 if len(issues) == 0 else 0.5,
            issues=issues,
            warnings=warnings,
            validated_by=context.requesting_agent_id
        )


class EvidenceBasedValidator(AnalysisValidationPolicy[T]):
    """Validates that claims are supported by evidence.

    Checks if result includes sufficient evidence for its claims.
    """

    async def validate(
        self,
        result: ScopeAwareResult[T],
        context: ValidationContext
    ) -> ValidationResult:
        """Validate evidence support.

        Args:
            result: Result to validate
            context: Validation context

        Returns:
            Validation result
        """
        issues = []
        warnings = []

        # Check if result has evidence
        if not result.scope.evidence:
            if context.require_evidence:
                issues.append(ValidationIssue(
                    severity="high",
                    issue_type="missing_evidence",
                    description="Result has no supporting evidence",
                    suggestion="Add evidence references (code locations, citations, etc.)"
                ))
            else:
                warnings.append("No evidence provided for claims")

        # Check confidence vs evidence ratio
        evidence_count = len(result.scope.evidence)
        if result.scope.confidence > 0.8 and evidence_count < 2:
            warnings.append(
                f"High confidence ({result.scope.confidence:.2f}) "
                f"with limited evidence ({evidence_count} items)"
            )

        # Check if assumptions are documented
        if not result.scope.assumptions and result.scope.confidence < 1.0:
            warnings.append("No assumptions documented for uncertain result")

        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=0.8 if len(issues) == 0 else 0.5,
            issues=issues,
            warnings=warnings,
            validated_by=context.requesting_agent_id
        )


class MultiLevelValidator(AnalysisValidationPolicy[T]):
    """Combines multiple validation strategies.

    Runs multiple validators and aggregates results.
    """

    def __init__(self, validators: list[AnalysisValidationPolicy[T]]):
        """Initialize with list of validators.

        Args:
            validators: List of validation policies to apply
        """
        self.validators = validators

    async def validate(
        self,
        result: ScopeAwareResult[T],
        context: ValidationContext
    ) -> ValidationResult:
        """Run all validators and aggregate results.

        Args:
            result: Result to validate
            context: Validation context

        Returns:
            Aggregated validation result
        """
        all_results: list[ValidationResult] = []

        # Run all validators
        for validator in self.validators:
            val_result = await validator.validate(result, context)
            all_results.append(val_result)

        # Aggregate results
        all_issues = []
        all_warnings = []
        all_suggestions = []

        for val_result in all_results:
            all_issues.extend(val_result.issues)
            all_warnings.extend(val_result.warnings)
            all_suggestions.extend(val_result.suggestions)

        # Result is valid only if all validators say it's valid
        is_valid = all(vr.is_valid for vr in all_results)

        # Aggregate confidence (minimum across validators)
        confidence = min(vr.confidence for vr in all_results) if all_results else 0.0

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=all_issues,
            warnings=all_warnings,
            suggestions=all_suggestions,
            validated_by=context.requesting_agent_id
        )


class ConsensusResult(BaseModel):
    """Result of consensus validation."""

    has_consensus: bool = Field(
        description="Whether consensus was reached"
    )

    confidence: float = Field(
        description="Aggregate confidence"
    )

    merged_result: ScopeAwareResult | None = Field(
        default=None,
        description="Merged result if consensus reached"
    )

    conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conflicts if no consensus"
    )

    participating_agents: list[str] = Field(
        default_factory=list,
        description="Agents that participated"
    )

    agreement_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Level of agreement (1.0 = perfect agreement)"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When consensus check was performed"
    )


class ConsensusValidator:
    """Validates results through consensus from multiple agents.

    Multiple agents analyze the same content and results are validated
    if they agree (reach consensus).
    """

    async def validate_consensus(
        self,
        results: list[ScopeAwareResult[T]],
        threshold: float = 0.7
    ) -> ConsensusResult:
        """Check if results agree (consensus).

        Args:
            results: Results from multiple agents
            threshold: Agreement threshold (0.7 = 70% agreement needed)

        Returns:
            Consensus result
        """
        if len(results) < 2:
            # Need at least 2 results for consensus
            return ConsensusResult(
                has_consensus=True,
                confidence=1.0,
                merged_result=results[0] if results else None
            )

        # Check for agreement
        # This is simplified - real implementation would use LLM to check
        # semantic agreement, not just field-by-field comparison

        # For now, check if confidences are similar
        confidences = [r.scope.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        max_deviation = max(abs(c - avg_confidence) for c in confidences)

        has_consensus = max_deviation < (1.0 - threshold)

        if has_consensus:
            # Merge results (simple average for now)
            # Real implementation would use appropriate MergePolicy
            merged = results[0]  # Placeholder

            return ConsensusResult(
                has_consensus=True,
                confidence=avg_confidence,
                merged_result=merged,
                participating_agents=[r.producer_agent_id for r in results if r.producer_agent_id],
                agreement_level=1.0 - max_deviation
            )
        else:
            # No consensus - identify conflicts
            conflicts = self._identify_conflicts(results)

            return ConsensusResult(
                has_consensus=False,
                confidence=avg_confidence,
                merged_result=None,
                conflicts=conflicts,
                participating_agents=[r.producer_agent_id for r in results if r.producer_agent_id],
                agreement_level=1.0 - max_deviation
            )

    def _identify_conflicts(
        self,
        results: list[ScopeAwareResult[T]]
    ) -> list[dict[str, Any]]:
        """Identify conflicts between results.

        Args:
            results: Results to compare

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # TODO: Implement detailed conflict identification logic

        # Compare confidence levels
        confidences = [r.scope.confidence for r in results]
        if max(confidences) - min(confidences) > 0.3:
            conflicts.append({
                "type": "confidence_mismatch",
                "description": f"Confidence varies from {min(confidences):.2f} to {max(confidences):.2f}",
                "severity": "medium"
            })

        # Compare completeness
        completeness = [r.scope.is_complete for r in results]
        if not all(c == completeness[0] for c in completeness):
            conflicts.append({
                "type": "completeness_mismatch",
                "description": "Some results are complete, others are not",
                "severity": "high"
            })

        return conflicts


class Contradiction(BaseModel):
    """Represents a contradiction between two results."""

    result1_id: str = Field(
        description="ID of first result"
    )

    result2_id: str = Field(
        description="ID of second result"
    )

    contradiction_type: str = Field(
        description="Type of contradiction"
    )

    details: str = Field(
        description="Description of contradiction"
    )

    resolution_suggestion: str | None = Field(
        default=None,
        description="Suggested resolution"
    )

    severity: str = Field(
        default="medium",
        description="Severity: 'critical', 'high', 'medium', 'low'"
    )


class ResolvedResult(BaseModel):
    """Result of contradiction resolution."""

    contradiction: Contradiction = Field(
        description="The contradiction that was addressed"
    )

    resolved: bool = Field(
        description="Whether contradiction was resolved"
    )

    resolution_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence used to resolve"
    )

    recommended_result: ScopeAwareResult | None = Field(
        default=None,
        description="Recommended result after resolution"
    )

    explanation: str | None = Field(
        default=None,
        description="Explanation of resolution"
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in resolution"
    )


class ContradictionResolver:
    """Detects and resolves contradictions in analysis results.

    When multiple results contradict each other, this resolver:
    1. Detects contradictions
    2. Generates queries to gather more evidence
    3. Resolves based on evidence
    """

    def __init__(self, agent: Agent):
        """Initialize resolver.

        Args:
            agent: Agent for contradiction detection
        """
        self.agent = agent

    async def detect_contradictions(
        self,
        results: list[ScopeAwareResult[T]]
    ) -> list[Contradiction]:
        """Detect contradictions between results.

        Args:
            results: Results to check

        Returns:
            List of contradictions found
        """
        contradictions = []

        # Compare all pairs
        for i, result1 in enumerate(results):
            for result2 in results[i+1:]:

                # TODO: Use LLM to check for contradictions. Simplified check for now

                # Check if confidences wildly differ for related results
                if result1.result_id in result2.scope.related_shards:
                    if abs(result1.scope.confidence - result2.scope.confidence) > 0.4:
                        contradictions.append(Contradiction(
                            result1_id=result1.result_id,
                            result2_id=result2.result_id,
                            contradiction_type="confidence_mismatch",
                            details="Related results have very different confidence levels",
                            severity="medium"
                        ))

        return contradictions

    async def resolve_contradiction(
        self,
        contradiction: Contradiction,
        result1: ScopeAwareResult[T],
        result2: ScopeAwareResult[T]
    ) -> ResolvedResult:
        """Resolve contradiction through additional analysis.

        Args:
            contradiction: Contradiction to resolve
            result1: First result
            result2: Second result

        Returns:
            Resolution result
        """
        # Generate queries to resolve contradiction
        resolution_queries = await self._generate_resolution_queries(contradiction)

        # TODO: Process queries and gather evidence
        # Then use LLM to resolve based on evidence

        return ResolvedResult(
            contradiction=contradiction,
            resolved=False,  # Placeholder
            resolution_evidence=[],
            recommended_result=None
        )

    async def _generate_resolution_queries(
        self,
        contradiction: Contradiction
    ) -> list[str]:
        """Generate queries to resolve contradiction.

        Args:
            contradiction: Contradiction to resolve

        Returns:
            List of query strings
        """
        # TODO: Use LLM to generate targeted queries
        return []


class ValidationCapability(AgentCapability):
    """Capability for validating analysis results.

    Wraps AnalysisValidationPolicy implementations and exposes:
    - validate_result: Validate a single result
    - validate_consensus: Check consensus across multiple results
    - detect_contradictions: Find contradictions between results
    - resolve_contradiction: Resolve a detected contradiction

    Uses dynamic lookup via self.agent.get_capability_by_type() for
    coordination with other capabilities.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "validation",
        capability_key: str = "validation",
    ):
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=None, capability_key=capability_key)
        self.validators: list[AnalysisValidationPolicy] = []
        self.contradiction_resolver: ContradictionResolver | None = None
        self._consensus_validator = ConsensusValidator()

    def get_action_group_description(self) -> str:
        return (
            "Analysis Result Validation — runs pluggable validators on results. "
            "Requires set_validators() before use. Validators run sequentially on each result. "
            "Can also check consensus across multiple results, detect contradictions between results, "
            "and resolve contradictions using LLM-assisted evidence gathering. "
            "Contradiction resolution is expensive (LLM call); detection is cheap."
        )

    def set_validators(self, validators: list[AnalysisValidationPolicy]) -> None:
        """Configure validators after instantiation.

        Args:
            validators: List of AnalysisValidationPolicy implementations
        """
        self.validators = validators

    def add_validator(self, validator: AnalysisValidationPolicy) -> None:
        """Add a validator to the list.

        Args:
            validator: AnalysisValidationPolicy to add
        """
        self.validators.append(validator)

    def set_contradiction_resolver(self, resolver: ContradictionResolver) -> None:
        """Configure the contradiction resolver.

        Args:
            resolver: ContradictionResolver instance
        """
        self.contradiction_resolver = resolver

    def _get_or_create_resolver(self) -> ContradictionResolver:
        """Get contradiction resolver, creating one if needed."""
        if self.contradiction_resolver is None:
            self.contradiction_resolver = ContradictionResolver(self.agent)
        return self.contradiction_resolver

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ValidationCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ValidationCapability")
        pass

    @action_executor()
    async def validate_result(
        self,
        result: ScopeAwareResult,
        context: ValidationContext | None = None
    ) -> ValidationResult:
        """Validate analysis result (plannable by LLM).

        Runs all configured validators and aggregates results.

        Args:
            result: Result to validate
            context: Optional validation context

        Returns:
            Validation result with issues and confidence
        """
        if context is None:
            context = ValidationContext()

        if not self.validators:
            # No validators configured - return trivially valid
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                validated_by=self.agent.agent_id
            )

        multi_validator = MultiLevelValidator(self.validators)
        return await multi_validator.validate(result, context)

    @action_executor()
    async def validate_consensus(
        self,
        results: list[ScopeAwareResult],
        threshold: float = 0.7
    ) -> ConsensusResult:
        """Check if multiple results agree (consensus validation).

        Args:
            results: Results from multiple sources/agents
            threshold: Agreement threshold (0.7 = 70% agreement needed)

        Returns:
            Consensus result indicating agreement level
        """
        return await self._consensus_validator.validate_consensus(results, threshold)

    @action_executor()
    async def detect_contradictions(
        self,
        results: list[ScopeAwareResult]
    ) -> list[Contradiction]:
        """Detect contradictions between results.

        Args:
            results: Results to check for contradictions

        Returns:
            List of contradictions found
        """
        resolver = self._get_or_create_resolver()
        return await resolver.detect_contradictions(results)

    @action_executor()
    async def resolve_contradiction(
        self,
        contradiction: Contradiction,
        result1: ScopeAwareResult,
        result2: ScopeAwareResult
    ) -> ResolvedResult:
        """Resolve a detected contradiction.

        Args:
            contradiction: The contradiction to resolve
            result1: First conflicting result
            result2: Second conflicting result

        Returns:
            Resolution result with recommended outcome
        """
        resolver = self._get_or_create_resolver()
        return await resolver.resolve_contradiction(contradiction, result1, result2)


# Utility functions

async def validate_with_multiple_policies(
    result: ScopeAwareResult[T],
    validators: list[AnalysisValidationPolicy[T]],
    context: ValidationContext | None = None
) -> ValidationResult:
    """Validate result with multiple policies.

    Args:
        result: Result to validate
        validators: List of validators
        context: Optional validation context

    Returns:
        Aggregated validation result
    """
    if context is None:
        context = ValidationContext()

    multi_validator = MultiLevelValidator(validators)
    return await multi_validator.validate(result, context)


def is_validation_passing(
    validation: ValidationResult,
    allow_warnings: bool = True
) -> bool:
    """Check if validation is passing.

    Args:
        validation: Validation result
        allow_warnings: Whether warnings are acceptable

    Returns:
        True if validation passes
    """
    if not validation.is_valid:
        return False

    if validation.has_critical_issues():
        return False

    if not allow_warnings and validation.warnings:
        return False

    return True

