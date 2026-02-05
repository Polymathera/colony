"""Core types for hypothesis-driven exploration and validation.

This module defines the data structures for the unified hypothesis game system:
- HypothesisContext: Flexible context for hypothesis formation
- Observation: Evidence that may lead to hypothesis formation
- Evidence: Supporting or contradicting evidence for hypotheses
- Formation triggers and filters

These types support diverse use cases:
- Code analysis (security vulnerabilities, architectural issues)
- Scientific research (experimental hypotheses)
- Debugging (root cause analysis)
- VCM analysis (cross-repository patterns)
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ...scope import ScopeAwareResult
    from ..state import GameOutcome
    from ...models import Hypothesis


# ============================================================================
# Subject Reference
# ============================================================================


class SubjectType(str, Enum):
    """Types of subjects for hypothesis formation."""

    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    REPOSITORY = "repository"
    DATASET = "dataset"
    SYSTEM = "system"
    BEHAVIOR = "behavior"
    CUSTOM = "custom"


class SubjectReference(BaseModel):
    """Reference to the subject of hypothesis formation.

    Polymorphic -- can reference different entity types depending on domain.
    The ``related_subjects`` field defines the initial search scope for evidence
    gathering strategies; each strategy interprets it according to its mechanism.

    Examples per domain::

        # Code analysis -- related files as initial search scope
        SubjectReference(
            subject_type=SubjectType.FILE,
            subject_id="src/auth/login.py",
            subject_data={"language": "python", "lines": 450},
            related_subjects=["src/auth/session.py", "src/auth/tokens.py"],
        )

        # VCM / repository analysis -- page IDs for context
        SubjectReference(
            subject_type=SubjectType.REPOSITORY,
            subject_id="org/payment-service",
            subject_data={"primary_language": "java"},
            related_subjects=["page:arch_overview", "page:api_design"],
        )

        # Scientific research -- related datasets for comparison
        SubjectReference(
            subject_type=SubjectType.DATASET,
            subject_id="experiment_2024_q3_results",
            subject_data={"sample_size": 1500, "variables": ["temperature", "pressure"]},
            related_subjects=["experiment_2024_q2_results", "baseline_data"],
        )

        # System debugging -- log/metrics sources to search
        SubjectReference(
            subject_type=SubjectType.BEHAVIOR,
            subject_id="intermittent_timeout_on_checkout",
            subject_data={"frequency": "5%", "affected_endpoint": "/api/checkout"},
            related_subjects=["logs:checkout_service", "metrics:db_latency"],
        )
    """

    subject_type: SubjectType = Field(
        description="Type of subject"
    )
    subject_id: str = Field(
        description="Identifier (path, name, ID)"
    )
    subject_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Subject-specific data (e.g., file content, function signature)"
    )
    related_subjects: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of related subjects that define the initial search scope for "
            "evidence gathering. For page-query strategies, these are the initial "
            "page IDs passed to IncrementalQueryCapability. For other strategies, "
            "they provide context-dependent scope (e.g., related files, datasets, "
            "or log sources to consider first)."
        ),
    )


# ============================================================================
# Observations
# ============================================================================


class ObservationType(str, Enum):
    """Types of observations that may lead to hypotheses."""

    PATTERN = "pattern"          # Recurring structure or behavior
    ANOMALY = "anomaly"          # Deviation from expected
    CORRELATION = "correlation"  # Statistical relationship
    GAP = "gap"                  # Missing expected element
    VULNERABILITY = "vulnerability"  # Security weakness
    DEPENDENCY = "dependency"    # Coupling or relationship
    CUSTOM = "custom"


class Observation(BaseModel):
    """A single observation that may lead to hypothesis formation.

    Observations are the raw inputs to hypothesis generation.
    They capture what was seen, where, and with what confidence.
    """

    observation_id: str = Field(
        default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}",
        description="Unique identifier"
    )
    observation_type: ObservationType = Field(
        description="Type of observation"
    )
    description: str = Field(
        description="What was observed"
    )
    source: str | None = Field(
        default=None,
        description="Where observed (file path, function name, etc.)"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in observation accuracy"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence (code snippets, line numbers)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional observation data"
    )


# ============================================================================
# Hypothesis Context
# ============================================================================


class HypothesisDomain(str, Enum):
    """Domains for hypothesis formation."""

    CODE_ANALYSIS = "code_analysis"
    SCIENTIFIC = "scientific"
    DEBUGGING = "debugging"
    ARCHITECTURAL = "architectural"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class HypothesisContext(BaseModel):
    """Flexible context for hypothesis formation.

    This is the primary input to HypothesisFormationStrategy.
    It captures everything needed to generate meaningful hypotheses.

    Use cases:
    - Code analysis: Subject=file/function, observations=patterns/vulnerabilities
    - Scientific: Subject=dataset, observations=correlations/anomalies
    - Debugging: Subject=system/behavior, observations=anomalies/traces
    - VCM analysis: Subject=repository, observations=architectural patterns
    """

    # Domain identification
    domain: HypothesisDomain = Field(
        description="Domain for hypothesis formation"
    )
    domain_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific configuration"
    )

    # Subject of investigation
    subject: SubjectReference = Field(
        description="What we're forming hypotheses about"
    )

    # Observations that may lead to hypotheses
    observations: list[Observation] = Field(
        default_factory=list,
        description="Observations that may warrant hypotheses"
    )

    # Existing knowledge
    prior_hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Previously formed hypotheses (avoid duplicates)"
    )
    known_facts: list[str] = Field(
        default_factory=list,
        description="Established facts that constrain hypotheses"
    )

    # Constraints and goals
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints on hypothesis formation"
    )
    investigation_goal: str | None = Field(
        default=None,
        description="What we're trying to understand/discover"
    )

    # Scope information (for integration with existing analysis)
    scope: ScopeAwareResult | None = Field(
        default=None,
        description="Scope-aware analysis result if available"
    )

    # Extensibility
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Evidence
# ============================================================================


class EvidenceType(str, Enum):
    """Types of evidence for hypothesis validation."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"


class Evidence(BaseModel):
    """Evidence for or against a hypothesis.

    Evidence is gathered during the GROUND phase of hypothesis games
    and used by arbiters to make final judgments.
    """

    evidence_id: str = Field(
        default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}",
        description="Unique identifier"
    )
    evidence_type: EvidenceType = Field(
        description="Whether evidence supports or contradicts"
    )
    description: str = Field(
        description="What the evidence shows"
    )
    source: str = Field(
        description="Where evidence was found"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in evidence validity"
    )
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw evidence data (code, measurements, etc.)"
    )


# ============================================================================
# Formation Triggers
# ============================================================================


class TriggerType(str, Enum):
    """Types of triggers that may cause hypothesis formation."""

    OBSERVATION = "observation"      # New observation detected
    GAME_OUTCOME = "game_outcome"    # Previous game completed
    CONTRADICTION = "contradiction"  # Contradiction detected
    SCHEDULED = "scheduled"          # Periodic formation
    EXPLICIT = "explicit"            # User-requested
    ANALYSIS_COMPLETE = "analysis_complete"  # Analysis finished


class HypothesisFormationTrigger(BaseModel):
    """Trigger that may cause hypothesis formation.

    Used by HypothesisFormationStrategy.should_form_hypothesis()
    to decide if new hypotheses should be generated.
    """

    trigger_type: TriggerType = Field(
        description="What triggered potential formation"
    )
    source: str | None = Field(
        default=None,
        description="Source of trigger (agent_id, game_id, etc.)"
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Trigger-specific data"
    )
    timestamp: float | None = Field(
        default=None,
        description="When trigger occurred"
    )


# ============================================================================
# Evaluation Results
# ============================================================================


class EvaluationDecision(str, Enum):
    """Possible evaluation decisions."""

    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"
    NEED_MORE_EVIDENCE = "need_more_evidence"


class EvaluationResult(BaseModel):
    """Result of hypothesis evaluation.

    Produced by HypothesisEvaluationStrategy and used by arbiters.
    """

    decision: EvaluationDecision = Field(
        description="Recommended decision"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in decision"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of decision"
    )
    supporting_evidence_count: int = Field(
        default=0,
        description="Number of supporting evidence items"
    )
    contradicting_evidence_count: int = Field(
        default=0,
        description="Number of contradicting evidence items"
    )
    unresolved_challenges: int = Field(
        default=0,
        description="Number of unaddressed challenges"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )


# ============================================================================
# Tracking Filters
# ============================================================================


class HypothesisStatus(str, Enum):
    """Status of a tracked hypothesis."""

    PENDING = "pending"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    REVISED = "revised"


class HypothesisFilter(BaseModel):
    """Filter for querying tracked hypotheses."""

    status: HypothesisStatus | None = Field(
        default=None,
        description="Filter by status"
    )
    game_id: str | None = Field(
        default=None,
        description="Filter by associated game"
    )
    domain: HypothesisDomain | None = Field(
        default=None,
        description="Filter by domain"
    )
    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    subject_type: SubjectType | None = Field(
        default=None,
        description="Filter by subject type"
    )
    subject_id: str | None = Field(
        default=None,
        description="Filter by specific subject"
    )
