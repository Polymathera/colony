
import uuid
from enum import Enum
from pydantic import BaseModel, Field
import time
from typing import Any

from polymathera.colony.agents.patterns import (
    ScopeAwareResult,
    RelationshipGraph,
)
from polymathera.colony.agents.blackboard import CausalityTimeline



class CaseInsensitiveEnum(str, Enum):
    """str Enum that accepts values case-insensitively (e.g., 'MODIFICATION' → 'modification')."""

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            lower = value.lower()
            for member in cls:
                if member.value == lower:
                    return member
        return None


class ChangeType(CaseInsensitiveEnum):
    """Types of code changes."""

    ADDITION = "addition"  # New code added
    MODIFICATION = "modification"  # Existing code changed
    DELETION = "deletion"  # Code removed
    REFACTORING = "refactoring"  # Structure changed, behavior same
    BUG_FIX = "bug_fix"  # Defect correction
    FEATURE = "feature"  # New functionality
    PERFORMANCE = "performance"  # Optimization
    SECURITY = "security"  # Security improvement
    DOCUMENTATION = "documentation"  # Doc changes


class ImpactType(CaseInsensitiveEnum):
    """Types of impact from changes."""

    FUNCTIONAL = "functional"  # Behavior changes
    PERFORMANCE = "performance"  # Speed/resource impact
    SECURITY = "security"  # Security implications
    API = "api"  # Interface changes
    DATA = "data"  # Data format/schema changes
    COMPATIBILITY = "compatibility"  # Breaking changes
    TEST = "test"  # Test suite impact
    DOCUMENTATION = "documentation"  # Doc updates needed
    DEPLOYMENT = "deployment"  # Deployment impact


class ImpactSeverity(CaseInsensitiveEnum):
    """Severity of change impact."""

    CRITICAL = "critical"  # System-wide impact
    HIGH = "high"  # Major component impact
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Minor impact
    MINIMAL = "minimal"  # Negligible impact


class CodeChange(BaseModel):
    """A code change to analyze."""

    change_id: str = Field(
        default_factory=lambda: f"change-{uuid.uuid4().hex[:8]}",
        description="Unique change identifier (auto-generated if not provided)"
    )

    file_path: str = Field(
        description="File being changed"
    )

    line_range: tuple[int, int] = Field(
        default=(0, 0),
        description="Lines affected (start, end). (0, 0) means the entire file."
    )

    change_type: ChangeType = Field(
        description="Type of change"
    )

    old_code: str | None = Field(
        default=None,
        description="Code before change"
    )

    new_code: str | None = Field(
        default=None,
        description="Code after change"
    )

    description: str = Field(
        description="What the change does"
    )

    author: str | None = Field(
        default=None,
        description="Who made the change"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When change was made"
    )


class ImpactedComponent(BaseModel):
    """A component impacted by changes."""

    component_id: str = Field(
        description="Component identifier"
    )

    component_type: str = Field(
        description="Type: function, class, module, test, etc."
    )

    file_path: str = Field(
        description="File containing component"
    )

    impact_types: list[ImpactType] = Field(
        default_factory=list,
        description="How component is impacted"
    )

    severity: ImpactSeverity = Field(
        description="Impact severity"
    )

    description: str = Field(
        description="Description of impact"
    )

    requires_update: bool = Field(
        default=False,
        description="Whether component needs updating"
    )

    confidence: float = Field(
        default=0.7,
        description="Confidence in impact assessment"
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence of impact"
    )


class ImpactStep(BaseModel):
    """A step in impact propagation."""

    from_component: str = Field(
        description="Component propagating impact"
    )

    to_component: str = Field(
        description="Component receiving impact"
    )

    relationship: str = Field(
        description="How they're related: calls, inherits, imports, etc."
    )

    impact_reason: str = Field(
        description="Why impact propagates"
    )


class ImpactPath(BaseModel):
    """Path of impact propagation."""

    path_id: str = Field(
        description="Path identifier"
    )

    source_change: str = Field(
        description="Change that starts the path"
    )

    steps: list[ImpactStep] = Field(
        default_factory=list,
        description="Steps in impact propagation"
    )

    final_impact: ImpactedComponent = Field(
        description="Final impacted component"
    )

    path_type: str = Field(
        default="direct",
        description="Path type: direct, indirect, transitive"
    )

    total_severity: ImpactSeverity = Field(
        description="Overall path severity"
    )


class ChangeImpactReport(BaseModel):
    """Complete change impact analysis report."""

    model_config = {"arbitrary_types_allowed": True}

    source_page_id: str | None = Field(
        default=None,
        description="VCM page ID that produced this report (for single-page reports)"
    )

    changes: list[CodeChange] = Field(
        description="Changes being analyzed"
    )

    impacted_components: list[ImpactedComponent] = Field(
        default_factory=list,
        description="All impacted components"
    )

    impact_paths: list[ImpactPath] = Field(
        default_factory=list,
        description="Paths of impact propagation"
    )

    impact_graph: RelationshipGraph | None = Field(
        default=None,
        description="Graph of impact relationships"
    )

    risk_assessment: dict[str, Any] = Field(
        default_factory=dict,
        description="Risk analysis of changes"
    )

    test_impact: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Tests affected by changes"
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for handling impact"
    )

    breaking_changes: list[str] = Field(
        default_factory=list,
        description="Identified breaking changes"
    )

    timeline: CausalityTimeline | None = Field(
        default=None,
        description="Timeline of cascading impacts"
    )


class ChangeImpactResult(ScopeAwareResult[ChangeImpactReport]):
    """Change impact analysis result with scope awareness."""

    pass

