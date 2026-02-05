
from enum import Enum
import time
from typing import Any
from pydantic import BaseModel, Field

from ....agents.patterns import ScopeAwareResult


class ComplianceType(str, Enum):
    """Types of compliance requirements."""

    LICENSE = "license"  # Open source licenses
    REGULATORY = "regulatory"  # Legal regulations (GDPR, HIPAA, etc.)
    SECURITY = "security"  # Security policies
    ORGANIZATIONAL = "organizational"  # Company policies
    INDUSTRY = "industry"  # Industry standards (PCI-DSS, ISO, etc.)
    ARCHITECTURAL = "architectural"  # Architecture guidelines
    QUALITY = "quality"  # Code quality standards


class ComplianceSeverity(str, Enum):
    """Severity of compliance issues."""

    CRITICAL = "critical"  # Must fix immediately
    HIGH = "high"  # Should fix soon
    MEDIUM = "medium"  # Should address
    LOW = "low"  # Nice to fix
    INFO = "info"  # Informational only


class ComplianceStatus(str, Enum):
    """Overall compliance status."""

    COMPLIANT = "compliant"  # Fully compliant
    PARTIALLY_COMPLIANT = "partially_compliant"  # Some issues
    NON_COMPLIANT = "non_compliant"  # Major violations
    UNKNOWN = "unknown"  # Cannot determine


class License(BaseModel):
    """Software license information."""

    name: str = Field(
        description="License name (e.g., MIT, GPL-3.0)"
    )

    spdx_id: str | None = Field(
        default=None,
        description="SPDX license identifier"
    )

    category: str = Field(
        default="permissive",
        description="Category: permissive, copyleft, proprietary"
    )

    permissions: list[str] = Field(
        default_factory=list,
        description="What the license permits"
    )

    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that must be met"
    )

    limitations: list[str] = Field(
        default_factory=list,
        description="What the license prohibits"
    )

    compatible_with: list[str] = Field(
        default_factory=list,
        description="Compatible licenses"
    )

    incompatible_with: list[str] = Field(
        default_factory=list,
        description="Incompatible licenses"
    )


class ComplianceViolation(BaseModel):
    """A compliance violation."""

    violation_id: str = Field(
        description="Unique violation identifier"
    )

    type: ComplianceType = Field(
        description="Type of compliance issue"
    )

    severity: ComplianceSeverity = Field(
        description="Severity level"
    )

    description: str = Field(
        description="What the violation is"
    )

    location: str = Field(
        description="Where violation occurs (file:line)"
    )

    rule: str | None = Field(
        default=None,
        description="Specific rule violated"
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence of violation"
    )

    remediation: str | None = Field(
        default=None,
        description="How to fix the violation"
    )

    risk: str | None = Field(
        default=None,
        description="Risk if not addressed"
    )

    confidence: float = Field(
        default=0.8,
        description="Confidence in violation"
    )


class ComplianceRequirement(BaseModel):
    """A compliance requirement to check."""

    requirement_id: str = Field(
        description="Requirement identifier"
    )

    type: ComplianceType = Field(
        description="Type of requirement"
    )

    description: str = Field(
        description="What must be complied with"
    )

    source: str = Field(
        description="Source of requirement (law, policy, etc.)"
    )

    checks: list[str] = Field(
        default_factory=list,
        description="Specific checks to perform"
    )

    exceptions: list[str] = Field(
        default_factory=list,
        description="Allowed exceptions"
    )


class ComplianceReport(BaseModel):
    """Complete compliance analysis report."""

    status: ComplianceStatus = Field(
        description="Overall compliance status"
    )

    violations: list[ComplianceViolation] = Field(
        default_factory=list,
        description="All violations found"
    )

    requirements_checked: list[ComplianceRequirement] = Field(
        default_factory=list,
        description="Requirements that were checked"
    )

    licenses_found: dict[str, License] = Field(
        default_factory=dict,
        description="Licenses detected in dependencies"
    )

    license_conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="License compatibility issues"
    )

    risk_assessment: dict[str, Any] = Field(
        default_factory=dict,
        description="Overall risk assessment"
    )

    recommendations: list[str] = Field(
        default_factory=list,
        description="Compliance recommendations"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When analysis was performed"
    )


class ComplianceResult(ScopeAwareResult[ComplianceReport]):
    """Compliance analysis result with scope awareness."""

    pass
