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
from enum import Enum

from pydantic import BaseModel, Field

from ....agents.patterns import ScopeAwareResult

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """High-level intent categories."""

    BUSINESS_LOGIC = "business_logic"  # Core business rules
    DATA_PROCESSING = "data_processing"  # Transform/filter data
    VALIDATION = "validation"  # Input/output validation
    ERROR_HANDLING = "error_handling"  # Exception management
    PERFORMANCE = "performance"  # Optimization code
    SECURITY = "security"  # Security measures
    INTEGRATION = "integration"  # External system interaction
    USER_INTERFACE = "user_interface"  # UI/UX logic
    INFRASTRUCTURE = "infrastructure"  # System setup/config
    TESTING = "testing"  # Test code
    DEBUGGING = "debugging"  # Debug/logging code
    UTILITY = "utility"  # Helper functions
    EXPERIMENTAL = "experimental"  # Prototype/WIP code


class IntentAlignment(str, Enum):
    """Alignment between code and apparent intent."""

    ALIGNED = "aligned"  # Code matches intent
    MISALIGNED = "misaligned"  # Code doesn't match intent
    PARTIALLY_ALIGNED = "partially_aligned"  # Some mismatch
    UNCLEAR = "unclear"  # Intent not clear


class CodeIntent(BaseModel):
    """Inferred intent for code segment."""

    segment_id: str = Field(
        description="Identifier for code segment"
    )

    file_path: str = Field(
        description="File containing segment"
    )

    line_start: int = Field(
        description="Starting line"
    )

    line_end: int = Field(
        description="Ending line"
    )

    primary_intent: str = Field(
        description="Main purpose in natural language"
    )

    secondary_intents: list[str] = Field(
        default_factory=list,
        description="Additional purposes"
    )

    categories: list[IntentCategory] = Field(
        default_factory=list,
        description="Intent categories"
    )

    business_goals: list[str] = Field(
        default_factory=list,
        description="Business goals served"
    )

    preconditions: list[str] = Field(
        default_factory=list,
        description="Assumed preconditions"
    )

    postconditions: list[str] = Field(
        default_factory=list,
        description="Intended outcomes"
    )

    alignment: IntentAlignment = Field(
        default=IntentAlignment.UNCLEAR,
        description="Code-intent alignment"
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Potential intent issues"
    )

    confidence: float = Field(
        default=0.7,
        description="Confidence in inference"
    )

    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting intent"
    )


class IntentGraph(BaseModel):
    """Graph of intents and their relationships."""

    nodes: dict[str, CodeIntent] = Field(
        default_factory=dict,
        description="Intent nodes by segment_id"
    )

    edges: list[IntentRelationship] = Field(
        default_factory=list,
        description="Relationships between intents"
    )

    hierarchies: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Parent-child intent relationships"
    )

    conflicts: list[IntentConflict] = Field(
        default_factory=list,
        description="Conflicting intents"
    )


class IntentRelationship(BaseModel):
    """Relationship between two intents."""

    source_id: str = Field(
        description="Source intent segment"
    )

    target_id: str = Field(
        description="Target intent segment"
    )

    relationship_type: str = Field(
        description="Type: supports, contradicts, implements, etc."
    )

    strength: float = Field(
        default=0.5,
        description="Relationship strength"
    )


class IntentConflict(BaseModel):
    """Conflict between intents."""

    intent1_id: str = Field(
        description="First conflicting intent"
    )

    intent2_id: str = Field(
        description="Second conflicting intent"
    )

    conflict_type: str = Field(
        description="Type of conflict"
    )

    severity: str = Field(
        default="medium",
        description="Conflict severity: low, medium, high"
    )

    resolution: str | None = Field(
        default=None,
        description="Suggested resolution"
    )


class IntentInferenceResult(ScopeAwareResult[IntentGraph]):
    """Intent inference result with scope awareness."""

    pass


class IntentInferenceContext(BaseModel):
    """Context for intent inference."""

    documentation: str = Field(
        description="Documentation for the codebase"
    )
    tests: str = Field(
        description="Tests for the codebase"
    )
    comments: str = Field(
        description="Comments for the codebase"
    )

