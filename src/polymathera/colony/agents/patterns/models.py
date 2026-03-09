
import time
import uuid
from typing import Any
from pydantic import BaseModel, Field

from .scope import AnalysisScope



class Critique(BaseModel):
    """Result of critique of action result quality.

    Example:
        ```python
        critique = Critique(
            quality_score=0.7,
            issues=["Incomplete coverage", "Missing error handling"],
            suggestions=["Analyze error paths", "Check edge cases"],
            requires_replanning=True
        )
        ```
    """

    critique_id: str = Field(
        default_factory=lambda: f"critique_{time.time_ns()}",
        description="Unique identifier for this critique"
    )
    valid_conclusions: list[str] = Field(
        default_factory=list,
        description="Conclusions that are logically sound"
    )
    invalid_conclusions: list[str] = Field(
        default_factory=list,
        description="Conclusions that don't follow from premises"
    )
    missing_premises: list[str] = Field(
        default_factory=list,
        description="Premises that should have been considered"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims made without sufficient evidence"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="General problems or concerns"
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality (0.0-1.0)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident in this critique?"
    )
    requires_replanning: bool = Field(
        default=False,
        description="Should we revise plan?"
    )
    requires_revision: bool = Field(
        default=False,
        description="Whether output should be revised"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific improvement suggestions"
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of critique"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the critique"
    )
    created_at: float = Field(
        default_factory=time.time,
        description="When this critique was created"
    )

    # -------------------------------------------------------------------------
    # Memory System Integration
    # -------------------------------------------------------------------------

    def get_blackboard_key(self, scope_id: str) -> str:
        """Generate blackboard key for storing this critique in memory.

        Args:
            scope_id: Memory scope ID (e.g., "agent:abc123:stm")

        Returns:
            Key like "agent:abc123:stm:critique:critique_123456789"
        """
        return f"{scope_id}:critique:{self.critique_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching all critiques in a scope.

        Args:
            scope_id: Memory scope ID

        Returns:
            Pattern like "agent:abc123:stm:critique:*"
        """
        return f"{scope_id}:critique:*"



class Reflection(BaseModel):
    """Reflection on action execution.

    Example:
        ```python
        reflection = Reflection(
            success=True,
            learned={
                "AuthManager_location": "page_042",
                "token_validation": "uses JWT library"
            },
            assumptions_violated=[],
            needs_more_info=False
        )
        ```
    """

    reflection_id: str = Field(
        default_factory=lambda: f"reflection_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this reflection"
    )
    success: bool
    learned: dict[str, Any] = Field(default_factory=dict)  # What did we learn?
    assumptions_violated: list[str] = Field(default_factory=list)  # Surprises?
    needs_more_info: bool = False  # Do we need more context?
    confidence: float = 1.0  # How confident are we? (0.0-1.0)
    reasoning: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    # -------------------------------------------------------------------------
    # Memory System Integration
    # -------------------------------------------------------------------------

    def get_blackboard_key(self, scope_id: str) -> str:
        """Generate blackboard key for storing this reflection in memory.

        Args:
            scope_id: Memory scope ID (e.g., "agent:abc123:ltm:semantic")

        Returns:
            Key like "agent:abc123:ltm:semantic:reflection:reflection_abc12345"
        """
        return f"{scope_id}:reflection:{self.reflection_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching all reflections in a scope.

        Args:
            scope_id: Memory scope ID

        Returns:
            Pattern like "agent:abc123:ltm:semantic:reflection:*"
        """
        return f"{scope_id}:reflection:*"


class Hypothesis(BaseModel):
    """A testable claim that can be validated through evidence gathering.

    Lifecycle:
        1. Formation: Created by HypothesisFormationStrategy with statement + test_queries
        2. Evidence gathering: EvidenceGatheringStrategy interprets test_queries
           according to its mechanism (page queries, code execution, LLM reasoning, etc.)
        3. Evaluation: Evidence assessed by HypothesisEvaluationStrategy or game arbiter
        4. Outcome: status updated to supported/refuted/uncertain

    The test_queries field is critical: these are the queries/specifications that
    the formation strategy generates to validate or refute this hypothesis. Each
    EvidenceGatheringStrategy interprets them according to its mechanism.
    """

    hypothesis_id: str = Field(
        default_factory=lambda: f"hypothesis_{int(time.time() * 1000)}",
        description="Unique hypothesis identifier"
    )

    statement: str = Field(
        description="The hypothesis statement (testable claim)"
    )

    test_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Queries/specifications to test this hypothesis. Each evidence "
            "gathering strategy interprets these according to its mechanism: "
            "QueryBasedEvidence uses them as page-graph search queries; "
            "LLMReasoningEvidence uses them as reasoning prompts; "
            "code execution strategies interpret them as test specifications; "
            "custom strategies interpret per implementation."
        ),
    )

    expected_evidence: str = Field(
        default="",
        description="Description of what evidence would support this hypothesis"
    )

    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting hypothesis"
    )

    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence refuting hypothesis"
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in hypothesis"
    )

    status: str = Field(
        default="untested",
        description="Status: untested, testing, supported, refuted, uncertain"
    )

    created_by: str | None = Field(
        default=None,
        description="Agent that formed hypothesis"
    )

    created_at: float = Field(
        default_factory=time.time,
        description="When hypothesis was formed"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional hypothesis metadata"
    )


class QueryAnswer(BaseModel):
    """Answer to a query with confidence and completeness tracking."""

    answer: str | dict[str, Any] = Field(
        description="The actual answer content"
    )

    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence in answer"
    )

    additional_pages_needed: list[str] = Field(
        default_factory=list,
        description="Descriptions of additional pages/context needed"
    )

    pages_used: list[str] = Field(
        default_factory=list,
        description="Page IDs used to answer"
    )

    reasoning: str | None = Field(
        default=None,
        description="Reasoning for the answer"
    )

    scope: AnalysisScope = Field(
        default_factory=AnalysisScope,
        description="Scope of the answer"
    )

