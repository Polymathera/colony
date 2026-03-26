"""Scope awareness for distributed analysis results.

This module implements the ScopeAwareResult pattern - a universal wrapper that
tracks completeness, missing context, and relationships for any analysis result.

Key Insight: Distributed analysis over large contexts requires managing partial
knowledge. Every result should explicitly track:
- Whether it's complete or needs more context
- What context is missing
- What other pages/shards it relates to
- Confidence level

This enables:
- Automatic query generation from scope gaps
- Incremental refinement as context arrives
- Cross-shard relationship tracking
- Confidence-aware decision making
"""

from __future__ import annotations

import time
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field


# Type variable for generic content
T = TypeVar('T')


class AnalysisScope(BaseModel):
    """Metadata about analysis result completeness and relationships.

    Every analysis result should include scope information that describes:
    - Completeness: Is this analysis complete or are we missing context?
    - Missing Context: What specific context is needed to complete this?
    - Related Shards: What other pages/shards are related to this result?
    - Confidence: How confident are we in this result?
    - Reasoning: Why did we reach these conclusions about scope?

    This is the meta-pattern that was identified in the old code analysis
    implementation and is now elevated to a first-class abstraction.

    Examples:
        Complete analysis:
        ```python
        scope = AnalysisScope(
            is_complete=True,
            related_shards=["page_042", "page_089"],
            confidence=0.95,
            reasoning=["All dependencies resolved", "No external references"]
        )
        ```

        Incomplete analysis needing context:
        ```python
        scope = AnalysisScope(
            is_complete=False,
            missing_context=["AuthManager.validate() implementation"],
            related_shards=["page_042"],
            external_refs=["external_lib.crypto"],
            confidence=0.65,
            reasoning=["Cannot verify auth flow without AuthManager", 
                      "External crypto library not in codebase"]
        )
        ```
    """

    # Completeness
    is_complete: bool = Field(
        default=False,
        description="Whether this analysis is complete or needs more context"
    )

    # Missing context
    missing_context: list[str] = Field(
        default_factory=list,
        description="Specific context needed to complete analysis (function names, files, etc.)"
    )

    # Relationships
    related_shards: list[str] = Field(
        default_factory=list,
        description="Related page IDs or shard IDs that should be analyzed together"
    )

    related_repos: list[str] = Field(
        default_factory=list,
        description="Related repositories (for cross-repo analysis)"
    )

    external_refs: list[str] = Field(
        default_factory=list,
        description="External dependencies or references not in analyzed scope"
    )

    # Confidence and quality
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence level in this result (0.0 = no confidence, 1.0 = certain)"
    )

    quality_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Quality of the analysis (0.0 = poor, 1.0 = excellent)"
    )

    # Reasoning trail
    reasoning: list[str] = Field(
        default_factory=list,
        description="Reasoning steps or justifications for scope assessment"
    )

    # Evidence
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting conclusions (code locations, citations, etc.)"
    )

    # Assumptions
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made in this analysis"
    )

    # Limitations
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this analysis"
    )

    # Extensible metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional scope metadata (domain-specific)"
    )

    # Timestamps
    created_at: float = Field(
        default_factory=time.time,
        description="When scope was created"
    )

    updated_at: float = Field(
        default_factory=time.time,
        description="When scope was last updated"
    )

    def needs_more_context(self) -> bool:
        """Check if more context is needed."""
        return not self.is_complete or len(self.missing_context) > 0

    def has_related_shards(self) -> bool:
        """Check if there are related shards to explore."""
        return len(self.related_shards) > 0 or len(self.external_refs) > 0

    def get_all_related_items(self) -> list[str]:
        """Get all related items (shards, repos, external refs)."""
        return self.related_shards + self.related_repos + self.external_refs

    def merge_with(self, other: AnalysisScope) -> AnalysisScope:
        """Merge two scopes (conservative - union of concerns).

        Args:
            other: Other scope to merge with

        Returns:
            Merged scope
        """
        return AnalysisScope(
            is_complete=self.is_complete and other.is_complete,
            missing_context=list(set(self.missing_context + other.missing_context)),
            related_shards=list(set(self.related_shards + other.related_shards)),
            related_repos=list(set(self.related_repos + other.related_repos)),
            external_refs=list(set(self.external_refs + other.external_refs)),
            confidence=min(self.confidence, other.confidence),  # Conservative
            quality_score=(self.quality_score + other.quality_score) / 2,
            reasoning=self.reasoning + other.reasoning,
            evidence=list(set(self.evidence + other.evidence)),
            assumptions=list(set(self.assumptions + other.assumptions)),
            limitations=list(set(self.limitations + other.limitations)),
            metadata={**self.metadata, **other.metadata},  # Later takes precedence
            created_at=min(self.created_at, other.created_at),
            updated_at=max(self.updated_at, other.updated_at)
        )

    def update_confidence(self, new_confidence: float, reason: str) -> None:
        """Update confidence with reasoning.

        Args:
            new_confidence: New confidence value
            reason: Reason for confidence update
        """
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.reasoning.append(f"Confidence updated to {new_confidence:.2f}: {reason}")
        self.updated_at = time.time()

    def mark_complete(self, reason: str) -> None:
        """Mark analysis as complete.

        Args:
            reason: Reason for marking complete
        """
        self.is_complete = True
        self.reasoning.append(f"Marked complete: {reason}")
        self.updated_at = time.time()

    def add_missing_context(self, context: str | list[str]) -> None:
        """Add missing context item(s).

        Args:
            context: Missing context item or list of items
        """
        if isinstance(context, str):
            context = [context]
        self.missing_context.extend(context)
        self.updated_at = time.time()

    def add_related_shard(self, shard_id: str) -> None:
        """Add related shard.

        Args:
            shard_id: ID of related shard/page
        """
        if shard_id not in self.related_shards:
            self.related_shards.append(shard_id)
            self.updated_at = time.time()


class ScopeAwareResult(BaseModel, Generic[T]):
    """Generic wrapper for analysis results with scope awareness.

    This is the fundamental abstraction for distributed analysis. Every result
    produced by an agent should be wrapped in ScopeAwareResult to track:
    - The actual content (analysis findings, dependencies, etc.)
    - Scope metadata (completeness, relationships, confidence)

    This enables:
    - Automatic detection of incomplete results
    - Query generation from scope gaps
    - Incremental refinement as context arrives
    - Cross-shard relationship tracking
    - Confidence-aware decision making

    Type Parameters:
        T: The type of the actual analysis result content

    Examples:
        Dependency analysis result:
        ```python
        result = ScopeAwareResult(
            content=DependencyGraph(
                nodes=["ModuleA", "ModuleB"],
                edges=[("ModuleA", "ModuleB", "imports")]
            ),
            scope=AnalysisScope(
                is_complete=False,
                missing_context=["ModuleC implementation"],
                related_shards=["page_042"],
                confidence=0.75
            )
        )

        # Check if needs more context
        if result.needs_refinement():
            queries = generate_queries_from_scope(result.scope)
        ```

        Security analysis result:
        ```python
        result = ScopeAwareResult(
            content=SecurityIssues(
                vulnerabilities=[
                    Vulnerability(type="SQL Injection", location="auth.py:42")
                ]
            ),
            scope=AnalysisScope(
                is_complete=True,
                confidence=0.90,
                evidence=["Line 42: unsanitized input in query"],
                assumptions=["Database is PostgreSQL"]
            )
        )
        ```
    """

    # The actual result content
    content: T = Field(
        description="The actual analysis result (dependency graph, findings, metrics, etc.)"
    )

    # Scope metadata
    scope: AnalysisScope = Field(
        default_factory=AnalysisScope,
        description="Metadata about result completeness and relationships"
    )

    # Result metadata
    result_id: str = Field(
        default_factory=lambda: f"result_{int(time.time() * 1000)}",
        description="Unique identifier for this result"
    )

    producer_agent_id: str | None = Field(
        default=None,
        description="ID of agent that produced this result"
    )

    result_type: str | None = Field(
        default=None,
        description="Type of result (for routing and merging)"
    )

    created_at: float = Field(
        default_factory=time.time,
        description="When result was created"
    )

    # Refinement history
    refinement_count: int = Field(
        default=0,
        description="Number of times this result has been refined"
    )

    previous_version_id: str | None = Field(
        default=None,
        description="ID of previous version if this is a refinement"
    )

    # Validation status
    validated: bool = Field(
        default=False,
        description="Whether this result has been validated"
    )

    validation_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Results from validation policies"
    )

    # Convenience methods

    def is_complete(self) -> bool:
        """Check if result is complete."""
        return self.scope.is_complete

    def get_missing_context(self) -> list[str]:
        """Get missing context items."""
        return self.scope.missing_context

    def get_related_pages(self) -> list[str]:
        """Get related page IDs for query routing."""
        return self.scope.related_shards

    def needs_refinement(self) -> bool:
        """Check if result needs refinement.

        Result needs refinement if:
        - Not complete
        - Low confidence
        - Has missing context
        """
        return (
            not self.scope.is_complete
            or self.scope.confidence < 0.8
            or len(self.scope.missing_context) > 0
        )

    def needs_validation(self) -> bool:
        """Check if result needs validation."""
        return not self.validated or self.scope.confidence < 0.7

    def get_confidence(self) -> float:
        """Get confidence level."""
        return self.scope.confidence

    def get_quality(self) -> float:
        """Get quality score."""
        return self.scope.quality_score

    def clone_for_refinement(self) -> ScopeAwareResult[T]:
        """Create a copy for refinement (preserves history).

        Returns:
            New result with incremented refinement count and link to previous
        """
        return ScopeAwareResult(
            content=self.content,
            scope=self.scope.model_copy(deep=True),
            producer_agent_id=self.producer_agent_id,
            result_type=self.result_type,
            refinement_count=self.refinement_count + 1,
            previous_version_id=self.result_id,
            validated=False,  # New refinement needs validation
            validation_results=[]
        )

    def mark_validated(self, validation_result: dict[str, Any]) -> None:
        """Mark result as validated.

        Args:
            validation_result: Validation result details
        """
        self.validated = True
        self.validation_results.append({
            "timestamp": time.time(),
            **validation_result
        })

    def update_scope(self, updates: dict[str, Any]) -> None:
        """Update scope metadata.

        Args:
            updates: Dictionary of scope field updates
        """
        for key, value in updates.items():
            if hasattr(self.scope, key):
                setattr(self.scope, key, value)
        self.scope.updated_at = time.time()

    def to_blackboard_entry(self) -> dict[str, Any]:
        """Convert to blackboard entry format.

        Returns:
            Dictionary suitable for blackboard storage
        """
        return {
            "result_id": self.result_id,
            "content": self.content,
            "scope": self.scope.model_dump(),
            "producer_agent_id": self.producer_agent_id,
            "result_type": self.result_type,
            "created_at": self.created_at,
            "refinement_count": self.refinement_count,
            "previous_version_id": self.previous_version_id,
            "validated": self.validated,
            "validation_results": self.validation_results
        }

    @classmethod
    def from_blackboard_entry(
        cls,
        entry: dict[str, Any],
        content_type: type[T]
    ) -> ScopeAwareResult[T]:
        """Reconstruct from blackboard entry.

        Args:
            entry: Blackboard entry data
            content_type: Type to deserialize content as

        Returns:
            Reconstructed ScopeAwareResult
        """
        # Handle content deserialization based on type
        content_data = entry["content"]
        if isinstance(content_data, dict) and hasattr(content_type, 'model_validate'):
            # Pydantic model
            content = content_type.model_validate(content_data)
        elif isinstance(content_data, dict) and hasattr(content_type, '__init__'):
            # Dataclass or regular class
            content = content_type(**content_data)
        else:
            # Primitive type or already correct type
            content = content_data

        return cls(
            content=content,
            scope=AnalysisScope(**entry["scope"]),
            result_id=entry["result_id"],
            producer_agent_id=entry.get("producer_agent_id"),
            result_type=entry.get("result_type"),
            created_at=entry["created_at"],
            refinement_count=entry.get("refinement_count", 0),
            previous_version_id=entry.get("previous_version_id"),
            validated=entry.get("validated", False),
            validation_results=entry.get("validation_results", [])
        )


def merge_scopes(scopes: list[AnalysisScope]) -> AnalysisScope:
    """Merge multiple analysis scopes (conservative union).

    Args:
        scopes: List of scopes to merge

    Returns:
        Merged scope with union of concerns

    Examples:
        ```python
        scope1 = AnalysisScope(
            is_complete=False,
            missing_context=["FunctionA"],
            confidence=0.7
        )
        scope2 = AnalysisScope(
            is_complete=True,
            missing_context=["FunctionB"],
            confidence=0.9
        )

        merged = merge_scopes([scope1, scope2])
        # merged.is_complete = False (conservative)
        # merged.missing_context = ["FunctionA", "FunctionB"]
        # merged.confidence = 0.7 (minimum)
        ```
    """
    if not scopes:
        return AnalysisScope()

    if len(scopes) == 1:
        return scopes[0].model_copy(deep=True)

    # Start with first scope
    merged = scopes[0].model_copy(deep=True)

    # Merge with remaining scopes
    for scope in scopes[1:]:
        merged = merged.merge_with(scope)

    return merged


def create_complete_scope(
    confidence: float = 1.0,
    reasoning: str | None = None
) -> AnalysisScope:
    """Create a complete scope with high confidence.

    Args:
        confidence: Confidence level
        reasoning: Optional reasoning

    Returns:
        Complete scope
    """
    return AnalysisScope(
        is_complete=True,
        confidence=confidence,
        quality_score=confidence,
        reasoning=[reasoning] if reasoning else []
    )


def create_incomplete_scope(
    missing_context: list[str],
    confidence: float = 0.5,
    reasoning: str | None = None
) -> AnalysisScope:
    """Create an incomplete scope with missing context.

    Args:
        missing_context: List of missing context items
        confidence: Confidence level (lower for incomplete)
        reasoning: Optional reasoning

    Returns:
        Incomplete scope
    """
    return AnalysisScope(
        is_complete=False,
        missing_context=missing_context,
        confidence=confidence,
        quality_score=confidence,
        reasoning=[reasoning] if reasoning else []
    )

