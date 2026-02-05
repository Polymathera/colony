"""Program Slicing - Extract minimal code subset affecting a target.

Qualitative LLM-based program slicing that approximates traditional
static slicing techniques using natural language reasoning about
data and control dependencies.

Traditional Approach:
- Build PDG (Program Dependence Graph) 
- Compute backward/forward slices
- Track data and control dependencies

LLM Approach:
- Reason about variable flows qualitatively
- Identify "likely influences" on slicing criterion
- Generate approximate slices with confidence scores
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from ....agents.patterns import ScopeAwareResult


logger = logging.getLogger(__name__)


class SliceType(str, Enum):
    """Types of program slices."""

    BACKWARD = "backward"  # Statements affecting criterion
    FORWARD = "forward"  # Statements affected by criterion
    CHOPPING = "chopping"  # Between source and sink
    DYNAMIC = "dynamic"  # Based on execution trace
    CONDITIONED = "conditioned"  # With path conditions


class SliceCriterion(BaseModel):
    """Slicing criterion specification."""

    file_path: str = Field(
        description="File containing the criterion"
    )

    line_number: int = Field(
        description="Line number of interest"
    )

    variable: str | None = Field(
        default=None,
        description="Variable of interest at the line"
    )

    expression: str | None = Field(
        default=None,
        description="Expression to slice on"
    )

    slice_type: SliceType = Field(
        default=SliceType.BACKWARD,
        description="Type of slice to compute"
    )


class DependencyEdge(BaseModel):
    """Edge in dependency graph."""

    from_line: int = Field(
        description="Source line"
    )

    to_line: int = Field(
        description="Target line"
    )

    dep_type: str = Field(
        description="Dependency type: data, control, or call"
    )

    variable: str | None = Field(
        default=None,
        description="Variable involved in data dependency"
    )

    condition: str | None = Field(
        default=None,
        description="Control flow condition"
    )

    confidence: float = Field(
        default=1.0,
        description="Confidence in this dependency"
    )


class ProgramSlice(BaseModel):
    """A program slice result."""

    criterion: SliceCriterion = Field(
        description="The slicing criterion"
    )

    included_lines: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Lines included in slice per file"
    )

    excluded_lines: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Lines excluded from slice per file"
    )

    dependencies: list[DependencyEdge] = Field(
        default_factory=list,
        description="Dependency edges in the slice"
    )

    entry_points: list[str] = Field(
        default_factory=list,
        description="Entry points into the slice"
    )

    exit_points: list[str] = Field(
        default_factory=list,
        description="Exit points from the slice"
    )

    interprocedural: bool = Field(
        default=False,
        description="Whether slice crosses function boundaries"
    )

    reasoning: str = Field(
        description="Explanation of slicing decisions"
    )


class SlicingResult(ScopeAwareResult[ProgramSlice]):
    """Slicing result with scope awareness."""

    pass

