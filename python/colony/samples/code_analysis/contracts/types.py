"""Contract Inference - Infer function contracts and invariants.

Qualitative LLM-based contract inference that approximates formal
specification inference using natural language reasoning about
preconditions, postconditions, and invariants.

Traditional Approach:
- Symbolic execution to derive constraints
- Daikon-style dynamic invariant detection  
- Houdini/ICE learning from examples
- Interpolation-based synthesis

LLM Approach:
- Reason about function intent and requirements
- Identify implicit assumptions and guarantees
- Generate likely invariants from patterns
- Express contracts in natural language or formal spec
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ....agents.patterns import ScopeAwareResult



logger = logging.getLogger(__name__)


class ContractType(str, Enum):
    """Types of contracts."""

    PRECONDITION = "precondition"  # Required before call
    POSTCONDITION = "postcondition"  # Guaranteed after call
    INVARIANT = "invariant"  # Always true
    ASSERTION = "assertion"  # Must hold at point
    ASSUMPTION = "assumption"  # Assumed to hold


class FormalismLevel(str, Enum):
    """Level of formalism for contracts."""

    NATURAL = "natural"  # Natural language
    SEMI_FORMAL = "semi_formal"  # Structured but not formal
    FORMAL = "formal"  # Formal logic (e.g., Z3, Dafny)
    CODE = "code"  # Executable assertions


class Contract(BaseModel):
    """A contract specification."""

    contract_type: ContractType = Field(
        description="Type of contract"
    )

    description: str = Field(
        description="Natural language description"
    )

    formal_spec: str | None = Field(
        default=None,
        description="Formal specification if available"
    )

    variables: list[str] = Field(
        default_factory=list,
        description="Variables involved"
    )

    confidence: float = Field(
        default=0.8,
        description="Confidence in this contract"
    )

    source: str = Field(
        default="inferred",
        description="How contract was derived: inferred, explicit, learned"
    )

    counterexamples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Known counterexamples"
    )


class FunctionContract(BaseModel):
    """Complete contract for a function."""

    function_name: str = Field(
        description="Function name"
    )

    file_path: str = Field(
        description="File containing function"
    )

    line_number: int = Field(
        description="Function definition line"
    )

    preconditions: list[Contract] = Field(
        default_factory=list,
        description="Preconditions"
    )

    postconditions: list[Contract] = Field(
        default_factory=list,
        description="Postconditions"
    )

    invariants: list[Contract] = Field(
        default_factory=list,
        description="Loop/class invariants"
    )

    modifies: list[str] = Field(
        default_factory=list,
        description="Variables/state modified"
    )

    pure: bool = Field(
        default=False,
        description="Whether function is pure (no side effects)"
    )

    termination: str | None = Field(
        default=None,
        description="Termination argument"
    )

    complexity: str | None = Field(
        default=None,
        description="Complexity bound (e.g., O(n))"
    )

    formalism: FormalismLevel = Field(
        default=FormalismLevel.NATURAL,
        description="Level of formalism"
    )


class ContractInferenceResult(ScopeAwareResult[list[FunctionContract]]):
    """Contract inference result with scope awareness."""

    pass

