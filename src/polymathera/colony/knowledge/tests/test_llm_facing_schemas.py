"""Every schema sent through the structured-output contract must obey
Anthropic's documented limitations — pinned HERE so violations fail in
CI, not as production 400s (2026-08-05: ProposalJudgement shipped with
``ge``/``le`` → ``minimum``/``maximum`` → first judge request died).

Adding a new LLM-facing schema? Add it to ``LLM_FACING_SCHEMAS``.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    CODE_CELL_SCHEMA,
)
from polymathera.colony.cluster.anthropic_deployment import (
    find_unsupported_schema_constraints,
)
from polymathera.colony.knowledge.extractors.claims import LLMClaimExtractor
from polymathera.colony.knowledge.vocabulary_revision import ProposalJudgement


LLM_FACING_SCHEMAS = {
    "ClaimList": LLMClaimExtractor.SCHEMA.model_json_schema(),
    "ProposalJudgement": ProposalJudgement.model_json_schema(),
    "CODE_CELL_SCHEMA": CODE_CELL_SCHEMA,
}


@pytest.mark.parametrize("name", sorted(LLM_FACING_SCHEMAS))
def test_llm_facing_schema_has_no_unsupported_constraints(name: str) -> None:
    problems = find_unsupported_schema_constraints(LLM_FACING_SCHEMAS[name])
    assert problems == [], (
        f"{name} would 400 at the Anthropic structured-outputs API: "
        f"{problems}. Move value constraints to field_validators "
        f"(see ExtractedClaim)."
    )


def test_checker_detects_the_2026_08_05_bug_shape() -> None:
    """The checker itself must catch the exact shape that shipped."""

    bad = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    problems = find_unsupported_schema_constraints(bad)
    assert "$.confidence: minimum" in problems
    assert "$.confidence: maximum" in problems
