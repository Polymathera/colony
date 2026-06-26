"""Tests for ``ClaimExtractor`` implementations.

The LLM extractor tests use a fake :data:`TypedLLMCallable` that
returns validated pydantic instances (as the real one does, backed by
the deployment's structured-output mechanism). The fake bypasses the
deployment layer so these tests do not need a live LLM cluster.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from polymathera.colony.cluster.errors import LLMCallDeadlineExceeded
from polymathera.colony.knowledge import (
    Chunk,
    CitationSpan,
    ClaimList,
    DeterministicClaimExtractor,
    ExtractedClaim,
    LLMClaimExtractor,
)


pytestmark = pytest.mark.asyncio


def _chunk(text: str, source_uri: str = "src:test") -> Chunk:
    return Chunk(
        text=text,
        token_count=max(1, len(text.split())),
        section_path="1",
        citation=CitationSpan(
            source_uri=source_uri,
            section_path="1",
            char_start=0,
            char_end=len(text),
        ),
        data_type="paper_section",
        source=source_uri,
    )


async def test_deterministic_is_a_claim() -> None:
    extractor = DeterministicClaimExtractor()
    text = "BGE-large-en-v1.5 is a transformer-based embedding model."
    claims = await extractor.extract(_chunk(text))
    assert any(
        c.predicate == "is_a" and "BGE" in c.subject
        for c in claims
    )


async def test_deterministic_requires_claim() -> None:
    extractor = DeterministicClaimExtractor()
    text = "JET requires deuterium-tritium fuel for fusion power."
    claims = await extractor.extract(_chunk(text))
    assert any(
        c.predicate == "requires" and "JET" in c.subject
        for c in claims
    )


async def test_deterministic_no_match_returns_empty() -> None:
    extractor = DeterministicClaimExtractor()
    claims = await extractor.extract(_chunk("Nothing matches here at all."))
    assert claims == ()


# ---------------------------------------------------------------------------
# LLMClaimExtractor — TypedLLMCallable contract
# ---------------------------------------------------------------------------


async def test_llm_extractor_returns_validated_claims() -> None:
    """Happy path: the typed callable returns a :class:`ClaimList`;
    the extractor turns each entry into a grounded :class:`Claim`."""

    captured: dict[str, Any] = {}

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        captured["prompt"] = prompt
        captured["schema"] = schema
        assert schema is ClaimList
        return ClaimList(
            claims=(
                ExtractedClaim(
                    subject="BGE",
                    predicate="is_a",
                    object="embedding model",
                    confidence=0.9,
                ),
                ExtractedClaim(
                    subject="BGE",
                    predicate="uses",
                    object="transformer",
                    confidence=0.8,
                ),
            ),
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(
        _chunk("BGE is an embedding model that uses transformer.")
    )
    assert "src:test" in captured["prompt"]
    assert {c.predicate for c in claims} == {"is_a", "uses"}
    assert all(c.confidence > 0 for c in claims)


async def test_llm_extractor_grounds_claims_in_chunk_citation() -> None:
    """Every emitted ``Claim`` carries the source chunk's citation —
    the LLM contract is purely the typed claim list; grounding is done
    by the extractor."""

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        return ClaimList(
            claims=(
                ExtractedClaim(
                    subject="X",
                    predicate="is_a",
                    object="thing",
                    confidence=0.7,
                ),
            ),
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    chunk = _chunk("X is a thing.", source_uri="src:special")
    claims = await extractor.extract(chunk)
    assert len(claims) == 1
    assert claims[0].citation.source_uri == "src:special"
    assert claims[0].chunk_id == chunk.chunk_id


async def test_llm_extractor_schema_validation_error_returns_empty() -> None:
    """A typed callable that raises :class:`ValidationError` (i.e. the
    deployment returned a schema-invalid payload) yields zero claims
    and logs the typed failure shape. No parsing happens above the
    deployment layer, so this is the canonical "the LLM produced
    something we can't validate" path."""

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        # Force the ValidationError surface — production code would
        # raise this from ``schema.model_validate_json(...)`` on a
        # malformed response. We synthesise it here without touching
        # JSON since the extractor never sees text.
        try:
            ExtractedClaim(subject="", predicate="", object="")
        except ValidationError as exc:
            raise exc
        # unreachable
        return ClaimList()  # pragma: no cover

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("..."))
    assert claims == ()


async def test_llm_extractor_deadline_exhaustion_returns_empty() -> None:
    """The typed callable raises :class:`LLMCallDeadlineExceeded` when
    the deployment-level wall-clock fires. The extractor catches the
    typed exception, returns ``()``, and the ingestion loop continues
    on the next chunk — no infinite retry, no silent stall."""

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        raise LLMCallDeadlineExceeded(
            request_id="claim_extract_test",
            deadline_s=0.05,
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("..."))
    assert claims == ()


async def test_llm_extractor_transport_error_returns_empty() -> None:
    """Generic deployment-level errors are swallowed for graceful
    degradation; the deterministic extractor still runs upstream."""

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        raise RuntimeError("simulated upstream failure")

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("..."))
    assert claims == ()


async def test_llm_extractor_permanent_failure_raises_to_caller() -> None:
    """R7-FIX-D: a permanent-category :class:`LLMInferenceError`
    (BILLING / AUTH) MUST raise to the caller — not warn-and-empty
    per-claim. Run7 had 25,254 per-claim warnings against an open
    breaker because this extractor swallowed the typed failure and
    the ingest pipeline kept calling it for every remaining claim of
    every remaining paper. Raising short-circuits the batch so the
    caller can abort cleanly."""

    from polymathera.colony.cluster.errors import (
        LLMErrorCategory,
        LLMInferenceError,
    )

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        raise LLMInferenceError(
            request_id="claim_extract_test",
            message="credit balance too low",
            category=LLMErrorCategory.BILLING,
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    with pytest.raises(LLMInferenceError) as exc_info:
        await extractor.extract(_chunk("..."))
    assert exc_info.value.category == LLMErrorCategory.BILLING


async def test_llm_extractor_transient_llm_error_still_returns_empty(
) -> None:
    """Counterpart: transient/unknown LLMInferenceError stays at
    warn-and-empty. A single chunk failing on a transient blip
    shouldn't poison the whole pipeline; only PERMANENT categories
    short-circuit."""

    from polymathera.colony.cluster.errors import (
        LLMErrorCategory,
        LLMInferenceError,
    )

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        raise LLMInferenceError(
            request_id="r",
            message="rate limit",
            category=LLMErrorCategory.TRANSIENT,
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("..."))
    assert claims == ()


async def test_llm_extractor_empty_response_is_zero_claims() -> None:
    """A schema-valid empty list is the legitimate "nothing
    high-confidence here" signal — distinct from
    ``ValidationError``-derived emptiness, which is the failure
    signal."""

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        return ClaimList(claims=())

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("Lorem ipsum."))
    assert claims == ()


def test_claim_list_schema_has_descriptive_title() -> None:
    """The schema's title makes the per-extractor identity visible in
    deployment logs and is part of the structured-outputs request
    payload; pin that the title is intentionally set."""

    schema = ClaimList.model_json_schema()
    assert schema.get("title") == "ClaimList"


def test_claim_list_schema_has_additional_properties_false_everywhere() -> None:
    """Anthropic's structured-outputs feature REJECTS any object
    schema that does not explicitly set ``additionalProperties: false``
    (see
    https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations).
    Pin that pydantic's ``extra="forbid"`` is propagating into BOTH the
    top-level ``ClaimList`` AND the nested ``ExtractedClaim`` — without
    this, the API rejects the request at the wire."""

    schema = ClaimList.model_json_schema()
    assert schema.get("additionalProperties") is False, schema
    # Pydantic places nested model schemas under ``$defs`` keyed by
    # the model name.
    defs = schema.get("$defs", {})
    assert "ExtractedClaim" in defs, defs
    assert defs["ExtractedClaim"].get("additionalProperties") is False


def test_claim_list_schema_has_no_unsupported_constraints() -> None:
    """Anthropic's structured-outputs feature does NOT support string
    ``minLength`` / ``maxLength`` nor numeric ``minimum`` / ``maximum``
    / ``multipleOf`` constraints. Pin that ``ExtractedClaim``'s schema
    is value-unconstrained — value-range enforcement lives in the
    pydantic field_validator and the extractor's grounding filter, NOT
    in the schema."""

    schema = ClaimList.model_json_schema()
    extracted = schema.get("$defs", {}).get("ExtractedClaim", {})
    properties = extracted.get("properties", {})
    for field_name in ("subject", "predicate", "object"):
        field_schema = properties.get(field_name, {})
        assert "minLength" not in field_schema, (field_name, field_schema)
        assert "maxLength" not in field_schema, (field_name, field_schema)
    confidence = properties.get("confidence", {})
    for forbidden in ("minimum", "maximum", "exclusiveMinimum",
                      "exclusiveMaximum", "multipleOf"):
        assert forbidden not in confidence, (forbidden, confidence)


def test_extracted_claim_confidence_clamps_to_unit_interval() -> None:
    """The JSON schema cannot declare ``[0, 1]`` bounds under
    Anthropic's structured-outputs limitations. So pydantic's
    field_validator clamps out-of-range values rather than rejecting
    them — one bad value should not poison the whole ``ClaimList``."""

    over = ExtractedClaim(
        subject="X", predicate="p", object="Y", confidence=1.5,
    )
    assert over.confidence == 1.0
    under = ExtractedClaim(
        subject="X", predicate="p", object="Y", confidence=-0.3,
    )
    assert under.confidence == 0.0
    normal = ExtractedClaim(
        subject="X", predicate="p", object="Y", confidence=0.42,
    )
    assert normal.confidence == 0.42


async def test_llm_extractor_drops_claims_with_empty_fields() -> None:
    """Anthropic's structured outputs cannot enforce ``minLength`` on
    strings, so the LLM may emit a valid-shape claim whose
    ``subject``/``predicate``/``object`` is empty. The extractor
    filters such claims at grounding time so one bad item does not
    poison the chunk."""

    async def fake_llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
        return ClaimList(
            claims=(
                ExtractedClaim(
                    subject="", predicate="is_a", object="thing", confidence=0.9,
                ),
                ExtractedClaim(
                    subject="X", predicate="", object="thing", confidence=0.9,
                ),
                ExtractedClaim(
                    subject="X", predicate="is_a", object="   ", confidence=0.9,
                ),
                ExtractedClaim(
                    subject="X", predicate="is_a", object="thing", confidence=0.9,
                ),
            ),
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("X is a thing."))
    assert len(claims) == 1
    assert claims[0].subject == "X"
    assert claims[0].predicate == "is_a"
    # Grounded ``Claim`` aliases ``object`` to ``object_`` in Python
    # (per the model in ``knowledge/models.py``).
    assert claims[0].object_ == "thing"


def test_llm_claim_extractor_schema_classvar_pins_claim_list() -> None:
    """Regression pin: every subclass of LLMClaimExtractor (and
    LLMClaimExtractor itself) MUST declare ``SCHEMA``. The base
    declares ``ClaimList``; any subclass that picks a different schema
    overrides it, but the contract is owned by the class, not the
    instance."""

    assert LLMClaimExtractor.SCHEMA is ClaimList
