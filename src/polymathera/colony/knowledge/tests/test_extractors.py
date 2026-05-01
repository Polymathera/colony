"""Tests for ``ClaimExtractor`` implementations."""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge import (
    Chunk,
    CitationSpan,
    DeterministicClaimExtractor,
    ExtractionPrompt,
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


async def test_llm_extractor_parses_typed_json() -> None:
    async def fake_llm(prompt: str) -> str:
        # Verify the prompt carries the chunk source_uri.
        assert "src:test" in prompt
        return (
            '[{"subject": "BGE", "predicate": "is_a", '
            '"object": "embedding model", "confidence": 0.9},'
            '{"subject": "BGE", "predicate": "uses", '
            '"object": "transformer", "confidence": 0.8}]'
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(
        _chunk("BGE is an embedding model that uses transformer.")
    )
    assert {c.predicate for c in claims} == {"is_a", "uses"}
    assert all(c.confidence > 0 for c in claims)


async def test_llm_extractor_handles_codefenced_response() -> None:
    async def fake_llm(prompt: str) -> str:
        return (
            "```json\n"
            '[{"subject": "X", "predicate": "is_a", '
            '"object": "thing", "confidence": 0.7}]\n'
            "```"
        )

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("X is a thing."))
    assert len(claims) == 1
    assert claims[0].subject == "X"


async def test_llm_extractor_malformed_returns_empty() -> None:
    async def fake_llm(prompt: str) -> str:
        return "this is not json at all"

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("..."))
    assert claims == ()


async def test_llm_extractor_drops_partial_claims() -> None:
    async def fake_llm(prompt: str) -> str:
        return '[{"subject": "X"}, {"subject": "Y", "predicate": "is_a", "object": "Z"}]'

    extractor = LLMClaimExtractor(llm=fake_llm)
    claims = await extractor.extract(_chunk("..."))
    assert len(claims) == 1
    assert claims[0].subject == "Y"


async def test_llm_extractor_timeout_returns_empty() -> None:
    import asyncio

    async def slow(prompt: str) -> str:
        await asyncio.sleep(2.0)
        return "[]"

    extractor = LLMClaimExtractor(llm=slow, timeout_s=0.05)
    claims = await extractor.extract(_chunk("..."))
    assert claims == ()
