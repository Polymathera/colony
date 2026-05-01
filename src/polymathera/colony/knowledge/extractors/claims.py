"""``ClaimExtractor`` ABC and two implementations.

The deterministic extractor is rule-based and makes no LLM calls; it
exists so the ingestor + tests have a deterministic claim source. The
LLM extractor binds to colony's existing LLM cluster via a small
``LLMCallable`` injection: callers pass a coroutine that, given a
prompt, returns a JSON string of typed claims. The extractor parses,
validates against the ``Claim`` schema, and emits.

The class hierarchy is intentionally narrow — the framework is
*not* the place to enumerate every domain's NER. Domain-specific
extractors (medical NER, regulatory clause extraction, ECCN tagging)
subclass ``ClaimExtractor`` and ship in CPS / per-domain packages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ..models import Chunk, CitationSpan, Claim


logger = logging.getLogger(__name__)


class ClaimExtractor(ABC):
    """Take a ``Chunk`` and emit ``Claim``s grounded in it."""

    @abstractmethod
    async def extract(self, chunk: Chunk) -> Sequence[Claim]:
        ...


# ---------------------------------------------------------------------------
# Deterministic extractor (rule-based, no LLM)
# ---------------------------------------------------------------------------


_DEFINITION_RE = re.compile(
    r"\b(?P<subject>[A-Z][A-Za-z0-9._-]+(?:\s+[A-Za-z0-9._-]+){0,5})"
    r"\s+is\s+(?:a|an|the)\s+(?P<obj>[a-z][a-z0-9 ._-]{3,80})",
)
"""Matches sentences like 'BGE-large-en-v1.5 is a transformer-based
embedding model'. Conservative, English-only; emits ``is_a`` claims.
Subjects are 1–6 tokens that may include digits, dots, hyphens, and
underscores (so version strings like ``v1.5`` parse)."""

_REQUIRES_RE = re.compile(
    r"\b(?P<subject>[A-Z][A-Za-z0-9._-]+(?:\s+[A-Za-z0-9._-]+){0,5})"
    r"\s+requires\s+(?P<obj>[a-z][a-z0-9 ._-]{3,80})",
)


class DeterministicClaimExtractor(ClaimExtractor):
    """Rule-based extractor — produces ``is_a`` and ``requires``
    claims. No LLM, no model downloads, no network calls. Used in
    tests and as a smoke fallback when the LLM is unavailable.
    """

    async def extract(self, chunk: Chunk) -> Sequence[Claim]:
        out: list[Claim] = []
        text = chunk.text
        out.extend(self._matches(text, _DEFINITION_RE, "is_a", chunk))
        out.extend(self._matches(text, _REQUIRES_RE, "requires", chunk))
        return tuple(out)

    @staticmethod
    def _matches(
        text: str, pattern: re.Pattern[str], predicate: str, chunk: Chunk,
    ) -> list[Claim]:
        results: list[Claim] = []
        for m in pattern.finditer(text):
            char_start = chunk.citation.char_start + m.start()
            char_end = chunk.citation.char_start + m.end()
            results.append(
                Claim(
                    subject=m.group("subject").strip(),
                    predicate=predicate,
                    object=m.group("obj").strip(),
                    confidence=0.5,
                    chunk_id=chunk.chunk_id,
                    citation=CitationSpan(
                        source_uri=chunk.citation.source_uri,
                        section_path=chunk.citation.section_path,
                        char_start=char_start,
                        char_end=char_end,
                        page_number=chunk.citation.page_number,
                    ),
                )
            )
        return results


# ---------------------------------------------------------------------------
# LLM extractor (typed-schema; binds to a callable)
# ---------------------------------------------------------------------------


LLMCallable = Callable[[str], Awaitable[str]]
"""Async callable: prompt -> raw JSON-shaped string of typed claims.

The framework doesn't bind directly to ``LLMCluster``; the caller
constructs a callable that does so (``async def llm(prompt): ...``).
This keeps the extractor unit-testable with a fake."""


@dataclass(frozen=True)
class ExtractionPrompt:
    """Template used by ``LLMClaimExtractor`` to ask the LLM for typed
    claims. Override ``system`` / ``user_template`` to tune."""

    system: str = (
        "You extract typed claims from a chunk of text. A claim is a "
        "(subject, predicate, object) triple grounded in the input. "
        "Predicates SHOULD use snake_case (e.g., 'is_a', 'requires', "
        "'measures', 'cites'). You produce only valid JSON."
    )
    user_template: str = (
        "Source URI: {source_uri}\n"
        "Section: {section_path}\n"
        "---\n"
        "{text}\n"
        "---\n"
        "Return a JSON array of claim objects, each with: "
        '{{"subject":..., "predicate":..., "object":..., "confidence":0..1}}. '
        "Output an empty array if no high-confidence claims are present. "
        "Return ONLY the JSON array, no prose."
    )


class LLMClaimExtractor(ClaimExtractor):
    """Typed-schema LLM-based extractor.

    ``llm`` is an async callable that takes a *single string prompt*
    (system + user concatenated with two newlines) and returns the
    LLM's raw response. The extractor parses the response as JSON
    and validates against the ``Claim`` shape; malformed responses
    yield zero claims (logged at WARN).

    The binding to ``polymathera.colony.cluster.cluster.LLMCluster``
    is the caller's responsibility — the ``Ingestor`` constructor
    accepts an LLMCluster handle and wires the callable.
    """

    def __init__(
        self,
        llm: LLMCallable,
        *,
        prompt: ExtractionPrompt | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._llm = llm
        self._prompt = prompt or ExtractionPrompt()
        self._timeout = timeout_s

    async def extract(self, chunk: Chunk) -> Sequence[Claim]:
        prompt = (
            f"{self._prompt.system}\n\n"
            + self._prompt.user_template.format(
                source_uri=chunk.citation.source_uri,
                section_path=chunk.citation.section_path,
                text=chunk.text,
            )
        )
        try:
            raw = await asyncio.wait_for(self._llm(prompt), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "LLMClaimExtractor: timeout extracting claims for %s",
                chunk.citation.source_uri,
            )
            return ()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLMClaimExtractor: LLM call failed for %s: %s",
                chunk.citation.source_uri, exc,
            )
            return ()
        return self._parse(raw, chunk)

    @staticmethod
    def _parse(raw: str, chunk: Chunk) -> Sequence[Claim]:
        # Tolerate code-fenced JSON.
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n", "", text)
            text = text.removesuffix("```").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "LLMClaimExtractor: malformed JSON from LLM for %s",
                chunk.citation.source_uri,
            )
            return ()
        if not isinstance(payload, list):
            return ()
        out: list[Claim] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            object_ = str(item.get("object", "")).strip()
            if not subject or not predicate or not object_:
                continue
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            out.append(
                Claim(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    confidence=confidence,
                    chunk_id=chunk.chunk_id,
                    citation=chunk.citation,
                )
            )
        return tuple(out)


__all__ = (
    "ClaimExtractor",
    "DeterministicClaimExtractor",
    "ExtractionPrompt",
    "LLMCallable",
    "LLMClaimExtractor",
)
