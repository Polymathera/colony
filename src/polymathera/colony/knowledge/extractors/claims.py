"""``ClaimExtractor`` ABC and two implementations.

The deterministic extractor is rule-based and makes no LLM calls; it
exists so the ingestor + tests have a deterministic claim source. The
LLM extractor binds to colony's existing LLM cluster via a small
:data:`TypedLLMCallable` injection: callers pass an async callable that
takes a prompt **and** a pydantic schema and returns an instance of
that schema, decoder-enforced by the underlying deployment
(Anthropic ``output_config.format`` with grammar-constrained sampling /
vLLM ``guided_json`` / OpenRouter ``response_format``). The extractor
returns the validated claims; no
JSON-text parsing happens on the consumer side, so the entire class of
"malformed JSON from LLM" failures is structurally unreachable.

The class hierarchy is intentionally narrow — the framework is
*not* the place to enumerate every domain's NER. Domain-specific
extractors (medical NER, regulatory clause extraction, ECCN tagging)
subclass ``ClaimExtractor`` and ship in CPS / per-domain packages.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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
# LLM extractor (typed-schema; binds to a typed callable)
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    """One claim as returned by the LLM, before grounding into a
    ``Claim`` (which carries chunk + citation metadata the LLM does
    not see). The schema is what is passed to the deployment as the
    decoder constraint.

    Note on field constraints: Anthropic's structured-outputs feature
    (grammar-constrained sampling) does NOT support ``minLength`` /
    ``maxLength`` (string constraints) or ``minimum`` / ``maximum`` /
    ``multipleOf`` (numeric constraints) on the JSON Schema —
    they would cause a 400 at request time. See
    https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
    So the SCHEMA is value-unconstrained; value-quality enforcement
    happens AFTER parse via the field_validator below and the
    extractor's per-claim grounding filter. ``extra="forbid"`` makes
    pydantic emit ``additionalProperties: false`` (REQUIRED by
    Anthropic's structured outputs)."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str = Field(
        description="snake_case predicate, e.g. 'is_a', 'requires', 'measures', 'cites'.",
    )
    object: str
    confidence: float = Field(default=0.5)

    @field_validator("confidence", mode="after")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Clamp out-of-range confidence values into ``[0, 1]``. The
        JSON Schema cannot declare ``ge``/``le`` under Anthropic's
        structured-outputs limitations, so the LLM might legitimately
        emit ``1.5`` or ``-0.2``; we coerce rather than reject so one
        out-of-range value does not poison the whole ``ClaimList``.
        Subject/predicate/object empty-string filtering is done at
        grounding time in :meth:`LLMClaimExtractor.extract` so one
        bad claim doesn't poison the chunk."""

        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v


class ClaimList(BaseModel):
    """The structured response shape for ``LLMClaimExtractor``.

    Provided as the decoder-level schema for Anthropic structured
    outputs / vLLM ``guided_json`` / OpenRouter ``response_format`` on
    every deployment. Future extractors follow the same shape: a
    single ``BaseModel`` whose
    :func:`pydantic.BaseModel.model_json_schema` becomes the LLM
    contract.

    ``extra="forbid"`` makes pydantic emit ``additionalProperties:
    false`` — REQUIRED on every object by Anthropic's structured
    outputs. Without it the API rejects the schema at request time."""

    model_config = ConfigDict(extra="forbid")

    claims: tuple[ExtractedClaim, ...] = Field(default_factory=tuple)


TypedLLMCallable = Callable[[str, type[BaseModel]], Awaitable[BaseModel]]
"""Async callable: ``(prompt, schema) -> validated pydantic instance``.

Implementations build an :class:`InferenceRequest` with
``json_schema=schema.model_json_schema()`` and validate the deployment's
returned ``APIResponse.content`` with ``schema.model_validate_json``.
The contract guarantees a typed return; failure surfaces as a typed
exception (``ValidationError`` for schema-shape failure;
``LLMCallDeadlineExceeded`` for timeout; deployment-level errors for
transport failure) rather than silent malformed JSON. See
:class:`LLMClaimExtractor.extract` for the canonical consumer."""


@dataclass(frozen=True)
class ExtractionPrompt:
    """Template used by ``LLMClaimExtractor`` to ask the LLM for typed
    claims. Override ``system`` / ``user_template`` to tune. The
    schema is enforced by the deployment, so the prompt no longer
    needs to instruct the LLM about JSON shape — it instructs about
    *what* to extract."""

    system: str = (
        "You extract typed claims from a chunk of text. A claim is a "
        "(subject, predicate, object) triple grounded in the input. "
        "Predicates SHOULD use snake_case (e.g., 'is_a', 'requires', "
        "'measures', 'cites'). Confidence is a float in [0, 1]. "
        "Return an empty list if no high-confidence claims are present."
    )
    user_template: str = (
        "Source URI: {source_uri}\n"
        "Section: {section_path}\n"
        "---\n"
        "{text}"
    )


class LLMClaimExtractor(ClaimExtractor):
    """Typed-schema LLM-based extractor.

    ``llm`` is a :data:`TypedLLMCallable` that takes a prompt + a
    pydantic schema and returns a validated instance of that schema.
    The binding to :class:`polymathera.colony.cluster.cluster.LLMCluster`
    is the caller's responsibility — the :class:`Ingestor` constructor
    accepts an ``LLMCluster`` handle and wires the callable. Decoder-
    level enforcement (Anthropic ``output_config.format`` with
    grammar-constrained sampling, vLLM ``guided_json``, OpenRouter
    ``response_format``) guarantees the returned object validates
    against the schema's SHAPE, so this extractor has no JSON-parsing
    or fence-stripping code. Note: value-range constraints (non-empty
    strings, ``[0,1]`` confidence) live in
    :class:`ExtractedClaim`'s ``field_validator`` and in this
    extractor's per-claim grounding filter, NOT in the JSON schema —
    Anthropic structured outputs do not support ``minLength`` /
    ``minimum`` / ``maximum``.
    """

    SCHEMA: ClassVar[type[BaseModel]] = ClaimList
    """The structured-output schema. Future extractors override with
    their own ``BaseModel`` subclass; the deployment honors it
    natively per :attr:`SCHEMA`."""

    def __init__(
        self,
        llm: TypedLLMCallable,
        *,
        prompt: ExtractionPrompt | None = None,
    ) -> None:
        self._llm = llm
        self._prompt = prompt or ExtractionPrompt()

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
            payload = await self._llm(prompt, type(self).SCHEMA)
        except ValidationError as exc:
            logger.warning(
                "LLMClaimExtractor: schema-invalid response for %s: %s",
                chunk.citation.source_uri, exc,
            )
            return ()
        except Exception as exc:  # noqa: BLE001 — typed failure paths above; broad guard covers deployment-level surprises
            logger.warning(
                "LLMClaimExtractor: LLM call failed for %s: %s (%s)",
                chunk.citation.source_uri,
                type(exc).__name__,
                exc,
            )
            return ()

        assert isinstance(payload, ClaimList), (
            f"TypedLLMCallable returned {type(payload).__name__}, expected ClaimList"
        )
        # Per-claim filtering for non-empty subject/predicate/object.
        # Anthropic's structured outputs cannot enforce ``minLength``
        # at the decoder level (see ``ExtractedClaim`` docstring) so
        # the LLM may emit an empty string for a field. Filter
        # per-claim rather than reject the whole ClaimList so one
        # malformed entry does not poison a chunk.
        grounded: list[Claim] = []
        for item in payload.claims:
            subject = item.subject.strip()
            predicate = item.predicate.strip()
            obj = item.object.strip()
            if not (subject and predicate and obj):
                continue
            grounded.append(Claim(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=item.confidence,
                chunk_id=chunk.chunk_id,
                citation=chunk.citation,
            ))
        return tuple(grounded)


__all__ = (
    "ClaimExtractor",
    "DeterministicClaimExtractor",
    "ExtractedClaim",
    "ClaimList",
    "ExtractionPrompt",
    "TypedLLMCallable",
    "LLMClaimExtractor",
)
