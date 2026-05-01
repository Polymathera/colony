"""Typed shapes for the knowledge corpus layer.

Per master §3.2 the knowledge layer carries documents through five
stages: read → chunk → extract → embed → store / index. Each stage
operates on a typed model defined here. The shapes are deliberately
narrow — domain-specific extensions (medical claims, regulatory
clauses, simulation provenance) live as ``extra: dict`` payloads on
the same models, not as parallel hierarchies.

The boundary the layer enforces (per master §3.2):

- The corpus is *durable* and *citation-aware*. Every chunk and claim
  records its provenance (source URI + section path + character span),
  so a downstream answer can show its work.
- The corpus is *open-set* in ``data_type`` and ``source`` so adding
  a new content kind (a vendor datasheet, an MRI metadata sidecar,
  a reqif file) does not require a schema migration.

The models are intentionally compatible with the C4 page-event
metadata: ``Chunk.data_type`` and ``Chunk.source`` mirror the
``PageMetadata`` fields, so a chunk paged into the VCM keeps the
subscription-engine indexes hot.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeFormat(str, Enum):
    """Open-set file-format tags. Source-specific formats add new
    members rather than overload existing ones."""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    JUPYTER = "jupyter"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    SOURCE_CODE = "source_code"
    REQIF = "reqif"
    UNKNOWN = "unknown"


class CorpusTier(str, Enum):
    """Master §3.2 tiered corpus. ``UNTIERED`` covers ad-hoc / per-tool
    content."""

    TIER_1_FOUNDATIONS = "tier_1_foundations"
    """Foundational textbooks; high retrieval weight."""

    TIER_2_STANDARDS = "tier_2_standards"
    """Standards / regulations; structured-requirement-aware."""

    TIER_3_RESEARCH = "tier_3_research"
    """Research literature; recency-weighted."""

    TIER_4_SOFTWARE_DOCS = "tier_4_software_docs"
    """Per-tool software docs and issue trackers; per-tool-scoped."""

    UNTIERED = "untiered"


_TIER_PRIORITY: dict[CorpusTier, int] = {
    CorpusTier.TIER_1_FOUNDATIONS: 4,
    CorpusTier.TIER_2_STANDARDS: 3,
    CorpusTier.TIER_3_RESEARCH: 2,
    CorpusTier.TIER_4_SOFTWARE_DOCS: 1,
    CorpusTier.UNTIERED: 0,
}
"""Total order over tiers. Foundational textbooks > standards >
research > software-docs > untiered. Used by the ``UPGRADE_TIER``
ingestion policy: a re-ingest with a *higher-priority* tier upgrades
existing chunks; a re-ingest with a *lower-or-equal* tier is a no-op
(the existing higher tier is preserved). This is the master §8.4
contract that physics / regulatory / standards content takes
precedence over ad-hoc re-ingestion."""


def tier_priority(tier: CorpusTier) -> int:
    """Return the ranking of ``tier`` (higher number = higher
    priority). See :data:`_TIER_PRIORITY` for the full ordering."""

    return _TIER_PRIORITY[tier]


class IngestionPolicy(str, Enum):
    """How the ``Ingestor`` handles a re-ingest of an already-known
    ``source_uri``.

    The default is ``SKIP_IF_PRESENT`` so that bulk-acquisition
    pipelines (master §6.6) cannot trample a user-seeded corpus. Pick
    ``OVERWRITE`` only when the caller has explicitly decided that the
    new content should replace the old (a paper was retracted, a
    standard was revised, the chunker was upgraded).
    """

    SKIP_IF_PRESENT = "skip_if_present"
    """If any chunk already exists for ``source_uri``, return without
    parsing / chunking / embedding. The default — safest for
    bulk-acquisition pipelines that may re-encounter user-seeded
    sources."""

    UPGRADE_TIER = "upgrade_tier"
    """If chunks exist and the *new* tier outranks the *existing*
    tier (per :func:`tier_priority`), update the persisted tier on
    every existing chunk in place. No re-chunking, no re-embedding.
    If the new tier is lower or equal, behaves as
    ``SKIP_IF_PRESENT``."""

    OVERWRITE = "overwrite"
    """Delete every chunk for the ``source_uri`` (exact match), then
    run the full ingestion pipeline. Use with care — this is the
    only policy that loses retrieval state."""


# ---------------------------------------------------------------------------
# Source -> raw -> parsed -> chunk -> claim -> embedded
# ---------------------------------------------------------------------------


class CitationSpan(BaseModel):
    """A character / token span inside a parsed source.

    Used everywhere a downstream artifact (chunk, claim, retrieval hit)
    needs to point back at where in the source it came from. The
    ``section_path`` is a slash-joined heading hierarchy
    (e.g., ``"3/3.1/3.1.4"`` for "Chapter 3 → 3.1 → 3.1.4") so a
    human reader can find the passage.
    """

    model_config = ConfigDict(frozen=True)

    source_uri: str
    """Stable URI of the parsed source (``"file://...pdf"``,
    ``"arxiv:2410.12345:v2"``, ``"semi:E123:rev2"``, …)."""

    section_path: str = ""
    """Heading hierarchy as a slash-joined string."""

    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)
    page_number: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    """One file / blob loaded from disk or a URL.

    The ``payload`` carries either decoded text (``str``) when the
    format is text-shaped or raw bytes (``bytes``) when the reader
    expects binary input (PDF, DOCX, Parquet). Readers are responsible
    for decoding bytes the right way.
    """

    model_config = ConfigDict(frozen=True)

    source_uri: str
    detected_format: KnowledgeFormat
    payload: str | bytes
    """Decoded text or raw bytes; readers decide."""

    encoding: str | None = None
    """For text payloads, the encoding used to decode (``"utf-8"`` etc.)."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary file-level metadata (mtime, size, mime, …)."""

    @property
    def is_text(self) -> bool:
        return isinstance(self.payload, str)

    @property
    def text(self) -> str:
        if not isinstance(self.payload, str):
            raise TypeError(
                f"RawDocument {self.source_uri} carries bytes, not text. "
                "Call the format reader before accessing .text.",
            )
        return self.payload

    @property
    def bytes_(self) -> bytes:
        if not isinstance(self.payload, (bytes, bytearray)):
            raise TypeError(
                f"RawDocument {self.source_uri} carries text, not bytes.",
            )
        return bytes(self.payload)


class ParsedSection(BaseModel):
    """One section of a parsed document.

    ``ParsedSection``s are produced by readers; chunkers consume them.
    Sections preserve heading hierarchy + citation spans so a
    downstream chunk can be traced back to its source-file character
    range.
    """

    model_config = ConfigDict(frozen=True)

    section_path: str = ""
    heading: str = ""
    text: str
    citation: CitationSpan
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrieval-sized chunk emitted by the chunker.

    Chunks are the smallest *retrieval* unit; they're embedded and
    stored in the ``VectorStore``. Each chunk carries enough metadata
    that the §6.4 retrieval modes can filter on `data_type` /
    `source` / `effective_at` without re-resolving the source.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        default_factory=lambda: f"chunk_{uuid.uuid4().hex[:16]}",
    )
    text: str
    token_count: int = 0
    """Estimated token count (``tiktoken`` cl100k_base by default).
    May be 0 when the producer hasn't called the token manager."""

    section_path: str = ""
    citation: CitationSpan
    data_type: str = "paper_section"
    """Mirrors C4's PageMetadata.data_type — open-set."""

    source: str = ""
    """Mirrors C4's PageMetadata.source — origin URI."""

    tier: CorpusTier = CorpusTier.UNTIERED
    effective_at: datetime | None = None
    """For time-versioned content (a standard clause as of a date,
    a paper section at a revision)."""

    language: str | None = None
    """ISO-639-1 (``"en"``) for prose; programming-language id
    (``"python"``, ``"rust"``) for source-code chunks."""

    extra: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """One typed claim extracted from a chunk.

    The dossiers (master §3.2 ingestion pipeline) treat claims as the
    structured units that populate the knowledge graph. A claim is
    deliberately small: subject + predicate + object + evidence,
    similar in shape to an RDF triple but with a citation tag glued
    on so the framework can show its work.
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(
        default_factory=lambda: f"claim_{uuid.uuid4().hex[:16]}",
    )
    subject: str
    predicate: str
    object_: str = Field(alias="object")
    """``object`` is a reserved name in some downstreams; aliased
    in JSON to ``object``, accessed as ``claim.object_`` in Python."""

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    citation: CitationSpan
    """Where in the source the claim came from."""

    chunk_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    """A chunk + its embedding vector. Used by the vector store."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    chunk: Chunk
    vector: tuple[float, ...]
    embedder: str = "unknown"
    """Identifier of the embedder that produced ``vector`` (e.g.,
    ``"colony.embedding:bge-large-en-v1.5"``); used by the vector
    store to refuse mixing dimensions / models in the same collection."""


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


class RetrievalQuery(BaseModel):
    """One query into the retrieval surface.

    All five §6.4 retrieval modes accept the same query shape; the
    fields each mode honours are documented per-mode. Fields not
    relevant to a mode are ignored, never an error.
    """

    model_config = ConfigDict(frozen=True)

    text: str = ""
    """Free-form natural-language query. Mandatory for embedding-based
    modes (scoped, grounded, budgeted, standards). Optional for graph
    queries that lean on `graph_query` instead."""

    graph_query: str | None = None
    """Cypher-style query string for graph retrieval mode."""

    data_types: tuple[str, ...] = Field(default_factory=tuple)
    """Open-set filter: only chunks with one of these `data_type`s
    are eligible. Empty = any."""

    source_prefix: str | None = None
    """Prefix filter on `Chunk.source` (e.g., ``"semi:"`` for
    SEMI-standards-only)."""

    tiers: tuple[CorpusTier, ...] = Field(default_factory=tuple)
    """Filter: only chunks from these tiers. Empty = any."""

    effective_at: datetime | None = None
    """For standards / regulatory retrieval — return only clauses
    whose ``effective_at`` ≤ this date (and which have not been
    superseded by a later clause)."""

    max_results: int = Field(default=10, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    """Token budget for the budget-aware retrieval mode. None means
    "no budget; honour `max_results` only"."""

    require_citations: bool = False
    """If True (the grounded-mode default), every hit must carry a
    `CitationSpan`. Hits that don't are dropped."""

    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    """One hit in a retrieval result set."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(default=0.0)
    """Mode-defined relevance score in [0, 1] when normalised."""

    rank: int = Field(default=0, ge=0)
    """Position in the result set (0-indexed)."""

    explanation: str = ""
    """Optional, free-form explanation (e.g., the matched substring,
    or the cypher path that produced a graph hit)."""


class RetrievalResult(BaseModel):
    """Result of a retrieval call."""

    model_config = ConfigDict(frozen=True)

    mode: str
    """Mode name (``"scoped"`` / ``"grounded"`` / ``"graph"`` /
    ``"budgeted"`` / ``"standards"``)."""

    hits: tuple[RetrievalHit, ...] = Field(default_factory=tuple)
    total_candidates: int = 0
    """How many candidates were considered before ranking + filtering."""

    used_tokens: int = 0
    """For budget-aware mode: cumulative token count of returned hits."""

    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ingestion record (one per source ingested)
# ---------------------------------------------------------------------------


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    REVIEW_QUEUED = "review_queued"
    COMPLETED = "completed"
    SKIPPED_ALREADY_PRESENT = "skipped_already_present"
    """The ``IngestionPolicy.SKIP_IF_PRESENT`` (default) or
    ``UPGRADE_TIER`` policy short-circuited because chunks already
    exist for the source URI. ``IngestionRecord.chunks_produced`` is
    0 when this is the terminal status; the existing chunks remain
    intact."""

    TIER_UPGRADED = "tier_upgraded"
    """The ``IngestionPolicy.UPGRADE_TIER`` policy bumped the
    persisted tier on every existing chunk for the source URI in
    place. ``IngestionRecord.chunks_produced`` reports how many
    chunks were updated (no new chunks were created)."""

    FAILED = "failed"


class IngestionRecord(BaseModel):
    """One row in the ingestion log: source URI + per-stage outcomes.

    The ``Ingestor`` writes one of these per source it processes;
    callers read it back to inspect failures, sampled-review status,
    and timing.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(
        default_factory=lambda: f"ingest_{uuid.uuid4().hex[:16]}",
    )
    source_uri: str
    status: IngestionStatus = IngestionStatus.PENDING
    detected_format: KnowledgeFormat = KnowledgeFormat.UNKNOWN
    tier: CorpusTier = CorpusTier.UNTIERED
    chunks_produced: int = 0
    claims_extracted: int = 0
    review_required: bool = False
    """Whether the sampled-human-review queue (master §3.2) flagged
    this source for review."""

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    finished_at: datetime | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    document_hash: str = ""
    """SHA-256 hex digest of the raw source payload, when the ingestor
    was run with a payload-bearing document. Used by callers that
    want to detect content changes between two ingestions of the
    same ``source_uri``. Empty string if the ingestor short-circuited
    before reading the payload."""

    policy: IngestionPolicy | None = None
    """The ``IngestionPolicy`` the ingestor ran under. ``None`` for
    pre-existing records produced before policy tracking was added."""


def deterministic_source_uri(*, scheme: str, parts: Sequence[str]) -> str:
    """Build a stable source URI from canonical parts.

    Used by readers when a source is locally-supplied (a file path)
    and there is no canonical URI. Hashes the parts so equivalent
    inputs produce the same URI deterministically.
    """

    body = ":".join(parts)
    if not body:
        body = hashlib.sha256(scheme.encode("utf-8")).hexdigest()[:16]
    return f"{scheme}:{body}"


__all__ = (
    "KnowledgeFormat",
    "CorpusTier",
    "tier_priority",
    "IngestionPolicy",
    "CitationSpan",
    "RawDocument",
    "ParsedSection",
    "Chunk",
    "Claim",
    "EmbeddedChunk",
    "RetrievalQuery",
    "RetrievalHit",
    "RetrievalResult",
    "IngestionStatus",
    "IngestionRecord",
    "deterministic_source_uri",
)
