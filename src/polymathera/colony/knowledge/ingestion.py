"""``Ingestor`` — orchestrates the master §6.3 pipeline end to end.

The pipeline (master §6.3):

    detect format → read → chunk → extract claims → embed → vector-store
    insert → graph-store insert → human-review queue (sampled)

Each stage is independently injectable: the constructor takes one
or more readers, a chunker pair (prose + code), one or more
extractors, an embedder, a vector store, a graph store, and an
optional review-queue callback. This keeps the ingestor unit-testable
without standing up GROBID / Kùzu / Qdrant.

The ingestor returns an ``IngestionRecord`` per source, recording
the per-stage outcomes. It does NOT re-page into the VCM — that's a
``ContextPageSource`` concern (master §3.1) and lives in C4 / C6
wiring.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..agents.blueprint import Blueprint, blueprint
from .chunking import ChunkerConfig, CodeChunker, ProseChunker
from .embedder import Embedder
from .extractors import ClaimExtractor
from .formats import detect_format
from .models import (
    Chunk,
    Claim,
    CorpusTier,
    EmbeddedChunk,
    IngestionPolicy,
    IngestionRecord,
    IngestionStatus,
    KnowledgeFormat,
    ParsedSection,
    RawDocument,
    deterministic_source_uri,
    tier_priority,
)
from .readers import FormatReader, FormatReaderError, ReaderRegistry, default_registry
from .stores import GraphStore, VectorStore


logger = logging.getLogger(__name__)


HumanReviewQueueCallback = Callable[[IngestionRecord, Sequence[Chunk]], Awaitable[None]]
"""Optional callback the ingestor invokes when the sampling policy
flags a record for human review. The callback typically writes to a
blackboard scope; for tests, a list-appending coroutine works."""


@blueprint
class Ingestor:
    """Per-source ingestion orchestrator.

    ``@blueprint`` adds a pickleable ``.bind()``. The heavy
    dependencies (``embedder``, ``vector_store``, ``graph_store``)
    accept either a real instance or a :class:`Blueprint` — the
    constructor resolves the latter via ``local_instance()`` so the
    same shape works in tests (real instances) and across the Ray
    boundary (blueprint chain). Same pattern as
    :class:`ConsciousnessStream`.
    """

    DEFAULT_REVIEW_SAMPLE_RATE = 0.05

    def __init__(
        self,
        *,
        readers: ReaderRegistry | None = None,
        prose_chunker: ProseChunker | None = None,
        code_chunker: CodeChunker | None = None,
        extractors: Sequence[ClaimExtractor] = (),
        embedder: Embedder | Blueprint,
        vector_store: VectorStore | Blueprint,
        graph_store: GraphStore | Blueprint | None = None,
        review_queue: HumanReviewQueueCallback | None = None,
        review_sample_rate: float = DEFAULT_REVIEW_SAMPLE_RATE,
        rng: random.Random | None = None,
    ) -> None:
        self._readers = readers or default_registry()
        self._prose = prose_chunker or ProseChunker()
        self._code = code_chunker or CodeChunker()
        self._extractors: tuple[ClaimExtractor, ...] = tuple(extractors)
        self._embedder = (
            embedder.local_instance() if isinstance(embedder, Blueprint) else embedder
        )
        self._vector_store = (
            vector_store.local_instance()
            if isinstance(vector_store, Blueprint)
            else vector_store
        )
        self._graph_store = (
            graph_store.local_instance()
            if isinstance(graph_store, Blueprint)
            else graph_store
        )
        self._review = review_queue
        self._review_rate = max(0.0, min(1.0, review_sample_rate))
        self._rng = rng or random.Random()

    # ---- Public API ----------------------------------------------------

    async def ingest_file(
        self,
        path: str | Path,
        *,
        tier: CorpusTier = CorpusTier.UNTIERED,
        data_type_override: str | None = None,
        source_uri: str | None = None,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> IngestionRecord:
        path_obj = Path(path)
        if not path_obj.is_file():
            return _fail_record(
                source_uri=source_uri or path_obj.as_uri(),
                error=f"File not found: {path_obj}",
            )
        payload: bytes = path_obj.read_bytes()
        fmt = detect_format(path=path_obj, payload=payload)
        is_text = fmt in _TEXT_FORMATS
        decoded: str | bytes
        encoding: str | None = None
        if is_text:
            try:
                decoded = payload.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                decoded = payload.decode("latin-1", errors="replace")
                encoding = "latin-1"
        else:
            decoded = payload

        document = RawDocument(
            source_uri=source_uri or path_obj.as_uri(),
            detected_format=fmt,
            payload=decoded,
            encoding=encoding,
            metadata={"path": str(path_obj), "size_bytes": len(payload)},
        )
        return await self.ingest_document(
            document, tier=tier, data_type_override=data_type_override,
            policy=policy,
        )

    async def ingest_text(
        self,
        text: str,
        *,
        source_uri: str | None = None,
        fmt: KnowledgeFormat = KnowledgeFormat.PLAIN_TEXT,
        tier: CorpusTier = CorpusTier.UNTIERED,
        data_type_override: str | None = None,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> IngestionRecord:
        document = RawDocument(
            source_uri=source_uri
            or deterministic_source_uri(scheme="text", parts=(text[:64],)),
            detected_format=fmt,
            payload=text,
            encoding="utf-8",
        )
        return await self.ingest_document(
            document, tier=tier, data_type_override=data_type_override,
            policy=policy,
        )

    async def ingest_document(
        self,
        document: RawDocument,
        *,
        tier: CorpusTier = CorpusTier.UNTIERED,
        data_type_override: str | None = None,
        policy: IngestionPolicy = IngestionPolicy.SKIP_IF_PRESENT,
    ) -> IngestionRecord:
        document_hash = _document_hash(document.payload)
        record = IngestionRecord(
            source_uri=document.source_uri,
            detected_format=document.detected_format,
            tier=tier,
            status=IngestionStatus.PARSING,
            document_hash=document_hash,
            policy=policy,
        )

        # ---- Idempotency pre-flight (master §6.6 / Q-S0a contract) ----
        # Run before parsing / chunking / embedding so SKIP_IF_PRESENT
        # is genuinely cheap when the source is already known.
        existing = await self._vector_store.list_chunks_for_source(
            document.source_uri,
        )
        if existing:
            if policy is IngestionPolicy.SKIP_IF_PRESENT:
                return _finish_record(
                    record,
                    status=IngestionStatus.SKIPPED_ALREADY_PRESENT,
                    chunks_produced=0,
                )
            if policy is IngestionPolicy.UPGRADE_TIER:
                existing_tier = existing[0].chunk.tier
                if tier_priority(tier) > tier_priority(existing_tier):
                    updated = await self._vector_store.update_tier_for_source(
                        document.source_uri, tier,
                    )
                    return _finish_record(
                        record,
                        status=IngestionStatus.TIER_UPGRADED,
                        chunks_produced=updated,
                    )
                # New tier does not outrank existing — preserve.
                return _finish_record(
                    record,
                    status=IngestionStatus.SKIPPED_ALREADY_PRESENT,
                    chunks_produced=0,
                )
            # OVERWRITE: drop all existing chunks for this source URI
            # before re-running the pipeline. Use exact-match delete via
            # chunk-id list rather than prefix-match to avoid collateral
            # damage to ``source_uri.bak`` siblings.
            chunk_ids = [item.chunk.chunk_id for item in existing]
            if chunk_ids:
                await self._vector_store.delete_by_chunk_ids(chunk_ids)

        reader = self._readers.reader_for(document.detected_format)
        if reader is None:
            return _finish_record(
                record,
                status=IngestionStatus.FAILED,
                error=(
                    f"No reader registered for format "
                    f"{document.detected_format.value}."
                ),
            )

        try:
            sections = await reader.read_async(document)
        except FormatReaderError as exc:
            return _finish_record(
                record, status=IngestionStatus.FAILED, error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Ingestor: reader %s failed on %s",
                type(reader).__name__, document.source_uri,
            )
            return _finish_record(
                record, status=IngestionStatus.FAILED,
                error=f"reader failed: {exc}",
            )

        record = record.model_copy(update={"status": IngestionStatus.CHUNKING})
        try:
            chunks = await asyncio.to_thread(
                self._chunk_sections,
                sections,
                document.detected_format,
                tier,
                data_type_override,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Ingestor: chunking failed for %s", document.source_uri,
            )
            return _finish_record(
                record, status=IngestionStatus.FAILED,
                error=f"chunking failed: {exc}",
            )
        if not chunks:
            return _finish_record(
                record, status=IngestionStatus.COMPLETED,
                chunks_produced=0,
            )

        record = record.model_copy(
            update={
                "status": IngestionStatus.EXTRACTING,
                "chunks_produced": len(chunks),
            },
        )

        claims = await self._run_extractors(chunks)

        record = record.model_copy(
            update={
                "status": IngestionStatus.EMBEDDING,
                "claims_extracted": len(claims),
            },
        )

        try:
            vectors = await self._embedder.embed([c.text for c in chunks])
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Ingestor: embedding failed for %s", document.source_uri,
            )
            return _finish_record(
                record, status=IngestionStatus.FAILED,
                error=f"embedding failed: {exc}",
            )

        if len(vectors) != len(chunks):
            return _finish_record(
                record, status=IngestionStatus.FAILED,
                error=(
                    f"embedder returned {len(vectors)} vectors for "
                    f"{len(chunks)} chunks"
                ),
            )

        embedded = tuple(
            EmbeddedChunk(
                chunk=chunk,
                vector=tuple(float(v) for v in vec),
                embedder=self._embedder.embedder_id,
            )
            for chunk, vec in zip(chunks, vectors)
        )

        record = record.model_copy(update={"status": IngestionStatus.INDEXING})

        try:
            await self._vector_store.upsert(embedded)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Ingestor: vector_store.upsert failed for %s",
                document.source_uri,
            )
            return _finish_record(
                record, status=IngestionStatus.FAILED,
                error=f"vector store insert failed: {exc}",
            )

        if self._graph_store is not None:
            for claim in claims:
                try:
                    await self._graph_store.add_claim(claim)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Ingestor: graph_store.add_claim failed for %s: %s",
                        document.source_uri, exc,
                    )

        review_required = self._sample_for_review()
        if review_required and self._review is not None:
            review_record = record.model_copy(
                update={
                    "status": IngestionStatus.REVIEW_QUEUED,
                    "review_required": True,
                },
            )
            try:
                await self._review(review_record, chunks)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Ingestor: review_queue callback failed for %s",
                    document.source_uri,
                )

        return _finish_record(
            record,
            status=IngestionStatus.COMPLETED,
            chunks_produced=len(chunks),
            claims_extracted=len(claims),
            review_required=review_required,
        )

    # ---- Internals -----------------------------------------------------

    def _chunk_sections(
        self,
        sections: Sequence[ParsedSection],
        fmt: KnowledgeFormat,
        tier: CorpusTier,
        data_type_override: str | None,
    ) -> Sequence[Chunk]:
        out: list[Chunk] = []
        for section in sections:
            if not section.text.strip():
                continue
            data_type = data_type_override or _default_data_type(fmt)
            if fmt is KnowledgeFormat.SOURCE_CODE:
                language = str(section.extra.get("language", "text"))
                out.extend(
                    self._code.chunk(
                        section,
                        language=language,
                        data_type=data_type,
                        tier=tier,
                    )
                )
            else:
                out.extend(
                    self._prose.chunk(
                        section,
                        data_type=data_type,
                        tier=tier,
                    )
                )
        return tuple(out)

    async def _run_extractors(
        self, chunks: Sequence[Chunk],
    ) -> Sequence[Claim]:
        if not self._extractors:
            return ()
        results: list[Claim] = []
        # Run extractors per chunk in series — parallelism is the
        # caller's choice (an LLM-bound extractor sets its own
        # concurrency by batching internally).
        for chunk in chunks:
            for extractor in self._extractors:
                try:
                    results.extend(await extractor.extract(chunk))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Ingestor: extractor %s failed on chunk %s: %s",
                        type(extractor).__name__, chunk.chunk_id, exc,
                    )
        return tuple(results)

    def _sample_for_review(self) -> bool:
        if self._review_rate <= 0.0:
            return False
        return self._rng.random() < self._review_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEXT_FORMATS: frozenset[KnowledgeFormat] = frozenset(
    {
        KnowledgeFormat.PLAIN_TEXT,
        KnowledgeFormat.MARKDOWN,
        KnowledgeFormat.HTML,
        KnowledgeFormat.JSONL,
        KnowledgeFormat.CSV,
        KnowledgeFormat.SOURCE_CODE,
        KnowledgeFormat.JUPYTER,
        KnowledgeFormat.REQIF,
    }
)


_DATA_TYPE_BY_FORMAT: dict[KnowledgeFormat, str] = {
    KnowledgeFormat.PLAIN_TEXT: "plain_text",
    KnowledgeFormat.MARKDOWN: "markdown",
    KnowledgeFormat.HTML: "html",
    KnowledgeFormat.JSONL: "dataset_record",
    KnowledgeFormat.CSV: "dataset_row",
    KnowledgeFormat.SOURCE_CODE: "code",
    KnowledgeFormat.JUPYTER: "notebook_cell",
    KnowledgeFormat.PDF: "paper_section",
    KnowledgeFormat.DOCX: "document_section",
    KnowledgeFormat.REQIF: "requirement",
}


def _default_data_type(fmt: KnowledgeFormat) -> str:
    return _DATA_TYPE_BY_FORMAT.get(fmt, "untyped")


def _document_hash(payload: str | bytes) -> str:
    """SHA-256 hex digest of a ``RawDocument.payload``. Used by the
    ``IngestionRecord`` so callers can detect content changes between
    two ingestions of the same ``source_uri`` (e.g., a paper revised
    upstream)."""

    if isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = payload
    return hashlib.sha256(data).hexdigest()


def _fail_record(*, source_uri: str, error: str) -> IngestionRecord:
    return IngestionRecord(
        source_uri=source_uri,
        status=IngestionStatus.FAILED,
        error=error,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )


def _finish_record(
    record: IngestionRecord,
    *,
    status: IngestionStatus,
    error: str | None = None,
    chunks_produced: int | None = None,
    claims_extracted: int | None = None,
    review_required: bool | None = None,
) -> IngestionRecord:
    update: dict[str, Any] = {
        "status": status,
        "finished_at": datetime.now(timezone.utc),
    }
    if error is not None:
        update["error"] = error
    if chunks_produced is not None:
        update["chunks_produced"] = chunks_produced
    if claims_extracted is not None:
        update["claims_extracted"] = claims_extracted
    if review_required is not None:
        update["review_required"] = review_required
    return record.model_copy(update=update)


__all__ = ("Ingestor", "HumanReviewQueueCallback")
