"""End-to-end tests for the ``Ingestor``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from polymathera.colony.knowledge import (
    CorpusTier,
    DeterministicClaimExtractor,
    InMemoryEmbedder,
    InMemoryGraphStore,
    InMemoryVectorStore,
    Ingestor,
    IngestionStatus,
    KnowledgeFormat,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def ingestor() -> Ingestor:
    return Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph_store=InMemoryGraphStore(),
        extractors=[DeterministicClaimExtractor()],
        review_sample_rate=0.0,
    )


async def test_ingest_markdown_text(ingestor: Ingestor) -> None:
    text = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Confinement\n\n"
        "JET requires deuterium-tritium fuel.\n"
    )
    record = await ingestor.ingest_text(
        text, source_uri="book:wesson:ch1",
        fmt=KnowledgeFormat.MARKDOWN,
        tier=CorpusTier.TIER_1_FOUNDATIONS,
    )
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced > 0
    assert record.claims_extracted >= 1
    assert record.tier is CorpusTier.TIER_1_FOUNDATIONS
    assert record.error is None


async def test_ingest_jsonl(tmp_path: Path, ingestor: Ingestor) -> None:
    p = tmp_path / "papers.jsonl"
    p.write_text(
        json.dumps({"title": "Paper A", "doi": "10.1/x"}) + "\n"
        + json.dumps({"title": "Paper B", "doi": "10.2/y"}) + "\n",
        encoding="utf-8",
    )
    record = await ingestor.ingest_file(p)
    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced >= 1
    assert record.detected_format is KnowledgeFormat.JSONL


async def test_ingest_unknown_format_fails(ingestor: Ingestor) -> None:
    record = await ingestor.ingest_text(
        "x", source_uri="thing:1", fmt=KnowledgeFormat.UNKNOWN,
    )
    assert record.status is IngestionStatus.FAILED
    assert record.error is not None


async def test_review_queue_called_when_sampled(tmp_path: Path) -> None:
    queue: list = []

    async def reviewer(record, chunks):
        queue.append((record, chunks))

    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_queue=reviewer,
        review_sample_rate=1.0,
    )
    record = await ing.ingest_text(
        "Body text.", source_uri="x:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert record.review_required is True
    assert len(queue) == 1


async def test_pipeline_state_visible_via_record_attributes(
    ingestor: Ingestor,
) -> None:
    text = "para one\n\npara two"
    record = await ingestor.ingest_text(
        text, source_uri="x:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert record.detected_format is KnowledgeFormat.PLAIN_TEXT
    assert record.started_at is not None
    assert record.finished_at is not None
    assert record.finished_at >= record.started_at


async def test_ingestor_persists_chunks_to_vector_store() -> None:
    vstore = InMemoryVectorStore()
    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=vstore,
        review_sample_rate=0.0,
    )
    await ing.ingest_text(
        "First paragraph.\n\nSecond paragraph.",
        source_uri="x:y", fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    assert await vstore.count() >= 1


# ---------------------------------------------------------------------------
# Concurrency contract (Fix A: chunk-level + Fix C: reader-level)
# ---------------------------------------------------------------------------


class _ConcurrencyProbeExtractor:
    """Extractor whose ``extract`` body sleeps for ``delay`` seconds and
    tracks the peak number of in-flight concurrent invocations. Pinning
    the peak proves the orchestrator (``_run_extractors``) fans out
    instead of awaiting each chunk before starting the next."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self._in_flight = 0
        self.peak_in_flight = 0
        self._lock = asyncio.Lock()

    async def extract(self, chunk):  # noqa: ANN001 — duck-typed in tests
        async with self._lock:
            self._in_flight += 1
            if self._in_flight > self.peak_in_flight:
                self.peak_in_flight = self._in_flight
        try:
            await asyncio.sleep(self.delay)
        finally:
            async with self._lock:
                self._in_flight -= 1
        return ()


async def test_run_extractors_fans_out_chunks_concurrently(
    tmp_path: Path,
) -> None:
    """Fix A regression pin: ``_run_extractors`` must run every
    (chunk × extractor) pair concurrently. Before the fix the
    inner ``for chunk in chunks: await extract`` loop serialised
    every Anthropic call; 88 calls × 11 s = 16 minutes wall in
    run7."""

    probe = _ConcurrencyProbeExtractor(delay=0.05)
    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        extractors=[probe],
        review_sample_rate=0.0,
    )
    # 8 separately-headed sections → 8 distinct chunks (the prose
    # chunker emits roughly one chunk per top-level heading at
    # default budget); enough to make the concurrency signal
    # unambiguous without slowing the suite.
    text = "\n\n".join(
        f"## Section {i}\n\nBody paragraph for section {i}."
        for i in range(8)
    )
    start = asyncio.get_event_loop().time()
    record = await ing.ingest_text(
        text, source_uri="probe:1", fmt=KnowledgeFormat.MARKDOWN,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert record.status is IngestionStatus.COMPLETED
    assert record.chunks_produced >= 4, (
        "Chunker must produce multiple chunks for the concurrency "
        "signal to be meaningful."
    )
    # Concurrent: peak should be ≥2 (typically equals chunk count).
    # Serial: peak would equal 1.
    assert probe.peak_in_flight >= 2, (
        f"Expected concurrent extractor calls; saw peak in-flight "
        f"= {probe.peak_in_flight} (serial-loop regression)."
    )
    # Wall-time sanity: a 0.05s-delay extractor run serially over N
    # chunks would take ≥ N×0.05s. With concurrency, total ≈ 0.05s
    # + overhead. Bound loosely at < N×0.05 / 2 so a slow CI box
    # doesn't flake.
    expected_serial = record.chunks_produced * 0.05
    assert elapsed < expected_serial / 2 + 0.5, (
        f"Wall time {elapsed:.3f}s suggests near-serial execution "
        f"(serial bound = {expected_serial:.3f}s); regression to "
        f"`for chunk in chunks: await` loop."
    )


async def test_run_extractors_logs_and_skips_per_chunk_failure(
    tmp_path: Path,
) -> None:
    """Fix A recovery pin: per-chunk failure in any one extractor
    must NOT poison the whole batch — same contract the serial loop
    enforced via try/except, preserved through the gather +
    return_exceptions transition."""

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, chunk):  # noqa: ANN001
            self.calls += 1
            if self.calls % 2 == 0:
                raise RuntimeError("synthetic per-chunk failure")
            return ()

    flaky = _Flaky()
    ing = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        extractors=[flaky],
        review_sample_rate=0.0,
    )
    text = "\n\n".join(
        f"## Section {i}\n\nBody for section {i}." for i in range(6)
    )
    record = await ing.ingest_text(
        text, source_uri="probe:flaky",
        fmt=KnowledgeFormat.MARKDOWN,
    )
    # Some chunks failed mid-extract but the ingest itself completes:
    # the surviving chunks land, the failed ones are logged and
    # dropped — same behaviour as the prior serial loop.
    assert record.status is IngestionStatus.COMPLETED
    assert flaky.calls >= 2


async def test_ingest_document_runs_readers_concurrently() -> None:
    """Fix C regression pin: the reader fall-through loop runs
    concurrently. Before the fix two LLM-driven PDF readers
    registered as fall-throughs (Anthropic + Mistral OCR, etc.)
    would serialise their per-document calls."""

    from polymathera.colony.knowledge.models import (
        ParsedSection, CitationSpan, RawDocument,
    )
    from polymathera.colony.knowledge.readers import (
        FormatReader, ReaderRegistry,
    )

    # Distinct classes — ReaderRegistry.register replaces same-class
    # entries (idempotent against double-init), so two instances of
    # one class would collapse to one. The contract under test is
    # fall-through across multiple registered readers — distinct
    # classes are the realistic shape.
    class _ReaderBase(FormatReader):
        name: str = "base"

        def __init__(self) -> None:
            super().__init__(handles=(KnowledgeFormat.PLAIN_TEXT,))
            self.invocations = 0

        def read(self, document: RawDocument):  # noqa: D401
            raise NotImplementedError

        async def read_async(self, document):  # noqa: ANN001
            self.invocations += 1
            await asyncio.sleep(0.1)
            return (
                ParsedSection(
                    text=f"sec from {self.name}",
                    format="text",
                    section_path=self.name,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=self.name,
                    ),
                ),
            )

    class _ReaderA(_ReaderBase):
        name = "r1"

    class _ReaderB(_ReaderBase):
        name = "r2"

    r1 = _ReaderA()
    r2 = _ReaderB()
    registry = ReaderRegistry()
    registry.register(r1)
    registry.register(r2)
    ing = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_sample_rate=0.0,
    )
    start = asyncio.get_event_loop().time()
    record = await ing.ingest_text(
        "body", source_uri="probe:two-readers",
        fmt=KnowledgeFormat.PLAIN_TEXT,
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert record.status is IngestionStatus.COMPLETED
    # Both readers invoked.
    assert r1.invocations == 1 and r2.invocations == 1
    # Concurrent execution: wall ≈ max(0.1, 0.1) ≈ 0.1s + overhead.
    # Serial regression: wall ≥ 0.2s. Bound loosely for CI.
    assert elapsed < 0.18, (
        f"Wall time {elapsed:.3f}s suggests reader fall-through "
        f"loop is serial; expected < 0.18s for two concurrent 0.1s "
        f"readers."
    )
