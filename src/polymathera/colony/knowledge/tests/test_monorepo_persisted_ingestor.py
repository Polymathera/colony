"""Tests for :class:`MonorepoPersistedIngestor`.

The wrapper's contract:

- First PDF ingest writes ``<dir>/.ingested/<stem>/{extracted.md,
  ingestion.json}`` next to the source and produces an
  :class:`IngestionRecord` from the chunked + embedded markdown.
- A re-ingest of an unchanged PDF skips the reader entirely
  (cache-hit on ``pdf_sha256``).
- An ``extracted.md`` whose mtime is newer than the source PDF is
  trusted (user-edited markdown) — the wrapper re-ingests from the
  edited markdown without re-running the reader.
- A PDF whose bytes change invalidates the sidecar; the wrapper
  re-runs the reader and rewrites the sidecar.
- Non-PDF inputs flow straight through to the underlying
  :class:`Ingestor` (no sidecar duplication of the readable artifact).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from polymathera.colony.knowledge import (
    CitationSpan,
    InMemoryEmbedder,
    InMemoryVectorStore,
    Ingestor,
    IngestionStatus,
    KnowledgeFormat,
    MonorepoPersistedIngestor,
    ParsedSection,
    ReaderRegistry,
    SidecarManifest,
)
from polymathera.colony.knowledge.monorepo_persisted_ingestor import (
    EXTRACTED_MD_NAME,
    INGESTION_JSON_NAME,
    SIDECAR_DIRNAME,
)
from polymathera.colony.knowledge.readers.base import FormatReader
from polymathera.colony.knowledge.readers import default_registry


pytestmark = pytest.mark.asyncio


# ---- Test doubles ----------------------------------------------------


class _CountingPdfReader(FormatReader):
    """Stub PDF reader that returns a fixed pair of sections and
    counts how many times it was invoked. Lets the tests assert the
    sidecar cache short-circuits the extractor on repeat ingests."""

    def __init__(self, text: str = "Intro paragraph about SERF.") -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        self.text = text
        self.calls = 0

    def read(self, document):  # type: ignore[override]
        self.calls += 1
        return [
            ParsedSection(
                section_path="1",
                heading="Introduction",
                text=self.text,
                citation=CitationSpan(
                    source_uri=document.source_uri, section_path="1",
                ),
                format="markdown",
            ),
            ParsedSection(
                section_path="2",
                heading="Method",
                text="Optical pumping at 894 nm.",
                citation=CitationSpan(
                    source_uri=document.source_uri, section_path="2",
                ),
                format="markdown",
            ),
        ]


def _registry_with(pdf_reader: FormatReader) -> ReaderRegistry:
    registry = default_registry()
    registry.register(pdf_reader)
    return registry


@pytest.fixture
def ingestor_and_readers() -> tuple[Ingestor, _CountingPdfReader]:
    reader = _CountingPdfReader()
    registry = _registry_with(reader)
    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_sample_rate=0.0,
    )
    return ingestor, reader


# ---- PDF path --------------------------------------------------------


async def test_first_pdf_ingest_writes_sidecar(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    ingestor, reader = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers, extractor_label="stub")

    pdf = tmp_path / "papers" / "allred_2002.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\nfake-pdf-bytes\n")

    rec = await mpi.ingest_file(pdf, source_uri="paper:allred_2002")
    assert rec.status is IngestionStatus.COMPLETED
    assert rec.chunks_produced > 0
    assert reader.calls == 1

    sidecar = pdf.parent / SIDECAR_DIRNAME / "allred_2002"
    assert (sidecar / EXTRACTED_MD_NAME).is_file()
    assert (sidecar / INGESTION_JSON_NAME).is_file()
    manifest = SidecarManifest.model_validate_json(
        (sidecar / INGESTION_JSON_NAME).read_text(encoding="utf-8"),
    )
    assert manifest.source_uri == "paper:allred_2002"
    assert manifest.pdf_sha256
    assert manifest.section_count == 2
    assert manifest.extractor == "stub"


async def test_reingest_unchanged_pdf_skips_reader(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    ingestor, reader = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nbytes\n")

    await mpi.ingest_file(pdf, source_uri="u")
    assert reader.calls == 1

    # The second ingest of the same bytes must NOT invoke the reader
    # again. The Ingestor's idempotency check returns SKIPPED_ALREADY_PRESENT
    # because the source_uri is already in the vector store; either way,
    # the extractor must stay at one call.
    rec2 = await mpi.ingest_file(pdf, source_uri="u")
    assert reader.calls == 1
    assert rec2.status in {
        IngestionStatus.COMPLETED,
        IngestionStatus.SKIPPED_ALREADY_PRESENT,
    }


async def test_user_edited_markdown_trumps_pdf_extraction(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    """When ``extracted.md`` mtime is newer than the PDF, the wrapper
    trusts the user's edit and re-ingests from the edited markdown
    without re-running the reader."""

    ingestor, reader = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nbytes\n")

    await mpi.ingest_file(pdf, source_uri="u:edited")
    assert reader.calls == 1

    # Edit the markdown after the PDF and ensure mtime is strictly
    # greater than the PDF's. Don't sleep — bump the mtime directly.
    extracted_md = (
        pdf.parent / SIDECAR_DIRNAME / "paper" / EXTRACTED_MD_NAME
    )
    edited_text = (
        "# Introduction\n\nUser-edited markdown — replaces extractor "
        "output.\n"
    )
    extracted_md.write_text(edited_text, encoding="utf-8")
    pdf_mtime = pdf.stat().st_mtime
    import os
    os.utime(extracted_md, (pdf_mtime + 5, pdf_mtime + 5))

    rec = await mpi.ingest_file(pdf, source_uri="u:edited:v2")
    assert reader.calls == 1  # reader NOT called again
    assert rec.status is IngestionStatus.COMPLETED


async def test_changed_pdf_invalidates_sidecar(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    ingestor, reader = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nv1\n")
    await mpi.ingest_file(pdf, source_uri="paper:v1")
    assert reader.calls == 1
    old_manifest = SidecarManifest.model_validate_json(
        (pdf.parent / SIDECAR_DIRNAME / "paper" / INGESTION_JSON_NAME).read_text(
            encoding="utf-8",
        ),
    )
    old_sha = old_manifest.pdf_sha256

    # Rewrite the PDF with different bytes — sha256 changes, sidecar
    # is invalidated, reader must re-run, manifest is overwritten.
    pdf.write_bytes(b"%PDF-1.4\nv2-different-bytes\n")
    await mpi.ingest_file(pdf, source_uri="paper:v2")
    assert reader.calls == 2
    new_manifest = SidecarManifest.model_validate_json(
        (pdf.parent / SIDECAR_DIRNAME / "paper" / INGESTION_JSON_NAME).read_text(
            encoding="utf-8",
        ),
    )
    assert new_manifest.pdf_sha256 != old_sha


# ---- Non-PDF passthrough --------------------------------------------


async def test_non_pdf_file_skips_sidecar(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    """Markdown / plain-text inputs flow straight through to the
    underlying ingestor — the file IS already the readable artifact,
    a sidecar would just duplicate it."""

    ingestor, _ = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)

    md = tmp_path / "note.md"
    md.write_text("# Title\n\nProse paragraph.\n", encoding="utf-8")
    rec = await mpi.ingest_file(md, source_uri="note:1")
    assert rec.status is IngestionStatus.COMPLETED
    assert not (tmp_path / SIDECAR_DIRNAME).exists()


async def test_missing_file_returns_failed_record(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    ingestor, _ = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)

    rec = await mpi.ingest_file(
        tmp_path / "nope.pdf", source_uri="missing:1",
    )
    assert rec.status is IngestionStatus.FAILED
