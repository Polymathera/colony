"""Tests for :class:`MonorepoPersistedIngestor`.

The wrapper's contract:

- First PDF ingest writes ``<dir>/.ingested/<stem>/{extracted.md,
  ingestion.json}`` next to the source and produces an
  :class:`IngestionRecord` from the chunked + embedded markdown.
- A re-ingest of an unchanged PDF skips the reader entirely
  (cache-hit on ``pdf_sha256`` + matching extractor label).
- An ``extracted.md`` whose CONTENT differs from the manifest's
  ``extracted_md_sha256`` is trusted (user-edited markdown) — the
  wrapper re-ingests from the edited markdown without re-running the
  reader. Figure bytes round-trip through the sidecar's ``images/``
  directory (externalized on write, rehydrated on skip).
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
    """When ``extracted.md``'s CONTENT differs from the manifest's
    ``extracted_md_sha256``, the wrapper trusts the user's edit and
    re-ingests from the edited markdown without re-running the reader.
    Content-based on purpose — mtimes are scrambled by git checkout
    on fresh clones, so they must play no part in the decision."""

    ingestor, reader = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nbytes\n")

    await mpi.ingest_file(pdf, source_uri="u:edited")
    assert reader.calls == 1

    extracted_md = (
        pdf.parent / SIDECAR_DIRNAME / "paper" / EXTRACTED_MD_NAME
    )
    edited_text = (
        "# Introduction\n\nUser-edited markdown — replaces extractor "
        "output.\n"
    )
    extracted_md.write_text(edited_text, encoding="utf-8")

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


# ---- Figure persistence + content-hash provenance (2026-08-03) -------


class _FigurePdfReader(FormatReader):
    """Stub multimodal reader: stores figure bytes in the ingestor's
    ImageStore (as real readers do) and emits markdown referencing the
    resulting ``colony-image://`` URI."""

    def __init__(self, image_store) -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        self._image_store = image_store
        self.calls = 0

    def read(self, document):  # type: ignore[override]
        raise NotImplementedError("async-only stub — use read_async")

    async def read_async(self, document):  # type: ignore[override]
        self.calls += 1
        uri = await self._image_store.put(b"\x89PNG-bytes", mime="image/png")
        return [
            ParsedSection(
                section_path="1",
                heading="Figure section",
                text=f"See the coil layout.\n\n![img-0.jpeg]({uri})\n",
                citation=CitationSpan(
                    source_uri=document.source_uri, section_path="1",
                ),
                format="markdown",
            ),
        ]


def _figure_ingestor() -> tuple[Ingestor, _FigurePdfReader]:
    from polymathera.colony.knowledge.stores.image import InMemoryImageStore

    store = InMemoryImageStore()
    reader = _FigurePdfReader(store)
    registry = _registry_with(reader)
    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        image_store=store,
        review_sample_rate=0.0,
    )
    return ingestor, reader


async def test_figures_externalized_into_sidecar(tmp_path: Path) -> None:
    """Figure bytes land in ``images/<sha>.png`` and the committed
    markdown references them RELATIVELY — no ``colony-image://`` URIs
    (the 2026-08-03 run committed 911 dangling ones)."""

    from polymathera.colony.knowledge.monorepo_persisted_ingestor import (
        IMAGES_DIRNAME,
    )

    ingestor, _reader = _figure_ingestor()
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfig\n")

    rec = await mpi.ingest_file(pdf, source_uri="fig:1")
    assert rec.status is IngestionStatus.COMPLETED

    sidecar = pdf.parent / SIDECAR_DIRNAME / "paper"
    md = (sidecar / EXTRACTED_MD_NAME).read_text(encoding="utf-8")
    assert "colony-image://" not in md
    assert f"({IMAGES_DIRNAME}/" in md
    images = list((sidecar / IMAGES_DIRNAME).iterdir())
    assert len(images) == 1 and images[0].suffix == ".png"
    assert images[0].read_bytes() == b"\x89PNG-bytes"

    manifest = SidecarManifest.model_validate_json(
        (sidecar / INGESTION_JSON_NAME).read_text(encoding="utf-8"),
    )
    assert manifest.image_count == 1
    assert manifest.extracted_md_sha256


async def test_missing_figure_bytes_fail_the_file_no_sidecar(
    tmp_path: Path,
) -> None:
    """If referenced figure bytes cannot be fetched, the file FAILS
    and no sidecar is written — dangling refs never reach git."""

    from polymathera.colony.knowledge.stores.image import InMemoryImageStore

    class _DanglingRefReader(FormatReader):
        def __init__(self) -> None:
            super().__init__(handles=(KnowledgeFormat.PDF,))

        def read(self, document):  # type: ignore[override]
            return [
                ParsedSection(
                    section_path="1",
                    heading="",
                    text="![fig](colony-image://" + "ab" * 32 + ")\n",
                    citation=CitationSpan(
                        source_uri=document.source_uri, section_path="1",
                    ),
                    format="markdown",
                ),
            ]

    registry = _registry_with(_DanglingRefReader())
    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        image_store=InMemoryImageStore(),  # empty — bytes never stored
        review_sample_rate=0.0,
    )
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfig\n")

    rec = await mpi.ingest_file(pdf, source_uri="fig:dangling")
    assert rec.status is IngestionStatus.FAILED
    assert "dangling" in (rec.error or "")
    assert not (pdf.parent / SIDECAR_DIRNAME).exists()


async def test_skip_path_rehydrates_figures_into_fresh_store(
    tmp_path: Path,
) -> None:
    """A fresh deployment (empty ImageStore) ingesting an unchanged
    committed sidecar must rehydrate the figure bytes into its store
    and feed the pipeline ``colony-image://`` URIs — without calling
    the reader."""

    ingestor, reader = _figure_ingestor()
    mpi = MonorepoPersistedIngestor(
        ingestor, ingestor.readers, extractor_label="stub",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfig\n")
    await mpi.ingest_file(pdf, source_uri="fig:rt")
    assert reader.calls == 1

    # Second deployment: same sidecar on disk, EMPTY image store,
    # empty vector store (so the pre-flight doesn't mask the skip).
    ingestor2, reader2 = _figure_ingestor()
    mpi2 = MonorepoPersistedIngestor(
        ingestor2, ingestor2.readers, extractor_label="stub",
    )
    rec = await mpi2.ingest_file(pdf, source_uri="fig:rt")
    assert rec.status is IngestionStatus.COMPLETED
    assert reader2.calls == 0  # extraction skipped

    from polymathera.colony.knowledge.stores.image import _build_uri
    import hashlib
    sha = hashlib.sha256(b"\x89PNG-bytes").hexdigest()
    assert await ingestor2.image_store.has(_build_uri(sha))


async def test_corrupt_sidecar_missing_image_falls_back_to_reextraction(
    tmp_path: Path,
) -> None:
    from polymathera.colony.knowledge.monorepo_persisted_ingestor import (
        IMAGES_DIRNAME,
    )

    ingestor, reader = _figure_ingestor()
    mpi = MonorepoPersistedIngestor(
        ingestor, ingestor.readers, extractor_label="stub",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfig\n")
    await mpi.ingest_file(pdf, source_uri="fig:corrupt")
    assert reader.calls == 1

    sidecar = pdf.parent / SIDECAR_DIRNAME / "paper"
    for img in (sidecar / IMAGES_DIRNAME).iterdir():
        img.unlink()

    ingestor2, reader2 = _figure_ingestor()
    mpi2 = MonorepoPersistedIngestor(
        ingestor2, ingestor2.readers, extractor_label="stub",
    )
    rec = await mpi2.ingest_file(pdf, source_uri="fig:corrupt:v2")
    assert rec.status is IngestionStatus.COMPLETED
    assert reader2.calls == 1  # self-healed via full re-extraction
    # Sidecar restored, images back in place.
    assert list((sidecar / IMAGES_DIRNAME).iterdir())


async def test_legacy_manifest_without_content_hash_reextracts(
    tmp_path: Path,
    ingestor_and_readers: tuple[Ingestor, _CountingPdfReader],
) -> None:
    """Pre-content-hash sidecars (e.g. the 34 degraded ones from the
    2026-08-03 run) have unknown provenance — re-extract rather than
    trust them."""

    ingestor, reader = ingestor_and_readers
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\nlegacy\n")
    await mpi.ingest_file(pdf, source_uri="legacy:1")
    assert reader.calls == 1

    # Strip the content hash, simulating a legacy manifest.
    manifest_path = (
        pdf.parent / SIDECAR_DIRNAME / "paper" / INGESTION_JSON_NAME
    )
    manifest = SidecarManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    legacy = manifest.model_copy(update={"extracted_md_sha256": ""})
    manifest_path.write_text(
        legacy.model_dump_json(indent=2) + "\n", encoding="utf-8",
    )

    await mpi.ingest_file(pdf, source_uri="legacy:2")
    assert reader.calls == 2  # re-extracted


async def test_manifest_page_count_from_typed_citation_field(
    tmp_path: Path,
) -> None:
    """``page_count`` comes from the typed ``CitationSpan.page_number``
    — the convention every PDF reader implements. (The 2026-08-03 run
    recorded ``page_count: 0`` in all 34 manifests because the old
    helper read ``section.extra['page']``, which no reader sets.)"""

    class _PagedReader(FormatReader):
        def __init__(self) -> None:
            super().__init__(handles=(KnowledgeFormat.PDF,))

        def read(self, document):  # type: ignore[override]
            return [
                ParsedSection(
                    section_path=f"page-{n}",
                    heading="",
                    text=f"Page {n} body text.",
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"page-{n}",
                        page_number=n,
                    ),
                    format="markdown",
                )
                for n in (1, 2, 3)
            ]

    registry = _registry_with(_PagedReader())
    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_sample_rate=0.0,
    )
    mpi = MonorepoPersistedIngestor(ingestor, ingestor.readers)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\npaged\n")

    await mpi.ingest_file(pdf, source_uri="paged:1")
    manifest = SidecarManifest.model_validate_json(
        (pdf.parent / SIDECAR_DIRNAME / "paper" / INGESTION_JSON_NAME)
        .read_text(encoding="utf-8"),
    )
    assert manifest.page_count == 3
