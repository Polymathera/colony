"""Required-reader enforcement — no silent fallback to free extraction.

Closes the 2026-08-03 hole found before the first OPM-MEG literature
ingest: with the paid body extractor (``knowledge.pdf_extractor.backend``)
down (e.g. Mistral OCR credit-out), the multi-reader loops accepted the
surviving free readers' sections (pypdf body text, GROBID metadata),
marked the file COMPLETED, and wrote a sidecar whose ``pdf_sha256``
then blocked the paid extraction from ever being retried.

Contract pinned here:

- ``ReaderRegistry.register(reader, required=True)`` marks a reader
  required; same-class replacement updates the flag.
- ``default_registry_with_pdf_extractor`` registers the configured
  body extractor as the ONLY PDF body reader (no pypdf sibling) and
  marks it required; the GROBID metadata sibling stays optional.
- A required reader's failure fails the document in BOTH multi-reader
  loops (``Ingestor.ingest_document`` and
  ``MonorepoPersistedIngestor._run_readers``); no sidecar is written,
  so the retry after the backend is restored re-runs the full ingest.
- A sidecar whose manifest ``extractor`` label differs from the
  wrapper's configured label is re-extracted (backend switch re-pays
  for quality instead of serving the old backend's output).
"""

from __future__ import annotations

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
    INGESTION_JSON_NAME,
    SIDECAR_DIRNAME,
)
from polymathera.colony.knowledge.readers import (
    GrobidMetadataReader,
    MistralOcrPdfReader,
    PdfReader,
    default_registry,
    default_registry_with_pdf_extractor,
)
from polymathera.colony.knowledge.readers.base import FormatReader


pytestmark = pytest.mark.asyncio


# ---- Test doubles ----------------------------------------------------


def _section(source_uri: str, text: str, path: str = "1") -> ParsedSection:
    return ParsedSection(
        section_path=path,
        heading="",
        text=text,
        citation=CitationSpan(source_uri=source_uri, section_path=path),
        format="markdown",
    )


class _BodyReader(FormatReader):
    """Paid-extractor stand-in: succeeds unless ``down`` is set,
    counting invocations either way (the counter is what
    distinguishes an extraction re-run from a cache skip)."""

    def __init__(self, text: str = "High-quality body text.") -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        self.text = text
        self.calls = 0
        self.down = False

    def read(self, document):  # type: ignore[override]
        self.calls += 1
        if self.down:
            raise RuntimeError("402 payment required: credit exhausted")
        return [_section(document.source_uri, self.text)]


class _MetaReader(FormatReader):
    """Free metadata sibling stand-in (GROBID-shaped): always
    succeeds with a metadata-only section."""

    def __init__(self) -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        self.calls = 0

    def read(self, document):  # type: ignore[override]
        self.calls += 1
        return [_section(document.source_uri, "Author: A. Uthor", path="meta")]


def _registry(
    body: FormatReader, *optional: FormatReader,
) -> ReaderRegistry:
    """Production-shaped registry: text/markdown readers present (the
    resume path re-ingests sidecar markdown through them), ``body``
    required, everything else optional."""

    registry = default_registry()
    registry.register(body, required=True)
    for reader in optional:
        registry.register(reader)
    return registry


def _ingestor(registry: ReaderRegistry) -> Ingestor:
    return Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_sample_rate=0.0,
    )


def _write_pdf(tmp_path: Path, content: bytes = b"%PDF-1.4\nv1\n") -> Path:
    pdf = tmp_path / "papers" / "paper.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(content)
    return pdf


# ---- ReaderRegistry required flag ------------------------------------


async def test_registry_required_flag_and_replacement() -> None:
    registry = ReaderRegistry()
    body = _BodyReader()
    meta = _MetaReader()
    registry.register(body, required=True)
    registry.register(meta)

    assert registry.is_required(KnowledgeFormat.PDF, body) is True
    assert registry.is_required(KnowledgeFormat.PDF, meta) is False
    assert registry.readers_for(KnowledgeFormat.PDF) == (body, meta)

    # Same-class replacement updates instance AND flag.
    body2 = _BodyReader()
    registry.register(body2)  # re-registered WITHOUT required
    assert registry.readers_for(KnowledgeFormat.PDF) == (body2, meta)
    assert registry.is_required(KnowledgeFormat.PDF, body2) is False
    # The replaced instance is no longer known to the registry.
    assert registry.is_required(KnowledgeFormat.PDF, body) is False


async def test_pdf_extractor_registry_no_pypdf_and_body_required() -> None:
    registry = default_registry_with_pdf_extractor(
        backend="mistral_ocr",
        image_store=object(),
        grobid_url="http://grobid:8070",
    )
    pdf_readers = registry.readers_for(KnowledgeFormat.PDF)
    assert not any(type(r) is PdfReader for r in pdf_readers)

    body = [r for r in pdf_readers if isinstance(r, MistralOcrPdfReader)]
    grobid = [r for r in pdf_readers if isinstance(r, GrobidMetadataReader)]
    assert len(body) == 1 and len(grobid) == 1
    assert registry.is_required(KnowledgeFormat.PDF, body[0]) is True
    assert registry.is_required(KnowledgeFormat.PDF, grobid[0]) is False


# ---- Ingestor.ingest_document enforcement ----------------------------


async def test_ingest_document_required_failure_fails_document(
    tmp_path: Path,
) -> None:
    """Paid reader down + free sibling up → FAILED, nothing indexed.
    The free sibling's sections must NOT become the document."""

    body = _BodyReader()
    body.down = True
    ingestor = _ingestor(_registry(body, _MetaReader()))

    pdf = _write_pdf(tmp_path)
    rec = await ingestor.ingest_file(pdf, source_uri="paper:x")
    assert rec.status is IngestionStatus.FAILED
    assert "required reader" in (rec.error or "")
    assert not await ingestor._vector_store.list_chunks_for_source("paper:x")


async def test_ingest_document_optional_failure_still_completes(
    tmp_path: Path,
) -> None:
    """Counterpart: a free sibling's failure keeps the existing
    warn-and-continue semantics when the required reader succeeded."""

    class _FailingMeta(_MetaReader):
        def read(self, document):  # type: ignore[override]
            self.calls += 1
            raise RuntimeError("grobid down")

    ingestor = _ingestor(_registry(_BodyReader(), _FailingMeta()))

    pdf = _write_pdf(tmp_path)
    rec = await ingestor.ingest_file(pdf, source_uri="paper:y")
    assert rec.status is IngestionStatus.COMPLETED
    assert rec.chunks_produced > 0


# ---- MonorepoPersistedIngestor enforcement + resume ------------------


async def test_mpi_required_failure_writes_no_sidecar_then_resumes(
    tmp_path: Path,
) -> None:
    """Credit-out run: FAILED record, NO sidecar (nothing to poison
    the cache). Recharged re-run: full re-extract + sidecar written."""

    body = _BodyReader()
    body.down = True
    ingestor = _ingestor(_registry(body, _MetaReader()))
    mpi = MonorepoPersistedIngestor(
        ingestor, ingestor.readers, extractor_label="stub",
    )

    pdf = _write_pdf(tmp_path)
    rec = await mpi.ingest_file(pdf, source_uri="paper:z")
    assert rec.status is IngestionStatus.FAILED
    assert "required reader" in (rec.error or "")
    assert not (pdf.parent / SIDECAR_DIRNAME).exists()
    assert not await ingestor._vector_store.list_chunks_for_source("paper:z")

    # Backend restored → the SAME call now runs the full pipeline.
    body.down = False
    rec2 = await mpi.ingest_file(pdf, source_uri="paper:z")
    assert rec2.status is IngestionStatus.COMPLETED
    assert (pdf.parent / SIDECAR_DIRNAME / "paper" / INGESTION_JSON_NAME).is_file()


# ---- Sidecar extractor-label invalidation ----------------------------


async def _ingest_once(
    pdf: Path, label: str, *, source_uri: str = "paper:label",
) -> tuple[MonorepoPersistedIngestor, _BodyReader]:
    body = _BodyReader()
    ingestor = _ingestor(_registry(body))
    mpi = MonorepoPersistedIngestor(
        ingestor, ingestor.readers, extractor_label=label,
    )
    await mpi.ingest_file(pdf, source_uri=source_uri)
    return mpi, body


async def test_mpi_same_label_skips_extraction(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    mpi, body = await _ingest_once(pdf, "mistral_ocr")
    assert body.calls == 1
    await mpi.ingest_file(pdf, source_uri="paper:label")
    assert body.calls == 1  # cache hit — extractor not re-run


async def test_mpi_label_mismatch_reextracts(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    await _ingest_once(pdf, "mistral_ocr")

    mpi2, body2 = await _ingest_once(pdf, "anthropic")
    # _ingest_once already ran one ingest with the new label — the
    # extractor MUST have run despite the sha256 cache hit.
    assert body2.calls == 1
    manifest = SidecarManifest.model_validate_json(
        (pdf.parent / SIDECAR_DIRNAME / "paper" / INGESTION_JSON_NAME)
        .read_text(encoding="utf-8"),
    )
    assert manifest.extractor == "anthropic"


async def test_mpi_unrecorded_label_reextracts(tmp_path: Path) -> None:
    """Legacy sidecar with an empty ``extractor`` field is treated as
    unknown provenance → re-extracted under a configured label."""

    pdf = _write_pdf(tmp_path)
    await _ingest_once(pdf, "")  # label unset → manifest.extractor == ""

    _, body2 = await _ingest_once(pdf, "mistral_ocr")
    assert body2.calls == 1


async def test_mpi_empty_wrapper_label_keeps_sha_skip(tmp_path: Path) -> None:
    """A wrapper with NO configured label cannot validate provenance —
    the sha256 skip stays authoritative (no spurious re-pay)."""

    pdf = _write_pdf(tmp_path)
    await _ingest_once(pdf, "mistral_ocr")

    _, body2 = await _ingest_once(pdf, "")
    assert body2.calls == 0
