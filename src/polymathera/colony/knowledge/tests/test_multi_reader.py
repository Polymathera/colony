"""Tests for the multi-reader-per-format contract.

Three contracts:

1. :class:`ReaderRegistry` lets different reader CLASSES coexist on
   the same format (additive); same-class re-registration replaces
   in place (idempotent against double-init).
2. :meth:`Ingestor.ingest_file` concatenates sections across readers
   for a format; one reader failing logs and skips, the rest still
   contribute. Ingest fails only if every reader returns nothing.
3. :func:`default_registry_with_pdf_extractor` registers the GROBID
   metadata-only sibling alongside the layout-aware body reader when
   ``grobid_url`` is supplied.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from polymathera.colony.knowledge.embedder import InMemoryEmbedder
from polymathera.colony.knowledge.ingestion import Ingestor
from polymathera.colony.knowledge.models import (
    CitationSpan, IngestionStatus, KnowledgeFormat, ParsedSection, RawDocument,
)
from polymathera.colony.knowledge.readers import (
    GrobidMetadataReader,
    default_registry_with_pdf_extractor,
)
from polymathera.colony.knowledge.readers.base import (
    FormatReader, FormatReaderError, ReaderRegistry,
)
from polymathera.colony.knowledge.stores.image import InMemoryImageStore
from polymathera.colony.knowledge.stores.vector import InMemoryVectorStore


class _SectionReader(FormatReader):
    """Tiny test double — emits one section with a configurable label
    so we can assert which reader's output reached the index."""

    def __init__(
        self,
        label: str,
        *,
        raise_exc: Exception | None = None,
        formats: tuple[KnowledgeFormat, ...] = (KnowledgeFormat.PDF,),
    ) -> None:
        super().__init__(handles=formats)
        self._label = label
        self._raise_exc = raise_exc
        self.calls = 0

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        self.calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        return [ParsedSection(
            section_path=self._label,
            text=f"section from {self._label}",
            citation=CitationSpan(source_uri=document.source_uri),
            extra={"label": self._label},
        )]


# ---------------------------------------------------------------------------
# ReaderRegistry — multi-reader semantics
# ---------------------------------------------------------------------------


def test_register_appends_for_different_classes() -> None:
    registry = ReaderRegistry()

    class _A(_SectionReader):
        pass

    class _B(_SectionReader):
        pass

    a = _A("a")
    b = _B("b")
    registry.register(a)
    registry.register(b)
    assert registry.readers_for(KnowledgeFormat.PDF) == (a, b)
    # ``reader_for`` keeps the legacy "last wins" contract.
    assert registry.reader_for(KnowledgeFormat.PDF) is b


def test_register_replaces_same_class_in_place() -> None:
    """Idempotency against double-init: registering a second instance
    of an already-registered class swaps it in, doesn't append."""
    registry = ReaderRegistry()

    class _A(_SectionReader):
        pass

    a1 = _A("a1")
    a2 = _A("a2")
    registry.register(a1)
    registry.register(a2)
    assert registry.readers_for(KnowledgeFormat.PDF) == (a2,)


def test_readers_for_unknown_format_returns_empty_tuple() -> None:
    assert ReaderRegistry().readers_for(KnowledgeFormat.PDF) == ()


# ---------------------------------------------------------------------------
# Ingestor — concat sections across readers, tolerate per-reader failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingestor_concatenates_sections_from_all_readers(
    tmp_path,
) -> None:
    """Two readers, both succeed: every chunk should reach the
    vector store (i.e., neither reader's output is dropped)."""

    class _Body(_SectionReader):
        pass

    class _Meta(_SectionReader):
        pass

    body = _Body("body")
    meta = _Meta("meta")
    registry = ReaderRegistry()
    registry.register(body)
    registry.register(meta)

    vector_store = InMemoryVectorStore()
    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=vector_store,
        image_store=InMemoryImageStore(),
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 (fake)")
    record = await ingestor.ingest_file(pdf)

    assert record.status == IngestionStatus.COMPLETED
    assert body.calls == 1
    assert meta.calls == 1
    # Both readers' single section produced one chunk each.
    assert record.chunks_produced >= 2


@pytest.mark.asyncio
async def test_ingestor_tolerates_one_reader_failing(tmp_path) -> None:
    """A transient failure on ONE reader (GROBID outage, 500 from
    metadata sibling) MUST NOT poison the body extractor's output."""

    class _Body(_SectionReader):
        pass

    class _Meta(_SectionReader):
        pass

    body = _Body("body")
    meta = _Meta("meta", raise_exc=FormatReaderError("transient grobid failure"))
    registry = ReaderRegistry()
    registry.register(body)
    registry.register(meta)

    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        image_store=InMemoryImageStore(),
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 (fake)")
    record = await ingestor.ingest_file(pdf)

    assert record.status == IngestionStatus.COMPLETED
    assert body.calls == 1
    assert meta.calls == 1
    assert record.chunks_produced >= 1


@pytest.mark.asyncio
async def test_ingestor_fails_when_every_reader_fails(tmp_path) -> None:
    """If no reader produces sections, the record is FAILED with the
    last reader's error message."""

    class _A(_SectionReader):
        pass

    class _B(_SectionReader):
        pass

    a = _A("a", raise_exc=FormatReaderError("a-bad"))
    b = _B("b", raise_exc=FormatReaderError("b-bad"))
    registry = ReaderRegistry()
    registry.register(a)
    registry.register(b)

    ingestor = Ingestor(
        readers=registry,
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        image_store=InMemoryImageStore(),
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 (fake)")
    record = await ingestor.ingest_file(pdf)

    assert record.status == IngestionStatus.FAILED
    assert "b-bad" in (record.error or "")


# ---------------------------------------------------------------------------
# default_registry_with_pdf_extractor — GROBID sibling registration
# ---------------------------------------------------------------------------


def test_pdf_extractor_registry_omits_grobid_when_url_unset() -> None:
    """No ``grobid_url`` → no metadata sibling. Backward compat with
    deployments that have not yet enabled GROBID."""
    registry = default_registry_with_pdf_extractor(
        backend="mistral_ocr",
        image_store=InMemoryImageStore(),
        backend_kwargs={"api_key": "fake"},
    )
    readers = registry.readers_for(KnowledgeFormat.PDF)
    assert not any(isinstance(r, GrobidMetadataReader) for r in readers)


def test_pdf_extractor_registry_registers_grobid_sibling_when_url_set() -> None:
    """``grobid_url`` set → :class:`GrobidMetadataReader` is registered
    ALONGSIDE the body extractor (multi-reader registry)."""
    registry = default_registry_with_pdf_extractor(
        backend="mistral_ocr",
        image_store=InMemoryImageStore(),
        backend_kwargs={"api_key": "fake"},
        grobid_url="http://grobid:8070",
    )
    readers = registry.readers_for(KnowledgeFormat.PDF)
    # Body extractor + pypdf shadow + grobid metadata sibling.
    assert any(isinstance(r, GrobidMetadataReader) for r in readers)
    assert len(readers) >= 2
