"""PDF reader: ``pypdf``-based fallback.

Per master §6.3 the canonical PDF path is GROBID (PDF → TEI XML →
sectioned text + refs + figures + tables). That landing is C1b — it
needs a running GROBID Docker service. This reader is the in-process
fallback: ``pypdf`` is pure-Python, ships with every colony install
under the ``knowledge`` extra, and produces page-grained sections
with no external service.

When GROBID is available (C1b), the ``GrobidPdfReader`` subclass
overrides ``read`` and uses TEI section structure; the registry
prefers it over this fallback by registering it last.

Sections here are *one per page*; the chunker turns those into
retrieval-sized chunks. The reader produces no headings — pypdf can't
recover document structure from a flat PDF.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..models import CitationSpan, KnowledgeFormat, ParsedSection, RawDocument
from .base import FormatReader, FormatReaderError


logger = logging.getLogger(__name__)


class PdfReader(FormatReader):
    """In-process PDF reader using ``pypdf``.

    Caller-supplied ``RawDocument.payload`` MUST be ``bytes``; PDFs
    are not text-shaped.
    """

    handles = (KnowledgeFormat.PDF,)

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        try:
            import pypdf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FormatReaderError(
                "PdfReader requires the 'pypdf' package. Install the "
                "'knowledge' extra: pip install polymathera-colony[knowledge].",
            ) from exc

        if document.is_text:
            raise FormatReaderError(
                f"PDF reader expected bytes for {document.source_uri}, "
                "got text.",
            )

        import io

        try:
            reader = pypdf.PdfReader(io.BytesIO(document.bytes_))
        except Exception as exc:  # noqa: BLE001
            raise FormatReaderError(
                f"pypdf failed to open {document.source_uri}: {exc}",
            ) from exc

        sections: list[ParsedSection] = []
        offset = 0
        for page_no, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PdfReader: page %d of %s failed: %s",
                    page_no, document.source_uri, exc,
                )
                continue
            text = text.strip()
            if not text:
                continue
            char_start = offset
            char_end = offset + len(text)
            sections.append(
                ParsedSection(
                    section_path=f"{page_no + 1}",
                    heading=f"page {page_no + 1}",
                    text=text,
                    citation=CitationSpan(
                        source_uri=document.source_uri,
                        section_path=f"{page_no + 1}",
                        char_start=char_start,
                        char_end=char_end,
                        page_number=page_no + 1,
                    ),
                    extra={"page_number": page_no + 1},
                )
            )
            offset = char_end + 1
        return tuple(sections)


__all__ = ("PdfReader",)
