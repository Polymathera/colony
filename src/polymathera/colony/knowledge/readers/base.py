"""``FormatReader`` ABC + ``ReaderRegistry``.

Per master §6.3 the ingestion pipeline starts with a per-format reader
that turns a raw blob into structured ``ParsedSection``s. Each reader
declares the formats it handles via the class-level
``handles: tuple[KnowledgeFormat, ...]`` attribute; the registry uses
that to resolve a reader for a given detected format.

The ABC is sync because the heavy work (PDF / DOCX / Jupyter parsing)
is CPU-bound and the colony runtime calls readers from worker threads
(``asyncio.to_thread``) when invoked under an event loop. Readers that
genuinely need async (e.g., a remote-fetching reader) override
``read_async``.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from ..models import KnowledgeFormat, ParsedSection, RawDocument


logger = logging.getLogger(__name__)


class FormatReaderError(RuntimeError):
    """Raised when a reader cannot parse its input."""


class PdfTooManyPagesError(FormatReaderError):
    """Raised when a PDF reader rejects a document because its page
    count exceeds the backend's hard limit (Mistral OCR: 1000 pages).

    Distinct from generic :class:`FormatReaderError` so a wrapping
    :class:`FallbackPdfReader` can route to a backend without that
    limit (Gemini, LlamaParse, self-hosted) instead of failing the
    ingest. Carries ``page_count`` / ``max_pages`` when the backend
    surfaces them, ``None`` when the reader inferred the limit from
    a less-structured error response.
    """

    def __init__(
        self,
        message: str,
        *,
        page_count: int | None = None,
        max_pages: int | None = None,
    ) -> None:
        super().__init__(message)
        self.page_count = page_count
        self.max_pages = max_pages


class FormatReader(ABC):
    """Read a ``RawDocument`` into a sequence of ``ParsedSection``s."""

    def __init__(self, *, handles: tuple[KnowledgeFormat, ...]) -> None:
        """
        ``handles`` declares the formats this reader handles. Subclasses set this; the registry
        indexes on it."""
        self.handles = handles

    @abstractmethod
    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        """Synchronous parse. CPU-bound by default; the registry runs
        readers in a worker thread under async callers."""

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        """Async parse. Defaults to wrapping ``read`` in
        ``asyncio.to_thread`` so subclasses don't have to. Override
        when a reader does its own async work (a remote API call)."""

        return await asyncio.to_thread(self.read, document)


class ReaderRegistry:
    """Registry that resolves readers for a detected format.

    Supports **multi-reader-per-format**: different reader CLASSES
    register additively (e.g. a layout-aware body extractor and a
    metadata-only GROBID sibling for ``KnowledgeFormat.PDF``); a
    re-registration of the SAME class replaces the previous instance
    (idempotent against double-init). Section concatenation across
    readers is the Ingestor's responsibility.

    Backward-compat: :meth:`reader_for` returns the LAST reader
    registered for a format (preserves the old "last write wins"
    contract used by tests and by ``default_registry_with_grobid``).
    Multi-reader callers use :meth:`readers_for`.
    """

    def __init__(self) -> None:
        self._by_format: dict[KnowledgeFormat, list[FormatReader]] = {}

    def register(self, reader: FormatReader) -> None:
        if not reader.handles:
            raise ValueError(
                f"Reader {type(reader).__name__} declares no handled formats.",
            )
        for fmt in reader.handles:
            readers = self._by_format.setdefault(fmt, [])
            for i, existing in enumerate(readers):
                if type(existing) is type(reader):
                    logger.info(
                        "ReaderRegistry: replacing %s for %s",
                        type(reader).__name__, fmt.value,
                    )
                    readers[i] = reader
                    break
            else:
                readers.append(reader)

    def readers_for(self, fmt: KnowledgeFormat) -> tuple[FormatReader, ...]:
        """All readers registered for ``fmt`` in registration order."""
        return tuple(self._by_format.get(fmt, ()))

    def reader_for(self, fmt: KnowledgeFormat) -> FormatReader | None:
        """The most-recently-registered reader for ``fmt`` (legacy
        single-reader contract). Returns ``None`` when no reader is
        registered. New code should use :meth:`readers_for`."""
        readers = self._by_format.get(fmt)
        if not readers:
            return None
        return readers[-1]

    def formats(self) -> Iterable[KnowledgeFormat]:
        return tuple(self._by_format.keys())


__all__ = (
    "FormatReader",
    "FormatReaderError",
    "PdfTooManyPagesError",
    "ReaderRegistry",
)
