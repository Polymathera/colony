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
from typing import ClassVar

from ..models import KnowledgeFormat, ParsedSection, RawDocument


logger = logging.getLogger(__name__)


class FormatReaderError(RuntimeError):
    """Raised when a reader cannot parse its input."""


class FormatReader(ABC):
    """Read a ``RawDocument`` into a sequence of ``ParsedSection``s."""

    handles: ClassVar[tuple[KnowledgeFormat, ...]] = ()
    """Formats this reader handles. Subclasses set this; the registry
    indexes on it."""

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
    """Registry that picks a reader for a detected format."""

    def __init__(self) -> None:
        self._by_format: dict[KnowledgeFormat, FormatReader] = {}

    def register(self, reader: FormatReader) -> None:
        if not reader.handles:
            raise ValueError(
                f"Reader {type(reader).__name__} declares no handled formats.",
            )
        for fmt in reader.handles:
            existing = self._by_format.get(fmt)
            if existing is not None and type(existing) is not type(reader):
                logger.info(
                    "ReaderRegistry: replacing %s for %s with %s",
                    type(existing).__name__, fmt.value, type(reader).__name__,
                )
            self._by_format[fmt] = reader

    def reader_for(self, fmt: KnowledgeFormat) -> FormatReader | None:
        return self._by_format.get(fmt)

    def formats(self) -> Iterable[KnowledgeFormat]:
        return tuple(self._by_format.keys())


__all__ = ("FormatReader", "FormatReaderError", "ReaderRegistry")
