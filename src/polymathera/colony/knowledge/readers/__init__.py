"""Format readers + the registry that resolves them.

Public API:

- ``FormatReader`` (ABC), ``FormatReaderError``, ``ReaderRegistry``.
- Concrete readers — ``PlainTextReader``, ``MarkdownReader``,
  ``HtmlReader``, ``JsonlReader``, ``CsvReader``, ``SourceCodeReader``,
  ``JupyterReader``, ``PdfReader``, ``GrobidPdfReader``.
- ``default_registry()`` — convenience that constructs a registry
  with every in-process reader pre-registered.
- ``default_registry_with_grobid(grobid_url=...)`` — same plus the
  GROBID-backed PDF reader (master §6.3 canonical path).
"""

from __future__ import annotations

from .base import FormatReader, FormatReaderError, ReaderRegistry
from .csv import CsvReader
from .grobid_pdf import GrobidPdfReader
from .html import HtmlReader
from .jsonl import JsonlReader
from .jupyter import JupyterReader
from .markdown import MarkdownReader
from .pdf import PdfReader
from .plain_text import PlainTextReader
from .source_code import SourceCodeReader


def default_registry() -> ReaderRegistry:
    """Return a ``ReaderRegistry`` with every in-process reader
    registered. The PDF reader is included; it raises a clear
    ``FormatReaderError`` if ``pypdf`` is not installed."""

    registry = ReaderRegistry()
    for reader in (
        PlainTextReader(),
        MarkdownReader(),
        HtmlReader(),
        JsonlReader(),
        CsvReader(),
        SourceCodeReader(),
        JupyterReader(),
        PdfReader(),
    ):
        registry.register(reader)
    return registry


def default_registry_with_grobid(
    *, grobid_url: str, timeout_s: float = 60.0,
) -> ReaderRegistry:
    """Return a ``ReaderRegistry`` like ``default_registry`` but with
    ``GrobidPdfReader`` shadowing ``PdfReader`` for the PDF format.

    Use this when a GROBID service is available (master §6.3 — the
    canonical path). The pypdf fallback remains available via the
    plain ``default_registry`` helper.
    """

    registry = ReaderRegistry()
    for reader in (
        PlainTextReader(),
        MarkdownReader(),
        HtmlReader(),
        JsonlReader(),
        CsvReader(),
        SourceCodeReader(),
        JupyterReader(),
        # ReaderRegistry is last-write-wins for a given KnowledgeFormat,
        # so registering PdfReader before GrobidPdfReader leaves the
        # GROBID-backed reader as the active one for PDFs (the pypdf
        # fallback stays importable for direct use).
        PdfReader(),
        GrobidPdfReader(base_url=grobid_url, timeout_s=timeout_s),
    ):
        registry.register(reader)
    return registry


__all__ = (
    "FormatReader",
    "FormatReaderError",
    "ReaderRegistry",
    "PlainTextReader",
    "MarkdownReader",
    "HtmlReader",
    "JsonlReader",
    "CsvReader",
    "SourceCodeReader",
    "JupyterReader",
    "PdfReader",
    "GrobidPdfReader",
    "default_registry",
    "default_registry_with_grobid",
)
