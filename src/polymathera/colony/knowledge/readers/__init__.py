"""Format readers + the registry that resolves them.

Public API:

- ``FormatReader`` (ABC), ``FormatReaderError``, ``ReaderRegistry``.
- Concrete readers — ``PlainTextReader``, ``MarkdownReader``,
  ``HtmlReader``, ``JsonlReader``, ``CsvReader``, ``SourceCodeReader``,
  ``JupyterReader``, ``PdfReader``, ``GrobidPdfReader``,
  ``GeminiPdfReader``, ``LlamaParsePdfReader``, ``MistralOcrPdfReader``.
- ``default_registry()`` — convenience that constructs a registry
  with every in-process reader pre-registered.
- ``default_registry_with_grobid(grobid_url=...)`` — same plus the
  GROBID-backed PDF reader (legacy / metadata-only canonical path).
- ``default_registry_with_pdf_extractor(backend=..., image_store=...)`` —
  registers a layout-aware multimodal PDF reader (Mistral OCR today;
  Marker / Docling / MinerU when their deployments land). Selected
  via the cluster config's ``knowledge.pdf_extractor.backend`` field.
  See :doc:`/architecture/multimodal-pdf-ingestion`.
"""

from __future__ import annotations

from typing import Any, Literal

from .anthropic_pdf import AnthropicPdfReader
from .base import FormatReader, FormatReaderError, ReaderRegistry
from .csv import CsvReader
from .gemini_pdf import GeminiPdfReader
from .grobid_pdf import GrobidMetadataReader, GrobidPdfReader
from .html import HtmlReader
from .jsonl import JsonlReader
from .jupyter import JupyterReader
from .llamaparse_pdf import LlamaParsePdfReader, LlamaParseTier
from .markdown import MarkdownReader
from .mistral_ocr_pdf import MistralOcrPdfReader
from .pdf import PdfReader
from .plain_text import PlainTextReader
from .remote_pdf import RemotePdfExtractorReader
from .source_code import SourceCodeReader


# Recognised values for the cluster config's
# ``knowledge.pdf_extractor.backend`` field. Three are reserved for
# the self-hosted deployments (planned in design doc §10 rows
# 1.5-1.7); two are live today.
PdfExtractorBackend = Literal[
    "mistral_ocr",
    "anthropic",
    "gemini",
    "llamaparse",
    "marker",
    "docling",
    "mineru",
]


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

    Use this when a GROBID service is available (legacy bibliographic
    canonical path). For new corpora prefer
    :func:`default_registry_with_pdf_extractor` — GROBID's content
    extraction is weak for figures, tables, and equations.
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


def default_registry_with_pdf_extractor(
    *,
    backend: PdfExtractorBackend,
    image_store: Any,
    backend_kwargs: dict[str, Any] | None = None,
) -> ReaderRegistry:
    """Return a ``ReaderRegistry`` whose PDF reader is the layout-aware
    multimodal extractor selected by ``backend``.

    Args:
        backend: Which PDF extractor wins for ``KnowledgeFormat.PDF``.
            Today only ``"mistral_ocr"`` is implemented; the other
            values are reserved for the self-hosted deployments
            (Marker / Docling / MinerU) and the Anthropic native PDF
            reader, all planned in design doc §10. Selecting a
            not-yet-implemented backend raises ``NotImplementedError``
            with a message pointing at the responsible row.
        image_store: The active :class:`ImageStore`. The reader
            REQUIRES one to land figure bytes; the cluster's
            :func:`set_knowledge_deps` resolves the right one
            (in-memory / local-FS) from
            ``polymathera_cluster.knowledge.image_dir``.
        backend_kwargs: Extra constructor kwargs forwarded to the
            selected reader. ``mistral_ocr`` accepts ``api_key``,
            ``api_base``, ``model``, ``timeout_s``, ``table_format``;
            see :class:`MistralOcrPdfReader` for the full list.

    Returns:
        A registry with the standard non-PDF readers + the chosen PDF
        extractor. Reading a non-PDF format goes through the existing
        in-process readers unchanged.
    """

    backend_kwargs = dict(backend_kwargs or {})

    if backend == "mistral_ocr":
        pdf_reader: FormatReader = MistralOcrPdfReader(
            image_store=image_store, **backend_kwargs,
        )
    elif backend == "anthropic":
        # Anthropic PDF support reads the document natively via the
        # Messages API document content block. The reader passes the
        # image_store through for symmetry with the other multimodal
        # readers but does not call ``put`` on it — Anthropic returns
        # text only.
        pdf_reader = AnthropicPdfReader(
            image_store=image_store, **backend_kwargs,
        )
    elif backend == "gemini":
        # Same shape as the Anthropic reader — Gemini returns text
        # only; the ``model`` kwarg is the cost / quality tier knob
        # (``gemini-2.5-flash`` ≈ $0.003/page; ``gemini-2.5-pro`` ≈
        # $0.010/page; etc.).
        pdf_reader = GeminiPdfReader(
            image_store=image_store, **backend_kwargs,
        )
    elif backend == "llamaparse":
        # LlamaParse is asynchronous (upload → poll → result) and
        # returns image bytes via presigned URLs. The ``tier`` kwarg
        # is the quality / cost knob (``fast`` / ``cost_effective``
        # / ``agentic`` / ``agentic_plus``).
        pdf_reader = LlamaParsePdfReader(
            image_store=image_store, **backend_kwargs,
        )
    elif backend in ("marker", "docling", "mineru"):
        # Self-hosted backends — the reader is the same generic
        # ``RemotePdfExtractorReader`` for all three; the
        # specialisation lives in the deployment class. The
        # operator must have deployed the corresponding
        # ``*ExtractorDeployment`` (this happens at cluster
        # bring-up via ``add_deployments_to_app``); the reader
        # resolves the handle on first use.
        pdf_reader = RemotePdfExtractorReader(
            backend=backend, image_store=image_store, **backend_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown pdf_extractor backend {backend!r}; choose one of "
            f"mistral_ocr / anthropic / marker / docling / mineru.",
        )

    registry = ReaderRegistry()
    for reader in (
        PlainTextReader(),
        MarkdownReader(),
        HtmlReader(),
        JsonlReader(),
        CsvReader(),
        SourceCodeReader(),
        JupyterReader(),
        # PdfReader registered first so the layout-aware reader
        # registered next replaces it for ``KnowledgeFormat.PDF`` via
        # the registry's last-write-wins rule. The pypdf fallback
        # stays importable for direct use in tests / one-offs.
        PdfReader(),
        pdf_reader,
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
    "GrobidMetadataReader",
    "MistralOcrPdfReader",
    "AnthropicPdfReader",
    "GeminiPdfReader",
    "LlamaParsePdfReader",
    "LlamaParseTier",
    "RemotePdfExtractorReader",
    "PdfExtractorBackend",
    "default_registry",
    "default_registry_with_grobid",
    "default_registry_with_pdf_extractor",
)
