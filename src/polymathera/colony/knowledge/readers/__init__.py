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
from .base import (
    FormatReader,
    FormatReaderError,
    PdfTooManyPagesError,
    ReaderRegistry,
)
from .csv import CsvReader
from .fallback import FallbackPdfReader
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


def _build_pdf_reader(
    *,
    backend: PdfExtractorBackend,
    image_store: Any,
    backend_kwargs: dict[str, Any] | None = None,
) -> FormatReader:
    """Construct the layout-aware multimodal PDF reader for one
    backend. Shared by :func:`default_registry_with_pdf_extractor` and
    by the fallback-reader wrapping path so the per-backend
    constructor logic lives in one place."""

    backend_kwargs = dict(backend_kwargs or {})

    if backend == "mistral_ocr":
        return MistralOcrPdfReader(
            image_store=image_store, **backend_kwargs,
        )
    if backend == "anthropic":
        # Anthropic PDF support reads the document natively via the
        # Messages API document content block. The reader passes the
        # image_store through for symmetry with the other multimodal
        # readers but does not call ``put`` on it — Anthropic returns
        # text only.
        return AnthropicPdfReader(
            image_store=image_store, **backend_kwargs,
        )
    if backend == "gemini":
        # Same shape as the Anthropic reader — Gemini returns text
        # only; the ``model`` kwarg is the cost / quality tier knob
        # (``gemini-2.5-flash`` ≈ $0.003/page; ``gemini-2.5-pro`` ≈
        # $0.010/page; etc.).
        return GeminiPdfReader(
            image_store=image_store, **backend_kwargs,
        )
    if backend == "llamaparse":
        # LlamaParse is asynchronous (upload → poll → result) and
        # returns image bytes via presigned URLs. The ``tier`` kwarg
        # is the quality / cost knob (``fast`` / ``cost_effective``
        # / ``agentic`` / ``agentic_plus``).
        return LlamaParsePdfReader(
            image_store=image_store, **backend_kwargs,
        )
    if backend in ("marker", "docling", "mineru"):
        # Self-hosted backends — the reader is the same generic
        # ``RemotePdfExtractorReader`` for all three; the
        # specialisation lives in the deployment class. The
        # operator must have deployed the corresponding
        # ``*ExtractorDeployment`` (this happens at cluster
        # bring-up via ``add_deployments_to_app``); the reader
        # resolves the handle on first use.
        return RemotePdfExtractorReader(
            backend=backend, image_store=image_store, **backend_kwargs,
        )
    raise ValueError(
        f"Unknown pdf_extractor backend {backend!r}; choose one of "
        f"mistral_ocr / anthropic / gemini / llamaparse / "
        f"marker / docling / mineru.",
    )


def default_registry_with_pdf_extractor(
    *,
    backend: PdfExtractorBackend,
    image_store: Any,
    backend_kwargs: dict[str, Any] | None = None,
    fallback_backend: PdfExtractorBackend | None = None,
    fallback_kwargs: dict[str, Any] | None = None,
    grobid_url: str | None = None,
    grobid_timeout_s: float = 60.0,
) -> ReaderRegistry:
    """Return a ``ReaderRegistry`` whose PDF readers cover the
    layout-aware multimodal body extractor and (optionally) a
    metadata-only GROBID sibling.

    Args:
        backend: Body extractor for ``KnowledgeFormat.PDF``.
        image_store: Active :class:`ImageStore` — readers REQUIRE one
            to land figure bytes; ``set_knowledge_deps`` resolves it
            from ``polymathera_cluster.knowledge.image_dir``.
        backend_kwargs: Per-backend constructor kwargs (forwarded
            verbatim — ``model`` / ``tier`` / ``api_base`` / etc.).
        fallback_backend: Optional backend to route to when
            ``backend`` raises :class:`PdfTooManyPagesError`. Wraps
            the body reader in :class:`FallbackPdfReader`. The
            fallback's ``image_store`` is the same instance as the
            primary's so figures land in one place regardless of
            which backend handled the document.
        fallback_kwargs: Per-backend kwargs for the fallback reader.
        grobid_url: When set, register a metadata-only GROBID reader
            (:class:`GrobidMetadataReader`) ALONGSIDE the body
            extractor. Both run on every PDF; sections concatenate
            (the multi-reader contract on :class:`ReaderRegistry`).
            Provides citation-graph + author / affiliation data on
            top of the layout-aware body chunks.
        grobid_timeout_s: HTTP timeout passed to
            :class:`GrobidMetadataReader`.

    Returns:
        A registry with the standard non-PDF readers plus, for
        ``KnowledgeFormat.PDF``, the body extractor (optionally
        fallback-wrapped) and the optional GROBID metadata sibling.
    """

    pdf_reader: FormatReader = _build_pdf_reader(
        backend=backend, image_store=image_store, backend_kwargs=backend_kwargs,
    )

    if fallback_backend is not None:
        fallback_reader = _build_pdf_reader(
            backend=fallback_backend,
            image_store=image_store,
            backend_kwargs=fallback_kwargs,
        )
        pdf_reader = FallbackPdfReader(primary=pdf_reader, fallback=fallback_reader)

    registry = ReaderRegistry()
    for reader in (
        PlainTextReader(),
        MarkdownReader(),
        HtmlReader(),
        JsonlReader(),
        CsvReader(),
        SourceCodeReader(),
        JupyterReader(),
        # PdfReader is the in-process pypdf fallback; it stays
        # registered for the no-multimodal case but is shadowed by
        # the layout-aware reader below for the same format.
        PdfReader(),
        pdf_reader,
    ):
        registry.register(reader)

    # GROBID metadata-only sibling. Runs alongside the body reader on
    # every PDF (multi-reader registry); the Ingestor concatenates
    # their sections. Each ``ParsedSection`` carries its own
    # ``metadata_origin`` so downstream consumers can distinguish
    # body chunks from the bibliographic envelope.
    if grobid_url:
        registry.register(
            GrobidMetadataReader(base_url=grobid_url, timeout_s=grobid_timeout_s),
        )

    return registry


__all__ = (
    "FallbackPdfReader",
    "FormatReader",
    "FormatReaderError",
    "PdfTooManyPagesError",
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
