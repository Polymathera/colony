"""``RemotePdfExtractorReader`` — generic reader over any
:class:`~polymathera.colony.cluster.extractors.PdfExtractorDeployment`.

One reader class that talks to a Ray serving deployment (Marker,
Docling, or MinerU) via its handle and converts the resulting
:class:`ExtractResult` into the canonical
``Sequence[ParsedSection]`` shape every other reader emits. Replaces
what would have been three near-identical reader subclasses; the
deployment IS the per-backend specialisation and the reader is the
backend-agnostic glue.

Lifecycle:

1. Reader is constructed with a backend name (used to resolve the
   :class:`DeploymentHandle` lazily on first call) and the active
   :class:`ImageStore`.
2. On ``read_async``, the reader calls ``handle.extract(pdf_bytes,
   options)`` over the Ray bus.
3. Each :class:`FigureBlob` lands in the image store; markdown
   image references keyed by ``blob_id`` are rewritten to
   ``colony-image://<sha>``; pages are split on ``<!-- page: N -->``
   markers (when present) into one :class:`ParsedSection` each.

The handle resolution is deferred to call time so the reader is
**picklable** — the dashboard process can ship the reader inside a
capability blueprint without already holding the cluster's serving
handle, and each Ray worker resolves its own.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from ...cluster.extractors import (
    ExtractOptions,
    ExtractResult,
    FigureBlob,
)
from ..models import (
    CitationSpan,
    FigureRef,
    KnowledgeFormat,
    ParsedSection,
    RawDocument,
)
from ..stores.image import ImageStore
from .base import FormatReader, FormatReaderError


logger = logging.getLogger(__name__)


_PAGE_MARKER_RE = re.compile(
    r"^<!--\s*page\s*:\s*(?P<page>\d+)\s*-->\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_MD_IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class RemotePdfExtractorReader(FormatReader):
    """PDF reader that delegates to a self-hosted extractor
    deployment over Ray's serving handle.

    Args:
        backend: Backend name (``"marker"``, ``"docling"``,
            ``"mineru"``). Used to resolve the deployment handle at
            call time via the colony's serving registry.
        image_store: REQUIRED — figure bytes have nowhere to land
            otherwise.
        app_name: Application name override. ``None`` resolves via
            the colony's default (the agent system's app).
        deployment_name: ``None`` defaults to the deployment class
            name (e.g. ``"MarkerExtractorDeployment"``).
        extract_options: Options forwarded to every extract call —
            controls table format, image extraction, page filter.
    """

    handles = (KnowledgeFormat.PDF,)

    _BACKEND_TO_DEPLOYMENT_NAME: dict[str, str] = {
        "marker": "MarkerExtractorDeployment",
        "docling": "DoclingExtractorDeployment",
        "mineru": "MinerUExtractorDeployment",
    }

    def __init__(
        self,
        *,
        backend: str,
        image_store: ImageStore,
        app_name: str | None = None,
        deployment_name: str | None = None,
        extract_options: ExtractOptions | None = None,
    ) -> None:
        if image_store is None:
            raise ValueError(
                "RemotePdfExtractorReader requires an image_store.",
            )
        if backend not in self._BACKEND_TO_DEPLOYMENT_NAME:
            raise ValueError(
                f"unknown self-hosted backend {backend!r}; choose one of "
                f"{sorted(self._BACKEND_TO_DEPLOYMENT_NAME)}",
            )
        self._backend = backend
        self._image_store = image_store
        self._app_name = app_name
        self._deployment_name = (
            deployment_name or self._BACKEND_TO_DEPLOYMENT_NAME[backend]
        )
        self._extract_options = extract_options or ExtractOptions()

    @property
    def backend(self) -> str:
        return self._backend

    # ----- FormatReader contract --------------------------------------

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        import asyncio
        return asyncio.run(self.read_async(document))

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        if document.is_text:
            raise FormatReaderError(
                f"RemotePdfExtractorReader expected bytes for "
                f"{document.source_uri}; got text.",
            )

        handle = await self._resolve_handle()
        try:
            result: ExtractResult = await handle.extract(
                pdf_bytes=document.bytes_, options=self._extract_options,
            )
        except Exception as exc:  # noqa: BLE001
            raise FormatReaderError(
                f"RemotePdfExtractorReader[{self._backend}] failed for "
                f"{document.source_uri}: {type(exc).__name__}: {exc}",
            ) from exc

        # The deployment may return a dict-shape (when serialisation
        # round-trips through the wire) or the typed model. Coerce
        # to the model so we have a stable surface.
        if isinstance(result, dict):
            result = ExtractResult.model_validate(result)

        id_to_uri = await self._store_figures(result.figures)
        return self._sections_from_extract(
            result, id_to_uri, document.source_uri,
        )

    # ----- Internals ---------------------------------------------------

    async def _resolve_handle(self) -> Any:
        """Resolve the deployment handle through the colony's
        serving registry. Imported lazily so unit tests can stub
        :meth:`_resolve_handle` without touching Ray."""
        try:
            from ...distributed.ray_utils import serving
        except ImportError as exc:  # pragma: no cover
            raise FormatReaderError(
                "RemotePdfExtractorReader: serving runtime unavailable.",
            ) from exc
        return await serving.get_deployment(
            app_name=self._app_name,
            deployment_name=self._deployment_name,
        )

    async def _store_figures(
        self, figures: tuple[FigureBlob, ...],
    ) -> dict[str, str]:
        """Land each :class:`FigureBlob`'s bytes in the image store
        and return a ``{blob_id: colony-image-uri}`` map for markdown
        rewriting."""
        id_to_uri: dict[str, str] = {}
        for blob in figures:
            if not isinstance(blob, FigureBlob):
                # Tolerant of dict-shaped wire output.
                blob = FigureBlob.model_validate(blob)
            if not blob.image_bytes:
                continue
            uri = await self._image_store.put(blob.image_bytes, mime=blob.mime)
            id_to_uri[blob.blob_id] = uri
        return id_to_uri

    def _sections_from_extract(
        self,
        result: ExtractResult,
        id_to_uri: dict[str, str],
        source_uri: str,
    ) -> Sequence[ParsedSection]:
        """Split the extract's markdown on ``<!-- page: N -->`` and
        emit one :class:`ParsedSection` per page (or one section
        when no markers are present)."""
        markdown = self._rewrite_markdown_image_refs(
            result.markdown, id_to_uri,
        )
        common_extra: dict[str, Any] = {
            "metadata_origin": result.backend or self._backend,
        }
        if result.extra:
            common_extra["extractor_extra"] = result.extra

        # Build per-blob FigureRef list once; each section claims
        # the figures whose URI appears in its rewritten text.
        all_refs: list[FigureRef] = []
        uri_to_ref: dict[str, FigureRef] = {}
        for blob_id, uri in id_to_uri.items():
            ref = FigureRef(image_uri=uri, label=blob_id, kind="figure")
            all_refs.append(ref)
            uri_to_ref[uri] = ref

        parts = _PAGE_MARKER_RE.split(markdown)

        sections: list[ParsedSection] = []
        char_cursor = 0

        if len(parts) == 1:
            text = parts[0].strip()
            if not text:
                return ()
            section_figures = self._figures_for_text(text, uri_to_ref)
            return (
                ParsedSection(
                    section_path="document",
                    text=text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path="document",
                        char_start=0,
                        char_end=len(text),
                        page_number=None,
                    ),
                    figures=tuple(section_figures),
                    format="markdown",
                    extra=common_extra,
                ),
            )

        prefix_text = parts[0].strip()
        if prefix_text:
            sections.append(
                ParsedSection(
                    section_path="page-prefix",
                    text=prefix_text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path="page-prefix",
                        char_start=char_cursor,
                        char_end=char_cursor + len(prefix_text),
                        page_number=None,
                    ),
                    figures=tuple(
                        self._figures_for_text(prefix_text, uri_to_ref),
                    ),
                    format="markdown",
                    extra=common_extra,
                )
            )
            char_cursor += len(prefix_text)

        for page_str, body in zip(parts[1::2], parts[2::2]):
            text = body.strip()
            if not text:
                continue
            try:
                page_no = int(page_str)
            except (TypeError, ValueError):
                page_no = None
            section_path = (
                f"page-{page_no}" if page_no is not None else "page-?"
            )
            sections.append(
                ParsedSection(
                    section_path=section_path,
                    text=text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path=section_path,
                        char_start=char_cursor,
                        char_end=char_cursor + len(text),
                        page_number=page_no,
                    ),
                    figures=tuple(self._figures_for_text(text, uri_to_ref)),
                    format="markdown",
                    extra=common_extra,
                )
            )
            char_cursor += len(text)

        return tuple(sections)

    @staticmethod
    def _figures_for_text(
        text: str, uri_to_ref: dict[str, FigureRef],
    ) -> list[FigureRef]:
        seen: set[str] = set()
        refs: list[FigureRef] = []
        for match in _MD_IMAGE_REF_RE.finditer(text):
            url = match.group(2)
            ref = uri_to_ref.get(url)
            if ref is not None and ref.image_uri not in seen:
                seen.add(ref.image_uri)
                refs.append(ref)
        return refs

    @staticmethod
    def _rewrite_markdown_image_refs(
        markdown: str, id_to_uri: dict[str, str],
    ) -> str:
        if not id_to_uri or not markdown:
            return markdown

        def _sub(match: re.Match[str]) -> str:
            alt, url = match.group(1), match.group(2)
            new_url = id_to_uri.get(url, url)
            return f"![{alt}]({new_url})"

        return _MD_IMAGE_REF_RE.sub(_sub, markdown)


__all__ = ("RemotePdfExtractorReader",)
