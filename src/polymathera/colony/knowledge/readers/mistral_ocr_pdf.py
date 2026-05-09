"""``MistralOcrPdfReader`` — PDF reader backed by Mistral OCR 3.

Mistral OCR 3 is the cheapest hosted PDF-to-Markdown extractor on
the market today (~$0.001-$0.002 per page, <1 s/page) and the most
direct fit for our :class:`ParsedSection(text=md, figures=(...))`
shape: it returns one ``markdown`` blob per page plus interleaved
image bytes keyed by an extractor-local ID, with markdown image
references already pointing at those keys. The reader's job is to
land each image in the active :class:`ImageStore`, rewrite the
markdown references to ``colony-image://<sha>`` URIs, and emit one
:class:`~polymathera.colony.knowledge.models.ParsedSection` per page.

See :doc:`/architecture/multimodal-pdf-ingestion` (Phase 1+2, row
1.10) for the design context. The reader is registered alongside
:class:`~polymathera.colony.knowledge.readers.grobid_pdf.GrobidPdfReader`
in the registry; the cluster config's ``pdf_extractor.backend``
field selects which one wins for ``KnowledgeFormat.PDF``.

Wire flow per source:

1. ``POST /v1/files``           — upload PDF bytes (multipart, ``purpose=ocr``).
2. ``GET /v1/files/{id}/url``   — fetch a signed URL Mistral can read.
3. ``POST /v1/ocr``             — submit the signed URL,
                                  ``include_image_base64=true``.
4. Per page in the response: store images via
   :meth:`ImageStore.put`, rewrite markdown refs, build
   :class:`~polymathera.colony.knowledge.models.FigureRef`s.

The reader is **picklable**: it holds only configuration (URL,
api_key, model name, timeouts) and a reference to the
:class:`ImageStore`. ``httpx.AsyncClient`` is constructed
per-request inside the async method so connection state never
crosses the cloudpickle boundary.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


_DEFAULT_API_BASE = "https://api.mistral.ai/v1"
_DEFAULT_MODEL = "mistral-ocr-latest"
_DEFAULT_TIMEOUT_S = 120.0
"""Documented Mistral OCR throughput is ~2,000 pages/min, so 120 s
is generous even for 600-page books. Bumped from the GROBID-style
60 s default since the OCR endpoint orchestrates upload + signed
URL + extract — three round trips serialised."""

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)
"""Mistral wraps image bytes in ``data:image/jpeg;base64,...`` per
RFC 2397. The reader strips the prefix before storing."""

_MD_IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
"""Markdown image syntax. We rewrite the URL component when it
matches a known Mistral image id; alt-text is preserved."""


class MistralOcrPdfReader(FormatReader):
    """PDF reader that calls Mistral OCR 3 via the public REST API.

    Caller-supplied ``RawDocument.payload`` MUST be ``bytes``. The
    reader uploads the bytes via the Files API, requests a signed
    URL, and invokes ``/v1/ocr``. Images come back inline as
    base64-encoded data URIs; the reader stores them in the active
    :class:`ImageStore` and rewrites markdown image references to
    the resulting ``colony-image://`` URIs.

    Args:
        image_store: The active image store. The reader REQUIRES
            one — without it, image bytes have nowhere to land. In
            production this is supplied by the registry factory
            from the colony's :class:`RetrievalDeps`.
        api_key: Mistral API key. ``None`` (the default) reads
            ``MISTRAL_API_KEY`` from the environment at call time
            (NOT at construction — keeps the reader picklable
            across nodes that don't have the env var set yet).
        api_base: Override for non-default deployments (Mistral on
            Azure, on-prem, the staging endpoint).
        model: ``mistral-ocr-latest`` by default. Pin a specific
            version (``mistral-ocr-2503``) for reproducibility.
        timeout_s: Per-HTTP-call timeout. The total per-document
            time budget can be up to 3× this (upload + signed URL +
            extract).
        table_format: How tables are serialised — passed straight
            to the OCR call.
    """

    handles = (KnowledgeFormat.PDF,)

    def __init__(
        self,
        *,
        image_store: ImageStore,
        api_key: str | None = None,
        api_base: str = _DEFAULT_API_BASE,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        table_format: str = "markdown",
    ) -> None:
        if image_store is None:
            raise ValueError(
                "MistralOcrPdfReader requires an image_store — figure "
                "bytes have nowhere to land otherwise.",
            )
        self._image_store = image_store
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._timeout_s = float(timeout_s)
        self._table_format = table_format

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def model(self) -> str:
        return self._model

    # ----- FormatReader contract --------------------------------------

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        """Sync entry point. The reader is async-native (HTTP I/O);
        ``FormatReader.read_async`` defaults to ``asyncio.to_thread``
        which would re-enter an event loop. We expose a sync facade
        for callers outside an event loop via ``asyncio.run``."""

        import asyncio
        return asyncio.run(self.read_async(document))

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - covered by knowledge extra
            raise FormatReaderError(
                "MistralOcrPdfReader requires the 'httpx' package; install via "
                "`pip install polymathera-colony[knowledge]` or `pip install httpx`.",
            ) from exc

        if document.is_text:
            raise FormatReaderError(
                f"MistralOcrPdfReader expected bytes for {document.source_uri}; "
                "got text.",
            )

        api_key = self._resolved_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            file_id = await self._upload(client, headers, document)
            signed_url = await self._signed_url(client, headers, file_id)
            ocr_payload = await self._invoke_ocr(client, headers, signed_url)

        return await self._sections_from_ocr(ocr_payload, document.source_uri)

    # ----- Internals ---------------------------------------------------

    def _resolved_api_key(self) -> str:
        """Resolve the API key at call time so the reader stays
        picklable across nodes that don't have ``MISTRAL_API_KEY``
        set when the blueprint is constructed."""
        api_key = self._api_key or os.environ.get("MISTRAL_API_KEY") or ""
        if not api_key:
            raise FormatReaderError(
                "MistralOcrPdfReader: no API key. Set MISTRAL_API_KEY in the "
                "environment or pass api_key= explicitly.",
            )
        return api_key

    async def _upload(
        self, client: Any, headers: dict[str, str], document: RawDocument,
    ) -> str:
        """Upload PDF bytes to ``POST /v1/files`` with ``purpose=ocr``.

        Returns the file id Mistral assigns. Errors are surfaced as
        :class:`FormatReaderError` with the response body truncated
        to 512 chars — enough to diagnose auth / rate-limit / quota
        failures without spamming the ingestion log.
        """
        filename = Path(document.source_uri).name or "document.pdf"
        # Mistral's Files API takes multipart with ``file`` and
        # ``purpose``. The Python SDK uses ``purpose="ocr"`` for OCR
        # uploads (vs ``purpose="batch"`` for batch jobs).
        response = await client.post(
            f"{self._api_base}/files",
            headers=headers,
            files={"file": (filename, document.bytes_, "application/pdf")},
            data={"purpose": "ocr"},
        )
        if response.status_code != 200:
            raise FormatReaderError(
                f"Mistral /v1/files returned HTTP {response.status_code} for "
                f"{document.source_uri}: {response.text[:512]!r}",
            )
        body = response.json()
        file_id = body.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise FormatReaderError(
                f"Mistral /v1/files response missing 'id' field for "
                f"{document.source_uri}: {body!r}",
            )
        return file_id

    async def _signed_url(
        self, client: Any, headers: dict[str, str], file_id: str,
    ) -> str:
        """Fetch a signed URL Mistral OCR can read.

        Mistral's OCR endpoint accepts ``document_url`` only; for
        files uploaded via the Files API the operator gets a signed
        URL by ``GET /v1/files/{file_id}/url``. The URL is short-
        lived but well within the OCR call's window.
        """
        response = await client.get(
            f"{self._api_base}/files/{file_id}/url",
            headers=headers,
        )
        if response.status_code != 200:
            raise FormatReaderError(
                f"Mistral /v1/files/{file_id}/url returned HTTP "
                f"{response.status_code}: {response.text[:512]!r}",
            )
        body = response.json()
        signed_url = body.get("url")
        if not isinstance(signed_url, str) or not signed_url:
            raise FormatReaderError(
                f"Mistral signed-url response missing 'url' field for "
                f"file_id={file_id}: {body!r}",
            )
        return signed_url

    async def _invoke_ocr(
        self, client: Any, headers: dict[str, str], signed_url: str,
    ) -> dict[str, Any]:
        """POST to ``/v1/ocr`` with ``include_image_base64=true``.

        ``table_format`` is passed straight through; Mistral picks
        per-table whether to use markdown or HTML. ``model`` is
        ``mistral-ocr-latest`` unless the operator pinned a version.
        """
        payload = {
            "model": self._model,
            "document": {
                "type": "document_url",
                "document_url": signed_url,
            },
            "include_image_base64": True,
            "table_format": self._table_format,
        }
        response = await client.post(
            f"{self._api_base}/ocr",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )
        if response.status_code != 200:
            raise FormatReaderError(
                f"Mistral /v1/ocr returned HTTP {response.status_code}: "
                f"{response.text[:512]!r}",
            )
        return response.json()

    async def _sections_from_ocr(
        self, ocr_payload: dict[str, Any], source_uri: str,
    ) -> Sequence[ParsedSection]:
        """Walk ``ocr_payload.pages`` and produce one
        :class:`ParsedSection` per page.

        Per page:
        - Decode each ``images[].image_base64`` data URI
        - Store via :meth:`ImageStore.put` → ``colony-image://<sha>``
        - Build a :class:`FigureRef` per stored image
        - Rewrite ``![label](mistral-id)`` markdown refs to point at
          ``colony-image://<sha>``

        ``page.index`` is 0-indexed in Mistral's response; the
        :class:`CitationSpan` field uses 1-indexed page numbers
        (matching every other reader and the dashboard's display
        convention).
        """
        pages = ocr_payload.get("pages") or ()
        if not isinstance(pages, list):
            raise FormatReaderError(
                f"Mistral OCR response for {source_uri} has malformed "
                f"'pages' field (got {type(pages).__name__}).",
            )

        sections: list[ParsedSection] = []
        char_cursor = 0  # global char offset across the document

        for page in pages:
            if not isinstance(page, dict):
                logger.warning(
                    "MistralOcrPdfReader: skipping non-dict page entry for %s: %r",
                    source_uri, page,
                )
                continue

            raw_index = page.get("index", 0)
            page_no_zero = int(raw_index) if isinstance(raw_index, (int, float)) else 0
            page_no = page_no_zero + 1
            md = str(page.get("markdown") or "")

            # Resolve images first — we need the (mistral_id → uri,
            # figure_id) map before rewriting the markdown.
            id_to_uri: dict[str, str] = {}
            id_to_figure_id: dict[str, str] = {}
            figure_refs: list[FigureRef] = []

            for image in (page.get("images") or ()):
                if not isinstance(image, dict):
                    continue
                blob = self._decode_data_uri(image.get("image_base64"))
                if blob is None:
                    continue
                payload, mime = blob
                uri = await self._image_store.put(payload, mime=mime)
                bbox = self._bbox_from_mistral(image)
                ref = FigureRef(
                    image_uri=uri,
                    page=page_no,
                    bbox=bbox,
                    caption_hint="",
                    kind="figure",
                    label=str(image.get("id") or ""),
                )
                figure_refs.append(ref)
                if image.get("id"):
                    id_to_uri[str(image["id"])] = uri
                    id_to_figure_id[str(image["id"])] = ref.figure_id

            rewritten_md = self._rewrite_markdown_image_refs(md, id_to_uri)

            section = ParsedSection(
                section_path=f"page-{page_no}",
                heading="",
                text=rewritten_md,
                citation=CitationSpan(
                    source_uri=source_uri,
                    section_path=f"page-{page_no}",
                    char_start=char_cursor,
                    char_end=char_cursor + len(rewritten_md),
                    page_number=page_no,
                ),
                figures=tuple(figure_refs),
                format="markdown",
                extra={
                    "metadata_origin": "mistral_ocr",
                    "model": self._model,
                    # Map preserved so the chunker can resolve in-text
                    # references like "Fig. 3" or "img-0.jpeg" back to
                    # a figure_id when the chunk text mentions them.
                    "figure_label_to_id": id_to_figure_id,
                },
            )
            sections.append(section)
            char_cursor += len(rewritten_md)

        return tuple(sections)

    @staticmethod
    def _decode_data_uri(value: Any) -> tuple[bytes, str] | None:
        """Decode a ``data:image/...;base64,...`` URI to ``(bytes,
        mime)``. Plain base64 (no data-uri wrapper) is also
        tolerated and assumed to be PNG.

        Returns ``None`` for malformed / empty values; the caller
        skips the figure rather than failing the whole page.
        """
        if not isinstance(value, str) or not value:
            return None
        match = _DATA_URI_RE.match(value)
        if match:
            mime = match.group("mime").strip().lower()
            data_str = match.group("data").strip()
        else:
            # No data-uri prefix; treat as raw base64, default to PNG.
            mime = "image/png"
            data_str = value.strip()
        try:
            raw = base64.b64decode(data_str, validate=False)
        except (ValueError, base64.binascii.Error) as exc:
            logger.warning(
                "MistralOcrPdfReader: failed to b64-decode image (%s); skipping",
                exc,
            )
            return None
        if not raw:
            return None
        return raw, mime

    @staticmethod
    def _bbox_from_mistral(image: dict[str, Any]) -> tuple[float, float, float, float] | None:
        """Convert Mistral's ``top_left_x/y`` + ``bottom_right_x/y``
        to our ``(x0, y0, x1, y1)`` tuple. Returns ``None`` if any
        component is missing — partial bboxes are not useful."""
        try:
            return (
                float(image["top_left_x"]),
                float(image["top_left_y"]),
                float(image["bottom_right_x"]),
                float(image["bottom_right_y"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _rewrite_markdown_image_refs(
        markdown: str, id_to_uri: dict[str, str],
    ) -> str:
        """Replace ``![alt](mistral-id)`` URLs with ``colony-image://``
        URIs. Refs whose URL is not in ``id_to_uri`` (external links,
        already-rewritten URIs) are left untouched.
        """
        if not id_to_uri or not markdown:
            return markdown

        def _sub(match: re.Match[str]) -> str:
            alt, url = match.group(1), match.group(2)
            new_url = id_to_uri.get(url, url)
            return f"![{alt}]({new_url})"

        return _MD_IMAGE_REF_RE.sub(_sub, markdown)


__all__ = ("MistralOcrPdfReader",)
