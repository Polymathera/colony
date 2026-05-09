"""``LlamaParsePdfReader`` — PDF reader backed by LlamaCloud's LlamaParse v2.

LlamaParse v2 (`API reference
<https://developers.llamaindex.ai/llamaparse/parse/guides/api-reference/>`_)
is the layout-aware extractor in the LlamaCloud RAG stack. The v2
API is fully **tier-based** — the operator picks one of four
quality / cost points up-front:

- ``fast`` — pure text extraction, no AI, cheapest.
- ``cost_effective`` — LLM-assisted layout reasoning, RAG-tuned
  Markdown. The recommended balanced default.
- ``agentic`` — Sonnet-class agent walks the document, best
  semantic structure preservation. ~10× the ``fast`` cost.
- ``agentic_plus`` — highest fidelity, very expensive. Reserve for
  documents that genuinely need it.

Unlike Mistral OCR's single-shot endpoint, LlamaParse is
**asynchronous**: upload → poll job status → fetch result. The
reader hides the polling behind ``read_async`` so the caller sees
the same ``Sequence[ParsedSection]`` shape as every other reader.

Images come back as **presigned S3 URLs**, not base64 — the reader
fetches each one, lands the bytes in the active
:class:`ImageStore`, and rewrites markdown image references to
``colony-image://<sha>`` so the chunker / KB tab resolve them
through the standard pipeline.

The reader is **picklable**: holds only configuration plus the
:class:`ImageStore`. ``httpx.AsyncClient`` is constructed inside
``read_async`` per request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

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


_DEFAULT_API_BASE = "https://api.cloud.llamaindex.ai/api/v2"
_DEFAULT_TIMEOUT_S = 300.0
"""Generous default for LlamaParse — agentic-tier jobs on a 30-page
paper can take 5–10 minutes. The reader caps both the per-HTTP
timeout and the overall poll budget at this value."""

_DEFAULT_POLL_INTERVAL_S = 2.0
"""Job-status polling cadence. Two seconds is enough to feel
responsive in the dashboard without hammering the LlamaCloud API."""

_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


LlamaParseTier = Literal["fast", "cost_effective", "agentic", "agentic_plus"]
"""Maps directly to LlamaParse's ``configuration.tier`` field. Each
tier is a different cost / quality point — see class docstring."""


_MD_IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
"""Same shape as :class:`MistralOcrPdfReader` uses — markdown image
syntax. We rewrite the URL component when it matches a known
LlamaParse filename; alt-text is preserved."""


class LlamaParsePdfReader(FormatReader):
    """PDF reader that calls LlamaParse v2.

    Args:
        image_store: REQUIRED — figure bytes have nowhere to land
            otherwise. The registry factory wires this from the
            colony's :class:`RetrievalDeps` automatically.
        api_key: ``None`` (default) reads ``LLAMA_CLOUD_API_KEY``
            at call time. Picklable across nodes.
        api_base: Override for non-default deployments
            (LlamaCloud Enterprise, on-prem).
        tier: Quality / cost knob. Default ``cost_effective``.
        timeout_s: Per-HTTP-call timeout AND overall poll budget.
            Agentic-tier jobs can take minutes; the default 300 s
            covers typical papers.
        poll_interval_s: How often to re-check job status.
        download_images: When True (default), figures are fetched
            from their presigned URLs and stored. When False the
            reader skips the image-fetch step and emits an empty
            ``figures`` tuple — useful for the ``fast`` tier where
            no images are extracted anyway.
    """

    handles = (KnowledgeFormat.PDF,)

    def __init__(
        self,
        *,
        image_store: ImageStore,
        api_key: str | None = None,
        api_base: str = _DEFAULT_API_BASE,
        tier: LlamaParseTier = "cost_effective",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        download_images: bool = True,
    ) -> None:
        if image_store is None:
            raise ValueError(
                "LlamaParsePdfReader requires an image_store — figure "
                "bytes have nowhere to land otherwise.",
            )
        self._image_store = image_store
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._tier: LlamaParseTier = tier
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._download_images = bool(download_images)

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def tier(self) -> LlamaParseTier:
        return self._tier

    # ----- FormatReader contract --------------------------------------

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        return asyncio.run(self.read_async(document))

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise FormatReaderError(
                "LlamaParsePdfReader requires the 'httpx' package; install via "
                "`pip install polymathera-colony[knowledge]`.",
            ) from exc

        if document.is_text:
            raise FormatReaderError(
                f"LlamaParsePdfReader expected bytes for "
                f"{document.source_uri}; got text.",
            )

        api_key = self._resolved_api_key()
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            job_id = await self._upload(client, headers, document)
            await self._wait_for_job(client, headers, job_id)
            result = await self._fetch_result(client, headers, job_id)
            id_to_uri = await self._download_figures(
                client, result, document.source_uri,
            )

        return self._sections_from_result(
            result, id_to_uri, document.source_uri,
        )

    # ----- Internals ---------------------------------------------------

    def _resolved_api_key(self) -> str:
        api_key = self._api_key or os.environ.get("LLAMA_CLOUD_API_KEY") or ""
        if not api_key:
            raise FormatReaderError(
                "LlamaParsePdfReader: no API key. Set LLAMA_CLOUD_API_KEY "
                "in the environment or pass api_key= explicitly.",
            )
        return api_key

    async def _upload(
        self, client: Any, headers: dict[str, str], document: RawDocument,
    ) -> str:
        """``POST /v2/parse/upload`` with PDF bytes inline + tier
        configuration. Returns the LlamaParse job id."""
        filename = Path(document.source_uri).name or "document.pdf"
        configuration = json.dumps({
            "tier": self._tier,
            "version": "latest",
        })
        response = await client.post(
            f"{self._api_base}/parse/upload",
            headers=headers,
            files={"file": (filename, document.bytes_, "application/pdf")},
            data={"configuration": configuration},
        )
        if response.status_code not in (200, 201):
            raise FormatReaderError(
                f"LlamaParse /parse/upload returned HTTP "
                f"{response.status_code} for {document.source_uri}: "
                f"{response.text[:512]!r}",
            )
        body = response.json()
        # The v2 response wraps the job under either ``id`` (job-only)
        # or ``job.id`` (full envelope). Tolerate both.
        job_id = body.get("id") or (body.get("job") or {}).get("id")
        if not isinstance(job_id, str) or not job_id:
            raise FormatReaderError(
                f"LlamaParse /parse/upload response missing job id for "
                f"{document.source_uri}: {body!r}",
            )
        return job_id

    async def _wait_for_job(
        self, client: Any, headers: dict[str, str], job_id: str,
    ) -> None:
        """Poll ``GET /v2/parse/{job_id}`` until status is terminal.

        Bounded by the reader's ``timeout_s`` — gives up cleanly
        rather than hanging if LlamaParse's queue is backed up.
        """
        deadline = asyncio.get_event_loop().time() + self._timeout_s
        while True:
            response = await client.get(
                f"{self._api_base}/parse/{job_id}",
                headers=headers,
            )
            if response.status_code != 200:
                raise FormatReaderError(
                    f"LlamaParse /parse/{job_id} status check returned "
                    f"HTTP {response.status_code}: {response.text[:512]!r}",
                )
            body = response.json()
            # The v2 schema has ``status`` either at the root or
            # nested under ``job``. Tolerate both.
            status = body.get("status") or (body.get("job") or {}).get("status")
            if status in _TERMINAL_STATUSES:
                if status != "COMPLETED":
                    error = (
                        body.get("error_message")
                        or (body.get("job") or {}).get("error_message")
                        or status
                    )
                    raise FormatReaderError(
                        f"LlamaParse job {job_id} ended with status "
                        f"{status!r}: {error}",
                    )
                return

            if asyncio.get_event_loop().time() > deadline:
                raise FormatReaderError(
                    f"LlamaParse job {job_id} did not complete within "
                    f"{self._timeout_s} s (last status={status!r}).",
                )
            await asyncio.sleep(self._poll_interval_s)

    async def _fetch_result(
        self, client: Any, headers: dict[str, str], job_id: str,
    ) -> dict[str, Any]:
        """``GET /v2/parse/{job_id}?expand=markdown,images_content_metadata,metadata``.

        Returns the full result envelope. ``expand`` is comma-
        separated; the reader asks for everything it might need so a
        single round trip is enough. ``images_content_metadata`` is
        skipped when ``download_images`` is False — saves a
        round-trip's worth of marshalling on the LlamaCloud side.
        """
        expand_fields = ["markdown", "metadata"]
        if self._download_images:
            expand_fields.append("images_content_metadata")
        params = {"expand": ",".join(expand_fields)}
        response = await client.get(
            f"{self._api_base}/parse/{job_id}",
            headers=headers,
            params=params,
        )
        if response.status_code != 200:
            raise FormatReaderError(
                f"LlamaParse result fetch for job {job_id} returned HTTP "
                f"{response.status_code}: {response.text[:512]!r}",
            )
        return response.json()

    async def _download_figures(
        self,
        client: Any,
        result: dict[str, Any],
        source_uri: str,
    ) -> dict[str, str]:
        """Walk ``result.images_content_metadata.images`` and store
        each presigned-URL-fetched image in the :class:`ImageStore`.

        Returns a ``{filename: colony-image-uri}`` map the markdown
        rewriter uses to swap in our content-addressed URIs.

        Failed downloads are logged and skipped — better to land
        the textual content than fail the whole document on a
        presigned-URL hiccup.
        """
        if not self._download_images:
            return {}

        images_meta = (
            result.get("images_content_metadata") or {}
        ).get("images") or ()
        id_to_uri: dict[str, str] = {}
        for image in images_meta:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            presigned = image.get("presigned_url")
            mime = image.get("content_type") or "image/png"
            if not filename or not presigned:
                continue
            try:
                resp = await client.get(presigned)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LlamaParsePdfReader: failed to fetch %s for %s "
                    "(%s); skipping",
                    filename, source_uri, exc,
                )
                continue
            if resp.status_code != 200:
                logger.warning(
                    "LlamaParsePdfReader: presigned URL for %s returned "
                    "HTTP %d; skipping", filename, resp.status_code,
                )
                continue
            uri = await self._image_store.put(resp.content, mime=mime)
            id_to_uri[filename] = uri
        return id_to_uri

    def _sections_from_result(
        self,
        result: dict[str, Any],
        id_to_uri: dict[str, str],
        source_uri: str,
    ) -> Sequence[ParsedSection]:
        """Walk the per-page markdown returned under ``markdown``
        and emit one :class:`ParsedSection` per page.

        ``markdown`` is a list of objects ``{page: int,
        markdown: str}``. If the upstream returns the older
        flat-string shape (single ``markdown_full`` field), fall
        back to a single section.
        """
        common_extra: dict[str, Any] = {
            "metadata_origin": "llamaparse",
            "tier": self._tier,
        }

        markdown_field = result.get("markdown")
        sections: list[ParsedSection] = []
        char_cursor = 0

        # Shape A: per-page list of {page, markdown}.
        if isinstance(markdown_field, list):
            for entry in markdown_field:
                if not isinstance(entry, dict):
                    continue
                raw_page = entry.get("page")
                page_no = (
                    int(raw_page) if isinstance(raw_page, (int, float)) else None
                )
                md = str(entry.get("markdown") or "")
                section = self._build_section(
                    md, id_to_uri, source_uri, char_cursor,
                    page_no=page_no, common_extra=common_extra,
                )
                if section is not None:
                    sections.append(section)
                    char_cursor += len(section.text)
            return tuple(sections)

        # Shape B: flat ``markdown_full`` string.
        flat = result.get("markdown_full") or (
            markdown_field if isinstance(markdown_field, str) else ""
        )
        section = self._build_section(
            str(flat or ""),
            id_to_uri,
            source_uri,
            0,
            page_no=None,
            common_extra=common_extra,
        )
        if section is None:
            return ()
        return (section,)

    def _build_section(
        self,
        markdown: str,
        id_to_uri: dict[str, str],
        source_uri: str,
        char_cursor: int,
        *,
        page_no: int | None,
        common_extra: dict[str, Any],
    ) -> ParsedSection | None:
        rewritten = self._rewrite_markdown_image_refs(markdown, id_to_uri)
        text = rewritten.strip()
        if not text:
            return None
        # Collect figure refs for THIS section by scanning rewritten
        # markdown for our image-uri scheme. Cross-section figures
        # are reported per-section so the chunker's existing
        # figure-id propagation logic works unchanged.
        figures: list[FigureRef] = []
        seen_uris: set[str] = set()
        for match in _MD_IMAGE_REF_RE.finditer(rewritten):
            url = match.group(2)
            if url.startswith("colony-image://") and url not in seen_uris:
                seen_uris.add(url)
                figures.append(
                    FigureRef(
                        image_uri=url,
                        page=page_no,
                        kind="figure",
                    )
                )
        section_path = (
            f"page-{page_no}" if page_no is not None else "document"
        )
        return ParsedSection(
            section_path=section_path,
            text=text,
            citation=CitationSpan(
                source_uri=source_uri,
                section_path=section_path,
                char_start=char_cursor,
                char_end=char_cursor + len(text),
                page_number=page_no,
            ),
            figures=tuple(figures),
            format="markdown",
            extra=common_extra,
        )

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


__all__ = ("LlamaParsePdfReader", "LlamaParseTier")
