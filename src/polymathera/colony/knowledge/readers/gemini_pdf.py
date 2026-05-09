"""``GeminiPdfReader`` — PDF reader backed by Google Gemini's native PDF support.

Gemini accepts ``application/pdf`` directly via the
``inline_data`` content part: each page is rasterised and tokenised
at ~258 tokens of input per page, then fed alongside the prompt
(`Gemini PDF docs
<https://ai.google.dev/gemini-api/docs/document-processing>`_). The
reader exploits this with the same "convert this PDF to clean
Markdown with ``<!-- page: N -->`` separators" prompt as
:class:`~polymathera.colony.knowledge.readers.anthropic_pdf.AnthropicPdfReader`,
then parses the response into one
:class:`~polymathera.colony.knowledge.models.ParsedSection` per page.

**Cost / quality tier knob.** ``model`` is the operator's lever:

- ``gemini-2.5-flash`` (default) — fast multimodal, ~$0.003 / page.
  The cheapest path that still does proper visual reasoning. Use
  for the bulk of the corpus.
- ``gemini-2.5-pro`` — premium multimodal, ~$0.010 / page. Pick
  when Flash misses something on a visually-dense doc.
- Any future ``gemini-N-flash-preview`` / ``gemini-N-pro`` slug
  works without code changes — pass it as ``model``.

**No image bytes returned**, same caveat as the Anthropic reader.
``ParsedSection.figures`` is always empty; figures are described
inline in the markdown. Operators who need clickable image
previews use the Mistral OCR or LlamaParse readers.

The reader is **picklable**: it holds only configuration. The
``google.genai.Client`` is constructed inside ``read_async`` so
connection state never crosses the cloudpickle boundary; the API
key is resolved at call time so the blueprint is pickle-clean even
when the worker doesn't have ``GOOGLE_API_KEY`` set yet.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from ..models import (
    CitationSpan,
    KnowledgeFormat,
    ParsedSection,
    RawDocument,
)
from ..stores.image import ImageStore
from .anthropic_pdf import _PAGE_MARKER_RE  # shared regex; same prompt shape
from .base import FormatReader, FormatReaderError


logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
_DEFAULT_TIMEOUT_S = 120.0
_API_KEY_ENV_VAR = "GOOGLE_API_KEY"


# Same template as the Anthropic reader; the page-marker discipline
# is portable across multimodal LLMs and lets the chunker do
# per-page citations without reader-specific parsing logic.
_DEFAULT_PROMPT = """Convert this PDF document to clean Markdown for downstream chunking and retrieval.

Strict requirements:
- Preserve LaTeX math: inline as `$...$`, display as `$$...$$`.
- Render tables as GFM markdown tables. If a table has merged cells the GFM rendering can't express, emit an HTML `<table>` instead.
- For each figure / chart / diagram, emit a brief description on its own line as a markdown image reference whose URL is the placeholder string `figure-N` (where N is the figure number, 1-indexed within the page) and whose alt text is the description, e.g. `![Block diagram of the control loop with PI feedback and anti-windup clamp.](figure-1)`. We do NOT need the image bytes, only the description.
- Preserve heading hierarchy with `#` / `##` / `###`.
- Preserve in-text references like "Fig. 3" or "Table 2" verbatim.
- Insert a literal page-break marker between pages, on its own line: `<!-- page: N -->` (1-indexed). The marker MUST appear immediately before the first content of page N.

Do NOT add commentary, summaries, or anything outside the Markdown body. Return ONLY the Markdown."""


class GeminiPdfReader(FormatReader):
    """PDF reader that calls Gemini's ``generateContent`` with an
    inline PDF part.

    Args:
        image_store: Accepted for symmetry with the other multimodal
            readers but unused — Gemini returns no image bytes.
        api_key: ``None`` (default) reads ``GOOGLE_API_KEY`` at call
            time. Picklable.
        model: Tier knob; see class docstring. Default
            ``gemini-2.5-flash``.
        max_output_tokens: Output budget. Bumped from the Gemini
            default so a long paper round-trips in one call.
        timeout_s: Per-call HTTP timeout.
        prompt: Override the default markdown-conversion prompt.
        cached_content_name: When set, attach an existing
            ``cachedContents`` resource to the request — the
            operator can pre-cache a frequently-re-ingested PDF and
            pay the 90% input discount on subsequent calls. Most
            operators leave this unset; the reader makes a fresh
            call per ingest.
    """

    def __init__(
        self,
        *,
        image_store: ImageStore | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        prompt: str | None = None,
        cached_content_name: str | None = None,
    ) -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        self._image_store = image_store
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = int(max_output_tokens)
        self._timeout_s = float(timeout_s)
        self._prompt = prompt or _DEFAULT_PROMPT
        self._cached_content = cached_content_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def has_image_extraction(self) -> bool:
        return False

    # ----- FormatReader contract --------------------------------------

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        import asyncio
        return asyncio.run(self.read_async(document))

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        # Validate the document shape and resolve the API key BEFORE
        # importing the heavy SDK — that way operators get a useful
        # ``GOOGLE_API_KEY``-style error message even when
        # ``google-genai`` is not installed in the worker.
        if document.is_text:
            raise FormatReaderError(
                f"GeminiPdfReader expected bytes for {document.source_uri}; "
                "got text.",
            )

        api_key = self._resolved_api_key()

        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types as genai_types  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise FormatReaderError(
                "GeminiPdfReader requires the 'google-genai' package; install via "
                "`pip install google-genai`.",
            ) from exc

        try:
            client = genai.Client(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            raise FormatReaderError(
                f"GeminiPdfReader: failed to construct google-genai client: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        markdown = await self._invoke_model(
            client, genai_types, document,
        )
        return self._sections_from_markdown(markdown, document.source_uri)

    # ----- Internals ---------------------------------------------------

    def _resolved_api_key(self) -> str:
        api_key = self._api_key or os.environ.get(_API_KEY_ENV_VAR)
        if not api_key:
            raise FormatReaderError(
                "GeminiPdfReader: no API key. Set GOOGLE_API_KEY in "
                "the environment or pass api_key= explicitly.",
            )
        return api_key

    async def _invoke_model(
        self, client: Any, genai_types: Any, document: RawDocument,
    ) -> str:
        """Call ``client.aio.models.generate_content`` with the PDF
        inlined as a ``Part`` with ``application/pdf`` mime + the
        prompt as a second text part. Returns the model's text body.
        """

        # ``Part.from_bytes`` is the documented helper for inline
        # documents on the google-genai SDK; falls back to the raw
        # ``inline_data`` shape on older SDK versions that pre-date
        # the helper.
        try:
            pdf_part = genai_types.Part.from_bytes(
                data=document.bytes_, mime_type="application/pdf",
            )
        except AttributeError:  # pragma: no cover - older SDK
            pdf_part = genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type="application/pdf", data=document.bytes_,
                ),
            )
        text_part = genai_types.Part.from_text(text=self._prompt)

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": self._max_output_tokens,
        }
        if self._cached_content:
            config_kwargs["cached_content"] = self._cached_content

        try:
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=[pdf_part, text_part],
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            raise FormatReaderError(
                f"GeminiPdfReader: API call failed for "
                f"{document.source_uri}: {type(exc).__name__}: {exc}",
            ) from exc

        return self._extract_text(response, document.source_uri)

    @staticmethod
    def _extract_text(response: Any, source_uri: str) -> str:
        """Concatenate every text part across every candidate.

        Robust to (a) blocked / safety-filtered responses (no
        candidates), (b) candidates with no parts (rare),
        (c) the ``response.text`` shortcut for the trivial case.
        """
        # Fast path: most successful responses have a flat ``.text``.
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None) or ()
        if not candidates:
            block_reason = getattr(
                getattr(response, "prompt_feedback", None),
                "block_reason", None,
            )
            raise FormatReaderError(
                f"GeminiPdfReader: empty response for {source_uri} "
                f"(block_reason={block_reason!r}).",
            )
        parts_text: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or ():
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    parts_text.append(value)
        merged = "\n".join(parts_text).strip()
        if not merged:
            raise FormatReaderError(
                f"GeminiPdfReader: response had no text parts for "
                f"{source_uri}.",
            )
        return merged

    def _sections_from_markdown(
        self, markdown: str, source_uri: str,
    ) -> Sequence[ParsedSection]:
        """Split on ``<!-- page: N -->`` markers, same shape as
        :meth:`AnthropicPdfReader._sections_from_markdown`. Falls
        back to one section if the model didn't emit markers.
        """
        parts = _PAGE_MARKER_RE.split(markdown)
        common_extra: dict[str, Any] = {
            "metadata_origin": "gemini",
            "model": self._model,
        }

        sections: list[ParsedSection] = []
        char_cursor = 0

        if len(parts) == 1:
            text = parts[0].strip()
            if not text:
                return ()
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
                    format="markdown",
                    extra=common_extra,
                )
            )
            char_cursor += len(text)

        return tuple(sections)


__all__ = ("GeminiPdfReader",)
