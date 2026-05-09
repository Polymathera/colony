"""``AnthropicPdfReader`` — PDF reader backed by Claude's native PDF support.

Anthropic's Messages API accepts ``application/pdf`` directly via the
``document`` content block (`Anthropic PDF support
<https://platform.claude.com/docs/en/build-with-claude/pdf-support>`_):
under the hood the platform rasterises every page as an image and
hands the page images + extracted text to the model. The model then
answers the prompt against the rasterised representation. We exploit
this by prompting for "convert this PDF to clean Markdown" and
parsing the resulting text into one
:class:`~polymathera.colony.knowledge.models.ParsedSection` per page.

The reader is the **second hosted backend** in the design (§5).
Compared to :class:`MistralOcrPdfReader`:

- *Higher fidelity* on visually-dense / complex docs that Mistral
  chokes on — Claude's visual-reasoning quality is the differentiator
  flagged in our library survey.
- *No figure bytes*. Anthropic returns text only; the reader cannot
  populate :class:`~polymathera.colony.knowledge.models.FigureRef.image_uri`.
  Sections come back with ``figures=()``; figures are described inline
  in the markdown (Claude writes prose / placeholder image refs).
  Operators who need clickable image previews use the Mistral OCR
  reader. This is a real tradeoff documented in the doc and the
  reader's class docstring.
- *Cost*. ~$0.009 / page (Sonnet 4.5 at typical density) — five to
  ten times Mistral OCR. The reader supports prompt-caching the
  document so re-runs of the same PDF (e.g. operator A/B'ing prompts)
  pay the prefix-cache discount.

Like :class:`MistralOcrPdfReader`, this reader holds only
configuration. The :class:`anthropic.AsyncAnthropic` client is
constructed inside ``read_async`` so connection state never crosses
the cloudpickle boundary; the API key is resolved at call time so
the blueprint is pickle-clean even when the worker doesn't have
``ANTHROPIC_API_KEY`` set yet.
"""

from __future__ import annotations

import base64
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
from .base import FormatReader, FormatReaderError


logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "claude-sonnet-4-5"
_DEFAULT_MAX_TOKENS = 8_192
_DEFAULT_TIMEOUT_S = 120.0


_DEFAULT_PROMPT = """Convert this PDF document to clean Markdown for downstream chunking and retrieval.

Strict requirements:
- Preserve LaTeX math: inline as `$...$`, display as `$$...$$`.
- Render tables as GFM markdown tables. If a table has merged cells the GFM rendering can't express, emit an HTML `<table>` instead.
- For each figure / chart / diagram, emit a brief description on its own line as a markdown image reference whose URL is the placeholder string `figure-N` (where N is the figure number, 1-indexed within the page) and whose alt text is the description, e.g. `![Block diagram of the control loop with PI feedback and anti-windup clamp.](figure-1)`. We do NOT need the image bytes, only the description.
- Preserve heading hierarchy with `#` / `##` / `###`.
- Preserve in-text references like "Fig. 3" or "Table 2" verbatim — they anchor cross-references.
- Insert a literal page-break marker between pages, on its own line: `<!-- page: N -->` (1-indexed). The marker MUST appear immediately before the first content of page N, including before page N's heading if any. This lets downstream chunking preserve per-page locality.

Do NOT add commentary, summaries, or anything outside the Markdown body. Return ONLY the Markdown."""


_PAGE_MARKER_RE = re.compile(
    r"^<!--\s*page\s*:\s*(?P<page>\d+)\s*-->\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class AnthropicPdfReader(FormatReader):
    """PDF reader that calls Claude's native PDF endpoint.

    Args:
        image_store: Accepted for symmetry with the other multimodal
            readers but unused — Anthropic returns no image bytes.
            Pass the active store anyway so the registry factory can
            wire every reader uniformly.
        api_key: ``None`` (default) reads ``ANTHROPIC_API_KEY`` at
            call time. Picklable across nodes.
        model: Defaults to ``claude-sonnet-4-5``. Pin a specific
            snapshot for reproducibility.
        max_tokens: Output budget. The reader returns once the model
            stops; large papers may need >8K. Bumped here from the
            default 1K of the Messages API.
        timeout_s: Per-call HTTP timeout.
        prompt: Override the default markdown-conversion prompt.
            Useful for operator A/B tests from the dashboard.
        prompt_cache: When ``True`` (default), attach
            ``cache_control={"type": "ephemeral"}`` to the document
            block so repeated calls on the same PDF (re-ingest with
            a different prompt, the dashboard's A/B button) pay the
            prefix-cache discount.
    """

    def __init__(
        self,
        *,
        image_store: ImageStore | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        prompt: str | None = None,
        prompt_cache: bool = True,
    ) -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        # image_store is accepted but stored only for diagnostics —
        # Anthropic's response carries no image bytes so the reader
        # has nothing to put. We still record it so the operator can
        # introspect via the reader's API surface.
        self._image_store = image_store
        self._api_key = api_key
        self._model = model
        self._max_tokens = int(max_tokens)
        self._timeout_s = float(timeout_s)
        self._prompt = prompt or _DEFAULT_PROMPT
        self._prompt_cache = bool(prompt_cache)

    @property
    def model(self) -> str:
        return self._model

    @property
    def has_image_extraction(self) -> bool:
        """``False`` — Anthropic returns text only. Documented here so
        consumers (the KB tab badge, operator-facing diagnostics) can
        signal the limitation without sniffing class names."""
        return False

    # ----- FormatReader contract --------------------------------------

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        import asyncio
        return asyncio.run(self.read_async(document))

    async def read_async(
        self, document: RawDocument,
    ) -> Sequence[ParsedSection]:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise FormatReaderError(
                "AnthropicPdfReader requires the 'anthropic' package; install via "
                "`pip install polymathera-colony[cpu]` or `pip install anthropic`.",
            ) from exc

        if document.is_text:
            raise FormatReaderError(
                f"AnthropicPdfReader expected bytes for {document.source_uri}; "
                "got text.",
            )

        api_key = self._resolved_api_key()
        client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=self._timeout_s,
        )

        try:
            markdown = await self._invoke_model(client, document)
        finally:
            # ``AsyncAnthropic`` holds an underlying httpx client.
            # Explicitly close it so the per-call object doesn't leak
            # connections through the asyncio loop.
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass

        return self._sections_from_markdown(markdown, document.source_uri)

    # ----- Internals ---------------------------------------------------

    def _resolved_api_key(self) -> str:
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if not api_key:
            raise FormatReaderError(
                "AnthropicPdfReader: no API key. Set ANTHROPIC_API_KEY in "
                "the environment or pass api_key= explicitly.",
            )
        return api_key

    async def _invoke_model(
        self, client: Any, document: RawDocument,
    ) -> str:
        """Build the Messages-API request, send it, return the
        model's text body.

        Raises :class:`FormatReaderError` on any non-success path so
        the ingestor records ``status=FAILED`` with a useful message
        rather than crashing the whole bulk-ingest call.
        """
        b64 = base64.standard_b64encode(document.bytes_).decode("ascii")
        document_block: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            },
        }
        if self._prompt_cache:
            document_block["cache_control"] = {"type": "ephemeral"}

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            document_block,
                            {"type": "text", "text": self._prompt},
                        ],
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            # Catches anthropic.APIError (status / auth / rate limit)
            # plus anything else httpx might raise. We surface the
            # message verbatim so the ingestion record's ``error``
            # field is actionable.
            raise FormatReaderError(
                f"AnthropicPdfReader: API call failed for "
                f"{document.source_uri}: {type(exc).__name__}: {exc}",
            ) from exc

        return self._extract_text(response, document.source_uri)

    @staticmethod
    def _extract_text(response: Any, source_uri: str) -> str:
        """Concatenate every text block in the response.

        Robust to (a) responses with multiple text blocks (common when
        the model intersperses thinking / tool use), (b) responses
        whose ``content`` list is empty (rare but documented).
        """
        content = getattr(response, "content", None)
        if not content:
            raise FormatReaderError(
                f"AnthropicPdfReader: empty response for {source_uri} "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r}).",
            )
        parts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                text_value = getattr(block, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
        merged = "\n".join(parts).strip()
        if not merged:
            raise FormatReaderError(
                f"AnthropicPdfReader: response had no text blocks for "
                f"{source_uri}.",
            )
        return merged

    def _sections_from_markdown(
        self, markdown: str, source_uri: str,
    ) -> Sequence[ParsedSection]:
        """Split the model's markdown into one
        :class:`ParsedSection` per ``<!-- page: N -->`` marker.

        If the model didn't emit markers (or emitted them
        inconsistently), the whole response becomes one section with
        ``page_number=None``. The chunker still produces sensible
        chunks via its paragraph-aware splitting; we just lose
        per-page citation precision in that case.
        """
        # Split on the page marker. ``re.split`` with a capturing group
        # interleaves the captured page numbers with the body text, so
        # zip() pairs them up cleanly.
        parts = _PAGE_MARKER_RE.split(markdown)
        # parts looks like:
        #   [<text-before-first-marker>, <page-N>, <text-after>, <page-M>, <text-after>, ...]
        # If no markers were found, parts == [markdown].

        sections: list[ParsedSection] = []
        char_cursor = 0
        common_extra: dict[str, Any] = {
            "metadata_origin": "anthropic",
            "model": self._model,
        }

        if len(parts) == 1:
            # No page markers — emit the whole response as one section.
            text = parts[0].strip()
            if not text:
                return ()
            sections.append(
                ParsedSection(
                    section_path="document",
                    heading="",
                    text=text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path="document",
                        char_start=0,
                        char_end=len(text),
                        page_number=None,
                    ),
                    figures=(),
                    format="markdown",
                    extra=common_extra,
                )
            )
            return tuple(sections)

        # Collect (page_no, text) pairs.
        # parts[0] is the prefix before the first marker — usually
        # empty (the prompt asks for the marker BEFORE page-1 content),
        # but if the model emits a preamble we treat it as page 0.
        prefix_text = parts[0].strip()
        if prefix_text:
            sections.append(
                ParsedSection(
                    section_path="page-prefix",
                    heading="",
                    text=prefix_text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path="page-prefix",
                        char_start=char_cursor,
                        char_end=char_cursor + len(prefix_text),
                        page_number=None,
                    ),
                    figures=(),
                    format="markdown",
                    extra=common_extra,
                )
            )
            char_cursor += len(prefix_text)

        # Remaining: pairs of (page-number-string, body).
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
                    heading="",
                    text=text,
                    citation=CitationSpan(
                        source_uri=source_uri,
                        section_path=section_path,
                        char_start=char_cursor,
                        char_end=char_cursor + len(text),
                        page_number=page_no,
                    ),
                    figures=(),
                    format="markdown",
                    extra=common_extra,
                )
            )
            char_cursor += len(text)

        return tuple(sections)


__all__ = ("AnthropicPdfReader",)
