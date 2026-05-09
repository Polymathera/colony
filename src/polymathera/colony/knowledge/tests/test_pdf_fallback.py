"""Tests for the Mistral 1000-page fallback path.

Two contracts:

1. :class:`MistralOcrPdfReader` translates the
   ``document_parser_too_many_pages`` HTTP 400 into the typed
   :class:`PdfTooManyPagesError` (carrying ``page_count`` /
   ``max_pages`` when the message is parseable).
2. :class:`FallbackPdfReader` routes ONLY that typed error to the
   fallback reader; other ``FormatReaderError`` subclasses propagate.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from polymathera.colony.knowledge.models import (
    CitationSpan, KnowledgeFormat, ParsedSection, RawDocument,
)
from polymathera.colony.knowledge.readers.base import (
    FormatReader, FormatReaderError, PdfTooManyPagesError,
)
from polymathera.colony.knowledge.readers.fallback import FallbackPdfReader
from polymathera.colony.knowledge.readers.mistral_ocr_pdf import (
    MistralOcrPdfReader,
    _parse_mistral_page_limits,
)
from polymathera.colony.knowledge.stores.image import InMemoryImageStore


_TOO_MANY_PAGES_BODY = (
    '{"object":"error","message":"This document has 2096 pages, which '
    'is more than the maximum allowed of 1000.",'
    '"type":"document_parser_too_many_pages","param":null,'
    '"code":"3730","raw_status_code":400}'
)


# ---------------------------------------------------------------------------
# _parse_mistral_page_limits — best-effort regex on the JSON message
# ---------------------------------------------------------------------------


def test_parse_mistral_page_limits_extracts_numbers() -> None:
    page_count, max_pages = _parse_mistral_page_limits(_TOO_MANY_PAGES_BODY)
    assert page_count == 2096
    assert max_pages == 1000


def test_parse_mistral_page_limits_returns_none_on_unparseable() -> None:
    """Best-effort: a future Mistral error format that omits the
    'has N pages' phrasing degrades to ``(None, None)`` so the typed
    exception still fires; we just lose the diagnostic detail."""
    assert _parse_mistral_page_limits('{"message":"different shape"}') == (None, None)
    assert _parse_mistral_page_limits("not even json") == (None, None)


# ---------------------------------------------------------------------------
# MistralOcrPdfReader — 400 → typed exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mistral_reader_promotes_too_many_pages_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapping :class:`FallbackPdfReader` only fires on the typed
    exception, so the Mistral reader MUST classify this 400 into
    :class:`PdfTooManyPagesError` rather than the generic
    :class:`FormatReaderError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-abc"})
        if path.endswith("/url"):
            return httpx.Response(200, json={"url": "https://signed.example/x"})
        if path.endswith("/ocr"):
            return httpx.Response(400, text=_TOO_MANY_PAGES_BODY)
        return httpx.Response(404, text=f"unmocked {path!r}")

    real_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        # Reader passes its own ``transport=None``; replace with our mock.
        kwargs.pop("transport", None)
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    reader = MistralOcrPdfReader(
        image_store=InMemoryImageStore(), api_key="fake",
    )
    document = RawDocument(
        source_uri="file:///tmp/big.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"%PDF-1.4 (fake)",
    )
    with pytest.raises(PdfTooManyPagesError) as exc_info:
        await reader.read_async(document)
    assert exc_info.value.page_count == 2096
    assert exc_info.value.max_pages == 1000


# ---------------------------------------------------------------------------
# FallbackPdfReader — routes typed error, propagates the rest
# ---------------------------------------------------------------------------


class _StubReader(FormatReader):
    def __init__(
        self, *, raise_exc: Exception | None = None, label: str = "stub",
    ) -> None:
        super().__init__(handles=(KnowledgeFormat.PDF,))
        self._raise_exc = raise_exc
        self._label = label
        self.calls = 0

    def read(self, document: RawDocument) -> Sequence[ParsedSection]:
        self.calls += 1
        if self._raise_exc is not None:
            raise self._raise_exc
        return [ParsedSection(
            section_path="body",
            text=f"hello from {self._label}",
            citation=CitationSpan(source_uri=document.source_uri),
            extra={"label": self._label},
        )]


@pytest.mark.asyncio
async def test_fallback_routes_too_many_pages_to_fallback_reader() -> None:
    primary = _StubReader(
        raise_exc=PdfTooManyPagesError(
            "rejected", page_count=2096, max_pages=1000,
        ),
        label="primary",
    )
    fallback = _StubReader(label="fallback")
    wrapper = FallbackPdfReader(primary=primary, fallback=fallback)

    document = RawDocument(
        source_uri="file:///x.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"",
    )
    sections = await wrapper.read_async(document)
    assert primary.calls == 1
    assert fallback.calls == 1
    assert len(sections) == 1
    assert sections[0].extra["label"] == "fallback"


@pytest.mark.asyncio
async def test_fallback_propagates_other_format_reader_errors() -> None:
    """A generic :class:`FormatReaderError` (auth failure, malformed
    response, …) MUST propagate — silently swapping backends would
    mask real errors."""
    primary = _StubReader(raise_exc=FormatReaderError("auth failed"), label="primary")
    fallback = _StubReader(label="fallback")
    wrapper = FallbackPdfReader(primary=primary, fallback=fallback)

    document = RawDocument(
        source_uri="file:///x.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"",
    )
    with pytest.raises(FormatReaderError, match="auth failed"):
        await wrapper.read_async(document)
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_fallback_returns_primary_sections_on_success() -> None:
    """Happy path — primary succeeds, fallback never runs."""
    primary = _StubReader(label="primary")
    fallback = _StubReader(label="fallback")
    wrapper = FallbackPdfReader(primary=primary, fallback=fallback)

    document = RawDocument(
        source_uri="file:///x.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"",
    )
    sections = await wrapper.read_async(document)
    assert primary.calls == 1
    assert fallback.calls == 0
    assert sections[0].extra["label"] == "primary"
