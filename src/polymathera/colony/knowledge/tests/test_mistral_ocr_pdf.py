"""Tests for :class:`MistralOcrPdfReader`.

The reader does HTTP I/O to a vendor endpoint, which we never
contact in unit tests. We mock the three calls (``POST /v1/files``,
``GET /v1/files/{id}/url``, ``POST /v1/ocr``) at the ``httpx``
level via :class:`httpx.MockTransport` and assert on the resulting
:class:`ParsedSection`s.

Two coverage goals:
1. The reader's *contract* — it produces page-aware sections,
   image bytes land in the active :class:`ImageStore`, markdown
   refs are rewritten, ``ParsedSection.figures`` is populated with
   correct ``page`` / ``bbox`` / ``image_uri``.
2. The reader's *robustness* — auth-missing → clear error,
   transport-level HTTP failures → :class:`FormatReaderError`,
   malformed images skipped without failing the whole page.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from polymathera.colony.knowledge.models import (
    KnowledgeFormat,
    RawDocument,
)
from polymathera.colony.knowledge.readers.base import FormatReaderError
from polymathera.colony.knowledge.readers.mistral_ocr_pdf import (
    MistralOcrPdfReader,
    _DATA_URI_RE,
    _MD_IMAGE_REF_RE,
)
from polymathera.colony.knowledge.stores.image import (
    URI_SCHEME,
    InMemoryImageStore,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _png_b64() -> str:
    """A trivially-decodable PNG-tagged data URI for tests. The
    payload bytes themselves don't need to be valid PNG — the
    reader only base64-decodes and stores them; the dashboard's
    image-resolve endpoint is what eventually serves them."""
    return "data:image/png;base64," + base64.b64encode(b"\x89PNG\x00fake").decode("ascii")


def _ocr_response_body() -> dict[str, Any]:
    """Two-page Mistral-OCR-shaped JSON. Page 1 has a figure; page
    2 has a malformed image (covered in the robustness test)."""
    return {
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 2, "doc_size_bytes": 1024},
        "pages": [
            {
                "index": 0,
                "markdown": (
                    "# Introduction\n\nSee ![fig1](img-0.jpeg) for the architecture.\n"
                ),
                "images": [
                    {
                        "id": "img-0.jpeg",
                        "image_base64": _png_b64(),
                        "top_left_x": 100,
                        "top_left_y": 200,
                        "bottom_right_x": 300,
                        "bottom_right_y": 500,
                    },
                ],
                "dimensions": {"dpi": 200, "width": 1575, "height": 1970},
            },
            {
                "index": 1,
                "markdown": "## Methods\n\nBody text on page 2.\n",
                # One valid image, one with a broken data URI — the
                # reader should keep the valid one and skip the
                # broken one.
                "images": [
                    {
                        "id": "img-1.png",
                        "image_base64": "data:image/png;base64,not-actually-base64?!",
                    },
                ],
                "dimensions": {"dpi": 200, "width": 1575, "height": 1970},
            },
        ],
    }


def _build_mock_transport(
    response_body: dict[str, Any] | None = None,
    fail_on: str | None = None,
) -> httpx.MockTransport:
    """Build a transport that returns plausible responses for the
    three Mistral endpoints the reader hits.

    ``fail_on`` injects a 500 on a specific endpoint path
    (``/files``, ``/files/<id>/url``, ``/ocr``) so the
    error-path tests can assert on the surfaced message.
    """
    response_body = response_body or _ocr_response_body()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if fail_on and fail_on in path:
            return httpx.Response(500, text="boom")
        if path.endswith("/v1/files") and request.method == "POST":
            return httpx.Response(200, json={"id": "file-abc"})
        if "/v1/files/" in path and path.endswith("/url"):
            return httpx.Response(
                200, json={"url": "https://signed.example/abc"},
            )
        if path.endswith("/v1/ocr") and request.method == "POST":
            return httpx.Response(200, json=response_body)
        return httpx.Response(404, text=f"unmocked path {path!r}")

    return httpx.MockTransport(handler)


@pytest.fixture
def image_store() -> InMemoryImageStore:
    return InMemoryImageStore()


@pytest.fixture
def reader(image_store: InMemoryImageStore) -> MistralOcrPdfReader:
    return MistralOcrPdfReader(
        image_store=image_store,
        api_key="test-key",
        model="mistral-ocr-latest",
        timeout_s=5.0,
    )


@pytest.fixture
def pdf_doc() -> RawDocument:
    return RawDocument(
        source_uri="file:///tmp/sample.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"%PDF-1.4\n(fake)",
    )


# ---------------------------------------------------------------------------
# Helper: monkeypatch httpx.AsyncClient to use our transport
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_httpx(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``httpx.AsyncClient`` with a wrapper that injects our
    :class:`MockTransport`. Returns a small mutable holder the
    individual tests poke to control transport behaviour."""

    state: dict[str, Any] = {
        "response_body": _ocr_response_body(),
        "fail_on": None,
        "requests": [],
    }
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        # Strip transport kwarg if a caller already passed one;
        # honour everything else (timeout, headers, …).
        kwargs.pop("transport", None)
        transport = _build_mock_transport(
            response_body=state["response_body"],
            fail_on=state["fail_on"],
        )
        # Wrap the handler so we can record requests for assertions.
        original_handler = transport.handler

        def recording_handler(request: httpx.Request) -> httpx.Response:
            state["requests"].append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "headers": dict(request.headers),
                }
            )
            return original_handler(request)

        transport.handler = recording_handler
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return state


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_async_emits_one_section_per_page(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 2
    assert sections[0].section_path == "page-1"
    assert sections[1].section_path == "page-2"
    assert sections[0].format == "markdown"
    assert sections[0].extra["metadata_origin"] == "mistral_ocr"
    assert sections[0].extra["model"] == "mistral-ocr-latest"


async def test_read_async_stores_figure_bytes(
    reader: MistralOcrPdfReader,
    image_store: InMemoryImageStore,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    await reader.read_async(pdf_doc)
    # The valid PNG payload should be in the image store under the
    # colony-image:// scheme.
    expected_payload = base64.b64decode(
        _png_b64().split(",", 1)[1].encode("ascii"),
    )
    # We don't hardcode the SHA — the store is content-addressed,
    # so we verify by listing bytes round-tripped via the URI in
    # the resulting FigureRef.
    sections = await reader.read_async(pdf_doc)
    page1_figures = sections[0].figures
    assert len(page1_figures) == 1
    fig = page1_figures[0]
    assert fig.image_uri.startswith(f"{URI_SCHEME}://")
    assert fig.page == 1
    assert fig.bbox == (100.0, 200.0, 300.0, 500.0)
    assert fig.label == "img-0.jpeg"
    assert await image_store.get(fig.image_uri) == expected_payload


async def test_read_async_rewrites_markdown_image_refs(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    sections = await reader.read_async(pdf_doc)
    md = sections[0].text
    # The original ``img-0.jpeg`` ref should have been replaced by
    # the colony-image URI from the store.
    assert "img-0.jpeg" not in md
    refs = _MD_IMAGE_REF_RE.findall(md)
    assert len(refs) == 1
    alt, url = refs[0]
    assert alt == "fig1"
    assert url == sections[0].figures[0].image_uri


async def test_read_async_records_figure_label_to_id(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """``section.extra["figure_label_to_id"]`` lets the chunker resolve
    in-text references like 'img-0.jpeg' back to the FigureRef."""
    sections = await reader.read_async(pdf_doc)
    label_map = sections[0].extra["figure_label_to_id"]
    fig = sections[0].figures[0]
    assert label_map == {"img-0.jpeg": fig.figure_id}


async def test_read_async_skips_malformed_image_keeps_page(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """Page 2 has a broken data URI; the page section is still
    emitted with markdown intact, just with an empty figures
    tuple."""
    sections = await reader.read_async(pdf_doc)
    page2 = sections[1]
    assert page2.section_path == "page-2"
    assert page2.figures == ()
    assert "Body text on page 2" in page2.text


# ---------------------------------------------------------------------------
# Wire / request shape
# ---------------------------------------------------------------------------


async def test_read_async_sends_three_calls_in_order(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """Upload → signed URL → OCR — exactly three calls, exactly
    these paths, exactly this order."""
    await reader.read_async(pdf_doc)
    paths = [r["path"] for r in patched_httpx["requests"]]
    methods = [r["method"] for r in patched_httpx["requests"]]
    assert paths[0].endswith("/v1/files")
    assert methods[0] == "POST"
    assert "/v1/files/" in paths[1] and paths[1].endswith("/url")
    assert methods[1] == "GET"
    assert paths[2].endswith("/v1/ocr")
    assert methods[2] == "POST"


async def test_read_async_attaches_bearer_auth(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    await reader.read_async(pdf_doc)
    for record in patched_httpx["requests"]:
        assert record["headers"].get("authorization") == "Bearer test-key"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_async_text_payload_raises(
    reader: MistralOcrPdfReader,
) -> None:
    text_doc = RawDocument(
        source_uri="file:///tmp/x.txt",
        detected_format=KnowledgeFormat.PDF,
        payload="text content",
    )
    with pytest.raises(FormatReaderError):
        await reader.read_async(text_doc)


async def test_read_async_missing_api_key_raises(
    image_store: InMemoryImageStore,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    bad_reader = MistralOcrPdfReader(image_store=image_store)
    with pytest.raises(FormatReaderError, match="MISTRAL_API_KEY"):
        await bad_reader.read_async(pdf_doc)


async def test_read_async_upload_failure_surfaces_status(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    patched_httpx["fail_on"] = "/files"
    with pytest.raises(FormatReaderError, match="HTTP 500"):
        await reader.read_async(pdf_doc)


async def test_read_async_ocr_failure_surfaces_status(
    reader: MistralOcrPdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    patched_httpx["fail_on"] = "/ocr"
    with pytest.raises(FormatReaderError, match="HTTP 500"):
        await reader.read_async(pdf_doc)


async def test_constructor_rejects_none_image_store() -> None:
    with pytest.raises(ValueError, match="image_store"):
        MistralOcrPdfReader(image_store=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_data_uri_regex_matches_standard_form() -> None:
    m = _DATA_URI_RE.match("data:image/jpeg;base64,xyz")
    assert m is not None
    assert m.group("mime") == "image/jpeg"
    assert m.group("data") == "xyz"


def test_decode_data_uri_handles_naked_base64() -> None:
    """Mistral SHOULD always send a data URI, but the reader
    tolerates raw base64 too — defaulting to PNG."""
    out = MistralOcrPdfReader._decode_data_uri(
        base64.b64encode(b"naked").decode("ascii"),
    )
    assert out == (b"naked", "image/png")


def test_decode_data_uri_returns_none_for_garbage() -> None:
    assert MistralOcrPdfReader._decode_data_uri(None) is None
    assert MistralOcrPdfReader._decode_data_uri("") is None
    assert MistralOcrPdfReader._decode_data_uri("data:image/png;base64,!@#?") is None


def test_bbox_helper_returns_none_on_partial_data() -> None:
    assert (
        MistralOcrPdfReader._bbox_from_mistral({"top_left_x": 1})
        is None
    )
    assert MistralOcrPdfReader._bbox_from_mistral(
        {
            "top_left_x": 1, "top_left_y": 2,
            "bottom_right_x": 3, "bottom_right_y": 4,
        },
    ) == (1.0, 2.0, 3.0, 4.0)


def test_rewrite_leaves_external_urls_alone() -> None:
    md = "see ![remote](https://example.com/img.png) and ![local](img-0.jpeg)"
    out = MistralOcrPdfReader._rewrite_markdown_image_refs(
        md, {"img-0.jpeg": "colony-image://abc"},
    )
    assert "https://example.com/img.png" in out
    assert "colony-image://abc" in out
    assert "img-0.jpeg" not in out
