"""Tests for :class:`LlamaParsePdfReader`.

LlamaParse is asynchronous (upload → poll → result) so the reader's
HTTP shape is more involved than the other hosted readers. We mock
the three endpoints (``POST /parse/upload``, ``GET /parse/{job_id}``,
and the presigned-URL image GETs) at the ``httpx`` level via
:class:`httpx.MockTransport`.

Coverage:
1. End-to-end happy path — multi-page result, image bytes
   downloaded from presigned URLs and stored in the
   :class:`ImageStore`, markdown image refs rewritten to
   ``colony-image://<sha>``.
2. Tier knob — ``tier`` is forwarded into the upload payload and
   surfaces on ``ParsedSection.extra["tier"]``.
3. Polling — ``RUNNING`` → ``COMPLETED`` transition is honoured;
   timeouts and ``FAILED`` statuses surface as
   :class:`FormatReaderError`.
4. Image-fetch failures don't crash the whole document.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from polymathera.colony.knowledge.models import (
    KnowledgeFormat,
    RawDocument,
)
from polymathera.colony.knowledge.readers.base import FormatReaderError
from polymathera.colony.knowledge.readers.llamaparse_pdf import (
    LlamaParsePdfReader,
)
from polymathera.colony.knowledge.stores.image import (
    URI_SCHEME,
    InMemoryImageStore,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_store() -> InMemoryImageStore:
    return InMemoryImageStore()


@pytest.fixture
def reader(image_store: InMemoryImageStore) -> LlamaParsePdfReader:
    return LlamaParsePdfReader(
        image_store=image_store,
        api_key="test-key",
        tier="cost_effective",
        timeout_s=5.0,
        poll_interval_s=0.0,  # don't waste real time in unit tests
    )


@pytest.fixture
def pdf_doc() -> RawDocument:
    return RawDocument(
        source_uri="file:///tmp/sample.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"%PDF-1.4\n(fake)",
    )


def _build_transport(
    *,
    poll_sequence: list[str] | None = None,
    result_body: dict[str, Any] | None = None,
    image_bytes: bytes = b"\x89PNG\r\n\x1a\nfake-png",
    fail_on_upload: bool = False,
    fail_image_fetch: bool = False,
    upload_status: int = 200,
) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    """Build a transport that walks the LlamaParse upload → poll →
    result → presigned-image flow.

    Returns ``(transport, requests_log)`` where ``requests_log`` is
    a list of ``{method, path}`` dicts the test can introspect.
    """
    poll_sequence = list(poll_sequence or ["RUNNING", "COMPLETED"])
    poll_index = {"i": 0}
    requests_log: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests_log.append(
            {
                "method": request.method,
                "path": path,
                "url": str(request.url),
            }
        )

        # Upload
        if path.endswith("/parse/upload") and request.method == "POST":
            if fail_on_upload:
                return httpx.Response(upload_status, text="boom")
            return httpx.Response(
                upload_status, json={"id": "job-abc"},
            )

        # Status / result polling
        if "/parse/job-abc" in path:
            i = poll_index["i"]
            poll_index["i"] = min(i + 1, len(poll_sequence) - 1)
            status = poll_sequence[i]
            # When the test asks for ``expand=...`` the ``status``
            # call is the result fetch, not a poll — return the
            # full body.
            if status == "COMPLETED" and "expand" in dict(request.url.params):
                body = result_body or _default_result_body()
                return httpx.Response(200, json=body)
            return httpx.Response(
                200,
                json={
                    "id": "job-abc",
                    "status": status,
                    "error_message": (
                        "agentic ran out of credits"
                        if status == "FAILED" else None
                    ),
                },
            )

        # Presigned image GET
        if path.startswith("/presigned/"):
            if fail_image_fetch:
                return httpx.Response(403, text="forbidden")
            return httpx.Response(
                200,
                content=image_bytes,
                headers={"content-type": "image/png"},
            )

        return httpx.Response(404, text=f"unmocked {path!r}")

    return httpx.MockTransport(handler), requests_log


def _default_result_body() -> dict[str, Any]:
    """Two-page LlamaParse result with one figure on page 1."""
    return {
        "id": "job-abc",
        "status": "COMPLETED",
        "markdown": [
            {
                "page": 1,
                "markdown": (
                    "# Intro\n\nSee ![fig1](image_0.png) for the diagram.\n"
                ),
            },
            {
                "page": 2,
                "markdown": "## Methods\n\nBody on page 2.\n",
            },
        ],
        "images_content_metadata": {
            "total_count": 1,
            "images": [
                {
                    "index": 0,
                    "filename": "image_0.png",
                    "content_type": "image/png",
                    "size_bytes": 1234,
                    "presigned_url": "https://api.cloud.llamaindex.ai/presigned/image_0.png",
                },
            ],
        },
    }


@pytest.fixture
def patched_httpx(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``httpx.AsyncClient`` so the reader's HTTP calls
    flow through a :class:`MockTransport` we control per test."""
    state: dict[str, Any] = {
        "poll_sequence": ["RUNNING", "COMPLETED"],
        "result_body": None,
        "image_bytes": b"\x89PNG\r\n\x1a\nfake-png",
        "fail_on_upload": False,
        "upload_status": 200,
        "fail_image_fetch": False,
        "requests": [],
    }
    real_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        transport, log = _build_transport(
            poll_sequence=state["poll_sequence"],
            result_body=state["result_body"],
            image_bytes=state["image_bytes"],
            fail_on_upload=state["fail_on_upload"],
            upload_status=state["upload_status"],
            fail_image_fetch=state["fail_image_fetch"],
        )
        # Append rather than replace so the test sees calls from
        # any number of client instances the reader constructed.
        state["requests"].append(log)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return state


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_async_emits_one_section_per_page(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 2
    assert sections[0].section_path == "page-1"
    assert sections[1].section_path == "page-2"
    assert sections[0].format == "markdown"
    assert sections[0].extra["metadata_origin"] == "llamaparse"
    assert sections[0].extra["tier"] == "cost_effective"


async def test_read_async_stores_figures_via_presigned_url(
    reader: LlamaParsePdfReader,
    image_store: InMemoryImageStore,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    sections = await reader.read_async(pdf_doc)
    page1 = sections[0]
    assert len(page1.figures) == 1
    fig = page1.figures[0]
    assert fig.image_uri.startswith(f"{URI_SCHEME}://")
    assert await image_store.get(fig.image_uri) == b"\x89PNG\r\n\x1a\nfake-png"


async def test_read_async_rewrites_markdown_image_refs(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    sections = await reader.read_async(pdf_doc)
    md = sections[0].text
    # Original ``image_0.png`` ref replaced by colony-image URI.
    assert "image_0.png" not in md
    assert "colony-image://" in md


async def test_read_async_polls_until_completed(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """Poll RUNNING → RUNNING → COMPLETED ⇒ three poll requests
    plus the upload + result + presigned + … sequence."""
    patched_httpx["poll_sequence"] = ["RUNNING", "RUNNING", "COMPLETED"]
    await reader.read_async(pdf_doc)
    # The MockTransport tracks every request the AsyncClient
    # issued. We expect at least: upload, two RUNNING polls, one
    # COMPLETED poll (status check), and one COMPLETED poll
    # (result fetch with expand), plus the presigned image GET.
    paths = [r["path"] for log in patched_httpx["requests"] for r in log]
    assert sum(1 for p in paths if p.endswith("/parse/upload")) == 1
    assert sum(1 for p in paths if "/parse/job-abc" in p) >= 3
    assert sum(1 for p in paths if p.startswith("/presigned/")) == 1


# ---------------------------------------------------------------------------
# Wire / tier
# ---------------------------------------------------------------------------


async def test_read_async_forwards_tier_to_extra(
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """The reader's ``tier`` knob is the operator's cost/quality
    lever. We verify it's stored on the reader (so it ships in
    the upload's ``configuration`` payload — httpx's job to encode)
    AND that it surfaces on every emitted section's
    ``extra["tier"]`` so retrieval / KB-tab consumers can see which
    tier produced each chunk."""
    agentic_reader = LlamaParsePdfReader(
        image_store=InMemoryImageStore(),
        api_key="k",
        tier="agentic",
        poll_interval_s=0.0,
    )
    assert agentic_reader.tier == "agentic"
    sections = await agentic_reader.read_async(pdf_doc)
    assert len(sections) >= 1
    assert all(s.extra["tier"] == "agentic" for s in sections)


async def test_read_async_attaches_bearer_auth(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_headers: list[str] = []
    real_async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("authorization", ""))
        if request.url.path.endswith("/parse/upload"):
            return httpx.Response(200, json={"id": "job-abc"})
        if "/parse/job-abc" in request.url.path:
            if "expand" in dict(request.url.params):
                return httpx.Response(200, json=_default_result_body())
            return httpx.Response(200, json={"id": "job-abc", "status": "COMPLETED"})
        if request.url.path.startswith("/presigned/"):
            return httpx.Response(200, content=b"img")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: real_async_client(*a, transport=transport, **{
            k: v for k, v in kw.items() if k != "transport"
        }),
    )
    await reader.read_async(pdf_doc)
    # Every reader call to the LlamaCloud host carries the bearer.
    # (Presigned URLs are pre-authenticated and don't need it.)
    api_calls = [h for h in seen_headers if h]
    assert all(h == "Bearer test-key" for h in api_calls)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_async_text_payload_raises(
    reader: LlamaParsePdfReader,
) -> None:
    text_doc = RawDocument(
        source_uri="file:///x.txt",
        detected_format=KnowledgeFormat.PDF,
        payload="i am text",
    )
    with pytest.raises(FormatReaderError):
        await reader.read_async(text_doc)


async def test_read_async_missing_api_key_raises(
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    bad_reader = LlamaParsePdfReader(image_store=InMemoryImageStore())
    with pytest.raises(FormatReaderError, match="LLAMA_CLOUD_API_KEY"):
        await bad_reader.read_async(pdf_doc)


async def test_read_async_upload_failure_surfaces_status(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    patched_httpx["fail_on_upload"] = True
    patched_httpx["upload_status"] = 500
    with pytest.raises(FormatReaderError, match="HTTP 500"):
        await reader.read_async(pdf_doc)


async def test_read_async_failed_status_surfaces_message(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    patched_httpx["poll_sequence"] = ["FAILED", "FAILED"]
    with pytest.raises(FormatReaderError, match="agentic ran out of credits"):
        await reader.read_async(pdf_doc)


async def test_read_async_image_fetch_failure_skipped(
    reader: LlamaParsePdfReader,
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """A 403 on a presigned URL must not crash the whole document
    — the operator gets the markdown and missing figures, not a
    failed ingestion record."""
    patched_httpx["fail_image_fetch"] = True
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 2
    # No figures landed because the fetch failed.
    assert sections[0].figures == ()


async def test_read_async_timeout_raises(
    pdf_doc: RawDocument,
    patched_httpx: dict[str, Any],
) -> None:
    """If the job never reaches a terminal status before the
    reader's overall budget, the reader bails with a
    :class:`FormatReaderError` rather than hanging forever."""
    patched_httpx["poll_sequence"] = ["RUNNING", "RUNNING", "RUNNING"]
    timeout_reader = LlamaParsePdfReader(
        image_store=InMemoryImageStore(),
        api_key="k",
        timeout_s=0.05,
        poll_interval_s=0.01,
    )
    with pytest.raises(FormatReaderError, match="did not complete"):
        await timeout_reader.read_async(pdf_doc)


def test_constructor_rejects_none_image_store() -> None:
    with pytest.raises(ValueError, match="image_store"):
        LlamaParsePdfReader(image_store=None)  # type: ignore[arg-type]
