"""Tests for :class:`RemotePdfExtractorReader`.

The reader is the generic glue between any
:class:`PdfExtractorDeployment` (Marker / Docling / MinerU) and the
canonical reader contract. We don't bring up Ray for the unit
tests; we mock ``_resolve_handle`` to return a fake handle whose
``extract`` returns synthesised :class:`ExtractResult` shapes.

Coverage:
1. Contract — figure blobs land in the image store, markdown
   image refs rewrite to ``colony-image://<sha>``, sections split
   on ``<!-- page: N -->`` markers.
2. Wire — the reader passes through ``ExtractOptions`` to the
   handle, surfaces deployment errors as
   :class:`FormatReaderError`.
3. Wire-shape resilience — the handle may return a dict
   (post-cloudpickle round-trip) instead of the typed model;
   the reader coerces.
"""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.cluster.extractors import (
    ExtractOptions,
    ExtractResult,
    FigureBlob,
)
from polymathera.colony.knowledge.models import (
    KnowledgeFormat,
    RawDocument,
)
from polymathera.colony.knowledge.readers.base import FormatReaderError
from polymathera.colony.knowledge.readers.remote_pdf import (
    RemotePdfExtractorReader,
)
from polymathera.colony.knowledge.stores.image import (
    URI_SCHEME,
    InMemoryImageStore,
)


pytestmark = pytest.mark.asyncio


class _FakeHandle:
    """Stand-in for a Ray serving deployment handle. Records every
    ``extract`` call and replays a programmable response."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def extract(
        self, *, pdf_bytes: bytes, options: ExtractOptions | None = None,
    ) -> Any:
        self.calls.append(
            {"pdf_bytes_len": len(pdf_bytes), "options": options},
        )
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _patch_handle(monkeypatch: pytest.MonkeyPatch, handle: Any) -> None:
    async def _resolve(self) -> Any:
        return handle

    monkeypatch.setattr(
        RemotePdfExtractorReader, "_resolve_handle", _resolve,
    )


@pytest.fixture
def image_store() -> InMemoryImageStore:
    return InMemoryImageStore()


@pytest.fixture
def reader(image_store: InMemoryImageStore) -> RemotePdfExtractorReader:
    return RemotePdfExtractorReader(
        backend="docling",
        image_store=image_store,
    )


@pytest.fixture
def pdf_doc() -> RawDocument:
    return RawDocument(
        source_uri="file:///tmp/sample.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"%PDF-1.4\n(fake)",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_async_emits_one_section_per_page_marker(
    reader: RemotePdfExtractorReader,
    image_store: InMemoryImageStore,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = FigureBlob(
        blob_id="picture-0",
        image_bytes=b"\x89PNG\r\n\x1a\nfake",
        mime="image/png",
        page=1,
        label="picture-0",
    )
    handle = _FakeHandle(
        ExtractResult(
            markdown=(
                "<!-- page: 1 -->\n"
                "# Intro\n\nSee ![diagram](picture-0) for the layout.\n\n"
                "<!-- page: 2 -->\n"
                "## Methods\n\nNo figure on this page.\n"
            ),
            figures=(blob,),
            backend="docling",
            page_count=2,
        ),
    )
    _patch_handle(monkeypatch, handle)

    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 2
    assert sections[0].section_path == "page-1"
    assert sections[1].section_path == "page-2"
    assert sections[0].format == "markdown"
    assert sections[0].extra["metadata_origin"] == "docling"

    # The figure landed in the image store and the markdown ref
    # was rewritten to point at it.
    assert len(sections[0].figures) == 1
    fig = sections[0].figures[0]
    assert fig.image_uri.startswith(f"{URI_SCHEME}://")
    assert "picture-0" not in sections[0].text
    assert fig.image_uri in sections[0].text
    assert await image_store.get(fig.image_uri) == blob.image_bytes


async def test_read_async_no_marker_emits_single_section(
    reader: RemotePdfExtractorReader,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _FakeHandle(
        ExtractResult(
            markdown="# Doc\n\nFlat output, no page markers.",
            figures=(),
            backend="docling",
            page_count=1,
        ),
    )
    _patch_handle(monkeypatch, handle)
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 1
    assert sections[0].section_path == "document"
    assert sections[0].citation.page_number is None


async def test_read_async_coerces_dict_response_from_wire(
    reader: RemotePdfExtractorReader,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``ExtractResult`` round-trips through cloudpickle / a
    Ray handle, it may come back as a plain dict. The reader
    coerces via ``model_validate`` so the rest of the pipeline
    sees a typed model."""
    dict_response: dict[str, Any] = {
        "markdown": "<!-- page: 1 -->\nbody",
        "figures": [],
        "backend": "marker",
        "page_count": 1,
    }
    handle = _FakeHandle(dict_response)
    _patch_handle(monkeypatch, handle)
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 1
    assert sections[0].extra["metadata_origin"] == "marker"


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


async def test_read_async_forwards_extract_options(
    image_store: InMemoryImageStore,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = ExtractOptions(
        extract_images=False, table_format="html",
    )
    reader = RemotePdfExtractorReader(
        backend="marker",
        image_store=image_store,
        extract_options=options,
    )
    handle = _FakeHandle(
        ExtractResult(markdown="body", backend="marker"),
    )
    _patch_handle(monkeypatch, handle)
    await reader.read_async(pdf_doc)
    assert len(handle.calls) == 1
    forwarded = handle.calls[0]["options"]
    assert forwarded.extract_images is False
    assert forwarded.table_format == "html"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_async_text_payload_raises(
    reader: RemotePdfExtractorReader,
) -> None:
    text_doc = RawDocument(
        source_uri="file:///x.txt",
        detected_format=KnowledgeFormat.PDF,
        payload="i am text",
    )
    with pytest.raises(FormatReaderError):
        await reader.read_async(text_doc)


async def test_read_async_handle_failure_surfaces_message(
    reader: RemotePdfExtractorReader,
    pdf_doc: RawDocument,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _FakeHandle(RuntimeError("docling crashed"))
    _patch_handle(monkeypatch, handle)
    with pytest.raises(FormatReaderError, match="docling crashed"):
        await reader.read_async(pdf_doc)


def test_constructor_rejects_unknown_backend(
    image_store: InMemoryImageStore,
) -> None:
    with pytest.raises(ValueError, match="unknown self-hosted backend"):
        RemotePdfExtractorReader(
            backend="not-a-backend",
            image_store=image_store,
        )


def test_constructor_rejects_none_image_store() -> None:
    with pytest.raises(ValueError, match="image_store"):
        RemotePdfExtractorReader(
            backend="marker",
            image_store=None,  # type: ignore[arg-type]
        )
