"""Tests for :class:`GeminiPdfReader`.

The reader speaks to Google's ``google-genai`` SDK
(``client.aio.models.generate_content``). We monkeypatch
``google.genai.Client`` for the test so the unit suite never hits
the real Gemini endpoint.

Coverage:
1. Contract — markdown response with ``<!-- page: N -->`` markers
   yields one :class:`ParsedSection` per page with
   ``metadata_origin="gemini"``, ``format="markdown"``,
   ``figures=()``.
2. Wire shape — the reader sends the PDF as an inline part with
   the right mime, attaches the prompt as a second text part,
   honours ``cached_content_name`` when set.
3. Robustness — missing API key, blocked / empty responses, and
   transport errors surface as :class:`FormatReaderError`.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from polymathera.colony.knowledge.models import (
    KnowledgeFormat,
    RawDocument,
)
from polymathera.colony.knowledge.readers.base import FormatReaderError
from polymathera.colony.knowledge.readers.gemini_pdf import (
    GeminiPdfReader,
)
from polymathera.colony.knowledge.stores.image import InMemoryImageStore


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fake google-genai SDK
# ---------------------------------------------------------------------------


class _FakeTextPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeContent:
    def __init__(self, parts: list[_FakeTextPart]) -> None:
        self.parts = parts


class _FakeCandidate:
    def __init__(self, text: str) -> None:
        self.content = _FakeContent([_FakeTextPart(text)])


class _FakeResponse:
    def __init__(
        self, text: str, *, block_reason: Any = None,
    ) -> None:
        # Attach both the flat ``.text`` shortcut (the SDK provides it)
        # and the structured ``candidates`` list. Tests can poke
        # either path.
        self.text = text
        self.candidates = [_FakeCandidate(text)] if text else []
        self.prompt_feedback = types.SimpleNamespace(
            block_reason=block_reason,
        )


class _FakeAioModels:
    def __init__(self, parent: "_FakeAio") -> None:
        self._parent = parent

    async def generate_content(self, **kwargs: Any) -> Any:
        self._parent.calls.append(kwargs)
        if isinstance(self._parent.next_response, BaseException):
            raise self._parent.next_response
        return self._parent.next_response


class _FakeAio:
    def __init__(self, parent: "_FakeClient") -> None:
        self._parent = parent
        self.models = _FakeAioModels(self)
        self.calls: list[dict[str, Any]] = []
        self.next_response: Any = None


class _FakeClient:
    def __init__(self, *, api_key: str | None = None, **_: Any) -> None:
        self.api_key = api_key
        self.aio = _FakeAio(self)


class _FakePart:
    """Stand-in for :class:`google.genai.types.Part`."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @classmethod
    def from_bytes(cls, *, data: bytes, mime_type: str) -> "_FakePart":
        return cls(kind="bytes", data=data, mime_type=mime_type)

    @classmethod
    def from_text(cls, *, text: str) -> "_FakePart":
        return cls(kind="text", text=text)


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``google.genai`` and ``google.genai.types`` so
    the reader's lazy import resolves to our stand-ins.

    Yields ``(set_response, instances)`` — ``set_response`` lets the
    test stuff a response (or exception) into the next
    ``generate_content`` call; ``instances`` records every
    ``Client`` constructed so we can inspect the api_key + calls.
    """
    instances: list[_FakeClient] = []

    fake_genai_mod = types.ModuleType("google.genai")
    fake_types_mod = types.ModuleType("google.genai.types")
    fake_google_mod = types.ModuleType("google")

    def _client_factory(**kwargs: Any) -> _FakeClient:
        client = _FakeClient(**kwargs)
        instances.append(client)
        return client

    fake_genai_mod.Client = _client_factory
    fake_types_mod.Part = _FakePart
    fake_types_mod.GenerateContentConfig = _FakeGenerateContentConfig

    monkeypatch.setitem(sys.modules, "google", fake_google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_mod)

    state: dict[str, Any] = {"next_response": None}

    def _set(response: Any) -> None:
        state["next_response"] = response
        for client in instances:
            client.aio.next_response = response

    return _set, instances, state


@pytest.fixture
def reader() -> GeminiPdfReader:
    return GeminiPdfReader(
        image_store=InMemoryImageStore(),
        api_key="test-key",
        model="gemini-2.5-flash",
        max_output_tokens=1_024,
    )


@pytest.fixture
def pdf_doc() -> RawDocument:
    return RawDocument(
        source_uri="file:///tmp/sample.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=b"%PDF-1.4\n(fake)",
    )


def _wire_response(fake_genai, response: Any) -> None:
    """Helper: ensure the next instantiated client returns
    ``response``. Because the reader builds its client inside
    ``read_async``, we have to plant the response on the SDK-level
    factory, not on a pre-created instance."""
    set_response, instances, state = fake_genai
    state["next_response"] = response
    # Patch the factory so any new client picks up the response.
    real_factory = sys.modules["google.genai"].Client

    def _new_factory(**kwargs: Any) -> _FakeClient:
        client = real_factory(**kwargs)
        client.aio.next_response = state["next_response"]
        return client

    sys.modules["google.genai"].Client = _new_factory


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_read_async_emits_one_section_per_page(
    reader: GeminiPdfReader,
    pdf_doc: RawDocument,
    fake_genai,
) -> None:
    _wire_response(
        fake_genai,
        _FakeResponse(
            "<!-- page: 1 -->\n# Intro\n\nBody on 1.\n\n"
            "<!-- page: 2 -->\n## Methods\n\nBody on 2.\n",
        ),
    )
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 2
    assert sections[0].section_path == "page-1"
    assert sections[0].format == "markdown"
    assert sections[0].extra["metadata_origin"] == "gemini"
    assert sections[0].extra["model"] == "gemini-2.5-flash"
    assert sections[0].figures == ()
    assert "# Intro" in sections[0].text
    assert sections[1].section_path == "page-2"


async def test_read_async_no_marker_emits_single_section(
    reader: GeminiPdfReader,
    pdf_doc: RawDocument,
    fake_genai,
) -> None:
    _wire_response(fake_genai, _FakeResponse("# Document body"))
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 1
    assert sections[0].section_path == "document"
    assert sections[0].citation.page_number is None


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


async def test_read_async_sends_pdf_as_inline_part(
    reader: GeminiPdfReader,
    pdf_doc: RawDocument,
    fake_genai,
) -> None:
    _, instances, _ = fake_genai
    _wire_response(fake_genai, _FakeResponse("<!-- page: 1 -->\nbody"))
    await reader.read_async(pdf_doc)
    assert instances
    client = instances[0]
    assert client.api_key == "test-key"
    assert len(client.aio.calls) == 1
    request = client.aio.calls[0]
    assert request["model"] == "gemini-2.5-flash"
    contents = request["contents"]
    # First part is the PDF (bytes), second is the prompt text.
    assert contents[0].kwargs["kind"] == "bytes"
    assert contents[0].kwargs["mime_type"] == "application/pdf"
    assert contents[0].kwargs["data"] == pdf_doc.bytes_
    assert contents[1].kwargs["kind"] == "text"
    assert contents[1].kwargs["text"]
    # Default config does NOT pin cached_content.
    cfg_kwargs = request["config"].kwargs
    assert cfg_kwargs["max_output_tokens"] == 1_024
    assert "cached_content" not in cfg_kwargs


async def test_read_async_attaches_cached_content_when_set(
    pdf_doc: RawDocument, fake_genai,
) -> None:
    reader = GeminiPdfReader(
        api_key="k", cached_content_name="cachedContents/foo",
    )
    _, instances, _ = fake_genai
    _wire_response(fake_genai, _FakeResponse("body"))
    await reader.read_async(pdf_doc)
    cfg_kwargs = instances[0].aio.calls[0]["config"].kwargs
    assert cfg_kwargs["cached_content"] == "cachedContents/foo"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_async_text_payload_raises(reader: GeminiPdfReader) -> None:
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
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    reader = GeminiPdfReader()
    with pytest.raises(FormatReaderError, match="GOOGLE_API_KEY"):
        await reader.read_async(pdf_doc)


async def test_read_async_api_failure_surfaces_message(
    reader: GeminiPdfReader, pdf_doc: RawDocument, fake_genai,
) -> None:
    _wire_response(fake_genai, RuntimeError("upstream 500"))
    with pytest.raises(FormatReaderError, match="upstream 500"):
        await reader.read_async(pdf_doc)


async def test_read_async_blocked_response_raises(
    reader: GeminiPdfReader, pdf_doc: RawDocument, fake_genai,
) -> None:
    _wire_response(
        fake_genai,
        _FakeResponse("", block_reason="SAFETY"),
    )
    with pytest.raises(FormatReaderError, match="empty response"):
        await reader.read_async(pdf_doc)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_reader_advertises_no_image_extraction() -> None:
    assert GeminiPdfReader(api_key="k").has_image_extraction is False
