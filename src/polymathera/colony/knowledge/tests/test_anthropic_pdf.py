"""Tests for :class:`AnthropicPdfReader`.

The reader speaks to Anthropic's Messages API via the official
``anthropic`` SDK. We don't ship a real API key in tests, so the
SDK is monkeypatched at ``messages.create`` (the only call the
reader makes) to return a synthesised :class:`anthropic.Message`-
shaped object — enough for the reader's parser to walk.

Two coverage goals:
1. Contract — the response is converted to one
   :class:`ParsedSection` per ``<!-- page: N -->`` marker, with
   ``format="markdown"``, ``metadata_origin="anthropic"``, and
   ``figures=()`` (Anthropic returns no image bytes).
2. Robustness — missing API key, empty content blocks, and
   markerless responses degrade gracefully (one section / clear
   error / typed exception).
"""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.knowledge.models import (
    KnowledgeFormat,
    RawDocument,
)
from polymathera.colony.knowledge.readers.anthropic_pdf import (
    AnthropicPdfReader,
    _PAGE_MARKER_RE,
)
from polymathera.colony.knowledge.readers.base import FormatReaderError
from polymathera.colony.knowledge.stores.image import InMemoryImageStore


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    """Minimal stand-in for :class:`anthropic.types.TextBlock`."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeMessage:
    """Minimal stand-in for the ``Message`` the SDK returns."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    """Captures the kwargs each test passes to ``messages.create``
    so assertions can inspect the document block + prompt without
    hitting the network."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class _FakeAnthropic:
    """Replaces ``anthropic.AsyncAnthropic`` for the duration of a
    test. Constructor records the api_key so we can assert the
    reader resolved it correctly."""

    def __init__(self, *, api_key: str, **_: Any) -> None:
        self.api_key = api_key
        self.messages = _FakeMessages(self._next_response)
        self.closed = False

    # Class-level slot the test patches before instantiating, so
    # tests don't need to thread a custom subclass through the
    # monkeypatch each time.
    _next_response: Any = None

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def patched_anthropic(monkeypatch: pytest.MonkeyPatch):
    """Replace ``anthropic.AsyncAnthropic`` with the recording fake.

    Yields a setter the test calls with the fake response (or an
    exception); the setter returns the most recently constructed
    :class:`_FakeAnthropic` so the test can inspect the request
    that the reader sent.
    """
    import anthropic  # type: ignore[import-not-found]

    instances: list[_FakeAnthropic] = []

    def _factory(**kwargs: Any) -> _FakeAnthropic:
        client = _FakeAnthropic(**kwargs)
        instances.append(client)
        return client

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _factory)

    def _set_response(response: Any) -> None:
        _FakeAnthropic._next_response = response

    yield _set_response, instances
    _FakeAnthropic._next_response = None


@pytest.fixture
def reader() -> AnthropicPdfReader:
    return AnthropicPdfReader(
        image_store=InMemoryImageStore(),
        api_key="test-key",
        model="claude-sonnet-4-5",
        max_tokens=2_048,
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
# Page-marker regex (sanity)
# ---------------------------------------------------------------------------


def test_page_marker_re_matches_well_formed_marker() -> None:
    body = "stuff\n<!-- page: 3 -->\nmore"
    parts = _PAGE_MARKER_RE.split(body)
    # split with a captured group: ['stuff\n', '3', '\nmore']
    assert parts[0].rstrip() == "stuff"
    assert parts[1] == "3"
    assert parts[2].lstrip() == "more"


def test_page_marker_re_tolerates_whitespace_and_case() -> None:
    body = "x\n<!--   PAGE  :  17  -->\ny"
    parts = _PAGE_MARKER_RE.split(body)
    assert parts[1] == "17"


# ---------------------------------------------------------------------------
# Happy path — paginated response
# ---------------------------------------------------------------------------


async def test_read_async_emits_one_section_per_page_marker(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, _ = patched_anthropic
    set_response(
        _FakeMessage(
            "<!-- page: 1 -->\n# Intro\n\nBody text on page 1.\n\n"
            "<!-- page: 2 -->\n## Methods\n\nBody text on page 2.\n",
        )
    )
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 2
    assert sections[0].section_path == "page-1"
    assert sections[0].citation.page_number == 1
    assert sections[0].format == "markdown"
    assert sections[0].extra["metadata_origin"] == "anthropic"
    assert sections[0].extra["model"] == "claude-sonnet-4-5"
    assert sections[0].figures == ()
    assert "# Intro" in sections[0].text
    assert sections[1].section_path == "page-2"
    assert sections[1].citation.page_number == 2


async def test_read_async_no_marker_returns_single_section(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, _ = patched_anthropic
    set_response(_FakeMessage("# Doc\n\nNo page markers in this response."))
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 1
    assert sections[0].section_path == "document"
    assert sections[0].citation.page_number is None
    assert sections[0].text.startswith("# Doc")


async def test_read_async_handles_preamble_before_first_marker(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, _ = patched_anthropic
    set_response(
        _FakeMessage(
            "Preamble Claude added.\n\n<!-- page: 1 -->\n\nPage 1 content.\n",
        )
    )
    sections = await reader.read_async(pdf_doc)
    # Two sections: preamble (no page) + page 1.
    assert len(sections) == 2
    assert sections[0].section_path == "page-prefix"
    assert sections[0].citation.page_number is None
    assert sections[1].section_path == "page-1"
    assert sections[1].citation.page_number == 1


async def test_read_async_skips_empty_page_bodies(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    """A page marker followed by only whitespace should not produce
    an empty section — common when Claude emits a marker on its own
    line but the page is blank."""
    set_response, _ = patched_anthropic
    set_response(
        _FakeMessage(
            "<!-- page: 1 -->\n\n   \n\n<!-- page: 2 -->\nReal content.\n",
        )
    )
    sections = await reader.read_async(pdf_doc)
    assert len(sections) == 1
    assert sections[0].section_path == "page-2"
    assert sections[0].citation.page_number == 2


# ---------------------------------------------------------------------------
# Wire shape (request)
# ---------------------------------------------------------------------------


async def test_read_async_sends_document_block_with_base64(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, instances = patched_anthropic
    set_response(_FakeMessage("<!-- page: 1 -->\nbody"))
    await reader.read_async(pdf_doc)
    assert len(instances) == 1
    client = instances[0]
    assert client.api_key == "test-key"
    assert client.closed is True  # reader closed the client
    assert len(client.messages.calls) == 1
    request = client.messages.calls[0]
    assert request["model"] == "claude-sonnet-4-5"
    assert request["max_tokens"] == 2_048
    content = request["messages"][0]["content"]
    # First block is the document, second is the prompt.
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"
    assert content[0]["source"]["type"] == "base64"
    # The base64 should round-trip back to the original bytes.
    import base64

    assert base64.b64decode(content[0]["source"]["data"]) == pdf_doc.bytes_
    # Prompt-cache breakpoint is set by default.
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[1]["type"] == "text"
    assert content[1]["text"]  # non-empty default prompt


async def test_read_async_respects_prompt_cache_disable(
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, instances = patched_anthropic
    set_response(_FakeMessage("<!-- page: 1 -->\nbody"))
    reader = AnthropicPdfReader(
        api_key="k", prompt_cache=False,
    )
    await reader.read_async(pdf_doc)
    request = instances[0].messages.calls[0]
    document_block = request["messages"][0]["content"][0]
    assert "cache_control" not in document_block


async def test_read_async_uses_custom_prompt(
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, instances = patched_anthropic
    set_response(_FakeMessage("<!-- page: 1 -->\nbody"))
    reader = AnthropicPdfReader(
        api_key="k", prompt="Just give me the title.",
    )
    await reader.read_async(pdf_doc)
    request = instances[0].messages.calls[0]
    assert (
        request["messages"][0]["content"][1]["text"]
        == "Just give me the title."
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_read_async_text_payload_raises(
    reader: AnthropicPdfReader,
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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reader = AnthropicPdfReader()
    with pytest.raises(FormatReaderError, match="ANTHROPIC_API_KEY"):
        await reader.read_async(pdf_doc)


async def test_read_async_api_failure_surfaces_message(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    set_response, _ = patched_anthropic
    set_response(RuntimeError("upstream 500"))
    with pytest.raises(FormatReaderError, match="upstream 500"):
        await reader.read_async(pdf_doc)


async def test_read_async_empty_response_raises(
    reader: AnthropicPdfReader,
    pdf_doc: RawDocument,
    patched_anthropic,
) -> None:
    """A response with no content blocks (Claude declines or the
    transport produced an empty payload) becomes a typed
    :class:`FormatReaderError` so the ingestion record carries a
    useful message."""

    class _Empty:
        content: list[Any] = []
        stop_reason = "end_turn"

    set_response, _ = patched_anthropic
    set_response(_Empty())
    with pytest.raises(FormatReaderError, match="empty response"):
        await reader.read_async(pdf_doc)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_reader_advertises_no_image_extraction() -> None:
    r = AnthropicPdfReader(api_key="k")
    assert r.has_image_extraction is False
