"""Unit tests for ``GrobidPdfReader`` (httpx mocked).

The reader's HTTP call is intercepted via ``httpx.MockTransport`` so
the real ``GrobidPdfReader`` code path runs end-to-end without a
network call. Integration tests against a live GROBID live in
``tests/integration/`` and are skipped when ``POLYMATHERA_GROBID_URL``
is unset.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge import (
    FormatReaderError,
    GrobidPdfReader,
    KnowledgeFormat,
    RawDocument,
    default_registry,
    default_registry_with_grobid,
)


_TEI_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Example Paper on Tokamak Confinement</title>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>This paper studies confinement in spherical tokamaks.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <pb n="1"/>
      <div>
        <head>Introduction</head>
        <p>Tokamaks confine plasma magnetically.</p>
        <p>This work focuses on MAST-U.</p>
      </div>
      <pb n="2"/>
      <div>
        <head>Methods</head>
        <p>We measured energy confinement time.</p>
        <div>
          <head>Equilibrium reconstruction</head>
          <p>EFIT was used for equilibrium reconstruction.</p>
        </div>
      </div>
      <pb n="3"/>
      <div>
        <head>Results</head>
        <p>Confinement time scaled as expected.</p>
      </div>
    </body>
  </text>
</TEI>
"""


def _bytes_doc(payload: bytes = b"%PDF-1.4 fake") -> RawDocument:
    return RawDocument(
        source_uri="file:///tmp/example.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=payload,
    )


def _patch_httpx(monkeypatch, handler):
    """Replace ``httpx.AsyncClient`` + ``httpx.Client`` with versions
    bound to a ``MockTransport`` so the real ``GrobidPdfReader`` runs
    end-to-end without a network call."""

    import httpx

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    def _async_client(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _client)
    monkeypatch.setattr(httpx, "AsyncClient", _async_client)


# ---- Construction ---------------------------------------------------------


def test_construction_requires_base_url() -> None:
    with pytest.raises(ValueError):
        GrobidPdfReader(base_url="")


def test_construction_handles_pdf() -> None:
    r = GrobidPdfReader(base_url="http://grobid:8070")
    assert KnowledgeFormat.PDF in r.handles


# ---- Parsing --------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_tei_extracts_title_and_abstract(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(200, text=_TEI_FIXTURE),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    sections = await reader.read_async(_bytes_doc())
    headings = [s.heading for s in sections]
    assert "title" in headings
    assert "abstract" in headings
    title_section = next(s for s in sections if s.heading == "title")
    assert "Tokamak Confinement" in title_section.text


@pytest.mark.asyncio
async def test_parse_tei_recovers_section_hierarchy(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(200, text=_TEI_FIXTURE),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    sections = await reader.read_async(_bytes_doc())
    paths = [s.section_path for s in sections]
    # Title + abstract are 0/0 and 0/1; numbered top-level sections
    # follow.
    assert "0/0" in paths and "0/1" in paths
    # At least one nested section path (e.g., '2/1' for "Equilibrium
    # reconstruction" under "Methods").
    assert any(
        "/" in p and not p.startswith("0/") and p[0].isdigit()
        for p in paths
    )


@pytest.mark.asyncio
async def test_parse_tei_assigns_page_numbers(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(200, text=_TEI_FIXTURE),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    sections = await reader.read_async(_bytes_doc())
    pages = {s.heading: s.citation.page_number for s in sections}
    assert pages.get("Introduction") == 1
    assert pages.get("Methods") == 2
    assert pages.get("Results") == 3


@pytest.mark.asyncio
async def test_text_payload_rejected(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(200, text=_TEI_FIXTURE),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    text_doc = RawDocument(
        source_uri="x://y",
        detected_format=KnowledgeFormat.PDF,
        payload="not bytes",
    )
    with pytest.raises(FormatReaderError):
        await reader.read_async(text_doc)


@pytest.mark.asyncio
async def test_grobid_error_status_raises(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(503, text="grobid is starting up"),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    with pytest.raises(FormatReaderError):
        await reader.read_async(_bytes_doc())


@pytest.mark.asyncio
async def test_malformed_xml_raises(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(200, text="<not><proper></xml"),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    with pytest.raises(FormatReaderError):
        await reader.read_async(_bytes_doc())


def test_sync_read_path(monkeypatch) -> None:
    import httpx

    _patch_httpx(
        monkeypatch,
        lambda request: httpx.Response(200, text=_TEI_FIXTURE),
    )
    reader = GrobidPdfReader(base_url="http://grobid:8070", timeout_s=5.0)
    sections = reader.read(_bytes_doc())
    assert any(s.heading == "Introduction" for s in sections)


# ---- Registry helper ------------------------------------------------------


def test_default_registry_with_grobid_overrides_pdf() -> None:
    reg = default_registry_with_grobid(grobid_url="http://grobid:8070")
    pdf_reader = reg.reader_for(KnowledgeFormat.PDF)
    assert isinstance(pdf_reader, GrobidPdfReader)


def test_default_registry_keeps_pypdf() -> None:
    reg = default_registry()
    pdf_reader = reg.reader_for(KnowledgeFormat.PDF)
    assert pdf_reader is not None
    assert not isinstance(pdf_reader, GrobidPdfReader)
