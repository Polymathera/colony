"""Tests for the GROBID demote — :class:`GrobidMetadataReader` +
:class:`GrobidPdfReader.mode='metadata_only'`.

Covers the structural change: in ``metadata_only`` mode the reader
emits ONLY the title + abstract sections (no body) and attaches a
``bibliographic`` payload to ``ParsedSection.extra``. The
``GrobidMetadataReader`` alias is the operator-facing entry point
with sensible defaults turned on (header consolidation, raw
citations).
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.readers.grobid_pdf import (
    GrobidMetadataReader,
    GrobidPdfReader,
)


# A trimmed TEI XML fixture covering the bits the metadata path
# must parse: title, authors with affiliations, abstract, body
# (which metadata-only mode MUST drop), and a reference list.
_TEI_FIXTURE = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>A Sample Paper on Multimodal Extraction</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Alice</forename>
                <surname>Author</surname>
              </persName>
              <affiliation>
                <orgName>MIT</orgName>
              </affiliation>
            </author>
            <author>
              <persName>
                <forename>Bob</forename>
                <surname>Builder</surname>
              </persName>
              <affiliation>
                <orgName>Stanford</orgName>
              </affiliation>
            </author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>Short abstract sentence one. Second sentence with detail.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>1. Introduction</head>
        <p>Body paragraph that metadata-only mode should NOT emit.</p>
      </div>
      <div>
        <head>2. Method</head>
        <p>Another body paragraph that should be skipped.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct>
            <analytic>
              <title>First reference paper</title>
            </analytic>
            <monogr>
              <imprint>
                <date type="published" when="2024">2024</date>
              </imprint>
            </monogr>
          </biblStruct>
          <biblStruct>
            <analytic>
              <title>Second reference paper</title>
            </analytic>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


def _reader(mode: str) -> GrobidPdfReader:
    return GrobidPdfReader(base_url="http://grobid:8070", mode=mode)


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="metadata_only"):
        GrobidPdfReader(base_url="http://grobid:8070", mode="kaboom")


def test_metadata_only_mode_skips_body_sections() -> None:
    reader = _reader("metadata_only")
    sections = reader._parse_tei(_TEI_FIXTURE, "file:///x.pdf")
    # Title + abstract = 2 sections; no body sections.
    assert len(sections) == 2
    headings = [s.heading for s in sections]
    assert "title" in headings
    assert "abstract" in headings
    # No section corresponds to "1. Introduction" or "2. Method".
    body_texts = [s.text.lower() for s in sections]
    assert not any("body paragraph" in t for t in body_texts)


def test_full_mode_keeps_body_sections() -> None:
    reader = _reader("full")
    sections = reader._parse_tei(_TEI_FIXTURE, "file:///x.pdf")
    # Title + abstract + body sections.
    assert len(sections) > 2
    body_texts = [s.text.lower() for s in sections]
    assert any("body paragraph" in t for t in body_texts)


def test_metadata_only_attaches_bibliographic_to_first_section() -> None:
    reader = _reader("metadata_only")
    sections = reader._parse_tei(_TEI_FIXTURE, "file:///x.pdf")
    bib = sections[0].extra.get("bibliographic")
    assert isinstance(bib, dict)
    # Two authors with affiliations.
    authors = bib.get("authors") or []
    assert len(authors) == 2
    names = sorted(a["name"] for a in authors)
    assert any("Alice" in n for n in names)
    assert any("Bob" in n for n in names)
    affs = sorted(a["affiliation"] for a in authors)
    assert "MIT" in affs[0] or "MIT" in affs[1]
    # Two references.
    references = bib.get("references") or []
    assert len(references) == 2
    titles = [r["title"] for r in references]
    assert "First reference paper" in titles
    assert "Second reference paper" in titles


def test_metadata_origin_marked_grobid_in_metadata_only_mode() -> None:
    reader = _reader("metadata_only")
    sections = reader._parse_tei(_TEI_FIXTURE, "file:///x.pdf")
    assert sections[0].extra.get("metadata_origin") == "grobid"


def test_metadata_reader_alias_uses_metadata_only_with_consolidation() -> None:
    """``GrobidMetadataReader`` is the operator-facing entry point —
    sensible defaults (header consolidation on, raw citations on)
    so its bibliographic payload is as rich as GROBID can make it."""
    reader = GrobidMetadataReader(base_url="http://grobid:8070")
    assert reader._mode == "metadata_only"
    assert reader._consolidate_header == 1
    assert reader._consolidate_citations == 1
    assert reader._include_raw_citations is True
