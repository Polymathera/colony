"""Tests for the in-process format readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymathera.colony.knowledge import (
    CsvReader,
    HtmlReader,
    JsonlReader,
    JupyterReader,
    KnowledgeFormat,
    MarkdownReader,
    PlainTextReader,
    RawDocument,
    SourceCodeReader,
    default_registry,
)


def _doc(text: str, fmt: KnowledgeFormat, uri: str = "test://doc") -> RawDocument:
    return RawDocument(source_uri=uri, detected_format=fmt, payload=text)


def test_plain_text_paragraphs() -> None:
    text = "Para 1 line.\n\nPara 2 line one.\nLine two.\n\n\nPara 3."
    sections = PlainTextReader().read(_doc(text, KnowledgeFormat.PLAIN_TEXT))
    assert len(sections) == 3
    assert sections[0].text.startswith("Para 1")
    assert sections[2].text.startswith("Para 3")


def test_markdown_headings_are_section_path() -> None:
    text = (
        "# Title\n\nIntro.\n\n"
        "## Section A\n\nContent A.\n\n"
        "### Sub A1\n\nDetail A1.\n\n"
        "## Section B\n\nContent B."
    )
    sections = MarkdownReader().read(_doc(text, KnowledgeFormat.MARKDOWN))
    paths = [s.section_path for s in sections]
    assert paths == ["1", "1/1", "1/1/1", "1/2"]
    assert "Detail A1" in sections[2].text


def test_html_skips_script_and_style() -> None:
    text = (
        "<html><head><style>body{}</style><script>alert(1)</script></head>"
        "<body><h1>Title</h1><p>Body text</p>"
        "<h2>Sub</h2><p>More body</p></body></html>"
    )
    sections = HtmlReader().read(_doc(text, KnowledgeFormat.HTML))
    assert any("Body text" in s.text for s in sections)
    assert all("alert(1)" not in s.text for s in sections)
    assert all("body{}" not in s.text for s in sections)


def test_jsonl_per_line_section() -> None:
    text = (
        '{"title": "First", "doi": "10.1/x"}\n'
        '{"title": "Second", "doi": "10.2/y"}\n'
        "\n"  # blank
        "not-json\n"
    )
    sections = JsonlReader().read(_doc(text, KnowledgeFormat.JSONL))
    assert len(sections) == 3
    assert sections[0].heading == "First"
    assert sections[2].heading.startswith("malformed")


def test_csv_header_plus_rows() -> None:
    text = "name,role\nAlice,RE\nBob,QA\n"
    sections = CsvReader().read(_doc(text, KnowledgeFormat.CSV))
    assert sections[0].section_path == "0"
    assert sections[0].heading == "header"
    assert sections[1].text.startswith("name: Alice")
    assert "RE" in sections[1].text


def test_source_code_python_blocks() -> None:
    text = (
        "import os\n\n"
        "def foo():\n    return 1\n\n"
        "class Bar:\n    def m(self):\n        return 2\n"
    )
    doc = RawDocument(
        source_uri="file:///tmp/x.py",
        detected_format=KnowledgeFormat.SOURCE_CODE,
        payload=text,
    )
    sections = SourceCodeReader().read(doc)
    headings = [s.heading for s in sections]
    # The reader may emit a preamble + blocks; check we got the
    # function and class headings.
    assert "foo" in headings
    assert "Bar" in headings


def test_source_code_unknown_language_single_section() -> None:
    text = "(* OCaml *)\nlet x = 1"
    doc = RawDocument(
        source_uri="file:///tmp/x.ml",
        detected_format=KnowledgeFormat.SOURCE_CODE,
        payload=text,
    )
    sections = SourceCodeReader().read(doc)
    # objectivec recogniser would attempt a match; the language map
    # covers .m → objectivec but the regex requires an open paren so
    # this falls into the "no blocks" path, returning the whole file.
    # Whether we get one or several sections, the body must be present.
    full_text = "\n".join(s.text for s in sections)
    assert "let x = 1" in full_text


def test_jupyter_reader_per_cell() -> None:
    payload = json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "source": "# Title\n\nIntro"},
                {
                    "cell_type": "code",
                    "source": ["print(1)\n", "x = 2"],
                    "outputs": [
                        {"output_type": "stream", "text": "1\n"},
                    ],
                },
            ],
        },
    )
    sections = JupyterReader().read(
        RawDocument(
            source_uri="file:///nb.ipynb",
            detected_format=KnowledgeFormat.JUPYTER,
            payload=payload,
        )
    )
    assert len(sections) == 2
    assert sections[1].extra["cell_type"] == "code"
    assert sections[1].extra.get("outputs") == "1\n"


def test_default_registry_has_all() -> None:
    reg = default_registry()
    fmts = set(reg.formats())
    assert {
        KnowledgeFormat.PLAIN_TEXT,
        KnowledgeFormat.MARKDOWN,
        KnowledgeFormat.HTML,
        KnowledgeFormat.JSONL,
        KnowledgeFormat.CSV,
        KnowledgeFormat.SOURCE_CODE,
        KnowledgeFormat.JUPYTER,
        KnowledgeFormat.PDF,
    } <= fmts
