"""Tests for ``detect_format`` + ``language_for_source_code``."""

from __future__ import annotations

from pathlib import Path

from polymathera.colony.knowledge import (
    KnowledgeFormat,
    detect_format,
    language_for_source_code,
)


def test_extension_detection() -> None:
    assert detect_format(path="paper.pdf") is KnowledgeFormat.PDF
    assert detect_format(path="readme.MD") is KnowledgeFormat.MARKDOWN
    assert detect_format(path="data.csv") is KnowledgeFormat.CSV
    assert detect_format(path="x.tsv") is KnowledgeFormat.CSV
    assert detect_format(path="x.ipynb") is KnowledgeFormat.JUPYTER
    assert detect_format(path="x.py") is KnowledgeFormat.SOURCE_CODE
    assert detect_format(path="x.unknownext") is KnowledgeFormat.UNKNOWN


def test_magic_byte_pdf() -> None:
    assert (
        detect_format(payload=b"%PDF-1.7\n%\xe2\xe3\xcf\xd3...")
        is KnowledgeFormat.PDF
    )


def test_magic_byte_parquet() -> None:
    payload = b"PAR1" + b"\x00" * 8 + b"PAR1"
    assert detect_format(payload=payload) is KnowledgeFormat.PARQUET


def test_mime_hint_overrides() -> None:
    assert (
        detect_format(path="x.txt", mime_hint="text/markdown")
        is KnowledgeFormat.MARKDOWN
    )


def test_text_payload_no_path_defaults_to_plain_text() -> None:
    assert detect_format(payload="hello world") is KnowledgeFormat.PLAIN_TEXT


def test_language_for_source_code() -> None:
    assert language_for_source_code("a.py") == "python"
    assert language_for_source_code("a.rs") == "rust"
    assert language_for_source_code("a.unknown") == "text"
