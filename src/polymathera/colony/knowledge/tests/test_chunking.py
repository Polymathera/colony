"""Tests for the chunkers (prose + code)."""

from __future__ import annotations

from polymathera.colony.knowledge import (
    ChunkerConfig,
    CitationSpan,
    CodeChunker,
    ParsedSection,
    ProseChunker,
)


def _section(text: str, source_uri: str = "doc:test") -> ParsedSection:
    return ParsedSection(
        section_path="1",
        heading="Test",
        text=text,
        citation=CitationSpan(
            source_uri=source_uri,
            section_path="1",
            char_start=0,
            char_end=len(text),
        ),
    )


def test_prose_chunker_paragraph_aware() -> None:
    text = "\n\n".join(["Paragraph %d." % i + " " * 50 for i in range(20)])
    chunker = ProseChunker(
        config=ChunkerConfig(target_tokens=20, overlap_tokens=4, min_tokens=4),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(_section(text))
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip()
        assert c.token_count > 0
        assert c.citation.source_uri == "doc:test"


def test_prose_chunker_short_text_one_chunk() -> None:
    chunker = ProseChunker(
        config=ChunkerConfig(target_tokens=200, min_tokens=1, overlap_tokens=0),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(_section("Just a single short paragraph here."))
    assert len(chunks) == 1
    assert "single short paragraph" in chunks[0].text


def test_prose_chunker_empty_text_zero_chunks() -> None:
    chunker = ProseChunker()
    chunks = chunker.chunk(_section("   "))
    assert chunks == ()


def test_prose_chunker_overlap_provides_continuity() -> None:
    text = "\n\n".join(f"Paragraph {i}." for i in range(10))
    chunker = ProseChunker(
        config=ChunkerConfig(target_tokens=4, overlap_tokens=2, min_tokens=1),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(_section(text))
    assert len(chunks) >= 2
    # Each successive chunk shares at least one paragraph with the
    # previous one (the overlap).
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_paras = {p.strip() for p in prev.text.split("\n\n") if p.strip()}
        next_paras = {p.strip() for p in nxt.text.split("\n\n") if p.strip()}
        assert prev_paras & next_paras


def test_code_chunker_handles_python() -> None:
    text = (
        "def f1():\n    return 1\n\n"
        + "def f2():\n    return 2\n\n"
        + "class C:\n    def m(self):\n        return 3\n"
    )
    chunker = CodeChunker(
        config=ChunkerConfig(target_tokens=80, overlap_tokens=4, min_tokens=1),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(_section(text), language="python", data_type="code")
    assert chunks, "expected at least one chunk"
    full = "\n".join(c.text for c in chunks)
    for needle in ("def f1", "def f2", "class C"):
        assert needle in full


def test_code_chunker_falls_back_for_unknown_language() -> None:
    text = "(* OCaml-like content *)\nlet x = 1\n" * 10
    chunker = CodeChunker(
        config=ChunkerConfig(target_tokens=20, min_tokens=1, overlap_tokens=2),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(_section(text), language="ocaml", data_type="code")
    assert chunks
    full = "\n".join(c.text for c in chunks)
    assert "OCaml-like content" in full
