"""Tests for the chunker's figure-id propagation + provenance.

The chunker stays text-driven — it never opens an :class:`ImageStore`
or pulls bytes — but it MUST forward the section's figure linkage
into ``Chunk.extra["figure_ids"]`` so retrieval-side consumers can
ask for image bytes by ID without re-parsing the source.

Two coverage goals:
1. Markdown sections with figures → chunks carry ``figure_ids`` for
   the figures their text actually references (not the ones from
   neighbour chunks in the same section).
2. Plain text sections with no figures and no ``metadata_origin``
   → ``Chunk.extra`` stays empty (bit-identical to the pre-design
   chunker output, so existing serialised chunks aren't invalidated).
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.chunking import (
    ChunkerConfig,
    ProseChunker,
    _chunk_extra_for,
)
from polymathera.colony.knowledge.models import (
    CitationSpan,
    FigureRef,
    ParsedSection,
)


def _section_with_figures() -> ParsedSection:
    """Two figures, one paragraph per figure, plus a paragraph that
    mentions both. The chunker's paragraph-aware splitting will
    place each into its own chunk so we can assert the per-chunk
    figure linkage."""
    fig1 = FigureRef(image_uri="colony-image://aaaa", label="Fig.1")
    fig2 = FigureRef(image_uri="colony-image://bbbb", label="Fig.2")
    text = (
        "Paragraph one mentions ![](colony-image://aaaa) only.\n\n"
        "Paragraph two mentions ![](colony-image://bbbb) only.\n\n"
        "Paragraph three mentions both ![](colony-image://aaaa) and "
        "![](colony-image://bbbb)."
    )
    return ParsedSection(
        section_path="page-1",
        text=text,
        citation=CitationSpan(
            source_uri="file:///tmp/p.pdf",
            section_path="page-1",
            char_start=0,
            char_end=len(text),
            page_number=1,
        ),
        figures=(fig1, fig2),
        format="markdown",
        extra={"metadata_origin": "mistral_ocr"},
    )


def test_chunk_extra_origin_only_when_set() -> None:
    """Sections with no ``metadata_origin`` produce empty extra."""
    plain = ParsedSection(
        text="a b c",
        citation=CitationSpan(source_uri="x", char_start=0, char_end=5),
    )
    assert _chunk_extra_for(plain, "a b c") == {}


def test_chunk_extra_origin_forwarded() -> None:
    section = ParsedSection(
        text="some text",
        citation=CitationSpan(source_uri="x", char_start=0, char_end=9),
        extra={"metadata_origin": "marker"},
    )
    extra = _chunk_extra_for(section, "some text")
    assert extra == {"metadata_origin": "marker"}


def test_chunk_extra_figure_ids_match_referenced_uris() -> None:
    section = _section_with_figures()
    fig1, fig2 = section.figures
    # A chunk whose text references only fig1 → figure_ids = [fig1]
    extra = _chunk_extra_for(
        section, "para mentioning ![](colony-image://aaaa) once",
    )
    assert extra["figure_ids"] == [fig1.figure_id]
    # A chunk that mentions both → both, in mention order, deduped
    extra_both = _chunk_extra_for(
        section,
        "first ![](colony-image://aaaa) then ![](colony-image://bbbb) "
        "and ![](colony-image://aaaa) again",
    )
    assert extra_both["figure_ids"] == [fig1.figure_id, fig2.figure_id]


def test_chunk_extra_no_figure_ids_when_none_referenced() -> None:
    section = _section_with_figures()
    extra = _chunk_extra_for(section, "this text mentions no images")
    assert "figure_ids" not in extra
    assert extra["metadata_origin"] == "mistral_ocr"


def test_prose_chunker_propagates_figure_ids_per_chunk() -> None:
    """End-to-end: the chunker walks the section, splits paragraphs
    into chunks, and per-chunk figure_ids reflect *that chunk's*
    references — not the section's full figures list."""
    section = _section_with_figures()
    fig1, fig2 = section.figures
    # Tight token budget so each paragraph lands in its own chunk.
    # Each paragraph is 5–7 word-tokens; target=4 forces one
    # paragraph per chunk via the chunker's "stop once we reach
    # target" rule.
    chunker = ProseChunker(
        ChunkerConfig(min_tokens=1, target_tokens=4, max_tokens=60, overlap_tokens=0),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(section)
    assert len(chunks) >= 3

    # Find the chunks by which figure URI they contain.
    only_fig1 = [c for c in chunks if "colony-image://aaaa" in c.text and "colony-image://bbbb" not in c.text]
    only_fig2 = [c for c in chunks if "colony-image://bbbb" in c.text and "colony-image://aaaa" not in c.text]
    both = [c for c in chunks if "colony-image://aaaa" in c.text and "colony-image://bbbb" in c.text]

    assert only_fig1 and only_fig1[0].extra["figure_ids"] == [fig1.figure_id]
    assert only_fig2 and only_fig2[0].extra["figure_ids"] == [fig2.figure_id]
    if both:
        assert set(both[0].extra["figure_ids"]) == {fig1.figure_id, fig2.figure_id}

    # Provenance: every chunk carries ``metadata_origin``.
    assert all(c.extra.get("metadata_origin") == "mistral_ocr" for c in chunks)


def test_prose_chunker_text_only_section_emits_empty_extra() -> None:
    """Text-only sections produce chunks with empty extra — keeps
    pre-design serialised chunks bit-identical."""
    section = ParsedSection(
        text="Para one.\n\nPara two.\n\nPara three.",
        citation=CitationSpan(
            source_uri="file:///plain.txt",
            section_path="",
            char_start=0,
            char_end=64,
        ),
    )
    chunker = ProseChunker(
        ChunkerConfig(min_tokens=1, target_tokens=10, max_tokens=60, overlap_tokens=0),
        token_counter=lambda s: max(1, len(s.split())),
    )
    chunks = chunker.chunk(section)
    assert chunks
    for c in chunks:
        assert c.extra == {}
