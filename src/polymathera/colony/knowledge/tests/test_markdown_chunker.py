"""Tests for :class:`MarkdownChunker`.

The chunker's invariant: fenced code blocks, GFM tables, and display
math (``$$...$$`` or ``\\[...\\]``) are atomic — they MUST NOT be
split mid-block by paragraph packing. Prose between blocks splits
on blank lines like the parent :class:`ProseChunker`.

Also covers:
- An oversized block stays as a single (oversized) chunk rather
  than be torn apart by sentence-splitting — preserving the block's
  semantics is more valuable than respecting ``max_tokens``.
- Code fences nested inside prose paragraphs still split the
  surrounding prose at the right boundaries.
- The chunker is wired into the :class:`Ingestor` for sections with
  ``format="markdown"``; plain-text sections still go through
  :class:`ProseChunker`.
"""

from __future__ import annotations

import pytest

from polymathera.colony.knowledge.chunking import (
    ChunkerConfig,
    MarkdownChunker,
    ProseChunker,
)
from polymathera.colony.knowledge.models import (
    CitationSpan,
    ParsedSection,
)


def _word_tokens(s: str) -> int:
    return max(1, len(s.split()))


def _section(text: str, fmt: str = "markdown") -> ParsedSection:
    return ParsedSection(
        section_path="page-1",
        text=text,
        citation=CitationSpan(
            source_uri="file:///x.pdf",
            section_path="page-1",
            char_start=0,
            char_end=len(text),
            page_number=1,
        ),
        format=fmt,  # type: ignore[arg-type]
    )


def _chunker() -> MarkdownChunker:
    return MarkdownChunker(
        ChunkerConfig(
            min_tokens=1, target_tokens=20, max_tokens=120, overlap_tokens=0,
        ),
        token_counter=_word_tokens,
    )


# ---------------------------------------------------------------------------
# Block-detection unit tests on the splitter
# ---------------------------------------------------------------------------


def test_split_paragraphs_keeps_fenced_code_atomic() -> None:
    text = (
        "Lead-in paragraph.\n\n"
        "```python\n"
        "def f():\n"
        "    return 42\n"
        "```\n\n"
        "Trailing paragraph."
    )
    paras = _chunker()._split_paragraphs(text)
    assert paras[0] == "Lead-in paragraph."
    # The middle paragraph is the entire fenced block (3 lines + fences).
    assert paras[1].startswith("```python")
    assert paras[1].endswith("```")
    assert "def f():" in paras[1]
    assert paras[2] == "Trailing paragraph."


def test_split_paragraphs_handles_tilde_fence() -> None:
    text = "Para.\n\n~~~\nbody\n~~~\n\nAfter."
    paras = _chunker()._split_paragraphs(text)
    assert any(p.startswith("~~~") and p.endswith("~~~") for p in paras)


def test_split_paragraphs_keeps_gfm_table_atomic() -> None:
    text = (
        "Before.\n\n"
        "| Col1 | Col2 |\n"
        "|------|------|\n"
        "| a    | b    |\n"
        "| c    | d    |\n\n"
        "After."
    )
    paras = _chunker()._split_paragraphs(text)
    table = next(p for p in paras if "| Col1 | Col2 |" in p)
    assert "| a    | b    |" in table
    assert "| c    | d    |" in table
    # Table should be a single paragraph element, not split per row.
    assert "After." not in table


def test_split_paragraphs_table_requires_separator_row() -> None:
    """A line with pipes that ISN'T followed by a separator row is
    just prose — don't accidentally swallow it as a one-row table.
    """
    text = "Sentence with a|b inside it.\n\nAnother sentence."
    paras = _chunker()._split_paragraphs(text)
    assert "Sentence with a|b inside it." in paras
    # Two prose paragraphs, no special table treatment.
    assert len(paras) == 2


def test_split_paragraphs_keeps_display_math_dollar_atomic() -> None:
    text = (
        "Set up.\n\n"
        "$$\n"
        "\\sum_{i=1}^n i = \\frac{n(n+1)}{2}\n"
        "$$\n\n"
        "Discussion."
    )
    paras = _chunker()._split_paragraphs(text)
    math_block = next(p for p in paras if p.startswith("$$"))
    assert "\\sum_{i=1}^n" in math_block
    assert math_block.count("$$") == 2  # opener + closer


def test_split_paragraphs_keeps_display_math_bracket_atomic() -> None:
    text = "Lead.\n\n\\[\nF = ma\n\\]\n\nFollow."
    paras = _chunker()._split_paragraphs(text)
    block = next(p for p in paras if p.startswith("\\["))
    assert "F = ma" in block
    assert block.endswith("\\]")


def test_split_paragraphs_unclosed_fence_consumes_to_eof() -> None:
    """If the model forgot the closing fence the chunker still
    treats the rest of the document as one block — better than
    silently re-paragraph-splitting half of a code listing."""
    text = "```\ncode line\nmore code"
    paras = _chunker()._split_paragraphs(text)
    assert paras == ["```\ncode line\nmore code"]


def test_split_paragraphs_back_to_back_blocks() -> None:
    """Two atomic blocks separated only by a blank line emerge as
    two paragraphs — important for page-aware Mistral output that
    interleaves figures + tables tightly."""
    text = (
        "```python\nx = 1\n```\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |"
    )
    paras = _chunker()._split_paragraphs(text)
    assert len(paras) == 2
    assert paras[0].startswith("```python")
    assert paras[1].startswith("| A | B |")


# ---------------------------------------------------------------------------
# End-to-end through ProseChunker.chunk()
# ---------------------------------------------------------------------------


def test_chunk_keeps_oversized_code_block_in_one_chunk() -> None:
    """A code block whose token count exceeds ``target_tokens``
    must still emerge as a single chunk — better an oversized chunk
    than a torn function definition."""
    code = "\n".join(f"line_{i}" for i in range(100))  # 100 word-tokens
    text = f"Intro.\n\n```python\n{code}\n```\n\nOutro."
    chunker = MarkdownChunker(
        ChunkerConfig(
            min_tokens=1, target_tokens=10, max_tokens=200, overlap_tokens=0,
        ),
        token_counter=_word_tokens,
    )
    chunks = chunker.chunk(_section(text))
    code_chunk = next(c for c in chunks if "line_50" in c.text)
    # Code block intact: contains both ends.
    assert "line_0" in code_chunk.text
    assert "line_99" in code_chunk.text
    # Token count exceeds target — that's intentional.
    assert code_chunk.token_count > 10


def test_chunk_splits_prose_around_atomic_blocks() -> None:
    """The blocks themselves are atomic, but surrounding prose
    still splits on blank lines / target tokens."""
    text = (
        "First long paragraph one.\n\n"
        "Second long paragraph two.\n\n"
        "```\nblock\n```\n\n"
        "Third long paragraph three."
    )
    chunker = MarkdownChunker(
        ChunkerConfig(
            min_tokens=1, target_tokens=4, max_tokens=120, overlap_tokens=0,
        ),
        token_counter=_word_tokens,
    )
    chunks = chunker.chunk(_section(text))
    assert len(chunks) >= 4
    # Code block emerges as its own chunk.
    code_chunks = [c for c in chunks if "block" in c.text and "```" in c.text]
    assert len(code_chunks) == 1


def test_chunk_text_section_falls_back_to_prose_chunker_behavior() -> None:
    """When ``format="text"`` (the default for legacy readers), the
    chunker is interchangeable with ``ProseChunker`` — block-aware
    treatment is opt-in via ``format="markdown"``."""
    text = "Para one.\n\nPara two."
    cfg = ChunkerConfig(
        min_tokens=1, target_tokens=3, max_tokens=50, overlap_tokens=0,
    )
    md_chunker = MarkdownChunker(cfg, token_counter=_word_tokens)
    prose_chunker = ProseChunker(cfg, token_counter=_word_tokens)

    # _split_paragraphs is the only behavioural divergence; for plain
    # prose with no blocks, the two splitters MUST produce identical
    # paragraph lists so chunk output stays bit-identical.
    assert md_chunker._split_paragraphs(text) == prose_chunker._split_paragraphs(text)


# ---------------------------------------------------------------------------
# Ingestor integration
# ---------------------------------------------------------------------------


def test_ingestor_picks_markdown_chunker_for_markdown_sections() -> None:
    """The Ingestor delegates to ``self._markdown`` for sections
    whose ``format`` is ``"markdown"``; the singleton sharing test
    confirms config + token counter are the same as ``self._prose``
    so the only behavioural difference is block-atomicity."""
    from polymathera.colony.knowledge.embedder import InMemoryEmbedder
    from polymathera.colony.knowledge.ingestion import Ingestor
    from polymathera.colony.knowledge.stores.image import InMemoryImageStore
    from polymathera.colony.knowledge.stores.vector import InMemoryVectorStore

    ingestor = Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        image_store=InMemoryImageStore(),
    )
    assert isinstance(ingestor._markdown, MarkdownChunker)
    # Shared config + token counter with the prose chunker.
    assert ingestor._markdown._config is ingestor._prose._config
    assert ingestor._markdown._count is ingestor._prose._count
