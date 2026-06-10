"""Tests for :func:`_merge_small_sections` — the Ingestor's pre-chunk
pack-small-adjacent-sections step that closes the 6.7× cost multiplier
observed on a 97KB design doc on 2026-06-09 (see
``colony/load_design_context_and_retry_storm_plan.md``)."""

from __future__ import annotations

from polymathera.colony.knowledge.ingestion import _merge_small_sections
from polymathera.colony.knowledge.models import CitationSpan, ParsedSection


def _section(
    *,
    section_path: str,
    heading: str,
    body: str,
    source_uri: str = "doc:test",
    char_start: int = 0,
    char_end: int | None = None,
    fmt: str = "markdown",
) -> ParsedSection:
    text = f"{heading}\n\n{body}" if heading and body else heading or body
    return ParsedSection(
        section_path=section_path,
        heading=heading,
        text=text,
        format=fmt,
        citation=CitationSpan(
            source_uri=source_uri,
            section_path=section_path,
            char_start=char_start,
            char_end=char_end if char_end is not None else char_start + len(text),
        ),
    )


def _tokens(s: str) -> int:
    return max(1, len(s.split()))


def test_two_small_siblings_under_target_merge_into_one() -> None:
    sections = (
        _section(
            section_path="1/1", heading="Sub one", body="alpha beta gamma",
            char_start=0,
        ),
        _section(
            section_path="1/2", heading="Sub two", body="delta epsilon zeta",
            char_start=40, char_end=80,
        ),
    )
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    assert len(merged) == 1
    assert "Sub one" in merged[0].text
    assert "Sub two" in merged[0].text
    # citation spans both sections' char ranges
    assert merged[0].citation.char_start == 0
    assert merged[0].citation.char_end == 80


def test_change_of_top_level_segment_breaks_pack() -> None:
    sections = (
        _section(section_path="1/1", heading="A1", body="foo"),
        _section(section_path="1/2", heading="A2", body="bar"),
        _section(section_path="2/1", heading="B1", body="baz"),
    )
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    assert len(merged) == 2
    assert "A1" in merged[0].text and "A2" in merged[0].text
    assert "B1" in merged[1].text


def test_section_larger_than_target_passes_through_unchanged() -> None:
    sections = (
        _section(
            section_path="1", heading="Huge",
            body=" ".join("word"+str(i) for i in range(500)),
        ),
    )
    merged = _merge_small_sections(
        sections, target_tokens=50, token_count=_tokens,
    )
    assert len(merged) == 1
    assert merged[0] is sections[0]  # passed through without copy


def test_pack_breaks_when_next_section_would_exceed_target() -> None:
    sections = (
        _section(section_path="1/1", heading="A", body="x " * 30),
        _section(section_path="1/2", heading="B", body="y " * 30),
        _section(section_path="1/3", heading="C", body="z " * 30),
    )
    # Each ~31 tokens (heading + body words). Target 80 fits two but
    # not three.
    merged = _merge_small_sections(
        sections, target_tokens=80, token_count=_tokens,
    )
    assert len(merged) == 2


def test_different_source_uris_never_merge() -> None:
    sections = (
        _section(section_path="1", heading="A", body="alpha", source_uri="docA"),
        _section(section_path="1", heading="B", body="beta", source_uri="docB"),
    )
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    assert len(merged) == 2


def test_different_formats_never_merge() -> None:
    sections = (
        _section(section_path="1/1", heading="A", body="x", fmt="markdown"),
        _section(section_path="1/2", heading="B", body="y", fmt="text"),
    )
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    assert len(merged) == 2


def test_empty_input_returns_empty() -> None:
    assert _merge_small_sections((), target_tokens=200, token_count=_tokens) == ()


def test_one_section_passes_through_unchanged() -> None:
    sections = (_section(section_path="1", heading="Only", body="some text"),)
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    assert merged == sections


def test_merged_text_uses_double_heading_prefix_for_each_sub() -> None:
    """Inside the merged ``text``, each merged sub-section's heading
    is prefixed with ``## `` so the downstream chunker (and LLM that
    consumes the chunk) can still see the structural boundaries."""

    sections = (
        _section(section_path="1/1", heading="First sub", body="alpha"),
        _section(section_path="1/2", heading="Second sub", body="beta"),
    )
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    assert "## First sub" in merged[0].text
    assert "## Second sub" in merged[0].text
    assert "alpha" in merged[0].text
    assert "beta" in merged[0].text


def test_merge_collapses_94_tiny_sections_into_few() -> None:
    """End-to-end shape: 94 tiny adjacent sections all under the
    same top-level path collapse to roughly ``ceil(total / target)``
    merged sections. Pins the 6.7× cost-multiplier fix.

    The pre-fix Ingestor would have called the chunker 94 times, each
    on a tiny section, producing 200+ chunks via the overlap bug.
    Post-fix: ``_merge_small_sections`` collapses them first, then
    the chunker (with the overlap fix) emits roughly target-sized
    chunks. Together: tens of chunks, not hundreds."""

    sections = tuple(
        _section(
            section_path=f"1/{i}", heading=f"Sub {i}",
            body=f"body text {i}",
        )
        for i in range(94)
    )
    merged = _merge_small_sections(
        sections, target_tokens=200, token_count=_tokens,
    )
    # Each tiny section ~5 tokens; target 200 should pack ~40 per
    # merged section → ~3 merged sections.
    assert 1 <= len(merged) <= 10
    # Every input section's body text shows up in some merged output.
    joined = "\n".join(s.text for s in merged)
    for i in range(94):
        assert f"body text {i}" in joined
