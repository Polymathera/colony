"""Tests for ``mention_routing.parser`` — the pure regex surface."""

from __future__ import annotations

from polymathera.colony.agents.patterns.capabilities.mention_routing.parser import (
    parse_mentions,
)


def test_bare_colony_matches() -> None:
    """``@colony`` at start of a word boundary matches; the handle
    captured is ``colony``."""

    [m] = parse_mentions("hey @colony please look at this")
    assert m.handle == "colony"
    assert m.offset == 4  # index of the '@'


def test_bare_polymath_matches() -> None:
    [m] = parse_mentions("@polymath thoughts?")
    assert m.handle == "polymath"
    assert m.offset == 0


def test_named_colony_handle_captured_with_suffix() -> None:
    """``@colony-roadmap`` captures the full ``colony-roadmap``
    handle — the suffix is preserved so a future per-handle
    dispatcher can branch on it."""

    [m] = parse_mentions("ping @colony-roadmap about milestone 3")
    assert m.handle == "colony-roadmap"


def test_named_polymath_handle_captured_with_suffix() -> None:
    [m] = parse_mentions("@polymath-experiments could you run X?")
    assert m.handle == "polymath-experiments"


def test_email_address_does_not_match() -> None:
    """The leading ``\\B`` (non-word-boundary before ``@``) prevents
    matching when ``@`` is preceded by a word char — so emails like
    ``foo@colony.com`` are NOT mentions."""

    assert parse_mentions("contact: alice@colony.com") == []
    assert parse_mentions("see notes@polymath.io") == []


def test_handle_at_end_of_string_matches() -> None:
    """``\\b`` after the handle works at end-of-string (string end
    is a word boundary)."""

    [m] = parse_mentions("cc @colony")
    assert m.handle == "colony"


def test_no_match_in_unrelated_text() -> None:
    assert parse_mentions("just a normal sentence about github") == []


def test_empty_or_none_body_returns_empty() -> None:
    assert parse_mentions(None) == []
    assert parse_mentions("") == []


def test_multiple_mentions_preserved_in_order() -> None:
    """Two mentions in one body → two ``ParsedMention`` entries in
    source-text order. Duplicates preserved."""

    text = (
        "@colony please review, then @polymath approve, then "
        "@colony again to deploy"
    )
    mentions = parse_mentions(text)
    assert [m.handle for m in mentions] == [
        "colony", "polymath", "colony",
    ]
    # Offsets strictly increasing (left-to-right).
    assert mentions[0].offset < mentions[1].offset < mentions[2].offset


def test_handle_in_middle_of_word_does_not_match() -> None:
    """Word-boundary after handle: ``@colonyfoo`` (no separator) is
    NOT a match — protects against e.g. accidental concatenation."""

    assert parse_mentions("@colonyfoo bar") == []
    assert parse_mentions("@polymathx") == []


def test_handle_with_hyphen_then_trailing_text_separated_by_space() -> None:
    """``@colony-foo bar`` captures ``colony-foo``; the boundary is
    at the space."""

    [m] = parse_mentions("@colony-foo bar")
    assert m.handle == "colony-foo"


def test_capitalization_is_case_sensitive() -> None:
    """The regex matches lowercase ``colony`` / ``polymath`` only.
    ``@Colony`` is NOT a match in v1 — GitHub usernames are
    case-insensitive, but we treat the handles as exact strings.
    Document the choice (a future change-of-mind is one regex flag away)."""

    assert parse_mentions("@Colony hi") == []
    assert parse_mentions("@Polymath hi") == []
