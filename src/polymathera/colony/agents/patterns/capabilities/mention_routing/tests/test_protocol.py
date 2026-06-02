"""Tests for ``MentionEventProtocol`` — key shape + round-trip."""

from __future__ import annotations

import pytest

from polymathera.colony.agents.blackboard.protocol import MentionEventProtocol


def test_event_key_with_comment_id() -> None:
    """Comment-borne mention → comment id baked into the key tail."""

    key = MentionEventProtocol.event_key("acme/widgets", 42, 99999)
    assert key == "mention:acme__widgets:42:99999"


def test_event_key_without_comment_id_uses_zero() -> None:
    """Issue/PR-body mention → no comment id → ``0`` placeholder so
    the colon-count of the key stays fixed at three."""

    key = MentionEventProtocol.event_key("acme/widgets", 7)
    assert key == "mention:acme__widgets:7:0"


def test_event_pattern_wildcard() -> None:
    assert MentionEventProtocol.event_pattern() == "mention:*"


def test_event_pattern_for_repo_encodes_slash() -> None:
    assert MentionEventProtocol.event_pattern_for_repo(
        "acme/widgets",
    ) == "mention:acme__widgets:*"


def test_parse_event_key_round_trip() -> None:
    key = MentionEventProtocol.event_key("acme/widgets", 42, 99999)
    parsed = MentionEventProtocol.parse_event_key(key)
    assert parsed == {
        "repo": "acme/widgets",
        "issue_number": "42",
        "comment_id": "99999",
    }


def test_parse_event_key_round_trip_no_comment() -> None:
    key = MentionEventProtocol.event_key("acme/widgets", 7)
    parsed = MentionEventProtocol.parse_event_key(key)
    assert parsed == {
        "repo": "acme/widgets",
        "issue_number": "7",
        "comment_id": "0",
    }


def test_parse_event_key_rejects_alien_key() -> None:
    with pytest.raises(ValueError, match="Not a MentionEvent"):
        MentionEventProtocol.parse_event_key("github:issue_opened:r:1")


def test_parse_event_key_rejects_malformed_tail() -> None:
    """Missing one of the three tail segments → ValueError. Pin the
    failure shape so a future regex tweak doesn't silently widen the
    accepted set."""

    with pytest.raises(ValueError, match="Malformed"):
        MentionEventProtocol.parse_event_key("mention:acme__widgets:42")
