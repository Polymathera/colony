"""Tests for ``MentionRoutingCapability._on_github_event``.

Stubs the colony blackboard so the emit path can be verified
without spinning up Redis. Mirrors P8b's
``test_capability.py`` shape — direct construction (catches the
abstract-method / signature / scope_id failures up front per the
P8b lesson), no ``from_blueprint`` mocking."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    GitHubEventProtocol,
    MentionEventProtocol,
)
from polymathera.colony.agents.blackboard.types import BlackboardEvent
from polymathera.colony.agents.patterns.capabilities.mention_routing.capability import (
    MentionRoutingCapability,
)


pytestmark = pytest.mark.asyncio


def _make_event(key: str, value: Any) -> BlackboardEvent:
    return BlackboardEvent(event_type="write", key=key, value=value)


def _make_capability_with_stub_blackboard() -> tuple[
    MentionRoutingCapability, AsyncMock,
]:
    """Detached capability + AsyncMock blackboard plumbed via the
    base-class ``_get_colony_blackboard`` accessor."""

    cap = MentionRoutingCapability(agent=None, scope_id="test_scope")
    bb = AsyncMock()
    bb.write = AsyncMock()

    async def _stub_get_colony_blackboard():
        return bb

    cap._get_colony_blackboard = _stub_get_colony_blackboard  # type: ignore[assignment]
    return cap, bb


# ---------------------------------------------------------------------------
# Emit path
# ---------------------------------------------------------------------------


async def test_comment_event_with_colony_mention_emits_one_event() -> None:
    """A ``github:issue_commented`` write whose ``body`` contains
    ``@colony`` → one ``MentionEventProtocol`` write with the right
    key shape + ``mention_kind=colony``."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 42)
    value = {
        "repo": "acme/widgets",
        "issue_number": 42,
        "comment_id": 99999,
        "body": "hey @colony please look at this",
        "author_login": "alice",
    }
    await cap._on_github_event(_make_event(key, value), None)

    bb.write.assert_awaited_once()
    call = bb.write.await_args
    written_key = call.args[0] if call.args else call.kwargs["key"]
    written_value = call.args[1] if len(call.args) > 1 else call.kwargs.get(
        "value"
    )
    assert written_key == MentionEventProtocol.event_key(
        "acme/widgets", 42, 99999,
    )
    assert written_value["mention_kind"] == "colony"
    assert written_value["commenter_login"] == "alice"
    assert written_value["repo"] == "acme/widgets"
    assert written_value["issue_number"] == 42
    assert written_value["comment_id"] == 99999
    assert written_value["source_github_key"] == key


async def test_issue_body_with_polymath_mention_uses_zero_comment_id() -> None:
    """Mention in an issue BODY (not a comment) → ``comment_id=0`` in
    the key + ``None`` in the value payload."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_opened_key("acme/widgets", 7)
    value = {
        "repo": "acme/widgets",
        "issue_number": 7,
        # No comment_id field for issue body events.
        "body": "@polymath we need a doc for this",
        "author_login": "bob",
    }
    await cap._on_github_event(_make_event(key, value), None)

    bb.write.assert_awaited_once()
    call = bb.write.await_args
    written_key = call.args[0]
    written_value = call.args[1]
    assert written_key == "mention:acme__widgets:7:0"
    assert written_value["mention_kind"] == "polymath"
    assert written_value["comment_id"] is None


async def test_multiple_mentions_emit_multiple_events() -> None:
    """Two mentions in one body → two blackboard writes (preserves
    the parser's source-order semantics)."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 1)
    value = {
        "repo": "acme/widgets",
        "issue_number": 1,
        "comment_id": 5,
        "body": "@colony review then @polymath approve",
        "author_login": "carol",
    }
    await cap._on_github_event(_make_event(key, value), None)
    assert bb.write.await_count == 2
    kinds = [
        c.args[1]["mention_kind"] for c in bb.write.await_args_list
    ]
    assert kinds == ["colony", "polymath"]


async def test_named_handle_preserved_in_payload() -> None:
    """``@colony-roadmap`` captures the full handle on the emitted
    event's ``mention_kind`` — guards against a future regex tweak
    that accidentally strips the suffix."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 1)
    value = {
        "repo": "acme/widgets",
        "issue_number": 1,
        "comment_id": 5,
        "body": "ping @colony-roadmap about milestone 3",
        "author_login": "carol",
    }
    await cap._on_github_event(_make_event(key, value), None)
    bb.write.assert_awaited_once()
    assert bb.write.await_args.args[1]["mention_kind"] == "colony-roadmap"


# ---------------------------------------------------------------------------
# No-emit paths
# ---------------------------------------------------------------------------


async def test_no_mention_in_body_skips_emit() -> None:
    """Plain comment body → zero writes (the capability is purely
    additive; events without mentions are dropped)."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 1)
    value = {
        "repo": "acme/widgets",
        "issue_number": 1,
        "comment_id": 5,
        "body": "looks good, ship it",
        "author_login": "carol",
    }
    await cap._on_github_event(_make_event(key, value), None)
    bb.write.assert_not_awaited()


async def test_missing_body_field_skips_emit() -> None:
    """Value missing the ``body`` field (e.g. a future event shape
    that doesn't carry one) → silent no-op, no emit, no crash."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_closed_key("acme/widgets", 1)
    value = {
        "repo": "acme/widgets",
        "issue_number": 1,
        # no body
    }
    await cap._on_github_event(_make_event(key, value), None)
    bb.write.assert_not_awaited()


async def test_email_in_body_is_not_a_mention() -> None:
    """End-to-end check that the parser's ``\\B`` guard reaches the
    capability — an email in the body doesn't trigger an emit."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 1)
    value = {
        "repo": "acme/widgets",
        "issue_number": 1,
        "comment_id": 5,
        "body": "cc alice@colony.com",
        "author_login": "carol",
    }
    await cap._on_github_event(_make_event(key, value), None)
    bb.write.assert_not_awaited()


async def test_malformed_payload_skips_silently() -> None:
    """Body has a mention but missing ``repo`` / ``issue_number``
    fields → silent skip rather than crash. Defensive against a
    future emitter shape change."""

    cap, bb = _make_capability_with_stub_blackboard()
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 1)
    value = {
        # no repo, no issue_number
        "body": "@colony please",
        "author_login": "carol",
    }
    await cap._on_github_event(_make_event(key, value), None)
    bb.write.assert_not_awaited()
