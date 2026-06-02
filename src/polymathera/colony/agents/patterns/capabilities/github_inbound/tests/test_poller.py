"""Tests for ``poller.poll_repo`` — the GraphQL → diff → emit chain.

Stub the GitHub client to return a canned GraphQL response and a
stub blackboard to record every ``write`` call. Verify:
- new issues / closed issues / updated issues map to the right
  ``GitHubEventProtocol`` key family per Decision C1
- new comments emit ``issue_commented_key`` writes
- cursor advances to ``max(updated_at)`` we saw
- pre-cursor entries are filtered (no double-emit on re-tick)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from polymathera.colony.agents.blackboard.protocol import GitHubEventProtocol
from polymathera.colony.agents.patterns.capabilities.github_inbound.cursor import (
    Cursor,
)
from polymathera.colony.agents.patterns.capabilities.github_inbound.poller import (
    poll_repo,
)


pytestmark = pytest.mark.asyncio


def _cursor(at: datetime, *, last_seen_id: str | None = None) -> Cursor:
    return Cursor(
        tenant_id="t1", colony_id="c1", repo="acme/widgets",
        channel="issues", last_updated=at,
        last_seen_id=last_seen_id,
    )


def _make_client(payload: dict) -> AsyncMock:
    """Stub ``GitHubClient`` with one canned GraphQL response."""

    client = AsyncMock()
    client.graphql = AsyncMock(return_value=payload)
    return client


def _make_blackboard() -> AsyncMock:
    bb = AsyncMock()
    bb.write = AsyncMock()
    return bb


_T_OLD = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
_T_MID = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_T_NEW = datetime(2026, 6, 1, 18, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def test_new_issue_emits_opened_key() -> None:
    """An issue created within the tick window (createdAt > cursor)
    → ``issue_opened_key`` write with the right value shape."""

    cursor = _cursor(_T_OLD)
    payload = {
        "repository": {
            "issues": {
                "nodes": [
                    {
                        "id": "I_new", "number": 42, "state": "OPEN",
                        "title": "fresh bug", "body": "reproduces every time",
                        "createdAt": _iso(_T_MID),
                        "updatedAt": _iso(_T_MID),
                        "closedAt": None,
                        "author": {"login": "alice"},
                        "comments": {"nodes": []},
                    },
                ],
            },
        },
    }
    client = _make_client(payload)
    bb = _make_blackboard()
    new_updated, new_seen, writes = await poll_repo(
        client=client, blackboard=bb, cursor=cursor,
    )
    assert writes == 1
    bb.write.assert_awaited_once()
    kwargs = bb.write.await_args.kwargs
    assert kwargs["key"] == GitHubEventProtocol.issue_opened_key(
        "acme/widgets", 42,
    )
    assert kwargs["value"]["change_kind"] == "opened"
    assert kwargs["value"]["author_login"] == "alice"
    assert kwargs["value"]["state"] == "open"
    assert new_updated == _T_MID
    assert new_seen == "I_new"


async def test_closed_issue_emits_closed_key() -> None:
    """An issue with ``state=CLOSED`` whose createdAt is BEFORE the
    cursor but updatedAt is AFTER → ``issue_closed_key`` write."""

    cursor = _cursor(_T_OLD)
    payload = {
        "repository": {
            "issues": {
                "nodes": [
                    {
                        "id": "I_old_closed", "number": 7,
                        "state": "CLOSED",
                        "title": "old bug fixed",
                        "body": "", "createdAt": "2026-01-01T00:00:00Z",
                        "updatedAt": _iso(_T_MID),
                        "closedAt": _iso(_T_MID),
                        "author": {"login": "alice"},
                        "comments": {"nodes": []},
                    },
                ],
            },
        },
    }
    client = _make_client(payload)
    bb = _make_blackboard()
    _new_updated, _new_seen, writes = await poll_repo(
        client=client, blackboard=bb, cursor=cursor,
    )
    assert writes == 1
    kwargs = bb.write.await_args.kwargs
    assert kwargs["key"] == GitHubEventProtocol.issue_closed_key(
        "acme/widgets", 7,
    )
    assert kwargs["value"]["change_kind"] == "closed"


async def test_pre_cursor_entries_filtered() -> None:
    """Issues with ``updatedAt <= cursor.last_updated`` are NOT
    emitted — guards against double-emit on re-tick."""

    cursor = _cursor(_T_MID)
    payload = {
        "repository": {
            "issues": {
                "nodes": [
                    {
                        "id": "I_stale", "number": 1, "state": "OPEN",
                        "title": "stale", "body": "",
                        "createdAt": _iso(_T_OLD),
                        "updatedAt": _iso(_T_OLD),
                        "closedAt": None,
                        "author": {"login": "x"},
                        "comments": {"nodes": []},
                    },
                ],
            },
        },
    }
    client = _make_client(payload)
    bb = _make_blackboard()
    _, _, writes = await poll_repo(
        client=client, blackboard=bb, cursor=cursor,
    )
    assert writes == 0
    bb.write.assert_not_awaited()


async def test_new_comment_emits_commented_key() -> None:
    """A comment with ``updatedAt > cursor`` → one
    ``issue_commented_key`` write. The parent issue's ``updatedAt``
    can also be > cursor; in that case BOTH writes fire (one per
    event)."""

    cursor = _cursor(_T_OLD)
    payload = {
        "repository": {
            "issues": {
                "nodes": [
                    {
                        "id": "I_with_comment", "number": 99, "state": "OPEN",
                        "title": "discussion", "body": "",
                        "createdAt": _iso(_T_OLD),  # pre-cursor → updated only
                        "updatedAt": _iso(_T_NEW),
                        "closedAt": None,
                        "author": {"login": "alice"},
                        "comments": {
                            "nodes": [
                                {
                                    "id": "C_new", "body": "fresh comment",
                                    "createdAt": _iso(_T_NEW),
                                    "updatedAt": _iso(_T_NEW),
                                    "author": {"login": "bob"},
                                },
                            ],
                        },
                    },
                ],
            },
        },
    }
    client = _make_client(payload)
    bb = _make_blackboard()
    new_updated, _, writes = await poll_repo(
        client=client, blackboard=bb, cursor=cursor,
    )
    assert writes == 2
    keys = [
        call.kwargs["key"] for call in bb.write.await_args_list
    ]
    assert GitHubEventProtocol.issue_opened_key("acme/widgets", 99) in keys
    assert GitHubEventProtocol.issue_commented_key("acme/widgets", 99) in keys
    assert new_updated == _T_NEW


async def test_cursor_advances_to_max_updated() -> None:
    """Cursor advances to the latest ``updated_at`` seen across all
    issues + comments — even if some are stale."""

    cursor = _cursor(_T_OLD)
    payload = {
        "repository": {
            "issues": {
                "nodes": [
                    {
                        "id": "I_mid", "number": 1, "state": "OPEN",
                        "title": "mid", "body": "",
                        "createdAt": _iso(_T_MID),
                        "updatedAt": _iso(_T_MID),
                        "closedAt": None,
                        "author": {"login": "a"},
                        "comments": {"nodes": []},
                    },
                    {
                        "id": "I_new", "number": 2, "state": "OPEN",
                        "title": "new", "body": "",
                        "createdAt": _iso(_T_NEW),
                        "updatedAt": _iso(_T_NEW),
                        "closedAt": None,
                        "author": {"login": "b"},
                        "comments": {"nodes": []},
                    },
                ],
            },
        },
    }
    client = _make_client(payload)
    bb = _make_blackboard()
    new_updated, new_seen, _writes = await poll_repo(
        client=client, blackboard=bb, cursor=cursor,
    )
    assert new_updated == _T_NEW
    assert new_seen == "I_new"


async def test_repo_404_returns_unchanged_cursor() -> None:
    """``repository: null`` in the GraphQL response (App lacks
    access, or repo deleted) → no writes, cursor unchanged."""

    cursor = _cursor(_T_MID, last_seen_id="I_x")
    payload = {"repository": None}
    client = _make_client(payload)
    bb = _make_blackboard()
    new_updated, new_seen, writes = await poll_repo(
        client=client, blackboard=bb, cursor=cursor,
    )
    assert writes == 0
    assert new_updated == _T_MID
    assert new_seen == "I_x"
    bb.write.assert_not_awaited()


async def test_graphql_called_with_since_iso() -> None:
    """The ``since:`` variable is the cursor's ``last_updated`` in
    ISO-8601 UTC — guards against a future regression where the
    timezone gets dropped + GitHub silently returns nothing."""

    cursor = _cursor(_T_MID)
    payload = {"repository": {"issues": {"nodes": []}}}
    client = _make_client(payload)
    bb = _make_blackboard()
    await poll_repo(client=client, blackboard=bb, cursor=cursor)
    call = client.graphql.await_args
    assert call.kwargs["variables"]["owner"] == "acme"
    assert call.kwargs["variables"]["name"] == "widgets"
    since = call.kwargs["variables"]["since"]
    assert since.startswith("2026-06-01T12:00:00")
    assert "+00:00" in since
