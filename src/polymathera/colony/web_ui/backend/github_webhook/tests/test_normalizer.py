"""Tests for the pure GitHub webhook payload → ``GitHubEventProtocol``
translation. Canned payloads minus the noise — every test pins one
key + value shape so a future field-rename in the value JSON shows
up immediately."""

from __future__ import annotations

from polymathera.colony.agents.blackboard.protocol import GitHubEventProtocol
from polymathera.colony.web_ui.backend.github_webhook.normalizer import (
    HANDLED_EVENT_TYPES,
    normalize,
    normalize_issue_comment,
    normalize_issues,
    normalize_pull_request,
)


# ---------------------------------------------------------------------------
# issues
# ---------------------------------------------------------------------------


def _issues_payload(*, action: str, number: int = 42, state: str = "open") -> dict:
    """Minimum valid ``issues`` webhook payload."""
    return {
        "action": action,
        "issue": {
            "number": number,
            "state": state,
            "title": "found a bug",
            "body": "reproduces every time",
            "user": {"login": "alice"},
            "created_at": "2026-06-01T10:00:00Z",
            "updated_at": "2026-06-01T10:00:00Z",
            "closed_at": (
                "2026-06-01T11:00:00Z" if state == "closed" else None
            ),
            "html_url": f"https://github.com/acme/widgets/issues/{number}",
        },
        "repository": {"full_name": "acme/widgets"},
        "installation": {"id": 12345},
    }


def test_issues_opened_maps_to_issue_opened_key() -> None:
    payload = _issues_payload(action="opened")
    result = normalize_issues(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.issue_opened_key("acme/widgets", 42)
    assert value["change_kind"] == "opened"
    assert value["author_login"] == "alice"
    assert value["title"] == "found a bug"


def test_issues_closed_maps_to_issue_closed_key() -> None:
    payload = _issues_payload(action="closed", state="closed")
    result = normalize_issues(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.issue_closed_key("acme/widgets", 42)
    assert value["change_kind"] == "closed"
    assert value["state"] == "closed"


def test_issues_labeled_rides_on_opened_key_with_change_kind() -> None:
    """Decision C1 — finer-grained issue actions (label/assign/etc.)
    ride on issue_opened_key with the action as ``change_kind``."""

    payload = _issues_payload(action="labeled")
    result = normalize_issues(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.issue_opened_key("acme/widgets", 42)
    assert value["change_kind"] == "labeled"


def test_issues_missing_repo_returns_none() -> None:
    payload = _issues_payload(action="opened")
    del payload["repository"]
    assert normalize_issues(payload) is None


def test_issues_missing_number_returns_none() -> None:
    payload = _issues_payload(action="opened")
    del payload["issue"]["number"]
    assert normalize_issues(payload) is None


# ---------------------------------------------------------------------------
# issue_comment
# ---------------------------------------------------------------------------


def _issue_comment_payload(*, action: str, on_pr: bool = False) -> dict:
    url_path = "pull" if on_pr else "issues"
    return {
        "action": action,
        "issue": {
            "number": 99,
            "html_url": f"https://github.com/acme/widgets/{url_path}/99",
        },
        "comment": {
            "id": 123456,
            "body": "+1",
            "user": {"login": "bob"},
            "created_at": "2026-06-01T12:00:00Z",
            "updated_at": "2026-06-01T12:00:00Z",
        },
        "repository": {"full_name": "acme/widgets"},
        "installation": {"id": 12345},
    }


def test_issue_comment_created_maps_to_issue_commented_key() -> None:
    payload = _issue_comment_payload(action="created")
    result = normalize_issue_comment(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.issue_commented_key("acme/widgets", 99)
    assert value["change_kind"] == "created"
    assert value["comment_id"] == 123456
    assert value["author_login"] == "bob"
    assert value["on_pr"] is False


def test_issue_comment_on_pr_sets_on_pr_flag() -> None:
    payload = _issue_comment_payload(action="created", on_pr=True)
    result = normalize_issue_comment(payload)
    assert result is not None
    _, value = result
    assert value["on_pr"] is True


def test_issue_comment_deleted_change_kind_preserved() -> None:
    payload = _issue_comment_payload(action="deleted")
    result = normalize_issue_comment(payload)
    assert result is not None
    _, value = result
    assert value["change_kind"] == "deleted"


# ---------------------------------------------------------------------------
# pull_request
# ---------------------------------------------------------------------------


def _pr_payload(*, action: str, merged: bool = False, number: int = 100) -> dict:
    state = "closed" if action == "closed" else "open"
    return {
        "action": action,
        "pull_request": {
            "number": number,
            "state": state,
            "title": "fix bug",
            "body": "this fixes it",
            "user": {"login": "carol"},
            "created_at": "2026-06-01T09:00:00Z",
            "updated_at": "2026-06-01T13:00:00Z",
            "closed_at": (
                "2026-06-01T13:00:00Z" if action == "closed" else None
            ),
            "merged": merged,
        },
        "repository": {"full_name": "acme/widgets"},
        "installation": {"id": 12345},
    }


def test_pr_opened_maps_to_pr_opened_key() -> None:
    payload = _pr_payload(action="opened")
    result = normalize_pull_request(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.pr_opened_key("acme/widgets", 100)
    assert value["change_kind"] == "opened"
    assert value["is_pr"] is True
    assert value["merged"] is False


def test_pr_closed_with_merge_maps_to_pr_merged_key() -> None:
    payload = _pr_payload(action="closed", merged=True)
    result = normalize_pull_request(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.pr_merged_key("acme/widgets", 100)
    assert value["merged"] is True
    assert value["change_kind"] == "closed"


def test_pr_closed_without_merge_maps_to_pr_opened_key() -> None:
    """No ``pr_closed_key`` in the protocol; closed-without-merge
    rides on ``pr_opened_key`` with ``change_kind: closed``."""

    payload = _pr_payload(action="closed", merged=False)
    result = normalize_pull_request(payload)
    assert result is not None
    key, value = result
    assert key == GitHubEventProtocol.pr_opened_key("acme/widgets", 100)
    assert value["merged"] is False
    assert value["change_kind"] == "closed"


def test_pr_review_requested_maps_to_pr_review_requested_key() -> None:
    payload = _pr_payload(action="review_requested")
    result = normalize_pull_request(payload)
    assert result is not None
    key, _ = result
    assert key == GitHubEventProtocol.pr_review_requested_key(
        "acme/widgets", 100,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_routes_each_handled_event_type() -> None:
    assert "issues" in HANDLED_EVENT_TYPES
    assert "issue_comment" in HANDLED_EVENT_TYPES
    assert "pull_request" in HANDLED_EVENT_TYPES

    assert normalize("issues", _issues_payload(action="opened")) is not None
    assert normalize(
        "issue_comment", _issue_comment_payload(action="created"),
    ) is not None
    assert normalize("pull_request", _pr_payload(action="opened")) is not None


def test_dispatcher_returns_none_for_unhandled_event_type() -> None:
    """v1 doesn't handle ``ping`` / ``discussion`` — these get
    ``status: ignored`` 200 at the route level."""

    assert normalize("ping", {"zen": "hi"}) is None
    assert normalize("discussion", {"action": "created"}) is None
    assert normalize("workflow_run", {}) is None
