"""Pure GitHub webhook payload → ``GitHubEventProtocol`` translation.

Separated from the route + publisher so the mapping logic can be
unit-tested against canned payloads with no FastAPI / Postgres /
blackboard / cluster connection in the loop.

P9 v1 covers three event types per design doc §10:

- ``issues`` (opened / closed / reopened / edited / assigned /
  unassigned / labeled / unlabeled / milestoned / demilestoned)
- ``issue_comment`` (created / edited / deleted)
- ``pull_request`` (opened / closed / reopened / synchronize /
  review_requested)

Discussions + ``pull_request_review_comment`` deferred to a
follow-up (the receiver returns 200 with ``{"status": "ignored"}``
for unhandled event types — no error).

Per Decision C1 (locked in P8a §5), finer-grained issue actions
(``assigned`` / ``labeled`` / etc.) ride on the existing
``issue_opened_key`` with a ``change_kind`` discriminator in the
value JSON rather than getting their own protocol key families.
The poller emits the same shape so subscribers don't have to
branch on poll-vs-webhook.
"""

from __future__ import annotations

from typing import Any

from polymathera.colony.agents.blackboard.protocol import GitHubEventProtocol


# Event types the receiver writes to the blackboard. Anything not in
# this set is logged + ignored at the route level. Keep narrow — the
# design doc §20 explicitly out-of-scopes wiki / release ingestion.
HANDLED_EVENT_TYPES: frozenset[str] = frozenset({
    "issues",
    "issue_comment",
    "pull_request",
})


def _author(payload_obj: dict[str, Any] | None) -> str | None:
    if not payload_obj:
        return None
    user = payload_obj.get("user") or {}
    return user.get("login")


def _repo_full_name(payload: dict[str, Any]) -> str | None:
    """``owner/repo`` from a GitHub webhook payload. Every event type
    we handle carries it on the same ``repository.full_name`` field."""

    repo = payload.get("repository") or {}
    full = repo.get("full_name")
    return full if isinstance(full, str) and "/" in full else None


def normalize_issues(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Normalize an ``X-GitHub-Event: issues`` payload.

    The ``action`` field discriminates the lifecycle event:
    - ``opened`` → ``issue_opened_key`` + ``change_kind: opened``
    - ``closed`` → ``issue_closed_key`` + ``change_kind: closed``
    - Everything else (``reopened`` / ``edited`` / ``assigned`` /
      ``labeled`` / etc.) → ``issue_opened_key`` with the action as
      ``change_kind`` (Decision C1 — ride on the opened key family).

    Returns ``(key, value)`` or ``None`` if the payload is malformed
    (missing repo, issue, etc.). Pure — no side effects."""

    action = payload.get("action")
    issue = payload.get("issue") or {}
    repo = _repo_full_name(payload)
    if not repo or not isinstance(action, str):
        return None
    number = issue.get("number")
    if not isinstance(number, int):
        return None

    if action == "closed":
        key = GitHubEventProtocol.issue_closed_key(repo, number)
    else:
        # ``opened`` AND all secondary actions land here per C1.
        key = GitHubEventProtocol.issue_opened_key(repo, number)

    value = {
        "repo": repo,
        "issue_number": number,
        "state": issue.get("state"),
        "title": issue.get("title"),
        "body": issue.get("body"),
        "author_login": _author(issue),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "change_kind": action,
    }
    return key, value


def normalize_issue_comment(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Normalize an ``X-GitHub-Event: issue_comment`` payload.

    Maps every comment lifecycle action (created / edited / deleted)
    to ``issue_commented_key`` with the action as ``change_kind``.
    Both true comments AND PR review comments fire ``issue_comment``
    when posted on the PR's main thread; in either case the
    ``issue.number`` is the right id.
    """

    action = payload.get("action")
    issue = payload.get("issue") or {}
    comment = payload.get("comment") or {}
    repo = _repo_full_name(payload)
    if not repo or not isinstance(action, str):
        return None
    number = issue.get("number")
    if not isinstance(number, int):
        return None

    key = GitHubEventProtocol.issue_commented_key(repo, number)
    value = {
        "repo": repo,
        "issue_number": number,
        "comment_id": comment.get("id"),
        "body": comment.get("body"),
        "author_login": _author(comment),
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "change_kind": action,
        # ``html_url`` on the issue payload distinguishes PR comments
        # from issue comments (PRs have ``/pull/`` in the URL).
        "on_pr": "/pull/" in (issue.get("html_url") or ""),
    }
    return key, value


def normalize_pull_request(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Normalize an ``X-GitHub-Event: pull_request`` payload.

    Maps:
    - ``opened`` / ``reopened`` / ``synchronize`` / etc. →
      ``pr_opened_key``
    - ``closed`` with ``merged: true`` → ``pr_merged_key``
    - ``closed`` without merge → ``pr_opened_key`` with
      ``change_kind: closed`` (the protocol has no pr_closed key)
    - ``review_requested`` → ``pr_review_requested_key``
    """

    action = payload.get("action")
    pr = payload.get("pull_request") or {}
    repo = _repo_full_name(payload)
    if not repo or not isinstance(action, str):
        return None
    number = pr.get("number")
    if not isinstance(number, int):
        return None

    if action == "closed" and pr.get("merged"):
        key = GitHubEventProtocol.pr_merged_key(repo, number)
    elif action == "review_requested":
        key = GitHubEventProtocol.pr_review_requested_key(repo, number)
    else:
        key = GitHubEventProtocol.pr_opened_key(repo, number)

    value = {
        "repo": repo,
        "issue_number": number,  # PRs are issues server-side
        "state": pr.get("state"),
        "title": pr.get("title"),
        "body": pr.get("body"),
        "author_login": _author(pr),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "closed_at": pr.get("closed_at"),
        "merged": bool(pr.get("merged")),
        "change_kind": action,
        "is_pr": True,
    }
    return key, value


_NORMALIZERS = {
    "issues": normalize_issues,
    "issue_comment": normalize_issue_comment,
    "pull_request": normalize_pull_request,
}


def normalize(
    event_type: str, payload: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Dispatch on ``X-GitHub-Event``. Returns ``None`` for event
    types we don't handle in v1 — the route surfaces that as
    ``{"status": "ignored"}`` 200."""

    fn = _NORMALIZERS.get(event_type)
    if fn is None:
        return None
    return fn(payload)
