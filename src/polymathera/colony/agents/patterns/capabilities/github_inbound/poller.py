"""GraphQL query + diff-against-cursor + emit logic for one repo tick.

Separated from :mod:`.capability` so the GraphQL/diff/emit chain is
testable against a stubbed ``GitHubClient`` + stubbed blackboard,
without spinning up the full ``AgentCapability`` machinery.

One ``poll_repo`` call ≈ one GraphQL query + one cursor update + N
blackboard writes. The capability's poll loop drives one call per
repo per tick.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from ....blackboard.protocol import GitHubEventProtocol
from .._github.client import GitHubClient, GitHubError, RateLimitError
from .cursor import Cursor


logger = logging.getLogger(__name__)


# GraphQL query: issues + their recent comments in one round-trip.
# ``filterBy.since`` is server-side and uses ISO-8601 — cheaper than
# fetching the full list + filtering client-side. Per-call cap of 100
# issues + 20 comments-per-issue keeps the point cost bounded; if a
# very-active repo exceeds either, the next tick picks up the rest
# because the cursor only advances to ``max(updated_at)`` we observed.
_ISSUES_QUERY = """
query($owner: String!, $name: String!, $since: DateTime!) {
  repository(owner: $owner, name: $name) {
    issues(
      filterBy: {since: $since},
      first: 100,
      orderBy: {field: UPDATED_AT, direction: ASC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id number state title body createdAt updatedAt closedAt
        author { login }
        comments(first: 20, orderBy: {field: UPDATED_AT, direction: ASC}) {
          nodes {
            id body createdAt updatedAt
            author { login }
          }
        }
      }
    }
  }
}
"""


class BlackboardWriter(Protocol):
    """Minimal interface the poller needs from a blackboard.

    Pinned to a Protocol so tests can pass a plain ``MagicMock`` with
    an ``async write(...)`` method instead of standing up an
    ``EnhancedBlackboard``."""

    async def write(self, *, key: str, value: dict[str, Any]) -> Any: ...


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # GraphQL returns "2026-06-01T12:34:56Z".
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _issue_change_kind(
    issue: dict[str, Any], cursor: Cursor,
) -> str:
    """Classify a returned issue into the existing ``GitHubEventProtocol``
    key family.

    Per Decision C1: the protocol's existing keys are
    ``issue_opened`` / ``issue_closed`` / ``issue_commented``. Finer-
    grained actions (label/assign/milestone) ride on ``issue_opened``
    with a ``change_kind`` discriminator in the value JSON — they
    show up here because GitHub bumped the issue's ``updatedAt``.
    """

    created_at = _parse_iso(issue.get("createdAt"))
    state = (issue.get("state") or "").upper()

    if (
        created_at is not None
        and created_at > cursor.last_updated
    ):
        # The issue was created within this tick's window.
        return "opened"
    if state == "CLOSED":
        return "closed"
    return "updated"


async def poll_repo(
    *,
    client: GitHubClient,
    blackboard: BlackboardWriter,
    cursor: Cursor,
) -> tuple[datetime, str | None, int]:
    """Run one GraphQL tick against ``cursor.repo``.

    Returns ``(new_last_updated, new_last_seen_id, writes_emitted)``.
    The caller is responsible for persisting the new cursor — keeping
    the IO concern out of the pure tick logic.

    Behaviour:

    - Issue with ``updatedAt > cursor.last_updated`` → emit one
      ``GitHubEventProtocol`` issue-key write (opened/closed/updated
      per :func:`_issue_change_kind`).
    - Each comment whose ``updatedAt > cursor.last_updated`` →
      emit one ``issue_commented_key`` write.
    - Rate-limit / network errors propagate; the capability's tick
      loop logs + backs off.
    """

    owner, name = cursor.repo.split("/", 1)
    since_iso = cursor.last_updated.astimezone(timezone.utc).isoformat()

    try:
        data = await client.graphql(
            _ISSUES_QUERY,
            variables={"owner": owner, "name": name, "since": since_iso},
        )
    except RateLimitError as exc:
        logger.warning(
            "GitHubInboundPoller: rate-limited on %s: %s",
            cursor.repo, exc,
        )
        raise
    except GitHubError as exc:
        logger.warning(
            "GitHubInboundPoller: GraphQL error on %s: %s",
            cursor.repo, exc,
        )
        raise

    repository = (data or {}).get("repository")
    if not repository:
        logger.warning(
            "GitHubInboundPoller: repository %s not visible (404 or "
            "App lacks access); skipping tick",
            cursor.repo,
        )
        return (cursor.last_updated, cursor.last_seen_id, 0)

    issues = ((repository.get("issues") or {}).get("nodes") or [])
    writes_emitted = 0
    new_last_updated = cursor.last_updated
    new_last_seen_id = cursor.last_seen_id

    for issue in issues:
        issue_number = issue.get("number")
        if issue_number is None:
            continue

        issue_updated = _parse_iso(issue.get("updatedAt"))
        if issue_updated is None:
            continue

        if issue_updated > cursor.last_updated:
            change_kind = _issue_change_kind(issue, cursor)
            if change_kind == "opened":
                key = GitHubEventProtocol.issue_opened_key(
                    cursor.repo, int(issue_number),
                )
            elif change_kind == "closed":
                key = GitHubEventProtocol.issue_closed_key(
                    cursor.repo, int(issue_number),
                )
            else:
                # Decision C1: finer-grained actions ride on the
                # opened key with ``change_kind`` discriminator.
                key = GitHubEventProtocol.issue_opened_key(
                    cursor.repo, int(issue_number),
                )

            value: dict[str, Any] = {
                "repo": cursor.repo,
                "issue_number": int(issue_number),
                "state": (issue.get("state") or "").lower(),
                "title": issue.get("title"),
                "body": issue.get("body"),
                "author_login": (
                    (issue.get("author") or {}).get("login")
                ),
                "created_at": issue.get("createdAt"),
                "updated_at": issue.get("updatedAt"),
                "closed_at": issue.get("closedAt"),
                "change_kind": change_kind,
            }
            await blackboard.write(key=key, value=value)
            writes_emitted += 1

            if issue_updated > new_last_updated:
                new_last_updated = issue_updated
                new_last_seen_id = issue.get("id")

        # Comments — emit per-comment if newer than cursor.
        comments = (
            (issue.get("comments") or {}).get("nodes") or []
        )
        for comment in comments:
            comment_updated = _parse_iso(comment.get("updatedAt"))
            if comment_updated is None:
                continue
            if comment_updated <= cursor.last_updated:
                continue

            comment_key = GitHubEventProtocol.issue_commented_key(
                cursor.repo, int(issue_number),
            )
            await blackboard.write(
                key=comment_key,
                value={
                    "repo": cursor.repo,
                    "issue_number": int(issue_number),
                    "comment_id": comment.get("id"),
                    "body": comment.get("body"),
                    "author_login": (
                        (comment.get("author") or {}).get("login")
                    ),
                    "created_at": comment.get("createdAt"),
                    "updated_at": comment.get("updatedAt"),
                },
            )
            writes_emitted += 1

            if comment_updated > new_last_updated:
                new_last_updated = comment_updated
                # Issue id (not comment id) — the cursor is per-issue
                # because next-tick's GraphQL re-fetches the issue's
                # full comment list and re-filters client-side via
                # the same updated_at check.
                new_last_seen_id = issue.get("id")

    return (new_last_updated, new_last_seen_id, writes_emitted)
