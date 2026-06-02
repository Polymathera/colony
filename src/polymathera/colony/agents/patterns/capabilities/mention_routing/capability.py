"""``MentionRoutingCapability`` — colony-singleton mention parser.

Mounted on the system session's ``SessionAgent`` (P8-0 foundation).
Subscribes to every ``github:*`` write on the colony-scoped
blackboard, runs the mention parser on the value's ``body`` field,
and emits one :class:`MentionEventProtocol` write per matched
``@colony`` / ``@polymath`` handle.

The capability is intentionally observation-only in v1 — emitting
the event makes mentions queryable (the InteractionLog write-through
mirrors ``mention:*`` writes to Postgres) and surfaceable in future
dashboards. Actual response-to-mention behaviour (LLM-judge intent,
post a comment back via ``GitHubCapability``) belongs to a separate
phase that needs more architectural choices (which agent owns the
reply identity? per-handle dispatcher? etc.) than fit in P10 v1.
"""

from __future__ import annotations

import logging
from typing import Any

from ...events import event_handler
from ....blackboard.protocol import (
    GitHubEventProtocol,
    MentionEventProtocol,
)
from ..colony_singleton_base import ColonySingletonCapabilityBase
from .parser import parse_mentions


logger = logging.getLogger(__name__)


class MentionRoutingCapability(ColonySingletonCapabilityBase):
    """Colony-singleton mention parser.

    Inherits scope_id derivation + no-op suspension methods from
    :class:`ColonySingletonCapabilityBase`. Adds only the
    ``github:*`` event handler.
    """

    # ------------------------------------------------------------------
    # Parser — subscribed to every GitHub event with a body field
    # ------------------------------------------------------------------

    @event_handler(pattern="github:*")
    async def _on_github_event(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Walk the value's ``body`` for ``@colony`` / ``@polymath``
        mentions; emit one :class:`MentionEventProtocol` per match.

        Best-effort per-write: a single bad payload (missing body,
        malformed value, blackboard write failure) is logged + the
        handler returns. The next event fires normally.
        """

        key = event.key
        value = event.value or {}

        body = value.get("body")
        if not isinstance(body, str) or not body:
            return

        mentions = parse_mentions(body)
        if not mentions:
            return

        repo = value.get("repo")
        issue_number = value.get("issue_number")
        if not isinstance(repo, str) or not isinstance(issue_number, int):
            # Defensive: the P8a poller + P9 webhook normalizers
            # always populate both, but a future emitter might not.
            return

        # ``comment_id`` is present on comment events; absent on
        # issue/PR body events. Match the protocol's optional shape.
        comment_id = value.get("comment_id")
        if not isinstance(comment_id, int):
            comment_id = None

        # GitHub HTML URL — best-effort. Comment events don't carry
        # html_url in the P8a/P9 value shape; PR/issue events neither.
        # For v1 we synthesize a deep-link from repo + number.
        html_url = (
            f"https://github.com/{repo}/issues/{issue_number}"
            if comment_id is None
            else f"https://github.com/{repo}/issues/{issue_number}"
            f"#issuecomment-{comment_id}"
        )

        try:
            blackboard = await self._get_colony_blackboard()
        except Exception:  # noqa: BLE001
            logger.exception(
                "MentionRoutingCapability: failed to acquire colony "
                "blackboard while processing %s; dropping %d mention(s)",
                key, len(mentions),
            )
            return

        for mention in mentions:
            mention_key = MentionEventProtocol.event_key(
                repo, issue_number, comment_id,
            )
            payload: dict[str, Any] = {
                "mention_kind": mention.handle,
                "repo": repo,
                "issue_number": issue_number,
                "comment_id": comment_id,
                "commenter_login": value.get("author_login"),
                "body": body,
                "html_url": html_url,
                "source_github_key": key,
                "mention_offset": mention.offset,
            }
            try:
                await blackboard.write(
                    mention_key, payload,
                    tags={"mention", "github_inbound"},
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "MentionRoutingCapability: blackboard.write failed "
                    "for mention=%s repo=%s#%s — dropping",
                    mention.handle, repo, issue_number,
                )

        logger.info(
            "MentionRoutingCapability: %s → %d mention(s) emitted",
            key, len(mentions),
        )
