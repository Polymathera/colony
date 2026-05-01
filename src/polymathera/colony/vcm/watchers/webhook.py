"""``WebhookEventBuilder`` — translate a git-push webhook payload into events.

Implements the *event-builder* half of master §5.6 transport (2). The
HTTP endpoint that receives the webhook (``POST
/api/v1/vcm/git_push_event``) lives in the Web UI's backend router and
is wired in Phase C6; this module is the *payload parser* the
endpoint hands its body to. Keeping the parser here lets:

- The convergence-runtime tests cover all the format variants
  (Gitea / GitLab / GitHub) without needing the HTTP layer.
- The endpoint stay a thin shim that delegates to this builder.

Each provider's payload has the same essential information — a
sequence of commits between ``before`` and ``after``, plus a list of
modified paths — but the JSON shape differs. The builder normalises
across them.

Verified-payload-handling (HMAC-SHA256 signature verification etc.) is
the endpoint's responsibility, not this module's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from ..page_events import PageChangeEvent

logger = logging.getLogger(__name__)


class WebhookEventBuilder:
    """Translate a git-push webhook payload into ``PageChangeEvent``s.

    Use:

    .. code-block:: python

        builder = WebhookEventBuilder(
            scope_id="program-1",
            source_uri="git:git@example/foo.git",
        )
        events = builder.build(payload, provider="github")
    """

    SUPPORTED_PROVIDERS: tuple[str, ...] = ("github", "gitlab", "gitea", "generic")

    def __init__(self, *, scope_id: str, source_uri: str, data_type: str | None = None) -> None:
        self._scope_id = scope_id
        self._source_uri = source_uri
        self._data_type = data_type

    def build(
        self,
        payload: Mapping[str, Any],
        *,
        provider: str = "generic",
    ) -> list[PageChangeEvent]:
        """Parse ``payload`` and return the implied page events."""

        provider = provider.lower()
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown webhook provider {provider!r}; "
                f"supported: {', '.join(self.SUPPORTED_PROVIDERS)}.",
            )
        commits = self._extract_commits(payload, provider)
        after_sha = self._extract_after_sha(payload, provider)
        added, modified, removed = self._collect_paths(commits)
        events: list[PageChangeEvent] = []
        source = (
            f"{self._source_uri}@{after_sha}" if after_sha else self._source_uri
        )
        for path in sorted(added):
            events.append(
                PageChangeEvent.page_added(
                    page_id=f"file:{path}",
                    source=source,
                    data_type=self._data_type,
                    scope_id=self._scope_id,
                    extra={"relative_path": path, "provider": provider},
                ),
            )
        for path in sorted(modified):
            page_id = f"file:{path}"
            events.append(
                PageChangeEvent.page_replaced(
                    old_page_id=page_id,
                    new_page_id=page_id,
                    source=source,
                    data_type=self._data_type,
                    scope_id=self._scope_id,
                    extra={"relative_path": path, "provider": provider},
                ),
            )
        for path in sorted(removed):
            events.append(
                PageChangeEvent.page_invalidated(
                    page_id=f"file:{path}",
                    source=source,
                    reason="source file deleted on remote",
                    data_type=self._data_type,
                    scope_id=self._scope_id,
                    extra={"relative_path": path, "provider": provider},
                ),
            )
        return events

    # ---- Provider-specific extraction ---------------------------------

    @staticmethod
    def _extract_commits(
        payload: Mapping[str, Any], provider: str,
    ) -> list[Mapping[str, Any]]:
        # GitHub / Gitea / GitLab all carry a top-level ``commits`` array.
        commits = payload.get("commits")
        if isinstance(commits, list):
            return [c for c in commits if isinstance(c, dict)]
        return []

    @staticmethod
    def _extract_after_sha(
        payload: Mapping[str, Any], provider: str,
    ) -> str | None:
        if provider in ("github", "gitea", "generic"):
            after = payload.get("after")
            if isinstance(after, str):
                return after
        if provider == "gitlab":
            after = payload.get("after") or payload.get("checkout_sha")
            if isinstance(after, str):
                return after
        return None

    @staticmethod
    def _collect_paths(
        commits: Sequence[Mapping[str, Any]],
    ) -> tuple[set[str], set[str], set[str]]:
        added: set[str] = set()
        modified: set[str] = set()
        removed: set[str] = set()
        for c in commits:
            for p in c.get("added") or ():
                if isinstance(p, str):
                    added.add(p)
            for p in c.get("modified") or ():
                if isinstance(p, str):
                    modified.add(p)
            for p in c.get("removed") or ():
                if isinstance(p, str):
                    removed.add(p)
        # If the same path appears in multiple categories across commits,
        # the latest-wins convention is approximated by precedence:
        # removed > added > modified (the path is gone now).
        modified -= added
        modified -= removed
        added -= removed
        return added, modified, removed


__all__ = ("WebhookEventBuilder",)
