"""``GitRemoteWatcher`` — periodic ``git fetch`` + ``git diff --name-only``.

Implements master §5.6 transport (3) — the poll-fallback story when no
webhook is configured. The watcher:

1. Fetches the configured remote at a fixed interval.
2. Compares the remote-tracking ref against the previously-observed
   commit.
3. For each path in the diff, emits a ``PageChangeEvent`` (kind
   inferred from ``A``/``M``/``D``).

The watcher does *not* update the working tree; that is the caller's
job. (For the design monorepo, the caller pulls / fast-forwards
explicitly via the ``DesignMonorepoClient``.) Emitting the events
without touching the working tree means downstream subscribers see the
remote-side change and can decide what to do.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..page_events import PageChangeEvent

if TYPE_CHECKING:
    from git import Repo

logger = logging.getLogger(__name__)


@dataclass
class GitRemoteWatcherConfig:
    """Tuning for ``GitRemoteWatcher``."""

    poll_interval_s: float = 30.0
    """Interval between ``git fetch`` calls. 30 s is the doc's
    recommended default for the 'no webhook' case."""

    remote_name: str = "origin"
    branch: str = "main"
    data_type: str | None = None


class GitRemoteWatcher:
    """Watch a git remote by polling ``ls-remote`` + ``fetch`` + diff."""

    static = False

    def __init__(
        self,
        *,
        repo_path: Path,
        scope_id: str,
        source_uri: str,
        config: GitRemoteWatcherConfig | None = None,
    ) -> None:
        self._repo_path = Path(repo_path).resolve()
        self._scope_id = scope_id
        self._source_uri = source_uri
        self._config = config or GitRemoteWatcherConfig()
        self._stopped = asyncio.Event()
        self._last_observed_sha: str | None = None

    @property
    def scope_id(self) -> str:
        return self._scope_id

    def stop(self) -> None:
        self._stopped.set()

    async def watch(self) -> AsyncIterator[PageChangeEvent]:
        from git import GitCommandError, Repo

        repo = Repo(str(self._repo_path))
        # Initialise last-observed from the current remote-tracking ref.
        ref = self._remote_tracking_ref()
        try:
            self._last_observed_sha = repo.git.rev_parse(ref).strip()
        except GitCommandError:
            self._last_observed_sha = None

        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(
                    self._stopped.wait(),
                    timeout=self._config.poll_interval_s,
                )
                break
            except asyncio.TimeoutError:
                pass
            try:
                async for event in self._poll_once(repo):
                    yield event
            except Exception:  # noqa: BLE001
                logger.exception(
                    "GitRemoteWatcher: poll iteration failed for %s",
                    self._repo_path,
                )

    def _remote_tracking_ref(self) -> str:
        return f"refs/remotes/{self._config.remote_name}/{self._config.branch}"

    async def _poll_once(self, repo: "Repo") -> AsyncIterator[PageChangeEvent]:
        from git import GitCommandError

        # Fetch in a worker thread; GitPython is synchronous.
        try:
            await asyncio.to_thread(
                repo.git.fetch,
                self._config.remote_name,
                self._config.branch,
                "--quiet",
            )
        except GitCommandError as exc:
            logger.info(
                "GitRemoteWatcher: fetch failed (%s); will retry next interval.",
                exc,
            )
            return
        ref = self._remote_tracking_ref()
        try:
            new_sha = repo.git.rev_parse(ref).strip()
        except GitCommandError:
            return
        if self._last_observed_sha is None:
            self._last_observed_sha = new_sha
            return
        if new_sha == self._last_observed_sha:
            return
        try:
            name_status = repo.git.diff(
                "--name-status", "-z", self._last_observed_sha, new_sha,
            )
        except GitCommandError:
            self._last_observed_sha = new_sha
            return
        events = self._parse_name_status(name_status, new_sha)
        self._last_observed_sha = new_sha
        for event in events:
            yield event

    def _parse_name_status(
        self, name_status_z: str, new_sha: str,
    ) -> list[PageChangeEvent]:
        out: list[PageChangeEvent] = []
        tokens = [t for t in name_status_z.split("\0") if t]
        i = 0
        source = f"{self._source_uri}@{new_sha}"
        while i < len(tokens):
            status = tokens[i]
            i += 1
            if not status:
                continue
            kind_letter = status[0]
            if kind_letter in ("R", "C"):
                if i + 1 >= len(tokens):
                    break
                old_path, new_path = tokens[i], tokens[i + 1]
                i += 2
                # Treat rename/copy as delete-old + add-new for change
                # propagation purposes.
                out.append(
                    PageChangeEvent.page_invalidated(
                        page_id=f"file:{old_path}",
                        source=source,
                        reason="renamed/copied to a new path",
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": old_path, "new_path": new_path},
                    )
                )
                out.append(
                    PageChangeEvent.page_added(
                        page_id=f"file:{new_path}",
                        source=source,
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": new_path, "from_rename": old_path},
                    )
                )
                continue
            if i >= len(tokens):
                break
            path = tokens[i]
            i += 1
            page_id = f"file:{path}"
            if kind_letter == "A":
                out.append(
                    PageChangeEvent.page_added(
                        page_id=page_id,
                        source=source,
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": path},
                    )
                )
            elif kind_letter == "D":
                out.append(
                    PageChangeEvent.page_invalidated(
                        page_id=page_id,
                        source=source,
                        reason="source file deleted on remote",
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": path},
                    )
                )
            else:
                out.append(
                    PageChangeEvent.page_replaced(
                        old_page_id=page_id,
                        new_page_id=page_id,
                        source=source,
                        data_type=self._config.data_type,
                        scope_id=self._scope_id,
                        extra={"relative_path": path},
                    )
                )
        return out


__all__ = ("GitRemoteWatcher", "GitRemoteWatcherConfig")
