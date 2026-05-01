"""Watcher implementations for ``ContextPageSource.watch()``.

Per master §5.6, three watch transports cover the field:

- ``LocalFsWatcher`` — ``watchdog`` filesystem watcher with debouncing,
  for sources whose backing store is a local clone (the design
  monorepo, an external repo cloned for analysis, a corpus directory).
- ``GitRemoteWatcher`` — periodic ``git fetch`` + ``git diff
  --name-only`` against a local clone, for the remote-driven
  push-fallback story when no webhook is available.
- ``SourcePollWatcher`` — generic interval poll wrapping any source's
  ``get_all_mapped_records`` snapshot; surfaces additions, removals,
  modifications as the corresponding ``PageChangeEvent`` kinds.

Plus a payload-to-events translator:

- ``WebhookEventBuilder`` — turns a Gitea / GitLab / GitHub git-push
  webhook payload into a sequence of ``PageChangeEvent``s. The HTTP
  endpoint that receives the webhook lives in the Web UI layer and is
  wired in Phase C6; the translator itself belongs here so the
  watcher contract stays in one place.

Each watcher publishes events onto the ``vcm:page_events:*`` topic on
the colony scope so the convergence runtime picks them up through the
forwarder.
"""

from __future__ import annotations

from .git_remote import GitRemoteWatcher, GitRemoteWatcherConfig
from .local_fs import LocalFsWatcher, LocalFsWatcherConfig
from .publisher import PageEventPublisher
from .source_poll import SourcePollWatcher, SourcePollWatcherConfig
from .webhook import WebhookEventBuilder


__all__ = (
    "GitRemoteWatcher",
    "GitRemoteWatcherConfig",
    "LocalFsWatcher",
    "LocalFsWatcherConfig",
    "PageEventPublisher",
    "SourcePollWatcher",
    "SourcePollWatcherConfig",
    "WebhookEventBuilder",
)
