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

Plus a composite + a payload-to-events translator:

- ``CompositeWatcher`` — merges N child watchers into one cancellable
  async iterator. Used when a single source needs more than one watch
  transport against the same backing store (e.g.,
  ``GitRepoContextPageSource`` couples a ``LocalFsWatcher`` and a
  ``GitRemoteWatcher`` against the cloned working tree).
- ``WebhookEventBuilder`` — turns a Gitea / GitLab / GitHub git-push
  webhook payload into a sequence of ``PageChangeEvent``s. The HTTP
  endpoint that receives the webhook lives in the Web UI layer and is
  wired in Phase C6; the translator itself belongs here so the
  watcher contract stays in one place.

Watchers expose events through their own ``watch()`` async iterator;
``VirtualContextManager`` drains the iterator and feeds events
straight into ``ConvergenceRuntimeDeployment.feed_page_event``
(KERNEL-ring path) — there is no intermediate blackboard topic.
"""

from __future__ import annotations

from .composite import CompositeWatcher
from .git_remote import GitRemoteWatcher, GitRemoteWatcherConfig
from .local_fs import LocalFsWatcher, LocalFsWatcherConfig
from .source_poll import SourcePollWatcher, SourcePollWatcherConfig
from .webhook import WebhookEventBuilder


__all__ = (
    "CompositeWatcher",
    "GitRemoteWatcher",
    "GitRemoteWatcherConfig",
    "LocalFsWatcher",
    "LocalFsWatcherConfig",
    "SourcePollWatcher",
    "SourcePollWatcherConfig",
    "WebhookEventBuilder",
)
