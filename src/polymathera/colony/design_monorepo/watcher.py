"""``DesignMonorepoWatcher`` — bridges the design monorepo (C5) to the
convergence runtime (C4).

The design monorepo is the canonical source of design state (master §8
+ ``git_as_design_state_engine.md``). The watcher attaches the two
transports the doc recommends for it (master §5.6 per-source roadmap)
to the convergence runtime's blackboard topic:

- ``LocalFsWatcher`` over the local working tree — picks up changes the
  agents themselves make (a file write from inside the colony fires
  the watcher within ``debounce_s``).
- ``GitRemoteWatcher`` polling the remote — picks up changes other
  Colony nodes pushed (a CFD output committed by an AWS sim node
  arrives within ``poll_interval_s``).

Both transports emit ``PageChangeEvent``s onto the colony's
``vcm:page_events:*`` topic via a ``PageEventPublisher``. The
convergence runtime's forwarder picks them up; capabilities subscribed
through ``ConvergenceCapability`` see them through the normal dispatch
flow.

The watcher is a long-running task: ``start()`` returns once both
transports are running; ``stop()`` cancels them; ``aclose()`` joins.
The doc's third transport (the webhook receiver) is wired in Phase C6
where the HTTP endpoint lives — its ``WebhookEventBuilder`` translator
already exists in ``vcm/watchers/webhook.py`` and uses the same
``PageEventPublisher``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from ..agents.blackboard import EnhancedBlackboard
from ..agents.scopes import BlackboardScope, get_scope_prefix
from ..vcm.page_events import PageChangeEvent
from ..vcm.watchers import (
    GitRemoteWatcher,
    GitRemoteWatcherConfig,
    LocalFsWatcher,
    LocalFsWatcherConfig,
    PageEventPublisher,
)
from .client import DesignMonorepoClient

logger = logging.getLogger(__name__)


class DesignMonorepoWatcher:
    """Long-running watcher attaching a ``DesignMonorepoClient`` to the
    convergence-runtime blackboard topic."""

    def __init__(
        self,
        *,
        client: DesignMonorepoClient,
        app_name: str,
        scope_id: str | None = None,
        local_fs_config: LocalFsWatcherConfig | None = None,
        remote_config: GitRemoteWatcherConfig | None = None,
        watch_remote: bool = True,
        backend_type: str | None = None,
        colony_blackboard: EnhancedBlackboard | None = None,
    ) -> None:
        self._client = client
        self._app_name = app_name
        self._backend_type = backend_type
        self._owned_blackboard = colony_blackboard is None
        self._colony_blackboard = colony_blackboard
        manifest = client.manifest
        self._scope_id = scope_id or f"design_monorepo:{manifest.program}"
        self._source_uri = self._build_source_uri(client)

        self._local_fs = LocalFsWatcher(
            root=client.working_dir,
            scope_id=self._scope_id,
            source_uri=self._source_uri,
            config=local_fs_config or LocalFsWatcherConfig(
                # Tag working-tree changes with a coarse data_type so
                # subscribers can filter without inspecting the path.
                data_type="design_monorepo_file",
            ),
        )
        self._remote: GitRemoteWatcher | None = None
        if watch_remote:
            cfg = remote_config or GitRemoteWatcherConfig(
                branch=manifest.default_branch,
                data_type="design_monorepo_file",
            )
            self._remote = GitRemoteWatcher(
                repo_path=client.working_dir,
                scope_id=self._scope_id,
                source_uri=self._source_uri,
                config=cfg,
            )

        self._publisher: PageEventPublisher | None = None
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def scope_id(self) -> str:
        return self._scope_id

    @property
    def source_uri(self) -> str:
        return self._source_uri

    async def start(self) -> None:
        """Start both watch loops. Idempotent."""

        if self._tasks:
            return
        if self._colony_blackboard is None:
            self._colony_blackboard = EnhancedBlackboard(
                app_name=self._app_name,
                scope_id=get_scope_prefix(BlackboardScope.COLONY),
                backend_type=self._backend_type,
            )
            await self._colony_blackboard.initialize()
        self._publisher = PageEventPublisher(
            self._colony_blackboard, source_id=self._scope_id,
        )
        self._tasks.append(
            asyncio.create_task(
                self._run_watch(self._local_fs.watch()),
                name=f"design-monorepo-watcher.local:{self._scope_id}",
            )
        )
        if self._remote is not None:
            self._tasks.append(
                asyncio.create_task(
                    self._run_watch(self._remote.watch()),
                    name=f"design-monorepo-watcher.remote:{self._scope_id}",
                )
            )

    async def stop(self) -> None:
        self._local_fs.stop()
        if self._remote is not None:
            self._remote.stop()
        for t in self._tasks:
            if not t.done():
                t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()
        if self._colony_blackboard is not None and self._owned_blackboard:
            try:
                await self._colony_blackboard.stop()
            except Exception:  # noqa: BLE001
                logger.exception("DesignMonorepoWatcher: blackboard stop failed")
            self._colony_blackboard = None
        self._publisher = None

    async def aclose(self) -> None:
        await self.stop()

    async def __aenter__(self) -> "DesignMonorepoWatcher":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    # ---- Internals -----------------------------------------------------

    async def _run_watch(
        self, source: AsyncIterator[PageChangeEvent],
    ) -> None:
        try:
            async for event in source:
                if self._publisher is None:
                    return
                await self._publisher.publish(event)
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.exception(
                "DesignMonorepoWatcher: watch loop crashed for %s",
                self._scope_id,
            )

    @staticmethod
    def _build_source_uri(client: DesignMonorepoClient) -> str:
        manifest = client.manifest
        try:
            sha = client.repo.head.commit.hexsha
        except Exception:  # noqa: BLE001
            sha = "HEAD"
        return f"git:{manifest.design_repo_url}:{manifest.default_branch}:{sha}"


__all__ = ("DesignMonorepoWatcher",)
