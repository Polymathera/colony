"""Pre-commit-callback registry for the design monorepo.

A small generic mechanism that lets process-singleton subscribers
mutate a clone's working tree right before
:func:`_commit_all_and_push` / :func:`_commit_paths_and_push` stage
files. The knowledge layer subscribes its KG snapshot writer so every
commit-and-push captures the current knowledge-graph state alongside
whatever else the action is committing; other subscribers may plug in
to write audit trails, lockfile updates, etc. without touching the
commit helpers themselves.

Callbacks are invoked in registration order. A callback that raises
aborts the commit (the helper re-raises) — surfacing failures loudly
is preferred over silently committing stale derived artefacts.

The registry is a process-wide singleton, so subscriptions made
during deps initialisation reach every commit attempt in the
process. Tests can reset state via :func:`reset_pre_commit_registry`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import DesignMonorepoClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreCommitContext:
    """State passed to every pre-commit callback.

    ``branch`` is the branch the commit will land on (``HEAD`` at
    call time). ``paths`` is the explicit path list the helper was
    invoked with, or ``None`` when the helper is in ``all_changes``
    mode — callbacks that need a strict path scope can inspect this
    to decide whether to extend it.
    """

    client: "DesignMonorepoClient"
    identity: Any
    message: str
    branch: str
    paths: list[Path] | None
    working_dir: Path


PreCommitCallback = Callable[[PreCommitContext], Awaitable[None]]


class PreCommitRegistry:
    """Process-singleton registry of pre-commit callbacks."""

    def __init__(self) -> None:
        self._callbacks: dict[str, PreCommitCallback] = {}

    def register(self, name: str, callback: PreCommitCallback) -> None:
        if name in self._callbacks:
            raise ValueError(
                f"PreCommitRegistry: a callback named {name!r} is already "
                "registered. Unregister first if you want to replace it.",
            )
        self._callbacks[name] = callback

    def unregister(self, name: str) -> None:
        self._callbacks.pop(name, None)

    def names(self) -> tuple[str, ...]:
        return tuple(self._callbacks)

    async def fire_all(self, ctx: PreCommitContext) -> None:
        """Invoke every registered callback in registration order.
        A raising callback aborts the sequence — the exception
        propagates to the commit helper which re-raises."""

        for name, cb in list(self._callbacks.items()):
            try:
                result = cb(ctx)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.error(
                    "PreCommitRegistry: callback %r raised; aborting commit.",
                    name,
                )
                raise


_registry: PreCommitRegistry = PreCommitRegistry()


def get_pre_commit_registry() -> PreCommitRegistry:
    return _registry


def reset_pre_commit_registry() -> None:
    """Test helper: replace the singleton with an empty registry.
    Not for production use."""

    global _registry
    _registry = PreCommitRegistry()


__all__ = (
    "PreCommitCallback",
    "PreCommitContext",
    "PreCommitRegistry",
    "get_pre_commit_registry",
    "reset_pre_commit_registry",
)
