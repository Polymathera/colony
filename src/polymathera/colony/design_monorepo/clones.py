"""Per-agent and shared clone-path resolution for design-monorepo
capabilities.

Layout under ``/mnt/shared`` (the colony-shared docker volume; survives
Ray actor restarts):

- ``/mnt/shared/agents/<agent_id>/clones/<scope_id>/``  per-agent
  writable clone. Each agent gets its own private working tree so
  branches and uncommitted edits do not collide across agents in the
  same process or across nodes.

- ``/mnt/shared/shared_clones/<scope_id>/``  one read-only clone per
  node, shared by every ``RepoStateProvider(read_only=True)`` on
  that node. Kept up-to-date by ``GitRemoteWatcher`` events feeding
  the VCM mapping; capabilities only read from it.

Callers should always go through :func:`resolve_clone_path` rather
than constructing paths inline so the layout stays canonical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base import Agent


logger = logging.getLogger(__name__)


SHARED_ROOT = Path("/mnt/shared")
"""Base of the colony-shared docker volume. The mount is created by
``docker-compose.yml``; production overrides via the
``COLONY_SHARED_ROOT`` env var if needed."""

PER_AGENT_SUBDIR = "agents"
SHARED_CLONES_SUBDIR = "shared_clones"


def shared_root() -> Path:
    """Resolve the colony-shared root, honouring an optional
    ``COLONY_SHARED_ROOT`` override (useful for local dev / tests).
    """

    import os

    override = os.environ.get("COLONY_SHARED_ROOT")
    return Path(override) if override else SHARED_ROOT


def resolve_clone_path(
    *,
    agent: "Agent | None",
    scope_id: str,
    read_only: bool = False,
) -> Path:
    """Pick the clone directory for a design-monorepo capability.

    Args:
        agent: Owning agent. Required for the per-agent path.
            ``None`` is only valid when ``read_only=True``.
        scope_id: Per-clone scope key. Typically the design monorepo's
            VCM scope_id, the colony id, or the design alternative
            branch name.
        read_only: ``True`` selects the shared per-node read-only
            clone; ``False`` selects the per-agent writable clone.

    Raises:
        ValueError: when ``read_only=False`` and ``agent`` is ``None``,
            since a writable clone must be agent-owned.
    """

    base = shared_root()
    if read_only:
        return base / SHARED_CLONES_SUBDIR / scope_id
    if agent is None:
        raise ValueError(
            "resolve_clone_path: writable clones require an owning "
            "agent. Pass read_only=True for the shared read-only clone.",
        )
    return base / PER_AGENT_SUBDIR / agent.agent_id / "clones" / scope_id


__all__ = (
    "PER_AGENT_SUBDIR",
    "SHARED_CLONES_SUBDIR",
    "SHARED_ROOT",
    "resolve_clone_path",
    "shared_root",
)
