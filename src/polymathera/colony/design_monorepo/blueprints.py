"""Composite blueprint helper for the design-monorepo capability trio.

Most agents that touch the design monorepo want all three subclasses
of :class:`_DesignMonorepoCapabilityBase` (read-only state queries,
checkpointing, tool scaffolding). Bind them all in one call instead
of repeating the three ``Capability.bind(...)`` rows in every agent
blueprint.

Usage::

    from polymathera.colony.design_monorepo import (
        design_monorepo_capability_blueprints,
    )

    SessionAgent.bind(
        ...,
        capability_blueprints=[
            ...,
            *design_monorepo_capability_blueprints(),
        ],
    )

By default each capability resolves a per-agent clone path under
``/mnt/shared/agents/<agent_id>/clones/<scope_id>/`` at agent-init
time (see :mod:`design_monorepo.clones`). Callers who want a
specific clone pass ``working_dir=...`` through their own
:meth:`Capability.bind` call instead of using this helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .capabilities import DesignCheckpointer, RepoStateProvider, ToolBuilder

if TYPE_CHECKING:
    from ..agents.blueprint import AgentCapabilityBlueprint


def design_monorepo_capability_blueprints(
    *,
    auto_checkpoint_on_quiescence: bool = True,
    read_only_state: bool = False,
    clone_scope_id: str | None = None,
) -> list["AgentCapabilityBlueprint"]:
    """Build the three design-monorepo blueprints with consistent
    clone-path resolution.

    Args:
        auto_checkpoint_on_quiescence: Forwarded to
            :class:`DesignCheckpointer`. ``True`` (default) tags an
            ``auto_quiescence_<iso8601>`` checkpoint whenever the
            convergence runtime settles with uncommitted changes.
        read_only_state: When ``True``, :class:`RepoStateProvider`
            uses the shared per-node read-only clone instead of a
            private writable per-agent clone. Use this for agents
            that only ever query design state and never mutate.
        clone_scope_id: Optional override for the clone-directory
            scope key. ``None`` lets each capability fall back to its
            own ``scope_id`` so the per-agent path naturally matches
            the capability's blackboard scope.
    """

    return [
        RepoStateProvider.bind(
            clone_scope_id=clone_scope_id,
            read_only=read_only_state,
        ),
        DesignCheckpointer.bind(
            clone_scope_id=clone_scope_id,
            auto_checkpoint_on_quiescence=auto_checkpoint_on_quiescence,
        ),
        ToolBuilder.bind(
            clone_scope_id=clone_scope_id,
        ),
    ]


__all__ = ("design_monorepo_capability_blueprints",)
