"""Discovery surface for the mission-type registry.

The legacy hardcoded ``MISSION_REGISTRY`` lives in
``polymathera.colony.cli.polymath`` (with the existing in-source TODO
"This registry should be allowed to be injected using a JSON or
Markdown file."). This module is that injection point: it walks the
``polymathera.mission_types`` entry-point group at runtime and
returns the union of the colony-builtin entries and any
user-package entries.

Pip-distributable packages register their coordinator missions by
adding a plugin to their ``pyproject.toml``:

.. code-block:: toml

    [tool.poetry.plugins."polymathera.mission_types"]
    <mission_id> = "<module.path>:<factory_callable>"

Each entry-point is a callable returning a dict matching
:class:`polymathera.colony.agents.configs.MissionSpec` — the single
source of truth for the mission-registry schema across every
registration path (this group, the colony-builtin dict, and L1-A's
:func:`polymathera.colony.design_monorepo.extensions.discover_missions`).
The entry-point's *name* (left side of ``=``) is the mission key.

Per-program missions that live in a design monorepo register through
L1-A instead of this group — see ``discover_missions`` for that path.

Consumers (currently
``polymathera.colony.web_ui.backend.routers.sessions.create_session``)
call :func:`get_mission_registry` instead of importing the
hardcoded dict directly. The SessionAgent's planner sees the union
in ``metadata.parameters['available_missions']`` and can dispatch
``AgentPoolCapability.create_agent`` against any registered
coordinator class — colony-builtin or domain-supplied.

Failures isolated: a broken plugin is logged + skipped (including
schema-validation failures from :class:`MissionSpec`); the rest of
the registry is unaffected.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from pydantic import ValidationError

from .configs import MissionSpec


logger = logging.getLogger(__name__)


MISSION_TYPES_ENTRY_POINT_GROUP = "polymathera.mission_types"
"""Entry-point group used by :func:`get_mission_registry`.

Stable name — consumed by domain packages' ``pyproject.toml``.
Don't rename without coordinating with every downstream package
(`polymathera-cps`, future racer / fusion / duv / cami / rocket).
"""


def get_mission_registry() -> dict[str, dict[str, Any]]:
    """Return the union of colony-builtin and plugin-registered missions.

    On every call, walks
    ``importlib.metadata.entry_points(group="polymathera.mission_types")``
    and merges the discovered entries into a fresh copy of the
    hardcoded ``MISSION_REGISTRY``. Plugin entries shadow builtins
    on key collision (last-write-wins) — domain packages can override
    a builtin if they have a strict reason to, but it's a noisy
    operation that the warning makes visible.

    Plugin loading failures (entry-point couldn't import, factory
    raised, factory returned a malformed dict) are logged at WARNING
    and the registry is built from the surviving entries. One bad
    plugin doesn't poison the whole registry.
    """

    # Lazy import to avoid the cli/polymath <-> agents circular
    # import (cli/polymath itself imports from polymathera.colony.agents
    # at module top, so the agents subpackage cannot top-level import
    # cli/polymath).
    from polymathera.colony.cli.polymath import MISSION_REGISTRY

    merged: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in MISSION_REGISTRY.items()
    }

    try:
        eps = entry_points(group=MISSION_TYPES_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — older Python returned a dict
        eps = ()

    for ep in eps:
        try:
            factory = ep.load()
            entry = factory()
        except Exception:  # noqa: BLE001
            logger.exception(
                "mission_registry: failed to load entry point %r from %r — skipping",
                ep.name, ep.value,
            )
            continue

        if not isinstance(entry, dict):
            logger.warning(
                "mission_registry: entry-point %r returned %s, expected dict — skipping",
                ep.name, type(entry).__name__,
            )
            continue
        try:
            MissionSpec.model_validate(entry)
        except ValidationError as exc:
            logger.warning(
                "mission_registry: entry-point %r failed schema validation — skipping. %s",
                ep.name, exc,
            )
            continue

        if ep.name in merged:
            logger.warning(
                "mission_registry: plugin entry %r shadows a colony-builtin entry",
                ep.name,
            )
        merged[ep.name] = entry

    return merged


__all__ = (
    "MISSION_TYPES_ENTRY_POINT_GROUP",
    "get_mission_registry",
)
