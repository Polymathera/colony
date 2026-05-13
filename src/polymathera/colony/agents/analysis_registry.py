"""Discovery surface for the analysis-type registry.

The legacy hardcoded ``ANALYSIS_REGISTRY`` lives in
``polymathera.colony.cli.polymath`` (with the existing in-source TODO
"This registry should be allowed to be injected using a JSON or
Markdown file."). This module is that injection point: it walks the
``polymathera.analysis_types`` entry-point group at runtime and
returns the union of the colony-builtin entries and any
user-package entries.

Pip-distributable packages register their coordinator analyses by
adding a plugin to their ``pyproject.toml``:

.. code-block:: toml

    [tool.poetry.plugins."polymathera.analysis_types"]
    <analysis_id> = "<module.path>:<factory_callable>"

Each entry-point is a callable returning a dict matching
:class:`polymathera.colony.agents.configs.AnalysisSpec` — the single
source of truth for the analysis-registry schema across every
registration path (this group, the colony-builtin dict, and L1-A's
:func:`polymathera.colony.design_monorepo.extensions.discover_analyses`).
The entry-point's *name* (left side of ``=``) is the analysis key.

Per-program analyses that live in a design monorepo register through
L1-A instead of this group — see ``discover_analyses`` for that path.

Consumers (currently
``polymathera.colony.web_ui.backend.routers.sessions.create_session``)
call :func:`get_analysis_registry` instead of importing the
hardcoded dict directly. The SessionAgent's planner sees the union
in ``metadata.parameters['available_analyses']`` and can dispatch
``AgentPoolCapability.create_agent`` against any registered
coordinator class — colony-builtin or domain-supplied.

Failures isolated: a broken plugin is logged + skipped (including
schema-validation failures from :class:`AnalysisSpec`); the rest of
the registry is unaffected.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from pydantic import ValidationError

from .configs import AnalysisSpec


logger = logging.getLogger(__name__)


ANALYSIS_TYPES_ENTRY_POINT_GROUP = "polymathera.analysis_types"
"""Entry-point group used by :func:`get_analysis_registry`.

Stable name — consumed by domain packages' ``pyproject.toml``.
Don't rename without coordinating with every downstream package
(`polymathera-cps`, future racer / fusion / duv / cami / rocket).
"""


def get_analysis_registry() -> dict[str, dict[str, Any]]:
    """Return the union of colony-builtin and plugin-registered analyses.

    On every call, walks
    ``importlib.metadata.entry_points(group="polymathera.analysis_types")``
    and merges the discovered entries into a fresh copy of the
    hardcoded ``ANALYSIS_REGISTRY``. Plugin entries shadow builtins
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
    from polymathera.colony.cli.polymath import ANALYSIS_REGISTRY

    merged: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in ANALYSIS_REGISTRY.items()
    }

    try:
        eps = entry_points(group=ANALYSIS_TYPES_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — older Python returned a dict
        eps = ()

    for ep in eps:
        try:
            factory = ep.load()
            entry = factory()
        except Exception:  # noqa: BLE001
            logger.exception(
                "analysis_registry: failed to load entry point %r from %r — skipping",
                ep.name, ep.value,
            )
            continue

        if not isinstance(entry, dict):
            logger.warning(
                "analysis_registry: entry-point %r returned %s, expected dict — skipping",
                ep.name, type(entry).__name__,
            )
            continue
        try:
            AnalysisSpec.model_validate(entry)
        except ValidationError as exc:
            logger.warning(
                "analysis_registry: entry-point %r failed schema validation — skipping. %s",
                ep.name, exc,
            )
            continue

        if ep.name in merged:
            logger.warning(
                "analysis_registry: plugin entry %r shadows a colony-builtin entry",
                ep.name,
            )
        merged[ep.name] = entry

    return merged


__all__ = (
    "ANALYSIS_TYPES_ENTRY_POINT_GROUP",
    "get_analysis_registry",
)
