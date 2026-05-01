"""Discovery surface for the analysis-type registry.

The legacy hardcoded ``ANALYSIS_REGISTRY`` lives in
``polymathera.colony.cli.polymath`` (with the existing in-source TODO
"This registry should be allowed to be injected using a JSON or
Markdown file."). This module is that injection point: it walks the
``polymathera.analysis_types`` entry-point group at runtime and
returns the union of the colony-builtin entries and any
user-package entries.

Domain packages register their coordinator analyses by adding a
plugin to their ``pyproject.toml``:

.. code-block:: toml

    [tool.poetry.plugins."polymathera.analysis_types"]
    opm_meg = "polymathera.cps.domains.quantum.registry:opm_meg_analysis_entry"

Each entry-point is a callable returning a dict in the same shape
as a hardcoded ``ANALYSIS_REGISTRY`` value (see the docstring on
that dict for the schema). The entry-point's *name* (left side of
``=``) is the analysis key — e.g. ``opm_meg``.

Consumers (currently
``polymathera.colony.web_ui.backend.routers.sessions.create_session``)
call :func:`get_analysis_registry` instead of importing the
hardcoded dict directly. The SessionAgent's planner sees the union
in ``metadata.parameters['available_analyses']`` and can dispatch
``AgentPoolCapability.create_agent`` against any registered
coordinator class — colony-builtin or domain-supplied.

Failures isolated: a broken plugin is logged + skipped; the rest
of the registry is unaffected.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any


logger = logging.getLogger(__name__)


ANALYSIS_TYPES_ENTRY_POINT_GROUP = "polymathera.analysis_types"
"""Entry-point group used by :func:`get_analysis_registry`.

Stable name — consumed by domain packages' ``pyproject.toml``.
Don't rename without coordinating with every downstream package
(`polymathera-cps`, future racer / fusion / duv / cami / rocket).
"""


_REQUIRED_REGISTRY_KEYS = (
    "label",
    "description",
)
"""Minimum keys an entry must declare to be accepted.

Matches the keys :func:`polymathera.colony.web_ui.backend.routers.sessions.create_session`
reads from each registry entry when it builds ``available_analyses``.
The ``coordinator_v2`` / ``worker`` keys are not strictly required
to render in the SessionAgent prompt, but the planner needs them
to actually spawn a coordinator — so plugins that omit them get a
warning."""


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
        missing = [k for k in _REQUIRED_REGISTRY_KEYS if k not in entry]
        if missing:
            logger.warning(
                "analysis_registry: entry-point %r missing required keys %r — skipping",
                ep.name, missing,
            )
            continue
        if "coordinator_v2" not in entry and "coordinator_v1" not in entry:
            logger.warning(
                "analysis_registry: entry-point %r has no coordinator_v1 / coordinator_v2 — "
                "the SessionAgent's planner will see it but cannot spawn it",
                ep.name,
            )

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
