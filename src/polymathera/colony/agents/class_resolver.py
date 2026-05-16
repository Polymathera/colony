"""Canonical string-to-class resolver for agent / capability spawning.

The agent spawn surface accepts class names as strings — the LLM-driven
:meth:`polymathera.colony.agents.patterns.capabilities.agent_pool.AgentPoolCapability.create_agent`
action, the REST endpoints ``/api/jobs/submit`` + ``/api/agents/spawn``,
and the CLI ``polymath cluster run-mission`` driver all need to turn
strings like ``"polymathera.cps.agents.IntegrationCapability"`` (or
the L4-shaped ``"opm_meg_coordinator.OPMMEGCoordinator"``) into the
actual class object before binding.

This module is the single source of truth for that resolution. Two
paths, tried in order:

1. ``importlib.import_module(<module_path>)`` then ``getattr`` — the
   canonical path for pip-installed classes (colony built-ins,
   CPS L2/L3, third-party-published packages).
2. ``fallback_registry[<class_name>]`` — for L4 classes authored
   under a design monorepo's ``.colony/{agents,deployments}/`` that
   :func:`polymathera.colony.design_monorepo.extensions.discover_agents`
   has loaded into a short-name-keyed dict. L1-A's loader
   deliberately keeps those modules out of ``sys.modules`` (extension
   files are discovery-scoped and re-read per access for mtime-based
   invalidation), so the importlib path cannot find them.

Every caller — CLI, REST endpoints, capability action surfaces, future
spawn flows — delegates here. Drift between paths breaks the L4
discovery story:

- A caller that uses only :func:`importlib.import_module` cannot
  spawn L4 classes.
- A caller that builds its own resolver duplicates the L1-A fallback
  semantics and the two implementations drift out of sync.

If you add a new spawn path, import :func:`resolve_class` here. Do
not re-implement.
"""

from __future__ import annotations

import importlib


def resolve_class(
    fully_qualified_name: str,
    *,
    fallback_registry: dict[str, type] | None = None,
) -> type:
    """Resolve a class from its fully qualified name.

    See module docstring for the dual-path lookup rationale. Raises
    :class:`ValueError` when the input is not a dotted-path string.
    Raises :class:`ImportError` or :class:`AttributeError` when
    neither path can resolve the name (the original importlib
    exception, so the caller sees the precise failure).
    """

    if not isinstance(fully_qualified_name, str) or "." not in fully_qualified_name:
        raise ValueError(
            f"Expected fully qualified class name (e.g., 'pkg.module.Class'), "
            f"got: {fully_qualified_name!r}"
        )
    module_path, class_name = fully_qualified_name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        if fallback_registry is not None:
            cls = fallback_registry.get(class_name)
            if cls is not None:
                return cls
        raise


__all__ = ("resolve_class",)
