"""Entry-point discovery for extension-supplied ``ConfigComponent``s.

Extensions (e.g. ``polymathera-cps``) register additional config components
by exposing a callable under the ``polymathera.config_components`` entry-point
group::

    [tool.poetry.plugins."polymathera.config_components"]
    cps = "polymathera.cps.config:register_components"

The callable takes no arguments. Its body is expected to import the modules
that declare new ``ConfigComponent`` subclasses (each decorated with
``@register_polymathera_config()``); the side-effect of those imports is
registration into the shared registry. Failures are logged and isolated —
one broken extension does not prevent others from loading.

The single existing legacy group ``polymathera.mission_types`` (consumed by
``agents.mission_registry.get_mission_registry``) is unrelated and remains
in place; both can coexist.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)


CONFIG_COMPONENTS_ENTRY_POINT_GROUP = "polymathera.config_components"


def discover_config_components() -> list[str]:
    """Walk the entry-point group; load each callable; return discovered names.

    Each entry-point's callable runs once. Side-effect: it registers any number
    of ``ConfigComponent`` subclasses with the global registry. The returned
    list is for logging/diagnostics only — callers should not rely on the
    set of names for behaviour.
    """
    try:
        eps = entry_points(group=CONFIG_COMPONENTS_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — older Python returned a dict
        eps = ()

    loaded: list[str] = []
    for ep in eps:
        try:
            register = ep.load()
            register()
        except Exception:
            logger.exception(
                "config_components: failed to load entry point %r from %r — skipping",
                ep.name, ep.value,
            )
            continue
        loaded.append(ep.name)

    if loaded:
        logger.info("config_components: registered extensions %s", loaded)
    return loaded


__all__ = (
    "CONFIG_COMPONENTS_ENTRY_POINT_GROUP",
    "discover_config_components",
)
