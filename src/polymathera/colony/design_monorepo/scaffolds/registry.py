"""Process-singleton registry of L2-F extension scaffolds.

L1-E (PR 2) ships *blank* templates: an agent scaffold produces
``class {name}(Agent): pass``. CPS (and any third-party extension
package) registers *domain-shaped* scaffold variants here at startup,
via the ``polymathera.config_components`` entry-point group's
``register_components`` hook.

The bootstrap actions on :class:`~polymathera.colony.design_monorepo.capabilities.ToolBuilder`
accept an optional ``scaffold`` parameter. When omitted, the L1-E
blank template renders (today's behavior). When set to a registered
``scaffold_id``, the corresponding registered template renders
instead — typically producing a subclass of a CPS L2-B base wired
with framework-specific imports and bindings.

One mechanism, one registry. Adding a new scaffold is a single
``register_extension_scaffold(...)`` call from any package's
``register_components()`` body — no parallel discovery path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..manifest import DEFAULT_SURFACE_DIRS


@dataclass(frozen=True)
class ExtensionScaffold:
    """One registered scaffold variant.

    ``scaffold_id`` is the stable identifier callers pass to
    :class:`ToolBuilder`'s ``bootstrap_<surface>`` action's
    ``scaffold`` parameter. ``surface`` must be one of the keys of
    :data:`DEFAULT_SURFACE_DIRS`; the registry validates this at
    registration so a mismatched binding fails at startup rather
    than at first use.

    ``template_path`` is the on-disk path to the ``string.Template``
    source file the scaffold renders. ``required_vars`` documents
    the names the caller must supply via the action's
    ``template_vars`` parameter (in addition to the standard
    ``name`` / ``name_snake`` / ``name_dash`` the renderer always
    supplies).
    """

    scaffold_id: str
    surface: str
    template_path: Path
    required_vars: frozenset[str] = frozenset()


_REGISTRY: dict[str, ExtensionScaffold] = {}


class ExtensionScaffoldRegistryError(RuntimeError):
    """Raised when a scaffold cannot be registered (duplicate id,
    unknown surface, missing template file) or looked up."""


def register_extension_scaffold(scaffold: ExtensionScaffold) -> None:
    """Add ``scaffold`` to the registry.

    Validates at registration time so misconfiguration is a startup
    error, not a render-time one:

    - ``scaffold_id`` must be unique across the whole process. CPS
      registering the same id Colony already declared (or two
      extension packages colliding) raises.
    - ``surface`` must match one of :data:`DEFAULT_SURFACE_DIRS`
      keys (otherwise no bootstrap action can target it).
    - ``template_path`` must exist on disk.
    """
    if scaffold.scaffold_id in _REGISTRY:
        raise ExtensionScaffoldRegistryError(
            f"scaffold_id {scaffold.scaffold_id!r} is already registered "
            f"(targeting surface "
            f"{_REGISTRY[scaffold.scaffold_id].surface!r})",
        )
    if scaffold.surface not in DEFAULT_SURFACE_DIRS:
        raise ExtensionScaffoldRegistryError(
            f"scaffold {scaffold.scaffold_id!r} targets surface "
            f"{scaffold.surface!r}, which is not in DEFAULT_SURFACE_DIRS "
            f"({sorted(DEFAULT_SURFACE_DIRS)})",
        )
    if not scaffold.template_path.is_file():
        raise ExtensionScaffoldRegistryError(
            f"scaffold {scaffold.scaffold_id!r} declares template at "
            f"{scaffold.template_path} but the file does not exist",
        )
    _REGISTRY[scaffold.scaffold_id] = scaffold


def get_extension_scaffold(scaffold_id: str) -> ExtensionScaffold:
    """Look up a registered scaffold by id. Raises when unregistered."""
    try:
        return _REGISTRY[scaffold_id]
    except KeyError:
        raise ExtensionScaffoldRegistryError(
            f"no scaffold registered with id {scaffold_id!r}; "
            f"available: {sorted(_REGISTRY)}",
        ) from None


def available_scaffolds(
    surface: str | None = None,
) -> tuple[ExtensionScaffold, ...]:
    """List every registered scaffold, optionally filtered by surface.
    Returned in registration order so CLI listings stay stable."""
    if surface is None:
        return tuple(_REGISTRY.values())
    return tuple(s for s in _REGISTRY.values() if s.surface == surface)


def reset_registry() -> None:
    """Drop every registered scaffold. For tests only — production
    only mutates the registry from ``register_components()`` at
    startup."""
    _REGISTRY.clear()


__all__ = (
    "ExtensionScaffold",
    "ExtensionScaffoldRegistryError",
    "available_scaffolds",
    "get_extension_scaffold",
    "register_extension_scaffold",
    "reset_registry",
)
