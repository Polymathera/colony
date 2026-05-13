"""L1-A: read the L4 extensions a design monorepo declares under ``.colony/``.

Six surfaces — for each, a single discover function:

- :func:`discover_plugins`     — ``SKILL.md``-shaped skills under the
                                  plugins surface; delegates to
                                  :mod:`polymathera.colony.agents.patterns.capabilities._plugin.discovery`
                                  so the convention is shared with
                                  ``UserPluginCapability``.
- :func:`discover_agents`      — ``Agent`` subclasses declared in
                                  ``*.py`` files under the agents surface.
- :func:`discover_deployments` — classes wrapped with ``@serving.deployment``
                                  in ``*.py`` files under the deployments
                                  surface. Detection is via the
                                  ``__deployment_config__`` attribute the
                                  decorator attaches.
- :func:`discover_tools`       — each ``*.py`` file under the tools surface
                                  may expose a top-level
                                  ``register(registry: ToolRegistry) -> None``
                                  callback; we call it. Adapters whose
                                  ``__init__`` takes arguments register
                                  themselves explicitly inside ``register``.
- :func:`discover_profiles`    — ``*.yaml`` files under the profiles surface,
                                  parsed and keyed by filename stem.
- :func:`discover_missions`    — each ``*.py`` file under the missions surface
                                  exposes a top-level ``mission_entry()``
                                  callable returning a dict matching
                                  :class:`polymathera.colony.agents.configs.MissionSpec`.
                                  The file stem is the mission key. This is
                                  the per-program counterpart to the
                                  ``polymathera.mission_types`` entry-point
                                  group (which is pip-distribution-time).

Surface directories come from :class:`ExtensionsConfig` on the v2 manifest
(or :data:`DEFAULT_SURFACE_DIRS` when the manifest is v1 / no extensions
block / a specific surface is omitted). A surface whose directory does
not exist on disk is empty — the function returns an empty container.

Discovery is best-effort: a single bad file logs at WARNING and is
skipped. Risk #5 in the alignment plan covers the deeper trust model;
it lives at the WRITE side (L1-E, PR 2). PR 1 is read-only.

:func:`discover_all` is the one-shot helper :class:`RepoStateProvider`
calls at init time.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..agents.base import Agent
from ..agents.configs import MissionSpec
from ..agents.patterns.capabilities._plugin.discovery import discover_skills
from ..agents.patterns.capabilities._plugin.schema import SkillSource, SkillSpec
from ..tools.registry import ToolRegistry
from .manifest import DEFAULT_SURFACE_DIRS, DesignMonorepoManifest


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surface-directory resolution
# ---------------------------------------------------------------------------


def resolve_surface_dirs(
    repo_root: Path, manifest: DesignMonorepoManifest | None,
) -> dict[str, Path]:
    """Return the absolute directory for every surface, honouring
    manifest overrides when present.

    The single source of truth for "where does this monorepo's
    <surface> live on disk" — used both by per-surface discoverers AND
    by :class:`RepoStateProvider`'s cache-invalidation fingerprint, so
    the two never disagree about which paths to watch.

    Surface dirs are NOT required to exist on disk — callers handle
    the empty case.
    """
    out: dict[str, Path] = {}
    has_overrides = manifest is not None and manifest.extensions is not None
    for surface, default_rel in DEFAULT_SURFACE_DIRS.items():
        if has_overrides:
            rel = getattr(manifest.extensions, surface).directory
        else:
            rel = default_rel
        out[surface] = (repo_root / rel).resolve()
    return out


def _surface_dir(
    repo_root: Path,
    surface: str,
    manifest: DesignMonorepoManifest | None,
) -> Path:
    """Single-surface variant of :func:`resolve_surface_dirs`."""
    if surface not in DEFAULT_SURFACE_DIRS:
        raise ValueError(f"unknown surface {surface!r}")
    return resolve_surface_dirs(repo_root, manifest)[surface]


# ---------------------------------------------------------------------------
# Per-file Python loading
# ---------------------------------------------------------------------------


def _load_py_module(path: Path) -> Any:
    """Load a single ``.py`` file as a stand-alone module.

    Reads the source directly from disk and executes it into a fresh
    :class:`types.ModuleType`. The deliberate divergence from
    ``importlib.util.spec_from_file_location``: that path consults
    Python's bytecode cache (``__pycache__``), keyed by (source mtime,
    source size). An in-place edit that doesn't change the size and
    lands within the same mtime tick as the prior import would be
    masked. ``_load_py_module`` re-reads the source on every call so
    invalidation always picks up the latest content. We do not insert
    into ``sys.modules`` either — these are discovery-scoped loads.

    Returns ``None`` on any error (logged at WARNING) so callers can
    skip the file without poisoning the rest of the surface.
    """
    try:
        source = path.read_text(encoding="utf-8")
        module = types.ModuleType(f"_l1a_ext_{path.stem}")
        module.__file__ = str(path)
        compiled = compile(source, str(path), "exec")
        exec(compiled, module.__dict__)
        return module
    except Exception as exc:  # noqa: BLE001 — best-effort load
        logger.warning(
            "L1-A: failed to import %s (%s: %s) — skipping",
            path, type(exc).__name__, exc,
        )
        return None


def _own_subclasses(module: Any, base: type) -> list[type]:
    """Classes defined IN ``module`` (not imported) that subclass ``base``.

    Filters out re-exported types so a file that ``from ..agents import
    Agent`` does not register ``Agent`` itself as a discovered agent.
    """
    out: list[type] = []
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type) or obj is base:
            continue
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, base):
            out.append(obj)
    return out


def _own_classes_with_attr(module: Any, attr: str) -> list[type]:
    """Classes defined IN ``module`` carrying ``attr`` — used for the
    ``@serving.deployment`` marker (``__deployment_config__``), which is
    not a subclass relationship but a decorator-attached attribute."""
    out: list[type] = []
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if obj.__module__ != module.__name__:
            continue
        if hasattr(obj, attr):
            out.append(obj)
    return out


# ---------------------------------------------------------------------------
# Per-surface discoverers
# ---------------------------------------------------------------------------


def discover_plugins(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> list[SkillSpec]:
    """Walk the plugins surface for ``SKILL.md``-shaped skills."""
    surface = _surface_dir(repo_root, "plugins", manifest)
    if not surface.is_dir():
        return []
    # The plugins surface lives inside the design monorepo's own
    # workspace tree, so we tag every discovered skill as ``SESSION``-
    # scoped (highest precedence in UserPluginCapability's resolution).
    result = discover_skills([(surface, SkillSource.SESSION)])
    return list(result.skills.values())


def discover_agents(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> dict[str, type[Agent]]:
    """Walk the agents surface, returning ``Agent`` subclasses keyed by
    class name. Caller is responsible for instantiation."""
    surface = _surface_dir(repo_root, "agents", manifest)
    if not surface.is_dir():
        return {}
    out: dict[str, type[Agent]] = {}
    for path in sorted(surface.glob("*.py")):
        module = _load_py_module(path)
        if module is None:
            continue
        for cls in _own_subclasses(module, Agent):
            out[cls.__name__] = cls
    return out


def discover_deployments(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> dict[str, type]:
    """Walk the deployments surface for classes wrapped with
    ``@serving.deployment``. Detection: the decorator attaches a
    ``__deployment_config__`` attribute to the wrapped class. Returned
    dict is keyed by class name."""
    surface = _surface_dir(repo_root, "deployments", manifest)
    if not surface.is_dir():
        return {}
    out: dict[str, type] = {}
    for path in sorted(surface.glob("*.py")):
        module = _load_py_module(path)
        if module is None:
            continue
        for cls in _own_classes_with_attr(module, "__deployment_config__"):
            out[cls.__name__] = cls
    return out


def discover_tools(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> ToolRegistry:
    """Walk the tools surface. Each ``*.py`` file may expose a top-level
    ``register(registry: ToolRegistry) -> None`` callable; we invoke it
    so the file controls its own adapter construction (adapters' ``__init__``
    signatures vary). Files without ``register`` are silently skipped."""
    registry = ToolRegistry()
    surface = _surface_dir(repo_root, "tools", manifest)
    if not surface.is_dir():
        return registry
    for path in sorted(surface.glob("*.py")):
        module = _load_py_module(path)
        if module is None:
            continue
        register = getattr(module, "register", None)
        if not callable(register):
            continue
        try:
            register(registry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "L1-A: %s.register(registry) raised (%s: %s) — skipping",
                path, type(exc).__name__, exc,
            )
    return registry


def discover_profiles(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> dict[str, dict[str, Any]]:
    """Walk the profiles surface, returning parsed YAML payloads keyed
    by filename stem. Each payload must be a top-level mapping; non-
    mappings are logged and skipped."""
    surface = _surface_dir(repo_root, "profiles", manifest)
    if not surface.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(surface.glob("*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "L1-A: failed to read profile %s (%s: %s) — skipping",
                path, type(exc).__name__, exc,
            )
            continue
        if not isinstance(data, dict):
            logger.warning(
                "L1-A: profile %s top-level is %s, expected mapping — skipping",
                path, type(data).__name__,
            )
            continue
        out[path.stem] = data
    return out


def discover_missions(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> dict[str, dict[str, Any]]:
    """Walk the missions surface. Each ``*.py`` file must expose a
    top-level ``mission_entry()`` callable returning a dict matching
    :class:`polymathera.colony.agents.configs.MissionSpec`; the file
    stem becomes the mission key. Files without a callable
    ``mission_entry``, factories that raise, and entries that fail
    :class:`MissionSpec` validation are logged and skipped.

    Shape parity with :func:`polymathera.colony.agents.mission_registry.get_mission_registry`
    is enforced through the same Pydantic model — drift between the
    entry-point-group path and this per-monorepo path surfaces as a
    validation error at load time, not silently."""
    surface = _surface_dir(repo_root, "missions", manifest)
    if not surface.is_dir():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(surface.glob("*.py")):
        module = _load_py_module(path)
        if module is None:
            continue
        factory = getattr(module, "mission_entry", None)
        if not callable(factory):
            logger.warning(
                "L1-A: mission %s exposes no callable mission_entry — skipping",
                path,
            )
            continue
        try:
            entry = factory()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "L1-A: mission_entry() in %s raised (%s: %s) — skipping",
                path, type(exc).__name__, exc,
            )
            continue
        if not isinstance(entry, dict):
            logger.warning(
                "L1-A: mission_entry() in %s returned %s, expected dict — skipping",
                path, type(entry).__name__,
            )
            continue
        try:
            MissionSpec.model_validate(entry)
        except ValidationError as exc:
            logger.warning(
                "L1-A: mission %s failed schema validation — skipping. %s",
                path, exc,
            )
            continue
        out[path.stem] = entry
    return out


# ---------------------------------------------------------------------------
# Bundle — one call site for RepoStateProvider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredExtensions:
    """Snapshot of all six surfaces, populated by :func:`discover_all`.

    Empty containers — not None — are the "surface declared but empty"
    state; absence of the surface directory entirely produces the same
    empty containers, intentionally indistinguishable.
    """

    plugins: list[SkillSpec] = field(default_factory=list)
    agents: dict[str, type[Agent]] = field(default_factory=dict)
    deployments: dict[str, type] = field(default_factory=dict)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    missions: dict[str, dict[str, Any]] = field(default_factory=dict)


def discover_all(
    repo_root: Path, manifest: DesignMonorepoManifest | None = None,
) -> DiscoveredExtensions:
    """Walk every surface; return a :class:`DiscoveredExtensions` snapshot.

    Convenience for :class:`RepoStateProvider`; per-surface tests still
    invoke the individual discoverers.
    """
    return DiscoveredExtensions(
        plugins=discover_plugins(repo_root, manifest),
        agents=discover_agents(repo_root, manifest),
        deployments=discover_deployments(repo_root, manifest),
        tools=discover_tools(repo_root, manifest),
        profiles=discover_profiles(repo_root, manifest),
        missions=discover_missions(repo_root, manifest),
    )


__all__ = (
    "DiscoveredExtensions",
    "discover_agents",
    "discover_all",
    "discover_deployments",
    "discover_missions",
    "discover_plugins",
    "discover_profiles",
    "discover_tools",
    "resolve_surface_dirs",
)
