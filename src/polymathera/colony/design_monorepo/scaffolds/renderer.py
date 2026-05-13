"""Render a scaffold into a target directory under the design monorepo.

Scaffolds are stored on disk as plain files alongside this module. Each
template directory holds:

- ``manifest.json`` — opt-in list of files in the scaffold (so the
  renderer skips any caches or ``__pycache__/`` produced by tests).
- The template files themselves, each subject to ``string.Template``
  substitution (``$name`` / ``${name}``). ``manifest.json`` controls
  which files are textual and which are byte-copied.

The renderer is intentionally restricted: no Jinja, no nested template
calls, no shell hooks. Anything more sophisticated belongs in the
tool-building agent's action policy, not in a static scaffold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from string import Template

from ..manifest import DEFAULT_SURFACE_DIRS


_THIS_DIR = Path(__file__).resolve().parent
_EXTENSION_TEMPLATES_DIR = _THIS_DIR / "monorepo_extensions"


AVAILABLE_TEMPLATES: tuple[str, ...] = (
    "python_lib",
    "c_library",
    "julia_module",
    "rust_crate",
    "cmake_project",
)


# Maps each L1-A surface to the on-disk template that L1-E renders into
# ``.colony/<surface>/``. Surface set must mirror
# :data:`polymathera.colony.design_monorepo.manifest.DEFAULT_SURFACE_DIRS`
# — the assertion below fails loudly at import time if the two drift.
_EXTENSION_TEMPLATE_FILE_BY_SURFACE: dict[str, str] = {
    "plugins": "plugin.SKILL.md",
    "agents": "agent.py.tmpl",
    "deployments": "deployment.py.tmpl",
    "tools": "tool_adapter.py.tmpl",
    "profiles": "profile.yaml.tmpl",
    "analyses": "analysis.py.tmpl",
}
if set(_EXTENSION_TEMPLATE_FILE_BY_SURFACE) != set(DEFAULT_SURFACE_DIRS):
    raise ImportError(
        "scaffolds.renderer: L1-E extension-template map is out of sync "
        f"with DEFAULT_SURFACE_DIRS — templates declare "
        f"{sorted(_EXTENSION_TEMPLATE_FILE_BY_SURFACE)}, surfaces declare "
        f"{sorted(DEFAULT_SURFACE_DIRS)}",
    )


class ScaffoldRenderError(RuntimeError):
    """Raised when a scaffold cannot be materialised."""


class _TemplateManifest:
    """Loaded ``manifest.json`` for a scaffold.

    Two lists:

    - ``text_files`` — files subject to ``$variable`` substitution.
    - ``binary_files`` — files copied verbatim.
    """

    def __init__(self, text_files: list[str], binary_files: list[str]) -> None:
        self.text_files = text_files
        self.binary_files = binary_files

    @classmethod
    def load(cls, template_root: Path) -> "_TemplateManifest":
        manifest_path = template_root / "manifest.json"
        if not manifest_path.is_file():
            raise ScaffoldRenderError(
                f"Scaffold {template_root.name!r} is missing manifest.json.",
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScaffoldRenderError(
                f"Scaffold {template_root.name!r} has malformed manifest.json: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise ScaffoldRenderError(
                f"Scaffold {template_root.name!r} manifest.json must be an object.",
            )
        text_files = list(payload.get("text_files", []))
        binary_files = list(payload.get("binary_files", []))
        if not text_files and not binary_files:
            raise ScaffoldRenderError(
                f"Scaffold {template_root.name!r} manifest.json declares no files.",
            )
        return cls(text_files, binary_files)


def _template_root(name: str) -> Path:
    if name not in AVAILABLE_TEMPLATES:
        raise ScaffoldRenderError(
            f"Unknown scaffold template {name!r}; "
            f"available: {', '.join(AVAILABLE_TEMPLATES)}.",
        )
    root = _THIS_DIR / name
    if not root.is_dir():
        raise ScaffoldRenderError(
            f"Scaffold {name!r} is registered but its directory is missing.",
        )
    return root


def list_template_files(template: str) -> tuple[str, ...]:
    """Return the relative paths of every file the scaffold ships."""
    root = _template_root(template)
    manifest = _TemplateManifest.load(root)
    return tuple(sorted(manifest.text_files + manifest.binary_files))


def _default_vars(
    *,
    name: str,
    purpose: str,
    license_id: str,
    description: str,
    template_vars: Mapping[str, str] | None,
) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    out: dict[str, str] = {
        "name": name,
        "name_snake": name.replace("-", "_"),
        "name_dash": name.replace("_", "-"),
        "purpose": purpose,
        "license": license_id,
        "description": description,
        "year": str(now.year),
        "iso_date": now.date().isoformat(),
        "author": "Polymathera Colony",
    }
    if template_vars:
        out.update({str(k): str(v) for k, v in template_vars.items()})
    return out


def render_template(
    template: str,
    target_dir: Path,
    *,
    name: str,
    purpose: str,
    license_id: str,
    description: str = "",
    template_vars: Mapping[str, str] | None = None,
    initial_files: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Render ``template`` into ``target_dir``.

    Returns the relative paths of every file written. If ``target_dir``
    already exists and contains files, raises ``ScaffoldRenderError`` —
    scaffold rendering must not overwrite an existing tool directory.

    ``initial_files`` is an optional ``{relative_path: content}`` map
    that overrides scaffold files post-render (used to drop in domain-
    specific stubs the agent has already authored).
    """

    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        raise ScaffoldRenderError(
            f"Refusing to render into non-empty directory {target_dir}.",
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    root = _template_root(template)
    manifest = _TemplateManifest.load(root)
    variables = _default_vars(
        name=name,
        purpose=purpose,
        license_id=license_id,
        description=description,
        template_vars=template_vars,
    )

    written: list[str] = []
    for rel in manifest.text_files:
        src = root / rel
        if not src.is_file():
            raise ScaffoldRenderError(
                f"Scaffold {template!r} declares text file {rel!r} but it does not exist.",
            )
        content = src.read_text(encoding="utf-8")
        try:
            rendered = Template(content).safe_substitute(variables)
        except (ValueError, KeyError) as exc:
            raise ScaffoldRenderError(
                f"Failed to render {template}/{rel}: {exc}",
            ) from exc
        # Filenames may also contain placeholders.
        rel_rendered = Template(rel).safe_substitute(variables)
        dest = target_dir / rel_rendered
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")
        written.append(rel_rendered)

    for rel in manifest.binary_files:
        src = root / rel
        if not src.is_file():
            raise ScaffoldRenderError(
                f"Scaffold {template!r} declares binary file {rel!r} but it does not exist.",
            )
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        written.append(rel)

    for rel, content in (initial_files or {}).items():
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        if rel not in written:
            written.append(rel)

    return tuple(written)


# ---------------------------------------------------------------------------
# L1-E monorepo-extension scaffolds
# ---------------------------------------------------------------------------


def _extension_dest_relative(surface: str, name: str) -> str:
    """Path of the rendered file relative to the resolved surface dir.

    Matches the L1-A discovery convention per surface (see
    ``design-monorepo-extensions.md``):

    - ``plugins``     → ``<name>/SKILL.md``     (directory-per-skill)
    - ``profiles``    → ``<name>.yaml``
    - others (agents, deployments, tools) → ``<name>.py``

    Routes through :data:`DEFAULT_SURFACE_DIRS` so the surface set
    never gets re-enumerated here.
    """
    if surface not in DEFAULT_SURFACE_DIRS:
        raise ScaffoldRenderError(
            f"Unknown extension surface {surface!r}; "
            f"available: {sorted(DEFAULT_SURFACE_DIRS)}",
        )
    if surface == "plugins":
        return f"{name}/SKILL.md"
    if surface == "profiles":
        return f"{name}.yaml"
    return f"{name}.py"


def render_extension_scaffold(
    surface: str,
    surface_dir: Path,
    name: str,
    *,
    template_vars: Mapping[str, str] | None = None,
    scaffold_id: str | None = None,
) -> Path:
    """Render the L1-E extension scaffold for ``surface`` into the file
    L1-A would discover at ``surface_dir / <surface convention for name>``.

    Returns the absolute path of the file written. Refuses to overwrite
    an existing destination file (sibling extensions in the same surface
    are fine — only the per-extension destination must be empty).

    When ``scaffold_id`` is ``None``, renders the blank L1-E template
    for the surface (the original behavior). When set, looks up the
    registered :class:`ExtensionScaffold` via the registry; the
    scaffold's bound surface must match ``surface``. This is the L2-F
    extension point: CPS (and any third-party extension package)
    registers domain-shaped scaffolds via
    :func:`register_extension_scaffold` at startup, and callers pick
    them by id.

    Variables available to ``string.Template`` substitution: ``name``,
    ``name_snake``, ``name_dash``, plus the union of caller-supplied
    ``template_vars`` (which override the defaults).
    """
    if surface not in _EXTENSION_TEMPLATE_FILE_BY_SURFACE:
        raise ScaffoldRenderError(
            f"Unknown extension surface {surface!r}; "
            f"available: {sorted(_EXTENSION_TEMPLATE_FILE_BY_SURFACE)}",
        )

    if scaffold_id is None:
        template_file = (
            _EXTENSION_TEMPLATES_DIR
            / _EXTENSION_TEMPLATE_FILE_BY_SURFACE[surface]
        )
    else:
        # Local import keeps the registry an optional collaborator
        # of the renderer (the blank-template path doesn't need it).
        from .registry import get_extension_scaffold
        scaffold = get_extension_scaffold(scaffold_id)
        if scaffold.surface != surface:
            raise ScaffoldRenderError(
                f"scaffold {scaffold_id!r} targets surface "
                f"{scaffold.surface!r}, but render was requested for "
                f"surface {surface!r}",
            )
        template_file = scaffold.template_path

    if not template_file.is_file():
        raise ScaffoldRenderError(
            f"Extension scaffold {template_file} is missing on disk.",
        )

    dest_rel = _extension_dest_relative(surface, name)
    dest = surface_dir / dest_rel
    if dest.exists():
        raise ScaffoldRenderError(
            f"Refusing to overwrite existing extension at {dest}.",
        )

    variables: dict[str, str] = {
        "name": name,
        "name_snake": name.replace("-", "_"),
        "name_dash": name.replace("_", "-"),
    }
    if template_vars:
        variables.update({str(k): str(v) for k, v in template_vars.items()})

    source = template_file.read_text(encoding="utf-8")
    try:
        rendered = Template(source).safe_substitute(variables)
    except (ValueError, KeyError) as exc:
        raise ScaffoldRenderError(
            f"Failed to render extension scaffold {surface}/{name}: {exc}",
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(rendered, encoding="utf-8")
    return dest


__all__ = (
    "AVAILABLE_TEMPLATES",
    "ScaffoldRenderError",
    "list_template_files",
    "render_extension_scaffold",
    "render_template",
)
