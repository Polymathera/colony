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


_THIS_DIR = Path(__file__).resolve().parent


AVAILABLE_TEMPLATES: tuple[str, ...] = (
    "python_lib",
    "c_library",
    "julia_module",
    "rust_crate",
    "cmake_project",
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


__all__ = (
    "AVAILABLE_TEMPLATES",
    "ScaffoldRenderError",
    "list_template_files",
    "render_template",
)
