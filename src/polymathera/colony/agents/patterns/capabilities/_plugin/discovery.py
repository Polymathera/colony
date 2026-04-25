"""Filesystem discovery for skills and plugins.

Layered over PyYAML (already a project dependency for
``sandbox-images.yaml``); we deliberately avoid adding
``python-frontmatter`` since the frontmatter format is straightforward
enough to parse inline.

Errors on individual skills (malformed YAML, missing fields) are
logged and skipped so one broken skill doesn't silence the rest — the
same philosophy applied in ``ImageRegistry``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import PluginSpec, SkillParam, SkillSource, SkillSpec

logger = logging.getLogger(__name__)


_FRONTMATTER_DELIM = "---"


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``SKILL.md``-style document into ``(frontmatter, body)``.

    Recognised form::

        ---
        key: value
        ...
        ---
        markdown body

    When the document has no frontmatter block, returns ``({}, text)``.
    A malformed frontmatter YAML block raises ``ValueError`` so the
    caller can log the specific skill path.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, text

    # Find the matching closing delimiter.
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            end = i
            break
    if end is None:
        raise ValueError("frontmatter block has no closing '---'")

    fm_raw = "\n".join(lines[1:end])
    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML in frontmatter: {e}") from e
    if not isinstance(fm, dict):
        raise ValueError(
            f"frontmatter must be a YAML mapping, got {type(fm).__name__}"
        )
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return fm, body


# ---------------------------------------------------------------------------
# Skill construction
# ---------------------------------------------------------------------------

def _coerce_params(raw: Any) -> tuple[SkillParam, ...]:
    """Accept both the list form and the dict form for ``params``.

    - List: ``[{name: foo, type: string, required: true}, ...]``
    - Dict: ``{foo: {type: string, required: true}, ...}``
    """
    if raw is None:
        return ()
    params: list[SkillParam] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError(f"param entry must be a mapping: {entry!r}")
            params.append(SkillParam.from_raw(entry))
    elif isinstance(raw, dict):
        for name, meta in raw.items():
            if isinstance(meta, dict):
                params.append(SkillParam.from_raw(meta, name=str(name)))
            else:
                # Bare "name: string" shorthand.
                params.append(SkillParam(
                    name=str(name), type=str(meta),
                ))
    else:
        raise ValueError(
            f"params must be a list or mapping, got {type(raw).__name__}"
        )
    return tuple(params)


def _coerce_globs(raw: Any) -> tuple[str, ...]:
    """``paths`` may be a comma-separated string or a list."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(s.strip() for s in raw.split(",") if s.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(s).strip() for s in raw if str(s).strip())
    raise ValueError(f"paths must be a string or list, got {type(raw).__name__}")


def _build_skill(
    *,
    skill_dir: Path,
    source: SkillSource,
    fm: dict[str, Any],
    body: str,
    plugin_name: str | None,
) -> SkillSpec:
    name = fm.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("SKILL.md frontmatter missing required 'name'")
    description = str(fm.get("description", "") or "")
    when_to_use = str(fm.get("when_to_use", "") or "")
    sandbox_image_role = fm.get("sandbox_image_role")
    if sandbox_image_role is not None:
        sandbox_image_role = str(sandbox_image_role)
    script = fm.get("script")
    if script is not None:
        script = str(script)
    params = _coerce_params(fm.get("params"))
    path_globs = _coerce_globs(fm.get("paths"))
    timeout_seconds = int(fm.get("timeout_seconds", 600))
    disable_model_invocation = bool(
        fm.get("disable-model-invocation", False)
    )
    return SkillSpec(
        name=name,
        source=source,
        directory=skill_dir,
        description=description,
        when_to_use=when_to_use,
        sandbox_image_role=sandbox_image_role,
        script=script,
        params=params,
        timeout_seconds=timeout_seconds,
        path_globs=path_globs,
        disable_model_invocation=disable_model_invocation,
        body=body,
        plugin_name=plugin_name,
        raw_frontmatter=dict(fm),
    )


# ---------------------------------------------------------------------------
# Discovery entry points
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryResult:
    """Outcome of one discovery pass.

    ``errors`` is a list of ``(path, reason)`` tuples for skills that
    were found on disk but could not be loaded. ``collisions`` records
    skills that were shadowed by a higher-priority source.
    """

    skills: dict[str, SkillSpec] = field(default_factory=dict)
    plugins: dict[str, PluginSpec] = field(default_factory=dict)
    errors: list[tuple[str, str]] = field(default_factory=list)
    collisions: list[tuple[str, str, str]] = field(default_factory=list)

    def add_skill(self, skill: SkillSpec) -> None:
        """Insert ``skill`` unless the qualified name already exists
        (later roots lose — we are called in priority order)."""
        key = skill.qualified_name
        if key in self.skills:
            prior = self.skills[key]
            self.collisions.append((
                key, prior.source.value, skill.source.value,
            ))
            logger.info(
                "UserPluginCapability: skill %r (source=%s) shadowed "
                "by higher-priority %s",
                key, skill.source.value, prior.source.value,
            )
            return
        self.skills[key] = skill


def resolve_skill_path(
    root: Path, skill_name: str,
) -> Path:
    """Default layout: ``<root>/<skill>/SKILL.md``."""
    return root / skill_name / "SKILL.md"


def _load_skill_md(
    skill_md: Path,
    *,
    source: SkillSource,
    plugin_name: str | None,
    result: DiscoveryResult,
) -> SkillSpec | None:
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        result.errors.append((str(skill_md), f"read failed: {e}"))
        return None
    try:
        fm, body = parse_frontmatter(text)
    except ValueError as e:
        result.errors.append((str(skill_md), str(e)))
        return None
    try:
        skill = _build_skill(
            skill_dir=skill_md.parent,
            source=source,
            fm=fm,
            body=body,
            plugin_name=plugin_name,
        )
    except ValueError as e:
        result.errors.append((str(skill_md), str(e)))
        return None
    return skill


def _iter_skill_dirs(skills_root: Path):
    """Yield ``<name>/SKILL.md`` candidates under ``skills_root``."""
    if not skills_root.is_dir():
        return
    for entry in sorted(skills_root.iterdir()):
        if not entry.is_dir():
            continue
        md = entry / "SKILL.md"
        if md.is_file():
            yield entry.name, md


def discover_skills(
    roots: list[tuple[Path, SkillSource]],
) -> DiscoveryResult:
    """Walk every ``(root, source)`` in priority order.

    ``root`` is the *skills root* (``.../skills``); each subdirectory
    is expected to contain a ``SKILL.md``. Higher-priority sources
    come first; duplicate names in later sources are recorded as
    collisions and skipped.
    """
    result = DiscoveryResult()
    for root, source in roots:
        for _name, md in _iter_skill_dirs(root):
            skill = _load_skill_md(
                md, source=source, plugin_name=None, result=result,
            )
            if skill is not None:
                result.add_skill(skill)
    return result


def discover_plugins(
    roots: list[tuple[Path, SkillSource]],
    *,
    into: DiscoveryResult | None = None,
) -> DiscoveryResult:
    """Walk every plugin root and materialise their skills.

    A plugin directory is identified by ``.claude-plugin/plugin.json``;
    skills live under ``<plugin>/skills/<name>/SKILL.md``. Skill
    qualified names are ``<plugin>/<name>`` to prevent cross-plugin
    collisions.
    """
    result = into if into is not None else DiscoveryResult()
    for root, source in roots:
        if not root.is_dir():
            continue
        for plugin_dir in sorted(root.iterdir()):
            if not plugin_dir.is_dir():
                continue
            manifest = plugin_dir / ".claude-plugin" / "plugin.json"
            if not manifest.is_file():
                continue
            try:
                raw = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as e:
                result.errors.append((str(manifest), f"invalid JSON: {e}"))
                continue
            name = str(raw.get("name") or plugin_dir.name)
            skills_root = plugin_dir / "skills"
            plugin_skills: list[SkillSpec] = []
            for _s_name, md in _iter_skill_dirs(skills_root):
                skill = _load_skill_md(
                    md, source=source, plugin_name=name, result=result,
                )
                if skill is None:
                    continue
                plugin_skills.append(skill)
                result.add_skill(skill)
            plugin_spec = PluginSpec(
                name=name,
                version=str(raw.get("version", "0.0.0")),
                description=str(raw.get("description", "")),
                author=str(raw.get("author", "")),
                directory=plugin_dir,
                source=source,
                skills=tuple(plugin_skills),
            )
            if name in result.plugins:
                result.collisions.append((
                    f"plugin:{name}",
                    result.plugins[name].source.value,
                    plugin_spec.source.value,
                ))
                continue
            result.plugins[name] = plugin_spec
    return result
