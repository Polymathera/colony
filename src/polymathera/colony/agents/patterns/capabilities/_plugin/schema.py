"""Typed records for skills and plugins discovered from disk.

The on-disk format deliberately overlaps with Claude Code's
``SKILL.md`` / ``plugin.json`` layout so the same directory can be
shared between Colony and Claude Code with minimal translation.

See ``colony_docs/markdown/plans/design_UserPluginCapability.md``
for the full schema and the Colony-specific extensions (in
particular, ``sandbox_image_role`` and the required ``script`` field).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SkillSource(str, Enum):
    """Where a skill was discovered, in priority order.

    The capability honours this order when resolving name collisions:
    the first source to contain a skill wins, later sources log a
    collision but are ignored.
    """

    SESSION = "session"      # <workspace>/.colony/skills/<name>/SKILL.md
    USER = "user"            # ~/.colony/skills/<name>/SKILL.md
    SYSTEM = "system"        # /etc/colony/skills/<name>/SKILL.md


@dataclass(frozen=True)
class SkillParam:
    """One declared parameter for a skill.

    Type strings intentionally mirror the Claude Code subset: plain
    names like ``"string"``, ``"integer"``, ``"boolean"``, ``"array"``.
    Unknown types are accepted but not validated — the LLM may pass
    arbitrary values through and the skill's script is responsible for
    handling them.
    """

    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    items: str | None = None  # element type for "array"

    @classmethod
    def from_raw(cls, raw: Any, *, name: str | None = None) -> "SkillParam":
        """Accept two input shapes:

        1. The Claude Code list form: ``[{name: x, type: y, ...}, ...]``
           — the caller provides ``name`` implicitly via position or
           via the dict.
        2. The Colony-introduced dict form for compactness:
           ``{foo: {type: string, required: true}, bar: {type: integer}}``
           — the parent dict's key is the param name.
        """
        if isinstance(raw, dict):
            merged_name = name or raw.get("name")
            if not merged_name:
                raise ValueError("skill param missing name")
            return cls(
                name=str(merged_name),
                type=str(raw.get("type", "string")),
                required=bool(raw.get("required", False)),
                description=str(raw.get("description", "")),
                items=raw.get("items"),
            )
        raise ValueError(f"unsupported param spec: {raw!r}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name, "type": self.type,
            "required": self.required,
        }
        if self.description:
            out["description"] = self.description
        if self.items is not None:
            out["items"] = self.items
        return out


@dataclass(frozen=True)
class SkillSpec:
    """One discovered skill.

    ``qualified_name`` is the globally unique identifier the caller
    uses with ``run_skill``:

    - Plugin-owned skills: ``"<plugin>/<skill>"``
    - Standalone skills: just ``"<skill>"``

    The capability accepts either form and falls back to simple
    ``name`` lookup when no slash is present.
    """

    name: str
    source: SkillSource
    directory: Path
    description: str
    when_to_use: str
    sandbox_image_role: str | None
    script: str | None
    params: tuple[SkillParam, ...]
    timeout_seconds: int
    path_globs: tuple[str, ...]
    disable_model_invocation: bool
    body: str
    plugin_name: str | None = None
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return (
            f"{self.plugin_name}/{self.name}"
            if self.plugin_name else self.name
        )

    def path_matches(self, file_path: str | Path) -> bool:
        """Whether any declared ``paths`` glob matches ``file_path``.

        The capability uses this to let future prompt enrichment
        highlight skills relevant to the agent's current working set
        (per design §5.6). v1 does not surface it yet; the hook exists
        so callers can filter ``list_skills`` by context.
        """
        if not self.path_globs:
            return False
        s = str(file_path)
        return any(fnmatch.fnmatch(s, pat) for pat in self.path_globs)

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "source": self.source.value,
            "plugin": self.plugin_name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "sandbox_image_role": self.sandbox_image_role,
            "timeout_seconds": self.timeout_seconds,
            "params": [p.to_dict() for p in self.params],
            "path_globs": list(self.path_globs),
            "disable_model_invocation": self.disable_model_invocation,
        }

    def to_detail(self) -> dict[str, Any]:
        out = self.to_summary()
        out["directory"] = str(self.directory)
        out["script"] = self.script
        out["body"] = self.body
        return out

    def required_params(self) -> list[str]:
        return [p.name for p in self.params if p.required]


@dataclass(frozen=True)
class PluginSpec:
    """One plugin directory.

    Plugins are the sharing / versioning envelope around a set of
    skills. The ``name`` field in ``plugin.json`` namespaces every
    skill under it (``<plugin>/<skill>``) so two plugins can ship
    skills with the same local name.
    """

    name: str
    version: str
    description: str
    author: str
    directory: Path
    source: SkillSource
    skills: tuple[SkillSpec, ...]

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "source": self.source.value,
            "skill_count": len(self.skills),
        }
