"""Shared schema for sandbox-image registry entries.

One source of truth for ``ScriptSpec`` + ``DockerImageSpec`` used by:

- :class:`~polymathera.colony.agents.configs.DockerImageRegistryConfig`
  (the Pydantic-validated operator-YAML schema).
- :class:`~polymathera.colony.agents.patterns.capabilities._sandbox.registry.DockerImageRegistry`
  (the in-memory registry the capability iterates).

Both classes are frozen Pydantic models — immutable after
construction (the registry trusts records don't change) with
Pydantic validation at the YAML boundary. ``extra="ignore"`` keeps
parsing tolerant: extra fields in legacy YAML files don't break the
load, they're just dropped. Tuple fields enforce immutability of
the collections themselves (frozen on a model with ``list[X]``
fields would still let callers mutate the list).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


class ScriptSpec(BaseModel):
    """One named, parameterised script registered against an image
    role.

    ``params`` is kept loose (dict of name → metadata) so it can
    describe an arbitrary JSON-schema-ish shape without pulling in
    ``jsonschema``.
    :meth:`SandboxedShellCapability._validate_script_args` enforces
    the ``required`` flag and a small set of type strings.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    description: str = ""
    cmd: tuple[str, ...] = ()
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    timeout_seconds: int = 300

    def to_summary(self) -> dict[str, Any]:
        """Dict suitable for returning to the LLM via ``list_scripts``."""
        return {
            "name": self.name,
            "description": self.description,
            "params": dict(self.params),
            "timeout_seconds": self.timeout_seconds,
        }


class DockerImageSpec(BaseModel):
    """One role entry in the sandbox image registry."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    role: str
    image: str
    description: str = ""
    scripts: tuple[ScriptSpec, ...] = ()
    required_env: tuple[str, ...] = ()
    """Env var names that :meth:`SandboxedShellCapability.run_script`
    resolves via sibling capabilities' ``resolve_value`` before
    dispatching a script in this image. If a required name has no
    resolver (or has multiple conflicting resolvers), ``run_script``
    raises with the missing / conflicting name."""

    script_template_packages: tuple[str, ...] = ()
    """Python package import paths whose ``.py`` files are valid
    ``template_name`` arguments to
    :meth:`SandboxedShellCapability.run_script(image_role=this_role,
    template_name=...)`. Read at runtime via
    :mod:`importlib.resources` so the agent doesn't need a
    filesystem path."""

    tags: tuple[str, ...] = ()
    """Free-form classification tags for this image. Used by
    :meth:`SandboxedShellCapability.list_images(tags=...)` for
    capability-discovery queries (e.g. ``{"data-analysis",
    "scientific-python"}``). All-match semantics on filter."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DockerImageSpec":
        """Tolerant parser for legacy YAML entries.

        Filters malformed script entries (missing ``name``, not a
        dict) and logs them rather than raising — preserves the
        registry's "tolerant load" semantics from before the
        consolidation. Operators who want stricter validation can
        call :meth:`model_validate` directly.
        """
        scripts_raw = raw.get("scripts") or []
        scripts: list[ScriptSpec] = []
        for s in scripts_raw:
            if not isinstance(s, dict) or "name" not in s:
                logger.warning(
                    "DockerImageSpec.from_dict: skipping malformed script "
                    "entry: %r", s,
                )
                continue
            scripts.append(ScriptSpec.model_validate(s))
        return cls(
            role=str(raw["role"]),
            image=str(raw["image"]),
            description=str(raw.get("description", "")),
            scripts=tuple(scripts),
            required_env=tuple(
                str(v) for v in (raw.get("required_env") or [])
            ),
            script_template_packages=tuple(
                str(v) for v in (raw.get("script_template_packages") or [])
            ),
            tags=tuple(str(v) for v in (raw.get("tags") or [])),
        )

    def script_by_name(self, name: str) -> ScriptSpec | None:
        for s in self.scripts:
            if s.name == name:
                return s
        return None

    def to_summary(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "image": self.image,
            "description": self.description,
            "scripts": [s.name for s in self.scripts],
            "required_env": list(self.required_env),
            "script_template_packages": list(self.script_template_packages),
            "tags": list(self.tags),
        }


__all__ = ("DockerImageSpec", "ScriptSpec")
