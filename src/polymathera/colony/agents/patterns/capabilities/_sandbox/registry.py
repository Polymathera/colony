"""Image + script registry for ``SandboxedShellCapability``.

The registry is the *allowed image list*. Agents pick a sandbox image
by role label (e.g., ``"code_analysis"``) rather than raw image name —
that way a misconfigured agent cannot run an untrusted image, and the
operator can swap images without touching agent code.

The on-disk format is a plain YAML file; the default location is
``/etc/colony/sandbox-images.yaml`` inside the ray-head container,
mounted from the repo at
``colony/src/polymathera/colony/cli/deploy/docker/sandbox-images.yaml``.
Tests load from a string directly so they do not touch disk.

Schema (tolerant dataclass-style; extra fields are ignored):

.. code-block:: yaml

    images:
      - role: code_analysis
        image: python:3.11-slim
        description: "Python 3.11 with pyright, ruff, mypy installed."
        scripts:
          - name: lint_python
            description: "Run ruff on a path"
            params:
              path: {type: string, required: true}
            cmd: ["bash", "-lc", "ruff check '{path}'"]

Scripts are referenced by name from ``SandboxedShellCapability.execute_script``.
Their ``cmd`` is a list of shell arguments with ``{param_name}``
placeholders that are filled in at execution time from the caller's
``args`` dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScriptSpec:
    """A single registered script.

    ``params`` is kept loose (dict of name → metadata) so it can describe
    an arbitrary JSON-schema-ish shape without pulling in ``jsonschema``.
    ``SandboxedShellCapability._validate_script_args`` enforces the
    ``required`` flag and a small set of type strings.
    """

    name: str
    description: str
    cmd: tuple[str, ...]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeout_seconds: int = 300

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScriptSpec":
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            cmd=tuple(raw.get("cmd", [])),
            params=dict(raw.get("params") or {}),
            timeout_seconds=int(raw.get("timeout_seconds", 300)),
        )

    def to_summary(self) -> dict[str, Any]:
        """Dict suitable for returning to the LLM via ``list_scripts``."""
        return {
            "name": self.name,
            "description": self.description,
            "params": dict(self.params),
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class ImageSpec:
    """One role entry in the image registry."""

    role: str
    image: str
    description: str
    scripts: tuple[ScriptSpec, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ImageSpec":
        scripts_raw = raw.get("scripts") or []
        scripts = tuple(
            ScriptSpec.from_dict(s) for s in scripts_raw
            if isinstance(s, dict) and "name" in s
        )
        return cls(
            role=str(raw["role"]),
            image=str(raw["image"]),
            description=str(raw.get("description", "")),
            scripts=scripts,
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
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ImageRegistry:
    """Loaded view of the sandbox image YAML.

    Immutable after construction. The capability re-reads the file on
    ``reload()`` (called by the operator-facing Settings endpoint in a
    future phase); today, loading happens once at capability init.
    """

    def __init__(self, images: list[ImageSpec]):
        self._images: dict[str, ImageSpec] = {i.role: i for i in images}

    # --- Construction ------------------------------------------------------

    @classmethod
    def from_yaml_text(cls, text: str) -> "ImageRegistry":
        data = yaml.safe_load(text) or {}
        raw_images = data.get("images") or []
        images: list[ImageSpec] = []
        for raw in raw_images:
            if not isinstance(raw, dict) or "role" not in raw or "image" not in raw:
                logger.warning(
                    "ImageRegistry: skipping malformed entry: %r", raw,
                )
                continue
            try:
                images.append(ImageSpec.from_dict(raw))
            except Exception as e:
                logger.warning(
                    "ImageRegistry: failed to parse %r: %s", raw, e,
                )
        return cls(images)

    @classmethod
    def from_path(cls, path: str | Path) -> "ImageRegistry":
        p = Path(path)
        if not p.exists():
            logger.warning(
                "ImageRegistry: registry file %s does not exist; "
                "starting with an empty registry",
                p,
            )
            return cls([])
        return cls.from_yaml_text(p.read_text())

    @classmethod
    def empty(cls) -> "ImageRegistry":
        return cls([])

    # --- Lookup -----------------------------------------------------------

    def get(self, role: str) -> ImageSpec | None:
        return self._images.get(role)

    def roles(self) -> list[str]:
        return list(self._images.keys())

    def summaries(self) -> list[dict[str, Any]]:
        return [i.to_summary() for i in self._images.values()]

    def scripts_for(self, role: str) -> list[ScriptSpec]:
        spec = self.get(role)
        return list(spec.scripts) if spec else []

    def find_script(
        self, script_name: str, *, image_role: str | None = None,
    ) -> tuple[ImageSpec, ScriptSpec] | None:
        """Locate a script by name, optionally scoped to one role.

        If ``image_role`` is None, searches all roles and returns the
        first match (names are scoped per-role, so in practice the
        caller should specify ``image_role`` when ambiguity is possible).
        """
        roles = [image_role] if image_role else list(self._images.keys())
        for r in roles:
            spec = self._images.get(r)
            if spec is None:
                continue
            found = spec.script_by_name(script_name)
            if found is not None:
                return spec, found
        return None

    def __len__(self) -> int:
        return len(self._images)
