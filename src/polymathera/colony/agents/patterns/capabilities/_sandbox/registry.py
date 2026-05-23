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
from pathlib import Path
from typing import Any

import yaml

# ``DockerImageSpec`` / ``ScriptSpec`` live in ``colony.agents.sandbox_images``
# — the single source of truth shared with ``DockerImageRegistryConfig``.
# See ``EXPERIMENTATION_AND_DATA_ANALYTICS_PLAN.md`` Stage E E-3
# follow-up for the consolidation history.
from ....sandbox_images import DockerImageSpec, ScriptSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DockerImageRegistry:
    """Loaded view of the sandbox image YAML.

    Immutable after construction. The capability re-reads the file on
    ``reload()`` (called by the operator-facing Settings endpoint in a
    future phase); today, loading happens once at capability init.
    """

    def __init__(self, images: list[DockerImageSpec]):
        self._images: dict[str, DockerImageSpec] = {i.role: i for i in images}

    # --- Construction ------------------------------------------------------

    @classmethod
    def from_yaml_text(cls, text: str) -> "DockerImageRegistry":
        data = yaml.safe_load(text) or {}
        raw_images = data.get("images") or []
        images: list[DockerImageSpec] = []
        for raw in raw_images:
            if not isinstance(raw, dict) or "role" not in raw or "image" not in raw:
                logger.warning(
                    "DockerImageRegistry: skipping malformed entry: %r", raw,
                )
                continue
            try:
                images.append(DockerImageSpec.from_dict(raw))
            except Exception as e:
                logger.warning(
                    "DockerImageRegistry: failed to parse %r: %s", raw, e,
                )
        return cls(images)

    @classmethod
    def from_path(cls, path: str | Path) -> "DockerImageRegistry":
        p = Path(path)
        if not p.exists():
            logger.warning(
                "DockerImageRegistry: registry file %s does not exist; "
                "starting with an empty registry",
                p,
            )
            return cls([])
        return cls.from_yaml_text(p.read_text())

    @classmethod
    def from_config(cls, config: Any) -> "DockerImageRegistry":
        """Build the registry from a ``DockerImageRegistryConfig`` instance.

        ``config.images`` is already ``list[DockerImageSpec]`` (the shared
        schema from ``colony.agents.sandbox_images`` — Pydantic-
        validated at YAML load time). No translation needed; the
        capability and the config component speak the same type.
        ``config`` is typed loosely to avoid a top-level import of
        ``agents.configs`` from this low-level subpackage.
        """
        return cls(list(config.images))

    @classmethod
    def empty(cls) -> "DockerImageRegistry":
        return cls([])

    # --- Lookup -----------------------------------------------------------

    def get(self, role: str) -> DockerImageSpec | None:
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
    ) -> tuple[DockerImageSpec, ScriptSpec] | None:
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
