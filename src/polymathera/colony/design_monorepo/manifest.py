"""``.colony/manifest.json`` — the per-program design-monorepo manifest.

The manifest carries the metadata that lets any Colony node clone a
program's design monorepo and reattach to its deployment context. It is
committed into the design monorepo itself so a clone alone is enough to
identify the program; secrets are kept out of it and live in the
per-deployment secrets store (master §3.1.6).

A manifest persists on disk as ``.colony/manifest.json`` with a small,
typed schema (``MANIFEST_SCHEMA_VERSION``). Subsequent schema versions
upgrade in place via ``DesignMonorepoManifest.load_path``; an unknown
version raises ``ManifestSchemaError`` rather than silently coercing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ImportedRemote


MANIFEST_SCHEMA_VERSION = 2
"""Bumped whenever the manifest's on-disk shape changes incompatibly.

v2 adds the optional ``extensions`` block (L1-A). v1 manifests still
parse cleanly — their ``extensions`` field defaults to ``None``. The
:mod:`polymathera.colony.tools.manifest_migrate` utility bumps existing
v1 manifests to v2 in place."""


MANIFEST_RELATIVE_PATH = ".colony/manifest.json"
"""Path within the design monorepo where the manifest lives."""


DEFAULT_SURFACE_DIRS: dict[str, str] = {
    "plugins": ".colony/plugins/",
    "agents": ".colony/agents/",
    "deployments": ".colony/deployments/",
    "tools": ".colony/tools/",
    "profiles": ".colony/profiles/",
    "missions": ".colony/missions/",
}
"""Per-surface default directories under the design-monorepo root.

A v2 manifest may override any of these via its ``extensions`` block; an
omitted surface (or an entire missing ``extensions`` block) means "use
the default — discover from this path if it exists, else surface is
empty"."""


class ManifestSchemaError(ValueError):
    """Raised when ``.colony/manifest.json`` is missing or malformed."""


class SurfaceConfig(BaseModel):
    """One ``extensions.<surface>`` entry: where to find that surface's
    extension files inside the design monorepo (path is relative to the
    repo root). Each surface kind shares the same shape today; per-surface
    knobs can grow here without re-bumping the schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: str


class ExtensionsConfig(BaseModel):
    """``extensions`` block on a v2+ manifest. Declares the per-surface
    directories L1-A discovery walks. A missing surface falls back to
    :data:`DEFAULT_SURFACE_DIRS`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plugins: SurfaceConfig = Field(
        default_factory=lambda: SurfaceConfig(directory=DEFAULT_SURFACE_DIRS["plugins"]),
    )
    agents: SurfaceConfig = Field(
        default_factory=lambda: SurfaceConfig(directory=DEFAULT_SURFACE_DIRS["agents"]),
    )
    deployments: SurfaceConfig = Field(
        default_factory=lambda: SurfaceConfig(directory=DEFAULT_SURFACE_DIRS["deployments"]),
    )
    tools: SurfaceConfig = Field(
        default_factory=lambda: SurfaceConfig(directory=DEFAULT_SURFACE_DIRS["tools"]),
    )
    profiles: SurfaceConfig = Field(
        default_factory=lambda: SurfaceConfig(directory=DEFAULT_SURFACE_DIRS["profiles"]),
    )
    missions: SurfaceConfig = Field(
        default_factory=lambda: SurfaceConfig(directory=DEFAULT_SURFACE_DIRS["missions"]),
    )


class LFSConfig(BaseModel):
    """LFS configuration for the design monorepo."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["same_remote", "separate", "disabled"] = "same_remote"
    separate_url: str | None = Field(
        default=None,
        description="Required when mode == 'separate'.",
    )
    credentials_ref: str | None = Field(
        default=None,
        description=(
            "Reference into the secrets store. Never the credentials "
            "themselves."
        ),
    )

    @model_validator(mode="after")
    def _check_separate_url(self) -> Self:
        if self.mode == "separate" and not self.separate_url:
            raise ValueError(
                "LFSConfig.separate_url is required when mode == 'separate'.",
            )
        return self


class WebhookConfig(BaseModel):
    """Configuration for inbound git push webhooks.

    The framework writes the webhook secret into the secrets store and
    references it here. The endpoint is the absolute URL the remote
    posts to (e.g. ``https://colony.example.com/api/v1/vcm/git_push_event``).
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    endpoint: str | None = None
    secret_ref: str | None = None


class DesignMonorepoManifest(BaseModel):
    """The top-level manifest committed at ``.colony/manifest.json``.

    All identity fields (tenant, colony, program, target_system) are
    free-form strings owned by the deploying organization. The framework
    treats them as opaque labels carried into provenance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=MANIFEST_SCHEMA_VERSION)

    # Identity
    tenant: str = Field(description="Tenant / organization label.")
    colony: str = Field(description="Colony deployment identifier.")
    program: str = Field(description="Program identifier (per design target).")
    target_system: str = Field(
        description=(
            "What the program is designing — free-form. E.g. "
            "'class-A fusion plant', 'CAMI ambulatory monitor', "
            "'F-class autonomous EV racer'."
        ),
    )

    # Topology
    topology: Literal["self_hosted_gitea", "external", "air_gapped"] = "external"
    design_repo_url: str = Field(
        description=(
            "Origin remote URL — https://, git@, or file://. The framework "
            "validates reachability at attach time, not at parse time."
        ),
    )
    credentials_ref: str | None = Field(
        default=None,
        description=(
            "Reference into the per-deployment secrets store. Stays None "
            "for file:// or for environments where the global git "
            "credential helper resolves auth."
        ),
    )

    # Cross-program tooling carry-over
    imports_tooling_from: tuple[ImportedRemote, ...] = Field(
        default_factory=tuple,
        description="Read-only tooling-monorepo carry-overs. Master §9.5.",
    )

    # LFS + webhook
    lfs: LFSConfig = Field(default_factory=LFSConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)

    # L1-A: per-monorepo extension surfaces. Optional — v1 manifests parse
    # with ``extensions=None`` (no L4 extensions surface declared); v2+
    # manifests carry an :class:`ExtensionsConfig` so operators can
    # override the per-surface directories.
    extensions: ExtensionsConfig | None = None

    # Default branch & identity convention
    default_branch: str = Field(default="main")
    agent_email_domain: str = Field(
        default="agent.colony.local",
        description=(
            "Suffix used to build per-agent git committer emails: "
            "'<agent_id>@<colony_id>.<agent_email_domain>'."
        ),
    )

    # ---- IO ------------------------------------------------------------

    @classmethod
    def load_path(cls, repo_root: Path) -> Self:
        """Load the manifest from ``<repo_root>/.colony/manifest.json``.

        Raises ``ManifestSchemaError`` if the file is missing, malformed,
        or written with a schema version this code does not understand.
        """

        manifest_path = repo_root / MANIFEST_RELATIVE_PATH
        if not manifest_path.is_file():
            raise ManifestSchemaError(
                f"Manifest not found at {manifest_path}. The repository "
                "may not be a Colony design monorepo.",
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestSchemaError(
                f"Manifest at {manifest_path} is not valid JSON: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise ManifestSchemaError(
                f"Manifest at {manifest_path} must be a JSON object.",
            )
        version = payload.get("schema_version", 1)
        if version > MANIFEST_SCHEMA_VERSION:
            raise ManifestSchemaError(
                f"Manifest schema_version={version} is newer than this "
                f"colony build understands ({MANIFEST_SCHEMA_VERSION}). "
                "Upgrade colony before reattaching.",
            )
        return cls.model_validate(payload)

    def write_path(self, repo_root: Path) -> Path:
        """Write the manifest to ``<repo_root>/.colony/manifest.json``.

        Returns the absolute path written. Caller is responsible for
        staging + committing the file.
        """

        manifest_path = repo_root / MANIFEST_RELATIVE_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                self.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    # ---- Convenience ---------------------------------------------------

    def with_imports(
        self, additional: Iterable[ImportedRemote],
    ) -> DesignMonorepoManifest:
        """Return a copy with extra imported tooling remotes appended.

        Duplicates (by URL + ref) are silently dropped. Used by the
        ``imports_tooling_from`` UI flow.
        """

        seen = {(r.url, r.ref) for r in self.imports_tooling_from}
        merged = list(self.imports_tooling_from)
        for r in additional:
            key = (r.url, r.ref)
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        return self.model_copy(update={"imports_tooling_from": tuple(merged)})


__all__ = (
    "DEFAULT_SURFACE_DIRS",
    "DesignMonorepoManifest",
    "ExtensionsConfig",
    "LFSConfig",
    "WebhookConfig",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_RELATIVE_PATH",
    "ManifestSchemaError",
    "SurfaceConfig",
)
