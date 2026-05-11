"""L1-A: idempotent migration of a design-monorepo manifest from
``schema_version: 1`` to the current schema (``v2`` today — adds the
optional ``extensions`` block).

Two modes:

- Pure-Python: :func:`migrate_manifest` reads, rewrites, and writes the
  manifest in place. The caller decides how to commit.
- Commit-via-client: pass ``commit_identity=...`` and we open a
  :class:`DesignMonorepoClient` on ``repo_root`` and stage + commit the
  changed file with the supplied identity — same machinery
  ``DesignCheckpointer`` uses internally.

Idempotent: running against a manifest already at the current schema
version returns :class:`MigrationResult` with ``was_migrated=False`` and
makes no disk writes (no commit either).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..design_monorepo.client import DesignMonorepoClient
from ..design_monorepo.identity import AgentIdentity
from ..design_monorepo.manifest import (
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    DesignMonorepoManifest,
    ExtensionsConfig,
)


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of a single :func:`migrate_manifest` call.

    ``from_version`` and ``to_version`` are always populated (even on a
    no-op, where they're equal). ``commit_sha`` is populated only when
    the caller supplied ``commit_identity`` AND a write happened.
    """

    repo_root: Path
    from_version: int
    to_version: int
    was_migrated: bool
    commit_sha: str | None = None


def migrate_manifest(
    repo_root: Path,
    *,
    commit_identity: AgentIdentity | None = None,
    commit_message: str = "L1-A: migrate design-monorepo manifest to schema v2",
) -> MigrationResult:
    """Bump ``<repo_root>/.colony/manifest.json`` to the current schema
    version, idempotently.

    A v1 manifest is rewritten with ``schema_version: 2`` and an
    ``extensions`` block populated with the per-surface defaults from
    :class:`ExtensionsConfig`. A manifest already at v2+ is left
    untouched and returned with ``was_migrated=False``.

    When ``commit_identity`` is supplied, the function opens a
    :class:`DesignMonorepoClient` against ``repo_root`` and commits the
    modified manifest with that identity. The commit is best-effort:
    if the working tree is dirty in unrelated ways, the function still
    commits only the manifest file (paths-scoped commit).
    """
    manifest = DesignMonorepoManifest.load_path(repo_root)
    from_version = manifest.schema_version
    if from_version >= MANIFEST_SCHEMA_VERSION:
        return MigrationResult(
            repo_root=repo_root,
            from_version=from_version,
            to_version=from_version,
            was_migrated=False,
        )

    # Force the schema_version onto the in-memory manifest and add the
    # default extensions block. ``model_copy`` respects the frozen
    # model. Existing fields are preserved verbatim — migration is
    # purely additive.
    upgraded = manifest.model_copy(update={
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "extensions": ExtensionsConfig(),
    })
    upgraded.write_path(repo_root)

    commit_sha: str | None = None
    if commit_identity is not None:
        client = DesignMonorepoClient.open(repo_root)
        commit_sha = client.commit_with_identity(
            commit_identity,
            commit_message,
            paths=[Path(MANIFEST_RELATIVE_PATH)],
        )

    return MigrationResult(
        repo_root=repo_root,
        from_version=from_version,
        to_version=MANIFEST_SCHEMA_VERSION,
        was_migrated=True,
        commit_sha=commit_sha,
    )


__all__ = ("MigrationResult", "migrate_manifest")
