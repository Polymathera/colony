"""Tests for the v1 → v2 manifest migrator.

Three contracts:

1. Idempotent — running against a v2+ manifest is a no-op (returns
   ``was_migrated=False``, no disk writes).
2. Additive — a v1 manifest is rewritten with ``schema_version=2`` and
   a default :class:`ExtensionsConfig` block; every other field is
   preserved verbatim (no data loss).
3. Commit hook — when ``commit_identity`` is passed AND a write
   happened, the migrator opens a :class:`DesignMonorepoClient` and
   commits the manifest file with the supplied identity. Skipped when
   the migration was a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from git import Actor, Repo

from polymathera.colony.design_monorepo.identity import AgentIdentity
from polymathera.colony.design_monorepo.manifest import (
    DEFAULT_SURFACE_DIRS,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    DesignMonorepoManifest,
    ExtensionsConfig,
)
from polymathera.colony.tools.manifest_migrate import (
    MigrationResult,
    migrate_manifest,
)


def _v1_payload() -> dict:
    return {
        "schema_version": 1,
        "tenant": "acme",
        "colony": "acme-colony",
        "program": "prog-1",
        "target_system": "x",
        "design_repo_url": "https://example.com/repo.git",
        "default_branch": "main",
        "agent_email_domain": "agent.colony.local",
    }


def _write_v1(repo_root: Path) -> Path:
    p = repo_root / MANIFEST_RELATIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_v1_payload()), encoding="utf-8")
    return p


def test_migrates_v1_to_current_schema(tmp_path: Path) -> None:
    _write_v1(tmp_path)
    result = migrate_manifest(tmp_path)

    assert isinstance(result, MigrationResult)
    assert result.was_migrated is True
    assert result.from_version == 1
    assert result.to_version == MANIFEST_SCHEMA_VERSION == 2
    assert result.commit_sha is None  # no commit_identity supplied

    reloaded = DesignMonorepoManifest.load_path(tmp_path)
    assert reloaded.schema_version == 2
    assert reloaded.extensions is not None
    # Defaults applied for every surface — operator can override later.
    assert reloaded.extensions.plugins.directory == DEFAULT_SURFACE_DIRS["plugins"]
    assert reloaded.extensions.tools.directory == DEFAULT_SURFACE_DIRS["tools"]


def test_migration_preserves_existing_fields(tmp_path: Path) -> None:
    """The migrator is purely additive — every original field must
    survive verbatim. Catches an accidental clobber on tenant / colony /
    LFS / webhook / default_branch / etc."""
    payload = _v1_payload()
    payload.update({
        "tenant": "preserved-tenant",
        "lfs": {"mode": "separate", "separate_url": "https://lfs.example.com"},
        "webhook": {"enabled": True, "endpoint": "https://hook.example.com", "secret_ref": "secret/xyz"},
        "agent_email_domain": "custom.local",
    })
    (tmp_path / MANIFEST_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / MANIFEST_RELATIVE_PATH).write_text(json.dumps(payload), encoding="utf-8")

    migrate_manifest(tmp_path)
    reloaded = DesignMonorepoManifest.load_path(tmp_path)
    assert reloaded.tenant == "preserved-tenant"
    assert reloaded.lfs.mode == "separate"
    assert reloaded.lfs.separate_url == "https://lfs.example.com"
    assert reloaded.webhook.enabled is True
    assert reloaded.webhook.endpoint == "https://hook.example.com"
    assert reloaded.agent_email_domain == "custom.local"


def test_idempotent_when_already_at_current_schema(tmp_path: Path) -> None:
    """Running twice in a row is a no-op the second time."""
    _write_v1(tmp_path)
    first = migrate_manifest(tmp_path)
    assert first.was_migrated is True

    mtime_before = (tmp_path / MANIFEST_RELATIVE_PATH).stat().st_mtime
    second = migrate_manifest(tmp_path)
    mtime_after = (tmp_path / MANIFEST_RELATIVE_PATH).stat().st_mtime

    assert second.was_migrated is False
    assert second.from_version == 2
    assert second.to_version == 2
    # No disk write on the no-op path.
    assert mtime_before == mtime_after


def test_idempotent_against_native_v2_manifest(tmp_path: Path) -> None:
    """A manifest authored directly at v2 (no prior v1) is also a
    no-op — the migrator doesn't blindly overwrite extensions block."""
    m = DesignMonorepoManifest(
        tenant="acme",
        colony="acme-colony",
        program="prog-1",
        target_system="x",
        design_repo_url="https://example.com/repo.git",
        extensions=ExtensionsConfig(),
    )
    m.write_path(tmp_path)
    before = (tmp_path / MANIFEST_RELATIVE_PATH).read_bytes()
    result = migrate_manifest(tmp_path)
    after = (tmp_path / MANIFEST_RELATIVE_PATH).read_bytes()
    assert result.was_migrated is False
    assert before == after


# ---------------------------------------------------------------------------
# Commit hook
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real git repo with a v1 manifest committed on main, so the
    migrator can re-commit the v2 result."""
    repo = Repo.init(tmp_path)
    # Initial commit so HEAD exists before the migration commit. Use
    # ``index.commit`` (in-process) rather than ``git commit``: the
    # CLI path needs ``user.name`` / ``user.email`` from git config,
    # which CI runners don't ship.
    actor = Actor("seed", "seed@example.com")
    (tmp_path / "README.md").write_text("seed\n")
    repo.index.add(["README.md"])
    repo.index.commit("seed", author=actor, committer=actor)
    _write_v1(tmp_path)
    repo.index.add([MANIFEST_RELATIVE_PATH])
    repo.index.commit("v1 manifest", author=actor, committer=actor)
    return tmp_path


def test_migration_with_commit_identity_writes_a_commit(git_repo: Path) -> None:
    repo = Repo(git_repo)
    head_before = repo.head.commit.hexsha

    identity = AgentIdentity(
        agent_id="migrator-test",
        role="ops",
        colony_id="local",
    )
    result = migrate_manifest(git_repo, commit_identity=identity)

    assert result.was_migrated is True
    assert result.commit_sha is not None
    assert result.commit_sha != head_before
    # The HEAD points at the new commit; the author/committer reflects
    # the supplied identity (commit_with_identity sets both).
    new_head = repo.head.commit
    assert new_head.author.name == identity.git_name
    assert new_head.author.email == identity.git_email


def test_no_commit_when_migration_is_noop(git_repo: Path) -> None:
    """First migrate brings v1→v2. Second migrate is a no-op — must NOT
    create a new commit even if commit_identity is passed."""
    identity = AgentIdentity(agent_id="m", role="ops", colony_id="local")
    migrate_manifest(git_repo, commit_identity=identity)

    repo = Repo(git_repo)
    head_before = repo.head.commit.hexsha
    result = migrate_manifest(git_repo, commit_identity=identity)
    assert result.was_migrated is False
    assert result.commit_sha is None
    assert repo.head.commit.hexsha == head_before
