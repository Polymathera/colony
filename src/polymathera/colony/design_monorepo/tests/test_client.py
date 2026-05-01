"""Tests for ``DesignMonorepoClient`` operations against a real local git repo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    AgentIdentity,
    BranchExistsError,
    Checkpoint,
    CheckpointNotFoundError,
    DECISIONS_DIR,
    DesignMonorepoClient,
)


def _write(repo: DesignMonorepoClient, rel: str, content: str) -> Path:
    path = repo.working_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_commit_with_identity_sha_returned(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    _write(bootstrapped_repo, "design/notes.txt", "hello\n")
    sha = bootstrapped_repo.commit_with_identity(
        identity, "add notes", paths=[Path("design/notes.txt")],
    )
    assert isinstance(sha, str) and len(sha) == 40
    assert bootstrapped_repo.current_state().is_fresh is False


def test_commit_with_identity_no_changes_returns_head(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    head = bootstrapped_repo.repo.head.commit.hexsha
    sha = bootstrapped_repo.commit_with_identity(
        identity, "noop", paths=[],
    )
    assert sha == head


def test_commit_authorship(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    _write(bootstrapped_repo, "design/x.txt", "x")
    sha = bootstrapped_repo.commit_with_identity(
        identity, "author check", paths=[Path("design/x.txt")],
    )
    commit = bootstrapped_repo.repo.commit(sha)
    assert commit.author.name == identity.git_name
    assert commit.author.email == identity.git_email
    assert commit.committer.name == identity.git_name
    assert commit.committer.email == identity.git_email


def test_tag_checkpoint_and_list(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    cp = bootstrapped_repo.tag_checkpoint(identity, "initial", "smoke")
    assert isinstance(cp, Checkpoint)
    cps = bootstrapped_repo.list_checkpoints()
    assert any(c.checkpoint_id == cp.checkpoint_id for c in cps)
    assert all(c.label for c in cps)


def test_fork_creates_branch(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    fork = bootstrapped_repo.fork(identity, "alt-1")
    assert fork.name == "fork/alt-1"
    state = bootstrapped_repo.current_state()
    assert state.current_branch == "fork/alt-1"
    assert any(f.name == "fork/alt-1" for f in state.forks)


def test_fork_existing_raises(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    bootstrapped_repo.fork(identity, "alt-1")
    with pytest.raises(BranchExistsError):
        bootstrapped_repo.fork(identity, "alt-1")


def test_restore_checkpoint_replace(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    cp = bootstrapped_repo.tag_checkpoint(identity, "before", "")
    _write(bootstrapped_repo, "design/y.txt", "later")
    sha_after = bootstrapped_repo.commit_with_identity(
        identity, "later", paths=[Path("design/y.txt")],
    )
    assert sha_after != cp.sha
    restored = bootstrapped_repo.restore_checkpoint(
        identity, cp.checkpoint_id, mode="replace",
    )
    assert restored == cp.checkpoint_id
    assert bootstrapped_repo.repo.head.commit.hexsha == cp.sha


def test_restore_checkpoint_fork(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    cp = bootstrapped_repo.tag_checkpoint(identity, "anchor", "")
    branch = bootstrapped_repo.restore_checkpoint(
        identity, cp.checkpoint_id, mode="fork", recovery_label="explore",
    )
    assert branch == "fork/explore"
    assert bootstrapped_repo.repo.active_branch.name == "fork/explore"


def test_restore_checkpoint_unknown(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    with pytest.raises(CheckpointNotFoundError):
        bootstrapped_repo.restore_checkpoint(identity, "checkpoint/missing")


def test_diff_between_refs(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    cp1 = bootstrapped_repo.tag_checkpoint(identity, "v1", "")
    _write(bootstrapped_repo, "design/added.txt", "new")
    bootstrapped_repo.commit_with_identity(
        identity, "add file", paths=[Path("design/added.txt")],
    )
    cp2 = bootstrapped_repo.tag_checkpoint(identity, "v2", "")
    diff = bootstrapped_repo.diff(cp1.checkpoint_id, cp2.checkpoint_id)
    assert diff.ref_a == cp1.checkpoint_id
    assert diff.ref_b == cp2.checkpoint_id
    paths = {e.path for e in diff.entries}
    assert "design/added.txt" in paths


def test_branch_topology(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    bootstrapped_repo.fork(identity, "left")
    # Switch back to main and make a fork of a different lifecycle prefix.
    bootstrapped_repo.repo.git.checkout("main")
    bootstrapped_repo.repo.create_head("session/abc", commit="HEAD")
    topo = bootstrapped_repo.get_branch_topology()
    names = {b.name for b in topo.branches}
    assert {"main", "fork/left", "session/abc"} <= names
    fork_node = [b for b in topo.branches if b.name == "fork/left"][0]
    assert fork_node.is_fork is True
    sess_node = [b for b in topo.branches if b.name == "session/abc"][0]
    assert sess_node.is_session is True


def test_list_recent_decisions(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    payload = {
        "decision_id": "d-001",
        "title": "Pick HBM3",
        "summary": "Memory choice",
        "rationale": "Bandwidth dominates.",
    }
    p = bootstrapped_repo.working_dir / DECISIONS_DIR / "d-001.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    bootstrapped_repo.commit_with_identity(
        identity, "add decision d-001", paths=[Path(f"{DECISIONS_DIR}/d-001.json")],
    )
    recent = bootstrapped_repo.list_recent_decisions(limit=10)
    assert len(recent) == 1
    assert recent[0].decision_id == "d-001"
    assert recent[0].title == "Pick HBM3"


def test_register_tool_round_trip(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    from polymathera.colony.design_monorepo import ToolEntry

    entry = ToolEntry(
        name="laptime",
        purpose="racer/laptime",
        capability="simulate_laptime",
        location="subdir:tools/racer/laptime",
        license="MIT",
    )
    sha = bootstrapped_repo.register_tool(identity, entry)
    assert isinstance(sha, str) and len(sha) == 40
    matches = bootstrapped_repo.find_existing_tool("simulate_laptime")
    assert len(matches) == 1
    assert matches[0].entry.name == "laptime"


def test_diff_against_unknown_checkpoint(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    with pytest.raises(CheckpointNotFoundError):
        bootstrapped_repo.diff_against_checkpoint("checkpoint/never")


def test_cherry_pick(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity,
) -> None:
    bootstrapped_repo.fork(identity, "feature")
    _write(bootstrapped_repo, "design/feature.txt", "f")
    sha = bootstrapped_repo.commit_with_identity(
        identity, "add feature", paths=[Path("design/feature.txt")],
    )
    bootstrapped_repo.repo.git.checkout("main")
    new = bootstrapped_repo.cherry_pick(identity, [sha])
    assert len(new) == 1
    head_files = bootstrapped_repo.repo.git.show(
        "--stat", "--pretty=", "HEAD"
    )
    assert "design/feature.txt" in head_files


def test_imports_remote_setup(
    bootstrapped_repo: DesignMonorepoClient, identity: AgentIdentity, tmp_path: Path,
) -> None:
    # Build a second repo and then add it as imports_tooling_from.
    second = tmp_path / "second"
    from polymathera.colony.design_monorepo import (
        DesignMonorepoManifest,
        bootstrap_design_monorepo,
        ImportedRemote,
    )
    other_manifest = DesignMonorepoManifest(
        tenant="acme",
        colony="acme-colony",
        program="other",
        target_system="x",
        design_repo_url="file:///does-not-matter",
    )
    bootstrap_design_monorepo(other_manifest, second, identity=identity)
    # Update the bootstrapped repo's manifest in memory and re-setup remotes.
    new_manifest = bootstrapped_repo.manifest.with_imports(
        [ImportedRemote(name="prev", url=str(second), ref="main")]
    )
    bootstrapped_repo._manifest = new_manifest
    bootstrapped_repo.setup_imported_remotes(fetch=False)
    remote_names = {r.name for r in bootstrapped_repo.repo.remotes}
    assert "prev" in remote_names
