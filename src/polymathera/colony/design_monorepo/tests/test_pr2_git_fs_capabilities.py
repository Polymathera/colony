"""Tests for the PR 2 additions to the design-monorepo capabilities.

Three groups:

- **Reads** on :class:`RepoStateProvider` — ``read_file``,
  ``read_lines``, ``list_directory``, ``stat_path``, ``grep_content``,
  ``git_log``, ``git_status``, ``diff_working_tree``.
- **Git writes** on :class:`DesignCheckpointer` — ``create_branch`` /
  ``delete_branch`` / ``checkout_branch``, ``stash_save`` / ``pop`` /
  ``list_stashes``, ``rebase_onto``, ``create_tag`` / ``delete_tag``.
  (``fetch_remote`` / ``pull_remote`` require a live remote and are
  exercised by the client-layer tests separately.)
- **FS writes** on :class:`ProjectAuthoringCapability` —
  ``make_directory``, ``remove_directory``, ``copy_file``,
  ``set_file_executable``.

Shared fixtures (``bootstrapped_repo``, ``manifest``, ``identity``)
come from ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DesignCheckpointer,
    DesignMonorepoClient,
    DesignMonorepoError,
    ProjectAuthoringCapability,
    RepoStateProvider,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures specific to PR 2 (capability instances bound to bootstrapped repo)
# ---------------------------------------------------------------------------


@pytest.fixture
def state_provider(bootstrapped_repo: DesignMonorepoClient) -> RepoStateProvider:
    cap = RepoStateProvider(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def checkpointer(bootstrapped_repo: DesignMonorepoClient) -> DesignCheckpointer:
    cap = DesignCheckpointer(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def authoring(bootstrapped_repo: DesignMonorepoClient) -> ProjectAuthoringCapability:
    cap = ProjectAuthoringCapability(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


def _seed_files(repo_root: Path) -> None:
    (repo_root / "src" / "polymathera").mkdir(parents=True, exist_ok=True)
    (repo_root / "src" / "polymathera" / "alpha.py").write_text(
        "def alpha():\n    return 'alpha'\n", encoding="utf-8",
    )
    (repo_root / "src" / "polymathera" / "beta.py").write_text(
        "def beta():\n    return 42\n",
        encoding="utf-8",
    )
    (repo_root / "tests").mkdir(parents=True, exist_ok=True)
    (repo_root / "tests" / "test_alpha.py").write_text(
        "import alpha\n\n\ndef test_alpha():\n    assert alpha.alpha() == 'alpha'\n",
        encoding="utf-8",
    )


def _commit_seed(client: DesignMonorepoClient) -> None:
    from polymathera.colony.design_monorepo import AgentIdentity

    identity = AgentIdentity(
        agent_id="seed", role="seed", colony_id="acme-colony",
    )
    client.commit_with_identity(identity, "seed test files", all_changes=True)


# ---------------------------------------------------------------------------
# Filesystem reads
# ---------------------------------------------------------------------------


async def test_read_file_returns_content_and_total_bytes(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    _seed_files(bootstrapped_repo.working_dir)
    res = await state_provider.read_file("src/polymathera/alpha.py")
    assert "def alpha()" in res.content
    assert res.truncated is False
    assert res.total_bytes > 0
    assert res.path == "src/polymathera/alpha.py"


async def test_read_file_truncates_when_over_max_bytes(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    big = bootstrapped_repo.working_dir / "big.txt"
    big.write_text("a" * 2048, encoding="utf-8")
    res = await state_provider.read_file("big.txt", max_bytes=100)
    assert len(res.content) == 100
    assert res.truncated is True
    assert res.total_bytes == 2048


async def test_read_file_rejects_outside_tree(
    state_provider: RepoStateProvider,
) -> None:
    with pytest.raises(DesignMonorepoError, match="escapes"):
        await state_provider.read_file("../outside")
    with pytest.raises(DesignMonorepoError, match=r"\.git"):
        await state_provider.read_file(".git/HEAD")


async def test_read_file_returns_exists_false_when_missing(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    """LLM planners (running in a sandbox REPL) can't catch raised
    exceptions cleanly. ``read_file`` on a missing path returns
    ``FileContent(exists=False, content='', total_bytes=0)`` so the
    planner branches on the result instead of trying to wrap each
    call in try/except. The bootstrap path of ``project_planning``
    relies on this — coordinator was thrashing with duplicate
    spawns when ``docs/ROADMAP.md`` was missing pre-2026-06-05."""
    res = await state_provider.read_file("docs/does-not-exist.md")
    assert res.exists is False
    assert res.content == ""
    assert res.total_bytes == 0
    assert res.path == "docs/does-not-exist.md"
    # Existing files still come back with exists=True.
    (bootstrapped_repo.working_dir / "extant.md").write_text("hi", encoding="utf-8")
    res2 = await state_provider.read_file("extant.md")
    assert res2.exists is True
    assert res2.content == "hi"


async def test_read_lines_head_and_tail(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    f = bootstrapped_repo.working_dir / "lines.txt"
    f.write_text("\n".join(str(i) for i in range(1, 21)) + "\n", encoding="utf-8")

    head = await state_provider.read_lines("lines.txt", start=1, count=3)
    assert head.start_line == 1
    assert head.end_line == 3
    assert head.total_lines == 20
    assert head.truncated is True
    assert head.content.splitlines() == ["1", "2", "3"]

    tail = await state_provider.read_lines("lines.txt", start=-3, count=3)
    assert tail.content.splitlines() == ["18", "19", "20"]
    assert tail.end_line == 20


async def test_read_lines_count_zero_reports_total_only(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    f = bootstrapped_repo.working_dir / "size.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    res = await state_provider.read_lines("size.txt", count=0)
    assert res.content == ""
    assert res.total_lines == 3


async def test_list_directory_recursive_with_glob(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    _seed_files(bootstrapped_repo.working_dir)
    entries = await state_provider.list_directory(
        ".", recursive=True, pattern="src/**/*.py",
    )
    paths = [e.path for e in entries]
    assert "src/polymathera/alpha.py" in paths
    assert "src/polymathera/beta.py" in paths
    # README.md (in tree from bootstrap) must not match the glob.
    assert all(p.endswith(".py") for p in paths)


async def test_list_directory_skips_dotgit_and_dotcolony(
    state_provider: RepoStateProvider,
) -> None:
    """``.git/`` and ``.colony/`` subtrees are framework territory and
    do not appear in listings. Top-level files like ``.gitattributes``
    (which sit *next to* ``.git/`` rather than inside it) are
    fine."""

    entries = await state_provider.list_directory(".", recursive=True)
    for entry in entries:
        assert not entry.path.startswith(".git/")
        assert not entry.path.startswith(".colony/")


async def test_list_directory_returns_empty_when_missing(
    state_provider: RepoStateProvider,
) -> None:
    """Mirrors ``read_file``'s ``exists=False`` contract — missing
    directory returns an empty list rather than raising, so LLM
    planners can branch on length without a try/except. Bootstrap
    flows (target directory not yet created) depend on this."""
    entries = await state_provider.list_directory("docs/nope")
    assert entries == []


async def test_stat_path_existing_and_missing(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    f = bootstrapped_repo.working_dir / "exists.txt"
    f.write_text("hello", encoding="utf-8")

    present = await state_provider.stat_path("exists.txt")
    assert present.exists is True
    assert present.is_file is True
    assert present.size_bytes == 5

    missing = await state_provider.stat_path("nope.txt")
    assert missing.exists is False
    assert missing.is_file is False


async def test_grep_content_finds_matches(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    _seed_files(bootstrapped_repo.working_dir)
    result = await state_provider.grep_content(
        r"def \w+", path="src", regex=True,
    )
    paths = {m.path for m in result.matches}
    assert "src/polymathera/alpha.py" in paths
    assert "src/polymathera/beta.py" in paths


async def test_grep_content_glob_filter(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    _seed_files(bootstrapped_repo.working_dir)
    result = await state_provider.grep_content(
        "def", path=".", glob="**/alpha*.py",
    )
    assert all("alpha" in m.path for m in result.matches)


async def test_grep_content_truncates_at_max_matches(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    big = bootstrapped_repo.working_dir / "hits.txt"
    big.write_text("\n".join(["match"] * 50) + "\n", encoding="utf-8")
    result = await state_provider.grep_content(
        "match", path=".", max_matches=10,
    )
    assert len(result.matches) == 10
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Git reads
# ---------------------------------------------------------------------------


async def test_git_log_returns_bootstrap_commit(
    state_provider: RepoStateProvider,
) -> None:
    rows = await state_provider.git_log(limit=5)
    assert len(rows) >= 1
    head = rows[0]
    assert head.sha
    assert head.message  # bootstrap commit has a non-empty message
    assert isinstance(head.paths_changed, tuple)


async def test_git_status_clean_after_bootstrap(
    state_provider: RepoStateProvider,
) -> None:
    status = await state_provider.git_status()
    assert status.is_clean is True


async def test_git_status_tracks_untracked_and_unstaged(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    (bootstrapped_repo.working_dir / "scratch.txt").write_text(
        "new file", encoding="utf-8",
    )
    status = await state_provider.git_status()
    assert "scratch.txt" in status.untracked


async def test_diff_working_tree_picks_up_uncommitted_edit(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
) -> None:
    _seed_files(bootstrapped_repo.working_dir)
    _commit_seed(bootstrapped_repo)
    alpha = bootstrapped_repo.working_dir / "src" / "polymathera" / "alpha.py"
    alpha.write_text(alpha.read_text() + "\n# trailing comment\n", encoding="utf-8")
    diff = await state_provider.diff_working_tree(paths=["src/polymathera/alpha.py"])
    assert "# trailing comment" in diff


# ---------------------------------------------------------------------------
# Git writes — branch / stash / rebase / tag
# ---------------------------------------------------------------------------


async def test_branch_create_checkout_delete_round_trip(
    checkpointer: DesignCheckpointer,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    name = await checkpointer.create_branch("wip/test-feature")
    assert name == "wip/test-feature"
    assert any(b.name == "wip/test-feature" for b in bootstrapped_repo.repo.branches)

    await checkpointer.checkout_branch("wip/test-feature")
    assert bootstrapped_repo.repo.active_branch.name == "wip/test-feature"

    await checkpointer.checkout_branch("main")
    await checkpointer.delete_branch("wip/test-feature")
    assert not any(
        b.name == "wip/test-feature" for b in bootstrapped_repo.repo.branches
    )


async def test_create_branch_rejects_reserved_prefix(
    checkpointer: DesignCheckpointer,
) -> None:
    for reserved in ("checkpoint/x", "fork/x", "session/x", "agent/x", "tool/x"):
        with pytest.raises(DesignMonorepoError, match="reserved prefix"):
            await checkpointer.create_branch(reserved)


async def test_checkout_branch_refuses_with_uncommitted_changes(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
) -> None:
    await checkpointer.create_branch("wip/other")
    (bootstrapped_repo.working_dir / "dirty.txt").write_text("x", encoding="utf-8")
    bootstrapped_repo.repo.git.add("--", "dirty.txt")
    with pytest.raises(DesignMonorepoError, match="uncommitted"):
        await checkpointer.checkout_branch("wip/other")


async def test_stash_save_pop_round_trip(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
) -> None:
    _seed_files(bootstrapped_repo.working_dir)
    _commit_seed(bootstrapped_repo)
    # Now create a working-tree change.
    edited = bootstrapped_repo.working_dir / "src" / "polymathera" / "alpha.py"
    edited.write_text("# stashed\n", encoding="utf-8")

    stashed = await checkpointer.stash_save("wip")
    assert stashed is True
    # After stash, tree is clean.
    assert not bootstrapped_repo.has_uncommitted_changes()

    entries = await checkpointer.list_stashes()
    assert len(entries) == 1
    assert "wip" in entries[0].message

    await checkpointer.stash_pop()
    assert "# stashed" in edited.read_text()


async def test_stash_save_returns_false_on_clean_tree(
    checkpointer: DesignCheckpointer,
) -> None:
    stashed = await checkpointer.stash_save("nothing")
    assert stashed is False


async def test_rebase_refuses_with_uncommitted_changes(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
) -> None:
    (bootstrapped_repo.working_dir / "dirty.txt").write_text(
        "x", encoding="utf-8",
    )
    bootstrapped_repo.repo.git.add("--", "dirty.txt")
    with pytest.raises(DesignMonorepoError, match="uncommitted"):
        await checkpointer.rebase_onto("HEAD")


async def test_tag_create_and_delete(
    checkpointer: DesignCheckpointer,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    name = await checkpointer.create_tag("v0.0.1", message="first")
    assert name == "v0.0.1"
    assert any(t.name == "v0.0.1" for t in bootstrapped_repo.repo.tags)
    await checkpointer.delete_tag("v0.0.1")
    assert not any(t.name == "v0.0.1" for t in bootstrapped_repo.repo.tags)


async def test_create_tag_rejects_reserved_checkpoint_prefix(
    checkpointer: DesignCheckpointer,
) -> None:
    with pytest.raises(DesignMonorepoError, match="reserved prefix"):
        await checkpointer.create_tag("checkpoint/manual")


async def test_pull_remote_rejects_unknown_strategy(
    checkpointer: DesignCheckpointer,
) -> None:
    with pytest.raises(DesignMonorepoError, match="strategy"):
        await checkpointer.pull_remote(strategy="cherry-pick")


# ---------------------------------------------------------------------------
# FS writes — make_directory / remove_directory / copy_file / chmod
# ---------------------------------------------------------------------------


async def test_make_directory_creates_gitkeep_and_commits(
    bootstrapped_repo: DesignMonorepoClient,
    authoring: ProjectAuthoringCapability,
) -> None:
    payload = await authoring.make_directory("data/new_bucket")
    target = bootstrapped_repo.working_dir / "data" / "new_bucket" / ".gitkeep"
    assert target.is_file()
    assert "data/new_bucket/.gitkeep" in payload.affected_paths
    assert payload.commit_sha


async def test_remove_directory_refuses_non_empty_without_recursive(
    bootstrapped_repo: DesignMonorepoClient,
    authoring: ProjectAuthoringCapability,
) -> None:
    target = bootstrapped_repo.working_dir / "bucket"
    target.mkdir()
    (target / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DesignMonorepoError, match="not empty"):
        await authoring.remove_directory("bucket")


async def test_remove_directory_recursive_clears_subtree(
    bootstrapped_repo: DesignMonorepoClient,
    authoring: ProjectAuthoringCapability,
) -> None:
    # Build a small tree, commit it, then remove recursively.
    await authoring.make_directory("doomed")
    await authoring.write_file("doomed/file.txt", "x\n")
    payload = await authoring.remove_directory("doomed", recursive=True)
    assert not (bootstrapped_repo.working_dir / "doomed").exists()
    assert payload.commit_sha


async def test_copy_file_refuses_existing_destination(
    bootstrapped_repo: DesignMonorepoClient,
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/source.txt", "hello\n")
    await authoring.write_file("src/dest.txt", "DO NOT CLOBBER\n")
    with pytest.raises(DesignMonorepoError, match="destination exists"):
        await authoring.copy_file("src/source.txt", "src/dest.txt")


async def test_copy_file_writes_and_commits(
    bootstrapped_repo: DesignMonorepoClient,
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("src/source.txt", "hello\n")
    payload = await authoring.copy_file("src/source.txt", "src/copied.txt")
    dest = bootstrapped_repo.working_dir / "src" / "copied.txt"
    assert dest.read_text() == "hello\n"
    assert "src/copied.txt" in payload.affected_paths


async def test_set_file_executable_flips_user_x_bit(
    bootstrapped_repo: DesignMonorepoClient,
    authoring: ProjectAuthoringCapability,
) -> None:
    await authoring.write_file("tools/run.sh", "#!/bin/sh\necho hi\n")
    payload = await authoring.set_file_executable("tools/run.sh", executable=True)
    target = bootstrapped_repo.working_dir / "tools" / "run.sh"
    import stat
    assert target.stat().st_mode & stat.S_IXUSR
    assert payload.action_kind == "set_file_executable"

    # Round-trip back to non-executable.
    await authoring.set_file_executable("tools/run.sh", executable=False)
    assert not (target.stat().st_mode & stat.S_IXUSR)


async def test_set_file_executable_refuses_missing_file(
    authoring: ProjectAuthoringCapability,
) -> None:
    with pytest.raises(DesignMonorepoError, match="not a regular file"):
        await authoring.set_file_executable("nope.sh")
