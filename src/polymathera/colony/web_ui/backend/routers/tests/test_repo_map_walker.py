"""Tests for the bounded tree walker that backs ``GET /repo-map/tree``.

The full HTTP surface needs a live cluster (auth middleware + colony
connection + ``GitFileStorage``); we don't have a router-level test
harness in this repo. The walker is the only non-trivial piece of
local logic, so unit-test it directly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.web_ui.backend.routers.repo_map import _walk_tree


def _make_tree(root: Path, files: list[str]) -> Path:
    for rel in files:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    return root


def test_walker_returns_root_node_with_children(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "r", [
        "a.py", "src/b.py", "src/c.py", "literature/paper.pdf",
    ])
    node = _walk_tree(root, max_nodes=1000, max_depth=4)
    assert node.path == "."
    assert node.is_dir is True
    top_level = {c.path for c in node.children}
    assert top_level == {"a.py", "src", "literature"}


def test_walker_orders_dirs_before_files(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "r", [
        "z_file", "a_dir/x", "b_dir/y",
    ])
    node = _walk_tree(root, max_nodes=1000, max_depth=4)
    names = [c.path for c in node.children]
    # Dirs (a_dir, b_dir) should precede the file (z_file).
    assert names[-1] == "z_file"
    assert set(names[:2]) == {"a_dir", "b_dir"}


def test_walker_skips_dot_git(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "r", [
        ".git/HEAD", ".git/config", "README.md",
    ])
    node = _walk_tree(root, max_nodes=1000, max_depth=4)
    top = {c.path for c in node.children}
    assert ".git" not in top
    assert "README.md" in top


def test_walker_respects_max_depth(tmp_path: Path) -> None:
    root = _make_tree(tmp_path / "r", [
        "a/b/c/d/e.txt",
    ])
    node = _walk_tree(root, max_nodes=1000, max_depth=2)
    # Depth 2 from root: root → a → b. b's children should be empty
    # because depth==max_depth at b.
    a = node.children[0]
    assert a.path == "a"
    b = a.children[0]
    assert b.path == "a/b"
    assert b.children == []


def test_walker_respects_max_nodes(tmp_path: Path) -> None:
    files = [f"d/f{i}.txt" for i in range(50)]
    root = _make_tree(tmp_path / "r", files)
    node = _walk_tree(root, max_nodes=10, max_depth=4)
    # Counting nodes: 1 (root) + 1 (d) + N children of d, capped overall
    # at max_nodes. We expect strictly fewer than 50 file children.
    d = node.children[0]
    assert len(d.children) < 50


# ---------------------------------------------------------------------------
# _refresh_cache_clone
# ---------------------------------------------------------------------------


def test_refresh_cache_clone_picks_up_new_origin_commits(tmp_path: Path) -> None:
    """The user-reported bug: an agent pushed ``initialize_repo_map``
    to origin, but the dashboard's "Load" still showed the old empty
    state because ``GitFileStorage.clone_or_retrieve_repository`` is
    idempotent and returns the cached path without fetching.

    ``_refresh_cache_clone`` is the dashboard-side fix: after the
    cache lookup, fetch from origin and reset the cached working tree
    to ``origin/<branch>`` so subsequent reads see what the agent
    just published.
    """

    import git

    from polymathera.colony.web_ui.backend.routers.repo_map import (
        _refresh_cache_clone,
    )

    upstream = tmp_path / "upstream.git"
    git.Repo.init(upstream, initial_branch="main", bare=True)

    # Seed upstream with commit X, so the dashboard cache has
    # SOMETHING to clone before the agent's push lands.
    seed = tmp_path / "seed"
    seed_repo = git.Repo.clone_from(f"file://{upstream}", str(seed))
    seed_repo.config_writer().set_value("user", "email", "s@s").release()
    seed_repo.config_writer().set_value("user", "name", "seed").release()
    (seed / "x").write_text("X\n", encoding="utf-8")
    seed_repo.index.add(["x"])
    seed_repo.index.commit("X")
    seed_repo.remote("origin").push(refspec="HEAD:refs/heads/main")

    # Dashboard cache cloned at X — pre-agent state.
    cache = tmp_path / "cache"
    git.Repo.clone_from(f"file://{upstream}", str(cache))
    assert (cache / "x").is_file()
    assert not (cache / "y").exists()

    # Agent's clone makes commit Y and pushes it.
    agent = tmp_path / "agent"
    agent_repo = git.Repo.clone_from(f"file://{upstream}", str(agent))
    agent_repo.config_writer().set_value("user", "email", "a@a").release()
    agent_repo.config_writer().set_value("user", "name", "agent").release()
    (agent / "y").write_text("from agent\n", encoding="utf-8")
    agent_repo.index.add(["y"])
    agent_repo.index.commit("Y — agent's bootstrap commit")
    agent_repo.remote("origin").push(refspec="HEAD:refs/heads/main")

    # Refresh the cache. After this, the cache's working tree should
    # reflect Y — both ``x`` and ``y`` present.
    _refresh_cache_clone(cache, branch="main", commit="HEAD")

    assert (cache / "x").is_file()
    assert (cache / "y").is_file()
    assert (cache / "y").read_text() == "from agent\n"


def test_refresh_cache_clone_respects_pinned_commit(tmp_path: Path) -> None:
    """When the caller asked for a specific revision (``commit !=
    "HEAD"``), refresh fetches but does NOT move HEAD — pinned
    commits are exactly the use case that wants stable snapshots.
    """

    import git

    from polymathera.colony.web_ui.backend.routers.repo_map import (
        _refresh_cache_clone,
    )

    upstream = tmp_path / "upstream.git"
    git.Repo.init(upstream, initial_branch="main", bare=True)

    seed = tmp_path / "seed"
    seed_repo = git.Repo.clone_from(f"file://{upstream}", str(seed))
    seed_repo.config_writer().set_value("user", "email", "s@s").release()
    seed_repo.config_writer().set_value("user", "name", "seed").release()
    (seed / "v1").write_text("v1", encoding="utf-8")
    seed_repo.index.add(["v1"])
    pinned = seed_repo.index.commit("v1")
    seed_repo.remote("origin").push(refspec="HEAD:refs/heads/main")

    cache = tmp_path / "cache"
    git.Repo.clone_from(f"file://{upstream}", str(cache))
    pinned_sha = pinned.hexsha

    # Upstream advances after the cache was populated.
    (seed / "v2").write_text("v2", encoding="utf-8")
    seed_repo.index.add(["v2"])
    seed_repo.index.commit("v2")
    seed_repo.remote("origin").push(refspec="HEAD:refs/heads/main")

    _refresh_cache_clone(cache, branch="main", commit=pinned_sha)

    cache_repo = git.Repo(str(cache))
    # HEAD didn't budge: pinned snapshot honoured.
    assert cache_repo.head.commit.hexsha == pinned_sha
    assert not (cache / "v2").exists()


def test_refresh_cache_clone_tolerates_missing_origin(tmp_path: Path) -> None:
    """A best-effort refresh: a local-only cache (no ``origin``) must
    not turn a successful Load into a 5xx. The dashboard logs a
    warning and serves the existing cached state."""

    import git

    from polymathera.colony.web_ui.backend.routers.repo_map import (
        _refresh_cache_clone,
    )

    cache = tmp_path / "cache"
    cache.mkdir()
    repo = git.Repo.init(cache, initial_branch="main")
    repo.config_writer().set_value("user", "email", "c@c").release()
    repo.config_writer().set_value("user", "name", "c").release()
    (cache / "f").write_text("hi", encoding="utf-8")
    repo.index.add(["f"])
    repo.index.commit("only commit")

    # Should not raise.
    _refresh_cache_clone(cache, branch="main", commit="HEAD")

    assert (cache / "f").is_file()
