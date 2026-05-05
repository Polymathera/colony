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
