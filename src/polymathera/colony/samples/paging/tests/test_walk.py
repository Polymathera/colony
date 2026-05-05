"""Walker tests — exercise :func:`walk_repo` directly against a small
in-memory git repo. The :class:`GitRepoContextPageSource` plumbing is
covered separately by the integration smoke tests so we don't pay the
full strategy/initialise cost here.
"""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from polymathera.colony.samples.paging._walk import PathFilter, walk_repo


def _make_repo(root: Path, files: dict[str, bytes]) -> Path:
    """Create a fresh git repo at ``root`` with the given files committed
    on ``main``. Returns ``root``."""

    repo = git.Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    repo.git.add(all=True)
    repo.index.commit("initial")
    return root


def _matched_relpaths(root: Path, files: list[str]) -> set[str]:
    return {str(Path(p).relative_to(root)) for p in files}


def test_unfiltered_walk_returns_every_blob(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "r", {
        "a.py": b"x = 1\n",
        "tools/b.py": b"y = 2\n",
        "literature/c.txt": b"hello\n",
    })
    out = walk_repo(str(root), PathFilter())
    assert _matched_relpaths(root, out) == {"a.py", "tools/b.py", "literature/c.txt"}


def test_start_dir_restricts_walk(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "r", {
        "a.py": b"x = 1\n",
        "tools/b.py": b"y = 2\n",
        "literature/c.txt": b"hello\n",
    })
    out = walk_repo(str(root), PathFilter(start_dir="tools"))
    assert _matched_relpaths(root, out) == {"tools/b.py"}


def test_exclude_globs_drop_matches(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "r", {
        "src/a.py": b"x\n",
        "src/build/b.py": b"y\n",
        "src/c.py": b"z\n",
    })
    out = walk_repo(
        str(root),
        PathFilter(exclude_globs=("**/build/**",)),
    )
    assert _matched_relpaths(root, out) == {"src/a.py", "src/c.py"}


def test_include_globs_restrict_to_matches(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "r", {
        "a.py": b"x\n",
        "b.md": b"y\n",
        "c.txt": b"z\n",
    })
    out = walk_repo(
        str(root),
        PathFilter(include_globs=("*.py",)),
    )
    assert _matched_relpaths(root, out) == {"a.py"}


def test_binary_policy_skip_drops_pdfs(tmp_path: Path) -> None:
    pdf_bytes = b"%PDF-1.4\n%\x00binary garbage"
    root = _make_repo(tmp_path / "r", {
        "doc.pdf": pdf_bytes,
        "readme.md": b"hello\n",
    })
    out = walk_repo(
        str(root),
        PathFilter(binary_policy="skip"),
    )
    assert _matched_relpaths(root, out) == {"readme.md"}


def test_binary_policy_include_keeps_pdfs(tmp_path: Path) -> None:
    pdf_bytes = b"%PDF-1.4\n%\x00binary garbage"
    root = _make_repo(tmp_path / "r", {
        "doc.pdf": pdf_bytes,
    })
    out = walk_repo(
        str(root),
        PathFilter(binary_policy="include"),
    )
    assert _matched_relpaths(root, out) == {"doc.pdf"}


def test_colonyignore_file_is_honored(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "r", {
        ".colonyignore": b"private/\n*.tmp\n",
        "src/a.py": b"x\n",
        "private/secret.py": b"y\n",
        "scratch.tmp": b"z\n",
    })
    out = walk_repo(str(root), PathFilter())
    assert _matched_relpaths(root, out) == {".colonyignore", "src/a.py"}


def test_ignore_files_can_be_disabled(tmp_path: Path) -> None:
    root = _make_repo(tmp_path / "r", {
        ".colonyignore": b"private/\n",
        "private/x.py": b"x\n",
    })
    out = walk_repo(str(root), PathFilter(ignore_files=()))
    # With ignore-file reading disabled, private/x.py is no longer dropped.
    assert "private/x.py" in _matched_relpaths(root, out)
