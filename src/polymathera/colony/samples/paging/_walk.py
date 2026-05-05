"""Repo-tree walker shared by :class:`GitRepoContextPageSource` and
:class:`LiteratureContextPageSource`.

Single concern: given a cloned repository, return the absolute paths of
the blobs that should become VCM pages, after applying:

- a sub-tree restriction (``start_dir``),
- gitignore-style include / exclude patterns,
- patterns read out of named ignore files inside the repo
  (``.gitignore``, ``.colonyignore``, …),
- a binary-file policy (``"skip"`` / ``"include"``).

The walker runs synchronously and is thread-safe — it constructs its
own :class:`git.Repo` so it can be dispatched via ``asyncio.to_thread``
without sharing the caller's git subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BinaryPolicy = Literal["skip", "include"]


@dataclass(frozen=True)
class PathFilter:
    """Declarative filter spec for :func:`walk_repo`.

    Args:
        start_dir: Repo-relative directory to root the walk at. ``None``
            means the repository root.
        include_globs: Gitignore-style include patterns (paths
            relative to ``start_dir`` after restriction). ``None``
            means *include everything not excluded*.
        exclude_globs: Gitignore-style exclude patterns. Always applied
            regardless of ``include_globs``.
        ignore_files: Filenames whose contents (gitignore syntax) are
            merged into the exclude pattern set when found anywhere in
            the walked subtree. Defaults to ``(".gitignore",
            ".colonyignore")``.
        binary_policy: ``"skip"`` (default) drops blobs whose
            ``mimetypes.guess_type`` is non-text and that contain a
            NUL byte in the first 8 KB; ``"include"`` keeps them
            (the literature source uses this).
    """

    start_dir: str | None = None
    include_globs: tuple[str, ...] | None = None
    exclude_globs: tuple[str, ...] = ()
    ignore_files: tuple[str, ...] = (".gitignore", ".colonyignore")
    binary_policy: BinaryPolicy = "skip"


_BINARY_PROBE_BYTES = 8192


def _is_binary_blob(abs_path: Path) -> bool:
    """Heuristic: NUL byte in the first 8 KB ⇒ binary.

    Cheaper than spinning up ``mimetypes`` for every blob and avoids
    misclassifying extension-less scripts. Caller has already filtered
    out everything ``include_globs`` rejected, so we only probe blobs
    that actually pay this cost.
    """

    try:
        with abs_path.open("rb") as fh:
            chunk = fh.read(_BINARY_PROBE_BYTES)
    except OSError:
        return False
    return b"\x00" in chunk


def _build_pathspec(patterns: tuple[str, ...]):
    """Compile gitignore-style patterns to a :class:`pathspec.PathSpec`.

    Imported lazily so callers that never use the filter don't pay the
    import cost (and so the dep stays in the ``code_analysis`` extra).
    """

    from pathspec import PathSpec
    from pathspec.patterns import GitWildMatchPattern

    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def _read_ignore_files(
    repo_root: Path, sub_root: Path, names: tuple[str, ...],
) -> list[str]:
    """Walk ``sub_root`` looking for files whose name matches
    ``names`` and accumulate their gitignore patterns. Each returned
    pattern is rewritten so it is anchored to the sub-tree (i.e. a
    ``.gitignore`` deeper in the tree applies only below its
    directory)."""

    if not names:
        return []
    patterns: list[str] = []
    for ignore_path in sub_root.rglob("*"):
        if not ignore_path.is_file() or ignore_path.name not in names:
            continue
        prefix = ignore_path.parent.relative_to(sub_root).as_posix()
        try:
            for line in ignore_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Anchor sub-tree patterns to their containing directory
                # so a nested ``.gitignore`` matches only below itself.
                # ``prefix == "."`` is the sub-root itself — do not anchor.
                if prefix and prefix != "." and not stripped.startswith("/"):
                    patterns.append(f"{prefix}/{stripped}")
                else:
                    patterns.append(stripped)
        except OSError:
            continue
    return patterns


def walk_repo(repo_path: str, path_filter: PathFilter) -> list[str]:
    """Return the absolute paths of git-tracked blobs under
    ``repo_path`` that pass ``path_filter``.

    The function uses its own :class:`git.Repo` instance so it is safe
    to call from ``asyncio.to_thread``. Paths are returned absolute
    (``str(repo_root / blob.path)``) to match the existing strategy
    contract.
    """

    import git

    repo = git.Repo(repo_path)
    repo_root = Path(repo_path)
    sub_root = repo_root / path_filter.start_dir if path_filter.start_dir else repo_root
    if not sub_root.is_dir():
        return []

    exclude_patterns = list(path_filter.exclude_globs) + _read_ignore_files(
        repo_root, sub_root, path_filter.ignore_files,
    )
    exclude_spec = _build_pathspec(tuple(exclude_patterns)) if exclude_patterns else None
    include_spec = (
        _build_pathspec(path_filter.include_globs)
        if path_filter.include_globs is not None else None
    )

    matched: list[str] = []
    for blob in repo.head.commit.tree.traverse():
        if blob.type != "blob":
            continue
        abs_path = repo_root / blob.path
        try:
            rel_to_sub = abs_path.relative_to(sub_root).as_posix()
        except ValueError:
            continue  # outside start_dir
        if exclude_spec is not None and exclude_spec.match_file(rel_to_sub):
            continue
        if include_spec is not None and not include_spec.match_file(rel_to_sub):
            continue
        if path_filter.binary_policy == "skip" and _is_binary_blob(abs_path):
            continue
        matched.append(str(abs_path))
    return matched
