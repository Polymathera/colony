"""Shared helpers for the design-monorepo merge drivers.

A driver is a small Python entry point invoked by git as

    python -m polymathera.colony.design_monorepo.git_merge.<name> %A %O %B %P

where ``%A``, ``%O``, ``%B``, ``%P`` are the four standard placeholders
documented in ``gitattributes(5)``. The helpers here normalize that
contract — argument parsing, file IO, conflict-marker fallback — so the
per-format drivers stay focused on the structural merge.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DriverArgs:
    """Parsed arguments to a merge driver.

    ``ours`` is the path the driver overwrites with the merged content.
    """

    ours: Path
    base: Path
    theirs: Path
    pathname: str


def parse_args(argv: list[str] | None = None) -> DriverArgs:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 4:
        print(
            f"merge-driver: expected 4 args (%A %O %B %P), got {len(args)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return DriverArgs(
        ours=Path(args[0]),
        base=Path(args[1]),
        theirs=Path(args[2]),
        pathname=args[3],
    )


def read_text(path: Path) -> str:
    """Read a temporary merge-input file, tolerant of missing files.

    git passes ``/dev/null`` (or its Windows equivalent) for the side
    that did not contribute the file (e.g. add/add or modify/delete
    cases). Returning an empty string keeps the driver code simple.
    """

    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return ""
    except UnicodeDecodeError:
        # Binary or oddly-encoded — let the caller fall back.
        return path.read_bytes().decode("utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_conflict_markers(
    args: DriverArgs,
    *,
    ours: str,
    theirs: str,
    base: str | None = None,
    label_ours: str = "ours",
    label_theirs: str = "theirs",
    label_base: str = "base",
) -> None:
    """Write standard ``<<<<<<< / ======= / >>>>>>>`` markers in-place.

    A driver calls this when it cannot merge cleanly; it writes the
    conflict-marker form into ``args.ours`` and exits non-zero so git
    flags the file as in conflict.
    """

    if base is not None:
        body = (
            f"<<<<<<< {label_ours}\n{ours}\n"
            f"||||||| {label_base}\n{base}\n"
            f"======= \n{theirs}\n>>>>>>> {label_theirs}\n"
        )
    else:
        body = f"<<<<<<< {label_ours}\n{ours}\n=======\n{theirs}\n>>>>>>> {label_theirs}\n"
    write_text(args.ours, body)


__all__ = (
    "DriverArgs",
    "parse_args",
    "read_text",
    "write_text",
    "write_conflict_markers",
)
