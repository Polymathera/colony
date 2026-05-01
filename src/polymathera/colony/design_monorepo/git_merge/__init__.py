"""Custom git merge drivers for the design monorepo's structured artifacts.

Drivers are registered in ``.gitattributes`` (committed) by file pattern
and resolved in ``.git/config`` (per-clone) to a command line. The
command convention follows git's standard merge-driver placeholders:

    python -m polymathera.colony.design_monorepo.git_merge.<name> %A %O %B %P

where:
- ``%A`` is the path to a temporary file holding the *current* (ours)
  version. The driver writes the merged content into this same file
  in-place;
- ``%O`` is the path to the temporary *base* version (the merge base);
- ``%B`` is the path to the temporary *theirs* version;
- ``%P`` is the path of the file being merged, relative to the repo
  root (informational; many drivers use it for logging only).

A driver exits 0 on a clean merge and non-zero on conflict. On a
non-zero exit, git falls back to its standard conflict resolution
behaviour (writing conflict markers into the working tree). Drivers
in this package are designed to *attempt* a structured merge first and
fall back to writing conflict markers themselves only when the merge is
genuinely ambiguous (e.g. two branches choose different values for the
same JSON key).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .common import (
    DriverArgs,
    parse_args,
    read_text,
    write_text,
    write_conflict_markers,
)


__all__ = (
    "DriverArgs",
    "parse_args",
    "read_text",
    "write_text",
    "write_conflict_markers",
)
