"""``page-graph-merge`` — regenerate ``**/page_graph.parquet`` from inputs.

Per master §8.4: "Recompute from the union of file-system inputs at the
merge point. Drops the merge driver's job and lets the convergence
runtime regenerate it."

The page graph is a derived index — its content is not authoritative
state but a *materialisation* of edges computed from the per-source
``ContextPageSource``s. After a merge, the only correct content is the
one produced by recomputing from the merged source state.

The driver therefore:

1. Removes the file from the working tree (so the index sees a clean
   slate).
2. Writes a sidecar marker ``<file>.regenerate_required`` so downstream
   runtime checks (the convergence runtime's startup probe) detect the
   need to rebuild before subscription dispatch.
3. Exits 0 — the merge is "successful" in git's sense; the
   regeneration is the runtime's job.

If the file pattern was wired by mistake to a *non-derived* artifact
(e.g. someone committed a hand-edited ``page_graph.parquet``), the
sidecar marker is still written and the regeneration step will simply
overwrite the (incorrect) file with the regenerated one. No data is
lost because the source content this graph was derived from is what
git already merged on its own per-source path.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import DriverArgs, parse_args, write_text


_REGENERATE_NOTE = (
    "# This file is a marker dropped by polymathera.colony.design_monorepo's\n"
    "# page-graph-merge driver. The page_graph at this path was invalidated by\n"
    "# a merge and must be regenerated from the per-source ContextPageSources\n"
    "# before subscription dispatch. The convergence runtime detects this\n"
    "# marker on startup and triggers regeneration; once regenerated, this\n"
    "# marker is removed.\n"
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = Path(args.ours)
    try:
        if target.exists():
            target.unlink()
    except OSError as exc:  # pragma: no cover - filesystem-dependent
        print(
            f"page-graph-merge: failed to remove {target}: {exc}",
            file=sys.stderr,
        )
        return 1
    marker = target.with_suffix(target.suffix + ".regenerate_required")
    write_text(marker, _REGENERATE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
