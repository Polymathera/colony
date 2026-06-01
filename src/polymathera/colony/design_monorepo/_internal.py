
from __future__ import annotations

from pathlib import Path
from typing import Any

# Per-query scan cap on the path-1 KG search. The existing GraphStore
# DSL (``MATCH (s)-[r]->(o) [WHERE …] [LIMIT n]``) doesn't support
# ``LIKE`` / prefix filters on edge properties, so ``_search_kuzu``
# pulls a bounded set of edges and post-filters by citation-URI
# prefix + text match. Generous for the deterministic extractor
# (roughly O(file_count) claims); P3d's LLM extractor will raise
# claim density, at which point a citation-URI-prefix index in the
# DSL should be considered.
SYSDES_KUZU_SCAN_LIMIT = 10_000

# Stable URI scheme for design-context ingestion. ``Claim`` /
# ``Chunk`` citations land with ``source_uri = f"design_context://
# {row.name}/{rel_path}"``, which is how downstream KG queries
# (``find_inconsistencies``, ``search_design_context(path='kuzu')``)
# filter "claims that came from design context" vs literature.
DESIGN_CONTEXT_URI_SCHEME = "design_context"


def parse_design_context_uri(uri: str) -> tuple[str, str]:
    """Parse a ``design_context://<source_name>/<rel_path>`` URI into
    ``(source_name, rel_path)``. Returns ``("", "")`` when the URI
    doesn't conform — callers should have filtered to design-context
    URIs upstream, so this is a defensive last resort."""

    prefix = f"{DESIGN_CONTEXT_URI_SCHEME}://"
    if not uri.startswith(prefix):
        return ("", "")
    rest = uri[len(prefix):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        return (parts[0], "") if parts else ("", "")
    return (parts[0], parts[1])



def sysdes_list_files(
    repo_root: Path, source: Any,
) -> list[Path]:
    """Walk ``repo_root`` once and return paths matching ``source``.

    Sorted deterministically so action output ordering is stable
    across runs (the planner caches plans by output hash).
    """

    matched: list[Path] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if source.matches(rel):
            matched.append(p)
    matched.sort()
    return matched

# Conservative ceilings for the v1 surface — avoid LLM context blowups for
# repos with hundreds of design-context files. The summary action truncates
# per-source file lists past this; the planner can re-call with a tighter
# ``source_names`` filter to drill into a specific row.
SYSDES_MAX_FILES_PER_SOURCE_IN_SUMMARY = 50
_SYSDES_MAX_HEADINGS_PEEK = 5
_SYSDES_SNIPPET_LINES_AROUND_MATCH = 2
_SYSDES_SUMMARY_BYTES_PER_FILE_READ = 4096


def sysdes_peek_headings(
    file_path: Path,
    max_headings: int = _SYSDES_MAX_HEADINGS_PEEK,
    max_bytes: int = _SYSDES_SUMMARY_BYTES_PER_FILE_READ,
) -> list[str]:
    """Read the first ``max_bytes`` of a file and return up to
    ``max_headings`` H1/H2 lines, stripped."""

    headings: list[str] = []
    try:
        with file_path.open("rb") as fh:
            chunk = fh.read(max_bytes)
    except OSError:
        return headings
    text = chunk.decode("utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            headings.append(stripped.rstrip())
            if len(headings) >= max_headings:
                break
    return headings


def sysdes_grep_file(
    file_path: Path,
    pattern: Any,
    max_hits: int,
    snippet_lines_around: int = _SYSDES_SNIPPET_LINES_AROUND_MATCH,
) -> list[tuple[int, str]]:
    """Read ``file_path`` and return up to ``max_hits`` matches as
    ``(line_no_1_based, snippet_with_surrounding_lines)``. Returns
    ``[]`` on read errors so the search can continue."""

    if max_hits <= 0:
        return []
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if pattern.search(line):
            start = max(0, i - snippet_lines_around)
            end = min(len(lines), i + snippet_lines_around + 1)
            out.append((i + 1, "\n".join(lines[start:end])))
            if len(out) >= max_hits:
                break
    return out
