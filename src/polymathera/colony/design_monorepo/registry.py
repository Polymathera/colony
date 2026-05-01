"""``.colony/tool-registry.json`` — typed read/write + capability resolution.

The tool registry is a per-monorepo JSON file listing every tool the
program has built or imported, alongside its location, capability,
licence, and headless / container metadata.

The capability layer queries it via two operations:

- ``find_existing_tool(capability_query)`` (master §3.5.1, §9.4) — a
  ranked search used by every tool-building pool *before* it considers
  bootstrapping a new tool.
- ``register_tool(entry)`` — appends a new entry and writes the file
  back. Used by ``ToolBuilder.bootstrap_repo`` after the scaffold is
  committed.

Imported tooling-monorepo remotes (master §9.5) layer on top: their
``tool-registry.json`` is read out of a fetched ref and folded into the
search index. Their entries are read-only by default.

The scoring function for ``find_existing_tool`` is intentionally simple:
exact capability match > capability prefix match > word-overlap on
purpose / name / description. The doc reserves the right to swap in a
better embedding-based matcher later — keeping the surface narrow now
prevents that swap from being a breaking change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from .models import ImportedRemote, ToolEntry, ToolMatch


REGISTRY_RELATIVE_PATH = ".colony/tool-registry.json"
REGISTRY_SCHEMA_VERSION = 1


class ToolRegistryError(ValueError):
    """Raised when the registry file is missing or malformed."""


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"schema_version": REGISTRY_SCHEMA_VERSION, "tools": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolRegistryError(
            f"Tool registry at {path} is not valid JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise ToolRegistryError(
            f"Tool registry at {path} must be a JSON object.",
        )
    version = payload.get("schema_version", 1)
    if version > REGISTRY_SCHEMA_VERSION:
        raise ToolRegistryError(
            f"Tool registry schema_version={version} is newer than this "
            f"colony build understands ({REGISTRY_SCHEMA_VERSION}).",
        )
    return payload


def load_registry(repo_root: Path) -> tuple[ToolEntry, ...]:
    """Read the local registry from ``<repo_root>/.colony/tool-registry.json``.

    Returns an empty tuple when the file does not exist (a fresh repo
    has no tools yet — that is not an error).
    """

    payload = _read(repo_root / REGISTRY_RELATIVE_PATH)
    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ToolRegistryError(
            "Tool registry 'tools' field must be a list.",
        )
    return tuple(ToolEntry.model_validate(item) for item in raw_tools)


def write_registry(repo_root: Path, entries: Sequence[ToolEntry]) -> Path:
    """Write the registry back. Caller stages + commits the file."""

    path = repo_root / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "tools": [e.model_dump(mode="json") for e in entries],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def upsert_tool(
    repo_root: Path,
    entry: ToolEntry,
) -> tuple[ToolEntry, ...]:
    """Insert or replace an entry by ``(purpose, name)``.

    Returns the new full registry tuple.
    """

    existing = list(load_registry(repo_root))
    key = (entry.purpose, entry.name)
    replaced = False
    for i, current in enumerate(existing):
        if (current.purpose, current.name) == key:
            existing[i] = entry
            replaced = True
            break
    if not replaced:
        existing.append(entry)
    write_registry(repo_root, existing)
    return tuple(existing)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _score(query: str, entry: ToolEntry) -> float:
    """Return a relevance score on [0, 1] for ``query`` against ``entry``.

    Scoring is intentionally narrow:

    - 1.0 — exact case-insensitive match against the capability key.
    - 0.85 — the capability key starts with the query (e.g. query
      ``simulate`` matches ``simulate_us_rf``) or the query starts with
      the capability key (a more-specific capability matches a broader
      query).
    - 0.0–0.7 — word-overlap fraction over (capability + name +
      purpose + description), measured by Jaccard on tokenised words.
    """

    q_lower = query.strip().lower()
    if not q_lower:
        return 0.0
    cap_lower = entry.capability.lower()
    if cap_lower == q_lower:
        return 1.0
    if cap_lower.startswith(q_lower) or q_lower.startswith(cap_lower):
        return 0.85
    q_tokens = _tokenize(query)
    e_tokens = _tokenize(
        " ".join(
            (
                entry.capability,
                entry.name,
                entry.purpose,
                str(entry.extra.get("description", "")),
            ),
        ),
    )
    if not q_tokens or not e_tokens:
        return 0.0
    overlap = q_tokens & e_tokens
    if not overlap:
        return 0.0
    union = q_tokens | e_tokens
    return min(0.7, len(overlap) / len(union))


def search(
    capability_query: str,
    *,
    local_entries: Iterable[ToolEntry],
    remote_entries: Iterable[ToolEntry] = (),
    require_writable: bool = False,
    min_score: float = 0.05,
) -> tuple[ToolMatch, ...]:
    """Score ``local_entries`` (writable) and ``remote_entries`` (read-only).

    Returns matches ordered by descending score. Entries with a score
    below ``min_score`` are dropped to keep the result list focused.
    When ``require_writable`` is True, remote entries are excluded.
    """

    matches: list[ToolMatch] = []
    for entry in local_entries:
        score = _score(capability_query, entry)
        if score < min_score:
            continue
        matches.append(ToolMatch(entry=entry, score=score, writable=True))
    if not require_writable:
        for entry in remote_entries:
            score = _score(capability_query, entry)
            if score < min_score:
                continue
            matches.append(ToolMatch(entry=entry, score=score, writable=False))
    matches.sort(key=lambda m: (-m.score, m.entry.name))
    return tuple(matches)


__all__ = (
    "REGISTRY_RELATIVE_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "ToolRegistryError",
    "load_registry",
    "write_registry",
    "upsert_tool",
    "search",
)
