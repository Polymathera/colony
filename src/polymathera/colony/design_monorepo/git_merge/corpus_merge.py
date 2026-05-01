"""``corpus-merge`` — merge ``corpora/papers/**/metadata.json``.

Per master §8.4: "Append-mostly; same ``(doi, version)`` with conflicting
fields → human review."

Schema. A corpus metadata file is one of:

1. A list of paper records, each a dict with at least ``doi`` and
   optional ``version``.
2. A dict with a top-level ``papers`` array of the same record shape.

The merge keeps every distinct ``(doi, version)`` from either side. For
records that share a ``(doi, version)`` between sides:

- If the records are equal — keep one copy.
- If the records differ but neither side modified the field relative
  to base — keep the side that did modify it.
- Otherwise — record-level conflict that becomes a ``_conflict``
  block in the output and a non-zero exit code.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .common import DriverArgs, parse_args, read_text, write_conflict_markers, write_text


def _records(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)], True
    if isinstance(payload, dict):
        if isinstance(payload.get("papers"), list):
            return (
                [r for r in payload["papers"] if isinstance(r, dict)],
                False,
            )
    return [], False


def _key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("doi", "")), str(record.get("version", ""))


def _three_way(
    base: list[dict[str, Any]],
    ours: list[dict[str, Any]],
    theirs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    base_map = {_key(r): r for r in base}
    ours_map = {_key(r): r for r in ours}
    theirs_map = {_key(r): r for r in theirs}
    keys = set(base_map) | set(ours_map) | set(theirs_map)
    output: list[dict[str, Any]] = []
    conflicts: list[tuple[str, str]] = []
    for k in sorted(keys):
        b, o, t = base_map.get(k), ours_map.get(k), theirs_map.get(k)
        if o == t:
            if o is not None:
                output.append(o)
            continue
        if o == b and t is not None:
            output.append(t)
            continue
        if t == b and o is not None:
            output.append(o)
            continue
        if o is None:
            output.append(t)
            continue
        if t is None:
            output.append(o)
            continue
        conflicts.append(k)
        output.append({
            "doi": k[0],
            "version": k[1],
            "_conflict": True,
            "ours": o,
            "theirs": t,
            "base": b,
        })
    return output, conflicts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ours_text = read_text(args.ours)
    base_text = read_text(args.base)
    theirs_text = read_text(args.theirs)
    try:
        ours_payload = json.loads(ours_text) if ours_text.strip() else []
        base_payload = json.loads(base_text) if base_text.strip() else []
        theirs_payload = json.loads(theirs_text) if theirs_text.strip() else []
    except json.JSONDecodeError as exc:
        print(
            f"corpus-merge: {args.pathname}: malformed JSON ({exc}); "
            "falling back to conflict markers.",
            file=sys.stderr,
        )
        write_conflict_markers(args, ours=ours_text, theirs=theirs_text, base=base_text)
        return 1

    ours_recs, list_shape_ours = _records(ours_payload)
    theirs_recs, list_shape_theirs = _records(theirs_payload)
    base_recs, _ = _records(base_payload)
    list_shape = list_shape_ours or list_shape_theirs or isinstance(ours_payload, list)

    merged, conflicts = _three_way(base_recs, ours_recs, theirs_recs)

    if list_shape:
        output: Any = merged
    else:
        output = {"papers": merged}

    write_text(
        args.ours,
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    if conflicts:
        print(
            f"corpus-merge: {args.pathname}: conflicts on (doi,version)="
            + ", ".join(f"({d!r},{v!r})" for d, v in conflicts),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
