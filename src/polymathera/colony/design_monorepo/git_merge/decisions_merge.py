"""``decisions-merge`` — merge ``design/decisions/*.json`` files.

Per master §8.4: "by ``decision_id``; same id with different choices →
conflict".

A decision file is one of two shapes:

1. A single decision object — a JSON object with a top-level
   ``decision_id`` field. This is the convention used when each
   decision lives in its own file (``design/decisions/<id>.json``).
2. A list of decision objects, each with a ``decision_id``. Used when
   a domain bundles decisions per topic.

The merge rule:

- Build maps keyed by ``decision_id`` for *base*, *ours*, and *theirs*.
- For each id present in either side: prefer the side that *changed*
  it relative to base (classic three-way merge).
- If both sides changed the same id and produced *non-equal* results →
  conflict on that id.
- Output is the merged map serialised back into the original shape.

Falls back to standard text conflict markers when the inputs are not
both well-formed JSON (so the format error surfaces instead of being
silently swallowed).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .common import DriverArgs, parse_args, read_text, write_conflict_markers, write_text


def _key_decisions(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "decision_id" in payload:
            return {str(payload["decision_id"]): payload}
        # Object-keyed map of decisions.
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    if isinstance(payload, list):
        out: dict[str, Any] = {}
        for item in payload:
            if isinstance(item, dict) and "decision_id" in item:
                out[str(item["decision_id"])] = item
        return out
    return {}


def _is_list_shape(*payloads: Any) -> bool:
    return any(isinstance(p, list) for p in payloads)


def _serialise(merged: dict[str, Any], list_shape: bool) -> str:
    if list_shape:
        return json.dumps(
            sorted(merged.values(), key=lambda d: str(d.get("decision_id", ""))),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n"
    if len(merged) == 1:
        only = next(iter(merged.values()))
        return json.dumps(only, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return json.dumps(
        merged, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"


def _merge_fields(
    b: Any, o: Any, t: Any, conflicts: list[str], path: str = ""
) -> Any:
    """Recursive three-way merge over nested dicts/lists/scalars.

    - dicts: per-key merge.
    - lists: equal-or-conflict (decisions don't recurse into list shape;
      they're typically string lists, not nested structures).
    - scalars: equal -> keep, else conflict if both modified.
    """

    if o == t:
        return o
    if o == b:
        return t
    if t == b:
        return o
    if isinstance(b, dict) and isinstance(o, dict) and isinstance(t, dict):
        merged: dict[str, Any] = {}
        for k in set(b) | set(o) | set(t):
            kp = f"{path}.{k}" if path else k
            merged[k] = _merge_fields(b.get(k), o.get(k), t.get(k), conflicts, kp)
        return merged
    if o is None and t is not None:
        return t
    if t is None and o is not None:
        return o
    conflicts.append(path or "<root>")
    return {"_conflict": True, "ours": o, "theirs": t, "base": b}


def _three_way(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Three-way merge keyed by ``decision_id``.

    For each id present in either side: prefer the side that changed
    it relative to base. When *both* sides changed a decision, recurse
    into its fields and merge per-field; conflicts surface only on
    fields both sides modified to different values (master §8.4).
    """

    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    keys = set(base) | set(ours) | set(theirs)
    for k in keys:
        b = base.get(k)
        o = ours.get(k)
        t = theirs.get(k)
        if o == t:
            if o is not None:
                merged[k] = o
            continue
        if o == b and t is not None:
            merged[k] = t
            continue
        if t == b and o is not None:
            merged[k] = o
            continue
        if o is None and t is not None:
            merged[k] = t
            continue
        if t is None and o is not None:
            merged[k] = o
            continue
        # Both sides changed; merge field-by-field.
        local_conflicts: list[str] = []
        merged_obj = _merge_fields(b, o, t, local_conflicts, path=str(k))
        if local_conflicts:
            conflicts.extend(local_conflicts)
        merged[k] = merged_obj
    return merged, conflicts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ours_text = read_text(args.ours)
    base_text = read_text(args.base)
    theirs_text = read_text(args.theirs)
    try:
        ours_payload = json.loads(ours_text) if ours_text.strip() else {}
        base_payload = json.loads(base_text) if base_text.strip() else {}
        theirs_payload = json.loads(theirs_text) if theirs_text.strip() else {}
    except json.JSONDecodeError as exc:
        print(
            f"decisions-merge: {args.pathname}: malformed JSON ({exc}); "
            "falling back to conflict markers.",
            file=sys.stderr,
        )
        write_conflict_markers(
            args, ours=ours_text, theirs=theirs_text, base=base_text,
        )
        return 1

    list_shape = _is_list_shape(ours_payload, base_payload, theirs_payload)
    merged, conflicts = _three_way(
        _key_decisions(base_payload),
        _key_decisions(ours_payload),
        _key_decisions(theirs_payload),
    )
    write_text(args.ours, _serialise(merged, list_shape))
    if conflicts:
        print(
            f"decisions-merge: {args.pathname}: conflicts on decision_id="
            + ", ".join(conflicts),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
