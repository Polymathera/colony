"""``budget-merge`` — merge ``**/budgets/*.yaml`` budget trees.

Per master §8.4: "Three-way merge over the budget tree; numeric leaves
are merged by max-of-allocations on increase, min on decrease
(configurable per budget); structural changes raise conflict."

A budget file is a YAML mapping of nested keys to numeric leaves. The
schema we honour:

```yaml
metadata:
  budget_id: weight
  policy: max_on_increase     # or min_on_increase
units: kg
tree:
  vehicle:
    chassis: 320.0
    powertrain: 410.0
    aerodynamics: 80.0
```

Merge rules:

1. ``metadata.policy`` resolves the leaf-merge direction:
   - ``max_on_increase`` (default): if both ours and theirs *increase*
     the leaf vs base, keep the larger; if they both *decrease*, keep
     the smaller (a conservative reduction); a mixed case (one
     increases, one decreases) is a conflict.
   - ``min_on_increase``: mirror.
2. Structural changes — adding or removing a sub-tree on one side
   while the other modifies the same path — raise conflict.
3. Non-numeric leaves merge with strict equality; disagreement is a
   conflict.

YAML parsing prefers ``ruamel.yaml`` if available (preserves comments
and key order); falls back to ``PyYAML``; falls back to writing
conflict markers if neither is available.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from .common import DriverArgs, parse_args, read_text, write_conflict_markers, write_text


def _load_yaml(text: str):
    if not text.strip():
        return None, None  # value, dumper
    # Prefer ruamel.yaml if installed.
    try:
        from ruamel.yaml import YAML  # type: ignore[import-not-found]

        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        from io import StringIO

        return yaml.load(StringIO(text)), ("ruamel", yaml)
    except Exception:  # noqa: BLE001
        pass
    try:
        import yaml as pyyaml  # type: ignore[import-not-found]

        return pyyaml.safe_load(text), ("pyyaml", pyyaml)
    except Exception:  # noqa: BLE001
        return None, None


def _dump_yaml(value: Any, dumper) -> str:
    if dumper is None:
        # Best effort: JSON is YAML-compatible for our schema.
        import json

        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    name, impl = dumper
    if name == "ruamel":
        from io import StringIO

        out = StringIO()
        impl.dump(value, out)
        return out.getvalue()
    if name == "pyyaml":
        return impl.safe_dump(value, sort_keys=False, allow_unicode=True)
    raise RuntimeError(f"Unknown YAML dumper: {name}")


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _merge(
    base: Any, ours: Any, theirs: Any, policy: str, conflicts: list[str], path: str = ""
) -> Any:
    if ours == theirs:
        return ours
    if _is_mapping(base) or _is_mapping(ours) or _is_mapping(theirs):
        if not (_is_mapping(ours) and _is_mapping(theirs)):
            conflicts.append(path or "<root>")
            return {"_conflict": True, "ours": ours, "theirs": theirs, "base": base}
        keys = set(ours) | set(theirs) | (set(base) if _is_mapping(base) else set())
        out: dict[str, Any] = {}
        for k in sorted(keys):
            b = base.get(k) if _is_mapping(base) else None
            o = ours.get(k) if _is_mapping(ours) else None
            t = theirs.get(k) if _is_mapping(theirs) else None
            if k not in ours and k in theirs and (not _is_mapping(base) or k not in base):
                out[k] = t
                continue
            if k not in theirs and k in ours and (not _is_mapping(base) or k not in base):
                out[k] = o
                continue
            if k in ours and k not in theirs and _is_mapping(base) and k in base:
                # Theirs deleted; ours kept or modified.
                if o == b:
                    continue  # delete prevails
                conflicts.append(f"{path}/{k}".lstrip("/"))
                out[k] = {"_conflict": "delete-vs-modify", "ours": o, "base": b}
                continue
            if k in theirs and k not in ours and _is_mapping(base) and k in base:
                if t == b:
                    continue
                conflicts.append(f"{path}/{k}".lstrip("/"))
                out[k] = {"_conflict": "delete-vs-modify", "theirs": t, "base": b}
                continue
            out[k] = _merge(b, o, t, policy, conflicts, f"{path}/{k}".lstrip("/"))
        return out
    # Numeric leaf merge.
    if isinstance(ours, (int, float)) and isinstance(theirs, (int, float)):
        b = base if isinstance(base, (int, float)) else None
        if b is None:
            # Both sides added a numeric leaf without a base — conflict
            # only if they disagree.
            conflicts.append(path or "<root>")
            return {"_conflict": True, "ours": ours, "theirs": theirs}
        ours_delta = ours - b
        theirs_delta = theirs - b
        if ours_delta == 0:
            return theirs
        if theirs_delta == 0:
            return ours
        # Mixed direction — conflict.
        if (ours_delta > 0) != (theirs_delta > 0):
            conflicts.append(path or "<root>")
            return {"_conflict": "mixed-direction", "ours": ours, "theirs": theirs, "base": b}
        # Same direction.
        if policy == "min_on_increase":
            if ours_delta > 0 and theirs_delta > 0:
                return min(ours, theirs)
            return max(ours, theirs)  # both decreasing, prefer larger (less conservative)
        # max_on_increase
        if ours_delta > 0 and theirs_delta > 0:
            return max(ours, theirs)
        return min(ours, theirs)
    # Non-numeric leaf disagreement.
    conflicts.append(path or "<root>")
    return {"_conflict": True, "ours": ours, "theirs": theirs, "base": base}


def _policy(payload: Any) -> str:
    if _is_mapping(payload):
        meta = payload.get("metadata") or {}
        if _is_mapping(meta):
            value = meta.get("policy")
            if value in ("max_on_increase", "min_on_increase"):
                return str(value)
    return "max_on_increase"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ours_text = read_text(args.ours)
    base_text = read_text(args.base)
    theirs_text = read_text(args.theirs)
    ours_payload, dumper = _load_yaml(ours_text)
    base_payload, _ = _load_yaml(base_text)
    theirs_payload, theirs_dumper = _load_yaml(theirs_text)
    if dumper is None:
        dumper = theirs_dumper
    if ours_payload is None and theirs_payload is None and base_payload is None:
        # Nothing to merge; let git handle empty/empty case.
        write_text(args.ours, "")
        return 0
    if dumper is None:
        # No YAML library available — fall back to conflict markers
        # rather than corrupt the file.
        print(
            f"budget-merge: {args.pathname}: no YAML library available; "
            "falling back to conflict markers.",
            file=sys.stderr,
        )
        write_conflict_markers(args, ours=ours_text, theirs=theirs_text, base=base_text)
        return 1

    policy = _policy(ours_payload) or _policy(theirs_payload) or _policy(base_payload)
    conflicts: list[str] = []
    merged = _merge(base_payload, ours_payload, theirs_payload, policy, conflicts)
    write_text(args.ours, _dump_yaml(merged, dumper))
    if conflicts:
        print(
            f"budget-merge: {args.pathname}: conflicts at "
            + ", ".join(conflicts),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
