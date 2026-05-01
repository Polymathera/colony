"""``kg-merge`` — merge ``**/*.kg.json`` knowledge-graph files.

Per master §8.4: "Three-way set-merge over RDF triples; conflicts only
when both branches assert contradictory triples about the same subject."

Concrete schema. A ``.kg.json`` file is a JSON object with a single
``triples`` array; each triple is a 3-element list ``[s, p, o]`` of
strings. (This is the colony convention; per-domain dossiers may extend
the file with additional metadata fields, which we preserve verbatim
when both sides agree on them.)

The merge:

1. Triples are treated as a set; the merged triple set is
   ``base ∪ ((ours \\ deleted_by_theirs) ∪ (theirs \\ deleted_by_ours))``.
2. A *contradiction* is two triples ``[s, p, o1]`` and ``[s, p, o2]``
   from opposite sides that:
     - share ``(s, p)`` with a *functional* predicate, and
     - did not share ``(s, p)`` in base.
   Functional predicates are listed in ``FUNCTIONAL_PREDICATES`` below.
   Contradictions raise a conflict on the file.
3. Non-triple top-level fields (e.g. ``namespaces``, ``version``) merge
   structurally with conflict on disagreement.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .common import DriverArgs, parse_args, read_text, write_conflict_markers, write_text


# Predicates that are functional by convention (one-to-one for each
# subject). New domain ontologies extend this list via per-domain merge
# drivers; the colony default is conservative.
FUNCTIONAL_PREDICATES: frozenset[str] = frozenset(
    {
        "rdf:type",
        "rdfs:label",
        "colony:created_at",
        "colony:created_by",
        "colony:supersedes",
        "colony:status",
    }
)


def _to_set(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for t in payload.get("triples", []):
        if not isinstance(t, list) or len(t) != 3:
            continue
        s, p, o = (str(x) for x in t)
        out.add((s, p, o))
    return out


def _detect_contradictions(
    base: set[tuple[str, str, str]],
    ours: set[tuple[str, str, str]],
    theirs: set[tuple[str, str, str]],
) -> list[tuple[str, str, str, str]]:
    """Return ``[(s, p, our_o, their_o), ...]`` for functional contradictions."""

    contradictions: list[tuple[str, str, str, str]] = []
    base_pairs = {(s, p) for (s, p, _) in base}
    ours_added = ours - base
    theirs_added = theirs - base
    by_pair_ours: dict[tuple[str, str], list[str]] = {}
    for s, p, o in ours_added:
        by_pair_ours.setdefault((s, p), []).append(o)
    for s, p, o_t in theirs_added:
        if p not in FUNCTIONAL_PREDICATES:
            continue
        if (s, p) in base_pairs:
            # Both sides may agree to overwrite a functional triple if
            # they pick the *same* new value; if the values differ that
            # is a separate conflict caught below.
            pass
        candidates = by_pair_ours.get((s, p), [])
        for o_o in candidates:
            if o_o != o_t:
                contradictions.append((s, p, o_o, o_t))
    return contradictions


def _merge_triples(
    base: set[tuple[str, str, str]],
    ours: set[tuple[str, str, str]],
    theirs: set[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    deleted_by_ours = base - ours
    deleted_by_theirs = base - theirs
    surviving_base = base - (deleted_by_ours | deleted_by_theirs)
    added = (ours - base) | (theirs - base)
    return surviving_base | added


def _merge_metadata(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Merge non-``triples`` keys with three-way semantics."""

    keys = (set(base) | set(ours) | set(theirs)) - {"triples"}
    out: dict[str, Any] = {}
    conflicts: list[str] = []
    for k in keys:
        b, o, t = base.get(k), ours.get(k), theirs.get(k)
        if o == t:
            if o is not None:
                out[k] = o
            continue
        if o == b and t is not None:
            out[k] = t
            continue
        if t == b and o is not None:
            out[k] = o
            continue
        # Genuine disagreement.
        conflicts.append(k)
        out[k] = {"_conflict": True, "ours": o, "theirs": t, "base": b}
    return out, conflicts


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
            f"kg-merge: {args.pathname}: malformed JSON ({exc}); "
            "falling back to conflict markers.",
            file=sys.stderr,
        )
        write_conflict_markers(args, ours=ours_text, theirs=theirs_text, base=base_text)
        return 1

    base_triples = _to_set(base_payload)
    ours_triples = _to_set(ours_payload)
    theirs_triples = _to_set(theirs_payload)

    contradictions = _detect_contradictions(
        base_triples, ours_triples, theirs_triples
    )
    merged_triples = _merge_triples(base_triples, ours_triples, theirs_triples)
    merged_meta, meta_conflicts = _merge_metadata(
        base_payload, ours_payload, theirs_payload
    )

    output: dict[str, Any] = dict(merged_meta)
    output["triples"] = sorted([list(t) for t in merged_triples])
    write_text(
        args.ours,
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    if contradictions or meta_conflicts:
        if contradictions:
            print(
                "kg-merge: {p}: contradictions: ".format(p=args.pathname)
                + "; ".join(
                    f"{s} {p} ours={o_o} theirs={o_t}"
                    for s, p, o_o, o_t in contradictions
                ),
                file=sys.stderr,
            )
        if meta_conflicts:
            print(
                f"kg-merge: {args.pathname}: metadata conflicts on keys="
                + ", ".join(meta_conflicts),
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
