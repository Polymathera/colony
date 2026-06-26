"""``kg-merge`` — merge ``**/*.kg.json`` knowledge-graph files.

Three-way set-merge over RDF triples: claims keyed by ``(subject, predicate,
object)``; conflicts only when both branches assert contradictory triples
about the same subject. Two on-disk shapes are accepted:

- **Claims schema (v1.x)**: ``{"version": "1.0", "namespaces": {...},
  "claims": [{subject, predicate, object, confidence, citation,
  provenance, ...}, ...]}``. Preserved verbatim through the merge
  except where the same triple appears on both sides — see
  ``_pick_winner`` for the deterministic tiebreak.
- **Triples schema (legacy)**: ``{"triples": [[s, p, o], ...]}``.
  Read for backward compatibility with operators who hand-authored
  files before the writer shipped. Emitted only when ALL inputs use
  it; mixing schemas across sides resolves to the richer claims
  shape.

Cross-version inputs (different non-empty ``version`` strings)
abort with a loud conflict — the driver refuses to silently merge
incompatible payloads.

Functional predicates (see :data:`FUNCTIONAL_PREDICATES`) flag a
contradiction when ours / theirs assign different objects to the
same ``(subject, predicate)`` pair that was NOT already present in
``base`` — surfaces extractor disagreement instead of silently
preferring one side.

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

from .common import parse_args, read_text, write_conflict_markers, write_text


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
        "colony:confidence",
    }
)


Triple = tuple[str, str, str]


def _claim_triple(claim: dict[str, Any]) -> Triple | None:
    try:
        return (
            str(claim["subject"]),
            str(claim["predicate"]),
            str(claim.get("object", claim.get("object_"))),
        )
    except KeyError:
        return None


def _coerce_to_claims(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise both shapes into a list of claim dicts. Triples
    schema entries become minimal claim dicts with no provenance."""

    out: list[dict[str, Any]] = []
    for raw in payload.get("claims", []):
        if isinstance(raw, dict) and _claim_triple(raw) is not None:
            out.append(dict(raw))
    for raw in payload.get("triples", []):
        if isinstance(raw, list) and len(raw) == 3:
            s, p, o = (str(x) for x in raw)
            out.append({"subject": s, "predicate": p, "object": o})
    return out


def _index_by_triple(
    claims: list[dict[str, Any]],
) -> dict[Triple, dict[str, Any]]:
    """When duplicates appear inside one payload, keep the winner per
    the same rules used cross-side (highest confidence; lexicographic
    run-id tiebreak)."""

    out: dict[Triple, dict[str, Any]] = {}
    for c in claims:
        key = _claim_triple(c)
        if key is None:
            continue
        existing = out.get(key)
        out[key] = c if existing is None else _pick_winner(existing, c)
    return out


def _pick_winner(
    a: dict[str, Any], b: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic merge of two claim dicts sharing the same
    ``(subject, predicate, object)``: higher confidence wins;
    on tie, lexicographic ``provenance.extractor_run_id`` wins
    (longer-id wins implicitly for partially-populated provenance)."""

    a_conf = float(a.get("confidence", 0.0) or 0.0)
    b_conf = float(b.get("confidence", 0.0) or 0.0)
    if a_conf != b_conf:
        return a if a_conf > b_conf else b
    a_run = str((a.get("provenance") or {}).get("extractor_run_id", ""))
    b_run = str((b.get("provenance") or {}).get("extractor_run_id", ""))
    return a if a_run >= b_run else b


def _detect_contradictions(
    base: dict[Triple, dict[str, Any]],
    ours: dict[Triple, dict[str, Any]],
    theirs: dict[Triple, dict[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """Functional-predicate disagreements between ours and theirs that
    were not already present in base."""

    contradictions: list[tuple[str, str, str, str]] = []
    added_ours = {k: v for k, v in ours.items() if k not in base}
    added_theirs = {k: v for k, v in theirs.items() if k not in base}
    by_pair_ours: dict[tuple[str, str], list[str]] = {}
    for (s, p, o), _ in added_ours.items():
        by_pair_ours.setdefault((s, p), []).append(o)
    for (s, p, o_t), _ in added_theirs.items():
        if p not in FUNCTIONAL_PREDICATES:
            continue
        for o_o in by_pair_ours.get((s, p), []):
            if o_o != o_t:
                contradictions.append((s, p, o_o, o_t))
    return contradictions


def _merge_claims(
    base: dict[Triple, dict[str, Any]],
    ours: dict[Triple, dict[str, Any]],
    theirs: dict[Triple, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Set-merge by triple key. ``base - ours`` is treated as
    deleted-by-ours and excluded from the result; same for theirs.
    Surviving entries pick the highest-confidence representative
    across sides."""

    deleted_by_ours = set(base) - set(ours)
    deleted_by_theirs = set(base) - set(theirs)
    surviving_base_keys = (
        set(base) - (deleted_by_ours | deleted_by_theirs)
    )
    merged: dict[Triple, dict[str, Any]] = {}
    for key in surviving_base_keys:
        merged[key] = base[key]
    for src in (ours, theirs):
        for key, claim in src.items():
            if key in base and key not in surviving_base_keys:
                continue
            existing = merged.get(key)
            merged[key] = claim if existing is None else _pick_winner(
                existing, claim,
            )
    return [
        merged[key]
        for key in sorted(merged.keys())
    ]


def _merge_metadata(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Merge non-claim top-level keys with three-way semantics."""

    reserved = {"claims", "triples"}
    keys = (set(base) | set(ours) | set(theirs)) - reserved
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


def _payload_uses_claims_schema(payload: dict[str, Any]) -> bool:
    return bool(payload.get("claims"))


def _versions_compatible(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any],
) -> tuple[bool, str | None]:
    """Cross-version refusal: any two sides with non-empty version
    strings that don't agree → loud conflict."""

    versions = {
        side.get("version")
        for side in (base, ours, theirs)
        if isinstance(side.get("version"), str) and side.get("version")
    }
    if len(versions) <= 1:
        return True, None
    return False, ", ".join(sorted(v for v in versions if v))


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
        write_conflict_markers(
            args, ours=ours_text, theirs=theirs_text, base=base_text,
        )
        return 1

    ok, mismatch = _versions_compatible(
        base_payload, ours_payload, theirs_payload,
    )
    if not ok:
        print(
            f"kg-merge: {args.pathname}: refusing to merge across "
            f"schema versions ({mismatch}). Upgrade the older "
            "snapshot to the new shape and retry.",
            file=sys.stderr,
        )
        write_conflict_markers(
            args, ours=ours_text, theirs=theirs_text, base=base_text,
        )
        return 1

    base_claims = _index_by_triple(_coerce_to_claims(base_payload))
    ours_claims = _index_by_triple(_coerce_to_claims(ours_payload))
    theirs_claims = _index_by_triple(_coerce_to_claims(theirs_payload))

    contradictions = _detect_contradictions(
        base_claims, ours_claims, theirs_claims,
    )
    merged_claims = _merge_claims(base_claims, ours_claims, theirs_claims)
    merged_meta, meta_conflicts = _merge_metadata(
        base_payload, ours_payload, theirs_payload,
    )

    output: dict[str, Any] = dict(merged_meta)
    use_claims_shape = (
        _payload_uses_claims_schema(ours_payload)
        or _payload_uses_claims_schema(theirs_payload)
        or _payload_uses_claims_schema(base_payload)
    )
    if use_claims_shape:
        output["claims"] = merged_claims
    else:
        output["triples"] = sorted(
            list(_claim_triple(c)) for c in merged_claims
            if _claim_triple(c) is not None
        )

    write_text(
        args.ours,
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
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
