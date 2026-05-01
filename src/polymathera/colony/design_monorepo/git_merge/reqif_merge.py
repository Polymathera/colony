"""``reqif-merge`` — merge ``**/requirements/*.reqif`` files.

Per master §8.4: "ReqIF-aware merge using doorstop's reconciliation."

ReqIF is an XML format. A real implementation in production would
delegate to ``doorstop`` or a ReqIF-aware library; in this baseline we
implement a structurally-aware fallback that:

1. Parses all three sides as XML.
2. Identifies elements by ``IDENTIFIER`` (the standard ReqIF id
   attribute on every spec object). Each unique IDENTIFIER becomes a
   merge unit.
3. Three-way merges per IDENTIFIER:
   - Both sides equal → keep one.
   - One side untouched → keep the other.
   - Both modified differently → conflict on that IDENTIFIER.
4. Reassembles the XML preserving the original document structure as
   much as possible.

When XML parsing fails on either side, falls back to standard text
conflict markers — the failure is surfaced rather than silently masked.

NOTE: this is intentionally narrower than a full ReqIF reconciler. It
covers the 95 % case (decisions and authoring identifiers stable; only
attribute or text changes per requirement) and lets the user resolve
the rare structural-rewrite cases by hand.
"""

from __future__ import annotations

import sys
from typing import Any
from xml.etree import ElementTree as ET

from .common import DriverArgs, parse_args, read_text, write_conflict_markers, write_text


_ID_ATTR = "IDENTIFIER"


def _parse(text: str) -> ET.ElementTree | None:
    if not text.strip():
        return None
    try:
        return ET.ElementTree(ET.fromstring(text))
    except ET.ParseError:
        return None


def _index_by_id(root: ET.Element) -> dict[str, ET.Element]:
    out: dict[str, ET.Element] = {}
    for el in root.iter():
        ident = el.attrib.get(_ID_ATTR)
        if ident:
            out[ident] = el
    return out


def _element_signature(el: ET.Element) -> str:
    """Build a normalised string signature for equality testing."""

    return ET.tostring(el, encoding="unicode")


def _three_way(
    base_root: ET.Element | None,
    ours_root: ET.Element,
    theirs_root: ET.Element,
) -> tuple[list[str], dict[str, ET.Element]]:
    """Return (conflict_ids, merged_id_map)."""

    base_idx = _index_by_id(base_root) if base_root is not None else {}
    ours_idx = _index_by_id(ours_root)
    theirs_idx = _index_by_id(theirs_root)
    keys = set(ours_idx) | set(theirs_idx) | set(base_idx)
    merged: dict[str, ET.Element] = {}
    conflicts: list[str] = []
    for k in keys:
        b = base_idx.get(k)
        o = ours_idx.get(k)
        t = theirs_idx.get(k)
        b_sig = _element_signature(b) if b is not None else None
        o_sig = _element_signature(o) if o is not None else None
        t_sig = _element_signature(t) if t is not None else None
        if o_sig == t_sig:
            if o is not None:
                merged[k] = o
            continue
        if o_sig == b_sig and t is not None:
            merged[k] = t
            continue
        if t_sig == b_sig and o is not None:
            merged[k] = o
            continue
        if o is None and t is not None:
            merged[k] = t
            continue
        if t is None and o is not None:
            merged[k] = o
            continue
        conflicts.append(k)
        # Pick ours as the placeholder; the conflict is reported separately.
        merged[k] = o if o is not None else t  # type: ignore[assignment]
    return conflicts, merged


def _replace_in_place(
    ours_root: ET.Element,
    theirs_root: ET.Element,
    merged: dict[str, ET.Element],
) -> None:
    """Apply the merge result to ``ours_root`` in-place.

    Three classes of update:

    1. An IDENTIFIER present in both sides — the merged element replaces
       the original ours element at its existing position.
    2. An IDENTIFIER deleted on ours but resurrected through merge (i.e.
       the merged map contains it but ours_root does not) — we look up
       the corresponding parent in theirs_root and append the merged
       element to the same-named parent in ours_root (best effort).
    3. An IDENTIFIER deleted on both sides — nothing to do.
    """

    ours_parents: dict[ET.Element, ET.Element] = {}
    for parent in ours_root.iter():
        for child in list(parent):
            ours_parents[child] = parent

    ours_index: dict[str, ET.Element] = {}
    for el in ours_root.iter():
        ident = el.attrib.get(_ID_ATTR)
        if ident:
            ours_index[ident] = el

    for ident, replacement in merged.items():
        existing = ours_index.get(ident)
        if existing is not None:
            if existing is replacement:
                continue
            parent = ours_parents.get(existing)
            if parent is None:
                continue
            idx = list(parent).index(existing)
            parent.remove(existing)
            parent.insert(idx, replacement)
            continue
        # New on theirs (or resurrected). Find the parent in theirs_root,
        # locate the matching parent in ours_root by tag/attrib, append.
        host = _find_parent_in_theirs(theirs_root, ident)
        target_parent = _match_parent(ours_root, host) if host is not None else ours_root
        target_parent.append(replacement)


def _find_parent_in_theirs(
    theirs_root: ET.Element, ident: str
) -> ET.Element | None:
    for parent in theirs_root.iter():
        for child in list(parent):
            if child.attrib.get(_ID_ATTR) == ident:
                return parent
    return None


def _match_parent(
    ours_root: ET.Element, theirs_parent: ET.Element
) -> ET.Element:
    """Best-effort match: return the first ours element that shares
    tag and (where present) IDENTIFIER with ``theirs_parent``; else the
    ours root."""

    theirs_id = theirs_parent.attrib.get(_ID_ATTR)
    for el in ours_root.iter():
        if el.tag != theirs_parent.tag:
            continue
        if theirs_id is None or el.attrib.get(_ID_ATTR) == theirs_id:
            return el
    return ours_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ours_text = read_text(args.ours)
    base_text = read_text(args.base)
    theirs_text = read_text(args.theirs)

    ours_tree = _parse(ours_text)
    theirs_tree = _parse(theirs_text)
    base_tree = _parse(base_text)
    if ours_tree is None or theirs_tree is None:
        print(
            f"reqif-merge: {args.pathname}: malformed XML on one side; "
            "falling back to conflict markers.",
            file=sys.stderr,
        )
        write_conflict_markers(args, ours=ours_text, theirs=theirs_text, base=base_text)
        return 1
    conflicts, merged_map = _three_way(
        base_tree.getroot() if base_tree is not None else None,
        ours_tree.getroot(),
        theirs_tree.getroot(),
    )
    _replace_in_place(ours_tree.getroot(), theirs_tree.getroot(), merged_map)
    serialized = ET.tostring(ours_tree.getroot(), encoding="unicode")
    write_text(args.ours, serialized + ("\n" if not serialized.endswith("\n") else ""))
    if conflicts:
        print(
            f"reqif-merge: {args.pathname}: conflicts on IDENTIFIERs: "
            + ", ".join(conflicts),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
