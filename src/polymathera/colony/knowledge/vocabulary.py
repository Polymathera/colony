"""Predicate vocabulary — a living controlled vocabulary over the KG's
open extraction (``colony/predicate_vocabulary_plan.md``).

Extraction mints predicates freely; every predicate is REGISTERED here
(``provisional`` on first sight) and consolidated over time through
typed, logged revision operations. The model composes four established
practices:

- SKOS: one canonical name per term, ``aliases`` (altLabels) absorbing
  merged surface forms, ``broader`` links forming the hierarchy.
- RDFS ``subPropertyOf`` semantics: ``broader`` gives query-time
  abstraction (subsumption closure) WITHOUT restricting minting.
- OBO lifecycle: terms are never deleted — ``deprecated`` with an
  optional ``replaced_by``; old data stays interpretable forever.
- Wikidata-style governance: open minting + curated consolidation;
  destructive operations are human-gated (enforced by callers via the
  approval machinery — this module is pure domain logic).

The registry serializes to ``.colony/colony.vocab.json`` in the design
monorepo (sibling of ``colony.kg.json``); the ``operations`` list is the
append-only audit trail. This module has NO I/O and NO knowledge-deps
imports — persistence wiring lives in :mod:`.persistence`.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


#: Path of the vocabulary registry relative to the design monorepo root.
VOCAB_FILE_RELATIVE_PATH = Path(".colony") / "colony.vocab.json"

VOCAB_SCHEMA_VERSION = "1.0"

#: Bound on ``replaced_by`` chains during resolution — a cycle in the
#: registry is a data error and must fail loud, not hang.
_MAX_RESOLUTION_HOPS = 20


class VocabError(RuntimeError):
    """Invalid vocabulary state or operation — always raised loudly;
    revision passes must surface bad proposals, never absorb them."""


class VocabTermStatus(str, Enum):
    PROVISIONAL = "provisional"
    """Minted by extraction, not yet reviewed. Excluded from the
    extractor's soft prior until promoted."""
    ACTIVE = "active"
    """Reviewed and kept — eligible for the extractor's soft prior."""
    DEPRECATED = "deprecated"
    """Retired (merged away or judged noise). Never deleted; resolution
    follows ``replaced_by`` when present."""


class VocabOpType(str, Enum):
    MERGE = "merge"            # a -> b; a deprecated with replaced_by=b
    RENAME = "rename"          # merge into a newly minted active term
    ADD_ALIAS = "add_alias"
    ADD_BROADER = "add_broader"
    REMOVE_BROADER = "remove_broader"
    DEPRECATE = "deprecate"    # retire; replaced_by optional
    PROMOTE = "promote"        # provisional -> active
    SPLIT = "split"            # recorded manually; claims reassigned by hand


#: Operations that shrink or redirect the vocabulary — callers MUST gate
#: these behind human approval. Additive operations may auto-apply.
DESTRUCTIVE_OP_TYPES = frozenset({
    VocabOpType.MERGE,
    VocabOpType.RENAME,
    VocabOpType.DEPRECATE,
    VocabOpType.SPLIT,
})


class VocabTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: VocabTermStatus = VocabTermStatus.PROVISIONAL
    replaced_by: str | None = None
    aliases: list[str] = Field(default_factory=list)
    broader: list[str] = Field(default_factory=list)
    description: str = ""
    provenance: dict[str, str] = Field(default_factory=dict)


class VocabOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op_id: str = Field(default_factory=lambda: f"vop_{uuid.uuid4().hex[:12]}")
    op_type: VocabOpType
    term: str
    """The term the operation acts on."""
    target: str | None = None
    """Second operand: merge/rename destination, broader parent,
    ``replaced_by`` for deprecate, new-term names for split."""
    rationale: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    proposed_by: str = ""
    approved_by: str = ""
    applied_at: str = ""
    """ISO-8601 UTC; empty until applied."""


class VocabFile(BaseModel):
    """On-disk schema for ``.colony/colony.vocab.json``."""

    model_config = ConfigDict(extra="forbid")

    version: str = VOCAB_SCHEMA_VERSION
    terms: dict[str, VocabTerm] = Field(default_factory=dict)
    operations: list[VocabOperation] = Field(default_factory=list)
    """Append-only audit trail of APPLIED operations."""

    def to_json(self) -> str:
        import json

        payload = self.model_dump(mode="json")
        payload["terms"] = dict(sorted(payload["terms"].items()))
        return json.dumps(
            payload, sort_keys=False, indent=2, ensure_ascii=False,
        ) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "VocabFile":
        if not text.strip():
            return cls()
        return cls.model_validate_json(text)


# ---------------------------------------------------------------------------
# Registration + resolution
# ---------------------------------------------------------------------------


def register_provisional(
    vocab: VocabFile, predicates: Iterable[str], *, minted_by: str = "extractor",
) -> int:
    """Register unseen predicate names as ``provisional`` terms.
    Returns the number newly registered. Known names (canonical or
    alias) are untouched — registration is idempotent."""

    known = set(vocab.terms)
    for term in vocab.terms.values():
        known.update(term.aliases)
    added = 0
    now = datetime.now(timezone.utc).isoformat()
    for name in predicates:
        if not name or name in known:
            continue
        vocab.terms[name] = VocabTerm(
            name=name,
            provenance={"minted_by": minted_by, "first_seen": now},
        )
        known.add(name)
        added += 1
    return added


def alias_index(vocab: VocabFile) -> dict[str, str]:
    """Surface form → canonical term name (aliases + own names)."""

    index: dict[str, str] = {}
    for term in vocab.terms.values():
        index[term.name] = term.name
        for alias in term.aliases:
            index[alias] = term.name
    return index


def resolve(vocab: VocabFile, name: str) -> str:
    """Canonical predicate for ``name``: alias lookup, then
    ``replaced_by`` chains (bounded; cycles are a loud data error).
    Unknown names resolve to themselves — the vocabulary never blocks
    an open-world predicate."""

    index = alias_index(vocab)
    current = index.get(name, name)
    for _ in range(_MAX_RESOLUTION_HOPS):
        term = vocab.terms.get(current)
        if term is None or not term.replaced_by:
            return current
        current = index.get(term.replaced_by, term.replaced_by)
    raise VocabError(
        f"replaced_by chain from {name!r} exceeded "
        f"{_MAX_RESOLUTION_HOPS} hops — registry contains a cycle.",
    )


def merge_mapping(vocab: VocabFile) -> dict[str, str]:
    """Every surface form whose resolution differs from itself →
    canonical. The input for claim rewriting."""

    mapping: dict[str, str] = {}
    names: set[str] = set()
    for term in vocab.terms.values():
        names.add(term.name)
        names.update(term.aliases)
    for name in names:
        canonical = resolve(vocab, name)
        if canonical != name:
            mapping[name] = canonical
    return mapping


def broader_closure(vocab: VocabFile, name: str) -> set[str]:
    """All ancestors of ``name`` via ``broader`` links (excludes the
    term itself). The seen-set traversal terminates even on a cyclic
    registry; cycle PREVENTION happens at edge creation
    (``apply_operation``'s reachability check refuses the edge)."""

    seen: set[str] = set()
    frontier = [resolve(vocab, name)]
    while frontier:
        nxt: list[str] = []
        for current in frontier:
            term = vocab.terms.get(current)
            if term is None:
                continue
            for parent in term.broader:
                if parent not in seen:
                    seen.add(parent)
                    nxt.append(parent)
        frontier = nxt
    return seen


def narrower_closure(vocab: VocabFile, name: str) -> set[str]:
    """All descendants of ``name`` (terms whose broader-closure
    contains it) — the expansion set for subsumption queries."""

    root = resolve(vocab, name)
    return {
        term.name for term in vocab.terms.values()
        if term.name != root and root in broader_closure(vocab, term.name)
    }


# ---------------------------------------------------------------------------
# Operation application
# ---------------------------------------------------------------------------


def _require_term(vocab: VocabFile, name: str, *, op: VocabOperation) -> VocabTerm:
    term = vocab.terms.get(name)
    if term is None:
        raise VocabError(
            f"{op.op_type.value}: term {name!r} is not in the registry "
            f"(op {op.op_id}).",
        )
    return term


def apply_operation(vocab: VocabFile, op: VocabOperation) -> None:
    """Apply one operation in place and append it to the audit log.
    Raises :class:`VocabError` on any invalid input — a bad proposal
    must surface, never half-apply. Callers gate
    :data:`DESTRUCTIVE_OP_TYPES` behind human approval BEFORE calling."""

    if op.op_type in DESTRUCTIVE_OP_TYPES and not op.approved_by:
        raise VocabError(
            f"{op.op_type.value} is destructive and carries no "
            f"approver (op {op.op_id}) — refuse to apply.",
        )

    if op.op_type in (VocabOpType.MERGE, VocabOpType.RENAME):
        if not op.target:
            raise VocabError(f"{op.op_type.value} requires a target (op {op.op_id}).")
        source = _require_term(vocab, op.term, op=op)
        if op.op_type is VocabOpType.RENAME and op.target not in vocab.terms:
            vocab.terms[op.target] = VocabTerm(
                name=op.target,
                status=VocabTermStatus.ACTIVE,
                description=source.description,
                provenance={"minted_by": f"rename:{op.op_id}"},
            )
        target = _require_term(vocab, op.target, op=op)
        if resolve(vocab, target.name) == source.name:
            raise VocabError(
                f"merge {source.name!r} -> {target.name!r} would create "
                f"a replaced_by cycle (op {op.op_id}).",
            )
        source.status = VocabTermStatus.DEPRECATED
        source.replaced_by = target.name
        if source.name not in target.aliases:
            target.aliases.append(source.name)
        for alias in source.aliases:
            if alias not in target.aliases:
                target.aliases.append(alias)
    elif op.op_type is VocabOpType.ADD_ALIAS:
        if not op.target:
            raise VocabError(f"add_alias requires a target (op {op.op_id}).")
        term = _require_term(vocab, op.term, op=op)
        if op.target not in term.aliases:
            term.aliases.append(op.target)
    elif op.op_type is VocabOpType.ADD_BROADER:
        if not op.target:
            raise VocabError(f"add_broader requires a target (op {op.op_id}).")
        term = _require_term(vocab, op.term, op=op)
        if op.target not in vocab.terms:
            # Purely organizational parents are normal in SKOS — mint
            # them active with provenance pointing at the op.
            vocab.terms[op.target] = VocabTerm(
                name=op.target,
                status=VocabTermStatus.ACTIVE,
                provenance={"minted_by": f"add_broader:{op.op_id}"},
            )
        # Refuse the edge if it would close a hierarchy cycle: the
        # child must not already be an ancestor of the parent.
        if (
            op.target == term.name
            or term.name in broader_closure(vocab, op.target)
        ):
            raise VocabError(
                f"add_broader {term.name!r} -> {op.target!r} would "
                f"create a hierarchy cycle (op {op.op_id}).",
            )
        if op.target not in term.broader:
            term.broader.append(op.target)
    elif op.op_type is VocabOpType.REMOVE_BROADER:
        term = _require_term(vocab, op.term, op=op)
        if op.target in term.broader:
            term.broader.remove(op.target)
    elif op.op_type is VocabOpType.DEPRECATE:
        term = _require_term(vocab, op.term, op=op)
        term.status = VocabTermStatus.DEPRECATED
        term.replaced_by = op.target or None
    elif op.op_type is VocabOpType.PROMOTE:
        term = _require_term(vocab, op.term, op=op)
        if term.status is VocabTermStatus.DEPRECATED:
            raise VocabError(
                f"promote: {term.name!r} is deprecated (op {op.op_id}).",
            )
        term.status = VocabTermStatus.ACTIVE
    elif op.op_type is VocabOpType.SPLIT:
        # Recorded for the audit trail; claim reassignment is a manual
        # guided pass (plan §4).
        _require_term(vocab, op.term, op=op)
    else:  # pragma: no cover — enum is closed
        raise VocabError(f"unknown op type {op.op_type!r}")

    op.applied_at = datetime.now(timezone.utc).isoformat()
    vocab.operations.append(op)


# ---------------------------------------------------------------------------
# Stats — the operator's decision-support payload
# ---------------------------------------------------------------------------


class VocabStats(BaseModel):
    """What the operator needs to decide whether a revision pass is
    worth running (surfaced in the KB tab)."""

    total_terms: int
    active: int
    provisional: int
    deprecated: int
    predicates_in_kg: int
    singleton_predicates: int
    """Predicates carrying exactly one claim — the join-poverty signal."""
    singleton_ratio: float
    unregistered_predicates: int
    """In the KG but absent from the registry (pre-vocabulary claims)."""
    top_predicates: list[tuple[str, int]]
    lexical_merge_candidates: int
    """Cheap upper-bound estimate: normalization clusters with >1
    member. Zero cost to compute; motivates (or spares) a full pass."""
    last_operation_at: str
    operations_applied: int


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_predicate(name: str) -> str:
    """Lexical normal form used for cheap merge-candidate clustering."""
    return _NORMALIZE_RE.sub("_", name.lower()).strip("_")


def vocab_stats(vocab: VocabFile, usage: Counter[str]) -> VocabStats:
    """Compute the decision-support stats from the registry + a
    predicate-usage counter (derived from the KG file by the caller —
    usage is never stored in the registry)."""

    by_status = Counter(t.status for t in vocab.terms.values())
    known = set(alias_index(vocab))
    singleton = sum(1 for _, n in usage.items() if n == 1)
    clusters = Counter(normalize_predicate(p) for p in usage)
    return VocabStats(
        total_terms=len(vocab.terms),
        active=by_status.get(VocabTermStatus.ACTIVE, 0),
        provisional=by_status.get(VocabTermStatus.PROVISIONAL, 0),
        deprecated=by_status.get(VocabTermStatus.DEPRECATED, 0),
        predicates_in_kg=len(usage),
        singleton_predicates=singleton,
        singleton_ratio=(singleton / len(usage)) if usage else 0.0,
        unregistered_predicates=sum(1 for p in usage if p not in known),
        top_predicates=usage.most_common(15),
        lexical_merge_candidates=sum(
            1 for _, n in clusters.items() if n > 1
        ),
        last_operation_at=(
            vocab.operations[-1].applied_at if vocab.operations else ""
        ),
        operations_applied=len(vocab.operations),
    )


def top_active_predicates(
    vocab: VocabFile, usage: Counter[str], *, k: int = 40,
) -> list[VocabTerm]:
    """The extractor's soft prior: the K most-used ACTIVE terms
    (provisional terms haven't earned reuse; deprecated never
    reappear). Plan §6 / operator decision O2."""

    active = [
        t for t in vocab.terms.values()
        if t.status is VocabTermStatus.ACTIVE
    ]
    active.sort(key=lambda t: (-usage.get(t.name, 0), t.name))
    return active[:k]


#: Rendered soft-prior text for the claim extractor, bound around an
#: ingest run (materialize sets it from the repo's registry; the
#: extractor reads it per chunk). ContextVar, not module state — safe
#: across concurrent async ingests (async-state discipline).
_vocabulary_prior: ContextVar[str | None] = ContextVar(
    "colony_vocabulary_prior", default=None,
)


@contextmanager
def set_vocabulary_prior(rendered: str | None):
    """Bind the extractor's soft prior for the enclosed ingest run."""

    token = _vocabulary_prior.set(rendered)
    try:
        yield
    finally:
        _vocabulary_prior.reset(token)


def get_vocabulary_prior() -> str | None:
    return _vocabulary_prior.get()


def render_prior(terms: list[VocabTerm]) -> str | None:
    """Render the soft prior block appended to the extraction prompt:
    prefer these predicates when apt, mint freely when none fits —
    convergence without closure (plan §6)."""

    if not terms:
        return None
    lines = "\n".join(
        f"- {t.name}" + (f": {t.description}" if t.description else "")
        for t in terms
    )
    return (
        "Preferred predicates (reuse when one fits the relation; mint "
        "a precise new snake_case predicate when none does):\n" + lines
    )


__all__ = (
    "DESTRUCTIVE_OP_TYPES",
    "VOCAB_FILE_RELATIVE_PATH",
    "VocabError",
    "VocabFile",
    "VocabOperation",
    "VocabOpType",
    "VocabStats",
    "VocabTerm",
    "VocabTermStatus",
    "alias_index",
    "apply_operation",
    "broader_closure",
    "get_vocabulary_prior",
    "merge_mapping",
    "narrower_closure",
    "normalize_predicate",
    "register_provisional",
    "render_prior",
    "resolve",
    "set_vocabulary_prior",
    "top_active_predicates",
    "vocab_stats",
)
