"""Knowledge-graph persistence: git-shared single-file snapshots.

Bridges the live :class:`GraphStore` (shared Kùzu instance, in-process
read cache) and a versioned JSON snapshot in the design monorepo at
``.colony/colony.kg.json``. Two operations:

- **snapshot**: exports the GraphStore's claims for the branch being
  committed and atomically MERGES them into the file (union on claim
  identity; fresh export wins a collision). Registered as a
  pre-commit callback (:mod:`..design_monorepo.commit_hooks`) so it
  fires automatically as part of any commit-and-push the design-
  monorepo capabilities issue. Merge, never replace: the live store
  is a working set, and a fresh deployment that never rehydrated
  must not clobber claims other runs paid for (nor may an empty
  export touch the file at all).

- **rehydrate**: loads the file from a branch's checked-out tree
  (or from ``origin/<branch>`` via ``git show``) and idempotently
  imports the claims into the shared GraphStore, tagging every
  touched node/edge with the source branch.

The file schema mirrors the :class:`PersistedClaim` shape one-to-one;
a top-level ``namespaces`` map and ``version`` field round-trip
through the kg-merge driver verbatim.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from git import Repo

from ..utils.files import atomic_write_text
from .deps import get_knowledge_deps
from .models import Claim, CitationSpan


logger = logging.getLogger(__name__)


#: Schema version stored in every persisted file. The merge driver
#: refuses to merge files with mismatched versions, so a stale clone
#: that hasn't been updated to handle a newer schema gets a loud
#: conflict instead of silently producing garbage.
SCHEMA_VERSION = "1.0"

#: Path of the snapshot file relative to the design monorepo root.
KG_FILE_RELATIVE_PATH = Path(".colony") / "colony.kg.json"

#: Name under which the snapshot callback registers itself in the
#: pre-commit registry. Stable so external callers can
#: :func:`~polymathera.colony.design_monorepo.commit_hooks.PreCommitRegistry.unregister`
#: it (e.g. tests, or a deployment that wants to take over snapshot
#: timing explicitly).
SNAPSHOT_CALLBACK_NAME = "knowledge.kg_snapshot"


class PersistedClaim(BaseModel):
    """Wire-shape claim — round-trips :class:`Claim` losslessly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    subject: str
    predicate: str
    object_: str = Field(alias="object")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    citation: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_claim(cls, claim: Claim) -> "PersistedClaim":
        return cls(
            subject=claim.subject,
            predicate=claim.predicate,
            object=claim.object_,
            confidence=claim.confidence,
            citation={
                "source_uri": claim.citation.source_uri,
                "section_path": claim.citation.section_path,
                "char_start": claim.citation.char_start,
                "char_end": claim.citation.char_end,
            },
            provenance=dict(claim.provenance),
        )

    def to_claim(self) -> Claim:
        cit = self.citation or {}
        return Claim(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object_,
            confidence=self.confidence,
            citation=CitationSpan(
                source_uri=str(cit.get("source_uri", "")),
                section_path=str(cit.get("section_path", "")),
                char_start=int(cit.get("char_start", 0) or 0),
                char_end=int(cit.get("char_end", 0) or 0),
            ),
            provenance=dict(self.provenance),
        )


class KgFile(BaseModel):
    """On-disk schema for ``.colony/colony.kg.json``."""

    model_config = ConfigDict(extra="forbid")

    version: str = SCHEMA_VERSION
    namespaces: dict[str, str] = Field(default_factory=dict)
    claims: list[PersistedClaim] = Field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "KgFile":
        if not text.strip():
            return cls()
        return cls.model_validate_json(text)


def _sorted_claims(claims: Iterable[PersistedClaim]) -> list[PersistedClaim]:
    """Stable claim ordering so byte-identical KGs produce
    byte-identical files (empty git diffs on unchanged commits)."""

    return sorted(claims, key=_claim_key)


def _claim_key(claim: PersistedClaim) -> tuple[str, str, str, str]:
    """Merge identity of a persisted claim — the same key
    ``_sorted_claims`` orders by and the kg-merge driver deduplicates
    on."""

    return (
        claim.subject,
        claim.predicate,
        claim.object_,
        str(claim.citation.get("source_uri", "")),
    )


async def snapshot_branch_to_file(
    working_dir: Path, branch: str,
) -> tuple[Path, int]:
    """Export every claim in the process-singleton GraphStore tagged
    with ``branch`` and MERGE them into
    ``<working_dir>/.colony/colony.kg.json`` atomically (union with
    the file's existing claims; on identity collision the freshly
    exported claim wins). Returns ``(path, claim_count_written)``.

    Merge, never replace: the process's live GraphStore is a WORKING
    SET, not the canonical record — a fresh deployment that ingested
    a subset and never rehydrated must not clobber claims other runs
    paid for. (2026-08-04: a replace-write dropped 20,387 claims of a
    prior run's extraction; recovered from git history.) Consequence:
    claim deletions do not propagate through snapshots — when a
    deletion story exists it needs an explicit tombstone mechanism,
    not a smaller snapshot.

    When the local GraphStore holds zero claims for ``branch``, the
    file is NOT touched — same no-clobber principle for the trivial
    case.
    """

    deps = get_knowledge_deps()
    claims: list[Claim] = []
    async for claim in deps.graph_store.export_claims(branch=branch):
        claims.append(claim)
    path = working_dir / KG_FILE_RELATIVE_PATH
    if not claims:
        return path, 0

    merged: dict[tuple[str, str, str, str], PersistedClaim] = {}
    if path.is_file():
        existing = KgFile.from_json(path.read_text(encoding="utf-8"))
        for pc in existing.claims:
            merged[_claim_key(pc)] = pc
    prior_count = len(merged)
    for claim in claims:
        pc = PersistedClaim.from_claim(claim)
        merged[_claim_key(pc)] = pc

    payload = KgFile(claims=_sorted_claims(merged.values()))
    atomic_write_text(path, payload.to_json())
    logger.info(
        "snapshot_branch_to_file: merged %d exported claims into %d "
        "existing → %d total (branch=%s)",
        len(claims), prior_count, len(merged), branch,
    )

    # Vocabulary registration rides the same snapshot: every predicate
    # in the canonical KG is registered (provisional on first sight) so
    # revision passes always see the full open vocabulary. Idempotent.
    _register_snapshot_predicates(working_dir, payload)

    return path, len(merged)


def _register_snapshot_predicates(
    working_dir: Path, kg: KgFile,
) -> None:
    from .vocabulary import (
        VOCAB_FILE_RELATIVE_PATH,
        VocabFile,
        register_provisional,
    )

    vocab_path = working_dir / VOCAB_FILE_RELATIVE_PATH
    vocab = (
        VocabFile.from_json(vocab_path.read_text(encoding="utf-8"))
        if vocab_path.is_file() else VocabFile()
    )
    added = register_provisional(
        vocab, (c.predicate for c in kg.claims),
    )
    if added:
        atomic_write_text(vocab_path, vocab.to_json())
        logger.info(
            "snapshot: registered %d new provisional predicates "
            "(vocabulary now %d terms).",
            added, len(vocab.terms),
        )


async def rewrite_claims_for_merges(working_dir: Path) -> dict[str, int]:
    """Apply the vocabulary's merge/rename resolutions to the canonical
    KG file: every claim whose predicate resolves to a different
    canonical is rewritten (original surface form preserved in
    ``provenance['predicate_as_extracted']``), and post-rewrite
    identity collisions deduplicate through the same union machinery
    the snapshot uses. Returns counts for the caller's report.

    File-level only by design: live stores refresh through the normal
    rehydrate path; the canonical record is the source of truth.
    """

    from .vocabulary import (
        VOCAB_FILE_RELATIVE_PATH,
        VocabFile,
        merge_mapping,
    )

    vocab_path = working_dir / VOCAB_FILE_RELATIVE_PATH
    kg_path = working_dir / KG_FILE_RELATIVE_PATH
    if not vocab_path.is_file() or not kg_path.is_file():
        return {"rewritten": 0, "deduplicated": 0, "total": 0}
    vocab = VocabFile.from_json(vocab_path.read_text(encoding="utf-8"))
    mapping = merge_mapping(vocab)
    kg = KgFile.from_json(kg_path.read_text(encoding="utf-8"))
    if not mapping:
        return {"rewritten": 0, "deduplicated": 0, "total": len(kg.claims)}

    merged: dict[tuple[str, str, str, str], PersistedClaim] = {}
    rewritten = 0
    for pc in kg.claims:
        canonical = mapping.get(pc.predicate)
        if canonical is not None:
            provenance = dict(pc.provenance)
            provenance.setdefault("predicate_as_extracted", pc.predicate)
            pc = pc.model_copy(
                update={"predicate": canonical, "provenance": provenance},
            )
            rewritten += 1
        merged[_claim_key(pc)] = pc

    payload = KgFile(claims=_sorted_claims(merged.values()))
    atomic_write_text(kg_path, payload.to_json())
    result = {
        "rewritten": rewritten,
        "deduplicated": len(kg.claims) - len(merged),
        "total": len(merged),
    }
    logger.info("rewrite_claims_for_merges: %s", result)
    return result


async def load_branch_from_text(text: str, branch: str) -> dict[str, int]:
    """Parse ``text`` as a :class:`KgFile` and import its claims into
    the process-singleton GraphStore tagged with ``branch``. Returns
    the per-call import counts (added / tagged / skipped / total
    parsed)."""

    file = KgFile.from_json(text)
    claims = [pc.to_claim() for pc in file.claims]
    deps = get_knowledge_deps()
    result = await deps.graph_store.import_claims(claims, branch=branch)
    return {
        "claims_in_file": len(claims),
        "claims_newly_added": result.added,
        "claims_newly_tagged": result.tagged,
        "claims_already_present": result.skipped,
    }


def normalize_branch_name(branch: str) -> str:
    """Strip well-known remote prefixes so branch annotations are
    comparable regardless of which remote the snapshot was read
    from. ``origin/main`` → ``main``; ``refs/heads/x`` → ``x``;
    everything else is returned unchanged."""

    for prefix in ("refs/remotes/origin/", "refs/heads/", "origin/"):
        if branch.startswith(prefix):
            return branch[len(prefix):]
    return branch


async def rehydrate_branch_from_repo(
    repo: Repo, branch: str,
) -> dict[str, Any]:
    """Read ``.colony/colony.kg.json`` from ``origin/<branch>`` via
    ``git show`` (no working-tree mutation) and import its claims into
    the process-wide GraphStore tagged with the normalised branch
    name. Returns the per-call counts plus the resolved source
    commit SHA so the caller can surface it.

    A missing file at the path is NOT an error — fresh branches just
    return zero claims. Any other ``git show`` failure (bad branch,
    corrupt object) propagates as :class:`GitCommandError`.
    """

    import asyncio

    from git import GitCommandError

    tag = normalize_branch_name(branch)
    rel = str(KG_FILE_RELATIVE_PATH).replace("\\", "/")

    def _read_text() -> tuple[str, str]:
        ref = f"origin/{tag}"
        try:
            text = repo.git.show(f"{ref}:{rel}")
        except GitCommandError as exc:
            stderr = (exc.stderr or "").lower()
            if "does not exist" in stderr or "exists on disk" in stderr:
                return "", ""
            raise
        sha = repo.git.rev_parse(ref).strip()
        return text, sha

    text, source_sha = await asyncio.to_thread(_read_text)
    if not text.strip():
        return {
            "branch": tag,
            "source_commit_sha": source_sha,
            "claims_in_file": 0,
            "claims_newly_added": 0,
            "claims_newly_tagged": 0,
            "claims_already_present": 0,
        }
    counts = await load_branch_from_text(text, tag)
    return {"branch": tag, "source_commit_sha": source_sha, **counts}


async def list_remote_branches(repo: Repo) -> list[str]:
    """List branches present on ``origin`` (after a fetch). Skips the
    ``HEAD`` ref and de-duplicates so each branch appears once with
    its normalised name."""

    import asyncio

    def _scan() -> list[str]:
        try:
            repo.git.fetch("origin", "--prune")
        except Exception:  # noqa: BLE001 — best-effort refresh; offline ok
            pass
        out: list[str] = []
        seen: set[str] = set()
        for ref in repo.remotes.origin.refs:
            name = normalize_branch_name(ref.name)
            if name in ("HEAD", "") or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return sorted(out)

    return await asyncio.to_thread(_scan)


def register_kg_snapshot_callback() -> None:
    """Idempotently register the KG snapshot pre-commit callback.
    Called from :func:`..deps.set_knowledge_deps` so every Ray
    process picks up the hook as part of its standard knowledge-deps
    bootstrap; safe to call multiple times."""

    from ..design_monorepo.commit_hooks import (
        PreCommitContext,
        get_pre_commit_registry,
    )

    registry = get_pre_commit_registry()
    if SNAPSHOT_CALLBACK_NAME in registry.names():
        return

    async def _callback(ctx: PreCommitContext) -> None:
        if not ctx.branch:
            return
        await snapshot_branch_to_file(ctx.working_dir, ctx.branch)

    registry.register(SNAPSHOT_CALLBACK_NAME, _callback)


__all__ = (
    "KG_FILE_RELATIVE_PATH",
    "KgFile",
    "PersistedClaim",
    "SCHEMA_VERSION",
    "SNAPSHOT_CALLBACK_NAME",
    "list_remote_branches",
    "load_branch_from_text",
    "normalize_branch_name",
    "register_kg_snapshot_callback",
    "rehydrate_branch_from_repo",
    "rewrite_claims_for_merges",
    "snapshot_branch_to_file",
)
