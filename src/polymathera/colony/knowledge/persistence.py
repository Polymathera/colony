"""Knowledge-graph persistence: git-shared single-file snapshots.

Bridges the live :class:`GraphStore` (shared Kùzu instance, in-process
read cache) and a versioned JSON snapshot in the design monorepo at
``.colony/colony.kg.json``. Two operations:

- **snapshot**: exports the GraphStore's claims for the branch being
  committed and atomically writes the file. Registered as a
  pre-commit callback (:mod:`..design_monorepo.commit_hooks`) so it
  fires automatically as part of any commit-and-push the design-
  monorepo capabilities issue. Never overwrites the file with an
  empty payload — a deployment that has never rehydrated must not
  destroy the canonical record on its first checkpoint.

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


def _sorted_claims(claims: Iterable[Claim]) -> list[PersistedClaim]:
    """Stable claim ordering so byte-identical KGs produce
    byte-identical files (empty git diffs on unchanged commits)."""

    persisted = [PersistedClaim.from_claim(c) for c in claims]
    persisted.sort(
        key=lambda c: (
            c.subject, c.predicate, c.object_,
            str(c.citation.get("source_uri", "")),
        ),
    )
    return persisted


async def snapshot_branch_to_file(
    working_dir: Path, branch: str,
) -> tuple[Path, int]:
    """Export every claim in the process-singleton GraphStore tagged
    with ``branch`` and write them to
    ``<working_dir>/.colony/colony.kg.json`` atomically. Returns
    ``(path, claim_count)``.

    When the local GraphStore holds zero claims for ``branch``, the
    file is NOT touched — a deployment whose Kùzu volume was just
    initialised and has never been rehydrated must not clobber the
    canonical record. Callers that want to materialise an empty
    branch must write the file directly.
    """

    deps = get_knowledge_deps()
    claims: list[Claim] = []
    async for claim in deps.graph_store.export_claims(branch=branch):
        claims.append(claim)
    path = working_dir / KG_FILE_RELATIVE_PATH
    if not claims:
        return path, 0
    payload = KgFile(claims=_sorted_claims(claims))
    atomic_write_text(path, payload.to_json())
    return path, len(claims)


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
    "snapshot_branch_to_file",
)
