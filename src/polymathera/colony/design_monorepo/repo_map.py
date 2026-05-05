"""``.colony/repo_map.yaml`` — declarative VCM-mapping manifest for a
design monorepo.

The repo map lives at the repo root under ``.colony/repo_map.yaml`` and
is version-controlled with the rest of the codebase. It declares **one
``ContextPageSource`` per source entry**; the materialiser issues one
``mmap_application_scope`` call per entry so a single repo can be
paged into VCM under multiple scopes (code subtree, literature
subdirectory, frozen submodule, ...).

This module is *VCM-mapping-only*. Knowledge-base ingestion is
agent-driven (chat → ``BulkAcquisitionCapability`` /
``KnowledgeCuratorCapability``); no auto-routing here. See
``colony/design_monorepo_integration_plan.md`` §5–§7.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1

REPO_MAP_FILENAME = "repo_map.yaml"
REPO_MAP_DIR = ".colony"

SourceType = Literal["git_repo", "literature"]
IngestDestination = Literal["knowledge_base", "vcm"]


class SourceSpec(BaseModel):
    """One row of ``sources:`` — a single VCM mapping declaration.

    Either ``submodule`` or ``origin_url`` must be set (not both). When
    ``submodule`` is given, :meth:`to_mmap_kwargs` resolves it against
    the design monorepo's ``.gitmodules`` and the gitlink commit
    pinned in the index, producing a frozen-commit mapping.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Human-readable scope label (also the suggested scope_id suffix).")
    type: SourceType = "git_repo"

    # Source location — exactly one of these:
    origin_url: str | None = None
    submodule: str | None = Field(
        default=None,
        description="Path to a submodule under .gitmodules. Resolved at materialise time.",
    )

    # Common page-source kwargs — passed through to the registered class.
    branch: str = "main"
    commit: str = "HEAD"
    start_dir: str | None = None
    include_globs: list[str] | None = None
    exclude_globs: list[str] | None = None
    binary_policy: Literal["skip", "include"] | None = None
    static: bool | None = None

    # Literature-specific.
    chunk_target_tokens: int | None = None
    chunk_overlap_tokens: int | None = None

    def to_mmap_kwargs(
        self,
        *,
        repo_root: Path,
        scope_id: str,
        fallback_origin_url: str,
        fallback_branch: str,
        fallback_commit: str,
    ) -> dict[str, Any]:
        """Translate this row into kwargs for
        :meth:`VirtualContextManager.mmap_application_scope`.

        ``fallback_*`` apply only when the row supplies neither
        ``origin_url`` nor ``submodule`` — e.g., the
        :func:`default_for_unmapped_repo` row that maps the whole
        outer repo. ``submodule`` rows always pin to the submodule's
        gitlink commit (frozen).
        """

        if self.origin_url and self.submodule:
            raise ValueError(
                f"SourceSpec[{self.name}]: origin_url and submodule are mutually exclusive.",
            )

        if self.submodule:
            sub = _resolve_submodule(repo_root, self.submodule)
            origin_url = sub.url
            branch = self.branch
            commit = sub.pinned_commit
        elif self.origin_url:
            origin_url = self.origin_url
            branch = self.branch
            commit = self.commit
        else:
            origin_url = fallback_origin_url
            branch = self.branch if self.branch != "main" else fallback_branch
            commit = self.commit if self.commit != "HEAD" else fallback_commit

        kwargs: dict[str, Any] = {
            "scope_id": scope_id,
            "source_type": self.type,
            "origin_url": origin_url,
            "branch": branch,
            "commit": commit,
        }
        if self.start_dir is not None:
            kwargs["start_dir"] = self.start_dir
        if self.include_globs is not None:
            kwargs["include_globs"] = list(self.include_globs)
        if self.exclude_globs is not None:
            kwargs["exclude_globs"] = list(self.exclude_globs)
        if self.binary_policy is not None and self.type == "git_repo":
            # LiteratureContextPageSource forces "include" — pass-through
            # only makes sense for git_repo.
            kwargs["binary_policy"] = self.binary_policy
        if self.static is not None:
            kwargs["static"] = self.static
        if self.type == "literature":
            if self.chunk_target_tokens is not None:
                kwargs["chunk_target_tokens"] = self.chunk_target_tokens
            if self.chunk_overlap_tokens is not None:
                kwargs["chunk_overlap_tokens"] = self.chunk_overlap_tokens
        return kwargs


class KnowledgeRoute(BaseModel):
    """One row of ``knowledge_routing:`` — declarative routing
    decision for a glob of literature paths.

    Each row carries an explicit ``ingest_to`` field naming the
    destination store. ``ingest_to: knowledge_base`` (the default for
    new literature) is the only value the materialiser acts on — it
    feeds matching files into :class:`Ingestor`.
    ``ingest_to: vcm`` rows are documentation-only: they declare that
    a path has been promoted to VCM and the materialiser skips KB
    ingestion for it. The actual VCM mapping for those paths still
    comes from a ``LiteratureContextPageSource`` (or other) row under
    ``sources:`` — the two lists are not redundant; they answer
    different questions.

    The dashboard's "Design Monorepo" tab promotes a file by
    flipping ``ingest_to`` on a row, so the action stays a single
    field edit instead of moving entries between two lists.
    """

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        description=(
            "Gitignore-style glob patterns relative to the repo root."
        ),
    )
    ingest_to: IngestDestination = Field(
        default="knowledge_base",
        description=(
            "Where matching files should land. ``knowledge_base`` "
            "(default) ingests via the process-singleton ``Ingestor``. "
            "``vcm`` is a no-op for the KB materialiser — it records "
            "that the path is intentionally excluded from KB "
            "ingestion because a ``sources:`` row covers it."
        ),
    )
    profile: str | None = Field(
        default=None,
        description=(
            "Optional ``data_type`` label propagated to the ingested "
            "chunks (e.g., ``scientific_paper``). Forwarded to "
            "``Ingestor.ingest_file(data_type_override=...)``. Ignored "
            "when ``ingest_to`` is ``vcm``."
        ),
    )

    def matches(self, rel_path: str) -> bool:
        """Return ``True`` when ``rel_path`` matches any of this row's
        ``paths`` patterns. Caller passes a forward-slash relative
        path; we delegate to :class:`pathspec.PathSpec` so pattern
        semantics match every other glob in the framework."""

        from pathspec import PathSpec
        from pathspec.patterns import GitWildMatchPattern

        spec = PathSpec.from_lines(GitWildMatchPattern, self.paths)
        return spec.match_file(rel_path)


class RepoMap(BaseModel):
    """Top-level schema for ``.colony/repo_map.yaml``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    sources: list[SourceSpec]
    knowledge_routing: list[KnowledgeRoute] = Field(default_factory=list)

    @classmethod
    def default_for_unmapped_repo(cls) -> "RepoMap":
        """Single ``git_repo`` source over the whole tree — the
        backwards-compatible fallback when no ``repo_map.yaml`` is
        present in the design monorepo. ``knowledge_routing`` defaults
        to empty so existing repos do not silently start ingesting."""

        return cls(sources=[SourceSpec(name="default", type="git_repo")])

    @classmethod
    def load(cls, repo_root: Path) -> "RepoMap":
        """Load ``<repo_root>/.colony/repo_map.yaml`` or return the
        default. Synchronous — IO is a single small file read."""

        path = Path(repo_root) / REPO_MAP_DIR / REPO_MAP_FILENAME
        if not path.is_file():
            return cls.default_for_unmapped_repo()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"{path}: invalid YAML — {exc}",
            ) from exc
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported schema_version "
                f"{data.get('schema_version')!r}; expected {SCHEMA_VERSION}.",
            )
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Submodule resolution
# ---------------------------------------------------------------------------


class _ResolvedSubmodule(BaseModel):
    """The two pieces a submodule ``SourceSpec`` row needs."""

    model_config = ConfigDict(frozen=True)
    url: str
    pinned_commit: str


def _resolve_submodule(repo_root: Path, submodule_path: str) -> _ResolvedSubmodule:
    """Resolve a submodule path against ``<repo_root>/.gitmodules`` +
    the parent repo's pinned gitlink. Raises if the submodule is not
    declared or its commit cannot be read."""

    gitmodules = Path(repo_root) / ".gitmodules"
    if not gitmodules.is_file():
        raise ValueError(
            f"submodule {submodule_path!r} requested but {gitmodules} is missing",
        )
    parser = ConfigParser()
    parser.read(gitmodules, encoding="utf-8")
    section = next(
        (
            s for s in parser.sections()
            if s.startswith("submodule ")
            and parser.get(s, "path", fallback=None) == submodule_path
        ),
        None,
    )
    if section is None:
        raise ValueError(
            f"submodule {submodule_path!r} not found in {gitmodules}",
        )
    url = parser.get(section, "url", fallback=None)
    if not url:
        raise ValueError(
            f"submodule {submodule_path!r} has no 'url' in {gitmodules}",
        )

    import git
    try:
        repo = git.Repo(str(repo_root))
        # ``ls-tree -z HEAD <path>`` outputs ``<mode> commit <sha>\t<path>``
        # for a gitlink entry. Parse the sha out of the second field.
        out = repo.git.ls_tree("HEAD", submodule_path).strip()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"submodule {submodule_path!r}: failed to read pinned commit "
            f"({exc})",
        ) from exc
    if not out:
        raise ValueError(
            f"submodule {submodule_path!r}: no gitlink in HEAD",
        )
    parts = out.split()
    if len(parts) < 3 or parts[1] != "commit":
        raise ValueError(
            f"submodule {submodule_path!r}: unexpected ls-tree output {out!r}",
        )
    return _ResolvedSubmodule(url=url, pinned_commit=parts[2])


__all__ = (
    "IngestDestination",
    "KnowledgeRoute",
    "REPO_MAP_DIR",
    "REPO_MAP_FILENAME",
    "RepoMap",
    "SCHEMA_VERSION",
    "SourceSpec",
)
