"""``.colony/repo_map.yaml`` — declarative manifest for a design monorepo.

Three independent sections:

- **``vcm_sources:``** declares VCM mapping. The materialiser issues
  one ``mmap_application_scope`` call per row so a single repo can be
  paged into VCM under multiple scopes (code subtree, literature
  subdirectory, frozen submodule, ...).
- **``knowledge_sources:``** declares KB ingestion. The materialiser
  walks each row's path globs and feeds matching files to the
  process-singleton :class:`Ingestor`.
- **``design_context_sources:``** (schema v3+) declares which markdown
  files form the project's design context. The materialiser registers
  one synthetic VCM scope per row (``source_type='literature'`` —
  prose chunker) and, if ``pin_in_vcm`` is true, holds long-lived
  page locks so the context survives eviction. No closed role
  vocabulary — agents infer structure from the markdown content.

The three sections are **orthogonal**: the same path can appear in
any combination of them. The dashboard renders the operator-facing
sections (``vcm_sources``, ``knowledge_sources``) as independent
checkbox lists; ``design_context_sources`` is configuration-only and
ingested wholesale once the operator declares it.
"""

from __future__ import annotations

import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..knowledge.models import CorpusTier


logger = logging.getLogger(__name__)


SCHEMA_VERSION = 3

REPO_MAP_FILENAME = "repo_map.yaml"
REPO_MAP_DIR = ".colony"

SourceType = Literal["git_repo", "literature"]


class VcmSource(BaseModel):
    """One row of ``vcm_sources:`` — a single VCM mapping declaration.

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

    # Per-source paging knobs — feed into ``MmapConfig`` at materialise
    # time. ``None`` means "use the materialiser's defaults". These are
    # only meaningful for ``git_repo`` sources today; literature rows
    # control chunk size via ``chunk_target_tokens`` instead.
    flush_threshold: int | None = Field(
        default=None,
        description=(
            "Number of records grouped per VCM page before flushing. "
            "Lower = more, smaller pages; higher = fewer, larger pages."
        ),
    )
    flush_token_budget: int | None = Field(
        default=None,
        description=(
            "Maximum tokens per VCM page. When exceeded, a new page "
            "starts. Controls how much code each agent sees at once."
        ),
    )
    pinned: bool | None = Field(
        default=None,
        description="Pin pages produced by this source in cache.",
    )

    def to_mmap_config_overrides(self) -> dict[str, Any]:
        """Return the subset of :class:`MmapConfig` fields this row
        overrides. Empty dict when the row uses defaults — callers
        layer this onto a base ``MmapConfig`` via ``model_copy``.

        Kept on the spec rather than in the materialiser so the rules
        (which knobs are per-source vs deployment-wide) are visible
        next to the field declarations.
        """
        overrides: dict[str, Any] = {}
        if self.flush_threshold is not None:
            overrides["flush_threshold"] = self.flush_threshold
        if self.flush_token_budget is not None:
            overrides["flush_token_budget"] = self.flush_token_budget
        if self.pinned is not None:
            overrides["pinned"] = self.pinned
        return overrides

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
                f"VcmSource[{self.name}]: origin_url and submodule are mutually exclusive.",
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


class AcquirerSpec(BaseModel):
    """Remote-source acquirer reference inside a ``knowledge_sources:``
    row. ``method`` selects an :class:`AcquirerStrategy` from the
    registry; ``args`` is the per-method payload the strategy consumes
    (e.g., ``{"arxiv_id": "2407.12345"}``)."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(
        description=(
            "Registry key — must match a strategy's ``method`` "
            "(e.g., ``arxiv_id``, ``doi``, ``http_url``)."
        ),
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-method arguments forwarded to the acquirer.",
    )


class KnowledgeSource(BaseModel):
    """One row of ``knowledge_sources:`` — a named KB ingestion target.

    Each row is one of two shapes:

    - **Local** (``paths`` set, ``acquirer`` unset): a glob bundle the
      materialiser walks and ingests directly.
    - **Remote** (``acquirer`` + ``destination`` set, ``paths`` unset):
      the materialiser runs the acquirer (which writes a file into
      ``<repo_root>/<destination>/``) and ingests that file. The
      written file is committed alongside the sidecar so re-ingest
      avoids the download.

    Independent from ``vcm_sources:``: presence here means "ingest
    these files into the KB"; absence (or unchecking in the dashboard)
    means "don't." The fact that a path may also appear in a
    ``vcm_sources:`` row is irrelevant — VCM mapping and KB ingestion
    are orthogonal.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "Human-readable label. Surfaces as the checkbox row in the "
            "Design Monorepo tab and as the filter key in "
            "``materialize_knowledge_sources(enabled_sources=...)``."
        ),
    )
    paths: list[str] | None = Field(
        default=None,
        description=(
            "Gitignore-style glob patterns relative to the repo root. "
            "Mutually exclusive with ``acquirer``."
        ),
    )
    acquirer: AcquirerSpec | None = Field(
        default=None,
        description=(
            "Remote-source fetcher. Mutually exclusive with ``paths``. "
            "When set, ``destination`` is required."
        ),
    )
    destination: str | None = Field(
        default=None,
        description=(
            "Repo-root-relative directory the acquirer writes its "
            "fetched file into. Required when ``acquirer`` is set, "
            "forbidden otherwise."
        ),
    )
    profile: str | None = Field(
        default=None,
        description=(
            "Optional ``data_type`` label propagated to the ingested "
            "chunks (e.g., ``scientific_paper``). Forwarded to "
            "``Ingestor.ingest_file(data_type_override=...)``."
        ),
    )
    tier: CorpusTier = Field(
        default=CorpusTier.UNTIERED,
        description=(
            "Corpus tier propagated to chunks. Lets ``UPGRADE_TIER`` "
            "policy decide whether to re-rank existing chunks when "
            "the same source is declared at a higher tier later."
        ),
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> "KnowledgeSource":
        has_paths = self.paths is not None and len(self.paths) > 0
        has_acquirer = self.acquirer is not None
        if has_paths == has_acquirer:
            raise ValueError(
                f"KnowledgeSource[{self.name}]: exactly one of "
                f"'paths' or 'acquirer' must be set.",
            )
        if has_acquirer and not self.destination:
            raise ValueError(
                f"KnowledgeSource[{self.name}]: 'destination' is "
                f"required when 'acquirer' is set.",
            )
        if not has_acquirer and self.destination:
            raise ValueError(
                f"KnowledgeSource[{self.name}]: 'destination' is only "
                f"valid alongside 'acquirer'.",
            )
        if self.destination is not None and Path(self.destination).is_absolute():
            raise ValueError(
                f"KnowledgeSource[{self.name}]: 'destination' must be "
                f"a repo-root-relative path, got {self.destination!r}.",
            )
        return self

    def matches(self, rel_path: str) -> bool:
        """Return ``True`` when ``rel_path`` matches any of this row's
        ``paths`` patterns. Caller passes a forward-slash relative
        path; we delegate to :class:`pathspec.PathSpec` so pattern
        semantics match every other glob in the framework.

        Returns ``False`` for acquirer-shaped rows (no globs to match)."""

        if not self.paths:
            return False

        from pathspec import PathSpec
        from pathspec.patterns import GitWildMatchPattern

        spec = PathSpec.from_lines(GitWildMatchPattern, self.paths)
        return spec.match_file(rel_path)


class DesignContextSource(BaseModel):
    """One row of ``design_context_sources:`` — declares a corpus of
    markdown files as the project's design context (objectives,
    constraints, alternatives, hypotheses, decisions, etc., in
    arbitrary mixes per file).

    The materialiser turns each row into a synthetic ``vcm_source``
    of ``type: literature`` (prose chunker), so agents read pages
    via the standard VCM page-load path. If ``pin_in_vcm`` is true,
    the materialiser additionally calls
    :meth:`~vcm.page_table.VirtualPageTable.lock_page` per
    materialised page and a renewer task refreshes the locks at
    ``6/7 * pin_lock_duration_days`` to outrun expiry.

    Free file layout — no closed role vocabulary. The
    (future-phase) Kuzu KG extractor infers structure from the
    markdown itself; this row is purely the "what bytes count as
    design context" declaration.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "Identifier; namespaces the materialiser output as "
            "``design_context.<name>`` (must be unique within "
            "``design_context_sources``)."
        ),
    )
    paths: list[str] = Field(
        description=(
            "Glob patterns relative to monorepo root (pathspec / "
            "GitWildMatch semantics — same as ``knowledge_sources.paths``)."
        ),
    )
    hint: str | None = Field(
        default=None,
        description=(
            "Free-form prose passed to the LLM claim-extractor as a "
            "'what is this corpus' cue. Not validated, not a gate — "
            "only an extractor hint. Omit if unsure."
        ),
    )
    pin_in_vcm: bool = Field(
        default=False,
        description=(
            "If true, materialised pages are locked via "
            "``VirtualPageTable.lock_page`` so they survive eviction. "
            "Use sparingly — pinned pages reduce the working-set "
            "budget available for transient context."
        ),
    )
    pin_lock_duration_days: int = Field(
        default=7,
        ge=1,
        description=(
            "Lock window the renewer uses. The renewer fires at "
            "``6/7 * pin_lock_duration_days`` to avoid expiry races. "
            "Only consulted when ``pin_in_vcm`` is true."
        ),
    )

    @model_validator(mode="after")
    def _check_paths(self) -> "DesignContextSource":
        if not self.paths:
            raise ValueError(
                f"DesignContextSource[{self.name}]: 'paths' must be "
                f"non-empty.",
            )
        return self

    def matches(self, rel_path: str) -> bool:
        """Return ``True`` when ``rel_path`` matches any of this row's
        ``paths`` patterns. Same delegation to ``pathspec`` as
        :meth:`KnowledgeSource.matches` so semantics stay aligned."""

        from pathspec import PathSpec
        from pathspec.patterns import GitWildMatchPattern

        spec = PathSpec.from_lines(GitWildMatchPattern, self.paths)
        return spec.match_file(rel_path)


class RepoMap(BaseModel):
    """Top-level schema for ``.colony/repo_map.yaml``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    vcm_sources: list[VcmSource]
    knowledge_sources: list[KnowledgeSource] = Field(default_factory=list)
    design_context_sources: list[DesignContextSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_design_context_sources_unique(self) -> "RepoMap":
        seen: set[str] = set()
        for src in self.design_context_sources:
            if src.name in seen:
                raise ValueError(
                    f"RepoMap: duplicate design_context_sources name "
                    f"{src.name!r}; each row must have a unique name.",
                )
            seen.add(src.name)
        return self

    @classmethod
    def default_for_unmapped_repo(cls) -> "RepoMap":
        """Single ``git_repo`` row over the whole tree — the
        backwards-compatible fallback when no ``repo_map.yaml`` is
        present in the design monorepo. ``knowledge_sources`` and
        ``design_context_sources`` default to empty so existing repos
        do not silently start ingesting."""

        return cls(vcm_sources=[VcmSource(name="default", type="git_repo")])

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
    """The two pieces a submodule ``VcmSource`` row needs."""

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
    "AcquirerSpec",
    "KnowledgeSource",
    "REPO_MAP_DIR",
    "REPO_MAP_FILENAME",
    "RepoMap",
    "SCHEMA_VERSION",
    "VcmSource",
)
