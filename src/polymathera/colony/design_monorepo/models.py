"""Typed models for the design monorepo.

These are the shapes that flow between the framework wrapper layer
(``DesignMonorepoClient``), the agent-facing capabilities
(``RepoStateProvider`` / ``DesignCheckpointer`` / ``ToolBuilder``), and
the action policies that consume them.

The models live here, not next to each consumer, so that the wrapper
layer and the capability layer share a single source of truth and so
that downstream multi-agent systems built on top of colony can
introspect the schema without importing GitPython.

Naming follows the design-automation architecture doc
(`colony_docs/markdown/apps/design_automation_architecture.md` §3.5.1
and §8) and the standalone analysis in
`colony_docs/markdown/apps/git_as_design_state_engine.md`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..tools.base import HeadlessReadiness


# ---------------------------------------------------------------------------
# Repo state — the snapshot returned by RepoStateProvider.get_repo_state()
# ---------------------------------------------------------------------------


class Checkpoint(BaseModel):
    """A tagged checkpoint in the design monorepo.

    Backed by an annotated git tag of the form
    ``checkpoint/<iso8601>-<short-sha>``. The label and rationale come
    from the tag's annotation; the SHA from the tag's target commit.
    """

    model_config = ConfigDict(frozen=True)

    checkpoint_id: str = Field(
        description=(
            "Tag name without the 'refs/tags/' prefix; canonical form "
            "'checkpoint/<iso8601>-<short_sha>'."
        ),
    )
    sha: str = Field(description="Full 40-char commit SHA.")
    label: str = Field(description="Human-readable label set at tag time.")
    rationale: str = Field(
        default="",
        description="Free-form rationale recorded in the tag annotation.",
    )
    author: str = Field(
        description=(
            "Tagger identity. For agent-produced checkpoints this is the "
            "agent's transactional commit identity."
        ),
    )
    created_at: datetime = Field(
        description="Tagger date in UTC."
    )


class ForkBranch(BaseModel):
    """A live ``fork/<label>`` branch — a design-space-exploration branch.

    Listed by ``DesignMonorepoClient.list_forks()`` and exposed to
    agents via ``RepoStateProvider.get_repo_state().forks``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Full branch name including 'fork/' prefix.")
    head_sha: str = Field(description="Current tip commit SHA.")
    diverged_from_sha: str = Field(
        description="Merge base with the parent branch (typically 'main').",
    )
    diverged_from_branch: str = Field(
        description="Branch this fork was created off (typically 'main').",
    )
    created_at: datetime = Field(
        description="Time the diverged_from_sha commit was authored.",
    )


class ToolEntry(BaseModel):
    """A tool registered in ``.colony/tool-registry.json``.

    Tools live at ``tools/<purpose>/<name>/`` in the design monorepo
    (master §9.3) or in an imported tooling-monorepo remote (master
    §9.5). The ``location`` discriminates.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    purpose: str = Field(
        description=(
            "Free-form purpose tag — e.g. 'racer/laptime', 'duv/hopkins', "
            "'shared/cad'. Determines the tools/<purpose>/ subdirectory."
        ),
    )
    capability: str = Field(
        description="Capability key the tool fulfils (queryable in find_existing_tool).",
    )
    version: str = Field(default="0.1.0")
    location: str = Field(
        description=(
            "Either 'subdir:tools/<purpose>/<name>' for a tool in this "
            "monorepo, or 'remote:<name>:tools/<purpose>/<name>' for a "
            "tool exposed through an imports_tooling_from remote."
        ),
    )
    license: str = Field(default="", description="SPDX licence identifier.")
    container_image: str | None = None
    headless: HeadlessReadiness = Field(
        default=HeadlessReadiness.NATIVE,
        description=(
            "Headless-readiness tier (see ``colony.tools.HeadlessReadiness``)."
        ),
    )
    extra: dict[str, object] = Field(default_factory=dict)


class ToolMatch(BaseModel):
    """One result from ``find_existing_tool``."""

    model_config = ConfigDict(frozen=True)

    entry: ToolEntry
    score: float = Field(
        description=(
            "Relevance score on [0, 1] from the capability_query match. "
            "Determined by the registry's scoring function."
        ),
    )
    writable: bool = Field(
        description=(
            "True iff the calling context can mutate this tool's "
            "directory in the local working tree. Tools imported via "
            "imports_tooling_from are read-only by default."
        ),
    )


class ImportedRemote(BaseModel):
    """An ``imports_tooling_from`` remote registered in the manifest.

    Added as a regular git remote (read-only by default) and surfaced in
    ``find_existing_tool`` searches.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Local git remote name (auto-generated from url).")
    url: str
    ref: str = Field(default="main", description="Refspec — branch, tag, or 'tag:<name>'.")
    scope: str = Field(
        default="tools/",
        description="Path prefix exposed to find_existing_tool.",
    )
    writable: bool = False


class DecisionEntry(BaseModel):
    """One design decision written to ``design/decisions/<id>.json``.

    Surfaced by ``RepoStateProvider.list_recent_decisions`` so an action
    policy can avoid proposing alternatives that have already been
    decided.
    """

    model_config = ConfigDict(frozen=True)

    decision_id: str
    sha: str = Field(description="Commit that authored this decision.")
    title: str
    summary: str = Field(default="")
    authored_at: datetime
    author: str
    rationale: str = Field(default="")
    relative_path: str = Field(
        description="Path of the JSON file under design/decisions/.",
    )


class DesignDiffEntry(BaseModel):
    """One per-file entry in a ``DesignDiff``."""

    model_config = ConfigDict(frozen=True)

    path: str
    change_type: Literal["added", "modified", "deleted", "renamed", "copied"]
    old_path: str | None = None
    insertions: int = 0
    deletions: int = 0


class DesignDiff(BaseModel):
    """Difference between two refs of the design monorepo."""

    model_config = ConfigDict(frozen=True)

    ref_a: str
    ref_b: str
    entries: tuple[DesignDiffEntry, ...] = Field(default_factory=tuple)
    raw_unified_diff: str = Field(
        default="",
        description=(
            "The raw `git diff a..b` text. Capped by the wrapper layer "
            "(default 256 KiB). Empty when the diff is too large; the "
            "entries summary still applies."
        ),
    )


class BranchNode(BaseModel):
    """One branch in a ``BranchTopology``."""

    model_config = ConfigDict(frozen=True)

    name: str
    head_sha: str
    parent_branch: str | None = Field(
        default=None,
        description=(
            "Best-effort parent branch (e.g. 'main' for fork/*). Computed "
            "from the merge-base graph; may be None if the branch was "
            "created with no clear parent."
        ),
    )
    diverged_from_sha: str | None = None
    is_fork: bool = False
    is_session: bool = False
    is_agent: bool = False
    is_tool: bool = False


class BranchTopology(BaseModel):
    """Snapshot of the active branch graph."""

    model_config = ConfigDict(frozen=True)

    main: str = Field(description="Name of the long-lived 'main' branch (usually 'main').")
    main_head_sha: str
    branches: tuple[BranchNode, ...] = Field(default_factory=tuple)


class RepoState(BaseModel):
    """Snapshot returned by ``RepoStateProvider.get_repo_state``.

    Designed so the action policy can switch on ``is_fresh``,
    ``uncommitted_changes`` and ``last_quiescence_at`` without a second
    round-trip to the wrapper layer.
    """

    model_config = ConfigDict(frozen=True)

    is_fresh: bool = Field(
        description=(
            "True iff the repo has only the bootstrap commit. Distinct "
            "from 'is empty': a freshly bootstrapped monorepo carries "
            "exactly one commit and is_fresh=True."
        ),
    )
    current_branch: str
    current_sha: str
    ahead_of_main_by: int = 0
    behind_main_by: int = 0
    checkpoints: tuple[Checkpoint, ...] = Field(default_factory=tuple)
    forks: tuple[ForkBranch, ...] = Field(default_factory=tuple)
    tools: tuple[ToolEntry, ...] = Field(default_factory=tuple)
    last_quiescence_at: datetime | None = None
    uncommitted_changes: bool = False
    imported_tooling_remotes: tuple[ImportedRemote, ...] = Field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Bootstrap spec — input to ToolBuilder.bootstrap_repo
# ---------------------------------------------------------------------------


_TARGET_PATTERN = (
    r"^(subdir_in_monorepo:tools/[^:]+|"
    r"branch_in_existing:[^:]+|"
    r"new_standalone:.+)$"
)


class RepoBootstrapSpec(BaseModel):
    """Input to ``ToolBuilder.bootstrap_repo``.

    The target string is the discriminator for *where* the scaffold
    goes:

    - ``subdir_in_monorepo:tools/<purpose>/<name>`` — canonical; the tool
      lives as a subdirectory of the design monorepo. This is what 99 %
      of tool-building pools should use.
    - ``branch_in_existing:<branch_name>`` — for tools developed on a
      side branch before they are merged back. Rare.
    - ``new_standalone:<git_url>`` — for tools intended for immediate
      external publication (the publish-to-PyPI gate). Requires explicit
      user approval (HITL frequency = APPROVAL_GATES) at the dispatcher
      level — the framework itself does not enforce that here.
    """

    model_config = ConfigDict(frozen=True)

    template: Literal[
        "python_lib",
        "c_library",
        "julia_module",
        "rust_crate",
        "cmake_project",
    ] = Field(description="Scaffold template name; must match a directory under design_monorepo/scaffolds/.")
    target: Annotated[str, Field(pattern=_TARGET_PATTERN)] = Field(
        description="Discriminator string; see class docstring.",
    )
    name: str = Field(description="Tool name (snake_case is preferred).")
    purpose: str = Field(
        description=(
            "tools/<purpose>/ subdirectory. Required when target is "
            "subdir_in_monorepo, optional otherwise (carried as metadata)."
        ),
    )
    license: str = Field(
        description="SPDX identifier. The pool's licence policy is checked at dispatch time.",
    )
    capability: str = Field(
        description="Capability key the tool fulfils (registered in tool-registry.json).",
    )
    description: str = Field(default="")
    initial_files: Mapping[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-file content overrides applied on top of the "
            "scaffold (relative path -> file content). Empty by default."
        ),
    )
    template_vars: Mapping[str, str] = Field(
        default_factory=dict,
        description=(
            "Variables available to the scaffold's str.Template "
            "substitutions in addition to the standard ones (name, "
            "purpose, license, description, year, author)."
        ),
    )


class BootstrapResult(BaseModel):
    """Return value of ``ToolBuilder.bootstrap_repo``."""

    model_config = ConfigDict(frozen=True)

    target: str = Field(description="Echo of the input spec's target.")
    relative_path: str = Field(
        description="Path within the working tree where the scaffold was written.",
    )
    sha: str = Field(description="Commit that contains the scaffold.")
    files_created: tuple[str, ...] = Field(default_factory=tuple)
    tool_entry: ToolEntry = Field(
        description=(
            "Newly registered tool entry (the registry has been updated "
            "and committed)."
        ),
    )


class ExtensionAuthoredPayload(BaseModel):
    """Audit payload for an L1-E ``bootstrap_<surface>`` call.

    Written to the blackboard under
    :meth:`DesignMonorepoEventProtocol.extension_authored_key` and
    returned by every ``ToolBuilder.bootstrap_<surface>`` action.
    Carries the provenance that lets a future session answer "which
    session authored this extension, when, in response to which user
    message" (Risk #5 in the alignment plan).

    ``user_message_id`` is reserved for future provenance plumbing —
    no entry-point sets it today, so the default is ``None``.
    """

    model_config = ConfigDict(frozen=True)

    surface: Literal[
        "plugins", "agents", "deployments", "tools", "profiles",
    ] = Field(
        description=(
            "Surface kind — must match a key in "
            "``DEFAULT_SURFACE_DIRS``."
        ),
    )
    name: str = Field(
        description="Agent-supplied extension name (file/directory stem).",
    )
    relative_path: str = Field(
        description=(
            "Path of the authored file or directory, relative to the "
            "design monorepo working tree."
        ),
    )
    commit_sha: str = Field(
        description="Commit that contains the new extension.",
    )
    template: str = Field(
        description="Scaffold template name that produced the file.",
    )
    files_created: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Paths written, relative to ``relative_path``.",
    )
    authored_at: datetime = Field(
        description="UTC timestamp at which the bootstrap action committed.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Session that requested the authoring — pulled from "
            "``get_current_session_id()`` at call time."
        ),
    )
    user_message_id: str | None = Field(
        default=None,
        description=(
            "Future-reserved: the user-message id that triggered the "
            "authoring. No provenance source populates this today."
        ),
    )


# ---------------------------------------------------------------------------
# Page-change events — yielded by ``ContextPageSource.watch()`` and fed
# directly into the convergence runtime by VCM's watch bridge.
# ---------------------------------------------------------------------------


class PageChangeEvent(BaseModel):
    """One page-graph mutation event.

    Five concrete kinds, distinguished by the ``kind`` literal. Defined
    as one model instead of five subclasses so the blackboard transport
    can carry them without dispatch.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal[
        "page_invalidated",
        "page_replaced",
        "page_graph_edge_added",
        "page_graph_edge_removed",
        "page_added",
    ]
    page_id: str = Field(description="Affected page's stable id.")
    related_page_ids: tuple[str, ...] = Field(default_factory=tuple)
    edge_type: str | None = None
    source: str = Field(
        description="Source URI, e.g. 'git:<remote>:<branch>:<commit>'.",
    )
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, object] = Field(default_factory=dict)


__all__ = (
    "Checkpoint",
    "ForkBranch",
    "ToolEntry",
    "ToolMatch",
    "ImportedRemote",
    "DecisionEntry",
    "DesignDiffEntry",
    "DesignDiff",
    "BranchNode",
    "BranchTopology",
    "RepoState",
    "RepoBootstrapSpec",
    "BootstrapResult",
    "PageChangeEvent",
)
