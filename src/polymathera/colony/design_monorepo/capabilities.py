"""Agent-facing capabilities over the design monorepo.

Three capabilities, each a subclass of ``AgentCapability``:

- ``RepoStateProvider`` — read-only view (master §3.5.1). Auto-installable
  on every agent when the deployment has a design monorepo configured.
- ``DesignCheckpointer`` — write-side wrappers for ``checkpoint_state``,
  ``restore_checkpoint``, ``fork_design``, ``merge_design``,
  ``cherry_pick_decisions``, ``commit_state``, ``tag_checkpoint``,
  ``list_checkpoints``, ``list_forks``, ``diff_design`` (master §8.1).
- ``ToolBuilder`` — ``bootstrap_repo`` (master §9.4). Tool-building pools
  install this on their pool agents.

All three share a small base class (``_DesignMonorepoCapabilityBase``)
that resolves the ``DesignMonorepoClient`` lazily, derives a per-call
``AgentIdentity`` from the owning agent, and routes git operations
through ``asyncio.to_thread`` so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from overrides import override

from ..agents.base import Agent, AgentCapability
from ..agents.blackboard import BlackboardEvent, ConvergenceQuiescenceProtocol
from ..agents.blackboard.protocol import (
    DesignMonorepoEventProtocol,
    VCMEventProtocol,
)
from ..agents.models import AgentSuspensionState
from ..agents.patterns.actions import action_executor
from ..agents.patterns.events import EventProcessingResult, event_handler
from .client import (
    DECISIONS_DIR,
    DesignMonorepoClient,
    DesignMonorepoError,
)
from .identity import (
    AgentIdentity,
    CommitIdentity,
    append_co_author_trailer,
    resolve_commit_identity,
)
from .manifest import MANIFEST_RELATIVE_PATH, DesignMonorepoManifest
from .repo_map import REPO_MAP_DIR, REPO_MAP_FILENAME
from .models import (
    BootstrapResult,
    BranchTopology,
    Checkpoint,
    DecisionEntry,
    DesignDiff,
    ForkBranch,
    RepoBootstrapSpec,
    RepoState,
    ToolEntry,
    ToolMatch,
)
from . import registry as registry_module
from . import scaffolds as scaffolds_module

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _commit_all(client: "DesignMonorepoClient", identity: AgentIdentity, message: str) -> str:
    """Worker for ``asyncio.to_thread`` — keyword-only kwargs aren't
    compatible with ``run_in_executor``'s positional-only call shape."""

    return client.commit_with_identity(identity, message, all_changes=True)


def _commit_paths(
    client: "DesignMonorepoClient",
    identity: AgentIdentity,
    message: str,
    paths: list[Path] | None,
    all_changes: bool,
) -> str:
    return client.commit_with_identity(
        identity, message, paths=paths, all_changes=all_changes,
    )


def _lfs_patterns_from_gitattributes(path: Path) -> list[str]:
    """Parse ``.gitattributes`` and return the patterns that route
    through LFS — i.e. lines whose attribute set contains
    ``filter=lfs``.

    Used by :meth:`DesignCheckpointer.initialize_repo_map` when the
    operator opts into ``migrate_existing_to_lfs=True``: ``git lfs
    migrate import`` needs an explicit ``--include=<csv>`` and we
    want to migrate exactly what ``.gitattributes`` declares
    (including any patterns the operator added beyond the default
    template).
    """
    if not path.is_file():
        return []
    patterns: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: ``<pattern> attr1=val1 attr2=val2 ...``
        parts = line.split()
        if len(parts) < 2:
            continue
        if any(p == "filter=lfs" for p in parts[1:]):
            patterns.append(parts[0])
    return patterns


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _DesignMonorepoCapabilityBase(AgentCapability):
    """Shared plumbing for the three design-monorepo capabilities.

    Holds a lazy ``DesignMonorepoClient`` keyed by the working_dir. The
    client is opened on first use and reused across calls; if the agent
    is suspended and resumed, the client opens fresh on the next call.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        working_dir: Path | str | None = None,
        clone_scope_id: str | None = None,
        read_only: bool = False,
        input_patterns: list[str] | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        # ``input_patterns`` defaults to ``None`` so the base class
        # auto-infers from ``@event_handler`` decorators on the
        # subclass — DesignCheckpointer's quiescence handler picks up
        # ``ConvergenceQuiescenceProtocol.quiescence_pattern()`` that
        # way. Pass ``input_patterns=[]`` explicitly to opt out (e.g.,
        # for the read-only RepoStateProvider / ToolBuilder).
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        if working_dir is None:
            # Resolve a per-agent (or shared read-only) clone path under
            # ``/mnt/shared`` so the layout survives Ray actor restarts.
            from .clones import resolve_clone_path
            resolved = resolve_clone_path(
                agent=agent,
                scope_id=clone_scope_id or self.scope_id,
                read_only=read_only,
            )
            self._working_dir = resolved
        else:
            self._working_dir = Path(working_dir)
        self._read_only = read_only
        self._client: DesignMonorepoClient | None = None

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    _DESIGN_MONOREPO_URL_KEY = "design_monorepo_url"

    def _client_sync(self) -> DesignMonorepoClient:
        if self._client is None:
            # Lazy-clone: if the per-agent ``working_dir`` does not yet
            # contain a git repo, clone the colony's configured design
            # monorepo into it. The URL is read from
            # ``agent.metadata.parameters[design_monorepo_url]``,
            # which the dashboard populates at session-creation time
            # (or which ``DesignMonorepoBootstrap.set_design_monorepo``
            # mutates in-place for chat-driven configuration). Auth is
            # the operator's responsibility — git's standard machinery
            # (credential helper, token-in-URL, ssh-agent) handles it
            # transparently and any failure surfaces verbatim.
            if not (self._working_dir / ".git").exists() and not self._read_only:
                self._lazy_clone_from_agent_metadata()
            self._client = DesignMonorepoClient.open(self._working_dir)
        return self._client

    def _lazy_clone_from_agent_metadata(self) -> None:
        """Issue a raw ``git clone`` into ``self._working_dir`` when an
        ``agent.metadata.parameters[design_monorepo_url]`` is present.
        No-op when the parameter is missing — :meth:`_client_sync`
        then falls back to ``open()`` and the caller sees the
        unmodified ``DesignMonorepoError``.
        """

        if self._agent is None:
            return
        params = getattr(self._agent.metadata, "parameters", None) or {}
        url = params.get(self._DESIGN_MONOREPO_URL_KEY)
        if not url:
            return
        from git import Repo  # local import — gitpython is in the design_monorepo extra
        from git.exc import GitCommandError

        from ..distributed.stores.git import _classify_git_clone_error

        # Authentication for github.com / gitlab.com flows through the
        # system-level credential helper baked into the container image
        # (see ``Dockerfile.local``). The helper reads
        # ``$GITHUB_TOKEN`` / ``$GITLAB_TOKEN`` from the process
        # environment and feeds them to git on demand. Pass the URL
        # bare; do NOT embed credentials.
        self._working_dir.mkdir(parents=True, exist_ok=True)
        try:
            cloned = Repo.clone_from(url, str(self._working_dir))
        except GitCommandError as exc:
            raise _classify_git_clone_error(exc) from exc
        # Activate Git LFS clean/smudge filters on this fresh clone.
        # Pointer files in the upstream are de-referenced into real
        # blobs by the smudge filter on checkout, and any large file
        # the agent later commits routes through clean. Without this
        # the agent could push plain-git blobs that bypass LFS and
        # bloat the repo (or trip GitHub's 100 MB hard limit). Best-
        # effort: ``git-lfs`` may not be installed on a bare-metal
        # dev box; degrade gracefully there.
        try:
            cloned.git.lfs("install", "--local")
        except Exception as exc:  # noqa: BLE001 - LFS may not be installed
            logger.warning(
                "git lfs install failed at %s (%s); LFS hooks are "
                "not active. Install git-lfs and re-run.",
                self._working_dir, exc,
            )

    async def _client_async(self) -> DesignMonorepoClient:
        return await asyncio.to_thread(self._client_sync)

    def _manifest(self) -> DesignMonorepoManifest:
        return self._client_sync().manifest

    def _identity(self) -> AgentIdentity:
        """Derive the per-call commit identity from the owning agent."""

        if self.is_detached:
            return AgentIdentity(
                agent_id=self.scope_id,
                role="external",
                colony_id="external",
                agent_email_domain=self._manifest().agent_email_domain,
            )
        agent = self.agent  # raises in detached mode, but we returned above
        role = "agent"
        try:
            md = getattr(agent, "metadata", None)
            if md is not None and getattr(md, "role", None):
                role = str(md.role)
        except Exception:  # noqa: BLE001
            pass
        colony_id = getattr(agent, "colony_id", "default") or "default"
        return AgentIdentity(
            agent_id=agent.agent_id,
            role=role,
            colony_id=colony_id,
            agent_email_domain=self._manifest().agent_email_domain,
        )

    # Live page-change events for the working tree flow through
    # ``GitRepoContextPageSource.watch()`` once the working tree
    # is mapped into the VCM (the source composes a LocalFsWatcher +
    # GitRemoteWatcher; VCM feeds the merged stream into the
    # convergence runtime). Capabilities here do not register a
    # separate watcher — that produced duplicate events and a
    # duplicate code path.

    # -- Suspension hooks --
    # The design monorepo's state is already durable on disk; the
    # ``working_dir`` is part of the agent's deployment configuration
    # rather than its serialised state, so suspension/resumption is a
    # no-op for these capabilities.

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# RepoStateProvider — read-only
# ---------------------------------------------------------------------------


class RepoStateProvider(_DesignMonorepoCapabilityBase):
    """Read-only query surface over the design monorepo.

    Auto-installable on every agent when the deployment has a design
    monorepo configured (master §3.5.1). The action policy uses these
    methods at the start of a session, at hypothesis-game boundaries,
    or whenever the convergence runtime signals quiescence, to decide
    what to do next without raising any concern about side effects.

    Pure action surface — declares no ``@event_handler`` methods, so
    ``input_patterns=[]`` is passed explicitly to opt out of the base
    ``AgentCapability``'s wildcard fallback. Otherwise the agent's
    own ``policy:action_started:*`` lifecycle writes would be fed
    back into the action policy's event queue.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        working_dir: Path | str | None = None,
        clone_scope_id: str | None = None,
        read_only: bool = False,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            working_dir=working_dir,
            clone_scope_id=clone_scope_id,
            read_only=read_only,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"design_state", "git", "read"})

    @action_executor(planning_summary="Snapshot the program's design-monorepo state.")
    async def get_repo_state(self) -> RepoState:
        """Return a snapshot of the design monorepo's state.

        ``is_fresh`` indicates a freshly-bootstrapped repo; ``checkpoints``
        / ``forks`` / ``tools`` are surfaced for the planner; the
        ``last_quiescence_at`` time and ``uncommitted_changes`` flag let
        the policy detect a dirty restart.
        """

        client = await self._client_async()
        return await asyncio.to_thread(client.current_state)

    @action_executor(planning_summary="Search registered tools by capability query.")
    async def find_existing_tool(
        self,
        capability_query: str,
        require_writable: bool = False,
    ) -> list[ToolMatch]:
        """Search ``tools/`` (this monorepo + imported tooling carry-over)
        for a tool that satisfies ``capability_query``.

        Used as the *first* step before any bootstrap (master §9.4).
        Returns ranked matches; ``writable=False`` indicates an entry
        that lives in an imported tooling-monorepo and must be promoted
        before it can be modified.
        """

        client = await self._client_async()
        matches = await asyncio.to_thread(
            client.find_existing_tool,
            capability_query,
            require_writable=require_writable,
        )
        return list(matches)

    @action_executor(
        planning_summary="List recent design decisions since a baseline ref.",
    )
    async def list_recent_decisions(
        self,
        since: str | None = None,
        limit: int = 50,
    ) -> list[DecisionEntry]:
        """Walk ``design/decisions/`` for decisions made since ``since``.

        ``since`` is a commit SHA, tag, or branch name; ``None`` means the
        full history of the current branch. Useful for cross-session
        continuity ("what did the previous session decide?") and to
        avoid proposing alternatives that are already settled.
        """

        client = await self._client_async()
        result = await asyncio.to_thread(
            client.list_recent_decisions,
            since=since,
            limit=limit,
        )
        return list(result)

    @action_executor(
        planning_summary="Diff the current design state against a checkpoint.",
    )
    async def diff_against_checkpoint(self, checkpoint_id: str) -> DesignDiff:
        client = await self._client_async()
        return await asyncio.to_thread(
            client.diff_against_checkpoint, checkpoint_id
        )

    @action_executor(planning_summary="Return the active branch graph.")
    async def get_branch_topology(self) -> BranchTopology:
        client = await self._client_async()
        return await asyncio.to_thread(client.get_branch_topology)


# ---------------------------------------------------------------------------
# DesignCheckpointer — write-side wrappers (master §8.1)
# ---------------------------------------------------------------------------


class DesignCheckpointer(_DesignMonorepoCapabilityBase):
    """Write-side capabilities over the design monorepo.

    Each operation is gated downstream by HITL policy at the dispatcher
    level (master §3.1 access-control discipline). Per-commit identity
    is derived from the owning agent so the audit trail is provably
    attributable (master §8.5).

    The capability also subscribes to
    ``ConvergenceQuiescenceProtocol.quiescence_pattern()`` and emits an
    ``auto_quiescence_<iso8601>`` checkpoint tag whenever an episode
    settles with uncommitted changes — the crash-recovery primitive
    master §8.1 / line 607 calls out (``restore_checkpoint(id=
    auto_quiescence_<timestamp>)``). The behaviour is opt-out via
    ``auto_checkpoint_on_quiescence=False``; nothing happens when the
    working tree is clean (HEAD already represents the settled state).
    """

    AUTO_CHECKPOINT_LABEL_FMT = "auto_quiescence_{timestamp}"

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        working_dir: Path | str | None = None,
        clone_scope_id: str | None = None,
        auto_checkpoint_on_quiescence: bool = True,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        # ``DesignCheckpointer`` writes checkpoint commits / tags / new
        # branches, so ``read_only`` is hard-wired to ``False``.
        #
        # When auto-checkpoint is disabled, also opt out of the
        # quiescence subscription. Otherwise the action policy's event
        # queue receives a wake-up on every episode boundary, which —
        # in ``reactive_only`` agents like SessionAgent — would trigger
        # a full LLM plan_step → action → settles → new quiescence
        # loop. The remote-change handler stays subscribed; it only
        # fires on actual upstream changes.
        if auto_checkpoint_on_quiescence:
            input_patterns: list[str] | None = None  # auto-infer
        else:
            input_patterns = [VCMEventProtocol.reindexed_pattern()]
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            working_dir=working_dir,
            clone_scope_id=clone_scope_id,
            read_only=False,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        self._auto_checkpoint_on_quiescence = auto_checkpoint_on_quiescence

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"design_state", "git", "write"})

    @action_executor(
        planning_summary="Commit current design state and tag a checkpoint.",
    )
    async def checkpoint_state(
        self,
        label: str,
        rationale: str = "",
        *,
        all_changes: bool = True,
    ) -> Checkpoint:
        """Commit any uncommitted state and create a ``checkpoint/<...>`` tag.

        Caller is responsible for waiting for convergence quiescence
        before invoking; the convergence runtime emits a quiescence
        event that the planner can listen for. ``rationale`` is recorded
        in the tag annotation alongside the label.
        """

        client = await self._client_async()
        identity = self._identity()
        if all_changes:
            await asyncio.to_thread(
                _commit_all,
                client,
                identity,
                f"checkpoint: {label}",
            )
        return await asyncio.to_thread(
            client.tag_checkpoint, identity, label, rationale,
        )

    @action_executor(
        planning_summary="Restore a tagged checkpoint, optionally as a new fork.",
    )
    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        mode: str = "replace",
        recovery_label: str | None = None,
    ) -> str:
        """Restore the working tree to ``checkpoint_id``.

        ``mode='replace'`` checks out the tag on the current branch;
        ``mode='fork'`` creates a new ``fork/<recovery_label>`` branch
        off the tag. Dirty state is auto-stashed onto a ``recovery/*``
        branch before the operation (master §8.6).
        """

        client = await self._client_async()
        identity = self._identity()
        return await asyncio.to_thread(
            client.restore_checkpoint,
            identity,
            checkpoint_id,
            mode=mode,
            recovery_label=recovery_label,
        )

    @action_executor(
        planning_summary="Create a fork/<label> branch off HEAD.",
    )
    async def fork_design(
        self,
        label: str,
        *,
        from_sha: str | None = None,
        checkout: bool = True,
    ) -> ForkBranch:
        client = await self._client_async()
        identity = self._identity()
        return await asyncio.to_thread(
            client.fork, identity, label, from_sha=from_sha, checkout=checkout,
        )

    @action_executor(
        planning_summary="Merge a source branch into the current branch.",
    )
    async def merge_design(
        self,
        source_branch: str,
        target_branch: str | None = None,
        message: str | None = None,
    ) -> str:
        """Merge ``source_branch`` into ``target_branch`` (default: current).

        Returns the new merge-commit SHA. Conflicts surface as
        ``DesignMonorepoError``; the caller resolves and commits.
        Selective merge ("take *these* decisions from fork/A") goes
        through ``cherry_pick_decisions`` instead.
        """

        client = await self._client_async()
        identity = self._identity()
        return await asyncio.to_thread(
            client.merge_full,
            identity,
            source_branch,
            target_branch=target_branch,
            message=message,
        )

    @action_executor(
        planning_summary="Cherry-pick a set of commits onto the current branch.",
    )
    async def cherry_pick_decisions(
        self,
        commit_shas: list[str],
        target_branch: str | None = None,
    ) -> list[str]:
        """Cherry-pick ``commit_shas`` (in order) onto ``target_branch``.

        The caller (typically the LLM-assisted decision-to-SHA resolver
        — master §8.3) selects the SHAs by walking ``design/decisions/``
        and identifying the commits that authored each chosen decision.
        """

        client = await self._client_async()
        identity = self._identity()
        result = await asyncio.to_thread(
            client.cherry_pick,
            identity,
            commit_shas,
            target_branch=target_branch,
        )
        return list(result)

    @action_executor(
        planning_summary="Commit specific paths under the agent's identity.",
    )
    async def commit_state(
        self,
        message: str,
        paths: list[str] | None = None,
        all_changes: bool = False,
    ) -> str:
        """Stage ``paths`` (or everything if ``all_changes``) and commit.

        Returns the new commit's SHA, or HEAD's SHA when nothing was
        staged. The commit is authored under the agent's transactional
        identity; the global git config is not mutated.
        """

        client = await self._client_async()
        identity = self._identity()
        path_objs: list[Path] | None = None
        if paths is not None:
            path_objs = [Path(p) for p in paths]
        return await asyncio.to_thread(
            _commit_paths,
            client,
            identity,
            message,
            path_objs,
            all_changes,
        )

    @action_executor(
        planning_summary="Create an annotated checkpoint tag at HEAD or a SHA.",
    )
    async def tag_checkpoint(
        self,
        label: str,
        rationale: str = "",
        sha: str | None = None,
    ) -> Checkpoint:
        client = await self._client_async()
        identity = self._identity()
        return await asyncio.to_thread(
            client.tag_checkpoint, identity, label, rationale, sha=sha,
        )

    @action_executor(planning_summary="List all checkpoint tags in the monorepo.")
    async def list_checkpoints(self) -> list[Checkpoint]:
        client = await self._client_async()
        result = await asyncio.to_thread(client.list_checkpoints)
        return list(result)

    @action_executor(planning_summary="List active fork/* branches.")
    async def list_forks(self) -> list[ForkBranch]:
        client = await self._client_async()
        result = await asyncio.to_thread(client.list_forks)
        return list(result)

    @action_executor(planning_summary="Diff between two refs.")
    async def diff_design(self, ref_a: str, ref_b: str) -> DesignDiff:
        client = await self._client_async()
        return await asyncio.to_thread(client.diff, ref_a, ref_b)

    @action_executor(
        planning_summary=(
            "Initialise the design monorepo: write a default "
            "``.colony/manifest.json`` and a commented "
            "``.colony/repo_map.yaml`` template, then commit them."
        ),
    )
    async def initialize_repo_map(
        self, *,
        push: bool = True,
        enable_lfs: bool = True,
        migrate_existing_to_lfs: bool = False,
    ) -> dict[str, Any]:
        """Bring an empty / freshly-cloned repo into a state where the
        rest of the design-monorepo capabilities work.

        Writes whichever of ``.colony/manifest.json`` and
        ``.colony/repo_map.yaml`` is missing, commits the new files
        under the agent's identity, and (when ``push=True``) pushes
        the commit to ``origin`` so the changes are visible on the
        upstream remote and to other clones of the repo (the
        dashboard's read-only inspection cache, other agents).
        **Idempotent**: existing files are never overwritten —
        operator edits stay intact.

        Why both files: ``manifest.json`` is the framework's required
        marker that "this is a Colony design monorepo"
        (:class:`DesignMonorepoClient.open` raises without it).
        ``repo_map.yaml`` declares VCM ingestion. A fresh clone has
        neither; the action creates whatever's needed in one shot so
        the operator doesn't trip a "Manifest not found" error from a
        sibling action.

        The manifest is built with sensible defaults: ``tenant`` /
        ``colony`` from the current execution context, ``program``
        defaulting to the colony id, ``target_system`` left as
        ``"unspecified"``, ``design_repo_url`` from the agent's
        ``design_monorepo_url`` parameter (or empty for a local-only
        bootstrap). Operator can edit any of these later — the file
        is plain JSON.

        The repo_map template is intentionally minimal: the only
        active row is the default ``git_repo`` source over the whole
        tree. Below that, every supported source type and
        ``knowledge_routing`` flavour ships as commented examples for
        the operator to un-comment and adapt. We do NOT auto-detect
        repo structure or scaffold ``tools/``, ``literature/``, etc.
        — prescribing a layout this early in the lifecycle costs
        more than the 30 lines of YAML the operator reads once.

        Bypasses :meth:`_client_async` deliberately: the client's
        ``open()`` requires a valid manifest, which is exactly what
        we may be about to create. Uses gitpython directly on the
        working tree instead.

        Args:
            push: When ``True`` (default), runs ``git push origin
                HEAD:<branch>`` after the commit. The local commit is
                already on disk by then, so a push failure (network,
                auth, branch protection) leaves the per-agent clone
                in a committed-but-unpushed state — the operator can
                re-push manually. The error string is returned in
                ``push_error`` so the planner can surface it. Set to
                ``False`` to commit locally only (e.g., for a
                file://-only test bootstrap or when the operator
                wants to review before publishing).
            enable_lfs: When ``True`` (default), activates Git LFS for
                the design monorepo:
                  - runs ``git lfs install --local`` on the working
                    tree (idempotent — installs the clean/smudge hooks);
                  - writes a default ``.gitattributes`` declaring LFS
                    patterns for documents, archives, scientific data,
                    images, CAD/3D, ML weights, and audio/video, **only
                    when no .gitattributes exists** (operator edits are
                    never overwritten);
                  - flips ``manifest.lfs.mode`` from ``"disabled"`` to
                    ``"same_remote"`` if needed, so other clones of
                    this repo also activate LFS.
                **Forward-only**: blobs that were already committed
                before ``.gitattributes`` was added stay as plain git
                objects in history. Use ``migrate_existing_to_lfs`` to
                convert them.
            migrate_existing_to_lfs: When ``True`` (default ``False``),
                runs ``git lfs migrate import --include=<patterns>``
                after the initial commit to convert already-committed
                files matching the LFS patterns into LFS pointers.
                **Rewrites history**: every commit SHA on the current
                branch changes, the next push needs ``--force``, and
                anyone else who already cloned the repo will need to
                re-clone. Off by default; only opt in on a fresh repo
                or when you're sure no one else has a working clone.
                Has no effect when ``enable_lfs`` is ``False``.
        """

        return await asyncio.to_thread(
            self._initialize_repo_map_sync,
            push=push,
            enable_lfs=enable_lfs,
            migrate_existing_to_lfs=migrate_existing_to_lfs,
        )

    def _initialize_repo_map_sync(
        self, *,
        push: bool,
        enable_lfs: bool,
        migrate_existing_to_lfs: bool,
    ) -> dict[str, Any]:
        """Synchronous body of :meth:`initialize_repo_map`. Lives on a
        thread (see the action wrapper) because gitpython operations
        are blocking."""

        from git import Repo

        repo_root = self._working_dir
        # Trigger the lazy-clone path that ``_client_sync`` would
        # normally run. We bypass ``_client_sync`` here because it
        # ends in ``DesignMonorepoClient.open`` which insists on a
        # manifest — the very file this action may be about to
        # create. But the clone still has to happen, so call the
        # cloner directly. ``_lazy_clone_from_agent_metadata`` is a
        # no-op when ``.git`` is already present or when no URL is
        # configured on agent metadata.
        if not (repo_root / ".git").is_dir():
            self._lazy_clone_from_agent_metadata()
        if not (repo_root / ".git").is_dir():
            raise DesignMonorepoError(
                f"{repo_root} is not a git repository and no "
                "``design_monorepo_url`` is configured on the agent's "
                "metadata — set the URL on the colony (Landing page → "
                "Colonies → pencil) and start a fresh session, or "
                "``git init`` the working tree manually if you mean "
                "to bootstrap a local-only design monorepo.",
            )

        manifest_path = repo_root / MANIFEST_RELATIVE_PATH
        repo_map_path = repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME
        gitattributes_path = repo_root / ".gitattributes"
        manifest_existed = manifest_path.is_file()
        repo_map_existed = repo_map_path.is_file()
        gitattributes_existed = gitattributes_path.is_file()

        files_changed: list[str] = []

        # ---- Manifest ---------------------------------------------------
        if not manifest_existed:
            manifest = self._build_default_manifest(enable_lfs=enable_lfs)
            manifest.write_path(repo_root)
            files_changed.append(MANIFEST_RELATIVE_PATH)
            agent_email_domain = manifest.agent_email_domain
        else:
            existing_manifest = DesignMonorepoManifest.load_path(repo_root)
            agent_email_domain = existing_manifest.agent_email_domain
            # If the operator is opting into LFS on a previously-
            # initialised repo whose manifest has ``lfs.mode="disabled"``,
            # flip it to ``"same_remote"`` so other clones of this repo
            # also activate LFS. Leave any non-disabled mode alone — the
            # operator may have configured ``"separate"`` deliberately.
            if enable_lfs and existing_manifest.lfs.mode == "disabled":
                from .manifest import LFSConfig
                updated = existing_manifest.model_copy(
                    update={"lfs": LFSConfig(mode="same_remote")},
                )
                updated.write_path(repo_root)
                files_changed.append(MANIFEST_RELATIVE_PATH)

        # ---- repo_map.yaml ---------------------------------------------
        if not repo_map_existed:
            template = (
                Path(__file__).parent / "templates" / "repo_map.template.yaml"
            ).read_text(encoding="utf-8")
            repo_map_path.parent.mkdir(parents=True, exist_ok=True)
            repo_map_path.write_text(template, encoding="utf-8")
            files_changed.append(f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}")

        # ---- LFS: hooks + .gitattributes -------------------------------
        # ``git lfs install --local`` writes the clean/smudge filter
        # config into ``.git/config`` and the ``pre-push`` hook into
        # ``.git/hooks/``. Idempotent — running twice is a no-op.
        # Without this, ``git add`` of a tracked path goes to plain git,
        # not LFS, and a subsequent push to GitHub will be rejected for
        # files > 100 MB. Tolerate ``git lfs`` not being installed (the
        # framework's container images ship it; bare-metal dev boxes
        # without it degrade gracefully — same pattern the bootstrap
        # path uses).
        if enable_lfs:
            try:
                Repo(str(repo_root)).git.lfs("install", "--local")
            except Exception as exc:  # noqa: BLE001 - LFS may not be installed
                logger.warning(
                    "git lfs install failed at %s (%s); "
                    "LFS hooks are NOT active. Install git-lfs and re-run.",
                    repo_root, exc,
                )
            if not gitattributes_existed:
                template = (
                    Path(__file__).parent / "templates" / "gitattributes.template"
                ).read_text(encoding="utf-8")
                gitattributes_path.write_text(template, encoding="utf-8")
                files_changed.append(".gitattributes")

        if not files_changed:
            return {
                "status": "already_initialized",
                "files_created": [],
                "committed_sha": None,
                "pushed": False,
                "push_error": None,
                "lfs_enabled": enable_lfs,
                "migrated_to_lfs": False,
                "migrate_error": None,
            }

        repo = Repo(str(repo_root))
        repo.index.add(files_changed)
        # Per-commit attribution. Principal becomes author/committer;
        # co-author (when set) is rendered as a ``Co-Authored-By:``
        # trailer. Configured per-colony via the landing-page UI and
        # plumbed onto agent metadata at session creation; defaults
        # are ``principal=colony, co_author=user``.
        principal_id, co_author_id = self._resolve_attribution(
            agent_email_domain=agent_email_domain,
        )
        message = append_co_author_trailer(
            "init: scaffold .colony/ (manifest + repo_map.yaml)",
            co_author_id,
        )
        actor = principal_id.actor()
        commit = repo.index.commit(
            message,
            author=actor,
            committer=actor,
        )

        # Push so the changes are visible on the upstream and so
        # other clones (dashboard cache, other agents) can pick them
        # up on their next fetch. We push HEAD:<active-branch> rather
        # than a bare ``push()`` because the per-agent clone may have
        # been cloned from a default branch the user later renamed —
        # being explicit avoids "src refspec doesn't match" surprises.
        # Failures (no ``origin``, no permission, branch protection,
        # offline) leave the local commit on disk and are surfaced in
        # the result so the planner / operator can act on them.
        pushed = False
        push_error: str | None = None
        if push:
            try:
                branch_name = repo.active_branch.name
            except TypeError:
                # Detached HEAD — no branch to push to. Rare in this
                # bootstrap path but possible if a prior action
                # checked out a tag.
                push_error = (
                    "HEAD is detached; cannot push. Switch to a branch "
                    "and re-run with ``push=True``."
                )
            else:
                try:
                    origin = repo.remote("origin")
                except ValueError as exc:
                    # No ``origin`` remote — typical of a local-only
                    # ``git init`` bootstrap. That's a legitimate
                    # workflow; just record it and move on.
                    push_error = f"no ``origin`` remote configured: {exc}"
                else:
                    try:
                        push_info = origin.push(
                            refspec=f"HEAD:refs/heads/{branch_name}",
                        )
                        # gitpython returns a list of PushInfo; any
                        # ERROR flag means the push didn't reach the
                        # remote even if the call returned normally.
                        from git import PushInfo
                        bad = [pi for pi in push_info if pi.flags & PushInfo.ERROR]
                        if bad:
                            push_error = "; ".join(
                                pi.summary.strip() or repr(pi.flags) for pi in bad
                            )
                        else:
                            pushed = True
                    except Exception as exc:  # noqa: BLE001
                        push_error = str(exc)

        # Optional history rewrite: convert already-committed blobs
        # matching the LFS patterns into LFS pointers. Off by default
        # because ``git lfs migrate import`` rewrites every commit
        # SHA on the migrated refs — anyone else's clones break and
        # the next push needs ``--force``. When the operator opts in,
        # we run the migration AFTER the bootstrap commit so the
        # newly-written ``.gitattributes`` is part of the migrated
        # history (otherwise the patterns wouldn't match anything).
        migrated = False
        migrate_error: str | None = None
        if enable_lfs and migrate_existing_to_lfs:
            patterns = _lfs_patterns_from_gitattributes(gitattributes_path)
            if not patterns:
                migrate_error = (
                    "No LFS patterns found in .gitattributes; "
                    "skipping migrate."
                )
            else:
                try:
                    repo.git.lfs(
                        "migrate", "import",
                        f"--include={','.join(patterns)}",
                        "--everything",
                    )
                    migrated = True
                except Exception as exc:  # noqa: BLE001
                    migrate_error = str(exc)

        return {
            "status": "initialized",
            "files_created": files_changed,
            "committed_sha": commit.hexsha,
            "pushed": pushed,
            "push_error": push_error,
            "lfs_enabled": enable_lfs,
            "migrated_to_lfs": migrated,
            "migrate_error": migrate_error,
        }

    def _build_default_manifest(
        self, *, enable_lfs: bool = True,
    ) -> DesignMonorepoManifest:
        """Construct a manifest with defaults pulled from execution
        context + agent metadata. Operator-tunable fields
        (``program``, ``target_system``, ``design_repo_url``) get
        sensible placeholders; the file is plain JSON for later
        editing.

        ``lfs.mode`` defaults to ``"same_remote"`` when ``enable_lfs``
        is True (the framework's default), or ``"disabled"`` when the
        operator explicitly opts out at init time. The mode is what
        other clones of this repo read to decide whether to activate
        LFS — a single-source-of-truth signal that survives the
        ephemeral per-agent clones.
        """

        from .manifest import LFSConfig
        from ..distributed.ray_utils import serving

        try:
            tenant = serving.get_tenant_id() or "unspecified"
        except Exception:  # noqa: BLE001 — no execution context (detached / test)
            tenant = "unspecified"
        try:
            colony = serving.get_colony_id() or "unspecified"
        except Exception:  # noqa: BLE001
            colony = "unspecified"

        url = ""
        if self._agent is not None:
            params = getattr(self._agent.metadata, "parameters", None) or {}
            url = params.get(self._DESIGN_MONOREPO_URL_KEY, "") or ""

        return DesignMonorepoManifest(
            tenant=tenant,
            colony=colony,
            program=colony,
            target_system="unspecified",
            design_repo_url=url,
            lfs=LFSConfig(mode="same_remote" if enable_lfs else "disabled"),
        )

    def _identity_for_bootstrap(self, agent_email_domain: str) -> AgentIdentity:
        """Build an :class:`AgentIdentity` without going through
        :meth:`_manifest` (which requires a working
        :class:`DesignMonorepoClient`). Mirrors :meth:`_identity` but
        takes ``agent_email_domain`` as an explicit input so the
        bootstrap path can use a freshly-built manifest's value."""

        if self.is_detached:
            return AgentIdentity(
                agent_id=self.scope_id,
                role="external",
                colony_id="external",
                agent_email_domain=agent_email_domain,
            )
        agent = self.agent
        role = "agent"
        try:
            md = getattr(agent, "metadata", None)
            if md is not None and getattr(md, "role", None):
                role = str(md.role)
        except Exception:  # noqa: BLE001
            pass
        colony_id = getattr(agent, "colony_id", "default") or "default"
        return AgentIdentity(
            agent_id=agent.agent_id,
            role=role,
            colony_id=colony_id,
            agent_email_domain=agent_email_domain,
        )

    _GIT_ATTRIBUTION_KEY = "git_attribution"

    def _resolve_attribution(
        self, *, agent_email_domain: str,
    ) -> tuple[CommitIdentity, CommitIdentity | None]:
        """Read the colony's commit-attribution config from agent
        metadata and resolve principal + co-author into concrete
        :class:`CommitIdentity` pairs.

        Returns ``(principal, co_author_or_None)``. Falls back to the
        framework defaults — ``principal=colony, co_author=user`` —
        when metadata is absent (detached / test contexts) and to
        ``principal=colony`` (no co-author) when ``user`` was selected
        but no name/email is configured (rather than failing the
        commit; we'd rather lose the trailer than block the operation).
        """

        params: dict[str, Any] = {}
        if self._agent is not None:
            params = getattr(self._agent.metadata, "parameters", None) or {}
        cfg = params.get(self._GIT_ATTRIBUTION_KEY) or {}
        principal_label = cfg.get("commit_principal") or "colony"
        co_author_label = cfg.get("commit_co_author")
        user_name = cfg.get("git_user_name")
        user_email = cfg.get("git_user_email")

        agent_id: str | None = None
        role: str | None = None
        colony_id: str = "default"
        if self._agent is not None:
            agent_id = getattr(self._agent, "agent_id", None)
            colony_id = getattr(self._agent, "colony_id", "default") or "default"
            try:
                md = getattr(self._agent, "metadata", None)
                if md is not None and getattr(md, "role", None):
                    role = str(md.role)
            except Exception:  # noqa: BLE001
                pass

        def _safe_resolve(label: str | None) -> CommitIdentity | None:
            if not label:
                return None
            try:
                return resolve_commit_identity(
                    label,
                    colony_id=colony_id,
                    agent_id=agent_id,
                    role=role,
                    user_name=user_name,
                    user_email=user_email,
                    agent_email_domain=agent_email_domain,
                )
            except ValueError:
                # 'user' picked but name/email missing, or 'agent'
                # picked in detached mode. Skip the trailer rather
                # than blocking the commit — operator can fix the
                # config and re-run.
                logger.warning(
                    "Skipping attribution for label %r: "
                    "name/email or agent context missing.", label,
                )
                return None

        principal = _safe_resolve(principal_label)
        if principal is None:
            # Last-resort fallback: synthesise a colony identity so
            # the commit always succeeds. Without a principal we
            # can't author at all.
            principal = resolve_commit_identity(
                "colony",
                colony_id=colony_id,
                agent_email_domain=agent_email_domain,
            )
        co_author = _safe_resolve(co_author_label)
        return principal, co_author

    # ---- Auto-checkpoint on convergence quiescence -------------------

    @event_handler(pattern=ConvergenceQuiescenceProtocol.quiescence_pattern())
    async def _on_quiescence(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> EventProcessingResult | None:
        """Tag an ``auto_quiescence_<iso8601>`` checkpoint when the
        runtime settles with uncommitted changes. No-op when the
        working tree is clean — HEAD already represents the settled
        state and tagging again would just create a duplicate
        checkpoint pointing to the same SHA."""

        if not self._auto_checkpoint_on_quiescence:
            return None
        try:
            episode_id = ConvergenceQuiescenceProtocol.parse_quiescence_key(
                event.key,
            )
        except ValueError:
            return None
        try:
            client = await self._client_async()
        except Exception:  # noqa: BLE001
            logger.exception(
                "DesignCheckpointer: client unavailable; skipping "
                "auto-checkpoint for episode %s", episode_id,
            )
            return None
        try:
            dirty = await asyncio.to_thread(client.has_uncommitted_changes)
        except Exception:  # noqa: BLE001
            logger.exception(
                "DesignCheckpointer: dirty-state probe failed for "
                "episode %s", episode_id,
            )
            return None
        if not dirty:
            return None
        identity = self._identity()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = self.AUTO_CHECKPOINT_LABEL_FMT.format(timestamp=timestamp)
        rationale = f"convergence quiescence (episode {episode_id})"
        try:
            await asyncio.to_thread(
                _commit_all, client, identity,
                f"checkpoint: {label}",
            )
            checkpoint = await asyncio.to_thread(
                client.tag_checkpoint, identity, label, rationale,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "DesignCheckpointer: auto-checkpoint failed for "
                "episode %s", episode_id,
            )
            return None
        logger.info(
            "DesignCheckpointer: tagged auto checkpoint %s at "
            "episode %s", checkpoint.checkpoint_id, episode_id,
        )
        return EventProcessingResult(
            context_key=f"auto_checkpoint:{episode_id}",
            context={
                "episode_id": episode_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "label": label,
            },
        )

    # ---- Translate VCM page-graph rebuilds into a coarser branch event

    @event_handler(pattern=VCMEventProtocol.reindexed_pattern())
    async def _on_remote_change(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> EventProcessingResult | None:
        """Bridge a VCM ``reindexed:<scope_id>`` event to a
        ``DesignMonorepoEventProtocol.branch_changed:<scope_id>``
        event the agent's planner can react to (checkout / merge /
        rebase the per-agent local clone against the new upstream).

        VCM rebuilds the page graph after a ``GitRemoteWatcher``
        observes upstream commits, so this is the right point to fan
        out a higher-level branch-update signal without re-deriving
        the underlying watcher event from raw blackboard state.
        """

        try:
            scope_id = VCMEventProtocol.parse_reindexed_key(event.key)
        except ValueError:
            return None
        return EventProcessingResult(
            context_key=DesignMonorepoEventProtocol.branch_changed_key(scope_id),
            context={
                "scope_id": scope_id,
                "source_event_key": event.key,
            },
        )


# ---------------------------------------------------------------------------
# ToolBuilder — bootstrap a new tool into tools/<purpose>/<name>/
# ---------------------------------------------------------------------------


class ToolBuilder(_DesignMonorepoCapabilityBase):
    """Scaffold a new tool into the design monorepo's ``tools/``.

    Pairs with ``RepoStateProvider.find_existing_tool``: a tool-building
    pool's standard sequence is *find first, then bootstrap if no
    writable match*. The capability does only the scaffold + register
    step; the implementation, validation, benchmarking, and merge steps
    are the pool's own action policy.

    Pure action surface — passes ``input_patterns=[]`` for the same
    reason as :class:`RepoStateProvider` (no ``@event_handler``
    methods, must opt out of the wildcard fallback).
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        working_dir: Path | str | None = None,
        clone_scope_id: str | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            working_dir=working_dir,
            clone_scope_id=clone_scope_id,
            read_only=False,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"design_state", "git", "tool_building", "write"})

    @action_executor(
        planning_summary="Scaffold a new tool into tools/<purpose>/<name>/.",
    )
    async def bootstrap_repo(self, spec: RepoBootstrapSpec) -> BootstrapResult:
        """Scaffold a tool from one of the registered templates.

        For ``target='subdir_in_monorepo:tools/<purpose>/<name>'`` (the
        canonical case), the scaffold is rendered into the design
        monorepo's working tree, registered in
        ``.colony/tool-registry.json``, and committed under the agent's
        identity.

        For ``target='branch_in_existing:<branch>'`` and
        ``target='new_standalone:<git_url>'`` — the alternates rare
        enough that the doc reserves them — a clear ``NotImplementedError``
        is raised. They require coordination flows (branch creation,
        out-of-tree clone) that a future revision will add.
        """

        if not spec.target.startswith("subdir_in_monorepo:tools/"):
            raise NotImplementedError(
                f"target={spec.target!r} is not yet supported. "
                "Use 'subdir_in_monorepo:tools/<purpose>/<name>' for now.",
            )
        return await asyncio.to_thread(self._bootstrap_subdir, spec)

    def _bootstrap_subdir(self, spec: RepoBootstrapSpec) -> BootstrapResult:
        rel_path = spec.target.removeprefix("subdir_in_monorepo:")
        target_dir = self.working_dir / rel_path
        if target_dir.exists() and any(target_dir.iterdir()):
            raise DesignMonorepoError(
                f"Bootstrap target {target_dir} already contains files; "
                "augment-on-branch instead of bootstrap.",
            )

        files_created = scaffolds_module.render_template(
            spec.template,
            target_dir,
            name=spec.name,
            purpose=spec.purpose,
            license_id=spec.license,
            description=spec.description,
            template_vars=spec.template_vars,
            initial_files=spec.initial_files,
        )

        client = self._client_sync()
        identity = self._identity()
        commit_message = (
            f"bootstrap tool {spec.purpose}/{spec.name} (template={spec.template})"
        )
        rel_paths = [Path(rel_path) / f for f in files_created]
        sha = client.commit_with_identity(
            identity,
            commit_message,
            paths=rel_paths,
        )

        tool_entry = ToolEntry(
            name=spec.name,
            purpose=spec.purpose,
            capability=spec.capability,
            location=f"subdir:{rel_path}",
            license=spec.license,
            extra={
                "description": spec.description,
                "template": spec.template,
                "bootstrapped_at_sha": sha,
            },
        )
        registry_sha = client.register_tool(identity, tool_entry)

        return BootstrapResult(
            target=spec.target,
            relative_path=rel_path,
            sha=registry_sha,
            files_created=tuple(files_created),
            tool_entry=tool_entry,
        )


__all__ = (
    "RepoStateProvider",
    "DesignCheckpointer",
    "ToolBuilder",
)
