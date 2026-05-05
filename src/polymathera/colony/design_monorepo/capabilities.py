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
from .identity import AgentIdentity
from .manifest import DesignMonorepoManifest
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
        from ..utils.git.utils import inject_github_token

        self._working_dir.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(inject_github_token(url), str(self._working_dir))

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
