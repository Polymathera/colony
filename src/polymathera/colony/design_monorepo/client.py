"""``DesignMonorepoClient`` — the GitPython wrapper layer.

A thin, typed wrapper that exposes exactly the operations the framework
needs on a per-program design monorepo (master §8.1):

- ``clone_or_open(manifest, working_dir)`` — clone from the manifest's
  ``design_repo_url`` or open an existing local clone.
- ``commit_with_identity`` / ``tag_checkpoint`` / ``list_checkpoints``.
- ``fork`` / ``list_forks`` / ``restore_checkpoint``.
- ``merge_full`` / ``cherry_pick``.
- ``diff`` / ``current_state`` / ``get_branch_topology``.
- ``find_existing_tool`` / ``register_tool`` / ``list_recent_decisions``.
- ``setup_imported_remotes`` / ``install_merge_drivers``.

The wrapper holds no agent-specific state; identity is supplied per
operation via ``AgentIdentity`` so two agents working on the same clone
do not clobber each other's authorship.

The client is intentionally synchronous — git is a process-bound
operation and the GitPython API is sync. Callers in async contexts run
it in a worker thread (``asyncio.to_thread``); the capability layer in
``capabilities.py`` does this. Wrapping every method in a coroutine
here would mean two layers of executor offloading and no benefit.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .identity import AgentIdentity, CommitIdentity
from .manifest import DesignMonorepoManifest, MANIFEST_RELATIVE_PATH
from .models import (
    BranchNode,
    BranchTopology,
    Checkpoint,
    CommitInfo,
    DecisionEntry,
    DesignDiff,
    DesignDiffEntry,
    ForkBranch,
    ImportedRemote,
    RepoState,
    StashEntry,
    ToolEntry,
    ToolMatch,
    WorkingTreeStatus,
)
from . import registry as registry_module

if TYPE_CHECKING:
    from git import Repo


logger = logging.getLogger(__name__)


# A commit / tag / merge / cherry-pick / fork can be authored under
# either shape:
# - ``AgentIdentity`` — the per-agent transactional identity used by
#   framework-internal paths (``bootstrap_design_monorepo``, recovery
#   branches, tests).
# - ``CommitIdentity`` — produced by
#   ``_DesignMonorepoCapabilityBase._commit_attribution`` so the
#   colony's UI-configured ``commit_principal`` (colony / user /
#   agent / agent-type label) lands as author/committer.
# Both classes expose ``.actor()`` and ``.signing_key``, so every
# commit-producing client method that takes ``Identity`` reads only
# those two attributes — duck-typed across both shapes.
Identity = AgentIdentity | CommitIdentity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_TAG_PREFIX = "checkpoint/"
"""All framework-managed checkpoint tags carry this prefix (master §8.2)."""

FORK_BRANCH_PREFIX = "fork/"
SESSION_BRANCH_PREFIX = "session/"
AGENT_BRANCH_PREFIX = "agent/"
TOOL_BRANCH_PREFIX = "tool/"

DECISIONS_DIR = "design/decisions"
"""All decision JSON files live here (master §5.1, §8.3)."""

MAX_DIFF_BYTES = 256 * 1024
"""Cap on the raw unified-diff text returned by ``DesignMonorepoClient.diff``."""

_LOG_PATHS_CAP = 100
"""Per-commit cap on ``CommitInfo.paths_changed`` length. Single sweeping
commits (huge merges, mass renames) otherwise blow the planner's
context. The truncation is reflected in ``paths_changed_truncated``."""

_USER_REF_RESERVED_PREFIXES: tuple[str, ...] = (
    CHECKPOINT_TAG_PREFIX,
    FORK_BRANCH_PREFIX,
    SESSION_BRANCH_PREFIX,
    AGENT_BRANCH_PREFIX,
    TOOL_BRANCH_PREFIX,
)
"""Prefixes the framework manages on its own (checkpoint tags, fork
branches, per-session branches, per-agent branches, per-tool
branches). User-driven branch/tag actions refuse these — those go
through the framework's own helpers (``tag_checkpoint``, ``fork``,
``restore_checkpoint``)."""

_STASH_LINE_RE = re.compile(
    r"^stash@\{(?P<index>\d+)\}:\s*(?P<branch>[^:]*):\s*(?P<message>.*)$",
)


# Maps file patterns in `.gitattributes` to the merge-driver entry-point
# specified in `git_merge/`. The bootstrap flow writes these into
# `.gitattributes` (committed) and `install_merge_drivers` writes the
# `merge.<name>.driver` lines into the local `.git/config` (per-clone).
MERGE_DRIVER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("**/*.kg.json", "kg-merge"),
    ("**/decisions/*.json", "decisions-merge"),
    ("**/budgets/*.yaml", "budget-merge"),
    ("**/budgets/*.yml", "budget-merge"),
    ("**/page_graph.parquet", "page-graph-merge"),
    ("**/requirements/*.reqif", "reqif-merge"),
)

# Drivers are invoked as `python -m polymathera.colony.design_monorepo.git_merge.<name> %A %O %B %P`
# where %A=ours, %O=base, %B=theirs, %P=path (git's standard merge-driver
# placeholders). See the gitattributes(5) manpage and `git_merge/__init__.py`.
MERGE_DRIVER_ENTRY_POINTS: tuple[tuple[str, str], ...] = (
    ("kg-merge", "kg_merge"),
    ("decisions-merge", "decisions_merge"),
    ("budget-merge", "budget_merge"),
    ("page-graph-merge", "page_graph_merge"),
    ("reqif-merge", "reqif_merge"),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DesignMonorepoError(RuntimeError):
    """Base error for the wrapper layer."""


class CheckpointNotFoundError(DesignMonorepoError):
    """Raised when a referenced checkpoint tag does not exist."""


class UncommittedChangesError(DesignMonorepoError):
    """Raised when an operation requires a clean working tree but it isn't."""


class BranchExistsError(DesignMonorepoError):
    """Raised when ``fork(label)`` / ``restore_checkpoint(mode='fork')`` would
    overwrite an existing branch."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _per_repo_identity(
    repo: "Repo", identity: Identity,
) -> "Iterator[None]":
    """Temporarily set the per-repo git config a commit-producing op
    needs from ``identity``, and revert on exit.

    Sets:

    - ``user.name`` / ``user.email`` — read by ``git tag -m``,
      ``git merge``, ``git cherry-pick``, and by GitPython's
      ``commit-tree`` invocation when signing a ``Repo.index.commit``.
      Always set.
    - ``commit.gpgsign`` = ``true`` and ``user.signingkey`` —
      only when ``identity.signing_key`` is non-None. Lets the
      framework sign tags / merges / cherry-picks under the same
      identity-config wrap as ordinary commits.

    Prior values are restored on exit; entries that didn't exist
    before are unset, so the wrap leaves no residue across calls.
    """

    actor = identity.actor()
    signing_key = getattr(identity, "signing_key", None)

    def _get_prev(section: str, key: str) -> str | None:
        return repo.git.config(
            f"{section}.{key}", with_exceptions=False, get=True,
        ) or None

    def _restore(section: str, key: str, prev: str | None) -> None:
        if prev is not None:
            repo.git.config(f"{section}.{key}", prev)
        else:
            repo.git.config(
                "--unset", f"{section}.{key}", with_exceptions=False,
            )

    prev_name = _get_prev("user", "name")
    prev_email = _get_prev("user", "email")
    prev_gpgsign: str | None = None
    prev_signingkey: str | None = None

    repo.git.config("user.name", actor.name)
    repo.git.config("user.email", actor.email)
    if signing_key is not None:
        prev_gpgsign = _get_prev("commit", "gpgsign")
        prev_signingkey = _get_prev("user", "signingkey")
        repo.git.config("commit.gpgsign", "true")
        repo.git.config("user.signingkey", signing_key)
    try:
        yield
    finally:
        _restore("user", "name", prev_name)
        _restore("user", "email", prev_email)
        if signing_key is not None:
            _restore("commit", "gpgsign", prev_gpgsign)
            _restore("user", "signingkey", prev_signingkey)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DesignMonorepoClient:
    """Wrapper over a single local clone of the design monorepo."""

    def __init__(self, repo: "Repo", manifest: DesignMonorepoManifest) -> None:
        self._repo = repo
        self._manifest = manifest

    # -- Construction ----------------------------------------------------

    @classmethod
    def open(cls, working_dir: Path) -> DesignMonorepoClient:
        """Open an existing local clone.

        ``working_dir`` must be the repo root (i.e. ``.git/`` is a child).
        Raises ``DesignMonorepoError`` if the directory is not a git repo
        or the manifest is missing.
        """

        from git import InvalidGitRepositoryError, NoSuchPathError, Repo

        try:
            repo = Repo(str(working_dir))
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise DesignMonorepoError(
                f"{working_dir} is not a git repository.",
            ) from exc
        manifest = DesignMonorepoManifest.load_path(Path(working_dir))
        client = cls(repo, manifest)
        client.install_merge_drivers()
        return client

    @classmethod
    def clone_or_open(
        cls,
        manifest: DesignMonorepoManifest,
        working_dir: Path,
        *,
        depth: int | None = None,
        single_branch: bool = False,
    ) -> DesignMonorepoClient:
        """Clone from ``manifest.design_repo_url`` if needed, or open existing.

        The manifest committed inside the cloned repo wins over the one
        passed in: the in-repo manifest carries the authoritative
        ``imports_tooling_from`` etc. The ``manifest`` argument is used
        only for the URL and (initial) credentials reference.

        Authentication is delegated to git's standard machinery (SSH
        agent, HTTPS credential helper, OAuth token in the URL). The
        framework's secrets-store layer is responsible for materialising
        credentials before this is called.
        """

        from git import GitCommandError, Repo

        working_dir = Path(working_dir)
        if (working_dir / ".git").exists():
            return cls.open(working_dir)

        working_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {}
        if depth is not None:
            kwargs["depth"] = depth
        if single_branch:
            kwargs["single_branch"] = True
            kwargs["branch"] = manifest.default_branch
        from ..distributed.stores.git import _classify_git_clone_error, GitAuthError
        try:
            # Authentication flows through the system-level git
            # credential helper baked into the container image (see
            # ``Dockerfile.local``); the URL stays bare.
            repo = Repo.clone_from(
                manifest.design_repo_url,
                str(working_dir),
                **kwargs,
            )
        except GitCommandError as exc:
            classified = _classify_git_clone_error(exc)
            # Auth failures propagate untouched so callers higher up
            # the stack (e.g. the dashboard router) can map them to
            # an actionable HTTP response instead of a generic 5xx.
            if isinstance(classified, GitAuthError):
                raise classified from exc
            raise DesignMonorepoError(
                f"Failed to clone {manifest.design_repo_url} into "
                f"{working_dir}: {exc}",
            ) from exc

        # Reload the manifest from the freshly-cloned repo.
        in_repo_manifest = DesignMonorepoManifest.load_path(working_dir)
        client = cls(repo, in_repo_manifest)
        client.install_merge_drivers()
        client.setup_imported_remotes(fetch=True)
        return client

    # -- Identity bound to the wrapping context --------------------------

    @property
    def repo(self) -> Repo:
        return self._repo

    @property
    def manifest(self) -> DesignMonorepoManifest:
        return self._manifest

    @property
    def working_dir(self) -> Path:
        return Path(self._repo.working_tree_dir or self._repo.working_dir)

    @property
    def active_branch(self) -> str:
        """Name of the currently-checked-out branch (raises ``TypeError``
        if HEAD is detached — same shape as ``Repo.active_branch.name``).
        Capabilities key per-branch distributed state on this so a
        ``fork`` / ``checkout`` automatically isolates working state."""
        return self._repo.active_branch.name

    # -- Local config: merge drivers + LFS hooks -------------------------

    def install_merge_drivers(self) -> None:
        """Register the framework's merge drivers in this clone's ``.git/config``.

        The driver patterns are committed in ``.gitattributes`` (master
        §8.4); the per-clone driver *commands* are local config. Idempotent.
        """

        cw = self._repo.config_writer(config_level="repository")
        try:
            python = sys.executable or "python"
            for driver_name, module_name in MERGE_DRIVER_ENTRY_POINTS:
                cmd = (
                    f"{python} -m "
                    f"polymathera.colony.design_monorepo.git_merge.{module_name} "
                    "%A %O %B %P"
                )
                cw.set_value(f'merge "{driver_name}"', "name", driver_name)
                cw.set_value(f'merge "{driver_name}"', "driver", cmd)
        finally:
            cw.release()

    def setup_imported_remotes(self, *, fetch: bool = False) -> None:
        """Register every ``imports_tooling_from`` entry as a git remote.

        The framework adds them as standard read-only remotes; the
        wrapper layer never pushes to them. ``find_existing_tool`` reads
        the registry from a fetched ref of each remote.

        Idempotent — re-running after the manifest changes adds new
        remotes and updates URLs of existing ones.
        """

        from git import GitCommandError

        existing = {r.name: r for r in self._repo.remotes}
        for entry in self._manifest.imports_tooling_from:
            local_name = self._remote_name_for_import(entry)
            if local_name in existing:
                if existing[local_name].url != entry.url:
                    existing[local_name].set_url(entry.url)
            else:
                self._repo.create_remote(local_name, entry.url)
            if fetch:
                try:
                    self._repo.git.fetch(local_name, "--quiet")
                except GitCommandError as exc:  # pragma: no cover - depends on remote reachability
                    logger.warning(
                        "imports_tooling_from: failed to fetch %s (%s): %s",
                        local_name,
                        entry.url,
                        exc,
                    )

    @staticmethod
    def _remote_name_for_import(entry: ImportedRemote) -> str:
        if entry.name:
            return entry.name
        # Build a deterministic, git-safe remote name from the URL.
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", entry.url.rsplit("/", 1)[-1])
        slug = slug.removesuffix(".git").strip("_") or "imported"
        return f"imported_{slug}"

    # -- State queries ---------------------------------------------------

    def current_state(self) -> RepoState:
        repo = self._repo
        head_commit = repo.head.commit
        current_branch = (
            repo.active_branch.name if not repo.head.is_detached else f"detached:{head_commit.hexsha[:8]}"
        )

        ahead = behind = 0
        main = self._manifest.default_branch
        if main in [b.name for b in repo.heads] and current_branch != main:
            try:
                ahead_str, behind_str = repo.git.rev_list(
                    "--left-right", "--count", f"{current_branch}...{main}"
                ).split()
                ahead = int(ahead_str)
                behind = int(behind_str)
            except Exception:  # noqa: BLE001 - GitPython raises subclasses
                ahead = behind = 0

        is_fresh = self._is_fresh()
        checkpoints = self.list_checkpoints()
        forks = self.list_forks()
        last_q = checkpoints[-1].created_at if checkpoints else None
        try:
            local_tools = registry_module.load_registry(self.working_dir)
        except registry_module.ToolRegistryError:
            local_tools = ()

        return RepoState(
            is_fresh=is_fresh,
            current_branch=current_branch,
            current_sha=head_commit.hexsha,
            ahead_of_main_by=ahead,
            behind_main_by=behind,
            checkpoints=checkpoints,
            forks=forks,
            tools=local_tools,
            last_quiescence_at=last_q,
            uncommitted_changes=self.has_uncommitted_changes(),
            imported_tooling_remotes=self._manifest.imports_tooling_from,
        )

    def _is_fresh(self) -> bool:
        # Exactly one commit reachable from HEAD.
        repo = self._repo
        # ``--count HEAD`` is more efficient than materialising every commit.
        count = int(repo.git.rev_list("--count", "HEAD"))
        return count == 1

    def has_uncommitted_changes(self) -> bool:
        repo = self._repo
        return bool(repo.is_dirty(untracked_files=True))

    def get_branch_topology(self) -> BranchTopology:
        repo = self._repo
        main = self._manifest.default_branch
        if main not in [h.name for h in repo.heads]:
            # The local clone may not have main checked out; fall back to
            # remote-tracking ref if present.
            try:
                main_sha = repo.git.rev_parse(f"origin/{main}")
            except Exception:  # noqa: BLE001
                main_sha = repo.head.commit.hexsha
        else:
            main_sha = repo.heads[main].commit.hexsha

        nodes: list[BranchNode] = []
        for h in repo.heads:
            name = h.name
            head_sha = h.commit.hexsha
            parent = main if name != main else None
            try:
                merge_base = (
                    repo.git.merge_base(name, main).strip() if name != main else None
                )
            except Exception:  # noqa: BLE001
                merge_base = None
            nodes.append(
                BranchNode(
                    name=name,
                    head_sha=head_sha,
                    parent_branch=parent,
                    diverged_from_sha=merge_base,
                    is_fork=name.startswith(FORK_BRANCH_PREFIX),
                    is_session=name.startswith(SESSION_BRANCH_PREFIX),
                    is_agent=name.startswith(AGENT_BRANCH_PREFIX),
                    is_tool=name.startswith(TOOL_BRANCH_PREFIX),
                )
            )
        return BranchTopology(main=main, main_head_sha=main_sha, branches=tuple(nodes))

    def list_checkpoints(self) -> tuple[Checkpoint, ...]:
        results: list[Checkpoint] = []
        for tag in self._repo.tags:
            if not tag.name.startswith(CHECKPOINT_TAG_PREFIX):
                continue
            tag_obj = tag.tag  # may be None for lightweight tags
            commit = tag.commit
            if tag_obj is not None:
                created = datetime.fromtimestamp(tag_obj.tagged_date, timezone.utc)
                # The annotation message format is "<label>\n\n<rationale>".
                msg = (tag_obj.message or "").rstrip()
                if "\n\n" in msg:
                    label, rationale = msg.split("\n\n", 1)
                else:
                    label, rationale = (msg, "")
                tagger = (
                    f"{tag_obj.tagger.name} <{tag_obj.tagger.email}>"
                    if tag_obj.tagger is not None
                    else ""
                )
            else:
                created = datetime.fromtimestamp(commit.authored_date, timezone.utc)
                label = tag.name.removeprefix(CHECKPOINT_TAG_PREFIX)
                rationale = ""
                tagger = f"{commit.author.name} <{commit.author.email}>"
            results.append(
                Checkpoint(
                    checkpoint_id=tag.name,
                    sha=commit.hexsha,
                    label=label.strip() or tag.name,
                    rationale=rationale.strip(),
                    author=tagger,
                    created_at=created,
                )
            )
        results.sort(key=lambda c: c.created_at)
        return tuple(results)

    def list_forks(self) -> tuple[ForkBranch, ...]:
        results: list[ForkBranch] = []
        main = self._manifest.default_branch
        for h in self._repo.heads:
            if not h.name.startswith(FORK_BRANCH_PREFIX):
                continue
            try:
                merge_base = self._repo.git.merge_base(h.name, main).strip()
            except Exception:  # noqa: BLE001
                continue
            base_commit = self._repo.commit(merge_base)
            results.append(
                ForkBranch(
                    name=h.name,
                    head_sha=h.commit.hexsha,
                    diverged_from_sha=merge_base,
                    diverged_from_branch=main,
                    created_at=datetime.fromtimestamp(
                        base_commit.authored_date, timezone.utc
                    ),
                )
            )
        results.sort(key=lambda f: f.created_at)
        return tuple(results)

    # -- Decisions -------------------------------------------------------

    def list_recent_decisions(
        self,
        *,
        since: str | None = None,
        limit: int = 50,
    ) -> tuple[DecisionEntry, ...]:
        """Walk ``design/decisions/`` for decisions made since a baseline.

        ``since`` is a commit SHA, tag, or branch name; ``None`` means
        the entire history of the current branch. Returns at most
        ``limit`` entries, ordered by authored-time ascending.
        """

        decisions_path = self.working_dir / DECISIONS_DIR
        if not decisions_path.is_dir():
            return ()

        # Build (decision_id -> DecisionEntry) by walking commits in the
        # range; later commits overwrite earlier ones for the same id.
        rev_range = "HEAD" if since is None else f"{since}..HEAD"
        try:
            commits = list(self._repo.iter_commits(rev_range, paths=DECISIONS_DIR))
        except Exception as exc:  # noqa: BLE001 - GitPython raises subclasses
            raise DesignMonorepoError(
                f"Could not walk decisions for {rev_range}: {exc}",
            ) from exc

        seen: dict[str, DecisionEntry] = {}
        for commit in commits:
            for blob_path in self._files_changed_in(commit):
                if not blob_path.startswith(DECISIONS_DIR + "/"):
                    continue
                if not blob_path.endswith(".json"):
                    continue
                # Read the file's content *as of this commit* — preserves
                # the decision body that the commit introduced.
                try:
                    blob_content = self._repo.git.show(f"{commit.hexsha}:{blob_path}")
                except Exception:  # noqa: BLE001
                    continue
                try:
                    data = json.loads(blob_content)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                decision_id = (
                    data.get("decision_id")
                    or Path(blob_path).stem
                )
                entry = DecisionEntry(
                    decision_id=str(decision_id),
                    sha=commit.hexsha,
                    title=str(data.get("title", Path(blob_path).stem)),
                    summary=str(data.get("summary", "")),
                    authored_at=datetime.fromtimestamp(
                        commit.authored_date, timezone.utc
                    ),
                    author=f"{commit.author.name} <{commit.author.email}>",
                    rationale=str(data.get("rationale", "")),
                    relative_path=blob_path,
                )
                seen[entry.decision_id] = entry
        ordered = sorted(seen.values(), key=lambda d: d.authored_at)
        return tuple(ordered[-limit:])

    def _files_changed_in(self, commit) -> Iterable[str]:
        # ``commit.stats.files`` returns a dict of path -> stats.
        return commit.stats.files.keys()

    # -- Tool registry / find ---------------------------------------------

    def load_local_registry(self) -> tuple[ToolEntry, ...]:
        try:
            return registry_module.load_registry(self.working_dir)
        except registry_module.ToolRegistryError as exc:
            raise DesignMonorepoError(str(exc)) from exc

    def load_imported_registries(self) -> tuple[ToolEntry, ...]:
        """Best-effort read of every imports_tooling_from remote's registry.

        Each remote is expected to be a Colony design monorepo with the
        same ``.colony/tool-registry.json`` schema. We read the registry
        out of the remote's fetched ref using ``git show``; any failure
        on a single remote is logged and skipped (one bad remote should
        not block the search).
        """

        results: list[ToolEntry] = []
        for entry in self._manifest.imports_tooling_from:
            ref = self._resolve_imported_ref(entry)
            if ref is None:
                continue
            try:
                payload = self._repo.git.show(f"{ref}:.colony/tool-registry.json")
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "imports_tooling_from(%s): no .colony/tool-registry.json at %s (%s)",
                    entry.url,
                    ref,
                    exc,
                )
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning(
                    "imports_tooling_from(%s): malformed registry at %s",
                    entry.url,
                    ref,
                )
                continue
            for raw in data.get("tools", []):
                try:
                    tool = ToolEntry.model_validate(raw)
                except Exception:  # noqa: BLE001 - pydantic ValidationError
                    continue
                # Rewrite location to mark the entry as remote-imported.
                results.append(
                    tool.model_copy(
                        update={
                            "location": f"remote:{self._remote_name_for_import(entry)}:"
                            f"tools/{tool.purpose}/{tool.name}",
                        }
                    )
                )
        return tuple(results)

    def _resolve_imported_ref(self, entry: ImportedRemote) -> str | None:
        """Resolve the entry's ``ref`` to a fully-qualified ref."""
        remote_name = self._remote_name_for_import(entry)
        ref = entry.ref
        if ref.startswith("tag:"):
            return f"refs/tags/{ref.removeprefix('tag:')}"
        # Default: assume a branch on the imported remote.
        return f"refs/remotes/{remote_name}/{ref}"

    def find_existing_tool(
        self,
        capability_query: str,
        *,
        require_writable: bool = False,
    ) -> tuple[ToolMatch, ...]:
        local = self.load_local_registry()
        remote = () if require_writable else self.load_imported_registries()
        return registry_module.search(
            capability_query,
            local_entries=local,
            remote_entries=remote,
            require_writable=require_writable,
        )

    def register_tool(
        self,
        identity: Identity,
        entry: ToolEntry,
        *,
        commit_message: str | None = None,
    ) -> str:
        """Upsert a tool entry and commit the registry change."""

        registry_module.upsert_tool(self.working_dir, entry)
        message = commit_message or f"register tool {entry.purpose}/{entry.name} ({entry.capability})"
        return self.commit_with_identity(
            identity,
            message,
            paths=[Path(registry_module.REGISTRY_RELATIVE_PATH)],
        )

    # -- Mutating operations ---------------------------------------------

    def commit_with_identity(
        self,
        identity: Identity,
        message: str,
        *,
        paths: Sequence[Path] | None = None,
        all_changes: bool = False,
    ) -> str:
        """Stage ``paths`` (or everything if ``all_changes``) and commit.

        Returns the new commit's SHA. Per master §8.5, the commit is
        authored and committed under the agent's transactional identity;
        the global git config is *not* mutated.
        """

        repo = self._repo
        if paths is None and not all_changes:
            raise ValueError("Either provide paths or set all_changes=True.")

        if all_changes:
            repo.git.add("-A")
        else:
            for p in paths or ():
                repo.git.add("--", str(p))

        if not repo.is_dirty(index=True, working_tree=False, untracked_files=False):
            # Nothing staged — return the current HEAD rather than make
            # an empty commit.
            return repo.head.commit.hexsha

        actor = identity.actor()
        with _per_repo_identity(repo, identity):
            commit = repo.index.commit(
                message,
                author=actor,
                committer=actor,
            )
        return commit.hexsha

    def tag_checkpoint(
        self,
        identity: Identity,
        label: str,
        rationale: str = "",
        *,
        sha: str | None = None,
    ) -> Checkpoint:
        """Create a ``checkpoint/<iso8601>-<short_sha>`` annotated tag.

        Caller is responsible for ensuring quiescence + cleanliness
        before calling. ``label`` and ``rationale`` are recorded in the
        tag annotation as ``"<label>\\n\\n<rationale>"``; the format is
        what ``list_checkpoints`` parses on read.

        Returns the new ``Checkpoint`` model.
        """

        repo = self._repo
        target_sha = sha or repo.head.commit.hexsha
        short = target_sha[:8]
        # ISO-8601 with no colons, since git refs forbid `:`.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag_name = f"{CHECKPOINT_TAG_PREFIX}{timestamp}-{short}"
        if any(t.name == tag_name for t in repo.tags):
            # Two checkpoints in the same second — disambiguate with a
            # short-lived suffix rather than failing.
            tag_name = f"{tag_name}-{label[:8].replace('/', '_') or 'extra'}"

        annotation = label if not rationale else f"{label}\n\n{rationale}"

        # Per-commit identity for the tagger as well.
        actor = identity.actor()
        with _per_repo_identity(repo, identity):
            tag = repo.create_tag(tag_name, ref=target_sha, message=annotation)

        # Append to the human-readable checkpoint log.
        log_path = self.working_dir / ".colony" / "checkpoints.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "tag": tag_name,
                        "sha": target_sha,
                        "label": label,
                        "rationale": rationale,
                        "tagger": actor.name,
                        "tagged_at": timestamp,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        # Commit the log update so the audit trail rides with the repo.
        self.commit_with_identity(
            identity,
            f"checkpoint log: {label}",
            paths=[log_path.relative_to(self.working_dir)],
        )

        return Checkpoint(
            checkpoint_id=tag.name,
            sha=target_sha,
            label=label,
            rationale=rationale,
            author=f"{actor.name} <{actor.email}>",
            created_at=datetime.now(timezone.utc),
        )

    def fork(
        self,
        identity: Identity,
        label: str,
        *,
        from_sha: str | None = None,
        checkout: bool = True,
    ) -> ForkBranch:
        """Create a ``fork/<label>`` branch off ``from_sha`` (default: HEAD)."""

        repo = self._repo
        branch_name = f"{FORK_BRANCH_PREFIX}{label}"
        if branch_name in [h.name for h in repo.heads]:
            raise BranchExistsError(f"Branch {branch_name} already exists.")
        target = from_sha or repo.head.commit.hexsha
        new_branch = repo.create_head(branch_name, commit=target)
        if checkout:
            new_branch.checkout()
        merge_base = repo.git.merge_base(branch_name, self._manifest.default_branch).strip()
        base_commit = repo.commit(merge_base)
        return ForkBranch(
            name=branch_name,
            head_sha=new_branch.commit.hexsha,
            diverged_from_sha=merge_base,
            diverged_from_branch=self._manifest.default_branch,
            created_at=datetime.fromtimestamp(base_commit.authored_date, timezone.utc),
        )

    def restore_checkpoint(
        self,
        identity: Identity,
        checkpoint_id: str,
        *,
        mode: str = "replace",
        recovery_label: str | None = None,
    ) -> str:
        """Restore the working tree to a checkpoint.

        - ``mode='replace'`` checks out the tagged commit on the current
          branch (using ``git checkout``). The caller MUST have first
          confirmed with the user.
        - ``mode='fork'`` creates ``fork/<recovery_label>`` off the tag.
          Returns the new branch name.

        On either mode, if the working tree is dirty, the dirty state
        is stashed onto a ``recovery/<timestamp>`` branch first (so it
        is never silently lost — master §8.6).
        """

        if mode not in ("replace", "fork"):
            raise ValueError("mode must be 'replace' or 'fork'.")
        repo = self._repo
        if not any(t.name == checkpoint_id for t in repo.tags):
            raise CheckpointNotFoundError(
                f"No tag named {checkpoint_id} in this repo.",
            )

        # Auto-stash dirty state to a recovery/* branch so nothing is lost.
        if self.has_uncommitted_changes():
            recovery_branch = self._auto_stash_recovery(identity)
            logger.info(
                "restore_checkpoint(%s): stashed dirty working tree to %s",
                checkpoint_id,
                recovery_branch,
            )

        if mode == "fork":
            label = recovery_label or f"from-{checkpoint_id.split('/', 1)[-1]}"
            branch_name = f"{FORK_BRANCH_PREFIX}{label}"
            if branch_name in [h.name for h in repo.heads]:
                raise BranchExistsError(f"Branch {branch_name} already exists.")
            new_branch = repo.create_head(branch_name, commit=checkpoint_id)
            new_branch.checkout()
            return branch_name

        repo.git.checkout(checkpoint_id)
        return checkpoint_id

    def _auto_stash_recovery(self, identity: Identity) -> str:
        repo = self._repo
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        branch_name = f"recovery/{timestamp}"
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        try:
            self.commit_with_identity(
                identity,
                f"recovery snapshot: {timestamp}",
                all_changes=True,
            )
        except Exception:  # noqa: BLE001
            # Fall through — even if there was nothing to commit, the
            # branch creation is the audit trail we wanted.
            pass
        return branch_name

    def merge_full(
        self,
        identity: Identity,
        source_branch: str,
        *,
        target_branch: str | None = None,
        message: str | None = None,
    ) -> str:
        """Standard merge of ``source_branch`` into ``target_branch``.

        Conflicts surface as a ``DesignMonorepoError`` after the working
        tree is left in the conflict state, exactly as ``git merge``
        does — the caller (or the user) resolves the conflict, then
        commits manually via ``commit_with_identity``.
        """

        from git import GitCommandError

        repo = self._repo
        target = target_branch or repo.active_branch.name
        if target != repo.active_branch.name:
            repo.git.checkout(target)
        args = ["--no-ff"]
        if message is not None:
            args.extend(["-m", message])
        with _per_repo_identity(repo, identity):
            try:
                repo.git.merge(source_branch, *args)
            except GitCommandError as exc:
                raise DesignMonorepoError(
                    f"Merge of {source_branch} into {target} failed: {exc}",
                ) from exc
        return repo.head.commit.hexsha

    def cherry_pick(
        self,
        identity: Identity,
        commit_shas: Sequence[str],
        *,
        target_branch: str | None = None,
    ) -> tuple[str, ...]:
        """Cherry-pick a sequence of commits onto ``target_branch``.

        Used by selective merge (master §8.3): the LLM-assisted resolver
        maps decision IDs to their owning commits; we replay them onto
        the target branch under the calling agent's identity.
        """

        from git import GitCommandError

        repo = self._repo
        if target_branch and target_branch != repo.active_branch.name:
            repo.git.checkout(target_branch)
        new_shas: list[str] = []
        with _per_repo_identity(repo, identity):
            for sha in commit_shas:
                try:
                    repo.git.cherry_pick(sha)
                except GitCommandError as exc:
                    raise DesignMonorepoError(
                        f"cherry-pick {sha} failed: {exc}",
                    ) from exc
                new_shas.append(repo.head.commit.hexsha)
        return tuple(new_shas)

    # -- Diffs -----------------------------------------------------------

    def diff(
        self,
        ref_a: str,
        ref_b: str,
        *,
        max_bytes: int = MAX_DIFF_BYTES,
    ) -> DesignDiff:
        """Return ``DesignDiff(ref_a, ref_b)`` with per-file entries + raw text.

        ``raw_unified_diff`` is capped at ``max_bytes`` and is empty
        beyond that — the entries summary still applies.
        """

        repo = self._repo
        # Per-file summary using --numstat + --name-status.
        numstat = repo.git.diff(ref_a, ref_b, "--numstat", "-z")
        name_status = repo.git.diff(ref_a, ref_b, "--name-status", "-z")
        entries = self._parse_diff(numstat, name_status)

        try:
            raw = repo.git.diff(ref_a, ref_b, no_color=True)
            if len(raw.encode("utf-8", errors="replace")) > max_bytes:
                raw = ""
        except Exception:  # noqa: BLE001
            raw = ""
        return DesignDiff(
            ref_a=ref_a,
            ref_b=ref_b,
            entries=entries,
            raw_unified_diff=raw,
        )

    @staticmethod
    def _parse_diff(numstat_z: str, name_status_z: str) -> tuple[DesignDiffEntry, ...]:
        """Parse ``git diff -z --numstat`` + ``-z --name-status`` outputs.

        ``-z`` separates fields with NUL. For numstat, each entry is
        ``<insertions>\\t<deletions>\\t<path>`` separated by NUL. For
        name-status, rename/copy entries spread across multiple NULs.
        """

        # numstat: simple parse
        ns_records: dict[str, tuple[int, int]] = {}
        for raw in numstat_z.split("\0"):
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) < 3:
                continue
            ins_str, del_str = parts[0], parts[1]
            path = "\t".join(parts[2:])
            try:
                ins = int(ins_str) if ins_str != "-" else 0
                dele = int(del_str) if del_str != "-" else 0
            except ValueError:
                ins, dele = 0, 0
            ns_records[path] = (ins, dele)

        # name-status: walk fields handling renames (R<score>) / copies (C<score>)
        tokens = [t for t in name_status_z.split("\0") if t]
        results: list[DesignDiffEntry] = []
        i = 0
        while i < len(tokens):
            status = tokens[i]
            i += 1
            if not status:
                continue
            kind_letter = status[0]
            if kind_letter in ("R", "C"):
                if i + 1 >= len(tokens):
                    break
                old_path = tokens[i]
                new_path = tokens[i + 1]
                i += 2
                ins, dele = ns_records.get(new_path, (0, 0))
                ct = "renamed" if kind_letter == "R" else "copied"
                results.append(
                    DesignDiffEntry(
                        path=new_path,
                        change_type=ct,
                        old_path=old_path,
                        insertions=ins,
                        deletions=dele,
                    )
                )
            else:
                if i >= len(tokens):
                    break
                path = tokens[i]
                i += 1
                if kind_letter == "A":
                    ct = "added"
                elif kind_letter == "M":
                    ct = "modified"
                elif kind_letter == "D":
                    ct = "deleted"
                else:
                    ct = "modified"
                ins, dele = ns_records.get(path, (0, 0))
                results.append(
                    DesignDiffEntry(
                        path=path,
                        change_type=ct,
                        insertions=ins,
                        deletions=dele,
                    )
                )
        return tuple(results)

    def diff_against_checkpoint(self, checkpoint_id: str) -> DesignDiff:
        if not any(t.name == checkpoint_id for t in self._repo.tags):
            raise CheckpointNotFoundError(
                f"No tag named {checkpoint_id} in this repo.",
            )
        head_sha = self._repo.head.commit.hexsha
        return self.diff(checkpoint_id, head_sha)

    # -- Push / fetch ----------------------------------------------------

    def fetch(self, remote: str = "origin") -> None:
        from git import GitCommandError

        try:
            self._repo.git.fetch(remote, "--quiet")
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to fetch from {remote}: {exc}",
            ) from exc

    def push(
        self,
        *,
        branch: str | None = None,
        remote: str = "origin",
        with_tags: bool = False,
    ) -> None:
        from git import GitCommandError

        args: list[str] = []
        if with_tags:
            args.append("--tags")
        try:
            ref = branch or self._repo.active_branch.name
            self._repo.git.push(remote, ref, *args)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to push to {remote}: {exc}",
            ) from exc

    # -- History / status / working-tree diff ----------------------------

    def log(
        self,
        *,
        paths: Sequence[str] | None = None,
        limit: int = 20,
        ref: str = "HEAD",
    ) -> tuple[CommitInfo, ...]:
        """Return up to ``limit`` commits reachable from ``ref``.

        When ``paths`` is set, restrict to commits that touched any of
        them (``git log -- <paths>`` semantics). Per-commit
        ``paths_changed`` is capped at ``_LOG_PATHS_CAP`` entries so a
        single sweeping merge commit can't blow the planner's token
        budget; ``paths_changed_truncated`` flags when capped.
        """

        if limit <= 0:
            return ()

        iter_kwargs: dict[str, object] = {"max_count": int(limit)}
        if paths:
            iter_kwargs["paths"] = list(paths)
        rows: list[CommitInfo] = []
        for commit in self._repo.iter_commits(ref, **iter_kwargs):
            changed = list(commit.stats.files.keys())
            truncated = len(changed) > _LOG_PATHS_CAP
            if truncated:
                changed = changed[:_LOG_PATHS_CAP]
            rows.append(
                CommitInfo(
                    sha=commit.hexsha,
                    author=str(commit.author),
                    committed_at=datetime.fromtimestamp(
                        commit.committed_date, tz=timezone.utc,
                    ),
                    message=str(commit.message).rstrip(),
                    paths_changed=tuple(changed),
                    paths_changed_truncated=truncated,
                ),
            )
        return tuple(rows)

    def status(self) -> WorkingTreeStatus:
        """Three-way split of ``git status``: staged, unstaged, untracked.

        Lighter than parsing ``--porcelain`` by hand — GitPython exposes
        each bucket directly. Paths are repo-root-relative
        forward-slash strings.
        """

        index = self._repo.index
        # ``diff(None)`` is index-vs-working-tree → unstaged paths.
        # ``diff("HEAD")`` is HEAD-vs-index → staged paths.
        try:
            staged_paths = {d.a_path or d.b_path for d in index.diff("HEAD")}
        except Exception:  # noqa: BLE001 — empty repo / no HEAD yet
            staged_paths = set()
        try:
            unstaged_paths = {d.a_path or d.b_path for d in index.diff(None)}
        except Exception:  # noqa: BLE001
            unstaged_paths = set()
        untracked = tuple(sorted(self._repo.untracked_files))
        # Filter out None just in case GitPython emits an unset path.
        return WorkingTreeStatus(
            staged=tuple(sorted(p for p in staged_paths if p)),
            unstaged=tuple(sorted(p for p in unstaged_paths if p)),
            untracked=untracked,
        )

    def diff_working_tree(
        self,
        *,
        paths: Sequence[str] | None = None,
        max_bytes: int = MAX_DIFF_BYTES,
    ) -> str:
        """Return the working-tree-vs-HEAD raw unified diff.

        Includes uncommitted changes (both staged and unstaged) — the
        symmetric "what would I commit if I ran ``commit -a``" view.
        Capped at ``max_bytes``; oversized diffs return the prefix
        with a trailing truncation marker.
        """

        from git import GitCommandError

        args: list[str] = ["HEAD"]
        if paths:
            args.append("--")
            args.extend(paths)
        try:
            text = self._repo.git.diff(*args)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to diff working tree: {exc}",
            ) from exc
        if len(text) > max_bytes:
            return (
                text[:max_bytes]
                + f"\n... (truncated at {max_bytes} bytes)\n"
            )
        return text

    # -- Branch ops ------------------------------------------------------

    def create_branch(
        self, name: str, *, base: str | None = None,
    ) -> str:
        """Create a new branch off ``base`` (default: current HEAD).

        Rejects names beginning with framework-reserved prefixes
        (``checkpoint/``, ``fork/``, ``session/``, ``agent/``,
        ``tool/``); those use the dedicated framework helpers
        (``tag_checkpoint``, ``fork``) so the branch namespace stays
        partitioned.
        """

        _validate_user_ref_name(name, kind="branch")
        from git import GitCommandError

        target = base or "HEAD"
        try:
            self._repo.git.branch(name, target)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to create branch {name!r} from {target!r}: {exc}",
            ) from exc
        return name

    def delete_branch(self, name: str, *, force: bool = False) -> None:
        """Delete a non-reserved branch.

        Refuses framework-reserved prefixes and refuses to delete the
        current HEAD.
        """

        _validate_user_ref_name(name, kind="branch")
        if self._repo.head.is_valid() and self._repo.active_branch.name == name:
            raise DesignMonorepoError(
                f"Cannot delete branch {name!r}: it is the current HEAD",
            )
        from git import GitCommandError

        try:
            self._repo.git.branch("-D" if force else "-d", name)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to delete branch {name!r}: {exc}",
            ) from exc

    def checkout_branch(self, name: str, *, create: bool = False) -> str:
        """Switch HEAD to ``name``. With ``create=True``, creates the
        branch off the current HEAD first.

        Refuses framework-reserved prefixes — those have dedicated
        helpers (``fork``, ``restore_checkpoint``) so the convention
        stays a single-writer thing.
        """

        _validate_user_ref_name(name, kind="branch")
        if self.has_uncommitted_changes():
            raise UncommittedChangesError(
                f"Cannot checkout {name!r}: working tree has uncommitted "
                "changes; commit or stash first.",
            )
        from git import GitCommandError

        try:
            if create:
                self._repo.git.checkout("-b", name)
            else:
                self._repo.git.checkout(name)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to checkout {name!r}: {exc}",
            ) from exc
        return name

    # -- Stash ops -------------------------------------------------------

    def stash_save(self, message: str = "") -> bool:
        """Save the working tree to a new stash. Returns ``True`` when
        something was stashed, ``False`` when the tree was already
        clean."""

        from git import GitCommandError

        args = ["push"]
        if message:
            args.extend(["-m", message])
        try:
            output = self._repo.git.stash(*args)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to stash: {exc}",
            ) from exc
        # ``git stash push`` prints "No local changes to save" on a
        # clean tree and exits 0 — disambiguate from a real stash here
        # so the action layer can surface it cleanly.
        return "No local changes" not in (output or "")

    def stash_pop(self) -> None:
        """Pop the most recent stash onto the current working tree."""

        from git import GitCommandError

        try:
            self._repo.git.stash("pop")
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Failed to pop stash: {exc}",
            ) from exc

    def list_stashes(self) -> tuple[StashEntry, ...]:
        """Return the current stash stack, most-recent first."""

        try:
            text = self._repo.git.stash("list")
        except Exception:  # noqa: BLE001
            return ()
        rows: list[StashEntry] = []
        # Each line: ``stash@{N}: <branch info>: <message>``.
        for line in (text or "").splitlines():
            m = _STASH_LINE_RE.match(line)
            if not m:
                continue
            rows.append(
                StashEntry(
                    index=int(m.group("index")),
                    message=m.group("message").strip(),
                    branch=m.group("branch").strip(),
                ),
            )
        return tuple(rows)

    # -- Rebase ----------------------------------------------------------

    def rebase_onto(self, target_ref: str) -> None:
        """Non-interactive ``git rebase <target_ref>`` on the current branch.

        Refuses to run with uncommitted changes (caller must stash or
        commit first). Conflict-mid-rebase leaves the repo in the
        standard git rebasing state for the operator to resolve.
        """

        if self.has_uncommitted_changes():
            raise UncommittedChangesError(
                "Cannot rebase: working tree has uncommitted changes; "
                "commit or stash first.",
            )
        from git import GitCommandError

        try:
            self._repo.git.rebase(target_ref)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Rebase onto {target_ref!r} failed: {exc}",
            ) from exc

    # -- Pull (fetch + ff-only / merge / rebase) -------------------------

    def pull(
        self,
        *,
        remote: str = "origin",
        branch: str | None = None,
        strategy: str = "ff_only",
    ) -> str:
        """Fetch from ``remote`` and integrate ``branch`` (default:
        current branch) into the working tree via ``strategy``:

        - ``"ff_only"`` (default): refuse non-fast-forward.
        - ``"merge"``: merge commit.
        - ``"rebase"``: replay local commits on top of the remote head.

        Returns the new HEAD SHA. Refuses to run with uncommitted
        changes so a conflict doesn't strand the caller mid-merge.
        """

        if strategy not in {"ff_only", "merge", "rebase"}:
            raise DesignMonorepoError(
                f"pull: unknown strategy {strategy!r} (expected "
                f"ff_only / merge / rebase)",
            )
        if self.has_uncommitted_changes():
            raise UncommittedChangesError(
                "Cannot pull: working tree has uncommitted changes; "
                "commit or stash first.",
            )
        from git import GitCommandError

        ref = branch or (
            self._repo.active_branch.name
            if self._repo.head.is_valid()
            else "main"
        )
        flag = {
            "ff_only": "--ff-only",
            "merge": "--no-rebase",
            "rebase": "--rebase",
        }[strategy]
        try:
            self._repo.git.pull(flag, remote, ref)
        except GitCommandError as exc:
            raise DesignMonorepoError(
                f"Pull from {remote}/{ref} ({strategy}) failed: {exc}",
            ) from exc
        return self._repo.head.commit.hexsha

    # -- Tag ops ---------------------------------------------------------

    def create_tag(
        self,
        identity: Identity,
        name: str,
        *,
        message: str = "",
    ) -> str:
        """Create a tag at the current HEAD.

        ``git tag -m`` (annotated tag) reads the committer from
        per-repo git config — wrapped in :func:`_per_repo_identity`
        for the same reason :meth:`tag_checkpoint`,
        :meth:`merge_full`, and :meth:`cherry_pick` are wrapped.
        Refuses the ``checkpoint/`` prefix — those go through
        :meth:`tag_checkpoint` so the convention stays a
        single-writer thing.
        """

        _validate_user_ref_name(name, kind="tag")
        from git import GitCommandError

        try:
            with _per_repo_identity(self._repo, identity):
                if message:
                    self._repo.create_tag(name, message=message)
                else:
                    self._repo.create_tag(name)
        except (GitCommandError, Exception) as exc:  # noqa: BLE001
            raise DesignMonorepoError(
                f"Failed to create tag {name!r}: {exc}",
            ) from exc
        return name

    def delete_tag(self, name: str) -> None:
        _validate_user_ref_name(name, kind="tag")
        from git import GitCommandError

        try:
            self._repo.delete_tag(name)
        except (GitCommandError, Exception) as exc:  # noqa: BLE001
            raise DesignMonorepoError(
                f"Failed to delete tag {name!r}: {exc}",
            ) from exc


def _validate_user_ref_name(name: str, *, kind: str) -> None:
    """Refuse empty names + framework-reserved prefixes.

    ``kind`` is ``"branch"`` / ``"tag"`` — purely for the error
    message. Used by every user-driven branch/tag-mutating client
    method so the framework's namespace conventions
    (:data:`_USER_REF_RESERVED_PREFIXES`) stay enforced.
    """

    if not name:
        raise DesignMonorepoError(f"{kind} name must be non-empty")
    for prefix in _USER_REF_RESERVED_PREFIXES:
        if name.startswith(prefix):
            raise DesignMonorepoError(
                f"{kind} name {name!r} starts with reserved prefix "
                f"{prefix!r}; use the framework helper instead "
                f"(tag_checkpoint / fork / etc.)",
            )


__all__ = (
    "DesignMonorepoClient",
    "DesignMonorepoError",
    "CheckpointNotFoundError",
    "UncommittedChangesError",
    "BranchExistsError",
    "CHECKPOINT_TAG_PREFIX",
    "FORK_BRANCH_PREFIX",
    "SESSION_BRANCH_PREFIX",
    "AGENT_BRANCH_PREFIX",
    "TOOL_BRANCH_PREFIX",
    "DECISIONS_DIR",
    "MAX_DIFF_BYTES",
    "MERGE_DRIVER_PATTERNS",
    "MERGE_DRIVER_ENTRY_POINTS",
)
