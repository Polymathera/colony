"""Unit tests for the lazy-clone path in
:class:`_DesignMonorepoCapabilityBase`.

The capability reads the URL from
``agent.metadata.parameters[design_monorepo_url]`` and clones into the
per-agent ``working_dir`` on first ``_client_sync``. We exercise the
clone with a hand-built file:// remote so the tests do not touch the
network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import git
import pytest

from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)


def _bootstrap_remote(remote_root: Path) -> Path:
    """Create a tiny on-disk git repo that doubles as the ``origin_url``.

    The capability calls ``DesignMonorepoClient.open(working_dir)``
    after cloning, which needs ``.colony/manifest.json`` to be present.
    We bake one into the remote so the open succeeds.
    """

    repo = git.Repo.init(remote_root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    (remote_root / ".colony").mkdir()
    (remote_root / ".colony" / "manifest.json").write_text(
        '{"schema_version": 1, "tenant": "t", "colony": "c", '
        '"program": "p", "target_system": "ts", "topology": "external", '
        '"default_branch": "main", "design_repo_url": "file://placeholder"}\n',
        encoding="utf-8",
    )
    repo.git.add(all=True)
    repo.index.commit("initial")
    return remote_root


def _make_writer_capability(
    *, agent_id: str, working_dir: Path, design_monorepo_url: str | None,
):
    """Build a ``DesignCheckpointer`` configured to lazy-clone from the
    URL on the agent's metadata. Returns inside an execution context
    because :class:`AgentCapability` captures one in ``__init__``."""

    from polymathera.colony.design_monorepo.capabilities import DesignCheckpointer

    agent = MagicMock()
    agent.agent_id = agent_id
    agent.metadata.parameters = (
        {"design_monorepo_url": design_monorepo_url}
        if design_monorepo_url is not None else {}
    )
    return DesignCheckpointer(
        agent=agent,
        scope_id="dm",
        working_dir=working_dir,
        auto_checkpoint_on_quiescence=False,
    )


def test_lazy_clone_from_agent_metadata(tmp_path: Path) -> None:
    remote_root = _bootstrap_remote(tmp_path / "remote")
    working_dir = tmp_path / "agent_clone"
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_writer_capability(
            agent_id="agent-A",
            working_dir=working_dir,
            design_monorepo_url=f"file://{remote_root}",
        )
        client = cap._client_sync()
    assert (working_dir / ".git").is_dir()
    assert client is not None
    assert (working_dir / ".colony" / "manifest.json").is_file()


def test_no_url_means_no_clone_and_open_raises(tmp_path: Path) -> None:
    """Without a URL on the agent's metadata, the capability falls
    through to ``DesignMonorepoClient.open`` and surfaces the
    missing-repo error verbatim — the framework does not silently
    swallow the misconfiguration."""

    working_dir = tmp_path / "empty"
    working_dir.mkdir()
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_writer_capability(
            agent_id="agent-A",
            working_dir=working_dir,
            design_monorepo_url=None,
        )
        # ``DesignMonorepoClient.open`` raises ``DesignMonorepoError`` on
        # ``InvalidGitRepositoryError``; an empty-but-existing dir trips
        # exactly that path.
        from polymathera.colony.design_monorepo.client import DesignMonorepoError
        with pytest.raises(DesignMonorepoError):
            cap._client_sync()
    assert not (working_dir / ".git").exists()


def test_classify_git_clone_error_recognises_auth_failures() -> None:
    """``_classify_git_clone_error`` returns a :class:`GitAuthError`
    for any stderr that matches a known auth-failure marker, and
    leaves other ``GitCommandError`` instances untouched. The retry
    decorator on ``clone_or_retrieve_repository`` keys off this
    type to skip retries — pinning the markers here prevents a
    silent regression that would resurrect the 3-attempt retry storm
    on permanent auth failures.
    """

    from git.exc import GitCommandError

    from polymathera.colony.distributed.stores.git import (
        GitAuthError,
        _classify_git_clone_error,
    )

    def _err(stderr: str) -> GitCommandError:
        e = GitCommandError(["git", "clone"], 128)
        e.stderr = stderr
        return e

    auth_messages = (
        "remote: Invalid username or token. Password authentication is "
        "not supported for Git operations.\nfatal: Authentication failed",
        "fatal: could not read Username for 'https://github.com'",
        "remote: Permission denied to user@github.com",
        "fatal: Authentication failed for 'https://gitlab.com/x.git'",
    )
    for msg in auth_messages:
        result = _classify_git_clone_error(_err(msg))
        assert isinstance(result, GitAuthError), msg
        # The actionable hint is in the message body — the user needs
        # to know which env var to fix and what scopes to check.
        text = str(result)
        assert "GITHUB_TOKEN" in text
        assert "scope" in text.lower()

    # Non-auth failures (network, missing repo, etc.) must NOT be
    # reclassified — those are still worth retrying.
    transient = _err("fatal: unable to access 'https://github.com/x/y.git/': "
                     "Could not resolve host: github.com")
    assert _classify_git_clone_error(transient) is transient


def _bootstrap_empty_remote(remote_root: Path) -> Path:
    """Empty git remote — no manifest, no repo_map. The exact shape an
    operator hits on a fresh empty GitHub repo before
    ``initialize_repo_map`` runs."""

    repo = git.Repo.init(remote_root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    # An initial empty commit so the remote has a HEAD to clone.
    repo.index.commit("initial")
    return remote_root


def _bootstrap_empty_bare_remote(remote_root: Path) -> Path:
    """Bare empty git remote that accepts pushes. Used for the push
    path of ``initialize_repo_map``: a non-bare remote rejects pushes
    to its currently-checked-out branch, so we use a bare repo to
    simulate the GitHub side of the conversation faithfully."""

    git.Repo.init(remote_root, initial_branch="main", bare=True)
    return remote_root


@pytest.mark.asyncio
async def test_initialize_repo_map_triggers_lazy_clone_on_empty_workdir(
    tmp_path: Path,
) -> None:
    """The user-facing reproducer: a fresh agent's working_dir exists
    (``resolve_clone_path`` made it) but is empty — no ``.git``,
    because ``_client_sync`` (which would have lazy-cloned) is
    bypassed by ``initialize_repo_map``. The action MUST trigger the
    lazy clone itself before checking for ``.git``; otherwise it
    fails with "not a git repository — clone the design monorepo
    first" exactly the way the user reported.

    End state: ``.git`` exists, ``.colony/manifest.json`` exists,
    ``.colony/repo_map.yaml`` exists, all committed.
    """

    from polymathera.colony.design_monorepo.manifest import (
        MANIFEST_RELATIVE_PATH,
    )
    from polymathera.colony.design_monorepo.repo_map import (
        REPO_MAP_DIR, REPO_MAP_FILENAME,
    )

    remote_root = _bootstrap_empty_remote(tmp_path / "remote")
    working_dir = tmp_path / "agent_clone"
    # Agent-init resolves the path so the directory exists but is
    # empty when the action fires.
    working_dir.mkdir(parents=True, exist_ok=True)
    assert not (working_dir / ".git").exists()

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_writer_capability(
            agent_id="agent-A",
            working_dir=working_dir,
            design_monorepo_url=f"file://{remote_root}",
        )
        # ``push=False`` because the helper above creates a non-bare
        # remote — fine for clone-from but not for push-to. Push
        # behaviour is exercised in
        # ``test_initialize_repo_map_pushes_to_origin``.
        result = await cap.initialize_repo_map(push=False)

    assert (working_dir / ".git").is_dir()
    assert (working_dir / MANIFEST_RELATIVE_PATH).is_file()
    assert (working_dir / REPO_MAP_DIR / REPO_MAP_FILENAME).is_file()
    assert result["status"] == "initialized"
    assert set(result["files_created"]) == {
        MANIFEST_RELATIVE_PATH,
        f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}",
    }


@pytest.mark.asyncio
async def test_initialize_repo_map_pushes_to_origin(
    tmp_path: Path,
) -> None:
    """The user-reported regression: ``initialize_repo_map`` succeeded
    locally but no changes appeared on GitHub because the action
    never pushed.

    With ``push=True`` (the default), the action must push the
    bootstrap commit so the upstream remote — and any other clone of
    it (the dashboard's read-only inspection cache, sibling agents)
    — sees the new files. We simulate the GitHub side with a bare
    file:// remote because non-bare remotes reject pushes to their
    currently-checked-out branch.
    """

    from polymathera.colony.design_monorepo.manifest import (
        MANIFEST_RELATIVE_PATH,
    )
    from polymathera.colony.design_monorepo.repo_map import (
        REPO_MAP_DIR, REPO_MAP_FILENAME,
    )

    remote_root = _bootstrap_empty_bare_remote(tmp_path / "remote.git")
    working_dir = tmp_path / "agent_clone"
    working_dir.mkdir(parents=True, exist_ok=True)

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_writer_capability(
            agent_id="agent-A",
            working_dir=working_dir,
            design_monorepo_url=f"file://{remote_root}",
        )
        result = await cap.initialize_repo_map()

    assert result["status"] == "initialized"
    assert result["pushed"] is True
    assert result["push_error"] is None

    # The remote now actually has the commit. Cloning it fresh into a
    # third location must yield the manifest and repo_map files —
    # this is what the dashboard's cache would see on its next
    # fetch from origin.
    inspector_dir = tmp_path / "inspector"
    git.Repo.clone_from(f"file://{remote_root}", str(inspector_dir))
    assert (inspector_dir / MANIFEST_RELATIVE_PATH).is_file()
    assert (inspector_dir / REPO_MAP_DIR / REPO_MAP_FILENAME).is_file()


@pytest.mark.asyncio
async def test_initialize_repo_map_errors_clearly_when_no_url_and_no_repo(
    tmp_path: Path,
) -> None:
    """Empty working_dir + no URL on agent metadata: the action can't
    do anything without manual ``git init``. It must raise an
    actionable :class:`DesignMonorepoError`, not the generic gitpython
    "not a git repository" — the message names the two ways forward
    so the planner can surface them to the user.
    """

    from polymathera.colony.design_monorepo.client import DesignMonorepoError

    working_dir = tmp_path / "agent_clone"
    working_dir.mkdir(parents=True, exist_ok=True)

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_writer_capability(
            agent_id="agent-A",
            working_dir=working_dir,
            design_monorepo_url=None,
        )
        with pytest.raises(DesignMonorepoError) as ei:
            await cap.initialize_repo_map()

    msg = str(ei.value)
    assert "not a git repository" in msg
    assert "design_monorepo_url" in msg
    assert "git init" in msg


def test_pre_existing_clone_is_opened_not_recloned(tmp_path: Path) -> None:
    """When ``working_dir`` already contains a git repo, the capability
    must open it directly — the lazy-clone branch must not run."""

    remote_root = _bootstrap_remote(tmp_path / "remote")
    working_dir = tmp_path / "agent_clone"
    git.Repo.clone_from(f"file://{remote_root}", str(working_dir))

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ):
        cap = _make_writer_capability(
            agent_id="agent-A",
            working_dir=working_dir,
            # Bogus URL — must not be hit; the existing clone wins.
            design_monorepo_url="file:///nonexistent",
        )
        client = cap._client_sync()
    assert client is not None
