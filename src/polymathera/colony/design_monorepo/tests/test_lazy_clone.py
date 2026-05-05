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


def test_inject_github_token_only_rewrites_github_https_urls(
    monkeypatch,
) -> None:
    """The helper rewrites bare ``https://github.com/...`` URLs when
    ``GITHUB_TOKEN`` is set, and leaves everything else alone."""

    from polymathera.colony.utils.git.utils import inject_github_token

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")

    rewritten = inject_github_token("https://github.com/o/r.git")
    assert rewritten == "https://x-access-token:ghp_secret@github.com/o/r.git"

    # ssh, file://, gitlab, and URLs that already carry credentials
    # must pass through untouched — git's standard auth machinery
    # handles those.
    assert (
        inject_github_token("git@github.com:o/r.git")
        == "git@github.com:o/r.git"
    )
    assert inject_github_token("file:///tmp/r") == "file:///tmp/r"
    assert (
        inject_github_token("https://gitlab.com/o/r.git")
        == "https://gitlab.com/o/r.git"
    )
    assert (
        inject_github_token("https://u:p@github.com/o/r.git")
        == "https://u:p@github.com/o/r.git"
    )

    # Without a token, every input is a no-op.
    monkeypatch.delenv("GITHUB_TOKEN")
    assert (
        inject_github_token("https://github.com/o/r.git")
        == "https://github.com/o/r.git"
    )


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
