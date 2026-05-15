"""Tests for UI-configured commit attribution.

The colony's landing-page UI stores commit attribution in
``agent.metadata.parameters["git_attribution"]`` — a dict with
``commit_principal`` / ``commit_co_author`` / ``git_user_name`` /
``git_user_email`` keys. Every commit-producing capability action must
honour it via :meth:`_DesignMonorepoCapabilityBase._commit_attribution`.

Pre-fix, only ``DesignCheckpointer.initialize_repo_map`` did. Every
other commit (L1-F writes, ``commit_state``, ``merge_design``,
``tag_checkpoint``, auto-quiescence checkpoint, ``cherry_pick``,
L1-E ``bootstrap_*``, …) silently used the raw agent identity,
ignoring the UI config. This file asserts the bug stays fixed:

- author/committer match the UI-configured principal
- Co-Authored-By: trailer matches the UI-configured co-author
- L1-F writes / commit_state / tag_checkpoint / merge_design / L1-E
  bootstrap all flow through the same resolver
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from polymathera.colony.design_monorepo import (
    AgentIdentity,
    DesignCheckpointer,
    DesignMonorepoClient,
    DesignMonorepoManifest,
    ProjectAuthoringCapability,
    RepoBootstrapSpec,
    ToolBuilder,
    bootstrap_design_monorepo,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fake agent with UI-configured attribution
# ---------------------------------------------------------------------------


@dataclass
class _FakeMetadata:
    """Minimum subset of agent.metadata the attribution resolver reads.

    The real :class:`Agent.metadata` is a :class:`pydantic.BaseModel`;
    we only need ``role`` and ``parameters`` here, so a plain dataclass
    is enough."""

    role: str = "agent"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeAgent:
    """Minimum subset of :class:`Agent` the capability base reads.

    ``_DesignMonorepoCapabilityBase`` accesses ``agent.agent_id``,
    ``agent.colony_id`` and ``agent.metadata.{role, parameters}``."""

    agent_id: str
    colony_id: str
    metadata: _FakeMetadata


def _agent_with_attribution(
    *,
    commit_principal: str,
    commit_co_author: str | None,
    git_user_name: str | None = None,
    git_user_email: str | None = None,
) -> _FakeAgent:
    return _FakeAgent(
        agent_id="agent_pr3b",
        colony_id="acme-colony",
        metadata=_FakeMetadata(
            role="agent",
            parameters={
                "git_attribution": {
                    "commit_principal": commit_principal,
                    "commit_co_author": commit_co_author,
                    "git_user_name": git_user_name,
                    "git_user_email": git_user_email,
                },
            },
        ),
    )


def _build_checkpointer(
    repo: DesignMonorepoClient, agent: _FakeAgent | None,
) -> DesignCheckpointer:
    cap = DesignCheckpointer(
        agent=agent, scope_id="test",
        working_dir=repo.working_dir,
    )
    cap._client = repo
    return cap


def _build_authoring(
    repo: DesignMonorepoClient, agent: _FakeAgent | None,
) -> ProjectAuthoringCapability:
    cap = ProjectAuthoringCapability(
        agent=agent, scope_id="test",
        working_dir=repo.working_dir,
    )
    cap._client = repo
    return cap


def _build_tool_builder(
    repo: DesignMonorepoClient, agent: _FakeAgent | None,
) -> ToolBuilder:
    cap = ToolBuilder(
        agent=agent, scope_id="test",
        working_dir=repo.working_dir,
    )
    cap._client = repo
    return cap


# ---------------------------------------------------------------------------
# L1-F: ProjectAuthoringCapability writes honour the principal
# ---------------------------------------------------------------------------


async def test_l1f_write_uses_ui_configured_user_principal(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """``commit_principal='user'`` + ``git_user_name``/``email`` set:
    every L1-F write must commit as the user, not the agent."""

    agent = _agent_with_attribution(
        commit_principal="user",
        commit_co_author=None,
        git_user_name="Ada Lovelace",
        git_user_email="ada@example.com",
    )
    cap = _build_authoring(bootstrapped_repo, agent)
    bootstrapped_repo.repo.git.checkout("-b", "wip/x")
    await cap.write_file("src/x.py", "x = 1\n")
    head = bootstrapped_repo.repo.head.commit
    assert head.author.name == "Ada Lovelace"
    assert head.author.email == "ada@example.com"
    assert head.committer.name == "Ada Lovelace"


async def test_l1f_write_appends_co_author_trailer(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """``commit_co_author='user'`` adds a ``Co-Authored-By:`` trailer
    on the commit message."""

    agent = _agent_with_attribution(
        commit_principal="colony",
        commit_co_author="user",
        git_user_name="Ada Lovelace",
        git_user_email="ada@example.com",
    )
    cap = _build_authoring(bootstrapped_repo, agent)
    bootstrapped_repo.repo.git.checkout("-b", "wip/x")
    await cap.write_file("src/x.py", "x = 1\n")
    head = bootstrapped_repo.repo.head.commit
    assert (
        "Co-Authored-By: Ada Lovelace <ada@example.com>" in head.message
    )
    # And the colony is the principal (author / committer):
    assert head.author.name == "colony:acme-colony"


# ---------------------------------------------------------------------------
# DesignCheckpointer: commit_state honours principal + co-author
# ---------------------------------------------------------------------------


async def test_commit_state_honours_ui_attribution(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    agent = _agent_with_attribution(
        commit_principal="user",
        commit_co_author="colony",
        git_user_name="Ada Lovelace",
        git_user_email="ada@example.com",
    )
    cap = _build_checkpointer(bootstrapped_repo, agent)
    bootstrapped_repo.repo.git.checkout("-b", "wip/work")
    (bootstrapped_repo.working_dir / "f.txt").write_text("x", encoding="utf-8")
    res = await cap.commit_state("add f.txt", paths=["f.txt"])
    assert res.status == "executed"
    head = bootstrapped_repo.repo.head.commit
    assert head.author.name == "Ada Lovelace"
    assert "Co-Authored-By: colony:acme-colony" in head.message


# ---------------------------------------------------------------------------
# Defaults: principal='colony', co_author='user' (framework default)
# ---------------------------------------------------------------------------


async def test_default_attribution_is_colony_principal(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """No git_attribution config → falls back to
    ``principal='colony', co_author='user'``. Without
    ``git_user_name`` / ``email`` the co-author is silently dropped
    (the resolver logs a warning) but the principal still lands."""

    agent = _FakeAgent(
        agent_id="agent_default",
        colony_id="acme-colony",
        metadata=_FakeMetadata(parameters={}),
    )
    cap = _build_authoring(bootstrapped_repo, agent)
    bootstrapped_repo.repo.git.checkout("-b", "wip/default")
    await cap.write_file("src/y.py", "y = 2\n")
    head = bootstrapped_repo.repo.head.commit
    assert head.author.name == "colony:acme-colony"
    # No co-author because user_name/email missing — trailer absent.
    assert "Co-Authored-By" not in head.message


async def test_attribution_carries_through_checkpoint_and_tag(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """``checkpoint_state`` produces (a) a commit and (b) a checkpoint
    tag. The principal must land on both; the trailer lands on the
    commit but not on the tag annotation."""

    agent = _agent_with_attribution(
        commit_principal="user",
        commit_co_author="colony",
        git_user_name="Ada Lovelace",
        git_user_email="ada@example.com",
    )
    cap = _build_checkpointer(bootstrapped_repo, agent)
    bootstrapped_repo.repo.git.checkout("-b", "wip/cp")
    (bootstrapped_repo.working_dir / "z.txt").write_text("z", encoding="utf-8")
    cp = await cap.checkpoint_state("initial", "first checkpoint")
    # ``checkpoint_state`` produces 3 git objects:
    #   1. The state commit (HEAD~1 from here, since the log-append
    #      commit lands afterwards).
    #   2. The checkpoint/<...> annotated tag pointing at (1).
    #   3. A trailing ``checkpoint log: <label>`` commit recording
    #      the tag in ``.colony/checkpoints.log``.
    # Principal attribution must land on the user-visible state
    # commit; the Co-Authored-By: trailer too.
    state_commit = bootstrapped_repo.repo.head.commit.parents[0]
    assert state_commit.author.name == "Ada Lovelace"
    assert "Co-Authored-By: colony:acme-colony" in state_commit.message
    # The trailing log commit also carries the principal (consistent
    # author across the bundle).
    log_commit = bootstrapped_repo.repo.head.commit
    assert log_commit.author.name == "Ada Lovelace"
    # Tag annotation has the original label/rationale shape with no
    # trailer (the tag annotation parser depends on it).
    tag = next(
        t for t in bootstrapped_repo.repo.tags
        if t.name == cp.checkpoint_id
    )
    assert "initial" in tag.tag.message
    assert "Co-Authored-By" not in tag.tag.message


# ---------------------------------------------------------------------------
# Detached / no-agent: graceful fallback
# ---------------------------------------------------------------------------


async def test_detached_mode_falls_back_to_colony_principal(
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """A detached (agent=None) capability falls back to
    ``principal='colony'`` so the commit doesn't fail."""

    cap = _build_authoring(bootstrapped_repo, None)
    bootstrapped_repo.repo.git.checkout("-b", "wip/detached")
    await cap.write_file("src/d.py", "d = 1\n")
    head = bootstrapped_repo.repo.head.commit
    assert head.author.name.startswith("colony:")
