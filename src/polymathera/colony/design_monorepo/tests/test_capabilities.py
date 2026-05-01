"""Tests for the agent-facing capability surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DesignCheckpointer,
    DesignMonorepoClient,
    RepoBootstrapSpec,
    RepoStateProvider,
    ToolBuilder,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def state_provider(bootstrapped_repo: DesignMonorepoClient) -> RepoStateProvider:
    cap = RepoStateProvider(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def checkpointer(bootstrapped_repo: DesignMonorepoClient) -> DesignCheckpointer:
    cap = DesignCheckpointer(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def tool_builder(bootstrapped_repo: DesignMonorepoClient) -> ToolBuilder:
    cap = ToolBuilder(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


async def test_get_repo_state(state_provider: RepoStateProvider) -> None:
    st = await state_provider.get_repo_state()
    assert st.is_fresh is True
    assert st.current_branch == "main"


async def test_find_existing_tool_finds_after_bootstrap(
    state_provider: RepoStateProvider, tool_builder: ToolBuilder,
) -> None:
    spec = RepoBootstrapSpec(
        template="python_lib",
        target="subdir_in_monorepo:tools/shared/widgets",
        name="widget_engine",
        purpose="shared/widgets",
        license="MIT",
        capability="render_widget",
        description="Renders widgets.",
    )
    result = await tool_builder.bootstrap_repo(spec)
    assert result.tool_entry.capability == "render_widget"
    matches = await state_provider.find_existing_tool("render_widget")
    assert len(matches) == 1
    assert matches[0].entry.name == "widget_engine"
    assert matches[0].writable is True


async def test_find_existing_tool_partial_match(
    state_provider: RepoStateProvider, tool_builder: ToolBuilder,
) -> None:
    await tool_builder.bootstrap_repo(RepoBootstrapSpec(
        template="python_lib",
        target="subdir_in_monorepo:tools/shared/widgets",
        name="widget_engine",
        purpose="shared/widgets",
        license="MIT",
        capability="render_widget",
        description="renders widgets in 3D",
    ))
    matches = await state_provider.find_existing_tool("render")
    assert any(m.entry.name == "widget_engine" for m in matches)


async def test_checkpoint_state_round_trip(
    checkpointer: DesignCheckpointer, state_provider: RepoStateProvider,
) -> None:
    cp = await checkpointer.checkpoint_state("v1", "first")
    assert cp.label == "v1"
    cps = await checkpointer.list_checkpoints()
    assert any(c.checkpoint_id == cp.checkpoint_id for c in cps)
    st = await state_provider.get_repo_state()
    assert any(c.checkpoint_id == cp.checkpoint_id for c in st.checkpoints)


async def test_fork_design_creates_branch(
    checkpointer: DesignCheckpointer, state_provider: RepoStateProvider,
) -> None:
    fork = await checkpointer.fork_design("explore-A")
    assert fork.name == "fork/explore-A"
    st = await state_provider.get_repo_state()
    assert st.current_branch == "fork/explore-A"


async def test_diff_design_against_checkpoint(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
    state_provider: RepoStateProvider,
) -> None:
    cp = await checkpointer.checkpoint_state("anchor", "")
    p = bootstrapped_repo.working_dir / "design" / "added.txt"
    p.write_text("x", encoding="utf-8")
    await checkpointer.commit_state(
        "add file",
        paths=["design/added.txt"],
    )
    diff = await state_provider.diff_against_checkpoint(cp.checkpoint_id)
    paths = {e.path for e in diff.entries}
    assert "design/added.txt" in paths


async def test_bootstrap_repo_unsupported_target(
    tool_builder: ToolBuilder,
) -> None:
    with pytest.raises(NotImplementedError):
        await tool_builder.bootstrap_repo(RepoBootstrapSpec(
            template="python_lib",
            target="new_standalone:https://example.com/x.git",
            name="x",
            purpose="p",
            license="MIT",
            capability="x",
        ))


async def test_get_branch_topology(
    checkpointer: DesignCheckpointer, state_provider: RepoStateProvider,
) -> None:
    await checkpointer.fork_design("alt1")
    topo = await state_provider.get_branch_topology()
    names = {b.name for b in topo.branches}
    assert "main" in names
    assert "fork/alt1" in names
