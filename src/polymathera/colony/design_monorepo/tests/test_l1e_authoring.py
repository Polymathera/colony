"""Tests for L1-E: ``ToolBuilder.bootstrap_<surface>`` authoring.

Round-trip discipline: every ``bootstrap_<surface>`` call must land a
file L1-A's ``discover_*`` immediately picks up. Plus: the AST allow-
list rejects disallowed surfaces; the surface set never gets re-
enumerated (one ``DEFAULT_SURFACE_DIRS`` for both halves).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.agents.blackboard.protocol import DesignMonorepoEventProtocol
from polymathera.colony.design_monorepo import (
    DEFAULT_SURFACE_DIRS,
    DesignMonorepoClient,
    DesignMonorepoError,
    ExtensionAuthoredPayload,
    RepoStateProvider,
    ToolBuilder,
)
from polymathera.colony.design_monorepo.ast_validator import (
    DISALLOWED_ATTRIBUTE_CALLS,
    DISALLOWED_BUILTIN_CALLS,
    DISALLOWED_FROM_OS,
    DISALLOWED_IMPORT_MODULES,
    validate_python_source,
)
from polymathera.colony.design_monorepo.extensions import discover_all
from polymathera.colony.design_monorepo.scaffolds import render_extension_scaffold


# ---------------------------------------------------------------------------
# Single-source-of-truth audit (Check #1 in feedback_pre_completion_audit.md)
# ---------------------------------------------------------------------------


def test_extension_template_set_matches_default_surface_dirs() -> None:
    """The L1-E scaffold map's surface keys must equal
    ``DEFAULT_SURFACE_DIRS`` — the renderer's import-time assertion is
    one half; this test is the public-facing half."""
    from polymathera.colony.design_monorepo.scaffolds.renderer import (
        _EXTENSION_TEMPLATE_FILE_BY_SURFACE,
    )
    assert set(_EXTENSION_TEMPLATE_FILE_BY_SURFACE) == set(DEFAULT_SURFACE_DIRS)


def test_extension_authored_payload_literal_matches_default_surface_dirs() -> None:
    """``ExtensionAuthoredPayload.surface``'s ``Literal[...]`` enumerates
    the surfaces a second time (the type system requires literal
    strings — it can't reference ``DEFAULT_SURFACE_DIRS``). Catch any
    drift by comparing the Literal args to the canonical set."""
    from typing import get_args

    annotation = ExtensionAuthoredPayload.model_fields["surface"].annotation
    assert set(get_args(annotation)) == set(DEFAULT_SURFACE_DIRS)


# ---------------------------------------------------------------------------
# AST allow-list — Risk #5 stopgap
# ---------------------------------------------------------------------------


def test_ast_validator_accepts_clean_source() -> None:
    src = (
        "from polymathera.colony.agents.base import Agent\n"
        "class Demo(Agent):\n    pass\n"
    )
    assert validate_python_source(src).ok


@pytest.mark.parametrize(
    "src",
    [
        "import os\nos.system('rm -rf /')\n",
        "import subprocess\nsubprocess.run(['rm', '-rf', '/'])\n",
        "from os import system\nsystem('whoami')\n",
        "eval('__import__(\"os\").system(\"ls\")')\n",
        "exec('print(1)')\n",
        "__import__('os').system('id')\n",
        "import ctypes\n",
        "import pickle\npickle.loads(b'')\n",
    ],
)
def test_ast_validator_rejects_disallowed(src: str) -> None:
    report = validate_python_source(src)
    assert not report.ok, src
    assert report.issues, f"expected at least one issue for: {src!r}"


def test_ast_validator_accepts_os_path_join() -> None:
    """``import os`` itself is allowed — only the shell-spawning names
    are rejected. ``os.path.join`` is universal."""
    src = "import os\nx = os.path.join('a', 'b')\n"
    assert validate_python_source(src).ok


def test_disallowed_constants_are_single_source() -> None:
    """The four DISALLOWED_* constants are the entire surface. No
    duplicate enumeration elsewhere — this test fails loudly if
    someone copy-pastes them into a second module."""
    # ``os.system`` should appear as the attribute pair AND as a
    # ``from os import system`` candidate, but the entries live in
    # ONE module each.
    assert ("os", "system") in DISALLOWED_ATTRIBUTE_CALLS
    assert "system" in DISALLOWED_FROM_OS
    assert "eval" in DISALLOWED_BUILTIN_CALLS
    assert "subprocess" in DISALLOWED_IMPORT_MODULES


# ---------------------------------------------------------------------------
# Renderer — refuses to overwrite, surface→destination mapping
# ---------------------------------------------------------------------------


def test_render_extension_scaffold_writes_canonical_path(tmp_path: Path) -> None:
    """plugins write to ``<name>/SKILL.md``; profiles to ``<name>.yaml``;
    agents/deployments/tools to ``<name>.py``."""
    surface_dir = tmp_path / "surface"
    surface_dir.mkdir()
    plugin = render_extension_scaffold("plugins", surface_dir, "my_skill")
    assert plugin == surface_dir / "my_skill" / "SKILL.md"
    assert plugin.is_file()

    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    out = render_extension_scaffold(
        "agents", agent_dir, "my_agent",
        template_vars={
            "class_name": "MyAgent",
            "base_class": "Agent",
            "base_module": "polymathera.colony.agents.base",
            "description": "stub",
        },
    )
    assert out == agent_dir / "my_agent.py"
    src = out.read_text(encoding="utf-8")
    assert "class MyAgent(Agent)" in src

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    out = render_extension_scaffold(
        "profiles", profile_dir, "p1",
        template_vars={
            "description": "x",
            "tags_yaml": " []",
            "embedding_strategy": "default",
        },
    )
    assert out == profile_dir / "p1.yaml"


def test_render_extension_scaffold_refuses_overwrite(tmp_path: Path) -> None:
    from polymathera.colony.design_monorepo.scaffolds import ScaffoldRenderError

    surface_dir = tmp_path / "agents"
    surface_dir.mkdir()
    render_extension_scaffold(
        "agents", surface_dir, "a1",
        template_vars={
            "class_name": "A1",
            "base_class": "Agent",
            "base_module": "polymathera.colony.agents.base",
            "description": "x",
        },
    )
    with pytest.raises(ScaffoldRenderError):
        render_extension_scaffold(
            "agents", surface_dir, "a1",
            template_vars={
                "class_name": "A1",
                "base_class": "Agent",
                "base_module": "polymathera.colony.agents.base",
                "description": "x",
            },
        )


# ---------------------------------------------------------------------------
# bootstrap_<surface> — write, validate, commit, round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_builder(bootstrapped_repo: DesignMonorepoClient) -> ToolBuilder:
    cap = ToolBuilder(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def state_provider(bootstrapped_repo: DesignMonorepoClient) -> RepoStateProvider:
    cap = RepoStateProvider(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.mark.asyncio
async def test_bootstrap_plugin_round_trips(
    tool_builder: ToolBuilder, bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await tool_builder.bootstrap_plugin(
        "serf_simulator", description="Toy SERF simulator.",
    )
    assert isinstance(payload, ExtensionAuthoredPayload)
    assert payload.surface == "plugins"
    assert payload.name == "serf_simulator"
    skill_md = bootstrapped_repo.working_dir / ".colony" / "plugins" / "serf_simulator" / "SKILL.md"
    assert skill_md.is_file()
    discovered = discover_all(bootstrapped_repo.working_dir)
    names = {s.name for s in discovered.plugins}
    assert "serf_simulator" in names


@pytest.mark.asyncio
async def test_bootstrap_agent_round_trips(
    tool_builder: ToolBuilder, bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await tool_builder.bootstrap_agent("opm_meg_planner")
    assert payload.surface == "agents"
    py = bootstrapped_repo.working_dir / ".colony" / "agents" / "opm_meg_planner.py"
    assert py.is_file()
    discovered = discover_all(bootstrapped_repo.working_dir)
    # The default class name is PascalCased from the snake-case input.
    assert "OpmMegPlanner" in discovered.agents
    cls = discovered.agents["OpmMegPlanner"]
    from polymathera.colony.agents.base import Agent
    assert issubclass(cls, Agent)


@pytest.mark.asyncio
async def test_bootstrap_deployment_round_trips(
    tool_builder: ToolBuilder, bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await tool_builder.bootstrap_deployment(
        "scoring_service", deployment_kwargs="",
    )
    assert payload.surface == "deployments"
    discovered = discover_all(bootstrapped_repo.working_dir)
    assert "ScoringService" in discovered.deployments


@pytest.mark.asyncio
async def test_bootstrap_tool_adapter_round_trips(
    tool_builder: ToolBuilder, bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await tool_builder.bootstrap_tool_adapter("quspin_adapter")
    assert payload.surface == "tools"
    py = bootstrapped_repo.working_dir / ".colony" / "tools" / "quspin_adapter.py"
    assert py.is_file()
    # ``discover_tools`` invokes the file's ``register()``; our stub
    # registers nothing, but the file must execute without error.
    registry = discover_all(bootstrapped_repo.working_dir).tools
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_bootstrap_profile_round_trips(
    tool_builder: ToolBuilder, bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await tool_builder.bootstrap_profile(
        "fda_510k_pattern",
        tags=["regulatory", "510k"],
        embedding_strategy="bm25",
        description="510(k) regulatory tag profile",
    )
    assert payload.surface == "profiles"
    discovered = discover_all(bootstrapped_repo.working_dir).profiles
    assert "fda_510k_pattern" in discovered
    body = discovered["fda_510k_pattern"]
    assert body["tags"] == ["regulatory", "510k"]
    assert body["embedding_strategy"] == "bm25"


@pytest.mark.asyncio
async def test_bootstrap_lands_session_attributed_commit(
    tool_builder: ToolBuilder, bootstrapped_repo: DesignMonorepoClient,
) -> None:
    payload = await tool_builder.bootstrap_agent("audit_agent")
    repo = bootstrapped_repo._repo
    commit = repo.commit(payload.commit_sha)
    assert "bootstrap agents/audit_agent (L1-E, scaffold=blank_agents)" in commit.message


@pytest.mark.asyncio
async def test_round_trip_state_provider_cache_invalidates(
    tool_builder: ToolBuilder, state_provider: RepoStateProvider,
) -> None:
    """RepoStateProvider's discovered_extensions cache must reflect the
    just-authored extension on the next access — the mtime fingerprint
    covers the surface dir, so adding a file bumps the dir mtime and
    forces a re-walk."""
    snap_before = state_provider.discovered_extensions
    assert snap_before.agents == {}
    await tool_builder.bootstrap_agent("freshly_authored")
    snap_after = state_provider.discovered_extensions
    assert "FreshlyAuthored" in snap_after.agents


# ---------------------------------------------------------------------------
# Blackboard audit event
# ---------------------------------------------------------------------------


def test_extension_authored_event_key_round_trips() -> None:
    key = DesignMonorepoEventProtocol.extension_authored_key("agents", "foo")
    surface, name = DesignMonorepoEventProtocol.parse_extension_authored_key(key)
    assert surface == "agents"
    assert name == "foo"
    assert DesignMonorepoEventProtocol.extension_authored_pattern().endswith("*")
    # Name containing a colon is preserved by ``split(":", 1)``.
    weird = DesignMonorepoEventProtocol.extension_authored_key("agents", "ns:bar")
    s2, n2 = DesignMonorepoEventProtocol.parse_extension_authored_key(weird)
    assert (s2, n2) == ("agents", "ns:bar")


def test_extension_authored_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        DesignMonorepoEventProtocol.parse_extension_authored_key("branch_changed:foo")
    with pytest.raises(ValueError):
        # Missing ``:name`` tail.
        DesignMonorepoEventProtocol.parse_extension_authored_key("extension_authored:agents")
