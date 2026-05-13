"""Tests for L1-A: ``colony.design_monorepo.extensions`` discovery.

For every surface: assert empty-dir → empty result, populated-dir →
typed result, and a bad file → logged and skipped (not fatal). Pulls
the surface-directory contract from the manifest's ``ExtensionsConfig``
defaults so per-surface override behaviour is checked too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DEFAULT_SURFACE_DIRS,
    DesignMonorepoManifest,
    ExtensionsConfig,
    SurfaceConfig,
)
from polymathera.colony.design_monorepo.extensions import (
    DiscoveredExtensions,
    discover_agents,
    discover_all,
    discover_missions,
    discover_deployments,
    discover_plugins,
    discover_profiles,
    discover_tools,
)


def _minimal_manifest(**overrides) -> DesignMonorepoManifest:
    return DesignMonorepoManifest(
        tenant="acme",
        colony="acme-colony",
        program="prog-1",
        target_system="x",
        design_repo_url="https://example.com/repo.git",
        **overrides,
    )


# ---------------------------------------------------------------------------
# Empty-dir handling — the OPM-MEG smoke-test shape
# ---------------------------------------------------------------------------


def test_all_discoverers_handle_missing_surface_dirs(tmp_path: Path) -> None:
    """No ``.colony/<surface>/`` at all → every discoverer returns an
    empty container. This is the freshly-bootstrapped monorepo state."""
    snap = discover_all(tmp_path)
    assert snap.plugins == []
    assert snap.agents == {}
    assert snap.deployments == {}
    assert len(snap.tools) == 0
    assert snap.profiles == {}
    assert snap.missions == {}


def test_all_discoverers_handle_empty_surface_dirs(tmp_path: Path) -> None:
    """Surface dirs exist but contain no extensions (post-bootstrap with
    ``.gitkeep`` per §2.4). Same observable behaviour as missing dirs."""
    for surface_rel in DEFAULT_SURFACE_DIRS.values():
        (tmp_path / surface_rel).mkdir(parents=True, exist_ok=True)
    snap = discover_all(tmp_path)
    assert snap.plugins == []
    assert snap.agents == {}
    assert snap.deployments == {}
    assert len(snap.tools) == 0
    assert snap.profiles == {}
    assert snap.missions == {}


# ---------------------------------------------------------------------------
# discover_agents — populated surface
# ---------------------------------------------------------------------------


def test_discover_agents_finds_agent_subclasses(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/agents"
    surface.mkdir(parents=True)
    (surface / "my_agent.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "\n"
        "class MyAgent(Agent):\n"
        "    pass\n"
        "\n"
        "class _NotExported:\n"
        "    pass\n"
    )
    found = discover_agents(tmp_path)
    assert "MyAgent" in found
    assert "_NotExported" not in found
    # Re-exported names from the imported base must not appear.
    assert "Agent" not in found


def test_discover_agents_skips_unimportable_file(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/agents"
    surface.mkdir(parents=True)
    (surface / "broken.py").write_text("this is not python\n")
    (surface / "good.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class GoodAgent(Agent):\n"
        "    pass\n"
    )
    found = discover_agents(tmp_path)
    assert set(found.keys()) == {"GoodAgent"}


# ---------------------------------------------------------------------------
# discover_deployments — @serving.deployment marker, not subclass
# ---------------------------------------------------------------------------


def test_discover_deployments_finds_decorated_classes(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/deployments"
    surface.mkdir(parents=True)
    (surface / "my_dep.py").write_text(
        "from polymathera.colony.distributed.ray_utils import serving\n"
        "\n"
        "@serving.deployment()\n"
        "class MyDeployment:\n"
        "    @serving.endpoint\n"
        "    async def ping(self) -> str:\n"
        "        return 'pong'\n"
        "\n"
        "class NotADeployment:\n"
        "    pass\n"
    )
    found = discover_deployments(tmp_path)
    assert "MyDeployment" in found
    assert "NotADeployment" not in found


# ---------------------------------------------------------------------------
# discover_tools — register(registry) callback
# ---------------------------------------------------------------------------


def test_discover_tools_calls_register_callback(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/tools"
    surface.mkdir(parents=True)
    (surface / "my_tool.py").write_text(
        "from typing import ClassVar\n"
        "from polymathera.colony.tools import (\n"
        "    ToolAdapter, ToolCall, ToolResult, ToolSpec,\n"
        "    CostModel, Determinism, HITLFrequency, HeadlessReadiness, Licensing,\n"
        ")\n"
        "\n"
        "class MyAdapter(ToolAdapter):\n"
        "    spec: ClassVar[ToolSpec] = ToolSpec(\n"
        "        name='my_tool', capabilities=('solve',),\n"
        "        headless=HeadlessReadiness.NATIVE,\n"
        "        hitl_frequency=HITLFrequency.AUTONOMOUS,\n"
        "        determinism=Determinism.DETERMINISTIC,\n"
        "        licensing=Licensing.MIT,\n"
        "        backend='in_process',\n"
        "        cost_model=CostModel(),\n"
        "    )\n"
        "    async def invoke(self, call: ToolCall) -> ToolResult:\n"
        "        return ToolResult.success(call=call, output={})\n"
        "\n"
        "def register(registry):\n"
        "    registry.register(MyAdapter())\n"
    )
    registry = discover_tools(tmp_path)
    names = [a.spec.name for a in registry.list_adapters()]
    assert names == ["my_tool"]


def test_discover_tools_skips_file_without_register(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/tools"
    surface.mkdir(parents=True)
    (surface / "no_register.py").write_text("x = 1\n")
    assert len(discover_tools(tmp_path)) == 0


# ---------------------------------------------------------------------------
# discover_profiles — *.yaml mappings
# ---------------------------------------------------------------------------


def test_discover_profiles_loads_yaml_mappings(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/profiles"
    surface.mkdir(parents=True)
    (surface / "vendor_command_reference.yaml").write_text(
        "tags:\n  - vendor\n  - command-reference\nembedding: default\n",
    )
    found = discover_profiles(tmp_path)
    assert set(found.keys()) == {"vendor_command_reference"}
    assert found["vendor_command_reference"]["tags"] == ["vendor", "command-reference"]


def test_discover_profiles_skips_non_mapping_top_level(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/profiles"
    surface.mkdir(parents=True)
    (surface / "list_top.yaml").write_text("- a\n- b\n")
    (surface / "good.yaml").write_text("k: v\n")
    found = discover_profiles(tmp_path)
    assert set(found.keys()) == {"good"}


# ---------------------------------------------------------------------------
# discover_missions — mission_entry() factory + MissionSpec validation
# ---------------------------------------------------------------------------


def _write_valid_mission_file(surface: Path, stem: str) -> None:
    """Write a ``.colony/missions/<stem>.py`` whose ``mission_entry()``
    returns a dict matching :class:`MissionSpec`."""
    (surface / f"{stem}.py").write_text(
        "def mission_entry():\n"
        "    return {\n"
        f"        'label': '{stem} mission',\n"
        "        'description': 'demo',\n"
        f"        'coordinator_v1': 'test.{stem}.Coordinator',\n"
        f"        'coordinator_v2': 'test.{stem}.Coordinator',\n"
        f"        'worker': 'test.{stem}.Worker',\n"
        "        'self_concept': {'description': 'stub'},\n"
        "    }\n",
    )


def test_discover_missions_loads_valid_factories(tmp_path: Path) -> None:
    surface = tmp_path / ".colony/missions"
    surface.mkdir(parents=True)
    _write_valid_mission_file(surface, "demo")
    _write_valid_mission_file(surface, "other")
    found = discover_missions(tmp_path)
    assert set(found.keys()) == {"demo", "other"}
    assert found["demo"]["label"] == "demo mission"
    assert found["demo"]["coordinator_v2"] == "test.demo.Coordinator"


def test_discover_missions_skips_file_without_factory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    surface = tmp_path / ".colony/missions"
    surface.mkdir(parents=True)
    _write_valid_mission_file(surface, "good")
    (surface / "nofactory.py").write_text("# no mission_entry defined\n")
    with caplog.at_level("WARNING", logger="polymathera.colony.design_monorepo.extensions"):
        found = discover_missions(tmp_path)
    assert set(found.keys()) == {"good"}
    assert any(
        "exposes no callable mission_entry" in r.getMessage()
        for r in caplog.records
    )


def test_discover_missions_skips_factory_that_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    surface = tmp_path / ".colony/missions"
    surface.mkdir(parents=True)
    _write_valid_mission_file(surface, "good")
    (surface / "boom.py").write_text(
        "def mission_entry():\n"
        "    raise RuntimeError('factory exploded')\n",
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.design_monorepo.extensions"):
        found = discover_missions(tmp_path)
    assert set(found.keys()) == {"good"}
    assert any(
        "mission_entry() in" in r.getMessage() and "raised" in r.getMessage()
        for r in caplog.records
    )


def test_discover_missions_rejects_unknown_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """``MissionSpec.extra='forbid'`` catches typos at discovery time —
    the cross-mechanism schema-drift guard that the strict spec exists
    to provide."""
    surface = tmp_path / ".colony/missions"
    surface.mkdir(parents=True)
    (surface / "typoed.py").write_text(
        "def mission_entry():\n"
        "    return {\n"
        "        'label': 't', 'description': 't',\n"
        "        'coordinator_v1': 'x.Y', 'coordinator_v2': 'x.Y', 'worker': 'x.W',\n"
        "        'self_concept': {'description': 'stub'},\n"
        "        'coordinator_capabilites': ['typo'],\n"
        "    }\n",
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.design_monorepo.extensions"):
        found = discover_missions(tmp_path)
    assert found == {}
    assert any(
        "failed schema validation" in r.getMessage()
        and "coordinator_capabilites" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Manifest-driven directory override
# ---------------------------------------------------------------------------


def test_surface_directory_override_from_manifest(tmp_path: Path) -> None:
    """An ``extensions.<surface>.directory`` override redirects discovery
    to a non-default path, leaving other surfaces on defaults."""
    custom_dir = tmp_path / "vendor/agents"
    custom_dir.mkdir(parents=True)
    (custom_dir / "x.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class VendorAgent(Agent):\n"
        "    pass\n",
    )
    # Default location stays empty — only the override should be walked.
    (tmp_path / ".colony/agents").mkdir(parents=True)

    manifest = _minimal_manifest(
        extensions=ExtensionsConfig(agents=SurfaceConfig(directory="vendor/agents/")),
    )
    found = discover_agents(tmp_path, manifest=manifest)
    assert set(found.keys()) == {"VendorAgent"}


# ---------------------------------------------------------------------------
# discover_all — bundle shape
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RepoStateProvider wiring — lazy + cached + manifest-aware
# ---------------------------------------------------------------------------


def test_repo_state_provider_exposes_discovered_extensions(tmp_path: Path) -> None:
    """``RepoStateProvider.discovered_extensions`` walks the working dir
    lazily and returns a typed snapshot. No manifest required.

    Cache identity check: with no mtime changes, a second access returns
    the SAME instance (memoization).
    """
    from polymathera.colony.design_monorepo import RepoStateProvider

    (tmp_path / ".colony/agents").mkdir(parents=True)
    (tmp_path / ".colony/agents/x.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class XAgent(Agent):\n    pass\n",
    )

    cap = RepoStateProvider(agent=None, scope_id="t", working_dir=tmp_path)
    snap = cap.discovered_extensions
    assert "XAgent" in snap.agents
    assert cap.discovered_extensions is snap


def test_repo_state_provider_auto_invalidates_when_new_file_added(
    tmp_path: Path,
) -> None:
    """The L1-E authoring path: a new file appears under ``.colony/<surface>/``
    and the next ``discovered_extensions`` access reflects it without an
    explicit invalidate call. Mtime-based fingerprint catches the dir-
    mtime bump from the file create."""
    import os
    from polymathera.colony.design_monorepo import RepoStateProvider

    (tmp_path / ".colony/agents").mkdir(parents=True)
    cap = RepoStateProvider(agent=None, scope_id="t", working_dir=tmp_path)
    assert cap.discovered_extensions.agents == {}

    # Bump the parent dir's mtime explicitly so the test is robust to
    # filesystem clock granularity (the previous discovery may have
    # happened in the same second as this write).
    (tmp_path / ".colony/agents/new_agent.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class NewAgent(Agent):\n    pass\n",
    )
    later = os.stat(tmp_path / ".colony/agents").st_mtime_ns + 1_000_000
    os.utime(tmp_path / ".colony/agents", ns=(later, later))

    snap = cap.discovered_extensions
    assert "NewAgent" in snap.agents


def test_repo_state_provider_invalidate_extensions_method(tmp_path: Path) -> None:
    """The explicit escape hatch: file contents change in place (no dir
    mtime bump), so caller must call ``invalidate_extensions()`` to see
    the new content."""
    from polymathera.colony.design_monorepo import RepoStateProvider

    (tmp_path / ".colony/agents").mkdir(parents=True)
    (tmp_path / ".colony/agents/x.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class V1Agent(Agent):\n    pass\n",
    )
    cap = RepoStateProvider(agent=None, scope_id="t", working_dir=tmp_path)
    assert "V1Agent" in cap.discovered_extensions.agents

    # Replace the file in place — dir mtime not bumped on Linux.
    # NOTE: importlib caches the module object; we use a different
    # module-loading id so the new content actually re-imports. See the
    # ``_load_py_module`` synthesised name in extensions.py — it tags
    # each import with ``id(path)``, which differs across Path objects
    # but the same Path is reused here, so this also implicitly tests
    # that subsequent imports get a fresh spec.
    (tmp_path / ".colony/agents/x.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class V2Agent(Agent):\n    pass\n",
    )

    # Without invalidation, cache is stale (file content edit doesn't
    # bump dir mtime).
    assert "V1Agent" in cap.discovered_extensions.agents
    assert "V2Agent" not in cap.discovered_extensions.agents

    cap.invalidate_extensions()
    assert "V2Agent" in cap.discovered_extensions.agents


def test_repo_state_provider_auto_invalidates_on_manifest_edit(
    tmp_path: Path,
) -> None:
    """Editing the manifest's ``extensions`` block (e.g., to redirect a
    surface to a non-default path) bumps the manifest mtime →
    fingerprint changes → re-discovery picks up the new override."""
    import os
    from polymathera.colony.design_monorepo import RepoStateProvider
    # First manifest: defaults — agent at .colony/agents/.
    manifest = _minimal_manifest(extensions=ExtensionsConfig())
    manifest.write_path(tmp_path)
    (tmp_path / ".colony/agents").mkdir(parents=True)
    (tmp_path / ".colony/agents/default.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class DefaultAgent(Agent):\n    pass\n",
    )
    cap = RepoStateProvider(agent=None, scope_id="t", working_dir=tmp_path)
    assert "DefaultAgent" in cap.discovered_extensions.agents

    # Rewrite manifest pointing agents at vendor/agents/.
    (tmp_path / "vendor/agents").mkdir(parents=True)
    (tmp_path / "vendor/agents/v.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class VendorAgent(Agent):\n    pass\n",
    )
    new_manifest = _minimal_manifest(
        extensions=ExtensionsConfig(agents=SurfaceConfig(directory="vendor/agents/")),
    )
    # Make sure the rewrite bumps the manifest mtime past the previous
    # value even on coarse-grained filesystems.
    manifest_path = tmp_path / ".colony/manifest.json"
    later = os.stat(manifest_path).st_mtime_ns + 1_000_000
    new_manifest.write_path(tmp_path)
    os.utime(manifest_path, ns=(later, later))

    snap = cap.discovered_extensions
    assert "VendorAgent" in snap.agents
    assert "DefaultAgent" not in snap.agents


def test_repo_state_provider_auto_invalidates_inside_override_dir(
    tmp_path: Path,
) -> None:
    """The fingerprint must stat the *resolved* surface dirs, not just
    the defaults. When the manifest points agents at vendor/agents/ and
    a NEW file appears there, ``discovered_extensions`` must pick it
    up without calling ``invalidate_extensions`` manually."""
    import os
    from polymathera.colony.design_monorepo import RepoStateProvider

    manifest = _minimal_manifest(
        extensions=ExtensionsConfig(agents=SurfaceConfig(directory="vendor/agents/")),
    )
    manifest.write_path(tmp_path)
    (tmp_path / "vendor/agents").mkdir(parents=True)

    cap = RepoStateProvider(agent=None, scope_id="t", working_dir=tmp_path)
    assert cap.discovered_extensions.agents == {}

    # New file in the OVERRIDE dir (not the default .colony/agents/).
    (tmp_path / "vendor/agents/v.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class VendorAgent(Agent):\n    pass\n",
    )
    later = os.stat(tmp_path / "vendor/agents").st_mtime_ns + 1_000_000
    os.utime(tmp_path / "vendor/agents", ns=(later, later))

    snap = cap.discovered_extensions
    assert "VendorAgent" in snap.agents


def test_repo_state_provider_honours_manifest_directory_override(
    tmp_path: Path,
) -> None:
    """When the manifest's ``extensions.agents.directory`` overrides the
    default, ``RepoStateProvider.discovered_extensions`` walks the override
    instead. End-to-end check that the wiring reads the manifest."""
    from polymathera.colony.design_monorepo import RepoStateProvider

    # Write a v2 manifest pointing the agents surface at a custom dir.
    manifest = _minimal_manifest(
        extensions=ExtensionsConfig(agents=SurfaceConfig(directory="vendor/agents/")),
    )
    manifest.write_path(tmp_path)

    (tmp_path / "vendor/agents").mkdir(parents=True)
    (tmp_path / "vendor/agents/v.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class VendorAgent(Agent):\n    pass\n",
    )
    # The default location stays empty — only the override should be walked.
    (tmp_path / ".colony/agents").mkdir(parents=True)

    cap = RepoStateProvider(agent=None, scope_id="t", working_dir=tmp_path)
    snap = cap.discovered_extensions
    assert set(snap.agents.keys()) == {"VendorAgent"}


def test_discover_all_aggregates_all_surfaces(tmp_path: Path) -> None:
    """A monorepo with one entry per surface produces a populated
    ``DiscoveredExtensions``. Smoke test that the bundle wiring works."""
    # agents
    (tmp_path / ".colony/agents").mkdir(parents=True)
    (tmp_path / ".colony/agents/a.py").write_text(
        "from polymathera.colony.agents.base import Agent\n"
        "class A(Agent):\n    pass\n",
    )
    # deployments
    (tmp_path / ".colony/deployments").mkdir(parents=True)
    (tmp_path / ".colony/deployments/d.py").write_text(
        "from polymathera.colony.distributed.ray_utils import serving\n"
        "@serving.deployment()\n"
        "class D:\n"
        "    @serving.endpoint\n"
        "    async def ping(self): return 'p'\n",
    )
    # profiles
    (tmp_path / ".colony/profiles").mkdir(parents=True)
    (tmp_path / ".colony/profiles/p.yaml").write_text("k: v\n")

    snap = discover_all(tmp_path)
    assert isinstance(snap, DiscoveredExtensions)
    assert "A" in snap.agents
    assert "D" in snap.deployments
    assert "p" in snap.profiles
    # Untouched surfaces are empty (not missing fields).
    assert snap.plugins == []
    assert len(snap.tools) == 0
