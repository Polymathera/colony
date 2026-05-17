"""Tests for the L2-F extension-scaffold registry + the renderer's
``scaffold_id`` dispatch.

The blank L1-E path (``scaffold_id=None``) is exercised by
``test_l1e_authoring.py``; this file covers the *registered* path
that CPS (and any third-party extension package) plugs into.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DesignMonorepoClient,
    ToolBuilder,
)
from polymathera.colony.design_monorepo.scaffolds import (
    ExtensionScaffold,
    ExtensionScaffoldRegistryError,
    available_scaffolds,
    get_extension_scaffold,
    register_extension_scaffold,
    render_extension_scaffold,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry_between_tests():
    """Keep tests independent: PR 4's production registration only
    runs at startup, so tests can populate freely and reset after."""
    reset_registry()
    yield
    reset_registry()


def _write_template(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "fake_agent.py.tmpl"
    f.write_text(body, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_then_lookup(tmp_path: Path) -> None:
    tmpl = _write_template(tmp_path, "x = 1\n")
    sc = ExtensionScaffold(
        scaffold_id="my_agent", surface="agents", template_path=tmpl,
    )
    register_extension_scaffold(sc)
    assert get_extension_scaffold("my_agent") is sc


def test_register_duplicate_raises(tmp_path: Path) -> None:
    tmpl = _write_template(tmp_path, "x = 1\n")
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="dup", surface="agents", template_path=tmpl,
    ))
    with pytest.raises(ExtensionScaffoldRegistryError, match="already registered"):
        register_extension_scaffold(ExtensionScaffold(
            scaffold_id="dup", surface="agents", template_path=tmpl,
        ))


def test_register_unknown_surface_raises(tmp_path: Path) -> None:
    tmpl = _write_template(tmp_path, "x = 1\n")
    with pytest.raises(ExtensionScaffoldRegistryError, match="DEFAULT_SURFACE_DIRS"):
        register_extension_scaffold(ExtensionScaffold(
            scaffold_id="bogus", surface="not_a_surface", template_path=tmpl,
        ))


def test_register_missing_template_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtensionScaffoldRegistryError, match="does not exist"):
        register_extension_scaffold(ExtensionScaffold(
            scaffold_id="ghost", surface="agents",
            template_path=tmp_path / "does_not_exist.tmpl",
        ))


def test_get_unregistered_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtensionScaffoldRegistryError, match="no scaffold"):
        get_extension_scaffold("never_registered")


def test_available_scaffolds_filters_by_surface(tmp_path: Path) -> None:
    tmpl = _write_template(tmp_path, "x = 1\n")
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="a1", surface="agents", template_path=tmpl,
    ))
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="a2", surface="agents", template_path=tmpl,
    ))
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="t1", surface="tools", template_path=tmpl,
    ))
    assert {s.scaffold_id for s in available_scaffolds("agents")} == {"a1", "a2"}
    assert {s.scaffold_id for s in available_scaffolds("tools")} == {"t1"}
    assert {s.scaffold_id for s in available_scaffolds()} == {"a1", "a2", "t1"}


# ---------------------------------------------------------------------------
# Renderer's scaffold_id dispatch
# ---------------------------------------------------------------------------


def test_render_with_scaffold_id_uses_registered_template(
    tmp_path: Path,
) -> None:
    tmpl = _write_template(
        tmp_path,
        "# domain-shaped agent for ${class_name}\n"
        "class ${class_name}: ...\n",
    )
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="domain_shaped", surface="agents", template_path=tmpl,
    ))
    surface_dir = tmp_path / "agents_dir"
    surface_dir.mkdir()
    out = render_extension_scaffold(
        "agents", surface_dir, "my_inst",
        template_vars={"class_name": "MyInst"},
        scaffold_id="domain_shaped",
    )
    text = out.read_text(encoding="utf-8")
    assert "# domain-shaped agent for MyInst" in text
    assert "class MyInst: ..." in text


def test_render_with_unknown_scaffold_id_raises(tmp_path: Path) -> None:
    from polymathera.colony.design_monorepo.scaffolds import ScaffoldRenderError

    surface_dir = tmp_path / "agents_dir"
    surface_dir.mkdir()
    with pytest.raises(ExtensionScaffoldRegistryError):
        render_extension_scaffold(
            "agents", surface_dir, "x", scaffold_id="never_registered",
        )


def test_render_scaffold_surface_mismatch_raises(tmp_path: Path) -> None:
    from polymathera.colony.design_monorepo.scaffolds import ScaffoldRenderError

    tmpl = _write_template(tmp_path, "x = 1\n")
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="agent_only", surface="agents", template_path=tmpl,
    ))
    surface_dir = tmp_path / "tools_dir"
    surface_dir.mkdir()
    # Render requests surface=tools but scaffold is bound to agents.
    with pytest.raises(ScaffoldRenderError, match="targets surface"):
        render_extension_scaffold(
            "tools", surface_dir, "x", scaffold_id="agent_only",
        )


def test_render_without_scaffold_id_uses_blank_template(tmp_path: Path) -> None:
    """Backwards compatibility: scaffold_id=None still renders the
    L1-E blank template that PR 2 shipped."""
    surface_dir = tmp_path / "agents_dir"
    surface_dir.mkdir()
    out = render_extension_scaffold(
        "agents", surface_dir, "x",
        template_vars={
            "class_name": "X", "base_class": "Agent",
            "base_module": "polymathera.colony.agents.base",
            "description": "stub",
        },
    )
    text = out.read_text(encoding="utf-8")
    # Blank template subclasses the abstract Agent.
    assert "class X(Agent)" in text


# ---------------------------------------------------------------------------
# End-to-end through ToolBuilder.bootstrap_<surface>(scaffold=...)
# ---------------------------------------------------------------------------


pytestmark_async = pytest.mark.asyncio


@pytest.fixture
def tool_builder(bootstrapped_repo: DesignMonorepoClient) -> ToolBuilder:
    cap = ToolBuilder(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.mark.asyncio
async def test_bootstrap_agent_uses_registered_scaffold(
    tmp_path: Path,
    tool_builder: ToolBuilder,
    bootstrapped_repo: DesignMonorepoClient,
) -> None:
    """The action surface routes ``scaffold=...`` through to the
    registered template; the rendered file reflects the scaffold's
    body (not the L1-E blank one)."""
    tmpl = tmp_path / "domain_agent.py.tmpl"
    tmpl.write_text(
        "# L2-F: domain-shaped agent\n"
        "class ${class_name}:\n"
        "    framework = '${framework_id}'\n",
        encoding="utf-8",
    )
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="my_domain_agent", surface="agents", template_path=tmpl,
    ))
    payload = await tool_builder.bootstrap_agent(
        "fda_audit",
        scaffold="my_domain_agent",
        template_vars={"framework_id": "fda_samd"},
    )
    body = (
        bootstrapped_repo.working_dir / ".colony" / "agents" / "fda_audit.py"
    ).read_text(encoding="utf-8")
    assert "# L2-F: domain-shaped agent" in body
    assert "class FdaAudit:" in body
    assert "framework = 'fda_samd'" in body
    # The payload records the scaffold id (not the surface name) so
    # the audit trail names exactly which template fired.
    assert payload.template == "my_domain_agent"


@pytest.mark.asyncio
async def test_bootstrap_surface_mismatch_with_scaffold_raises(
    tmp_path: Path,
    tool_builder: ToolBuilder,
) -> None:
    from polymathera.colony.design_monorepo import DesignMonorepoError

    tmpl = _write_template(tmp_path, "x = 1\n")
    register_extension_scaffold(ExtensionScaffold(
        scaffold_id="agent_bound", surface="agents", template_path=tmpl,
    ))
    # Caller picks a tools-surface bootstrap with an agents-bound
    # scaffold — the renderer rejects it cleanly.
    with pytest.raises((Exception,)) as exc_info:
        await tool_builder.bootstrap_tool_capability(
            "wrong", scaffold="agent_bound",
        )
    assert "agent_bound" in str(exc_info.value) or "targets surface" in str(
        exc_info.value,
    )
