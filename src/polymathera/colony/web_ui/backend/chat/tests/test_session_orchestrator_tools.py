"""Tests for ``SessionOrchestratorCapability._refresh_available_tools``.

C9 of the tool-framework retrofit
(``colony/STAGE_B_TOOL_FRAMEWORK_RETROFIT_PLAN.md``): the SessionAgent's
LLM planner reads ``metadata.parameters["available_tools"]`` to know
which L4 tool capabilities the operator's design monorepo declares,
so when it spawns a coordinator via ``spawn_mission`` it can mount
the right tool capabilities on the workers.

The refresh:

- Pulls fresh entries from ``RepoStateProvider.discovered_extensions.tools``
  on every call (mirrors ``_refresh_available_missions``).
- Skips catalog-only stubs (entries with empty ``capability_fqn``) —
  those are visible to the build-vs-buy advisor via
  ``RepoStateProvider.find_existing_tool`` but cannot be mounted.
- Projects the entry to a planner-shaped dict with
  ``purpose / location / capability / capability_fqn``.
- Mutates ``self._agent.metadata.parameters["available_tools"]`` in place.

These tests pin each rule. They reuse the same detached-capability
+ stub-agent fixture as ``test_session_orchestrator_missions``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.design_monorepo.models import ToolEntry
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


@pytest.fixture
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _make_cap(_exec_ctx) -> SessionOrchestratorCapability:
    return SessionOrchestratorCapability(
        agent=None,
        scope=BlackboardScope.SESSION,
        namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        capability_key="orchestrator_test",
        app_name="test_app",
    )


def _attach_fake_agent(
    cap: SessionOrchestratorCapability,
    *,
    discovered_tools: dict[str, ToolEntry] | None = None,
) -> SimpleNamespace:
    """Wire a synthetic agent + RepoStateProvider stub onto ``cap``.

    ``discovered_tools`` shape is ``{tool_name: ToolEntry}``; pass
    ``None`` for the "no monorepo mounted" case (which produces
    ``{}`` in the parameters dict)."""
    metadata = SimpleNamespace(parameters={})
    if discovered_tools is None:
        agent = SimpleNamespace(
            agent_id="session_agent_xyz",
            metadata=metadata,
            get_capability_by_type=lambda _t: None,
        )
    else:
        provider = MagicMock()
        provider.ensure_materialized = MagicMock(return_value=True)
        provider.discovered_extensions = SimpleNamespace(
            tools=discovered_tools,
            # Mission refresh tests sometimes peek at the same
            # snapshot; populate empty maps to keep the shape stable.
            missions={},
        )
        from polymathera.colony.design_monorepo import RepoStateProvider

        def _gcbt(t):
            return provider if t is RepoStateProvider else None

        agent = SimpleNamespace(
            agent_id="session_agent_xyz",
            metadata=metadata,
            get_capability_by_type=_gcbt,
        )
    cap._agent = agent
    return agent


# ---------------------------------------------------------------------------
# No design monorepo → empty dict (preserve seed shape)
# ---------------------------------------------------------------------------


def test_no_design_monorepo_yields_empty_dict(_exec_ctx) -> None:
    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(cap)
    cap._refresh_available_tools()
    assert agent.metadata.parameters["available_tools"] == {}


def test_detached_capability_is_no_op(_exec_ctx) -> None:
    cap = _make_cap(_exec_ctx)
    cap._agent = None
    # Must not raise; nothing to refresh into.
    cap._refresh_available_tools()


# ---------------------------------------------------------------------------
# L4 tools surface — only entries with capability_fqn become mountable
# ---------------------------------------------------------------------------


def test_tools_with_capability_fqn_appear_in_planner_dict(_exec_ctx) -> None:
    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(
        cap,
        discovered_tools={
            "openems_fdtd": ToolEntry(
                name="openems_fdtd",
                purpose="opm_meg/em",
                location="subdir:tools/opm_meg/em/openems_fdtd",
                capability="run_em_fdtd",
                capability_fqn=(
                    "polymathera.cps.tools.em.openems.OpenEMSFdtdCapability"
                ),
            ),
        },
    )
    cap._refresh_available_tools()
    available = agent.metadata.parameters["available_tools"]
    assert set(available) == {"openems_fdtd"}
    entry = available["openems_fdtd"]
    assert entry["purpose"] == "opm_meg/em"
    assert entry["location"] == "subdir:tools/opm_meg/em/openems_fdtd"
    assert entry["capability"] == "run_em_fdtd"
    assert entry["capability_fqn"] == (
        "polymathera.cps.tools.em.openems.OpenEMSFdtdCapability"
    )


def test_catalog_only_stubs_are_omitted_from_planner_dict(_exec_ctx) -> None:
    """Entries with empty ``capability_fqn`` are build-vs-buy candidates,
    not mountable — they MUST not appear in the planner-visible dict."""
    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(
        cap,
        discovered_tools={
            "future_solver": ToolEntry(
                name="future_solver",
                purpose="opm_meg/future",
                location="subdir:tools/opm_meg/future/future_solver",
                capability="future_capability",
                capability_fqn="",  # ← stub
            ),
            "shipped_solver": ToolEntry(
                name="shipped_solver",
                purpose="opm_meg/em",
                location="subdir:tools/opm_meg/em/shipped",
                capability="solve_shipped",
                capability_fqn="polymathera.cps.tools.em.shipped.ShippedSolverCapability",
            ),
        },
    )
    cap._refresh_available_tools()
    available = agent.metadata.parameters["available_tools"]
    assert set(available) == {"shipped_solver"}
    assert "future_solver" not in available


def test_refresh_overwrites_previous_snapshot(_exec_ctx) -> None:
    cap = _make_cap(_exec_ctx)
    agent = _attach_fake_agent(
        cap,
        discovered_tools={
            "v1_tool": ToolEntry(
                name="v1_tool",
                purpose="opm_meg/v1",
                location="subdir:tools/opm_meg/v1",
                capability="cap_v1",
                capability_fqn="pkg.v1.Cap",
            ),
        },
    )
    cap._refresh_available_tools()
    assert set(agent.metadata.parameters["available_tools"]) == {"v1_tool"}

    # Subsequent refresh with a different catalog replaces, not merges.
    provider = agent.get_capability_by_type(None)
    # Re-attach a fresh snapshot.
    agent2 = _attach_fake_agent(
        cap,
        discovered_tools={
            "v2_tool": ToolEntry(
                name="v2_tool",
                purpose="opm_meg/v2",
                location="subdir:tools/opm_meg/v2",
                capability="cap_v2",
                capability_fqn="pkg.v2.Cap",
            ),
        },
    )
    cap._refresh_available_tools()
    assert set(agent2.metadata.parameters["available_tools"]) == {"v2_tool"}
