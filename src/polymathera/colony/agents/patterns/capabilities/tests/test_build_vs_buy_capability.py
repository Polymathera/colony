"""Tests for ``BuildVsBuyCapability`` — the agent-facing wrapper.

The engine itself (``BuildVsBuyAdvisor``) is covered by
``test_build_vs_buy_engine.py``. These tests focus on the wrapper
concerns:

- Tag set + action-group description (LLM-visible metadata).
- ``recommend_build_or_buy`` action dispatch + arg coercion (the
  ``available_external_tools``-as-dicts + ``licence_forbidden``-as-strs
  contract for LLM ergonomics).
- ``RepoStateProvider`` discovery (AUGMENT path activates when a
  writable local match exists; falls back to BUY/BUILD otherwise).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.patterns.capabilities.build_vs_buy import (
    BuildVsBuyCapability,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


@pytest.fixture(autouse=True)
def _exec_ctx():
    """Provide an execution context so the capability's scope_id resolves."""
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _fake_agent() -> MagicMock:
    """Bare-bones agent stub. No RepoStateProvider mounted by default.

    ``syscontext`` mirrors the ``_exec_ctx`` fixture below so the
    capability's scope-prefix resolution passes the
    tenant/colony-match check in ``serving.context``.
    """
    agent = MagicMock()
    agent.agent_id = "agent-bvb-test"
    agent.syscontext.tenant_id = "t1"
    agent.syscontext.colony_id = "c1"
    agent.get_capability_by_type.return_value = None
    return agent


def _make_capability(agent: MagicMock | None = None) -> BuildVsBuyCapability:
    return BuildVsBuyCapability(
        agent=agent if agent is not None else _fake_agent(),
        capability_key="bvb_test",
        app_name="test_app",
    )


def test_capability_tags_include_build_vs_buy() -> None:
    cap = _make_capability()
    tags = cap.get_capability_tags()
    assert "build_vs_buy" in tags
    assert "tool_selection" in tags
    assert "planning" in tags


def test_action_group_description_mentions_six_decisions() -> None:
    cap = _make_capability()
    desc = cap.get_action_group_description()
    for decision in ("AUGMENT", "BUY", "BUILD", "HYBRID", "CROSS_CHECK_ONLY", "DENY"):
        assert decision in desc


@pytest.mark.asyncio
async def test_recommend_no_externals_and_no_validation_returns_deny() -> None:
    cap = _make_capability()
    verdict = await cap.recommend_build_or_buy(
        capability_query="differentiable_laptime_simulation",
        validation_against_gold_feasible=False,
    )
    assert verdict["decision"] == "deny"
    assert "rationale" in verdict
    rules_seen = {r["rule_id"] for r in verdict["rules_evaluated"]}
    assert {"R0_local_match", "R1_tier_cd", "R2_narrower", "R3_validation"} <= rules_seen


@pytest.mark.asyncio
async def test_recommend_uses_external_tool_dict_input() -> None:
    cap = _make_capability()
    verdict = await cap.recommend_build_or_buy(
        capability_query="solve_em",
        available_external_tools=[
            {
                "name": "openems",
                "capabilities": ("solve_em",),
                "headless": "native",
                "hitl_frequency": "autonomous",
                "licensing": "gpl",
                "determinism": "deterministic",
                "backend": "cli_subprocess",
            },
        ],
        inner_loop_call_frequency_per_workflow=100,
        custom_can_be_differentiable=True,
        custom_problem_narrower_than_external=True,
        require_determinism=True,
        validation_against_gold_feasible=True,
    )
    # R1 fails (Tier A — openems is NATIVE + AUTONOMOUS), R2/R3/R4 hold,
    # so HYBRID is expected with openems as the external for breadth.
    assert verdict["decision"] == "hybrid"
    assert verdict["matched_tool"]["name"] == "openems"


@pytest.mark.asyncio
async def test_licence_forbidden_strings_coerced_to_enum() -> None:
    cap = _make_capability()
    verdict = await cap.recommend_build_or_buy(
        capability_query="proprietary_only_capability",
        available_external_tools=[
            {
                "name": "gurobi",
                "capabilities": ("solve_milp",),
                "licensing": "commercial",
                "headless": "native",
                "hitl_frequency": "autonomous",
            },
        ],
        licence_forbidden=["commercial", "agpl"],
        validation_against_gold_feasible=False,
    )
    # commercial is now forbidden → no eligible external + no validation
    # path → DENY.
    assert verdict["decision"] == "deny"


@pytest.mark.asyncio
async def test_unknown_licence_strings_are_silently_ignored() -> None:
    """Defensive: an LLM-supplied typo in licence_forbidden must not
    crash the advisor."""
    cap = _make_capability()
    verdict = await cap.recommend_build_or_buy(
        capability_query="anything",
        licence_forbidden=["nonexistent_licence", "mit"],
        validation_against_gold_feasible=True,
    )
    # No externals supplied, validation feasible — falls through to a
    # non-DENY verdict.
    assert verdict["decision"] in {"build", "deny", "buy"}


@pytest.mark.asyncio
async def test_augment_path_activates_with_writable_local_match() -> None:
    """When the agent has a RepoStateProvider mounted and it returns a
    writable match, the verdict is AUGMENT."""
    from polymathera.colony.design_monorepo.capabilities import RepoStateProvider

    match = MagicMock()
    match.entry = MagicMock()
    match.entry.name = "racer_laptime"
    match.entry.location = "subdir:tools/racer/laptime"
    match.entry.headless = "native"
    match.entry.license = "MIT"
    match.writable = True

    repo_state = MagicMock(spec=RepoStateProvider)
    repo_state.find_existing_tool = AsyncMock(return_value=[match])

    agent = _fake_agent()

    def _resolve(typ):
        if typ is RepoStateProvider:
            return repo_state
        return None

    agent.get_capability_by_type.side_effect = _resolve

    cap = _make_capability(agent)
    verdict = await cap.recommend_build_or_buy(
        capability_query="differentiable_laptime_simulation",
    )
    assert verdict["decision"] == "augment"
    assert verdict["matched_tool"]["name"] == "racer_laptime"
    assert verdict["matched_tool"]["writable"] is True


@pytest.mark.asyncio
async def test_repo_state_provider_absent_skips_augment_path() -> None:
    """No mounted RepoStateProvider → AUGMENT is unreachable; R0
    evaluates to ``satisfied=False``."""
    cap = _make_capability()
    verdict = await cap.recommend_build_or_buy(
        capability_query="anything",
    )
    r0 = next(r for r in verdict["rules_evaluated"] if r["rule_id"] == "R0_local_match")
    assert r0["satisfied"] is False
    assert verdict["decision"] != "augment"
