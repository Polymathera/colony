"""Tests for ``BuildVsBuyAdvisor``."""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.agents.patterns.capabilities.build_vs_buy_engine import (
    BuildVsBuyAdvisor,
    BuildVsBuyContext,
    BuildVsBuyDecision,
    INNER_LOOP_FREQUENCY_THRESHOLD,
    TeamTrackRecord,
)
from polymathera.colony.tools import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolSpec,
)


pytestmark = pytest.mark.asyncio


def _gui_external() -> ToolSpec:
    return ToolSpec(
        name="commercial_gui_tool",
        capabilities=("simulate",),
        headless=HeadlessReadiness.GUI_PRIMARY,
        hitl_frequency=HITLFrequency.HUMAN_PRIMARY,
        licensing=Licensing.COMMERCIAL,
        determinism=Determinism.STOCHASTIC,
        backend="gui",
    )


def _native_external() -> ToolSpec:
    return ToolSpec(
        name="open_native_tool",
        capabilities=("simulate",),
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        licensing=Licensing.MIT,
        determinism=Determinism.DETERMINISTIC,
        backend="in_process",
    )


async def test_all_rules_satisfied_picks_cross_check() -> None:
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_gui_external(),),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_can_be_differentiable=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    # Cross-check: build is justified, but the external is kept as a
    # benchmark.
    assert v.decision is BuildVsBuyDecision.CROSS_CHECK_ONLY
    assert v.matched_tool is not None
    assert v.matched_tool.name == "commercial_gui_tool"
    assert v.estimated_effort_months and v.estimated_effort_months > 0


async def test_no_external_picks_pure_build() -> None:
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    assert v.decision is BuildVsBuyDecision.BUILD


async def test_native_external_no_inner_loop_picks_buy() -> None:
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_native_external(),),
        inner_loop_call_frequency_per_workflow=5,  # below threshold
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    assert v.decision is BuildVsBuyDecision.BUY
    assert v.matched_tool is not None
    assert v.matched_tool.name == "open_native_tool"


async def test_inner_loop_with_licence_clean_native_external_picks_hybrid() -> None:
    """Tier-A licence-clean external + inner-loop + narrower + validation:
    BUILD is out (R1 ∧ R5 both fail) but the hot-loop + narrower + gold
    benchmark justify a narrow custom path while still keeping the
    external for breadth → HYBRID (master §4.3 spirit)."""

    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_native_external(),),
        inner_loop_call_frequency_per_workflow=INNER_LOOP_FREQUENCY_THRESHOLD,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    assert v.decision is BuildVsBuyDecision.HYBRID
    assert v.matched_tool is not None
    assert v.matched_tool.name == "open_native_tool"


async def test_no_external_no_validation_picks_deny() -> None:
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=False,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    assert v.decision is BuildVsBuyDecision.DENY


async def test_licence_forbidden_excludes_external() -> None:
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_native_external(),),
        inner_loop_call_frequency_per_workflow=5,
        validation_against_gold_feasible=False,
        licence_forbidden=frozenset({Licensing.MIT}),
    )
    v = await advisor.recommend(ctx)
    assert v.decision is BuildVsBuyDecision.DENY


async def test_verdict_carries_per_rule_trace() -> None:
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_gui_external(),),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    rule_ids = {r.rule_id for r in v.rules_evaluated}
    assert rule_ids >= {
        "R0_local_match", "R1_tier_cd", "R2_narrower",
        "R3_validation", "R4_inner_loop", "R5_licence_or_determinism",
        "R6_differentiable",
    }


async def test_local_writable_match_picks_augment() -> None:
    class _StubMatch:
        def __init__(self, entry, writable):
            self.entry = entry
            self.writable = writable

    class _StubEntry:
        def __init__(self, name, location, headless, license_):
            self.name = name
            self.location = location
            self.headless = headless
            self.license = license_

    class _StubProvider:
        async def find_existing_tool(self, q, require_writable=False):
            assert q == "simulate"
            return [
                _StubMatch(
                    _StubEntry("local_tool", "subdir:tools/x/y", "native", "MIT"),
                    writable=True,
                ),
            ]

    advisor = BuildVsBuyAdvisor(repo_state_provider=_StubProvider())
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_gui_external(),),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    assert v.decision is BuildVsBuyDecision.AUGMENT
    assert v.matched_tool is not None
    assert v.matched_tool.name == "local_tool"
    assert v.matched_tool.writable is True


async def test_local_match_not_writable_skipped() -> None:
    class _StubMatch:
        def __init__(self, entry, writable):
            self.entry = entry
            self.writable = writable

    class _StubEntry:
        def __init__(self, name):
            self.name = name
            self.location = ""
            self.headless = "native"
            self.license = "MIT"

    class _StubProvider:
        async def find_existing_tool(self, q, require_writable=False):
            return [_StubMatch(_StubEntry("read_only"), writable=False)]

    advisor = BuildVsBuyAdvisor(repo_state_provider=_StubProvider())
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(_gui_external(),),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    v = await advisor.recommend(ctx)
    assert v.decision is not BuildVsBuyDecision.AUGMENT


async def test_team_competence_scales_effort() -> None:
    advisor = BuildVsBuyAdvisor()
    base_ctx = dict(
        capability_query="simulate",
        available_external_tools=(),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
    )
    inexperienced = await advisor.recommend(BuildVsBuyContext(
        **base_ctx,
        team=TeamTrackRecord(custom_tools_built=4, custom_tools_validated_against_gold=0),
    ))
    experienced = await advisor.recommend(BuildVsBuyContext(
        **base_ctx,
        team=TeamTrackRecord(custom_tools_built=4, custom_tools_validated_against_gold=4),
    ))
    assert inexperienced.estimated_effort_months > experienced.estimated_effort_months


async def test_require_determinism_against_stochastic_ext_fires_r5() -> None:
    """A Tier-A licence-clean *but stochastic* external satisfies R5
    when ``require_determinism=True``. With a Tier-A external R1 still
    fails (so BUILD is out) but R2∧R3∧R4 hold and the external is
    eligible → HYBRID."""

    stochastic_native = ToolSpec(
        name="stochastic_native",
        capabilities=("simulate",),
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        licensing=Licensing.MIT,
        determinism=Determinism.STOCHASTIC,
        backend="in_process",
    )
    advisor = BuildVsBuyAdvisor()
    ctx = BuildVsBuyContext(
        capability_query="simulate",
        available_external_tools=(stochastic_native,),
        inner_loop_call_frequency_per_workflow=200,
        validation_against_gold_feasible=True,
        custom_problem_narrower_than_external=True,
        require_determinism=True,
    )
    v = await advisor.recommend(ctx)
    by_id = {r.rule_id: r for r in v.rules_evaluated}
    assert by_id["R5_licence_or_determinism"].satisfied is True
    assert "determinism" in by_id["R5_licence_or_determinism"].reason
    assert v.decision is BuildVsBuyDecision.HYBRID
