"""Tests for the new ``RuntimeGuardrail`` subclasses introduced by the
action-preconditions plan
(``colony/mission_and_action_guardrails_plan.md`` Part 2)."""

from __future__ import annotations

import re
import time

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    ApprovalRequiredGuardrail,
    ArgsAwareOrderingRule,
    ArgsAwareTemporalOrderGuardrail,
    CapabilityBoundaryGuardrail,
    CompositeGuardrail,
    NoGuardrail,
    TemporalOrderGuardrail,
)
from polymathera.colony.agents.patterns.planning.models import CallRecord


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# call_history shape: existing guardrails read ``c.action_key``
# ---------------------------------------------------------------------------


async def test_temporal_order_guardrail_reads_call_record_action_key() -> None:
    """``TemporalOrderGuardrail`` was migrated from ``list[str]`` to
    ``list[CallRecord]``; the rule still matches when the prior call's
    action_key contains the required prefix."""

    g = TemporalOrderGuardrail(ordering_rules=[("analyze", "synthesize")])
    history = [CallRecord(
        action_key="domain.analyze_pages",
        params={},
        end_wall=time.time(),
        status="ok",
    )]
    d = await g.check(
        action_key="domain.synthesize",
        params={},
        call_history=history,
    )
    assert d.allowed

    d2 = await g.check(
        action_key="domain.synthesize",
        params={},
        call_history=[],
    )
    assert not d2.allowed
    assert "requires" in d2.reason


async def test_capability_boundary_guardrail_unchanged_by_migration() -> None:
    """``CapabilityBoundaryGuardrail`` doesn't read call_history at all
    — pin that it stays unchanged after the schema migration."""

    g = CapabilityBoundaryGuardrail(allowed_prefixes=["Analysis"])
    d = await g.check(
        action_key="domain.AnalysisCapability.run", params={}, call_history=[],
    )
    assert d.allowed
    d2 = await g.check(
        action_key="domain.OtherCapability.run", params={}, call_history=[],
    )
    assert not d2.allowed


# ---------------------------------------------------------------------------
# ArgsAwareTemporalOrderGuardrail
# ---------------------------------------------------------------------------


_AGENT_RE = re.compile(r"agent-[0-9a-f]+")


def _status_claim_rule() -> ArgsAwareOrderingRule:
    """The status-claim rule the SessionAgent mounts: gate
    ``respond_to_user`` calls whose content mentions an agent_id on
    a recent matching ``get_agent_status``."""

    return ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=lambda p: bool(_AGENT_RE.search(p.get("content", ""))),
        required_prior="get_agent_status",
        max_age_calls=10,
        prior_params_match=lambda prior, target: any(
            agent_id in (prior.get("agent_ids") or [])
            for agent_id in _AGENT_RE.findall(target.get("content", ""))
        ),
        suggestion=(
            "Call AgentPoolCapability.get_agent_status with the "
            "agent_id you're about to reference, then report."
        ),
    )


async def test_args_aware_allows_when_applies_when_returns_false() -> None:
    """Free-form ``respond_to_user`` ('Hi there!') doesn't match the
    applies_when predicate → the rule doesn't fire even with empty
    history."""

    g = ArgsAwareTemporalOrderGuardrail(rules=[_status_claim_rule()])
    d = await g.check(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params={"content": "Hi there!"},
        call_history=[],
    )
    assert d.allowed


async def test_args_aware_blocks_when_no_prior_match() -> None:
    """Content references agent_id; history is empty → blocked."""

    g = ArgsAwareTemporalOrderGuardrail(rules=[_status_claim_rule()])
    d = await g.check(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params={"content": "agent-abc123 is running"},
        call_history=[],
    )
    assert not d.allowed
    assert "recent" in d.reason
    assert "agent_id" in d.suggestion


async def test_args_aware_blocks_when_prior_matches_different_args() -> None:
    """Recent get_agent_status for a DIFFERENT agent_id doesn't unblock
    a claim about THIS agent_id."""

    g = ArgsAwareTemporalOrderGuardrail(rules=[_status_claim_rule()])
    history = [CallRecord(
        action_key="AgentPoolCapability.get_agent_status",
        params={"agent_ids": ["agent-def456"]},
        end_wall=time.time(),
        status="ok",
    )]
    d = await g.check(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params={"content": "agent-abc123 is running"},
        call_history=history,
    )
    assert not d.allowed


async def test_args_aware_allows_when_prior_matches_args() -> None:
    """Recent get_agent_status for THIS agent_id unblocks the claim."""

    g = ArgsAwareTemporalOrderGuardrail(rules=[_status_claim_rule()])
    history = [CallRecord(
        action_key="AgentPoolCapability.get_agent_status",
        params={"agent_ids": ["agent-abc123"]},
        end_wall=time.time(),
        status="ok",
    )]
    d = await g.check(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params={"content": "agent-abc123 is running"},
        call_history=history,
    )
    assert d.allowed


async def test_args_aware_respects_max_age_calls() -> None:
    """A prior call older than ``max_age_calls`` doesn't count."""

    rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=None,
        required_prior="get_agent_status",
        max_age_calls=2,
        prior_params_match=None,
        suggestion="...",
    )
    g = ArgsAwareTemporalOrderGuardrail(rules=[rule])
    history = [
        CallRecord(
            action_key="AgentPoolCapability.get_agent_status",
            params={}, end_wall=time.time(), status="ok",
        ),
        CallRecord(
            action_key="foo.unrelated", params={},
            end_wall=time.time(), status="ok",
        ),
        CallRecord(
            action_key="foo.also_unrelated", params={},
            end_wall=time.time(), status="ok",
        ),
    ]
    # max_age_calls=2 → only the LAST 2 entries are eligible; the
    # get_agent_status one is 3rd-from-last → falls outside the window.
    d = await g.check(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params={"content": "x"},
        call_history=history,
    )
    assert not d.allowed


async def test_args_aware_respects_max_age_seconds() -> None:
    """A prior call older than ``max_age_seconds`` doesn't count, even
    when within the call-count window."""

    rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=None,
        required_prior="get_agent_status",
        max_age_calls=None,
        max_age_seconds=0.5,
        prior_params_match=None,
        suggestion="...",
    )
    g = ArgsAwareTemporalOrderGuardrail(rules=[rule])
    history = [CallRecord(
        action_key="AgentPoolCapability.get_agent_status",
        params={}, end_wall=time.time() - 5.0,
        start_wall=time.time() - 5.0, status="ok",
    )]
    d = await g.check(
        action_key="SessionOrchestratorCapability.respond_to_user",
        params={"content": "x"},
        call_history=history,
    )
    assert not d.allowed


# ---------------------------------------------------------------------------
# ApprovalRequiredGuardrail
# ---------------------------------------------------------------------------


class _FakeCap:
    """Stub for ``HumanApprovalCapability.has_active_approval_for``."""

    def __init__(self, allow: bool = False, rid: str | None = None) -> None:
        self.allow = allow
        self.rid = rid
        self.queried: list[str] = []

    async def has_active_approval_for(
        self, action_key: str,
    ) -> tuple[bool, str | None]:
        self.queried.append(action_key)
        return self.allow, self.rid


def _bind_cap(g: ApprovalRequiredGuardrail, cap) -> None:
    class _A:
        agent_id = "agent-test"

        def get_capability_by_type(self, _cls):
            return cap
    g.bind_agent(_A())


async def test_approval_guardrail_dry_run_is_always_allowed() -> None:
    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=[
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
        ],
    )
    d = await g.check(
        action_key=(
            "DesignProcessCapability.bootstrap_roadmap_from_objectives"
        ),
        params={"dry_run": True},
        call_history=[],
    )
    assert d.allowed


async def test_approval_guardrail_blocks_apply_without_binding() -> None:
    """No bound capability → no approval state → block."""

    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=[
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
        ],
    )
    d = await g.check(
        action_key=(
            "DesignProcessCapability.bootstrap_roadmap_from_objectives"
        ),
        params={"dry_run": False},
        call_history=[],
    )
    assert not d.allowed
    assert "no recorded approval" in d.reason
    assert "request_human_approval" in d.suggestion


async def test_approval_guardrail_blocks_when_cap_reports_no_approval() -> None:
    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=[
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
        ],
    )
    cap = _FakeCap(allow=False)
    _bind_cap(g, cap)
    d = await g.check(
        action_key=(
            "DesignProcessCapability.bootstrap_roadmap_from_objectives"
        ),
        params={"dry_run": False},
        call_history=[],
    )
    assert not d.allowed
    assert cap.queried == [
        "DesignProcessCapability.bootstrap_roadmap_from_objectives",
    ]


async def test_approval_guardrail_allows_when_cap_reports_approval() -> None:
    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=[
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
        ],
    )
    _bind_cap(g, _FakeCap(allow=True, rid="appr_xyz"))
    d = await g.check(
        action_key=(
            "DesignProcessCapability.bootstrap_roadmap_from_objectives"
        ),
        params={"dry_run": False},
        call_history=[],
    )
    assert d.allowed


async def test_approval_guardrail_bind_owner_none_clears_resolver() -> None:
    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=["X.gated"],
    )
    _bind_cap(g, _FakeCap(allow=True))
    g.bind_agent(None)
    d = await g.check(
        action_key="X.gated.apply",
        params={"dry_run": False},
        call_history=[],
    )
    assert not d.allowed


async def test_approval_guardrail_ungated_action_passes() -> None:
    """An action whose key doesn't match any gated prefix is always
    allowed."""

    g = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=["X.gated"],
    )
    d = await g.check(
        action_key="Y.ungated", params={}, call_history=[],
    )
    assert d.allowed


# ---------------------------------------------------------------------------
# CompositeGuardrail
# ---------------------------------------------------------------------------


async def test_composite_short_circuits_on_first_block() -> None:
    """First non-allowed decision wins — guardrails further down don't
    run. The decision's ``reason`` is the blocker's."""

    g = CompositeGuardrail(
        ApprovalRequiredGuardrail(
            approval_required_action_prefixes=["X.gated"],
        ),
        ArgsAwareTemporalOrderGuardrail(rules=[_status_claim_rule()]),
    )
    d = await g.check(
        action_key="X.gated.apply",
        params={"dry_run": False},
        call_history=[],
    )
    assert not d.allowed
    assert "gated" in d.reason  # approval guardrail's message


async def test_composite_allows_when_every_inner_allows() -> None:
    g = CompositeGuardrail(NoGuardrail(), NoGuardrail())
    d = await g.check(
        action_key="anything", params={}, call_history=[],
    )
    assert d.allowed


async def test_composite_requires_at_least_one_guardrail() -> None:
    with pytest.raises(ValueError):
        CompositeGuardrail()


async def test_composite_runs_guardrails_in_order() -> None:
    """The order of inner guardrails matters: blocking happens on the
    first hit. Swap the order and a different blocker wins."""

    rule = _status_claim_rule()
    args_aware = ArgsAwareTemporalOrderGuardrail(rules=[rule])
    approval = ApprovalRequiredGuardrail(
        approval_required_action_prefixes=["X.gated"],
    )

    # Apply call mentioning an agent_id → both guardrails would
    # block. With approval FIRST, the approval message wins.
    g1 = CompositeGuardrail(approval, args_aware)
    d1 = await g1.check(
        action_key="X.gated.respond_to_user",
        params={
            "dry_run": False,
            "content": "agent-abc123 is running",
        },
        call_history=[],
    )
    assert not d1.allowed
    assert "gated behind human approval" in d1.reason

    # With the args_aware FIRST, its message wins.
    g2 = CompositeGuardrail(args_aware, approval)
    d2 = await g2.check(
        action_key="X.gated.respond_to_user",
        params={
            "dry_run": False,
            "content": "agent-abc123 is running",
        },
        call_history=[],
    )
    assert not d2.allowed
    assert "recent" in d2.reason
