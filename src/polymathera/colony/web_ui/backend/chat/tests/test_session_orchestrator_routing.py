"""Routing-priority regression test for ``SessionOrchestratorCapability.stream_events_to_queue``.

The session orchestrator subscribes to three cross-scope event
patterns. Two of them are pure side-effect mirrors (lifecycle ->
chat banners; human approval -> chat agent_question) and must reach
the policy's **high-priority** queue so the dedicated
``_run_high_priority_loop`` task drains them WITHOUT triggering
main-loop planning iterations. The third (agent_diagnostic) returns
``EventProcessingResult`` planner context and MUST go through the
normal queue so the planner sees it.

Regression captured: an earlier refactor routed every cross-scope
subscription to the normal queue, creating an infinite feedback loop
on this agent's own ``policy:action_started:*`` /
``policy:action_completed:*`` events (each emission re-entered
``plan_step`` with the user's last message still bound as context,
so the planner replayed the welcome message until it hit
max-iterations). See
``colony/session_agent_lifecycle_feedback_loop_audit.md``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.blackboard.protocol import (
    ActionPolicyLifecycleProtocol,
    AgentDiagnosticProtocol,
    HumanApprovalProtocol,
)
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _make_recording_blackboard() -> MagicMock:
    """A stand-in blackboard whose ``stream_events_to_queue`` is a
    sync ``MagicMock`` we can introspect. The capability calls it
    once per (queue, pattern) pair."""

    bb = MagicMock()
    bb.stream_events_to_queue = MagicMock()
    return bb


async def _build_cap(
    agent_bb: MagicMock,
    human_bb: MagicMock,
    diag_bb: MagicMock,
    chat_bb: MagicMock,
) -> SessionOrchestratorCapability:
    """Wire a SessionOrchestratorCapability whose
    ``get_blackboard(scope_id=...)`` calls return the supplied
    recording stand-ins, keyed by the namespace embedded in
    ``scope_id``."""

    cap = SessionOrchestratorCapability(
        agent=None,
        scope=BlackboardScope.SESSION,
        namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        capability_key="orch_test",
        app_name="test_app",
    )

    fake_agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        get_blackboard=AsyncMock(return_value=agent_bb),
    )
    cap._agent = fake_agent

    async def _get_bb(scope_id=None, **_kw):
        if scope_id is None or scope_id == cap.scope_id:
            return chat_bb
        if "human_approval" in scope_id:
            return human_bb
        if "agent_diagnostic" in scope_id:
            return diag_bb
        return chat_bb

    cap.get_blackboard = _get_bb  # type: ignore[method-assign]
    return cap


def _patterns_routed_to(bb: MagicMock, queue: asyncio.Queue) -> list[str]:
    """Project the recorded ``stream_events_to_queue`` calls on
    ``bb`` down to the patterns whose first positional arg was
    ``queue``. Lets each test assert routing without caring about
    call order."""

    out: list[str] = []
    for call in bb.stream_events_to_queue.call_args_list:
        args, kwargs = call
        target = args[0] if args else kwargs.get("queue")
        if target is queue:
            out.append(kwargs.get("pattern") or (args[1] if len(args) > 1 else ""))
    return out


async def test_lifecycle_routes_to_high_priority_queue(_exec_ctx) -> None:
    """``ActionPolicyLifecycleProtocol`` events on the agent's primary
    scope MUST reach the high-priority queue. Routing them to the
    normal queue forms a feedback loop because the policy emits the
    very events the handler subscribes to."""

    agent_bb = _make_recording_blackboard()
    human_bb = _make_recording_blackboard()
    diag_bb = _make_recording_blackboard()
    chat_bb = _make_recording_blackboard()
    cap = await _build_cap(agent_bb, human_bb, diag_bb, chat_bb)

    normal_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    high_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    await cap.stream_events_to_queue(normal_q, high_priority_queue=high_q)

    on_high = _patterns_routed_to(agent_bb, high_q)
    on_normal = _patterns_routed_to(agent_bb, normal_q)
    assert ActionPolicyLifecycleProtocol.all_pattern() in on_high
    assert ActionPolicyLifecycleProtocol.all_pattern() not in on_normal


async def test_human_approval_routes_to_high_priority_queue(_exec_ctx) -> None:
    """``HumanApprovalProtocol`` requests are translated into a chat
    record only (no planner context). They belong on the high-priority
    lane so the planner is not iterated on every approval request."""

    agent_bb = _make_recording_blackboard()
    human_bb = _make_recording_blackboard()
    diag_bb = _make_recording_blackboard()
    chat_bb = _make_recording_blackboard()
    cap = await _build_cap(agent_bb, human_bb, diag_bb, chat_bb)

    normal_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    high_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    await cap.stream_events_to_queue(normal_q, high_priority_queue=high_q)

    on_high = _patterns_routed_to(human_bb, high_q)
    on_normal = _patterns_routed_to(human_bb, normal_q)
    assert HumanApprovalProtocol.request_pattern() in on_high
    assert HumanApprovalProtocol.request_pattern() not in on_normal


async def test_agent_diagnostic_routes_to_normal_queue(_exec_ctx) -> None:
    """``AgentDiagnosticProtocol`` events produce
    ``EventProcessingResult`` planner context (see
    :meth:`handle_agent_diagnostic` — guardrail_block_streak branch).
    They MUST reach the main planning loop, i.e. the normal queue."""

    agent_bb = _make_recording_blackboard()
    human_bb = _make_recording_blackboard()
    diag_bb = _make_recording_blackboard()
    chat_bb = _make_recording_blackboard()
    cap = await _build_cap(agent_bb, human_bb, diag_bb, chat_bb)

    normal_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    high_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    await cap.stream_events_to_queue(normal_q, high_priority_queue=high_q)

    on_high = _patterns_routed_to(diag_bb, high_q)
    on_normal = _patterns_routed_to(diag_bb, normal_q)
    assert AgentDiagnosticProtocol.event_pattern() in on_normal
    assert AgentDiagnosticProtocol.event_pattern() not in on_high


async def test_lifecycle_falls_back_to_event_queue_when_no_high_queue(
    _exec_ctx,
) -> None:
    """When the caller does not supply ``high_priority_queue``, the
    override must mirror the base method's degraded-but-functional
    fallback: high-priority patterns land on the single ``event_queue``
    so older callers (and tests) keep working. The feedback-loop risk
    only applies when the policy IS providing the high-priority lane —
    the policy always does in production."""

    agent_bb = _make_recording_blackboard()
    human_bb = _make_recording_blackboard()
    diag_bb = _make_recording_blackboard()
    chat_bb = _make_recording_blackboard()
    cap = await _build_cap(agent_bb, human_bb, diag_bb, chat_bb)

    only_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    await cap.stream_events_to_queue(only_q)

    assert ActionPolicyLifecycleProtocol.all_pattern() in _patterns_routed_to(
        agent_bb, only_q,
    )
    assert HumanApprovalProtocol.request_pattern() in _patterns_routed_to(
        human_bb, only_q,
    )
    assert AgentDiagnosticProtocol.event_pattern() in _patterns_routed_to(
        diag_bb, only_q,
    )


async def test_handler_priorities_are_consistent_with_routing(
    _exec_ctx,
) -> None:
    """The ``@event_handler(priority=...)`` decorations and the
    override's routing must agree. If a handler's priority is changed
    without updating the routing (or vice versa), the feedback-loop
    fix silently regresses."""

    cap = SessionOrchestratorCapability(
        agent=None, scope=BlackboardScope.SESSION,
        capability_key="orch_check", app_name="test_app",
    )
    high = set(cap.high_priority_patterns)
    normal = set(cap.normal_priority_patterns)

    assert ActionPolicyLifecycleProtocol.all_pattern() in high
    assert HumanApprovalProtocol.request_pattern() in high
    assert AgentDiagnosticProtocol.event_pattern() in normal
