"""Tests for FIX-B: ``SessionOrchestratorCapability`` subscribes to
``LifecycleSignalProtocol.terminated_pattern()`` on the colony
control-plane lifecycle scope and surfaces child-coordinator
terminations to the planner via ``EventProcessingResult`` — no
polling of ``get_agent_status``.

The payload-side change adds ``parent_agent_id`` + ``stop_reason``
to ``AgentTerminationEvent`` so subscribers can filter on parent
without an extra ``fetch_agent_info`` round-trip. The handler
filters to "agents I spawned" before binding planner context, so
the SessionAgent doesn't react to every colony-wide termination.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.blackboard.protocol import (
    LifecycleSignalProtocol,
)
from polymathera.colony.agents.scopes import BlackboardScope, MemoryScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _exec_ctx():
    """``MemoryScope.colony_control_plane`` resolves from the ambient
    syscontext AND ``AgentTerminationEvent``'s ``syscontext`` field
    is constructed via ``require_execution_context``; every test in
    this module needs one set."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _make_cap(parent_agent_id: str = "session_agent_xyz") -> SessionOrchestratorCapability:
    cap = SessionOrchestratorCapability(
        agent=None, scope=BlackboardScope.SESSION,
        capability_key="orch_test",
        app_name="test_app",
    )
    cap._agent = SimpleNamespace(agent_id=parent_agent_id)
    return cap


def _terminated_event(
    *,
    terminated_agent_id: str,
    parent_agent_id: str | None,
    agent_type: str = "ProjectPlanningCoordinator",
    stop_reason: str = "policy_completed",
) -> BlackboardEvent:
    return BlackboardEvent(
        event_type="write",
        key=LifecycleSignalProtocol.terminated_key(terminated_agent_id),
        value={
            "agent_id": terminated_agent_id,
            "agent_type": agent_type,
            "parent_agent_id": parent_agent_id,
            "stop_reason": stop_reason,
            "timestamp": 1234567890.0,
            "memory_scopes": [],
        },
    )


# ---------------------------------------------------------------------------
# Handler — filtering + context shape
# ---------------------------------------------------------------------------


async def test_handler_returns_planner_context_when_parent_matches(_exec_ctx) -> None:
    """A termination whose ``parent_agent_id`` matches this
    SessionAgent's ``agent_id`` produces an EventProcessingResult
    whose context binding carries the terminated agent's id +
    stop_reason so the planner LLM sees on its next iteration that
    the child is gone."""

    cap = _make_cap(parent_agent_id="session_agent_xyz")
    event = _terminated_event(
        terminated_agent_id="agent-0e22639f",
        parent_agent_id="session_agent_xyz",
    )

    result = await cap.handle_child_agent_terminated(event, None)

    assert result is not None
    assert (
        getattr(result, "context_key", None)
        == "child_agent_terminated:agent-0e22639f"
    )
    ctx = getattr(result, "context", {}) or {}
    assert ctx["terminated_agent_id"] == "agent-0e22639f"
    assert ctx["agent_type"] == "ProjectPlanningCoordinator"
    assert ctx["stop_reason"] == "policy_completed"


async def test_handler_skips_when_parent_does_not_match(_exec_ctx) -> None:
    """The colony-wide stream carries every termination; the handler
    MUST skip events whose ``parent_agent_id`` does not match this
    SessionAgent. Otherwise every chat session would receive every
    other session's terminations."""

    from polymathera.colony.agents.patterns.events import PROCESSED
    cap = _make_cap(parent_agent_id="session_agent_xyz")
    event = _terminated_event(
        terminated_agent_id="agent-other",
        parent_agent_id="session_agent_other",  # different parent
    )

    result = await cap.handle_child_agent_terminated(event, None)

    assert result is PROCESSED


async def test_handler_skips_when_parent_field_missing(_exec_ctx, caplog) -> None:
    """D8a: a termination event with no ``parent_agent_id`` skips
    binding context (it isn't actionable as a child termination), but
    MUST log a WARN — ``None`` can mean either a legitimate root agent
    OR a spawn-path bug that silently failed to populate
    ``parent_agent_id``. The prior silent-skip masked the second case
    and let spawn-site bugs hide. Pin the warn so the dev sees the
    offending agent_id in logs."""

    import logging
    from polymathera.colony.agents.patterns.events import PROCESSED
    cap = _make_cap(parent_agent_id="session_agent_xyz")
    event = _terminated_event(
        terminated_agent_id="agent-root",
        parent_agent_id=None,
    )

    caplog.set_level(
        logging.WARNING,
        logger="polymathera.colony.web_ui.backend.chat.session_agent",
    )
    result = await cap.handle_child_agent_terminated(event, None)

    assert result is PROCESSED
    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    assert any(
        "agent-root" in m and "parent_agent_id=None" in m
        for m in warnings
    ), (
        "D8a: termination event with parent_agent_id=None must log a "
        "WARN naming the offending agent so the spawn-site bug surfaces."
    )


# ---------------------------------------------------------------------------
# stream_events_to_queue — subscription is wired on the right scope
# ---------------------------------------------------------------------------


async def test_lifecycle_subscription_lands_on_colony_control_plane_scope(
    _exec_ctx,
) -> None:
    """The subscription MUST resolve the BB at the colony
    control-plane lifecycle scope — that's where
    ``LifecycleSignalProtocol.terminated_key`` is written by
    ``MemoryLifecycleHooks._emit_termination_event``. A scope
    mismatch returns an empty stream and the handler never fires."""

    import asyncio

    cap = _make_cap()
    # Stand-in agent + capability blackboards so the existing
    # subscriptions in the override don't crash.
    agent_bb = MagicMock()
    agent_bb.stream_events_to_queue = MagicMock()
    cap._agent = SimpleNamespace(
        agent_id="session_agent_xyz",
        get_blackboard=AsyncMock(return_value=agent_bb),
    )

    captured_scopes: list[str] = []
    captured_streams: list[tuple[str, str]] = []  # (scope_id, pattern)

    async def _get_bb(scope_id: str | None = None, **_kw: Any) -> MagicMock:
        captured_scopes.append(scope_id or "<default>")
        bb = MagicMock()
        # Record what pattern was streamed against this scope.
        def _stream(_queue, *, pattern, event_types=None, **__):
            captured_streams.append((scope_id or "<default>", pattern))
        bb.stream_events_to_queue = _stream
        return bb

    cap.get_blackboard = _get_bb  # type: ignore[method-assign]

    normal_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    high_q: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
    await cap.stream_events_to_queue(normal_q, high_priority_queue=high_q)

    expected_scope = MemoryScope.colony_control_plane("lifecycle")
    expected_pattern = LifecycleSignalProtocol.terminated_pattern()

    assert (expected_scope, expected_pattern) in captured_streams, (
        f"Lifecycle subscription not wired on the expected scope. "
        f"Captured: {captured_streams}"
    )


# ---------------------------------------------------------------------------
# Producer side — AgentTerminationEvent payload carries parent + reason
# ---------------------------------------------------------------------------


def test_agent_termination_event_carries_parent_agent_id_and_reason() -> None:
    """The producer must include ``parent_agent_id`` and
    ``stop_reason`` so the SessionAgent's filter is a payload check,
    not an ``fetch_agent_info`` round-trip. Without these fields the
    handler would have to call back into the registry for every
    termination event in the colony — defeating the point of the
    event-driven path."""

    from polymathera.colony.agents.patterns.memory.lifecycle import (
        AgentTerminationEvent,
    )

    fields = set(AgentTerminationEvent.model_fields.keys())
    assert "parent_agent_id" in fields
    assert "stop_reason" in fields

    # And both default to None so the model still constructs for
    # standalone / pre-fix-B callers.
    sample = AgentTerminationEvent(
        agent_id="a",
        agent_type="T",
        timestamp=0.0,
        memory_scopes=[],
    )
    assert sample.parent_agent_id is None
    assert sample.stop_reason is None


def test_emit_termination_event_passes_parent_and_reason_from_agent() -> None:
    """Source-pin: ``MemoryLifecycleHooks._emit_termination_event``
    constructs the payload with ``parent_agent_id`` from
    ``agent.metadata.parent_agent_id`` and ``stop_reason`` from the
    hook context's call args (D2 fix — was previously reading from
    ``agent._stop_reason`` which a BEFORE hook sees as the stale
    initial value because Agent.stop's body hasn't run yet). A future
    refactor that drops either field silently breaks the
    SessionAgent's filter."""

    from pathlib import Path
    # parents[0]=tests/ parents[1]=chat/ parents[2]=backend/
    # parents[3]=web_ui/ parents[4]=colony/
    src = (
        Path(__file__).resolve().parents[4]
        / "agents" / "patterns" / "memory" / "lifecycle.py"
    ).read_text(encoding="utf-8")
    assert (
        "parent_agent_id=agent.metadata.parent_agent_id" in src
    ), "Producer must pass parent_agent_id from agent metadata."
    assert (
        "ctx.kwargs.get(\"reason\")" in src
    ), (
        "D2: producer MUST read stop_reason from HookContext kwargs, "
        "NOT from agent._stop_reason — BEFORE-hook timing prevents "
        "the instance field from being set when this runs."
    )
    # D2: agent._stop_reason MUST NOT be ASSIGNED to the payload
    # field — that's the bug shape we just fixed (BEFORE hook reads
    # the still-empty initial value). The substring may legitimately
    # appear in the comment that explains the bug; match the
    # assignment pattern specifically.
    assert (
        "stop_reason=agent._stop_reason" not in src
    ), (
        "D2: agent._stop_reason MUST NOT be assigned as the payload "
        "field — BEFORE-hook timing prevents the instance field from "
        "being set when this runs. Read from ctx.kwargs / ctx.args."
    )


@pytest.mark.asyncio
async def test_emit_termination_event_reads_reason_from_hook_kwargs(
) -> None:
    """D2 runtime pin: feed a synthesised HookContext with
    ``kwargs={'reason': 'policy_completed'}`` and verify the
    constructed AgentTerminationEvent carries that value. The
    bug-shape: BEFORE hook reads agent._stop_reason (still empty)
    instead of ctx.kwargs['reason'] (the actual reason being
    passed to Agent.stop)."""

    from unittest.mock import AsyncMock, MagicMock
    from types import SimpleNamespace

    from polymathera.colony.agents.patterns.memory.lifecycle import (
        MemoryLifecycleHooks,
    )

    fake_agent = MagicMock()
    fake_agent.agent_id = "agent-test"
    fake_agent.agent_type = "TestAgent"
    fake_agent.syscontext = SimpleNamespace(
        tenant_id="t1", colony_id="c1", session_id="s1",
    )
    fake_agent.metadata = MagicMock()
    fake_agent.metadata.parent_agent_id = "parent-X"
    fake_agent.metadata.session_id = "s1"
    fake_agent.metadata.tenant_id = "t1"
    fake_agent.metadata.colony_id = "c1"
    fake_agent.metadata.created_at = 0.0
    fake_agent._stop_reason = ""  # the BUG: empty when BEFORE hook fires

    captured: dict = {}

    async def _write(*, key, value, **kw):
        captured["key"] = key
        captured["value"] = value

    fake_bb = MagicMock()
    fake_bb.write = _write
    fake_agent.get_blackboard = AsyncMock(return_value=fake_bb)

    hooks = MemoryLifecycleHooks.__new__(MemoryLifecycleHooks)
    hooks._agent = fake_agent

    ctx = MagicMock()
    ctx.instance = fake_agent
    ctx.args = ()
    ctx.kwargs = {"reason": "policy_completed"}

    await hooks._emit_termination_event(ctx)

    assert captured["value"]["stop_reason"] == "policy_completed"
    assert captured["value"]["parent_agent_id"] == "parent-X"


@pytest.mark.asyncio
async def test_emit_termination_event_reads_reason_from_positional_args(
) -> None:
    """D2 fallback: if Agent.stop is called positionally (e.g.
    ``agent.stop("error")``), the hook reads ``ctx.args[0]`` instead
    of ``ctx.kwargs['reason']``. Both forms must work because
    Python's bound-method dispatch can present args either way."""

    from unittest.mock import AsyncMock, MagicMock
    from types import SimpleNamespace

    from polymathera.colony.agents.patterns.memory.lifecycle import (
        MemoryLifecycleHooks,
    )

    fake_agent = MagicMock()
    fake_agent.agent_id = "agent-test"
    fake_agent.agent_type = "TestAgent"
    fake_agent.syscontext = SimpleNamespace(
        tenant_id="t1", colony_id="c1", session_id="s1",
    )
    fake_agent.metadata = MagicMock()
    fake_agent.metadata.parent_agent_id = None
    fake_agent._stop_reason = ""

    captured: dict = {}

    async def _write(*, key, value, **kw):
        captured["value"] = value

    fake_bb = MagicMock()
    fake_bb.write = _write
    fake_agent.get_blackboard = AsyncMock(return_value=fake_bb)

    hooks = MemoryLifecycleHooks.__new__(MemoryLifecycleHooks)
    hooks._agent = fake_agent

    ctx = MagicMock()
    ctx.instance = fake_agent
    ctx.args = ("error",)  # positional reason
    ctx.kwargs = {}

    await hooks._emit_termination_event(ctx)

    assert captured["value"]["stop_reason"] == "error"
