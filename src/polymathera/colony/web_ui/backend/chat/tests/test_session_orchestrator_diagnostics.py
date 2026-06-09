"""Tests for the SessionOrchestratorCapability's diagnostic-event
handler (item 2c of
``colony/approval_gate_persistence_and_diagnostic_events_plan.md``)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK,
)
from polymathera.colony.agents.patterns.events import PROCESSED
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _make_orchestrator(self_agent_id: str = "agent-session"):
    cap = SessionOrchestratorCapability(
        agent=None,
        scope=BlackboardScope.SESSION,
        namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        capability_key="orchestrator_test",
        app_name="test_app",
    )
    cap._agent = MagicMock()
    cap._agent.agent_id = self_agent_id
    return cap


async def test_guardrail_block_streak_surfaces_as_planner_context() -> None:
    cap = _make_orchestrator()
    event = SimpleNamespace(
        key=AgentDiagnosticProtocol.event_key(
            "agent-child", DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK, 1,
        ),
        value={
            "agent_id": "agent-child",
            "action_key": "DesignProcessCapability.create_decomposition",
            "count": 3,
            "reason": "no recorded approval",
            "suggestion": "request_human_approval(action_type=...)",
        },
    )
    result = await cap.handle_agent_diagnostic(event, None)
    assert result is not None
    assert result.context_key == (
        "agent_diagnostic:agent-child:guardrail_block_streak:1"
    )
    assert result.context["producer_agent_id"] == "agent-child"
    assert result.context["action_key"] == (
        "DesignProcessCapability.create_decomposition"
    )
    assert result.context["count"] == 3


async def test_ignores_diagnostics_about_self() -> None:
    cap = _make_orchestrator(self_agent_id="agent-session")
    event = SimpleNamespace(
        key=AgentDiagnosticProtocol.event_key(
            "agent-session", DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK, 1,
        ),
        value={
            "action_key": "X", "count": 3,
            "reason": "r", "suggestion": "s",
        },
    )
    assert await cap.handle_agent_diagnostic(event, None) is PROCESSED


async def test_ignores_unknown_kind() -> None:
    cap = _make_orchestrator()
    event = SimpleNamespace(
        key=AgentDiagnosticProtocol.event_key(
            "agent-child", "some_future_kind", 1,
        ),
        value={},
    )
    assert await cap.handle_agent_diagnostic(event, None) is PROCESSED


async def test_ignores_alien_key() -> None:
    """Keys that don't match the decorator's pattern filter never
    reach the handler body — the ``@event_handler`` wrapper returns
    None, not PROCESSED."""

    cap = _make_orchestrator()
    event = SimpleNamespace(
        key="human_approval:request:appr_xyz",
        value={},
    )
    assert await cap.handle_agent_diagnostic(event, None) is None
