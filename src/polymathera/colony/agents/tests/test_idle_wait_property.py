"""Tests for the ``Agent.idle_wait_counter`` field + the
``Agent.is_idle_waiting`` derived property.

Item 6 of ``colony/decompose_and_session_recovery_fixes_plan.md``: the
counter is the agent-loop's signal to skip an iteration's count
toward ``max_iterations`` when the step was a passive poll. The
capability-level integration is covered in
``test_idle_wait_counter.py``; this file pins the field contract
(default 0, ``ge=0``, ``is_idle_waiting`` derived correctly)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from polymathera.colony.agents.base import Agent
from polymathera.colony.agents.models import AgentMetadata
from polymathera.colony.distributed.ray_utils.serving.context import (
    user_execution_context,
)


@pytest.fixture
def _exec_ctx():
    with user_execution_context(
        tenant_id="tenant_test",
        colony_id="colony_test",
        session_id="session_test",
        origin="test",
    ) as ctx:
        yield ctx


def _make_agent(_exec_ctx) -> Agent:
    return Agent(
        agent_id="agent-test",
        agent_type="test_agent",
        metadata=AgentMetadata(),
    )


def test_idle_wait_counter_defaults_to_zero(_exec_ctx) -> None:
    agent = _make_agent(_exec_ctx)
    assert agent.idle_wait_counter == 0
    assert agent.is_idle_waiting is False


def test_is_idle_waiting_true_when_counter_positive(_exec_ctx) -> None:
    agent = _make_agent(_exec_ctx)
    agent.idle_wait_counter = 1
    assert agent.is_idle_waiting is True

    agent.idle_wait_counter = 5
    assert agent.is_idle_waiting is True

    agent.idle_wait_counter = 0
    assert agent.is_idle_waiting is False


def test_idle_wait_counter_rejects_negative_at_construction(_exec_ctx) -> None:
    """Pydantic's ``ge=0`` validator must catch negative values at
    construction — a misbehaving caller can't create an agent
    with a negative counter."""

    with pytest.raises(ValidationError):
        Agent(
            agent_id="agent-test",
            agent_type="test_agent",
            metadata=AgentMetadata(),
            idle_wait_counter=-1,
        )


def test_idle_wait_counter_excluded_from_serialization(_exec_ctx) -> None:
    """Counter is exclude=True so it doesn't travel through Pydantic's
    JSON serialization (its lifetime is per-agent-instance, not
    persistent state)."""

    agent = _make_agent(_exec_ctx)
    agent.idle_wait_counter = 3
    dumped = agent.model_dump()
    assert "idle_wait_counter" not in dumped
