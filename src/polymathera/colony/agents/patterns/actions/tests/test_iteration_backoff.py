"""Integration test: ``BaseActionPolicy._execute_iteration_inner``
catches :class:`LLMInferenceError` raised by ``plan_step`` and
returns an idle iteration result after backoff.

This is the wire-up that closes the 2026-06-09 storm: without it the
unit tests on :class:`LLMFailureBackoff` pass but the live system
still hammers a credit-out cluster. With it, ``plan_step``'s LLM
failure becomes idle-wait, the outer agent loop skips the iteration
counter (per ``AgentMetadata.idle_wait_counter`` contract), and the
next iteration only fires after ``next_delay_s`` seconds.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymathera.colony.agents.patterns.actions.policies import (
    BaseActionPolicy,
)
from polymathera.colony.cluster.errors import LLMInferenceError


pytestmark = pytest.mark.asyncio


def _agent() -> SimpleNamespace:
    agent = MagicMock()
    agent.agent_id = "agent-backoff-test"
    agent.metadata = SimpleNamespace(idle_wait_counter=0)
    return agent


async def test_execute_iteration_catches_llm_inference_error_returns_idle() -> None:
    policy = BaseActionPolicy(agent=_agent())
    # Aggressive timings so the test sleep is negligible.
    policy._llm_failure_backoff._initial_delay_s = 0.001
    policy._llm_failure_backoff._cap_delay_s = 0.001
    policy._create_action_dispatcher = AsyncMock()

    async def _bad_plan_step(_state):
        raise LLMInferenceError(request_id="r1", message="credit balance is too low")

    policy.plan_step = _bad_plan_step  # type: ignore[assignment]
    state = MagicMock()
    state.custom = {}

    with patch("asyncio.sleep") as fake_sleep:
        result = await policy._execute_iteration_inner(state)

    assert result.idle is True
    assert result.policy_completed is False
    assert result.action_executed is None
    assert policy.agent.metadata.idle_wait_counter == 1
    fake_sleep.assert_awaited()


async def test_execute_iteration_resets_backoff_on_success() -> None:
    """When ``plan_step`` succeeds after a failure streak, the backoff
    must drop the idle-wait token + reset its delay."""

    policy = BaseActionPolicy(agent=_agent())
    policy._llm_failure_backoff._initial_delay_s = 0.001
    policy._llm_failure_backoff._cap_delay_s = 0.001
    policy._create_action_dispatcher = AsyncMock()
    state = MagicMock()
    state.custom = {}

    # First: failure → idle-wait incremented.
    async def _bad_plan_step(_state):
        raise LLMInferenceError(request_id="r1", message="x")

    policy.plan_step = _bad_plan_step  # type: ignore[assignment]
    with patch("asyncio.sleep"):
        await policy._execute_iteration_inner(state)
    assert policy.agent.metadata.idle_wait_counter == 1

    # Then: success → idle-wait released.
    async def _good_plan_step(_state):
        return None  # idle-empty plan path; returns None and policy keeps running

    policy.plan_step = _good_plan_step  # type: ignore[assignment]
    await policy._execute_iteration_inner(state)
    assert policy.agent.metadata.idle_wait_counter == 0
    snap = policy._llm_failure_backoff.snapshot()
    assert snap["in_backoff_streak"] is False
    assert snap["next_delay_s"] == 0.0


async def test_non_llm_exception_propagates_no_backoff() -> None:
    """Generic exceptions in ``plan_step`` (e.g. a bug in user code)
    must NOT be swallowed by the backoff path — only typed
    :class:`LLMInferenceError` triggers it. Other failures should
    surface so the existing error-handling path catches them."""

    policy = BaseActionPolicy(agent=_agent())
    policy._create_action_dispatcher = AsyncMock()
    state = MagicMock()
    state.custom = {}

    async def _raise(_state):
        raise RuntimeError("unrelated bug")

    policy.plan_step = _raise  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="unrelated bug"):
        await policy._execute_iteration_inner(state)
    assert policy.agent.metadata.idle_wait_counter == 0


async def test_status_snapshot_includes_backoff_state() -> None:
    """``/status`` needs to surface 'waiting for LLM cluster recovery,
    next attempt in 32s' — that data comes from the backoff snapshot
    propagated into ``get_status_snapshot``."""

    policy = BaseActionPolicy(agent=_agent())
    policy._llm_failure_backoff._initial_delay_s = 0.001
    policy._llm_failure_backoff._cap_delay_s = 0.001
    with patch("asyncio.sleep"):
        await policy._llm_failure_backoff.handle_failure(
            LLMInferenceError(request_id="r1", message="oops"),
        )
    snap = policy.get_status_snapshot()
    assert "llm_failure_backoff" in snap
    assert snap["llm_failure_backoff"]["in_backoff_streak"] is True
    assert snap["llm_failure_backoff"]["failure_count"] == 1
