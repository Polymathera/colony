"""Tests for the LLM-cluster-failure backoff (item 3 of
``colony/load_design_context_and_retry_storm_plan.md``).

The 2026-06-09 live run: the API credit ran out at 03:01:55 and the
coordinator made 4365 retry attempts at median 1s intervals over the
following 1h 36min — ~530 ERROR log lines per minute, no backoff,
``max_iterations`` never tripped (each attempt counted). The fix:
the policy catches :class:`LLMInferenceError`, calls
:meth:`LLMFailureBackoff.handle_failure`, and the framework's
``idle_wait_counter`` mechanism absorbs the wait without consuming
the agent's iteration budget.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from polymathera.colony.agents.patterns.actions.llm_failure_backoff import (
    LLMFailureBackoff,
)
from polymathera.colony.cluster.errors import LLMInferenceError


pytestmark = pytest.mark.asyncio


def _make_agent() -> SimpleNamespace:
    return SimpleNamespace(
        agent_id="agent-test",
        metadata=SimpleNamespace(idle_wait_counter=0),
    )


async def test_first_failure_increments_idle_wait_counter() -> None:
    """The framework's iteration accounting reads
    ``idle_wait_counter`` at the end of each ``run_step`` and skips
    the iteration counter when truthy. The first failure of a streak
    must move the agent into idle-wait so the credit-out window
    doesn't burn through ``max_iterations``."""

    agent = _make_agent()
    backoff = LLMFailureBackoff(agent, initial_delay_s=0.01, cap_delay_s=0.02)
    with patch("asyncio.sleep") as fake_sleep:
        await backoff.handle_failure(LLMInferenceError(request_id="r1", message="x"))
    assert agent.metadata.idle_wait_counter == 1
    fake_sleep.assert_awaited_once()


async def test_streak_increments_idle_wait_only_once() -> None:
    """Repeated failures in the same streak must NOT bump the counter
    each time — that would only deepen the polling debt without
    matching decrements."""

    agent = _make_agent()
    backoff = LLMFailureBackoff(agent, initial_delay_s=0.01, cap_delay_s=0.02)
    with patch("asyncio.sleep"):
        for _ in range(5):
            await backoff.handle_failure(
                LLMInferenceError(request_id="r", message="x"),
            )
    assert agent.metadata.idle_wait_counter == 1


async def test_success_decrements_idle_wait_counter() -> None:
    """First success after a failure streak releases the
    ``idle_wait_counter`` token + resets backoff."""

    agent = _make_agent()
    backoff = LLMFailureBackoff(agent, initial_delay_s=0.01, cap_delay_s=0.02)
    with patch("asyncio.sleep"):
        await backoff.handle_failure(LLMInferenceError(request_id="r1", message="x"))
        await backoff.handle_failure(LLMInferenceError(request_id="r2", message="x"))
    backoff.record_success()
    assert agent.metadata.idle_wait_counter == 0
    assert backoff.snapshot()["in_backoff_streak"] is False
    assert backoff.snapshot()["next_delay_s"] == 0.0


async def test_delay_doubles_then_caps() -> None:
    """Backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (cap), 60s, ..."""

    agent = _make_agent()
    backoff = LLMFailureBackoff(
        agent, initial_delay_s=1.0, cap_delay_s=60.0,
    )
    captured: list[float] = []

    async def _fake_sleep(s: float) -> None:
        captured.append(s)

    with patch(
        "polymathera.colony.agents.patterns.actions.llm_failure_backoff."
        "asyncio.sleep",
        side_effect=_fake_sleep,
    ):
        for _ in range(8):
            await backoff.handle_failure(
                LLMInferenceError(request_id="r", message="x"),
            )

    assert captured == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


async def test_record_success_with_no_open_streak_is_noop() -> None:
    """Calling ``record_success`` when no failure happened must not
    decrement the counter (it was never incremented) or otherwise
    drift state."""

    agent = SimpleNamespace(
        agent_id="agent-test",
        metadata=SimpleNamespace(idle_wait_counter=5),
    )
    backoff = LLMFailureBackoff(agent)
    backoff.record_success()
    assert agent.metadata.idle_wait_counter == 5


async def test_snapshot_shape() -> None:
    """``get_status_snapshot`` includes the backoff state under
    ``llm_failure_backoff`` so ``/status`` can show 'waiting for LLM
    cluster recovery, next attempt in 32s'."""

    agent = _make_agent()
    backoff = LLMFailureBackoff(agent, initial_delay_s=0.01, cap_delay_s=0.02)
    with patch("asyncio.sleep"):
        await backoff.handle_failure(
            LLMInferenceError(request_id="r1", message="credit balance"),
        )
    snap = backoff.snapshot()
    assert snap["in_backoff_streak"] is True
    assert snap["failure_count"] == 1
    assert snap["next_delay_s"] > 0
    assert snap["last_failure_at"] is not None
    assert "credit balance" in snap["last_error_message"]


async def test_permanent_category_uses_recovery_floor(
) -> None:
    """R7-FIX-B: a permanent-category failure (BILLING / AUTH) raises
    the next_delay to the breaker's recovery floor (300s) instead of
    the exponential 1s start. Run7 had ~5,364 spin-loop retries at
    sub-second cadence into a permanently-open breaker; this floor
    is the consumer-side answer that aligns the retry cadence with
    the breaker's RECOVERY_TIMEOUT."""

    from polymathera.colony.cluster.errors import LLMErrorCategory

    agent = _make_agent()
    backoff = LLMFailureBackoff(agent, initial_delay_s=0.01, cap_delay_s=0.02)
    sleeps: list[float] = []
    with patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        await backoff.handle_failure(
            LLMInferenceError(
                request_id="r1",
                message="credit balance too low",
                category=LLMErrorCategory.BILLING,
            ),
        )
    # The category floor is 300s — far above the configured
    # initial_delay_s=0.01 and cap_delay_s=0.02.
    assert sleeps == [LLMFailureBackoff.PERMANENT_FAILURE_FLOOR_S]
    assert backoff._next_delay_s == LLMFailureBackoff.PERMANENT_FAILURE_FLOOR_S


async def test_transient_category_uses_exponential_not_floor(
) -> None:
    """Counterpart: a transient failure keeps the existing
    exponential backoff. The floor is reserved for permanent
    categories so transient blips don't pause the agent for 5 min."""

    from polymathera.colony.cluster.errors import LLMErrorCategory

    agent = _make_agent()
    backoff = LLMFailureBackoff(agent, initial_delay_s=0.01, cap_delay_s=0.02)
    sleeps: list[float] = []
    with patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        await backoff.handle_failure(
            LLMInferenceError(
                request_id="r1",
                message="rate limited",
                category=LLMErrorCategory.TRANSIENT,
            ),
        )
    assert sleeps == [0.01]  # initial_delay_s, NOT 300s
