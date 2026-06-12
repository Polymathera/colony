"""Tests for :class:`EmptyIterationTracker` — the soft backstop that
nudges the LLM toward ``wait_for_next_event`` when it burns empty
planning iterations."""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_EMPTY_ITERATION_STREAK,
)
from polymathera.colony.agents.patterns.actions.empty_iteration_tracker import (
    EmptyIterationTracker,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    user_execution_context,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_ctx():
    with user_execution_context(
        tenant_id="tenant_test",
        colony_id="colony_test",
        session_id="session_test",
        origin="test",
    ) as ctx:
        yield ctx


class _FakeAgent:
    """Minimal agent that records blackboard writes."""

    def __init__(self, agent_id: str = "agent-empty") -> None:
        self.agent_id = agent_id
        self.writes: list[tuple[str, dict[str, Any]]] = []

    async def get_blackboard(self, *, scope_id: str | None = None):
        agent = self

        class _BB:
            async def write(
                self,
                key: str,
                value: dict[str, Any],
                *,
                tags=None,
                metadata=None,
            ):
                agent.writes.append((key, value))

        return _BB()


def _new_tracker(*, threshold: int = 3) -> tuple[EmptyIterationTracker, _FakeAgent]:
    agent = _FakeAgent()
    return EmptyIterationTracker(agent, threshold=threshold), agent


async def test_streak_increments_on_empty_iteration() -> None:
    """An iteration with no actions AND no event observed counts as
    empty and increments the streak."""
    tracker, _agent = _new_tracker()
    await tracker.observe_iteration(
        actions_called_count=0,
        queue_was_empty_at_observation=True,
    )
    assert tracker.snapshot()["streak"] == 1


async def test_streak_resets_on_actions_called() -> None:
    """Any iteration that called at least one action resets the
    streak — the LLM was doing something useful."""
    tracker, _agent = _new_tracker()
    await tracker.observe_iteration(
        actions_called_count=0,
        queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=2,
        queue_was_empty_at_observation=True,
    )
    assert tracker.snapshot()["streak"] == 0


async def test_streak_resets_on_event_observation() -> None:
    """An iteration where the queue had an event (even if no action
    was called) resets the streak — the agent is processing
    incoming events, not idle-spinning."""
    tracker, _agent = _new_tracker()
    await tracker.observe_iteration(
        actions_called_count=0,
        queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=0,
        queue_was_empty_at_observation=False,
    )
    assert tracker.snapshot()["streak"] == 0


async def test_diagnostic_fires_exactly_once_per_streak() -> None:
    """The diagnostic is emitted when the streak crosses threshold,
    and ONLY once per streak — a fourth empty iteration in the same
    streak does not refire."""
    tracker, agent = _new_tracker(threshold=3)
    for _ in range(5):
        await tracker.observe_iteration(
            actions_called_count=0,
            queue_was_empty_at_observation=True,
        )
    assert len(agent.writes) == 1


async def test_diagnostic_can_refire_after_streak_reset() -> None:
    """A productive iteration resets both the streak and the
    "diagnostic already fired" flag, so the NEXT streak gets its
    own diagnostic."""
    tracker, agent = _new_tracker(threshold=2)
    # First streak: fires.
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    assert len(agent.writes) == 1
    # Productive iteration breaks the streak.
    await tracker.observe_iteration(
        actions_called_count=1, queue_was_empty_at_observation=True,
    )
    # New streak: fires again.
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    assert len(agent.writes) == 2


async def test_diagnostic_payload_carries_streak_and_suggestion() -> None:
    """The emitted diagnostic carries the streak length, the
    threshold, and a suggestion pointing the LLM at
    ``wait_for_next_event``."""
    tracker, agent = _new_tracker(threshold=2)
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    assert len(agent.writes) == 1
    key, value = agent.writes[0]
    assert DIAGNOSTIC_EMPTY_ITERATION_STREAK in key
    assert value["kind"] == DIAGNOSTIC_EMPTY_ITERATION_STREAK
    assert value["streak"] == 2
    assert value["threshold"] == 2
    assert value["agent_id"] == "agent-empty"
    assert "wait_for_next_event" in value["suggestion"]


async def test_threshold_is_configurable() -> None:
    """A non-default threshold fires at the configured count."""
    tracker, agent = _new_tracker(threshold=5)
    for _ in range(4):
        await tracker.observe_iteration(
            actions_called_count=0,
            queue_was_empty_at_observation=True,
        )
    assert len(agent.writes) == 0
    await tracker.observe_iteration(
        actions_called_count=0,
        queue_was_empty_at_observation=True,
    )
    assert len(agent.writes) == 1


async def test_event_key_matches_protocol_format() -> None:
    """The key is well-formed under ``AgentDiagnosticProtocol``."""
    tracker, agent = _new_tracker(threshold=1)
    await tracker.observe_iteration(
        actions_called_count=0,
        queue_was_empty_at_observation=True,
    )
    key, _ = agent.writes[0]
    parsed = AgentDiagnosticProtocol.parse_event_key(key)
    assert parsed["agent_id"] == "agent-empty"
    assert parsed["kind"] == DIAGNOSTIC_EMPTY_ITERATION_STREAK


async def test_snapshot_reflects_internal_state() -> None:
    """The ``snapshot`` returns the streak, threshold, and the
    per-streak fired flag for ``get_status_snapshot`` consumers."""
    tracker, _agent = _new_tracker(threshold=2)
    snap = tracker.snapshot()
    assert snap == {
        "streak": 0,
        "threshold": 2,
        "diagnostic_fired_for_streak": False,
    }
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    snap = tracker.snapshot()
    assert snap["streak"] == 2
    assert snap["diagnostic_fired_for_streak"] is True


async def test_productive_iteration_with_event_observed_resets_streak() -> None:
    """Both "actions called" AND "event observed" reset — they
    are alternative signals of productivity."""
    tracker, _agent = _new_tracker()
    await tracker.observe_iteration(
        actions_called_count=0, queue_was_empty_at_observation=True,
    )
    await tracker.observe_iteration(
        actions_called_count=3, queue_was_empty_at_observation=False,
    )
    assert tracker.snapshot()["streak"] == 0
