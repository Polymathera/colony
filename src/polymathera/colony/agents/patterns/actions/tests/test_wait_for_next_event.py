"""Tests for the ``wait_for_next_event`` action on
:class:`EventDrivenActionPolicy` — the consume-none idle primitive
the LLM calls from generated code when it has no work to do."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.policies import (
    EventDrivenActionPolicy,
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
    """Minimal agent stub for direct policy-method testing."""

    def __init__(self, agent_id: str = "agent-wait") -> None:
        self.agent_id = agent_id
        self.idle_wait_counter: int = 0
        self.metadata = MagicMock()
        self.metadata.goals = []
        self.metadata.role = "test"
        self.metadata.parameters = {}
        self.metadata.run_id = None
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

    def get_capabilities(self):
        return []


def _make_policy() -> EventDrivenActionPolicy:
    """Build a policy without subscribing capabilities; we test the
    wait action's interaction with the queue pair directly."""
    return EventDrivenActionPolicy(agent=_FakeAgent())


async def test_returns_immediately_when_normal_queue_nonempty() -> None:
    """Already-queued normal event ⇒ wait returns immediately."""
    policy = _make_policy()
    policy._queues.normal.put_nowait("ev")
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=0.05,
    )
    assert result == {"ok": True, "timed_out": False}
    # Consume-none: event is still queued.
    assert policy._queues.normal.qsize() == 1


async def test_returns_immediately_when_high_queue_nonempty() -> None:
    """Already-queued high-priority event ⇒ wait returns immediately."""
    policy = _make_policy()
    policy._queues.high.put_nowait("ev-h")
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=0.05,
    )
    assert result == {"ok": True, "timed_out": False}
    assert policy._queues.high.qsize() == 1


async def test_blocks_until_event_arrives_on_normal_lane() -> None:
    """Empty queues ⇒ the wait blocks; arrival wakes it."""
    policy = _make_policy()

    async def delayed_put() -> None:
        await asyncio.sleep(0.03)
        policy._queues.normal.put_nowait("ev")

    asyncio.create_task(delayed_put())
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=1.0,
    )
    assert result == {"ok": True, "timed_out": False}


async def test_blocks_until_event_arrives_on_high_lane() -> None:
    """High-priority arrival also wakes the wait."""
    policy = _make_policy()

    async def delayed_put_high() -> None:
        await asyncio.sleep(0.03)
        policy._queues.high.put_nowait("ev-h")

    asyncio.create_task(delayed_put_high())
    result = await asyncio.wait_for(
        policy.wait_for_next_event(), timeout=1.0,
    )
    assert result == {"ok": True, "timed_out": False}


async def test_timeout_returns_timed_out_envelope() -> None:
    """No event before deadline ⇒ ``{ok: True, timed_out: True}``."""
    policy = _make_policy()
    result = await policy.wait_for_next_event(timeout_seconds=0.05)
    assert result == {"ok": True, "timed_out": True}


async def test_idle_wait_counter_pair_balanced_on_normal_wakeup() -> None:
    """Counter increments on entry and decrements on exit. Pairs
    with the existing ``idle_wait_counter`` accounting used by
    ``HumanApprovalCapability.get_response``."""
    policy = _make_policy()
    agent = policy.agent
    assert agent.idle_wait_counter == 0

    async def delayed_put() -> None:
        # Confirm counter is incremented while the wait is in
        # progress — sample mid-wait before delivering the event.
        await asyncio.sleep(0.02)
        assert agent.idle_wait_counter == 1
        policy._queues.normal.put_nowait("ev")

    asyncio.create_task(delayed_put())
    await policy.wait_for_next_event()
    assert agent.idle_wait_counter == 0


async def test_idle_wait_counter_balanced_on_timeout() -> None:
    """Counter returns to 0 even when the wait times out."""
    policy = _make_policy()
    agent = policy.agent
    assert agent.idle_wait_counter == 0
    await policy.wait_for_next_event(timeout_seconds=0.02)
    assert agent.idle_wait_counter == 0


async def test_idle_wait_counter_balanced_on_cancellation() -> None:
    """Counter returns to 0 when the wait task is cancelled — the
    /abort path relies on this so the agent's idle accounting does
    not leak when the user aborts an in-flight wait."""
    policy = _make_policy()
    agent = policy.agent
    waiter = asyncio.create_task(policy.wait_for_next_event())
    await asyncio.sleep(0.02)
    assert agent.idle_wait_counter == 1
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert agent.idle_wait_counter == 0


async def test_consume_none_event_is_not_pulled_from_queue() -> None:
    """The wait observes nonempty but does NOT consume; the next
    plan_step's drain (or any other consumer) sees the event
    exactly once."""
    policy = _make_policy()
    policy._queues.normal.put_nowait("ev-1")
    policy._queues.high.put_nowait("ev-h-1")
    await policy.wait_for_next_event()
    assert policy._queues.normal.qsize() == 1
    assert policy._queues.high.qsize() == 1


def test_action_is_discoverable_via_action_executor_decorator() -> None:
    """``ActionDispatcher._create_default_action_executors`` walks
    ``[self.agent, self.action_policy]`` (see ``dispatcher.py:877``)
    and discovers methods decorated with ``@action_executor`` via
    ``_action_input_schema`` / ``_action_output_schema`` attributes.
    Pin that ``wait_for_next_event`` carries those markers so a
    future refactor that drops the decorator surfaces here."""
    method = EventDrivenActionPolicy.wait_for_next_event
    assert callable(method)
    # The decorator at dispatcher.py:751-752 sets these attributes
    # on the wrapped function — they're the canonical signal the
    # dispatcher uses to discover action executors.
    assert hasattr(method, "_action_input_schema")
    assert hasattr(method, "_action_output_schema")
