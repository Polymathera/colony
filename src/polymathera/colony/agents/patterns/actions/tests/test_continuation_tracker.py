"""Tests for :class:`ContinuationTracker` — the producer of
``AgentDiagnosticProtocol`` continuation-budget-exhausted events."""

from __future__ import annotations

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED,
)
from polymathera.colony.agents.patterns.actions.continuation_tracker import (
    ContinuationTracker,
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
    def __init__(self) -> None:
        self.agent_id = "agent-prod"
        self.writes: list[tuple[str, dict]] = []

    async def get_blackboard(self, *, scope_id: str | None = None):
        agent = self

        class _BB:
            async def write(
                self,
                key: str,
                value: dict,
                *,
                tags=None,
                metadata=None,
            ):
                agent.writes.append((key, value))

        return _BB()


def _new_tracker(*, max_per_burst: int = 5) -> ContinuationTracker:
    return ContinuationTracker(_FakeAgent(), max_per_burst=max_per_burst)


async def test_record_continuation_within_budget_returns_true() -> None:
    """While the consecutive count is below ``max_per_burst``, every
    call returns True and increments the counter. No diagnostic is
    emitted."""

    t = _new_tracker(max_per_burst=3)
    for i in range(3):
        allowed = await t.record_continuation(reason=f"step {i}")
        assert allowed is True
    assert t._consecutive_count == 3
    assert t._last_reason == "step 2"
    assert t.agent.writes == []


async def test_record_continuation_at_cap_returns_false() -> None:
    """Once the counter has reached ``max_per_burst``, the next call
    returns False and emits ONE ``continuation_budget_exhausted``
    diagnostic. The counter is NOT incremented past the cap."""

    t = _new_tracker(max_per_burst=2)
    assert await t.record_continuation(reason="a") is True
    assert await t.record_continuation(reason="b") is True
    allowed = await t.record_continuation(reason="c")
    assert allowed is False
    assert t._consecutive_count == 2
    assert len(t.agent.writes) == 1
    key, payload = t.agent.writes[0]
    parsed = AgentDiagnosticProtocol.parse_event_key(key)
    assert parsed["agent_id"] == "agent-prod"
    assert parsed["kind"] == DIAGNOSTIC_CONTINUATION_BUDGET_EXHAUSTED
    assert payload["consecutive_count"] == 2
    assert payload["max_per_burst"] == 2
    assert payload["last_reason"] == "b"
    assert payload["attempted_reason"] == "c"


async def test_reset_clears_consecutive_count() -> None:
    """``reset()`` clears the burst counter and last_reason so the
    next burst starts fresh — modelling the arrival of a NEW external
    event that breaks the self-triggered continuation chain."""

    t = _new_tracker(max_per_burst=3)
    await t.record_continuation(reason="a")
    await t.record_continuation(reason="b")
    assert t._consecutive_count == 2
    t.reset()
    assert t._consecutive_count == 0
    assert t._last_reason is None
    # Budget restored: three more continuations should all be allowed.
    for i in range(3):
        assert await t.record_continuation(reason=f"r{i}") is True
    assert t.agent.writes == []


async def test_snapshot_shape() -> None:
    """``snapshot()`` is the read-only view consumed by
    ``get_status_snapshot``. Shape: consecutive_count, max_per_burst,
    last_reason, exhausted."""

    t = _new_tracker(max_per_burst=2)
    snap = t.snapshot()
    assert snap == {
        "consecutive_count": 0,
        "max_per_burst": 2,
        "last_reason": None,
        "exhausted": False,
    }
    await t.record_continuation(reason="first")
    await t.record_continuation(reason="second")
    snap = t.snapshot()
    assert snap == {
        "consecutive_count": 2,
        "max_per_burst": 2,
        "last_reason": "second",
        "exhausted": True,
    }
