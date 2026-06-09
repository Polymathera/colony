"""Tests for :class:`BlockStreakTracker` — the producer of
``AgentDiagnosticProtocol`` guardrail-block-streak events."""

from __future__ import annotations

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK,
)
from polymathera.colony.agents.patterns.actions.code_generation import (
    BlockStreakTracker,
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


def _new_tracker() -> BlockStreakTracker:
    return BlockStreakTracker(_FakeAgent())


async def test_no_emission_below_threshold() -> None:
    t = _new_tracker()
    for _ in range(2):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    assert t.agent.writes == []


async def test_emission_at_threshold() -> None:
    t = _new_tracker()
    for _ in range(3):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    assert len(t.agent.writes) == 1
    key, payload = t.agent.writes[0]
    parsed = AgentDiagnosticProtocol.parse_event_key(key)
    assert parsed["agent_id"] == "agent-prod"
    assert parsed["kind"] == DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK
    assert payload["action_key"] == "X.gated"
    assert payload["count"] == 3


async def test_emission_re_fires_every_threshold() -> None:
    t = _new_tracker()
    for _ in range(9):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    # Threshold 3 → emit at 3, 6, 9.
    assert len(t.agent.writes) == 3
    counts = [v["count"] for _, v in t.agent.writes]
    assert counts == [3, 6, 9]


async def test_different_action_key_resets_streak() -> None:
    t = _new_tracker()
    for _ in range(2):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    await t.track(action_key="Y.gated", reason="r", suggestion="s")
    # Streak reset → no emission yet.
    assert t.agent.writes == []
    # Two more Y.gated → reach threshold.
    for _ in range(2):
        await t.track(action_key="Y.gated", reason="r", suggestion="s")
    assert len(t.agent.writes) == 1
    assert t.agent.writes[0][1]["action_key"] == "Y.gated"


async def test_explicit_reset_streak() -> None:
    """Successful dispatches in the policy call ``reset_streak`` —
    after which the next two blocks should NOT yet emit."""

    t = _new_tracker()
    for _ in range(2):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    t.reset_streak()
    for _ in range(2):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    assert t.agent.writes == []


async def test_sequence_numbers_unique() -> None:
    t = _new_tracker()
    for _ in range(6):
        await t.track(action_key="X.gated", reason="r", suggestion="s")
    seqs = [
        AgentDiagnosticProtocol.parse_event_key(k)["sequence"]
        for k, _ in t.agent.writes
    ]
    assert len(set(seqs)) == len(seqs)
