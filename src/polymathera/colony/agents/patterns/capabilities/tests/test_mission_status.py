"""Tests for :class:`MissionStatusCapability`.

The capability's contract: one ``@action_executor`` that writes the
current narrative status to a SINGLETON SESSION-scoped blackboard key
keyed by the agent's own ``agent_id`` (the framework-known mission
identifier). The key is the canonical ``SessionChatProtocol``
``mission_status_key`` so the chat router's relay forwards it to the
frontend without bespoke wiring. The action is decorated
``emits_lifecycle=False`` so its own dispatch does NOT generate a
"running: emit_mission_status" spinner row.
"""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.agents.patterns.capabilities.mission_status import (
    MissionStatusCapability,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubBlackboard:
    """Records every ``write`` call so the test can assert key + payload."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, Any], frozenset[str] | set[str] | None, dict[str, Any] | None]] = []

    async def write(
        self,
        key: str,
        value: dict[str, Any],
        *,
        tags: Any = None,
        metadata: Any = None,
    ) -> None:
        self.writes.append((key, value, tags, metadata))


class _StubAgent:
    """Minimal agent with the surface the capability touches."""

    def __init__(self, agent_id: str = "agent-mission-test") -> None:
        self.agent_id = agent_id
        self.blackboard = _StubBlackboard()
        self.scope_id_observed: str | None = None

    async def get_blackboard(self, *, scope_id: str | None = None) -> _StubBlackboard:
        self.scope_id_observed = scope_id
        return self.blackboard

    def get_capabilities(self):  # pragma: no cover — capability framework optional
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_capability(agent: _StubAgent) -> MissionStatusCapability:
    """Bypass the capability framework's blueprint resolution and
    construct directly. We test the action body, not the registration
    plumbing."""

    cap = MissionStatusCapability.__new__(MissionStatusCapability)
    cap._agent = agent
    cap.scope_id = f"polymathera:test:agent:{agent.agent_id}:session_chat"
    return cap


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


async def test_emit_writes_singleton_key_under_agent_id() -> None:
    """The ``mission_id`` is the agent's own ``agent_id`` — typed,
    framework-known, never threaded through an LLM kwarg. The key is
    the canonical ``SessionChatProtocol.mission_status_key``."""

    from polymathera.colony.agents.blackboard.protocol import (
        MissionStatusProtocol,
    )

    agent = _StubAgent(agent_id="agent-coord-42")
    cap = _new_capability(agent)

    result = await cap.emit_mission_status(message="loading...")
    assert result == {"ok": True, "mission_id": "agent-coord-42"}
    assert len(agent.blackboard.writes) == 1
    key, payload, tags, metadata = agent.blackboard.writes[0]
    assert key == MissionStatusProtocol.status_key("agent-coord-42")
    assert payload["mission_id"] == "agent-coord-42"
    assert payload["agent_id"] == "agent-coord-42"
    assert payload["message"] == "loading..."
    assert payload["details"] == {}
    assert tags == {"mission_status"}
    assert metadata == {"mission_id": "agent-coord-42"}


async def test_emit_uses_session_scoped_blackboard() -> None:
    """The capability resolves to a SESSION-scoped blackboard via
    its ``scope_id``; subsequent emits stay on the same key so the
    UI sees a singleton, not a history."""

    agent = _StubAgent()
    cap = _new_capability(agent)
    await cap.emit_mission_status(message="step 1")
    await cap.emit_mission_status(message="step 2")
    assert agent.scope_id_observed == cap.scope_id
    # Both writes target the SAME key — the protocol is singleton-per-mission.
    keys = {w[0] for w in agent.blackboard.writes}
    assert len(keys) == 1


async def test_emit_round_trips_details_dict() -> None:
    """Structured ``details`` are forwarded verbatim — additive UI
    context the frontend renders alongside the message."""

    agent = _StubAgent()
    cap = _new_capability(agent)
    await cap.emit_mission_status(
        message="classifying",
        details={"total": 14, "decomposable_so_far": 3},
    )
    payload = agent.blackboard.writes[0][1]
    assert payload["details"] == {"total": 14, "decomposable_so_far": 3}


async def test_emit_includes_monotonic_timestamp() -> None:
    """The payload carries a server-side timestamp so the frontend
    can order overlapping emissions deterministically."""

    agent = _StubAgent()
    cap = _new_capability(agent)
    await cap.emit_mission_status(message="x")
    payload = agent.blackboard.writes[0][1]
    assert isinstance(payload["timestamp"], float)
    assert payload["timestamp"] > 0.0


def test_emit_action_carries_emits_lifecycle_false() -> None:
    """Regression pin: the action must declare ``emits_lifecycle=False``
    so its own dispatch does not produce a spinner row labelled
    ``running: emit_mission_status``. Without this flag the primitive
    would echo into the very channel it is meant to replace — the
    assumption-review check Change 5 unlocks."""

    method = MissionStatusCapability.emit_mission_status
    assert getattr(method, "_action_emits_lifecycle", True) is False


def test_planning_summary_describes_use_contract_not_phase_names() -> None:
    """The decorator's ``planning_summary`` tells the LLM WHY the
    primitive exists (opaque-waiting alternative); it does NOT list
    specific phase names that would nudge the planner toward a fixed
    pipeline. Per [[primitives-not-pipelines]]."""

    summary = (
        MissionStatusCapability.emit_mission_status._action_planning_summary
        or ""
    )
    # WHY framing is present
    assert "opaque waiting" in summary or "narrative" in summary
    # Not a baked sequence — the docstring's parenthetical examples
    # are illustrative single-actions, NOT a numbered or arrow-
    # connected pipeline. Pin the absence of ordering markers.
    for forbidden in ("Step 1", "step (1)", "->", "→"):
        assert forbidden not in summary, summary
