"""Tests for the SessionAgent's runtime guardrail composition.

Covers the speaker-aware status-claim predicate (item 4 of
``colony/decompose_and_session_recovery_fixes_plan.md``): a
``respond_to_user`` content that mentions ONLY the SessionAgent's own
``agent-<hex>`` id must pass without a prior ``get_agent_status`` call,
while a content that mentions OTHER agent_ids must still gate on a
status check that covers them.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from polymathera.colony.agents.patterns.planning.models import (
    CallRecord,
)
from polymathera.colony.web_ui.backend.chat.session_agent_guardrails import (
    build_session_agent_runtime_guardrail,
)


pytestmark = pytest.mark.asyncio


class _FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id


def _status_call(
    *, agent_ids: list[str], when: float | None = None,
) -> CallRecord:
    return CallRecord(
        action_key="AgentPoolCapability.AgentPoolCapability.get_agent_status",
        params={"agent_ids": agent_ids},
        end_wall=when or time.time(),
        status="ok",
        result={"ok": True, "agents": []},
    )


async def _check(
    guardrail, *, content: str, history: list[CallRecord],
) -> Any:
    return await guardrail.check(
        action_key=(
            "SessionOrchestratorCapability."
            "SessionOrchestratorCapability.respond_to_user"
        ),
        params={"content": content},
        call_history=history,
    )


async def test_respond_to_user_without_agent_mention_passes() -> None:
    """Plain content with no ``agent-<hex>`` mention bypasses the
    rule's ``applies_when`` — no status check required."""

    guardrail = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    decision = await _check(
        guardrail, content="Hello!", history=[],
    )
    assert decision.allowed


async def test_respond_to_user_mentioning_other_agent_requires_status_check() -> None:
    """Content referencing a NON-speaker agent_id without a prior
    status check fires the gate — the canonical original failure
    mode the rule was written to catch."""

    guardrail = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    guardrail.bind_speaker(_FakeAgent("agent-aaaaaaaa"))
    decision = await _check(
        guardrail,
        content="The coordinator agent-bbbbbbbb is now running.",
        history=[],
    )
    assert not decision.allowed
    assert "get_agent_status" in decision.suggestion


async def test_respond_to_user_mentioning_only_speakers_own_id_passes() -> None:
    """Item 4 fix — the SessionAgent talking about ITSELF doesn't
    need a status check. Content that mentions only the speaker's
    own ``agent-<hex>`` id passes without any prior
    ``get_agent_status`` call.

    Pre-fix the 2026-06-07 live run blocked exactly this shape and
    the SessionAgent never sent its closing summary."""

    guardrail = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    guardrail.bind_speaker(_FakeAgent("agent-52c36b64"))
    decision = await _check(
        guardrail,
        content=(
            "I (agent-52c36b64) finished spawning the coordinator."
        ),
        history=[],
    )
    assert decision.allowed


async def test_respond_to_user_mentioning_speaker_plus_other_still_gates_on_other() -> None:
    """Content that mentions BOTH the speaker AND another agent must
    still gate — the gate is for the non-speaker. A prior status
    check that covers the non-speaker is enough."""

    guardrail = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    guardrail.bind_speaker(_FakeAgent("agent-52c36b64"))

    # Without a status check on the OTHER agent → blocked.
    decision_blocked = await _check(
        guardrail,
        content=(
            "I (agent-52c36b64) spawned coordinator agent-38220aa6."
        ),
        history=[],
    )
    assert not decision_blocked.allowed

    # With a status check that covers the OTHER agent → allowed.
    decision_allowed = await _check(
        guardrail,
        content=(
            "I (agent-52c36b64) spawned coordinator agent-38220aa6."
        ),
        history=[_status_call(agent_ids=["agent-38220aa6"])],
    )
    assert decision_allowed.allowed


async def test_bind_speaker_with_none_falls_back_to_pre_fix_behaviour() -> None:
    """When the policy never binds a speaker (or binds None), the
    predicate degrades safely: every ``agent-<hex>`` mention must be
    covered, just like before item 4. Belt-and-braces for any code
    path that constructs the guardrail without going through the
    code-generation policy's init hook."""

    guardrail = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    # Deliberately do NOT call bind_speaker.
    decision = await _check(
        guardrail,
        content="My id is agent-aaaaaaaa.",
        history=[],
    )
    assert not decision.allowed


async def test_composite_propagates_bind_speaker_to_inner_guardrails() -> None:
    """The ``CompositeGuardrail.bind_speaker`` override must call
    each inner guardrail's ``bind_speaker`` — otherwise the SessionAgent
    fix in ``ArgsAwareTemporalOrderGuardrail`` would never see the
    speaker through the composite wrapper."""

    guardrail = build_session_agent_runtime_guardrail(
        approval_required_action_prefixes=[],
    )
    speaker_id = "agent-deadbeef"
    guardrail.bind_speaker(_FakeAgent(speaker_id))

    decision = await _check(
        guardrail,
        content=f"Self-reference only: {speaker_id} is running.",
        history=[],
    )
    assert decision.allowed
