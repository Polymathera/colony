"""Tests for :class:`GuardrailWaiverCapability` + the
:class:`GuardrailWaiverProtocol` shape.

Pins the contract the asking agent and the dashboard endpoints
both depend on:

- Request key / response key / patterns roundtrip via the typed
  protocol helpers.
- The action ``request_guardrail_waiver`` writes a request entry
  on session BB carrying the typed payload (constraint_id,
  justification, requester_agent_id, waiver_id).
- The ``@event_handler`` on the response pattern returns an
  ``EventProcessingResult`` with the right planner-context binding
  (``approved`` flag preserved, constraint_id round-tripped) so the
  asking agent's next iteration sees the decision and branches.
- Auto-mount: ``Agent._create_action_policy``'s Phase 1d adds the
  capability when the runtime_guardrail contains a
  :class:`SemanticConstraintGuardrail` (directly or wrapped in
  ``CompositeGuardrail``), and DOES NOT add it otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.blackboard.protocol import (
    GuardrailWaiverProtocol,
)
from polymathera.colony.agents.patterns.capabilities.guardrail_waiver import (
    GuardrailWaiverCapability,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_protocol_request_response_keys_and_patterns() -> None:
    rk = GuardrailWaiverProtocol.request_key("w1")
    assert rk == "guardrail_waiver:request:w1"
    assert GuardrailWaiverProtocol.parse_request_key(rk) == "w1"
    rp = GuardrailWaiverProtocol.response_key("w1")
    assert rp == "guardrail_waiver:response:w1"
    assert GuardrailWaiverProtocol.parse_response_key(rp) == "w1"

    assert (
        GuardrailWaiverProtocol.request_pattern()
        == "guardrail_waiver:request:*"
    )
    assert (
        GuardrailWaiverProtocol.response_pattern()
        == "guardrail_waiver:response:*"
    )


def test_protocol_parse_rejects_alien_key() -> None:
    with pytest.raises(ValueError):
        GuardrailWaiverProtocol.parse_request_key("chat:user:x")
    with pytest.raises(ValueError):
        GuardrailWaiverProtocol.parse_response_key("chat:user:x")


# ---------------------------------------------------------------------------
# Action — request_guardrail_waiver writes the typed request key
# ---------------------------------------------------------------------------


def _make_capability(
    agent_id: str = "agent-1",
) -> tuple[GuardrailWaiverCapability, MagicMock]:
    cap = GuardrailWaiverCapability.__new__(GuardrailWaiverCapability)
    cap._agent = SimpleNamespace(agent_id=agent_id)
    cap._outstanding_waiver_ids = set()
    bb = MagicMock()
    bb.write = AsyncMock()
    cap.get_blackboard = AsyncMock(return_value=bb)
    return cap, bb


async def test_request_writes_typed_request_key_with_payload() -> None:
    cap, bb = _make_capability(agent_id="agent-X")
    result = await cap.request_guardrail_waiver(
        constraint_id="no_unverified_agent_state_claims",
        justification="LLM judge keeps misreading the verified call.",
    )
    assert result["ok"] is True
    waiver_id = result["waiver_id"]
    assert waiver_id.startswith("waiver_")
    bb.write.assert_awaited_once()
    args, kwargs = bb.write.call_args
    written_key = args[0]
    written_value = args[1]
    assert written_key == GuardrailWaiverProtocol.request_key(waiver_id)
    assert written_value["waiver_id"] == waiver_id
    assert written_value["constraint_id"] == "no_unverified_agent_state_claims"
    assert written_value["justification"].startswith("LLM judge keeps")
    assert written_value["requester_agent_id"] == "agent-X"


async def test_request_rejects_empty_constraint_id() -> None:
    cap, bb = _make_capability()
    result = await cap.request_guardrail_waiver(
        constraint_id="   ", justification="real reason",
    )
    assert result["ok"] is False
    bb.write.assert_not_awaited()


async def test_request_rejects_empty_justification() -> None:
    cap, bb = _make_capability()
    result = await cap.request_guardrail_waiver(
        constraint_id="some_rule", justification="",
    )
    assert result["ok"] is False
    bb.write.assert_not_awaited()


async def test_request_with_no_agent_context_fails() -> None:
    """Without an owner agent, the capability cannot stamp
    requester_agent_id; refuse rather than write a malformed payload."""
    cap = GuardrailWaiverCapability.__new__(GuardrailWaiverCapability)
    cap._agent = None
    result = await cap.request_guardrail_waiver(
        constraint_id="r", justification="r",
    )
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# Response handler — binds planner context
# ---------------------------------------------------------------------------


async def test_on_response_binds_planner_context_for_approved() -> None:
    cap, _ = _make_capability()
    event = BlackboardEvent(
        event_type="write",
        key=GuardrailWaiverProtocol.response_key("w42"),
        value={
            "waiver_id": "w42",
            "constraint_id": "rule_x",
            "approved": True,
            "decided_by": "user_abc",
            "reason": "rule legitimately doesn't fit",
        },
    )
    result = await cap._on_response(event, None)
    assert result is not None
    assert result.context_key == (
        f"{GuardrailWaiverCapability.RESPONSE_CONTEXT_KEY_PREFIX}w42"
    )
    ctx = result.context
    assert ctx["approved"] is True
    assert ctx["constraint_id"] == "rule_x"
    assert ctx["decided_by"] == "user_abc"
    assert ctx["reason"] == "rule legitimately doesn't fit"


async def test_on_response_binds_planner_context_for_rejected() -> None:
    cap, _ = _make_capability()
    event = BlackboardEvent(
        event_type="write",
        key=GuardrailWaiverProtocol.response_key("w42"),
        value={
            "waiver_id": "w42",
            "constraint_id": "rule_x",
            "approved": False,
            "decided_by": "user_abc",
            "reason": "no — this rule must stand",
        },
    )
    result = await cap._on_response(event, None)
    assert result is not None
    assert result.context["approved"] is False
    assert result.context["reason"] == "no — this rule must stand"


async def test_on_response_skips_alien_key() -> None:
    cap, _ = _make_capability()
    event = BlackboardEvent(
        event_type="write",
        key="chat:user:msg_X",  # wrong protocol
        value={"approved": True},
    )
    result = await cap._on_response(event, None)
    assert result is None


async def test_on_response_skips_malformed_value() -> None:
    cap, _ = _make_capability()
    event = BlackboardEvent(
        event_type="write",
        key=GuardrailWaiverProtocol.response_key("w42"),
        value="not a dict",
    )
    result = await cap._on_response(event, None)
    assert result is None


# ---------------------------------------------------------------------------
# Auto-mount detection: Agent._create_action_policy Phase 1d
# ---------------------------------------------------------------------------


def test_auto_mount_helper_detects_direct_semantic_constraint() -> None:
    from polymathera.colony.agents.patterns.actions.defaults import (
        _runtime_guardrail_has_semantic_constraints,
    )
    from polymathera.colony.agents.patterns.actions.semantic_constraints import (
        SemanticConstraintGuardrail,
    )

    g = SemanticConstraintGuardrail(constraints=[])
    assert _runtime_guardrail_has_semantic_constraints(g) is True


def test_auto_mount_helper_detects_composite_wrapped_semantic_constraint(
) -> None:
    from polymathera.colony.agents.patterns.actions.defaults import (
        _runtime_guardrail_has_semantic_constraints,
    )
    from polymathera.colony.agents.patterns.actions.code_constraints import (
        CompositeGuardrail,
        NoGuardrail,
    )
    from polymathera.colony.agents.patterns.actions.semantic_constraints import (
        SemanticConstraintGuardrail,
    )

    g = CompositeGuardrail(
        NoGuardrail(),
        SemanticConstraintGuardrail(constraints=[]),
    )
    assert _runtime_guardrail_has_semantic_constraints(g) is True


def test_auto_mount_helper_false_when_no_semantic_constraint() -> None:
    from polymathera.colony.agents.patterns.actions.defaults import (
        _runtime_guardrail_has_semantic_constraints,
    )
    from polymathera.colony.agents.patterns.actions.code_constraints import (
        NoGuardrail,
    )

    assert _runtime_guardrail_has_semantic_constraints(NoGuardrail()) is False
    assert _runtime_guardrail_has_semantic_constraints(None) is False
    assert _runtime_guardrail_has_semantic_constraints("nonsense") is False
