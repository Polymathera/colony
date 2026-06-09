"""Tests for ``AgentDiagnosticProtocol``."""

from __future__ import annotations

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
    DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK,
)


def test_event_key_shape() -> None:
    key = AgentDiagnosticProtocol.event_key(
        "agent-aaa", DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK, 1,
    )
    assert key == "agent:diagnostic:agent-aaa:guardrail_block_streak:1"


def test_event_pattern_for_specific_agent() -> None:
    assert (
        AgentDiagnosticProtocol.event_pattern("agent-aaa")
        == "agent:diagnostic:agent-aaa:*:*"
    )


def test_event_pattern_wildcard() -> None:
    assert (
        AgentDiagnosticProtocol.event_pattern()
        == "agent:diagnostic:*:*:*"
    )


def test_parse_event_key_round_trips() -> None:
    key = AgentDiagnosticProtocol.event_key(
        "agent-bbb", "polling_timeout", 7,
    )
    parsed = AgentDiagnosticProtocol.parse_event_key(key)
    assert parsed == {
        "agent_id": "agent-bbb",
        "kind": "polling_timeout",
        "sequence": "7",
    }


def test_parse_event_key_rejects_alien_prefix() -> None:
    with pytest.raises(ValueError):
        AgentDiagnosticProtocol.parse_event_key(
            "human_approval:request:appr_xyz",
        )


def test_parse_event_key_rejects_malformed_suffix() -> None:
    with pytest.raises(ValueError):
        AgentDiagnosticProtocol.parse_event_key(
            "agent:diagnostic:agent-aaa:kind",  # missing sequence
        )
