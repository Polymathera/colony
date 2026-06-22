"""Tests for the dashboard endpoints + protocol shape that back the
operator runtime override of semantic constraints.

The override path is intentionally event-handler-free: the dashboard
POST writes a canonical key on the session BB, and
:meth:`SemanticConstraintGuardrail.check` reads the same keys live
each iteration. So this file pins only the two things still on the
critical path:

1. The :class:`OperatorOverrideProtocol` key/pattern/parse shape — the
   single contract both writer (endpoint) and reader (guardrail) bind
   to.
2. The dashboard endpoints exist and write to the canonical key.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Protocol shape
# ---------------------------------------------------------------------------


def test_operator_override_protocol_keys_and_patterns() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        OperatorOverrideProtocol,
    )
    key = OperatorOverrideProtocol.semantic_constraint_key("my_rule")
    assert key == "operator_override:semantic_constraint:my_rule"
    pattern = OperatorOverrideProtocol.semantic_constraint_pattern()
    assert pattern == "operator_override:semantic_constraint:*"
    parsed = OperatorOverrideProtocol.parse_semantic_constraint_key(key)
    assert parsed == "my_rule"


def test_operator_override_protocol_parse_rejects_alien_key() -> None:
    import pytest
    from polymathera.colony.agents.blackboard.protocol import (
        OperatorOverrideProtocol,
    )
    with pytest.raises(ValueError):
        OperatorOverrideProtocol.parse_semantic_constraint_key("chat:user:x")


# ---------------------------------------------------------------------------
# 2. Dashboard endpoints
# ---------------------------------------------------------------------------


def test_disable_and_enable_endpoints_registered() -> None:
    """The two endpoints are registered on the sessions router so
    the dashboard can POST to them."""

    from polymathera.colony.web_ui.backend.routers import sessions
    routes = {
        getattr(r, "path", None)
        for r in sessions.router.routes
    }
    assert (
        "/sessions/{session_id}/constraints/{constraint_id}/disable"
        in routes
    )
    assert (
        "/sessions/{session_id}/constraints/{constraint_id}/enable"
        in routes
    )


def test_endpoints_write_canonical_operator_override_key() -> None:
    """Source-pin: both endpoints write to the canonical
    ``OperatorOverrideProtocol.semantic_constraint_key`` — the same
    key the guardrail's ``_read_disabled_ids`` reads back."""

    src = (
        Path(__file__).resolve().parents[1]
        / "routers" / "sessions.py"
    ).read_text(encoding="utf-8")
    assert (
        "OperatorOverrideProtocol.semantic_constraint_key(constraint_id)"
        in src
    )
    # Payload shape includes the three load-bearing fields.
    assert '"disabled":' in src
    assert '"set_at":' in src
    assert '"set_by":' in src
