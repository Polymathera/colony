"""Tests for the dashboard endpoints that translate an operator's
waiver decision into the typed
:class:`GuardrailWaiverProtocol` response key (always written) and
the :class:`OperatorOverrideProtocol` key (written only on approve,
to actually disable the constraint via the PR5-B path).

The asking agent's :class:`GuardrailWaiverCapability` subscribes to
the response key — that's its wake signal. The override key is what
makes :meth:`SemanticConstraintGuardrail._read_disabled_ids` see the
constraint as disabled on the next ``.check()``. Both must be on BB
by the time the agent's planner runs, so the approve endpoint
writes both atomically (override first, then response).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def test_approve_and_reject_endpoints_registered() -> None:
    """Both routes are on the sessions router so the dashboard can POST."""

    from polymathera.colony.web_ui.backend.routers import sessions
    routes = {
        getattr(r, "path", None)
        for r in sessions.router.routes
    }
    assert (
        "/sessions/{session_id}/waivers/{waiver_id}/approve"
        in routes
    )
    assert (
        "/sessions/{session_id}/waivers/{waiver_id}/reject"
        in routes
    )


# ---------------------------------------------------------------------------
# Source pins: both endpoints write to the canonical keys
# ---------------------------------------------------------------------------


def test_approve_path_writes_both_override_and_response_keys() -> None:
    """Source pin: the approve path MUST write BOTH the
    OperatorOverrideProtocol key (to make the guardrail let the
    asking agent's next action through) AND the
    GuardrailWaiverProtocol response key (to wake the asking
    agent). Order matters: override first, so by the time the
    response key fires the agent's @event_handler the override is
    already on BB."""

    src = (
        Path(__file__).resolve().parents[1]
        / "routers" / "sessions.py"
    ).read_text(encoding="utf-8")
    assert (
        "OperatorOverrideProtocol.semantic_constraint_key(" in src
    ), "approve path must write the operator-override key"
    assert (
        "GuardrailWaiverProtocol.response_key(waiver_id)" in src
    ), "both paths must write the waiver response key"
    # The approve guard is in _write_guardrail_waiver_response; check
    # the write-override branch is gated on ``approved and constraint_id``.
    assert "if approved and constraint_id:" in src, (
        "override write must be gated on approved=True; reject path "
        "MUST NOT write the override (no leakage on rejection)."
    )


def test_response_payload_carries_approved_constraint_decided_by() -> None:
    """Source pin: the waiver response payload carries the three
    fields the agent's @event_handler reads:
    ``approved``, ``constraint_id``, ``decided_by``. Drift here
    would break the planner-context binding contract."""

    src = (
        Path(__file__).resolve().parents[1]
        / "routers" / "sessions.py"
    ).read_text(encoding="utf-8")
    # Find the response-write block and assert all three field
    # assignments are present in the payload literal.
    assert '"approved": approved,' in src
    assert '"constraint_id": constraint_id,' in src
    assert '"decided_by": decided_by,' in src


# ---------------------------------------------------------------------------
# Runtime: approve writes 2 BB entries, reject writes 1
# ---------------------------------------------------------------------------


class _FakeBB:
    def __init__(self) -> None:
        self.writes: list[tuple[str, dict]] = []

    async def initialize(self) -> None:
        return None

    async def write(self, key: str, value: dict, **_kw: Any) -> None:
        self.writes.append((key, value))


def _build_app(
    *,
    tenant_id: str = "t1",
    colony_id: str = "c1",
    session_id: str = "s1",
) -> tuple[Any, dict]:
    """Build a FastAPI app with the sessions router + every dependency
    stubbed. Returns (app, capture) where ``capture['bb']`` is the
    fake blackboard the endpoint writes to."""

    from fastapi import FastAPI

    from polymathera.colony.web_ui.backend.routers import sessions
    from polymathera.colony.web_ui.backend.dependencies import get_colony
    from polymathera.colony.web_ui.backend.auth.middleware import require_auth

    app = FastAPI()
    app.include_router(sessions.router, prefix="/api/v1")

    fake_bb = _FakeBB()
    capture: dict = {"bb": fake_bb, "_originals": {}}

    # Stub the EnhancedBlackboard constructor at the call site.
    from polymathera.colony.agents import blackboard as bb_mod

    capture["_originals"]["EnhancedBlackboard"] = bb_mod.EnhancedBlackboard

    def _stub_bb(*args, **kwargs):
        return fake_bb

    bb_mod.EnhancedBlackboard = _stub_bb  # type: ignore[assignment]

    # Stub the session_manager lookup.
    fake_session = SimpleNamespace(
        tenant_id=tenant_id, colony_id=colony_id,
    )
    fake_sm = SimpleNamespace(
        get_session=AsyncMock(return_value=fake_session),
    )
    fake_colony = SimpleNamespace(
        app_name="test-app",
        is_connected=True,
        get_session_manager=AsyncMock(return_value=fake_sm),
        _db_pool=object(),
    )
    app.dependency_overrides[get_colony] = lambda: fake_colony
    app.dependency_overrides[require_auth] = lambda: {
        "tenant_id": tenant_id, "sub": "user_test",
    }

    return app, capture


def _restore(capture: dict) -> None:
    from polymathera.colony.agents import blackboard as bb_mod
    bb_mod.EnhancedBlackboard = capture["_originals"]["EnhancedBlackboard"]


async def test_approve_writes_override_then_response() -> None:
    """End-to-end: POST approve → fake BB receives TWO writes, the
    override key first (so it's visible when the agent wakes), then
    the response key."""

    from fastapi.testclient import TestClient
    app, capture = _build_app()
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/sessions/s1/waivers/w_abc/approve",
            json={"constraint_id": "rule_X", "reason": "valid"},
        )
        assert resp.status_code == 200
        writes = capture["bb"].writes
        assert len(writes) == 2
        first_key, first_val = writes[0]
        second_key, second_val = writes[1]
        # Override first.
        assert first_key == "operator_override:semantic_constraint:rule_X"
        assert first_val["disabled"] is True
        assert first_val["waiver_id"] == "w_abc"
        # Response second.
        assert second_key == "guardrail_waiver:response:w_abc"
        assert second_val["approved"] is True
        assert second_val["constraint_id"] == "rule_X"
        assert second_val["decided_by"] == "user_test"
        assert second_val["reason"] == "valid"
    finally:
        _restore(capture)


async def test_reject_writes_only_response_no_override() -> None:
    """Reject MUST NOT write the operator-override key — the
    constraint stays active. Only the typed response lands so the
    agent's planner wakes and sees ``approved=False``."""

    from fastapi.testclient import TestClient
    app, capture = _build_app()
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/sessions/s1/waivers/w_xyz/reject",
            json={"constraint_id": "rule_X", "reason": "no"},
        )
        assert resp.status_code == 200
        writes = capture["bb"].writes
        assert len(writes) == 1
        key, val = writes[0]
        assert key == "guardrail_waiver:response:w_xyz"
        assert val["approved"] is False
        assert val["reason"] == "no"
        # Explicitly no operator-override write.
        assert not any(
            k.startswith("operator_override:semantic_constraint:")
            for k, _ in writes
        )
    finally:
        _restore(capture)
