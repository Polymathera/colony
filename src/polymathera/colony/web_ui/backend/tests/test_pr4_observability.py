"""Tests for PR4 (R12 ROOT-CAUSE-B) — observability sweep.

Pins the four silent-failure fixes shipped together:

- B5: dispatcher output-validation failure → ActionResult(success=False)
  (was: log+swallow, returned success=True with unvalidated dict)
- B6: chat router listener tasks → add_done_callback that logs + notifies
  the browser (was: fire-and-forget, silent chat hang on crash)
- B10: github_inbound poll crash → diagnostic blackboard emit
  (was: only log; webhook pipeline silently dies)
- B12: handle_user_reply → honest user-facing message naming the
  correct surface (was: "Reply acknowledged" lying about routing)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from polymathera.colony.agents.models import ActionResult


# ---------------------------------------------------------------------------
# B5: dispatcher output-validation failure surfaces as success=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_output_validation_failure_returns_success_false() -> None:
    """Output-validation failure used to log+swallow and return
    success=True with the unvalidated dict. The LLM then consumed
    garbage as schema-valid. Pin the post-fix shape: success=False
    with the validation error in ``error``."""

    from pydantic import BaseModel
    from polymathera.colony.agents.models import Action
    from polymathera.colony.agents.patterns.actions.dispatcher import (
        MethodWrapperActionExecutor,
    )

    class _Out(BaseModel):
        x: int

    class _Target:
        async def m(self) -> dict:
            return {"x": "not an int"}  # will fail validation

    wrapper = MethodWrapperActionExecutor(
        object=_Target(),
        method=_Target.m,
        action_key="t.bad",
        output_schema=_Out,
    )
    action = Action(
        action_id="a1",
        agent_id="agent-test",
        action_type="t.bad",
        parameters={},
    )
    result: ActionResult = await wrapper.execute(action)
    assert result.success is False
    assert result.error is not None
    assert "validation failed" in result.error.lower()


# ---------------------------------------------------------------------------
# B6: chat router wires add_done_callback on listener tasks
# ---------------------------------------------------------------------------


def test_chat_router_listener_tasks_have_done_callback() -> None:
    """Source-pin: each listener task created in the WebSocket
    handler MUST get an ``add_done_callback`` so a crash is logged
    and surfaced to the browser (not silently swallowed)."""

    src = (
        Path(__file__).resolve().parents[1]
        / "routers"
        / "chat.py"
    ).read_text(encoding="utf-8")
    # The three task creations and matching add_done_callback wires.
    for stream in (
        "_listen_for_agent_messages",
        "_listen_for_action_status",
        "_listen_for_mission_status",
    ):
        assert stream in src
    # The callback factory exists and is wired three times.
    assert src.count(".add_done_callback(_listener_done(") == 3


# ---------------------------------------------------------------------------
# B10: github_inbound emits diagnostic on poll-loop crash
# ---------------------------------------------------------------------------


def test_github_inbound_emits_diagnostic_on_poll_crash() -> None:
    """Source-pin: the poll-loop crash branch calls
    ``_emit_quiesced_diagnostic(exc)`` so operator-facing tools see
    the inbound went dark. Without it, the webhook pipeline silently
    dies and the operator only notices via missing issues."""

    src = (
        Path(__file__).resolve().parents[3]
        / "agents"
        / "patterns"
        / "capabilities"
        / "github_inbound"
        / "capability.py"
    ).read_text(encoding="utf-8")
    assert "poll_loop_crashed" in src
    assert "_emit_quiesced_diagnostic" in src
    assert "github_inbound_quiesced" in src


# ---------------------------------------------------------------------------
# B12: handle_user_reply ack is honest about the routing gap
# ---------------------------------------------------------------------------


def test_handle_user_reply_routes_to_typed_protocol() -> None:
    """``handle_user_reply`` must actually route the user's chat
    reply to the matching typed protocol response key, not just
    print a sorry-this-isn't-wired message. Help replies map to
    HumanHelpResponse(guidance=...) on the human_help namespace;
    approval replies surface an error directing the user to the
    typed buttons (freeform can't produce a discrete choice).
    Unknown request_ids surface explicitly so nothing rots silently."""

    from polymathera.colony.web_ui.backend.chat import session_agent
    src = inspect.getsource(
        session_agent.SessionOrchestratorCapability.handle_user_reply,
    )
    # The help routing: looks up the request, writes a response.
    assert "HumanHelpProtocol.request_key(request_id)" in src
    assert "HumanHelpProtocol.response_key(request_id)" in src
    assert "HumanHelpResponse(guidance=content)" in src
    # Approval: detects the request, tells user to use buttons.
    assert "HumanApprovalProtocol.request_key(request_id)" in src
    assert "approve" in src and "reject" in src
    # Unknown id: explicit error, no silent drop.
    assert "No live request found for" in src
    # Operator-visible logging on every branch.
    assert "logger.warning" in src
    assert "logger.info" in src
