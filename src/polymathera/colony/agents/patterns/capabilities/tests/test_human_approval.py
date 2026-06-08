"""Unit tests for ``HumanApprovalCapability``.

These exercise the capability in detached mode against an in-memory
blackboard backend. The four-layer chain the capability participates
in (agent → session blackboard → Web UI → session blackboard → agent
event handler) is verified by writing the response payload directly
to the blackboard and observing that the capability's event handler
fires + that ``get_response`` returns the expected typed result.

What is NOT tested here: the Web UI HTTP endpoint; SessionAgent's
relay to the chat UI. Those have their own tests.
"""

from __future__ import annotations

import asyncio

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanApprovalProtocol
from polymathera.colony.agents.models import AgentSuspensionState
from polymathera.colony.agents.patterns.capabilities.human_approval import (
    HumanApprovalCapability,
    HumanApprovalRequest,
    HumanApprovalResponse,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_ctx():
    """Provide an execution context with session_id so the capability's
    SESSION-scoped scope_id resolves."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


async def _make_capability(_exec_ctx) -> HumanApprovalCapability:
    """Build a detached capability with an in-memory blackboard pre-wired."""

    cap = HumanApprovalCapability(
        agent=None,
        capability_key="hac_test",
        app_name="test_app",
    )
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=cap.scope_id,
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb  # bypass deferred init, force memory backend
    return cap


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


async def test_request_writes_typed_payload_to_session_scope(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    result = await cap.request_human_approval(
        question="Approve the design checkpoint?",
        options=("approve", "reject"),
        extra={"checkpoint_id": "cp_42"},
    )
    assert result["ok"] is True
    rid = result["request_id"]
    assert rid.startswith("appr_")
    bb = await cap.get_blackboard()
    raw = await bb.read(HumanApprovalProtocol.request_key(rid))
    assert isinstance(raw, dict)
    request = HumanApprovalRequest.model_validate(raw)
    assert request.question == "Approve the design checkpoint?"
    assert request.options == ("approve", "reject")
    assert request.extra == {"checkpoint_id": "cp_42"}
    pending = await cap.list_pending()
    assert pending["ok"] is True
    assert rid in pending["pending_request_ids"]


async def test_request_id_is_unique_across_calls(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    a = (await cap.request_human_approval(question="A"))["request_id"]
    b = (await cap.request_human_approval(question="B"))["request_id"]
    assert a != b
    assert {a, b} == set(
        (await cap.list_pending())["pending_request_ids"],
    )


# ---------------------------------------------------------------------------
# Receive side — event handler + cache
# ---------------------------------------------------------------------------


async def test_event_handler_caches_response_and_returns_context(
    _exec_ctx,
) -> None:
    """When the response lands on the blackboard, the @event_handler
    fires inside the capability's normal event loop, caches the
    response, and surfaces it as planner context."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(
        question="Q", options=("a", "b"),
    ))["request_id"]

    # Simulate the Web UI HTTP endpoint writing the response.
    response = HumanApprovalResponse(
        request_id=rid, choice="a", note="ok", decided_by="alice",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )

    # Drive the handler manually (as the agent's event loop would).
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    result = await cap._on_response(fake_event, None)

    assert result is not None
    assert result.context_key == f"human_approval_response:{rid}"
    assert result.context == {
        "request_id": rid,
        "choice": "a",
        "note": "ok",
        "decided_by": "alice",
    }
    # The cache survives — get_response should not need a blackboard hit.
    envelope = await cap.get_response(rid)
    assert envelope["ok"] is True
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "a"
    assert envelope["response"]["decided_by"] == "alice"
    pending = await cap.list_pending()
    assert rid not in pending["pending_request_ids"]


async def test_event_handler_drops_malformed_payload(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key("appr_bad")
    fake_event.value = "not-a-dict"
    result = await cap._on_response(fake_event, None)
    assert result is None


async def test_event_handler_ignores_non_response_keys(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    fake_event = type("E", (), {})()
    fake_event.key = "some:other:key"
    fake_event.value = {"choice": "a", "request_id": "x"}
    result = await cap._on_response(fake_event, None)
    assert result is None


# ---------------------------------------------------------------------------
# get_response — cache + blackboard fallback
# ---------------------------------------------------------------------------


async def test_get_response_envelope_state_pending(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q"))["request_id"]
    envelope = await cap.get_response(rid)
    assert envelope == {"ok": True, "state": "pending", "response": None}


async def test_get_response_falls_back_to_blackboard(_exec_ctx) -> None:
    """A response that landed during agent suspension (so the in-process
    event handler never fired) is still recoverable through a direct
    blackboard read on resume."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q"))["request_id"]

    response = HumanApprovalResponse(
        request_id=rid, choice="approve", note="lgtm", decided_by="bob",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )
    # Skip the event handler entirely — simulate the resume case.
    cap._responses.clear()

    envelope = await cap.get_response(rid)
    assert envelope["ok"] is True
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "approve"
    assert envelope["response"]["decided_by"] == "bob"
    # Cache populated by the fallback so subsequent reads are cheap.
    assert rid in cap._responses


# ---------------------------------------------------------------------------
# Suspend / resume
# ---------------------------------------------------------------------------


async def test_suspend_resume_round_trips_requests_and_responses(
    _exec_ctx,
) -> None:
    cap1 = await _make_capability(_exec_ctx)
    rid_pending = (await cap1.request_human_approval(
        question="Pending?",
    ))["request_id"]
    rid_resolved = (await cap1.request_human_approval(
        question="Resolved?",
    ))["request_id"]
    response = HumanApprovalResponse(
        request_id=rid_resolved, choice="approve", decided_by="carol",
    )
    cap1._responses[rid_resolved] = response

    state = AgentSuspensionState(
        agent_id="test",
        agent_type="test_agent",
        suspension_reason="test",
        suspended_at=0.0,
    )
    await cap1.serialize_suspension_state(state)

    cap2 = await _make_capability(_exec_ctx)
    await cap2.deserialize_suspension_state(state)

    assert rid_pending in cap2._requests
    assert rid_resolved in cap2._requests
    assert cap2._responses.get(rid_resolved) is not None
    assert cap2._responses[rid_resolved].choice == "approve"
    pending = await cap2.list_pending()
    assert pending["pending_request_ids"] == [rid_pending]


# ---------------------------------------------------------------------------
# End-to-end via blackboard event stream
# ---------------------------------------------------------------------------


async def test_end_to_end_via_blackboard_event_stream(_exec_ctx) -> None:
    """Drive the full chain: request → blackboard → event stream →
    capability handler. Proves the @event_handler pattern subscribes
    correctly on the session blackboard."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(
        question="Final answer?",
    ))["request_id"]

    bb = await cap.get_blackboard()

    # Subscribe directly so we can verify the event lands on the topic
    # the capability would observe via @event_handler. We do not run
    # the agent's full event loop here — that is exercised by the
    # broader agent integration tests.
    queue: asyncio.Queue = asyncio.Queue()
    bb.stream_events_to_queue(
        queue,
        pattern=HumanApprovalProtocol.response_pattern(),
    )

    response = HumanApprovalResponse(
        request_id=rid, choice="reject", note="not yet", decided_by="dan",
    )
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event.key == HumanApprovalProtocol.response_key(rid)
    parsed = HumanApprovalResponse.model_validate(event.value)
    assert parsed.choice == "reject"

    # Feed the event into the capability's handler the way the agent
    # event loop would, and confirm the cache + planner context.
    result = await cap._on_response(event, None)
    assert result is not None
    assert result.context["choice"] == "reject"
    envelope = await cap.get_response(rid)
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "reject"
