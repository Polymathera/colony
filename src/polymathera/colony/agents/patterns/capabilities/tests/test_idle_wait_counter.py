"""Tests for the idle-wait counter integration between
``HumanApprovalCapability`` and ``Agent``.

Item 6 of ``colony/decompose_and_session_recovery_fixes_plan.md``:
the agent's ``idle_wait_counter`` is incremented by the capability on
the FIRST pending poll for a given request_id, decremented when that
request_id resolves. Repeated pending polls for the same id are
idempotent. Resolutions arriving via either ``get_response`` or the
event-handler path land the decrement.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.patterns.capabilities.human_approval import (
    HumanApprovalCapability,
    HumanApprovalProtocol,
    HumanApprovalResponse,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_ctx():
    """Reuse the same execution-context fixture pattern as the
    sibling test_human_approval.py file."""

    from polymathera.colony.distributed.ray_utils.serving.context import (
        user_execution_context,
    )

    with user_execution_context(
        tenant_id="tenant_test",
        colony_id="colony_test",
        session_id="session_test",
        origin="test",
    ) as ctx:
        yield ctx


class _FakeAgent:
    """Minimal stand-in for ``Agent``: holds an ``idle_wait_counter``
    and exposes the ``is_idle_waiting`` derived property the loop
    reads. Production ``Agent`` uses a Pydantic field with
    ``ge=0``; here we keep an int for simplicity (the bookkeeping
    contract is the same)."""

    def __init__(self) -> None:
        self.idle_wait_counter: int = 0
        self.agent_id = "agent-test"

    @property
    def is_idle_waiting(self) -> bool:
        return self.idle_wait_counter > 0


async def _make_capability(_exec_ctx) -> HumanApprovalCapability:
    from polymathera.colony.agents.blackboard import EnhancedBlackboard

    cap = HumanApprovalCapability(agent=None, scope_id="test")
    cap._agent = _FakeAgent()
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id="test",
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb
    return cap


async def test_first_pending_poll_increments_counter(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q?"))["request_id"]
    assert cap._agent.idle_wait_counter == 0

    envelope = await cap.get_response(rid)
    assert envelope["state"] == "pending"
    assert cap._agent.idle_wait_counter == 1
    assert cap._agent.is_idle_waiting


async def test_repeated_pending_polls_are_idempotent(_exec_ctx) -> None:
    """The agent loop will poll the same pending id many times; the
    counter must stay at 1, not climb."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q?"))["request_id"]

    for _ in range(5):
        envelope = await cap.get_response(rid)
        assert envelope["state"] == "pending"
    assert cap._agent.idle_wait_counter == 1


async def test_resolved_via_get_response_decrements_counter(_exec_ctx) -> None:
    """A resolution observed by ``get_response`` (blackboard
    fallback path) lands the decrement."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q?"))["request_id"]
    await cap.get_response(rid)  # pending → counter 1
    assert cap._agent.idle_wait_counter == 1

    # Land the response on the blackboard directly (simulates the
    # UI POST). Skip the event handler and use the get_response
    # blackboard-fallback path so we exercise that code's
    # _on_resolved hook.
    response = HumanApprovalResponse(
        request_id=rid, choice="approve", decided_by="alice",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )
    cap._responses.clear()  # force the fallback path

    envelope = await cap.get_response(rid)
    assert envelope["state"] == "ready"
    assert cap._agent.idle_wait_counter == 0
    assert not cap._agent.is_idle_waiting


async def test_resolved_via_event_handler_decrements_counter(_exec_ctx) -> None:
    """A resolution observed by the ``@event_handler`` path lands
    the decrement too — the agent loop never has to poll
    ``get_response`` a second time."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q?"))["request_id"]
    await cap.get_response(rid)  # pending → counter 1
    assert cap._agent.idle_wait_counter == 1

    response = HumanApprovalResponse(
        request_id=rid, choice="reject", decided_by="bob",
        explanation="Out of scope for this run.",
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    await cap._on_response(fake_event, None)

    assert cap._agent.idle_wait_counter == 0


async def test_repeated_resolutions_are_idempotent(_exec_ctx) -> None:
    """A second cached read of a resolved response must not
    decrement the counter further (would underflow)."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Q?"))["request_id"]
    await cap.get_response(rid)  # pending → counter 1

    response = HumanApprovalResponse(
        request_id=rid, choice="approve", decided_by="alice",
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    await cap._on_response(fake_event, None)
    assert cap._agent.idle_wait_counter == 0

    # Subsequent reads of the now-cached response: no underflow.
    for _ in range(5):
        envelope = await cap.get_response(rid)
        assert envelope["state"] == "ready"
    assert cap._agent.idle_wait_counter == 0


async def test_n_concurrent_pollers_compose_independently(_exec_ctx) -> None:
    """Two distinct request_ids → counter climbs to 2 then comes
    back down as each resolves — the counter naturally composes
    across N concurrent pollers without naming concerns."""

    cap = await _make_capability(_exec_ctx)
    rid_a = (await cap.request_human_approval(question="A?"))["request_id"]
    rid_b = (await cap.request_human_approval(question="B?"))["request_id"]

    await cap.get_response(rid_a)
    await cap.get_response(rid_b)
    assert cap._agent.idle_wait_counter == 2

    # Resolve A.
    bb = await cap.get_blackboard()
    resp_a = HumanApprovalResponse(
        request_id=rid_a, choice="approve", decided_by="x",
    )
    await bb.write(
        HumanApprovalProtocol.response_key(rid_a),
        resp_a.model_dump(mode="json"),
    )
    cap._responses.clear()
    await cap.get_response(rid_a)
    assert cap._agent.idle_wait_counter == 1

    # Resolve B.
    resp_b = HumanApprovalResponse(
        request_id=rid_b, choice="reject", decided_by="x",
        explanation="rejected for test",
    )
    await bb.write(
        HumanApprovalProtocol.response_key(rid_b),
        resp_b.model_dump(mode="json"),
    )
    cap._responses.clear()
    await cap.get_response(rid_b)
    assert cap._agent.idle_wait_counter == 0


async def test_detached_capability_no_op_on_counter(_exec_ctx) -> None:
    """When ``self._agent is None`` (test/detached construction), the
    counter hooks are no-ops — never raise on missing agent."""

    cap = HumanApprovalCapability(agent=None, scope_id="test")
    from polymathera.colony.agents.blackboard import EnhancedBlackboard

    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id="test",
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb

    rid = (await cap.request_human_approval(question="Q?"))["request_id"]
    # Should NOT raise.
    envelope = await cap.get_response(rid)
    assert envelope["state"] == "pending"
