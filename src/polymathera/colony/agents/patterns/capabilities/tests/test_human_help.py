"""Unit tests for :class:`HumanHelpCapability` (Bucket B.8 — the
``request_help`` escalation primitive).

The capability is exercised in detached mode against an in-memory
blackboard backend; the four-layer chain (agent → session blackboard
→ Web UI → session blackboard → agent event handler) is verified by
writing the operator's response payload directly to the blackboard
and observing that the capability's event handler fires + that
``get_response`` returns the expected typed result.

Not tested here: the Web UI HTTP endpoint and the SessionAgent's
relay to the chat UI — those land with the frontend work and have
their own tests.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanHelpProtocol
from polymathera.colony.agents.patterns.capabilities.human_help import (
    HumanHelpCapability,
    HumanHelpRequest,
    HumanHelpResponse,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


async def _make_capability(_exec_ctx) -> HumanHelpCapability:
    cap = HumanHelpCapability(
        agent=None,
        capability_key="hhc_test",
        app_name="test_app",
    )
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=cap.scope_id,
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb
    return cap


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


async def test_request_writes_typed_payload_to_session_scope(
    _exec_ctx,
) -> None:
    cap = await _make_capability(_exec_ctx)
    result = await cap.request_help(
        question="Which of two designs should I pick?",
        context="A failed; B partially worked.",
        options=("A", "B", "restart"),
    )
    assert result["ok"] is True
    rid = result["request_id"]
    assert rid.startswith("help_")
    bb = await cap.get_blackboard()
    raw = await bb.read(HumanHelpProtocol.request_key(rid))
    assert isinstance(raw, dict)
    request = HumanHelpRequest.model_validate(raw)
    assert request.question == "Which of two designs should I pick?"
    assert request.context == "A failed; B partially worked."
    assert request.options == ("A", "B", "restart")


async def test_request_id_is_unique_across_calls(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    a = (await cap.request_help(question="Q1"))["request_id"]
    b = (await cap.request_help(question="Q2"))["request_id"]
    assert a != b


async def test_request_rejects_empty_question(_exec_ctx) -> None:
    """``request_help`` with no question has no signal for the
    operator; failing here is better than surfacing a blank
    escalation card."""

    cap = await _make_capability(_exec_ctx)
    with pytest.raises(ValueError) as exc_info:
        await cap.request_help(question="")
    assert "empty" in str(exc_info.value).lower()


async def test_request_rejects_whitespace_question(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    with pytest.raises(ValueError):
        await cap.request_help(question="   \t\n  ")


async def test_request_accepts_brief_question_without_context_or_options(
    _exec_ctx,
) -> None:
    """Help requests can be made without ``context`` or ``options``
    — the agent may not have either in pathological cases, and
    blocking on them would discourage the escalation."""

    cap = await _make_capability(_exec_ctx)
    result = await cap.request_help(question="What now?")
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# Receive side — event handler + cache
# ---------------------------------------------------------------------------


async def test_event_handler_caches_response_and_returns_context(
    _exec_ctx,
) -> None:
    """When the operator's response lands on the blackboard, the
    @event_handler fires inside the capability's normal event loop,
    caches the response, and surfaces it as planner context the next
    iteration reads."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_help(
        question="Q", options=("A", "B"),
    ))["request_id"]

    response = HumanHelpResponse(
        request_id=rid,
        chosen_option="A",
        guidance="Go with A; B is too expensive.",
        decided_by="alice",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanHelpProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )

    # Drive the handler manually as the agent's event loop would.
    fake_event = type("E", (), {})()
    fake_event.key = HumanHelpProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    result = await cap._on_response(fake_event, None)

    assert result is not None
    assert result.context_key == (
        f"{HumanHelpCapability.RESPONSE_CONTEXT_KEY_PREFIX}{rid}"
    )
    assert result.context == {
        "request_id": rid,
        "chosen_option": "A",
        "guidance": "Go with A; B is too expensive.",
        "decided_by": "alice",
        "decided_at": response.decided_at.isoformat(),
    }
    # The cache survives — get_response should not need a blackboard
    # hit.
    envelope = await cap.get_response(rid)
    assert envelope["ok"] is True
    assert envelope["state"] == "ready"
    assert envelope["response"]["chosen_option"] == "A"
    assert envelope["response"]["guidance"] == "Go with A; B is too expensive."


async def test_event_handler_supports_free_form_guidance_without_option(
    _exec_ctx,
) -> None:
    """The operator can write free-form guidance without picking an
    option — useful when the agent's enumerated options didn't
    capture the right move."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_help(
        question="Stuck on design X. Suggestions?",
        options=("retry", "abandon"),
    ))["request_id"]

    response = HumanHelpResponse(
        request_id=rid,
        chosen_option=None,
        guidance="Neither — switch to design Y; the constraints changed.",
        decided_by="bob",
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanHelpProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    result = await cap._on_response(fake_event, None)

    assert result is not None
    assert result.context["chosen_option"] is None
    assert "design Y" in result.context["guidance"]


async def test_event_handler_drops_malformed_payload(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    fake_event = type("E", (), {})()
    fake_event.key = HumanHelpProtocol.response_key("help_bad")
    fake_event.value = "not-a-dict"
    assert await cap._on_response(fake_event, None) is None


async def test_event_handler_ignores_non_response_keys(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    fake_event = type("E", (), {})()
    fake_event.key = "some:other:key"
    fake_event.value = {"chosen_option": "A", "request_id": "x"}
    assert await cap._on_response(fake_event, None) is None


# ---------------------------------------------------------------------------
# get_response — cache + blackboard fallback
# ---------------------------------------------------------------------------


async def test_get_response_envelope_state_pending(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_help(question="Q"))["request_id"]
    envelope = await cap.get_response(rid)
    assert envelope == {"ok": True, "state": "pending", "response": None}


async def test_get_response_falls_back_to_blackboard(_exec_ctx) -> None:
    """A response that lands during agent suspension (so the in-process
    event handler never fires) is still recoverable through a direct
    blackboard read on resume."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_help(question="Q"))["request_id"]
    response = HumanHelpResponse(
        request_id=rid,
        chosen_option="A",
        guidance="picked A",
        decided_by="alice",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanHelpProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )
    # Skip the event handler — simulate the resume case.
    cap._responses.clear()

    envelope = await cap.get_response(rid)
    assert envelope["ok"] is True
    assert envelope["state"] == "ready"
    assert envelope["response"]["chosen_option"] == "A"


async def test_get_response_marks_malformed_payload(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_help(question="Q"))["request_id"]
    bb = await cap.get_blackboard()
    await bb.write(
        HumanHelpProtocol.response_key(rid),
        {"request_id": rid, "chosen_option": None,
         "guidance": object()},
    )
    envelope = await cap.get_response(rid)
    assert envelope["ok"] is False
    assert envelope["error"] == "malformed_response_payload"


# ---------------------------------------------------------------------------
# Idle-wait counter bookkeeping
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal stand-in providing the ``idle_wait_counter`` attribute
    the capability mutates. Keeps the test independent of the
    AgentBase hierarchy."""

    def __init__(self) -> None:
        self.agent_id = "stub-agent"
        self.idle_wait_counter = 0


async def test_pending_polls_increment_idle_counter_idempotently(
    _exec_ctx,
) -> None:
    """Mirrors :class:`HumanApprovalCapability` — repeated pending
    polls of the same request_id increment the agent's idle-wait
    counter ONCE, not on every call."""

    cap = await _make_capability(_exec_ctx)
    cap._agent = _StubAgent()
    rid = (await cap.request_help(question="Q"))["request_id"]
    assert cap._agent.idle_wait_counter == 0
    await cap.get_response(rid)  # pending → counter 1
    await cap.get_response(rid)  # pending again → still 1
    await cap.get_response(rid)
    assert cap._agent.idle_wait_counter == 1


async def test_response_arrival_decrements_idle_counter(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    cap._agent = _StubAgent()
    rid = (await cap.request_help(question="Q"))["request_id"]
    await cap.get_response(rid)
    assert cap._agent.idle_wait_counter == 1

    response = HumanHelpResponse(
        request_id=rid, chosen_option="A",
        guidance="g", decided_by="x",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanHelpProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanHelpProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    await cap._on_response(fake_event, None)
    assert cap._agent.idle_wait_counter == 0


async def test_response_context_key_prefix_is_classvar(_exec_ctx) -> None:
    """Single source of truth — referenced by-attribute by any
    advisor / guardrail that needs to quote the prefix. Pinned so a
    rename surfaces here, not in a string-match downstream."""

    assert HumanHelpCapability.RESPONSE_CONTEXT_KEY_PREFIX == (
        "human_help_response:"
    )


# ---------------------------------------------------------------------------
# HumanHelpResponse validator — chosen_option / guidance exclusivity
# ---------------------------------------------------------------------------


def test_response_rejects_both_fields_empty() -> None:
    """The validator catches the empty-empty case at the data-shape
    boundary so the REST endpoint surfaces a 422 and the chat-UI
    form can refuse to submit. The blackboard write never happens
    for an empty response."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        HumanHelpResponse(
            request_id="rid",
            chosen_option=None,
            guidance="",
        )
    assert "at least one" in str(exc_info.value).lower()


def test_response_rejects_whitespace_guidance_when_no_option() -> None:
    """Whitespace-only ``guidance`` is empty in spirit — the agent
    can't translate ``"   "`` into typed state. The validator strips
    before deciding."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HumanHelpResponse(
            request_id="rid",
            chosen_option=None,
            guidance="   \t\n  ",
        )


def test_response_accepts_chosen_option_only() -> None:
    r = HumanHelpResponse(
        request_id="rid",
        chosen_option="A",
        guidance="",
    )
    assert r.chosen_option == "A"
    assert r.guidance == ""


def test_response_accepts_guidance_only() -> None:
    r = HumanHelpResponse(
        request_id="rid",
        chosen_option=None,
        guidance="Pick option C from the docs.",
    )
    assert r.chosen_option is None
    assert "option C" in r.guidance


def test_response_accepts_both() -> None:
    """The operator may pick an option AND add a guidance note — the
    agent's translator step can read both."""

    r = HumanHelpResponse(
        request_id="rid",
        chosen_option="A",
        guidance="Also, downstream caveat: watch for the X edge case.",
    )
    assert r.chosen_option == "A"
    assert "edge case" in r.guidance


def test_response_rejects_empty_string_chosen_option_alone() -> None:
    """``chosen_option=""`` is functionally equivalent to ``None`` —
    a picked-but-empty option carries no signal. Combined with empty
    guidance the validator must refuse."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HumanHelpResponse(
            request_id="rid",
            chosen_option="",
            guidance="",
        )
