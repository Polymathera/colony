"""Tests for the human-help relay on ``SessionOrchestratorCapability``.

Sibling of ``test_human_approval_relay.py``. The relay subscribes to
``human_help:request:*`` on the session's ``human_help`` scope and
translates each into a chat ``agent_question`` record on the chat
scope, stamped with ``kind="human_help"`` so the frontend routes the
operator's response via the dedicated
``/sessions/{id}/human_help/{request_id}/respond`` HTTP endpoint.

The capability is exercised in detached mode against an in-memory
blackboard backend; the live ``stream_events`` loop is covered by the
broader integration tests against a running cluster.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanHelpProtocol
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


async def _make_capability_and_chat_bb(_exec_ctx):
    """Build a detached SessionOrchestratorCapability with a fake
    ``self._agent`` (so ``handle_human_help_request`` has an
    agent_id fallback) and an in-memory chat blackboard."""

    cap = SessionOrchestratorCapability(
        agent=None,
        scope=BlackboardScope.SESSION,
        namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        capability_key="orchestrator_test",
        app_name="test_app",
    )
    cap._agent = SimpleNamespace(agent_id="session_agent_xyz")
    chat_bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=cap.scope_id,
        backend_type="memory",
        enable_events=False,
    )
    await chat_bb.initialize()
    cap._blackboard = chat_bb
    return cap, chat_bb


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


async def test_translates_request_into_chat_agent_question(_exec_ctx) -> None:
    """Round-trip the canonical shape: question + context + options +
    requester travels through; ``kind='human_help'`` is set; the
    ``context`` rides in ``extra`` so the frontend renders it above
    the response surface; ``awaiting_reply`` is True so the chat WS
    relay forwards as a typed ``agent_question``."""

    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    request_id = "help_xyz123"
    event = SimpleNamespace(
        key=HumanHelpProtocol.request_key(request_id),
        value={
            "request_id": request_id,
            "question": "Which decomposition strategy should I use?",
            "context": (
                "Tried 'too high-level' default. Issues #34/#36 didn't "
                "split — the LLM said they're already focused."
            ),
            "options": [
                "Require multiple independent deliverables",
                "Require explicit cross-team coordination",
            ],
            "requester_agent_id": "project_planning_coord_001",
        },
    )
    await cap.handle_human_help_request(event, None)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    payload = entries[0].value
    assert payload["awaiting_reply"] is True
    assert payload["kind"] == "human_help"
    assert payload["request_id"] == request_id
    assert payload["agent_type"] == "human_help"
    assert (
        payload["content"]
        == "Which decomposition strategy should I use?"
    )
    assert payload["agent_id"] == "project_planning_coord_001"
    assert payload["response_options"] == [
        "Require multiple independent deliverables",
        "Require explicit cross-team coordination",
    ]
    # Context rides in extra.context so the frontend renders it above
    # the response surface.
    assert payload["extra"]["context"].startswith(
        "Tried 'too high-level' default."
    )


async def test_translates_request_with_no_options(_exec_ctx) -> None:
    """When the agent has no candidate options to suggest, the
    operator gets a free-text-only card. The relay must not crash on
    an absent ``options`` field; the frontend gates option-button
    rendering on a non-empty list."""

    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    request_id = "help_no_opts"
    event = SimpleNamespace(
        key=HumanHelpProtocol.request_key(request_id),
        value={
            "request_id": request_id,
            "question": "What now?",
            # No options, no context.
            "requester_agent_id": "agent_q",
        },
    )
    await cap.handle_human_help_request(event, None)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    payload = entries[0].value
    assert payload["response_options"] == []
    assert payload["extra"] == {"context": ""}


async def test_translator_falls_back_to_session_agent_id_when_requester_missing(
    _exec_ctx,
) -> None:
    """When the request payload omits ``requester_agent_id`` (legacy
    callers or detached requests), the chat record falls back to the
    SessionAgent's own ``agent_id`` so the WebSocket relay still has a
    sender id to render."""

    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    request_id = "help_no_requester"
    event = SimpleNamespace(
        key=HumanHelpProtocol.request_key(request_id),
        value={
            "request_id": request_id,
            "question": "Help?",
        },
    )
    await cap.handle_human_help_request(event, None)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    payload = entries[0].value
    assert payload["agent_id"] == "session_agent_xyz"


async def test_translator_drops_malformed_request_key(_exec_ctx) -> None:
    """A key that does not match the protocol's request shape is
    dropped silently — same recovery as ``handle_human_approval_request``."""

    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    event = SimpleNamespace(
        key="not_a_human_help_key",
        value={"question": "ignored"},
    )
    await cap.handle_human_help_request(event, None)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert entries == []


async def test_translator_tolerates_non_dict_payload(_exec_ctx) -> None:
    """Defensive: a corrupt payload (non-dict value) must not crash
    the relay; the chat message just gets fallback strings."""

    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    request_id = "help_corrupt"
    event = SimpleNamespace(
        key=HumanHelpProtocol.request_key(request_id),
        value="this is not a dict",
    )
    await cap.handle_human_help_request(event, None)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    payload = entries[0].value
    assert payload["content"] == "(empty help request)"
    assert payload["awaiting_reply"] is True
    assert payload["kind"] == "human_help"
