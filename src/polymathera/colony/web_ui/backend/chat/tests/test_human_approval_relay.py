"""Tests for the human-approval relay on ``SessionOrchestratorCapability``.

The relay subscribes to ``human_approval:request:*`` on the session's
``human_approval`` scope and translates each into a chat
``agent_question`` record on the chat scope. The chat WebSocket relay
(:func:`_listen_for_agent_messages`) then forwards it to the browser.

These tests exercise the translation method directly with synthesized
``BlackboardEvent``s — the live ``stream_events`` loop is covered by
the broader integration tests against a running cluster.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanApprovalProtocol
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
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
    ``self._agent`` (so ``_handle_human_approval_request`` has an
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
    return cap, chat_bb


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


async def test_translates_request_into_chat_agent_question(_exec_ctx) -> None:
    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    request_id = "appr_xyz"
    event = SimpleNamespace(
        key=HumanApprovalProtocol.request_key(request_id),
        value={
            "request_id": request_id,
            "question": "Approve checkpoint cp_42?",
            "options": ["approve", "reject"],
            "requester_agent_id": "physics_agent_001",
            "extra": {"checkpoint_id": "cp_42"},
        },
    )
    await cap._handle_human_approval_request(event, chat_bb)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    payload = entries[0].value
    assert payload["awaiting_reply"] is True
    assert payload["kind"] == "human_approval"
    assert payload["request_id"] == request_id
    assert payload["response_options"] == ["approve", "reject"]
    assert payload["content"] == "Approve checkpoint cp_42?"
    assert payload["agent_id"] == "physics_agent_001"
    assert payload["extra"] == {"checkpoint_id": "cp_42"}


async def test_falls_back_to_session_agent_id_when_requester_missing(
    _exec_ctx,
) -> None:
    """When the request payload does not carry a ``requester_agent_id``
    (older payloads or detached writes), the chat record is attributed
    to the SessionAgent so it still renders in the chat panel."""

    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    event = SimpleNamespace(
        key=HumanApprovalProtocol.request_key("appr_a"),
        value={
            "request_id": "appr_a",
            "question": "Pick one",
            "options": ["a", "b"],
        },
    )
    await cap._handle_human_approval_request(event, chat_bb)

    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    assert entries[0].value["agent_id"] == "session_agent_xyz"


async def test_provides_default_options_when_payload_omits_them(
    _exec_ctx,
) -> None:
    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    event = SimpleNamespace(
        key=HumanApprovalProtocol.request_key("appr_b"),
        value={
            "question": "Proceed?",
            "requester_agent_id": "physics_agent_002",
        },
    )
    await cap._handle_human_approval_request(event, chat_bb)
    entries = await chat_bb.query(namespace="chat:agent:*")
    assert entries[0].value["response_options"] == ["approve", "reject"]


async def test_ignores_non_request_keys(_exec_ctx) -> None:
    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    event = SimpleNamespace(
        key="some:other:key",
        value={"question": "ignored"},
    )
    await cap._handle_human_approval_request(event, chat_bb)
    entries = await chat_bb.query(namespace="chat:agent:*")
    assert entries == []


async def test_tolerates_non_dict_payload(_exec_ctx) -> None:
    cap, chat_bb = await _make_capability_and_chat_bb(_exec_ctx)
    event = SimpleNamespace(
        key=HumanApprovalProtocol.request_key("appr_c"),
        value="not-a-dict",
    )
    # Should not raise; emits a placeholder agent_question with the
    # default options.
    await cap._handle_human_approval_request(event, chat_bb)
    entries = await chat_bb.query(namespace="chat:agent:*")
    assert len(entries) == 1
    assert entries[0].value["response_options"] == ["approve", "reject"]


# ---------------------------------------------------------------------------
# Scope discipline
# ---------------------------------------------------------------------------


async def test_human_approval_scope_id_matches_capability_default(
    _exec_ctx,
) -> None:
    """The relay constructs its blackboard handle against the
    ``human_approval`` namespace; this test pins the scope_id format
    so future scope renames break loudly here rather than silently
    breaking the relay."""

    from polymathera.colony.agents.patterns.capabilities.human_approval import (
        HumanApprovalCapability,
    )

    expected = get_scope_prefix(
        BlackboardScope.SESSION,
        namespace=HumanApprovalCapability.DEFAULT_NAMESPACE,
    )
    assert "human_approval" in expected
    cap_scope = HumanApprovalCapability(
        agent=None, capability_key="hac", app_name="test_app",
    ).scope_id
    assert cap_scope == expected
