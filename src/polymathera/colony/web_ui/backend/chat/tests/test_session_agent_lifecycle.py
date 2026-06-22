"""Tests for the SessionAgent stop-callback in
``web_ui/backend/chat/session_agent_lifecycle.py``.

The callback is wired into ``SessionAgent.bind(...).stop_callbacks``
for both the user-chat SessionAgent (``routers/sessions.py``) and the
colony system SessionAgent (``chat/system_session.py``). When an
agent terminates for any reason the framework's ``Agent.stop()``
fires the callback with ``(agent, reason)``; the callback writes a
typed ``chat:agent:*`` system_failure message AND emits an
``AgentDiagnosticProtocol`` event so the user (and operator-facing
tools) see what happened instead of a silently-locked chat.

What we pin:

1. UNEXPECTED reasons (max_iterations, max_code_iterations,
   policy_completed, error, unknown, anything else) produce a typed
   ``chat:agent:*`` system_failure write AND a diagnostic event.
2. EXPECTED reasons (stop_requested, cancelled) produce NEITHER —
   operator-initiated shutdown isn't a failure.
3. The chat-message payload carries the verbatim ``stop_reason`` so
   operator-facing log grep keeps working.
4. The chat-message body translates the internal reason string into
   plain language for the user (e.g. ``max_code_iterations`` →
   "I exhausted my code-generation iteration budget on this turn.").
5. A missing ``SessionOrchestratorCapability`` (misconfigured
   blueprint) doesn't crash the callback — the chat write is
   skipped, the diagnostic emit still runs.
6. Defensive: a blackboard exception in EITHER the chat write OR the
   diagnostic emit is caught and logged; the other emit still
   attempts. ``Agent.stop()`` already catches+logs callback
   exceptions, but the inner defense keeps a partial degradation
   from becoming a full silent death.
"""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
from polymathera.colony.web_ui.backend.chat.chat_protocol import (
    SessionChatProtocol,
)
from polymathera.colony.web_ui.backend.chat.session_agent import (
    SessionOrchestratorCapability,
)
from polymathera.colony.web_ui.backend.chat.session_agent_lifecycle import (
    DIAGNOSTIC_SESSION_AGENT_STOPPED,
    session_agent_stopped,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def _exec_ctx():
    """``session_agent_stopped`` resolves the diagnostic scope via
    ``get_scope_prefix(BlackboardScope.SESSION, ...)`` which reads
    the ambient execution context. Provide one for every test."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


async def _make_chat_bb(scope_suffix: str) -> EnhancedBlackboard:
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=f"test:{scope_suffix}",
        backend_type="memory",
        enable_events=False,
    )
    await bb.initialize()
    return bb


class _FakeOrchestratorCapability:
    """Stub the capability surface ``session_agent_stopped`` reads —
    just the ``get_blackboard()`` method that returns the (pre-built)
    chat blackboard for the session."""

    def __init__(self, chat_bb: EnhancedBlackboard) -> None:
        self._chat_bb = chat_bb

    async def get_blackboard(self) -> EnhancedBlackboard:
        return self._chat_bb


class _FakeAgent:
    """Stub the agent surface the callback reads:
    ``agent_id``, ``agent_type``, ``get_capability_by_type``,
    ``get_blackboard(scope_id=...)``."""

    def __init__(
        self,
        *,
        chat_bb: EnhancedBlackboard | None,
        diag_bb: EnhancedBlackboard | None,
        agent_id: str = "agent-test-session",
        agent_type: str = "SessionAgent",
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        if chat_bb is not None:
            self._cap = _FakeOrchestratorCapability(chat_bb)
        else:
            self._cap = None
        self._diag_bb = diag_bb

    def get_capability_by_type(self, capability_type: type) -> Any:
        if (
            capability_type is SessionOrchestratorCapability
            and self._cap is not None
        ):
            return self._cap
        return None

    async def get_blackboard(
        self, scope_id: str | None = None, **_: Any,
    ) -> EnhancedBlackboard | None:
        return self._diag_bb


async def _chat_messages(
    chat_bb: EnhancedBlackboard,
) -> list[tuple[str, dict]]:
    """Return ``[(key, payload), ...]`` for every ``chat:agent:*`` entry
    on the bb. Uses the public ``query`` API to avoid coupling to
    backend internals."""

    entries = await chat_bb.query(namespace="chat:agent:*")
    return [(e.key, e.value) for e in entries]


async def _diag_events(
    diag_bb: EnhancedBlackboard,
) -> list[tuple[str, dict]]:
    entries = await diag_bb.query(namespace="agent:diagnostic:*")
    return [(e.key, e.value) for e in entries]


# ---------------------------------------------------------------------------
# 1. Unexpected reasons → chat + diagnostic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        "max_iterations",
        "max_code_iterations",
        "policy_completed",
        "idle_timeout",
        "error",
        "unknown",
        "something_brand_new_we_didnt_anticipate",
    ],
)
async def test_unexpected_reason_writes_chat_system_failure(
    _exec_ctx, reason: str,
) -> None:
    """Every non-expected reason produces a typed system_failure
    chat message. The set defaults to user-visible — new failure
    modes are surfaced automatically."""

    chat_bb = await _make_chat_bb(f"chat_{reason}")
    diag_bb = await _make_chat_bb(f"diag_{reason}")
    agent = _FakeAgent(chat_bb=chat_bb, diag_bb=diag_bb)

    await session_agent_stopped(agent, reason)

    msgs = await _chat_messages(chat_bb)
    assert len(msgs) == 1
    key, payload = msgs[0]
    assert key.startswith(f"chat:agent:{agent.agent_id}:msg_")
    assert payload["kind"] == "system_failure"
    assert payload["agent_id"] == agent.agent_id
    assert payload["agent_type"] == agent.agent_type
    assert payload["stop_reason"] == reason
    # Reason is rendered VERBATIM in the body too so log/UI grep works.
    assert reason in payload["content"]


@pytest.mark.parametrize(
    "reason,expected_plain_fragment",
    [
        ("max_iterations", "planning iteration budget"),
        ("max_code_iterations", "code-generation iteration budget"),
        ("policy_completed", "action policy reported completion"),
        ("idle_timeout", "idled past my session timeout"),
        ("error", "internal error stopped"),
        ("unknown", "stopped for an unknown reason"),
    ],
)
async def test_chat_message_translates_internal_reason_into_plain_language(
    _exec_ctx, reason: str, expected_plain_fragment: str,
) -> None:
    """The user reads the body, not the reason field. Plain language
    is the durable shape — the operator can still grep ``stop_reason``
    or the trailing ``Internal stop reason: <reason>`` line."""

    chat_bb = await _make_chat_bb(f"chat_{reason}")
    diag_bb = await _make_chat_bb(f"diag_{reason}")
    agent = _FakeAgent(chat_bb=chat_bb, diag_bb=diag_bb)

    await session_agent_stopped(agent, reason)

    msgs = await _chat_messages(chat_bb)
    assert expected_plain_fragment in msgs[0][1]["content"]


async def test_unexpected_reason_emits_diagnostic_event(
    _exec_ctx,
) -> None:
    """The operator-facing surface: a typed AgentDiagnosticProtocol
    event lets dashboard/log adapters / future auto-respawn loops
    react to SessionAgent death without polling agent state."""

    chat_bb = await _make_chat_bb("chat_diag_emit")
    diag_bb = await _make_chat_bb("diag_emit")
    agent = _FakeAgent(chat_bb=chat_bb, diag_bb=diag_bb)

    await session_agent_stopped(agent, "max_code_iterations")

    events = await _diag_events(diag_bb)
    assert len(events) == 1
    key, payload = events[0]
    parsed = AgentDiagnosticProtocol.parse_event_key(key)
    assert parsed["agent_id"] == agent.agent_id
    assert parsed["kind"] == DIAGNOSTIC_SESSION_AGENT_STOPPED
    assert payload["agent_id"] == agent.agent_id
    assert payload["kind"] == DIAGNOSTIC_SESSION_AGENT_STOPPED
    assert payload["stop_reason"] == "max_code_iterations"


# ---------------------------------------------------------------------------
# 2. Expected reasons → no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["stop_requested", "cancelled"])
async def test_expected_reason_writes_nothing(
    _exec_ctx, reason: str,
) -> None:
    """Operator-initiated shutdown (and graceful cancel) are not
    failures — surfacing them as system_failure would be chat noise.
    No chat write, no diagnostic emit."""

    chat_bb = await _make_chat_bb(f"chat_{reason}")
    diag_bb = await _make_chat_bb(f"diag_{reason}")
    agent = _FakeAgent(chat_bb=chat_bb, diag_bb=diag_bb)

    await session_agent_stopped(agent, reason)

    assert await _chat_messages(chat_bb) == []
    assert await _diag_events(diag_bb) == []


# ---------------------------------------------------------------------------
# 3. Defensive paths
# ---------------------------------------------------------------------------


async def test_missing_orchestrator_capability_still_emits_diagnostic(
    _exec_ctx,
) -> None:
    """A misconfigured blueprint (SessionAgent without
    SessionOrchestratorCapability) shouldn't crash the callback.
    The chat write is skipped (no surface), but the operator-facing
    diagnostic emit still runs so the misconfiguration is visible."""

    diag_bb = await _make_chat_bb("diag_no_cap")
    # No chat_bb → agent.get_capability_by_type returns None
    agent = _FakeAgent(chat_bb=None, diag_bb=diag_bb)

    await session_agent_stopped(agent, "max_code_iterations")

    # Diagnostic still emitted.
    events = await _diag_events(diag_bb)
    assert len(events) == 1


async def test_chat_write_exception_does_not_propagate(_exec_ctx) -> None:
    """``Agent.stop`` already catches+logs callback exceptions, but
    a permanent silent chat is strictly worse than a partial
    degradation. Pin defense-in-depth: a raising chat-bb still
    lets the diagnostic emit attempt + the callback returns
    cleanly."""

    class _RaisingChatBB:
        async def write(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("simulated backend down")

    cap = _FakeOrchestratorCapability(_RaisingChatBB())  # type: ignore[arg-type]
    diag_bb = await _make_chat_bb("diag_after_chat_err")

    class _Agent:
        agent_id = "agent-test-defense"
        agent_type = "SessionAgent"

        def get_capability_by_type(self, t: type) -> Any:
            return cap if t is SessionOrchestratorCapability else None

        async def get_blackboard(self, scope_id: str | None = None, **_: Any):
            return diag_bb

    await session_agent_stopped(_Agent(), "max_code_iterations")  # must not raise

    # Diagnostic still emitted even though chat write failed.
    assert len(await _diag_events(diag_bb)) == 1


async def test_diagnostic_emit_exception_does_not_propagate(_exec_ctx) -> None:
    """Symmetric to the prior test: a raising diagnostic backend
    cannot block the callback. Chat write still landed; the
    diagnostic failure is logged-and-swallowed."""

    chat_bb = await _make_chat_bb("chat_after_diag_err")

    class _RaisingDiagBB:
        async def write(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("simulated diag-bb down")

    class _Agent:
        agent_id = "agent-test-defense2"
        agent_type = "SessionAgent"
        _cap = _FakeOrchestratorCapability(chat_bb)

        def get_capability_by_type(self, t: type) -> Any:
            return self._cap if t is SessionOrchestratorCapability else None

        async def get_blackboard(self, scope_id: str | None = None, **_: Any):
            return _RaisingDiagBB()

    await session_agent_stopped(_Agent(), "policy_completed")  # must not raise

    # Chat message still landed.
    msgs = await _chat_messages(chat_bb)
    assert len(msgs) == 1


# ---------------------------------------------------------------------------
# 4. Wiring pins — confirm the callback is mounted on both blueprints
# ---------------------------------------------------------------------------


def test_user_session_blueprint_mounts_callback(_exec_ctx) -> None:
    """Source-level pin: the user SessionAgent blueprint registers
    the stop callback. PR1-B moved the blueprint construction from
    ``routers/sessions.py`` into ``chat/user_session_factory.py``
    so both create-session AND respawn share the same shape — the
    pin follows the code."""

    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "user_session_factory.py"
    ).read_text(encoding="utf-8")
    assert "session_agent_stopped" in src
    assert "stop_callbacks=[session_agent_stopped]" in src


def test_system_session_blueprint_mounts_callback(_exec_ctx) -> None:
    """Source-level pin for the colony system session — same
    death-notification path as the user session."""

    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1]
        / "system_session.py"
    ).read_text(encoding="utf-8")
    assert "session_agent_stopped" in src
    assert "stop_callbacks=[session_agent_stopped]" in src


async def test_diagnostic_emit_uses_colony_scope(_exec_ctx) -> None:
    """The diagnostic event MUST be emitted on COLONY scope so the
    colony-mounted ``InteractionLogCapability._on_agent_diagnostic``
    handler captures it. A regression that flips it back to SESSION
    scope leaves the health dashboard dark — Postgres never sees the
    event because the InteractionLog handler is scoped colony-side.

    Captures the ``scope_id`` value the callback passes to
    ``get_blackboard``."""

    captured: list[str | None] = []

    class _ScopeRecordingAgent:
        agent_id = "agent-scope-pin"
        agent_type = "SessionAgent"

        def get_capability_by_type(self, t: type) -> Any:
            return None  # skip chat write — focus on the diagnostic scope

        async def get_blackboard(
            self, scope_id: str | None = None, **_: Any,
        ) -> EnhancedBlackboard:
            captured.append(scope_id)
            # Return a real bb so the write succeeds and the callback
            # exits cleanly.
            return await _make_chat_bb("scope_pin")

    await session_agent_stopped(
        _ScopeRecordingAgent(),  # type: ignore[arg-type]
        "max_iterations",
    )

    # Exactly one diagnostic get_blackboard call; its scope_id is the
    # canonical colony scope. The exact string is what ``get_scope_prefix
    # (BlackboardScope.COLONY)`` produces for the ambient exec context.
    from polymathera.colony.agents.scopes import (
        BlackboardScope, get_scope_prefix,
    )
    expected = get_scope_prefix(BlackboardScope.COLONY)
    assert captured == [expected], captured
