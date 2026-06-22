"""Lifecycle stop-callbacks wired into ``SessionAgent.bind(...).stop_callbacks``.

The dashboard chat router has NO dead-agent detection — when a
SessionAgent terminates for any reason, the chat WebSocket's listener
tasks remain bound to a backend stream that nobody feeds anymore.
The user types, nothing reads, the chat appears hung. This is the
"user-visible silent death" pattern (R12-audit / B4) — the symptom
of every termination path that doesn't currently surface a
user-facing signal (max_code_iterations cap, policy_completed,
exception → FAILED, etc.).

The callbacks defined here run from ``Agent.stop()`` AFTER the agent
has flipped to ``STOPPED``. They:

1. Write a typed ``chat:agent:{agent_id}:msg_*`` system_failure
   message into the session chat blackboard. The existing
   ``_listen_for_agent_messages`` in ``routers/chat.py`` already
   relays ``chat:agent:*`` writes to every connected WebSocket
   without any further wiring — the user immediately sees the
   message in chat.

2. Emit an ``AgentDiagnosticProtocol`` event of kind
   ``session_agent_stopped`` on the session's diagnostic scope.
   Operator-facing tools (dashboard health view, log adapters,
   future auto-respawn loops) can subscribe to this event to react
   without polling agent state.

Why a stop_callback and not a hook on ``Agent.stop`` itself: the
callback list is the explicit, per-spawn surface for cross-cutting
lifecycle concerns (see ``Agent.stop_callbacks`` docstring at
``base.py:1980``). Mission coordinators register their own ledger /
billing callbacks here; this is the same shape for chat-surface
death-notification. The framework stays free of any specific
"surface this in chat" concern.

Both the user-session and system-session SessionAgents wire this
callback at their blueprint construction sites:
- :func:`routers/sessions.py` (user chat session)
- :func:`web_ui/backend/chat/system_session.py` (colony system session)

PR1-B will add auto-respawn as a SECOND callback that runs after
this one — when it lands, the chat message will read "restarting…"
instead of "no longer being orchestrated."
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from polymathera.colony.agents.blackboard.protocol import (
    AgentDiagnosticProtocol,
)
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix

from .chat_protocol import SessionChatProtocol

if TYPE_CHECKING:
    from polymathera.colony.agents.base import Agent


logger = logging.getLogger(__name__)


#: Diagnostic ``kind`` for an unexpected SessionAgent termination.
#: Routed via :class:`AgentDiagnosticProtocol` on the session's
#: ``agent_diagnostic`` scope. Operator-facing tools subscribe via
#: ``@event_handler(pattern=AgentDiagnosticProtocol.event_pattern())``
#: and filter on ``parse_event_key(key)["kind"]``.
DIAGNOSTIC_SESSION_AGENT_STOPPED = "session_agent_stopped"


# Stop reasons the framework treats as EXPECTED — operator-initiated
# shutdown, graceful cancellation, normal session-close. Surfacing a
# system_failure for these would be noise. Every other reason
# (max_iterations, max_code_iterations, policy_completed, error,
# unknown, ...) is treated as unexpected and surfaced to the user.
# The set is deliberately small: new failure modes default to
# user-visible until proven benign.
_EXPECTED_STOP_REASONS: frozenset[str] = frozenset({
    "stop_requested",
    "cancelled",
})


def _is_unexpected_stop(reason: str) -> bool:
    return reason not in _EXPECTED_STOP_REASONS


def _format_user_facing_message(*, reason: str) -> str:
    """One short, human-readable line the chat UI shows. Keep it
    plain-language: the user is not the operator and the reason
    string is internal jargon (``max_code_iterations``,
    ``policy_completed``). The body translates rather than echoes.

    The reason string IS rendered verbatim in a code-span so an
    operator who's watching can still grep for it; the prose around
    it explains what it means to the user."""

    plain = {
        "max_iterations": (
            "I exhausted my planning iteration budget on this turn."
        ),
        "max_code_iterations": (
            "I exhausted my code-generation iteration budget on this turn."
        ),
        "policy_completed": (
            "My action policy reported completion (likely a stuck "
            "iteration cap inside the policy)."
        ),
        "idle_timeout": (
            "I idled past my session timeout."
        ),
        "error": (
            "An internal error stopped my planning loop."
        ),
        "unknown": (
            "I stopped for an unknown reason."
        ),
    }.get(reason, f"I stopped (reason: `{reason}`).")
    return (
        f"⚠️ **Session agent terminated.** {plain}\n\n"
        f"This session is no longer being orchestrated. Until "
        f"auto-respawn lands (PR1-B), reload the page or start a new "
        f"session to continue. Any messages you send now will not be "
        f"processed.\n\n"
        f"*Internal stop reason:* `{reason}`"
    )


async def session_agent_stopped(agent: "Agent", reason: str) -> None:
    """Stop-callback wired into ``SessionAgent.bind(...).stop_callbacks``.

    Writes a typed ``chat:agent:*`` system_failure message AND emits
    an ``AgentDiagnosticProtocol`` event of kind
    :data:`DIAGNOSTIC_SESSION_AGENT_STOPPED` on the session's
    diagnostic scope.

    Fully defensive: ``Agent.stop`` already catches and logs
    callback exceptions, but a permanent silent chat is strictly
    worse than a degraded death-message, so this layer also
    catches + logs and returns. Blackboard failures cannot be
    allowed to bubble.

    Skips emission entirely for reasons in
    :data:`_EXPECTED_STOP_REASONS` (operator-initiated shutdown,
    graceful cancellation) — surfacing those as a failure would be
    chat noise.
    """

    if not _is_unexpected_stop(reason):
        return

    # Resolve the SessionOrchestratorCapability — it owns the
    # chat-blackboard scope (``BlackboardScope.SESSION`` +
    # ``namespace=session_chat``). Using the capability avoids
    # re-deriving the scope_id by hand at the callback site (which
    # would couple this module to scope-construction internals).
    #
    # A SessionAgent without SessionOrchestratorCapability has no
    # chat surface to write to — skip silently rather than fail
    # loudly. The only way this branch fires is on a misconfigured
    # blueprint; the operator-facing diagnostic emit below still
    # runs (different scope, no capability dependency).
    from .session_agent import SessionOrchestratorCapability

    try:
        cap = agent.get_capability_by_type(SessionOrchestratorCapability)
    except Exception as e:
        logger.error(
            "session_agent_stopped: failed to resolve "
            "SessionOrchestratorCapability for agent %s: %s",
            agent.agent_id, e,
        )
        cap = None

    if cap is not None:
        try:
            chat_bb = await cap.get_blackboard()
            mid = f"msg_{uuid.uuid4().hex[:12]}"
            await chat_bb.write(
                SessionChatProtocol.agent_message_key(agent.agent_id, mid),
                {
                    "content": _format_user_facing_message(reason=reason),
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "message_id": mid,
                    "timestamp": time.time(),
                    "kind": "system_failure",
                    "stop_reason": reason,
                },
            )
            logger.warning(
                "SessionAgent %s stopped (reason=%s); wrote "
                "user-visible system_failure chat message %s",
                agent.agent_id, reason, mid,
            )
        except Exception as e:
            logger.error(
                "session_agent_stopped: chat-message write failed "
                "for agent %s (reason=%s): %s",
                agent.agent_id, reason, e,
            )

    # Operator-facing diagnostic emit. Scope is the session's
    # ``agent_diagnostic`` namespace (the same scope handler-block
    # mirror SessionAgent subscribes to for cross-agent diagnostic
    # events; see session_agent.py:683-687). The key shape is
    # ``agent:diagnostic:<agent_id>:<kind>:<sequence>``; sequence
    # uses a monotonic-enough nanosecond timestamp because there is
    # no per-(agent_id, kind) counter accessible from a free
    # function.
    # Colony scope (not session scope) so the InteractionLog
    # capability mounted on the colony's system session mirrors
    # this event into Postgres for the health-monitoring dashboard
    # to surface across sessions. session_agent_stopped emitted on
    # session scope alone would be invisible to the cross-session
    # health panel (and the session itself is dead by then anyway).
    try:
        diag_scope_id = get_scope_prefix(BlackboardScope.COLONY)
        diag_bb = await agent.get_blackboard(scope_id=diag_scope_id)
        seq = time.time_ns()
        diag_key = AgentDiagnosticProtocol.event_key(
            agent_id=agent.agent_id,
            kind=DIAGNOSTIC_SESSION_AGENT_STOPPED,
            sequence=seq,
        )
        await diag_bb.write(
            diag_key,
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "kind": DIAGNOSTIC_SESSION_AGENT_STOPPED,
                "stop_reason": reason,
                "timestamp": time.time(),
            },
        )
    except Exception as e:
        logger.error(
            "session_agent_stopped: diagnostic emit failed for "
            "agent %s (reason=%s): %s",
            agent.agent_id, reason, e,
        )


__all__ = (
    "DIAGNOSTIC_SESSION_AGENT_STOPPED",
    "session_agent_stopped",
)
