"""WebSocket chat endpoint for session-based communication.

Messages go to the session's SessionAgent via the blackboard. The session
agent decides how to handle them (respond directly, spawn coordinators,
route to specific agents). Agent responses flow back via the same blackboard
and are relayed to the WebSocket client.

If a session has no session agent (spawn failed or legacy session), falls
back to listing agents and direct agent routing (Phase 1 behavior).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import time
from typing import Any
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

from polymathera.colony.agents.blackboard import EnhancedBlackboard


logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/chat/{session_id}")
async def session_chat(
    websocket: WebSocket,
    session_id: str,
):
    """WebSocket for session chat.

    Client sends:
        { "type": "message", "content": "...", "controls": {...} }
        { "type": "reply", "content": "...", "request_id": "...", "agent_id": "..." }
        { "type": "cancel_run", "run_id": "..." }
        { "type": "list_agents" }
        { "type": "history", "before": "...", "limit": 50 }

    Server sends:
        { "type": "message", "message": {...} }
        { "type": "agent_question", "message": {...} }
        { "type": "run_event", "run_id": "...", "event_type": "...", "data": {...} }
        { "type": "tab_activity", "tab_id": "...", "count": 1 }
        { "type": "agents_list", "agents": [...] }
        { "type": "error", "message": "..." }
    """
    await websocket.accept()

    # Authenticate via cookie (same JWT as HTTP endpoints)
    from ..auth.middleware import ACCESS_COOKIE
    from ..auth.service import decode_token
    from polymathera.colony.distributed.ray_utils.serving.context import Ring, execution_context

    token = websocket.cookies.get(ACCESS_COOKIE)
    user_payload = decode_token(token) if token else None
    if not user_payload or user_payload.get("type") != "access":
        await websocket.send_json({"type": "error", "message": "Authentication required"})
        await websocket.close()
        return

    tenant_id = user_payload.get("tenant_id", "")
    user_id = user_payload.get("user_id", "")

    # Get colony connection from app state
    colony: ColonyConnection = websocket.app.state.colony

    if not colony.is_connected:
        await websocket.send_json({"type": "error", "message": "Not connected to cluster"})
        await websocket.close()
        return

    # Chat message persistence
    chat_store = getattr(websocket.app.state, "chat_store", None)

    # Look up session metadata (agent ID, tenant, colony for execution context)
    session_info = await _get_session_info(colony, session_id)

    # Refuse to attach chat to a colony-singleton system session. The
    # system session hosts always-on capabilities (P8 GitHub inbound +
    # InteractionLog; P9+ webhook / mention routing) and has no user
    # behind it; chat attempts here would never receive a response.
    # Traces tab paths are independent and stay open. Pre-P8-0
    # sessions without the field default to ``user`` via the
    # _SessionInfo dataclass and pass through unaffected.
    if session_info is not None and session_info.session_kind == "system":
        await websocket.send_json({
            "type": "error",
            "message": "Cannot attach chat to a system session.",
        })
        await websocket.close()
        return

    session_agent_id = session_info.session_agent_id if session_info else None

    # colony_id: try WebSocket header first, fall back to session's colony_id.
    # WebSocket connections don't go through apiFetch so X-Colony-Id may be absent.
    colony_id = (
        websocket.headers.get("X-Colony-Id")
        or (session_info.colony_id if session_info else "")
    )

    # Set execution context for the entire WebSocket connection.
    # All blackboard operations within this connection use the
    # authenticated user's tenant/colony/session context.
    with execution_context(
        ring=Ring.USER,
        tenant_id=tenant_id,
        colony_id=colony_id,
        session_id=session_id,
        origin="dashboard_chat",
    ):

        # Backfill chat history from the session blackboard before
        # loading from chat_store. Persistence to chat_store happens
        # inside ``_listen_for_agent_messages`` — i.e., only AFTER a
        # WebSocket has connected and started listening. Any agent
        # message written *before* the first WebSocket connection
        # (e.g., the very first ``run_step`` of the session agent
        # almost always emits a welcome ``respond_to_user`` while the
        # browser is still negotiating the socket) lives only on the
        # blackboard. Without backfill, the user would have to type
        # something first to trigger any subsequent agent activity
        # before the welcome appeared — exactly the symptom reported.
        #
        # The blackboard query is ``ON CONFLICT DO NOTHING`` via
        # ``chat_store.save_message``, so backfill is idempotent and
        # safe to run on every connect (reconnects don't duplicate).
        if chat_store and session_agent_id:
            try:
                await _backfill_chat_history_from_blackboard(
                    colony, session_id, session_info, chat_store,
                )
            except Exception as e:
                logger.warning(
                    "Failed to backfill chat history from blackboard for session %s: %s",
                    session_id, e,
                )

        # Send chat history on connect
        if chat_store:
            try:
                history = await chat_store.get_history(session_id, limit=50)
                if history:
                    await websocket.send_json({"type": "history", "messages": history, "has_more": len(history) >= 50})
            except Exception as e:
                logger.warning("Failed to load chat history for session %s: %s", session_id, e)

        # Background task for streaming blackboard events from the session agent
        # back to the WebSocket client.
        event_listener_task: asyncio.Task | None = None
        action_status_task: asyncio.Task | None = None
        mission_status_task: asyncio.Task | None = None
        if session_agent_id:
            event_listener_task = asyncio.create_task(
                _listen_for_agent_messages(websocket, colony, session_id, session_info, chat_store)
            )
            action_status_task = asyncio.create_task(
                _listen_for_action_status(websocket, colony, session_id, session_info)
            )
            mission_status_task = asyncio.create_task(
                _listen_for_mission_status(websocket, colony, session_id, session_info)
            )

        # Track active streaming tasks so we can cancel on disconnect
        active_tasks: dict[str, asyncio.Task] = {}

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "message":
                    content = msg.get("content", "")
                    controls = msg.get("controls")

                    # Persist user message
                    if chat_store and content:
                        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                        await chat_store.save_message({
                            "id": msg_id,
                            "session_id": session_id,
                            "role": "user",
                            "user_id": user_id,
                            "content": content,
                            "timestamp": time.time(),
                            "controls": controls,
                        })

                    if session_agent_id and session_info:
                        # Route through session agent via blackboard
                        await _post_user_message(
                            colony, session_id, session_info, content, controls,
                        )
                    else:
                        # Fallback: direct agent routing (no session agent)
                        agent_id = msg.get("agent_id")
                        if agent_id:
                            task = asyncio.create_task(
                                _stream_direct_agent_response(
                                    websocket, colony, session_id, agent_id, content,
                                )
                            )
                            active_tasks[agent_id] = task
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": "No session agent available. Provide agent_id for direct routing.",
                            })

                elif msg_type == "reply":
                    content = msg.get("content", "")
                    request_id = msg.get("request_id", "")
                    agent_id = msg.get("agent_id", "")

                    if session_agent_id and session_info and request_id:
                        await _post_user_reply(
                            colony, session_id, session_info, content, request_id, agent_id,
                        )
                    else:
                        await websocket.send_json({"type": "error", "message": "request_id required for reply"})

                elif msg_type == "cancel_run":
                    run_id = msg.get("run_id")
                    # Cancel active streaming task if any
                    for task_id, task in list(active_tasks.items()):
                        if not task.done():
                            task.cancel()
                    active_tasks.clear()
                    await websocket.send_json({
                        "type": "cancelled",
                        "run_id": run_id or "",
                    })

                elif msg_type == "list_agents":
                    await _handle_list_agents(websocket, colony)

                elif msg_type == "history":
                    if chat_store:
                        before = msg.get("before")
                        limit = min(msg.get("limit", 50), 100)
                        before_ts = float(before) if before else None
                        history = await chat_store.get_history(session_id, limit=limit, before_timestamp=before_ts)
                        await websocket.send_json({"type": "history", "messages": history, "has_more": len(history) >= limit})
                    else:
                        await websocket.send_json({"type": "history", "messages": [], "has_more": False})

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

        except WebSocketDisconnect:
            logger.info("Chat WebSocket disconnected for session %s", session_id)
        except Exception as e:
            logger.error("Chat WebSocket error for session %s: %s", session_id, e)
        finally:
            if event_listener_task and not event_listener_task.done():
                event_listener_task.cancel()
            if action_status_task and not action_status_task.done():
                action_status_task.cancel()
            if mission_status_task and not mission_status_task.done():
                mission_status_task.cancel()
            # Cancel all active streams
            for task in active_tasks.values():
                if not task.done():
                    task.cancel()


# ---------------------------------------------------------------------------
# Session agent communication via blackboard
# ---------------------------------------------------------------------------

@dataclass
class _SessionInfo:
    """Cached session metadata for the WebSocket connection."""
    session_agent_id: str | None
    tenant_id: str
    colony_id: str
    # ``user`` (chat-bound human) or ``system`` (colony singleton —
    # chat-attach is refused). Defaults to ``user`` so any pre-P8-0
    # serialized session that lacks the field passes the guard.
    session_kind: str = "user"


async def _get_session_info(colony: ColonyConnection, session_id: str) -> _SessionInfo | None:
    """Look up session metadata needed for the chat WebSocket."""
    try:
        with colony.kernel_execution_context(origin="dashboard_chat"):
            sm = await colony.get_session_manager()
            session = await sm.get_session(session_id=session_id)
            if session is None:
                return None

            from polymathera.colony.agents.sessions.models import Session as SessionModel
            if isinstance(session, dict):
                session = SessionModel(**session)

            return _SessionInfo(
                session_agent_id=session.session_agent_id or None,
                tenant_id=session.tenant_id,
                colony_id=session.colony_id,
                session_kind=session.session_kind,
            )
    except Exception as e:
        logger.warning("Failed to look up session info for %s: %s", session_id, e)
        return None


async def _get_session_chat_blackboard(colony: ColonyConnection, session_info: _SessionInfo) -> EnhancedBlackboard | None:
    """Get the session-scoped blackboard for the session agent.

    Resolves the chat namespace via
    ``SessionOrchestratorCapability.DEFAULT_NAMESPACE`` so the literal
    lives in exactly one place — the capability class — and any future
    rename happens in one file rather than three.
    """
    try:
        from ..chat import SessionOrchestratorCapability
        from polymathera.colony.agents import AgentHandle
        from polymathera.colony.agents.scopes import BlackboardScope

        handle = await AgentHandle.from_agent_id(session_info.session_agent_id, app_name=colony.app_name)
        cap = handle.get_capability(
            SessionOrchestratorCapability,
            scope=BlackboardScope.SESSION,
            namespace=SessionOrchestratorCapability.DEFAULT_NAMESPACE,
        )
        bb: EnhancedBlackboard = await cap.get_blackboard()
        logger.info("Got session chat blackboard: scope_id=%s, backend_type=%s", bb.scope_id, bb.backend_type)
        return bb
    except Exception as e:
        logger.error("Failed to get session chat blackboard: %s", e)
        return None


async def _post_user_message(
    colony: ColonyConnection,
    session_id: str,
    session_info: _SessionInfo,
    content: str,
    controls: dict | None,
) -> None:
    """Post a user message to the session agent via the session-scoped blackboard.

    Slash commands listed in ``_HIGH_PRIORITY_COMMANDS`` are routed
    to ``chat:control:*`` (high-priority lane on the agent's policy)
    so they bypass any in-flight long-running action. Plain messages
    and unrecognised commands stay on ``chat:user:*``.

    Caller must have set the execution context (session_chat does this).
    """
    try:
        from ..chat import SessionChatProtocol

        bb = await _get_session_chat_blackboard(colony, session_info)
        if bb is None:
            logger.error("Failed to get session chat blackboard — bb is None")
            return

        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        is_control = _is_control_command(content)
        if is_control:
            key = SessionChatProtocol.control_message_key(message_id)
        else:
            key = SessionChatProtocol.user_message_key(message_id)

        logger.info(
            "Writing %s message to blackboard: scope_id=%s, key=%s, backend_type=%s",
            "CONTROL" if is_control else "user",
            bb.scope_id, key, bb.backend_type,
        )

        await bb.write(key, {
            "content": content,
            "command": _extract_command(content) if is_control else None,
            "message_id": message_id,
            "controls": controls,
            "timestamp": time.time(),
        })
        logger.info("User message %s written to blackboard key=%s", message_id, key)
    except Exception as e:
        logger.error("Failed to post user message to session agent: %s", e, exc_info=True)


# Slash commands routed to the high-priority lane. Anything not in
# this set goes through the normal lane (so /help, /agents etc. still
# do their pre-existing rule-based handling on the main loop).
#
# - ``/status`` / ``/whatdoing``: read-only inspection, must not block.
# - ``/abort`` / ``/cancel``: cancel the in-flight action so the user
#   regains control without waiting for the action to complete.
# - ``/replace <new request>``: pre-emptive re-prioritisation —
#   abort the current action AND queue the new request as the next
#   user message so the planner picks it up immediately.
_HIGH_PRIORITY_COMMANDS: frozenset[str] = frozenset({
    "/status",
    "/whatdoing",
    "/abort",
    "/cancel",
    "/replace",
})


def _extract_command(content: str) -> str | None:
    """Return the leading slash command (lowercased) or ``None``.

    A bare slash (``"/"`` or ``" / "``) is NOT a command — the user
    has to type at least one character after the slash.
    """
    stripped = content.strip()
    if not stripped.startswith("/") or len(stripped) < 2:
        return None
    head = stripped.split(None, 1)[0]
    if head == "/":
        return None
    return head.lower()


def _is_control_command(content: str) -> bool:
    cmd = _extract_command(content)
    return cmd is not None and cmd in _HIGH_PRIORITY_COMMANDS


async def _post_user_reply(
    colony: ColonyConnection,
    session_id: str,
    session_info: _SessionInfo,
    content: str,
    request_id: str,
    agent_id: str,
) -> None:
    """Post a user reply to an agent question via the session-scoped blackboard.

    Caller must have set the execution context (session_chat does this).
    """
    try:
        from ..chat import SessionChatProtocol

        bb = await _get_session_chat_blackboard(colony, session_info)

        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        key = SessionChatProtocol.reply_key(request_id, message_id)

        await bb.write(key, {
            "content": content,
            "message_id": message_id,
            "request_id": request_id,
            "agent_id": agent_id,
            "user_id": "",  # TODO: pass from auth context
            "timestamp": time.time(),
        })
    except Exception as e:
        logger.error("Failed to post user reply: %s", e)


async def _backfill_chat_history_from_blackboard(
    colony: ColonyConnection,
    session_id: str,
    session_info: _SessionInfo,
    chat_store: Any,
) -> int:
    """Persist any agent/user chat messages on the session blackboard
    that are not yet in ``chat_store`` (which is only written to by the
    live WS-bound listener — so anything written before the first WS
    connect would otherwise be invisible to the UI).

    ``chat_store.save_message`` uses ``ON CONFLICT (id) DO NOTHING``,
    so calling this on every connect is safe — duplicates from prior
    backfills are dropped at the DB level.

    Returns the number of records inspected (for logging).
    """
    from ..chat import SessionChatProtocol

    bb = await _get_session_chat_blackboard(colony, session_info)
    if bb is None:
        return 0

    # Both agent messages and user messages live on the chat
    # blackboard. We backfill both so the history shown on a fresh
    # connect matches what would have been streamed live had the
    # WebSocket been connected when the messages were written. The
    # ``query`` is a single backend round-trip per pattern.
    count = 0
    for pattern, role in (
        (SessionChatProtocol.agent_message_pattern(), "agent"),
        (SessionChatProtocol.user_message_pattern(), "user"),
    ):
        entries = await bb.query(namespace=pattern, limit=500)
        for entry in entries:
            payload = entry.value if isinstance(entry.value, dict) else {}
            chat_msg = {
                "id": payload.get("message_id") or entry.key,
                "session_id": session_id,
                "run_id": payload.get("run_id"),
                "role": role,
                "agent_id": (
                    payload.get("agent_id")
                    if role == "agent"
                    else None
                ) or (session_info.session_agent_id if role == "agent" else None),
                "agent_type": payload.get("agent_type", "") if role == "agent" else None,
                "user_id": payload.get("user_id") if role == "user" else None,
                "username": payload.get("username") if role == "user" else None,
                "content": payload.get("content", ""),
                "timestamp": payload.get("timestamp", entry.created_at),
                "request_id": payload.get("request_id"),
                "response_options": payload.get("response_options"),
                "awaiting_reply": payload.get("awaiting_reply", False),
                "run_status": payload.get("run_status"),
                "controls": payload.get("controls"),
                "kind": payload.get("kind"),
                "action_type": payload.get("action_type"),
            }
            try:
                await chat_store.save_message(chat_msg)
                count += 1
            except Exception as e:
                logger.debug(
                    "Backfill: failed to persist message %s: %s",
                    chat_msg["id"], e,
                )
    if count:
        logger.info(
            "Backfilled %d chat messages from blackboard for session %s",
            count, session_id,
        )
    return count


async def _listen_for_agent_messages(
    websocket: WebSocket,
    colony: ColonyConnection,
    session_id: str,
    session_info: _SessionInfo,
    chat_store: Any | None = None,
) -> None:
    """Background task: subscribe to agent messages on the blackboard and relay to WebSocket.

    Listens for agent messages posted via SessionChatProtocol and forwards
    them to the client. Also persists agent messages to PostgreSQL.

    Inherits execution context from the caller (asyncio.create_task copies contextvars).
    """
    try:
        from ..chat import SessionChatProtocol

        bb = await _get_session_chat_blackboard(colony, session_info)

        async for event in bb.stream_events(
            pattern=SessionChatProtocol.agent_message_pattern(),
            timeout=None,
        ):
            payload = event.value if isinstance(event.value, dict) else {}

            chat_msg = {
                "id": payload.get("message_id", event.key),
                "session_id": session_id,
                "run_id": payload.get("run_id"),
                "role": "agent",
                "agent_id": payload.get("agent_id", session_info.session_agent_id),
                "agent_type": payload.get("agent_type", ""),
                "content": payload.get("content", ""),
                "timestamp": payload.get("timestamp", time.time()),
                "request_id": payload.get("request_id"),
                "response_options": payload.get("response_options"),
                "awaiting_reply": payload.get("awaiting_reply", False),
                "run_status": payload.get("run_status"),
                # ``kind`` distinguishes a typed human-approval gate
                # (rendered with option buttons that POST to the
                # approval HTTP endpoint) from a freeform
                # ``agent_question`` (replies routed via the chat
                # WebSocket). Absent for legacy messages.
                "kind": payload.get("kind"),
                # Short action name for typed approvals; drives the
                # 4-choice button labels on the frontend (Approve once
                # / Approve all / Reject / Abort). Absent for legacy
                # untyped approve/reject requests.
                "action_type": payload.get("action_type"),
                # Structured attachments emitted by ``respond_to_user``
                # / ``respond_to_user_with_table`` / ``respond_to_user_with_diff``. The
                # chat UI dispatches each attachment to a typed
                # renderer (code block, table, diff). Absent for
                # plain-text messages.
                "attachments": payload.get("attachments"),
                # Free-form bag the requesting agent stamps on
                # typed-question payloads. For ``kind="human_approval"``
                # callers pass the proposal diff / summary / affected-
                # pages list here; for ``kind="human_help"`` the
                # SessionAgent's ``handle_human_help_request`` translator
                # stamps the agent's ``context`` (what it has tried +
                # observed) so the operator card surfaces it above the
                # response surface. Absent for legacy messages.
                "extra": payload.get("extra"),
            }

            # Persist agent message
            if chat_store:
                try:
                    await chat_store.save_message(chat_msg)
                except Exception as e:
                    logger.warning("Failed to persist agent message: %s", e)

            msg_type = "agent_question" if payload.get("awaiting_reply") else "message"
            await websocket.send_json({"type": msg_type, "message": chat_msg})

    except asyncio.CancelledError:
        logger.debug("Agent message listener cancelled for session %s", session_id)
    except Exception as e:
        logger.error("Agent message listener error for session %s: %s", session_id, e)


async def _listen_for_action_status(
    websocket: WebSocket,
    colony: ColonyConnection,
    session_id: str,
    session_info: _SessionInfo,
) -> None:
    """Background task: relay action-status records to the WebSocket.

    The session agent's ``CodeGenerationActionPolicy`` publishes a
    ``running`` record before every action and a ``complete``/``failed``
    record after. The frontend tracks them in a Map keyed by
    ``action_id`` and renders a small badge while any are running.

    These records are NOT persisted to the chat history — they are
    transient UI cues, not real chat messages.
    """
    try:
        from ..chat import SessionChatProtocol

        bb = await _get_session_chat_blackboard(colony, session_info)

        async for event in bb.stream_events(
            pattern=SessionChatProtocol.action_status_pattern(),
            timeout=None,
        ):
            payload = event.value if isinstance(event.value, dict) else {}
            await websocket.send_json({
                "type": "action_status",
                "agent_id": payload.get("agent_id"),
                "action_id": payload.get("action_id"),
                "action_key": payload.get("action_key"),
                "status": payload.get("status"),
                "started_at": payload.get("started_at"),
                "ended_at": payload.get("ended_at"),
                "wall_time_ms": payload.get("wall_time_ms"),
                "error": payload.get("error"),
            })
    except asyncio.CancelledError:
        logger.debug(
            "Action-status listener cancelled for session %s", session_id,
        )
    except Exception as e:
        logger.error(
            "Action-status listener error for session %s: %s", session_id, e,
        )


async def _listen_for_mission_status(
    websocket: WebSocket,
    colony: ColonyConnection,
    session_id: str,
    session_info: _SessionInfo,
) -> None:
    """Background task: relay mission-status records to the WebSocket.

    Sibling of :func:`_listen_for_action_status`. Coordinators publish
    ``chat:mission_status:{mission_id}`` records via
    ``MissionStatusCapability.emit_mission_status`` whenever they want
    the chat UI to surface a one-line narrative ("loading design
    context...", "classifying issues...") in place of the spinner.
    The frontend keys by ``mission_id`` (the coordinator's
    ``agent_id``) and replaces the prior status with the latest — a
    singleton, not a history.

    On reconnect, the relay snapshot-reads the current key BEFORE
    streaming so the client immediately sees the latest status
    without waiting for the next emit. This is the assumption-review
    correction baked in: pure pub/sub would leave the user staring at
    nothing until the coordinator's next narrative tick.

    Lifetime is framework-owned, not LLM-owned: the relay forwards a
    synthetic ``{"cleared": true}`` payload when the chat router
    observes the mission's terminal state (the coordinator's
    ``policy:action_completed`` for the final ``signal_completion`` /
    ``respond_to_user`` or a non-running run state). The planner is
    NOT expected to clear; the framework does, on these boundaries.
    """

    try:
        # ``MissionStatusProtocol`` lives in ``agents/blackboard`` so
        # producers (capabilities in ``agents/``) and this consumer in
        # ``web_ui/`` agree on the same canonical key/pattern owner.
        # The dependency direction is ``web_ui/`` → ``agents/`` —
        # downstream, never the reverse.
        from polymathera.colony.agents.blackboard.protocol import (
            MissionStatusProtocol,
        )

        bb = await _get_session_chat_blackboard(colony, session_info)

        # Snapshot-read the current mission_status singletons so a
        # reconnecting client doesn't wait for the next emit to see
        # state. ``read_keys_matching`` is the existing primitive
        # used by the chat history loader; we reuse it here for
        # symmetric reconnect semantics with action-status.
        try:
            existing = await bb.read_keys_matching(
                MissionStatusProtocol.status_pattern(),
            )
        except AttributeError:
            existing = []
        for key, payload in existing:
            if not isinstance(payload, dict):
                continue
            try:
                mission_id = MissionStatusProtocol.parse_status_key(key)
            except ValueError:
                continue
            await websocket.send_json({
                "type": "mission_status",
                "mission_id": mission_id,
                "agent_id": payload.get("agent_id"),
                "message": payload.get("message", ""),
                "details": payload.get("details") or {},
                "timestamp": payload.get("timestamp"),
            })

        async for event in bb.stream_events(
            pattern=MissionStatusProtocol.status_pattern(),
            timeout=None,
        ):
            payload = event.value if isinstance(event.value, dict) else {}
            try:
                mission_id = MissionStatusProtocol.parse_status_key(
                    event.key,
                )
            except ValueError:
                continue
            await websocket.send_json({
                "type": "mission_status",
                "mission_id": mission_id,
                "agent_id": payload.get("agent_id"),
                "message": payload.get("message", ""),
                "details": payload.get("details") or {},
                "timestamp": payload.get("timestamp"),
            })
    except asyncio.CancelledError:
        logger.debug(
            "Mission-status listener cancelled for session %s", session_id,
        )
    except Exception as e:
        logger.error(
            "Mission-status listener error for session %s: %s", session_id, e,
        )


# ---------------------------------------------------------------------------
# Fallback: direct agent routing (no session agent)
# ---------------------------------------------------------------------------

async def _handle_list_agents(websocket: WebSocket, colony: ColonyConnection) -> None:
    """List active agents and send to client."""
    with colony.kernel_execution_context(origin="dashboard_chat"):
        try:
            handle = await colony.get_agent_system()
            agent_ids: list[str] = await handle.list_all_agents()

            agents = []
            for agent_id in agent_ids[:50]:  # Limit to 50
                try:
                    info = await handle.get_agent_info(agent_id=agent_id)
                except Exception:
                    agents.append({"agent_id": agent_id, "agent_type": "", "state": "unknown"})
                    continue
                if info is None:
                    agents.append({"agent_id": agent_id, "agent_type": "", "state": "unregistered"})
                    continue
                # ``AgentRegistrationInfo`` fields are typed + required
                # (see models.py:2819). Read directly per
                # [[no-getattr-defaults]] — ``info.state.name`` is the
                # uppercase enum name (e.g. ``"RUNNING"`` /
                # ``"STOPPED"``), no ``str(...)`` coercion needed.
                agents.append({
                    "agent_id": agent_id,
                    "agent_type": info.agent_type,
                    "state": info.state.name,
                })

            await websocket.send_json({"type": "agents_list", "agents": agents})
        except Exception as e:
            await websocket.send_json({"type": "error", "message": f"Failed to list agents: {e}"})


async def _stream_direct_agent_response(
    websocket: WebSocket,
    colony: ColonyConnection,
    session_id: str,
    agent_id: str,
    content: str,
) -> None:
    """Fallback: stream agent response directly when no session agent exists."""
    with colony.kernel_execution_context(origin="dashboard_chat"):
        try:
            from polymathera.colony.agents import AgentHandle

            handle = await AgentHandle.from_agent_id(
                agent_id, app_name=colony.app_name,
            )

            run_id = f"chat_{uuid.uuid4().hex[:8]}"

            async for event in handle.run_streamed(
                input_data={"query": content, "source": "dashboard_chat"},
                timeout=120.0,
                session_id=session_id,
                run_id=run_id,
            ):
                await websocket.send_json({
                    "type": "agent_event",
                    "agent_id": agent_id,
                    "event_type": event.event_type,
                    "data": event.data if isinstance(event.data, dict) else {"raw": str(event.data)},
                    "timestamp": getattr(event, "timestamp", None),
                })

                if event.event_type in ("completed", "error", "timeout"):
                    break

        except asyncio.CancelledError:
            logger.debug("Direct chat stream cancelled for agent %s", agent_id)
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "agent_event",
                    "agent_id": agent_id,
                    "event_type": "error",
                    "data": {"error": str(e)},
                })
            except Exception:
                pass  # WebSocket may already be closed
