"""WebSocket chat endpoint for real-time agent communication.

Bridges the browser WebSocket to Colony's agent system:
- User sends a message → AgentHandle.run_streamed() on selected agent
- Agent responses streamed back via WebSocket events
- Supports agent selection and concurrent conversations
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/chat/{session_id}")
async def agent_chat(
    websocket: WebSocket,
    session_id: str,
):
    """WebSocket for real-time agent communication.

    Protocol:
        Client sends JSON messages:
        {
            "type": "message",
            "agent_id": "agent_123",
            "content": "Analyze the auth module",
            "namespace": "analysis"  // optional
        }
        {
            "type": "list_agents"
        }

        Server sends JSON events:
        {
            "type": "agent_event",
            "agent_id": "agent_123",
            "event_type": "progress" | "completed" | "error",
            "data": {...}
        }
        {
            "type": "agents_list",
            "agents": [{"agent_id": "...", "agent_type": "...", "state": "..."}]
        }
        {
            "type": "error",
            "message": "..."
        }
    """
    await websocket.accept()

    # Get colony connection from app state
    colony: ColonyConnection = websocket.app.state.colony

    if not colony.is_connected:
        await websocket.send_json({"type": "error", "message": "Not connected to cluster"})
        await websocket.close()
        return

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

            if msg_type == "list_agents":
                await _handle_list_agents(websocket, colony)

            elif msg_type == "message":
                agent_id = msg.get("agent_id")
                content = msg.get("content", "")
                namespace = msg.get("namespace", "")

                if not agent_id:
                    await websocket.send_json({"type": "error", "message": "agent_id required"})
                    continue

                # Cancel any previous streaming task for this agent
                prev = active_tasks.pop(agent_id, None)
                if prev and not prev.done():
                    prev.cancel()

                # Start streaming in background
                task = asyncio.create_task(
                    _stream_agent_response(
                        websocket, colony, session_id, agent_id, content, namespace,
                    )
                )
                active_tasks[agent_id] = task

            elif msg_type == "cancel":
                agent_id = msg.get("agent_id")
                if agent_id:
                    task = active_tasks.pop(agent_id, None)
                    if task and not task.done():
                        task.cancel()
                    await websocket.send_json({
                        "type": "cancelled",
                        "agent_id": agent_id,
                    })

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
        # Cancel all active streams
        for task in active_tasks.values():
            if not task.done():
                task.cancel()


async def _handle_list_agents(
    websocket: WebSocket,
    colony: ColonyConnection,
) -> None:
    """List active agents and send to client."""
    with colony.kernel_execution_context(origin="dashboard"):
        try:
            handle = colony.get_agent_system()
            agent_ids: list[str] = await handle.list_all_agents()

            agents = []
            for agent_id in agent_ids[:50]:  # Limit to 50
                try:
                    info = await handle.get_agent_info(agent_id=agent_id)
                    agents.append({
                        "agent_id": agent_id,
                        "agent_type": getattr(info, "agent_type", "") if info else "",
                        "state": str(getattr(info, "state", "")) if info else "unknown",
                    })
                except Exception:
                    agents.append({"agent_id": agent_id, "agent_type": "", "state": "unknown"})

            await websocket.send_json({"type": "agents_list", "agents": agents})
        except Exception as e:
            await websocket.send_json({"type": "error", "message": f"Failed to list agents: {e}"})


async def _stream_agent_response(
    websocket: WebSocket,
    colony: ColonyConnection,
    session_id: str,
    agent_id: str,
    content: str,
    namespace: str,
) -> None:
    """Stream agent response events to the WebSocket client.

    Uses AgentHandle.run_streamed() to send input and receive events.
    """
    with colony.kernel_execution_context(origin="dashboard"):
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
                namespace=namespace,
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
            logger.debug("Chat stream cancelled for agent %s", agent_id)
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
