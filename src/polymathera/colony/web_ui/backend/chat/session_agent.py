"""Session agent — per-session orchestrator for user interactions.

Each session gets a SessionAgent spawned via AgentHandle.from_blueprint().
It receives user messages via the SessionChatProtocol on the blackboard,
decides how to handle them (respond directly, spawn coordinators, route to
specific agents), and relays agent progress back to the user.
"""

from __future__ import annotations

import logging
import time
import uuid
import re
from typing import Any

from overrides import override

from polymathera.colony.agents.base import Agent, AgentCapability, AgentHandle
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.agents.models import AgentMetadata, AgentSuspensionState, RunContext, PolicyREPL
from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.patterns.events import event_handler, EventProcessingResult, PROCESSED
from polymathera.colony.agents.patterns.actions import action_executor
from .chat_protocol import SessionChatProtocol

logger = logging.getLogger(__name__)


class SessionOrchestratorCapability(AgentCapability):
    """Core capability for the session agent.

    Handles incoming user messages posted to the blackboard via
    SessionChatProtocol and orchestrates responses.

    Supports:
    - /commands (analyze, map, abort, status, set, help, agents, context)
    - @agent routing (forward to specific agent)
    - Plain messages (session agent handles directly)
    - User replies to agent questions (route back to requesting agent)
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "session_chat",
        input_patterns: list[str] | None = None,
        capability_key: str = "session_orchestrator",
    ):
        """Initialize session orchestrator capability.

        Args:
            agent: Owning agent (the SessionAgent). None for detached mode.
            scope: Blackboard scope (SESSION — shared partition for the session's chat)
            namespace: Namespace within the scope
            input_patterns: Event patterns to subscribe to. If None, auto-inferred
                from @event_handler decorators.
            capability_key: Unique key for the capability
        """
        scope_id = get_scope_prefix(scope, agent, namespace=namespace) if agent is not None else None
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    def get_action_group_description(self) -> str:
        return (
            "Session Orchestrator — handles user chat interactions. "
            "respond_to_user sends a text response back to the user in the chat. "
            "Use this when the user asks a question you can answer directly, "
            "or to acknowledge a request before spawning agents."
        )

    @action_executor()
    async def respond_to_user(self, content: str, **extra: Any) -> dict[str, Any]:
        """Send a text response to the user in the chat.

        Use this action to respond to user questions, provide status updates,
        or acknowledge requests. For analysis tasks, use AgentPoolCapability's
        create_agent instead.

        Args:
            content: The message text to send to the user (supports markdown).
            **extra: Additional fields to include in the message payload
                (e.g., request_id, response_options, awaiting_reply).

        Returns:
            Dict with message_id of the posted message.
        """
        bb = await self.get_blackboard()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        key = SessionChatProtocol.agent_message_key(self.agent.agent_id, message_id)
        await bb.write(key, {
            "content": content,
            "agent_id": self.agent.agent_id,
            "agent_type": self.agent.agent_type,
            "message_id": message_id,
            "timestamp": time.time(),
            **extra,
        })
        return {"message_id": message_id}

    async def _post_response(self, content: str, **extra: Any) -> None:
        """Internal helper — posts a response without going through action dispatch.

        Used by command handlers and @mention routing which are rule-based
        (don't need LLM planning). For LLM-planned responses, the planner
        calls respond_to_user directly.
        """
        await self.respond_to_user(content, **extra)

    @event_handler(pattern=SessionChatProtocol.user_message_pattern())
    async def handle_user_message(self, event: BlackboardEvent, repl: PolicyREPL) -> EventProcessingResult:
        """Handle an incoming user message from the chat."""
        payload = event.value if isinstance(event.value, dict) else {}
        content = payload.get("content", "").strip()
        controls = payload.get("controls")

        if not content:
            return PROCESSED

        # /command dispatch
        if content.startswith("/"):
            await self._handle_command(content, controls)
            return PROCESSED

        # @agent routing — extract @mentions and forward to specific agents
        at_mentions = re.findall(r"@(tool:)?(\w[\w.-]*)", content)
        if at_mentions:
            await self._handle_at_mention(content, at_mentions, controls)
            return PROCESSED

        # Plain message — provide context to the LLM planner so it can decide
        # what action to take (respond_to_user, create_agent, etc.)
        return EventProcessingResult(
            context_key="user_chat_message",
            context={
                "user_message": content,
                "controls": controls,
                "session_id": self.agent.metadata.session_id,
            },
        )

    async def _handle_at_mention(
        self, content: str, mentions: list[tuple[str, str]], controls: dict | None,
    ) -> None:
        """Route a message containing @mentions to the appropriate target.

        Mentions are tuples of (tool_prefix, name) from the regex.
        - ("tool:", "web-search") → tool request
        - ("", "impact-coordinator") → agent type
        - ("", "agent_abc123") → agent ID
        """
        for tool_prefix, name in mentions:
            if tool_prefix:
                # @tool:name — request a tool
                await self._post_response(
                    f"Tool `{name}` requested. "
                    f"(Tool execution not yet implemented — coming in a future update.)"
                )
                return

            # @agent_type or @agent_id — forward message to that agent
            try:
                from polymathera.colony.system import get_agent_system
                agent_system = get_agent_system()

                # Try to find by ID first, then by type
                agent_ids = await agent_system.list_all_agents()
                target_id = None

                for aid in agent_ids:
                    if aid.startswith(name) or aid == name:
                        target_id = aid
                        break

                if not target_id:
                    # Try matching by agent_type suffix
                    for aid in agent_ids:
                        info = await agent_system.get_agent_info(agent_id=aid)
                        agent_type = getattr(info, "agent_type", "") if info else ""
                        if agent_type.split(".")[-1].lower() == name.lower():
                            target_id = aid
                            break

                if target_id:
                    # Strip the @mention from content and forward
                    clean_content = re.sub(r"@(tool:)?\w[\w.-]*\s*", "", content).strip()
                    await self._post_response(
                        f"Forwarding to agent `{target_id[:16]}`: {clean_content}\n"
                        f"(Agent message forwarding not yet fully implemented.)"
                    )
                else:
                    await self._post_response(f"Agent `{name}` not found.")
            except Exception as e:
                await self._post_response(f"Failed to route to `{name}`: {e}")
            return

        # Fallback — shouldn't reach here
        await self._post_response("Could not process @mention.")

    @event_handler(pattern=SessionChatProtocol.reply_pattern())
    async def handle_user_reply(self, event: BlackboardEvent, repl: PolicyREPL) -> EventProcessingResult:
        """Handle a user reply to an agent question."""
        request_id, _ = SessionChatProtocol.parse_reply_key(event.key)
        payload = event.value if isinstance(event.value, dict) else {}
        content = payload.get("content", "")

        logger.info("SessionAgent received reply to request %s: %s", request_id, content[:100])

        # TODO: Route reply back to the requesting agent's blackboard.
        await self._post_response(f"Reply to {request_id} acknowledged: {content}")

        return PROCESSED

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _handle_command(self, raw: str, controls: dict | None) -> None:
        """Parse and dispatch a /command."""
        parts = raw.split(None, 1)
        command = parts[0].lstrip("/").lower()
        args_str = parts[1] if len(parts) > 1 else ""

        handler = {
            "help": self._cmd_help,
            "status": self._cmd_status,
            "agents": self._cmd_agents,
            "abort": self._cmd_abort,
            "analyze": self._cmd_analyze,
            "map": self._cmd_map,
            "set": self._cmd_set,
            "context": self._cmd_context,
        }.get(command)

        if handler:
            await handler(args_str, controls)
        else:
            await self._post_response(f"Unknown command: /{command}. Type /help for available commands.")

    async def _cmd_help(self, args: str, controls: dict | None) -> None:
        """Show available commands."""
        commands = {
            "/analyze <type>": "Start an analysis run (impact, compliance, intent, contracts, slicing, basic)",
            "/map <url>": "Map a repository or content to VCM",
            "/abort [run_id]": "Abort the current or specified run",
            "/status": "Show current session and run status",
            "/agents": "List active agents in this session",
            "/set <param>=<value>": "Set session parameters (max-agents, effort, etc.)",
            "/context [add|remove] [#ref]": "Show or modify active VCM context",
            "/help [command]": "Show this help or details for a specific command",
        }

        if args.strip():
            # Help for specific command
            key = f"/{args.strip()}"
            for cmd, desc in commands.items():
                if cmd.startswith(key):
                    await self._post_response(f"**{cmd}**\n{desc}")
                    return
            await self._post_response(f"Unknown command: {key}")
            return

        lines = ["**Available commands:**", ""]
        for cmd, desc in commands.items():
            lines.append(f"  `{cmd}` — {desc}")
        lines.append("")
        lines.append("**Syntax:** `@agent_type` to address agents, `#repo:name` to reference VCM assets")

        await self._post_response("\n".join(lines))

    async def _cmd_status(self, args: str, controls: dict | None) -> None:
        """Show session/run status."""
        session_id = self.agent.metadata.session_id
        await self._post_response(
            f"**Session:** {session_id}\n"
            f"**Agent:** {self.agent.agent_id}\n"
            f"**State:** {self.agent.state.value}"
        )

    async def _cmd_agents(self, args: str, controls: dict | None) -> None:
        """List active agents."""
        # Query agent system for agents in this session
        try:
            from polymathera.colony.system import get_agent_system
            agent_system = get_agent_system()
            agent_ids = await agent_system.list_all_agents()

            if not agent_ids:
                await self._post_response("No active agents.")
                return

            lines = [f"**Active agents:** ({len(agent_ids)})"]
            for aid in agent_ids[:20]:
                try:
                    info = await agent_system.get_agent_info(agent_id=aid)
                    agent_type = getattr(info, "agent_type", "unknown") if info else "unknown"
                    state = str(getattr(info, "state", "unknown")) if info else "unknown"
                    lines.append(f"  `{aid[:16]}` — {agent_type.split('.')[-1]} ({state})")
                except Exception:
                    lines.append(f"  `{aid[:16]}` — unknown")

            if len(agent_ids) > 20:
                lines.append(f"  ... and {len(agent_ids) - 20} more")

            await self._post_response("\n".join(lines))
        except Exception as e:
            await self._post_response(f"Failed to list agents: {e}")

    async def _cmd_abort(self, args: str, controls: dict | None) -> None:
        """Abort a run."""
        # TODO: Implement run cancellation via InterruptionProtocol
        await self._post_response("Abort not yet implemented. Coming in a future update.")

    async def _cmd_analyze(self, args: str, controls: dict | None) -> None:
        """Start an analysis run by spawning coordinator agents."""
        # TODO: Parse analysis type and parameters, spawn coordinators
        # via AgentPoolCapability (same flow as jobs.py _run_job).
        if not args.strip():
            await self._post_response(
                "Usage: `/analyze <type> [--max-agents N]`\n"
                "Types: impact, compliance, intent, contracts, slicing, basic"
            )
            return

        analysis_type = args.split()[0]
        valid_types = {"impact", "compliance", "intent", "contracts", "slicing", "basic"}
        if analysis_type not in valid_types:
            await self._post_response(f"Unknown analysis type: {analysis_type}. Valid: {', '.join(sorted(valid_types))}")
            return

        await self._post_response(
            f"Starting **{analysis_type}** analysis...\n"
            f"(Coordinator spawning not yet implemented — coming in the next update.)"
        )

    async def _cmd_map(self, args: str, controls: dict | None) -> None:
        """Map content to VCM."""
        if not args.strip():
            await self._post_response("Usage: `/map <url> [--branch name]`")
            return

        # TODO: Call VCM deployment handle to trigger mapping
        url = args.split()[0]
        await self._post_response(
            f"Mapping `{url}` to VCM...\n"
            f"(VCM mapping via session agent not yet implemented — use the VCM tab for now.)"
        )

    async def _cmd_set(self, args: str, controls: dict | None) -> None:
        """Set session parameters."""
        if not args.strip() or "=" not in args:
            await self._post_response("Usage: `/set <param>=<value>` (e.g., `/set max-agents=5`)")
            return

        # TODO: Update session configuration
        await self._post_response(f"Set: {args.strip()}")

    async def _cmd_context(self, args: str, controls: dict | None) -> None:
        """Show or modify VCM context."""
        # TODO: Query VCM for current context, allow add/remove
        await self._post_response("VCM context management not yet implemented.")


class SessionAgent(Agent):
    """Per-session orchestrator agent.

    Spawned via AgentHandle.from_blueprint() when a session is created.
    Receives user messages via blackboard, decides execution strategy,
    and orchestrates runs by spawning child coordinator agents.

    Bound capabilities:
    - SessionOrchestratorCapability: chat message handling and command routing
    - AgentPoolCapability: spawning and managing child coordinator agents
    """

    agent_type: str = "polymathera.colony.agents.sessions.session_agent.SessionAgent"
