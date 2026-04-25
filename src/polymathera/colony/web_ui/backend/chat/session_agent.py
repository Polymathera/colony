"""Session agent — per-session orchestrator for user interactions.

Each session gets a SessionAgent spawned via AgentHandle.from_blueprint().
It receives user messages via the SessionChatProtocol on the blackboard,
decides how to handle them (respond directly, spawn coordinators, route to
specific agents), and relays agent progress back to the user.
"""

from __future__ import annotations

import asyncio
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
from polymathera.colony.agents.blackboard.protocol import ActionPolicyLifecycleProtocol
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
    - Bridges generic ``ActionPolicyLifecycleProtocol`` events emitted
      by the agent's action policy into chat-side updates the WebSocket
      relay can forward to the browser.
    """

    #: Default sub-namespace under the session blackboard for chat
    #: traffic. Single source of truth — the chat router and any other
    #: consumer reach this constant rather than repeating the literal.
    DEFAULT_NAMESPACE = "session_chat"

    def __init__(
        self,
        agent: Agent | None = None,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = DEFAULT_NAMESPACE,
        input_patterns: list[str] | None = None,
        capability_key: str = "session_orchestrator",
        app_name: str | None = None,
    ):
        """Initialize session orchestrator capability.

        NOTE: scope cannot be BlackboardScope.AGENT in detached mode (agent=None).

        Args:
            agent: Owning agent (the SessionAgent). None for detached mode.
            scope: Blackboard scope (SESSION — shared partition for the session's chat)
            namespace: Namespace within the scope
            input_patterns: Event patterns to subscribe to. If None, auto-inferred
                from @event_handler decorators.
            capability_key: Unique key for the capability
            app_name: The `serving.Application` name where the agent system resides.
                    Required when creating detached handles from outside any `serving.deployment`.
        """
        scope_id = get_scope_prefix(scope, agent, namespace=namespace)
        logger.info(
            "SessionOrchestratorCapability init: scope=%s, namespace=%s, scope_id=%s",
            scope, namespace, scope_id,
        )
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=input_patterns,
            capability_key=capability_key,
            app_name=app_name,
        )
        self._lifecycle_relay_task: asyncio.Task | None = None

    @override
    async def initialize(self) -> None:
        """Spin up the policy→chat lifecycle bridge.

        ``ActionPolicyLifecycleProtocol`` events are emitted by the
        agent's action policy on the agent's primary blackboard. We
        cannot route them through the policy's own event queue (that
        would re-enter ``plan_step`` on every emission and form a
        feedback loop), so a dedicated background task subscribes
        directly and translates each event into the corresponding
        chat-blackboard record. The chat WebSocket relay then
        forwards those records to the browser.

        The task is cancelled on ``shutdown``. Failures inside the
        loop are logged but never propagated — the chat UI is a
        downstream consumer; its absence must not break the agent.
        """
        await super().initialize()
        if self._agent is None:
            # Detached mode: no agent blackboard to subscribe to.
            return
        self._lifecycle_relay_task = asyncio.create_task(
            self._relay_policy_lifecycle_to_chat(),
            name=f"policy_lifecycle_relay:{self._agent.agent_id}",
        )

    async def shutdown(self) -> None:
        """Cancel the lifecycle relay task. Idempotent."""
        if self._lifecycle_relay_task is not None:
            self._lifecycle_relay_task.cancel()
            try:
                await self._lifecycle_relay_task
            except (asyncio.CancelledError, Exception):  # pragma: no cover
                pass
            self._lifecycle_relay_task = None

    async def _relay_policy_lifecycle_to_chat(self) -> None:
        """Subscribe to policy lifecycle events on the agent's primary
        blackboard and translate each into a chat-blackboard write.

        See :class:`ActionPolicyLifecycleProtocol` for the event
        catalogue. The translation here is the only place that knows
        about both worlds — the policy emits generic events, the chat
        UI consumes chat-shaped records, and this method bridges them.
        """
        try:
            agent_bb = await self._agent.get_blackboard()
            chat_bb = await self.get_blackboard()
        except Exception as e:
            logger.warning(
                "SessionOrchestratorCapability: lifecycle relay "
                "failed to acquire blackboards (%s); chat-side action "
                "status will not appear", e,
            )
            return
        try:
            async for event in agent_bb.stream_events(
                pattern=ActionPolicyLifecycleProtocol.all_pattern(),
                event_types={"write"},
                timeout=None,
            ):
                try:
                    await self._handle_lifecycle_event(event, chat_bb)
                except Exception as e:  # pragma: no cover — defensive
                    logger.debug(
                        "lifecycle relay translation failed for %s: %s",
                        event.key, e,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover — defensive
            logger.error(
                "SessionOrchestratorCapability: lifecycle relay loop "
                "crashed: %s", e, exc_info=True,
            )

    async def _handle_lifecycle_event(
        self, event: BlackboardEvent, chat_bb,
    ) -> None:
        """Translate one lifecycle event into a chat-blackboard write.

        Action started/completed → ``chat:action_status:*`` (drives
        the spinner badge in ``ChatPanel.ActionStatusBanner``).

        Codegen retry → ``chat:action_status:*`` with a synthetic
        ``codegen_recovery`` action_id so the banner persists across
        the whole retry streak.

        Codegen failed (retries exhausted) → both a final
        ``chat:action_status:*`` clearing the banner AND a
        ``chat:agent:*`` system_failure message that lands in the
        user's chat history alongside normal agent replies.
        """
        key = event.key
        payload = event.value if isinstance(event.value, dict) else {}
        agent_id = payload.get("agent_id") or self._agent.agent_id

        if key.startswith(ActionPolicyLifecycleProtocol._ACTION_STARTED):
            action_id = payload.get("action_id") or "?"
            await chat_bb.write(
                f"chat:action_status:{agent_id}:{action_id}",
                {
                    "agent_id": agent_id,
                    "action_id": action_id,
                    "action_key": payload.get("action_key", ""),
                    "status": "running",
                    "started_at": payload.get("started_at"),
                },
            )
            return

        if key.startswith(ActionPolicyLifecycleProtocol._ACTION_COMPLETED):
            action_id = payload.get("action_id") or "?"
            await chat_bb.write(
                f"chat:action_status:{agent_id}:{action_id}",
                {
                    "agent_id": agent_id,
                    "action_id": action_id,
                    "action_key": payload.get("action_key", ""),
                    "status": "complete" if payload.get("success") else "failed",
                    "started_at": payload.get("started_at"),
                    "ended_at": payload.get("ended_at"),
                    "wall_time_ms": payload.get("wall_time_ms"),
                    "error": payload.get("error"),
                },
            )
            return

        if ActionPolicyLifecycleProtocol.is_codegen_retry_key(key):
            # ``codegen_retry`` carries both progress events (``finished=False``)
            # and the terminal success event (``finished=True, succeeded=True``)
            # that ``_emit_codegen_recovery_banner`` fires when a regenerated
            # cell finally validates. The banner must clear on the terminal
            # event — otherwise it stays "running" forever even though the
            # policy is idle, and ``/abort`` then correctly reports "nothing
            # to abort" while the user stares at a stuck spinner.
            attempt = payload.get("attempt", 0)
            max_attempts = payload.get("max_attempts", 0)
            finished = bool(payload.get("finished"))
            succeeded = bool(payload.get("succeeded"))
            if finished:
                await chat_bb.write(
                    f"chat:action_status:{agent_id}:codegen_recovery",
                    {
                        "agent_id": agent_id,
                        "action_id": "codegen_recovery",
                        "action_key": (
                            f"codegen — recovered on attempt {attempt}"
                            if succeeded
                            else f"codegen — gave up after {attempt} attempts"
                        ),
                        "status": "complete" if succeeded else "failed",
                        "started_at": payload.get("ts"),
                        "ended_at": payload.get("ts"),
                        "error": payload.get("error"),
                    },
                )
            else:
                await chat_bb.write(
                    f"chat:action_status:{agent_id}:codegen_recovery",
                    {
                        "agent_id": agent_id,
                        "action_id": "codegen_recovery",
                        "action_key": (
                            f"codegen — regenerating after invalid output "
                            f"(attempt {attempt}/{max_attempts})"
                        ),
                        "status": "running",
                        "started_at": payload.get("ts"),
                    },
                )
            return

        if ActionPolicyLifecycleProtocol.is_codegen_failed_key(key):
            attempts = payload.get("attempts", 0)
            error = payload.get("error", "")
            # Clear any lingering codegen_recovery banner.
            await chat_bb.write(
                f"chat:action_status:{agent_id}:codegen_recovery",
                {
                    "agent_id": agent_id,
                    "action_id": "codegen_recovery",
                    "action_key": (
                        f"codegen — gave up after {attempts} attempts"
                    ),
                    "status": "failed",
                    "started_at": payload.get("ts"),
                    "ended_at": payload.get("ts"),
                    "error": error,
                },
            )
            # And surface a regular chat message the user can see in
            # their history.
            mid = f"msg_{uuid.uuid4().hex[:12]}"
            await chat_bb.write(
                SessionChatProtocol.agent_message_key(agent_id, mid),
                {
                    "content": (
                        f"⚠️ I couldn't produce valid code for your "
                        f"last request after {attempts} attempts.\n\n"
                        f"**Last error:** `{error[:300]}`\n\n"
                        f"This usually means the LLM is producing output "
                        f"the validator can't parse (e.g., markdown "
                        f"fences, prose between statements, or "
                        f"hallucinated action keys). Try rephrasing or "
                        f"open an issue if it keeps happening."
                    ),
                    "agent_id": agent_id,
                    "agent_type": self._agent.agent_type,
                    "message_id": mid,
                    "timestamp": time.time(),
                    "kind": "system_failure",
                },
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

    @event_handler(
        pattern=SessionChatProtocol.control_message_pattern(),
        priority="high",
    )
    async def handle_control_command(
        self, event: BlackboardEvent, _repl: Any,
    ) -> EventProcessingResult:
        """Handle a high-priority control command from the user.

        Runs on the policy's concurrent high-priority lane, so it is
        NOT blocked by long-running actions in the main planning
        loop. Strict read-only contract:

        - MAY read agent / policy state via
          ``policy.get_status_snapshot()``.
        - MAY write to its own (chat) blackboard via
          ``self._post_response``.
        - MUST NOT dispatch actions through the action dispatcher.
        - MUST NOT mutate policy internals; cancellation paths are
          dedicated APIs added in Phase 2 of the design (see
          ``design_event_priority_and_action_interruption.md``).
        """
        payload = event.value if isinstance(event.value, dict) else {}
        command = (payload.get("command") or "").lower()

        if command in ("/status", "/whatdoing"):
            await self._respond_with_status_snapshot()
            return PROCESSED

        if command in ("/abort", "/cancel"):
            await self._handle_abort_command(reason=command)
            return PROCESSED

        if command == "/replace":
            content = (payload.get("content") or "").strip()
            await self._handle_replace_command(content)
            return PROCESSED

        # Unknown control: acknowledge so the user knows the command
        # was received on the high-priority lane.
        await self._post_response(
            f"⚠️ Unknown control command `{command}`. "
            f"Available: `/status`, `/whatdoing`, `/abort`, `/cancel`, `/replace`.",
            kind="control_ack",
        )
        return PROCESSED

    async def _handle_abort_command(self, *, reason: str) -> None:
        """Cancel the action policy's currently-executing action.

        Always called from the high-priority lane, so a long-running
        action does NOT keep us from getting here. We:

        1. Pull the live action policy off the agent. If there isn't one
           yet (agent still warming up) the user gets a clear message;
           there is nothing to abort.
        2. Call ``policy.abort_current(reason=...)`` — the policy is
           responsible for resetting any per-policy state (codegen
           recovery counters, in-flight LLM call) AND cancelling the
           dispatcher-tracked action. Returns True iff *something* was
           actually interrupted.
        3. Surface the outcome to the user. ``False`` is not a failure
           — it just means the agent was idle when /abort landed.
        """
        policy = getattr(self.agent, "action_policy", None)
        if policy is None:
            await self._post_response(
                "⚠️ No action policy attached to this agent — nothing to abort.",
                kind="control_ack",
            )
            return

        try:
            aborted = await policy.abort_current(reason=reason)
        except Exception as e:  # pragma: no cover — defensive
            logger.exception("abort_current() raised")
            await self._post_response(
                f"⚠️ Abort failed: {e}",
                kind="control_ack",
            )
            return

        if aborted:
            await self._post_response(
                f"⏹️ Aborted current action ({reason}).",
                kind="control_ack",
            )
        else:
            await self._post_response(
                "ℹ️ Nothing to abort — the agent is idle.",
                kind="control_ack",
            )

    async def _handle_replace_command(self, full_content: str) -> None:
        """Pre-emptive re-prioritisation: abort current action, queue new request.

        ``/replace <new request>`` is the user saying "stop what you're doing,
        do this instead". Two-step:

        1. Cancel the in-flight action via ``policy.abort_current()`` so the
           planner doesn't continue spending time on something the user has
           already discarded.
        2. Post the new request as a regular ``chat:user:*`` message on the
           same chat blackboard, so it picks up via the *normal* event lane
           in ``plan_step`` exactly as if the user had sent a fresh message.
           This keeps the planning path single-sourced — the high-priority
           lane never tries to *plan*, it only kicks the planner with new
           input.

        Empty payload is rejected with a clear message — ``/replace`` with
        no body is almost always a typo, and we'd rather flag it than
        silently abort + queue an empty message that the planner ignores.
        """
        # ``full_content`` is the full chat message including the leading
        # "/replace " — strip the command itself, preserving any whitespace
        # / newlines in the actual request.
        body = full_content[len("/replace"):].lstrip()
        if not body:
            await self._post_response(
                "⚠️ `/replace` requires a new request. "
                "Usage: `/replace <what to do instead>`.",
                kind="control_ack",
            )
            return

        policy = getattr(self.agent, "action_policy", None)
        aborted = False
        if policy is not None:
            try:
                aborted = await policy.abort_current(reason="/replace")
            except Exception:  # pragma: no cover — defensive
                logger.exception("abort_current() raised during /replace")

        try:
            await self._post_user_message_on_normal_lane(body)
        except Exception as e:
            logger.exception("/replace failed to enqueue new request")
            await self._post_response(
                f"⚠️ Aborted current action but failed to queue replacement: {e}",
                kind="control_ack",
            )
            return

        if aborted:
            await self._post_response(
                "🔁 Aborted current action and queued your new request.",
                kind="control_ack",
            )
        else:
            await self._post_response(
                "🔁 Agent was idle — queued your new request.",
                kind="control_ack",
            )

    async def _post_user_message_on_normal_lane(self, content: str) -> None:
        """Inject ``content`` as if the user had typed it as a fresh chat
        message. Used by ``/replace`` to seed the next iteration of the
        planner with the new request.

        Routes to ``chat:user:*`` on this capability's own scoped
        blackboard — same key shape and namespace the chat router
        produces, so the existing ``handle_user_message`` event handler
        on the normal lane picks it up unchanged.
        """
        blackboard = await self.get_blackboard()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        await blackboard.write(
            SessionChatProtocol.user_message_key(message_id),
            {
                "content": content,
                "command": None,
                "message_id": message_id,
                "controls": None,
                "timestamp": time.time(),
                "source": "control:/replace",
            },
        )

    async def _respond_with_status_snapshot(self) -> None:
        """Read the policy's status snapshot and post it as a chat
        message. No action dispatch, no LLM call — pure read."""
        try:
            policy = getattr(self.agent, "action_policy", None)
            snapshot = policy.get_status_snapshot() if policy else {
                "error": "no action policy attached to agent",
            }
        except Exception as e:  # pragma: no cover — defensive
            await self._post_response(
                f"⚠️ Could not read agent status: {e}",
            )
            return

        lines = ["**Agent status**", ""]
        # Render the snapshot as a small markdown table. Stable order
        # so the user can scan repeated /status calls easily.
        for k in (
            "policy_class", "agent_id", "mode",
            "in_recovery", "recovery_attempts", "max_recovery_attempts",
            "code_iteration_count",
            "queue_depth_normal", "queue_depth_high",
            "high_priority_loop_running",
            "reactive_only", "has_pending_work",
            "complete_signaled",
        ):
            if k in snapshot:
                lines.append(f"- **{k}**: `{snapshot[k]}`")
        # Surface anything else the snapshot carried that we didn't
        # explicitly list — defensive against future field additions.
        for k, v in snapshot.items():
            if k not in {
                "policy_class", "agent_id", "mode",
                "in_recovery", "recovery_attempts", "max_recovery_attempts",
                "code_iteration_count",
                "queue_depth_normal", "queue_depth_high",
                "high_priority_loop_running",
                "reactive_only", "has_pending_work",
                "complete_signaled",
            }:
                lines.append(f"- **{k}**: `{v}`")
        await self._post_response("\n".join(lines), kind="status")

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
