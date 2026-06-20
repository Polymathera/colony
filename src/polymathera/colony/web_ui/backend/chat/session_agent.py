"""Session agent — per-session orchestrator for user interactions.

Each session gets a SessionAgent spawned via AgentHandle.from_blueprint().
It receives user messages via the SessionChatProtocol on the blackboard,
decides how to handle them (respond directly, spawn coordinators, route to
specific agents), and relays agent progress back to the user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pprint
import time
import uuid
import re
from typing import Any

from overrides import override

from polymathera.colony.agents.base import Agent, AgentCapability, AgentHandle
from polymathera.colony.agents.metadata_parameters import (
    ParameterScope,
    ParameterSpec,
)
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.agents.models import AgentMetadata, AgentSuspensionState, RunContext, PolicyREPL
from polymathera.colony.agents.blackboard import BlackboardEvent
from polymathera.colony.agents.blackboard.protocol import (
    ActionPolicyLifecycleProtocol,
    AgentDiagnosticProtocol,
    CHAT_BLACKBOARD_NAMESPACE,
    DIAGNOSTIC_EMPTY_ITERATION_STREAK,
    DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK,
    HumanApprovalProtocol,
    HumanHelpProtocol,
    SELF_RELEVANT_DIAGNOSTIC_KINDS,
)
from polymathera.colony.agents.patterns.events import event_handler, EventProcessingResult, PROCESSED
from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.actions.repl import (
    REPL_GUIDANCE_OVERRIDE_KEY,
)
from .chat_protocol import SessionChatProtocol

logger = logging.getLogger(__name__)


def _render_caller_parameters(
    caller_parameters: list[Any],
) -> list[dict[str, Any]]:
    """Project a mission entry's ``caller_parameters`` list into the
    JSON-friendly shape the LLM planner reads from
    ``metadata.parameters['available_missions'][<m>]['mission_params']``.

    Accepts both ``ParameterSpec`` instances (the typed form on
    ``MissionSpec.caller_parameters``) and plain dicts (raw
    registry entries before model_validate). Tolerating both lets
    the L4-discovered mission path (which keeps entries as dicts in
    the L4 cache) and the colony-builtin path (which has them as
    typed specs once routed through ``MissionSpec``) share the same
    projector.

    Each rendered entry carries ``name`` / ``description`` /
    ``required`` / ``json_type`` / ``default`` so the planner sees
    a signature, not bare strings.
    """

    rendered: list[dict[str, Any]] = []
    for spec in caller_parameters:
        if isinstance(spec, dict):
            name = spec["name"]
            description = spec.get("description", "")
            json_type = spec.get("json_type", "string")
            has_default = "default" in spec or "default_factory" in spec
            default = spec.get("default") if has_default else None
        else:
            name = spec.name
            description = spec.description
            json_type = spec.json_type
            has_default = not spec.required
            default = spec.default if has_default else None
        entry: dict[str, Any] = {
            "name": name,
            "description": description,
            "json_type": json_type,
            "required": not has_default,
        }
        if has_default:
            entry["default"] = default
        rendered.append(entry)
    return rendered


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
    #: traffic. Single source of truth lives at
    #: ``CHAT_BLACKBOARD_NAMESPACE`` in
    #: ``agents/blackboard/protocol.py`` — exported here so
    #: chat-specific call sites that already imported
    #: ``SessionOrchestratorCapability`` keep one indirection level.
    DEFAULT_NAMESPACE = CHAT_BLACKBOARD_NAMESPACE

    # Key constants for the SESSION-scoped planner-loop state this
    # capability owns and refreshes. Read by ``_refresh_available_*``
    # writers + by every consumer (REPL, system-prompt builder).
    # ``REPL_GUIDANCE_OVERRIDE_KEY`` lives at the reader site (the REPL
    # action module — see ``patterns/actions/repl.py``) and is
    # re-exported here for the spec; importing from the reader keeps a
    # single source of truth without creating a web_ui→agents inversion.
    AVAILABLE_MISSIONS_KEY = "available_missions"
    AVAILABLE_TOOLS_KEY = "available_tools"
    REPL_GUIDANCE_OVERRIDE_KEY = REPL_GUIDANCE_OVERRIDE_KEY

    AGENT_METADATA_PARAMS = (
        ParameterSpec(
            name=AVAILABLE_MISSIONS_KEY,
            scope=ParameterScope.SESSION,
            description=(
                "Per-session map of {mission_type: {label, description, "
                "coordinator_class, worker_class, mission_params}} the "
                "LLM planner reads from the system prompt when picking "
                "a ``spawn_mission`` target. Rebuilt at every user "
                "message by ``_refresh_available_missions`` as the "
                "union of ``get_mission_registry()`` and the L4 "
                "design-monorepo mission cache."
            ),
            json_type="object",
            default_factory=dict,
        ),
        ParameterSpec(
            name=AVAILABLE_TOOLS_KEY,
            scope=ParameterScope.SESSION,
            description=(
                "Per-session map of {tool_capability_fqn: ToolEntry} "
                "the LLM planner reads to know which tools the agent "
                "can mount on a freshly-spawned worker via "
                "``AgentPoolCapability.create_agent``. Rebuilt from the "
                "L4 design monorepo's ``.colony/tool-registry.json`` "
                "by ``_refresh_available_tools``."
            ),
            json_type="object",
            default_factory=dict,
        ),
        ParameterSpec(
            name=REPL_GUIDANCE_OVERRIDE_KEY,
            scope=ParameterScope.SESSION,
            description=(
                "Optional Markdown block that overrides the default "
                "REPL guidance section of the planner's system prompt. "
                "Used by chat sessions to tailor the REPL contract for "
                "session-driven planning (vs the worker-driven default)."
            ),
            json_type="string",
            default=None,
        ),
    )

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

    @override
    async def initialize(self) -> None:
        """Spin up the policy→chat lifecycle bridge.

        ``ActionPolicyLifecycleProtocol`` events are emitted by the
        agent's action policy on the agent's primary blackboard. They
        cannot route through the policy's own normal event queue —
        that would re-enter ``plan_step`` on every emission and form a
        feedback loop on this agent's own actions. Instead the
        :meth:`stream_events_to_queue` override below subscribes them
        on the policy's high-priority queue, where
        :meth:`handle_lifecycle_event` (decorated
        ``priority="high"``) translates each event into the
        corresponding chat-blackboard record on the dedicated
        ``_run_high_priority_loop`` task. The chat WebSocket relay
        then forwards those records to the browser.
        """
        await super().initialize()
        if self._agent is None:
            # Detached mode: no agent blackboard to subscribe to.
            return

    def _refresh_available_missions(self) -> None:
        """Rebuild ``self.agent.metadata.parameters["available_missions"]``
        as the union of:

        - colony-builtin missions + ``polymathera.mission_types``
          entry-point group (via :func:`get_mission_registry`), AND
        - L4 missions discovered under the parent agent's per-agent
          design-monorepo clone (via :func:`get_l4_extensions`, the
          shared L4 lookup helper).

        L4 entries shadow builtins on key collision (last-write-wins),
        mirroring the convention in
        :func:`get_mission_registry` (plugin entry-points shadow
        colony-builtins).

        The output dict matches the shape
        :func:`polymathera.colony.web_ui.backend.routers.sessions.create_session`
        produces at session-create time — single source of truth for
        the four planner-visible fields per mission
        (``label`` / ``description`` / ``coordinator_class`` /
        ``worker_class``).

        Falls back to the entry-points-only registry when L4 is not
        available (no monorepo URL, no provider mounted, clone
        failure). Failure modes are logged inside
        :func:`get_l4_extensions` so the operator sees the cause in
        container logs rather than a silently-empty mission list.

        Synchronous because the underlying cache
        (``RepoStateProvider.discovered_extensions``) is sync; callers
        run this inside ``loop.run_in_executor`` when blocking on
        the lazy clone is undesirable.
        """

        if self._agent is None:
            return
        from polymathera.colony.agents.mission_registry import (
            get_mission_registry,
        )
        from polymathera.colony.design_monorepo.extensions import (
            get_l4_extensions,
        )

        merged = dict(get_mission_registry())

        snapshot = get_l4_extensions(self._agent)
        if snapshot is not None:
            for key, entry in snapshot.missions.items():
                if key in merged:
                    logger.warning(
                        "SessionOrchestratorCapability: L4 mission %r "
                        "shadows a colony-builtin / entry-point mission",
                        key,
                    )
                merged[key] = entry

        # Project to the planner-visible shape: the four core fields
        # plus the CALLER-scoped ``mission_params`` signature so the
        # planner sees per-mission expectations (name + description +
        # optionality) rather than having to guess from prose.
        # ``caller_parameters`` entries are rendered as plain dicts
        # for prompt readability — the registry validates them via
        # ``MissionSpec`` at load time, so by this point they're
        # well-formed.
        available = {
            atype: {
                "label": reg["label"],
                "description": reg.get("description", ""),
                "coordinator_class": reg.get("coordinator_v2", ""),
                "worker_class": reg.get("worker", ""),
                "mission_params": _render_caller_parameters(
                    reg.get("caller_parameters", []),
                ),
            }
            for atype, reg in merged.items()
        }
        self._agent.metadata.parameters[self.AVAILABLE_MISSIONS_KEY] = available

    def _refresh_available_tools(self) -> None:
        """Rebuild ``self.agent.metadata.parameters["available_tools"]``
        from the L4 design monorepo's
        :func:`polymathera.colony.design_monorepo.registry.load_registry`
        catalog.

        The LLM planner reads this dict to know which tool capabilities
        the agent CAN mount in a freshly-spawned worker
        (``AgentPoolCapability.create_agent`` resolves the FQN via the
        canonical ``class_resolver`` fallback registry). Entries with
        empty ``capability_fqn`` are catalog-only stubs and are
        omitted from the planner-visible dict — the planner cannot
        mount what doesn't exist yet — but the same stub stays
        discoverable to the ``BuildVsBuyCapability`` advisor through
        ``RepoStateProvider.find_existing_tool``.

        Synchronous + thread-safe for the same reasons as
        :meth:`_refresh_available_missions`. Callers run this inside
        ``loop.run_in_executor`` to avoid blocking the event loop on
        the first lazy clone.
        """
        if self._agent is None:
            return
        from polymathera.colony.design_monorepo.extensions import (
            get_l4_extensions,
        )

        snapshot = get_l4_extensions(self._agent)
        available: dict[str, dict[str, str]] = {}
        if snapshot is not None:
            for name, entry in snapshot.tools.items():
                if not entry.capability_fqn:
                    continue
                available[name] = {
                    "purpose": entry.purpose,
                    "location": entry.location,
                    "capability": entry.capability,
                    "capability_fqn": entry.capability_fqn,
                }
        self._agent.metadata.parameters[self.AVAILABLE_TOOLS_KEY] = available

    async def _refresh_monorepo_extensions(self) -> None:
        """Refresh the L4 extension snapshot in metadata.parameters.

        The LLM planner reads from this snapshot when deciding what tools
        and missions are available, so keeping it up to date ensures the
        LLM can pick up mid-session additions. Async wrapper around the
        sync ``_refresh_available_missions`` / ``_refresh_available_tools``
        helpers; this method hops them to a thread so the first call's
        blocking ``git clone`` of the L4 design monorepo does not stall
        the event loop.
        """
        # Refresh the dynamic L4 mission / agent registry the LLM
        # planner reads from ``metadata.parameters["available_missions"]``
        # right before the planner runs. Cheap when nothing changed
        # (mtime fingerprint hit on the cached
        # ``RepoStateProvider.discovered_extensions``); picks up
        # mid-session L1-E mission additions on the next stat tick.
        # Failure here must NOT block the user's message — a refresh
        # error logs and the planner sees the previous snapshot.

        # First refresh of available_missions + available_tools against
        # the L4 design monorepo's ``.colony/`` (if any). Runs in a thread
        # because the underlying lazy clone of the design monorepo is
        # blocking IO; we don't want to stall the event loop on a
        # multi-second git clone.
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._refresh_available_missions,
            )
        except Exception:  # noqa: BLE001 — refresh must not block init
            logger.exception(
                "SessionOrchestratorCapability: "
                "available_missions refresh failed; planner will see "
                "the static snapshot",
            )

        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._refresh_available_tools,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "SessionOrchestratorCapability: "
                "available_tools refresh failed; planner will see "
                "the previous snapshot",
            )

    @event_handler(
        pattern=HumanApprovalProtocol.request_pattern(),
        priority="high",
    )
    async def handle_human_approval_request(
        self, event: BlackboardEvent, _repl: Any,
    ) -> EventProcessingResult | None:
        """Translate one ``HumanApprovalRequest`` into a chat agent_question.

        The chat scope's existing WebSocket relay
        (:func:`_listen_for_agent_messages`) forwards every
        ``chat:agent:*`` write whose payload carries
        ``awaiting_reply=True`` as a typed ``agent_question`` message;
        the ``kind`` field tells the frontend to send the response via
        the HTTP endpoint rather than the WebSocket reply lane.
        """

        try:
            request_id = HumanApprovalProtocol.parse_request_key(event.key)
        except ValueError:
            return PROCESSED
        chat_bb = await self.get_blackboard()
        payload = event.value if isinstance(event.value, dict) else {}
        question = payload.get("question") or "(empty approval request)"
        options = payload.get("options") or ["approve", "reject"]
        action_type = payload.get("action_type") or None
        requester_agent_id = (
            payload.get("requester_agent_id")
            or self._agent.agent_id
        )

        message_id = f"appr_msg_{uuid.uuid4().hex[:12]}"
        await chat_bb.write(
            SessionChatProtocol.agent_message_key(
                requester_agent_id, message_id,
            ),
            {
                "message_id": message_id,
                "agent_id": requester_agent_id,
                "agent_type": "human_approval",
                "content": question,
                "request_id": request_id,
                "response_options": list(options),
                "action_type": action_type,
                "awaiting_reply": True,
                "kind": "human_approval",
                "timestamp": time.time(),
                "extra": payload.get("extra") or {},
            },
        )
        return PROCESSED

    @event_handler(
        pattern=HumanHelpProtocol.request_pattern(),
        priority="high",
    )
    async def handle_human_help_request(
        self, event: BlackboardEvent, _repl: Any,
    ) -> EventProcessingResult | None:
        """Translate one ``HumanHelpRequest`` into a chat
        ``agent_question`` carrying ``kind='human_help'``.

        Sibling of :meth:`handle_human_approval_request`. The chat
        scope's existing WebSocket relay forwards every
        ``chat:agent:*`` write whose payload carries
        ``awaiting_reply=True`` as a typed ``agent_question``
        message; the ``kind`` field tells the frontend to route the
        operator's response via the
        ``/sessions/{id}/human_help/{request_id}/respond`` HTTP
        endpoint rather than the WebSocket reply lane or the
        ``human_approval`` endpoint. ``options`` is the agent's
        candidate-action list (rendered as buttons); a free-text
        ``guidance`` field is always available so the operator can
        write open guidance when none of the options fit.
        ``context`` accompanies the question so the operator sees
        what the agent has tried + observed."""

        try:
            request_id = HumanHelpProtocol.parse_request_key(event.key)
        except ValueError:
            return PROCESSED
        chat_bb = await self.get_blackboard()
        payload = event.value if isinstance(event.value, dict) else {}
        question = payload.get("question") or "(empty help request)"
        options = payload.get("options") or []
        context_text = payload.get("context") or ""
        requester_agent_id = (
            payload.get("requester_agent_id")
            or self._agent.agent_id
        )

        message_id = f"help_msg_{uuid.uuid4().hex[:12]}"
        await chat_bb.write(
            SessionChatProtocol.agent_message_key(
                requester_agent_id, message_id,
            ),
            {
                "message_id": message_id,
                "agent_id": requester_agent_id,
                "agent_type": "human_help",
                "content": question,
                "request_id": request_id,
                "response_options": list(options),
                "awaiting_reply": True,
                "kind": "human_help",
                "timestamp": time.time(),
                "extra": {"context": context_text},
            },
        )
        return PROCESSED

    @event_handler(
        pattern=ActionPolicyLifecycleProtocol.all_pattern(),
        priority="high",
    )
    async def handle_lifecycle_event(
        self, event: BlackboardEvent, _repl: Any,
    ) -> EventProcessingResult | None:
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
        chat_bb = await self.get_blackboard()
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
            return PROCESSED

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
            return PROCESSED

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
            return PROCESSED

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
            return PROCESSED

        return PROCESSED

    @override
    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        *,
        high_priority_queue: asyncio.Queue[BlackboardEvent] | None = None,
    ) -> None:
        """Add the cross-scope subscriptions this capability's
        ``@event_handler`` methods need.

        Routing mirrors the base method's ``target_for_high`` rule so
        ``priority="high"`` patterns reach the dedicated
        ``_run_high_priority_loop`` task (side-effect mirrors that
        must NOT re-enter the planner) while normal-priority patterns
        reach the main planning queue (real planner triggers).

        - Agent's primary scope → ``ActionPolicyLifecycleProtocol``
          events for :meth:`handle_lifecycle_event` (HIGH — same-scope
          side-effect mirror; routing through the main queue would
          feed the planner its own action_started/completed events
          and form an infinite re-iteration loop, see
          ``session_agent_lifecycle_feedback_loop_audit.md``).
        - Session's ``human_approval`` scope →
          ``HumanApprovalProtocol`` requests for
          :meth:`handle_human_approval_request` (HIGH — pure chat
          mirror, no planner context bound).
        - Session's ``agent_diagnostic`` scope →
          ``AgentDiagnosticProtocol`` events for
          :meth:`handle_agent_diagnostic` (NORMAL — produces
          ``EventProcessingResult`` planner context the LLM must see
          on its next iteration; producers are CHILD agents so there
          is no feedback loop).
        """

        await super().stream_events_to_queue(
            event_queue, high_priority_queue=high_priority_queue,
        )
        from polymathera.colony.agents.patterns.actions.code_generation import (
            BlockStreakTracker,
        )
        from polymathera.colony.agents.patterns.capabilities.human_approval import (
            HumanApprovalCapability,
        )

        target_for_high = high_priority_queue or event_queue

        agent_bb = await self._agent.get_blackboard()
        agent_bb.stream_events_to_queue(
            target_for_high,
            pattern=ActionPolicyLifecycleProtocol.all_pattern(),
            event_types={"write"},
        )

        human_bb = await self.get_blackboard(
            scope_id=get_scope_prefix(
                BlackboardScope.SESSION,
                namespace=HumanApprovalCapability.DEFAULT_NAMESPACE,
            ),
            enable_events=True,
        )
        human_bb.stream_events_to_queue(
            target_for_high,
            pattern=HumanApprovalProtocol.request_pattern(),
            event_types={"write"},
        )

        diag_bb = await self.get_blackboard(
            scope_id=get_scope_prefix(
                BlackboardScope.SESSION,
                namespace=BlockStreakTracker.DIAGNOSTIC_NAMESPACE,
            ),
            enable_events=True,
        )
        diag_bb.stream_events_to_queue(
            event_queue,
            pattern=AgentDiagnosticProtocol.event_pattern(),
            event_types={"write"},
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

    @staticmethod
    def _format_code_value(value: Any, lang: str | None) -> str:
        """Pretty-print a structured value into a string suitable for a
        ``kind: "code"`` attachment.

        Lives here because three turns of prompt iteration showed that
        agents reliably reach for ``str(out)`` instead of
        ``pprint.pformat(out)`` even when explicitly told. Doing the
        formatting framework-side is the durable fix: the agent can
        pass ``content=out`` (raw dict, list, ActionResult.output) and
        get readable multi-line output regardless of which serializer
        habit the LLM falls into.

        Dispatches on ``lang`` so the same attachment shape works for
        Python, JSON, and YAML payloads. Any other lang value falls
        back to ``pprint.pformat`` — a safe default that produces
        well-indented Python repr.
        """

        normalised_lang = (lang or "python").lower()
        if normalised_lang == "json":
            try:
                return json.dumps(value, indent=2, default=str, sort_keys=True)
            except Exception:  # noqa: BLE001
                return pprint.pformat(value, width=88)
        if normalised_lang in ("yaml", "yml"):
            try:
                import yaml  # type: ignore[import-not-found]
                return yaml.safe_dump(value, default_flow_style=False, sort_keys=False)
            except Exception:  # noqa: BLE001
                return pprint.pformat(value, width=88)
        return pprint.pformat(value, width=88)

    @classmethod
    def _normalize_attachments(
        cls, attachments: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Apply server-side formatting to attachment payloads.

        Today this only touches ``kind: "code"`` attachments whose
        ``content`` is not already a string — those get pretty-printed
        via :meth:`_format_code_value`. Other kinds are passed through
        unchanged. Returns a *new* list so the caller's input is not
        mutated.
        """

        if not attachments:
            return attachments
        out: list[dict[str, Any]] = []
        for att in attachments:
            kind = att.get("kind")
            if kind == "code" and not isinstance(att.get("content"), str):
                copy = dict(att)
                copy["content"] = cls._format_code_value(
                    att.get("content"), att.get("lang"),
                )
                out.append(copy)
            else:
                out.append(att)
        return out

    @action_executor()
    async def spawn_mission(
        self,
        *,
        mission_type: str,
        mission_params: dict[str, Any] | None = None,
        max_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Spawn a mission coordinator for ``mission_type`` — the
        recommended way to start a mission task from chat.

        Why this exists (rather than the LLM calling ``create_agent``
        directly): the LLM only needs to pick a ``mission_type`` key
        from ``available_missions``; this action does the
        coordinator-class lookup, metadata construction, and dispatch
        to :meth:`AgentPoolCapability.create_agent`. No inference
        chain "extract coordinator_class from a dict literal, then
        pass it as agent_type" — which is brittle for any LLM and
        broken for weaker ones.

        Pipeline:

        1. Look up ``mission_type`` in the LIVE merged mission
           registry (``get_mission_registry()`` ∪
           ``get_l4_extensions(agent).missions``). This is the same
           source the static snapshot in
           ``metadata.parameters["available_missions"]`` mirrors;
           re-reading live ensures mid-session L1-E mission additions
           are reachable.
        2. Build :class:`AgentMetadata` populated from the registry
           entry's ``self_concept`` and any caller-supplied
           ``mission_params`` (e.g. OPM-MEG's
           ``noise_floor_target_fT_rt_hz``).
        3. Dispatch to :meth:`AgentPoolCapability.create_agent` with
           the registry's ``coordinator_v2`` class path.
           ``create_agent`` handles L4 class resolution via its
           discovered-agents fallback, so L4 coordinators authored
           under ``<monorepo>/.colony/agents/`` work transparently.

        Args:
            mission_type: A key from ``available_missions`` (rendered
                in this agent's system prompt). Must match a
                registered mission; mismatch raises ``ValueError``.
            mission_params: Optional domain-specific parameters
                threaded into the coordinator's
                ``metadata.parameters``. Pass values matching the
                mission's declared ``caller_parameters`` (rendered
                into the system prompt as ``mission_params``).
            max_iterations: Reasoning-loop cap for the coordinator
                (default 20).

        Returns:
            A dict whose ``outcome`` field is the canonical
            discriminator. Branch on it before reading per-outcome
            fields:

            - ``outcome="spawned"``: coordinator just started. Use
              ``result["agent_id"]`` to address it.
            - ``outcome="return_existing"``: a coordinator is ALREADY
              running for this mission_type + scope; the gate handed
              it back instead of starting a duplicate. Use
              ``result["agent_id"]`` (same field as ``spawned``);
              do NOT call ``spawn_mission`` again.
              ``result["reason"]`` carries the gate's rationale.
            - ``outcome="rejected"``: the spawn gate refused (e.g.
              cap reached under ``on_concurrency_violation="reject"``,
              or a sibling reservation never resolved within the
              await window). Tell the user
              ``result["error"]``; ``result["suggested_action"]``
              is the gate's hint for next steps. Do NOT retry blindly.
            - ``outcome="error"``: configuration / dispatch / mission-
              registry failure (unknown mission_type, missing
              ``coordinator_v2``, no AgentPoolCapability mounted).
              Tell the user ``result["error"]``. One retry is
              reasonable; then escalate.

            Legacy ``result["created"]`` (bool) and
            ``result["mission_gate"]`` are preserved as computed
            properties on :class:`SpawnOutcome` for back-compat with
            consumers that predate the discriminator; new code SHOULD
            branch on ``outcome``.
        """

        from polymathera.colony.agents.mission_registry import (
            get_mission_registry,
        )
        from polymathera.colony.design_monorepo.extensions import (
            get_l4_extensions,
        )
        from polymathera.colony.agents.patterns.capabilities.agent_pool import (
            AgentPoolCapability,
        )
        from polymathera.colony.agents.missions.execution_ledger import (
            SpawnOutcome,
        )
        from polymathera.colony.agents.models import AgentMetadata

        # 1) Look up the mission in the LIVE registry — the chat-side
        # static snapshot can lag if a mission was added since the
        # last refresh.
        registry = dict(get_mission_registry())
        if self._agent is not None:
            snapshot = get_l4_extensions(self._agent)
            if snapshot is not None:
                registry.update(snapshot.missions)
        if mission_type not in registry:
            available = sorted(registry.keys())
            return SpawnOutcome(
                outcome="error",
                mission_type=mission_type,
                coordinator_class="",
                label="",
                error=(
                    f"Unknown mission type {mission_type!r}. "
                    f"Available: {available}"
                ),
            ).model_dump()
        reg = registry[mission_type]
        coord_class = reg.get("coordinator_v2") or reg.get("coordinator_v1") or ""
        if not coord_class:
            return SpawnOutcome(
                outcome="error",
                mission_type=mission_type,
                coordinator_class="",
                label=reg.get("label", ""),
                error=(
                    f"Mission {mission_type!r} has no coordinator_v2 "
                    f"or coordinator_v1 in its registry entry."
                ),
            ).model_dump()

        # 2) Build the coordinator's metadata.
        #
        # * ``self_concept`` — stamped by the shared
        #   :func:`build_coordinator_self_concept` helper (single
        #   source of truth for the
        #   ``MissionSelfConcept`` → ``AgentSelfConcept`` bridge;
        #   both this call site and the REST ``jobs.start_run``
        #   route through it).
        # * ``parameters`` — just the LLM-supplied mission_params
        #   plus the ``mission_type`` tag. COLONY / SESSION-scoped
        #   keys (``design_monorepo_url``, ``git_attribution``,
        #   ``github_identity``, …) are no longer threaded here —
        #   the central inheritance gate in
        #   ``AgentPoolCapability.create_agent`` walks them off this
        #   SessionAgent's metadata using the typed
        #   ``AGENT_METADATA_PARAMS`` registry, so every spawn
        #   anywhere in the codebase gets them automatically.
        from polymathera.colony.agents.configs import (
            build_coordinator_self_concept,
            resolve_effective_max_iterations,
            resolve_mission_execution_policy,
        )
        from polymathera.colony.agents.class_resolver import resolve_class

        params: dict[str, Any] = dict(mission_params or {})
        params.setdefault("mission_type", mission_type)
        # Resolve the coordinator's declared mission-execution policy
        # so we can apply ``policy.max_iterations`` as the metadata
        # default when the caller didn't pass an explicit override.
        # Precedence rules (caller > policy > schema-default 20) live
        # in :func:`resolve_effective_max_iterations` — one source of
        # truth shared with the REST path in ``routers/jobs.py``.
        try:
            coord_cls_obj = resolve_class(coord_class)
        except (ImportError, AttributeError, ValueError):
            coord_cls_obj = None
        coord_policy = resolve_mission_execution_policy(
            spec=reg, coordinator_class=coord_cls_obj,
        )
        effective_max_iterations = resolve_effective_max_iterations(
            caller_override=max_iterations, policy=coord_policy,
        )
        # The coordinator runs in this SessionAgent's runtime, so
        # the syscontext default_factory captures the right session
        # context (the SessionAgent itself is spawned inside a
        # ``user_execution_context(session_id=...)`` block in
        # ``routers/sessions.py::create_session``). The prior
        # ``session_id=self.agent.metadata.session_id`` kwarg was
        # silently dropped — ``session_id`` is a read-only @property
        # delegating to ``syscontext.session_id``, not a writable
        # field.
        coord_metadata = AgentMetadata(
            role=f"{reg.get('label', mission_type)} coordinator",
            goals=[f"Run {reg.get('label', mission_type)} mission"],
            max_iterations=effective_max_iterations,
            self_concept=build_coordinator_self_concept(
                reg, mission_type=mission_type,
            ),
            parameters=params,
        )

        # 3) Dispatch to AgentPoolCapability.create_agent. The pool
        # capability is mounted on this same agent (every
        # chat-spawned SessionAgent gets AgentPoolCapability.bind()
        # at session-create time), so the lookup is safe.
        pool = self.agent.get_capability_by_type(AgentPoolCapability)
        if pool is None:
            return SpawnOutcome(
                outcome="error",
                mission_type=mission_type,
                coordinator_class=coord_class,
                label=reg.get("label", ""),
                error=(
                    "AgentPoolCapability is not mounted on this agent; "
                    "spawn_mission requires it for dispatch."
                ),
            ).model_dump()
        # Funnel through the mission spawn-gate
        # (``admit_and_spawn``), which both this chat path AND the
        # REST ``routers/jobs.py::_run_job`` path share. The gate
        # consults the cluster-shared :class:`MissionExecutionLedger`
        # (Redis-backed via ``StateManager``) so concurrency caps
        # enforce uniformly across workers — the
        # ``AgentPoolCapability.create_agent`` primitive
        # stays mission-unaware. Return shape is exactly what the
        # helper produces, so the LLM has one stable schema to
        # branch on: ``created`` + optional ``mission_gate``
        # ("return_existing" | "rejected") + ``reason`` /
        # ``suggested_action`` on the gate paths.
        from polymathera.colony.agents.missions.execution_ledger import (
            admit_and_spawn,
        )
        mode = params.get("mode")
        return await admit_and_spawn(
            parent_agent=self.agent,
            pool=pool,
            agent_type=coord_class,
            metadata=coord_metadata,
            mission_type=mission_type,
            mode=str(mode) if mode is not None else None,
            label=reg.get("label", ""),
        )

    @action_executor()
    async def list_spawned_missions(
        self,
        *,
        mission_type: str | None = None,
    ) -> dict[str, Any]:
        """Query the cluster-shared ledger for missions this session has
        already spawned.

        Read primitive backing the "before calling ``spawn_mission``,
        check if there's already a live coordinator for this kind of
        work" loop. Reads :class:`MissionExecutionLedger` directly —
        the authoritative source for "what missions are running where"
        — so the answer reflects every Ray worker in the cluster, not
        just this process.

        Scope resolution: when ``mission_type`` is given, the mission's
        declared ``concurrency_scope`` decides which bucket to read
        (``project_planning`` is SESSION-scoped, others may differ).
        When ``mission_type`` is ``None``, the query defaults to the
        SessionAgent's own SESSION bucket — the natural concern of the
        chat planner. Wider visibility (colony, tenant) is a future
        opt-in via an explicit ``scope=`` kwarg.

        Args:
            mission_type: A key from ``available_missions``. ``None``
                returns every mission running in this SessionAgent's
                SESSION scope.

        Returns:
            ``{"missions": [{"agent_id", "mission_type", "mode",
            "started_at"}, ...]}``. Empty list when nothing matches
            (NOT an error — "no live missions" is a normal answer the
            planner branches on).
        """

        from polymathera.colony.agents.mission_registry import (
            get_mission_registry,
        )
        from polymathera.colony.design_monorepo.extensions import (
            get_l4_extensions,
        )
        from polymathera.colony.agents.configs import (
            MissionConcurrencyScope,
            resolve_mission_execution_policy,
        )
        from polymathera.colony.agents.class_resolver import resolve_class
        from polymathera.colony.agents.missions.execution_ledger import (
            get_mission_execution_ledger,
            resolve_scope_id,
        )

        # Resolve scope per registry entry when mission_type is given;
        # else default to SESSION (the SessionAgent's natural concern).
        if mission_type is None:
            scope = MissionConcurrencyScope.SESSION
        else:
            registry = dict(get_mission_registry())
            if self._agent is not None:
                snapshot = get_l4_extensions(self._agent)
                if snapshot is not None:
                    registry.update(snapshot.missions)
            if mission_type not in registry:
                available = sorted(registry.keys())
                return {
                    "missions": [],
                    "error": (
                        f"Unknown mission type {mission_type!r}. "
                        f"Available: {available}"
                    ),
                }
            reg = registry[mission_type]
            coord_class = (
                reg.get("coordinator_v2") or reg.get("coordinator_v1") or ""
            )
            try:
                coord_cls_obj = (
                    resolve_class(coord_class) if coord_class else None
                )
            except (ImportError, AttributeError, ValueError):
                coord_cls_obj = None
            policy = resolve_mission_execution_policy(
                spec=reg, coordinator_class=coord_cls_obj,
            )
            scope = policy.concurrency_scope

        scope_id = resolve_scope_id(scope, self.agent)
        ledger = await get_mission_execution_ledger(self._app_name)
        pairs = await ledger.list_for_scope(
            scope=scope, scope_id=scope_id, mission_type=mission_type,
        )

        # ``mission_type`` comes from the KEY (not the entry). ``mode``
        # and ``started_at`` come from the ENTRY — direct attribute
        # access, no getattr defaults: every field is a pydantic
        # default-factory field, so it's always present.
        return {
            "missions": [
                {
                    "agent_id": entry.agent_id,
                    "mission_type": key.mission_type,
                    "mode": entry.mode,
                    "started_at": entry.started_at,
                }
                for key, entry in pairs
            ],
        }

    @action_executor()
    async def respond_to_user(
        self,
        content: str,
        attachments: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Send a text response to the user in the chat.

        Use this action to respond to user questions, provide status updates,
        or acknowledge requests. For mission tasks, use ``spawn_mission``
        (NOT ``create_agent`` directly).

        Args:
            content: The message text to send to the user (supports markdown).
            attachments: Optional list of structured attachments rendered
                below ``content``. Each attachment is a dict with a
                ``kind`` discriminator the chat UI dispatches on. Three
                kinds are supported today:

                - ``{"kind": "code", "lang": "python", "content": <value>,
                   "label": "..."?}`` — fenced code block, collapsible
                  past 12 lines / 800 chars. ``content`` accepts EITHER
                  a pre-formatted string OR a raw Python value (dict,
                  list, ``ActionResult.output``). When non-string, the
                  framework pretty-prints it via ``pprint.pformat`` /
                  ``json.dumps`` / ``yaml.safe_dump`` based on
                  ``lang``. Pass the raw value — DO NOT call
                  ``str(value)`` yourself; that produces a single-line
                  blob that the chat cannot lay out cleanly.
                - ``{"kind": "table", "rows": [...], "columns": [...]?,
                   "label": "..."?}`` — see :meth:`respond_to_user_with_table`.
                - ``{"kind": "diff", "before": "...", "after": "...",
                   "lang": "..."?, "label": "..."?}`` — see
                  :meth:`respond_to_user_with_diff`.

                Future kinds (image, plot, error_trace, …) plug in
                without changing the action surface — only the
                renderer registry on the frontend grows.
            **extra: Additional fields to include in the message payload
                (e.g., request_id, response_options, awaiting_reply).

        Returns:
            Dict with message_id of the posted message.
        """
        bb = await self.get_blackboard()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        key = SessionChatProtocol.agent_message_key(self.agent.agent_id, message_id)
        payload: dict[str, Any] = {
            "content": content,
            "agent_id": self.agent.agent_id,
            "agent_type": self.agent.agent_type,
            "message_id": message_id,
            "timestamp": time.time(),
            **extra,
        }
        normalised = self._normalize_attachments(attachments)
        if normalised:
            payload["attachments"] = normalised
        await bb.write(key, payload)
        return {"message_id": message_id}

    @action_executor()
    async def respond_to_user_with_table(
        self,
        *,
        summary: str,
        rows: list[dict[str, Any]],
        columns: list[str] | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat response that pairs a one-line summary with a
        rendered table — for action results that are naturally
        tabular (per-file outcomes, per-source counts, side-by-side
        comparisons).

        Args:
            summary: Short headline rendered as plain markdown above
                the table. The user reads this first.
            rows: Each dict is one row. Keys whose values are missing
                in a given row render as empty cells.
            columns: Column order. Defaults to the union of keys
                across all rows, sorted alphabetically.
            label: Optional caption shown above the table.

        Returns:
            Dict with message_id of the posted message.
        """
        if columns is None:
            seen: dict[str, None] = {}
            for row in rows:
                for k in row:
                    seen.setdefault(k, None)
            columns = sorted(seen)
        attachment: dict[str, Any] = {
            "kind": "table",
            "rows": list(rows),
            "columns": list(columns),
        }
        if label:
            attachment["label"] = label
        return await self.respond_to_user(content=summary, attachments=[attachment])

    @action_executor()
    async def respond_to_user_with_diff(
        self,
        *,
        summary: str,
        before: str,
        after: str,
        lang: str = "text",
        label: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat response that pairs a one-line summary with a
        before/after diff — for action results that describe a state
        change (config edit, file rewrite, schema migration).

        Args:
            summary: Short headline rendered as plain markdown above
                the diff. The user reads this first.
            before: Pre-change content (multi-line strings preferred).
            after: Post-change content.
            lang: Language hint for syntax highlighting on each side
                (``"yaml"``, ``"python"``, ``"text"``).
            label: Optional caption (e.g., file path the diff is over).

        Returns:
            Dict with message_id of the posted message.
        """
        attachment: dict[str, Any] = {
            "kind": "diff",
            "before": before,
            "after": after,
            "lang": lang,
        }
        if label:
            attachment["label"] = label
        return await self.respond_to_user(content=summary, attachments=[attachment])

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

        await self._refresh_monorepo_extensions()  # in case the user just added a new mission or tool

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
                agent_system = await get_agent_system()

                # Try to find by ID first, then by type
                agent_ids = await agent_system.list_all_agents()
                target_id = None

                for aid in agent_ids:
                    if aid.startswith(name) or aid == name:
                        target_id = aid
                        break

                if not target_id:
                    # Try matching by agent_type suffix
                    from polymathera.colony.system import fetch_agent_info
                    for aid in agent_ids:
                        info = await fetch_agent_info(aid)
                        if info is None:
                            continue
                        if info.agent_type.split(".")[-1].lower() == name.lower():
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

    @event_handler(pattern=AgentDiagnosticProtocol.event_pattern())
    async def handle_agent_diagnostic(
        self, event: BlackboardEvent, _repl: Any,
    ) -> EventProcessingResult | None:
        """Translate one diagnostic event into planner context.

        The relay task ``_relay_agent_diagnostic_to_planner`` bridges
        events from the session's ``agent_diagnostic`` scope into
        this policy's event queue; this handler picks them up by
        pattern and surfaces a structured context binding the LLM
        sees on its next iteration.
        """

        try:
            parsed = AgentDiagnosticProtocol.parse_event_key(event.key)
        except ValueError:
            return PROCESSED
        producer_id = parsed["agent_id"]
        kind = parsed["kind"]
        is_self = self._agent is not None and producer_id == self._agent.agent_id
        if is_self and kind not in SELF_RELEVANT_DIAGNOSTIC_KINDS:
            return PROCESSED
        payload = event.value if isinstance(event.value, dict) else {}
        if kind == DIAGNOSTIC_GUARDRAIL_BLOCK_STREAK:
            return EventProcessingResult(
                context_key=(
                    f"agent_diagnostic:{producer_id}:{kind}:{parsed['sequence']}"
                ),
                context={
                    "producer_agent_id": producer_id,
                    "kind": kind,
                    "action_key": payload.get("action_key"),
                    "count": payload.get("count"),
                    "reason": payload.get("reason"),
                    "suggestion": payload.get("suggestion"),
                },
            )
        if kind == DIAGNOSTIC_EMPTY_ITERATION_STREAK:
            return EventProcessingResult(
                context_key=(
                    f"agent_diagnostic:{producer_id}:{kind}:{parsed['sequence']}"
                ),
                context={
                    "producer_agent_id": producer_id,
                    "kind": kind,
                    "streak": payload.get("streak"),
                    "threshold": payload.get("threshold"),
                    "suggestion": payload.get("suggestion"),
                },
            )
        return PROCESSED

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
        policy = self.agent.action_policy
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

        policy = self.agent.action_policy
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
            policy = self.agent.action_policy
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
            "has_pending_work",
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
                "has_pending_work",
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
            "roles": self._cmd_roles,
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
            "/analyze <type>": "Start a mission run (impact, compliance, intent, contracts, slicing, basic)",
            "/map <url>": "Map a repository or content to VCM",
            "/abort [run_id]": "Abort the current or specified run",
            "/status": "Show current session and run status",
            "/agents": "List active agents in this session",
            "/roles": "List the colony-generic agent roles available to spawn",
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
            from polymathera.colony.system import fetch_agent_info
            agent_system = await get_agent_system()
            agent_ids = await agent_system.list_all_agents()

            if not agent_ids:
                await self._post_response("No active agents.")
                return

            lines = [f"**Active agents:** ({len(agent_ids)})"]
            for aid in agent_ids[:20]:
                try:
                    info = await fetch_agent_info(aid)
                except Exception:
                    lines.append(f"  `{aid[:16]}` — unknown")
                    continue
                if info is None:
                    lines.append(f"  `{aid[:16]}` — unregistered")
                    continue
                # ``AgentRegistrationInfo`` fields are typed + required
                # (see models.py:2819); read directly per
                # [[no-getattr-defaults]]. ``state.name`` returns the
                # enum NAME (uppercase: RUNNING / STOPPED / ...);
                # ``agent_type`` is a dotted FQN whose tail segment
                # is the human-readable class name.
                lines.append(
                    f"  `{aid[:16]}` — "
                    f"{info.agent_type.split('.')[-1]} ({info.state.name})"
                )

            if len(agent_ids) > 20:
                lines.append(f"  ... and {len(agent_ids) - 20} more")

            await self._post_response("\n".join(lines))
        except Exception as e:
            await self._post_response(f"Failed to list agents: {e}")

    async def _cmd_roles(self, args: str, controls: dict | None) -> None:
        """List the colony-generic agent roles the user can spawn (Phase C3).

        This is a *registry view* of the framework's known agent
        classes (master §3.5), distinct from ``/agents`` which lists
        the agents currently *running* in the session. The registry
        is filled at import time from
        ``polymathera.colony.agents.roles``; CPS-shared roles add
        themselves the same way in their own package.
        """

        try:
            from polymathera.colony.agents.roles import (
                DataCurationAgent,
                KnowledgeCuratorAgent,
            )
        except ImportError as exc:
            await self._post_response(
                f"Cannot load colony agent roles: {exc}",
            )
            return

        roles = (
            ("KnowledgeCuratorAgent",
             KnowledgeCuratorAgent.agent_type,
             "Source ingestion, KG maintenance, sampled review queue."),
            ("DataCurationAgent",
             DataCurationAgent.agent_type,
             "Dataset registration, content-hash versioning, lineage."),
        )
        lines = ["**Generic colony agent roles:**"]
        for short, agent_type, summary in roles:
            lines.append(f"  • **{short}** — {summary}")
            lines.append(f"    `{agent_type}`")
        await self._post_response("\n".join(lines))

    async def _cmd_abort(self, args: str, controls: dict | None) -> None:
        """Abort a run."""
        # TODO: Implement run cancellation via InterruptionProtocol
        await self._post_response("Abort not yet implemented. Coming in a future update.")

    async def _cmd_analyze(self, args: str, controls: dict | None) -> None:
        """Start a mission run by spawning coordinator agents."""
        # TODO: Parse mission type and parameters, spawn coordinators
        # via AgentPoolCapability (same flow as jobs.py _run_job).
        if not args.strip():
            await self._post_response(
                "Usage: `/analyze <type> [--max-agents N]`\n"
                "Types: impact, compliance, intent, contracts, slicing, basic"
            )
            return

        mission_type = args.split()[0]
        valid_types = {"impact", "compliance", "intent", "contracts", "slicing", "basic"}
        if mission_type not in valid_types:
            await self._post_response(f"Unknown mission type: {mission_type}. Valid: {', '.join(sorted(valid_types))}")
            return

        await self._post_response(
            f"Starting **{mission_type}** mission...\n"
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
