"""Dynamic game participation capability.

Enables any agent to join and leave games at runtime via blackboard
invitation events, replacing the rigid pattern of game-specific agent
classes (``HypothesisGameAgent``, etc.).

Usage::

    # In any agent's initialization or capability blueprints:
    agent.add_capability_blueprints([DynamicGameCapability.bind()])

    # A coordinator creates a game by writing an invitation:
    await dynamic_cap.create_game(
        game_type="hypothesis_game",
        participants={"agent-1": "proposer", "agent-2": "skeptic"},
        game_config={"use_llm_reasoning": True},
        initial_data={"hypothesis": {...}},
    )

    # Each invited agent's DynamicGameCapability auto-joins, creating the
    # appropriate GameProtocolCapability and adding it to the agent.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ...base import AgentCapability
from ...blackboard.protocol import GameInvitationProtocol
from ...blackboard.types import BlackboardEvent
from ...scopes import BlackboardScope, get_scope_prefix
from ..capabilities.agent_pool import AgentPoolCapability
from ..actions.dispatcher import action_executor
from ..events import EventProcessingResult, event_handler
from .registry import GameProtocolRegistry
from .state import GameInvitation

logger = logging.getLogger(__name__)


class DynamicGameCapability(AgentCapability):
    """Enables runtime game participation via blackboard invitation events.

    Listens at colony scope for ``GameInvitationProtocol`` events.
    When the owning agent is listed in an invitation's ``participants``,
    creates the appropriate ``GameProtocolCapability`` subclass, initializes
    it, and adds it to the agent.  When a game reaches terminal state,
    the corresponding capability is automatically cleaned up.

    Supports concurrent participation in multiple games — each gets its
    own ``GameProtocolCapability`` instance with a unique ``capability_key``
    of the form ``"{game_type}:{game_id}"``.
    """

    def __init__(
        self,
        agent: Any,
        *,
        registry: GameProtocolRegistry | None = None,
        auto_accept: bool = True,
        default_game_config: dict[str, dict[str, Any]] | None = None,
        capability_key: str = "dynamic_game",
    ):
        """Initialize DynamicGameCapability.

        Args:
            agent: Owning agent.
            registry: Game protocol registry. Uses the singleton if None.
            auto_accept: If True, automatically join when invited.
                If False, invitations are surfaced as context for the
                LLM planner to decide.
            default_game_config: Per-game-type default constructor kwargs.
                Merged (invitation overrides) before protocol creation.
                Example: ``{"negotiation": {"strategy": "hardball"}}``
            capability_key: Unique key for this capability.
        """
        scope_id = get_scope_prefix(
            BlackboardScope.COLONY, agent, namespace="game_invitations",
        )
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=[GameInvitationProtocol.invitation_pattern()],
            capability_key=capability_key,
        )
        self._registry = registry or GameProtocolRegistry.instance()
        self._auto_accept = auto_accept
        self._default_game_config = default_game_config or {}
        # game_id → capability_key of the installed GameProtocolCapability
        self._active_games: dict[str, str] = {}

    def get_action_group_description(self) -> str:
        active = len(self._active_games)
        return (
            f"Dynamic Game Participation — join/leave multi-agent games at "
            f"runtime ({active} active).\n\n"
            "Listens for game invitation events on the colony blackboard. "
            "When invited, creates the appropriate game protocol capability "
            "(hypothesis, negotiation, consensus, contract-net, coalition) "
            "and adds it to this agent. The protocol's action executors then "
            "appear in your available actions for the duration of the game.\n\n"
            "Use ``create_game`` to initiate a new game and invite other agents. "
            "Games are automatically cleaned up when they reach terminal state."
        )

    # ------------------------------------------------------------------
    # Event handler — process game invitations
    # ------------------------------------------------------------------

    @event_handler(pattern=GameInvitationProtocol.invitation_pattern())
    async def handle_invitation(
        self,
        event: BlackboardEvent,
        repl: Any,
    ) -> EventProcessingResult | None:
        """Process a game invitation event.

        If this agent is listed in the invitation's participants and
        ``auto_accept`` is True, joins the game immediately.
        """
        try:
            invitation = GameInvitation.model_validate(event.value)
        except Exception:
            logger.debug("Ignoring malformed game invitation: %s", event.key)
            return None

        agent_id = self.agent.agent_id
        if agent_id not in invitation.participants:
            return None  # Not for this agent

        if invitation.game_id in self._active_games:
            return None  # Already participating

        role = invitation.participants[agent_id]

        if self._auto_accept:
            await self._join_game(invitation, role)
            return EventProcessingResult(
                context_key="game_joined",
                context={
                    "game_id": invitation.game_id,
                    "game_type": invitation.game_type,
                    "role": role,
                    "participants": invitation.participants,
                },
            )

        # Surface as pending decision for the LLM planner
        return EventProcessingResult(
            context_key="game_invitation_pending",
            context=invitation.model_dump(),
        )

    # ------------------------------------------------------------------
    # Internal — join/leave logic
    # ------------------------------------------------------------------

    async def _join_game(self, invitation: GameInvitation, role: str) -> None:
        """Create and attach a GameProtocolCapability for this game."""
        # Merge defaults with invitation-specific config
        config = {
            **self._default_game_config.get(invitation.game_type, {}),
            **invitation.game_config,
        }

        cap_key = f"{invitation.game_type}:{invitation.game_id}"

        capability = self._registry.create_protocol(
            game_type=invitation.game_type,
            agent=self.agent,
            game_id=invitation.game_id,
            role=role,
            capability_key=cap_key,
            **config,
        )
        await capability.initialize()

        # Register terminal cleanup callback
        capability._on_terminal_callbacks.append(self._cleanup_game)

        # Add to agent and refresh action policy
        self.agent.add_capability(capability)
        if self.agent.action_policy:
            self.agent.action_policy.use_agent_capabilities([cap_key])

        self._active_games[invitation.game_id] = cap_key

        logger.info(
            "Agent %s joined game %s as %s (type=%s, cap_key=%s)",
            self.agent.agent_id, invitation.game_id, role,
            invitation.game_type, cap_key,
        )

    async def _cleanup_game(self, game_id: str) -> None:
        """Remove the game protocol capability after terminal state."""
        cap_key = self._active_games.pop(game_id, None)
        if not cap_key:
            return

        self.agent.remove_capability(cap_key)
        if self.agent.action_policy:
            self.agent.action_policy.disable_agent_capabilities([cap_key])

        logger.info(
            "Agent %s left game %s (capability %s removed)",
            self.agent.agent_id, game_id, cap_key,
        )

    def _get_agent_pool(self) -> AgentPoolCapability | None:
        """Get AgentPoolCapability for spawning participants."""
        return self.agent.get_capability_by_type(AgentPoolCapability)

    # ------------------------------------------------------------------
    # Action executors — exposed to the LLM planner
    # ------------------------------------------------------------------

    @action_executor()
    async def spawn_game_participants(
        self,
        roles: dict[str, int],
        role_capabilities: dict[str, list[str]] | None = None,
        base_capabilities: list[str] | None = None,
        agent_type: str = "polymathera.colony.agents.base.Agent",
    ) -> dict[str, Any]:
        """Spawn agents to participate in a game.

        Creates generic agents equipped with ``DynamicGameCapability``
        (plus any additional capabilities) so they are ready to receive
        a game invitation.  Call this before ``create_game``.

        Args:
            roles: Role name → number of agents to create.
                Example: ``{"proposer": 1, "skeptic": 2, "arbiter": 1}``
            role_capabilities: Per-role additional capability FQNs.
                Example: ``{"skeptic": ["polymathera...CriticCapability"]}``
                These are added alongside ``DynamicGameCapability``.
            base_capabilities: Capability FQNs added to ALL agents
                regardless of role.
            agent_type: Fully qualified agent class path
                (default: generic ``Agent``).

        Returns:
            Dict with:
            - agents: role → list of created agent IDs
            - total: total agents spawned
            - failed: list of (role, error) for any spawn failures
        """
        from ..capabilities.agent_pool import AgentPoolCapability

        pool = self._get_agent_pool()
        if pool is None:
            return {"agents": {}, "total": 0, "failed": [],
                    "error": "AgentPoolCapability required for spawning"}

        my_fqn = f"{DynamicGameCapability.__module__}.{DynamicGameCapability.__qualname__}"
        role_capabilities = role_capabilities or {}
        base_capabilities = base_capabilities or []

        result_agents: dict[str, list[str]] = {}
        failed: list[dict[str, str]] = []

        for role, count in roles.items():
            result_agents[role] = []
            # Combine: DynamicGameCapability + base + role-specific
            caps = list(dict.fromkeys(
                [my_fqn] + base_capabilities + role_capabilities.get(role, [])
            ))

            for _ in range(count):
                spawn_result = await pool.create_agent(
                    agent_type=agent_type,
                    capabilities=caps,
                    label=role,
                )
                if spawn_result.get("created"):
                    result_agents[role].append(spawn_result["agent_id"])
                else:
                    failed.append({"role": role, "error": spawn_result.get("error", "unknown")})

        total = sum(len(ids) for ids in result_agents.values())
        logger.info(
            "Spawned %d game participants: %s",
            total, {r: len(ids) for r, ids in result_agents.items()},
        )
        return {"agents": result_agents, "total": total, "failed": failed}

    @action_executor()
    async def create_game(
        self,
        game_type: str,
        participants: dict[str, str],
        game_config: dict[str, Any] | None = None,
        initial_data: dict[str, Any] | None = None,
        game_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new game and broadcast the invitation.

        Writes a ``GameInvitation`` to the colony-scoped blackboard.
        All agents with ``DynamicGameCapability`` that are listed in
        ``participants`` will auto-join (if ``auto_accept=True``).

        Args:
            game_type: Registered game type (e.g., "hypothesis_game").
            participants: Agent ID → role mapping for invited agents.
                This agent should be included if it wants to participate.
            game_config: Constructor kwargs for the game protocol.
            initial_data: Initial game data (hypothesis, issue, etc.).
            game_id: Explicit game ID (auto-generated if None).

        Returns:
            Dict with ``game_id`` and ``game_type``.
        """
        game_id = game_id or f"game_{uuid.uuid4().hex[:12]}"

        invitation = GameInvitation(
            game_id=game_id,
            game_type=game_type,
            creator_agent_id=self.agent.agent_id,
            participants=participants,
            game_config=game_config or {},
            initial_data=initial_data or {},
        )

        blackboard = await self.get_blackboard()
        await blackboard.write(
            key=GameInvitationProtocol.invitation_key(game_id),
            value=invitation.model_dump(),
            created_by=self.agent.agent_id,
            tags={"game_invitation", game_type},
            metadata={"game_type": game_type, "creator": self.agent.agent_id},
        )

        logger.info(
            "Game invitation created: %s (type=%s, participants=%s)",
            game_id, game_type, list(participants.keys()),
        )
        return {"game_id": game_id, "game_type": game_type}

    @action_executor()
    async def join_game(
        self,
        game_id: str,
        game_type: str,
        role: str,
        game_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Manually join a game without an invitation event.

        Use this for ad-hoc joining or when ``auto_accept=False``.

        Args:
            game_id: Game instance identifier.
            game_type: Registered game type.
            role: Role to take in the game.
            game_config: Constructor kwargs for the game protocol.

        Returns:
            Dict with ``game_id`` and ``capability_key``.
        """
        if game_id in self._active_games:
            return {
                "game_id": game_id,
                "capability_key": self._active_games[game_id],
                "already_joined": True,
            }

        invitation = GameInvitation(
            game_id=game_id,
            game_type=game_type,
            creator_agent_id="manual",
            participants={self.agent.agent_id: role},
            game_config=game_config or {},
        )
        await self._join_game(invitation, role)
        return {
            "game_id": game_id,
            "capability_key": self._active_games[game_id],
        }

    @action_executor()
    async def leave_game(self, game_id: str) -> dict[str, Any]:
        """Leave an active game and clean up the capability.

        Args:
            game_id: Game to leave.

        Returns:
            Dict with ``left`` boolean and ``game_id``.
        """
        if game_id not in self._active_games:
            return {"left": False, "game_id": game_id, "error": "Not in this game"}

        await self._cleanup_game(game_id)
        return {"left": True, "game_id": game_id}

    @action_executor(exclude_from_planning=True)
    async def get_active_games(self) -> dict[str, Any]:
        """List currently active games for this agent.

        Returns:
            Dict with ``games`` mapping game_id → capability_key,
            and ``count``.
        """
        return {
            "games": dict(self._active_games),
            "count": len(self._active_games),
        }
