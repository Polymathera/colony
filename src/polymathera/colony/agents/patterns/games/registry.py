"""Game protocol registry — maps game_type strings to protocol classes.

Used by ``DynamicGameCapability`` to instantiate the correct
``GameProtocolCapability`` subclass when an agent joins a game.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...base import Agent
    from .state import GameProtocolCapability

logger = logging.getLogger(__name__)


class GameProtocolRegistry:
    """Singleton registry mapping ``game_type`` → ``GameProtocolCapability`` subclass.

    Default game types are lazily registered on first access to avoid
    import cycles.

    Usage::

        registry = GameProtocolRegistry.instance()
        registry.register("my_custom_game", MyCustomGameProtocol)

        cap = registry.create_protocol(
            game_type="hypothesis_game",
            agent=agent,
            game_id="game_001",
            role="proposer",
        )
    """

    _instance: GameProtocolRegistry | None = None

    def __init__(self) -> None:
        self._registry: dict[str, type[GameProtocolCapability]] = {}
        self._defaults_registered = False

    @classmethod
    def instance(cls) -> GameProtocolRegistry:
        """Get or create the singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_defaults(self) -> None:
        """Lazily register built-in game types on first use."""
        if self._defaults_registered:
            return
        self._defaults_registered = True

        from .hypothesis.capabilities import HypothesisGameProtocol
        from .negotiation.capabilities import NegotiationGameProtocol
        from .consensus_game import ConsensusGameProtocol
        from .contract_net import ContractNetGameCapability
        from .coalition_formation import CoalitionFormationProtocol

        self.register("hypothesis_game", HypothesisGameProtocol)
        self.register("negotiation", NegotiationGameProtocol)
        self.register("consensus_game", ConsensusGameProtocol)
        self.register("contract_net", ContractNetGameCapability)
        self.register("coalition_formation", CoalitionFormationProtocol)

    def register(
        self, game_type: str, protocol_cls: type[GameProtocolCapability]
    ) -> None:
        """Register a game type → protocol class mapping."""
        self._registry[game_type] = protocol_cls

    def get(self, game_type: str) -> type[GameProtocolCapability] | None:
        """Look up the protocol class for a game type."""
        self._ensure_defaults()
        return self._registry.get(game_type)

    def registered_types(self) -> list[str]:
        """Return all registered game type names."""
        self._ensure_defaults()
        return list(self._registry.keys())

    def create_protocol(
        self,
        game_type: str,
        agent: Agent,
        game_id: str,
        role: str | None = None,
        capability_key: str | None = None,
        **kwargs: Any,
    ) -> GameProtocolCapability:
        """Create a protocol instance from the registry.

        Args:
            game_type: Registered game type string.
            agent: Owning agent.
            game_id: Game instance identifier.
            role: Agent's role in the game.
            capability_key: Override the default capability key.
                Defaults to ``"{game_type}:{game_id}"`` to support
                concurrent participation in multiple games.
            **kwargs: Additional constructor kwargs forwarded to the
                protocol subclass (e.g., ``strategy``, ``voting_method``).

        Returns:
            An initialized (but not yet ``initialize()``-d) protocol instance.

        Raises:
            ValueError: If ``game_type`` is not registered.
        """
        self._ensure_defaults()
        protocol_cls = self._registry.get(game_type)
        if protocol_cls is None:
            raise ValueError(
                f"Unknown game type: {game_type!r}. "
                f"Registered: {list(self._registry.keys())}"
            )

        cap_key = capability_key or f"{game_type}:{game_id}"
        return protocol_cls(
            agent=agent,
            game_id=game_id,
            role=role,
            capability_key=cap_key,
            **kwargs,
        )
