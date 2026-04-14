"""Negotiation Game — structured multi-party negotiation protocol.

Game participation is handled dynamically via ``DynamicGameCapability`` —
agents no longer need to extend game-specific base classes.  A coordinator
writes a ``GameInvitation`` to the colony blackboard, and any agent with
``DynamicGameCapability`` that is listed as a participant auto-joins.
"""

from __future__ import annotations

import logging
import uuid
from pydantic import BaseModel, Field

from ..state import GameInvitation, GameOutcome
from ....base import Agent, CapabilityResultFuture
from ...actions import action_executor
from ..dynamic import DynamicGameCapability
from .capabilities import (
    NegotiationIssue,
    NegotiationStrategy,
    NegotiationGameProtocol,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Game Configuration
# ============================================================================


class NegotiationGameConfig(BaseModel):
    """Configuration for a negotiation game instance.

    Used as structured input for ``GameInvitation.game_config``.
    """

    game_id: str = Field(description="Unique game identifier")
    issue: NegotiationIssue = Field(description="The negotiation issue")
    strategy: NegotiationStrategy = Field(
        default=NegotiationStrategy.COMPROMISING,
        description="Negotiation strategy to use"
    )
    min_acceptable_utility: float = Field(
        default=0.3,
        description="Minimum utility threshold (BATNA)"
    )
    use_llm_reasoning: bool = Field(
        default=False,
        description="Whether to use LLM for decision making"
    )
    llm_temperature: float = Field(
        default=0.7,
        description="Temperature setting for LLM responses"
    )
    llm_max_tokens: int = Field(
        default=1000,
        description="Max tokens for LLM responses"
    )
    participants: dict[str, str] = Field(
        default_factory=dict,
        description="Agent ID -> role mapping for all participants"
    )


# ============================================================================
# Public API Functions
# ============================================================================


@action_executor(writes=["game_future"])
async def run_negotiation_game(
    *,
    owner: Agent,
    issue: NegotiationIssue,
    coordinator_id: str | None = None,
    participant_ids: list[str] | None = None,
    mediator_id: str | None = None,
    strategies: dict[str, NegotiationStrategy] | None = None,
    use_llm_reasoning: bool = False,
    game_id: str | None = None,
    num_participants: int = 2,
) -> CapabilityResultFuture:
    """Run a complete negotiation game via DynamicGameCapability.

    Writes a ``GameInvitation`` to the colony blackboard.  All participant
    agents must have ``DynamicGameCapability`` — they will auto-join and
    receive ``NegotiationGameProtocol`` at runtime.

    Args:
        owner: Parent agent (must have DynamicGameCapability)
        issue: The negotiation issue to resolve
        coordinator_id: Optional coordinator agent ID
        participant_ids: Optional list of participant agent IDs
        mediator_id: Optional mediator agent ID
        strategies: Optional per-agent strategy mapping
        use_llm_reasoning: Whether to use LLM for decision making
        game_id: Optional game ID (generated if not provided)
        num_participants: Number of participants if not provided

    Returns:
        CapabilityResultFuture handle for monitoring/awaiting completion
    """
    game_id = game_id or f"negotiation_game_{uuid.uuid4().hex[:8]}"

    # Build participants mapping
    participants: dict[str, str] = {}
    coordinator_id = coordinator_id or f"{game_id}_coordinator"
    participants[coordinator_id] = "coordinator"

    if participant_ids is None:
        participant_ids = [f"{game_id}_participant_{i}" for i in range(num_participants)]
    for pid in participant_ids:
        participants[pid] = "participant"

    if mediator_id:
        participants[mediator_id] = "mediator"

    # Write game invitation via DynamicGameCapability
    dynamic_cap = owner.get_capability_by_type(DynamicGameCapability)
    if dynamic_cap is None:
        raise RuntimeError(
            "Owner agent must have DynamicGameCapability to create games."
        )

    default_strategy = (strategies or {}).get(
        coordinator_id, NegotiationStrategy.COMPROMISING
    )
    await dynamic_cap.create_game(
        game_type="negotiation",
        participants=participants,
        game_config={
            "strategy": default_strategy.value,
            "use_llm_reasoning": use_llm_reasoning,
        },
        initial_data={"issue": issue.model_dump()},
        game_id=game_id,
    )

    # Create result handle
    game_protocol = NegotiationGameProtocol(agent=owner, game_id=game_id)
    await game_protocol.initialize()
    result = await game_protocol.get_result_future()

    logger.info(
        "Negotiation game %s created with %d participants%s",
        game_id, len(participant_ids),
        f" and mediator {mediator_id}" if mediator_id else "",
    )

    return result
