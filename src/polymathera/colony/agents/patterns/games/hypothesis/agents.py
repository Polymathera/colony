"""Hypothesis Game for hallucination control.

Game structure (Extensive Form):
- Roles: Proposer, Skeptic(s), Grounder(s), Arbiter
- Phases: PROPOSE → CHALLENGE → GROUND → DEFEND → ARBITRATE → TERMINAL
- Purpose: Combat hallucination through structured challenge and evidence requirements

Game participation is handled dynamically via ``DynamicGameCapability`` —
agents no longer need to extend game-specific base classes.  A coordinator
writes a ``GameInvitation`` to the colony blackboard, and any agent with
``DynamicGameCapability`` that is listed as a participant auto-joins.
"""

from __future__ import annotations

import uuid
from pydantic import BaseModel, Field
from logging import getLogger

from ....base import Agent, CapabilityResultFuture
from ..state import GameInvitation, GameOutcome
from ...models import Hypothesis
from ...actions import action_executor
from ..dynamic import DynamicGameCapability
from .capabilities import HypothesisGameProtocol

logger = getLogger(__name__)


# ============================================================================
# Game Configuration
# ============================================================================


class HypothesisGameConfig(BaseModel):
    """Configuration for a hypothesis game instance.

    Used as structured input for ``GameInvitation.game_config``.
    """

    game_id: str = Field(description="Unique game identifier")
    hypothesis: Hypothesis = Field(description="The hypothesis being validated")
    use_llm_reasoning: bool = Field(
        default=False,
        description="Whether to use LLM for decision making"
    )
    participants: dict[str, str] = Field(
        default_factory=dict,
        description="Agent ID -> role mapping for all participants"
    )


# ============================================================================
# Public API Functions
# ============================================================================


@action_executor(writes=["game_future"])
async def run_hypothesis_game(
    *,
    owner: Agent,
    hypothesis: Hypothesis,
    proposer_id: str | None = None,
    skeptic_ids: list[str] | None = None,
    grounder_ids: list[str] | None = None,
    arbiter_id: str | None = None,
    use_llm_reasoning: bool = False,
    game_id: str | None = None,
    num_skeptics: int = 2,
    num_grounders: int = 1,
) -> CapabilityResultFuture:
    """Run a complete hypothesis validation game via DynamicGameCapability.

    Writes a ``GameInvitation`` to the colony blackboard.  All participant
    agents must have ``DynamicGameCapability`` — they will auto-join and
    receive ``HypothesisGameProtocol`` at runtime.

    Args:
        owner: Parent agent (must have DynamicGameCapability)
        hypothesis: Hypothesis to validate
        proposer_id: Optional proposer agent ID
        skeptic_ids: Optional list of skeptic agent IDs
        grounder_ids: Optional list of grounder agent IDs
        arbiter_id: Optional arbiter agent ID
        use_llm_reasoning: Whether to use LLM for decision making
        game_id: Optional game ID (generated if not provided)
        num_skeptics: Number of skeptics if skeptic_ids not provided
        num_grounders: Number of grounders if grounder_ids not provided

    Returns:
        CapabilityResultFuture handle for monitoring/awaiting completion
    """
    game_id = game_id or f"hypothesis_game_{uuid.uuid4().hex[:8]}"

    # Build participants mapping
    participants: dict[str, str] = {}

    proposer_id = proposer_id or f"{game_id}_proposer"
    participants[proposer_id] = "proposer"

    if skeptic_ids is None:
        skeptic_ids = [f"{game_id}_skeptic_{i}" for i in range(num_skeptics)]
    for sid in skeptic_ids:
        participants[sid] = "skeptic"

    if grounder_ids is None:
        grounder_ids = [f"{game_id}_grounder_{i}" for i in range(num_grounders)]
    for gid in grounder_ids:
        participants[gid] = "grounder"

    arbiter_id = arbiter_id or f"{game_id}_arbiter"
    participants[arbiter_id] = "arbiter"

    # Write game invitation via DynamicGameCapability
    dynamic_cap = owner.get_capability_by_type(DynamicGameCapability)
    if dynamic_cap is None:
        raise RuntimeError(
            "Owner agent must have DynamicGameCapability to create games. "
            "Add it via agent.add_capability_blueprints([DynamicGameCapability.bind()])"
        )

    await dynamic_cap.create_game(
        game_type="hypothesis_game",
        participants=participants,
        game_config={"use_llm_reasoning": use_llm_reasoning},
        initial_data={"hypothesis": hypothesis.model_dump()},
        game_id=game_id,
    )

    # Create result handle using the same game_id scope
    game_protocol = HypothesisGameProtocol(agent=owner, game_id=game_id)
    await game_protocol.initialize()
    result = await game_protocol.get_result_future()

    logger.info(
        "Hypothesis game %s created with %d skeptics, %d grounders",
        game_id, len(skeptic_ids), len(grounder_ids),
    )

    return result


async def ground_hypothesis(
    hypothesis: Hypothesis,
    agent: Agent,
    use_llm: bool = False,
) -> tuple[bool, Hypothesis, GameOutcome]:
    """Validate hypothesis using hypothesis game (legacy API).

    For new code, use ``run_hypothesis_game`` instead.
    """
    result = await run_hypothesis_game(
        owner=agent,
        hypothesis=hypothesis,
        use_llm_reasoning=use_llm,
    )

    outcome = await result.wait(timeout=120.0)

    if outcome is None:
        return (False, hypothesis, GameOutcome(
            outcome_type="timeout",
            success=False,
            participants=[],
            rounds_played=0,
            messages_exchanged=0,
        ))

    final_hypothesis = hypothesis
    if outcome.result and isinstance(outcome.result, dict):
        try:
            final_hypothesis = Hypothesis(**outcome.result)
        except Exception:
            pass

    return (outcome.success, final_hypothesis, outcome)
