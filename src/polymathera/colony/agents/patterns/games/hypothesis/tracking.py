"""Hypothesis tracking capability for managing hypotheses across games.

This module provides HypothesisTrackingCapability which:
- Tracks hypotheses and their validation status
- Manages hypothesis-game associations
- Delegates contradiction detection to ValidationCapability
- Uses scope_id for flexible sharing patterns
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING
from overrides import override
from pydantic import BaseModel, Field

from ....base import AgentCapability
from ....models import AgentSuspensionState
from ...actions.policies import action_executor
from .types import (
    HypothesisFilter,
    HypothesisStatus,
    HypothesisDomain,
)
from .....utils import setup_logger

if TYPE_CHECKING:
    from ....base import Agent
    from ...models import Hypothesis
    from ..state import GameOutcome
    from ...capabilities.validation import ValidationCapability, Contradiction


logger = setup_logger(__name__)


class TrackedHypothesis(BaseModel):
    """A hypothesis with tracking metadata."""

    hypothesis: Hypothesis = Field(description="The hypothesis")
    status: HypothesisStatus = Field(default=HypothesisStatus.PENDING)
    game_id: str | None = Field(default=None, description="Associated game")
    domain: HypothesisDomain | None = Field(default=None)
    registered_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    outcome_summary: str | None = Field(default=None)


class HypothesisTrackingCapability(AgentCapability):
    """Capability for tracking hypotheses across games.

    Uses scope_id for flexible sharing:
    - scope_id = agent_id: Per-agent tracking (default)
    - scope_id = coordinator_id: Per-coordinator (shared across games)
    - scope_id = "global": Cross-agent tracking
    - scope_id = tenant_id: Multi-tenant isolation

    Delegates contradiction detection to ValidationCapability.
    Stores hypothesis data on blackboard for persistence and sharing.
    """

    def __init__(
        self,
        agent: Agent,
        scope_id: str | None = None,
    ):
        """Initialize tracking capability.

        Args:
            agent: Owning agent
            scope_id: Scope for sharing (defaults to agent.agent_id)
        """
        super().__init__(agent=agent, scope_id=scope_id or agent.agent_id)

        # Local cache (authoritative data is on blackboard)
        self._cache: dict[str, TrackedHypothesis] = {}
        self._game_mapping: dict[str, list[str]] = {}  # game_id -> hypothesis_ids

    def get_action_group_description(self) -> str:
        return (
            "Hypothesis Tracking — tracks hypothesis lifecycle across games. "
            "Statuses: PENDING → SUPPORTED/REFUTED/UNCERTAIN. Confidence adjusts with outcomes. "
            "Delegates contradiction detection to ValidationCapability. "
            "should_retry_hypothesis returns true for UNCERTAIN (confidence>0.4) or REFUTED (confidence<0.3). "
            "Blackboard-backed for persistence and cross-agent sharing via scope_id."
        )

    def _get_validation_capability(self) -> ValidationCapability | None:
        """Get ValidationCapability for contradiction detection."""
        from ...capabilities.validation import ValidationCapability
        return self.agent.get_capability_by_type(ValidationCapability)

    def _get_hypotheses_key(self) -> str:
        """Get blackboard key for tracked hypotheses."""
        return f"{self.scope_id}:tracked_hypotheses"

    def _get_games_key(self) -> str:
        """Get blackboard key for game mappings."""
        return f"{self.scope_id}:hypothesis_games"

    async def _load_from_blackboard(self) -> None:
        """Load cached data from blackboard."""
        blackboard = await self.get_blackboard()

        # Load hypotheses
        hypotheses_data = await blackboard.read(self._get_hypotheses_key())
        if hypotheses_data:
            for hyp_id, tracked_data in hypotheses_data.items():
                self._cache[hyp_id] = TrackedHypothesis(**tracked_data)

        # Load game mappings
        games_data = await blackboard.read(self._get_games_key())
        if games_data:
            self._game_mapping = games_data

    async def _save_to_blackboard(self) -> None:
        """Save cached data to blackboard."""
        blackboard = await self.get_blackboard()

        # Save hypotheses
        hypotheses_data = {
            hyp_id: tracked.model_dump()
            for hyp_id, tracked in self._cache.items()
        }
        await blackboard.write(
            self._get_hypotheses_key(),
            hypotheses_data,
            created_by=self.agent.agent_id,
        )

        # Save game mappings
        await blackboard.write(
            self._get_games_key(),
            self._game_mapping,
            created_by=self.agent.agent_id,
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for HypothesisTrackingCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for HypothesisTrackingCapability")
        pass

    @action_executor()
    async def register_hypothesis(
        self,
        hypothesis: Hypothesis,
        game_id: str | None = None,
        domain: HypothesisDomain | None = None,
    ) -> dict[str, Any]:
        """Register a hypothesis for tracking.

        Args:
            hypothesis: Hypothesis to track
            game_id: Associated game ID (if in a game)
            domain: Hypothesis domain

        Returns:
            Registration result with hypothesis_id
        """
        await self._load_from_blackboard()

        tracked = TrackedHypothesis(
            hypothesis=hypothesis,
            status=HypothesisStatus.PENDING,
            game_id=game_id,
            domain=domain,
        )

        self._cache[hypothesis.hypothesis_id] = tracked

        # Update game mapping
        if game_id:
            if game_id not in self._game_mapping:
                self._game_mapping[game_id] = []
            if hypothesis.hypothesis_id not in self._game_mapping[game_id]:
                self._game_mapping[game_id].append(hypothesis.hypothesis_id)

        await self._save_to_blackboard()

        logger.info(
            f"Registered hypothesis {hypothesis.hypothesis_id} "
            f"(game={game_id}, domain={domain})"
        )

        return {
            "registered": True,
            "hypothesis_id": hypothesis.hypothesis_id,
            "game_id": game_id,
        }

    @action_executor()
    async def update_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus | str,
        outcome: GameOutcome | None = None,
    ) -> dict[str, Any]:
        """Update hypothesis status.

        Args:
            hypothesis_id: Hypothesis to update
            status: New status
            outcome: Game outcome (if from game)

        Returns:
            Update result
        """
        await self._load_from_blackboard()

        if hypothesis_id not in self._cache:
            return {
                "updated": False,
                "hypothesis_id": hypothesis_id,
                "error": "hypothesis_not_found",
            }

        if isinstance(status, str):
            status = HypothesisStatus(status)

        tracked = self._cache[hypothesis_id]
        tracked.status = status
        tracked.updated_at = time.time()

        if outcome:
            tracked.outcome_summary = outcome.summary
            # Update hypothesis confidence from outcome
            if hasattr(tracked.hypothesis, "confidence"):
                if outcome.success:
                    tracked.hypothesis.confidence = max(tracked.hypothesis.confidence, 0.8)
                else:
                    tracked.hypothesis.confidence = min(tracked.hypothesis.confidence, 0.3)

        await self._save_to_blackboard()

        logger.info(f"Updated hypothesis {hypothesis_id} status to {status.value}")

        return {
            "updated": True,
            "hypothesis_id": hypothesis_id,
            "status": status.value,
        }

    @action_executor()
    async def get_hypotheses(
        self,
        filter: HypothesisFilter | None = None,
    ) -> list[Hypothesis]:
        """Get tracked hypotheses with optional filter.

        Args:
            filter: Optional filter criteria

        Returns:
            List of matching hypotheses
        """
        await self._load_from_blackboard()

        results = []

        for tracked in self._cache.values():
            # Apply filters
            if filter:
                if filter.status and tracked.status != filter.status:
                    continue
                if filter.game_id and tracked.game_id != filter.game_id:
                    continue
                if filter.domain and tracked.domain != filter.domain:
                    continue
                if filter.min_confidence and tracked.hypothesis.confidence < filter.min_confidence:
                    continue

            results.append(tracked.hypothesis)

        return results

    @action_executor()
    async def get_supported_hypotheses(self) -> list[Hypothesis]:
        """Get all supported hypotheses."""
        return await self.get_hypotheses(
            HypothesisFilter(status=HypothesisStatus.SUPPORTED)
        )

    @action_executor()
    async def get_refuted_hypotheses(self) -> list[Hypothesis]:
        """Get all refuted hypotheses."""
        return await self.get_hypotheses(
            HypothesisFilter(status=HypothesisStatus.REFUTED)
        )

    @action_executor()
    async def get_uncertain_hypotheses(self) -> list[Hypothesis]:
        """Get all uncertain hypotheses."""
        return await self.get_hypotheses(
            HypothesisFilter(status=HypothesisStatus.UNCERTAIN)
        )

    @action_executor()
    async def get_active_games(self) -> list[str]:
        """Get IDs of games with tracked hypotheses."""
        await self._load_from_blackboard()
        return list(self._game_mapping.keys())

    @action_executor()
    async def get_hypotheses_for_game(
        self,
        game_id: str,
    ) -> list[Hypothesis]:
        """Get all hypotheses associated with a game.

        Args:
            game_id: Game ID

        Returns:
            List of hypotheses in that game
        """
        return await self.get_hypotheses(
            HypothesisFilter(game_id=game_id)
        )

    @action_executor()
    async def detect_contradictions(
        self,
        hypothesis_ids: list[str] | None = None,
    ) -> list[Contradiction]:
        """Detect contradictions between hypotheses.

        Delegates to ValidationCapability.detect_contradictions().

        Args:
            hypothesis_ids: Specific hypotheses (None = all tracked)

        Returns:
            List of contradictions found
        """
        validation_cap = self._get_validation_capability()
        if not validation_cap:
            logger.warning(
                "ValidationCapability not available for contradiction detection"
            )
            return []

        await self._load_from_blackboard()

        # Get hypotheses to check
        if hypothesis_ids:
            hypotheses = [
                self._cache[hid].hypothesis
                for hid in hypothesis_ids
                if hid in self._cache
            ]
        else:
            hypotheses = [tracked.hypothesis for tracked in self._cache.values()]

        if len(hypotheses) < 2:
            return []  # Need at least 2 for contradiction

        # Convert to ScopeAwareResult for ValidationCapability
        from ...scope import ScopeAwareResult, AnalysisScope

        results = []
        for hyp in hypotheses:
            result = ScopeAwareResult(
                result_id=hyp.hypothesis_id,
                content={"hypothesis": hyp.statement},
                scope=AnalysisScope(
                    is_complete=hyp.status in ("supported", "refuted"),
                    confidence=hyp.confidence,
                    evidence=hyp.supporting_evidence,
                ),
            )
            results.append(result)

        return await validation_cap.detect_contradictions(results)

    @action_executor()
    async def find_related(
        self,
        hypothesis: Hypothesis,
        max_results: int = 5,
    ) -> list[Hypothesis]:
        """Find hypotheses related to the given one.

        Uses simple heuristics: same game, similar statements.
        Could be enhanced with LLM-based similarity.

        Args:
            hypothesis: Reference hypothesis
            max_results: Maximum results to return

        Returns:
            Related hypotheses
        """
        await self._load_from_blackboard()

        related = []

        # Find hypotheses in same game
        for tracked in self._cache.values():
            if tracked.hypothesis.hypothesis_id == hypothesis.hypothesis_id:
                continue

            # Same game is strongly related
            source_tracked = self._cache.get(hypothesis.hypothesis_id)
            if source_tracked and tracked.game_id == source_tracked.game_id:
                related.append((tracked.hypothesis, 1.0))
                continue

            # Simple keyword overlap for relatedness
            source_words = set(hypothesis.statement.lower().split())
            target_words = set(tracked.hypothesis.statement.lower().split())
            overlap = len(source_words & target_words)
            if overlap > 2:
                similarity = overlap / max(len(source_words), len(target_words))
                related.append((tracked.hypothesis, similarity))

        # Sort by relatedness and return top results
        related.sort(key=lambda x: x[1], reverse=True)
        return [hyp for hyp, _ in related[:max_results]]

    @action_executor()
    async def should_retry_hypothesis(
        self,
        hypothesis_id: str,
    ) -> bool:
        """Determine if a hypothesis should be retried.

        Args:
            hypothesis_id: Hypothesis to check

        Returns:
            True if should retry
        """
        await self._load_from_blackboard()

        if hypothesis_id not in self._cache:
            return False

        tracked = self._cache[hypothesis_id]

        # Retry if uncertain and confidence is moderate
        if tracked.status == HypothesisStatus.UNCERTAIN:
            return tracked.hypothesis.confidence > 0.4

        # Retry if refuted but confidence is low (might be wrong)
        if tracked.status == HypothesisStatus.REFUTED:
            return tracked.hypothesis.confidence < 0.3

        return False

    async def cleanup_completed_game(
        self,
        game_id: str,
        remove_hypotheses: bool = False,
    ) -> None:
        """Clean up tracking data for a completed game.

        Args:
            game_id: Game that completed
            remove_hypotheses: Whether to remove hypotheses (default: keep)
        """
        await self._load_from_blackboard()

        if game_id in self._game_mapping:
            if remove_hypotheses:
                for hyp_id in self._game_mapping[game_id]:
                    self._cache.pop(hyp_id, None)
            del self._game_mapping[game_id]

        await self._save_to_blackboard()
