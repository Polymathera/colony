"""Incremental synthesis with feedback loops.

This module implements synthesis patterns that combine results progressively:
- IncrementalSynthesizer: Synthesizes results as they arrive with feedback
- SynthesisUpdate: Tracks synthesis progress
- Feedback propagation for refinement
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field
from overrides import override

from ..scope import ScopeAwareResult
from .merge import MergePolicy, MergeContext
from .refinement import RefinementPolicy, RefinementContext
from ...base import Agent, AgentCapability
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix
from ...models import AgentSuspensionState
from ....utils import setup_logger
from ..actions import action_executor


logger = setup_logger(__name__)

T = TypeVar('T')


class SynthesisUpdate(BaseModel):
    """Update from incremental synthesis."""

    current_synthesis: ScopeAwareResult[T] | None = Field(
        default=None,
        description="Current synthesized result"
    )

    refinements: list[tuple[str, ScopeAwareResult]] = Field(
        default_factory=list,
        description="Results that were refined: (result_id, refined_result)"
    )

    is_complete: bool = Field(
        default=False,
        description="Whether synthesis is complete"
    )

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Synthesis progress (0.0-1.0)"
    )

    timestamp: float = Field(
        default_factory=time.time,
        description="When update was generated"
    )


class IncrementalSynthesizer(Generic[T]):
    """Synthesizes results incrementally with feedback loops.

    As results arrive:
    1. Adds to partial results
    2. Checks if new result provides context for existing results
    3. Refines existing results if needed (by applying refinement policy to each partial result)
    4. Updates overall synthesis (by applying merge policy to all partial results)
    5. Detects significant changes between current and previous synthesis that trigger further refinement

    This enables progressive synthesis without waiting for all results.
    """

    def __init__(
        self,
        merge_policy: MergePolicy[T],
        refinement_policy: RefinementPolicy[T] | None = None
    ):
        """Initialize synthesizer.

        Args:
            merge_policy: Policy for merging results
            refinement_policy: Optional policy for refining results
        """
        self.merge_policy = merge_policy
        self.refinement_policy = refinement_policy
        self.partial_results: dict[str, ScopeAwareResult[T]] = {}
        self.synthesis_history: list[dict[str, Any]] = []

    async def add_result(
        self,
        result_id: str,
        result: ScopeAwareResult[T]
    ) -> SynthesisUpdate:
        """Add new result and update synthesis.

        Args:
            result_id: Unique ID for this result
            result: Result to add

        Returns:
            Synthesis update
        """
        self.partial_results[result_id] = result

        # Check if we can refine existing results with this new one
        refinements: list[tuple[str, ScopeAwareResult]] = []

        if self.refinement_policy:
            for key, existing in self.partial_results.items():
                if key == result_id:
                    continue

                # Check if new result provides context for existing
                if self._provides_context(result, existing):
                    refinement_context = RefinementContext(
                        new_evidence={"related_result": result.to_blackboard_entry()}
                    )

                    if await self.refinement_policy.should_refine(existing, refinement_context):
                        refined = await self.refinement_policy.refine(existing, refinement_context)
                        self.partial_results[key] = refined
                        refinements.append((key, refined))

        # Update overall synthesis if we have multiple results
        current_synthesis = None
        if len(self.partial_results) >= 2:
            # Merge all partial results
            current_synthesis = await self.merge_policy.merge(
                list(self.partial_results.values()),
                MergeContext()  # TODO: populate context as needed.
            )

            # Check if synthesis changed significantly
            if self.synthesis_history:
                prev_synthesis = self.synthesis_history[-1]["synthesis"]
                if self._has_significant_change(prev_synthesis, current_synthesis):
                    # Trigger feedback refinement
                    await self._trigger_refinement_feedback(current_synthesis)

            # Record synthesis
            self.synthesis_history.append({
                "timestamp": time.time(),
                "synthesis": current_synthesis,
                "sources": list(self.partial_results.keys()),
                "refinements": refinements
            })

        # Calculate progress
        complete_count = sum(1 for r in self.partial_results.values() if r.scope.is_complete)
        progress = complete_count / len(self.partial_results) if self.partial_results else 0.0

        return SynthesisUpdate(
            current_synthesis=current_synthesis,
            refinements=refinements,
            is_complete=all(r.scope.is_complete for r in self.partial_results.values()),
            progress=progress
        )

    def _provides_context(
        self,
        new_result: ScopeAwareResult[T],
        existing_result: ScopeAwareResult[T]
    ) -> bool:
        """Check if new result provides context for existing.

        Args:
            new_result: New result
            existing_result: Existing result

        Returns:
            True if new result helps existing
        """
        # Check if new result is in existing's missing context
        for missing in existing_result.scope.missing_context:
            if missing in str(new_result.content):  # FIXME - TODO: Implement better matching than string containment
                return True

        # Check if they share related shards
        # TODO: More sophisticated context overlap checks can be added.
        # But why does context overlap matter? Because if they share context,
        # then insights from one can inform the other?
        shared = set(new_result.scope.related_shards) & set(existing_result.scope.related_shards)
        return len(shared) > 0

    def _has_significant_change(
        self,
        prev_synthesis: ScopeAwareResult[T],
        current_synthesis: ScopeAwareResult[T]
    ) -> bool:
        """Check if synthesis changed significantly.

        Args:
            prev_synthesis: Previous synthesis
            current_synthesis: Current synthesis

        Returns:
            True if significant change
        """
        # TODO: Implement more sophisticated change detection, or allow policies to define this.
        # Check confidence change
        confidence_delta = abs(
            current_synthesis.scope.confidence - prev_synthesis.scope.confidence
        )
        if confidence_delta > 0.1:
            return True

        # Check completeness change
        if current_synthesis.scope.is_complete != prev_synthesis.scope.is_complete:
            return True

        return False

    async def _trigger_refinement_feedback(
        self,
        synthesis: ScopeAwareResult[T]
    ) -> None:
        """Trigger feedback refinement of earlier results based on insights from new synthesis.

        Args:
            synthesis: New synthesis that triggered feedback
        """
        # TODO: Implement feedback refinement logic
        pass

    def get_current_synthesis(self) -> ScopeAwareResult[T] | None:
        """Get current synthesis.

        Returns:
            Current synthesis or None if not yet synthesized
        """
        if self.synthesis_history:
            return self.synthesis_history[-1]["synthesis"]
        return None

    def get_synthesis_history(self) -> list[dict[str, Any]]:
        """Get full synthesis history.

        Returns:
            List of synthesis snapshots
        """
        return self.synthesis_history.copy()


class SynthesisCapability(AgentCapability):
    """Capability for incremental synthesis with feedback loops.

    Looks up MergeCapability, RefinementCapability, and ValidationCapability
    dynamically via self.agent.get_capability_by_type().

    Exposes @action_executor methods for:
    - add_result: Add new result and update synthesis
    - get_current_synthesis: Get current synthesized result
    - get_synthesis_progress: Get synthesis progress metrics
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "synthesis_final",
        capability_key: str = "synthesis"
    ):
        super().__init__(agent=agent, scope_id=get_scope_prefix(scope, agent, namespace=namespace), input_patterns=[], capability_key=capability_key)
        # Internal state only - no capability references stored
        self.partial_results: dict[str, ScopeAwareResult] = {}
        self.synthesis_history: list[SynthesisUpdate] = []

    def get_action_group_description(self) -> str:
        return (
            "Incremental Synthesis — builds up a synthesized result as partials arrive. "
            "add_result triggers a multi-step pipeline: check if new result refines existing ones "
            "(RefinementCapability), merge all partials (MergeCapability, REQUIRED), "
            "validate merged result (ValidationCapability, optional). "
            "Call get_synthesis_progress to check confidence/completeness before deciding to continue."
        )

    # -------------------------------------------------------------------------
    # Capability Lookup Helpers (dynamic lookup at call time)
    # -------------------------------------------------------------------------

    def _get_merge_capability(self):
        """Get MergeCapability from agent (REQUIRED)."""
        from .merge import MergeCapability
        cap = self.agent.get_capability_by_type(MergeCapability)
        if cap is None:
            raise RuntimeError(
                "SynthesisCapability requires MergeCapability. "
                "Add MergeCapability to agent's capabilities list."
            )
        return cap

    def _get_refinement_capability(self):
        """Get RefinementCapability from agent (optional)."""
        from .refinement import RefinementCapability
        return self.agent.get_capability_by_type(RefinementCapability)

    def _get_validation_capability(self):
        """Get ValidationCapability from agent (optional)."""
        from .validation import ValidationCapability
        return self.agent.get_capability_by_type(ValidationCapability)

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for SynthesisCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for SynthesisCapability")
        pass

    # -------------------------------------------------------------------------
    # Action Executors
    # -------------------------------------------------------------------------

    @action_executor()
    async def add_result(
        self,
        result_id: str,
        result: ScopeAwareResult
    ) -> SynthesisUpdate:
        """Add new result and update synthesis (plannable by LLM).

        1. Stores the new partial result
        2. Checks if new result provides context for existing results
        3. Refines existing results if needed (via RefinementCapability if present)
        4. Merges all partial results (via MergeCapability - required)
        5. Validates merged result (via ValidationCapability if present)

        Args:
            result_id: Unique ID for this result
            result: Result to add

        Returns:
            SynthesisUpdate with current synthesis and refinements
        """
        self.partial_results[result_id] = result

        # Step 1: Check refinements (lookup RefinementCapability dynamically)
        refinements = []
        refine_cap = self._get_refinement_capability()
        if refine_cap:
            for existing_id, existing in self.partial_results.items():
                if existing_id == result_id:
                    continue
                if self._provides_context(result, existing):
                    context = RefinementContext(
                        new_evidence={"related_result": result.to_blackboard_entry()}
                    )
                    if await refine_cap.should_refine(existing, context):
                        refined = await refine_cap.refine_result(existing, context)
                        self.partial_results[existing_id] = refined
                        refinements.append((existing_id, refined))

        # Step 2: Merge if we have multiple results (lookup MergeCapability dynamically)
        current_synthesis = None
        if len(self.partial_results) >= 2:
            merge_cap = self._get_merge_capability()  # Required - raises if missing
            current_synthesis = await merge_cap.merge_results(
                list(self.partial_results.values()),
                MergeContext()
            )

            # Step 3: Validate if capability present (lookup ValidationCapability dynamically)
            validate_cap = self._get_validation_capability()
            if validate_cap and current_synthesis:
                validation = await validate_cap.validate_result(current_synthesis)
                # Could trigger further refinement if validation fails
                if not validation.is_valid:
                    # Store validation issues for potential use
                    current_synthesis.scope.metadata["validation_issues"] = [
                        issue.model_dump() for issue in validation.issues
                    ]

        # Calculate progress
        complete_count = sum(1 for r in self.partial_results.values() if r.scope.is_complete)
        progress = complete_count / len(self.partial_results) if self.partial_results else 0.0

        update = SynthesisUpdate(
            current_synthesis=current_synthesis,
            refinements=refinements,
            is_complete=all(r.scope.is_complete for r in self.partial_results.values()),
            progress=progress
        )
        self.synthesis_history.append(update)
        return update

    @action_executor()
    async def get_current_synthesis(self) -> ScopeAwareResult | None:
        """Get current synthesized result.

        Returns:
            Current synthesis or None if not yet synthesized
        """
        if self.synthesis_history:
            return self.synthesis_history[-1].current_synthesis
        return None

    @action_executor()
    async def get_synthesis_progress(self) -> dict:
        """Get synthesis progress metrics.

        Returns:
            Dict with progress information
        """
        return {
            "partial_results_count": len(self.partial_results),
            "complete_count": sum(1 for r in self.partial_results.values() if r.scope.is_complete),
            "history_length": len(self.synthesis_history),
            "current_progress": self.synthesis_history[-1].progress if self.synthesis_history else 0.0
        }

    def _provides_context(
        self,
        new_result: ScopeAwareResult,
        existing_result: ScopeAwareResult
    ) -> bool:
        """Check if new result provides context for existing result.

        Args:
            new_result: New result
            existing_result: Existing result

        Returns:
            True if new result might help refine existing
        """
        # Check if new result addresses missing context
        for missing in existing_result.scope.missing_context:
            if missing in str(new_result.content):
                return True
        # Check shared related shards
        shared = set(new_result.scope.related_shards) & set(existing_result.scope.related_shards)
        return len(shared) > 0


# Utility functions

async def synthesize_incrementally(
    results: list[ScopeAwareResult[T]],
    merge_policy: MergePolicy[T],
    refinement_policy: RefinementPolicy[T] | None = None
) -> ScopeAwareResult[T]:
    """Synthesize results incrementally.

    Args:
        results: Results to synthesize
        merge_policy: Merge policy
        refinement_policy: Optional refinement policy

    Returns:
        Final synthesis
    """
    synthesizer = IncrementalSynthesizer(merge_policy, refinement_policy)

    final_update = None
    for i, result in enumerate(results):
        update = await synthesizer.add_result(f"result_{i}", result)
        final_update = update

    return final_update.current_synthesis if final_update else None

