"""Result capability for partial result management.

Provides @action_executor methods for storing, merging, validating, and
synthesizing results from multiple sources/agents.

Stores partial results in blackboard for cluster-wide visibility.
Delegates to existing MergeCapability, SynthesisCapability, ValidationCapability
for actual merge/synthesis/validation operations.

Usage:
    # Add capability to agent
    result_cap = ResultCapability(agent=self)
    self.add_capability(result_cap)

    # ActionPolicy can now use these actions:
    # - store_partial(result_id, result, source_agent, source_pages)
    # - get_partials(filter_type, filter_agent)
    # - merge_results(result_ids, merge_policy)
    # - validate_results(result_ids)
    # - synthesize(result_ids, synthesis_goal)
    # - detect_contradictions(result_ids)
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING
from overrides import override

from ...base import AgentCapability
from ...blackboard.protocol import ResultStorageProtocol
from ...scopes import BlackboardScope, get_scope_prefix
from ...models import AgentSuspensionState
from ..actions import action_executor

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class ResultCapability(AgentCapability):
    """Manages partial results with cluster-wide visibility.

    Stores partial results in blackboard for cluster-wide access.
    Delegates to MergeCapability, SynthesisCapability, ValidationCapability
    for actual operations via agent.get_capability_by_type().

    This provides a unified interface for result management that
    ActionPolicies can use without knowing about the underlying
    capability implementations.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "result_store",
        capability_key: str = "result_capability",
        app_name: str | None = None,
    ):
        """Initialize result capability.

        Args:
            agent: Owning agent
            scope: Blackboard scope (defaults to COLONY)
            namespace: Namespace for the result store (defaults to "result_store")
            capability_key: Key to identify this capability within the agent (default "result_capability")
            app_name: The `serving.Application` name where the agent system resides.
                      Required when creating detached handles from outside any `serving.deployment`.
        """
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name
        )

    def get_action_group_description(self) -> str:
        return (
            "Partial Result Management — cluster-wide storage for intermediate results. "
            "Orchestrates MergeCapability, ValidationCapability, and SynthesisCapability. "
            "Typical flow: store_partial as results arrive → get_partials to review → "
            "merge/validate/synthesize when ready. Results are blackboard-backed and visible cluster-wide. "
            "detect_contradictions before merging to catch conflicts early."
        )

    def _get_partial_key(self, result_id: str) -> str:
        """Get blackboard key for a partial result."""
        return ResultStorageProtocol.partial_key(result_id)

    def _get_index_key(self) -> str:
        """Get blackboard key for results index."""
        return ResultStorageProtocol.index_key()

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for ResultCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for ResultCapability")
        pass

    # === Action Executors ===

    @action_executor()
    async def store_partial(
        self,
        result_id: str,
        result: dict[str, Any],
        source_agent: str | None = None,
        source_pages: list[str] | None = None,
        result_type: str = "analysis",
    ) -> dict[str, Any]:
        """Store a partial result in cluster-wide storage.

        Results are stored in blackboard so all agents can access them.

        Args:
            result_id: Unique result identifier
            result: The result data
            source_agent: Agent that produced this result
            source_pages: Pages this result is based on
            result_type: Type of result ("analysis", "query_answer", "validation")

        Returns:
            Dict with:
            - stored: Whether storage succeeded
            - result_id: The result ID
            - total_partials: Total partial results stored
        """
        blackboard = await self.get_blackboard()

        # Store the result
        result_entry = {
            "result_id": result_id,
            "result": result,
            "source_agent": source_agent or self.agent.agent_id,
            "source_pages": source_pages or [],
            "result_type": result_type,
            "stored_at": time.time(),
            "stored_by": self.agent.agent_id,
        }

        key = self._get_partial_key(result_id)
        await blackboard.write(
            key,
            result_entry,
            created_by=self.agent.agent_id if self.agent else None
        )

        # Update index atomically (multiple workers may store_partial concurrently)
        index_key = self._get_index_key()
        async with blackboard.transaction() as txn:
            index = await txn.read(index_key) or {"result_ids": [], "count": 0}
            if result_id not in index["result_ids"]:
                index["result_ids"].append(result_id)
                index["count"] = len(index["result_ids"])
            await txn.write(index_key, index, created_by=self.agent.agent_id)

        logger.debug(
            f"ResultCapability: stored partial result {result_id} "
            f"(type={result_type}, total={index['count']})"
        )

        return {
            "stored": True,
            "result_id": result_id,
            "total_partials": index["count"],
        }

    @action_executor()
    async def get_partials(
        self,
        filter_type: str | None = None,
        filter_agent: str | None = None,
        filter_pages: list[str] | None = None,
        result_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get partial results with optional filters.

        Args:
            filter_type: Filter by result type
            filter_agent: Filter by source agent
            filter_pages: Filter by source pages (any overlap)
            result_ids: Specific result IDs to get (overrides other filters)

        Returns:
            Dict with:
            - results: List of partial result entries
            - count: Number of results returned
        """
        blackboard = await self.get_blackboard()

        # Get result IDs from index or use provided
        if result_ids is None:
            index_key = self._get_index_key()
            index = await blackboard.read(index_key) or {"result_ids": [], "count": 0}
            result_ids = index["result_ids"]

        # Fetch and filter results
        results = []
        for rid in result_ids:
            key = self._get_partial_key(rid)
            entry = await blackboard.read(key)
            if entry is None:
                continue

            # Apply filters
            if filter_type and entry.get("result_type") != filter_type:
                continue
            if filter_agent and entry.get("source_agent") != filter_agent:
                continue
            if filter_pages:
                entry_pages = set(entry.get("source_pages", []))
                if not entry_pages.intersection(filter_pages):
                    continue

            results.append(entry)

        return {
            "results": results,
            "count": len(results),
        }

    @action_executor()
    async def merge_results(
        self,
        result_ids: list[str],
        merge_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge multiple partial results.

        Delegates to MergeCapability if available, otherwise does
        simple concatenation.

        Args:
            result_ids: IDs of results to merge
            merge_context: Context for merge (hints, constraints)

        Returns:
            Dict with:
            - merged: Merged result data
            - source_ids: IDs of results that were merged
            - merge_method: How merge was performed
        """
        # Get the partial results
        partials_result = await self.get_partials(result_ids=result_ids)
        partials = partials_result.get("results", [])

        if not partials:
            return {
                "merged": None,
                "source_ids": result_ids,
                "merge_method": "none",
                "error": "no_results_found",
            }

        # Try to use MergeCapability
        try:
            from .merge import MergeCapability, MergeContext
            from ..scope import ScopeAwareResult, AnalysisScope

            merge_cap = self.agent.get_capability_by_type(MergeCapability)
            if merge_cap:
                # Convert to ScopeAwareResults for merge
                scope_results = []
                for p in partials:
                    result_data = p.get("result", {})
                    scope_results.append(ScopeAwareResult(
                        content=result_data,
                        scope=AnalysisScope(
                            agent_id=p.get("source_agent"),
                            related_shards=p.get("source_pages", []),
                        ),
                    ))

                ctx = MergeContext(**(merge_context or {}))
                merged_result = await merge_cap.merge_results(scope_results, ctx)

                return {
                    "merged": merged_result.content if hasattr(merged_result, 'content') else merged_result,
                    "source_ids": result_ids,
                    "merge_method": "merge_capability",
                }

        except Exception as e:
            logger.debug(f"MergeCapability not available or failed: {e}")

        # Fallback: simple concatenation
        merged_data = {
            "partial_results": [p.get("result") for p in partials],
            "source_agents": list(set(p.get("source_agent") for p in partials if p.get("source_agent"))),
            "source_pages": list(set(
                page
                for p in partials
                for page in p.get("source_pages", [])
            )),
            "merged_at": time.time(),
        }

        return {
            "merged": merged_data,
            "source_ids": result_ids,
            "merge_method": "concatenation",
        }

    @action_executor()
    async def validate_results(
        self,
        result_ids: list[str],
    ) -> dict[str, Any]:
        """Validate partial results for consistency and evidence.

        Delegates to ValidationCapability if available.

        Args:
            result_ids: IDs of results to validate

        Returns:
            Dict with:
            - is_valid: Overall validity
            - issues: List of validation issues
            - validated_count: Number of results validated
        """
        # Get the partial results
        partials_result = await self.get_partials(result_ids=result_ids)
        partials = partials_result.get("results", [])

        if not partials:
            return {
                "is_valid": True,
                "issues": [],
                "validated_count": 0,
                "error": "no_results_found",
            }

        # Try to use ValidationCapability
        try:
            from .validation import ValidationCapability
            from ..scope import ScopeAwareResult, AnalysisScope

            validation_cap = self.agent.get_capability_by_type(ValidationCapability)
            if validation_cap:
                all_issues = []
                for p in partials:
                    result_data = p.get("result", {})
                    scope_result = ScopeAwareResult(
                        content=result_data,
                        scope=AnalysisScope(
                            agent_id=p.get("source_agent"),
                            related_shards=p.get("source_pages", []),
                        ),
                    )

                    validation_result = await validation_cap.validate_result(scope_result)
                    all_issues.extend(validation_result.issues)

                return {
                    "is_valid": len(all_issues) == 0,
                    "issues": [str(i) for i in all_issues],
                    "validated_count": len(partials),
                    "validation_method": "validation_capability",
                }

        except Exception as e:
            logger.debug(f"ValidationCapability not available or failed: {e}")

        # Fallback: basic validation (check non-empty)
        issues = []
        for p in partials:
            if not p.get("result"):
                issues.append(f"Result {p.get('result_id')} is empty")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "validated_count": len(partials),
            "validation_method": "basic",
        }

    @action_executor()
    async def synthesize(
        self,
        result_ids: list[str],
        synthesis_goal: str | None = None,
    ) -> dict[str, Any]:
        """Synthesize final output from partial results.

        Delegates to SynthesisCapability if available.

        Args:
            result_ids: IDs of results to synthesize
            synthesis_goal: Goal for synthesis

        Returns:
            Dict with:
            - synthesis: Synthesized output
            - source_ids: IDs of results used
            - synthesis_method: How synthesis was performed
        """
        # Get the partial results
        partials_result = await self.get_partials(result_ids=result_ids)
        partials = partials_result.get("results", [])

        if not partials:
            return {
                "synthesis": None,
                "source_ids": result_ids,
                "synthesis_method": "none",
                "error": "no_results_found",
            }

        # Try to use SynthesisCapability
        try:
            from .synthesis import SynthesisCapability
            from ..scope import ScopeAwareResult, AnalysisScope

            synthesis_cap = self.agent.get_capability_by_type(SynthesisCapability)
            if synthesis_cap:
                # Add results one by one
                for p in partials:
                    result_data = p.get("result", {})
                    scope_result = ScopeAwareResult(
                        content=result_data,
                        scope=AnalysisScope(
                            agent_id=p.get("source_agent"),
                            related_shards=p.get("source_pages", []),
                        ),
                    )
                    await synthesis_cap.add_result(p.get("result_id"), scope_result)

                current = await synthesis_cap.get_current_synthesis()
                return {
                    "synthesis": current.content if current and hasattr(current, 'content') else current,
                    "source_ids": result_ids,
                    "synthesis_method": "synthesis_capability",
                }

        except Exception as e:
            logger.debug(f"SynthesisCapability not available or failed: {e}")

        # Fallback: merge results
        merged = await self.merge_results(result_ids)
        return {
            "synthesis": merged.get("merged"),
            "source_ids": result_ids,
            "synthesis_method": "merge_fallback",
        }

    @action_executor()
    async def detect_contradictions(
        self,
        result_ids: list[str],
    ) -> dict[str, Any]:
        """Detect contradictions between results.

        Delegates to ValidationCapability if available.

        Args:
            result_ids: IDs of results to check

        Returns:
            Dict with:
            - contradictions: List of contradictions found
            - count: Number of contradictions
        """
        # Get the partial results
        partials_result = await self.get_partials(result_ids=result_ids)
        partials = partials_result.get("results", [])

        if len(partials) < 2:
            return {
                "contradictions": [],
                "count": 0,
                "message": "need_at_least_two_results",
            }

        # Try to use ValidationCapability
        try:
            from .validation import ValidationCapability
            from ..scope import ScopeAwareResult, AnalysisScope

            validation_cap: ValidationCapability = self.agent.get_capability_by_type(ValidationCapability)
            if validation_cap:
                scope_results = []
                for p in partials:
                    result_data = p.get("result", {})
                    scope_results.append(ScopeAwareResult(
                        content=result_data,
                        scope=AnalysisScope(
                            agent_id=p.get("source_agent"),
                            related_shards=p.get("source_pages", []),
                        ),
                    ))

                contradictions = await validation_cap.detect_contradictions(scope_results)
                return {
                    "contradictions": [c.model_dump() if hasattr(c, 'model_dump') else str(c) for c in contradictions],
                    "count": len(contradictions),
                    "detection_method": "validation_capability",
                }

        except Exception as e:
            logger.debug(f"ValidationCapability not available or failed: {e}")

        # Fallback: no contradiction detection
        return {
            "contradictions": [],
            "count": 0,
            "detection_method": "none",
            "message": "validation_capability_not_available",
        }

    @action_executor()
    async def clear_partials(
        self,
        result_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Clear partial results from storage.

        Args:
            result_ids: Specific results to clear (None = all)

        Returns:
            Dict with:
            - cleared: Number of results cleared
        """
        blackboard = await self.get_blackboard()

        # Get result IDs to clear
        if result_ids is None:
            index_key = self._get_index_key()
            index = await blackboard.read(index_key) or {"result_ids": [], "count": 0}
            result_ids = index["result_ids"]

        # Delete each result
        cleared = 0
        for rid in result_ids:
            key = self._get_partial_key(rid)
            try:
                await blackboard.delete(key)
                cleared += 1
            except Exception:
                pass

        # Update index
        if result_ids:
            index_key = self._get_index_key()
            index = await blackboard.read(index_key) or {"result_ids": [], "count": 0}
            index["result_ids"] = [r for r in index["result_ids"] if r not in result_ids]
            index["count"] = len(index["result_ids"])
            await blackboard.write(
                index_key,
                index,
                created_by=self.agent.agent_id if self.agent else None
            )

        logger.info(f"ResultCapability: cleared {cleared} partial results")

        return {
            "cleared": cleared,
        }
