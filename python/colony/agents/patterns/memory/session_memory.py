"""Session-aware Memory Capability.

The SessionMemoryCapability provides session-scoped memory for agents. Unlike
regular MemoryCapability instances (which are per-agent), SessionMemoryCapability
is a single instance per tenant that stores entries tagged with session_id.

Key concepts:
- ONE instance per tenant (not per session)
- Entries are tagged with session_id from the current session context
- Retrieval automatically filters by the current session_id
- Falls back to unfiltered retrieval when not in a session context

This allows an Agent to handle multiple sessions for a tenant, with each
session having isolated memory that persists across agent invocations.

Example:
    ```python
    from polymathera.colony.agents.patterns.memory.session_memory import (
        SessionMemoryCapability,
    )
    from polymathera.colony.agents.sessions import session_context

    # Create session memory capability (one per tenant)
    session_memory = SessionMemoryCapability(
        agent=agent,
        tenant_id="my-tenant",
    )
    await session_memory.initialize()
    agent.add_capability(session_memory)

    # Use within session context
    async with session_context(session):
        # Store memory - automatically tagged with session_id
        await session_memory.store(observation)

        # Recall memory - automatically filtered by session_id
        memories = await session_memory.recall(query)
    ```
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from .capability import MemoryCapability
from .scopes import MemoryScope
from .types import MemoryQuery, RetrievalContext, ScoredEntry
from .protocols import (
    MaintenancePolicy,
    RetrievalStrategy,
    StorageBackendFactory,
    UtilityScorer,
    MemoryIngestPolicy,
    RecencyRetrieval,
)
from ...blackboard.types import BlackboardEntry
from ..actions.policies import action_executor

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


def _get_current_session_id() -> str | None:
    """Get the current session ID from context.

    Lazy import to avoid circular dependencies.
    """
    from ...sessions.context import get_current_session_id
    return get_current_session_id()


class SessionMemoryCapability(MemoryCapability):
    """Session-scoped memory capability.

    A specialized `MemoryCapability` that:
    1. Is a single instance per tenant (not per session)
    2. Tags all entries with the current `session_id`
    3. Filters retrieval by the current `session_id`

    This enables session-isolated memory without creating separate
    capability instances or storage scopes per session.

    Attributes:
        `tenant_id`: The tenant this capability serves
        `include_cross_session`: If `True`, cross-session memories are
            included with lower ranking (default: False)
    """

    def __init__(
        self,
        agent: "Agent",
        tenant_id: str,
        *,
        # Session-specific options
        include_cross_session: bool = False,
        cross_session_weight: float = 0.3,

        # Standard MemoryCapability options (subset)
        ttl_seconds: float | None = None,
        max_entries: int | None = None,
        storage_backend_factory: StorageBackendFactory | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
        maintenance_policies: list[MaintenancePolicy] | None = None,
        maintenance_interval_seconds: float = 60.0,
        utility_scorer: UtilityScorer | None = None,
    ):
        """Initialize session memory capability.

        Args:
            `agent`: Agent that owns this capability
            `tenant_id`: Tenant this capability serves

            `include_cross_session`: Whether to include cross-session memories
                in retrieval (with lower weight). Default: False.
            `cross_session_weight`: Weight for cross-session entries when
                `include_cross_session` is True (0.0-1.0). Default: 0.3.

            `ttl_seconds`: Default TTL for stored memories
            `max_entries`: Maximum entries before eviction
            `storage_backend_factory`: Factory for creating storage backends
            `retrieval_strategy`: Strategy for memory retrieval
            `maintenance_policies`: List of maintenance policies to run
            `maintenance_interval_seconds`: How often to run maintenance
            `utility_scorer`: Scorer for memory utility
        """
        # Use tenant-level session scope
        scope_id = MemoryScope.session(tenant_id)

        super().__init__(
            agent=agent,
            scope_id=scope_id,
            # No producers or ingestion policy - session memory is explicit
            producers=None,
            ingestion_policy=MemoryIngestPolicy(),  # On-demand only
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            storage_backend_factory=storage_backend_factory,
            retrieval_strategy=retrieval_strategy or RecencyRetrieval(),
            maintenance_policies=maintenance_policies,
            maintenance_interval_seconds=maintenance_interval_seconds,
            utility_scorer=utility_scorer,
        )

        self.tenant_id = tenant_id
        self.include_cross_session = include_cross_session
        self.cross_session_weight = cross_session_weight

    @action_executor(action_key="session_memory_store", planning_summary="Store data in session-scoped memory with optional tags.")
    async def store(
        self,
        data: str | dict[str, Any] | BaseModel,
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory with automatic session tagging.

        Extends `MemoryCapability`.store to automatically tag entries with
        the current `session_id` from context.

        Args:
            data: Memory content to store (dict or Pydantic model)
            tags: Tags for categorization and retrieval
            ttl_seconds: TTL override (uses level default if None)
            metadata: Additional metadata

        Returns:
            Key under which the memory was stored

        Note:
            If not in a session context, stores without `session_id` tag.
            A warning is logged in this case. Session tagging is handled
            automatically by EnhancedBlackboard.write().
        """
        session_id = _get_current_session_id()

        if not session_id:
            logger.warning(
                f"SessionMemoryCapability.store called outside session context. "
                f"Entry will not be session-tagged."
            )

        # Session_id is automatically added to tags/metadata by EnhancedBlackboard.write()
        return await super().store(
            data=data,
            tags=tags,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )

    @action_executor(action_key="session_memory_recall_with_scores", planning_summary="Recall session memories with relevance scores, optionally including cross-session.")
    async def recall_with_scores(
        self,
        query: MemoryQuery | str | None = None,
        lens: str | None = None,
        context: RetrievalContext | None = None,
    ) -> list[ScoredEntry]:
        """Recall memories with automatic session filtering.

        Extends MemoryCapability.recall_with_scores to automatically filter
        by the current session_id from context.

        Args:
            query: Query string or MemoryQuery object for filtering/ranking.
                LLM planners typically pass a plain string which is auto-wrapped.
            lens: Name of a predefined lens to apply
            context: Retrieval context (goal, agent state) for relevance

        Returns:
            List of matching ScoredEntry objects, filtered by session_id
        """
        if isinstance(query, str):
            query = MemoryQuery(query=query)
        elif isinstance(query, dict):
            query = MemoryQuery(**query)
        session_id = _get_current_session_id()

        # Build session-filtered query
        effective_query = query or MemoryQuery()

        if session_id:
            # Add session tag filter
            session_tags = set(effective_query.tags) if effective_query.tags else set()
            session_tags.add(f"session:{session_id}")
            effective_query = MemoryQuery(
                query=effective_query.query,
                tags=session_tags,
                max_results=effective_query.max_results,
                min_relevance=effective_query.min_relevance,
                include_expired=effective_query.include_expired,
                max_age_seconds=effective_query.max_age_seconds,
            )

            # Get session-filtered results
            session_results = await super().recall_with_scores(
                query=effective_query,
                lens=lens,
                context=context,
            )

            # Optionally include cross-session results with lower weight
            if self.include_cross_session:
                # Query without session filter
                cross_query = query or MemoryQuery()
                all_results = await super().recall_with_scores(
                    query=cross_query,
                    lens=lens,
                    context=context,
                )

                # Filter to only cross-session entries
                session_keys = {sr.entry.key for sr in session_results}
                cross_results = [
                    sr for sr in all_results
                    if sr.entry.key not in session_keys
                ]

                # Apply cross-session weight penalty
                # ScoredEntry is a dataclass, so we create new instances with adjusted scores
                weighted_cross_results = []
                for sr in cross_results:
                    weighted_cross_results.append(
                        ScoredEntry(
                            entry=sr.entry,
                            score=sr.score * self.cross_session_weight,
                            components={
                                **sr.components,
                                "cross_session_penalty": self.cross_session_weight,
                            },
                        )
                    )

                # Merge and re-sort
                combined = session_results + weighted_cross_results
                combined.sort(key=lambda x: x.score, reverse=True)
                return combined[:effective_query.max_results]

            return session_results
        else:
            # Not in session context - return all entries
            logger.debug(
                "SessionMemoryCapability.recall called outside session context. "
                "Returning unfiltered results."
            )
            return await super().recall_with_scores(
                query=effective_query,
                lens=lens,
                context=context,
            )

    @action_executor(action_key="session_memory_recall", planning_summary="Recall session memories matching a query.")
    async def recall(
        self,
        query: MemoryQuery | str | None = None,
        lens: str | None = None,
        context: RetrievalContext | None = None,
    ) -> list[BlackboardEntry]:
        """Recall memories with automatic session filtering.

        Args:
            query: Query string or MemoryQuery object.
            lens: Name of a predefined lens to apply
            context: Retrieval context

        Returns:
            List of matching BlackboardEntry objects, filtered by session_id
        """
        scored = await self.recall_with_scores(query=query, lens=lens, context=context)
        return [se.entry for se in scored]

    @action_executor(action_key="session_memory_forget", planning_summary="Delete session memories by key, tags, or age.")
    async def forget(
        self,
        keys: list[str] | None = None,
        tags: set[str] | None = None,
        older_than_seconds: float | None = None,
    ) -> int:
        """Forget memories with automatic session scoping.

        When in a session context, only forgets memories from the current
        session (unless explicit keys are provided).

        Args:
            keys: Specific keys to delete (bypasses session filter)
            tags: Delete entries matching any of these tags (within session)
            older_than_seconds: Delete entries older than this (within session)

        Returns:
            Number of memories forgotten
        """
        session_id = _get_current_session_id()

        # If explicit keys provided, use parent implementation directly
        if keys:
            return await super().forget(keys=keys)

        # Add session tag filter when in session context
        if session_id:
            session_tags = set(tags) if tags else set()
            session_tags.add(f"session:{session_id}")
            return await super().forget(
                tags=session_tags,
                older_than_seconds=older_than_seconds,
            )
        else:
            return await super().forget(
                tags=tags,
                older_than_seconds=older_than_seconds,
            )

    async def get_session_stats(self, session_id: str | None = None) -> dict[str, Any]:
        """Get statistics for a specific session's memory.

        Args:
            session_id: Session ID to get stats for. Uses current session
                if None and in session context.

        Returns:
            Dictionary with session memory statistics
        """
        target_session_id = session_id or _get_current_session_id()

        if not target_session_id:
            return {
                "error": "No session_id provided and not in session context",
                "total_entries": 0,
            }

        # Query for session entries
        entries = await self.storage.query(limit=10000)
        session_entries = [
            e for e in entries
            if f"session:{target_session_id}" in e.tags
        ]

        # Calculate stats
        if session_entries:
            oldest = min(e.created_at for e in session_entries)
            newest = max(e.created_at for e in session_entries)
        else:
            oldest = newest = None

        return {
            "session_id": target_session_id,
            "total_entries": len(session_entries),
            "oldest_entry_at": oldest,
            "newest_entry_at": newest,
            "age_span_seconds": (newest - oldest) if oldest and newest else 0,
        }

    async def cleanup_session(self, session_id: str) -> int:
        """Remove all memories for a specific session.

        Called when a session is closed/archived to clean up its memories.

        Args:
            session_id: Session ID to clean up

        Returns:
            Number of entries removed
        """
        return await super().forget(tags={f"session:{session_id}"})
