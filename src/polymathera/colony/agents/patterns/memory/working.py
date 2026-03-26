"""Working Memory Capability - manages the agent's current task context.

Working memory is a specialized memory capability that:
- Stores data relevant to the current task (PolicyPythonREPL/REPLCapability, plan, observations)
- Operates in append-only mode (benefits vLLM KV caching)
- Automatically compacts when approaching token limits
- Drains to STM when tasks complete

Unlike regular MemoryCapability, WorkingMemoryCapability tracks
token usage and triggers compaction when approaching the configured budget.

Example:
    ```python
    working = WorkingMemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_working(agent),
        max_tokens=8000,
        compaction_threshold=0.9,  # Compact at 90% capacity
    )
    await working.initialize()
    agent.add_capability(working)

    # Store observations (append-only)
    await working.store(observation)

    # Check if compaction needed
    if await working.needs_compaction():
        await working.compact(current_task="Implement auth feature")

    # Drain to STM on task completion (via MemoryLifecycleHooks)
    await working.drain_to_stm(stm_scope_id)
    ```
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from ...blackboard.types import BlackboardEntry
from ..actions.policies import action_executor
from .capability import MemoryCapability
from .types import MemoryLens, MaintenanceConfig, MemoryProducerConfig
from .protocols import (
    StorageBackendFactory,
    RetrievalStrategy,
    MaintenancePolicy,
    UtilityScorer,
    MemoryIngestPolicy,
)

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class WorkingMemoryCapability(MemoryCapability):
    """Capability for managing working memory with context compaction.

    Working memory extends MemoryCapability with:
    - Token budget tracking
    - Automatic compaction when approaching limits
    - Append-only semantics (no overwrites, only additions)
    - Drain-to-STM for task completion

    Attributes:
        max_tokens: Maximum token budget for this working memory
        compaction_threshold: Fraction of max_tokens that triggers compaction
        current_tokens: Estimated current token usage
    """

    def __init__(
        self,
        agent: "Agent",
        scope_id: str,
        *,
        # === INGESTION (Pull data INTO this scope) ===
        producers: list[MemoryProducerConfig] | None = None,
        ingestion_policy: MemoryIngestPolicy | None = None,

        # === STORAGE ===
        ttl_seconds: float | None = None,
        max_entries: int | None = None,
        storage_backend_factory: StorageBackendFactory | None = None,

        # === RETRIEVAL (Retrieval is the bottleneck) ===
        retrieval_strategy: RetrievalStrategy | None = None,
        lenses: list[MemoryLens] | None = None,

        # === MAINTENANCE (Keep this scope healthy) ===
        maintenance_policies: list[MaintenancePolicy] | None = None,
        maintenance_interval_seconds: float = 60.0,

        # === UTILITY SCORING (Memory as self-optimizing) ===
        utility_scorer: UtilityScorer | None = None,

        # === LEGACY COMPAT ===
        maintenance: MaintenanceConfig | None = None,  # TODO: Remove

        # === WORKING MEMORY CONFIG ===
        max_tokens: int = 8000,
        compaction_threshold: float = 0.9,

        # === IDENTITY ===
        capability_key: str | None = None,
    ):
        """Initialize memory capability.

        Args:
            agent: Agent that owns this capability
            scope_id: Blackboard scope for this memory level

            producers: Hook-based memory capture configurations
            ingestion_policy: Dataflow edges from other scopes, and
                when to trigger ingestion (default: OnDemand) and
                how to consolidate the entries. Each subscription listens
                and collects entries into pending queue. ALL pending
                entries are then transformed together before writing
                to this scope. If transformer=None, entries stored as-is.

            ttl_seconds: Default TTL for stored memories (None = no expiration)
            max_entries: Maximum entries before eviction (None = no limit)
            storage_backend_factory: Factory for creating storage backends for
                any scope (e.g., this scope or other scopes). Defaults to
                BlackboardStorageBackendFactory.

            retrieval_strategy: Strategy for memory retrieval (default: RecencyRetrieval)
            lenses: Named query configurations

            maintenance_policies: List of maintenance policies to run
                Use ConsolidationMaintenancePolicy for within-scope consolidation.
            maintenance_interval_seconds: How often to run maintenance (default: 60s)

            utility_scorer: Scorer for memory utility

            maintenance: Legacy MaintenanceConfig (deprecated, use maintenance_policies)

            max_tokens: Token budget (default 8000)
            compaction_threshold: Fraction triggering compaction (default 0.9)
        """
        # Use default maintenance config for working memory (no decay, no pruning)
        default_maintenance = MaintenanceConfig(
            decay_rate=0.0,  # No decay in working memory
            prune_threshold=0.0,  # No pruning
            track_access=False,  # No access tracking
        )

        super().__init__(
            agent=agent,
            scope_id=scope_id,
            producers=producers,
            ingestion_policy=ingestion_policy,
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            storage_backend_factory=storage_backend_factory,
            retrieval_strategy=retrieval_strategy,
            lenses=lenses,
            maintenance_policies=maintenance_policies,
            maintenance_interval_seconds=maintenance_interval_seconds,
            utility_scorer=utility_scorer,
            maintenance=maintenance or default_maintenance,
            map_to_vcm=False,
            vcm_config=None,
            capability_key=capability_key,
        )
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self._current_tokens = 0
        self._compaction_count = 0

    @property
    def current_tokens(self) -> int:
        """Estimated current token usage."""
        return self._current_tokens

    @property
    def token_usage_fraction(self) -> float:
        """Current token usage as fraction of max."""
        if self.max_tokens <= 0:
            return 0.0
        return self._current_tokens / self.max_tokens

    # -------------------------------------------------------------------------
    # LLM-Plannable Actions
    # -------------------------------------------------------------------------

    @action_executor(action_key="working_memory_store", planning_summary="Store data in working memory (append-only, token-bounded).")
    async def store(
        self,
        data: str | dict[str, Any] | BaseModel,
        tags: list[str] | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store data in working memory (append-only).

        Updates token count estimate. Does NOT overwrite existing entries.
        Raw strings/dicts (from LLM-planned actions) are auto-wrapped by the
        parent ``MemoryCapability.store()`` into a :class:`MemoryRecord`.

        Args:
            data: Content to store (string, dict, or Pydantic model with record_id)
            tags: Tags for categorization
            ttl_seconds: TTL override
            metadata: Additional metadata

        Returns:
            Key under which data was stored
        """
        # Auto-wrap strings/dicts before token estimation (parent.store does
        # the same, but we need the wrapped object for accurate token counting).
        if isinstance(data, str):
            data = {"text": data}
        if isinstance(data, dict):
            from .types import MemoryRecord
            data = MemoryRecord(content=data, tags=tags or set())

        # Estimate tokens for this entry
        entry_tokens = self._estimate_tokens(data)
        self._current_tokens += entry_tokens

        # Add token count to metadata
        entry_metadata = {
            **(metadata or {}),
            "estimated_tokens": entry_tokens,
            "append_order": await self._get_next_append_order(),
        }

        # Store using parent method
        key = await super().store(
            data=data,
            tags=tags,
            ttl_seconds=ttl_seconds,
            metadata=entry_metadata,
        )

        logger.debug(
            f"Working memory: stored {entry_tokens} tokens, "
            f"total {self._current_tokens}/{self.max_tokens}"
        )

        return key

    @action_executor(action_key="working_memory_needs_compaction", planning_summary="Check if working memory needs compaction.")
    async def needs_compaction(self) -> bool:
        """Check if working memory needs compaction.

        Returns True if token usage exceeds the compaction threshold.

        Returns:
            True if compaction should be triggered
        """
        return self.token_usage_fraction >= self.compaction_threshold

    @action_executor(action_key="working_memory_compact", planning_summary="Compact working memory by summarizing old entries to free token budget.")
    async def compact(
        self,
        current_task: str | None = None,
        preserve_recent_seconds: float = 300,
    ) -> dict[str, Any]:
        """Compact working memory to free up token budget.

        Compaction strategy:
        1. Keep recent entries (within preserve_recent_seconds)
        2. Summarize older entries (task-aware if current_task provided)
        3. Replace old entries with summary
        4. Record compaction event

        Args:
            current_task: Current task description for task-aware summarization
            preserve_recent_seconds: Keep entries newer than this (default 5 min)

        Returns:
            Dict with compaction statistics
        """
        entries = await self.storage.query(limit=1000)  # TODO: Make configurable

        if not entries:
            return {"status": "empty", "freed_tokens": 0}

        now = time.time()
        recent_entries = []
        old_entries = []

        for entry in entries:
            age = now - entry.created_at
            if age < preserve_recent_seconds:
                recent_entries.append(entry)
            else:
                old_entries.append(entry)

        if not old_entries:
            return {"status": "nothing_to_compact", "freed_tokens": 0}

        # Calculate tokens to free
        old_tokens = sum(
            entry.metadata.get("estimated_tokens", self._estimate_tokens_from_entry(entry))
            for entry in old_entries
        )

        # Create compaction summary
        # TODO: Use LLM for task-aware summarization
        summary_content = self._create_compaction_summary(
            old_entries,
            current_task=current_task,
        )

        # Delete old entries
        for entry in old_entries:
            await self.storage.delete(entry.key)

        # Store summary as a special compaction entry
        summary_data = CompactionSummary(
            id=f"compaction_{self._compaction_count}",
            content=summary_content,
            source_count=len(old_entries),
            source_tokens=old_tokens,
            timestamp=now,
        )

        summary_tokens = self._estimate_tokens(summary_data)

        # Store using parent method (updates token count)
        await super().store(
            data=summary_data,
            tags={"compaction", "summary"},
            metadata={
                "estimated_tokens": summary_tokens,
                "compaction_number": self._compaction_count,
            },
        )

        # Update token count
        freed_tokens = old_tokens - summary_tokens
        self._current_tokens = max(0, self._current_tokens - freed_tokens)
        self._compaction_count += 1

        result = {
            "status": "compacted",
            "entries_removed": len(old_entries),
            "old_tokens": old_tokens,
            "summary_tokens": summary_tokens,
            "freed_tokens": freed_tokens,
            "current_tokens": self._current_tokens,
            "compaction_number": self._compaction_count,
        }

        logger.info(f"Working memory compacted: {result}")
        return result

    @action_executor(action_key="working_memory_drain", planning_summary="Drain working memory to STM on task completion, then clear.")
    async def drain_to_stm(
        self,
        stm_scope_id: str,
    ) -> int:
        """Drain working memory to STM on task completion.

        Transfers all working memory entries to STM with task completion
        metadata, then clears working memory.

        Args:
            stm_scope_id: Target STM scope ID

        Returns:
            Number of entries drained
        """
        entries = await self.storage.query(limit=1000)  # TODO: Make configurable

        if not entries:
            return 0

        # Get STM storage backend via factory
        stm_storage = await self._storage_backend_factory.create_for_scope(stm_scope_id)

        # Transfer entries to STM
        now = time.time()
        for entry in entries:
            # Preserve the entry with task completion metadata
            new_metadata = {
                **entry.metadata,
                "source_scope": self.scope_id,
                "drained_at": now,
                "from_working_memory": True,
            }

            await stm_storage.write(
                key=entry.key.replace(self.scope_id, stm_scope_id),
                value=entry.value,
                metadata=new_metadata,
                tags=entry.tags | {"from_working_memory"},
            )

            # Delete from working memory
            await self.storage.delete(entry.key)

        # Reset token count
        self._current_tokens = 0

        logger.info(f"Drained {len(entries)} entries from working memory to STM")
        return len(entries)

    @action_executor(action_key="working_memory_clear", planning_summary="Clear all working memory entries.")
    async def clear(self) -> int:
        """Clear all working memory entries.

        Use this when starting a completely new task without preserving context.

        Returns:
            Number of entries cleared
        """
        count = await self.forget(older_than_seconds=0)  # Delete all
        self._current_tokens = 0
        logger.info(f"Cleared {count} entries from working memory")
        return count

    # -------------------------------------------------------------------------
    # Token Estimation
    # -------------------------------------------------------------------------

    def _estimate_tokens(self, data: BaseModel) -> int:
        """Estimate token count for a data object.

        Simple heuristic: ~4 characters per token.
        Override for more accurate estimation.

        Args:
            data: Pydantic model to estimate

        Returns:
            Estimated token count
        """
        # Convert to string and estimate
        if hasattr(data, "model_dump_json"):
            text = data.model_dump_json()
        elif hasattr(data, "json"):
            text = data.json()
        else:
            text = str(data)

        # ~4 characters per token (rough estimate)
        return len(text) // 4

    def _estimate_tokens_from_entry(self, entry: BlackboardEntry) -> int:
        """Estimate tokens from a blackboard entry.

        Args:
            entry: Entry to estimate

        Returns:
            Estimated token count
        """
        import json
        try:
            text = json.dumps(entry.value)
        except Exception:
            text = str(entry.value)
        return len(text) // 4

    async def _get_next_append_order(self) -> int:
        """Get the next append order number for ordering entries."""
        entries = await self.storage.query(limit=10000)
        if not entries:
            return 0

        max_order = max(
            entry.metadata.get("append_order", 0)
            for entry in entries
        )
        return max_order + 1

    def _create_compaction_summary(
        self,
        entries: list[BlackboardEntry],
        current_task: str | None = None,
    ) -> str:
        """Create a summary of entries for compaction.

        TODO: Use LLM for intelligent, task-aware summarization.
        Current implementation is a simple concatenation.

        Args:
            entries: Entries to summarize
            current_task: Current task for context-aware summarization

        Returns:
            Summary string
        """
        # Sort by append_order or created_at
        sorted_entries = sorted(
            entries,
            key=lambda e: e.metadata.get("append_order", e.created_at),
        )

        # Simple concatenation for now
        parts = []
        if current_task:
            parts.append(f"[Task: {current_task}]")
        parts.append(f"[Compacted {len(entries)} entries]")

        for entry in sorted_entries[:5]:  # Show first 5
            value_str = str(entry.value)[:200]  # Truncate
            parts.append(f"- {value_str}")

        if len(entries) > 5:
            parts.append(f"... and {len(entries) - 5} more entries")

        return "\n".join(parts)


class CompactionSummary(BaseModel):
    """Summary of compacted working memory entries."""

    id: str
    """Unique ID for this summary."""

    content: str
    """Summarized content from compacted entries."""

    source_count: int
    """Number of entries that were compacted."""

    source_tokens: int
    """Estimated tokens from source entries."""

    timestamp: float
    """When compaction occurred."""

