"""Pluggable protocols for the unified MemoryCapability.

This module defines the core protocols (interfaces) that enable pluggable
behavior in the memory system:

- StorageBackend: Where memories are stored (blackboard, vector DB, etc.)
- RetrievalStrategy: How memories are retrieved and ranked
- MaintenancePolicy: How memories are maintained (decay, prune, etc.)
- ConsolidationTransformer: How memories are consolidated/abstracted

These protocols allow users to customize memory behavior without changing
the core MemoryCapability class.

Design justifications from research (<mark> highlights):
- <mark>Non-parametric storage works; parametric NOT solved online</mark>
- <mark>Retrieval quality is the bottleneck</mark>
- <mark>Forgetting is heuristic—TTL, LRU, importance all have tradeoffs</mark>
- <mark>Consolidation is task-specific, unsolved</mark>
"""

from __future__ import annotations

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable, Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel


if TYPE_CHECKING:
    from ...base import Agent
    from .types import (
        MemoryQuery,
        RetrievalContext,
        ScoredEntry,
        MaintenanceResult,
        ConsolidationContext,
        MemorySubscription,
    )
from ...blackboard.protocol import MemoryRecordProtocol
from ...blackboard.types import BlackboardEntry, BlackboardEvent, KeyPatternFilter

# =============================================================================
# Storage Backend Protocol
# =============================================================================

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage interface for memory entries.

    Why a protocol: Memory storage is unsolved. Different levels may need:
    - Blackboard: Fast, key-value, tags (sensory, working)
    - Vector DB: Semantic search (STM, LTM episodic/semantic)
    - Graph DB: Relational queries (LTM semantic)
    - Parametric: LoRA adapters (LTM procedural, experimental)

    Justification: <mark>Non-parametric storage works; parametric NOT solved online</mark>

    Contract:
    - All write operations MUST emit events that can be subscribed to via
      `stream_events_to_queue`. This is essential for the subscription-based
      dataflow architecture where higher memory levels subscribe to lower levels.
    - TODO: The event key pattern needs to follow a pattern.

    Implementations:
    - BlackboardStorageBackend: Default, delegates to EnhancedBlackboard
    - VectorStorageBackend: ChromaDB/FAISS/Pinecone (TODO: future)
    - HybridStorageBackend: Blackboard + Vector (TODO: future)
    """

    @property
    def scope_id(self) -> str:
        """Scope ID this backend is bound to."""
        ...

    async def write(
        self,
        key: str,
        value: dict[str, Any],
        metadata: dict[str, Any],
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Write an entry to storage.

        Args:
            key: Storage key
            value: Serialized data (dict from model_dump())
            metadata: Entry metadata (relevance, timestamps, etc.)
            tags: Tags for categorization and filtering
            ttl_seconds: Time to live (None = no expiration)
        """
        ...

    async def read(self, key: str) -> BlackboardEntry | None:
        """Read a single entry by key.

        Args:
            key: Storage key

        Returns:
            Entry if found, None otherwise
        """
        ...

    async def query(
        self,
        pattern: str | None = None,
        tags: set[str] | None = None,
        time_range: tuple[float, float] | None = None,
        limit: int = 100,
    ) -> list[BlackboardEntry]:
        """Query entries by pattern, tags, or time range.

        Args:
            pattern: Key pattern (e.g., "scope:*")
            tags: Filter by tags (entries must have ALL tags)
            time_range: (start_timestamp, end_timestamp)
            limit: Maximum entries to return

        Returns:
            List of matching entries
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete an entry by key.

        Args:
            key: Storage key

        Returns:
            True if deleted, False if not found
        """
        ...

    async def count(self) -> int:
        """Count total entries in this scope.

        Returns:
            Number of entries
        """
        ...

    async def clear(self) -> int:
        """Delete all entries in this scope.

        Returns:
            Number of entries deleted
        """
        ...

    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        key_pattern: str,
        consumer_group: str | None = None,
        consumer_name: str | None = None,
    ) -> None:
        """Subscribe to write events matching a key pattern.

        This method is essential for the subscription-based dataflow architecture.
        All `write()` operations MUST emit events that are delivered to subscribers.

        When ``consumer_group`` is provided, uses Redis Streams consumer groups
        for exactly-once delivery across multiple consumers (e.g., VCM replicas).
        Each event is delivered to exactly one consumer in the group.
        When not provided, uses standard pub-sub (all subscribers receive all events).

        Args:
            event_queue: Queue to receive events
            key_pattern: Pattern to filter events (e.g., ``"scope:DataType:*"``)
            consumer_group: Optional Redis Streams consumer group name for
                exactly-once delivery across consumers. Used by VCM's
                ``BlackboardContextPageSource`` for multi-replica deduplication.
            consumer_name: This consumer's name within the group (e.g., replica ID).
                Required when ``consumer_group`` is provided.
        """
        ...


@runtime_checkable
class StorageBackendFactory(Protocol):
    """Factory for creating storage backends for arbitrary scopes.

    This enables MemoryCapability to access other scopes (e.g., for transfer_to)
    without hardcoding a specific StorageBackend implementation.

    The default implementation creates BlackboardStorageBackend instances.
    Users can provide custom factories for vector DBs or other storage types.
    """

    async def create_for_scope(self, scope_id: str) -> StorageBackend:
        """Create a storage backend for the given scope.

        Args:
            scope_id: Scope ID to create backend for

        Returns:
            StorageBackend instance for the scope
        """
        ...


# =============================================================================
# Retrieval Strategy Protocol
# =============================================================================


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Abstract retrieval interface.

    Why pluggable: <mark>Retrieval quality is the bottleneck.</mark>
    No single strategy works for all cases:
    - Recency: Good for sensory/working memory
    - Similarity: Good for semantic search
    - Generative Agents: Balanced recency + importance + relevance
    - LLM-guided: Let LLM refine queries iteratively

    Implementations:
    - RecencyRetrieval: Most recent first (default)
    - SimilarityRetrieval: Embedding-based similarity (TODO)
    - CompositeRetrieval: Weighted combination (TODO)
    - LLMGuidedRetrieval: LLM refines queries (TODO)
    """

    async def retrieve(
        self,
        query: "MemoryQuery",
        backend: StorageBackend,
        context: "RetrievalContext | None" = None,
    ) -> list["ScoredEntry"]:
        """Retrieve memories matching the query.

        Args:
            query: Query parameters (tags, max_results, etc.)
            backend: Storage backend to query
            context: Optional retrieval context (goal, agent state)

        Returns:
            List of scored entries, sorted by score descending
        """
        ...


# =============================================================================
# Maintenance Policy Protocol
# =============================================================================


@runtime_checkable
class MaintenancePolicy(Protocol):
    """Abstract maintenance interface.

    Why pluggable: <mark>Forgetting is heuristic. No good solution.</mark>
    - <mark>TTL-based: Loses important old memories</mark>
    - <mark>LRU: Loses rarely-accessed but critical info</mark>
    - <mark>Importance-based: Requires expensive scoring</mark>

    Implementations:
    - TTLMaintenancePolicy: Remove expired entries
    - LRUPolicy: Remove least recently used
    - DecayMaintenancePolicy: Reduce relevance over time
    - CapacityMaintenancePolicy: Enforce max_entries limit
    - DeduplicationPolicy: Merge similar entries
    - UtilityPolicy: Remove lowest utility entries
    """

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Check if this policy should run now.

        Args:
            backend: Storage backend to check
            last_run: Timestamp of last run (None if never)

        Returns:
            True if policy should run
        """
        ...

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Execute the maintenance policy.

        Args:
            backend: Storage backend to maintain

        Returns:
            Result with counts of processed/removed/modified entries
        """
        ...


# =============================================================================
# Consolidation Transformer Protocol
# =============================================================================

# Type variables for transformer source and target data types
TSource = TypeVar('TSource', bound=BaseModel)
TTarget = TypeVar('TTarget', bound=BaseModel)


class ConsolidationTransformer(ABC, Generic[TSource, TTarget]):
    """Abstract consolidation interface. Transforms memories during transfer between levels.

    Why pluggable: Consolidation is task-specific, unsolved.
    - Abstraction is task-specific—no universal approach
    - When to consolidate? Too early loses info, too late wastes

    Consolidation transforms multiple memories into fewer, more abstract ones.
    This is how the memory system builds hierarchy (observations → insights).

    TODO: Implementations:
    - SummarizingTransformer: LLM summarizes multiple memories
    - ClusteringTransformer: Group similar, create cluster summary
    - AbstractionTransformer: Extract patterns/schemas
    - SkillExtractionTransformer: Extract procedural patterns

    Transformers implement the data processing in memory edges:
    - Summarization: Combine multiple memories into a summary
    - Abstraction: Extract higher-level concepts
    - Filtering: Drop irrelevant memories
    - Enrichment: Add metadata, embeddings, tags

    Returns `list[BlackboardEntry]` where:
    - `entry.value` is a BaseModel instance with `record_id` property
    - `entry.tags` are the tags for the target entry
    - `entry.metadata` is the metadata for the target entry
    - `entry.ttl_seconds` is the TTL for the target entry (optional)

    The transformer decides how to transfer/merge source tags and metadata to target.

    Example:
        ```python
        class ObservationSummarizer(ConsolidationTransformer[Observation, ObservationSummary]):
            async def consolidate(self, entries: list[BlackboardEntry], context: "ConsolidationContext") -> list[BlackboardEntry]:
                # Combine source tags
                all_tags = set()
                for e in entries:
                    all_tags.update(e.tags)
                all_tags.add("summary")

                # Create target data
                summary = ObservationSummary(...)

                return [BlackboardEntry(
                    key="",  # Will be set by transfer capability
                    value=summary,
                    tags=all_tags,
                    metadata={"source_count": len(source_entries)},
                )]
        ```
    """

    @abstractmethod
    async def consolidate(
        self,
        entries: list[BlackboardEntry],
        context: ConsolidationContext,
    ) -> list[BlackboardEntry]:
        """Consolidate entries into fewer, more abstract entries.
        Transform source entries into target entries.

        Args:
            entries: Source entries from source scope (all of same TSource type) to consolidate
            context: Consolidation context (source/target scopes, goals)

        Returns:
            Consolidated entries to write to target scope, where entry.value is a TTarget instance ready for target scope.
            Each entry.value must have record_id.
            The TTarget must have `record_id` instance property.
            The entry.key field is ignored (set by memory capability).
        """
        ...




class IdentityConsolidationTransformer(ConsolidationTransformer[TSource, TSource]):
    """No-op transformer that passes data through unchanged.

    Use this for simple memory forwarding without modification.
    Entries from the blackboard may have serialized (dict) values;
    these are re-wrapped into MemoryRecord to satisfy the contract.
    """

    async def consolidate(
        self,
        entries: list[BlackboardEntry],
        context: ConsolidationContext,
    ) -> list[BlackboardEntry]:
        """Pass through unchanged, ensuring values satisfy the contract."""
        from .types import MemoryRecord

        result = []
        for entry in entries:
            value = entry.value
            if not hasattr(value, "record_id"):
                if isinstance(value, str):
                    value = MemoryRecord(content={"text": value}, tags=entry.tags)
                elif isinstance(value, dict):
                    value = MemoryRecord(content=value, tags=entry.tags)
                else:
                    raise TypeError(
                        f"IdentityConsolidationTransformer: entry.value is "
                        f"{type(value).__name__}, expected BaseModel with "
                        f"record_id or dict/str for auto-wrapping."
                    )
            result.append(BlackboardEntry(
                key=entry.key,
                value=value,
                version=entry.version,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                created_by=entry.created_by,
                tags=entry.tags,
                metadata=entry.metadata,
            ))
        return result


class SummarizingTransformer(ConsolidationTransformer[TSource, TTarget]):
    """Summarize multiple memories into one using LLM.

    Different memory transfers require different summarization strategies:
    - Working → STM: Summarize recent actions/observations into coherent narrative
    - STM → LTM Episodic: Extract key events, outcomes, and experiences
    - STM → LTM Semantic: Distill facts, concepts, and patterns
    - LTM Episodic → Procedural: Extract action patterns, strategies, lessons
    - Within-scope consolidation: Compress similar entries

    The `prompt` parameter customizes the summarization instruction for each use case.

    Args:
        agent: Agent instance for LLM access
        prompt: Summarization instruction describing what kind of summary to produce.
                This is critical for getting useful consolidation results.
        max_tokens: Maximum tokens in the summary output
    """

    DEFAULT_PROMPT = "Summarize the following memories into a coherent, concise summary."

    def __init__(
        self,
        agent: "Agent",
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 500,
    ):
        self.agent = agent
        self.prompt = prompt
        self.max_tokens = max_tokens

    async def consolidate(
        self,
        entries: list[BlackboardEntry],
        context: ConsolidationContext,
    ) -> list[BlackboardEntry]:
        """Summarize source entries using LLM with the configured prompt."""
        if not entries:
            return []

        # Format entries for LLM
        entries_text = "\n\n".join(
            f"[{i+1}] {entry.value}"
            for i, entry in enumerate(entries)
        )

        # Build the full prompt with context
        full_prompt = f"""{self.prompt}

Source memories:
{entries_text}

Summary:"""

        # TODO: Replace with actual LLM call: self.agent.infer(full_prompt, max_tokens=self.max_tokens)
        # For now, create a placeholder that concatenates key points
        summary_text = f"[Consolidated {len(entries)} entries]"

        # Collect all tags from source entries
        all_tags: set[str] = set()
        for entry in entries:
            all_tags.update(entry.tags)
        all_tags.add("summary")
        all_tags.add("consolidated")

        from .types import MemoryRecord
        summary_record = MemoryRecord(
            content={"summary": summary_text},
            tags=all_tags,
        )

        return [BlackboardEntry(
            key="",  # Key set by memory capability via record_id
            value=summary_record,
            tags=all_tags,
            metadata={
                "transformer": "SummarizingTransformer",
                "source_count": len(entries),
                "source_keys": [e.key for e in entries],
                "prompt_used": self.prompt[:100],  # Truncate for metadata
            },
        )]


class FilteringTransformer(ConsolidationTransformer[TSource, TTarget]):
    """Filter memories based on criteria.

    Args:
        min_relevance: Minimum relevance score (0-1)
        required_tags: Tags that must be present
        excluded_tags: Tags that must NOT be present
    """

    def __init__(
        self,
        min_relevance: float = 0.0,
        required_tags: set[str] | None = None,
        excluded_tags: set[str] | None = None,
    ):
        self.min_relevance = min_relevance
        self.required_tags = required_tags or set()
        self.excluded_tags = excluded_tags or set()

    async def consolidate(
        self,
        entries: list[BlackboardEntry],
        context: ConsolidationContext,
    ) -> list[BlackboardEntry]:
        """Filter entries based on criteria."""
        from .types import MemoryRecord

        results = []

        for entry in entries:
            # Check relevance
            relevance = entry.metadata.get("relevance", 1.0)
            if relevance < self.min_relevance:
                continue

            # Check required tags
            if self.required_tags and not self.required_tags.issubset(entry.tags):
                continue

            # Check excluded tags
            if self.excluded_tags and self.excluded_tags.intersection(entry.tags):
                continue

            # Re-wrap serialized values to satisfy the contract
            value = entry.value
            if not hasattr(value, "record_id"):
                if isinstance(value, str):
                    value = MemoryRecord(content={"text": value}, tags=entry.tags)
                elif isinstance(value, dict):
                    value = MemoryRecord(content=value, tags=entry.tags)
                else:
                    raise TypeError(
                        f"FilteringTransformer: entry.value is "
                        f"{type(value).__name__}, expected BaseModel with "
                        f"record_id or dict/str for auto-wrapping."
                    )

            results.append(BlackboardEntry(
                key=entry.key,
                value=value,
                version=entry.version,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                created_by=entry.created_by,
                tags=entry.tags,
                metadata=entry.metadata,
            ))

        return results



# =============================================================================
# Memory Ingestion Policies
# =============================================================================


class MemoryIngestPolicyTrigger(ABC):
    """Policy that determines when to trigger a transfer between memory scopes.

    Transfer is not just periodic. Different policies enable:
    - Periodic: Regular consolidation intervals
    - Threshold: When enough data accumulates
    - On-demand: Explicit trigger only
    - TODO: Make it event-driven: When specific events occur
    """

    @abstractmethod
    async def should_transfer(
        self,
        source_entries: list[BlackboardEntry],
        last_transfer_time: float | None,
    ) -> bool:
        """Check if transfer should be triggered.

        Args:
            source_entries: Current entries in source scopes
            last_transfer_time: Timestamp of last transfer (None if never)

        Returns:
            True if transfer should be triggered
        """
        ...


class OnDemandMemoryIngestPolicyTrigger(MemoryIngestPolicyTrigger):
    """Transfer only when explicitly triggered.

    Use this for manual consolidation controlled by LLM or user.
    """

    async def should_transfer(
        self,
        source_entries: list[BlackboardEntry],
        last_transfer_time: float | None,
    ) -> bool:
        """Never auto-transfer; requires explicit trigger."""
        return False


class PeriodicMemoryIngestPolicyTrigger(MemoryIngestPolicyTrigger):
    """Transfer at regular intervals.

    Args:
        interval_seconds: Minimum seconds between transfers
    """

    def __init__(self, interval_seconds: float = 60.0):
        self.interval_seconds = interval_seconds

    async def should_transfer(
        self,
        source_entries: list[BlackboardEntry],
        last_transfer_time: float | None,
    ) -> bool:
        """Transfer if interval has elapsed and there are pending entries."""
        if last_transfer_time is None:
            return len(source_entries) > 0
        return time.time() - last_transfer_time >= self.interval_seconds


class ThresholdMemoryIngestPolicyTrigger(MemoryIngestPolicyTrigger):
    """Transfer when enough entries accumulate.

    Args:
        min_items: Minimum items before transfer
        max_items: Optional maximum (force transfer if exceeded)
    """

    def __init__(self, min_items: int = 10, max_items: int | None = None):
        self.min_items = min_items
        self.max_items = max_items

    async def should_transfer(
        self,
        source_entries: list[BlackboardEntry],
        last_transfer_time: float | None,
    ) -> bool:
        """Transfer if threshold is reached."""
        count = len(source_entries)
        if self.max_items and count >= self.max_items:
            return True
        return count >= self.min_items


class CompositeMemoryIngestPolicyTrigger(MemoryIngestPolicyTrigger):
    """Combine multiple policies with AND/OR logic.

    Args:
        policies: Policies to combine
        require_all: If True, all must agree (AND). If False, any (OR).
    """

    def __init__(
        self,
        policies: list[MemoryIngestPolicyTrigger] = [],
        require_all: bool = False
    ):
        self.policies = policies
        self.require_all = require_all

    async def should_transfer(
        self,
        source_entries: list[BlackboardEntry],
        last_transfer_time: float | None,
    ) -> bool:
        """Check all policies."""
        results = [
            await p.should_transfer(source_entries, last_transfer_time)
            for p in self.policies
        ]
        if self.require_all:
            return all(results)
        return any(results)


@dataclass
class MemoryIngestPolicy:
    """Policy that determines what subscriptions to use and how to consolidate the data when ingesting.
    """
    trigger: MemoryIngestPolicyTrigger = field(default_factory=OnDemandMemoryIngestPolicyTrigger)
    subscriptions: list[MemorySubscription] = field(default_factory=list)
    """Subscriptions to use for ingestion."""
    transformer: ConsolidationTransformer | None = field(default_factory=IdentityConsolidationTransformer)
    ingestion_check_interval_seconds: float = 30.0


# =============================================================================
# Utility Scorer Protocol
# =============================================================================


@runtime_checkable
class UtilityScorer(Protocol):
    """Scores memory utility for prioritization.

    Justification: <mark>Memory as active, self-optimizing</mark>—SEDM's
    empirical utility ranking. Memories that were useful should be kept;
    those that weren't should be forgotten.

    Implementations:
    - CompositeUtilityScorer: Weighted recency + importance + relevance + utility
    """

    async def score(
        self,
        entry: BlackboardEntry,
        context: dict[str, Any],
    ) -> float:
        """Score the utility of an entry.

        Args:
            entry: Entry to score
            context: Scoring context (current goal, recent actions, etc.)

        Returns:
            Utility score (0-1)
        """
        ...


# =============================================================================
# Default Implementations
# =============================================================================


class RecencyRetrieval:
    """Default retrieval strategy: routes by query type.

    Supports three query modes based on MemoryQuery fields:
    - Semantic (query.has_semantic): vector similarity via HybridStorageBackend
    - Logical (query.has_logical): tag/time/pattern filters via blackboard
    - Hybrid (both): semantic search + logical post-filtering
    - Neither: return all entries scored by recency

    When the backend is a HybridStorageBackend (has search_semantic),
    semantic queries are routed to ChromaDB. Otherwise, only logical
    queries are supported and semantic queries return empty results.
    """

    async def retrieve(
        self,
        query: "MemoryQuery",
        backend: StorageBackend,
        context: "RetrievalContext | None" = None,
    ) -> list["ScoredEntry"]:
        """Retrieve entries based on query type."""
        import time
        from .types import ScoredEntry

        # Route semantic queries to the vector backend if available
        if query.has_semantic and hasattr(backend, "search_semantic"):
            scored = await backend.search_semantic(
                query.query, query.max_results * 2,
            )

            # Hybrid: apply logical filters to semantic results
            if query.has_logical:
                scored = self._apply_logical_filters(scored, query)

            # Apply min_relevance threshold
            if query.min_relevance > 0:
                scored = [s for s in scored if s.score >= query.min_relevance]

            # Apply TTL/expiration filtering
            now = time.time()
            scored = self._apply_temporal_filters(scored, query, now)

            return scored[:query.max_results]

        # Logical query path (or fallback when no vector backend)
        # Resolve tags for the blackboard query (blackboard only supports all_of)
        effective_tags = None
        if not query.tag_filter.is_empty and query.tag_filter.all_of:
            effective_tags = query.tag_filter.all_of

        entries = await backend.query(
            pattern=query.key_pattern,
            tags=effective_tags,
            time_range=query.time_range,
            limit=query.max_results * 2,  # Over-fetch to allow filtering
        )

        now = time.time()
        scored: list[ScoredEntry] = []

        for entry in entries:
            # Apply any_of and none_of filters (blackboard only handles all_of)
            if not query.tag_filter.is_empty:
                if query.tag_filter.any_of and not query.tag_filter.any_of.intersection(entry.tags):
                    continue
                if query.tag_filter.none_of and query.tag_filter.none_of.intersection(entry.tags):
                    continue

            # Filter expired unless include_expired
            if not query.include_expired and entry.ttl_seconds:
                if now > entry.created_at + entry.ttl_seconds:
                    continue

            # Filter by age
            if query.max_age_seconds is not None:
                if now - entry.created_at > query.max_age_seconds:
                    continue

            # Score by recency (newer = higher score)
            age = now - entry.created_at
            # Exponential decay: e^(-age/3600) gives 1.0 at age=0, 0.37 at 1 hour
            recency_score = max(0.0, min(1.0, 2.718 ** (-age / 3600)))

            scored.append(ScoredEntry(
                entry=entry,
                score=recency_score,
                components={"recency": recency_score},
            ))

        # Sort by recency (most recent first)
        scored.sort(key=lambda s: s.score, reverse=True)

        return scored[:query.max_results]

    def _apply_logical_filters(
        self, scored: list["ScoredEntry"], query: "MemoryQuery",
    ) -> list["ScoredEntry"]:
        """Apply tag_filter constraints to pre-scored semantic results."""
        result = []
        for se in scored:
            if not query.tag_filter.matches(se.entry.tags):
                continue
            result.append(se)
        return result

    def _apply_temporal_filters(
        self, scored: list["ScoredEntry"], query: "MemoryQuery", now: float,
    ) -> list["ScoredEntry"]:
        """Apply TTL and age filters to scored entries."""
        result = []
        for se in scored:
            entry = se.entry
            if not query.include_expired and entry.ttl_seconds:
                if now > entry.created_at + entry.ttl_seconds:
                    continue
            if query.max_age_seconds is not None:
                if now - entry.created_at > query.max_age_seconds:
                    continue
            result.append(se)
        return result


class TTLMaintenancePolicy:
    """Maintenance policy that removes expired entries.

    Simple and predictable. Runs every interval and removes entries
    whose TTL has expired.
    """

    def __init__(self, check_interval_seconds: float = 60.0):
        self.check_interval = check_interval_seconds

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Run if interval has elapsed."""
        import time
        if last_run is None:
            return True
        return time.time() - last_run >= self.check_interval

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Remove expired entries."""
        import time
        from .types import MaintenanceResult

        start = time.time()
        entries = await backend.query(limit=10000)

        now = time.time()
        removed = 0

        for entry in entries:
            if entry.ttl_seconds is not None:
                if now > entry.created_at + entry.ttl_seconds:
                    if await backend.delete(entry.key):
                        removed += 1

        return MaintenanceResult(
            entries_processed=len(entries),
            entries_removed=removed,
            entries_modified=0,
            duration_seconds=time.time() - start,
        )


class CapacityMaintenancePolicy:
    """Maintenance policy that enforces max_entries limit.

    When over capacity, removes oldest entries (by created_at).
    """

    def __init__(self, max_entries: int, check_interval_seconds: float = 60.0):
        self.max_entries = max_entries
        self.check_interval = check_interval_seconds

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Run if interval has elapsed."""
        import time
        if last_run is None:
            return True
        return time.time() - last_run >= self.check_interval

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Remove oldest entries if over capacity."""
        import time
        from .types import MaintenanceResult

        start = time.time()
        count = await backend.count()

        if count <= self.max_entries:
            return MaintenanceResult(
                entries_processed=0,
                entries_removed=0,
                duration_seconds=time.time() - start,
            )

        # Need to remove excess
        num_to_remove = count - self.max_entries
        entries = await backend.query(limit=count)

        # Sort by created_at (oldest first)
        entries.sort(key=lambda e: e.created_at)

        removed = 0
        for entry in entries[:num_to_remove]:
            if await backend.delete(entry.key):
                removed += 1

        return MaintenanceResult(
            entries_processed=len(entries),
            entries_removed=removed,
            entries_modified=0,
            duration_seconds=time.time() - start,
        )


class DecayMaintenancePolicy:
    """Maintenance policy that applies temporal decay to reduce relevance scores over time.

    Implements exponential decay: relevance *= (1 - decay_rate)^minutes

    Reduces the relevance/salience of memories over time based on
    the configured decay_rate. This implements cognitive forgetting
    where older, less-accessed memories fade.

    Attributes:
        decay_rate: Decay rate per minute (e.g., 0.01 = 1%)
        decay_min: Floor value for relevance (won't decay below this)
        check_interval_seconds: How often to run
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        decay_min: float = 0.0,
        check_interval_seconds: float = 60.0,
    ):
        self.decay_rate = decay_rate
        self.decay_min = decay_min
        self.check_interval = check_interval_seconds

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Run if interval has elapsed."""
        if self.decay_rate <= 0:
            return False

        import time
        if last_run is None:
            return True
        return time.time() - last_run >= self.check_interval

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Apply decay to relevance scores."""
        import time
        from .types import MaintenanceResult

        start = time.time()
        entries = await backend.query(limit=10000)  # TODO: Make configurable

        now = time.time()
        modified = 0

        for entry in entries:
            age_minutes = (now - entry.updated_at) / 60.0
            relevance = entry.metadata.get("relevance", 1.0)

            # Exponential decay
            new_relevance = max(
                self.decay_min,
                relevance * (1 - self.decay_rate) ** age_minutes
            )

            if new_relevance < relevance:
                entry.metadata["relevance"] = new_relevance
                entry.metadata["last_decay"] = now
                await backend.write(
                    key=entry.key,
                    value=entry.value,
                    metadata=entry.metadata,
                    tags=entry.tags,
                )
                modified += 1

        logger.debug(f"Decayed {modified} memories")
        return MaintenanceResult(
            entries_processed=len(entries),
            entries_removed=0,
            entries_modified=modified,
            duration_seconds=time.time() - start,
        )



class PruneMaintenancePolicy:
    """Maintenance policy that removes memories below the relevance threshold.

    Pruning removes memories that have decayed below a threshold
    or are no longer useful. This prevents memory bloat.
    """
    def __init__(self, prune_threshold: float = 0.1, check_interval_seconds: float = 60.0):
        self.prune_threshold = prune_threshold
        self.check_interval = check_interval_seconds

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Run if interval has elapsed."""
        if self.prune_threshold <= 0:
            return False

        import time
        if last_run is None:
            return True
        return time.time() - last_run >= self.check_interval

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Remove memories below the relevance threshold.

        Pruning removes memories below a relevance threshold
        that are no longer useful. This prevents memory bloat.

        Returns:
            MaintenanceResult with counts of processed/removed/modified entries
        """
        import time
        from .types import MaintenanceResult

        start = time.time()
        entries = await backend.query(limit=10000)  # TODO: Make configurable

        now = time.time()
        removed = 0

        for entry in entries:
            relevance = entry.metadata.get("relevance", 1.0)

            if relevance < self.prune_threshold:
                if await backend.delete(entry.key):
                    removed += 1

        logger.debug(f"Pruned {removed} memories")
        return MaintenanceResult(
            entries_processed=len(entries),
            entries_removed=removed,
            entries_modified=0,
            duration_seconds=time.time() - start,
        )


class DeduplicationMaintenancePolicy:
    """Maintenance policy that deduplicates memories. This prevents memory bloat.

    TODO: Implement semantic deduplication using embeddings.
    Currently a no-op placeholder.
    """

    def __init__(self, deduplication_threshold: float = 0.95, check_interval_seconds: float = 60.0):
        self.deduplication_threshold = deduplication_threshold
        self.check_interval = check_interval_seconds

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Run if interval has elapsed."""
        if self.deduplication_threshold <= 0:
            return False
        if last_run is None:
            return True
        return time.time() - last_run >= self.check_interval

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Deduplicate memories.

        TODO: Implement semantic deduplication using embeddings.
        For now, this is a no-op placeholder.
        """
        from .types import MaintenanceResult

        start = time.time()
        # TODO: Implement actual deduplication
        # 1. Get all entries
        # 2. Compute embeddings (or use cached)
        # 3. Find pairs above similarity threshold
        # 4. Merge duplicates (keep most recent, combine metadata)
        # 5. Delete duplicates

        logger.debug(f"Deduplication requested for {backend.scope_id} (not yet implemented)")
        return MaintenanceResult(
            entries_processed=0,
            entries_removed=0,
            entries_modified=0,
            duration_seconds=time.time() - start,
        )


class ConsolidationMaintenancePolicy:
    """Maintenance policy that consolidates entries within a scope.

    This handles INTRA_SCOPE consolidation as a maintenance task:
    1. Query entries from the scope
    2. Apply transformer to create consolidated entries
    3. Write consolidated entries back
    4. Delete originals

    Use this for within-scope consolidation (summarization, abstraction).
    For inter-scope transfer, the receiving scope should subscribe and
    configure an ingestion_transformer.

    Args:
        transformer: ConsolidationTransformer to apply
        threshold: Min entries before consolidation runs
        check_interval_seconds: How often to check
        delete_originals: Whether to delete source entries after consolidation
    """

    def __init__(
        self,
        transformer: "ConsolidationTransformer",
        threshold: int = 10,
        check_interval_seconds: float = 300.0,
        delete_originals: bool = True,
    ):
        self.transformer = transformer
        self.threshold = threshold
        self.check_interval = check_interval_seconds
        self.delete_originals = delete_originals

    async def should_run(
        self,
        backend: StorageBackend,
        last_run: float | None,
    ) -> bool:
        """Run if interval has elapsed and enough entries exist."""
        if last_run is not None and time.time() - last_run < self.check_interval:
            return False
        count = await backend.count()
        return count >= self.threshold

    async def execute(
        self,
        backend: StorageBackend,
    ) -> "MaintenanceResult":
        """Consolidate entries within the scope."""
        from .types import MaintenanceResult, ConsolidationContext

        start = time.time()
        entries = await backend.query(limit=self.threshold)

        if not entries:
            return MaintenanceResult(
                entries_processed=0,
                entries_removed=0,
                entries_modified=0,
                duration_seconds=time.time() - start,
            )

        # Apply transformer
        ctx = ConsolidationContext(
            source_scope=backend.scope_id,
            target_scope=backend.scope_id,
        )
        consolidated = await self.transformer.consolidate(entries, ctx)

        # Write consolidated entries
        written = 0
        for entry in consolidated:
            data = entry.value
            key = MemoryRecordProtocol.consolidated_key(int(time.time()), written)

            value = data.model_dump() if hasattr(data, "model_dump") else data

            await backend.write(
                key=key,
                value=value,
                metadata={
                    **entry.metadata,
                    "consolidated_at": time.time(),
                },
                tags=entry.tags,
            )
            written += 1

        # Delete originals
        removed = 0
        if self.delete_originals:
            for entry in entries:
                if await backend.delete(entry.key):
                    removed += 1

        logger.debug(
            f"Consolidated {len(entries)} → {written} entries in {backend.scope_id}"
        )
        return MaintenanceResult(
            entries_processed=len(entries),
            entries_removed=removed,
            entries_modified=written,
            duration_seconds=time.time() - start,
        )


class IdentityConsolidationTransformer:
    """No-op consolidation transformer that passes entries through unchanged.

    Use this for simple memory forwarding without modification.
    """

    async def consolidate(
        self,
        entries: list[BlackboardEntry],
        context: ConsolidationContext,
    ) -> list[BlackboardEntry]:
        """Pass through unchanged."""
        return entries
