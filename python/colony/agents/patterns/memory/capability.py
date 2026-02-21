"""Unified Memory Capability

The `MemoryCapability` is a container for a memory level (scope) that:
- Manages ONE scope (a memory level in the dataflow graph)
- INGESTS data from sources (subscriptions to other scopes + hook-based producers)
- Transforms ALL pending entries together via `ingestion_policy.transformer`
- MAINTAINS the scope to keep the memory level healthy (decay, prune, dedupe, within-scope consolidation): Background (subconscious) cognitive memory processes
- Exposes LLM-plannable actions (store, recall, forget, transfer_raw)
- Provides query interface for recall

A memory level is a scope (or namespace in the storage backend) and it is a "node" in the memory dataflow graph.

Architecture (Pull Model):
    If CapabilityB wants data from CapabilityA's scope:
    → CapabilityB subscribes to CapabilityA's scope
    → CapabilityB's ingestion transformer consolidates the data
    → CapabilityA's maintenance (TTL, capacity) cleans up old entries

    Each capability only manages its own scope. No "push" logic.


This enables:
- STM → LTM consolidation (summarize recent memories)
- Perception → STM ingestion (filter and annotate raw events)
- Cross-agent memory sharing (transfer memories to shared scope)

The memory dataflow graph:
- Nodes = Memory scopes (working, STM, LTM-episodic, etc.)
- Edges = Subscriptions (a capability subscribes to another scope)

Design justifications from research:
- Memory as active, self-optimizing → background maintenance + utility scoring
- Hook-based formation → producers (memory observes agent behavior)
- Evolution, not just storage → ingestion_policy.transformer consolidates inputs
- Retrieval is the bottleneck → pluggable RetrievalStrategy
- Forgetting is heuristic → pluggable MaintenancePolicy list
- Transformation is task-specific → pluggable ConsolidationTransformer
- Lenses for different contexts → predefined MemoryLens views

Example:
    ```python
    from polymathera.colony.agents.patterns.memory import (
        MemoryCapability,
        MemoryScope,
        MemorySubscription,
        MemoryProducerConfig,
        SummarizingTransformer,
    )
    from polymathera.colony.agents.patterns.hooks import Pointcut

    # STM subscribes to working memory and consolidates inputs
    stm = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_stm(agent_id),
        ingestion_policy=MemoryIngestPolicy(
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_working(agent_id)),
            ],
            trigger=PeriodicMemoryIngestPolicyTrigger(
                interval_seconds=120.0,  # Every 2 minutes
            ),
            transformer=SummarizingTransformer(
                agent=agent,
                prompt="Summarize recent actions into a coherent narrative for STM.",
            ),
        ),
        ttl_seconds=86400,  # 1 day
        max_entries=500,
    )

    # Working memory with hook-based capture (no extractor = store as-is)
    working = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_working(agent_id),
        ttl_seconds=3600,  # 1 hour
        max_entries=50,
        producers=[
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
                extractor=extract_action_from_dispatch,
                ttl_seconds=3600,  # 1 hour
            ),
        ],
    )

    await stm.initialize()
    await working.initialize()
    agent.add_capability(stm)
    agent.add_capability(working)

    # Store a memory explicitly (data instance provides its own key via `get_blackboard_key`)
    observation = Observation(content="User asked about auth", timestamp=time.time())
    await working.store(observation)

    # Recall memories
    memories = await stm.recall(MemoryQuery(tags={"authentication"}))
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from overrides import override

from pydantic import BaseModel

from ...base import AgentCapability, CapabilityResultFuture
from ...models import AgentSuspensionState
from ...blackboard.types import BlackboardEntry, BlackboardEvent
from ....vcm.models import MmapConfig
from ....vcm.sources import BuilInContextPageSourceType
from ..actions.policies import action_executor
from ..hooks.types import HookContext, HookType, ErrorMode
from .types import (
    MemoryQuery,
    MemorySubscription,
    MaintenanceConfig,
    MemoryProducerConfig,
    RetrievalContext,
    ScoredEntry,
    ConsolidationContext,
    MemoryLens,
)
from .protocols import (
    StorageBackend,
    StorageBackendFactory,
    RetrievalStrategy,
    MaintenancePolicy,
    ConsolidationTransformer,
    UtilityScorer,
    RecencyRetrieval,
    TTLMaintenancePolicy,
    CapacityMaintenancePolicy,
    DecayMaintenancePolicy,
    PruneMaintenancePolicy,
    DeduplicationMaintenancePolicy,
    MemoryIngestPolicy,
    OnDemandMemoryIngestPolicyTrigger,
)
from .backends import BlackboardStorageBackend, BlackboardStorageBackendFactory

if TYPE_CHECKING:
    from ...base import Agent


# =============================================================================
# Pending Entry with Subscription Info
# =============================================================================


@dataclass
class PendingEntry:
    """Entry pending transfer with its subscription info.

    Used to track entries collected from subscriptions before
    the transfer policy triggers consolidation.
    """

    entry: BlackboardEntry
    """The raw entry from the source scope."""

    subscription: MemorySubscription
    """The subscription that collected this entry."""

    collected_at: float
    """Timestamp when the entry was collected."""



logger = logging.getLogger(__name__)


class MemoryCapability(AgentCapability):
    """Unified memory capability: ingestion + storage + maintenance (cognitive processes).

    A MemoryCapability manages ONE memory scope. It:
    1. INGESTS data from sources (subscriptions + producers)
        - Zero or more producers (hook-based capture of memory-producing methods)
        - Zero or more subscriptions (pull/push from other scopes)
    2. Transforms ALL inputs together via `ingestion_policy.transformer`
    3. Writes transformed data to THIS scope
    4. MAINTAINS this scope in background cognitive processes (decay, prune, dedupe, consolidation)
    5. Provides conscious LLM-plannable actions (store, recall, forget, transfer)
    6. Provides query interface for recall

    Pull model: if another scope wants data from this scope, it subscribes.
    No push/export logic. Each capability manages its own scope.

    This pull model (observer pattern) is motivated by evolutionary history of the brain.
    - A new memory layer evolves to observe and consolidate data from existing layers.
    - An existing memory layer does not have to be aware of new layers, which makes the memory system more extensible.

    The `producers` parameter enables hook-based memory capture: the memory
    capability registers AFTER hooks on specified methods to automatically
    capture their outputs. This implements the principle: "memory is an
    observer of agent behavior."

    Attributes:
        scope_id: Blackboard scope for this memory level
        producers: Hook-based memory capture configs
        ingestion_policy: Ingestion policy including memory scopes to
            listen to and pull data from, and when to trigger ingestion
            (default: OnDemand) and how to consolidate the entries.
        ttl_seconds: Default TTL for stored memories
        max_entries: Maximum entries before eviction
        retrieval: Strategy for memory retrieval
        lenses: Named query configurations
        maintenance_policies: List of maintenance policies
        utility_scorer: Scorer for memory utility (SEDM-inspired)
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

        # === VCM MAPPING ===
        map_to_vcm: bool = False,
        vcm_config: MmapConfig | None = None,
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

            map_to_vcm: If True, map this scope into VCM pages during initialize().
                Enables attention-based discovery of this scope's contents.
            vcm_config: Configuration for VCM mapping (controls flushing, locality, etc.)
                Only used if map_to_vcm=True. Defaults to MmapConfig() if None.
        """
        super().__init__(agent=agent, scope_id=scope_id)

        # Ingestion: sources and transformation
        self.producers = producers or []

        # Storage
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._storage_backend = None  # Resolved in initialize()
        self._storage_backend_factory = storage_backend_factory  # Resolved in initialize()

        # Retrieval
        self._retrieval_strategy = retrieval_strategy or RecencyRetrieval()
        self._lenses: dict[str, MemoryLens] = {lens.name: lens for lens in (lenses or [])}

        # Maintenance
        self._maintenance_policies = maintenance_policies
        self._maintenance_interval = maintenance_interval_seconds
        self._legacy_maintenance = maintenance  # For backwards compat

        # VCM mapping
        self.map_to_vcm = map_to_vcm
        self.vcm_config = vcm_config

        # Ingestion policy (when to process pending entries)
        self._ingestion_policy = ingestion_policy or MemoryIngestPolicy()

        # Utility
        self._utility_scorer = utility_scorer

        # Pending entries from subscriptions (collected, not yet transformed)
        self._pending_entries: list[PendingEntry] = []

        # Background task handles
        self._maintenance_task: asyncio.Task | None = None
        self._subscription_tasks: list[asyncio.Task] = []
        self._ingestion_task: asyncio.Task | None = None  # Checks policy, triggers ingestion
        self._producer_hook_ids: list[str] = []  # Track registered hooks for cleanup
        self._running = False
        self._initialized = False

        # Track last maintenance run times
        self._last_maintenance_run: dict[int, float] = {}
        self._last_ingestion_time: float | None = None

    @property
    def storage(self) -> StorageBackend:
        """Storage backend for this capability.

        Raises:
            RuntimeError: If accessed before initialize()
        """
        if self._storage_backend is None:
            raise RuntimeError(
                f"MemoryCapability {self.scope_id} not initialized. "
                f"Call initialize() first."
            )
        return self._storage_backend

    @property
    def retrieval(self) -> RetrievalStrategy:
        """Retrieval strategy for this capability."""
        return self._retrieval_strategy

    @property
    def lenses(self) -> dict[str, MemoryLens]:
        """Named query configurations."""
        return self._lenses

    @property
    def maintenance_policies(self) -> list[MaintenancePolicy]:
        """Active maintenance policies."""
        if self._maintenance_policies is not None:
            return self._maintenance_policies

        # Build from legacy config or defaults
        policies: list[MaintenancePolicy] = []
        if self.ttl_seconds:
            policies.append(TTLMaintenancePolicy())
        if self.max_entries:
            policies.append(CapacityMaintenancePolicy(self.max_entries))
        return policies

    @property
    def utility_scorer(self) -> UtilityScorer | None:
        """Utility scorer for memory prioritization."""
        return self._utility_scorer

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize storage backend and start background processes.

        - Resolves storage backend factory (defaults to BlackboardStorageBackendFactory)
        - Resolves storage backend (defaults to BlackboardStorageBackend)
        - Registers producer hooks for hook-based memory capture
        - Starts background maintenance task
        - Starts subscription listeners (collect into pending entries)
        - Starts ingestion task (checks policy, processes pending entries)
        """
        if self._initialized:
            return

        # Resolve storage backend factory (for accessing other scopes)
        if self._storage_backend_factory is None:
            self._storage_backend_factory = BlackboardStorageBackendFactory(self.agent)

        # Resolve storage backend for THIS scope
        if self._storage_backend is None:
            self._storage_backend = await self._storage_backend_factory.create_for_scope(
                self.scope_id
            )

        # Register producer hooks (Hook-based formation)
        await self._register_producer_hooks()

        # Start background tasks
        self._running = True

        # Maintenance task (decay, prune, dedupe, within-scope consolidation)
        if self.maintenance_policies:
            self._maintenance_task = asyncio.create_task(
                self._run_maintenance_loop()
            )

        # Subscription tasks: collect entries into _pending_entries
        for subscription in self._ingestion_policy.subscriptions:
            task = asyncio.create_task(
                self._run_subscription(subscription)
            )
            self._subscription_tasks.append(task)

        # Ingestion task: checks policy, processes pending entries
        # Only run if we have subscriptions and not on-demand policy
        if self._ingestion_policy.subscriptions and not isinstance(self._ingestion_policy.trigger, OnDemandMemoryIngestPolicyTrigger):
            self._ingestion_task = asyncio.create_task(
                self._run_ingestion_loop()
            )

        # VCM mapping: request VCM to page this scope's contents
        if self.map_to_vcm:
            try:
                from ....system import get_vcm

                vcm_handle = get_vcm()
                # TODO: This only works if the memory storage backend is BlackboardStorageBackend. Generalize?
                result = await vcm_handle.mmap_application_scope(
                    scope_id=self.scope_id,
                    group_id=self._agent.group_id,
                    tenant_id=self._agent.tenant_id,
                    source_type=BuilInContextPageSourceType.BLACKBOARD.value,
                    config=self.vcm_config or MmapConfig(),
                )
                logger.info(
                    f"MemoryCapability[{self.scope_id}]: VCM mapping status={result.status}"
                )
            except Exception as e:
                logger.warning(
                    f"MemoryCapability[{self.scope_id}]: failed to map to VCM: {e}"
                )

        self._initialized = True
        logger.info(
            f"MemoryCapability initialized: scope={self.scope_id}, "
            f"subscriptions={len(self._ingestion_policy.subscriptions)}, "
            f"producers={len(self.producers)}"
        )

    async def shutdown(self) -> None:
        """Stop background tasks and cleanup hooks."""
        self._running = False

        # Cancel background tasks
        all_tasks = self._subscription_tasks + [
            t for t in [
                self._maintenance_task,
                self._ingestion_task,
            ]
            if t is not None
        ]

        for task in all_tasks:
            task.cancel()

        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
            # TODO: Handle exceptions. Check if it is just asyncio.CancelledError.

        self._subscription_tasks = []
        self._maintenance_task = None
        self._ingestion_task = None

        # Clear pending entries
        self._pending_entries = []

        # Unregister hooks
        if self._producer_hook_ids:
            hook_registry = self.agent.hooks
            for hook_id in self._producer_hook_ids:
                hook_registry.unregister(hook_id)
            self._producer_hook_ids = []

        self._initialized = False
        logger.info(f"MemoryCapability shutdown: scope={self.scope_id}")

    # =========================================================================
    # AgentCapability Abstract Methods
    # =========================================================================

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for MemoryCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for MemoryCapability")
        pass

    @override
    async def stream_events_to_queue(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
    ) -> None:
        """Stream memory events from this scope to the given queue.

        Streams all write events from this memory capability's scope.

        Args:
            event_queue: Queue to stream events to
        """
        await self.storage.stream_events_to_queue(event_queue, f"{self.scope_id}:*")

    @override
    async def get_result_future(self) -> CapabilityResultFuture:
        """Memory capabilities are persistent services without a single result.

        Raises:
            NotImplementedError: Always, use stream_events_to_queue() instead
        """
        raise NotImplementedError(
            "MemoryCapability is a persistent service without a single result. "
            "Use stream_events_to_queue() to monitor memory changes."
        )

    # =========================================================================
    # CONSCIOUS ACTIONS (LLM-Plannable via @action_executor)
    # =========================================================================

    @action_executor(action_key="memory_store")
    async def store(
        self,
        data: BaseModel,
        tags: set[str] | None = None,
        ttl_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory explicitly (conscious write).

        Called by the agent action policy when it decides to deliberately store something.
        This action is exposed to the agent action policy for explicit memory storage.
        The agent might want to store insights, conclusions, or important
        observations for later retrieval.
        The data instance MUST have `get_blackboard_key(scope_id)` method.

        Args:
            data: Memory data to store (Pydantic model with get_blackboard_key)
            tags: Tags for categorization and retrieval
            ttl_seconds: TTL override (uses level default if None)
            metadata: Additional metadata

        Returns:
            Key under which the memory was stored
        """
        if not hasattr(data, "get_blackboard_key"):
            raise ValueError(
                f"Data instance of type {type(data).__name__} must have "
                f"get_blackboard_key(scope_id) instance method"
            )

        key = data.get_blackboard_key(self.scope_id)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

        # Serialize to dict
        value = data.model_dump() if hasattr(data, "model_dump") else dict(data)

        # Build metadata (session_id is auto-added by EnhancedBlackboard.write())
        entry_metadata = {
            "data_type": type(data).__name__,
            "relevance": 1.0,  # Initial relevance
            **(metadata or {}),
        }

        await self.storage.write(
            key=key,
            value=value,
            agent_id=self.agent.agent_id,
            metadata=entry_metadata,
            tags=tags,
            ttl_seconds=effective_ttl,
        )

        # Check if we need to evict
        if self.max_entries:
            await self._maybe_evict()

        logger.debug(f"MemoryCapability: stored {key}")
        return key

    @action_executor(action_key="memory_recall_with_scores")
    async def recall_with_scores(
        self,
        query: MemoryQuery | None = None,
        lens: str | None = None,
        context: RetrievalContext | None = None,
    ) -> list[ScoredEntry]:
        """Recall memories (conscious read) with detailed scoring for debugging/analysis.

        This action is exposed to the agent action policy for memory retrieval. The agent
        can search for relevant memories using semantic similarity, tags,
        or recency filtering.

        Supports:
        - Query-based filtering (tags, recency, etc.)
        - Lens-based views (predefined query configurations)
        - Goal-aware retrieval via RetrievalContext

        Args:
            query: Query parameters for filtering/ranking
            lens: Name of a predefined lens to apply
            context: Retrieval context (goal, agent state) for relevance

        Returns:
            List of matching ScoredEntry objects with score breakdowns
        """
        effective_query = query or MemoryQuery()

        # Apply lens if specified
        if lens and lens in self._lenses:
            effective_query = self._apply_lens(self._lenses[lens], effective_query)

        # Use retrieval strategy
        scored_entries = await self.retrieval.retrieve(
            query=effective_query,
            backend=self.storage,
            context=context,
        )

        # Track access if configured
        if self._legacy_maintenance and self._legacy_maintenance.track_access:
            for se in scored_entries[:effective_query.max_results]:
                await self._track_access(se.entry.key)

        return scored_entries

    @action_executor(action_key="memory_recall")
    async def recall(
        self,
        query: MemoryQuery | None = None,
        lens: str | None = None,
        context: RetrievalContext | None = None,
    ) -> list[BlackboardEntry]:
        """Recall memories (conscious read).

        Args:
            query: Query parameters
            lens: Name of a predefined lens to apply
            context: Retrieval context (goal, agent state) for relevance

        Returns:
            List of matching BlackboardEntry objects
        """
        scored_entries = await self.recall_with_scores(
            query=query,
            lens=lens,
            context=context,
        )
        return [se.entry for se in scored_entries]

    @action_executor(action_key="memory_forget")
    async def forget(
        self,
        keys: list[str] | None = None,
        tags: set[str] | None = None,
        older_than_seconds: float | None = None,
    ) -> int:
        """Forget memories (conscious deletion).

        This action is exposed to the agent action policy for explicit memory deletion.
        The agent might want to forget outdated, incorrect, or irrelevant memories or
        memories that are no longer relevant to the current task to maintain a clean memory state.

        Args:
            keys: Specific keys to delete
            tags: Delete entries matching any of these tags
            older_than_seconds: Delete entries older than this

        Returns:
            Number of memories forgotten
        """
        count = 0

        # Delete by keys
        if keys:
            for key in keys:
                if await self.storage.delete(key):
                    count += 1

        # Delete by tags or age
        if tags or older_than_seconds is not None:
            entries = await self.storage.query(
                limit=1000,  # TODO: Make configurable
            )
            now = time.time()

            for entry in entries:
                should_delete = False

                # Check tags
                if tags and entry.tags.intersection(tags):
                    should_delete = True

                # Check age
                if older_than_seconds is not None:
                    if now - entry.created_at > older_than_seconds:
                        should_delete = True

                if should_delete:
                    if await self.storage.delete(entry.key):
                        count += 1

        logger.debug(f"MemoryCapability: forgot {count} memories from {self.scope_id}")
        return count

    @action_executor(action_key="memory_deduplicate")
    async def deduplicate(self, similarity_threshold: float = 0.95) -> int:
        """Deduplicate similar memories (conscious maintenance).

        TODO: Implement semantic deduplication using embeddings.

        This finds memories with high similarity and merges them,
        keeping the most recent and combining metadata.

        Args:
            similarity_threshold: Threshold for considering entries duplicates

        Returns:
            Number of duplicates removed
        """
        dedup_policy = DeduplicationMaintenancePolicy(
            deduplication_threshold=similarity_threshold,
        )
        result = await dedup_policy.execute(self.storage)
        logger.debug(f"Deduplicated {result.entries_removed} memories from {self.scope_id}")
        return result.entries_removed

    @action_executor(action_key="memory_prune")
    async def prune(self, relevance_threshold: float = 0.1) -> int:
        """Remove memories below the relevance threshold.

        Pruning removes memories that have decayed below a threshold
        or are no longer useful. This prevents memory bloat.

        Args:
            relevance_threshold: Minimum relevance score to keep (default 0.1)

        Returns:
            Number of memories pruned
        """
        prune_policy = PruneMaintenancePolicy(
            prune_threshold=relevance_threshold,
        )
        result = await prune_policy.execute(self.storage)
        logger.debug(f"Pruned {result.entries_removed} memories from {self.scope_id}")
        return result.entries_removed

    @action_executor(action_key="memory_transfer_raw")
    async def transfer_to(
        self,
        target_scope: str,
        query: MemoryQuery | None = None,
        delete_after: bool = False,
    ) -> int:
        """Transfer memories to another scope WITHOUT transformation (raw copy/move).

        This is a conscious action for the LLM to explicitly copy/move memories
        between scopes without any transformation. Use cases:
        - Archiving memories to a backup scope
        - Sharing memories with other agents (cross-agent sharing)
        - Manual organization of memories

        Note: For transformation (summarization, abstraction, merging), the
        receiving scope should configure an ingestion_policy.transformer.

        Args:
            target_scope: Scope to write transferred memories to
            query: Query to select which memories to transfer
            delete_after: Whether to delete originals after transfer (move vs copy)

        Returns:
            Number of entries transferred
        """
        entries = await self.recall(query=query)

        if not entries:
            return 0

        # Create storage backend for target scope via factory
        target_storage = await self._storage_backend_factory.create_for_scope(target_scope)

        count = 0
        for entry in entries:
            # Generate new key for target scope
            data = entry.value
            if hasattr(data, "get_blackboard_key"):
                new_key = data.get_blackboard_key(target_scope)
            else:
                # Fallback: replace scope prefix in key
                new_key = entry.key.replace(self.scope_id, target_scope, 1)

            # Build transfer metadata (session_id is auto-added by EnhancedBlackboard.write())
            await target_storage.write(
                key=new_key,
                value=entry.value,
                metadata={
                    **entry.metadata,
                    "transferred_from": self.scope_id,
                    "transferred_at": time.time(),
                },
                tags=entry.tags,
            )
            count += 1

        if delete_after:
            await self.forget(keys=[e.key for e in entries])

        logger.info(
            f"MemoryCapability: raw-transferred {count} entries "
            f"from {self.scope_id} to {target_scope}"
        )
        return count

    @action_executor(action_key="memory_ingest_now")
    async def ingest_now(self) -> int:
        """Process pending entries from subscriptions (on-demand ingestion).

        When using OnDemandMemoryIngestPolicyTrigger (the default), this action allows
        the LLM to explicitly trigger processing of pending entries.

        Flow:
        1. Collect ALL pending entries from ALL subscriptions
        2. Apply ingestion_policy.transformer to ALL entries together
        3. Write transformed entries to THIS scope

        For automatic processing, use a different ingestion_policy
        (e.g., PeriodicMemoryIngestPolicyTrigger, ThresholdMemoryIngestPolicyTrigger).

        Returns:
            Number of entries written after processing
        """
        return await self._execute_ingestion()

    # =========================================================================
    # UTILITY FEEDBACK (Memory as self-optimizing)
    # =========================================================================

    async def register_retrieval_utility(
        self,
        entry_key: str,
        utility: float,
    ) -> None:
        """Record utility feedback for a retrieved memory.

        Justification: SEDM's empirical utility ranking—memories
        that were useful should be kept; those that weren't should decay.

        Args:
            entry_key: Key of the entry to update
            utility: Utility score (0-1) based on whether the memory was helpful
        """
        entry = await self.storage.read(entry_key)
        if entry is None:
            return

        # Exponential moving average for utility
        old_utility = entry.metadata.get("retrieval_utility", 0.5)
        entry.metadata["retrieval_utility"] = old_utility * 0.9 + utility * 0.1
        entry.metadata["access_count"] = entry.metadata.get("access_count", 0) + 1
        entry.metadata["last_accessed_at"] = time.time()

        # session_id is auto-added by EnhancedBlackboard.write()
        await self.storage.write(
            key=entry.key,
            value=entry.value,
            metadata=entry.metadata,
            tags=entry.tags,
        )

    # =========================================================================
    # SUBCONSCIOUS PROCESSES (Background Tasks)
    # =========================================================================

    async def _run_subscription(self, subscription: MemorySubscription) -> None:
        """Background task to collect entries from a subscription into pending entries.

        This method does NOT transform immediately. It collects entries into
        `_pending_entries` and waits for the transfer policy to trigger
        processing via `_run_transfer_loop()`.

        """
        event_queue: asyncio.Queue[BlackboardEvent] = asyncio.Queue()

        # Get key pattern from data type
        if subscription.data_type and hasattr(subscription.data_type, "get_key_pattern"):
            pattern = subscription.data_type.get_key_pattern(subscription.source_scope_id)
        else:
            # Fallback for data types without get_key_pattern
            pattern = f"{subscription.source_scope_id}:*"

        # Get storage backend for source scope and subscribe to events
        # The StorageBackend contract requires all writes to emit events.
        source_storage = await self._storage_backend_factory.create_for_scope(
            subscription.source_scope_id
        )
        await source_storage.stream_events_to_queue(event_queue, pattern)

        logger.debug(
            f"MemoryCapability {self.scope_id}: subscribed to "
            f"{subscription.data_type.__name__ if subscription.data_type else 'all'} "
            f"from {subscription.source_scope_id}"
        )

        # Collect events into pending entries (NO transformation yet)
        while self._running:
            try:
                event = await event_queue.get()

                if event.event_type != "write":
                    continue

                # Create entry from event
                entry = BlackboardEntry(
                    key=event.key,
                    value=event.value,
                    version=event.version,
                    created_at=event.timestamp,
                    updated_at=event.timestamp,
                    created_by=event.agent_id,
                    tags=event.tags,
                    metadata={
                        **event.metadata,
                        "source_scope": subscription.source_scope_id,
                        "source_type": (
                            subscription.data_type.__name__
                            if subscription.data_type else "unknown"
                        ),
                    },
                )

                # Apply filters
                if subscription.filters:
                    if not all(f(entry) for f in subscription.filters):
                        continue

                # Add to pending entries (ingestion_policy.transformer applied later)
                self._pending_entries.append(PendingEntry(
                    entry=entry,
                    subscription=subscription,
                    collected_at=time.time(),
                ))

                logger.debug(
                    f"MemoryCapability {self.scope_id}: collected pending entry "
                    f"from {subscription.source_scope_id}, total pending: {len(self._pending_entries)}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Error collecting subscription event in {self.scope_id}: {e}"
                )

    async def _run_maintenance_loop(self) -> None:
        """Background task for periodic maintenance.

        Runs each configured maintenance policy at its configured interval.
        """
        while self._running:
            try:
                await asyncio.sleep(self._maintenance_interval)

                for i, policy in enumerate(self.maintenance_policies):
                    last_run = self._last_maintenance_run.get(i)

                    if await policy.should_run(self.storage, last_run):
                        result = await policy.execute(self.storage)
                        self._last_maintenance_run[i] = time.time()

                        if result.entries_removed > 0 or result.entries_modified > 0:
                            logger.debug(
                                f"MemoryCapability {self.scope_id}: "
                                f"maintenance policy {type(policy).__name__} "
                                f"removed={result.entries_removed}, "
                                f"modified={result.entries_modified}"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop for {self.scope_id}: {e}")

    async def _run_ingestion_loop(self) -> None:
        """Background task to check ingestion policy and process pending entries.

        When the ingestion policy triggers:
        1. Collects ALL pending entries from all subscriptions
        2. Applies ingestion_policy.transformer to ALL entries together
            - It is up to the transformer to separate the entries by their type if it wants to.
        3. Writes transformed entries to this scope
        4. Clears processed pending entries
        """
        while self._running:
            try:
                await asyncio.sleep(self._ingestion_policy.ingestion_check_interval_seconds)

                if not self._pending_entries:
                    continue

                # Check if ingestion policy triggers
                pending_as_entries = [p.entry for p in self._pending_entries]
                should_ingest = await self._ingestion_policy.should_transfer(
                    pending_as_entries,
                    self._last_ingestion_time,
                )

                if should_ingest:
                    await self._execute_ingestion()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ingestion loop for {self.scope_id}: {e}")

    async def _execute_ingestion(self) -> int:
        """Process ALL pending entries: transform together and write to this scope.

        This is the core of the pull model:
        1. Collect ALL pending entries from ALL subscriptions
        2. Apply ONE ingestion_policy.transformer to ALL entries together
        3. Write transformed entries to THIS scope

        Returns:
            Number of entries written
        """
        if not self._pending_entries:
            return 0

        # Collect ALL entries from ALL subscriptions
        all_entries = [p.entry for p in self._pending_entries]
        source_count = len(all_entries)

        # Apply ingestion transformer to ALL entries together (or pass through)
        if self._ingestion_policy.transformer:
            # Build context with all source scopes
            source_scopes = list({p.subscription.source_scope_id for p in self._pending_entries})
            transformed = await self._ingestion_policy.transformer.consolidate(
                all_entries,
                ConsolidationContext(
                    source_scope=",".join(source_scopes),  # Multiple sources
                    target_scope=self.scope_id,
                ),
            )
        else:
            # No transformer = pass-through
            transformed = all_entries

        # Write transformed entries to this scope
        # session_id is auto-added by EnhancedBlackboard.write()
        count = 0
        for entry in transformed:
            data = entry.value
            key = data.get_blackboard_key(self.scope_id)

            value = data.model_dump() if hasattr(data, "model_dump") else data

            await self.storage.write(
                key=key,
                value=value,
                metadata=entry.metadata,
                tags=entry.tags,
                ttl_seconds=self.ttl_seconds,
            )
            count += 1

        # Clear pending and record time
        self._pending_entries = []
        self._last_ingestion_time = time.time()

        logger.info(
            f"MemoryCapability {self.scope_id}: ingested {source_count} → {count} entries"
        )

        return count

    async def _register_producer_hooks(self) -> None:
        """Register AFTER hooks for all configured memory producers.

        Each producer config specifies a pointcut (which methods to observe)
        and an extractor (how to extract storable data from the result).
        """
        if not self.producers:
            return

        hook_registry = self.agent.hooks

        for producer in self.producers:
            # Create the hook handler for this producer
            handler = self._create_producer_hook_handler(producer)

            # Register the hook
            hook_id = hook_registry.register(
                pointcut=producer.pointcut,
                handler=handler,
                hook_type=HookType.AFTER,
                priority=producer.priority,
                on_error=ErrorMode.SUPPRESS,  # Memory failures should not affect main flow
                owner=self,
            )

            self._producer_hook_ids.append(hook_id)
            logger.debug(
                f"MemoryCapability {self.scope_id}: registered producer hook "
                f"{hook_id} for {producer.pointcut!r}"
            )

    def _create_producer_hook_handler(self, producer: MemoryProducerConfig):
        """Create an AFTER hook handler for a memory producer.

        The handler extracts data from the hook context and result,
        then stores it to this memory level.
        """

        async def handler(ctx: HookContext, result: Any) -> Any:
            """AFTER hook handler for memory capture."""

            try:
                # Skip if result is None
                if result is None:
                    return result

                # Extract storable data
                if producer.extractor is not None:
                    # Call extractor (may be sync or async)
                    # Returns MemoryTagsMetadataTuple: (data, tags, metadata)
                    # or list[MemoryTagsMetadataTuple] for batch results
                    extracted = producer.extractor(ctx, result)
                    if asyncio.iscoroutine(extracted):
                        extracted = await extracted
                else:
                    # Default: use result directly with empty tags/metadata
                    # Result must be a BaseModel with get_blackboard_key
                    extracted = (result, set(), {})

                # Handle list results
                if not isinstance(extracted, list):
                    extracted = [extracted]

                for item in extracted:
                    # Unpack tuple: (data, tags, metadata)
                    if isinstance(item, tuple) and len(item) == 3:
                        data, tags, metadata = item
                    else:
                        # Fallback: item is just data
                        data, tags, metadata = item, set(), {}

                    # Skip if no data to store
                    if data is None:
                        continue

                    # Verify data is storable (has get_blackboard_key)
                    if not hasattr(data, "get_blackboard_key"):
                        logger.debug(
                            f"Memory producer data from {ctx.join_point} does not have "
                            f"get_blackboard_key, skipping storage"
                        )
                        continue

                    # Store to this memory level
                    await self.store(
                        data=data,
                        tags=tags,
                        metadata=metadata,
                        ttl_seconds=producer.ttl_seconds,
                    )

            except Exception as e:
                # Log but don't propagate - memory should not affect main flow
                logger.debug(f"Memory producer hook failed for {ctx.join_point}: {e}")

            # Always return the original result unchanged
            return result

        return handler

    def _apply_lens(self, lens: MemoryLens, query: MemoryQuery) -> MemoryQuery:
        """Apply lens configuration to query."""
        return MemoryQuery(
            query=query.query,
            tags=query.tags | (lens.tags_include or set()),
            max_results=min(query.max_results, lens.max_results),
            min_relevance=query.min_relevance,
            include_expired=query.include_expired,
            max_age_seconds=(
                -lens.time_range[0] if lens.time_range else query.max_age_seconds
            ),
        )

    async def _maybe_evict(self) -> None:
        """Evict entries if over max_entries limit."""
        if self.max_entries is None:
            return

        count = await self.storage.count()
        if count <= self.max_entries:
            return

        # Get all entries and sort by access time (oldest first) for LRU
        entries = await self.storage.query(limit=count + 100)
        entries_with_access = [
            (e, e.metadata.get("last_accessed_at", e.created_at))
            for e in entries
        ]
        entries_with_access.sort(key=lambda x: x[1])

        # Evict oldest entries
        num_to_evict = len(entries) - self.max_entries
        for entry, _ in entries_with_access[:num_to_evict]:
            await self.storage.delete(entry.key)

        logger.debug(f"MemoryCapability {self.scope_id}: evicted {num_to_evict} entries")

    async def _track_access(self, key: str) -> None:
        """Track access to a memory entry."""
        entry = await self.storage.read(key)
        if entry is None:
            return

        entry.metadata["last_accessed_at"] = time.time()
        entry.metadata["access_count"] = entry.metadata.get("access_count", 0) + 1

        # session_id is auto-added by EnhancedBlackboard.write()
        await self.storage.write(
            key=key,
            value=entry.value,
            metadata=entry.metadata,
            tags=entry.tags,
            expected_version=entry.version,
        )


