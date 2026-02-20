"""Blackboard → VCM page source: event-driven ingestion of blackboard scopes into VCM pages.

This module implements the BlackboardContextPageSource, which runs inside the VCM
deployment and pages any EnhancedBlackboard scope into VirtualContextPages via a
pluggable IngestionPolicy.

Analogous to FileGrouperContextPageSource (which pages git repo files), this pages
blackboard/memory scope contents. The key difference:
- FileGrouperContextPageSource: static content (files), loaded once
- BlackboardContextPageSource: dynamic content (live writes), event-driven

Architecture:
- Subscribes to EnhancedBlackboard.stream_events_to_queue() for live events
- On initialization, backfills existing entries (full scope scan)
- Delegates ALL record-to-page transformation to IngestionPolicy
- Tokenizes via TokenizerProtocol (from cluster/tokenization.py)
- Stores pages via PageStorage (VCM's existing instance — EFS/S3 + PostgreSQL)

All policies (IngestionPolicy, FlushPolicy, LocalityPolicy, etc.) are defined
here to keep the VCM integration self-contained.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from typing import Any, TYPE_CHECKING
from uuid import uuid4
from overrides import override

import networkx as nx

from ....vcm.sources import (
    ContextPageSource,
    ContextPageSourceFactory,
    BuilInContextPageSourceType
)
from ....vcm.models import VirtualContextPage, ContextPageId, MmapConfig
from ....distributed.ray_utils import serving

if TYPE_CHECKING:
    from ..types import BlackboardEvent, BlackboardScope
    from ..blackboard import EnhancedBlackboard
    from ....cluster.tokenization import TokenizerProtocol
    from ....vcm.page_storage import PageStorage, PageStorageConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class PendingRecord:
    """A record waiting to be grouped into a VCM page."""

    key: str  # Blackboard key
    value: dict[str, Any]  # Serialized record value
    tags: set[str] = dataclass_field(default_factory=set)  # Tags for grouping and metadata
    timestamp: float = 0.0  # When the record was written



# =============================================================================
# SimpleTokenizer — Fallback when VCM has no real tokenizer
# =============================================================================


class SimpleTokenizer:
    """Word-count-based tokenizer fallback.

    Used when the VCM hasn't acquired a real tokenizer from the LLM cluster.
    Implements the same interface as TokenizerProtocol (encode, decode,
    count_tokens) using a simple word-splitting heuristic.

    Approximation: ~1.3 tokens per whitespace-separated word (for English).
    """

    TOKENS_PER_WORD = 1.3

    def encode(self, text: str) -> list[int]:
        """Encode text to fake token IDs (one per word)."""
        words = text.split()
        # Generate deterministic fake token IDs from word hashes
        return [hash(w) % 100000 for w in words]

    def decode(self, token_ids: list[int]) -> str:
        """Decode is not reversible with this tokenizer."""
        return f"<SimpleTokenizer: {len(token_ids)} tokens>"

    def count_tokens(self, text: str) -> int:
        """Estimate token count from word count."""
        word_count = len(text.split())
        return int(word_count * self.TOKENS_PER_WORD)


# =============================================================================
# Locality Policy — HOW to group records into pages
# =============================================================================


class LocalityPolicy(ABC):
    """Assigns records to locality groups.

    Determines which records should be co-located in the same VCM page.
    This is the paging equivalent of FileGrouper's sharding strategy.
    """

    @abstractmethod
    def assign_group(self, value: dict[str, Any], tags: set[str] | None) -> str:
        """Assign a record to a locality group.

        Args:
            value: The record's serialized value (dict from blackboard entry)
            tags: The record's tags

        Returns:
            A locality key string. Records with the same key are
            grouped into the same page.
        """
        ...


class TagLocalityPolicy(LocalityPolicy):
    """Group records by their primary tags (most specific tags).

    Records tagged {"security", "auth", "critical"} and
    {"security", "auth", "medium"} share locality key "auth:security".
    """

    def assign_group(self, value: dict[str, Any], tags: set[str] | None) -> str:
        if not tags:
            return "untagged"
        # Use sorted tags (excluding meta-tags) as locality key
        content_tags = sorted(
            t for t in tags
            if not t.startswith("type:") and not t.startswith("from:")
        )
        return ":".join(content_tags[:3]) if content_tags else "untagged"


class TemporalLocalityPolicy(LocalityPolicy):
    """Group records by time window (e.g., all records within 5 minutes)."""

    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds

    def assign_group(self, value: dict[str, Any], tags: set[str] | None) -> str:
        bucket = int(time.time() / self.window_seconds)
        return f"time_bucket:{bucket}"


# =============================================================================
# Flush Policy — WHEN to create a VCM page from pending records
# =============================================================================


class FlushPolicy(ABC):
    """Decides when a pending group should be flushed to a VCM page.

    Controls the trade-off between locality (grouping more records per
    page) and latency (how long until a record is visible in the VCM).
    """

    @abstractmethod
    def should_flush(
        self,
        group: list[PendingRecord],
        tokenizer: "TokenizerProtocol",
    ) -> bool:
        """Return True if the group should be flushed now."""
        ...


class ThresholdFlushPolicy(FlushPolicy):
    """Flush when record count OR token budget is reached. Default policy.

    Good locality when traffic is high. Unbounded latency for slow scopes
    (pair with PeriodicFlushPolicy for bounded latency).
    """

    def __init__(self, record_threshold: int = 20, token_budget: int = 4096):
        self.record_threshold = record_threshold
        self.token_budget = token_budget

    def should_flush(self, group: list[PendingRecord], tokenizer: "TokenizerProtocol") -> bool:
        if len(group) >= self.record_threshold:
            return True
        estimated_tokens = sum(
            tokenizer.count_tokens(json.dumps(r.value, default=str))
            for r in group
        )
        return estimated_tokens >= self.token_budget


class PeriodicFlushPolicy(FlushPolicy):
    """Flush when the oldest pending record exceeds an age threshold.

    Guarantees bounded latency: a record becomes visible in the VCM
    within at most ``interval_seconds`` of being written.
    """

    def __init__(self, interval_seconds: float = 60.0):
        self.interval_seconds = interval_seconds

    def should_flush(self, group: list[PendingRecord], tokenizer: "TokenizerProtocol") -> bool:
        if not group:
            return False
        oldest = min(r.timestamp for r in group)
        return (time.time() - oldest) >= self.interval_seconds


class ImmediateFlushPolicy(FlushPolicy):
    """Flush every record immediately (one page per record).

    Use sparingly — trades locality for zero latency. Appropriate
    for critical results that must be visible immediately.
    """

    def should_flush(self, group: list[PendingRecord], tokenizer: "TokenizerProtocol") -> bool:
        return len(group) >= 1


# =============================================================================
# Serialization Policy — HOW to render records into page content
# =============================================================================


class SerializationPolicy(ABC):
    """Controls how records are serialized into VCM page text content.

    The output text is then tokenized by TokenizerProtocol and stored
    as a VirtualContextPage. Different serialization strategies affect
    how LLMs "read" the page content.
    """

    @abstractmethod
    def serialize_group(
        self,
        locality_key: str,
        records: list[PendingRecord],
    ) -> str:
        """Serialize a group of records into page text content."""
        ...


class JsonSerializationPolicy(SerializationPolicy):
    """Serialize records as structured JSON with headers. Default policy."""

    def serialize_group(self, locality_key: str, records: list[PendingRecord]) -> str:
        parts = []
        all_tags: set[str] = set()
        for r in records:
            parts.append(json.dumps(r.value, indent=2, default=str))
            all_tags |= r.tags
        return (
            f"# Page: {locality_key}\n"
            f"# Records: {len(records)} | Tags: {', '.join(sorted(all_tags))}\n\n"
            + "\n\n---\n\n".join(parts)
        )


# =============================================================================
# Page Update Policy — HOW to handle updates to already-paged records
# =============================================================================


class PageUpdatePolicy(ABC):
    """Handles updates to records that are already in a VCM page.

    When a blackboard key is overwritten, the page source receives a
    write event with old_value set. This policy decides what to do.
    """

    @abstractmethod
    async def handle_update(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
        ingestion_policy: "GroupAndFlushIngestionPolicy",
    ) -> None:
        ...


class AppendOnlyUpdatePolicy(PageUpdatePolicy):
    """Treat updates as new records. Default policy.

    The old version remains in its existing page; the new version
    enters the pending buffer. Consumers see both versions until
    the old page is rebuilt during maintenance.
    Simple and low-overhead. Best for append-heavy workloads.
    """

    async def handle_update(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
        ingestion_policy: "GroupAndFlushIngestionPolicy",
    ) -> None:
        # Treat as a new record
        record = PendingRecord(
            key=event.key,
            value=event.value or {},
            tags=event.tags or set(),
            timestamp=event.timestamp or time.time(),
        )
        await ingestion_policy.ingest_record(record)


class RebuildPageUpdatePolicy(PageUpdatePolicy):
    """Mark the page containing the old record as stale and schedule
    a rebuild. The updated record enters the pending buffer.

    More consistent (consumers see freshness indicator on stale pages)
    but higher overhead.
    """

    async def handle_update(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
        ingestion_policy: "GroupAndFlushIngestionPolicy",
    ) -> None:
        old_page_id = record_to_page.get(event.key)
        if old_page_id:
            await ingestion_policy._mark_page_stale(old_page_id)
            del record_to_page[event.key]
        # Re-ingest as new record
        record = PendingRecord(
            key=event.key,
            value=event.value or {},
            tags=event.tags or set(),
            timestamp=event.timestamp or time.time(),
        )
        await ingestion_policy.ingest_record(record)


# =============================================================================
# Page Eviction Policy — HOW to handle deletions
# =============================================================================


class PageEvictionPolicy(ABC):
    """Handles deletion of records that are already in a VCM page.

    When a blackboard key is deleted, the page source receives a
    delete event. This policy decides how to handle the stale page.
    """

    @abstractmethod
    async def handle_delete(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
        ingestion_policy: "GroupAndFlushIngestionPolicy",
    ) -> None:
        ...


class LazyEvictionPolicy(PageEvictionPolicy):
    """Do nothing on deletion. Default policy.

    Pages become stale naturally and are cleaned up during periodic
    VCM maintenance. Lowest overhead.
    """

    async def handle_delete(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
        ingestion_policy: "GroupAndFlushIngestionPolicy",
    ) -> None:
        record_to_page.pop(event.key, None)


class MarkStaleEvictionPolicy(PageEvictionPolicy):
    """Mark the page containing the deleted record as stale.

    Consumers can check the 'stale' flag in page metadata before
    relying on content. More responsive than Lazy but no rebuild.
    """

    async def handle_delete(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
        ingestion_policy: "GroupAndFlushIngestionPolicy",
    ) -> None:
        page_id = record_to_page.pop(event.key, None)
        if page_id:
            await ingestion_policy._mark_page_stale(page_id)


# =============================================================================
# IngestionPolicy ABC + GroupAndFlushIngestionPolicy
# =============================================================================


class IngestionPolicy(ABC):
    """Controls how blackboard records are organized into VCM pages.

    This is the core extension point for BlackboardContextPageSource.
    Different implementations can use entirely different backends and
    organization strategies:

    - GroupAndFlushIngestionPolicy (default): Groups records by locality
      tags, buffers until record count or token budget is reached, then
      flushes to a VCM page.
    - KnowledgeGraphIngestionPolicy: Ingests records into a knowledge
      graph (e.g., Neo4j), creates pages from graph neighborhoods.
    - SemanticClusterIngestionPolicy: Uses vector embeddings to cluster
      records into semantically coherent pages.
    - Custom: Any backend (vector DB, relational DB, etc.) that can
      produce VCM pages from blackboard records.

    The policy receives VCM's PageStorage and TokenizerProtocol during
    initialization, so it can create and store pages directly.
    """

    @abstractmethod
    async def initialize(
        self,
        scope_id: str,
        page_storage: "PageStorage",
        tokenizer: "TokenizerProtocol",
        record_to_page: dict[str, str],
    ) -> None:
        """Initialize the policy with VCM resources.

        Called once when the BlackboardContextPageSource is initialized.
        Implementations can use this to set up backend connections,
        load existing state, etc.

        Args:
            scope_id: The scope being paged.
            page_storage: VCM's PageStorage for storing pages.
            tokenizer: Tokenizer for token counting/encoding.
            record_to_page: Shared mutable dict mapping record keys to page IDs.
        """
        ...

    @abstractmethod
    async def ingest_record(self, record: PendingRecord) -> list[str]:
        """Ingest a new record.

        Returns:
            List of page_ids of any VCM pages created (empty if record
            was buffered and no flush occurred).
        """
        ...

    @abstractmethod
    async def handle_update(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
    ) -> None:
        """Handle an update to an already-ingested record."""
        ...

    @abstractmethod
    async def handle_delete(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
    ) -> None:
        """Handle deletion of an already-ingested record."""
        ...

    @abstractmethod
    async def flush_all(self) -> list[str]:
        """Flush all pending records. Returns page_ids of pages created."""
        ...


class GroupAndFlushIngestionPolicy(IngestionPolicy):
    """Default IngestionPolicy: group-by-locality -> buffer -> flush -> serialize.

    Composes sub-policies for fine-grained control:
    - LocalityPolicy: How to group records (by tags, time, etc.)
    - FlushPolicy: When to flush a group (record count, token budget, time)
    - SerializationPolicy: How to serialize records into page text
    - PageUpdatePolicy: How to handle updates to already-paged records
    - PageEvictionPolicy: How to handle deletions

    This is the right policy for most use cases. For fundamentally different
    backends (knowledge graphs, vector DBs), implement IngestionPolicy directly.
    """

    def __init__(
        self,
        source: ContextPageSource,
        locality_policy: LocalityPolicy | None = None,
        flush_policy: FlushPolicy | None = None,
        serialization_policy: SerializationPolicy | None = None,
        update_policy: PageUpdatePolicy | None = None,
        eviction_policy: PageEvictionPolicy | None = None,
    ):
        self.source: ContextPageSource = source
        self.locality_policy = locality_policy or TagLocalityPolicy()
        self.flush_policy = flush_policy or ThresholdFlushPolicy()
        self.serialization_policy = serialization_policy or JsonSerializationPolicy()
        self.update_policy = update_policy or AppendOnlyUpdatePolicy()
        self.eviction_policy = eviction_policy or LazyEvictionPolicy()

        # Set during initialize()
        self._page_storage: PageStorage | None = None
        self._tokenizer: TokenizerProtocol | None = None
        self._record_to_page: dict[str, str] = {}
        self._pending_groups: dict[str, list[PendingRecord]] = {}
        # Page graph reference — set by BlackboardContextPageSource after init
        self._page_graph: nx.DiGraph | None = None

    async def initialize(
        self,
        page_storage: "PageStorage",
        tokenizer: "TokenizerProtocol",
        record_to_page: dict[str, str],
    ) -> None:
        self._page_storage = page_storage
        self._tokenizer = tokenizer
        self._record_to_page = record_to_page

    async def ingest_record(self, record: PendingRecord) -> list[str]:
        locality_key = self.locality_policy.assign_group(record.value, record.tags)
        group = self._pending_groups.setdefault(locality_key, [])
        group.append(record)

        if self.flush_policy.should_flush(group, self._tokenizer):
            page_id = await self._flush_group(locality_key)
            return [page_id] if page_id else []
        return []

    async def handle_update(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
    ) -> None:
        await self.update_policy.handle_update(event, record_to_page, self)

    async def handle_delete(
        self,
        event: "BlackboardEvent",
        record_to_page: dict[str, str],
    ) -> None:
        await self.eviction_policy.handle_delete(event, record_to_page, self)

    async def flush_all(self) -> list[str]:
        page_ids = []
        for locality_key in list(self._pending_groups.keys()):
            pid = await self._flush_group(locality_key)
            if pid:
                page_ids.append(pid)
        return page_ids

    async def _flush_group(self, locality_key: str) -> str:
        """Create a VCM page from all pending records in a locality group.

        Steps:
        1. Serialize all records via SerializationPolicy
        2. Tokenize via TokenizerProtocol
        3. Create VirtualContextPage
        4. Store via PageStorage
        5. Track record->page mapping for update/delete handling
        6. Update page graph if available

        Returns:
            page_id of the created page, or empty string if group was empty.
        """
        group = self._pending_groups.pop(locality_key, [])
        if not group:
            return ""

        # 1. Serialize using pluggable policy
        content_text = self.serialization_policy.serialize_group(
            locality_key=locality_key,
            records=group,
        )

        # 2. Tokenize using framework tokenizer
        tokens = self._tokenizer.encode(content_text)

        # 3. Create page
        page_id = f"bb:{self.source.scope_id}:{locality_key}:{uuid4().hex[:8]}"
        all_tags = set().union(*(r.tags for r in group))

        page = VirtualContextPage(
            page_id=page_id,
            tokens=tokens,
            text=content_text,
            size=len(tokens),
            group_id=self.source.group_id, # f"bb:{self.source.scope_id}:{locality_key}",
            tenant_id=self.source.tenant_id,
            metadata={
                "source": BlackboardContextPageSource.get_source_metadata(self.source.scope_id),
                "scope_id": self.source.scope_id,
                "locality_key": locality_key,
                "record_count": len(group),
                "record_keys": [r.key for r in group],
                "tags": list(all_tags),
                "created_at": time.time(),
            },
            created_by="blackboard_page_source",
            isolation_level="shared",
        )

        # 4. Store via PageStorage
        await self._page_storage.store_page(page)

        # 5. Track record->page mapping
        for record in group:
            self._record_to_page[record.key] = page_id

        # 6. Update page graph if available
        if self._page_graph is not None:
            self._page_graph.add_node(page_id, **page.metadata)

        logger.info(
            f"GroupAndFlushIngestionPolicy[{self.source.scope_id}]: created page "
            f"{page_id} ({len(group)} records, {len(tokens)} tokens, "
            f"locality={locality_key})"
        )
        return page_id

    async def _mark_page_stale(self, page_id: str) -> None:
        """Mark a page as stale (used by update/eviction policies)."""
        if self._page_graph is not None and page_id in self._page_graph:
            self._page_graph.nodes[page_id]["stale"] = True
        logger.debug(
            f"GroupAndFlushIngestionPolicy[{self.source.scope_id}]: "
            f"marked page {page_id} as stale"
        )


# =============================================================================
# BlackboardContextPageSource — Main class
# =============================================================================


@ContextPageSourceFactory.register_new_source_type(BuilInContextPageSourceType.BLACKBOARD.value)
class BlackboardContextPageSource(ContextPageSource):
    """Pages any EnhancedBlackboard scope into VCM pages via IngestionPolicy.

    Runs inside the VirtualContextManager deployment. Created by
    ``mmap_application_scope()`` — never instantiated directly by agents.

    Analogous to FileGrouperContextPageSource (which pages git repo files),
    this pages blackboard/memory scope contents. The key difference:
    - FileGrouperContextPageSource: static content (files), loaded once
    - BlackboardContextPageSource: dynamic content (live writes), event-driven

    This creates the actual infrastructure needed to watch a blackboard
    scope and page its contents into VCM:
    1. Creates an EnhancedBlackboard for the scope
    2. Wraps it in a BlackboardStorageBackend
    3. Creates the configured IngestionPolicy
    4. Initializes with consumer group for event deduplication

    Architecture:
    - Subscribes to EnhancedBlackboard.stream_events_to_queue() for live events
    - On initialization, backfills existing entries (full scope scan)
    - Delegates ALL record-to-page transformation to IngestionPolicy
    - Tokenizes via TokenizerProtocol (from cluster/tokenization.py)
    - Stores pages via PageStorage (VCM's existing instance — EFS/S3 + PostgreSQL)

    This class does NOT create a VCM page per record. The IngestionPolicy
    controls buffering and flushing behavior.
    """

    def __init__(
        self,
        *,
        scope_id: str,
        group_id: str,
        tenant_id: str,
        mmap_config: MmapConfig,
    ):
        """Initialize page source for a storage scope.

        Called by VCM's mmap_application_scope() — not by agents directly.

        Args:
            scope_id: The scope being paged (e.g., "tenant:acme:discoveries")
            group_id: Identifier for grouping related context sources (e.g., VMR ID)
            tenant_id: The tenant ID for the scope
            mmap_config: Configuration for the memory-mapped page source
        """
        super().__init__(scope_id=scope_id, group_id=group_id, tenant_id=tenant_id, mmap_config=mmap_config)
        #    ingestion_policy: How records are organized into pages.
        #        Default: GroupAndFlushIngestionPolicy (tag-based locality,
        #        threshold-based flushing, JSON serialization).
        self.tokenizer: TokenizerProtocol | None = None
        self._page_storage: PageStorage | None = None

        # Pluggable ingestion policy (controls the entire record->page pipeline)
        self.ingestion_policy: IngestionPolicy | None = None # ingestion_policy or GroupAndFlushIngestionPolicy()

        # Create blackboard for the scope (SHARED scope, distributed backend)
        self.app_name = serving.get_my_app_name()
        self.blackboard = EnhancedBlackboard(
            app_name=self.app_name,
            scope=BlackboardScope.SHARED,
            scope_id=scope_id,
            enable_events=True,
        )

        # Internal state
        self._page_graph: nx.DiGraph = nx.DiGraph()  # TODO: This should be loaded from PageStorage dynamically
        self._event_queue: asyncio.Queue[BlackboardEvent] | None = None
        self._event_loop_task: asyncio.Task | None = None
        # Track which records are in which pages (for update/delete handling)
        self._record_to_page: dict[str, str] = {}  # record_key -> page_id

    @override
    async def initialize(self) -> None:
        """Initialize the page source.

        1. Initialize the IngestionPolicy with VCM resources
        2. Backfill existing entries from the scope (full scan)
        3. Start event subscription loop for live writes

        """

        # Initialize with consumer group for cross-replica deduplication
        # consumer_group: Redis Streams consumer group name for event
        #     deduplication across VCM replicas. When set, each event is
        #     delivered to exactly one replica in the group. If None,
        #     uses regular pub-sub (all replicas see all events).
        # consumer_name: This replica's name within the consumer group.
        #     Typically serving.get_my_replica_id().
        consumer_group = f"cg:mmap:{self.scope_id}"
        consumer_name = serving.get_my_replica_id()

        await self.blackboard.initialize()

        # Initialize tokenizer for scope-to-VCM mapping
        # Try to get a real tokenizer from the LLM cluster; fall back to SimpleTokenizer
        try:
            if self.llm_cluster_handle:
                self.tokenizer = await self.llm_cluster_handle.get_tokenizer()  # TODO - FIXME: This get_tokenizer method does not exist.
                logger.info("Acquired tokenizer from LLM cluster for scope mapping")
        except Exception as e:
            logger.info(f"Could not acquire tokenizer from LLM cluster ({e}), using SimpleTokenizer fallback")

        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer()
            logger.info("Using SimpleTokenizer fallback for scope mapping")

        from ....system import get_vcm
        vcm_handle = get_vcm()
        config: PageStorageConfig | None = vcm_handle.get_page_storage_config()
        if not config:
            raise ValueError("Missing PageStorageConfig in VCM")

        self._page_storage = PageStorage(
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            s3_bucket=config.s3_bucket,
        )
        await self._page_storage.initialize()

        # Resolve ingestion policy from config
        self.ingestion_policy = self._resolve_ingestion_policy()

        # Initialize the ingestion policy with shared record_to_page dict
        await self.ingestion_policy.initialize(
            scope_id=self.scope_id,
            page_storage=self._page_storage,
            tokenizer=self.tokenizer,
            record_to_page=self._record_to_page,
        )

        # Share the page graph reference with the ingestion policy
        # (so _flush_group can update nodes directly)
        if isinstance(self.ingestion_policy, GroupAndFlushIngestionPolicy):
            self.ingestion_policy._page_graph = self._page_graph

        # Backfill: query all existing entries in the scope
        existing = await self.blackboard.query(namespace=f"{self.scope_id}:*")
        for entry in existing:
            record = PendingRecord(
                key=entry.key,
                value=entry.value,
                tags=entry.tags or set(),
                timestamp=entry.metadata.get("created_at", time.time())
                if entry.metadata else time.time(),
            )
            page_ids = await self.ingestion_policy.ingest_record(record)
            for pid in page_ids:
                self._page_graph.add_node(pid)

        # Flush any remaining buffered records from backfill
        flush_ids = await self.ingestion_policy.flush_all()
        for pid in flush_ids:
            self._page_graph.add_node(pid)

        # TODO: This does not update the edges in the page graph or store the updated graph in PageStorage
        # Consider adding logic to update edges based on relationships between pages
        # and persist the updated graph to PageStorage.

        logger.info(
            f"BlackboardContextPageSource[{self.scope_id}]: backfilled "
            f"{len(existing)} entries, {len(self._page_graph.nodes)} pages"
        )

        # Start listening for new events
        self._event_queue = asyncio.Queue()
        if consumer_group is not None:
            await self.blackboard.stream_events_via_consumer_group(
                self._event_queue,
                consumer_group,
                consumer_name,
            )
        else:
            await self.blackboard.stream_events_to_queue(
                self._event_queue,
                pattern=f"{self.scope_id}:*",
            )

        self._event_loop_task = asyncio.create_task(self._event_loop())

    def _resolve_ingestion_policy(self) -> GroupAndFlushIngestionPolicy:
        """Create an IngestionPolicy from MmapConfig.

        Returns:
            Configured IngestionPolicy instance
        """
        # Resolve locality policy
        locality: LocalityPolicy
        if self.mmap_config.locality_policy_type == "temporal":
            locality = TemporalLocalityPolicy()
        else:
            locality = TagLocalityPolicy()

        # Resolve flush policy
        flush: FlushPolicy
        if self.mmap_config.flush_policy_type == "periodic":
            flush = PeriodicFlushPolicy(
                interval_seconds=self.mmap_config.flush_interval_seconds,
            )
        elif self.mmap_config.flush_policy_type == "immediate":
            flush = ImmediateFlushPolicy()
        else:
            flush = ThresholdFlushPolicy(
                record_threshold=self.mmap_config.flush_threshold,
                token_budget=self.mmap_config.flush_token_budget,
            )

        return GroupAndFlushIngestionPolicy(
            source=self,
            locality_policy=locality,
            flush_policy=flush,
        )

    @override
    async def shutdown(self) -> None:
        """Shutdown the page source (called by VCM's munmap_application_scope).

        Flushes all pending records and stops the event loop.
        """
        # Flush all pending records via the ingestion policy
        page_ids = await self.ingestion_policy.flush_all()
        for pid in page_ids:
            self._page_graph.add_node(pid)

        # TODO: This does not update the edges in the page graph or store the updated graph in PageStorage

        # Stop event loop
        if self._event_loop_task:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass
            self._event_loop_task = None

        logger.info(
            f"BlackboardContextPageSource[{self.scope_id}]: shutdown complete"
        )

    @override
    async def claim_orphaned_events(self) -> None:
        """Claim orphaned events from crashed replicas via XAUTOCLAIM.

        Events that were delivered to a consumer but not acknowledged
        (because the replica crashed) are claimed by this replica and
        re-enqueued into the page source's event queue.
        """
        claimed_messages = await self.blackboard.event_bus.claim_orphaned_events()
        for msg_id, msg_data in claimed_messages:
            await self.re_enqueue_event(msg_id, msg_data)

    async def re_enqueue_event(self, msg_id: bytes, msg_data: dict) -> None:
        """Re-enqueue an orphaned event from XAUTOCLAIM into the event queue.

        Called by VCM's reconciliation loop when it claims orphaned messages
        from crashed replicas via XAUTOCLAIM. The message data is parsed
        into a BlackboardEvent and put into the local event queue for
        processing by the ingestion policy.

        Args:
            msg_id: Redis Stream message ID.
            msg_data: Raw message data dict from XAUTOCLAIM.
        """
        from ..types import BlackboardEvent

        if self._event_queue is None:
            logger.warning(
                f"BlackboardContextPageSource[{self.scope_id}]: "
                f"cannot re-enqueue event {msg_id}, event queue not initialized"
            )
            return

        try:
            event_data = {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in msg_data.items()
            }
            event = BlackboardEvent(
                event_type=event_data.get("event_type", "write"),
                key=event_data.get("key", ""),
                value=json.loads(event_data.get("value", "{}")),
                old_value=(
                    json.loads(event_data["old_value"])
                    if event_data.get("old_value")
                    else None
                ),
                timestamp=float(event_data.get("timestamp", 0)),
                agent_id=event_data.get("agent_id") or None,
                tags=(
                    set(json.loads(event_data["tags"]))
                    if event_data.get("tags")
                    else None
                ),
                metadata=(
                    json.loads(event_data["metadata"])
                    if event_data.get("metadata")
                    else None
                ),
            )
            await self._event_queue.put(event)
        except Exception as e:
            logger.warning(
                f"BlackboardContextPageSource[{self.scope_id}]: "
                f"failed to re-enqueue event {msg_id}: {e}"
            )

    @override
    async def get_page_id_for_record(self, record_id: str) -> ContextPageId | None:
        """Get the page ID associated with a specific record ID, if any."""
        return self._record_to_page.get(record_id)

    @override
    async def get_record_ids_for_page(self, page_id: ContextPageId) -> list[str]:
        """Get all record IDs associated with a specific page ID."""
        return [record_id for record_id, pid in self._record_to_page.items() if pid == page_id]

    @override
    async def get_all_mapped_records(self) -> dict[str, ContextPageId]:
        """Get a mapping of all record IDs to their associated page IDs."""
        return dict(self._record_to_page)  # Return a copy to prevent external mutation

    @override
    async def get_all_mapped_pages(self) -> dict[ContextPageId, list[str]]:
        """Get a mapping of all page IDs to their associated record IDs."""
        return {
            page_id: [record_id for record_id, pid in self._record_to_page.items() if pid == page_id]
            for page_id in set(self._record_to_page.values())
        }

    # -------------------------------------------------------------------------
    # Event loop
    # -------------------------------------------------------------------------

    async def _event_loop(self) -> None:
        """Process events from the EnhancedBlackboard.

        Dispatches to the IngestionPolicy based on event type:
        - write (no old_value): new record -> ingest_record()
        - write (with old_value): update -> handle_update()
        - delete: removal -> handle_delete()
        """
        while True:
            try:
                event = await self._event_queue.get()

                if event.event_type == "write":
                    if event.old_value is not None:
                        # UPDATE: old_value present means this key was overwritten
                        await self.ingestion_policy.handle_update(
                            event=event,
                            record_to_page=self._record_to_page,
                        )
                    else:
                        # NEW WRITE
                        record = PendingRecord(
                            key=event.key,
                            value=event.value or {},
                            tags=event.tags or set(),
                            timestamp=event.timestamp or time.time(),
                        )
                        page_ids = await self.ingestion_policy.ingest_record(record)
                        for pid in page_ids:
                            self._page_graph.add_node(pid)

                        # TODO: This does not update the edges in the page graph or store the updated graph in PageStorage

                elif event.event_type == "delete":
                    await self.ingestion_policy.handle_delete(
                        event=event,
                        record_to_page=self._record_to_page,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"BlackboardContextPageSource[{self.scope_id}]: "
                    f"error processing event: {e}",
                    exc_info=True,
                )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _keyword_relevance(query: str, text: str) -> float:
        """Compute keyword-based relevance score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & text_words)
        return overlap / len(query_words)
