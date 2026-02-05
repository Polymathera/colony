"""Virtual Context Manager (VCM) data models.

This module defines the core data models for the VCM layer, including:
- VirtualContextPage: A chunk of tokens that is persisted and can be loaded into (and evicted from) KV cache
- PageLocation: Tracks which LLM replica has which page (physical location)
- PageGroup: Groups of related virtual context pages for spatial locality (usually from same high-level context unit such as git repo)
- PageFault: Represents a request to load a page
- VirtualPageTableState: Distributed state for page table
- PageAllocation/Eviction/Migration: Request/response models for page management
- ClusterPageStats: Cluster-wide statistics

All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
import sqlmodel as sqlm

from ..distributed.state_management import SharedState

# Limits to prevent unbounded queue growth in long-running jobs
MAX_PENDING_PAGE_FAULTS = 10000  # Maximum page faults in queue
MAX_PAGE_FAULT_WARNING = 8000  # Warn when approaching limit

logger = logging.getLogger(__name__)

# Type aliases for ID types - simple strings with semantic meaning
ContextPageId = str
"""Unique identifier for a context page."""

BranchId = str
"""Unique identifier for a VCM branch (for copy-on-write semantics)."""



class VirtualContextPageMetadata(sqlm.SQLModel, table=True):
    """Database model for virtual context page metadata.

    This stores queryable metadata while tokens are stored separately in EFS/S3.
    """
    __tablename__ = "virtual_context_pages"

    page_id: str = sqlm.Field(primary_key=True, description="Unique page identifier")
    tenant_id: str = sqlm.Field(index=True, description="Tenant ID for multi-tenancy")
    source: str = sqlm.Field(index=True, description="Source identifier (e.g., git repo URL)")
    created_at: datetime = sqlm.Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = sqlm.Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")
    size: int = sqlm.Field(description="Number of tokens in page")
    metadata_json: str = sqlm.Field(default="{}", description="JSON-encoded metadata")
    storage_location: str = sqlm.Field(description="Storage location (efs:// or s3://)")
    storage_backend: str = sqlm.Field(description="Storage backend type (efs or s3)")

    # Optional fields from VirtualContextPage
    group_id: Optional[str] = sqlm.Field(None, index=True, description="Page group ID for spatial locality")
    created_by: Optional[str] = sqlm.Field(None, index=True, description="Creator ID for finer-grained tracking within tenant (agent_id, session_id, run_id, etc.)")
    expires_at: Optional[datetime] = sqlm.Field(None, description="Optional expiration timestamp")


class VirtualContextPage(BaseModel):
    """A contiguous chunk of tokens that can be loaded into KV cache of a VLLMDeployment replica.

    This is a generic abstraction not tied to any higher-level context units (e.g., git repos) - can represent any
    tokenized data (code files, documents, tool descriptions, etc.).

    Pages are immutable (read-only) once created to simplify reasoning about
    shared state and caching.

    Typical page size: 20k-40k tokens to balance granularity and overhead.

    Multi-Tenancy Security:
    - tenant_id is included in KV cache hash to prevent cross-tenant leakage
    - Pages with different tenant_ids cannot share KV cache blocks
    - Access control validated on every load_page() and infer() call

    Attributes:
        page_id: Unique identifier for this page
        tokens: The actual token sequence
        size: Number of tokens (must be greater than or equal to len(tokens))
        metadata: Arbitrary metadata (source file, keywords, etc.)
        group_id: Optional group ID for spatial locality
        storage_uri: Where the raw data is stored (S3, DB, etc.)
        created_at: Timestamp when page was created
        expires_at: Optional expiration timestamp for automatic cleanup
        tenant_id: Required - identifies data owner for multi-tenancy isolation
        created_by: Optional - identifies creator (agent_id, session_id, run_id, etc.) for finer-grained tracking within tenant
        isolation_level: 'shared' (tenant-isolated KV cache) or 'isolated' (dedicated instance)
        allowed_tenant_ids: Set of tenant IDs allowed to access this page (for sharing)
        sensitivity_level: Data sensitivity classification
    """

    page_id: ContextPageId = Field(..., description="Unique page identifier")
    tokens: list[int] = Field(..., description="Token sequence")
    size: int = Field(..., description="Number of tokens")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    # Optional grouping for spatial locality
    group_id: str | None = Field(None, description="Page group ID for co-loading")

    # Storage location
    storage_uri: str | None = Field(None, description="Where raw data is stored")

    # Timestamps
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    expires_at: float | None = Field(None, description="Optional expiration timestamp")

    # Multi-tenancy fields
    tenant_id: str = Field(default="default", description="Tenant ID for multi-tenancy isolation")
    created_by: str | None = Field(None, description="Creator ID for finer-grained tracking within tenant (agent_id, session_id, run_id, etc.)")
    isolation_level: str = Field(
        default="shared",
        description="'shared' (tenant-isolated KV cache) or 'isolated' (dedicated instance)"
    )
    allowed_tenant_ids: set[str] = Field(
        default_factory=set,
        description="Set of tenant IDs allowed to access this page"
    )

    # Security metadata
    sensitivity_level: str = Field(
        default="internal",
        description="Data sensitivity classification: 'public', 'internal', 'confidential', 'restricted'"
    )

    # Branch support for copy-on-write semantics
    branch_id: BranchId = Field(
        default="main",
        description="Branch this page belongs to (for copy-on-write)"
    )
    parent_page_id: ContextPageId | None = Field(
        None,
        description="Parent page ID for CoW lineage (None if original)"
    )
    is_overlay: bool = Field(
        default=False,
        description="Whether this is a CoW overlay page"
    )
    base_version: int | None = Field(
        None,
        description="Base page version when this overlay was forked"
    )

    def __init__(self, **data):
        """Initialize page and auto-compute size if not provided."""
        if "size" not in data and "tokens" in data:
            data["size"] = len(data["tokens"])
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        """Validate size matches tokens length and ensure tenant_id is in allowed_tenant_ids."""
        # Validate that reserved capacity is sufficient for actual tokens
        if self.size < len(self.tokens):
            raise ValueError(
                f"Page capacity ({self.size}) is less than actual token count ({len(self.tokens)}). "
                f"Capacity must be >= actual tokens."
            )

        # Ensure tenant_id is in allowed_tenant_ids
        if self.tenant_id and self.tenant_id not in self.allowed_tenant_ids:
            self.allowed_tenant_ids.add(self.tenant_id) # TODO: Should we raise an error instead?

    def can_access(self, tenant_id: str) -> bool:
        """Check if a tenant can access this page.

        Args:
            tenant_id: Tenant ID requesting access

        Returns:
            True if access is allowed, False otherwise
        """
        return tenant_id in self.allowed_tenant_ids

    def is_expired(self) -> bool:
        """Check if this page has expired.

        Returns:
            True if page is expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class VCMBranch(BaseModel):
    """A branch in the VCM representing a version lineage for copy-on-write.

    VCM branches are analogous to Git branches - changes can be made independently
    and later merged. Each branch has its own view of VCM pages via copy-on-write.

    Key concepts:
    - A branch inherits all pages from its parent branch (via base_snapshot)
    - Modifications create overlay pages on the branch (via overlays dict)
    - New pages can be created on a branch (via new_pages set)
    - Branches can be merged back together

    Copy-on-write mechanics:
    - When reading a page: check overlays first, then walk up parent chain
    - When writing a page: create overlay if not already present
    - When creating a page: add to new_pages set

    Attributes:
        branch_id: Unique identifier
        tenant_id: Owning tenant
        parent_branch_id: Parent branch (None for root/main)
        name: Human-readable name
        created_at: Creation timestamp
        forked_at_version: Parent's overlay count at fork time
        base_snapshot: Page IDs inherited from parent at fork time
        overlays: original_page_id -> overlay_page_id mapping
        new_pages: Pages created on this branch (no parent page)
        merged_into: Target branch if this branch was merged
        state: Branch lifecycle state
    """

    branch_id: BranchId = Field(
        default_factory=lambda: f"branch_{uuid.uuid4().hex[:12]}",
        description="Unique branch identifier"
    )
    tenant_id: str = Field(..., description="Owning tenant")
    parent_branch_id: BranchId | None = Field(
        None,
        description="Parent branch for lineage (None for root/main)"
    )
    name: str = Field(default="main", description="Human-readable name")

    # Timestamps
    created_at: float = Field(default_factory=time.time, description="Branch creation timestamp")
    forked_at_version: int | None = Field(
        None,
        description="Parent's overlay count at fork time"
    )

    # Page tracking
    base_snapshot: set[str] = Field(
        default_factory=set,
        description="Page IDs inherited from parent at fork time"
    )
    overlays: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original_page_id -> overlay_page_id"
    )
    new_pages: set[str] = Field(
        default_factory=set,
        description="Pages created on this branch with no parent"
    )

    # Lifecycle
    merged_into: BranchId | None = Field(
        None,
        description="Target branch if this branch was merged"
    )
    state: str = Field(
        default="active",
        description="Branch state: 'active', 'merged', 'archived'"
    )

    def has_overlay(self, page_id: str) -> bool:
        """Check if this branch has an overlay for a page."""
        return page_id in self.overlays

    def get_overlay_id(self, page_id: str) -> str | None:
        """Get the overlay page ID for a base page, if it exists."""
        return self.overlays.get(page_id)

    def is_active(self) -> bool:
        """Check if branch is active."""
        return self.state == "active"



class PagePriority(str, Enum):
    """Priority levels for page allocation and loading."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PageLocation:
    """Physical location where a virtual page is loaded in the LLM cluster.

    Represents a specific instance of a virtual page loaded into a specific
    VLLMDeployment replica's KV cache. Used by VirtualPageTable to track
    page locations across the cluster.

    Attributes:
        page_id: Virtual page identifier
        deployment_name: Name of the VLLMDeployment
        client_id: ID of the specific replica within the deployment
        replica_id: Alias for client_id (for backward compatibility)
        load_time: Timestamp when page was loaded
        last_access_time: Timestamp of most recent access
        access_count: Number of times this page has been accessed
        size: Page size in tokens
        tenant_id: Tenant that owns this page
        has_appended_context: Whether this page has appended task-specific context
        appended_size: Size of appended context in tokens (0 if none)
    """

    page_id: str
    deployment_name: str
    client_id: str
    load_time: float
    last_access_time: float
    access_count: int = 0
    size: int = 0
    tenant_id: str = "default"

    # Appended context tracking
    has_appended_context: bool = False
    appended_size: int = 0

    @property
    def replica_id(self) -> str:
        """Alias for client_id (backward compatibility)."""
        return self.client_id

    @property
    def loaded_at(self) -> float:
        """Alias for load_time (backward compatibility)."""
        return self.load_time

    @property
    def last_accessed(self) -> float:
        """Alias for last_access_time (backward compatibility)."""
        return self.last_access_time

    def record_access(self, current_time: float | None = None) -> None:
        """Record an access to this page (updates last_access_time and access_count).

        Args:
            current_time: Optional timestamp (defaults to now)
        """
        self.last_access_time = current_time or time.time()
        self.access_count += 1

    def update_access(self, current_time: float) -> None:
        """Update access statistics (alias for record_access).

        Args:
            current_time: Current timestamp
        """
        self.record_access(current_time)


class PageLock(BaseModel):
    """Lock on a page to prevent eviction during critical operations.

    Locks are temporary and expire after a specified duration. This ensures
    that pages needed for long-running operations (e.g., multi-turn agent
    workflows, extended code analysis) remain in memory.

    Attributes:
        page_id: ID of the locked page
        locked_by: Identifier of who locked the page (agent_id, session_id, run_id, etc.)
        lock_expires_at: Timestamp when lock expires (Unix time)
        reason: Human-readable reason for the lock
        created_at: When the lock was created
    """

    page_id: str = Field(..., description="ID of the locked page")
    locked_by: str = Field(..., description="Identifier of who locked the page")
    lock_expires_at: float = Field(..., description="Lock expiration timestamp")
    reason: str = Field(default="", description="Reason for locking")
    created_at: float = Field(default_factory=time.time, description="Lock creation timestamp")

    def is_expired(self, current_time: float | None = None) -> bool:
        """Check if this lock has expired.

        Args:
            current_time: Optional timestamp (defaults to now)

        Returns:
            True if lock is expired, False otherwise
        """
        check_time = current_time if current_time is not None else time.time()
        return check_time >= self.lock_expires_at

    def remaining_time_s(self, current_time: float | None = None) -> float:
        """Get remaining lock time in seconds.

        Args:
            current_time: Optional timestamp (defaults to now)

        Returns:
            Remaining seconds (0 if expired)
        """
        check_time = current_time if current_time is not None else time.time()
        remaining = self.lock_expires_at - check_time
        return max(0.0, remaining)


class PageGroup(BaseModel):
    """Group of related pages that should be loaded together.

    Page groups support spatial locality by co-loading related pages (e.g.,
    related files from the same module or related sections of a document).

    Attributes:
        group_id: Unique group identifier
        page_ids: List of pages in this group
        priority: Load priority (higher = more urgent)
        metadata: Arbitrary metadata
    """

    group_id: str
    page_ids: list[str] = Field(default_factory=list)
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageFault(BaseModel):
    """Request to load virtual context pages that aren't currently in any LLM replica's KV cache.

    Page faults don't block - they add to a priority queue and are processed
    asynchronously by background tasks. This matches OS virtual memory behavior
    where a page fault increases loading priority but doesn't guarantee
    immediate loading.

    The background page fault processor can batch multiple faults together,
    reorder based on priorities, and optimize for spatial locality.

    Attributes:
        fault_id: Unique identifier for this fault (for tracking and waiting)
        page_ids: Which pages to load (can be multiple for batching)
        requesting_agent_id: Which agent/router requested it (optional)
        priority: Load priority (higher = more urgent)
        requested_at: When the fault occurred
        group_id: If part of a group, load entire group
        tenant_id: Tenant that owns these pages
        completed: Whether this fault has been processed
        completed_at: When the fault was completed (None if not completed)
        lock_duration_s: If set, lock pages after loading for this duration (seconds)
        lock_reason: Reason for locking (if lock_duration_s is set)
    """

    fault_id: str = Field(default_factory=lambda: f"fault-{uuid.uuid4().hex[:12]}")
    page_ids: list[str]
    requesting_agent_id: str | None = None
    priority: int = 0
    requested_at: float = Field(default_factory=time.time)
    group_id: str | None = None
    tenant_id: str = "default"
    completed: bool = False
    completed_at: float | None = None

    # Page locking after load
    lock_duration_s: float | None = Field(None, description="Lock pages after loading (seconds)")
    lock_reason: str = Field(default="", description="Reason for locking pages")

    # Backward compatibility: single page_id
    @property
    def page_id(self) -> str:
        """Get first page ID (for backward compatibility)."""
        return self.page_ids[0] if self.page_ids else ""

    def __lt__(self, other: PageFault) -> bool:
        """Compare by priority (higher priority first), then by timestamp (older first)."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.requested_at < other.requested_at  # Older first


# =============================================================================
# Scope-to-VCM Mapping Models (Blackboard ↔ VCM Integration)
# =============================================================================


class MmapConfig(BaseModel):
    """Configuration for mapping a blackboard/memory scope into VCM pages.

    Analogous to Linux mmap() — maps a storage scope into the VCM's virtual
    context space. The VCM creates a ``BlackboardContextPageSource`` that
    watches the scope's event stream and pages its contents.

    Attributes:
        ingestion_policy_type: Type of IngestionPolicy to use.
            ``"group_and_flush"`` (default) groups records by locality and
            flushes when thresholds are met.
        locality_policy_type: Type of LocalityPolicy for record grouping.
            ``"tag"`` (default) groups by content tags.
            ``"temporal"`` groups by time window.
        flush_policy_type: Type of FlushPolicy for page creation timing.
            ``"threshold"`` (default) flushes on count/token budget.
            ``"periodic"`` flushes on time interval.
            ``"immediate"`` flushes every record (one page per record).
        flush_threshold: Number of records before flushing (threshold policy).
        flush_token_budget: Token count before flushing (threshold policy).
        flush_interval_seconds: Time interval for periodic flush.
        pinned: If True, pages from this scope are pinned (never evicted).
            Analogous to mlock() — useful for critical collective knowledge.
    """

    ingestion_policy_type: str = "group_and_flush"
    locality_policy_type: str = "tag"
    flush_policy_type: str = "threshold"
    flush_threshold: int = 20
    flush_token_budget: int = 4096
    flush_interval_seconds: float = 60.0
    pinned: bool = False


class MmapResult(BaseModel):
    """Result of mmap_blackboard_scope() or munmap_blackboard_scope() call.

    Attributes:
        status: One of ``"mapped"``, ``"already_mapped"``, ``"not_mapped"``,
            ``"unmapped"``, ``"error"``.
        scope_id: The scope that was (un)mapped.
        message: Optional human-readable message with details.
    """

    status: str
    scope_id: str
    message: str = ""


class MappedScopeConfig(BaseModel):
    """Configuration for a scope-to-VCM mapping stored in shared state.

    Stored in ``VirtualPageTableState.mapped_scopes`` so all VCM replicas
    can materialize the mapping locally.

    Attributes:
        scope_id: The blackboard/memory scope being mapped.
        config: The MmapConfig used for this mapping.
        tenant_id: Tenant that owns this mapping (for multi-tenancy isolation).
        created_at: Timestamp when the mapping was created.
    """

    scope_id: str
    config: MmapConfig
    tenant_id: str | None = None
    created_at: float = Field(default_factory=time.time)


class VirtualPageTableState(SharedState):
    """Distributed state for page table shared across VCM replicas.

    This state is managed by StateManager and shared across all
    VirtualContextManager replicas for consistent view of page locations.

    Supports:
    - Multiple physical locations per page (replication)
    - Tenant-based indexing for multi-tenancy
    - Client-based indexing using (deployment_name, client_id)
    - Comprehensive statistics tracking
    - Page fault queue management
    - Page group management

    Attributes:
        entries: Forward index (page_id -> PageTableEntry with list of locations)
        client_pages: Reverse index ((deployment_name, client_id) -> set of page_ids)
        tenant_pages: Tenant index (tenant_id -> set of page_ids)
        page_groups: All registered page groups
        pending_faults: Priority queue of page faults to process
        total_loads: Total number of page loads across cluster
        total_evictions: Total number of page evictions
        total_migrations: Total number of page migrations
    """

    # Forward index: page_id -> PageTableEntry (supports multiple physical locations)
    entries: dict[str, PageTableEntry] = Field(default_factory=dict)

    # Reverse index: (deployment_name, client_id) -> set of page_ids
    # NOTE: Pydantic serializes tuple keys as "('dep', 'client')" strings
    client_pages: dict[str, set[str]] = Field(default_factory=dict)

    # Tenant index: tenant_id -> set of page_ids
    tenant_pages: dict[str, set[str]] = Field(default_factory=dict)

    # Page groups
    page_groups: dict[str, PageGroup] = Field(default_factory=dict)

    # Pending page faults (sorted by priority)
    pending_faults: list[PageFault] = Field(default_factory=list)

    # Page locks (prevent eviction during critical operations)
    locked_pages: dict[str, PageLock] = Field(default_factory=dict)
    """Maps page_id to PageLock for pages that should not be evicted"""

    # Statistics
    total_loads: int = 0
    total_evictions: int = 0
    total_migrations: int = 0

    # Backward compatibility: maintain old replica_pages index
    # Maps replica_id (client_id) to list of page_ids
    replica_pages: dict[str, list[str]] = Field(default_factory=dict)

    # Scope-to-VCM mappings (scope_id -> MappedScopeConfig)
    # Each entry represents a blackboard/memory scope that is being paged into
    # VCM by a BlackboardContextPageSource. All VCM replicas materialize local
    # page sources from this shared mapping during reconciliation.
    mapped_scopes: dict[str, MappedScopeConfig] = Field(default_factory=dict)

    # Branch tracking for copy-on-write semantics
    branches: dict[str, VCMBranch] = Field(
        default_factory=dict,
        description="Registered branches (branch_id -> VCMBranch)"
    )
    branch_pages: dict[str, set[str]] = Field(
        default_factory=dict,
        description="Pages created on each branch (branch_id -> set of page_ids)"
    )

    @classmethod
    def get_state_key(cls, app_name: str) -> str:
        """Generate state key for this page table."""
        return f"polymathera:serving:{app_name}:vcm:page_table"

    @staticmethod
    def _client_key(deployment_name: str, client_id: str) -> str:
        """Generate client key for indexing.

        Args:
            deployment_name: Deployment name
            client_id: Client ID

        Returns:
            String key for dict indexing
        """
        return f"{deployment_name}:{client_id}"

    def register_page_load(
        self,
        virtual_page_id: str,
        location: PageLocation,
        tenant_id: str = "default",
        size: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register that a virtual page has been loaded at a physical location.

        This method is idempotent - calling it multiple times with the same
        location updates the existing entry.

        Args:
            virtual_page_id: ID of the virtual page
            location: Physical location where page is loaded
            tenant_id: Tenant that owns this page
            size: Page size in tokens
            metadata: Additional metadata
        """
        # Get or create page table entry
        if virtual_page_id not in self.entries:
            self.entries[virtual_page_id] = PageTableEntry(
                virtual_page_id=virtual_page_id,
                tenant_id=tenant_id,
                size=size,
                created_at=time.time(),
                metadata=metadata or {},
            )

        entry = self.entries[virtual_page_id]

        # Add physical location to entry
        entry.add_location(location)

        # Update reverse index (client → pages)
        client_key = self._client_key(location.deployment_name, location.client_id)
        if client_key not in self.client_pages:
            self.client_pages[client_key] = set()
        self.client_pages[client_key].add(virtual_page_id)

        # Update tenant index
        if tenant_id not in self.tenant_pages:
            self.tenant_pages[tenant_id] = set()
        self.tenant_pages[tenant_id].add(virtual_page_id)

        # Update backward compatibility index
        if location.client_id not in self.replica_pages:
            self.replica_pages[location.client_id] = []
        if virtual_page_id not in self.replica_pages[location.client_id]:
            self.replica_pages[location.client_id].append(virtual_page_id)

        # Update statistics
        self.total_loads += 1

        logger.debug(
            f"Registered page load: {virtual_page_id} at "
            f"{location.deployment_name}/{location.client_id}"
        )

    def register_page_eviction(
        self,
        virtual_page_id: str,
        deployment_name: str,
        client_id: str,
    ) -> bool:
        """Register that a page has been evicted from a physical location.

        Args:
            virtual_page_id: ID of the virtual page
            deployment_name: Deployment where page was evicted
            client_id: Client where page was evicted

        Returns:
            True if page was found and evicted, False otherwise
        """
        if virtual_page_id not in self.entries:
            logger.warning(f"Attempted to evict unknown page: {virtual_page_id}")
            return False

        entry = self.entries[virtual_page_id]

        # Remove physical location
        removed = entry.remove_location(deployment_name, client_id)
        if not removed:
            logger.warning(
                f"Attempted to evict page {virtual_page_id} from "
                f"{deployment_name}/{client_id} but location not found"
            )
            return False

        # Update reverse index
        client_key = self._client_key(deployment_name, client_id)
        if client_key in self.client_pages:
            self.client_pages[client_key].discard(virtual_page_id)
            # Clean up empty entries
            if not self.client_pages[client_key]:
                del self.client_pages[client_key]

        # Update backward compatibility index
        if client_id in self.replica_pages:
            if virtual_page_id in self.replica_pages[client_id]:
                self.replica_pages[client_id].remove(virtual_page_id)
            if not self.replica_pages[client_id]:
                del self.replica_pages[client_id]

        # If page has no more physical locations, remove from tenant index
        if not entry.is_loaded():
            tenant_id = entry.tenant_id
            if tenant_id in self.tenant_pages:
                self.tenant_pages[tenant_id].discard(virtual_page_id)
                if not self.tenant_pages[tenant_id]:
                    del self.tenant_pages[tenant_id]

            # Remove entry entirely
            del self.entries[virtual_page_id]

        # Update statistics
        self.total_evictions += 1

        logger.debug(
            f"Registered page eviction: {virtual_page_id} from "
            f"{deployment_name}/{client_id}"
        )

        return True

    def register_page_access(
        self,
        virtual_page_id: str,
        deployment_name: str,
        client_id: str,
        current_time: float,
    ) -> None:
        """Update access statistics for a page.

        Args:
            virtual_page_id: ID of the virtual page
            deployment_name: Deployment where page was accessed
            client_id: Client where page was accessed
            current_time: Current timestamp
        """
        if virtual_page_id not in self.entries:
            logger.warning(f"Attempted to record access for unknown page: {virtual_page_id}")
            return

        entry = self.entries[virtual_page_id]
        location = entry.get_location(deployment_name, client_id)

        if location:
            location.update_access(current_time)
            entry.total_access_count += 1
        else:
            logger.warning(
                f"Attempted to record access for page {virtual_page_id} at "
                f"{deployment_name}/{client_id} but location not found"
            )

    def get_page_locations(self, virtual_page_id: str) -> list[PageLocation]:
        """Get all physical locations where a page is loaded.

        Args:
            virtual_page_id: ID of the virtual page

        Returns:
            List of physical locations (empty if page not loaded)
        """
        if virtual_page_id not in self.entries:
            return []
        return self.entries[virtual_page_id].physical_locations.copy()

    def get_pages_on_client(self, deployment_name: str, client_id: str) -> set[str]:
        """Get all virtual pages loaded on a specific client.

        Args:
            deployment_name: Deployment name
            client_id: Client ID

        Returns:
            Set of virtual page IDs (empty if none)
        """
        client_key = self._client_key(deployment_name, client_id)
        return self.client_pages.get(client_key, set()).copy()

    def get_pages_for_tenant(self, tenant_id: str) -> set[str]:
        """Get all virtual pages belonging to a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Set of virtual page IDs (empty if none)
        """
        return self.tenant_pages.get(tenant_id, set()).copy()

    def find_clients_with_page(self, virtual_page_id: str) -> list[tuple[str, str]]:
        """Find all clients that have a specific page loaded.

        Args:
            virtual_page_id: ID of the virtual page

        Returns:
            List of (deployment_name, client_id) tuples
        """
        if virtual_page_id not in self.entries:
            return []

        entry = self.entries[virtual_page_id]
        return [
            (loc.deployment_name, loc.client_id)
            for loc in entry.physical_locations
        ]

    def get_page_entry(self, virtual_page_id: str) -> PageTableEntry | None:
        """Get the full page table entry for a page.

        Args:
            virtual_page_id: ID of the virtual page

        Returns:
            PageTableEntry if page exists, None otherwise
        """
        return self.entries.get(virtual_page_id)

    def is_page_loaded(self, virtual_page_id: str) -> bool:
        """Check if a page is loaded anywhere in the cluster.

        Args:
            virtual_page_id: ID of the virtual page

        Returns:
            True if page has at least one physical location
        """
        if virtual_page_id not in self.entries:
            return False
        return self.entries[virtual_page_id].is_loaded()

    def get_replication_factor(self, virtual_page_id: str) -> int:
        """Get the replication factor (number of physical copies) of a page.

        Args:
            virtual_page_id: ID of the virtual page

        Returns:
            Number of physical locations (0 if page not loaded)
        """
        if virtual_page_id not in self.entries:
            return 0
        return self.entries[virtual_page_id].replication_factor()

    def get_all_loaded_pages(self) -> list[str]:
        """Get all pages currently loaded across all replicas.

        Returns:
            List of page IDs that are currently loaded (have at least one physical location)
        """
        return [
            page_id
            for page_id, entry in self.entries.items()
            if entry.is_loaded()
        ]

    # === Backward Compatibility Methods ===

    def add_page_location(self, location: PageLocation) -> None:
        """Add a page location (backward compatibility wrapper).

        Args:
            location: Page location to add
        """
        self.register_page_load(
            virtual_page_id=location.page_id,
            location=location,
            tenant_id=location.tenant_id,
            size=location.size,
        )

    def remove_page_location(self, page_id: str) -> None:
        """Remove a page from all locations (backward compatibility).

        Args:
            page_id: Page identifier
        """
        if page_id not in self.entries:
            return

        entry = self.entries[page_id]
        # Remove from all locations
        for location in entry.physical_locations.copy():
            self.register_page_eviction(
                virtual_page_id=page_id,
                deployment_name=location.deployment_name,
                client_id=location.client_id,
            )

    def add_page_fault(self, fault: PageFault) -> None:
        """Add a page fault to the priority queue (maintains sort order).

        Enforces queue size limits to prevent OOM in long-running jobs.
        """
        # Check if page fault already exists for this page
        existing = [f for f in self.pending_faults if f.page_id == fault.page_id]
        if existing:
            # Update priority if new fault has higher priority
            # TODO: Replacing existing fault may not be ideal - consider merging requester IDs
            if fault.priority > existing[0].priority:
                self.pending_faults.remove(existing[0])
                self.pending_faults.append(fault)
                self.pending_faults.sort()
        else:
            # Enforce queue size limit
            if len(self.pending_faults) >= MAX_PENDING_PAGE_FAULTS:
                # Drop lowest priority fault (last in sorted list)
                self.pending_faults.sort()  # Ensure sorted
                dropped = self.pending_faults.pop()  # Remove lowest priority
                logger.warning(
                    f"Page fault queue full ({MAX_PENDING_PAGE_FAULTS}), "
                    f"dropped lowest priority fault for page {dropped.page_id} "
                    f"(priority={dropped.priority})"
                )
            elif len(self.pending_faults) >= MAX_PAGE_FAULT_WARNING:
                logger.warning(
                    f"Page fault queue approaching limit: "
                    f"{len(self.pending_faults)}/{MAX_PENDING_PAGE_FAULTS}"
                )

            self.pending_faults.append(fault)
            self.pending_faults.sort()

    def pop_next_fault(self) -> PageFault | None:
        """Pop the highest priority page fault."""
        if self.pending_faults:
            return self.pending_faults.pop(0)
        return None

    def get_replica_page_count(self, replica_id: str) -> int:
        """Get number of pages loaded on a replica (backward compatibility)."""
        return len(self.replica_pages.get(replica_id, []))

    def get_cluster_stats(self, top_n: int = 10) -> ClusterPageStats:
        """Compute cluster-wide statistics about page distribution.

        Args:
            top_n: Number of hot/cold pages to include

        Returns:
            ClusterPageStats with comprehensive statistics
        """
        # Calculate basic statistics
        total_virtual_pages = len(self.entries)
        total_physical_pages = sum(
            entry.replication_factor()
            for entry in self.entries.values()
        )
        average_replication = (
            total_physical_pages / total_virtual_pages
            if total_virtual_pages > 0
            else 0.0
        )

        # Calculate cache statistics
        # Note: We don't have direct access to cache capacity here,
        # so we'll need to aggregate this from VLLMDeploymentState
        # For now, we calculate based on page sizes
        total_cache_used = sum(
            entry.size * entry.replication_factor()
            for entry in self.entries.values()
        )

        # Pages per tenant
        pages_per_tenant = {
            tenant_id: len(pages)
            for tenant_id, pages in self.tenant_pages.items()
        }

        # Pages per deployment (physical pages)
        pages_per_deployment: dict[str, int] = {}
        for entry in self.entries.values():
            for loc in entry.physical_locations:
                pages_per_deployment[loc.deployment_name] = (
                    pages_per_deployment.get(loc.deployment_name, 0) + 1
                )

        # Find hot and cold pages by access count
        pages_by_access = sorted(
            self.entries.values(),
            key=lambda e: e.total_access_count,
            reverse=True,
        )
        hot_pages = [
            (entry.virtual_page_id, entry.total_access_count)
            for entry in pages_by_access[:top_n]
        ]
        cold_pages = [
            (entry.virtual_page_id, entry.total_access_count)
            for entry in pages_by_access[-top_n:]
        ]

        return ClusterPageStats(
            total_virtual_pages=total_virtual_pages,
            total_physical_pages=total_physical_pages,
            average_replication_factor=average_replication,
            total_cache_capacity=0,  # TODO: Get from VLLMDeploymentState
            total_cache_used=total_cache_used,
            cache_utilization=0.0,  # TODO: Calculate when we have capacity
            pages_per_tenant=pages_per_tenant,
            pages_per_deployment=pages_per_deployment,
            hot_pages=hot_pages,
            cold_pages=cold_pages,
        )

    # === Page Lock Management ===

    def lock_page(
        self,
        page_id: str,
        locked_by: str,
        lock_duration_s: float,
        reason: str = "",
        current_time: float | None = None,
    ) -> PageLock:
        """Lock a page to prevent eviction during critical operations.

        Args:
            page_id: ID of the page to lock
            locked_by: Identifier of who is locking the page (agent_id, session_id, run_id, etc.)
            lock_duration_s: Lock duration in seconds
            reason: Human-readable reason for the lock
            current_time: Optional current timestamp (defaults to now)

        Returns:
            The created PageLock

        Raises:
            ValueError: If lock_duration_s is negative or zero
        """
        # TODO: Check if page is already locked and handle accordingly

        if lock_duration_s <= 0:
            raise ValueError(f"Lock duration must be positive, got {lock_duration_s}")

        now = current_time if current_time is not None else time.time()
        lock = PageLock(
            page_id=page_id,
            locked_by=locked_by,
            lock_expires_at=now + lock_duration_s,
            reason=reason,
            created_at=now,
        )
        self.locked_pages[page_id] = lock
        logger.info(
            f"Locked page {page_id} for {lock_duration_s}s by {locked_by} "
            f"(expires at {lock.lock_expires_at})"
        )
        return lock

    def unlock_page(self, page_id: str) -> bool:
        """Unlock a page, allowing it to be evicted.

        Args:
            page_id: ID of the page to unlock

        Returns:
            True if page was locked and is now unlocked, False if page wasn't locked
        """
        # TODO: Optionally check if locked_by matches?

        if page_id in self.locked_pages:
            del self.locked_pages[page_id]
            logger.info(f"Unlocked page {page_id}")
            return True
        return False

    def extend_page_lock(
        self,
        page_id: str,
        additional_duration_s: float,
        current_time: float | None = None,
    ) -> bool:
        """Extend the lock duration for a locked page.

        Args:
            page_id: ID of the locked page
            additional_duration_s: Additional seconds to add to lock duration
            current_time: Optional current timestamp (defaults to now)

        Returns:
            True if lock was extended, False if page wasn't locked

        Raises:
            ValueError: If additional_duration_s is negative
        """
        if additional_duration_s < 0:
            raise ValueError(f"Additional duration cannot be negative, got {additional_duration_s}")

        if page_id not in self.locked_pages:
            return False

        lock = self.locked_pages[page_id]
        now = current_time if current_time is not None else time.time()

        # Extend from current expiration time, or from now if already expired
        new_expires_at = max(lock.lock_expires_at, now) + additional_duration_s
        lock.lock_expires_at = new_expires_at

        logger.info(
            f"Extended lock for page {page_id} by {additional_duration_s}s "
            f"(new expiration: {new_expires_at})"
        )
        return True

    def is_page_locked(self, page_id: str, current_time: float | None = None) -> bool:
        """Check if a page is currently locked (and lock hasn't expired).

        Args:
            page_id: ID of the page to check
            current_time: Optional current timestamp (defaults to now)

        Returns:
            True if page is locked and lock hasn't expired, False otherwise
        """
        if page_id not in self.locked_pages:
            return False

        lock = self.locked_pages[page_id]
        return not lock.is_expired(current_time)

    def get_page_lock(self, page_id: str) -> PageLock | None:
        """Get the lock information for a page.

        Args:
            page_id: ID of the page

        Returns:
            PageLock if page is locked, None otherwise
        """
        return self.locked_pages.get(page_id)

    def cleanup_expired_locks(self, current_time: float | None = None) -> int:
        """Remove expired locks from the state.

        Args:
            current_time: Optional current timestamp (defaults to now)

        Returns:
            Number of locks removed
        """
        now = current_time if current_time is not None else time.time()
        expired_page_ids = [
            page_id
            for page_id, lock in self.locked_pages.items()
            if lock.is_expired(now)
        ]

        for page_id in expired_page_ids:
            del self.locked_pages[page_id]

        if expired_page_ids:
            logger.info(f"Cleaned up {len(expired_page_ids)} expired locks")

        return len(expired_page_ids)

    def get_locked_pages(self, include_expired: bool = False, current_time: float | None = None) -> list[str]:
        """Get list of currently locked page IDs.

        Args:
            include_expired: If True, include expired locks; if False, only active locks
            current_time: Optional current timestamp (defaults to now)

        Returns:
            List of locked page IDs
        """
        if include_expired:
            return list(self.locked_pages.keys())

        now = current_time if current_time is not None else time.time()
        return [
            page_id
            for page_id, lock in self.locked_pages.items()
            if not lock.is_expired(now)
        ]

    # === Branch Management Methods ===

    def register_branch(self, branch: VCMBranch) -> None:
        """Register a new branch.

        Args:
            branch: VCMBranch to register
        """
        self.branches[branch.branch_id] = branch
        self.branch_pages[branch.branch_id] = set()
        logger.debug(f"Registered branch {branch.branch_id} for tenant {branch.tenant_id}")

    def get_branch(self, branch_id: str) -> VCMBranch | None:
        """Get a branch by ID.

        Args:
            branch_id: Branch identifier

        Returns:
            VCMBranch if found, None otherwise
        """
        return self.branches.get(branch_id)

    def register_overlay(
        self,
        branch_id: str,
        original_page_id: str,
        overlay_page_id: str,
    ) -> None:
        """Register a CoW overlay for a page on a branch.

        Args:
            branch_id: Branch to register overlay on
            original_page_id: Original page being overlaid
            overlay_page_id: New overlay page ID

        Raises:
            ValueError: If branch doesn't exist
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")

        self.branches[branch_id].overlays[original_page_id] = overlay_page_id
        self.branch_pages[branch_id].add(overlay_page_id)
        logger.debug(f"Registered overlay {overlay_page_id} for page {original_page_id} on branch {branch_id}")

    def register_new_page_on_branch(
        self,
        branch_id: str,
        page_id: str,
    ) -> None:
        """Register a new page created on a branch (not an overlay).

        Args:
            branch_id: Branch the page was created on
            page_id: New page ID

        Raises:
            ValueError: If branch doesn't exist
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")

        self.branches[branch_id].new_pages.add(page_id)
        self.branch_pages[branch_id].add(page_id)
        logger.debug(f"Registered new page {page_id} on branch {branch_id}")

    def get_effective_page_id(self, base_page_id: str, branch_id: str) -> str:
        """Get the effective page ID for a branch (resolves overlays).

        Walks up the branch lineage looking for overlays. Returns the
        overlay page ID if found, otherwise the original page ID.

        Args:
            base_page_id: Original page ID to look up
            branch_id: Branch to resolve from

        Returns:
            Effective page ID (overlay or original)
        """
        branch = self.branches.get(branch_id)
        while branch:
            if base_page_id in branch.overlays:
                return branch.overlays[base_page_id]
            if branch.parent_branch_id:
                branch = self.branches.get(branch.parent_branch_id)
            else:
                break
        return base_page_id  # No overlay found, use original

    def get_branches_for_tenant(self, tenant_id: str) -> list[VCMBranch]:
        """Get all branches belonging to a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of VCMBranch objects for this tenant
        """
        return [
            branch for branch in self.branches.values()
            if branch.tenant_id == tenant_id
        ]

    def get_branch_effective_pages(self, branch_id: str) -> set[str]:
        """Get all pages effectively visible from a branch.

        Includes:
        - Pages from base_snapshot (inherited from parent)
        - Overlay pages (modifications on this branch)
        - New pages (created on this branch)

        Args:
            branch_id: Branch identifier

        Returns:
            Set of effective page IDs visible from this branch
        """
        branch = self.branches.get(branch_id)
        if not branch:
            return set()

        # Start with base snapshot (inherited pages)
        pages = branch.base_snapshot.copy()

        # Add overlays (these replace originals in effective view)
        pages.update(branch.overlays.values())

        # Add new pages created on this branch
        pages.update(branch.new_pages)

        return pages


# === Page Management Request/Response Models ===


class PageTableEntry(BaseModel):
    """Entry in the cluster page table.

    Tracks all physical locations where a virtual page is loaded,
    along with metadata for management and optimization.

    Attributes:
        virtual_page_id: ID of the virtual page
        physical_locations: List of physical locations where page is loaded
        tenant_id: Tenant that owns this page
        size: Page size in tokens
        created_at: When this page was first registered
        total_access_count: Total accesses across all physical locations
        metadata: Additional metadata (e.g., source info, page group)
    """

    virtual_page_id: str
    physical_locations: list[PageLocation] = Field(default_factory=list)
    tenant_id: str = "default"
    size: int = 0
    created_at: float = 0.0
    total_access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True  # Allow PageLocation dataclass

    def add_location(self, location: PageLocation) -> None:
        """Add a physical location for this page.

        Args:
            location: New physical location
        """
        # Check if location already exists (same deployment + client)
        for existing in self.physical_locations:
            if (existing.deployment_name == location.deployment_name and
                existing.client_id == location.client_id):
                # Update existing location
                existing.load_time = location.load_time
                existing.last_access_time = location.last_access_time
                return

        # Add new location
        self.physical_locations.append(location)

    def remove_location(self, deployment_name: str, client_id: str) -> bool:
        """Remove a physical location.

        Args:
            deployment_name: Deployment name
            client_id: Client ID

        Returns:
            True if location was removed, False if not found
        """
        for i, loc in enumerate(self.physical_locations):
            if loc.deployment_name == deployment_name and loc.client_id == client_id:
                self.physical_locations.pop(i)
                return True
        return False

    def get_location(self, deployment_name: str, client_id: str) -> PageLocation | None:
        """Get a specific physical location.

        Args:
            deployment_name: Deployment name
            client_id: Client ID

        Returns:
            PageLocation if found, None otherwise
        """
        for loc in self.physical_locations:
            if loc.deployment_name == deployment_name and loc.client_id == client_id:
                return loc
        return None

    def is_loaded(self) -> bool:
        """Check if page is loaded anywhere.

        Returns:
            True if page has at least one physical location
        """
        return len(self.physical_locations) > 0

    def replication_factor(self) -> int:
        """Get number of physical locations (replication factor).

        Returns:
            Number of replicas where this page is loaded
        """
        return len(self.physical_locations)


class PageAllocationRequest(BaseModel):
    """Request to allocate (load) virtual pages into physical memory.

    Attributes:
        virtual_page_ids: List of virtual page IDs to allocate
        tenant_id: Tenant requesting the allocation
        priority: Allocation priority
        preferred_deployment: Optional preferred deployment name
        target_client_id: Optional target replica ID (bypasses allocation strategy)
        affinity_pages: Optional list of page IDs that should be co-located
        max_replication: Maximum replication factor (default: 1)
        metadata: Additional request metadata
    """

    virtual_page_ids: list[str]
    tenant_id: str = "default"
    priority: PagePriority = PagePriority.NORMAL
    preferred_deployment: str | None = None
    target_client_id: str | None = Field(
        None,
        description="Target specific replica for page loading (overrides allocation strategy, used by routers)"
    )
    affinity_pages: list[str] = Field(default_factory=list)
    max_replication: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageAllocationResponse(BaseModel):
    """Response from page allocation request.

    Attributes:
        allocated_locations: Mapping of virtual page ID to allocated locations
        failed_pages: Pages that couldn't be allocated
        evicted_pages: Pages that were evicted to make room
        allocation_time_ms: Time taken for allocation in milliseconds
    """

    allocated_locations: dict[str, list[PageLocation]] = Field(
        default_factory=dict
    )
    failed_pages: list[str] = Field(default_factory=list)
    evicted_pages: list[str] = Field(default_factory=list)
    allocation_time_ms: float = 0.0

    class Config:
        arbitrary_types_allowed = True  # Allow PageLocation dataclass


class PageEvictionRequest(BaseModel):
    """Request to evict pages from physical memory.

    Attributes:
        virtual_page_ids: Optional specific pages to evict (if None, use policy)
        num_pages: Number of pages to evict (if virtual_page_ids is None)
        tenant_id: Optional tenant filter for eviction
        deployment_name: Optional deployment filter for eviction
    """

    virtual_page_ids: list[str] | None = None
    num_pages: int = 1
    tenant_id: str | None = None
    deployment_name: str | None = None


class PageMigrationRequest(BaseModel):
    """Request to migrate a page between replicas.

    Attributes:
        virtual_page_id: Page to migrate
        source_deployment: Source deployment name
        source_client_id: Source client ID
        target_deployment: Target deployment name (None to auto-select)
        target_client_id: Target client ID (None to auto-select)
        keep_source: Whether to keep source copy (replication) or remove it (migration)
    """

    virtual_page_id: str
    source_deployment: str
    source_client_id: str
    target_deployment: str | None = None
    target_client_id: str | None = None
    keep_source: bool = False  # False = move, True = replicate


class ClusterPageStats(BaseModel):
    """Statistics about page distribution across the cluster.

    Attributes:
        total_virtual_pages: Total number of unique virtual pages tracked
        total_physical_pages: Total number of physical page instances (including replicas)
        average_replication_factor: Average number of replicas per page
        total_cache_capacity: Total KV cache capacity across all replicas (tokens)
        total_cache_used: Total KV cache used across all replicas (tokens)
        cache_utilization: Percentage of cache used
        pages_per_tenant: Number of pages per tenant
        pages_per_deployment: Number of physical pages per deployment
        hot_pages: Top N most accessed pages
        cold_pages: Top N least accessed pages
    """

    total_virtual_pages: int
    total_physical_pages: int
    average_replication_factor: float
    total_cache_capacity: int
    total_cache_used: int
    cache_utilization: float
    pages_per_tenant: dict[str, int] = Field(default_factory=dict)
    pages_per_deployment: dict[str, int] = Field(default_factory=dict)
    hot_pages: list[tuple[str, int]] = Field(default_factory=list)
    cold_pages: list[tuple[str, int]] = Field(default_factory=list)


class PageAccessPattern(BaseModel):
    """Page access pattern analysis for optimization.

    Attributes:
        page_id: Virtual page ID
        access_frequency: Accesses per second
        access_recency: Time since last access (seconds)
        co_accessed_pages: Pages frequently accessed together
        requesting_tenants: Tenants that accessed this page
    """

    page_id: str
    access_frequency: float  # accesses per second
    access_recency: float  # seconds since last access
    co_accessed_pages: list[tuple[str, float]] = Field(default_factory=list)  # (page_id, correlation)
    requesting_tenants: set[str] = Field(default_factory=set)

