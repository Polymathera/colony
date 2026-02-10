"""Virtual Context Manager deployment.

The VirtualContextManager is the main API for the VCM layer. It manages
virtual context pages, handles page faults, and coordinates with the LLMCluster
layer for physical page loading.

This is a scalable deployment with multiple replicas sharing a distributed
page table via StateManager.

Key Features:
- Page creation and management
- Page loading on demand (page fault handling)
- Page groups for spatial locality
- Integration with LLMCluster for physical operations
- Background page fault processing

Example:
    ```python
    from colony.distributed.ray_utils import serving
    from polymathera.colony.vcm import VirtualContextManager

    app = serving.Application(name="vcm-app")
    app.add_deployment(
        VirtualContextManager.bind(
            caching_policy="LRU",
            allocation_strategy=DEFAULT_ALLOCATION_STRATEGY,
            page_storage_backend_type="efs",
            page_storage_path="colony/context_pages",
            reconciliation_interval_s=30.0
        )
    )
    await app.start()
    ```
"""

import asyncio
import logging
import time
from typing import Any, Literal

from ..distributed.ray_utils import serving
from .models import (
    BranchId,
    ContextPageId,
    MappedScopeConfig,
    MmapConfig,
    MmapResult,
    PageAllocationRequest,
    PageAllocationResponse,
    PageFault,
    PageGroup,
    PageLocation,
    PageLock,
    VCMBranch,
    VirtualContextPage,
)
from .sources.blackboard_page_source import (
    BlackboardContextPageSource,
    GroupAndFlushIngestionPolicy,
    MappedScope,
    SimpleTokenizer,
)
from .page_storage import PageStorage, PageStorageConfig
from .page_table import VirtualPageTable
from .allocation import AllocationStrategy, DEFAULT_ALLOCATION_STRATEGY
from ..deployment_names import get_deployment_names

logger = logging.getLogger(__name__)


@serving.deployment
class VirtualContextManager:
    """Main VCM deployment - manages virtual context pages and routing to LLMCluster.

    Scalable deployment with multiple replicas sharing distributed page table.
    Provides API for page management, page fault handling, and integration
    with LLMCluster and agents.

    Attributes:
        caching_policy: Caching policy for page eviction ("LRU" or "LFU")
    """

    def __init__(
        self,
        caching_policy: Literal["LRU", "LFU"] = "LRU",
        allocation_strategy: AllocationStrategy | None = None,
        page_storage_backend_type: Literal["efs", "s3"] = "efs",  # Default to EFS for fast access
        page_storage_path: str = "colony/context_pages",
        page_storage_s3_bucket: str = "polymathera-context-pages",
        reconciliation_interval_s: float = 30.0
    ):
        """Initialize VirtualContextManager.

        Args:
            caching_policy: Page eviction policy ("LRU" or "LFU")
            allocation_strategy: Strategy for page allocation decisions
            page_storage_backend_type: Backend type for page storage ("efs" or "s3")
            page_storage_path: Path for page storage (EFS directory or S3 prefix)
            page_storage_s3_bucket: S3 bucket name for page storage (if using S3 backend)
            reconciliation_interval_s: Interval for periodic reconciliation (seconds)
        """
        self.caching_policy = caching_policy
        self.allocation_strategy = allocation_strategy
        self.page_storage_backend_type = page_storage_backend_type
        self.page_storage_path = page_storage_path
        self.page_storage_s3_bucket = page_storage_s3_bucket

        # Initialized in initialize
        self.app_name: str | None = None
        self.page_table: VirtualPageTable | None = None
        self.page_storage: PageStorage | None = None
        self.llm_cluster_handle: serving.DeploymentHandle | None = None

        # Cache of deployment state managers for updating VLLMDeploymentState
        # Maps deployment_name -> StateManager[VLLMDeploymentState]
        self._deployment_state_managers: dict[str, Any] = {}

        # Page fault event tracking for efficient waiting
        # Maps fault_id -> asyncio.Event for notifying waiters when fault is processed
        self._page_fault_events: dict[str, asyncio.Event] = {}

        # Event subscription infrastructure (initialized in initialize())
        self.redis_client = None  # RedisClient
        self.event_subscribers: dict[str, Any] = {}  # deployment_name -> DistributedStateSubscriber
        self._reconciliation_task: asyncio.Task | None = None
        self._reconciliation_interval_s = reconciliation_interval_s

        # Blackboard ↔ VCM integration (scope mappings)
        self._local_mapped_scopes: dict[str, MappedScope] = {}
        self._tokenizer: Any = None  # TokenizerProtocol — set during initialize()

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize state manager and components after deployment starts."""
        # Get app name from environment
        self.app_name = serving.get_my_app_name()
        logger.info(f"Initializing VirtualContextManager for app {self.app_name}")

        # Initialize page table
        self.page_table = VirtualPageTable()
        await self.page_table.initialize()

        names = get_deployment_names()

        # Get LLMCluster handle for allocation decisions and page loading
        try:
            self.llm_cluster_handle = serving.get_deployment(
                app_name=self.app_name,
                deployment_name=names.llm_cluster,
            )
            logger.info("Connected to LLMCluster deployment")
        except Exception as e:
            logger.warning(f"LLMCluster deployment not found (allocation will use default strategies): {e}")

        # Initialize allocation strategy
        if self.allocation_strategy is None:
            self.allocation_strategy = DEFAULT_ALLOCATION_STRATEGY
            logger.info("Using default allocation strategy (BalancedAllocationStrategy)")
        else:
            logger.info(f"Using custom allocation strategy: {type(self.allocation_strategy).__name__}")

        # Initialize persistent page storage
        from ..distributed import get_polymathera

        polymathera = get_polymathera()
        storage_backend = await polymathera.get_storage()

        self.page_storage = PageStorage(
            storage_backend=storage_backend,
            backend_type=self.page_storage_backend_type,
            storage_path=self.page_storage_path,
        )
        await self.page_storage.initialize()
        logger.info("Initialized PageStorage with EFS backend")

        # Initialize Redis and event subscriptions
        try:
            self.redis_client = await polymathera.get_redis_client()
            await self._subscribe_to_page_events()
            logger.info("Initialized Redis event subscriptions for page state reconciliation")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis event subscriptions: {e}. Layer 2 reconciliation will rely on periodic scans only.")
            self.redis_client = None

        # Initialize tokenizer for scope-to-VCM mapping (BlackboardContextPageSource)
        # Try to get a real tokenizer from the LLM cluster; fall back to SimpleTokenizer
        try:
            if self.llm_cluster_handle:
                self._tokenizer = await self.llm_cluster_handle.get_tokenizer()
                logger.info("Acquired tokenizer from LLM cluster for scope mapping")
        except Exception as e:
            logger.info(f"Could not acquire tokenizer from LLM cluster ({e}), using SimpleTokenizer fallback")
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
            logger.info("Using SimpleTokenizer fallback for scope mapping")

        # Materialize any existing scope mappings from shared state
        try:
            await self._reconcile_scope_mappings()
        except Exception as e:
            logger.warning(f"Failed to reconcile scope mappings during init: {e}")

        # Start periodic reconciliation task
        self._reconciliation_task = asyncio.create_task(self._periodic_reconciliation_loop())
        logger.info(f"Started periodic reconciliation task (interval={self._reconciliation_interval_s}s)")

        logger.info(
            f"VirtualContextManager initialized with caching_policy={self.caching_policy}, "
            f"allocation_strategy={type(self.allocation_strategy).__name__}"
        )

    # === Page Management API ===
    @serving.endpoint
    def get_page_storage_config(self):
        return PageStorageConfig(
            backend_type=self.page_storage_backend_type,
            storage_path=self.page_storage_path,
            s3_bucket=self.page_storage_s3_bucket,
        )

    @serving.endpoint
    async def create_virtual_page(
        self,
        tokens: list[int],
        page_id: ContextPageId | None = None,
        metadata: dict[str, Any] | None = None,
        group_id: str | None = None,
    ) -> VirtualContextPage:
        """Create a new virtual context page and persist it to storage.
        This method does not preload the newly created page into an LLM instance.

        Args:
            tokens: Token sequence for this page
            page_id: Optional ID (generated if None)
            metadata: Optional metadata
            group_id: Optional group ID for spatial locality

        Returns:
            Created VirtualContextPage
        """
        if page_id is None:
            import uuid

            page_id = f"page-{uuid.uuid4().hex[:8]}"

        page = VirtualContextPage(
            page_id=page_id,
            tokens=tokens,
            size=len(tokens),
            metadata=metadata or {},
            group_id=group_id,
        )

        # Persist to durable storage (EFS or S3)
        await self.page_storage.store_page(page)

        # TODO: Add the page to the page group if group_id is provided

        logger.info(f"Created and persisted virtual page {page_id} with {len(tokens)} tokens")

        return page

    # === Physical Page Allocation (from LLMCluster) ===

    @serving.endpoint
    async def allocate_pages(self, request: PageAllocationRequest) -> PageAllocationResponse:
        """Allocate pages into physical memory.

        This method:
        1. Consults the allocation strategy to decide where to place pages
        2. Evicts pages if necessary to make room
        3. Coordinates with VLLMDeployment replicas to load pages
        4. Updates the page table

        Args:
            request: Page allocation request

        Returns:
            Allocation response with results

        Raises:
            RuntimeError: If allocation fails critically
        """
        start_time = time.time()
        logger.info(
            f"Allocating {len(request.virtual_page_ids)} pages for tenant {request.tenant_id}"
        )

        try:
            # Get client states from LLMCluster for allocation decisions
            client_states_snapshot = {}
            if self.llm_cluster_handle:
                try:
                    client_states_snapshot = await self.llm_cluster_handle.get_all_client_states()
                    logger.debug(f"Queried client states from {len(client_states_snapshot)} clients")
                except Exception as e:
                    logger.warning(f"Failed to query client states from LLMCluster: {e}")
                    client_states_snapshot = {}
            else:
                logger.debug("LLMCluster handle not available, using default allocation")
                client_states_snapshot = {}

            # Prepare page sizes
            # Get actual sizes from virtual pages in cache
            page_sizes = {}
            for page_id in request.virtual_page_ids:
                page = await self.get_virtual_page(page_id)
                if page:
                    page_sizes[page_id] = page.size # TODO: How was size set? Based on LLM name?
                else:
                    # Default size if page not found
                    page_sizes[page_id] = 40000 # TODO: Make configurable or derived from LLM model type.
                    logger.warning(f"Page {page_id} not found in cache, using default size 40000")

            # Make allocation decisions
            decisions = await self.allocation_strategy.make_allocation_decisions(
                request=request,
                page_table=self.page_table,
                client_states=client_states_snapshot,
                page_sizes=page_sizes,
            )

            # Execute allocations
            allocated_locations = {}
            failed_pages = []
            evicted_pages = []

            for decision in decisions:
                try:
                    # Evict pages if needed
                    for evict_page_id in decision.evict_pages:
                        success = await self._evict_page_from_client(
                            virtual_page_id=evict_page_id,
                            deployment_name=decision.target_deployment,
                            client_id=decision.target_client_id,
                        )
                        if success:
                            evicted_pages.append(evict_page_id)
                        else:
                            logger.warning(
                                f"Failed to evict page {evict_page_id} from "
                                f"{decision.target_deployment}/{decision.target_client_id}"
                            )

                    # Load the page
                    success = await self._load_page_on_client(
                        virtual_page_id=decision.virtual_page_id,
                        deployment_name=decision.target_deployment,
                        client_id=decision.target_client_id,
                        page_size=page_sizes.get(decision.virtual_page_id, 0),
                        tenant_id=request.tenant_id,
                    )

                    if success:
                        # Create location record
                        location = PageLocation(
                            page_id=decision.virtual_page_id,
                            deployment_name=decision.target_deployment,
                            client_id=decision.target_client_id,
                            load_time=time.time(),
                            last_access_time=time.time(),
                            access_count=0,
                            size=page_sizes.get(decision.virtual_page_id, 0),
                            tenant_id=request.tenant_id,
                        )
                        allocated_locations[decision.virtual_page_id] = [location]
                    else:
                        failed_pages.append(decision.virtual_page_id)
                        logger.warning(
                            f"Failed to load page {decision.virtual_page_id} on "
                            f"{decision.target_deployment}/{decision.target_client_id}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error allocating page {decision.virtual_page_id}: {e}",
                        exc_info=True,
                    )
                    failed_pages.append(decision.virtual_page_id)

            allocation_time_ms = (time.time() - start_time) * 1000

            response = PageAllocationResponse(
                allocated_locations=allocated_locations,
                failed_pages=failed_pages,
                evicted_pages=evicted_pages,
                allocation_time_ms=allocation_time_ms,
            )

            logger.info(
                f"Allocation complete: {len(allocated_locations)} succeeded, "
                f"{len(failed_pages)} failed, {len(evicted_pages)} evicted, "
                f"time={allocation_time_ms:.2f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Fatal error during page allocation: {e}", exc_info=True)
            raise RuntimeError(f"Page allocation failed: {e}") from e

    @serving.endpoint
    async def request_page_load(
        self,
        page_id: ContextPageId,
        priority: int = 0,
        agent_id: str | None = None,
        tenant_id: str = "default",
        lock_duration_s: float | None = None,
        lock_reason: str = "",
    ) -> str | None:
        """Request a page to be loaded (adds to page fault queue if not loaded).

        This is non-blocking - it adds the request to a priority queue and
        returns immediately. The page will be loaded asynchronously by a
        background task.

        If lock_duration_s is specified, the page will be locked after loading to
        prevent eviction during critical operations (e.g., multi-turn agent workflows).

        Args:
            page_id: Page identifier
            priority: Load priority (higher = more urgent)
            agent_id: Requesting agent ID (optional)
            tenant_id: Tenant ID (optional)
            lock_duration_s: If set, lock page after loading for this duration (seconds)
            lock_reason: Reason for locking (if lock_duration_s is set)

        Returns:
            None if page already loaded, fault_id (Unique identifier for tracking this fault) if added to fault queue
        """
        # Check if already loaded
        if await self.page_table.is_page_loaded(page_id):
            # Update access time
            await self.page_table.update_page_access(page_id)

            # Lock if requested
            if lock_duration_s is not None and lock_duration_s > 0:
                await self.page_table.lock_page(
                    page_id=page_id,
                    locked_by=agent_id or "unknown",
                    lock_duration_s=lock_duration_s,
                    reason=lock_reason,
                )

            return True

        # Add to page fault queue (with lock request if specified)
        fault_id = self.issue_page_fault(
            page_ids=[page_id],
            requester_id=agent_id or "unknown",
            priority=priority,
            tenant_id=tenant_id,
            lock_duration_s=lock_duration_s,
            lock_reason=lock_reason,
        )

        return fault_id

    @serving.endpoint
    async def request_group_load(
        self,
        group_id: str,
        priority: int | None = None,
        agent_id: str | None = None,
        tenant_id: str = "default",
    ) -> None:
        """Request all pages in a group to be loaded.

        Args:
            group_id: Group identifier
            priority: Optional priority override (uses group priority if None)
            agent_id: Requesting agent ID (optional)
            tenant_id: Tenant ID (optional)
        """
        # Get group
        group = await self.page_table.get_page_group(group_id)
        if not group:
            raise ValueError(f"Group {group_id} not found")

        # Use group priority if not overridden
        load_priority = priority if priority is not None else group.priority

        # Request each page
        # TODO: The page loader should support group loading directly for efficiency
        for page_id in group.page_ids:
            await self.request_page_load(page_id, load_priority, agent_id, tenant_id=tenant_id)

        logger.info(
            f"Requested load for group {group_id} ({len(group.page_ids)} pages, "
            f"priority={load_priority})"
        )

    # === Page Fault Management (for Routers) ===

    @serving.endpoint
    async def issue_page_fault(
        self,
        page_ids: list[str],
        requester_id: str,
        priority: int = 10,
        tenant_id: str = "default",
        lock_duration_s: float | None = None,
        lock_reason: str = "",
    ) -> str:
        """Issue a page fault for missing pages (called by routers).

        This is the primary mechanism for routers to request pages that aren't loaded.
        The fault is queued and processed asynchronously by the background task.

        Args:
            page_ids: List of page IDs that need to be loaded
            requester_id: ID of the requester (router, agent, etc.)
            priority: Fault priority (higher = more urgent, default=10 for router requests)
            tenant_id: Tenant that owns these pages
            lock_duration_s: If set, lock pages after loading for this duration (seconds)
            lock_reason: Reason for locking (if lock_duration_s is set)

        Returns:
            fault_id: Unique identifier for tracking this fault
        """
        # Create page fault with batched pages
        fault = PageFault(
            page_ids=page_ids,
            requesting_agent_id=requester_id,
            priority=priority,
            tenant_id=tenant_id,
            lock_duration_s=lock_duration_s,
            lock_reason=lock_reason,
        )

        # Create event for waiters
        event = asyncio.Event()
        self._page_fault_events[fault.fault_id] = event

        # Add to queue
        await self.page_table.add_page_fault(fault)

        logger.info(
            f"Issued page fault {fault.fault_id} for {len(page_ids)} pages "
            f"(priority={priority}, requester={requester_id}, tenant={tenant_id}, lock_duration={lock_duration_s}s)"
        )

        return fault.fault_id

    @serving.endpoint
    async def wait_for_pages(
        self,
        fault_id: str,
        timeout_s: float = 30.0,
    ) -> bool:
        """Wait for a page fault to be processed (called by routers after issuing fault).

        This uses asyncio.Event for efficient waiting without polling.

        Args:
            fault_id: Fault ID returned by issue_page_fault()
            timeout_s: Maximum time to wait (default: 30 seconds)

        Returns:
            True if pages were loaded successfully, False if timeout or fault not found
        """
        event = self._page_fault_events.get(fault_id)
        if not event:
            # Fault already processed or doesn't exist
            logger.debug(f"Page fault {fault_id} already processed or not found")
            return False

        try:
            # Wait with timeout
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
            logger.debug(f"Page fault {fault_id} completed successfully")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Page fault {fault_id} timed out after {timeout_s}s")
            return False
        finally:
            # Clean up event
            if fault_id in self._page_fault_events:
                del self._page_fault_events[fault_id]

    # === Page Retrieval ===

    @serving.endpoint
    def get_page_locations(self, virtual_page_id: str) -> list[PageLocation]:
        """Get all locations where a virtual page is loaded.

        Args:
            virtual_page_id: Virtual page identifier
        Returns:
            List of PageLocation records
        """
        return self.page_table.get_page_locations(virtual_page_id)

    @serving.endpoint
    async def get_virtual_page(self, page_id: ContextPageId) -> VirtualContextPage | None:
        """Get a virtual page by ID from persistent storage.

        Args:
            page_id: Page identifier

        Returns:
            VirtualContextPage if exists, None otherwise
        """
        return await self.page_storage.retrieve_page(page_id)

    @serving.endpoint
    async def is_page_loaded(self, page_id: str) -> bool:
        """Check if page is loaded in any replica.

        Args:
            page_id: Page identifier

        Returns:
            True if loaded
        """
        return await self.page_table.is_page_loaded(page_id)

    @serving.endpoint
    async def get_page_location(self, page_id: str) -> PageLocation | None:
        """Get which replica has this page loaded.

        Args:
            page_id: Page identifier

        Returns:
            PageLocation if loaded, None otherwise
        """
        return await self.page_table.get_page_location(page_id)

    @serving.endpoint
    async def get_all_loaded_pages(self) -> list[str]:
        """Get all pages currently loaded across all replicas.

        Returns:
            List of page IDs that are currently loaded in VCM
        """
        return await self.page_table.get_all_loaded_pages()

    async def _page_is_on_target(
        self,
        page_id: str,
        target_deployment_name: str,
        target_client_id: str,
    ) -> bool:
        """Check if page is loaded on specific target replica.

        Args:
            page_id: Virtual page identifier
            target_deployment_name: Target VLLM deployment name
            target_client_id: Target VLLM replica ID
        Returns:
            True if page is on target replica
        """
        locations = await self.page_table.get_page_locations(page_id)
        for loc in locations:
            if (loc.deployment_name == target_deployment_name and
                loc.client_id == target_client_id):
                return True
        return False

    @serving.endpoint
    async def replicate_page(
        self,
        page_id: str,
        target_deployment_name: str,
        target_client_id: str,
        priority: int = 20,
        timeout_s: float = 60.0,
    ) -> bool:
        """Replicate a page to a specific VLLM replica.

        This is used by routers (e.g., SoftPageAffinityRouter) to ensure
        pages are loaded on specific replicas for agent scheduling.

        This method:
        1. Checks if page already exists on target replica → return True
        2. Retrieves page from storage
        3. Issues targeted page allocation request
        4. Waits for page to be loaded (with timeout)

        Args:
            page_id: Virtual page identifier
            target_deployment_name: Target VLLM deployment name
            target_client_id: Target VLLM replica ID
            priority: Page loading priority (default: 20 for agent spawning)
            timeout_s: Maximum time to wait for replication

        Returns:
            True if page was replicated successfully, False otherwise

        Raises:
            ValueError: If page doesn't exist in storage
        """
        # Check if page already loaded on target
        if await self._page_is_on_target(
            page_id,
            target_deployment_name,
            target_client_id,
        ):
            logger.debug(
                f"Page {page_id} already loaded on "
                f"{target_deployment_name}/{target_client_id}"
            )
            return True

        # Verify page exists in storage
        page = await self.page_storage.retrieve_page(page_id)
        if not page:
            raise ValueError(f"Page {page_id} not found in storage")

        logger.info(
            f"Replicating page {page_id} to "
            f"{target_deployment_name}/{target_client_id}"
        )

        # Issue targeted page load request
        # This uses the allocation system but targets a specific client
        allocation_request = PageAllocationRequest(
            virtual_page_ids=[page_id],
            tenant_id="system",  # System-initiated replication
            priority=priority,
            preferred_deployment=target_deployment_name,
            target_client_id=target_client_id,  # Override allocation strategy
        )

        try:
            response = await self.allocate_pages(allocation_request)

            # Check if page was loaded successfully
            if page_id in response.failed_pages:
                logger.error(
                    f"Failed to replicate page {page_id}: "
                    f"{response.failed_pages[page_id]}"
                )
                return False

            # Verify page is now on target
            if await self._page_is_on_target(
                page_id,
                target_deployment_name,
                target_client_id,
            ):
                logger.info(
                    f"Successfully replicated page {page_id} to "
                    f"{target_deployment_name}/{target_client_id}"
                )
                return True

            logger.error(
                f"Page {page_id} not found on target after allocation "
                f"(allocated to: {response.allocated_locations.get(page_id)})"
            )
            return False

        except Exception as e:
            logger.error(f"Error replicating page {page_id}: {e}", exc_info=True)
            return False

    # === Page Group Management ===

    @serving.endpoint
    async def create_page_group(
        self,
        page_ids: list[str],
        group_id: str | None = None,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PageGroup:
        """Create a page group for spatial locality.

        Args:
            page_ids: List of page IDs in this group
            group_id: Optional group ID (generated if None)
            priority: Load priority for entire group
            metadata: Optional metadata

        Returns:
            Created PageGroup
        """
        if group_id is None:
            import uuid

            group_id = f"group-{uuid.uuid4().hex[:8]}"

        group = PageGroup(
            group_id=group_id,
            page_ids=page_ids,
            priority=priority,
            metadata=metadata or {},
        )

        await self.page_table.register_page_group(group)

        logger.info(f"Created page group {group_id} with {len(page_ids)} pages")

        return group

    # === Helper Methods ===

    async def _get_deployment_state_manager(self, deployment_name: str):
        """Get or create a state manager for a deployment's VLLMDeploymentState.

        This is cached to avoid creating multiple state managers for the same deployment.

        Args:
            deployment_name: Name of the VLLM deployment

        Returns:
            StateManager[VLLMDeploymentState] for this deployment
        """
        if deployment_name not in self._deployment_state_managers:
            from ..cluster.models import VLLMDeploymentState
            from ..distributed import get_initialized_polymathera

            polymathera = await get_initialized_polymathera()
            state_key = VLLMDeploymentState.get_state_key(self.app_name, deployment_name)
            state_manager = await polymathera.get_state_manager(
                state_type=VLLMDeploymentState,
                state_key=state_key,
            )
            self._deployment_state_managers[deployment_name] = state_manager
            logger.debug(f"Created state manager for deployment {deployment_name}: {state_key}")

        return self._deployment_state_managers[deployment_name]

    async def _load_page_on_client(
        self,
        virtual_page_id: ContextPageId,
        deployment_name: str,
        client_id: str,
        page_size: int,
        tenant_id: str,
    ) -> bool:
        """Load a page on a specific client.

        This method coordinates with the VLLMDeployment to load the page
        and updates the page table.

        Args:
            virtual_page_id: Virtual page ID
            deployment_name: Target deployment
            client_id: Target client
            page_size: Page size in tokens
            tenant_id: Tenant ID

        Returns:
            True if successful
        """
        try:
            # Get actual page from cache (we ARE the VCM)
            page = await self.get_virtual_page(virtual_page_id)
            if not page:
                raise ValueError(f"Page {virtual_page_id} not found in VCM cache")

            # Load page on specific client via LLMCluster
            # This ensures the page is loaded on the correct replica as decided by allocation strategy
            success = await self.llm_cluster_handle.load_page(
                page=page,
                deployment_name=deployment_name,
                client_id=client_id,
            )
            if not success:
                return False

            # Update VCM page table (Layer 2)
            # Note: Layer 1 (VLLMDeploymentState) is updated by VLLMDeployment.load_page()
            # In Phase 2, this Layer 2 update will be moved to event handler
            await self.page_table.register_loaded_page(
                page_id=virtual_page_id,
                replica_id=client_id,
                deployment_name=deployment_name,
                tenant_id=tenant_id,
                size=page_size,
            )

            logger.debug(
                f"Loaded page {virtual_page_id} on {deployment_name}/{client_id} "
                f"(VCM page table updated, deployment state updated by VLLMDeployment)"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error loading page {virtual_page_id} on {deployment_name}/{client_id}: {e}",
                exc_info=True,
            )
            return False

    async def _evict_page_from_client(
        self,
        virtual_page_id: ContextPageId,
        deployment_name: str,
        client_id: str,
    ) -> bool:
        """Evict a page from a specific client.

        Args:
            virtual_page_id: Virtual page ID
            deployment_name: Deployment name
            client_id: Client ID

        Returns:
            True if successful
        """
        try:
            # Get VLLMDeployment handle and call evict_page
            deployment_handle = serving.get_deployment(self.app_name, deployment_name)
            success = await deployment_handle.evict_page(virtual_page_id)

            if not success:
                logger.warning(
                    f"Failed to evict page {virtual_page_id} from {deployment_name}/{client_id}"
                )
                return False

            # Update VCM page table (Layer 2)
            # Note: Layer 1 (VLLMDeploymentState) is updated by VLLMDeployment.evict_page()
            # In Phase 2, this Layer 2 update will be moved to event handler
            async for state in self.page_table.state_manager.write_transaction():
                state.register_page_eviction(
                    virtual_page_id=virtual_page_id,
                    deployment_name=deployment_name,
                    client_id=client_id,
                )

            logger.debug(
                f"Evicted page {virtual_page_id} from {deployment_name}/{client_id} "
                f"(VCM page table updated, deployment state updated by VLLMDeployment)"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error evicting page {virtual_page_id} from {deployment_name}/{client_id}: {e}",
                exc_info=True,
            )
            return False

    # === Background Tasks ===

    @serving.periodic_health_check(interval_s=5.0)
    async def process_page_faults_background(self):
        """Periodic task to process page fault queue.

        This background task:
        1. Pops faults from the priority queue
        2. Uses allocate_pages to load missing pages
        3. Signals asyncio.Event to notify waiters
        4. Supports batching and optimization

        Processes up to 10 page faults every 5 seconds.
        """
        try:
            processed = 0

            # TODO: For efficiency, implement optimized batching of faults
            # based on priority, tenant, spatial locality, previous rounds of
            # inference, etc.
            for _ in range(10):  # Process up to 10 faults per cycle
                # Pop next fault from queue
                fault = await self.page_table.get_next_page_fault()
                if not fault:
                    break  # No more faults

                logger.debug(
                    f"Processing page fault {fault.fault_id} for {len(fault.page_ids)} pages "
                    f"(priority={fault.priority})"
                )

                try:
                    # Check which pages are already loaded
                    pages_to_load = []
                    for page_id in fault.page_ids:
                        if not await self.page_table.is_page_loaded(page_id):
                            pages_to_load.append(page_id)

                    if not pages_to_load:
                        logger.debug(
                            f"All pages for fault {fault.fault_id} already loaded"
                        )
                        # Signal success even though nothing was done
                        if fault.fault_id in self._page_fault_events:
                            self._page_fault_events[fault.fault_id].set()
                        processed += 1
                        continue

                    # Create allocation request
                    allocation_request = PageAllocationRequest(
                        virtual_page_ids=pages_to_load,
                        tenant_id=fault.tenant_id,
                        priority=fault.priority,
                        # Could add affinity_pages here for spatial locality optimization
                    )

                    # Allocate pages
                    response = await self.allocate_pages(allocation_request)

                    # Check if all pages were allocated successfully
                    success = len(response.failed_pages) == 0

                    # Lock pages if requested in the fault
                    if fault.lock_duration_s is not None and fault.lock_duration_s > 0:
                        for page_id in response.allocated_locations.keys():
                            try:
                                await self.page_table.lock_page(
                                    page_id=page_id,
                                    locked_by=fault.requesting_agent_id or "unknown",
                                    lock_duration_s=fault.lock_duration_s,
                                    reason=fault.lock_reason,
                                )
                                logger.debug(
                                    f"Locked page {page_id} for {fault.lock_duration_s}s "
                                    f"after loading (fault {fault.fault_id})"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to lock page {page_id} after loading: {e}")

                    if success:
                        logger.info(
                            f"Successfully processed page fault {fault.fault_id}: "
                            f"{len(pages_to_load)} pages loaded, "
                            f"{len(response.evicted_pages)} pages evicted"
                        )
                    else:
                        logger.warning(
                            f"Partially processed page fault {fault.fault_id}: "
                            f"{len(response.allocated_locations)} succeeded, "
                            f"{len(response.failed_pages)} failed"
                        )

                    # Signal event regardless of success/failure
                    # Waiters will check if pages are actually loaded
                    if fault.fault_id in self._page_fault_events:
                        self._page_fault_events[fault.fault_id].set()

                    processed += 1

                except Exception as e:
                    logger.error(
                        f"Error processing page fault {fault.fault_id}: {e}",
                        exc_info=True,
                    )
                    # Signal event even on error so waiters don't hang forever
                    if fault.fault_id in self._page_fault_events:
                        self._page_fault_events[fault.fault_id].set()

            if processed > 0:
                logger.info(f"Processed {processed} page faults")

        except Exception as e:
            logger.error(f"Error in page fault background processor: {e}", exc_info=True)

    @serving.periodic_health_check(interval_s=30.0)
    async def collect_metrics(self):
        """Collect VCM metrics for monitoring.

        Logs statistics every 30 seconds.
        """
        try:
            stats = await self.page_table.get_stats()
            logger.info(
                f"VCM Stats: {stats['total_pages_loaded']} pages loaded across "
                f"{stats['num_replicas']} replicas, {stats['pending_faults']} pending faults"
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    @serving.periodic_health_check(interval_s=60.0)
    async def cleanup_expired_page_locks(self):
        """Clean up expired page locks periodically.

        Runs every 60 seconds to remove expired locks and free up memory.
        """
        try:
            cleaned_count = await self.page_table.cleanup_expired_locks()
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired page locks")
        except Exception as e:
            logger.error(f"Error cleaning up expired locks: {e}")

    # === Page Locking Management ===

    @serving.endpoint
    async def lock_page(
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
        return await self.page_table.lock_page(
            page_id=page_id,
            locked_by=locked_by,
            lock_duration_s=lock_duration_s,
            reason=reason,
            current_time=current_time,
        )

    @serving.endpoint
    async def unlock_page(self, page_id: str) -> bool:
        """Unlock a page, allowing it to be evicted.

        Args:
            page_id: Page identifier

        Returns:
            True if page was unlocked, False if page wasn't locked
        """
        return await self.page_table.unlock_page(page_id)

    @serving.endpoint
    async def extend_page_lock(
        self,
        page_id: str,
        additional_duration_s: float,
    ) -> bool:
        """Extend the lock duration for a locked page.

        Args:
            page_id: Page identifier
            additional_duration_s: Additional seconds to add to lock duration

        Returns:
            True if lock was extended, False if page wasn't locked

        Raises:
            ValueError: If additional_duration_s is negative
        """
        return await self.page_table.extend_page_lock(page_id, additional_duration_s)

    @serving.endpoint
    async def get_page_lock_info(self, page_id: str) -> dict[str, Any] | None:
        """Get lock information for a page.

        Args:
            page_id: Page identifier

        Returns:
            Lock info dict if page is locked, None otherwise
        """
        lock = await self.page_table.get_page_lock(page_id)
        if not lock:
            return None

        return {
            "page_id": lock.page_id,
            "locked_by": lock.locked_by,
            "lock_expires_at": lock.lock_expires_at,
            "remaining_time_s": lock.remaining_time_s(),
            "reason": lock.reason,
            "created_at": lock.created_at,
        }

    # === Monitoring and Statistics ===

    @serving.endpoint
    async def get_stats(self) -> dict[str, Any]:
        """Get VCM statistics.

        Returns:
            Dictionary with comprehensive stats
        """
        page_table_stats = await self.page_table.get_stats()
        storage_stats = await self.page_storage.get_storage_stats()

        return {
            "app_name": self.app_name,
            "caching_policy": self.caching_policy,
            "page_table": page_table_stats,
            "storage": storage_stats,
        }

    # === Branch Management API (Copy-on-Write Support) ===

    @serving.endpoint
    async def create_branch(
        self,
        tenant_id: str,
        parent_branch_id: BranchId | None = None,
        name: str | None = None,
    ) -> VCMBranch:
        """Create a new branch for copy-on-write operations.

        When a branch is created:
        1. If parent_branch_id is None, create root branch (main)
        2. If parent exists, fork from parent's current state
        3. Child branch inherits parent's pages (no copying yet)
        4. First modification triggers copy-on-write

        Args:
            tenant_id: Owning tenant
            parent_branch_id: Parent branch to fork from (None for root)
            name: Human-readable name

        Returns:
            Created VCMBranch
        """
        import uuid

        branch_id = f"branch_{uuid.uuid4().hex[:12]}"

        # Get parent branch if forking
        parent = None
        base_snapshot: set[str] = set()

        if parent_branch_id:
            parent = await self.page_table.get_branch(parent_branch_id)
            if not parent:
                raise ValueError(f"Parent branch {parent_branch_id} not found")
            # Inherit parent's effective pages (including its inherited pages)
            base_snapshot = await self.page_table.get_branch_effective_pages(parent_branch_id)

        branch = VCMBranch(
            branch_id=branch_id,
            tenant_id=tenant_id,
            parent_branch_id=parent_branch_id,
            name=name or f"branch_{branch_id[:8]}",
            base_snapshot=base_snapshot,
            forked_at_version=len(parent.overlays) if parent else 0,
        )

        # Register branch in page table
        await self.page_table.register_branch(branch)

        logger.info(
            f"Created branch {branch_id} for tenant {tenant_id}"
            f"{f' (forked from {parent_branch_id})' if parent_branch_id else ''}"
        )

        return branch

    @serving.endpoint
    async def get_branch(self, branch_id: BranchId) -> VCMBranch | None:
        """Get a branch by ID.

        Args:
            branch_id: Branch identifier

        Returns:
            VCMBranch if found, None otherwise
        """
        return await self.page_table.get_branch(branch_id)

    @serving.endpoint
    async def list_branches(self, tenant_id: str) -> list[VCMBranch]:
        """List all branches for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of VCMBranch objects for this tenant
        """
        return await self.page_table.list_branches(tenant_id)

    @serving.endpoint
    async def get_page_for_branch(
        self,
        page_id: str,
        branch_id: BranchId,
    ) -> VirtualContextPage | None:
        """Get a page as seen from a specific branch.

        Resolves copy-on-write overlays: if this branch or any ancestor
        has an overlay for this page, return the overlay. Otherwise,
        return the original page.

        Args:
            page_id: Base page ID
            branch_id: Branch to read from

        Returns:
            VirtualContextPage (overlay or original)
        """
        # Resolve effective page ID through overlay chain
        effective_page_id = await self.page_table.get_effective_page_id(page_id, branch_id)

        # Retrieve from storage
        return await self.page_storage.retrieve_page(effective_page_id)

    @serving.endpoint
    async def modify_page_on_branch(
        self,
        page_id: str,
        branch_id: BranchId,
        new_tokens: list[int],
        modifier_id: str,
        metadata_updates: dict[str, Any] | None = None,
    ) -> VirtualContextPage:
        """Modify a page on a branch with copy-on-write.

        On first modification:
        1. Create a new overlay page with the modifications
        2. Register overlay in branch's overlay map
        3. Original page remains unchanged

        Subsequent modifications update the existing overlay.

        Args:
            page_id: Base page ID to modify
            branch_id: Branch to modify on
            new_tokens: New token content
            modifier_id: ID of who is modifying (agent_id, session_id)
            metadata_updates: Additional metadata to merge

        Returns:
            The overlay page (new or updated)
        """
        # Get branch
        branch = await self.page_table.get_branch(branch_id)
        if not branch:
            raise ValueError(f"Branch {branch_id} not found")

        # Check if we already have an overlay on this exact branch
        existing_overlay_id = branch.overlays.get(page_id)

        if existing_overlay_id:
            # Update existing overlay
            overlay = await self.page_storage.retrieve_page(existing_overlay_id)
            if not overlay:
                raise ValueError(f"Overlay page {existing_overlay_id} not found in storage")

            # Create updated overlay with new tokens
            updated_overlay = VirtualContextPage(
                page_id=overlay.page_id,
                tokens=new_tokens,
                size=len(new_tokens),
                tenant_id=overlay.tenant_id,
                branch_id=branch_id,
                parent_page_id=overlay.parent_page_id,
                is_overlay=True,
                base_version=overlay.base_version,
                metadata={
                    **overlay.metadata,
                    "last_modified_by": modifier_id,
                    "last_modification_time": time.time(),
                    **(metadata_updates or {}),
                },
            )

            await self.page_storage.store_page(updated_overlay)
            logger.info(f"Updated overlay {existing_overlay_id} on branch {branch_id}")
            return updated_overlay

        else:
            # Create new overlay (copy-on-write)
            # First, get the effective page (could be parent's overlay or original)
            effective_page_id = await self.page_table.get_effective_page_id(page_id, branch_id)

            base_page = await self.page_storage.retrieve_page(effective_page_id)
            if not base_page:
                raise ValueError(f"Base page {effective_page_id} not found")

            # Generate overlay ID
            import uuid
            overlay_id = f"{page_id}_overlay_{uuid.uuid4().hex[:8]}"

            overlay = VirtualContextPage(
                page_id=overlay_id,
                tokens=new_tokens,
                size=len(new_tokens),
                tenant_id=branch.tenant_id,
                branch_id=branch_id,
                parent_page_id=effective_page_id,
                is_overlay=True,
                base_version=base_page.metadata.get("version", 0),
                metadata={
                    **base_page.metadata,
                    "forked_from": effective_page_id,
                    "modified_by": modifier_id,
                    "modification_time": time.time(),
                    **(metadata_updates or {}),
                },
            )

            # Store overlay page
            await self.page_storage.store_page(overlay)

            # Register overlay in branch
            await self.page_table.register_overlay(branch_id, page_id, overlay_id)

            logger.info(f"Created CoW overlay {overlay_id} for page {page_id} on branch {branch_id}")
            return overlay

    @serving.endpoint
    async def create_page_on_branch(
        self,
        branch_id: BranchId,
        tokens: list[int],
        creator_id: str,
        metadata: dict[str, Any] | None = None,
        page_id: str | None = None,
    ) -> VirtualContextPage:
        """Create a new page on a branch (not an overlay of an existing page).

        Unlike modify_page_on_branch, this creates a truly new page that
        doesn't have a parent. Used for adding new content to a branch.

        Args:
            branch_id: Branch to create page on
            tokens: Token content
            creator_id: ID of who is creating the page
            metadata: Optional metadata
            page_id: Optional page ID (generated if None)

        Returns:
            Created VirtualContextPage
        """
        import uuid

        # Get branch
        branch = await self.page_table.get_branch(branch_id)
        if not branch:
            raise ValueError(f"Branch {branch_id} not found")

        # Generate page ID if not provided
        if page_id is None:
            page_id = f"page_{uuid.uuid4().hex[:12]}"

        page = VirtualContextPage(
            page_id=page_id,
            tokens=tokens,
            size=len(tokens),
            tenant_id=branch.tenant_id,
            branch_id=branch_id,
            parent_page_id=None,  # Not an overlay
            is_overlay=False,
            metadata={
                "created_by": creator_id,
                "creation_time": time.time(),
                **(metadata or {}),
            },
        )

        # Store page
        await self.page_storage.store_page(page)

        # Register as new page on branch
        await self.page_table.register_new_page_on_branch(branch_id, page_id)

        logger.info(f"Created new page {page_id} on branch {branch_id}")
        return page

    @serving.endpoint
    async def merge_branches(
        self,
        source_branch_id: BranchId,
        target_branch_id: BranchId,
        strategy: str = "last_write_wins",
    ) -> dict[str, Any]:
        """Merge source branch into target branch.

        Merge strategies:
        - "last_write_wins": Source overlays overwrite target overlays
        - "first_write_wins": Keep target overlays, ignore source conflicts
        - "fail_on_conflict": Raise error if both have overlays for same page

        Args:
            source_branch_id: Branch to merge from
            target_branch_id: Branch to merge into
            strategy: Merge strategy

        Returns:
            Merge result with conflicts and applied changes
        """
        source = await self.page_table.get_branch(source_branch_id)
        target = await self.page_table.get_branch(target_branch_id)

        if not source or not target:
            return {"success": False, "error": "Source or target branch not found"}

        conflicts: list[dict[str, Any]] = []
        merged: list[str] = []
        new_pages_merged: list[str] = []

        # Merge overlays (modifications to existing pages)
        for original_id, overlay_id in source.overlays.items():
            if original_id in target.overlays:
                # Conflict: both branches modified same page
                if strategy == "last_write_wins":
                    target.overlays[original_id] = overlay_id
                    merged.append(original_id)
                elif strategy == "first_write_wins":
                    pass  # Keep target's overlay
                elif strategy == "fail_on_conflict":
                    conflicts.append({
                        "page_id": original_id,
                        "source_overlay": overlay_id,
                        "target_overlay": target.overlays[original_id],
                    })
            else:
                # No conflict: apply source's overlay
                target.overlays[original_id] = overlay_id
                merged.append(original_id)

        # Merge new pages (pages created on source branch with no parent)
        for page_id in source.new_pages:
            if page_id in target.new_pages:
                # Both branches created page with same ID (unlikely with UUIDs)
                conflicts.append({
                    "page_id": page_id,
                    "type": "new_page_conflict",
                    "source_page": page_id,
                    "target_page": page_id,
                })
            else:
                target.new_pages.add(page_id)
                new_pages_merged.append(page_id)

        if conflicts and strategy == "fail_on_conflict":
            return {
                "success": False,
                "error": "Merge conflicts detected",
                "conflicts": conflicts,
            }

        # Mark source as merged
        source.merged_into = target_branch_id
        source.state = "merged"

        # Update branches in page table
        await self.page_table.register_branch(target)
        await self.page_table.register_branch(source)

        logger.info(
            f"Merged branch {source_branch_id} into {target_branch_id}: "
            f"{len(merged)} overlays, {len(new_pages_merged)} new pages"
        )

        return {
            "success": len(conflicts) == 0 or strategy != "fail_on_conflict",
            "merged_pages": merged,
            "new_pages_merged": new_pages_merged,
            "conflicts": conflicts,
        }

    # === Blackboard ↔ VCM Scope Mapping API ===

    @serving.endpoint
    async def mmap_blackboard_scope(
        self,
        scope_id: str,
        config: MmapConfig | None = None,
        tenant_id: str | None = None,
    ) -> MmapResult:
        """Map a blackboard/memory scope into VCM pages.

        This is the main entry point for the Blackboard ↔ VCM integration.
        When a scope is mapped, a BlackboardContextPageSource is created to
        watch the scope for writes and automatically page them into VCM pages.

        The mapping is recorded in shared state so all VCM replicas can
        materialize it during reconciliation. Each replica joins the same
        Redis Streams consumer group for event deduplication.

        Args:
            scope_id: The scope to map (e.g., "tenant:acme:discoveries")
            config: Mapping configuration (controls flushing, locality, etc.)
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            MmapResult with status and message
        """
        config = config or MmapConfig()

        # Check if already mapped
        async for state in self.page_table.state_manager.read_transaction():
            if scope_id in state.mapped_scopes:
                return MmapResult(
                    status="already_mapped",
                    scope_id=scope_id,
                    message=f"Scope {scope_id} is already mapped",
                )

        # Record mapping in shared state (visible to all replicas)
        mapping_config = MappedScopeConfig(
            scope_id=scope_id,
            config=config,
            tenant_id=tenant_id,
        )

        async for state in self.page_table.state_manager.write_transaction():
            state.mapped_scopes[scope_id] = mapping_config

        # Materialize locally on this replica
        try:
            await self._materialize_scope_mapping(scope_id, config, tenant_id)
        except Exception as e:
            logger.error(f"Failed to materialize scope mapping for {scope_id}: {e}", exc_info=True)
            return MmapResult(
                status="error",
                scope_id=scope_id,
                message=f"Mapping recorded but materialization failed: {e}",
            )

        logger.info(f"Mapped scope {scope_id} into VCM (tenant={tenant_id})")
        return MmapResult(
            status="mapped",
            scope_id=scope_id,
            message=f"Scope {scope_id} mapped successfully",
        )

    @serving.endpoint
    async def munmap_blackboard_scope(self, scope_id: str) -> MmapResult:
        """Unmap a blackboard/memory scope from VCM.

        Flushes any pending records, shuts down the page source, and
        removes the mapping from shared state.

        Args:
            scope_id: The scope to unmap

        Returns:
            MmapResult with status and message
        """
        # Check if mapped
        is_mapped = False
        async for state in self.page_table.state_manager.read_transaction():
            is_mapped = scope_id in state.mapped_scopes

        if not is_mapped:
            return MmapResult(
                status="not_mapped",
                scope_id=scope_id,
                message=f"Scope {scope_id} is not mapped",
            )

        # Shut down local page source if materialized
        if scope_id in self._local_mapped_scopes:
            mapped = self._local_mapped_scopes.pop(scope_id)
            await mapped.source.shutdown()
            logger.info(f"Shut down local page source for scope {scope_id}")

        # Remove from shared state
        async for state in self.page_table.state_manager.write_transaction():
            state.mapped_scopes.pop(scope_id, None)

        logger.info(f"Unmapped scope {scope_id} from VCM")
        return MmapResult(
            status="unmapped",
            scope_id=scope_id,
            message=f"Scope {scope_id} unmapped successfully",
        )

    @serving.endpoint
    async def get_pages_for_scope(
        self,
        scope_id: str,
        include_metadata: bool = False,
    ) -> list[dict]:
        """List all VCM pages created from a blackboard scope.

        Uses PageStorage.query_pages_by_metadata() to find pages with
        source=f"bb:{scope_id}".

        Args:
            scope_id: The scope to query
            include_metadata: Whether to include full page metadata

        Returns:
            List of page info dicts
        """
        pages = await self.page_storage.query_pages_by_metadata(
            filters={"scope_id": scope_id},
        )
        result = []
        for page in pages:
            info = {
                "page_id": page.page_id,
                "size": page.size,
                "group_id": page.group_id,
            }
            if include_metadata:
                info["metadata"] = page.metadata
            result.append(info)
        return result

    @serving.endpoint
    async def is_blackboard_scope_mapped(self, scope_id: str) -> bool:
        """Check if a blackboard scope is currently mapped into VCM.

        Args:
            scope_id: The scope to check

        Returns:
            True if mapped, False otherwise
        """
        async for state in self.page_table.state_manager.read_transaction():
            return scope_id in state.mapped_scopes
        return False

    @serving.endpoint
    async def get_blackboard_scope_mapping_status(self, scope_id: str) -> dict | None:
        """Get detailed status of a blackboard scope mapping.

        Args:
            scope_id: The scope to check

        Returns:
            Dict with mapping details, or None if not mapped
        """
        async for state in self.page_table.state_manager.read_transaction():
            mapping = state.mapped_scopes.get(scope_id)
            if mapping is None:
                return None
            result = {
                "scope_id": mapping.scope_id,
                "config": mapping.config.model_dump(),
                "tenant_id": mapping.tenant_id,
                "created_at": mapping.created_at,
                "materialized_locally": scope_id in self._local_mapped_scopes,
            }
            # Add page count if materialized
            if scope_id in self._local_mapped_scopes:
                source = self._local_mapped_scopes[scope_id].source
                result["page_count"] = len(source._page_graph.nodes)
                result["tracked_records"] = len(source._record_to_page)
            return result
        return None

    @serving.endpoint
    async def get_blackboard_records_in_page(self, page_id: str) -> list[str]:
        """Get the blackboard record keys that were ingested into a VCM page.

        This is a provenance/debugging endpoint for understanding what
        blackboard records are in a given page.

        Args:
            page_id: VCM page ID

        Returns:
            List of blackboard record keys in the page
        """
        # Search across all local mapped scopes for the record_to_page mapping
        for _scope_id, mapped in self._local_mapped_scopes.items():
            records = [
                key for key, pid in mapped.source._record_to_page.items()
                if pid == page_id
            ]
            if records:
                return records

        # Fallback: check page metadata for record_keys
        page = await self.page_storage.retrieve_page(page_id)
        if page and page.metadata:
            return page.metadata.get("record_keys", [])
        return []

    # === Event-Driven State Reconciliation (Phase 2) ===

    async def _subscribe_to_page_events(self):
        """Subscribe to page lifecycle events from all deployments.

        This sets up Redis pub/sub subscriptions to receive PageLoadedEvent,
        PageEvictedEvent, and PageLoadFailedEvent from all VLLMDeployments.
        """
        # Get all deployment names
        try:
            deployment_names = await self.llm_cluster_handle.get_all_deployment_names()
            logger.info(f"Subscribing to page events from {len(deployment_names)} deployments")
        except Exception as e:
            logger.warning(f"Failed to get deployment names: {e}")
            deployment_names = []

        from ..distributed.redis_utils.redis_om import RedisOM

        for deployment_name in deployment_names:
            try:
                event_namespace = f"vllm_events:{deployment_name}"

                # Create RedisOM for this deployment's event channel
                redis_om = RedisOM(
                    redis_client=self.redis_client,
                    namespace=event_namespace,
                )

                # Subscribe to "vcm_page_events" topic
                subscriber = redis_om.subscribe_to_state_updates("vcm_page_events")

                # Start listening with our event handler
                await subscriber.start(callback=self._handle_page_event)

                self.event_subscribers[deployment_name] = subscriber

                logger.info(f"Subscribed to page events from deployment: {deployment_name}")
            except Exception as e:
                logger.error(f"Failed to subscribe to events from {deployment_name}: {e}", exc_info=True)

    async def _handle_page_event(self, update: Any, error: Exception | None) -> bool:
        """Handle incoming page lifecycle event.

        This is called by DistributedStateSubscriber when events arrive from Redis.

        Args:
            update: State update from Redis pub/sub
            error: Error if subscription failed

        Returns:
            True to continue subscription, False to stop
        """
        if error:
            logger.error(f"Error in page event subscription: {error}")
            return True  # Continue subscription despite error

        try:
            from .events import PageLoadedEvent, PageEvictedEvent, PageLoadFailedEvent

            # Extract event data from update
            event_data = update.data.get("event_data")
            if not event_data:
                logger.warning(f"Received update without event_data: {update}")
                return True

            event_type = event_data.get("event_type")

            # Dispatch to appropriate handler
            if event_type == "page_loaded":
                event = PageLoadedEvent(**event_data)
                await self._on_page_loaded(event)
            elif event_type == "page_evicted":
                event = PageEvictedEvent(**event_data)
                await self._on_page_evicted(event)
            elif event_type == "page_load_failed":
                event = PageLoadFailedEvent(**event_data)
                await self._on_page_load_failed(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")

            return True  # Continue subscription

        except Exception as e:
            logger.error(f"Error handling page event: {e}", exc_info=True)
            return True  # Continue subscription despite error

    async def _on_page_loaded(self, event: Any):
        """Handle PageLoadedEvent - reconcile Layer 2 (VirtualPageTableState).

        This is triggered by VLLMDeployment after it updates Layer 1.
        """
        logger.info(
            f"VCM received PageLoadedEvent: page={event.page_id}, "
            f"deployment={event.deployment_name}, client={event.client_id}"
        )

        try:
            # Update VCM page table (Layer 2)
            await self.page_table.register_loaded_page(
                page_id=event.page_id,
                replica_id=event.client_id,
                deployment_name=event.deployment_name,
                tenant_id=event.tenant_id,
                size=event.size,
            )

            logger.debug(f"Updated VCM page table for page {event.page_id}")

            # Check if this resolves any pending page faults
            await self._resolve_page_faults_for_page(event.page_id)

        except Exception as e:
            logger.error(f"Error handling PageLoadedEvent: {e}", exc_info=True)

    async def _on_page_evicted(self, event: Any):
        """Handle PageEvictedEvent - reconcile Layer 2."""
        logger.info(
            f"VCM received PageEvictedEvent: page={event.page_id}, "
            f"deployment={event.deployment_name}, client={event.client_id}"
        )

        try:
            # Update VCM page table (Layer 2)
            async for state in self.page_table.state_manager.write_transaction():
                state.register_page_eviction(
                    virtual_page_id=event.page_id,
                    deployment_name=event.deployment_name,
                    client_id=event.client_id,
                )

            logger.debug(f"Updated VCM page table for evicted page {event.page_id}")

        except Exception as e:
            logger.error(f"Error handling PageEvictedEvent: {e}", exc_info=True)

    async def _on_page_load_failed(self, event: Any):
        """Handle PageLoadFailedEvent - update fault tracking."""
        logger.warning(
            f"VCM received PageLoadFailedEvent: page={event.page_id}, "
            f"error={event.error}"
        )

        # Log for debugging - could extend to update page fault tracking
        logger.error(
            f"Page load failed for {event.page_id} on "
            f"{event.deployment_name}/{event.client_id}: {event.error}"
        )

    async def _resolve_page_faults_for_page(self, page_id: str):
        """Resolve any pending page faults for a specific page.

        Called when a page is successfully loaded.
        """
        # Signal any pending events for this page
        # Note: We'd need to track which fault_id corresponds to which page_id
        # For now, this simplified implementation signals all pending faults
        events_to_signal = list(self._page_fault_events.keys())

        for fault_id in events_to_signal:
            if fault_id in self._page_fault_events:
                self._page_fault_events[fault_id].set()
                logger.debug(f"Resolved page fault {fault_id} due to page {page_id} loaded")

    # === Blackboard ↔ VCM Scope Mapping Internals ===

    async def _materialize_scope_mapping(
        self,
        scope_id: str,
        config: MmapConfig,
        tenant_id: str | None = None,
    ) -> None:
        """Create a BlackboardContextPageSource for a scope on this replica.

        This creates the actual infrastructure needed to watch a blackboard
        scope and page its contents into VCM:
        1. Creates an EnhancedBlackboard for the scope
        2. Wraps it in a BlackboardStorageBackend
        3. Creates a BlackboardContextPageSource with IngestionPolicy
        4. Initializes with consumer group for event deduplication
        5. Tracks in _local_mapped_scopes

        Args:
            scope_id: The scope being mapped
            config: Mapping configuration
            tenant_id: Tenant ID for multi-tenancy
        """
        if scope_id in self._local_mapped_scopes:
            logger.debug(f"Scope {scope_id} already materialized locally, skipping")
            return

        # Resolve ingestion policy from config
        ingestion_policy = self._resolve_ingestion_policy(config)

        # Create page source
        source = BlackboardContextPageSource(
            scope_id=scope_id,
            tokenizer=self._tokenizer,
            page_storage=self.page_storage,
            ingestion_policy=ingestion_policy,
        )

        # Initialize with consumer group for cross-replica deduplication
        consumer_group = f"cg:mmap:{scope_id}"
        consumer_name = serving.get_my_replica_id()
        await source.initialize(
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )

        # Track locally
        self._local_mapped_scopes[scope_id] = MappedScope(
            source=source,
            config=config,
            tenant_id=tenant_id,
        )

        # Pin pages if configured
        if config.pinned:
            await self._pin_scope_pages(scope_id)

        logger.info(
            f"Materialized scope mapping: scope={scope_id}, "
            f"consumer_group={consumer_group}, consumer={consumer_name}"
        )

    async def _reconcile_scope_mappings(self) -> None:
        """Synchronize local scope mappings with shared state.

        Called periodically (via _reconcile_page_state) and during init.
        - Materializes any new mappings that exist in shared state but
          not locally (e.g., added by another replica's mmap_blackboard_scope call).
        - Cleans up local mappings that no longer exist in shared state
          (e.g., removed by another replica's munmap_blackboard_scope call).
        - Claims orphaned events via XAUTOCLAIM for any mappings where
          a previous replica crashed mid-processing.
        """
        # Read shared state
        shared_scopes: dict[str, MappedScopeConfig] = {}
        async for state in self.page_table.state_manager.read_transaction():
            shared_scopes = dict(state.mapped_scopes)

        # Materialize new mappings
        for scope_id, mapping in shared_scopes.items():
            if scope_id not in self._local_mapped_scopes:
                try:
                    await self._materialize_scope_mapping(
                        scope_id=scope_id,
                        config=mapping.config,
                        tenant_id=mapping.tenant_id,
                    )
                    logger.info(f"Reconciliation: materialized scope mapping {scope_id}")
                except Exception as e:
                    logger.warning(
                        f"Reconciliation: failed to materialize scope {scope_id}: {e}"
                    )

        # Clean up stale local mappings
        stale = [
            sid for sid in self._local_mapped_scopes
            if sid not in shared_scopes
        ]
        for scope_id in stale:
            mapped = self._local_mapped_scopes.pop(scope_id)
            try:
                await mapped.source.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down stale scope mapping {scope_id}: {e}")
            logger.info(f"Reconciliation: cleaned up stale scope mapping {scope_id}")

        # XAUTOCLAIM orphaned events from crashed replicas
        if self.redis_client:
            for scope_id, mapped in self._local_mapped_scopes.items():
                await mapped.source.claim_orphaned_events()

    async def _pin_scope_pages(self, scope_id: str) -> None:
        """Lock all pages from a scope to prevent eviction.

        Used when MmapConfig.pinned=True. Locks each page using the
        existing page_table.lock_page() mechanism with a very long
        duration.

        Args:
            scope_id: Scope whose pages should be pinned
        """
        pages = await self.page_storage.query_pages_by_metadata(
            filters={"scope_id": scope_id},
        )
        for page in pages:
            try:
                await self.page_table.lock_page(
                    page_id=page.page_id,
                    locked_by=f"mmap:pinned:{scope_id}",
                    lock_duration_s=86400 * 365,  # ~1 year - TODO: Make configurable?
                    reason=f"Pinned scope: {scope_id}",
                )
            except Exception as e:
                logger.warning(f"Failed to pin page {page.page_id}: {e}")

        if pages:
            logger.info(f"Pinned {len(pages)} pages for scope {scope_id}")

    def _resolve_ingestion_policy(self, config: MmapConfig) -> GroupAndFlushIngestionPolicy:
        """Create an IngestionPolicy from MmapConfig.

        Args:
            config: Mapping configuration

        Returns:
            Configured IngestionPolicy instance
        """
        from .blackboard_page_source import (
            FlushPolicy,
            ImmediateFlushPolicy,
            LocalityPolicy,
            PeriodicFlushPolicy,
            TagLocalityPolicy,
            TemporalLocalityPolicy,
            ThresholdFlushPolicy,
        )

        # Resolve locality policy
        locality: LocalityPolicy
        if config.locality_policy_type == "temporal":
            locality = TemporalLocalityPolicy()
        else:
            locality = TagLocalityPolicy()

        # Resolve flush policy
        flush: FlushPolicy
        if config.flush_policy_type == "periodic":
            flush = PeriodicFlushPolicy(
                interval_seconds=config.flush_interval_seconds,
            )
        elif config.flush_policy_type == "immediate":
            flush = ImmediateFlushPolicy()
        else:
            flush = ThresholdFlushPolicy(
                record_threshold=config.flush_threshold,
                token_budget=config.flush_token_budget,
            )

        return GroupAndFlushIngestionPolicy(
            locality_policy=locality,
            flush_policy=flush,
        )

    async def _periodic_reconciliation_loop(self):
        """Periodically reconcile Layer 2 (VCM) with Layer 1 (Deployments).

        This handles edge cases where events are lost or processing fails.
        Runs every 30 seconds (configurable).
        """
        while True:
            try:
                await asyncio.sleep(self._reconciliation_interval_s)
                await self._reconcile_page_state()
            except asyncio.CancelledError:
                logger.info("Reconciliation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}", exc_info=True)

    async def _reconcile_page_state(self):
        """Reconcile VCM page table (Layer 2) with deployment states (Layer 1).

        This scans all deployments and ensures Layer 2 matches Layer 1.
        """
        logger.debug("Starting page state reconciliation")

        try:
            # Get all deployment states (Layer 1)
            deployment_states = await self._get_all_deployment_states()

            reconciled_count = 0

            for deployment_name, deployment_state in deployment_states.items():
                for client_id, client_state in deployment_state.client_states.items():
                    # Pages in Layer 1 (deployment state)
                    layer1_pages = client_state.loaded_page_ids

                    # Pages in Layer 2 (VCM page table) for this client
                    layer2_pages = await self.page_table.get_pages_on_client(
                        deployment_name=deployment_name,
                        client_id=client_id,
                    )

                    # Find discrepancies
                    missing_in_layer2 = layer1_pages - layer2_pages
                    missing_in_layer1 = layer2_pages - layer1_pages

                    # Reconcile: Layer 1 is source of truth for physical state
                    for page_id in missing_in_layer2:
                        logger.warning(
                            f"Reconciling: Adding page {page_id} to Layer 2 "
                            f"(present in Layer 1: {deployment_name}/{client_id})"
                        )
                        await self.page_table.register_loaded_page(
                            page_id=page_id,
                            replica_id=client_id,
                            deployment_name=deployment_name,
                            tenant_id="unknown",  # TODO: Best effort? Load page metadata?
                            size=0,  # TODO: Unknown, will be corrected on next load?
                        )
                        reconciled_count += 1

                    for page_id in missing_in_layer1:
                        logger.warning(
                            f"Reconciling: Removing page {page_id} from Layer 2 "
                            f"(absent in Layer 1: {deployment_name}/{client_id})"
                        )
                        async for state in self.page_table.state_manager.write_transaction():
                            state.register_page_eviction(
                                virtual_page_id=page_id,
                                deployment_name=deployment_name,
                                client_id=client_id,
                            )
                        reconciled_count += 1

            if reconciled_count > 0:
                logger.info(f"Reconciliation complete: {reconciled_count} pages reconciled")
            else:
                logger.debug("Reconciliation complete: no discrepancies found")

            # Reconcile blackboard ↔ VCM scope mappings
            try:
                await self._reconcile_scope_mappings()
            except Exception as e:
                logger.warning(f"Error reconciling scope mappings: {e}")

        except Exception as e:
            logger.error(f"Error in page state reconciliation: {e}", exc_info=True)

    async def _get_all_deployment_states(self) -> dict[str, Any]:
        """Get all VLLMDeploymentState instances.

        Returns:
            Dictionary mapping deployment_name to VLLMDeploymentState
        """
        deployment_states = {}
        deployment_names = await self.llm_cluster_handle.get_all_deployment_names()

        for deployment_name in deployment_names:
            try:
                state_manager = await self._get_deployment_state_manager(deployment_name)
                async for state in state_manager.read_transaction():
                    deployment_states[deployment_name] = state
            except Exception as e:
                logger.warning(f"Failed to get state for deployment {deployment_name}: {e}")

        return deployment_states

    async def cleanup(self):
        """Cleanup VCM resources including event subscriptions."""
        logger.info("Cleaning up VirtualContextManager")

        # Cancel reconciliation task
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass

        # Cancel all event subscriptions
        for deployment_name, subscriber in self.event_subscribers.items():
            logger.info(f"Unsubscribing from events: {deployment_name}")
            try:
                await subscriber.cancel()
            except Exception as e:
                logger.warning(f"Error cancelling subscriber for {deployment_name}: {e}")

        # Flush and shut down all scope mappings
        for scope_id, mapped in list(self._local_mapped_scopes.items()):
            try:
                await mapped.source.shutdown()
                logger.info(f"Shut down scope mapping: {scope_id}")
            except Exception as e:
                logger.warning(f"Error shutting down scope mapping {scope_id}: {e}")
        self._local_mapped_scopes.clear()

        logger.info("VirtualContextManager cleanup complete")

