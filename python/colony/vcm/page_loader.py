"""Page Loader for managing page loading and eviction.

The PageLoader implements caching policies (LRU, LFU) and processes page faults
from the priority queue. It integrates with the LLMCluster layer to load/evict pages
from VLLM replicas.

Key Responsibilities:
- Load virtual pages into LLMCluster replicas
- Evict pages based on caching policy when capacity is reached
- Process page faults asynchronously from priority queue
- Select best replica for loading a page

Caching Policies:
- LRU (Least Recently Used): Evict page with oldest last_accessed
- LFU (Least Frequently Used): Evict page with lowest access_count
"""

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from ..distributed.ray_utils import serving
from .models import PageAllocationRequest, PageAllocationResponse, PagePriority
from .models import VirtualContextPage, PageLocation, ContextPageId
from .page_table import VirtualPageTable
from ..deployment_names import get_deployment_names

logger = logging.getLogger(__name__)


# TODO: This is currently not used.
class PageLoader:
    """Handles loading and evicting pages from VLLM replicas.

    Implements caching policies and manages page fault queue.

    Example:
        ```python
        loader = PageLoader(
            page_table=page_table,
            policy="LRU"
        )

        # Load a page
        replica_id = await loader.load_page(virtual_page)

        # Process page faults
        processed = await loader.process_page_faults(max_faults=10)
        ```
    """

    def __init__(
        self,
        page_table: VirtualPageTable,
        policy: Literal["LRU", "LFU"] = "LRU",
    ):
        """Initialize page loader.

        Args:
            page_table: Virtual context page table for tracking locations
            policy: Caching policy ("LRU" or "LFU")
        """
        self.page_table = page_table
        self.policy = policy.upper()

        # Get VCM handle
        self.vcm_handle: serving.DeploymentHandle | None = None  # Deprecated, for backward compatibility
        try:
            app_name = serving.get_my_app_name()
            names = get_deployment_names()
            self.vcm_handle = serving.get_deployment(
                app_name=app_name,
                deployment_name=names.vcm,
            )
            logger.info("Connected to VCM deployment.")
        except Exception as e:
            logger.debug(f"VCM deployment not found (expected after merge): {e}")

        if self.policy not in ["LRU", "LFU"]:
            raise ValueError(f"Invalid policy: {policy}. Must be 'LRU' or 'LFU'")

        # TODO: Add storage backend for persisting virtual pages
        # For now, assume pages are provided directly
        self._page_cache: dict[str, VirtualContextPage] = {}

    def cache_virtual_page(self, page: VirtualContextPage) -> None:
        """Cache a virtual page in memory (temporary solution).

        TODO: Replace with proper storage backend (S3, database, etc.)

        Args:
            page: Virtual context page to cache
        """
        self._page_cache[page.page_id] = page

    async def get_virtual_page(self, page_id: ContextPageId) -> VirtualContextPage | None:
        """Get a virtual page by ID.

        Args:
            page_id: Page identifier

        Returns:
            VirtualContextPage if exists, None otherwise

        TODO: Load from persistent storage instead of memory cache
        """
        return self._page_cache.get(page_id)

    async def load_page(
        self,
        page: VirtualContextPage,
        replica_id: str | None = None,
        tenant_id: str = "default",
        priority: int = 0,
    ) -> str:
        """Load a virtual context page into a replica's KV cache via VCM.

        Calls VCM's allocate_pages() endpoint which coordinates the actual
        loading with VLLM deployments.

        Args:
            page: Virtual context page to load
            replica_id: If specified, load on this replica. Otherwise VCM decides. Currently unused.
            tenant_id: Tenant ID for multi-tenancy
            priority: Load priority (0-100, higher = more urgent)

        Returns:
            replica_id where page was loaded

        Raises:
            ValueError: If no suitable replica found or allocation fails
        """
        # Cache the page for future reference
        self.cache_virtual_page(page)

        # Map integer priority to PagePriority enum
        if priority >= 75:
            page_priority = PagePriority.CRITICAL
        elif priority >= 50:
            page_priority = PagePriority.HIGH
        elif priority >= 25:
            page_priority = PagePriority.NORMAL
        else:
            page_priority = PagePriority.LOW

        # Call VCM's allocate_pages endpoint
        try:
            request = PageAllocationRequest(
                virtual_page_ids=[page.page_id],
                tenant_id=tenant_id,
                priority=page_priority,
            )

            response: PageAllocationResponse = await self.vcm_handle.allocate_pages(request)

            # Check if allocation succeeded
            if page.page_id in response.failed_pages:
                raise ValueError(
                    f"VCM failed to allocate page {page.page_id}: "
                    f"evicted={response.evicted_pages}, time={response.allocation_time_ms:.2f}ms"
                )

            # Extract the location where page was loaded
            locations = response.allocated_locations.get(page.page_id, [])
            if not locations:
                raise ValueError(f"VCM returned no locations for page {page.page_id}")

            # Use the first location (VCM may replicate to multiple locations)
            location = locations[0]
            allocated_replica_id = location.client_id

            # Register in page table
            await self.page_table.register_loaded_page(page.page_id, allocated_replica_id)

            logger.info(
                f"Loaded page {page.page_id} ({page.size} tokens) on "
                f"{location.deployment_name}/{allocated_replica_id} via VCM "
                f"(time={response.allocation_time_ms:.2f}ms, "
                f"evicted={len(response.evicted_pages)} pages)"
            )

            return allocated_replica_id

        except Exception as e:
            logger.error(f"Failed to load page {page.page_id} via VCM: {e}")
            raise

    async def evict_page(self, page_id: str) -> None:
        """Evict a page from whichever replica has it.

        Args:
            page_id: Page identifier

        Raises:
            ValueError: If page is not loaded anywhere
        """
        # Get page location
        location = await self.page_table.get_page_location(page_id)
        if not location:
            raise ValueError(f"Page {page_id} is not loaded anywhere")

        # TODO: Call LLMCluster's unload_page endpoint when implemented
        # For now, just unregister from page table
        # await self.vllm_handle.unload_page(page_id)

        await self.page_table.unregister_page(page_id)

        logger.info(f"Evicted page {page_id} from replica {location.replica_id}")

    async def select_eviction_candidate(self, replica_id: str) -> str | None:
        """Select a page to evict from a replica based on policy.

        Args:
            replica_id: Replica identifier

        Returns:
            Page ID to evict, or None if no pages on replica
        """
        # Get all pages on this replica
        page_ids = await self.page_table.get_replica_pages(replica_id)
        if not page_ids:
            return None

        # Get locations for all pages
        locations: list[PageLocation] = []
        for page_id in page_ids:
            location = await self.page_table.get_page_location(page_id)
            if location:
                locations.append(location)

        if not locations:
            return None

        # Select based on policy
        if self.policy == "LRU":
            # Evict page with oldest last_accessed
            victim = min(locations, key=lambda loc: loc.last_accessed)
        else:  # LFU
            # Evict page with lowest access_count
            victim = min(locations, key=lambda loc: loc.access_count)

        logger.debug(
            f"Selected page {victim.page_id} for eviction from {replica_id} "
            f"(policy={self.policy}, last_access={victim.last_accessed}, "
            f"access_count={victim.access_count})"
        )

        return victim.page_id

    async def process_page_faults(self, max_faults: int = 10) -> int:
        """Process up to max_faults from the priority queue.

        This is typically called periodically by a background task.

        Args:
            max_faults: Maximum number of faults to process

        Returns:
            Number of faults processed
        """
        processed = 0

        for _ in range(max_faults):
            # Get next fault
            fault = await self.page_table.get_next_page_fault()
            if not fault:
                break  # No more faults

            # Check if page is already loaded
            if await self.page_table.is_page_loaded(fault.page_id):
                logger.debug(f"Page {fault.page_id} already loaded, skipping fault")
                processed += 1
                continue

            # Get the virtual page
            # TODO: Load from storage backend
            if fault.page_id not in self._page_cache:
                logger.warning(
                    f"Page {fault.page_id} not in cache, cannot process fault"
                )
                continue

            virtual_page = self._page_cache[fault.page_id]

            try:
                # Load the page
                # TODO: Pass tenant_id from fault or agent
                await self.load_page(virtual_page, tenant_id="default")
                processed += 1

                logger.info(
                    f"Processed page fault for {fault.page_id} "
                    f"(priority={fault.priority}, agent={fault.requesting_agent_id})"
                )

            except Exception as e:
                logger.error(f"Failed to process fault for {fault.page_id}: {e}")
                # Put fault back in queue with lower priority?
                # For now, just skip it

        return processed

    async def _select_best_replica(self, page_id: str) -> str:
        """Select the best replica to load a page.

        Selection criteria (in order):
        1. Replica with most capacity available
        2. Replica with least pages loaded
        3. Round-robin (use page_id hash)

        Args:
            page_id: Page identifier (used for hashing if needed)

        Returns:
            Selected replica_id

        Raises:
            ValueError: If no replicas available

        TODO: Implement actual replica selection logic.
        For now, returns a placeholder.
        """
        # TODO: Get replica stats from page table or VLLM deployment
        # For now, return placeholder - this will be replaced when we integrate
        # with actual VLLM deployment and can query replica states

        # stats = await self.page_table.get_stats()
        # if not stats["pages_per_replica"]:
        #     raise ValueError("No replicas available")

        # For now, use a fixed replica ID
        # In production, this would query VLLM deployment for available replicas
        return "replica-0"

    async def get_stats(self) -> dict[str, Any]:
        """Get page loader statistics.

        Returns:
            Dictionary with loader stats
        """
        page_table_stats = await self.page_table.get_stats()

        return {
            "policy": self.policy,
            "cached_pages": len(self._page_cache),
            **page_table_stats,
        }
