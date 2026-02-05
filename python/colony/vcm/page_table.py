"""Virtual Page Table for tracking context page locations across LLM replicas.

The VirtualPageTable maintains a distributed mapping between virtual context pages and
their physical locations (which VLLM replica has them loaded). It uses
StateManager for distributed coordination across multiple VCM replicas.

Key Operations:
- is_page_loaded: Check if a page is loaded anywhere
- get_page_location: Get which replica has a page
- register_loaded_page: Record that a page was loaded
- unregister_page: Record that a page was evicted
- add_page_fault: Add request to load a page
- get_next_page_fault: Get highest priority page fault

All operations are atomic via StateManager transactions.
"""

import logging
import time

from ...distributed import get_initialized_polymathera
from ..distributed.state_management import StateManager
from .models import PageFault, PageGroup, PageLocation, PageLock, VCMBranch, VirtualPageTableState
from ..distributed.ray_utils import serving

logger = logging.getLogger(__name__)


class VirtualPageTable:
    """Manages mapping between virtual context pages and physical locations.

    Thread-safe operations using StateManager for distributed coordination.
    All VirtualContextManager replicas share the same page table via
    distributed state.

    Example:
        ```python
        page_table = VirtualPageTable()
        await page_table.initialize()

        # Check if page is loaded
        if await page_table.is_page_loaded("page-123"):
            location = await page_table.get_page_location("page-123")
            print(f"Page loaded on replica {location.replica_id}")

        # Register a loaded page
        await page_table.register_loaded_page("page-456", "replica-1")

        # Add page fault
        fault = PageFault(page_id="page-789", priority=10)
        await page_table.add_page_fault(fault)
        ```
    """

    def __init__(self):
        """Initialize page table."""
        self.app_name: str | None = None
        self.state_manager: StateManager[VirtualPageTableState] | None = None

    async def initialize(self) -> None:
        """Initialize page table state manager."""
        self.app_name = serving.get_my_app_name()
        # Get Polymathera for state management
        polymathera = await get_initialized_polymathera()

        # Initialize StateManager for page table
        self.state_manager = await polymathera.get_state_manager(
            state_type=VirtualPageTableState,
            state_key=VirtualPageTableState.get_state_key(self.app_name),
        )

    async def is_page_loaded(self, page_id: str) -> bool:
        """Check if page is loaded anywhere in the cluster.

        Args:
            page_id: Page identifier

        Returns:
            True if page is loaded on at least one replica
        """
        async for state in self.state_manager.read_transaction():
            return state.is_page_loaded(page_id)

    async def get_replication_factor(self, virtual_page_id: str) -> int:
        """Get the replication factor (number of physical copies) of a page.

        Args:
            virtual_page_id: ID of the virtual page

        Returns:
            Number of physical locations (0 if page not loaded)
        """
        async for state in self.state_manager.read_transaction():
            return state.get_replication_factor(virtual_page_id)

    async def is_page_loaded_on_client(
        self,
        page_id: str,
        deployment_name: str,
        client_id: str,
    ) -> bool:
        """Check if page is loaded on a specific client (replica).

        Args:
            page_id: Page identifier
            deployment_name: Deployment name
            client_id: Client (replica) ID

        Returns:
            True if page is loaded on this specific client
        """
        async for state in self.state_manager.read_transaction():
            entry = state.get_page_entry(page_id)
            if not entry:
                return False
            return entry.get_location(deployment_name, client_id) is not None

    async def get_page_location(self, page_id: str) -> PageLocation | None:
        """Get first location where page is loaded (for backward compatibility).

        Args:
            page_id: Page identifier

        Returns:
            PageLocation if loaded, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            locations = state.get_page_locations(page_id)
            return locations[0] if locations else None

    async def get_page_locations(
        self,
        page_id: str,
        deployment_name: str | None = None,
        tenant_id: str | None = None,
    ) -> list[PageLocation]:
        """Get all locations where a page is loaded, with optional filters.

        Args:
            page_id: Page identifier
            deployment_name: Optional filter by deployment
            tenant_id: Optional filter by tenant

        Returns:
            List of PageLocation objects matching filters
        """
        async for state in self.state_manager.read_transaction():
            locations = state.get_page_locations(page_id)

            # Apply filters
            if deployment_name:
                locations = [loc for loc in locations if loc.deployment_name == deployment_name]
            if tenant_id:
                locations = [loc for loc in locations if loc.tenant_id == tenant_id]

            return locations

    async def find_replicas_with_page(
        self,
        page_id: str,
        deployment_name: str | None = None,
    ) -> list[tuple[str, str]]:
        """Find all replicas that have a specific page loaded.

        Args:
            page_id: Page identifier
            deployment_name: Optional filter by deployment

        Returns:
            List of (deployment_name, client_id) tuples
        """
        async for state in self.state_manager.read_transaction():
            clients = state.find_clients_with_page(page_id)

            # Apply deployment filter
            if deployment_name:
                clients = [(dep, client) for dep, client in clients if dep == deployment_name]

            return clients

    async def get_pages_on_client(
        self,
        deployment_name: str,
        client_id: str,
    ) -> set[str]:
        """Get all pages loaded on a specific client (deployment-aware).

        Args:
            deployment_name: Deployment name
            client_id: Client (replica) ID

        Returns:
            Set of page IDs loaded on this client
        """
        async for state in self.state_manager.read_transaction():
            return state.get_pages_on_client(deployment_name, client_id)

    async def get_replica_pages(self, replica_id: str) -> set[str]:
        """Get all pages loaded on a specific replica (backward compatibility).

        Args:
            replica_id: Replica identifier

        Returns:
            Set of page IDs loaded on this replica
        """
        async for state in self.state_manager.read_transaction():
            return set(state.replica_pages.get(replica_id, []))

    async def get_replica_page_count(self, replica_id: str) -> int:
        """Get number of pages loaded on a replica.

        Args:
            replica_id: Replica identifier

        Returns:
            Number of pages
        """
        async for state in self.state_manager.read_transaction():
            return state.get_replica_page_count(replica_id)

    async def register_loaded_page(
        self,
        page_id: str,
        replica_id: str,
        deployment_name: str,
        tenant_id: str = "default",
        size: int = 0,
    ) -> None:
        """Register that a page has been loaded on a LLM replica.

        Updates both forward (page_id -> location) and reverse
        (replica_id -> page_ids) indices atomically.

        Args:
            page_id: Page identifier
            replica_id: Replica identifier (client_id)
            deployment_name: Deployment name (required - no default)
            tenant_id: Tenant that owns this page
            size: Page size in tokens
        """
        import time

        if deployment_name is None:
            raise ValueError("deployment_name must be provided to register_loaded_page")

        async for state in self.state_manager.write_transaction():
            location = PageLocation(
                page_id=page_id,
                deployment_name=deployment_name,
                client_id=replica_id,
                load_time=time.time(),
                last_access_time=time.time(),
                access_count=0,
                size=size,
                tenant_id=tenant_id,
            )
            state.register_page_load(
                virtual_page_id=page_id,
                location=location,
                tenant_id=tenant_id,
                size=size,
            )
            logger.info(f"Registered page {page_id} on {deployment_name}/{replica_id}")

    async def unregister_page(self, page_id: str) -> None:
        """Unregister a page from all locations (when evicted).

        Removes from both forward and reverse indices atomically.

        Args:
            page_id: Page identifier
        """
        async for state in self.state_manager.write_transaction():
            # Use backward compatibility wrapper that removes from all locations
            state.remove_page_location(page_id)
            logger.info(f"Unregistered page {page_id} from all locations")

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
        async for state in self.state_manager.write_transaction():
            return state.lock_page(
                page_id=page_id,
                locked_by=locked_by,
                lock_duration_s=lock_duration_s,
                reason=reason,
                current_time=current_time,
            )

    async def unlock_page(self, page_id: str) -> bool:
        """Unlock a page, allowing it to be evicted.

        Args:
            page_id: ID of the page to unlock

        Returns:
            True if page was locked and is now unlocked, False if page wasn't locked
        """
        async for state in self.state_manager.write_transaction():
            return state.unlock_page(page_id)

    async def update_page_access(
        self,
        page_id: str,
        deployment_name: str | None = None,
        client_id: str | None = None,
    ) -> None:
        """Update last access time and count for eviction policies.

        Args:
            page_id: Page identifier
            deployment_name: Optional specific deployment to update
            client_id: Optional specific client to update
        """
        import time

        async for state in self.state_manager.write_transaction():
            if deployment_name and client_id:
                # Update specific location
                state.register_page_access(page_id, deployment_name, client_id, time.time())
            else:
                # Update all locations for this page
                entry = state.get_page_entry(page_id)
                if entry:
                    for loc in entry.physical_locations:
                        state.register_page_access(
                            page_id, loc.deployment_name, loc.client_id, time.time()
                        )
                logger.debug(f"Updated access for page {page_id}")

    async def add_page_fault(self, fault: PageFault) -> None:
        """Add a page fault request to the priority queue.

        If a fault already exists for this page, updates priority to the
        higher of the two.

        Args:
            fault: Page fault request
        """
        async for state in self.state_manager.write_transaction():
            state.add_page_fault(fault)
            logger.info(
                f"Added page fault for {fault.page_id} "
                f"(priority={fault.priority}, agent={fault.requesting_agent_id})"
            )

    async def get_next_page_fault(self) -> PageFault | None:
        """Get highest priority page fault to handle next.

        Removes the fault from the queue.

        Returns:
            Highest priority PageFault, or None if queue is empty
        """
        async for state in self.state_manager.write_transaction():
            fault = state.pop_next_fault()
            if fault:
                logger.info(f"Popped page fault for {fault.page_id} (priority={fault.priority})")
            return fault

    async def get_pending_fault_count(self) -> int:
        """Get number of pending page faults.

        Returns:
            Number of faults in the queue
        """
        async for state in self.state_manager.read_transaction():
            return len(state.pending_faults)

    # === Page Group Management ===

    async def register_page_group(self, group: PageGroup) -> None:
        """Register a virtual page group for spatial locality.

        Args:
            group: Virtual page group to register
        """
        async for state in self.state_manager.write_transaction():
            state.page_groups[group.group_id] = group
            # TODO: Ensure that pages in group are linked to the group
            logger.info(f"Registered virtual page group {group.group_id} with {len(group.page_ids)} pages")

    async def get_page_group(self, group_id: str) -> PageGroup | None:
        """Get a virtual page group by ID.

        Args:
            group_id: Group identifier

        Returns:
            PageGroup if exists, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            return state.page_groups.get(group_id)

    async def get_page_groups_for_page(self, page_id: str) -> list[PageGroup]:
        """Get all groups that contain a specific page.

        Args:
            page_id: Page identifier

        Returns:
            List of PageGroups containing this page
        """
        async for state in self.state_manager.read_transaction():
            groups = []
            for group in state.page_groups.values():
                if page_id in group.page_ids:
                    groups.append(group)
            return groups

    # === Statistics and Monitoring ===

    async def get_all_loaded_pages(self) -> list[str]:
        """Get all pages currently loaded across all replicas.

        Returns:
            List of page IDs that are currently loaded
        """
        async for state in self.state_manager.read_transaction():
            return state.get_all_loaded_pages()

    async def get_stats(self) -> dict:
        """Get page table statistics.

        Returns:
            Dictionary with stats (total_pages_loaded, replicas, pending_faults, etc.)
        """
        async for state in self.state_manager.read_transaction():
            # Get cluster stats from enhanced state
            cluster_stats = state.get_cluster_stats()

            return {
                "total_pages_loaded": len(state.entries),
                "total_physical_pages": cluster_stats.total_physical_pages,
                "average_replication_factor": cluster_stats.average_replication_factor,
                "num_replicas": len(state.replica_pages),
                "pending_faults": len(state.pending_faults),
                "num_groups": len(state.page_groups),
                "pages_per_replica": {
                    replica_id: len(page_ids)
                    for replica_id, page_ids in state.replica_pages.items()
                },
                "pages_per_tenant": cluster_stats.pages_per_tenant,
                "pages_per_deployment": cluster_stats.pages_per_deployment,
                "total_loads": state.total_loads,
                "total_evictions": state.total_evictions,
                "total_migrations": state.total_migrations,
                "num_branches": len(state.branches),
            }

    # === Branch Management ===

    async def get_branch(self, branch_id: str) -> VCMBranch | None:
        """Get a branch by ID.

        Args:
            branch_id: Branch identifier

        Returns:
            VCMBranch if found, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            return state.get_branch(branch_id)

    async def list_branches(self, tenant_id: str) -> list[VCMBranch]:
        """List all branches for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of VCMBranch objects for this tenant
        """
        async for state in self.state_manager.read_transaction():
            return state.get_branches_for_tenant(tenant_id)

    async def get_branch_pages(self, branch_id: str) -> set[str]:
        """Get all page IDs owned by a branch (overlays and new pages).

        Args:
            branch_id: Branch identifier

        Returns:
            Set of page IDs created on this branch
        """
        async for state in self.state_manager.read_transaction():
            return state.branch_pages.get(branch_id, set()).copy()

    async def get_effective_page_id(
        self,
        base_page_id: str,
        branch_id: str,
    ) -> str:
        """Resolve a page ID to its effective ID on a branch.

        Walks up the branch lineage to find overlays. Returns the overlay
        page ID if found, otherwise the original page ID.

        Args:
            base_page_id: Original page ID to look up
            branch_id: Branch to resolve from

        Returns:
            Effective page ID (overlay or original)
        """
        async for state in self.state_manager.read_transaction():
            return state.get_effective_page_id(base_page_id, branch_id)

    async def get_branch_effective_pages(self, branch_id: str) -> set[str]:
        """Get all pages effectively visible from a branch.

        Includes inherited pages, overlays, and new pages.

        Args:
            branch_id: Branch identifier

        Returns:
            Set of effective page IDs visible from this branch
        """
        async for state in self.state_manager.read_transaction():
            return state.get_branch_effective_pages(branch_id)

    async def register_branch(self, branch: VCMBranch) -> None:
        """Register a new branch.

        Args:
            branch: VCMBranch to register
        """
        async for state in self.state_manager.write_transaction():
            state.register_branch(branch)
            logger.info(f"Registered branch {branch.branch_id} for tenant {branch.tenant_id}")

    async def register_overlay(
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
        """
        async for state in self.state_manager.write_transaction():
            state.register_overlay(branch_id, original_page_id, overlay_page_id)
            logger.info(
                f"Registered overlay {overlay_page_id} for page {original_page_id} "
                f"on branch {branch_id}"
            )

    async def register_new_page_on_branch(
        self,
        branch_id: str,
        page_id: str,
    ) -> None:
        """Register a new page created on a branch (not an overlay).

        Args:
            branch_id: Branch the page was created on
            page_id: New page ID
        """
        async for state in self.state_manager.write_transaction():
            state.register_new_page_on_branch(branch_id, page_id)
            logger.info(f"Registered new page {page_id} on branch {branch_id}")
