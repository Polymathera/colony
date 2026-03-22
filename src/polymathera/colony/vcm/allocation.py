"""Page allocation and eviction strategies for Virtual Context Manager.

This module provides various strategies for deciding where to load pages
and which pages to evict when KV cache capacity is exhausted.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from overrides import override
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cluster.models import LLMClientState
    from .models import (
        PageAllocationRequest,
        PageLocation,
        ContextPageId,
        VirtualPageTableState,
    )
    from .page_table import VirtualPageTable

from ..distributed.ray_utils import serving

logger = logging.getLogger(__name__)


def _check_page_allocation_request(request: PageAllocationRequest):
    tenant_id = serving.require_tenant_id()
    colony_id = serving.require_colony_id()

    if request.syscontext.colony_id != colony_id or request.syscontext.tenant_id != tenant_id:
        raise ValueError(
            f"Request colony_id {request.syscontext.colony_id} / tenant_id {request.syscontext.tenant_id} does not match current colony_id {colony_id} / tenant_id {tenant_id}"
        )


class AllocationDecision:
    """Decision about where to allocate a page.

    Attributes:
        page_id: The page to allocate
        target_deployment: Deployment to load page into
        target_client_id: Client (replica) to load page into
        evict_pages: Pages to evict to make room (if needed), each represented as a tuple (page_id, colony_id, tenant_id)
        reason: Human-readable reason for this decision
    """

    def __init__(
        self,
        page_id: ContextPageId,
        target_deployment: str,
        target_client_id: str,
        evict_pages: list[tuple[str, str, str]] | None = None,
        reason: str = "",
    ):
        self.page_id = page_id
        self.target_deployment = target_deployment
        self.target_client_id = target_client_id
        self.evict_pages = evict_pages or []
        self.reason = reason


class AllocationStrategy(ABC):
    """Abstract base class for page allocation strategies.

    Subclasses implement different policies for deciding where to place pages
    in the cluster's physical memory (KV caches).
    """

    @abstractmethod
    async def make_allocation_decisions(
        self,
        request: PageAllocationRequest,
        page_table: VirtualPageTable,
        client_states: dict[str, LLMClientState],
        page_sizes: dict[ContextPageId, int],
    ) -> list[AllocationDecision]:
        """Make allocation decisions for a request.

        Args:
            request: Allocation request with pages to allocate
            page_table: Current page table
            client_states: Current state of all clients
            page_sizes: Sizes of pages to allocate (in tokens)

        Returns:
            List of allocation decisions (one per page)
        """
        pass


class BalancedAllocationStrategy(AllocationStrategy):
    """Balanced allocation strategy.

    This strategy:
    1. Prefers clients with more available capacity
    2. Balances load across clients
    3. Uses LRU eviction when capacity is exhausted
    4. Respects tenant isolation
    5. Honors preferred deployment hints

    This is a good general-purpose strategy for most workloads.
    """

    @override
    async def make_allocation_decisions(
        self,
        request: PageAllocationRequest,
        page_table: VirtualPageTable,
        client_states: dict[str, LLMClientState],
        page_sizes: dict[ContextPageId, int],
    ) -> list[AllocationDecision]:
        """Make balanced allocation decisions.

        Args:
            request: Allocation request
            page_table: Current page table
            client_states: Current client states
            page_sizes: Page sizes

        Returns:
            List of allocation decisions
        """
        _check_page_allocation_request(request)

        decisions = []

        # Filter clients by tenant isolation if needed
        available_clients = self._filter_clients_by_tenant(
            client_states, request.syscontext.tenant_id
        )

        # Filter by preferred deployment if specified
        if request.preferred_deployment:
            available_clients = {
                cid: state for cid, state in available_clients.items()
                if state.deployment_name == request.preferred_deployment
            }

        if not available_clients:
            logger.warning(
                f"No available clients for tenant {request.syscontext.tenant_id}. "
                f"Total clients: {len(client_states)}"
            )
            return []

        # Allocate each page
        for page_id in request.virtual_page_ids:
            page_size = page_sizes.get(page_id, 0)

            # Check if page is already loaded
            if await page_table.is_page_loaded(page_id):
                # Page already loaded, check if we need more replicas
                current_replication = await page_table.get_replication_factor(page_id)
                if current_replication >= request.max_replication:
                    logger.debug(
                        f"Page {page_id} already loaded with sufficient replication "
                        f"({current_replication}/{request.max_replication})"
                    )
                    continue

            # Find best client for this page
            best_client_id = await self._select_best_client(
                page_id=page_id,
                page_size=page_size,
                available_clients=available_clients,
                page_table=page_table,
                affinity_pages=request.affinity_pages,
            )

            if best_client_id is None:
                logger.warning(f"Could not find suitable client for page {page_id}")
                continue

            client_state = available_clients[best_client_id]

            # Check if we need to evict pages
            evict_pages: list[tuple[str, str, str]] = []
            available_capacity = client_state.get_available_cache_capacity()

            if available_capacity < page_size:
                # Need to evict pages
                evict_pages = await self._select_pages_to_evict(
                    client_id=best_client_id,
                    required_capacity=page_size - available_capacity,
                    page_table=page_table,
                    client_state=client_state,
                )

            # Create allocation decision
            decision = AllocationDecision(
                page_id=page_id,
                target_deployment=client_state.deployment_name,
                target_client_id=best_client_id,
                evict_pages=evict_pages,
                reason=f"Balanced allocation to {best_client_id} with "
                       f"{len(evict_pages)} evictions",
            )
            decisions.append(decision)

        return decisions

    def _filter_clients_by_tenant(
        self,
        client_states: dict[str, LLMClientState],
        tenant_id: str,
    ) -> dict[str, LLMClientState]:
        """Filter clients based on tenant isolation.

        For now, all clients are available to all tenants (tenant isolation
        is enforced at the page level via KV cache hashing in vLLM).

        Args:
            client_states: All client states
            tenant_id: Tenant ID

        Returns:
            Filtered client states
        """
        # TODO: Implement tenant-specific client pools if needed
        return {
            cid: state for cid, state in client_states.items()
            if state.is_healthy
        }

    async def _select_best_client(
        self,
        page_id: ContextPageId,
        page_size: int,
        available_clients: dict[str, LLMClientState],
        page_table: VirtualPageTable,
        affinity_pages: list[ContextPageId],
    ) -> str | None:
        """Select the best client to load a page.

        Scoring factors:
        - Available capacity (higher is better)
        - Current load (lower is better)
        - Affinity with other pages (higher is better)

        Args:
            page_id: Page to allocate
            page_size: Size of page in tokens
            available_clients: Available client states
            page_table: Current page table
            affinity_pages: Pages that should be co-located

        Returns:
            Client ID of best client, or None if no suitable client
        """
        CAPACITY_WEIGHT = 1.0
        LOAD_WEIGHT = 0.5
        AFFINITY_WEIGHT = 2.0

        best_client_id = None
        best_score = float('-inf')

        for client_id, client_state in available_clients.items():
            # Check if client has enough capacity (including potential evictions)
            if client_state.kv_cache_capacity < page_size:
                # Client can never fit this page
                continue

            # Calculate capacity score (0 to 1)
            available_capacity = client_state.get_available_cache_capacity()
            capacity_score = available_capacity / client_state.kv_cache_capacity

            # Calculate load score (0 to 1, inverted)
            # Assume max load of 100 for normalization
            load_score = 1.0 - min(1.0, client_state.pending_requests / 100.0)

            # Calculate affinity score (0 to 1)
            affinity_score = 0.0
            if affinity_pages:
                # Check how many affinity pages are on this client
                client_pages = await page_table.get_pages_on_client(
                    client_state.deployment_name,
                    client_id,
                )
                affinity_count = sum(
                    1 for aff_page in affinity_pages
                    if aff_page in client_pages
                )
                affinity_score = affinity_count / len(affinity_pages)

            # Compute total score
            score = (
                CAPACITY_WEIGHT * capacity_score +
                LOAD_WEIGHT * load_score +
                AFFINITY_WEIGHT * affinity_score
            )

            if score > best_score:
                best_score = score
                best_client_id = client_id

        return best_client_id

    async def _select_pages_to_evict(
        self,
        client_id: str,
        required_capacity: int,
        page_table: VirtualPageTable,
        client_state: LLMClientState,
    ) -> list[tuple[str, str, str]]:
        """Select pages to evict using LRU policy.

        Args:
            client_id: Client to evict from
            required_capacity: Amount of capacity needed (tokens)
            page_table: Current page table
            client_state: Client state

        Returns:
            List of page IDs to evict
        """
        # Get all pages on this client
        client_pages = await page_table.get_pages_on_client(
            client_state.deployment_name,
            client_id,
        )

        # Get locations for these pages on this client
        page_locations: list[tuple[str, PageLocation]] = []
        for page_ref in client_pages:
            ref = VirtualPageTableState.parse_page_ref(page_ref)
            page_id = ref["page_id"]
            colony_id = ref["colony_id"]
            tenant_id = ref["tenant_id"]
            with serving.execution_context(
                colony_id=colony_id,
                tenant_id=tenant_id,
                origin="allocation_strategy",
            ):
                locations = await page_table.get_page_locations(
                    page_id,
                    deployment_name=client_state.deployment_name,
                )
                # Find the location on this specific client
                for loc in locations:
                    if loc.client_id == client_id:
                        page_locations.append((page_id, loc))
                        break

        # Sort by last access time (LRU)
        page_locations.sort(key=lambda x: x[1].last_access_time)

        # Get locked pages to exclude them from eviction
        locked_page_keys = await page_table.get_locked_pages()
        locked_pages = [key[0] for key in locked_page_keys]

        # Select pages to evict (skip locked pages)
        evict_pages: list[tuple[str, str, str]] = []
        freed_capacity = 0

        for page_id, location in page_locations:
            if freed_capacity >= required_capacity:
                break

            # Skip locked pages
            if page_id in locked_pages:
                logger.debug(f"Skipping locked page {page_id} during eviction")
                continue

            evict_pages.append((page_id, location.syscontext.colony_id, location.syscontext.tenant_id))
            freed_capacity += location.size

        # Warn if we couldn't free enough capacity due to locked pages
        if freed_capacity < required_capacity and evict_pages:
            logger.warning(
                f"Could not free enough capacity ({freed_capacity}/{required_capacity} tokens) "
                f"on client {client_id} - some pages are locked"
            )

        return evict_pages


class LocalityAwareAllocationStrategy(AllocationStrategy):
    """Locality-aware allocation strategy.

    This strategy optimizes for data locality by:
    1. Co-locating pages that are frequently accessed together
    2. Placing pages near other pages from the same source (e.g., same git repo)
    3. Minimizing page faults by predicting access patterns

    This is useful for workloads with strong locality patterns (e.g., code analysis).
    """

    @override
    async def make_allocation_decisions(
        self,
        request: PageAllocationRequest,
        page_table: VirtualPageTable,
        client_states: dict[str, LLMClientState],
        page_sizes: dict[ContextPageId, int],
    ) -> list[AllocationDecision]:
        """Make locality-aware allocation decisions.

        For now, this delegates to BalancedAllocationStrategy with enhanced
        affinity handling. Future enhancements can add predictive placement.

        Args:
            request: Allocation request
            page_table: Current page table
            client_states: Current client states
            page_sizes: Page sizes

        Returns:
            List of allocation decisions
        """
        _check_page_allocation_request(request)

        # Use balanced strategy as base
        # In the future, we can add:
        # - Page group analysis (from metadata)
        # - Access pattern prediction
        # - Proactive replication of hot pages
        balanced_strategy = BalancedAllocationStrategy()
        return await balanced_strategy.make_allocation_decisions(
            request, page_table, client_states, page_sizes
        )


# Default strategy
DEFAULT_ALLOCATION_STRATEGY = BalancedAllocationStrategy()
