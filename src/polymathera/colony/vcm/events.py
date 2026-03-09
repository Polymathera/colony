"""Events for page lifecycle state changes.

This module defines events emitted by VLLMDeployment when pages are loaded,
evicted, or fail to load. These events are published via Redis pub/sub and
consumed by VirtualContextManager to maintain Layer 2 (VCM page table) state.

Event Flow:
    1. VLLMDeployment performs physical operation (load/evict page)
    2. VLLMDeployment updates Layer 1 (VLLMDeploymentState)
    3. VLLMDeployment emits event via Redis pub/sub
    4. VCM receives event and updates Layer 2 (VirtualPageTableState)
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PageEvent(BaseModel):
    """Base class for all page lifecycle events.

    All events include basic identifying information about the page,
    deployment, replica, and tenant.
    """

    event_type: str = Field(..., description="Type of event")
    page_id: str = Field(..., description="Virtual page identifier")
    deployment_name: str = Field(..., description="Deployment name")
    client_id: str = Field(..., description="LLM client/replica ID")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
    timestamp: float = Field(..., description="Event timestamp (Unix time)")


class PageLoadedEvent(PageEvent):
    """Event emitted when a page is successfully loaded into KV cache.

    Emitted by VLLMDeployment.load_page() after:
    - Page is loaded into vLLM KV cache (physical operation)
    - Layer 1 (VLLMDeploymentState) is updated
    - Layer 3 (local loaded_pages dict) is updated

    Triggers VCM to:
    - Update Layer 2 (VirtualPageTableState)
    - Resolve any pending page faults for this page
    """

    event_type: Literal["page_loaded"] = "page_loaded"
    size: int = Field(..., description="Page size in tokens")
    kv_cache_slot: int = Field(..., description="KV cache slot allocated")
    load_duration_ms: float = Field(..., description="Time taken to load page (ms)")


class PageEvictedEvent(PageEvent):
    """Event emitted when a page is evicted from KV cache.

    Emitted by VLLMDeployment.evict_page() after:
    - Page is removed from vLLM KV cache (physical operation)
    - Layer 1 (VLLMDeploymentState) is updated
    - Layer 3 (local loaded_pages dict) is updated

    Triggers VCM to:
    - Update Layer 2 (VirtualPageTableState)
    - Remove page location from page table
    """

    event_type: Literal["page_evicted"] = "page_evicted"
    size: int = Field(..., description="Page size in tokens that was freed")
    reason: str = Field(..., description="Eviction reason: 'manual', 'capacity', 'expired'")


class PageLoadFailedEvent(PageEvent):
    """Event emitted when a page load fails.

    Emitted by VLLMDeployment.load_page() when:
    - Physical load operation fails
    - No state updates are made (neither Layer 1 nor Layer 3)

    Triggers VCM to:
    - Mark page fault as failed
    - Log error for debugging
    - Potentially retry with different allocation
    """

    event_type: Literal["page_load_failed"] = "page_load_failed"
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type/class")
    retry_count: int = Field(0, description="Number of retries attempted")
