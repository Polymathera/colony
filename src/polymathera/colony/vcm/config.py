"""Configuration for Virtual Context Manager (VCM)."""

from dataclasses import dataclass
from typing import Literal

from ..distributed.ray_utils import serving
from .allocation import AllocationStrategy


@dataclass
class VCMConfig:
    """Configuration for VirtualContextManager deployment.

    Attributes:
        caching_policy: Page eviction policy ("LRU" or "LFU")
        page_fault_processing_interval_s: How often to process page fault queue (seconds)
        metrics_collection_interval_s: How often to collect metrics (seconds)
    """

    caching_policy: str = "LRU"
    page_fault_processing_interval_s: float = 5.0
    metrics_collection_interval_s: float = 30.0
    allocation_strategy: AllocationStrategy | None = None
    page_storage_backend_type: Literal["efs", "s3"] = "efs"  # Default to EFS for fast access
    page_storage_path: str = "colony/context_pages"
    reconciliation_interval_s: float = 30.0

    def __post_init__(self):
        """Validate configuration."""
        if self.caching_policy not in ["LRU", "LFU"]:
            raise ValueError(f"Invalid caching_policy: {self.caching_policy}. Must be 'LRU' or 'LFU'")

    def add_deployments_to_app(self, app: serving.Application, top_level: bool) -> None:
        if not top_level:
            from . import VirtualContextManager

            app.add_deployment(
                VirtualContextManager.bind(
                    caching_policy=self.caching_policy,
                    allocation_strategy=self.allocation_strategy,
                    page_storage_backend_type=self.page_storage_backend_type,
                    page_storage_path=self.page_storage_path,
                    reconciliation_interval_s=self.reconciliation_interval_s,
                ),
                name="vcm",
            )
