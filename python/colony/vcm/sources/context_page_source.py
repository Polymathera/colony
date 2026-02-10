"""Context page source: Interface for page clustering and retrieval.

This module provides abstractions for grouping virtual context pages
and finding relevant pages based on queries (key-query-value attention).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


from ..models import ContextPageId
from ...vcm.models import MmapConfig

logger = logging.getLogger(__name__)


class PageCluster(BaseModel):
    """A cluster of related pages."""
    cluster_id: str
    page_ids: list[str]
    relationship_score: float = Field(ge=0.0, le=1.0, description="Average relationship strength")
    cluster_type: str  # "file_group", "semantic", "hybrid", etc.
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPageSource(ABC):
    """Abstract interface for page clustering and retrieval.
    This provides a mapping of application-level records to VCM pages.
    Application-level records include files, blackboard entries, knowledge graphs, etc.

    Implementations can use different strategies:
    - File-based: Uses FileGrouper from sharding infrastructure
    - Semantic: Uses embedding-based clustering
    - Hybrid: Combines multiple signals
    - LLM-learned: Uses LLM to determine clusters
    """

    def __init__(
        self,
        scope_id: str,
        tenant_id: str,
        mmap_config: MmapConfig,
    ):
        self.scope_id = scope_id
        self.tenant_id = tenant_id
        self.mmap_config = mmap_config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the page source (load or build page graph)."""
        pass

    @abstractmethod
    async def claim_orphaned_events(self) -> None:
        """Claim any orphaned events for this source (e.g., from previous instance)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and gracefully shut down the source."""
        pass

    @abstractmethod
    async def get_page_id_for_record(self, record_id: str) -> ContextPageId | None:
        """Get the page ID associated with a specific record ID, if any."""
        pass

    @abstractmethod
    async def get_record_ids_for_page(self, page_id: ContextPageId) -> list[str]:
        """Get all record IDs associated with a specific page ID."""
        pass

    @abstractmethod
    async def get_all_mapped_records(self) -> dict[str, ContextPageId]:
        """Get a mapping of all record IDs to their associated page IDs."""
        pass

    @abstractmethod
    async def get_all_mapped_pages(self) -> dict[ContextPageId, list[str]]:
        """Get a mapping of all page IDs to their associated record IDs."""
        pass


class BuilInContextPageSourceType(str, Enum):
    """Built-in context page source types for easy reference.
    Users can also register custom types via ContextPageSourceFactory.
    """
    FILE_GROUPER = "file_grouper"
    BLACKBOARD = "blackboard"


class ContextPageSourceFactory:
    """Factory for creating ContextPageSource instances."""

    _registry: dict[str, type[ContextPageSource]] = {}

    @staticmethod
    def register_new_source_type(source_type: str):
        """A decorator to register a new ContextPageSource type."""
        def decorator(source_class: type[ContextPageSource]):
            if not hasattr(ContextPageSourceFactory, "_registry"):
                ContextPageSourceFactory._registry = {}
            ContextPageSourceFactory._registry[source_type] = source_class
            return source_class
        return decorator

    @staticmethod
    def create(
        source_type: str,
        scope_id: str,
        tenant_id: str,
        mmap_config: MmapConfig,
        *args: Any,
        **kwargs: Any
    ) -> ContextPageSource:
        """Create and initialize a ContextPageSource.

        Args:
            source_type: Type of source to create ("file_grouper", etc.)
            *args: Positional arguments for source constructor
            **kwargs: Keyword arguments for source constructor

        Returns:
            Initialized ContextPageSource instance
        """
        if source_type in ContextPageSourceFactory._registry:
            return ContextPageSourceFactory._registry[source_type](
                scope_id=scope_id,
                tenant_id=tenant_id,
                mmap_config=mmap_config,
                *args,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown ContextPageSource type: {source_type}")


