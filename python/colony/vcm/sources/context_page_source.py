"""Context page source: Interface for page clustering and retrieval.

This module provides abstractions for grouping virtual context pages
and finding relevant pages based on queries (key-query-value attention).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import networkx as nx
from pydantic import BaseModel, Field

from ..page_storage import PageStorage

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

    Implementations can use different strategies:
    - File-based: Uses FileGrouper from sharding infrastructure
    - Semantic: Uses embedding-based clustering
    - Hybrid: Combines multiple signals
    - LLM-learned: Uses LLM to determine clusters
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the page source (load or build page graph)."""
        pass

    @abstractmethod
    async def get_page_storage(self) -> PageStorage:
        """Get the page storage instance."""
        pass

    @abstractmethod
    async def load_page_graph(self) -> nx.DiGraph:
        """Load the page graph."""
        pass

    @abstractmethod
    async def get_page_cluster(
        self,
        cluster_size: int = 10,
        cluster_type: str | None = None
    ) -> PageCluster:
        """Get a cluster of related pages.

        Args:
            cluster_size: Desired number of pages in cluster
            cluster_type: Optional filter for cluster type

        Returns:
            PageCluster with related pages
        """
        pass

    @abstractmethod
    async def get_all_clusters(
        self,
        max_cluster_size: int = 10,
        min_cluster_size: int = 2
    ) -> AsyncIterator[PageCluster]:
        """Iterate over all page clusters.

        Args:
            max_cluster_size: Maximum pages per cluster
            min_cluster_size: Minimum pages per cluster

        Yields:
            PageCluster objects
        """
        pass

    @abstractmethod
    async def update_page_graph(
        self,
        page_relationships: dict[tuple[str, str], dict[str, Any]]
    ) -> None:
        """Update page relationship graph based on LLM inference patterns.

        Args:
            page_relationships: Dict mapping (source_page, target_page) → relationship info
        """
        pass

    @abstractmethod
    async def get_page_neighbors(
        self,
        page_id: str,
        max_neighbors: int = 5,
        relationship_types: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Get nearest neighbor pages for a given page.

        Args:
            page_id: Page to find neighbors for
            max_neighbors: Maximum number of neighbors
            relationship_types: Filter by relationship types

        Returns:
            List of (neighbor_page_id, relationship_score) tuples
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get configuration dict for recreating this source.

        Returns:
            Configuration dictionary
        """
        pass


