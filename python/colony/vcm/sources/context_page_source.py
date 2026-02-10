"""Context page source: Interface for page clustering and retrieval.

This module provides abstractions for grouping virtual context pages
and finding relevant pages based on queries (key-query-value attention).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from pydantic import BaseModel, Field

import networkx as nx

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
        source_type: str = "file_grouper",
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
            return ContextPageSourceFactory._registry[source_type](*args, **kwargs)
        else:
            raise ValueError(f"Unknown ContextPageSource type: {source_type}")


