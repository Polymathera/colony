"""Virtual Context Manager (VCM) layer.

This module provides virtual memory management for extremely long contexts
(up to billions of tokens) by paging context into GPU KV caches on demand.

Key Components:
- VirtualContextPage: A chunk of tokens (20k-40k) that can be loaded into KV cache
- VirtualPageTable: Tracks which pages are loaded where
- VirtualContextManager: Main deployment that manages the VCM
"""

from .config import VCMConfig
from .manager import VirtualContextManager
from .models import (
    PageFault,
    PageGroup,
    PageLocation,
    VirtualContextPage,
    VirtualPageTableState,
)
from .page_table import VirtualPageTable

__all__ = [
    "VCMConfig",
    "VirtualContextPage",
    "PageLocation",
    "PageGroup",
    "PageFault",
    "VirtualPageTableState",
    "VirtualPageTable",
    "VirtualContextManager",
]
