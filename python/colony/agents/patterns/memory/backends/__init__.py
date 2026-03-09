"""Storage backends for the memory system.

This module provides storage backend implementations:
- BlackboardStorageBackend: Default, delegates to EnhancedBlackboard (logical queries)
- ChromaStorageBackend: Vector storage via embedded ChromaDB (semantic search)
- HybridStorageBackend: Blackboard + ChromaDB dual-write (logical + semantic)
- HybridStorageBackendFactory: Factory that creates hybrid backends with graceful fallback
"""

from .blackboard import BlackboardStorageBackend, BlackboardStorageBackendFactory
from .hybrid import HybridStorageBackend, HybridStorageBackendFactory

__all__ = [
    "BlackboardStorageBackend",
    "BlackboardStorageBackendFactory",
    "HybridStorageBackend",
    "HybridStorageBackendFactory",
]
