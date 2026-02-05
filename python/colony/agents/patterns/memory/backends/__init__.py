"""Storage backends for the memory system.

This module provides storage backend implementations:
- BlackboardStorageBackend: Default, delegates to EnhancedBlackboard
- BlackboardStorageBackendFactory: Factory for creating backends for arbitrary scopes

Future backends (TODO):
- VectorStorageBackend: For semantic search via vector DBs
- HybridStorageBackend: Blackboard + Vector for dual access
- ParametricStorageBackend: LoRA adapters (experimental)
"""

from .blackboard import BlackboardStorageBackend, BlackboardStorageBackendFactory

__all__ = ["BlackboardStorageBackend", "BlackboardStorageBackendFactory"]

