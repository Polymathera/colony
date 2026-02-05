"""Enhanced Blackboard: Production-grade working memory abstraction.

This package provides a powerful, policy-driven, event-driven blackboard
implementation for multi-agent systems.

Key Features:
- Pluggable backends (memory, distributed, Redis)
- Event-driven notifications via Redis pub-sub
- Policy-based customization (access, eviction, validation)
- Transactions with optimistic locking
- Rich metadata (TTL, tags, versioning, audit trail)
- Efficient backend-specific queries

Quick Start:
    ```python
    from polymathera.colony.agents.blackboard import (
        EnhancedBlackboard,
        BlackboardScope,
        KeyPatternFilter,
    )

    # Create blackboard
    board = EnhancedBlackboard(
        app_name="my-app",
        scope=BlackboardScope.SHARED,
        scope_id="team-1",
    )
    await board.initialize()

    # Write with metadata
    await board.write(
        "analysis_results",
        my_results,
        created_by="agent-123",
        tags={"analysis", "final"},
        ttl_seconds=3600,
    )

    # Subscribe to changes
    async def on_result_updated(event):
        print(f"Result updated: {event.value}")

    board.subscribe(on_result_updated, filter=KeyPatternFilter("*_results"))

    # Atomic transaction
    async with board.transaction() as txn:
        counter = await txn.read("counter") or 0
        await txn.write("counter", counter + 1)

    # Query by namespace
    results = await board.query(namespace="agent:*:results")
    ```
"""

# Main blackboard interface
from .blackboard import EnhancedBlackboard

# Core types
from .types import (
    AccessPolicy,
    BlackboardEntry,
    BlackboardEvent,
    BlackboardScope,
    EventFilter,
    EvictionPolicy,
    ValidationPolicy,
    # Event filters
    KeyPatternFilter,
    EventTypeFilter,
    AgentFilter,
    CombinationFilter,
    # Policy implementations
    NoOpAccessPolicy,
    LRUEvictionPolicy,
    LFUEvictionPolicy,
    NoOpValidationPolicy,
    TypeValidationPolicy,
)

from .task_graph import TaskGraph, Task, TaskStatus
from .obligation_graph import ObligationGraph, ComplianceRelationship
from .causality_timeline import (
    VectorClock,
    EventType,
    CausalRelation,
    CausalEvent,
    CausalityMatrix,
    ConcurrencyConflict,
    CausalityTimeline,
)

# Backend protocol and transaction
from .backend import (
    BlackboardBackend,
    BlackboardTransaction,
    ConcurrentModificationError,
)

# Backend implementations
from .backends import (
    InMemoryBackend,
    DistributedBackend,
    RedisBackend,
)

# Event bus
from .events import EventBus

__all__ = [
    # Main interface
    "EnhancedBlackboard",
    # Core types
    "BlackboardEntry",
    "BlackboardEvent",
    "BlackboardScope",
    # Policy ABCs
    "AccessPolicy",
    "EvictionPolicy",
    "ValidationPolicy",
    # Policy implementations
    "NoOpAccessPolicy",
    "LRUEvictionPolicy",
    "LFUEvictionPolicy",
    "NoOpValidationPolicy",
    "TypeValidationPolicy",
    # Event filters
    "EventFilter",
    "KeyPatternFilter",
    "EventTypeFilter",
    "AgentFilter",
    "CombinationFilter",
    # Task graph
    "TaskGraph",
    "Task",
    "TaskStatus",
    # Obligation graph
    "ObligationGraph",
    "ComplianceRelationship",
    # Causality timeline
    "VectorClock",
    "EventType",
    "CausalRelation",
    "CausalEvent",
    "CausalityMatrix",
    "ConcurrencyConflict",
    "CausalityTimeline",
    # Backend protocol
    "BlackboardBackend",
    "BlackboardTransaction",
    "ConcurrentModificationError",
    # Backend implementations
    "InMemoryBackend",
    "DistributedBackend",
    "RedisBackend",
    # Event bus
    "EventBus",
]