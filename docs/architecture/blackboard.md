# Blackboard

The `EnhancedBlackboard` is the single source of truth for all agent state in Colony. It provides a shared, observable, transactional key-value store backed by Redis, with event-driven notifications and policy-based customization.

## Core Philosophy

The blackboard design follows five principles:

1. **Composability over inheritance**: Behaviors are composed from pluggable policies, not inherited from base classes
2. **Policy-based design**: Access control, eviction, and validation are pluggable policies using duck-typed protocols
3. **Observable by default**: Every write produces an event; subscribers react to state changes
4. **Transactional integrity**: Optimistic concurrency control prevents lost updates
5. **Strongly typed**: Entries carry metadata, tags, and versioning information

## EnhancedBlackboard

Defined in `polymathera.colony.agents.blackboard.blackboard`:

```python
class EnhancedBlackboard:
    """Production-grade blackboard.

    Features:
    - Pluggable backends (memory, distributed, Redis)
    - Event-driven notifications via Redis pub-sub
    - Policy-based customization (access, eviction, validation)
    - Transactions with optimistic locking
    - Rich metadata (TTL, tags, versioning)
    - Efficient backend-specific queries
    """
```

### Scopes

Blackboard instances are scoped via `BlackboardScope`:

- **Agent-private scope**: Only the owning agent can read/write
- **Shared scope**: Accessible to all agents in the application
- **Task scope**: Scoped to a specific task or coordination group

The `scope_id` parameter determines the namespace for all keys in that blackboard instance.

### Operations

| Operation | Description |
|-----------|-------------|
| `write(key, value, ...)` | Write entry with optional tags, metadata, TTL |
| `read(key)` | Read single entry |
| `query(pattern)` | Query entries by key glob pattern |
| `delete(key)` | Remove an entry |
| `subscribe(filter)` | Subscribe to events matching a filter |
| `transaction()` | Start an optimistic transaction |

## Event System

Every blackboard write produces a `BlackboardEvent` distributed via Redis pub-sub. Events carry:

- **Event type**: Write, delete, expire, update
- **Key**: The affected blackboard key
- **Value**: The new value (for write/update events)
- **Metadata**: Tags, timestamp, version, created_by

Subscribers register via `EventFilter` which supports key patterns and event type filtering. This enables reactive architectures -- capabilities and memory levels respond to state changes without polling.

```mermaid
sequenceDiagram
    participant Agent as Agent A
    participant BB as EnhancedBlackboard
    participant Redis as Redis Pub-Sub
    participant Mem as MemoryCapability
    participant Agent2 as Agent B

    Agent->>BB: write("scope:result:123", data)
    BB->>Redis: PUBLISH event
    Redis->>Mem: Event notification
    Mem->>Mem: Ingest into memory scope
    Redis->>Agent2: Event notification
    Agent2->>Agent2: React to new data
```

## Optimistic Concurrency Control

The blackboard uses optimistic concurrency control (OCC) for transactions. Each entry has a version number. When a transaction commits:

1. The transaction reads entries and records their versions
2. Modifications are prepared locally
3. At commit time, versions are checked against current state
4. If any version has changed (another writer intervened), the transaction is retried

!!! warning "Write transaction pattern"
    When using `async for state in state_manager.write_transaction()`, never `return` or `break` from inside the loop after modifying state. Python async generators skip post-yield cleanup code (the `compare_and_swap`) when the caller exits early. Use a result variable, let the loop body complete naturally, and return after the loop.

## Storage Backends

The blackboard supports pluggable storage backends:

| Backend | Use Case |
|---------|----------|
| **InMemory** | Development, testing, single-process scenarios |
| **Redis** | Production distributed deployment |
| **VCM-backed** | Page-mapped data that should be discoverable via VCM |

Backend selection is configured via cluster config and can be overridden per-blackboard instance.

## Blackboard as Memory Foundation

The memory system is built entirely on top of the blackboard:

- Each memory level (working, STM, LTM) is a blackboard scope
- Memory queries translate to blackboard key pattern matching + tag filtering
- Memory events are blackboard events
- Memory subscriptions are blackboard event subscriptions with transformation

This means the blackboard is not just a coordination mechanism -- it is the foundation of the entire memory architecture.

## Key Patterns and Namespacing

Blackboard keys follow a hierarchical namespace convention:

```
{scope_id}:{data_type}:{identifier}
```

Examples:
- `agent:abc123:working:record:action+success:a1b2c3d4` -- A working memory record
- `game:hypothesis-42:state:current` -- Current state of a hypothesis game
- `task:analysis-7:result:summary` -- Task result summary

The `BlackboardPublishable` protocol and legacy `get_blackboard_key()` method on data models provide automatic key generation.

## KeyPatternFilter and EventFilter

`KeyPatternFilter` (in `polymathera.colony.agents.blackboard.types`) supports glob-style pattern matching for querying entries. `EventFilter` extends this with event type filtering for subscriptions.

These filters define memory scope boundaries -- a memory level is specified by a list of one or more `KeyPatternFilter` instances that define its scope within the blackboard.

## Policy Protocols

Policies use Python protocols (duck typing), not abstract base classes:

- **Access policy**: Controls read/write permissions per key/scope
- **Eviction policy**: Determines which entries to evict under memory pressure
- **Validation policy**: Validates entries before write (schema, constraints)

This enables flexible composition -- multiple policies without inheritance hierarchies, easy testing with simple mock objects, and runtime swapping of behavior.

## VCM Integration

When a blackboard scope is VCM-mapped via `mmap_application_scope()`, writes to that scope are automatically picked up by the `BlackboardContextPageSource` running inside the VCM. The data eventually appears in a VCM page and becomes discoverable by other agents via `QueryAttentionCapability`.

This bridges the two halves of the Extended VCM: the blackboard provides read-write coordination state, and the VCM makes that state available as context pages for deep reasoning.
