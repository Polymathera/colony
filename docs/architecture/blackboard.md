# Blackboard

The `EnhancedBlackboard` is the single source of truth for all agent state in Colony. It provides a shared, observable, transactional key-value store backed by Redis, with event-driven notifications and policy-based customization.

## Core Philosophy

The blackboard design follows five principles:

1. **Composability over inheritance**: Behaviors are composed from pluggable policies, not inherited from base classes
2. **Policy-based design**: Access control, eviction, and validation are pluggable policies using duck-typed protocols
3. **Observable by default**: Every write produces an event; subscribers react to state changes
4. **Transactional integrity**: Optimistic concurrency control prevents lost updates
5. **Strongly typed**: Entries carry metadata, tags, and versioning information

## `EnhancedBlackboard`

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

    def __init__(
        self,
        app_name: str,
        scope: BlackboardScope = BlackboardScope.LOCAL,
        scope_id: str = "default",
        # Policy customization points
        access_policy: AccessPolicy | None = None,
        eviction_policy: EvictionPolicy | None = None,
        validation_policy: ValidationPolicy | None = None,
        # Backend selection
        backend: BlackboardBackend | None = None,
        backend_type: str | None = None,  # "memory", "distributed", "redis"
        # Event system
        enable_events: bool = True,
        max_event_queue_size: int = 1000,
        # Resource limits
        max_entries: int | None = None,
    ): ...
```

### Scopes

Blackboard instances are scoped via `BlackboardScope`:

```python
class BlackboardScope(str, Enum):
    LOCAL = "local"       # Agent-local (in-memory, not shared)
    SHARED = "shared"     # Shared among specific agents
    GLOBAL = "global"     # Shared among all agents in app
    PERSISTENT = "persistent"  # Persisted to VCM/disk
```

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

```python
board = EnhancedBlackboard(
    app_name="my-app",
    scope=BlackboardScope.SHARED,
    scope_id="team-1",
    access_policy=MyAccessPolicy(),
    validation_policy=SchemaValidator(MySchema),
)
await board.initialize()

# Write with metadata and TTL
await board.write(
    "analysis_results",
    my_results,
    created_by="agent-123",
    tags={"analysis", "final"},
    ttl_seconds=3600,
)

# Read
value = await board.read("analysis_results")

# Read full entry with metadata
entry = await board.read_entry("analysis_results")
print(entry.version, entry.tags, entry.updated_at)

# Query by namespace pattern and tags
entries = await board.query(
    namespace="agent:*:results",
    tags={"analysis"},
    limit=50,
)

# Batch operations
values = await board.read_batch(["key1", "key2", "key3"])
await board.write_batch(
    {"key1": val1, "key2": val2},
    created_by="agent-123",
    tags={"batch"},
)

# Ambient transaction -- read/write/delete transparently route
# through transaction buffers while the context is active
async with board.transaction() as txn:
    counter = await board.read("counter") or 0   # routed through txn
    await board.write("counter", counter + 1)     # buffered until commit
    # commit happens automatically on __aexit__
```

## Event System

Every blackboard write produces a `BlackboardEvent` distributed via Redis pub-sub. Events carry:

- **Event type**: Write, delete, expire, update
- **Key**: The affected blackboard key
- **Value**: The new value (for write/update events)
- **Metadata**: Tags, timestamp, version, created_by

Subscribers register via `EventFilter` which supports key patterns and event type filtering. This enables reactive architectures -- capabilities and memory levels respond to state changes without polling.

```python
@dataclass
class BlackboardEvent:
    event_type: str              # "write", "delete", "clear"
    key: str | None              # None for clear events
    value: Any | None            # None for delete/clear events
    event_id: str                # Auto-generated unique ID
    version: int = 0
    old_value: Any | None = None # Previous value (for updates)
    timestamp: float             # Auto-set to time.time()
    agent_id: str | None = None
    tags: set[str]               # For querying
    metadata: dict[str, Any]     # Extensible
```

Subscribing to events:

```python
# Callback-based subscription
async def on_result_updated(event: BlackboardEvent):
    print(f"Result updated by {event.agent_id}: {event.key}")

board.subscribe(on_result_updated, filter=KeyPatternFilter("*:results:*"))

# Async iterator -- long-running background monitoring
async for event in board.stream_events(
    filter=KeyPatternFilter("scope:*:analysis:*"),
    until=lambda: self._stopped,
):
    await process(event)

# Queue-based -- feed events into an asyncio.Queue for plan_step
event_queue = board.stream_events_to_queue(
    pattern="agent:*:result:*",
    event_types={"write"},
)
```

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

```python
async with board.transaction() as txn:
    # Reads record version tokens for optimistic check at commit
    entry = await txn.read("shared_counter")
    current = entry.value if entry else 0

    # Writes are buffered locally
    await txn.write("shared_counter", BlackboardEntry(
        key="shared_counter", value=current + 1, version=(entry.version + 1) if entry else 0,
        created_by="agent-123", updated_at=time.time(), created_at=entry.created_at if entry else time.time(),
    ))
    # On __aexit__: versions are checked via compare-and-swap.
    # If another writer modified "shared_counter" after our read,
    # ConcurrentModificationError is raised and the transaction must be retried.
```

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

## `BlackboardEntry`

Every value stored in the blackboard is wrapped in a `BlackboardEntry` with rich metadata:

```python
class BlackboardEntry(BaseModel):
    key: str
    value: Any = None
    version: int = 0                       # Incremented on each write
    created_at: float                      # Unix timestamp
    updated_at: float                      # Unix timestamp
    created_by: str | None = None          # Agent ID
    updated_by: str | None = None          # Agent ID
    ttl_seconds: float | None = None       # Time-to-live
    tags: set[str] = set()                 # For querying
    metadata: dict[str, Any] = {}          # Extensible
```

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

## `KeyPatternFilter` and `EventFilter`

`KeyPatternFilter` (in `polymathera.colony.agents.blackboard.types`) supports glob-style pattern matching for querying entries. `EventFilter` extends this with event type filtering for subscriptions.

```python
class EventFilter(ABC):
    @abstractmethod
    def matches(self, event: BlackboardEvent) -> bool: ...

# Filter by key glob pattern
@dataclass
class KeyPatternFilter(EventFilter):
    pattern: str  # e.g., "agent:*:results"
    def matches(self, event: BlackboardEvent) -> bool:
        return event.key and fnmatch.fnmatch(event.key, self.pattern)

# Filter by event type
@dataclass
class EventTypeFilter(EventFilter):
    event_types: set[str]  # e.g., {"write", "delete"}
    def matches(self, event: BlackboardEvent) -> bool:
        return event.event_type in self.event_types

# Filter by agent ID
@dataclass
class AgentFilter(EventFilter):
    agent_ids: set[str]
    def matches(self, event: BlackboardEvent) -> bool:
        return event.agent_id in self.agent_ids

# Combine key pattern + event type
@dataclass
class CombinationFilter(EventFilter):
    pattern: str
    event_types: set[str]
    def matches(self, event: BlackboardEvent) -> bool:
        return (event.key and fnmatch.fnmatch(event.key, self.pattern)
                and event.event_type in self.event_types)
```

These filters define memory scope boundaries -- a memory level is specified by a list of one or more `KeyPatternFilter` instances that define its scope within the blackboard.

## Policy Protocols

Policies use Python protocols (duck typing), not abstract base classes:

- **Access policy**: Controls read/write permissions per key/scope
- **Eviction policy**: Determines which entries to evict under memory pressure
- **Validation policy**: Validates entries before write (schema, constraints)

```python
class AccessPolicy(ABC):
    @abstractmethod
    async def can_read(self, agent_id: str, key: str, scope_id: str) -> bool: ...
    @abstractmethod
    async def can_write(self, agent_id: str, key: str, value: Any, scope_id: str) -> bool: ...
    @abstractmethod
    async def can_delete(self, agent_id: str, key: str, scope_id: str) -> bool: ...

class EvictionPolicy(ABC):
    @abstractmethod
    async def get_eviction_candidates(
        self, entries: list[BlackboardEntry], num_to_evict: int
    ) -> list[str]: ...

class ValidationPolicy(ABC):
    @abstractmethod
    async def validate(self, key: str, value: Any, metadata: dict[str, Any]) -> None: ...
```

Built-in implementations:

```python
# Access: allow everything (default)
NoOpAccessPolicy()

# Eviction: least recently used (default)
LRUEvictionPolicy()  # sorts by updated_at ascending

# Eviction: least frequently used (requires access_count in metadata)
LFUEvictionPolicy()

# Validation: enforce type constraints per key pattern
TypeValidationPolicy({"config.*": dict, "count.*": int})

# Validation: no-op (default)
NoOpValidationPolicy()
```

This enables flexible composition -- multiple policies without inheritance hierarchies, easy testing with simple mock objects, and runtime swapping of behavior.

## VCM Integration

When a blackboard scope is VCM-mapped via `mmap_application_scope()`, writes to that scope are automatically picked up by the `BlackboardContextPageSource` running inside the VCM. The data eventually appears in a VCM page and becomes discoverable by other agents via `QueryAttentionCapability`.

This bridges the two halves of the Extended VCM: the blackboard provides read-write coordination state, and the VCM makes that state available as context pages for deep reasoning.

## Communication Protocols

The blackboard is a KV store, not an append-only log. When multiple agents share a scope, they need to agree on **key format** so writers and readers match. Protocols formalize this agreement.

### The Problem Protocols Solve

Without protocols, every reader/writer pair invents its own key format:

- `AgentHandle.run()` uses `AgentRunProtocol.request_key(req_id, namespace=ns)` → `{ns}:request:run:{ns}:{req_id}`
- `AgentCapability.send_request()` writes `request_id:{X}:request_type:{Y}:sender:{Z}`
- A coordinator reads `result:*` but a worker writes `agent_id:{id}:result_type:final`

These never match. Protocols eliminate this by providing a single shared vocabulary.

### Keys Are Scope-Relative

!!! warning "Important"
    All blackboard keys are **scope-relative**. The `scope_id` is the partition — it lives in the Redis namespace, not in the key itself. Never include `scope_id` in keys, patterns, or queries.

    ```python
    # WRONG — scope_id prefix in key
    await blackboard.write(f"{self.scope_id}:result:123", data)
    await blackboard.query(namespace=f"{self.scope_id}:*")

    # CORRECT — scope-relative key
    await blackboard.write("result:123", data)
    await blackboard.query(namespace="*")
    ```

    The blackboard already knows its scope from `__init__`. All operations automatically happen within that scope.

### `BlackboardProtocol` Base Class

Every protocol subclasses `BlackboardProtocol` and declares:

- **`scope`**: Which `BlackboardScope` level it operates at (agent, session, colony)
- **Key methods**: How to construct keys for each message type
- **Pattern methods**: How to construct subscription patterns

```python
from polymathera.colony.agents.blackboard.protocol import BlackboardProtocol
from polymathera.colony.agents.scopes import BlackboardScope, ScopeUtils

class MyProtocol(BlackboardProtocol):
    scope = BlackboardScope.COLONY

    @staticmethod
    def request_key(request_id: str, agent_id: str) -> str:
        return ScopeUtils.format_key(agent_id=agent_id, my_request=request_id)

    @staticmethod
    def request_pattern(agent_id: str | None = None) -> str:
        return ScopeUtils.pattern_key(agent_id=agent_id, my_request=None)
```

### Built-In Protocols

Colony provides protocols for common communication patterns:

#### Request/Result with Streaming (`AgentRunProtocol`)

Used by `AgentHandle.run()` and `run_streamed()` for parent-child agent communication. Every call requires a `namespace` parameter that identifies which capability should handle the request. This prevents interference when multiple capabilities share a scope or when an agent has multiple capabilities using the same protocol.

```python
from polymathera.colony.agents.blackboard.protocol import AgentRunProtocol

# Every protocol method requires a namespace
key = AgentRunProtocol.request_key("req_abc123", namespace="compliance")
# -> "request:run:compliance:req_abc123"

key = AgentRunProtocol.result_key("req_abc123", namespace="compliance")
# -> "result:run:compliance:req_abc123"

# Subscription patterns — only match this capability's keys
AgentRunProtocol.request_pattern(namespace="compliance")  # -> "compliance:request:run:compliance:*"
AgentRunProtocol.result_pattern(namespace="compliance")   # -> "compliance:result:run:compliance:*"

# AgentHandle.run() passes the namespace to reach the right capability
run = await handle.run(input_data, namespace="compliance")
```

#### Work Assignment (`WorkAssignmentProtocol`)

Used by `AgentPoolCapability` for coordinator-worker communication. Colony-level scope — `agent_id` **is** in the key because multiple workers share the partition.

```python
from polymathera.colony.agents.blackboard.protocol import WorkAssignmentProtocol

# Coordinator assigns work to a specific worker
key = WorkAssignmentProtocol.assignment_key(agent_id="worker-1", request_id="task-42")

# Worker writes result back
key = WorkAssignmentProtocol.result_key(agent_id="worker-1", result_type="final")

# Coordinator subscribes to all worker results
pattern = WorkAssignmentProtocol.result_pattern(namespace="pool")
# -> "agent_id:*:result_type:*"
```

#### Lifecycle Signals (`LifecycleSignalProtocol`)

Used by `MemoryLifecycleHooks` to emit agent creation/termination signals on the colony control plane. `agent_id` in the key prevents overwrites when multiple agents terminate concurrently.

```python
from polymathera.colony.agents.blackboard.protocol import LifecycleSignalProtocol

# Agent emits termination signal
key = LifecycleSignalProtocol.terminated_key("agent-xyz")

# MemoryManagementAgent subscribes to all terminations
pattern = LifecycleSignalProtocol.terminated_pattern(namespace="lifecycle")
# -> "agent_id:*:scope:agent_terminated"
```

#### Other Built-In Protocols

| Protocol | Scope | Use Case |
|----------|-------|----------|
| `GameStateProtocol` | Colony | Game state publication via OCC |
| `CritiqueProtocol` | Colony | Peer/parent/child critique exchange |
| `ConsistencyCheckProtocol` | Colony | Consistency check requests/results |
| `GroundingProtocol` | Agent | Grounding claim requests/results |
| `GoalAlignmentProtocol` | Colony | Goal alignment and joint goal registration |
| `WorkingSetStateProtocol` | Colony | VCM working set state publication |
| `PlanProtocol` | Colony | Colony-wide plan publication |

### Declaring Protocols on Capabilities

Capabilities declare which protocols they support via `input_patterns`:

```python
from polymathera.colony.agents.base import AgentCapability
from polymathera.colony.agents.blackboard.protocol import AgentRunProtocol

class MyWorkerCapability(AgentCapability):
    """Handles incoming compliance requests."""

    input_patterns = [AgentRunProtocol.request_pattern(namespace="compliance")]
    # -> ["request:run:compliance:*"]

    @event_handler(pattern=AgentRunProtocol.request_pattern(namespace="compliance"))
    async def handle_request(self, event, repl):
        request_id = AgentRunProtocol.parse_request_key(event.key, namespace="compliance")
        # ... process ...
        await blackboard.write(AgentRunProtocol.result_key(request_id, namespace="compliance"), result)
```

The default `stream_events_to_queue()` uses `input_patterns` to subscribe only to relevant events instead of `"*"`. This prevents colony-scoped capabilities from flooding the action policy queue with every event from every agent.

### Key Validation

The protocol module includes key validation that catches common mistakes:

```python
from polymathera.colony.agents.blackboard.protocol import validate_key, validate_pattern

validate_key("request:run:req_abc")      # OK
validate_key("polymathera:tenant:...")   # KeyValidationError: scope-absolute prefix
validate_key("result:*")                 # KeyValidationError: wildcard in key
validate_key("")                         # KeyValidationError: empty key

validate_pattern("request:*")            # OK
validate_pattern("polymathera:tenant:*") # KeyValidationError: scope-absolute prefix
```

### When Agent-Level Scope Needs `agent_id` in Keys (and When It Doesn't)

The decision depends on whether the scope is **shared** or **private**:

| Scope type | Multiple writers? | `agent_id` in key? | Example |
|------------|-------------------|-------------------|---------|
| Agent-level | No (one agent) | No — scope IS the mailbox | `AgentRunProtocol`: `{ns}:request:run:{ns}:{req_id}` |
| Colony-level | Yes (all agents) | Yes — prevents overwrites | `WorkAssignmentProtocol`: `agent_id:{id}:work_assignment:{req_id}` |
| Game-level | Yes (participants) | Depends — game state has one canonical key per game | `GameStateProtocol`: `state:{game_id}` |
| Control plane | Yes (all agents) | Yes — concurrent signals | `LifecycleSignalProtocol`: `agent_id:{id}:scope:agent_terminated` |

### Writing Custom Protocols

To define a new protocol for a custom capability:

```python
class MyAnalysisProtocol(BlackboardProtocol):
    """Protocol for my custom analysis workflow."""
    scope = BlackboardScope.SESSION
    _TASK_PREFIX = "analysis_task:"
    _RESULT_PREFIX = "analysis_result:"

    @staticmethod
    def task_key(task_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{MyAnalysisProtocol._TASK_PREFIX}{task_id}")

    @staticmethod
    def result_key(task_id: str, namespace: str) -> str:
        return BlackboardProtocol._ns(namespace, f"{MyAnalysisProtocol._RESULT_PREFIX}{task_id}")

    @staticmethod
    def task_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, f"{MyAnalysisProtocol._TASK_PREFIX}*")

    @staticmethod
    def result_pattern(namespace: str) -> str:
        return BlackboardProtocol._ns_pattern(namespace, f"{MyAnalysisProtocol._RESULT_PREFIX}*")

    @staticmethod
    def parse_task_key(key: str, namespace: str) -> str:
        """Extract task_id from a task key."""
        key = key[len(namespace) + 1:]  # strip namespace prefix
        return key[len(MyAnalysisProtocol._TASK_PREFIX):]


class MyAnalysisCapability(AgentCapability):
    protocols = [MyAnalysisProtocol]
    input_patterns = [MyAnalysisProtocol.task_pattern(namespace="my_analysis")]

    @event_handler(pattern=MyAnalysisProtocol.task_pattern(namespace="my_analysis"))
    async def handle_task(self, event, repl):
        task_id = MyAnalysisProtocol.parse_task_key(event.key, namespace="my_analysis")
        # ... run analysis ...
        blackboard = await self.get_blackboard()
        await blackboard.write(
            MyAnalysisProtocol.result_key(task_id, namespace="my_analysis"),
            result_data,
        )
```

### Best Practices

1. **Use protocol methods, not raw strings.** `AgentRunProtocol.request_key(req_id, namespace="compliance")` is self-documenting. `f"request:run:{req_id}"` requires reading the protocol code to understand.

2. **Declare `input_patterns`.** Every capability should declare what events it monitors. This prevents the `"*"` fallback which creates excessive noise in colony-scoped blackboards.

3. **Never include `scope_id` in keys or patterns.** The blackboard partition already provides isolation.

4. **Use exact keys for `CapabilityResultFuture`.** Never `"result:*"` — use `protocol.result_key(request_id)` with the specific request ID to avoid resolving unrelated concurrent operations.

5. **Colony-scoped keys must disambiguate writers.** Include `agent_id`, `page_id`, or another unique identifier in keys when multiple agents share the scope.

<div style="margin:1.5rem 0;">

<style>
/* ── Blackboard Protocol Diagram ── */
.bb-svg text { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.bb-svg .r-agent { fill: #f5f3ff; stroke: #8b5cf6; }
.bb-svg .r-bb    { fill: #eff6ff; stroke: #3b82f6; }
.bb-svg .r-proto { fill: #ecfdf5; stroke: #10b981; }
.bb-svg .t-title { fill: #1e1b4b; }
.bb-svg .t-body  { fill: #374151; }
.bb-svg .t-muted { fill: #6b7280; }
.bb-svg .t-proto { fill: #064e3b; }
[data-md-color-scheme="slate"] .bb-svg .r-agent { fill: #1e1338; stroke: #7c3aed; }
[data-md-color-scheme="slate"] .bb-svg .r-bb    { fill: #0c1929; stroke: #2563eb; }
[data-md-color-scheme="slate"] .bb-svg .r-proto { fill: #052e16; stroke: #059669; }
[data-md-color-scheme="slate"] .bb-svg .t-title { fill: #c4b5fd; }
[data-md-color-scheme="slate"] .bb-svg .t-body  { fill: #d1d5db; }
[data-md-color-scheme="slate"] .bb-svg .t-muted { fill: #9ca3af; }
[data-md-color-scheme="slate"] .bb-svg .t-proto { fill: #6ee7b7; }
</style>

<svg class="bb-svg" viewBox="0 0 740 340" xmlns="http://www.w3.org/2000/svg">
  <!-- Parent agent -->
  <rect class="r-agent" x="20" y="20" width="200" height="100" rx="8" stroke-width="1.5"/>
  <text class="t-title" x="120" y="50" text-anchor="middle" font-size="13" font-weight="600">Parent Agent</text>
  <text class="t-body" x="120" y="72" text-anchor="middle" font-size="11">AgentHandle.run()</text>
  <text class="t-muted" x="120" y="92" text-anchor="middle" font-size="10">protocol=AgentRunProtocol</text>

  <!-- Protocol box -->
  <rect class="r-proto" x="270" y="30" width="200" height="80" rx="8" stroke-width="1.5"/>
  <text class="t-proto" x="370" y="55" text-anchor="middle" font-size="12" font-weight="600">AgentRunProtocol</text>
  <text class="t-proto" x="370" y="75" text-anchor="middle" font-size="10">request:run:{req_id}</text>
  <text class="t-proto" x="370" y="90" text-anchor="middle" font-size="10">result:run:{req_id}</text>

  <!-- Child blackboard -->
  <rect class="r-bb" x="520" y="20" width="200" height="100" rx="8" stroke-width="1.5"/>
  <text class="t-title" x="620" y="50" text-anchor="middle" font-size="13" font-weight="600">Child Blackboard</text>
  <text class="t-body" x="620" y="72" text-anchor="middle" font-size="11">scope: agent:{child_id}</text>
  <text class="t-muted" x="620" y="92" text-anchor="middle" font-size="10">keys are scope-relative</text>

  <!-- Arrows: parent -> protocol -> blackboard -->
  <line x1="220" y1="55" x2="268" y2="55" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#bb-arrow)"/>
  <line x1="470" y1="55" x2="518" y2="55" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#bb-arrow-blue)"/>

  <!-- Child agent -->
  <rect class="r-agent" x="520" y="160" width="200" height="100" rx="8" stroke-width="1.5"/>
  <text class="t-title" x="620" y="190" text-anchor="middle" font-size="13" font-weight="600">Child Agent</text>
  <text class="t-body" x="620" y="212" text-anchor="middle" font-size="11">@event_handler</text>
  <text class="t-muted" x="620" y="232" text-anchor="middle" font-size="10">pattern="request:run:*"</text>

  <!-- Arrow: blackboard -> child (event) -->
  <line x1="620" y1="120" x2="620" y2="158" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#bb-arrow-blue)"/>
  <text class="t-muted" x="635" y="143" font-size="9">write event</text>

  <!-- Arrow: child -> blackboard (result) -->
  <line x1="540" y1="160" x2="540" y2="122" stroke="#10b981" stroke-width="1.5" marker-end="url(#bb-arrow-green)"/>
  <text class="t-muted" x="500" y="143" font-size="9" text-anchor="end">result:run:*</text>

  <!-- Arrow: blackboard -> parent (result event) -->
  <line x1="520" y1="90" x2="222" y2="90" stroke="#10b981" stroke-width="1.5" marker-end="url(#bb-arrow-green-left)"/>
  <text class="t-muted" x="370" y="84" text-anchor="middle" font-size="9">result event resolves future</text>

  <!-- Input patterns box -->
  <rect class="r-proto" x="20" y="180" width="200" height="80" rx="8" stroke-width="1.5"/>
  <text class="t-proto" x="120" y="205" text-anchor="middle" font-size="12" font-weight="600">input_patterns</text>
  <text class="t-proto" x="120" y="225" text-anchor="middle" font-size="10">["request:run:*"]</text>
  <text class="t-proto" x="120" y="242" text-anchor="middle" font-size="10">Filters event subscription</text>

  <!-- Caption -->
  <text class="t-muted" x="370" y="320" text-anchor="middle" font-size="11" font-style="italic">AgentRunProtocol: parent and child agree on key format via shared protocol</text>

  <!-- Arrow markers -->
  <defs>
    <marker id="bb-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#7c3aed"/>
    </marker>
    <marker id="bb-arrow-blue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#3b82f6"/>
    </marker>
    <marker id="bb-arrow-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#10b981"/>
    </marker>
    <marker id="bb-arrow-green-left" markerWidth="8" markerHeight="6" refX="1" refY="3" orient="auto">
      <path d="M8,0 L0,3 L8,6" fill="#10b981"/>
    </marker>
  </defs>
</svg>

</div>
