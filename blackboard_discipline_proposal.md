# Blackboard Programming Model — Proposal

The blackboard is the primary inter-agent communication medium. Scope isolation (which blackboard partition to use) is correct — `scope_id` maps to a separate storage namespace. But **key discipline within a scope** is ad-hoc, with multiple incompatible key formats for the same logical operation. 

## Architecture as it stands

**The blackboard is a partitioned key-value store with event notifications and optimistic concurrency control**.

**Two-level addressing**:
- **Scope ID** determines which storage partition (Redis namespace / StateManager key). Computed by `ScopeUtils.get_*_level_scope()` or `MemoryScope.*`. Two agents sharing the same `scope_id` can read/write each other's keys.
- **Key** is the address within a scope. When multiple agents share a scope, the key MUST encode enough information to avoid overwrites (**the blackboard is not append-only — writing the same key overwrites the previous value**).

**Key construction**: `ScopeUtils.format_key(**kwargs)` sorts kwargs alphabetically and joins as `ns1:val1:ns2:val2`. `ScopeUtils.pattern_key(**kwargs)` does the same but maps `None` values to `*` for glob matching.

**Event delivery**: `EnhancedBlackboard.write()` emits a `BlackboardEvent` containing the full key, new value, old value, `agent_id`, tags, and metadata. Events fan out to ALL subscribers whose filter matches. The built-in filters are `KeyPatternFilter` (`fnmatch` on key), `EventTypeFilter` (write/delete), `AgentFilter` (`agent_id`), and `CombinationFilter` (key pattern + event type + optional checker callback). There is no built-in `TagFilter` for real-time event subscriptions, even though `query()` supports tag-based filtering server-side.

**This asymmetry is why keys encode context**: Key-pattern matching is the only reliable zero-cost filter for real-time events. Encoding `agent_id`, `request_type`, etc. in keys enables targeted subscriptions without custom filter objects. This is a deliberate trade-off, not a mistake.

---

## Problems (grouped by severity)

### A. Protocol mismatches — things that don't work

**A1. `AgentHandle.run()` and capabilities use incompatible key formats**

`AgentHandle.run()` writes key `{child_id}:request:{req_id}` and listens for `{child_id}:result:{req_id}` on the child's agent-level scope.

`AgentCapability.send_request()` writes key `request_id:{req_id}:request_type:{type}:sender:{sender_id}` (alphabetically sorted by `format_key`). `AgentCapability.get_result_future()` listens for `result:{req_id}`.

These never match. A child agent whose capabilities use `send_request`/`get_result_future` cannot interoperate with a parent using `AgentHandle.run()`.

**A2. Sample capabilities use scopes that don't match `AgentHandle.run()`**

`AgentHandle.run()` writes to the child's agent-level scope: `...session:{s}:agent:{child_id}`.

Sample capabilities like `ContractInferenceCapability` construct `scope_id = f"...agent:{id}:contract_inference:{id}"` — a sub-scope. Their `EventBus` channel is different from the agent-level scope's channel. Events written by `AgentHandle.run()` are never delivered to these capabilities.

**A3. `CriticCapability._send_critique_request` calls a nonexistent method**

Line 400: `self.agent.get_agent(scope_id=...)`. `Agent` has no `get_agent()` method. The entire peer/parent critique request path is broken.

**A4. `_setup_resumption_trigger` is dead code**

- Subscribes to scope `"plan_coordination"` — no code in the codebase writes to this scope.
- Expected key pattern `plan:{child_id}` — no code produces this.
- Loop-variable closure bug: all callbacks capture the last `child_id`.

**A5. `scope_id` leaks into keys, patterns, and queries — but stored keys are scope-relative**

All three backends store keys exactly as the caller provides — no `scope_id` prefix is prepended. The `scope_id` is the partition boundary (the Redis namespace or `StateManager` state key), not a key prefix within that partition. Keys are **scope-relative**.

But multiple places in the codebase prefix `scope_id` into keys, query patterns, and event subscription patterns as if keys were **scope-absolute**:

- `@event_handler(pattern="{scope_id}:request:*")` — resolves to `polymathera:tenant:T:colony:C:session:S:agent:A:request:*`. Since stored keys are just `request:*` (no scope prefix), **this pattern never matches** in the default attached-mode case.
- `blackboard.query(namespace=f"{self.scope_id}:*")` — in `blackboard_page_source.py:780`. Will return nothing if keys are scope-relative.
- `stream_events_to_queue(pattern=f"{self.scope_id}:*")` — in `blackboard_page_source.py:818`, `memory/capability.py:539`.
- `effective_pattern = pattern or f"{self._scope_id}:*"` — in `memory/backends/blackboard.py:132`.
- `result_id=f"{self.scope_id}:{page_id}"` — in `vcm_analysis.py:781,859,895,921`.
- `f"{self.scope_id}:consistency:*"` — in `consistency.py:185`.

These only "work" when writers also incorrectly prefix `scope_id` into their keys — two bugs compensating for each other. The root cause: callers should NEVER include `scope_id` in read/write/query/subscribe calls. The `scope_id` is set at `EnhancedBlackboard.__init__` and all operations automatically happen within that scope.

### B. Correctness risks — things that may silently produce wrong results

**B1. `CapabilityResultFuture` with wildcard keys resolves all waiters**

`ConsistencyCapability.get_result_future()` returns key `consistency_check_result:*`. `CapabilityResultFuture._monitor_result()` uses this as an fnmatch pattern. The first write matching `consistency_check_result:*` from ANY source resolves ALL pending futures listening on that pattern. Concurrent consistency checks will get each other's results.

**B2. `ResultCapability.results:index` has a read-modify-write race**

Multiple workers call `store_partial()` concurrently. Each reads the index dict, appends its `result_id`, writes it back. No CAS/transaction protection. Concurrent writes silently drop entries.

**B3. `HypothesisTrackingCapability` last-write-wins in shared scopes**

When scope is `COLONY` or `SESSION`, `_save_to_blackboard()` writes the entire in-memory `_cache` dict to key `tracked_hypotheses:True`. Agent A loads, modifies, saves. Agent B loaded before A saved, modifies, saves — overwrites A's changes.

**B4. Tags lost through Redis pub-sub path**

`EventBus.emit()` (`events.py` line 147-155) builds `event_dict` without `tags`. The receiving side reconstructs `BlackboardEvent` with `tags=set()` (default). Tags survive only through the Redis Streams path (XADD), not the pub-sub path. Subscribers relying on `event.tags` get empty sets for cross-node events.

**B5. `CombinationFilter.checker` raises instead of filtering**

Line 164 in `types.py`: when the checker returns `False`, it raises `ValueError` instead of returning `False` from `matches()`. This makes the checker unusable as a filter — it crashes the dispatch loop instead of silently skipping.

**B6. `VCMAnalysisCapability` and `AgentPoolCapability` share colony scope with overlapping key prefixes**

Both use colony-level scope. Both can produce `result:*` and `state:*` keys. A wildcard subscription on `result:*` will match entries from both capabilities.

### C. Inefficiency — things that work but waste resources

**C1. Colony-scoped capabilities default to `"*"` event subscription**

`AgentPoolCapability` and `WorkingSetCapability` inherit the base `stream_events_to_queue(pattern="*")`. In a colony with many agents, every write to the colony-level scope (from any capability using that scope) goes into these capabilities' event queues. This is unnecessary noise.

**C2. `MemoryLifecycleHooks` subscribes to the wrong scope**

Its `stream_events_to_queue()` subscribes to the agent-local scope `{agent_scope}:memory_lifecycle`. Lifecycle events are emitted to `{colony_scope}:control_plane:lifecycle`. The capability's subscription sees nothing. (The actual monitoring works only because `AgentMemoryRecycler` independently subscribes to the control plane scope.)

### D. Fragility — things that work but are easy to break

**D1. `ScopeUtils.format_key` is silently order-dependent on kwarg names**

`format_key(agent_id=X, result=Y)` → `agent_id:X:result:Y`. A reader using `pattern_key(result=None, agent_id=None)` → `agent_id:*:result:*` — works (alphabetical sort is deterministic). But `format_key(agent=X)` (typo, missing `_id`) produces `agent:X` which no reader will match. No validation, no error — just silent mismatch.

**D2. No shared vocabulary between `AgentHandle.run()` and capabilities**

`AgentHandle.run()` uses hand-written f-strings: `f"{child_id}:request:{req_id}"`. Capabilities use `ScopeUtils.format_key(sender=..., request_type=..., request_id=...)`. These are two independent key construction paths with no shared definition of what a "request" or "result" key looks like.

**D3. Event handlers use inconsistent pattern styles**

Some use `{scope_id}:request:*` (bakes scope into key pattern). Others use `request:*` (raw key). Whether `scope_id` prefix is needed depends on whether the key was written with that prefix — which depends on the writer's code, not the handler's code.

---

## Proposed Changes

### Change 1: Per-capability protocols for request/result exchange

The root cause of A1, A2, and D2 is that `AgentHandle.run()` and `AgentCapability.send_request()` are two independent protocols with no shared contract. The fix is not a single global `RequestProtocol` — different capabilities need different protocols. A capability may even need multiple protocols (e.g., `AgentPoolCapability` has one protocol for work assignment and another for work cancellation).

**Define a `BlackboardProtocol` base class** for any structured communication pattern on the blackboard:

```python
# agents/blackboard/protocol.py

class BlackboardProtocol:
    """Base class for blackboard communication protocols.

    A protocol defines the key format and scope level for a specific type
    of structured communication. This is not limited to request/result —
    it covers any pattern where writers and readers must agree on key format:

    - Request/result: bidirectional, correlated by request_id
    - Streaming events: unidirectional stream correlated by request_id
    - State publication: unidirectional, no request_id (game state, working set)
    - Signal/notification: fire-and-forget (lifecycle events)
    - Command: one-way instruction, no result expected

    Subclass this per interaction type — not per capability. Multiple
    capabilities can share the same protocol. One capability can support
    multiple protocols for different actions.

    Built on top of ScopeUtils — protocols use ScopeUtils to compute
    scope IDs and format_key to construct keys within those scopes.
    All keys produced are scope-relative (no scope_id prefix).
    """

    # The scope level this protocol operates at.
    scope: BlackboardScope = BlackboardScope.AGENT

    @classmethod
    def key(cls, **context) -> str:
        """Construct a specific key for writing.

        Args:
            **context: All fields needed to construct the key. What fields
                       are required depends on the protocol and scope level.
                       Agent-scoped protocols may need only request_id.
                       Colony-scoped protocols may also need agent_id.
        """
        raise NotImplementedError

    @classmethod
    def pattern(cls, **context) -> str:
        """Construct a glob pattern for subscribing/querying.

        Args:
            **context: Fields to match on. None values become '*' wildcards.
                       Omitted fields are wildcarded.
        """
        raise NotImplementedError
```

Concrete protocols subclass this for each communication pattern. A protocol can define multiple key types (request, result, event, state) as classmethods:

**Example: agent-level run protocol** (request/result + streaming events):

```python
class AgentRunProtocol(BlackboardProtocol):
    """Protocol for AgentHandle.run() <-> child agent communication.

    Operates at agent-level scope. No agent_id in keys (the scope IS
    the agent's mailbox). request_id provides uniqueness.

    Key types:
    - request:{request_id} — parent writes request
    - result:{request_id} — child writes result
    - event:{request_id}:{event_name} — child streams incremental events
    """
    scope = BlackboardScope.AGENT

    # --- Key construction ---

    @classmethod
    def request_key(cls, request_id: str) -> str:
        return f"request:run:{request_id}"

    @classmethod
    def result_key(cls, request_id: str) -> str:
        return f"result:run:{request_id}"

    @classmethod
    def event_key(cls, request_id: str, event_name: str) -> str:
        return f"event:run:{request_id}:{event_name}"

    # --- Subscription patterns ---

    @classmethod
    def request_pattern(cls) -> str:
        return "request:run:*"

    @classmethod
    def result_pattern(cls) -> str:
        return "result:run:*"

    @classmethod
    def event_pattern(cls, request_id: str) -> str:
        return f"event:run:{request_id}:*"

    # --- Base class interface ---

    @classmethod
    def key(cls, *, key_type: str = "request", request_id: str, event_name: str | None = None, **context) -> str:
        if key_type == "request":
            return cls.request_key(request_id)
        elif key_type == "result":
            return cls.result_key(request_id)
        elif key_type == "event":
            return cls.event_key(request_id, event_name)
        raise ValueError(f"Unknown key_type: {key_type}")

    @classmethod
    def pattern(cls, *, key_type: str = "request", request_id: str | None = None, **context) -> str:
        if key_type == "request":
            return cls.request_pattern()
        elif key_type == "result":
            return cls.result_pattern()
        elif key_type == "event" and request_id:
            return cls.event_pattern(request_id)
        raise ValueError(f"Unknown key_type: {key_type}")
```

**Example: colony-level work assignment protocol** (command + result):

```python
class WorkAssignmentProtocol(BlackboardProtocol):
    """Protocol for coordinator <-> worker communication.

    Operates at colony-level scope. agent_id IS needed in keys because
    multiple workers share the scope.
    """
    scope = BlackboardScope.COLONY

    @classmethod
    def assignment_key(cls, agent_id: str, request_id: str) -> str:
        return ScopeUtils.format_key(agent_id=agent_id, work_assignment=request_id)

    @classmethod
    def result_key(cls, agent_id: str, result_type: str = "final") -> str:
        return ScopeUtils.format_key(agent_id=agent_id, result_type=result_type)

    @classmethod
    def assignment_pattern(cls, agent_id: str | None = None) -> str:
        return ScopeUtils.pattern_key(agent_id=agent_id, work_assignment=None)

    @classmethod
    def result_pattern(cls, agent_id: str | None = None) -> str:
        return ScopeUtils.pattern_key(agent_id=agent_id, result_type=None)
```

**Example: state publication protocol** (no request_id, singleton state):

```python
class WorkingSetStateProtocol(BlackboardProtocol):
    """Protocol for publishing/observing VCM working set state.

    Colony-scoped. Single shared state key — any agent can read,
    only WorkingSetCapability writes.
    """
    scope = BlackboardScope.COLONY

    STATE_KEY = "state:working_set:cluster"
    PAGE_STATUS_KEY = "state:working_set:page_status"

    @classmethod
    def state_pattern(cls) -> str:
        return "state:working_set:*"
```

**Example: signal/notification protocol** (fire-and-forget):

```python
class LifecycleSignalProtocol(BlackboardProtocol):
    """Protocol for agent lifecycle signals on colony control plane.

    Colony-scoped. agent_id in key prevents overwrites when multiple
    agents terminate concurrently.
    """
    scope = BlackboardScope.COLONY

    @classmethod
    def created_key(cls, agent_id: str) -> str:
        return ScopeUtils.format_key(scope="agent_created", agent_id=agent_id)

    @classmethod
    def terminated_key(cls, agent_id: str) -> str:
        return ScopeUtils.format_key(scope="agent_terminated", agent_id=agent_id)

    @classmethod
    def created_pattern(cls) -> str:
        return ScopeUtils.pattern_key(scope="agent_created", agent_id=None)

    @classmethod
    def terminated_pattern(cls) -> str:
        return ScopeUtils.pattern_key(scope="agent_terminated", agent_id=None)
```

**`AgentHandle.run()` takes a protocol parameter**:

```python
class AgentHandle:
    async def run(
        self,
        input_data: dict,
        *,
        protocol: type[BlackboardProtocol] = AgentRunProtocol,
        ...
    ) -> AgentRun:
        request_key = protocol.request_key(request_id)
        result_key = protocol.result_key(request_id)
        ...

    async def run_streamed(
        self,
        input_data: dict,
        *,
        protocol: type[BlackboardProtocol] = AgentRunProtocol,
        ...
    ) -> AsyncIterator[AgentRunEvent]:
        request_key = protocol.request_key(request_id)
        event_pattern = protocol.event_pattern(request_id)
        ...
```

**Capabilities declare which protocols they listen for** (ties into Change 2's `input_patterns`):

```python
class GroundingCapability(AgentCapability):
    protocols = [AgentRunProtocol]

    @property
    def input_patterns(self) -> list[str]:
        return [p.request_pattern() for p in self.protocols]

class AgentPoolCapability(AgentCapability):
    protocols = [WorkAssignmentProtocol]

    @property
    def input_patterns(self) -> list[str]:
        return [WorkAssignmentProtocol.result_pattern()]

class MemoryLifecycleHooks(AgentCapability):
    protocols = [LifecycleSignalProtocol]

    @property
    def input_patterns(self) -> list[str]:
        return [
            LifecycleSignalProtocol.created_pattern(),
            LifecycleSignalProtocol.terminated_pattern(),
        ]
```

**`CapabilityResultFuture` uses the protocol** to construct exact result keys (fixes B1):

```python
def get_result_future(self, protocol: type[BlackboardProtocol], request_id: str) -> CapabilityResultFuture:
    result_key = protocol.result_key(request_id)  # exact key, no wildcards
    return CapabilityResultFuture(result_key=result_key, blackboard=self.get_blackboard())
```

**Key design points**:
- `BlackboardProtocol` is not limited to request/result. It covers any communication pattern: state publication, signals, commands, streaming events.
- Protocols are per-interaction-type, not per-capability. Multiple capabilities can share `AgentRunProtocol`. One capability can support multiple protocols for different actions.
- Protocols know their scope level (`BlackboardScope.AGENT` vs `COLONY`) and what context fields are needed in keys (`agent_id` for colony-scoped, nothing for agent-scoped).
- Protocols are built on `ScopeUtils.format_key`/`pattern_key` — they don't replace it.
- All keys are scope-relative. No protocol ever includes `scope_id` in its keys.

### Change 2: Capabilities declare `input_patterns`

Add a class variable to `AgentCapability`:

```python
class AgentCapability:
    # Subclasses override these to declare their key contracts
    input_patterns: ClassVar[list[str]] = []   # glob patterns this capability monitors
```

Subclasses override:

```python
class GroundingCapability(AgentCapability):
    input_patterns = [RequestProtocol.REQUEST_PATTERN]  # "request:*"

class AgentPoolCapability(AgentCapability):
    input_patterns = [RequestProtocol.RESULT_PATTERN]   # "result:*"

class WorkingSetCapability(AgentCapability):
    input_patterns = ["state:working_set:*"]
```

Change the default `stream_events_to_queue()`:

```python
async def stream_events_to_queue(self, event_queue, ...) -> asyncio.Queue:
    blackboard = await self.get_blackboard()
    patterns = self.input_patterns or ["*"]  # fallback for legacy capabilities
    for pattern in patterns:
        blackboard.stream_events_to_queue(event_queue, pattern=pattern, event_types={"write"})
    return event_queue
```

**Impact**: Colony-scoped capabilities stop receiving every event in the colony. `AgentPoolCapability` only sees `result:*`, `WorkingSetCapability` only sees `state:working_set:*`.

### Change 3: Add `TagFilter` to the event system

Add a built-in tag filter:

```python
# blackboard/types.py

class TagFilter(EventFilter):
    """Filter events by tags."""
    def __init__(self, required_tags: set[str]):
        self.required_tags = required_tags

    def matches(self, event: BlackboardEvent) -> bool:
        return self.required_tags.issubset(event.tags)
```

And fix `CombinationFilter.checker` to return `False` instead of raising.

And fix the tags-lost-through-pub-sub bug: include `"tags": list(event.tags)` in `event_dict` in `EventBus.emit()`.

**Why**: This doesn't replace key-based filtering (which remains the primary mechanism). But it enables capabilities that share a scope to differentiate events by tag without encoding everything in the key. It also makes the event system consistent with the query system.

### Change 4: Fix concrete bugs

| Bug | Fix |
|---|---|
| A3: `CriticCapability._send_critique_request` | Replace `self.agent.get_agent(scope_id=...)` with `self.agent.get_blackboard(scope_id=...)` |
| A4: `_setup_resumption_trigger` | Delete it. Child completion detection should go through `AgentPoolCapability`'s `result:*` subscription or through `AgentHandle.run()`'s built-in result waiting |
| B1: `CapabilityResultFuture` wildcard | Override `ConsistencyCapability.get_result_future()` to use exact key `result:{request_id}` instead of `consistency_check_result:*` |
| B2: `ResultCapability.results:index` race | Wrap in `blackboard.transaction()` with version-based CAS |
| B4: Tags lost through pub-sub | Add `"tags": list(event.tags)` to `event_dict` in `EventBus.emit()` |
| B5: `CombinationFilter.checker` raises | Return `False` instead of raising `ValueError` |
| B6: Scope collision | Add a namespace prefix to `VCMAnalysisCapability` keys: `vcm_analysis:result:{page_id}` vs `pool:result:{agent_id}` |
| C2: `MemoryLifecycleHooks` wrong scope | Override `stream_events_to_queue` to subscribe to `MemoryScope.colony_control_plane("lifecycle")` |
| A5: `scope_id` in patterns/queries | Remove `{scope_id}` prefix from all `@event_handler` patterns, `query(namespace=...)` calls, `stream_events_to_queue(pattern=...)` calls, and keys passed to `read`/`write`. Keys are scope-relative — the scope is already the partition. Deprecate the `{scope_id}` template variable in `@event_handler`. Specific files: `blackboard_page_source.py:780,818`, `memory/capability.py:539`, `memory/backends/blackboard.py:132`, `vcm_analysis.py:781,859,895,921`, `consistency.py:185,350`, `critique.py:428`, all sample capability `@event_handler` patterns |

### Change 5: Fix `run_streamed()` event protocol

`run_streamed()` listens for `{child_id}:event:{req_id}:*` and expects events ending in `:complete`, `:result`, `:error`. No existing capability writes this format. Migrate to `RequestProtocol.event_key(request_id, event_name)` and document the streaming event contract for capabilities that want to support `run_streamed()`.

---

## What this does NOT change

- **Scope isolation model**: `scope_id` still determines which partition. `ScopeUtils.get_*_level_scope()` and `MemoryScope` remain the way to compute scope IDs.
- **`ScopeUtils.format_key` for non-request/result keys**: Capabilities that use `format_key` for their own internal state keys (game state, hypothesis tracking, working set) continue to do so. Only the request/result protocol is standardized.
- **`EnhancedBlackboard` API**: `read`, `write`, `delete`, `query`, `list_keys`, `stream_events_to_queue` — all unchanged.
- **Event delivery model**: Fan-out to all matching subscribers. `EventBus` internals unchanged.
- **Pydantic `BaseModel` validation on payloads**: Already enforced. Not changed.
- **`@event_handler` decorator**: Continues to match on key patterns. The `{scope_id}` template variable in `_resolve_pattern()` is removed — it produces patterns that don't match scope-relative keys. All patterns should be scope-relative.

---

## Implementation order

### Phase 1: Foundation (no behavior change)
1. Create `agents/blackboard/protocol.py` with `BlackboardProtocol` base class and `AgentRunProtocol`.
2. Add `input_patterns: ClassVar[list[str]] = []` and `protocols: ClassVar[list[type[BlackboardProtocol]]] = []` to `AgentCapability`.
3. Add `TagFilter` to `types.py`.

### Phase 2: Bug fixes (immediate value)
4. Fix A3 (`CriticCapability`), A4 (`_setup_resumption_trigger`), B4 (tags in pub-sub), B5 (`CombinationFilter.checker`), C2 (`MemoryLifecycleHooks`).
5. Fix B1 (`CapabilityResultFuture` exact keys).
6. Fix B2 (`ResultCapability` CAS), B6 (scope collision prefixes).

### Phase 3: Remove `scope_id` from keys and patterns (A5)
7. Audit and fix all `@event_handler(pattern="{scope_id}:...")` — replace with scope-relative patterns.
8. Audit and fix all `query(namespace=f"{self.scope_id}:...")`, `stream_events_to_queue(pattern=f"{self.scope_id}:...")`, and `read`/`write` calls that prefix `scope_id` into keys.
9. Remove the `{scope_id}` template variable from `@event_handler` `_resolve_pattern()`. It produces patterns that contradict scope-relative key storage.

### Phase 4: Protocol unification
10. Create concrete protocol classes: `AgentRunProtocol`, `WorkAssignmentProtocol`, `CritiqueProtocol`, etc.
11. Migrate `AgentHandle.run()` and `run_streamed()` to use `protocol` parameter with `AgentRunProtocol` default.
12. Migrate `AgentCapability.send_request()` and `get_result_future()` to use protocols.
13. Migrate sample capabilities to declare `protocols` and use protocol-derived `input_patterns`.
14. Populate `input_patterns` on all capabilities. Update `stream_events_to_queue` default.

### Phase 5: Validation (optional, deferred)
15. Add a registration-time check: when a capability is attached to an agent, warn if `input_patterns` is empty (legacy) or if two capabilities on the same agent have overlapping `input_patterns` on the same scope.
