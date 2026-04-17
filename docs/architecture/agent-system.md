# Agent System

The agent system defines how autonomous computational entities are created, managed, and coordinated. Every agent follows the same core lifecycle but can be specialized through capabilities and action policies.

## Guiding Principle

> Agent control flow and all decisions should be driven by a reasoning LLM given sufficient context, **not hardcoded logic**. But the actions available to the LLM planner are as important as the context it reasons over. The framework provides a rich ecosystem of capabilities that expose different actions and cognitive processes to the LLM, enabling emergent behavior from the combinatorial explosion of possible action interleavings.

The framework provides structure (lifecycle, capabilities, blackboard access), but the LLM decides what to do next. Meta-choices available to the LLM include delegating to another agent, meta-reasoning about its own strategy, initiating multi-agent deliberation, and building new tools.

## Agent Base Class

`polymathera.colony.agents.base.Agent` is the base class for all agents. It provides:

- **Lifecycle management**: State transitions through `AgentState`
- **Blackboard access**: Read/write shared state via `EnhancedBlackboard`
- **VCM access**: Read virtual context pages
- **Inference submission**: Submit requests to vLLM replicas
- **Capability management**: Add, remove, and query `AgentCapability` instances
- **Hook registry**: `HookRegistry` for AOP interception
- **Session tracking**: Every `run()` call creates an `AgentRun` tracked in the session

## Agent Lifecycle States

Defined in `polymathera.colony.agents.models.AgentState`:

```
INITIALIZED --> RUNNING --> IDLE --> STOPPED
                  |                    ^
                  v                    |
               WAITING ----------------
                  |
                  v
              SUSPENDED --> RUNNING (resumed)
                  |
                  v
               FAILED
```

| State | Description |
|-------|-------------|
| `INITIALIZED` | Agent created, not yet started |
| `RUNNING` | Actively executing an action policy iteration |
| `WAITING` | Blocked on input, tool result, or child agent |
| `LOADED` | VCM-bound agent with pages resident in cache |
| `UNLOADED` | VCM-bound agent with pages evicted from cache |
| `IDLE` | Between tasks, not actively executing |
| `SUSPENDED` | Resources freed, state serialized, can be resumed |
| `STOPPED` | Gracefully terminated |
| `FAILED` | Terminated due to error |

## Agent Types

### VCM-Bound Agents
Loaded and unloaded together with their assigned VCM pages. When their pages are evicted from cache, these agents are suspended. When pages are reloaded, agents resume from their serialized state.

### Unbound Agents
Dynamically select which VCM pages to load. They are not tied to specific pages and can request page loads as needed during execution.

### Service Agents
Always-running agents that provide services to other agents (e.g., a `MemoryManagementAgent` that consolidates memories across the system).

### Supervisor Agents
Independent of page state. They monitor and coordinate other agents, make delegation decisions, and handle escalation.

## `AgentCapability`

`polymathera.colony.agents.base.AgentCapability` is the extension point for agent functionality. Each capability encapsulates a specific aspect or protocol:

```python
class AgentCapability(ABC):
    """Base class for capabilities.

    Operates in four modes:
    1. Local: runs within the owning agent
    2. Remote: parent communicates with child via shared scope
    3. Shared scope: multiple agents share a namespace
    4. Detached: standalone without agent context
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        blackboard: EnhancedBlackboard | None = None,
        capability_key: str | None = None,
    ): ...
```

The four modes in practice:

```python
# 1. Local mode — capability runs within its owning agent
memory = MemoryCapability(agent=self)  # scope_id defaults to agent.agent_id

# 2. Remote mode — parent monitors a child agent's progress
child_cap = ResultCapability(agent=parent, scope_id=child_agent_id)
await child_cap.stream_events_to_queue(self.get_event_queue())
result = await asyncio.wait_for(child_cap.get_result_future(), timeout=30.0)

# 3. Shared scope — all game participants see each other's events
game_cap = NegotiationGameProtocol(agent=self, scope_id=game_id)

# 4. Detached mode — external system interacts with agents via blackboard
external_cap = MyCapability(agent=None, scope_id=target_agent_id)
```

Capabilities provide:

- **Action executors**: Methods decorated with `@action_executor` that the `ActionPolicy` can invoke (conscious cognitive processes)
- **Hookables**: Methods marked `@hookable` that other components can intercept
- **Hooks** (*method call interceptors*): Hooks the capability registers on other components (via `hook_handler()`)
- **Event streams**: Capabilities can publish events to their scoped blackboard, which other agents can subscribe to (e.g., for game protocols or parent-child communication)
- **Event handlers**: Capabilities can subscribe to events from other capabilities or agents, enabling reactive behavior and emergent coordination patterns
- **Services** or **background processes**: Subconscious *cognitive* processes (consolidation, rehearsal)

```python
class PageGraphCapability(AgentCapability):
    """Example capability with action executors and hooks."""

    @action_executor()
    async def traverse(
        self, start_pages: list[str], strategy: str = "bfs", max_depth: int = 5
    ) -> dict[str, Any]:
        """Auto-discovered by ActionPolicy — the LLM planner can invoke this."""
        graph = await self._get_page_graph()
        ...

    @action_executor(exclude_from_planning=True)
    async def update_edge(self, source: str, target: str, weight: float) -> None:
        """exclude_from_planning=True: only invoked programmatically, not by the LLM."""
        ...
```

!!! tip "Capabilities as AOP Aspects"
    Each `AgentCapability` is an "aspect" in the aspect-oriented programming sense. The `ActionPolicy` plays the role of the "aspect weaver," deciding which capabilities to activate and in what order. Emergent behavior arises from the combinatorial explosion of possible capability interleavings -- the framework does not model all paths explicitly.


!!! bug "Merge with `docs/design-insights/capabilities-as-aspects.md` and reference here."


!!! tip "LLM-Decidable Actions"
    The LLM should be able to reason about when/how to use each `AgentCapability` action.


!!! bug "Add subconscious processes to planning prompt"
    Subconscious processes (e.g., memory consolidation) are not directly invoked by the LLM, but they affect the agent's state and capabilities. The LLM should be aware of these processes and their effects when making decisions. Add explanation of how to include subconscious processes in the planning prompt and how they interact with conscious actions.



### Scope-Based Communication

Capabilities use `scope_id` for flexible communication patterns:

- `agent.agent_id` (default): Agent-local scope
- `child_agent_id`: Parent-to-child communication
- `game_id` or `task_id`: Shared scope for group coordination

The `publish()` method writes records to the capability's scoped blackboard. If the scope is VCM-mapped, writes are automatically discoverable by other agents via the VCM.

```python
# publish() replaces manual blackboard key construction:
#   key = f"{self.scope_id}:analysis:result:{result.result_id}"
#   await bb.write(key=key, value=result.model_dump(), created_by=...)
# With:
await self.publish(result, tags={"analysis"})
# Key is resolved automatically via BlackboardPublishable protocol on the record.
```

## `ActionPolicy`

`polymathera.colony.agents.base.ActionPolicy` is the abstract base for decision-making. It receives execution state and produces iteration results:

```python
class ActionPolicy(ABC):
    async def execute_iteration(
        self, state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        ...
```

The policy manages which capabilities are active via `use_agent_capabilities()` and `disable_agent_capabilities()`. See [Action Policies](action-policies.md) for the full planning architecture.

The agent's main execution loop calls `run_step()` repeatedly. Each step invokes the action policy, which gathers context from capabilities, asks the LLM what to do next, and dispatches the chosen action:

```python
@tracing(publish_key=lambda self: self.agent_id)
class Agent:
    @hookable
    async def run_step(self) -> None:
        """Execute one iteration of the agent's reasoning loop.

        @hookable: capabilities can register BEFORE/AFTER/AROUND hooks
        to prepare context, post-process results, or wrap execution.

        Uses repeated run_step() instead of a single long-running run()
        to facilitate suspension and state management across replicas.
        """
        if self.state not in (AgentState.RUNNING, AgentState.IDLE):
            return
        result = await self.action_policy.execute_iteration(self._build_state())
        await self._apply_iteration_result(result)

    def add_capability(self, capability: AgentCapability, *,
                       include_actions: list[str] | None = None,
                       exclude_actions: list[str] | None = None) -> None:
        """Add a capability. Must still call action_policy.use_agent_capabilities()
        to include it in planning."""

    def get_capability(self, name: str) -> AgentCapability | None:
        """Retrieve a capability by name."""
```

## `AgentHandle` — Parent-Child Communication

`AgentHandle` is the interface for sending work to an agent and receiving results. It works in both **owned mode** (parent agent communicating with a child) and **detached mode** (external code like CLI communicating with an agent).

### `run()` — Request/Result

```python
handle = AgentHandle(child_agent_id="agent-xyz", owner=parent_agent)

# namespace scopes the blackboard partition (baked into scope_id)
# protocol defines the key format (defaults to AgentRunProtocol)
run = await handle.run(
    {"query": "analyze this code"},
    protocol=AgentRunProtocol,  # default — defines key format
    namespace="compliance",     # scopes the blackboard partition
    timeout=60,
)
print(run.output_data)
```

The `namespace` parameter is **required** — it identifies which capability on the child agent should handle the request. This prevents interference when an agent has multiple capabilities using the same protocol. The namespace is appended to the `scope_id`, giving each capability its own blackboard partition (e.g., `...colony:C:namespace:compliance`).

The `protocol` parameter (default `AgentRunProtocol`) defines the key format for request/result exchange within that partition:

- Parent writes to scope `...colony:C:namespace:compliance` with key `request:run:{request_id}`
- Child's `ComplianceCapability` has the same scope (set in its `__init__` via the same namespace)
- Child detects requests via `@event_handler(pattern=AgentRunProtocol.request_pattern())`
- Child writes result with key `result:run:{request_id}` on the same scope
- Parent listens for `result:run:{request_id}` on the same scope

The blackboard `scope` parameter (default `BlackboardScope.AGENT`) determines the scope level. `AGENT`-scoped protocols write to the child's agent-level blackboard. `COLONY`-scoped capabilities pass `scope=BlackboardScope.COLONY` to write to the shared colony blackboard.

See [Blackboard Protocols](blackboard.md#communication-protocols) for the full protocol design.

### `run_streamed()` — Streaming Events

```python
async for event in handle.run_streamed(
    {"query": "analyze"},
    protocol=AgentRunProtocol,
    namespace="compliance",
    timeout=300,
):
    print(f"{event.event_type}: {event.data}")
    if event.event_type == "completed":
        break
```

Uses `AgentRunProtocol.event_pattern(request_id)` to subscribe to incremental events on the namespace-scoped blackboard. The child emits events via `AgentRunProtocol.event_key(request_id, event_name)`.

### Custom Protocols

For specialized communication patterns, pass a different protocol:

```python
from polymathera.colony.agents.blackboard.protocol import WorkAssignmentProtocol
from polymathera.colony.agents.scopes import BlackboardScope

# Use a colony-scoped protocol for coordinator→worker communication
run = await handle.run(
    work_unit,
    protocol=WorkAssignmentProtocol,
    scope=BlackboardScope.COLONY,
    namespace="pool",
)
```

## Suspension and Resumption

When an agent cannot proceed — blocked on children, out of resources, or by explicit request — it is **suspended**. Suspension persists the agent's full state (plan progress, working set, page access patterns) to Redis and frees its resources. The agent is deleted from the replica.

Resumption is handled by `AgentSystemDeployment._resource_monitor_loop()`, which evaluates structured `ResumptionCondition`s:

| Condition | When met | Example |
|-----------|----------|---------|
| `CHILDREN_COMPLETED` | All blocking child agents reach STOPPED/FAILED | Agent waiting for workers to finish |
| `RESOURCE_AVAILABLE` | Resources freed (checked by attempting spawn) | Agent evicted for capacity |
| `PAGES_AVAILABLE` | Required VCM pages loaded on some replica | Cache-aware resumption (future) |
| `IMMEDIATE` | Always — resume ASAP | Explicit suspension lifted |
| `CUSTOM` | Never (requires external trigger) | Application-specific logic |

Action executors signal blocking with structured metadata:

```python
return ActionResult(
    success=False,
    blocked=True,
    blocked_reason="Waiting for worker agents",
    blocking_agent_ids=["agent-abc", "agent-def"],  # structured, not text
)
```

The system constructs a `CHILDREN_COMPLETED` condition from `blocking_agent_ids` and stores it in the suspension state. The monitor loop checks `AgentSystemState` for child completion before attempting resume.

## `ResourceExhausted` Handling

When a replica has insufficient resources for a new agent, the system follows an ordered strategy:

1. **Hard page affinity**: Only place on replicas with required pages
2. **Soft affinity**: Try replicas with preferred pages, fall back to others
3. **Retry later**: Queue the agent for later placement
4. **Suspend existing agents**: Free resources by suspending lower-priority agents

The `ResourceExhausted` exception (in `polymathera.colony.agents.base`) triggers this cascade.

## Agent Actions Taxonomy

Actions are typed via `polymathera.colony.agents.models.ActionType`:

| Category | Actions |
|----------|---------|
| **Planning** | `PLAN_CREATE`, `PLAN_REVISE`, `PLAN_BACKTRACK` |
| **Reasoning** | `ANALYZE`, `HYPOTHESIS_TEST`, `DECISION_MAKE` |
| **Tools** | `TOOL_DISCOVER`, `TOOL_USE`, `TOOL_FIX`, `TOOL_CREATE` |
| **Context** | `CONTEXT_FETCH`, `CONTEXT_COMPACT`, `CONTEXT_SUMMARIZE` |
| **Communication** | `MESSAGE_SEND`, `MESSAGE_RECEIVE`, `NEGOTIATE` |
| **Memory** | `MEMORY_SEARCH`, `MEMORY_READ`, `MEMORY_WRITE` |
| **Orchestration** | `AGENT_SPAWN`, `AGENT_TERMINATE`, `AGENT_MONITOR` |
| **Output** | `OUTPUT_WRITE`, `OUTPUT_FORMAT`, `REPORT_GENERATE` |

Long-running actions should be idempotent, pausable, resumable, checkpointable, and cancellable. Complex actions are best implemented as separate agents managed by the `AgentSystemDeployment`.

## Blueprints

!!! bug "This section is incomplete and needs expansion"
    Add explanation of how blueprints work for serializable agent specifications, validation, and instantiation with dependency injection.


Agent configuration uses the blueprint pattern for serializable, deployable specifications:

- `AgentBlueprint`: Full agent specification (capabilities, policy, resources)
- `AgentCapabilityBlueprint`: Capability class + constructor kwargs
- `ActionPolicyBlueprint`: Policy class + constructor kwargs (created via `ActionPolicy.bind()`)

Blueprints are validated for serializability at creation time and instantiated at deployment time with injected dependencies (agent reference, blackboard, etc.).
