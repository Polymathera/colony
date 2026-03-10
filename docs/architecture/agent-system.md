# Agent System

The agent system defines how autonomous computational entities are created, managed, and coordinated. Every agent follows the same core lifecycle but can be specialized through capabilities and action policies.

## Guiding Principle

> Agent control flow and all decisions should be driven by a reasoning LLM given sufficient context, **not hardcoded logic**.

The framework provides structure (lifecycle, capabilities, blackboard access), but the LLM decides what to do next. Meta-choices available to the LLM include delegating to another agent, meta-reasoning about its own strategy, initiating multi-agent deliberation, and building new tools.

## Agent Base Class

`polymathera.colony.agents.base.Agent` is the base class for all agents. It provides:

- **Lifecycle management**: State transitions through `AgentState`
- **Blackboard access**: Read/write shared state via `EnhancedBlackboard`
- **VCM access**: Read virtual context pages
- **Inference submission**: Submit requests to vLLM replicas
- **Capability management**: Add, remove, and query `AgentCapability` instances
- **Hook registry**: Per-agent `AgentHookRegistry` for AOP interception
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

## AgentCapability

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
```

Capabilities provide:

- **Action executors**: Methods the `ActionPolicy` can invoke (conscious cognitive processes)
- **Hookables**: Methods marked `@hookable` that other components can intercept
- **Hooks**: Hooks the capability registers on other components
- **Background processes**: Subconscious cognitive processes (consolidation, rehearsal)

!!! tip "Capabilities as AOP Aspects"
    Each `AgentCapability` is an "aspect" in the aspect-oriented programming sense. The `ActionPolicy` plays the role of the "aspect weaver," deciding which capabilities to activate and in what order. Emergent behavior arises from the combinatorial explosion of possible capability interleavings -- the framework does not model all paths explicitly.

### Scope-Based Communication

Capabilities use `scope_id` for flexible communication patterns:

- `agent.agent_id` (default): Agent-local scope
- `child_agent_id`: Parent-to-child communication
- `game_id` or `task_id`: Shared scope for group coordination

The `publish()` method writes records to the capability's scoped blackboard. If the scope is VCM-mapped, writes are automatically discoverable by other agents via the VCM.

## ActionPolicy

`polymathera.colony.agents.base.ActionPolicy` is the abstract base for decision-making. It receives execution state and produces iteration results:

```python
class ActionPolicy(ABC):
    async def execute_iteration(
        self, state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        ...
```

The policy manages which capabilities are active via `use_agent_capabilities()` and `disable_agent_capabilities()`. See [Action Policies](action-policies.md) for the full planning architecture.

## ResourceExhausted Handling

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

Agent configuration uses the blueprint pattern for serializable, deployable specifications:

- `AgentBlueprint`: Full agent specification (capabilities, policy, resources)
- `AgentCapabilityBlueprint`: Capability class + constructor kwargs
- `ActionPolicyBlueprint`: Policy class + constructor kwargs (created via `ActionPolicy.bind()`)

Blueprints are validated for serializability at creation time and instantiated at deployment time with injected dependencies (agent reference, blackboard, etc.).
