# Hook System

Colony uses aspect-oriented programming (AOP) to handle cross-cutting concerns -- token tracking, rate limiting, memory capture, checkpointing, and retry logic -- without polluting core agent logic. The hook system is implemented in `polymathera.colony.agents.patterns.hooks`.

## Core Concepts

### @hookable Decorator

The `hookable` decorator (in `polymathera.colony.agents.patterns.hooks.decorator`) marks a method as an interception point. When a hookable method is called, it checks the owning agent's hook registry for matching hooks and executes them in the appropriate order.

```python
from polymathera.colony.agents.patterns.hooks.decorator import hookable

class MyCapability(AgentCapability):
    @hookable
    async def analyze(self, data: dict) -> AnalysisResult:
        """This method can be intercepted by hooks."""
        ...
```

### Hook Types

Defined in `polymathera.colony.agents.patterns.hooks.types.HookType`:

| Type | Execution | Use Case |
|------|-----------|----------|
| `BEFORE` | Before the method. Can modify args/kwargs. | Input validation, rate limiting, context injection |
| `AFTER` | After the method. Receives the return value. | Logging, metric collection, memory capture |
| `AROUND` | Wraps the method. Controls whether it executes. | Caching, retry logic, circuit breaking |

### HookContext

Every hook handler receives a `HookContext` (in `polymathera.colony.agents.patterns.hooks.types`):

```python
@dataclass
class HookContext:
    join_point: str      # e.g., "MyCapability.analyze"
    instance: Any        # The object whose method was called
    args: tuple          # Positional arguments
    kwargs: dict         # Keyword arguments
    agent: Agent | None  # Owning agent (if available)
```

## Pointcut Expressions

`Pointcut` (in `polymathera.colony.agents.patterns.hooks.pointcuts`) determines which method invocations a hook intercepts. Pointcuts match against both the join point string (e.g., `"MyCapability.analyze"`) and the actual instance.

### Pattern Matching

```python
# Match a specific method
Pointcut.pattern("ActionDispatcher.dispatch")

# Match all methods on a class
Pointcut.pattern("MyCapability.*")

# Match a method across all classes
Pointcut.pattern("*.analyze")
```

### Combinators

Pointcuts compose with logical operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `&` | AND | `Pointcut.pattern("*.dispatch") & Pointcut.class_filter(ActionDispatcher)` |
| `\|` | OR | `Pointcut.pattern("*.analyze") \| Pointcut.pattern("*.synthesize")` |
| `~` | NOT | `~Pointcut.pattern("*.internal_*")` |

## AgentHookRegistry

Each agent has its own `AgentHookRegistry` (in `polymathera.colony.agents.patterns.hooks.registry`). Hooks registered on one agent do not affect other agents.

```python
class AgentHookRegistry:
    """Per-agent registry for hooks.

    Each agent has its own hook registry. Hooks registered on an agent
    apply to all components of that agent (capabilities, policies, etc.)
    but not to other agents.
    """
```

### Hook Registration

Hooks are registered with a pointcut, type, handler function, and optional priority:

```python
registry.register(
    pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
    hook_type=HookType.AFTER,
    handler=my_tracking_handler,
    priority=100,  # Higher = runs later
)
```

### Auto-Registration

The `auto_register_hooks` function scans a capability for declared hooks and registers them with the agent's registry. This is called during capability initialization.

## RegisteredHook

A `RegisteredHook` (in `polymathera.colony.agents.patterns.hooks.types`) bundles the hook configuration:

- `hook_id`: Unique identifier
- `pointcut`: Which methods to intercept
- `hook_type`: BEFORE, AFTER, or AROUND
- `handler`: The async callable
- `priority`: Execution order (lower runs first)
- `error_mode`: How to handle exceptions (`ErrorMode`)

## Error Handling

`ErrorMode` controls hook failure behavior:

- **Propagate**: Exception from the hook propagates to the caller
- **Suppress**: Exception is logged but swallowed; execution continues
- **Fallback**: A fallback value is returned if the hook fails

## Use Cases

### Token Tracking

An AFTER hook on inference methods tracks token consumption without modifying any inference code:

```python
@after(Pointcut.pattern("*.submit_inference"))
async def track_tokens(ctx: HookContext, result: InferenceResponse):
    agent = ctx.agent
    usage = result.usage
    await agent.update_resource_usage(tokens=usage.total_tokens)
```

### Rate Limiting

A BEFORE hook throttles inference requests:

```python
@before(Pointcut.pattern("*.submit_inference"))
async def rate_limit(ctx: HookContext):
    if ctx.agent.requests_this_minute > MAX_REQUESTS:
        await asyncio.sleep(backoff_duration)
```

### Memory Capture

The memory system uses AFTER hooks via `MemoryProducerConfig` to observe agent behavior and automatically store memories. See [Memory System](memory-system.md) for details.

### Checkpointing

An AFTER hook on plan execution saves state for recovery:

```python
@after(Pointcut.pattern("CacheAwareActionPolicy.execute_iteration"))
async def checkpoint(ctx: HookContext, result: ActionPolicyIterationResult):
    await ctx.agent.serialize_suspension_state()
```

## Design Rationale

The hook system exists because many concerns cut across the agent/capability/policy hierarchy:

- Token tracking touches every inference call across all capabilities
- Rate limiting applies to all external API interactions
- Memory capture spans action execution, planning, reflection, and games
- Checkpointing applies at multiple granularities

Without hooks, each of these would require modifications in dozens of methods across the codebase. The AOP approach keeps core logic clean and cross-cutting concerns modular.

!!! tip "Capabilities as aspects"
    Each `AgentCapability` can declare hooks via `MemoryProducerConfig` or direct registration. The capability itself is an AOP aspect, and the `ActionPolicy` acts as the aspect weaver -- deciding which capabilities (and therefore which hooks) are active at any given time.
