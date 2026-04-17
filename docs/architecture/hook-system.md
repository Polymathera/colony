# Hook System

!!! bug "Aspect-Oriented Programming for Agents"
    Create a new article on aspect-oriented programming (AOP) principles in Colony (hook system and agent capabilities) to allow cross-cutting concerns like observability, memory capture, and rate limiting to be modularly implemented without polluting core agent logic.


Colony uses aspect-oriented programming (AOP) to handle cross-cutting concerns, most importantly, **observability** -- token tracking, rate limiting, agent memory capture, checkpointing, and retry logic -- without polluting core agent logic. The hook system is implemented in `polymathera.colony.distributed.hooks`.

## Core Concepts

### `@hookable` Decorator

The `hookable` decorator (in `polymathera.colony.distributed.hooks.decorator`) marks a method as an interception point. When a hookable method is called, it checks the owning agent's hook registry for matching hooks and executes them in the appropriate order.

```python
from polymathera.colony.distributed.hooks.decorator import hookable

@tracing(publish_key=lambda self: self.agent.agent_id)
class MyCapability(AgentCapability):
    @hookable
    async def analyze(self, data: dict) -> AnalysisResult:
        """This method can be intercepted by hooks."""
        ...
```

### Hook Types

Defined in `polymathera.colony.distributed.hooks.types.HookType`:

| Type | Execution | Use Case |
|------|-----------|----------|
| `BEFORE` | Before the method. Can modify args/kwargs. | Input validation, rate limiting, context injection |
| `AFTER` | After the method. Receives the return value. | Logging, metric collection, memory capture |
| `AROUND` | Wraps the method. Controls whether it executes. | Caching, retry logic, circuit breaking |

### `HookContext`

Every hook handler receives a `HookContext` (in `polymathera.colony.distributed.hooks.types`):

```python
@dataclass
class HookContext:
    join_point: str      # e.g., "MyCapability.analyze"
    instance: Any        # The object whose method was called
    args: tuple          # Positional arguments
    kwargs: dict         # Keyword arguments
    agent: Agent | None  # Owning agent (if available)
```

### Hook Registration

#### Declarative Registration

`AgentCapabilities` can declare **hooks** (*handlers*) using the `@hook_handler` decorator. These are auto-discovered and registered with the capability's parent agent during initialization. A hook declaration includes a pointcut, type, and optional priority:

```python
from polymathera.colony.distributed.hooks import hook_handler, tracing

@tracing(subscribe_key=lambda self: self.agent.agent_id)
class TokenTrackingCapability(AgentCapability):
    @hook_handler(
        pointcut=Pointcut.pattern("*.infer"),
        hook_type=HookType.AFTER,
        priority=100,
    )
    async def track_tokens(self, ctx: HookContext, result: Any) -> Any:
        usage = result.usage
        await ctx.agent.update_resource_usage(tokens=usage.total_tokens)
        return result
```

The `install_hook_handlers` function (called by `AgentCapability.initialize`) scans a capability for methods decorated with `@hook_handler` and registers them with the agent's registry.

Alternatively, an arbitrary **handler function** can be directly registered as a hook by calling an agent's `registry.register` method:

```python
registry.register(
    pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
    hook_type=HookType.AFTER,
    handler=my_tracking_handler,
    priority=100,  # Higher = runs later
)
```

#### `HookRegistry`

Each agent has its own `HookRegistry` (in `polymathera.colony.distributed.hooks.registry`). Hooks registered on one agent do not affect other agents.

```python
class HookRegistry:
    """Registry for hooks associated with one domain key.

    A domain is a group of hook handlers or listeners that can be
    assigned to any group of hookable methods (e.g., all capabilities
    of an agent) that they send notifications to.

    Each agent can have its own hook registry. Hook handlers registered on an agent
    apply to all components of that agent (capabilities, policies, etc.)
    but not to other agents.
    """
```


#### `RegisteredHook`

A `RegisteredHook` (in `polymathera.colony.distributed.hooks.types`) bundles the hook configuration:

- `hook_id`: Unique identifier
- `pointcut`: Which methods to intercept
- `hook_type`: `BEFORE`, `AFTER`, or `AROUND`
- `handler`: The async callable
- `priority`: Execution order (lower runs first)
- `error_mode`: How to handle exceptions (`ErrorMode`)



## Pointcut Expressions

`Pointcut` (in `polymathera.colony.distributed.hooks.pointcuts`) determines which method invocations a hook intercepts. Pointcuts match against both the join point string (e.g., `"MyCapability.analyze"`) and the actual instance.

#### Pattern Matching

```python
# Match a specific method
Pointcut.pattern("ActionDispatcher.dispatch")

# Match all methods on a class
Pointcut.pattern("MyCapability.*")

# Match a method across all classes
Pointcut.pattern("*.analyze")
```

#### Combinators

Pointcuts compose with logical operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `&` | AND | `Pointcut.pattern("*.dispatch") & Pointcut.class_filter(ActionDispatcher)` |
| `\|` | OR | `Pointcut.pattern("*.analyze") \| Pointcut.pattern("*.synthesize")` |
| `~` | NOT | `~Pointcut.pattern("*.internal_*")` |

Additional factory methods:

```python
Pointcut.cls(MyCapability)              # Match all methods on instances of a class
Pointcut.instance(specific_cap)         # Match only this specific instance (weak ref)
Pointcut.method("analyze")              # Exact method name matching
Pointcut.decorated_with("_is_hookable") # Match methods with a decorator marker
```


## Hook Execution Chain

When a `@hookable` method is called, hooks execute in this order:

1. **`BEFORE`** hooks (highest priority first) -- can modify `ctx.args`/`ctx.kwargs`
2. **`AROUND`** hooks build a wrapper chain (highest priority = outermost)
3. The original method executes inside the AROUND wrapper
4. **`AFTER`** hooks (highest priority first) -- can modify the return value

```python
# Handler type signatures:
BeforeHookHandler = Callable[[HookContext], Awaitable[HookContext]]
AfterHookHandler  = Callable[[HookContext, Any], Awaitable[Any]]
AroundHookHandler = Callable[[HookContext, Callable[[], Awaitable[Any]]], Awaitable[Any]]
```

## Error Handling

`ErrorMode` (in `polymathera.colony.distributed.hooks.types`) controls hook failure behavior:

```python
class ErrorMode(str, Enum):
    FAIL_FAST = "fail_fast"  # First error aborts entire chain (default)
    CONTINUE = "continue"    # Log error, continue to next hook
    SUPPRESS = "suppress"    # Silently ignore errors
```

## Use Cases

### Token Tracking

An `AFTER` hook on inference methods tracks token consumption without modifying any inference code:

```python
@after(Pointcut.pattern("*.submit_inference"))
async def track_tokens(ctx: HookContext, result: InferenceResponse):
    agent = ctx.agent
    usage = result.usage
    await agent.update_resource_usage(tokens=usage.total_tokens)
```

### Rate Limiting

A `BEFORE` hook throttles inference requests:

```python
@before(Pointcut.pattern("*.submit_inference"))
async def rate_limit(ctx: HookContext):
    if ctx.agent.requests_this_minute > MAX_REQUESTS:
        await asyncio.sleep(backoff_duration)
```

### Memory Capture

The memory system uses `AFTER` hooks via `MemoryProducerConfig` to observe agent behavior and automatically store memories. See [Memory System](agent-memory-system.md) for details.

### Checkpointing

An `AFTER` hook on plan execution saves state for recovery:

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
