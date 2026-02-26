# Proposal: Deferred Binding for Agents, Capabilities, and Policies

## The Pattern

This proposal introduces a unified **deferred binding** pattern for all user-extensible types in Colony: `Agent`, `AgentCapability`, and `ActionPolicy`. The pattern is a form of *partial application* for constructors — `cls.bind(*args, **kwargs)` captures a class and its constructor arguments into a lightweight `Bound[T]` object that can be transported across Ray node boundaries (via `cloudpickle`) and instantiated later on the target node.

This is the same pattern used by `WrappedDeployment.bind()` → `BoundDeployment` in Colony's serving framework and by Ray Serve's deployment graph API.

> In type theory, `Bound[T]` is a *thunk* — a suspended computation `() → T` that, when applied, produces an instance. In practical terms, it is `functools.partial` for class constructors, made explicit as a first-class object so it can be named, stored, serialized, and composed.

Since Ray distributes Python packages to workers via `runtime_env.py_modules` (set up in `PolymatheraApp.setup_ray`), any user-defined subclass of `Agent`, `AgentCapability`, or `ActionPolicy` is importable on every node. `Bound[T]` carries the class itself (not a string path) — `cloudpickle` handles the rest.

## Problem Statement

`AgentCapability.get_capability_name()` is a `@classmethod` returning `cls.__name__`. This is used as the key in `Agent._capabilities: dict[str, AgentCapability]`. When multiple instances of the same class are added (e.g., six `MemoryCapability` instances for working/stm/ltm-episodic/ltm-semantic/ltm-procedural/context), later instances silently overwrite earlier ones — only the last one survives.

Additionally, `_create_action_policy()` constructs capabilities from `capability_classes` with `cap_class(self)` — there is no way to pass constructor arguments like `scope_id`, `ttl_seconds`, or `max_entries`. Parent agents cannot specify how child agents should configure their capabilities.

The same limitation applies to agents and action policies: `AgentSpawnSpec` carries string class paths and has no way to encode constructor arguments. The `action_policy` field is a string class path. There is no uniform mechanism to fully specify the construction of an object at one Ray node and actually construct it on another.

## Root Causes

1. **Class-level naming**: `get_capability_name()` returns `cls.__name__`, so all instances of the same class have the same key.
2. **No binding mechanism**: `capability_classes: list[type[AgentCapability]]` only carries classes, not constructor arguments. Same for `AgentSpawnSpec.agent_type: str` and `AgentSpawnSpec.action_policy: str`.
3. **String-based class resolution**: `_resolve_class_from_identifier()` imports classes by string path — fragile, no constructor args, no type safety.
4. **`spawn_agents()` drops capabilities**: `AgentSystemDeployment.spawn_agents()` never passes `spec.capabilities` to `start_agent()`.

## Design

### 0. `Bound[T]` — Abstract Deferred Construction

**File:** `colony/python/colony/distributed/ray_utils/binding.py` (new)

A generic base for all bound types. Kept minimal — no framework dependencies.

```python
from typing import TypeVar, Generic, Any

T = TypeVar("T")

class Bound(Generic[T]):
    """Deferred constructor: captures cls + args + kwargs for later instantiation.

    Serialized across Ray boundaries via cloudpickle (same as BoundDeployment).
    Subclasses add domain-specific fields (key, action filters, etc.).
    """

    def __init__(self, cls: type[T], args: tuple = (), kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs or {}

    def instantiate(self, *prefix_args: Any) -> T:
        """Create the instance. prefix_args are prepended (e.g., agent for capabilities)."""
        return self.cls(*prefix_args, *self.args, **self.kwargs)

    def __repr__(self) -> str:
        name = self.cls.__name__
        parts = [repr(a) for a in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"Bound({name}, {', '.join(parts)})" if parts else f"Bound({name})"
```

Each domain type extends this with its own fields:

### 1. `BoundCapability` — Capability + Constructor Arguments

```python
# colony/python/colony/agents/base.py

class BoundCapability(Bound[AgentCapability]):
    """A capability class bound with constructor arguments.

    The `key` uniquely identifies this capability instance within an agent.
    If not provided, defaults to cls.__name__ (single-instance behavior).
    """

    def __init__(
        self,
        cls: type[AgentCapability],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        *,
        key: str | None = None,
        include_actions: list[str] | None = None,
        exclude_actions: list[str] | None = None,
        events_only: bool = False,
    ):
        super().__init__(cls, args, kwargs)
        self.key = key or cls.__name__
        self.include_actions = include_actions
        self.exclude_actions = exclude_actions
        self.events_only = events_only
```

### 1b. `BoundAgent` — Agent + Constructor Arguments

Replaces `AgentSpawnSpec` as the unit of agent creation.

```python
# colony/python/colony/agents/base.py (or models.py)

class BoundAgent(Bound[Agent]):
    """An agent class bound with constructor arguments and spawn configuration.

    Replaces AgentSpawnSpec. All fields that were on AgentSpawnSpec
    are either constructor kwargs (metadata, bound_pages, etc.)
    or spawn-routing fields on this class (requirements, resource_requirements).
    """

    def __init__(
        self,
        cls: type[Agent],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
        bound_pages: list[str] | None = None,
        requirements: Any | None = None,
        resource_requirements: AgentResourceRequirements | None = None,
    ):
        super().__init__(cls, args, kwargs)
        self.agent_id = agent_id
        self.bound_pages = bound_pages or []
        self.requirements = requirements
        self.resource_requirements = resource_requirements or AgentResourceRequirements()

    def has_deployment_affinity(self) -> bool:
        return bool(self.bound_pages) or (self.requirements is not None)
```

### 1c. `BoundActionPolicy` — Policy + Constructor Arguments

```python
# colony/python/colony/agents/base.py

class BoundActionPolicy(Bound[ActionPolicy]):
    """An action policy class bound with constructor arguments.

    The `agent` argument is injected at instantiation time (not bound).
    """
    pass  # No extra fields needed — Bound[ActionPolicy] suffices.
```

### 2. `.bind()` Class Methods

Each extensible type gets a `bind()` classmethod. These are the user-facing API.

```python
# AgentCapability (base.py)
@classmethod
def bind(cls, *args, key: str | None = None, **kwargs) -> BoundCapability:
    return BoundCapability(cls, args, kwargs, key=key)

# Agent (base.py)
@classmethod
def bind(cls, *args, **kwargs) -> BoundAgent:
    return BoundAgent(cls, args, kwargs)

# ActionPolicy (base.py)
@classmethod
def bind(cls, *args, **kwargs) -> BoundActionPolicy:
    return BoundActionPolicy(cls, args, kwargs)
```

Usage:

```python
# Parent agent specifying how to create a child agent:
child = ComplianceAnalysisWorker.bind(
    agent_id="worker-1",
    bound_pages=["repo-123-context"],
    metadata=AgentMetadata(role="worker", tenant_id="acme"),
    bound_capabilities=[
        MemoryCapability.bind(scope_id=MemoryScope.agent_stm(worker_id), key="stm", ttl_seconds=3600),
        ComplianceVCMCapability.bind(key="vcm"),
    ],
    action_policy=CacheAwareActionPolicy.bind(planner=my_planner),
)

# Spawning — BoundAgent travels across Ray boundaries via cloudpickle:
agent_ids = await agent_system.spawn_agents([child])
```

### 3. Instance-Level `capability_key` Property

Replace the class-level `get_capability_name()` as the dict key with an instance-level `capability_key`:

```python
# On AgentCapability (base.py)

def __init__(self, agent=None, scope_id=None, *, blackboard=None, capability_key=None):
    # ... existing init ...
    self._capability_key: str | None = capability_key

@property
def capability_key(self) -> str:
    """Unique key for this instance within an agent's _capabilities dict.

    Set by BoundCapability or directly in constructor.
    Falls back to get_capability_name() (cls.__name__) for single-instance capabilities.
    """
    return self._capability_key or self.get_capability_name()
```

`get_capability_name()` remains a `@classmethod` — it's still useful for class-level lookups (e.g., "does this agent have any `MemoryCapability`?"). But it's no longer the dict key.

### 4. Changes to `Agent._capabilities` Dict

**Key change**: `_capabilities` is keyed by `capability_key` (instance-level), not `get_capability_name()` (class-level).

```python
# Agent.add_capability() — line 1673
def add_capability(self, capability, *, include_actions=None, exclude_actions=None, events_only=False):
    cap_key = capability.capability_key  # CHANGED: was capability.get_capability_name()
    if cap_key in self._capabilities:
        raise ValueError(f"Capability '{cap_key}' already added to agent {self.agent_id}")
    # ... filter setup unchanged ...
    self._capabilities[cap_key] = capability

# Agent.remove_capability() — line 1726
def remove_capability(self, capability_key: str) -> AgentCapability | None:
    capability = self._capabilities.pop(capability_key, None)
    # ... hook cleanup unchanged ...
    return capability

# Agent.get_capability() — line 1742
def get_capability(self, capability_key: str) -> AgentCapability | None:
    return self._capabilities.get(capability_key)

# Agent.has_capability() — line 1785
def has_capability(self, capability_key: str) -> bool:
    return capability_key in self._capabilities
```

`get_capability_by_type()` is unchanged — it does `isinstance()` scan over values, which is type-based and doesn't depend on keys.

### 5. New: `get_capabilities_by_class()`

For callers that need "all `MemoryCapability` instances":

```python
def get_capabilities_by_class(self, capability_class: type[AgentCapability]) -> list[AgentCapability]:
    """Get all capabilities that are instances of the given class."""
    return [cap for cap in self._capabilities.values() if isinstance(cap, capability_class)]
```

This replaces the pattern of `agent.get_capability(MemoryCapability.get_capability_name())` which only finds one. `AgentContextEngine.initialize()` already uses `isinstance()` to discover memory capabilities, so it's already correct.

### 6. Replace `capability_classes` with `bound_capabilities` on `Agent`

```python
# Agent model field (line 1584)
# BEFORE:
capability_classes: list[type[AgentCapability]] = Field(default_factory=list)

# AFTER:
bound_capabilities: list[BoundCapability] = Field(default_factory=list)
```

Replace `add_capability_classes` with `add_bound_capabilities` everywhere:

```python
def add_bound_capabilities(self, capabilities: list[BoundCapability]) -> None:
    """Add bound capabilities to be instantiated during _create_action_policy."""
    for bound in capabilities:
        if not any(b.key == bound.key for b in self.bound_capabilities):
            self.bound_capabilities.append(bound)
```

Delete `add_capability_classes()`. All callers migrate:

```python
# BEFORE:
self.add_capability_classes([ComplianceVCMCapability, MergeCapability])

# AFTER:
self.add_bound_capabilities([
    ComplianceVCMCapability.bind(),
    MergeCapability.bind(),
])
```

### 7. Replace `action_policy_class` with `bound_action_policy` on `Agent`

```python
# Agent model field (line 1596)
# BEFORE:
action_policy_class: type[ActionPolicy] | None = None

# AFTER:
bound_action_policy: BoundActionPolicy | None = None
```

### 8. Update `_create_action_policy()`

```python
async def _create_action_policy(self) -> None:
    if self.action_policy:
        return

    # Phase 1: Instantiate bound capabilities
    for bound in self.bound_capabilities:
        if not self.has_capability(bound.key):
            capability = bound.instantiate(self)
            capability._capability_key = bound.key
            self.add_capability(
                capability,
                include_actions=bound.include_actions,
                exclude_actions=bound.exclude_actions,
                events_only=bound.events_only,
            )

    # Phase 2: Initialize all capabilities
    for capability in list(self._capabilities.values()):
        await capability.initialize()

    # Phase 3: Create action policy
    if self.bound_action_policy:
        # Bound policy — instantiate with agent + action_providers + bound args
        self.action_policy = self.bound_action_policy.instantiate(
            self,
            action_providers=list(self._capabilities.values()),
        )
    else:
        from .patterns.actions.policies import create_default_action_policy
        self.action_policy = await create_default_action_policy(
            agent=self,
            action_map={},
            action_providers=list(self._capabilities.values()),
            max_iterations=self.metadata.max_iterations,
            **self.metadata.action_policy_config
        )

    # Phase 4: Mark bound capabilities as "used"
    self.action_policy.use_agent_capabilities(
        [bound.key for bound in self.bound_capabilities]
    )
    await self.action_policy.initialize()
```

### 9. Replace `AgentSpawnSpec` with `BoundAgent`

`AgentSpawnSpec` is deleted. `spawn_agents` takes `list[BoundAgent]`:

```python
# system.py — AgentSystemDeployment
@serving.endpoint
async def spawn_agents(
    self,
    agent_specs: list[BoundAgent],   # CHANGED from list[AgentSpawnSpec]
    session_id: str | None = None,
    run_id: str | None = None,
    soft_affinity: bool = True,
    suspend_agents: bool = False,
) -> list[str]:
```

Routing uses `spec.has_deployment_affinity()` (same method, now on `BoundAgent`).

### 10. Rewrite `_create_agent_instance()` — No More String Resolution

`_create_agent_instance` no longer resolves strings. It receives a `BoundAgent` directly:

```python
def _create_agent_instance(self, bound_agent: BoundAgent, agent_id: str, metadata: AgentMetadata) -> Agent:
    """Create an agent from a BoundAgent."""
    extra_fields: dict[str, Any] = {}
    if metadata.tenant_id:
        extra_fields["tenant_id"] = metadata.tenant_id
    if metadata.group_id:
        extra_fields["group_id"] = metadata.group_id

    # BoundAgent.kwargs may contain bound_capabilities, bound_action_policy, etc.
    # These are passed through to the Agent constructor.
    return bound_agent.instantiate(
        agent_id=agent_id,
        metadata=metadata,
        resource_requirements=bound_agent.resource_requirements,
        bound_pages=bound_agent.bound_pages,
        **extra_fields,
    )
```

`_resolve_class_from_identifier` is no longer needed for agents/capabilities/policies. It can remain for other uses or be removed.

### 11. `start_agent()` Signature Change

```python
# BEFORE:
async def start_agent(
    self, agent_class_id: str, agent_id: str | None = None,
    capabilities: list[str] | None = None,
    action_policy_id: str | None = None, ...
) -> str:

# AFTER:
async def start_agent(
    self, bound_agent: BoundAgent,
    agent_id: str | None = None, ...
) -> str:
```

The `BoundAgent` carries the class, constructor args, capabilities, action policy — everything needed. No string-based import resolution.

### 12. Update `AgentPoolCapability.create_agent()`

The LLM-facing `create_agent` action still takes a string `agent_type` (the LLM cannot construct Python objects). The capability resolves it to a `BoundAgent`:

```python
@action_executor()
async def create_agent(
    self,
    agent_type: str,
    bound_pages: list[str] | None = None,
    metadata: AgentMetadata | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    # Resolve agent_type string to class (for LLM-initiated spawning)
    agent_class = self._resolve_agent_type(agent_type)

    bound = BoundAgent(
        agent_class,
        kwargs=dict(
            metadata=metadata or AgentMetadata(
                tenant_id=self.agent.tenant_id,
                group_id=self.agent.group_id,
                parent_agent_id=self.agent.agent_id,
            ),
        ),
        bound_pages=bound_pages or [],
    )

    handles = await self.agent.spawn_child_agents(
        agent_specs=[bound],
        return_handles=True,
    )
```

### 13. Update `create_default_memory_hierarchy()`

Pass `capability_key=` to each memory capability constructor:

```python
working = WorkingMemoryCapability(
    agent=agent,
    scope_id=MemoryScope.agent_working(agent_id),
    capability_key="working",
    ...
)
stm = MemoryCapability(
    agent=agent,
    scope_id=MemoryScope.agent_stm(agent_id),
    capability_key="stm",
    ...
)
ltm_episodic = MemoryCapability(
    agent=agent,
    scope_id=MemoryScope.agent_ltm_episodic(agent_id),
    capability_key="ltm:episodic",
    ...
)
# etc.
```

### 14. Update `ActionPolicy.use_agent_capability_types`

Rename to `use_bound_capabilities`. Takes `list[BoundCapability]`:

```python
async def use_bound_capabilities(self, capabilities: list[BoundCapability]) -> None:
    for bound in capabilities:
        if not self.agent.has_capability(bound.key):
            instance = bound.instantiate(self.agent)
            instance._capability_key = bound.key
            await instance.initialize()
            self.agent.add_capability(instance)
        self.use_agent_capabilities([bound.key])
```

`use_agent_capabilities(list[str])` — unchanged, still works with capability keys.

## Callers Impacted — Full Audit

### Class-level `get_capability_name()` calls (must remain working)

These call `SomeClass.get_capability_name()` on the class, not an instance. They use it to look up capabilities by class identity. With the new design, these still work for **single-instance capabilities** (where key == class name). For multi-instance capabilities, callers should use `get_capability_by_type()` or `get_capabilities_by_class()`.

| File | Line | Code | Change needed? |
|------|------|------|----------------|
| `base.py` | 120 | `capability_cls.get_capability_name()` in `use_agent_capability_types` | Yes — method renamed to `use_bound_capabilities`, see section 14 |
| `base.py` | 133 | Same method | Yes — same |
| `base.py` | 967 | `capability_type.get_capability_name()` in `AgentHandle.get_capability` | No — AgentHandle creates its own instances keyed by class name, doesn't touch agent._capabilities |
| `base.py` | 1932 | `cap_class.get_capability_name()` in `_create_action_policy` | Replaced — uses `bound.key` |
| `base.py` | 1978 | `cap.get_capability_name() for cap in self.capability_classes` | Replaced — uses `bound.key for bound in self.bound_capabilities` |
| `patterns/games/state.py` | 818, 822 | `HypothesisGameCapability.get_capability_name()` | No change — single instance per agent, key == class name |
| `patterns/games/negotiation/capabilities.py` | 337 | `NegotiationGameProtocol.get_capability_name()` | No change — same reason |
| `patterns/capabilities/reflection.py` | 247 | `MemoryCapability.get_capability_name()` | **Yes — should use `get_capability_by_type(MemoryCapability)`** |
| `patterns/capabilities/reflection.py` | 372 | `ConsciousnessCapability.get_capability_name()` | No change — single instance |
| `patterns/capabilities/critique.py` | 785 | `MemoryCapability.get_capability_name()` | **Yes — should use `get_capability_by_type(MemoryCapability)`** |
| `patterns/attention/adaptive.py` | 47 | `AdaptiveQueryGenerator.get_capability_name()` | No change — single instance |
| `patterns/attention/multi_hop.py` | 276 | `MultiHopSearchCapability.get_capability_name()` | No change — single instance |
| `patterns/attention/incremental.py` | 249 | `IncrementalQueryCapability.get_capability_name()` | No change — single instance |
| `patterns/meta_agents/reputation_agent.py` | 90 | `HypothesisGameProtocol.get_capability_name()` | No change — single instance |
| `patterns/meta_agents/reputation_agent.py` | 99 | `ReputationCapability.get_capability_name()` | No change — single instance |
| `patterns/meta_agents/goal_alignment.py` | 91, 96 | `ObjectiveGuardCapability.get_capability_name()` | No change — single instance |
| `patterns/meta_agents/grounding.py` | 90 | `capability_type.get_capability_name()` | No change — iterates types |
| `patterns/memory/__init__.py` | 118 | `AgentContextEngine.get_capability_name()` | No change — single instance |

### `add_capability_classes()` callers → migrate to `add_bound_capabilities()`

All callers switch from class lists to bound capability lists. Clean break — no backward compat wrapper.

| File | Before | After |
|------|--------|-------|
| `samples/code_analysis/compliance/agents.py` | `self.add_capability_classes([ComplianceVCMCapability, MergeCapability])` | `self.add_bound_capabilities([ComplianceVCMCapability.bind(), MergeCapability.bind()])` |
| `patterns/meta_agents/reputation_agent.py` | `self.add_capability_classes([...])` | `self.add_bound_capabilities([...bind()])` |
| `patterns/meta_agents/goal_alignment.py` | `self.add_capability_classes([...])` | `self.add_bound_capabilities([...bind()])` |

### `add_capability()` callers

These already create instances manually, so they just need to set `capability_key`.

| File | Code | Migration |
|------|------|-----------|
| `patterns/memory/defaults.py` | `agent.add_capability(working)` etc. | Set `capability_key=` in constructor |
| `base.py` | `_create_action_policy` | Replaced with `bound.instantiate()` |

### `AgentSpawnSpec` construction sites → migrate to `BoundAgent`

| File | Before | After |
|------|--------|-------|
| `cli/polymath.py:1564` | `AgentSpawnSpec(agent_type=coord_class, capabilities=..., metadata=..., bound_pages=[])` | `coord_class.bind(metadata=..., bound_capabilities=[...])` with `BoundAgent(cls=coord_class, ...)` |
| `samples/.../code_analysis_agent_example.py:67` | `AgentSpawnSpec(agent_type="...", metadata=..., bound_pages=[])` | `CodeAnalysisCoordinator.bind(metadata=..., bound_pages=[])` |
| `samples/.../coordinator.py:427` | `AgentSpawnSpec(agent_type="...", metadata=..., bound_pages=..., resource_requirements=...)` | `ClusterAnalyzer.bind(metadata=..., bound_pages=..., resource_requirements=...)` |
| `agents/system.py:704` (resumption) | `AgentSpawnSpec(agent_type=..., bound_pages=..., resource_requirements=..., metadata=...)` | `BoundAgent(cls=agent_class, bound_pages=..., resource_requirements=..., kwargs={"metadata": ...})` |
| `patterns/capabilities/agent_pool.py:136` | `AgentSpawnSpec(agent_type=..., capability_types=..., ...)` (BUG) | `BoundAgent(cls=resolved_class, ...)` |
| `patterns/memory/management.py:852` | `AgentSpawnSpec(agent_type="...", ...)` | `MemoryManagementAgent.bind(...)` |

### `action_policy_class` → `bound_action_policy`

| File | Before | After |
|------|--------|-------|
| `base.py:1965` | `self.action_policy_class(agent=self, action_providers=..., **self.metadata.action_policy_config)` | `self.bound_action_policy.instantiate(self, action_providers=...)` |
| `_create_agent_instance` | `action_policy_class=action_policy_class` | `bound_action_policy=bound_agent.kwargs.get("bound_action_policy")` or from `BoundAgent` |

### `spawn_child_agents()` callers

| File | Code | Migration |
|------|------|-----------|
| `patterns/capabilities/agent_pool.py` | `create_agent()` | Construct `BoundAgent` instead of `AgentSpawnSpec` |
| Any coordinator agent | `self.spawn_child_agents(specs)` | `specs` is now `list[BoundAgent]` |

## Existing Bugs Fixed by This Refactoring

1. **Memory capabilities overwriting each other**: Six `MemoryCapability` instances all use key `"MemoryCapability"` → only one survives. Fixed by instance-level `capability_key`.

2. **`AgentSpawnSpec.capabilities` never forwarded**: `spawn_agents()` drops `spec.capabilities`. Eliminated — `BoundAgent` carries everything and `spawn_agents` instantiates from it.

3. **`AgentPoolCapability` wrong field name**: Passes `capability_types=` (nonexistent field on `AgentSpawnSpec`). Eliminated — `AgentSpawnSpec` is deleted; `BoundAgent` has no string field name confusion.

4. **`_create_action_policy` can't pass constructor args**: `cap_class(self)` only passes agent. Fixed by `bound.instantiate(self)` which passes all bound args.

5. **Duplicate `get_capability_names()`**: Defined twice on `Agent` (lines 1777 and 1795). Remove the duplicate during this refactoring.

6. **`spawn_child_agents` roles default bug**: `roles: list[str] = []` fails validation when `agent_specs` is non-empty. Fix: make the check conditional on `roles` being non-empty.

7. **String-based class resolution fragility**: `_resolve_class_from_identifier` guesses import paths. Eliminated — `Bound[T]` carries the class directly, cloudpickle handles serialization.

## Execution Order

### Phase 1: `Bound[T]` Infrastructure

1. Create `colony/python/colony/distributed/ray_utils/binding.py` with `Bound[T]`
2. Add `BoundCapability`, `BoundAgent`, `BoundActionPolicy` (in `base.py` or `models.py`)
3. Add `capability_key` param to `AgentCapability.__init__()`, add `capability_key` property
4. Add `.bind()` classmethod on `AgentCapability`, `Agent`, `ActionPolicy`
5. Add `Agent.get_capabilities_by_class()` method

### Phase 2: Capabilities — Switch Dict Key and Field

6. Change `add_capability()` to use `capability.capability_key`
7. Update `create_default_memory_hierarchy()` to set `capability_key=` on each instance
8. Fix callers in `reflection.py` and `critique.py` → use `get_capability_by_type()`
9. Remove duplicate `get_capability_names()` definition
10. Replace `capability_classes` field with `bound_capabilities` on `Agent`
11. Delete `add_capability_classes()`, add `add_bound_capabilities()`
12. Update `_create_action_policy()` to iterate `bound_capabilities`
13. Rename `use_agent_capability_types()` → `use_bound_capabilities()`, update callers

### Phase 3: Agents — Replace `AgentSpawnSpec` with `BoundAgent`

14. Replace `action_policy_class` field with `bound_action_policy` on `Agent`
15. Delete `AgentSpawnSpec` from `models.py`
16. Rewrite `spawn_agents()` to accept `list[BoundAgent]`
17. Rewrite `start_agent()` to accept `BoundAgent`
18. Rewrite `_create_agent_instance()` — no string resolution, just `bound_agent.instantiate()`
19. Update `spawn_child_agents()` to accept `list[BoundAgent]`
20. Fix `spawn_child_agents` roles default bug
21. Update `AgentPoolCapability.create_agent()` to construct `BoundAgent`

### Phase 4: Migrate All Callers

22. `polymath.py` — construct `BoundAgent` instead of `AgentSpawnSpec`
23. `code_analysis_agent_example.py` — same
24. `coordinator.py` — same
25. `system.py` resumption path — same
26. `memory/management.py` — same
27. All `add_capability_classes` callers → `add_bound_capabilities`
28. All `action_policy_class=` callers → `bound_action_policy=`
29. Verify `AgentContextEngine.initialize()` still discovers all memory capabilities (it uses `isinstance`, so it should)

## Testing

1. **Unit test**: Create agent with multiple `MemoryCapability.bind(key="stm", ...)` and verify all are stored and retrievable.
2. **Unit test**: `BoundAgent` and `BoundCapability` survive cloudpickle round-trip (simulating Ray transport).
3. **Integration test**: Full compliance analysis run — verify all memory scopes present, no overwrites.
4. **Regression**: `get_capability_by_type(MemoryCapability)` returns first match (unchanged behavior).

