# Proposal: Blueprint[T] — Deferred Binding for Agents, Capabilities, and Policies

## The Pattern

This proposal introduces a unified **deferred binding** pattern for all user-extensible types in Colony: `Agent`, `AgentCapability`, and `ActionPolicy`. The pattern captures a class and its **constructor arguments only** into a lightweight `Blueprint[T]` object that can be transported across Ray node boundaries (via `cloudpickle`) and instantiated later on the target node.

This is the same pattern used by `WrappedDeployment.bind()` → `BoundDeployment` in Colony's serving framework (Akka calls this `Props`; Erlang/OTP calls it `child_spec`).

> In type theory, `Blueprint[T]` is a *thunk* — a suspended computation `() → T` that, when applied, produces an instance. In practical terms, it is `functools.partial` for class constructors, made explicit as a first-class object so it can be named, stored, serialized, and composed.

Since Ray distributes Python packages to workers via `runtime_env.py_modules` (set up in `PolymatheraApp.setup_ray`), any user-defined subclass of `Agent`, `AgentCapability`, or `ActionPolicy` is importable on every node. `Blueprint[T]` carries the class itself (not a string path) — `cloudpickle` handles the rest.

### Three-Layer Separation

`Blueprint[T]` enforces a strict separation of concerns:

| Layer | What it captures | API | Example |
|-------|------------------|-----|---------|
| **Constructor args** | Arguments to `T.__init__()` | `cls.bind(**kwargs)` → `Blueprint[T]` | `metadata=..., bound_pages=[...]` |
| **Composition** | How an instance is used by its parent | `.with_composition(...)` on `AgentCapabilityBlueprint` | `key="stm", events_only=True` |
| **Deployment** | Where/how to spawn remotely | `bp.remote_instance(...)` on `AgentBlueprint` | `requirements=..., session_id=...` |

**Critical invariant**: `Blueprint[T]` captures **only** the constructor arguments of `T` — never deployment params (`suspend_agents`, `max_iterations`, `requirements`) or composition metadata (`key`, `include_actions`). These belong at separate layers.

### Validation at Bind Time

`Blueprint[T]` validates at bind time (fail-fast):
1. **For pydantic BaseModel subclasses** (e.g., `Agent`): validates kwargs are recognized `model_fields`, rejects excluded fields (`agent_id`, `state`, `created_at`).
2. **For all classes**: validates all kwargs are serializable via `ray.cloudpickle.dumps`.
3. **Nested Blueprint objects** in kwargs are validated recursively.

## Problem Statement

### Original Problems (unchanged)
1. **Capability key collision**: `AgentCapability.get_capability_name()` is a `@classmethod` returning `cls.__name__`. This is used as the key in `Agent._capabilities: dict[str, AgentCapability]`. When multiple instances of the same class are added (e.g., six `MemoryCapability` instances for working/stm/ltm-episodic/ltm-semantic/ltm-procedural/context), later instances silently overwrite earlier ones — only the last one survives.
2. **No constructor args**: `capability_classes: list[type[AgentCapability]]` only carries classes, not constructor arguments.
3. **String-based class resolution**: `_resolve_class_from_identifier()` is fragile — no type safety.
4. **`spawn_agents()` drops capabilities**: Never forwards `spec.capabilities` to `start_agent()`.

`_create_action_policy()` constructs capabilities from `capability_classes` with `cap_class(self)` — there is no way to pass constructor arguments like `scope_id`, `ttl_seconds`, or `max_entries`. Parent agents cannot specify how child agents should configure their capabilities.

The same limitation applies to agents and action policies: `AgentSpawnSpec` carries string class paths and has no way to encode constructor arguments. The `action_policy` field is a string class path. There is no uniform mechanism to fully specify the construction of an object at one Ray node and actually construct it on another.

### Additional Problems (identified during first implementation)
5. **Mixed concerns in `Bound[T]`**: The first implementation's `BoundAgent` stored both Agent constructor args (`metadata`, `bound_pages`, `resource_requirements`) and unrelated deployment params (`suspend_agents`, `max_iterations`). `start_agent` accepted redundant `metadata=` and `resource_requirements=` overrides alongside the bound object. `Bound[T]` became a grab-bag instead of a disciplined constructor thunk.
6. **No validation at bind time**: `Bound.__init__` accepted arbitrary `*args, **kwargs` with no type checking against the target class, no serializability check. Runtime errors surfaced on remote nodes instead of at the call site.

## Root Causes

1. **Class-level naming**: `get_capability_name()` returns `cls.__name__`, so all instances of the same class have the same key.
2. **No binding mechanism**: `capability_classes: list[type[AgentCapability]]` only carries classes, not constructor arguments. Same for `AgentSpawnSpec.agent_type: str` and `AgentSpawnSpec.action_policy: str`.
3. **String-based class resolution**: `_resolve_class_from_identifier()` imports classes by string path — fragile, no constructor args, no type safety.
4. **`spawn_agents()` drops capabilities**: `AgentSystemDeployment.spawn_agents()` never passes `spec.capabilities` to `start_agent()`.

## Design

### 0. `Blueprint[T]` — Abstract Deferred Construction

**File:** `colony/python/colony/agents/blueprint.py` (new)

A generic base for all blueprint types. Captures **only** constructor kwargs — no positional args, no deployment params.

```python
from typing import TypeVar, Generic, Any, ClassVar
from pydantic import BaseModel

T = TypeVar("T")

class Blueprint(Generic[T]):
    """Deferred constructor: captures cls + keyword args for later instantiation.

    Serialized across Ray boundaries via cloudpickle (same as BoundDeployment).
    Subclasses add domain-specific fields (key, action filters, etc.).

    Validates at bind time:
    - Serializability via ray.cloudpickle.dumps
    - For pydantic BaseModel subclasses: kwargs are valid model fields

    Invariants:
    - self.cls is the actual class (not a string path)
    - self.kwargs contains ONLY constructor kwargs for cls.__init__
    - No positional args stored (prefix_args injected at local_instance time)
    """
    __slots__ = ("cls", "kwargs")

    def __init__(self, cls: type[T], kwargs: dict[str, Any]):
        self.cls = cls
        self.kwargs = kwargs

    def validate_serializable(self) -> None:
        """Validate all kwargs are cloudpickle-serializable. Called at bind time.
        Recursively validates nested Blueprint objects."""
        import ray.cloudpickle as cloudpickle
        for key, value in self.kwargs.items():
            if isinstance(value, Blueprint):
                value.validate_serializable()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Blueprint):
                        item.validate_serializable()
                    else:
                        try: cloudpickle.dumps(item)
                        except Exception as e:
                            raise BlueprintSerializationError(
                                f"List item in '{key}' for {self.cls.__name__} "
                                f"not serializable: {e}") from e
            else:
                try: cloudpickle.dumps(value)
                except Exception as e:
                    raise BlueprintSerializationError(
                        f"Argument '{key}' (type={type(value).__name__}) for "
                        f"{self.cls.__name__} not serializable: {e}") from e

    def local_instance(self, *prefix_args: Any, **extra_kwargs: Any) -> T:
        """Construct T locally. Called on the target deployment node.

        Does NOT auto-resolve nested Blueprints in kwargs — domain code
        (e.g., Agent._create_action_policy) handles that explicitly.

        Args:
            *prefix_args: Prepended positional args (e.g., agent for capabilities).
            **extra_kwargs: Merged with bound kwargs (overrides on conflict).
        """
        merged = {**self.kwargs, **extra_kwargs}
        return self.cls(*prefix_args, **merged)

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"Blueprint({self.cls.__name__}, {', '.join(parts)})" if parts else f"Blueprint({self.cls.__name__})"
```

**Key changes from previous `Bound[T]`:**
1. **No positional `args: tuple`** — for pydantic models all construction is keyword-based; for `AgentCapability(agent, scope_id)`, `agent` is a prefix_arg injected at `local_instance` time — never bound.
2. **`validate_serializable()`** — validates all kwargs via cloudpickle at bind time (fail-fast).
3. **`local_instance()` replaces `instantiate()`** — clearer name indicating where construction happens.
4. **No auto-resolution of nested Blueprints** — `capability_blueprints` and `action_policy_blueprint` are instantiated by `_create_action_policy` with domain-specific logic.

### 1. `AgentCapabilityBlueprint` — Constructor Args + Composition Metadata

```python
# colony/python/colony/agents/base.py

class AgentCapabilityBlueprint(Blueprint[AgentCapability]):
    """Blueprint for an AgentCapability.
    A capability class bound with constructor arguments.

    kwargs: constructor args (scope_id, blackboard, ttl_seconds, etc.)
    Composition fields: how this capability is added to the agent (via add_capability).
    Set via .with_composition() — NOT passed to the capability constructor.
    """
    __slots__ = ("key", "include_actions", "exclude_actions", "events_only")

    def __init__(
        self,
        cls: type[AgentCapability],
        kwargs: dict[str, Any] | None = None,
        *,
        key: str | None = None,
        include_actions: list[str] | None = None,
        exclude_actions: list[str] | None = None,
        events_only: bool = False,
    ):
        super().__init__(cls, kwargs)
        self.key = key or cls.__name__
        self.include_actions = include_actions
        self.exclude_actions = exclude_actions
        self.events_only = events_only

    def with_composition(self, *, key=None, include_actions=None,
                         exclude_actions=None, events_only=None) -> "AgentCapabilityBlueprint":
        """Return a copy with updated composition metadata. Immutable."""
        return AgentCapabilityBlueprint(
            self.cls, self.kwargs,
            key=key if key is not None else self.key,
            include_actions=include_actions if include_actions is not None else self.include_actions,
            exclude_actions=exclude_actions if exclude_actions is not None else self.exclude_actions,
            events_only=events_only if events_only is not None else self.events_only)
```

### 1b. `AgentBlueprint` — Pydantic Validation + `remote_instance`

Replaces `AgentSpawnSpec` as the unit of agent creation. Constructor args validated against `model_fields`.

```python
# colony/python/colony/agents/base.py (or models.py)

class AgentBlueprint(Blueprint[Agent]):
    """Blueprint for an Agent class bound with constructor arguments
    with pydantic model_fields validation.

    At bind time:
    - Rejects kwargs that aren't Agent model fields
    - Rejects excluded fields (agent_id, state, created_at, etc.)
    - Validates serializability via cloudpickle
    Full pydantic validation deferred to local_instance (needs agent_id).
    """
    _EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "agent_id", "state", "created_at",
        "action_policy", "action_policy_state",
        "page_storage", "child_agents",
    })

    def __init__(
        self,
        cls: type[Agent],
        kwargs: dict[str, Any] | None = None
    ):
        model_fields = set(cls.model_fields.keys())
        allowed = model_fields - self._EXCLUDED_FIELDS
        invalid = set(kwargs.keys()) - allowed
        if invalid:
            raise BlueprintValidationError(
                f"Invalid bind() arguments for {cls.__name__}: {invalid}. "
                f"Allowed: {sorted(allowed)}")
        super().__init__(cls, kwargs)

    def has_deployment_affinity(self) -> bool:
        """Check if agent needs VLLM routing (has bound pages)."""
        return bool(self.kwargs.get("bound_pages"))

    async def remote_instance(
        self, *,
        requirements: Any | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        max_iterations: int | None = None,
        app_name: str | None = None,
    ) -> str:
        """Spawn this agent on a remote deployment.

        Deployment parameters (NOT Agent constructor args) go here.
        Catches remote exceptions and raises BlueprintRemoteError
        with the remote traceback attached.

        Args:
            requirements: LLMClientRequirements for VLLM deployment routing
            agent_id: Optional agent ID (auto-generated if None)
            session_id: Session tracking
            run_id: Run tracking
            soft_affinity: Allow spawning without all pages loaded
            suspend_agents: Allow suspending existing agents for resources
            max_iterations: Max iterations for the agent's action policy
            app_name: Application name (for cross-app spawning)

        Returns:
            The spawned agent_id

        Raises:
            BlueprintRemoteError: If remote instantiation fails
        """
        from ..system import get_agent_system
        agent_system = get_agent_system(app_name)
        try:
            return await agent_system.spawn_from_blueprint(
                blueprint=self, requirements=requirements, agent_id=agent_id,
                session_id=session_id, run_id=run_id, soft_affinity=soft_affinity,
                suspend_agents=suspend_agents, max_iterations=max_iterations)
        except Exception as e:
            raise BlueprintRemoteError(
                f"Failed to spawn {self.cls.__name__} remotely: {e}",
                remote_traceback=getattr(e, 'traceback', str(e))) from e
```

### 1c. `ActionPolicyBlueprint` — No Extra Fields

```python
class ActionPolicyBlueprint(Blueprint[ActionPolicy]):
    """Blueprint for an ActionPolicy.

    The agent and action_providers arguments are injected via local_instance prefix/extra args at instantiation time (not bound).
    """
    pass  # No extra fields needed — Blueprint[ActionPolicy] suffices.
```

### 1d. Error Types

```python
class BlueprintValidationError(Exception):
    """Bind-time validation failure (invalid field name, wrong type)."""

class BlueprintSerializationError(BlueprintValidationError):
    """Argument not serializable via cloudpickle."""

class BlueprintRemoteError(Exception):
    """Remote instantiation failure with remote traceback."""
    def __init__(self, message, remote_traceback=None):
        super().__init__(message)
        self.remote_traceback = remote_traceback
```

### 2. `.bind()` Class Methods + `blueprint()` Decorator

Each base class defines `.bind()` directly — all subclasses inherit it:

```python
# AgentCapability (base.py) — bind() returns constructor-only blueprint
@classmethod
def bind(cls, **kwargs) -> AgentCapabilityBlueprint:
    bp = AgentCapabilityBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp

# Agent (base.py) — bind() validates against model_fields
@classmethod
def bind(cls, **kwargs) -> AgentBlueprint:
    bp = AgentBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp

# ActionPolicy (base.py)
@classmethod
def bind(cls, **kwargs) -> ActionPolicyBlueprint:
    bp = ActionPolicyBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp
```

The `blueprint()` decorator is available for non-standard classes:

```python
def blueprint(cls: type[T]) -> type[T]:
    """Add .bind() to any class. No-op if cls already has bind."""
    if hasattr(cls, 'bind'):
        return cls
    @classmethod
    def bind(klass, **kwargs) -> Blueprint:
        bp = Blueprint(klass, kwargs)
        bp.validate_serializable()
        return bp
    cls.bind = bind
    return cls
```

Usage:

```python
# Constructor args go in bind() — composition via .with_composition() — deployment via .remote_instance()
agent_bp = ComplianceAnalysisCoordinator.bind(
    agent_type="compliance_coordinator",
    bound_pages=["repo-123-context"],
    resource_requirements=AgentResourceRequirements(cpu_cores=0.5, memory_mb=1024),
    metadata=AgentMetadata(tenant_id="acme", role="coordinator"),
    capability_blueprints=[
        MemoryCapability.bind(scope_id="agent:stm:coord", ttl_seconds=3600).with_composition(key="stm"),
        MemoryCapability.bind(scope_id="agent:ltm:coord").with_composition(key="ltm:episodic", events_only=True),
        ComplianceVCMCapability.bind().with_composition(key="vcm"),
    ],
    action_policy_blueprint=CacheAwareActionPolicy.bind(planner=my_planner),
)

# Deployment params are separate from constructor args:
agent_id = await agent_bp.remote_instance(
    requirements=LLMClientRequirements(model_family="llama", min_context_window=32000),
    session_id="sess-123",
    run_id="run-456",
    max_iterations=50,
)
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

    Set by AgentCapabilityBlueprint or directly in constructor.
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

### 6. Replace `capability_classes` with `capability_blueprints` on `Agent`

```python
# Agent model field (line 1584)
# BEFORE:
capability_classes: list[type[AgentCapability]] = Field(default_factory=list)

# AFTER:
capability_blueprints: list[AgentCapabilityBlueprint] = Field(default_factory=list)
```

Replace `add_capability_classes` with `add_capability_blueprints` everywhere:

```python
def add_capability_blueprints(self, capabilities: list[AgentCapabilityBlueprint]) -> None:
    """Add capability blueprints to be instantiated during _create_action_policy."""
    for blueprint in capabilities:
        if not any(b.key == blueprint.key for b in self.capability_blueprints):
            self.capability_blueprints.append(blueprint)
```

Delete `add_capability_classes()`. All callers migrate:

```python
# BEFORE:
self.add_capability_classes([ComplianceVCMCapability, MergeCapability])

# AFTER:
self.add_capability_blueprints([
    ComplianceVCMCapability.bind(),
    MergeCapability.bind(),
])
```

### 7. Replace `action_policy_class` with `action_policy_blueprint` on `Agent`

```python
# Agent model field (line 1596)
# BEFORE:
action_policy_class: type[ActionPolicy] | None = None

# AFTER:
action_policy_blueprint: ActionPolicyBlueprint | None = None
```

### 8. Update `_create_action_policy()`

```python
async def _create_action_policy(self) -> None:
    if self.action_policy:
        return

    # Phase 1: Instantiate capability blueprints and add to agent
    for bp in self.capability_blueprints:
        if not self.has_capability(bp.key):
            capability = bp.local_instance(self)
            capability._capability_key = bp.key
            self.add_capability(
                capability,
                include_actions=bp.include_actions,
                exclude_actions=bp.exclude_actions,
                events_only=bp.events_only,
            )

    # Phase 2: Initialize all capabilities
    for capability in list(self._capabilities.values()):
        await capability.initialize()

    # Phase 3: Create action policy
    if self.action_policy_blueprint:
        # Blueprint policy — construct with agent + action_providers + policy config
        self.action_policy = self.action_policy_blueprint.local_instance(
            self,
            action_providers=list(self._capabilities.values()),
            **self.metadata.action_policy_config,
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

    # Phase 4: Mark capability blueprints as "used"
    self.action_policy.use_agent_capabilities(
        [bp.key for bp in self.capability_blueprints]
    )
    await self.action_policy.initialize()
```

### 9. Replace `AgentSpawnSpec` with `AgentBlueprint`

`AgentSpawnSpec` is deleted. `spawn_agents` takes `list[AgentBlueprint]`:

```python
# system.py — AgentSystemDeployment
@serving.endpoint
async def spawn_agents(
    self,
    agent_blueprints: list[AgentBlueprint],   # CHANGED from list[AgentSpawnSpec]
    session_id: str | None = None,
    run_id: str | None = None,
    soft_affinity: bool = True,
    suspend_agents: bool = False,
) -> list[str]:
```

Routing uses `blueprint.has_deployment_affinity()` (same method, now on `AgentBlueprint`).

### 10. Rewrite `_create_agent_instance()` — No More String Resolution

`_create_agent_instance` is deleted. `start_agent` calls `blueprint.local_instance()` directly:

```python
# In start_agent (AgentManagerBase):
agent = agent_blueprint.local_instance(agent_id=agent_id)
```

All constructor args (including `capability_blueprints`, `action_policy_blueprint`, `metadata`, `bound_pages`, `resource_requirements`) are already in `blueprint.kwargs` — set at `bind()` time. `agent_id` is the only extra kwarg injected at instantiation.

`_resolve_class_from_identifier` is kept for the resume path and `AgentPoolCapability` (LLM-initiated string-based spawning).

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
    self, agent_blueprint: AgentBlueprint,
    agent_id: str | None = None,
    suspend_agents: bool = False,
    max_iterations: int | None = None,
) -> str:
```

The `AgentBlueprint` carries the class and all constructor args (`metadata`, `capability_blueprints`, `action_policy_blueprint`, `bound_pages`, `resource_requirements`). Deployment params (`suspend_agents`, `max_iterations`) remain on `start_agent`. No string-based import resolution.

### 12. Update `AgentPoolCapability.create_agent()`

The LLM-facing `create_agent` action still takes a string `agent_type` (the LLM cannot construct Python objects). The capability resolves it to an `AgentBlueprint`:

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

    blueprint = agent_class.bind(
        metadata=metadata or AgentMetadata(
            tenant_id=self.agent.tenant_id,
            group_id=self.agent.group_id,
            parent_agent_id=self.agent.agent_id,
        ),
        bound_pages=bound_pages or [],
    )

    handles = await self.agent.spawn_child_agents(
        agent_blueprints=[blueprint],
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

Rename to `use_capability_blueprints`. Takes `list[AgentCapabilityBlueprint]`:

```python
async def use_capability_blueprints(self, capabilities: list[AgentCapabilityBlueprint]) -> None:
    for bp in capabilities:
        if not self.agent.has_capability(bp.key):
            instance = bp.local_instance(self.agent)
            instance._capability_key = bp.key
            await instance.initialize()
            self.agent.add_capability(instance)
        self.use_agent_capabilities([bp.key])
```

`use_agent_capabilities(list[str])` — unchanged, still works with capability keys.

## Callers Impacted — Full Audit

### Class-level `get_capability_name()` calls (must remain working)

These call `SomeClass.get_capability_name()` on the class, not an instance. They use it to look up capabilities by class identity. With the new design, these still work for **single-instance capabilities** (where key == class name). For multi-instance capabilities, callers should use `get_capability_by_type()` or `get_capabilities_by_class()`.

| File | Line | Code | Change needed? |
|------|------|------|----------------|
| `base.py` | 120 | `capability_cls.get_capability_name()` in `use_agent_capability_types` | Yes — method renamed to `use_capability_blueprints`, see section 14 |
| `base.py` | 133 | Same method | Yes — same |
| `base.py` | 967 | `capability_type.get_capability_name()` in `AgentHandle.get_capability` | No — AgentHandle creates its own instances keyed by class name, doesn't touch agent._capabilities |
| `base.py` | 1932 | `cap_class.get_capability_name()` in `_create_action_policy` | Replaced — uses `bp.key` |
| `base.py` | 1978 | `cap.get_capability_name() for cap in self.capability_classes` | Replaced — uses `bp.key for bp in self.capability_blueprints` |
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

### `add_capability_classes()` callers → migrate to `add_capability_blueprints()`

All callers switch from class lists to bound capability lists. Clean break — no backward compat wrapper.

| File | Before | After |
|------|--------|-------|
| `samples/code_analysis/compliance/agents.py` | `self.add_capability_classes([ComplianceVCMCapability, MergeCapability])` | `self.add_capability_blueprints([ComplianceVCMCapability.bind(), MergeCapability.bind()])` |
| `patterns/meta_agents/reputation_agent.py` | `self.add_capability_classes([...])` | `self.add_capability_blueprints([...bind()])` |
| `patterns/meta_agents/goal_alignment.py` | `self.add_capability_classes([...])` | `self.add_capability_blueprints([...bind()])` |

### `add_capability()` callers

These already create instances manually, so they just need to set `capability_key`.

| File | Code | Migration |
|------|------|-----------|
| `patterns/memory/defaults.py` | `agent.add_capability(working)` etc. | Set `capability_key=` in constructor |
| `base.py` | `_create_action_policy` | Replaced with `bp.local_instance()` |

### `AgentSpawnSpec` construction sites → migrate to `AgentBlueprint`

| File | Before | After |
|------|--------|-------|
| `cli/polymath.py:1564` | `AgentSpawnSpec(agent_type=coord_class, capabilities=..., metadata=..., bound_pages=[])` | `coord_class.bind(metadata=..., capability_blueprints=[...], bound_pages=[])` |
| `samples/.../code_analysis_agent_example.py:67` | `AgentSpawnSpec(agent_type="...", metadata=..., bound_pages=[])` | `CodeAnalysisCoordinator.bind(metadata=..., bound_pages=[])` |
| `samples/.../coordinator.py:427` | `AgentSpawnSpec(agent_type="...", metadata=..., bound_pages=..., resource_requirements=...)` | `ClusterAnalyzer.bind(metadata=..., bound_pages=..., resource_requirements=...)` |
| `agents/system.py:704` (resumption) | `AgentSpawnSpec(agent_type=..., bound_pages=..., resource_requirements=..., metadata=...)` | `agent_class.bind(bound_pages=..., resource_requirements=..., metadata=...)` |
| `patterns/capabilities/agent_pool.py:136` | `AgentSpawnSpec(agent_type=..., capability_types=..., ...)` (BUG) | `agent_class.bind(metadata=..., bound_pages=...)` |
| `patterns/memory/management.py:852` | `AgentSpawnSpec(agent_type="...", ...)` | `MemoryManagementAgent.bind(...)` |

### `action_policy_class` → `action_policy_blueprint`

| File | Before | After |
|------|--------|-------|
| `base.py:1965` | `self.action_policy_class(agent=self, action_providers=..., **self.metadata.action_policy_config)` | `self.action_policy_blueprint.local_instance(self, action_providers=..., **self.metadata.action_policy_config)` |
| Callers | `action_policy_class=SomePolicy` | `action_policy_blueprint=SomePolicy.bind(...)` |

### `spawn_child_agents()` callers

| File | Code | Migration |
|------|------|-----------|
| `patterns/capabilities/agent_pool.py` | `create_agent()` | Construct `AgentBlueprint` via `agent_class.bind(...)` instead of `AgentSpawnSpec` |
| Any coordinator agent | `self.spawn_child_agents(specs)` | `specs` is now `list[AgentBlueprint]` |

## Existing Bugs Fixed by This Refactoring

1. **Memory capabilities overwriting each other**: Six `MemoryCapability` instances all use key `"MemoryCapability"` → only one survives. Fixed by instance-level `capability_key`.

2. **`AgentSpawnSpec.capabilities` never forwarded**: `spawn_agents()` drops `spec.capabilities`. Eliminated — `AgentBlueprint` carries everything via `capability_blueprints` in kwargs.

3. **`AgentPoolCapability` wrong field name**: Passes `capability_types=` (nonexistent field on `AgentSpawnSpec`). Eliminated — `AgentSpawnSpec` is deleted; `AgentBlueprint.bind()` validates kwargs against `model_fields`.

4. **`_create_action_policy` can't pass constructor args**: `cap_class(self)` only passes agent. Fixed by `bp.local_instance(self)` which passes all bound kwargs.

5. **Duplicate `get_capability_names()`**: Defined twice on `Agent` (lines 1777 and 1795). Remove the duplicate during this refactoring.

6. **`spawn_child_agents` roles default bug**: `roles: list[str] = []` fails validation when `agent_blueprints` is non-empty. Fix: make the check conditional on `roles` being non-empty.

7. **String-based class resolution fragility**: `_resolve_class_from_identifier` guesses import paths. Eliminated for normal spawning — `Blueprint[T]` carries the class directly, cloudpickle handles serialization. Kept only for resume path and LLM-initiated spawning.

## Execution Order

### Phase 1: `Blueprint[T]` Infrastructure

1. Create `colony/python/colony/agents/blueprint.py` with `Blueprint[T]`, `AgentCapabilityBlueprint`, `AgentBlueprint`, `PolicyBlueprint`, error types
2. Add `.bind()` classmethod on `Agent`, `AgentCapability`, `ActionPolicy` (in `base.py`)
3. Add `capability_key` param to `AgentCapability.__init__()`, add `capability_key` property
4. Add `blueprint()` decorator in `blueprint.py`
5. Add `Agent.get_capabilities_by_class()` method

### Phase 2: Capabilities — Switch Dict Key and Field

6. Change `add_capability()` to use `capability.capability_key`
7. Update `create_default_memory_hierarchy()` to set `capability_key=` on each instance
8. Fix callers in `reflection.py` and `critique.py` → use `get_capability_by_type()`
9. Remove duplicate `get_capability_names()` definition
10. Replace `capability_classes` field with `capability_blueprints` on `Agent`
11. Delete `add_capability_classes()`, add `add_capability_blueprints()`
12. Update `_create_action_policy()` to iterate `capability_blueprints` using `bp.local_instance(self)`
13. Rename `use_agent_capability_types()` → `use_capability_blueprints()`, update callers

### Phase 3: Agents — Replace `AgentSpawnSpec` with `AgentBlueprint`

14. Replace `action_policy_class` field with `action_policy_blueprint` on `Agent`
15. Delete `AgentSpawnSpec` from `models.py`
16. Rewrite `spawn_agents()` to accept `list[AgentBlueprint]`; add `spawn_from_blueprint()` endpoint
17. Rewrite `start_agent()` to accept `AgentBlueprint`; call `blueprint.local_instance(agent_id=...)` directly
18. Delete `_create_agent_instance()` — no longer needed
19. Update `spawn_child_agents()` to accept `list[AgentBlueprint]`
20. Fix `spawn_child_agents` roles default bug
21. Update `AgentPoolCapability.create_agent()` to use `agent_class.bind(...)`

### Phase 4: Migrate All Callers

22. `polymath.py` — use `coord_class.bind(...)` instead of `AgentSpawnSpec`
23. `code_analysis_agent_example.py` — same
24. `coordinator.py` — same
25. `system.py` resumption path — resolve class then `.bind()`
26. `memory/management.py` — same
27. All `add_capability_classes` callers → `add_capability_blueprints`
28. All `action_policy_class=` callers → `action_policy_blueprint=`
29. Verify `AgentContextEngine.initialize()` still discovers all memory capabilities (it uses `isinstance`, so it should)

## Testing

1. **Unit test**: Create agent with multiple `MemoryCapability.bind(...).with_composition(key="stm")` and verify all are stored and retrievable.
2. **Unit test**: `AgentBlueprint` and `AgentCapabilityBlueprint` survive cloudpickle round-trip (simulating Ray transport).
3. **Unit test**: `AgentBlueprint` rejects unknown kwargs and excluded fields at bind time.
4. **Unit test**: `validate_serializable()` catches non-picklable args at bind time.
5. **Integration test**: Full compliance analysis run — verify all memory scopes present, no overwrites.
6. **Regression**: `get_capability_by_type(MemoryCapability)` returns first match (unchanged behavior).



----

> **Grok Prompt**: I am build a Python multi-agent library that runs on a Ray cluster. One agent is allowed to spawn another agent on a different Ray node. The child agent can build an arbitrarily complex hierarchy of complex components. So, we have a `Bound[T]` class to be used by the parent agent to specify the construction of the child agent and its components on its local Ray node and passes it to a `spawn_agent` function to materialize the child agent on the remote Ray node. The child agent can then use the `Bount[U]` objects associated with its components to materialize those components recursively on the same Ray node where the child agent is running. This pattern allows us to construct complex agent hierarchies across a Ray cluster while keeping the programming model intuitive and easy to use for the open-source community. I wonder if this pattern is common in the literature and if it has a name. Is it related to monads in any way? The `Bound[T]` class is intended to disguise the creation of a remote object as a local proxy object by capturing the constructor arguments and then passing it to a `spawn` function to construct the actual object (and its component hierarchy) on some remote node while enforcing the same strong typing and serialization requirements on the captured arguments as if they were passed to the eventual agent/component classes. I am bothered that the child agent class needs to be "aware" of this `Bound[T]` class and needs to explicitly call it to materialize its components. I am not sure if we can do better than this pattern, but I want to make sure we are not reinventing the wheel here. So, please do some research and let me know if this pattern has been studied before and if there are any known variations or improvements to it. Also, I want to make sure that this pattern is robust and can handle all the edge cases that may arise in distributed agent construction and management. So, please think through all the potential issues and propose any necessary modifications or improvements to the pattern to ensure its robustness and reliability in our multi-agent system.


**Your `Bound[T]` pattern is a standard and proven technique in distributed actor systems.** It closely matches the widely adopted <mark>"Props" (or **actor creation blueprint/descriptor**) pattern used in mature frameworks for safely specifying and materializing remote object hierarchies while preserving type safety and serialization guarantees.</mark>

Research across actor-model literature and frameworks (Akka, Erlang/OTP, Ray, Proto.Actor, and others) confirms this approach is common, intuitive, and battle-tested for building complex, cross-node agent/component trees. It is **not directly a monad**, though it can be viewed loosely as a <mark>delayed/bound computation</mark>; a separate <mark>"Remote Monad" pattern</mark> exists in Haskell for batching RPC calls, which is unrelated to object construction.

<mark>The requirement for child agents to explicitly recognize and materialize `Bound[U]` objects is **standard practice** (not a flaw) and promotes clarity,</mark> as seen in every major actor system where spawning uses explicit context methods. No major framework hides this entirely without sacrificing debuggability or type safety.

**Core Similarities to Existing Patterns**
Your design—<mark>capturing constructor arguments in a serializable proxy (`Bound[T]`), passing it to a `spawn_agent` materializer on a target node, and allowing recursive materialization of sub-components</mark>—is essentially identical to Akka's `Props` class. `Props` encapsulates the actor class, arguments, deployment info (e.g., remote node), and is passed to `actorOf(Props(...))` (local or remote). Children spawn via `context.actorOf(childProps)`. The same pattern appears in Erlang/OTP `child_spec` tuples (module + function + args) for supervised remote processes and in Ray's tree-of-actors supervisors (though <mark>Ray uses direct `@ray.remote` calls rather than an explicit bound object</mark>).

This enables **location-transparent hierarchies** without the parent needing to know remote details upfront, while enforcing serialization and typing exactly as you describe.

**Why This Pattern Persists**
Direct instantiation (`new Agent(...)`) fails remotely because the object must be created on the target node. <mark>A **bound descriptor** solves this by deferring construction, mirroring how factories or blueprints work in distributed settings (e.g., Kubernetes pod specs at the infrastructure level or Swift distributed actors' factory initializers).</mark>

**Relation to Monads**
No direct link. Monads structure sequenced computations with bind/return (e.g., for effects or state). Your `Bound[T]` is a data descriptor (immutable recipe), not a computation container. The closest functional analogue is a "thunk" or partial application, but actor frameworks treat it as a configuration object. The unrelated "Remote Monad" (Haskell) bundles multiple remote procedure calls into one packet to amortize latency—it does not construct object hierarchies.

**Robustness Assessment and Edge Cases**
The pattern is robust for most distributed agent use cases on Ray but requires explicit handling of a few distributed-system realities:

- **Serialization & Type Safety**: Captured args must be Ray-serializable (cloudpickle). Non-serializable closures or node-local objects will fail on spawn.
- **Code Availability**: Remote nodes must have identical class definitions (standard Ray cluster assumption).
- **Placement & Resources**: Without explicit options, Ray scheduler decides nodes; deep hierarchies may overload nodes.
- **Failures & Lifecycle**: Spawn exceptions, node crashes, or parent death need propagation and cleanup. Ray actor reference counting helps, but no built-in supervision tree.
- **Recursion & Scale**: Deep materialization can hit Python recursion limits; cycles in hierarchy specs are possible but undetected.
- **Explicit Awareness**: Child "knowing" `Bound` is fine and matches Akka/Ray norms—hiding it entirely (e.g., via heavy metaprogramming) often reduces maintainability for open-source users.

**Proposed Improvements (Minimal & Backward-Compatible)**
1. **Enhance `Bound[T]`**: Add `.options(resources=..., placement_group=..., node_affinity=...)` mirroring Ray's actor options—pass these to `spawn_agent`. <mark>Make `Bound` immutable and implement `__reduce__` for custom serialization.</mark>
2. **Supervision**: Include an optional `supervisor_strategy` in `Bound` (`one_for_one`, etc., like Akka/Erlang) and auto-wire it during materialization.
3. **Iterative Materialization**: Provide a non-recursive `materialize_tree(bound)` helper using a queue to handle arbitrary depth.
4. **Base Class Mixin**: Offer `AgentBase` with `self.materialize(b: Bound[U]) -> U` so children call a clean API without importing `Bound` directly (still explicit, but abstracted).
5. **Lazy/On-Demand Components**: For very complex trees, store `Bound` internally and materialize sub-components only on first access via descriptors.
6. **Error Wrapping & Retries**: `spawn_agent` should catch, wrap with remote-node info, and support configurable retries.
7. **Validation**: Runtime check that all nested `Bound` args are serializable before sending.

These keep the API intuitive for open-source users while addressing distributed edge cases. Your current design is already strong—no wheel reinvention; you're aligning with decades of actor-model best practices.

---

Your described `Bound[T]` abstraction for declarative, type-safe, serializable construction of remote agent hierarchies on a Ray cluster represents a canonical solution to <mark>a well-understood problem in distributed computing: how to specify and materialize complex object graphs across address spaces without violating encapsulation, typing, or serialization invariants.</mark> This pattern appears throughout actor-model literature and production frameworks, where direct object creation is impossible or unsafe for remote targets. It enables intuitive parent-driven specification of child construction (including arbitrary sub-component trees) while deferring actual instantiation to the target node—precisely as you implemented.

### Historical and Conceptual Context
The actor model, introduced by Carl Hewitt in 1973, inherently supports hierarchical creation: actors spawn children as part of their behavior. In distributed settings, this requires a "blueprint" mechanism because the spawning actor cannot execute the child's constructor locally. Early systems like Erlang/OTP formalized this with `child_spec` tuples of the form `{Id, {M, F, A}, Restart, Shutdown, Type, Modules}`, where `{M, F, A}` is a bound module-function-arguments descriptor passed to supervisors for remote spawning via `spawn(Node, M, F, A)`. This supports supervised hierarchies across nodes with restart policies.

Akka (and `Akka.NET/Pekko`) elevated this to the `Props` class—an immutable, serializable configuration object that captures the exact actor type, constructor arguments, deployment scope (local or remote), dispatcher, mailbox, router, and supervisor strategy. Actors are never created with `new`; instead, `system.actorOf(props)` or `context.actorOf(childProps)` is used. For remote deployment, `Props` can embed a `RemoteScope(address)` or rely on `application.conf` declarations (e.g., `/parent/child = { remote = "akka.tcp://system@host:port" }`). Hierarchies are fully supported: a remotely deployed parent can still spawn children via its context, with paths resolved across nodes.

Key properties of `Props` that mirror your `Bound[T]` exactly:
- Encapsulates constructor args (verified at creation).
- Immutable and freely shareable.
- Serializable for transmission to remote daemons (with strict rules: no closures capturing `this`; use companion-object factories).
- Supports remote materialization without code shipping (class must exist on target classpath—identical to Ray cluster assumptions).
- Children remain "aware" of the spawning API (`context.actorOf`), which is considered a feature for explicitness and debuggability.

Ray's native actor API is simpler (`@ray.remote class Agent: ...; agent = Agent.remote(*args)`), but the community "tree-of-actors" pattern uses a supervisor actor to spawn and manage worker sub-actors, exactly as you envision for agent hierarchies. Supervisors hold references to children created within their methods; Ray automatically terminates children if the supervisor dies (reference counting). No explicit `Bound` object exists, but your addition provides stronger typing and recursive sub-component support, which Ray lacks natively.

Other examples include Proto.Actor (Go/.NET) `Props` with remote kind registration, and Swift's distributed actors (factory initializers that run on the target system). In all cases, the pattern prioritizes a serializable descriptor over direct construction.

### Detailed Comparison of Patterns Across Frameworks

| Framework       | Creation Mechanism                  | Remote Support                          | Child/ Hierarchy Spawning                  | Serialization Requirements                  | Explicit Child Awareness? |
|-----------------|-------------------------------------|-----------------------------------------|--------------------------------------------|---------------------------------------------|---------------------------|
| Akka / Pekko   | `Props` (class + args + Deploy)    | `RemoteScope` or config; full hierarchy | `context.actorOf(props)`                   | Full (no enclosing closures); configurable | Yes (`context.actorOf`)  |
| Erlang/OTP     | `child_spec` tuple `{M,F,A,...}`   | `spawn(Node, M, F, A)`                 | Supervisor child specs (recursive)         | Erlang terms (built-in)                     | Yes (supervisor API)     |
| Ray (native)   | `@ray.remote().remote(*args)`      | Implicit scheduler; placement groups    | Manual in supervisor actor methods         | cloudpickle (lambdas supported)             | Implicit (direct calls)  |
| Your `Bound[T]`| Generic bound constructor proxy    | Explicit `spawn_agent` + Ray node       | Recursive `materialize` on child node      | Ray cloudpickle + strong typing             | Yes (explicit materialize) |
| Proto.Actor    | `PropsFromProducer` + remote kinds | gRPC-based remote spawning             | `context.Spawn(props)`                     | Custom serializer                           | Yes                      |

This table illustrates that your design is not only common but sits at the more sophisticated end by combining strong generics with recursive component binding—ideal for open-source multi-agent libraries.

### Why Children Must Be "Aware" of `Bound` (and Why That's Beneficial)
Every production system requires explicit spawning to integrate with supervision, naming, lifecycle, and placement. Hiding it (e.g., via heavy metaclasses that auto-materialize every attribute) introduces magic that complicates debugging, testing, and extension. <mark>Your explicit approach aligns with Akka's philosophy: "actors know they are actors."</mark> For usability, a thin mixin (`class Agent: def materialize(self, b: Bound[U]) -> U: ...`) reduces boilerplate without losing transparency.

### Thorough Edge-Case Analysis and Recommended Mitigations
Distributed construction introduces issues beyond single-node Python:

1. **Serialization Failures**: Deeply nested or non-cloudpickleable args (e.g., open files, unhashable custom objects).
   **Mitigation**: Add `Bound.validate_serializable()` using `ray.cloudpickle.dumps`; document factory pattern (companion methods returning `Bound`); support Ray `ObjectRef` args that are already in the object store.

2. **Code/Class Version Mismatch**: Remote node lacks updated agent class.
   **Mitigation**: Rely on Ray's cluster-wide environment consistency; add optional version hash in `Bound` for runtime check.

3. **Node/Placement Control**: Default scheduler may not respect "different Ray node" intent.
   **Mitigation**: Extend `Bound[T].options(placement_group=..., resources={"CPU": 2}, node_id=...)` and forward to Ray actor options in `spawn_agent`.

4. **Spawn Failures & Exceptions**: Constructor raises on remote, or node down.
   **Mitigation**: Wrap `spawn_agent` in try/except; return `Result[Agent, SpawnError]` or raise enriched exception with remote traceback; support retry policy in `Bound`.

5. **Hierarchy Depth & Cycles**: Recursion limit or infinite spawn loops.
   **Mitigation**: Internal iterative BFS materializer with depth limit and cycle detection (track materialized IDs).

6. **Lifecycle & Cleanup**: Parent dies → orphan children; no supervision.
   **Mitigation**: Implement optional supervision tree (inspired by Akka/Erlang) where `Bound` carries strategy; use Ray's `ray.kill` recursively on hierarchy roots.

7. **Concurrency & Ordering**: Multiple simultaneous spawns from one parent.
   **Mitigation**: Ray handles naturally; add optional `spawn_order: "sequential" | "parallel"` in `Bound`.

8. **Resource Leaks & Monitoring**: Long-lived hierarchies consume memory without visibility.
   **Mitigation**: Integrate with Ray dashboard; add `Bound` metadata for custom tags.

9. **Security/Isolation**: Arbitrary code in constructors.
   **Mitigation**: Internal library only; document that `Bound` args should be data-only.

With the minimal enhancements outlined earlier (options, supervision, iterative materialize, mixin), the pattern becomes production-grade and handles all common distributed edge cases while remaining approachable for the open-source community.

### Conclusion and Recommendations for Your Library
You are not reinventing the wheel—you are applying a time-tested, literature-backed pattern in a Python/Ray context with modern generics for better ergonomics. Document it explicitly as "inspired by Akka Props and Erlang child_spec" to build user trust. Start with the core `Bound[T] + spawn_agent` as-is, then layer the improvements as optional extensions (e.g., `Bound.with_options(...)`). This will make your multi-agent library both innovative in its recursive component support and reassuringly familiar to users coming from actor frameworks.

The pattern scales to arbitrarily complex hierarchies across Ray nodes exactly as you intend, and with the proposed robustness tweaks it will be reliable for real-world multi-agent workloads.

**Key Citations**
- Akka Classic Actors Documentation (Props and remote creation): https://doc.akka.io/libraries/akka-core/current/actors.html
- Akka Remoting Documentation (Props for remote deployment and hierarchies): https://doc.akka.io/libraries/akka-core/current/remoting-artery.html
- Akka.NET Props API Reference: https://getakka.net/api/Akka.Actor.Props.html
- Ray Tree-of-Actors Pattern: https://docs.ray.io/en/latest/ray-core/patterns/tree-of-actors.html
- Remote Monad Design Pattern Paper (for contrast): https://ku-fpg.github.io/files/Gill-15-RemoteMonad.pdf
- Erlang/OTP System Documentation (child_spec): https://www.erlang.org/docs/20/pdf/otp-system-documentation.pdf
- Ray Actors Documentation: https://docs.ray.io/en/latest/ray-core/actors.html

