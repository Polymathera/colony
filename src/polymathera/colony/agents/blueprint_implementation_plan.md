# Plan: Amend `capability_bind_proposal.md` — Blueprint[T] Redesign

> **First implementation step:** Amend `colony/python/colony/agents/capability_bind_proposal.md` with this redesign.

## Context

The previous `Bound[T]` implementation (git-stashed) had two fundamental problems:

1. **Mixed concerns**: `BoundAgent` stored Agent constructor args (`metadata`, `bound_pages`) alongside unrelated deployment params (`suspend_agents`, `max_iterations`). The `start_agent` method accepted both `BoundAgent` and deployment params like `metadata` and `resource_requirements` redundantly. `Bound[T]` became a grab-bag instead of a disciplined constructor thunk.

2. **No validation**: `Bound.__init__` accepted arbitrary `*args, **kwargs` — no type checking against the target class, no serializability check. Errors surfaced on remote nodes at construction time instead of at bind-site.

The user's instructions: rename to `Blueprint[T]`, add a `blueprint` decorator, enforce strong typing and serialization validation, and cleanly separate constructor args from deployment parameters via a `bind() → remote_instance()` API.

## Design: Three-Layer Separation

| Layer | What it captures | API | Example |
|-------|-----------------|-----|---------|
| **Constructor args** | Arguments to `T.__init__()` | `cls.bind(**kwargs)` → `Blueprint[T]` | `metadata=..., bound_pages=[...]` |
| **Composition** | How an instance is used by its parent | Fields on `AgentCapabilityBlueprint` | `key="stm", events_only=True` |
| **Deployment** | Where/how to spawn remotely | `bp.remote_instance(...)` | `requirements=..., session_id=...` |

## New File: `colony/python/colony/agents/blueprint.py`

### `Blueprint[T]` — Generic Base

```python
T = TypeVar("T")

class Blueprint(Generic[T]):
    """Deferred constructor: captures cls + keyword args for later instantiation.

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
                                f"List item in '{key}' for {self.cls.__name__} not serializable: {e}") from e
            else:
                try: cloudpickle.dumps(value)
                except Exception as e:
                    raise BlueprintSerializationError(
                        f"Argument '{key}' (type={type(value).__name__}) for "
                        f"{self.cls.__name__} not serializable: {e}") from e

    def local_instance(self, *prefix_args: Any, **extra_kwargs: Any) -> T:
        """Construct T locally. Called on the target deployment node.

        Does NOT auto-resolve nested Blueprints — domain code
        (Agent._create_action_policy) handles that explicitly.

        Args:
            *prefix_args: Prepended positional args (e.g., agent for capabilities).
            **extra_kwargs: Merged with bound kwargs (overrides on conflict).
        """
        merged = {**self.kwargs, **extra_kwargs}
        return self.cls(*prefix_args, **merged)
```

**Key decisions:**
- **No positional `args: tuple`**: For pydantic models, all construction is keyword-based. For `AgentCapability(agent, scope_id)`, `agent` is a prefix_arg injected at `local_instance` time — never bound.
- **`local_instance` does NOT auto-resolve nested Blueprints**: `capability_blueprints` and `action_policy_blueprint` on Agent are instantiated by `_create_action_policy` with domain-specific logic (composition metadata, action_providers injection).

### `AgentCapabilityBlueprint` — Constructor Args + Composition Metadata

```python
class AgentCapabilityBlueprint(Blueprint["AgentCapability"]):
    __slots__ = ("key", "include_actions", "exclude_actions", "events_only")

    def __init__(self, cls, kwargs, *, key=None,
                 include_actions=None, exclude_actions=None, events_only=False):
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

`bind()` captures ONLY constructor kwargs. Composition metadata is set via `.with_composition()`:

```python
# AgentCapability.bind() classmethod:
@classmethod
def bind(cls, **kwargs) -> AgentCapabilityBlueprint:
    bp = AgentCapabilityBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp
```

Usage — constructor args and composition are explicitly separated:
```python
MemoryCapability.bind(scope_id="agent:stm:w1", ttl_seconds=3600).with_composition(key="stm")
#                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
#                     constructor args → kwargs                    composition → self.key
```

### `ActionPolicyBlueprint` — No Extra Fields

```python
class ActionPolicyBlueprint(Blueprint["ActionPolicy"]):
    pass  # agent and action_providers injected via local_instance prefix/extra args
```

### `AgentBlueprint` — Pydantic Validation + `remote_instance`

```python
class AgentBlueprint(Blueprint["Agent"]):
    _EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "agent_id", "state", "created_at",
        "action_policy", "action_policy_state",
        "page_storage", "child_agents",
    })

    def __init__(self, cls: type[Agent], kwargs: dict[str, Any]):
        model_fields = set(cls.model_fields.keys())
        allowed = model_fields - self._EXCLUDED_FIELDS
        invalid = set(kwargs.keys()) - allowed
        if invalid:
            raise BlueprintValidationError(
                f"Invalid bind() arguments for {cls.__name__}: {invalid}. "
                f"Allowed: {sorted(allowed)}")
        super().__init__(cls, kwargs)

    def has_deployment_affinity(self) -> bool:
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
        Deployment parameters (not Agent constructor args) go here.
        Raises BlueprintRemoteError with remote traceback on failure."""
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

### `blueprint()` Decorator

```python
def blueprint(cls: type[T]) -> type[T]:
    """Add .bind() to any class. No-op if cls already has bind (via base class).
    For Agent/AgentCapability/ActionPolicy subclasses: returns cls as-is.
    For other classes: adds a generic Blueprint .bind() classmethod."""
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

Primary path: `Agent`, `AgentCapability`, `ActionPolicy` define `.bind()` directly on the base class — all subclasses inherit. `blueprint()` is available for non-standard classes: `blueprint(UserDefinedClass).bind(...)`.

### Error Types

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

## Changes to `colony/python/colony/agents/base.py`

### `.bind()` Classmethods

```python
# On Agent (BaseModel):
@classmethod
def bind(cls, **kwargs) -> "AgentBlueprint":
    from .blueprint import AgentBlueprint
    bp = AgentBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp

# On AgentCapability (ABC):
@classmethod
def bind(cls, **kwargs) -> "AgentCapabilityBlueprint":
    from .blueprint import AgentCapabilityBlueprint
    bp = AgentCapabilityBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp

# On ActionPolicy (ABC):
@classmethod
def bind(cls, **kwargs) -> "ActionPolicyBlueprint":
    from .blueprint import ActionPolicyBlueprint
    bp = ActionPolicyBlueprint(cls, kwargs)
    bp.validate_serializable()
    return bp
```

### Agent Model Field Changes

```python
# BEFORE:
capability_classes: list[type[AgentCapability]] = Field(default_factory=list)
action_policy_class: type[ActionPolicy] | None = None

# AFTER:
capability_blueprints: list[AgentCapabilityBlueprint] = Field(default_factory=list)
action_policy_blueprint: ActionPolicyBlueprint | None = None
```

### `capability_key` Property on AgentCapability

```python
def __init__(self, agent=None, scope_id=None, *, blackboard=None, capability_key=None):
    ...
    self._capability_key: str | None = capability_key

@property
def capability_key(self) -> str:
    return self._capability_key or self.get_capability_name()
```

`add_capability()` uses `capability.capability_key` as the `_capabilities` dict key.

### Updated `_create_action_policy`

```python
async def _create_action_policy(self) -> None:
    if self.action_policy: return

    # Phase 1: Instantiate capabilities from blueprints
    for bp in self.capability_blueprints:
        if not self.has_capability(bp.key):
            capability = bp.local_instance(self)  # agent as prefix_arg
            capability._capability_key = bp.key
            self.add_capability(capability,
                include_actions=bp.include_actions, exclude_actions=bp.exclude_actions,
                events_only=bp.events_only)

    # Phase 2: Initialize all capabilities
    for capability in list(self._capabilities.values()):
        await capability.initialize()

    # Phase 3: Create action policy
    if self.action_policy_blueprint:
        self.action_policy = self.action_policy_blueprint.local_instance(
            self, action_providers=list(self._capabilities.values()),
            **self.metadata.action_policy_config)
    else:
        from .patterns.actions.policies import create_default_action_policy
        self.action_policy = await create_default_action_policy(
            agent=self, action_map={},
            action_providers=list(self._capabilities.values()),
            max_iterations=self.metadata.max_iterations,
            **self.metadata.action_policy_config)

    self.action_policy.use_agent_capabilities([bp.key for bp in self.capability_blueprints])
    await self.action_policy.initialize()
```

### `start_agent` — Accept `AgentBlueprint`

```python
@serving.endpoint(router_class=SoftPageAffinityRouter,
    router_kwargs={"strip_routing_params": ["soft_affinity"]})
async def start_agent(
    self,
    agent_blueprint: AgentBlueprint,
    *,
    agent_id: str | None = None,
    suspend_agents: bool = False,
    max_iterations: int | None = None,
) -> str:
    async with self._agent_lock:
        if agent_id is None:
            agent_id = f"agent-{uuid.uuid4().hex[:8]}"

        resource_requirements = agent_blueprint.kwargs.get(
            "resource_requirements", AgentResourceRequirements())

        # ... resource checking (unchanged) ...

        agent = agent_blueprint.local_instance(agent_id=agent_id)
        agent.set_manager(self)
        await agent.initialize()
        self._agents[agent_id] = agent
        # ... track resources, start loop, register (unchanged) ...
```

`agent_id` is an `extra_kwarg` to `local_instance()` — the blueprint never contains it (`_EXCLUDED_FIELDS`). Metadata enrichment (session_id, run_id) happens in `spawn_from_blueprint` upstream.

## Changes to `colony/python/colony/agents/system.py`

### `spawn_from_blueprint` (replaces `spawn_agents`)

```python
@serving.endpoint
async def spawn_from_blueprint(
    self, blueprint: AgentBlueprint, *,
    requirements: Any | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    soft_affinity: bool = True,
    suspend_agents: bool = False,
    max_iterations: int | None = None,
) -> str:
    agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"

    # Enrich metadata with session/run tracking
    metadata = blueprint.kwargs.get("metadata", AgentMetadata())
    if isinstance(metadata, AgentMetadata):
        metadata = metadata.model_copy()
    if session_id: metadata.session_id = session_id
    if run_id: metadata.run_id = run_id
    if max_iterations is not None: metadata.max_iterations = max_iterations

    # Tenant quota check
    if metadata.tenant_id:
        resource_req = blueprint.kwargs.get("resource_requirements", AgentResourceRequirements())
        await self._check_tenant_quota(metadata.tenant_id, resource_req)

    # Create enriched blueprint copy (with session/run metadata)
    enriched_bp = AgentBlueprint(blueprint.cls, {**blueprint.kwargs, "metadata": metadata})

    # Route based on affinity
    has_affinity = bool(blueprint.kwargs.get("bound_pages")) or (requirements is not None)
    if has_affinity:
        vllm_name = await llm_cluster_handle.select_deployment(requirements=requirements)
        return await get_vllm_deployment(vllm_name).start_agent(
            enriched_bp, agent_id=agent_id, suspend_agents=suspend_agents,
            max_iterations=max_iterations, soft_affinity=soft_affinity)
    else:
        return await get_standalone_agents().start_agent(
            enriched_bp, agent_id=agent_id, suspend_agents=suspend_agents,
            max_iterations=max_iterations)
```

Batch `spawn_agents` calls `spawn_from_blueprint` in a loop. Resource exhaustion handlers updated to pass `AgentBlueprint` instead of `AgentSpawnSpec`.

## User-Facing API — Complete Example

```python
# Build agent blueprint — ONLY constructor args
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
    action_policy_blueprint=CacheAwareActionPolicy.bind(
        planner=my_planner,
        replanning_policy=PeriodicReplanningPolicy(),
    ),
)

# Spawn remotely — DEPLOYMENT params separate from constructor args
agent_id = await agent_bp.remote_instance(
    requirements=LLMClientRequirements(model_family="llama", min_context_window=32000),
    session_id="sess-123",
    run_id="run-456",
    max_iterations=50,
)
```

## Execution Order

### Phase 1: Blueprint Infrastructure
1. Create `colony/python/colony/agents/blueprint.py` — `Blueprint[T]`, `AgentCapabilityBlueprint`, `ActionPolicyBlueprint`, `AgentBlueprint`, `blueprint()`, error types
2. Add `.bind()` classmethod on `Agent`, `AgentCapability`, `ActionPolicy`
3. Add `capability_key` param + property on `AgentCapability`
4. Add `get_capabilities_by_class()` on `Agent`

### Phase 2: Agent Model + Policy
5. `capability_classes` → `capability_blueprints: list[AgentCapabilityBlueprint]`
6. `action_policy_class` → `action_policy_blueprint: ActionPolicyBlueprint | None`
7. `add_capability_classes()` → `add_capability_blueprints()`
8. `add_capability()` keyed by `capability.capability_key`
9. Rewrite `_create_action_policy()` to iterate `capability_blueprints`
10. `use_agent_capability_types()` → `use_capability_blueprints()`
11. Update `create_default_memory_hierarchy()` with `capability_key=`

### Phase 3: Spawn Pipeline
12. Replace `start_agent` on `AgentManagerBase` to accept `AgentBlueprint`
13. Add `spawn_from_blueprint` on `AgentSystemDeployment`
14. Replace `spawn_agents` to accept `list[AgentBlueprint]`
15. Update `spawn_child_agents`, `AgentHandle.from_agent_type`
16. Update resume path (`importlib` → `.bind()`)
17. Update resource exhaustion handlers
18. Delete `_create_agent_instance`, `AgentSpawnSpec`

### Phase 4: Migrate Callers
19. All `add_capability_classes` → `add_capability_blueprints`
20. All `AgentSpawnSpec(...)` → `cls.bind(...)`
21. All `action_policy_class=` → `action_policy_blueprint=`
22. `polymath.py`, sample agents, meta agents, game agents

## Verification

1. `python -c "import ast; ast.parse(open(f).read())"` for all modified files
2. `colony-env build`
3. Verify: `AgentBlueprint` rejects unknown kwargs at bind time
4. Verify: `validate_serializable()` catches non-picklable args at bind time
5. Verify: Nested blueprints (capabilities, policies) survive cloudpickle round-trip
