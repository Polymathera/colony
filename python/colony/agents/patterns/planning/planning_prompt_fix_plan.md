# Plan: Fix LLM Planning Prompt — Three Problems


## Context

When a `ComplianceAnalysisCoordinator` agent plans "Run Compliance Analysis on my-project", the LLM receives a prompt with:
1. A bare instruction: "Break down these goals into a hierarchical structure"
2. Goals: just the raw string
3. **Only memory actions** (store, recall, compact, drain, etc.)
4. Memory architecture guidance (verbose, ~8k tokens)

Missing: agent identity, domain actions, task parameters, constraints. The LLM responds with memory-only plans.

**Three root causes, three fixes:**

## Fix 1: Inherited Action Discovery (MRO Bug)

**Root cause:** `ActionDispatcher._create_object_action_group()` at [`policies.py:740`](colony/python/colony/agents/patterns/actions/policies.py#L740) uses `obj.__class__.__dict__.items()` which only scans the **immediate class**, not parent classes. `ComplianceVCMCapability` inherits 25+ `@action_executor` methods from `VCMAnalysisCapability` — none are discovered.

**File:** [`colony/python/colony/agents/patterns/actions/policies.py`](colony/python/colony/agents/patterns/actions/policies.py)

**Change:** Replace `obj.__class__.__dict__.items()` with an MRO walk:

```python
# line 740 — BEFORE:
for name, method in obj.__class__.__dict__.items():

# AFTER: Walk MRO to discover inherited @action_executor methods
seen_names: set[str] = set()
for cls in type(obj).__mro__:
    if cls is object:
        continue
    for name, method in cls.__dict__.items():
        if name in seen_names:
            continue  # Most-derived class wins
        seen_names.add(name)
        if hasattr(method, '_action_key'):
            # ... existing action_key filtering and executor creation logic ...
```

The `seen_names` set ensures that if a subclass overrides a parent method, the subclass version wins (standard MRO shadowing). The existing `include_filter`/`exclude_filter` logic applies unchanged.

## Fix 2: Agent Identity in Planning Prompt

**Root cause:** `_get_planning_context()` at [`policies.py:2400`](colony/python/colony/agents/patterns/actions/policies.py#L2400) only populates `goals`, `action_descriptions`, `recalled_memories`, and `memory_guidance`. It ignores `agent.metadata.role`, `agent.__class__.__doc__`, `agent.metadata.parameters`, and `constraints`. The `TopDownPlanningStrategy` at [`strategies.py:548`](colony/python/colony/agents/patterns/planning/strategies.py#L548) uses its own minimal inline prompt that never calls `_build_planning_prompt()`.

### 2a. Add `system_prompt` field to `PlanningContext`

**File:** [`colony/python/colony/agents/models.py`](colony/python/colony/agents/models.py) (line 563)

```python
class PlanningContext(BaseModel):
    # ... existing fields ...

    system_prompt: str = Field(
        default="",
        description="Stable agent identity context (role, capabilities, action reference). "
        "Separated for future KV-cache optimization.",
    )
```

### 2b. Populate `system_prompt` and `constraints` in `_get_planning_context()`

**File:** [`colony/python/colony/agents/patterns/actions/policies.py`](colony/python/colony/agents/patterns/actions/policies.py) (line 2400)

Add `_build_system_prompt()` and `_get_constraints()` to `CacheAwareActionPolicy`, call them from `_get_planning_context`:

```python
def _build_system_prompt(self) -> str:
    """Build stable agent identity prompt."""
    parts = []
    agent = self.agent

    # Agent identity
    identity = f"You are {agent.__class__.__name__}"
    if agent.metadata.role:
        identity += f" (role: {agent.metadata.role})"
    identity += f", a {agent.agent_type} agent."
    parts.append(identity)

    # Class docstring (first paragraph)
    doc = agent.__class__.__doc__
    if doc:
        parts.append(doc.strip().split('\n\n')[0].strip())

    # Capability summary
    cap_names = agent.get_capability_names()
    if cap_names:
        parts.append(f"Your capabilities: {', '.join(cap_names)}")

    # Task parameters (repo_id, analysis_type, etc.)
    params = agent.metadata.parameters
    if params:
        param_lines = [f"- {k}: {v}" for k, v in params.items()
                       if not k.startswith("_") and k != "planning_params"]
        if param_lines:
            parts.append("Task parameters:\n" + "\n".join(param_lines))

    return "\n\n".join(parts)

def _get_constraints(self) -> dict[str, Any]:
    """Extract execution constraints from agent metadata."""
    constraints = {}
    meta = self.agent.metadata
    if meta.max_iterations:
        constraints["max_iterations"] = meta.max_iterations
    params = meta.parameters
    if "max_agents" in params:
        constraints["max_parallel_workers"] = params["max_agents"]
    if "quality_threshold" in params:
        constraints["quality_threshold"] = params["quality_threshold"]
    return constraints
```

Update `_get_planning_context` to call both and populate `system_prompt` + `constraints`.

### 2c. All strategy prompts prepend `system_prompt`

**File:** [`colony/python/colony/agents/patterns/planning/strategies.py`](colony/python/colony/agents/patterns/planning/strategies.py)

**`_build_planning_prompt` (line 345):** Replace the generic "You are creating a plan to achieve multiple goals" opener with `{planning_context.system_prompt}`.

**`TopDownPlanningStrategy.generate_plan` (line 548):** Replace the two inline prompts. Add a shared `_build_decomposition_prompt()` to `PlanningStrategyPolicy`:

```python
def _build_decomposition_prompt(self, planning_context: PlanningContext, params: PlanningParameters) -> str:
    action_descriptions = self._format_action_descriptions(planning_context)
    custom_data_section = self._format_custom_data(planning_context)
    constraints_section = self._format_constraints(planning_context)

    return f"""{planning_context.system_prompt}

## Goals

{', '.join(planning_context.goals)}

## Available Actions

{action_descriptions}
{constraints_section}
{custom_data_section}

Break down the goals into a hierarchical structure.
Each leaf goal should map to one or more of the available actions.

The response must be valid JSON matching the expected schema."""
```

Similarly add `_build_action_planning_prompt()` for the step 2 call, and update `replan_horizon()` in all three strategies to prepend `system_prompt`.


## Fix 3: Prompt Length — Concise Action Descriptions for Planning

**Root cause:** Each memory action has a verbose multi-line description (~200-500 tokens). With 13+ memory actions + domain actions, this dominates the prompt. `MethodWrapperActionExecutor.get_action_description()` at [`policies.py:205`](colony/python/colony/agents/patterns/actions/policies.py#L205) returns the full docstring.

**Approach:** Add `planning_summary` to the `@action_executor` decorator and pipe it through.

### 3a. `@action_executor` decorator (line 491)

**File:** [`colony/python/colony/agents/patterns/actions/policies.py`](colony/python/colony/agents/patterns/actions/policies.py)

```python
def action_executor(
    action_key: str | ActionType | None = None,
    *,
    ...,
    planning_summary: str | None = None,  # NEW
):
    def decorator(func):
        ...
        func._action_planning_summary = planning_summary
        return func
    return decorator
```

### 3b. `MethodWrapperActionExecutor.__init__` (line 132)

Add `planning_summary` param, store as `self.planning_summary`:

```python
def __init__(self, ..., planning_summary: str | None = None):
    ...
    self.planning_summary = planning_summary
```

### 3c. `_create_object_action_group` (line 752)

Pass it through when creating `MethodWrapperActionExecutor`:

```python
executor = MethodWrapperActionExecutor(
    ...,
    planning_summary=getattr(method, '_action_planning_summary', None),
)
```

### 3d. `get_action_description()` (line 205)

Prefer `planning_summary` over full docstring:

```python
async def get_action_description(self) -> str:
    if self.planning_summary:
        return self.planning_summary
    docstring = inspect.getdoc(self.method)
    if not docstring:
        raise ValueError(...)
    return docstring
```

### 3e. Add `planning_summary` to verbose memory actions

**File:** [`colony/python/colony/agents/patterns/memory/capability.py`](colony/python/colony/agents/patterns/memory/capability.py) (where `MemoryCapability` defines its `@action_executor` methods)

Add `planning_summary=` to each verbose `@action_executor`. Example:

```python
@action_executor(action_key="store", planning_summary="Store content in memory with optional tags and metadata.")
```

No changes to method bodies — just a decorator parameter on the verbose ones. Do the same for `FunctionWrapperActionExecutor` for consistency.

## Files Modified

| File | Change |
|------|--------|
| [`policies.py`](colony/python/colony/agents/patterns/actions/policies.py) | Fix MRO walk in `_create_object_action_group`. Add `planning_summary` to `@action_executor`, `MethodWrapperActionExecutor`, `FunctionWrapperActionExecutor`. Add `_build_system_prompt()`, `_get_constraints()` to `CacheAwareActionPolicy`. |
| [`models.py`](colony/python/colony/agents/models.py) | Add `system_prompt` field to `PlanningContext` |
| [`strategies.py`](colony/python/colony/agents/patterns/planning/strategies.py) | Prepend `system_prompt` in all prompts. Add `_build_decomposition_prompt()`, `_build_action_planning_prompt()`. Remove `memory_architecture_guidance` hard validation. |
| [`capability.py`](colony/python/colony/agents/patterns/memory/capability.py) | Add `planning_summary=` to verbose memory `@action_executor` decorators |

## Execution Order

1. **Fix 1 first** (MRO bug) — this is the most impactful and simplest change
2. **Fix 2** (prompt structure) — depends on Fix 1 being done so we can verify domain actions appear
3. **Fix 3** (prompt length) — polish pass, can be done incrementally

## Verification

1. `colony-env build`
2. `colony-env run /path/to/codebase --config test.yaml`
3. Check logs for the planning prompt:
   - Domain actions (spawn_worker, merge_results, build_obligation_graph, etc.) appear alongside memory actions
   - Agent identity block at top: "You are ComplianceAnalysisCoordinator (role: coordinator)..."
   - Class docstring, capability list, task parameters, constraints all present
   - Total prompt length reduced from ~14k to ~8-10k tokens
