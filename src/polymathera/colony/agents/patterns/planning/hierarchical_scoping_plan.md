# Plan: Hierarchical Action Scoping — Two-Phase Planning Prompt Reduction

## Context

The planning LLM prompt includes ALL action descriptions from ALL capabilities on every agent loop iteration. With 16+ capabilities and 92+ `@action_executor` methods, the "Available Actions" section alone is ~15-25k tokens. Combined with memory guidance (~8-12k), recalled memories (~3-10k), and execution context, prompts reach 30-50k tokens per call. This is called every iteration — extremely expensive.

**Goal**: Reduce the planning prompt to ~7-20k tokens via a two-phase approach: (1) a lightweight scope selection call where the LLM picks relevant action groups, (2) the actual planning call with only the selected groups' actions.

## Design: Two-Phase Planning with Scope Selection

### Phase 1 — Scope Selection (~2-5k tokens)
- Prompt includes ONLY group-level summaries (one line per capability: key, description, action count, tags)
- LLM selects which groups are relevant for the current goals
- Fast call: `max_tokens=512`, `temperature=0.1`

### Phase 2 — Action Planning (~5-15k tokens)
- Prompt includes ONLY actions from selected groups + full context (memories, guidance, etc.)
- This is the existing planning prompt with a filtered action set

### Skip Threshold
When total groups ≤ `SCOPE_SELECTION_THRESHOLD` (default 6), skip Phase 1 entirely — the overhead of the extra call exceeds the savings.

### Replanning
Cache `selected_groups` on the `ActionPlan`. Reuse for `replan_horizon()`. Re-select only when goals change or execution context shifts significantly.

---

## Changes

### 1. Add `group_key` and `tags` to `ActionGroup`

**File**: `colony/python/colony/agents/patterns/actions/policies.py:616`

```python
class ActionGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    group_key: str = Field(description="Stable identifier for this action group.")
    description: str = Field(description="Description of the action group.")
    executors: dict[str, ActionExecutor] = Field(default_factory=dict)
    tags: frozenset[str] = Field(default_factory=frozenset, description="Domain/modality/cost tags for filtering.")
```

`group_key` is set in `_create_object_action_group()` from `capability_key` (for capabilities) or `f"{ClassName}.{dispatch_key}"` (for agent/policy objects).

### 2. Add `get_capability_tags()` to `AgentCapability`

**File**: `colony/python/colony/agents/base.py` — on `AgentCapability` class

```python
@classmethod
def get_capability_tags(cls) -> frozenset[str]:
    """Return domain/modality tags for hierarchical action scoping.

    Override to provide tags. Tags are free-form strings:
    "memory", "analysis", "coordination", "synthesis", "expensive", etc.
    """
    return frozenset()
```

Class method — tags describe the type, not the instance. Subclasses override. No constructor changes.

### 3. Add `tags` and `group_description` to `AgentCapabilityBlueprint`

**File**: `colony/python/colony/agents/blueprint.py:162`

Add `tags` and `group_description` to `__slots__` and `__init__`. `tags` overrides `cls.get_capability_tags()`. `group_description` overrides the instance's `get_action_group_description()` return value — this is critical because multiple instances of the same capability (e.g., MemoryCapability for STM vs LTM) need distinct descriptions for the scope selection LLM. Also update `with_composition()`.

### 4. Propagate tags in `_create_object_action_group()`

**File**: `colony/python/colony/agents/patterns/actions/policies.py:770`

After building the `ActionGroup`, resolve tags and description from the capability:
- Tags: check `obj._action_tags` (blueprint override) first, then `obj.get_capability_tags()` if available
- Description: check `obj._action_group_description` (blueprint override) first, then `obj.get_action_group_description()`
- Set both on `ActionGroup`

Also propagate blueprint metadata in `Agent.add_capability()` (base.py, in `_create_action_policy` Phase 1):
- `capability_instance._action_tags = bp.tags` (when tags is not None)
- `capability_instance._action_group_description = bp.group_description` (when group_description is not None)

### 5. Add `tags` to `@action_executor` decorator

**File**: `colony/python/colony/agents/patterns/actions/policies.py:538`

Add optional `tags: frozenset[str] | None = None`. Store as `func._action_tags`. This is for future per-action tag filtering — not used in scope selection but available for the lattice extension.

### 6. Add `get_action_group_summaries()` to `ActionDispatcher`

**File**: `colony/python/colony/agents/patterns/actions/policies.py` — after `get_action_descriptions()` (line 914)

```python
def get_action_group_summaries(self) -> list[tuple[str, str, frozenset[str], int]]:
    """Lightweight group summaries for scope selection.

    Returns:
        List of (group_key, description, tags, plannable_action_count).
    """
    summaries = []
    for group in self.action_map:
        plannable = sum(1 for e in group.executors.values()
                        if not getattr(e, 'exclude_from_planning', False))
        if plannable > 0:
            summaries.append((group.group_key, group.description, group.tags, plannable))
    return summaries
```

### 7. Add `selected_groups` filter to `get_action_descriptions()`

**File**: `colony/python/colony/agents/patterns/actions/policies.py:863`

Add optional `selected_groups: list[str] | None = None` parameter. When provided, skip groups whose `group_key` is not in the list. REPL descriptions are always included (**meta-capability**, never filtered).

Also update the pass-through on `BaseActionPolicy.get_action_descriptions()` at line 1503.

### 8. Replace `action_descriptions` tuple type with `ActionGroupDescription` model

**File**: `colony/python/colony/agents/models.py`

Add new model and update `PlanningContext`:

```python
class ActionGroupDescription(BaseModel):
    """Describes one action group for the planning LLM.

    Each group corresponds to a capability or provider instance.
    Used both for full action descriptions (action_descriptions populated)
    and for lightweight summaries (action_descriptions empty, action_count set).
    """
    group_key: str = Field(description="Stable identifier for this action group.")
    group_description: str = Field(description="Human-readable description of the group.")
    action_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of full action names to their descriptions."
    )
    tags: frozenset[str] = Field(default_factory=frozenset)
    action_count: int = Field(default=0, description="Number of plannable actions in this group.")
```

```python
# Before:
action_descriptions: list[tuple[str, dict[str, str]]]

# After:
action_descriptions: list[ActionGroupDescription]
```

Update `ActionDispatcher.get_action_descriptions()` to return `list[ActionGroupDescription]`.

Update `_format_action_descriptions()` in `strategies.py:173`:

```python
for group in planning_context.action_descriptions:
    if group.group_description:
        sections.append(f"\n### {group.group_description}")
    for action_key, action_desc in group.action_descriptions.items():
        sections.append(f"- {action_key}: {action_desc}")
```

Also add `get_action_group_summaries()` on `ActionDispatcher` that returns `list[ActionGroupDescription]` with empty `action_descriptions` but populated `action_count` — same model, lightweight variant.

### 9. Add `action_group_summaries` and `selected_groups` to `PlanningContext`

**File**: `colony/python/colony/agents/models.py` — on `PlanningContext`

```python
action_group_summaries: list[tuple[str, str, frozenset[str], int]] = Field(
    default_factory=list,
    description="Lightweight group summaries: (group_key, description, tags, action_count)."
)

selected_groups: list[str] | None = Field(
    default=None,
    description="Group keys selected by scope selection. None = all groups."
)
```

### 10. Add `selected_groups` to `ActionPlan`

**File**: wherever `ActionPlan` is defined (planning models)

Add `selected_groups: list[str] | None = None` field. Cached from scope selection, reused by `replan_horizon()`.

### 11. Add scope selection to `PlanningStrategyPolicy` base class

**File**: `colony/python/colony/agents/patterns/planning/strategies.py`

#### New response model (after existing response models, ~line 97):

```python
class ScopeSelectionResponse(BaseModel):
    reasoning: str = Field(description="Brief reasoning for scope selection")
    selected_groups: list[str] = Field(description="group_key values for relevant action groups")
```

#### New constant:

```python
SCOPE_SELECTION_THRESHOLD: int = 6
```

#### New methods on `PlanningStrategyPolicy`:

**`_format_action_group_summaries(planning_context)`** — static, formats one line per group:
```
- [group_key] (N actions, tags: tag1, tag2): description
```

**`_build_scope_selection_prompt(planning_context)`** — lightweight prompt with: system_prompt, goals, minimal execution context, group summaries. NO action descriptions, NO memory guidance, NO recalled memories. Asks LLM to select relevant groups.

**`_should_use_scope_selection(planning_context)`** — returns `len(action_group_summaries) > SCOPE_SELECTION_THRESHOLD`.

**`_run_scope_selection(planning_context)`** — calls `agent.infer()` with scope selection prompt, parses `ScopeSelectionResponse`, returns `list[str]` of selected group keys. On parse failure, falls back to all groups.

### 12. Wire up `_get_planning_context` to populate summaries

**File**: `colony/python/colony/agents/patterns/actions/policies.py:2550`

In `_get_planning_context()`:
1. Call `self._action_dispatcher.get_action_group_summaries()` → store in context
2. Default: still fetch all `action_descriptions` (backward compatible)
3. Add an overload/parameter for pre-filtered fetch: when `selected_groups` is passed, call `self.get_action_descriptions(selected_groups=selected_groups)` instead

### 13. Integrate scope selection into strategy `generate_plan()` methods

**MPC** (`strategies.py:556`):
```python
async def generate_plan(self, planning_context, params, ...):
    # Phase 1: Scope selection (if beneficial)
    if self._should_use_scope_selection(planning_context):
        selected = await self._run_scope_selection(planning_context)
        # Re-fetch filtered action descriptions
        planning_context.action_descriptions = [
            (k, d, a) for k, d, a in planning_context.action_descriptions
            if k in selected
        ]
        planning_context.selected_groups = selected

    # Phase 2: Action planning (with filtered actions)
    prompt = self._build_planning_prompt(...)
    ...
```

**TopDown** (`strategies.py:720`):
- Step 1 (decomposition): Use a new `_build_decomposition_prompt_with_summaries()` that shows group summaries instead of full actions. This is cheaper than even scope selection — no extra LLM call needed for decomposition.
- Step 2 (action planning): Run scope selection, filter, then `_build_action_planning_prompt()`.

**BottomUp** (`strategies.py:819`): Same pattern as MPC.

### 14. Integrate scope selection into `replan_horizon()` methods

For all three strategies:
- If `plan.selected_groups` exists, filter `planning_context.action_descriptions` using it (no extra LLM call)
- If `plan.selected_groups` is None, skip filtering (backward compatible)
- For future: detect goal drift → re-run scope selection (not in this PR)

---

## Files Modified (summary)

| File | Changes |
|------|---------|
| `agents/base.py` | Add `get_capability_tags()` classmethod to `AgentCapability`. Propagate `_action_tags` in `add_capability()`. |
| `agents/blueprint.py` | Add `tags` slot to `AgentCapabilityBlueprint`, update `with_composition()`. |
| `agents/models.py` | Change `action_descriptions` tuple type (add group_key). Add `action_group_summaries`, `selected_groups` to `PlanningContext`. |
| `agents/patterns/actions/policies.py` | Add `group_key`, `tags` to `ActionGroup`. Add `get_action_group_summaries()`. Add `selected_groups` filter to `get_action_descriptions()`. Wire summaries in `_get_planning_context()`. |
| `agents/patterns/planning/strategies.py` | Add `ScopeSelectionResponse`, `SCOPE_SELECTION_THRESHOLD`, `_format_action_group_summaries()`, `_build_scope_selection_prompt()`, `_should_use_scope_selection()`, `_run_scope_selection()`, `_build_decomposition_prompt_with_summaries()`. Integrate into `generate_plan()` and `replan_horizon()` for all 3 strategies. |
| `agents/patterns/planning/__init__.py` | Export `ScopeSelectionResponse`, `SCOPE_SELECTION_THRESHOLD`. |
| Planning models (ActionPlan) | Add `selected_groups: list[str] | None = None`. |

## What Is NOT Changed

- `_format_custom_data()` — completely untouched
- `memory_architecture_guidance` validation — untouched
- `ActionDispatcher.dispatch()` — unchanged (all executors remain registered, scope selection only affects the prompt)
- Existing `@action_executor` usage — the new `tags` parameter is optional
- `SequentialPlanner` — no LLM planning, unchanged

## Execution Order

1. **Data layer**: `ActionGroup.group_key/tags`, `@action_executor` tags, `get_capability_tags()`, `AgentCapabilityBlueprint.tags`, propagation in `_create_object_action_group()`
2. **Query layer**: `get_action_group_summaries()`, filtered `get_action_descriptions(selected_groups=)`, `action_descriptions` type change, `_format_action_descriptions()` update
3. **Model layer**: `PlanningContext` new fields, `ActionPlan.selected_groups`
4. **Selection layer**: `ScopeSelectionResponse`, `_build_scope_selection_prompt()`, `_run_scope_selection()`, `_should_use_scope_selection()`
5. **Integration**: Wire into `_get_planning_context()`, `generate_plan()`, `replan_horizon()` for all 3 strategies, `_build_decomposition_prompt_with_summaries()` for TopDown

## Verification

1. `python -c "import ast; ast.parse(open(f).read())"` for all modified files
2. Verify `ActionGroup` construction in `_create_object_action_group()` sets `group_key` correctly
3. Verify `get_action_group_summaries()` returns correct count excluding `exclude_from_planning` actions
4. Verify `get_action_descriptions(selected_groups=["key1"])` filters correctly
5. Verify `_format_action_descriptions()` handles new 3-tuple format
6. `colony-env build` to verify full build
7. End-to-end: agent with 10+ capabilities → observe scope selection in logs → verify reduced prompt length
