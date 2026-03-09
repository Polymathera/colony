# Plan: Four Issues — Scope Selection Context, Action History, `spawn_workers` Validation, UI Tab State

## Issue 1: Scope Selection Prompt Lacks Agent Context

### Problem

`_build_scope_selection_prompt` (`strategies.py`:470-509) produces a bare prompt with only goals and group summaries. It omits:
- System prompt (agent identity, role, constraints, world model)
- Self-concept
- Recalled memories
- Constraints
- Custom data (including `memory_architecture_guidance`)

The main planning prompt (`_build_planning_prompt`, line 881) includes all of these. Without role/identity context, the scope selector doesn't know what kind of agent it's filtering for — a compliance agent vs an impact analysis agent would need entirely different action groups, but the scope selector can't distinguish them.

### Root Cause

The scope selection prompt was designed to be "lightweight" to minimize token overhead. But "lightweight" was taken too far — it excluded the agent's identity entirely.

### Fix

Add `planning_context.system_prompt` to the scope selection prompt. The system prompt already contains the agent's role, identity, constraints, and goals — it's the single source of identity context. This is the minimal addition that fixes the problem without bloating the prompt.

**File**: `colony/python/colony/agents/patterns/planning/strategies.py`

In `_build_scope_selection_prompt` (~line 487), change the prompt to include the system prompt:

```python
return f"""{planning_context.system_prompt}

## Task: Select Relevant Action Groups

{exec_hint}

## Available Action Groups

{summaries_str}

## Instructions

Select the action groups that are relevant to the current goals. Include groups whose actions
might be needed for planning. When in doubt, include the group — it is better to include an
extra group than to miss a needed one.

Respond with ONLY a JSON object (no markdown fences, no surrounding text):

{{
  "reasoning": "<brief reasoning for your selection>",
  "selected_groups": ["<group_key_1>", "<group_key_2>", ...]
}}"""
```

**Why not add recalled memories / custom data?** The system prompt already contains goals and identity. Recalled memories are better reserved for Phase 2 (action planning) where the LLM needs detailed context to plan specific actions. Scope selection only needs to know *what kind of agent* is asking.

**Update docstring** to reflect the new design:
```python
"""Build a lightweight prompt for scope selection (Phase 1).

Includes: system prompt (agent identity, role, goals, constraints), minimal
execution context, and group summaries.
Omits: full action descriptions, recalled memories, custom data — those go in Phase 2.
"""
```

---

## Issue 2: Action History Not Reaching Planning Prompts

### Problem

When replanning, the planner repeats already-executed actions because it doesn't see what was already done.

### What the Design Intended

Per the memory specs (`MEMORY_SYSTEM_ITERATIONS.md`, `MEMORY_MAP.md`):
1. `ActionDispatcher.dispatch()` completes → `@hookable` fires
2. `WorkingMemoryCapability` has a `MemoryProducerConfig` with `Pointcut.pattern("ActionDispatcher.dispatch")` and `extract_action_from_dispatch` extractor
3. The hook handler stores the Action (with result) into working memory
4. `AgentContextEngine.gather_context()` queries all memory capabilities including working memory
5. Action history appears in `recalled_memories` in the planning prompt

### What Actually Happens

The infrastructure is fully wired:
- `create_default_memory_hierarchy()` creates `WorkingMemoryCapability` with the `ActionDispatcher.dispatch` producer hook (defaults.py:176-182)
- `_register_producer_hooks()` registers AFTER hooks on the agent's hook registry (`capability.py`:1145-1174)
- `_create_producer_hook_handler()` extracts and stores data (`capability.py`:1176+)
- `Agent.initialize()` calls `initialize_memory_hierarchy()` which calls `create_default_memory_hierarchy()` (`base.py`:2041-2042)

**So actions SHOULD be flowing into working memory and appearing in `recalled_memories`.** The question is: are they actually showing up in the planning prompt?

### Investigation Needed

Before coding a fix, we need to verify whether the pipeline is actually broken or just formatting poorly. The issue could be:

1. **Hook not firing**: `dispatch()` might not be `@hookable` or the pointcut pattern might not match
2. **Extractor returning None**: `extract_action_from_dispatch` skips if `result is None` (`capability.py`:1188)
3. **`gather_context` query too narrow**: The `MemoryQuery(max_results=50)` in `_gather_planning_context()` might not match action entries
4. **Formatting hides actions**: `_format_recalled_memories` formats entries generically — action entries might not have `content` or `text` keys, so they'd show as raw dict strings
5. **Timing**: Actions stored after the planning context is gathered

### Diagnostic Step

Add a temporary log in `_gather_planning_context()` (`policies.py`:2353) to see what `gather_context()` returns:

```python
entries = await ctx_engine.gather_context(query=MemoryQuery(max_results=50))
logger.warning(f"gather_context returned {len(entries)} entries: {[e.key for e in entries]}")
```

### Likely Fix

If actions ARE in memory but not formatted well, fix `_format_recalled_memories` to handle Action objects:

```python
@staticmethod
def _format_recalled_memories(planning_context: PlanningContext) -> str:
    if not planning_context.recalled_memories:
        return ""
    lines = ["## Recalled Memories"]
    for mem in planning_context.recalled_memories[:10]:
        # Handle action entries stored by memory producer hooks
        value = mem.get("value")
        if hasattr(value, "action_type"):
            status = "completed" if (value.result and value.result.success) else "failed"
            lines.append(f"- [action] {value.action_type}: {value.description} → {status}")
            continue
        content = mem.get("content", mem.get("text", str(mem)))
        scope = mem.get("scope", "")
        prefix = f"[{scope}] " if scope else ""
        lines.append(f"- {prefix}{content}")
    return "\n".join(lines)
```

If actions are NOT in memory at all, the hook pipeline is broken and we need to debug `_register_producer_hooks` → pointcut matching → handler invocation.

### The Replanning Prompt Already Has Action History (Partially)

`_build_replanning_prompt` (line 1130) already includes completed actions from `plan.actions`. But:
- It only shows the last 3 action results (line 950)
- It shows action type + description + success/fail, not the actual findings
- For **initial planning** after a previous plan completed, execution_context starts empty

The deeper issue: when `_replan_horizon()` is called (per the lifecycle redesign plan), `revise_plan()` preserves completed actions. But `generate_plan()` for a fresh plan has no history. The memory system is supposed to bridge this gap.

---

## Issue 3: `spawn_workers` `**domain_params` Validation Error

### Problem

```
Parameter validation failed: 1 validation error for spawn_workers_Input
domain_params Field required [type=missing, ...]
```

### Root Cause

`_infer_input_schema()` (policies.py:419-470) iterates `sig.parameters` but doesn't check `param.kind`. When it encounters `**domain_params` (a `VAR_KEYWORD` parameter), it treats it as a regular required field because `param.default is inspect.Parameter.empty`. The generated Pydantic model requires a `domain_params` field, but the LLM generates flat kwargs like `{page_ids: [...], quality_threshold: 0.7}`.

### Fix

**File**: `colony/python/colony/agents/patterns/actions/policies.py`

In `_infer_input_schema()` (~line 435), skip `VAR_KEYWORD` and `VAR_POSITIONAL` parameters:

```python
for name, param in sig.parameters.items():
    if name == 'self':
        continue
    # Skip *args and **kwargs — these can't be represented as fixed schema fields
    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        continue

    field_type = hints.get(name, Any)
    if param.default is inspect.Parameter.empty:
        fields[name] = (field_type, ...)
    else:
        fields[name] = (field_type, param.default)
```

**But wait** — this means `domain_params` content (like `quality_threshold`, `granularity`) won't be in the schema at all. The LLM won't know to include them, and Pydantic will reject them as extra fields.

**Better fix**: Skip `VAR_KEYWORD` in schema generation AND configure the generated model to allow extra fields:

```python
if not fields:
    return None

# If the function accepts **kwargs, allow extra fields in the model
has_var_keyword = any(
    p.kind == inspect.Parameter.VAR_KEYWORD
    for p in sig.parameters.values()
)

try:
    if has_var_keyword:
        # Allow arbitrary extra fields to flow through as **kwargs
        model = create_model(
            f"{func.__name__}_Input",
            __config__=type("Config", (), {"extra": "allow"}),
            **fields,
        )
    else:
        model = create_model(f"{func.__name__}_Input", **fields)
```

This way:
- The schema includes the named parameters (`page_ids`, `cache_affine`, `max_parallel`)
- Extra fields from the LLM (`quality_threshold`, `granularity`) pass validation and flow through as `**kwargs`

### Action Description Fix

The action description for `spawn_workers` should also document the domain-specific parameters so the LLM knows what to include. This is a separate concern — the description comes from the docstring and `@action_executor` metadata.

---

## Issue 4: UI Trace Tab Loses Expand/Collapse State

### Problem

`AppShell.tsx` uses a switch statement in `TabContent` (line 30-57) to render tab components. Switching tabs unmounts the old component and mounts the new one. The `collapsed` state in `TraceWaterfallView` (TracesTab.tsx:733) is `useState<Set<string>>` — component-local state that's destroyed on unmount.

### Fix

**Approach**: Keep all tab components mounted but hide inactive ones with CSS. This is the simplest fix that preserves ALL state for ALL tabs (not just traces), costs nothing to implement, and avoids introducing state management infrastructure.

**File**: `colony/python/colony/web_ui/frontend/src/components/layout/AppShell.tsx`

Replace the switch-based `TabContent` with render-all-hide-inactive:

```tsx
function TabContent({ activeTab }: { activeTab: string }) {
  return (
    <>
      <div style={{ display: activeTab === "overview" ? "block" : "none" }}><OverviewTab /></div>
      <div style={{ display: activeTab === "agents" ? "block" : "none" }}><AgentsTab /></div>
      <div style={{ display: activeTab === "sessions" ? "block" : "none" }}><SessionsTab /></div>
      <div style={{ display: activeTab === "vcm" ? "block" : "none" }}><VCMTab /></div>
      <div style={{ display: activeTab === "graph" ? "block" : "none" }}><PageGraphTab /></div>
      <div style={{ display: activeTab === "blackboard" ? "block" : "none" }}><BlackboardTab /></div>
      <div style={{ display: activeTab === "interact" ? "block" : "none" }}><InteractTab /></div>
      <div style={{ display: activeTab === "logs" ? "block" : "none" }}><LogsTab /></div>
      <div style={{ display: activeTab === "traces" ? "block" : "none" }}><TracesTab /></div>
      <div style={{ display: activeTab === "metrics" ? "block" : "none" }}><MetricsTab /></div>
      <div style={{ display: activeTab === "settings" ? "block" : "none" }}><SettingsTab /></div>
    </>
  );
}
```

**Consideration**: All tabs mount on initial load. If some tabs make expensive API calls on mount, they'll fire immediately. If this is a concern, use lazy mounting:

```tsx
function TabContent({ activeTab }: { activeTab: string }) {
  const [mounted, setMounted] = useState<Set<string>>(new Set(["overview"]));

  useEffect(() => {
    setMounted(prev => {
      if (prev.has(activeTab)) return prev;
      const next = new Set(prev);
      next.add(activeTab);
      return next;
    });
  }, [activeTab]);

  return (
    <>
      {mounted.has("overview") && <div style={{ display: activeTab === "overview" ? "block" : "none" }}><OverviewTab /></div>}
      {mounted.has("agents") && <div style={{ display: activeTab === "agents" ? "block" : "none" }}><AgentsTab /></div>}
      {/* ... etc for all tabs */}
    </>
  );
}
```

This only mounts a tab the first time it's visited, then keeps it alive. Best of both worlds.

---

## Files Summary

| File | Change | Issue |
|------|--------|-------|
| `agents/patterns/planning/strategies.py` | Add `system_prompt` to `_build_scope_selection_prompt` | #1 |
| `agents/patterns/planning/strategies.py` | Fix `_format_recalled_memories` to handle Action entries (pending diagnostic) | #2 |
| `agents/patterns/actions/policies.py` | Skip `VAR_KEYWORD`/`VAR_POSITIONAL` in `_infer_input_schema`, allow extra fields | #3 |
| `web_ui/frontend/src/components/layout/AppShell.tsx` | Lazy-mount tabs with `display:none` instead of unmounting | #4 |

## Recommended Order

1. **Issue 3** (`_infer_input_schema` **kwargs) — straightforward, no investigation needed
2. **Issue 4** (UI tab state) — straightforward, no investigation needed
3. **Issue 1** (scope selection context) — straightforward addition of system_prompt
4. **Issue 2** (action history) — needs diagnostic step first to determine where the pipeline breaks
