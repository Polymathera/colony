# Plan: Agent Capability API Refactoring

## Problem

The capability registration APIs are split between `Agent` and `ActionPolicy`, causing:

1. **Prompt duplication**: Capabilities are passed to `ActionDispatcher` from TWO redundant sources:
   - `self._action_providers` (set at construction from `agent._capabilities.values()`)
   - `get_used_capabilities()` (resolves `_used_agent_capabilities` string list back to `agent._capabilities`)
   - Combined at `policies.py:1662`: `capability_providers + self._action_providers` — same objects appear twice

2. **Confusing dual-registration**: `Agent._create_action_policy()` both passes capabilities as `action_providers` (line 2053) AND calls `use_agent_capabilities()` (line 2077) for the same set.

3. **`_action_providers` goes stale**: It's a snapshot of `agent._capabilities.values()` at construction time. Dynamic additions via `use_agent_capabilities()` / `use_capability_blueprints()` correctly update `_used_agent_capabilities` and reset the dispatcher, but `_action_providers` still holds the stale list.

4. **Event handler duplication**: `BlackboardEventActionPolicy._get_event_handlers()` iterates BOTH `agent.get_capabilities()` AND `self._action_providers` without deduplication.

5. **API surface confusion**: Users see `ActionPolicy.use_agent_capabilities()`, `ActionPolicy.use_capability_blueprints()`, `ActionPolicy.disable_agent_capabilities()` alongside `Agent.add_capability()`, `Agent.add_capability_blueprints()`, `Agent.remove_capability()` — unclear which to use when.

---

## Key Insight: `Agent` is the Single Source of Truth

`Agent._capabilities` dict is already the authoritative registry. `ActionPolicy._used_agent_capabilities` is just a filter (which subset of agent capabilities the policy exposes as actions). `_action_providers` is a redundant copy that causes duplication.

The design should be:
- **`Agent`** owns capability instances (`_capabilities` dict) — registration, lifecycle, lookup
- **`ActionPolicy`** owns the filter (`_used_agent_capabilities`) — which agent capabilities are exposed as actions
- **`ActionPolicy`** also accepts non-capability action providers (standalone functions, arbitrary objects) — these are NOT capabilities, just action sources
- **No redundant copy** — the dispatcher resolves capabilities from the agent at creation time, filtered by `_used_agent_capabilities`

---

## Changes

### 1. Remove `_action_providers` capability overlap in `_create_action_dispatcher()`

**File**: `colony/python/colony/agents/patterns/actions/policies.py` (~line 1650-1665)

The `_action_providers` list currently holds ALL agent capabilities (passed at construction). It should ONLY hold **non-capability** action providers (standalone functions, external objects). Capability-based providers come exclusively from `get_used_capabilities()`.

**Current**:
```python
async def _create_action_dispatcher(self) -> None:
    if self._action_dispatcher:
        return
    capability_providers = self.get_used_capabilities()
    self._action_dispatcher = ActionDispatcher(
        agent=self.agent,
        action_policy=self,
        action_map=self._action_map,
        action_providers=capability_providers + self._action_providers,
    )
    await self._action_dispatcher.initialize()
```

**New**:
```python
async def _create_action_dispatcher(self) -> None:
    if self._action_dispatcher:
        return
    capability_providers = self.get_used_capabilities()
    # self._action_providers may contain AgentCapability instances that
    # overlap with capability_providers (passed at construction from
    # agent._capabilities). Deduplicate by identity.
    seen = set(id(p) for p in capability_providers)
    extra_providers = [p for p in self._action_providers if id(p) not in seen]
    self._action_dispatcher = ActionDispatcher(
        agent=self.agent,
        action_policy=self,
        action_map=self._action_map,
        action_providers=capability_providers + extra_providers,
    )
    await self._action_dispatcher.initialize()
```

This is the minimal fix that eliminates duplication while preserving backward compatibility. The `_action_providers` list still exists for non-capability providers (standalone functions).

### 2. Fix `_get_event_handlers()` duplication

**File**: `colony/python/colony/agents/patterns/actions/policies.py` (~line 1907-1923)

**Current** (no deduplication):
```python
def _get_event_handlers(self) -> list[Callable]:
    handlers = []
    for cap in self.agent.get_capabilities():
        handlers.extend(self._get_object_event_handlers(cap))
    for provider in self._action_providers:
        handlers.extend(self._get_object_event_handlers(provider))
    return handlers
```

**New** (deduplicate by identity, same as `initialize()` already does):
```python
def _get_event_handlers(self) -> list[Callable]:
    handlers = []
    seen: set[int] = set()
    for source in list(self.agent.get_capabilities()) + list(self._action_providers):
        if id(source) in seen:
            continue
        seen.add(id(source))
        handlers.extend(self._get_object_event_handlers(source))
    return handlers
```

### 3. Stop passing capabilities as `action_providers` in `Agent._create_action_policy()`

**File**: `colony/python/colony/agents/base.py` (~line 2046-2078)

The root cause of the duplication is that `_create_action_policy()` passes capabilities via BOTH mechanisms. Since `use_agent_capabilities()` (line 2077) is the correct path for capabilities, stop passing them as `action_providers`.

**Current**:
```python
# Phase 3: Create action policy
if not self.action_policy_blueprint:
    self.action_policy = await create_default_action_policy(
        agent=self,
        action_map={},
        action_providers=list(self._capabilities.values()),  # ALL capabilities
        ...
    )
else:
    self.action_policy = self.action_policy_blueprint.local_instance(
        self,
        action_providers=list(self._capabilities.values()),  # ALL capabilities
        ...
    )
# Phase 4: Mark capability blueprints as "used" by the action policy
self.action_policy.use_agent_capabilities([bp.key for bp in self.capability_blueprints])
```

**New**:
```python
# Phase 3: Create action policy
if not self.action_policy_blueprint:
    self.action_policy = await create_default_action_policy(
        agent=self,
        action_map={},
        action_providers=[],  # Capabilities come via use_agent_capabilities()
        ...
    )
else:
    self.action_policy = self.action_policy_blueprint.local_instance(
        self,
        action_providers=[],  # Capabilities come via use_agent_capabilities()
        ...
    )
# Phase 4: Mark ALL capabilities as "used" by the action policy
self.action_policy.use_agent_capabilities(list(self._capabilities.keys()))
```

**Important**: Phase 4 currently only marks *blueprint* capabilities as used (`self.capability_blueprints`). But capabilities can also be added directly via `add_capability()` during the agent's `initialize()` (before `super().initialize()` calls `_create_action_policy()`). We should mark ALL capabilities in `_capabilities` as used, not just blueprint ones.

This change means:
- `_action_providers` starts empty (unless the user explicitly passes non-capability providers)
- `get_used_capabilities()` is the sole path for capabilities into the dispatcher
- Dynamic additions via `use_agent_capabilities()` / `use_capability_blueprints()` work as before
- Non-capability `action_providers` (standalone functions) still work via `_action_providers`

### 4. Make `use_agent_capabilities` accept ALL capabilities by default

The current Phase 4 only marks *blueprint* capabilities:
```python
self.action_policy.use_agent_capabilities([bp.key for bp in self.capability_blueprints])
```

But some capabilities are added directly in `Agent.initialize()` before `super().initialize()`, e.g.:
```python
# In ChangeImpactAnalysisCoordinator.initialize():
if not self.has_capability(...):
    cap = WorkingSetCapability(...)
    await cap.initialize()
    self.add_capability(cap)
```

These direct-add capabilities end up in `_capabilities` but NOT in `_used_agent_capabilities`.

**Fix**: Change Phase 4 to use all capability keys:
```python
self.action_policy.use_agent_capabilities(list(self._capabilities.keys()))
```

This ensures every capability in `_capabilities` at initialization time is exposed to the planner.

---

## What This Does NOT Change

- `Agent.add_capability()`, `Agent.remove_capability()`, `Agent.get_capability*()` — untouched
- `Agent.add_capability_blueprints()` — untouched
- `ActionPolicy.use_agent_capabilities()`, `.disable_agent_capabilities()` — untouched (these are the correct filter APIs)
- `ActionPolicy.use_capability_blueprints()` — untouched (dynamic blueprint instantiation)
- `ActionDispatcher` — untouched (it just receives a list of providers)
- Action filters (`include_actions`, `exclude_actions`, `events_only`) — untouched
- Non-capability `action_providers` (standalone functions) — still work
- Blueprint system — untouched

---

## Files Summary

| File | Change |
|------|--------|
| `colony/python/colony/agents/base.py` | Stop passing capabilities as `action_providers` in `_create_action_policy()`; mark ALL capabilities (not just blueprints) as used |
| `colony/python/colony/agents/patterns/actions/policies.py` | Deduplicate in `_create_action_dispatcher()`; fix `_get_event_handlers()` duplication |

---

## Verification

1. **No duplicate actions in prompt**: Run an agent and check the INFER span prompt — each action key should appear exactly once.
2. **Dynamic capability addition still works**: Add a capability via `use_capability_blueprints()` at runtime — it should appear in the next plan.
3. **`disable_agent_capabilities()` still works**: Disable a capability — it should disappear from the prompt.
4. **Non-capability action providers still work**: Pass a standalone function as `action_providers` — it should appear in the prompt.
5. **Action filters still work**: Add a capability with `events_only=True` — its actions should NOT appear in the prompt.
6. **Capabilities added directly in `Agent.initialize()` appear**: Capabilities added via `self.add_capability()` before `super().initialize()` should be in the prompt.
