# Spawn Pipeline Audit — Blueprint Parameter Separation

## Principle

Three layers, no duplication:

| Layer | What it captures | Where it lives |
|-------|-----------------|----------------|
| **Constructor args** | Arguments to `T.__init__()` | `.bind(**kwargs)` → blueprint.kwargs |
| **Composition** | How instance relates to parent | `.with_composition()` on capability blueprints |
| **Deployment** | Where/how to route and spawn | `spawn_child_agents(soft_affinity=, suspend_agents=)` / `remote_instance()` |

**NOT deployment parameters** (they're metadata):
- `session_id`, `run_id` — tracking, not routing
- `max_iterations` — agent behavior, not routing
- `roles` — parent-child relationship, not routing
- `capabilities` — agent composition, not routing

## Issues Found

### 1. `AgentPoolCapability.create_agent` — `capabilities` parameter is dead code

**File:** `patterns/capabilities/agent_pool.py:100`

The `capabilities: list[str] | None = None` parameter is accepted but **never used** anywhere. It was previously passed as `capability_types=[capabilities]` to `spawn_child_agents`, which was removed. Now it's dead code.

**Fix:** The `capabilities` parameter is a list of capability *class names* (strings) that should be attached to the spawned child. Since `create_agent` dynamically imports the agent class and calls `.bind()`, capabilities should be resolved to actual `AgentCapabilityBlueprint` objects and passed as `capability_blueprints` in the blueprint. Since the current usage in samples passes capability class name strings (e.g., `capabilities=["ClusterAnalyzerCapabilityV2"]`), we need to either:

(a) Resolve the string to a class and `.bind()` it → pass as `capability_blueprints` in the blueprint kwargs
(b) Pass the strings into `metadata.parameters` for the agent to handle itself

Option (a) is cleaner — it uses the Blueprint pattern properly. The agent_pool already imports dynamically, so we can do the same for capabilities.

### 2. `Agent.spawn_child_agents` — `roles` parameter is NOT a deployment parameter

**File:** `base.py:2617`

`roles` determines the parent-child tracking relationship (stored in `self.child_agents[role] = child_id`). This is metadata about the relationship, not about deployment routing.

**Fix:** Remove `roles` from `spawn_child_agents`. Callers should track their children themselves — most already do (e.g., `agent_pool.py:158`). The docstring example already shows `roles=["grounding"]` which is misleading since no actual callers use it (only the docstring).

The `roles` tracking logic:
```python
if roles is not None:
    for role, child_id in zip(roles, child_ids):
        self.child_agents[role] = child_id
```
This is trivially done by callers after receiving child IDs.

### 3. Stale `session_id`/`run_id` params in sample code `spawn_agents` calls

**File:** `samples/code_analysis/examples/code_analysis_agent_example.py:78-79`
```python
agent_ids = await agent_system.spawn_agents(
    blueprints=[coordinator_bp],
    session_id=session_id,  # STALE — this param was removed from spawn_agents
    run_id=run_id,          # STALE — this param was removed from spawn_agents
    soft_affinity=False
)
```

**File:** `samples/code_analysis/basic/coordinator.py:452-453`
```python
agent_ids = await agent_system.spawn_agents(
    blueprints=blueprints,
    session_id=self.agent.metadata.session_id,  # STALE
    run_id=self.agent.metadata.run_id,          # STALE
    soft_affinity=True,
    suspend_agents=False
)
```

**Fix:** Remove `session_id=` and `run_id=` from these calls. The blueprints' metadata already contains session_id/run_id (set at bind time via `AgentMetadata(session_id=..., run_id=...)`).

### 4. Stale docstring examples referencing old API

**Files:**
- `meta_agents/consistency.py:19-23` — references `AgentBlueprint(agent_type=...)` and `capability_types=[...]`
- `meta_agents/grounding.py:27-34` — references `AgentBlueprint(agent_id=..., agent_type=..., capability_types=[...])`
- `base.py:2648` — `spawn_child_agents` docstring example uses `roles=["grounding"]`

**Fix:** Update docstring examples to use `.bind()` pattern with `capability_blueprints`.

### 5. Sample code `capabilities=["..."]` in `create_agent` calls

**File:** `samples/code_analysis/basic/coordinator.py:680`
```python
result = await self.agent_pool_cap.create_agent(
    agent_type="...",
    capabilities=["ClusterAnalyzerCapabilityV2"],  # Dead — never reaches the agent
    ...
)
```

**File:** `samples/code_analysis/impact/coordinator.py:1097`
```python
result = await self.agent_pool_cap.create_agent(
    agent_type="...",
    capabilities=["ChangeImpactAnalysisCapability"],  # Dead — never reaches the agent
    ...
)
```

**Fix:** These will be fixed by fixing `create_agent` itself (issue #1).

## Execution Plan

### Step 1: Fix `agent_pool.py` — make `capabilities` work
- Resolve capability class names to classes via `importlib`
- Call `.bind()` on each → `AgentCapabilityBlueprint`
- Pass as `capability_blueprints=` in the agent's `.bind()` call

### Step 2: Fix `spawn_child_agents` — remove `roles`
- Remove `roles` parameter
- Move role-tracking to callers
- Update docstring/example to not use `roles=`
- Search for any callers that pass `roles=` and update them

### Step 3: Fix stale sample code calls
- Remove `session_id=`, `run_id=` from `agent_system.spawn_agents()` calls
- The blueprint metadata already has these

### Step 4: Update stale docstring examples
- `consistency.py` module docstring
- `grounding.py` module docstring
- `spawn_child_agents` docstring example

### Step 5: Verify
- `python -c "import ast; ast.parse(open(f).read())"` for all modified files
