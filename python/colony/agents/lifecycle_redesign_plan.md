# Plan: Agent Lifecycle Redesign — Plan Exhaustion as Replanning Trigger

## Context

When `CacheAwareActionPolicy.plan_step()` exhausts all actions (`current_action_index >= len(actions)`), it immediately sets `policy_complete = True`, which propagates through `execute_iteration` → `run_step` → agent enters `STOPPED` state and terminates the agent. This means:
1. **Plan exhaustion = agent death** — no opportunity to replan for new work
2. **One run = one lifetime** — agents can't serve multiple sequential requests
3. **No IDLE state** — `AgentState.IDLE` exists in the enum but is never used
4. **No extensibility between runs** — no hook point for memory consolidation, intrinsic goals, etc.

This is wrong because:

1. **Plan exhaustion ≠ goal satisfaction** — the planning horizon may have been reached, but the agent's goals aren't met yet. The planner should decide if more work is needed.
2. **The existing replanning infrastructure is bypassed** — Colony already has `_replan_horizon()`, `revise_plan()`, `ReplanningPolicy`, `RevisionTrigger`, and `RevisionStrategy`. Plan exhaustion should feed into this mechanism, not bypass it.
3. **Completed plan context is destroyed** — the exhausted plan's `execution_context` (completed actions, results, findings) is invaluable for the planner to generate a **continuation**, but it's thrown away.
4. **No IDLE state** — `AgentState.IDLE` exists in the enum but is never used. After ALL work is truly done (planner confirms goal satisfied), agents should be able to idle rather than die.

**Goal**: Plan exhaustion triggers replanning via the existing MPC mechanism. Only when the planner produces no new actions (goal satisfied) does the agent consider stopping or idling. **The behavior at each decision point is configurable**.

---

## Key Insight: `revise_plan()` Already Handles Exhaustion

`CacheAwareActionPlanner.revise_plan()` (planner.py:265-268):
```python
current_plan.actions = (
    current_plan.actions[: current_plan.current_action_index] + new_actions
)
```
When `current_action_index == len(actions)`, this keeps ALL completed actions and appends new ones. **The infrastructure already supports plan continuation at exhaustion — it's just never called there.**

---

## Changes

### 1. Add `PLAN_EXHAUSTED` revision trigger

**File**: `colony/python/colony/agents/models.py` (~line 349)

Add to `RevisionTrigger` enum:
```python
PLAN_EXHAUSTED = "plan_exhausted"  # All actions executed, horizon reached
```

This is the natural place — it's a trigger like `FAILURE` or `PERIODIC`, not a special case.

### 2. Add `PlanExhaustionBehavior` enum and `LifecycleMode` enum

**File**: `colony/python/colony/agents/models.py`

Add new enum near `AgentState` (~line 1648):

```python
class PlanExhaustionBehavior(str, Enum):
    """What to do when all actions in the current plan have been executed."""
    REPLAN = "replan"            # Ask planner for continuation (default for MPC agents)
    STOP = "stop"                # Terminate immediately (legacy behavior)

class LifecycleMode(str, Enum):
    """What to do when the planner confirms no more work is needed."""
    ONE_SHOT = "one_shot"        # Goal satisfied → STOPPED (default, backward-compatible)
    CONTINUOUS = "continuous"    # Goal satisfied → IDLE, wait for new events/runs
```

**Separation of concerns**:
- `PlanExhaustionBehavior` — policy-level: what happens when actions run out? (REPLAN or STOP)
- `LifecycleMode` — agent-level: what happens when work is genuinely done? (STOP or IDLE)

### 3. Add fields to `AgentMetadata`

**File**: `colony/python/colony/agents/models.py` (~line 2580)

```python
lifecycle_mode: LifecycleMode = LifecycleMode.ONE_SHOT
idle_sleep_interval: float = Field(default=1.0, description="Seconds between idle checks")
idle_timeout: float | None = Field(default=None, description="Seconds in IDLE before auto-stop (None=forever)")
```

### 4. Add `PlanExhaustionReplanningPolicy` (IMPLEMENTED)

**File**: `colony/python/colony/agents/patterns/planning/replanning.py`

Plan exhaustion logic lives in a proper `ReplanningPolicy`, not inline in `plan_step()`.
`PlanExhaustionReplanningPolicy` owns `plan_exhaustion_behavior` and `max_replan_cycles`.
`CompositeReplanningPolicy` was updated to propagate `plan_exhausted` from sub-policies.
`PlanExhaustionReplanningPolicy` is exported from `planning/__init__.py`.

### 5. Rewrite plan exhaustion in `CacheAwareActionPolicy.plan_step()` (IMPLEMENTED)

**File**: `colony/python/colony/agents/patterns/actions/policies.py`

The inline plan exhaustion check was removed. `plan_step()` now has a **single** `evaluate_replanning_need` call via `self.replanning_policy` (a `CompositeReplanningPolicy`) that handles ALL triggers: periodic, failure, AND plan exhaustion.

The default `replanning_policy` is now:
```python
CompositeReplanningPolicy([
    PeriodicReplanningPolicy(...),
    PlanExhaustionReplanningPolicy(),
])
```

`plan_exhaustion_behavior` and `max_replan_cycles` were removed from `CacheAwareActionPolicy.__init__()`.

The `plan_exhausted` field on `ReplanningDecision` drives true completion logic in `plan_step()`.

### 6. Add `idle` field to `ActionPolicyIterationResult`

State transitions are the policy's decision, communicated through `ActionPolicyIterationResult`. The existing pattern:
- `policy_completed=True` → agent transitions to STOPPED
- `blocked_reason` → agent transitions to SUSPENDED

Add the same pattern for IDLE:

**File**: `colony/python/colony/agents/models.py` — `ActionPolicyIterationResult` (~line 1661)

```python
idle: bool = False  # Policy requests IDLE state (no work available)
```

**File**: `colony/python/colony/agents/patterns/actions/policies.py` — `BaseActionPolicy.execute_iteration()` (~line 1723)

Add an idle check alongside the existing `policy_complete` check:
```python
if next_action is None:
    if state.custom.get("policy_complete"):
        return ActionPolicyIterationResult(success=True, policy_completed=True)
    if state.custom.get("idle"):
        return ActionPolicyIterationResult(success=True, policy_completed=False, idle=True)
    return ActionPolicyIterationResult(success=True, policy_completed=False)
```

This follows the exact same `state.custom` → result field pattern already established for `policy_complete`.

### 7. Modify `Agent.run_step()` — no separate `_idle_step()` or `has_pending_work()`

IDLE agents still go through `execute_iteration()` → `plan_step()`. The policy decides if there's real work via its own internal logic. No separate code path, no reaching into policy internals.

**File**: `colony/python/colony/agents/base.py`

**7a.** Allow `run_step()` to accept IDLE state (~line 2440):

```python
if self.state not in (AgentState.RUNNING, AgentState.IDLE):
    logger.warning(f"Agent {self.agent_id} in state {self.state}, cannot run step")
    return
```

**7b.** After `execute_iteration()` returns (~line 2477), handle the `idle` signal:

```python
iteration_result = await self.action_policy.execute_iteration(self.action_policy_state)

# Policy-driven state transitions
if iteration_result.policy_completed:
    self.state = AgentState.STOPPED
elif iteration_result.idle:
    if self.state != AgentState.IDLE:
        # Entering IDLE — record timestamp
        self._idle_since = time.time()
    self.state = AgentState.IDLE
elif self.state == AgentState.IDLE:
    # Policy returned work (action executed) while we were IDLE → wake up
    self.state = AgentState.RUNNING
    self._idle_since = None

# Idle timeout check
if self.state == AgentState.IDLE:
    timeout = self.metadata.idle_timeout
    if timeout is not None and hasattr(self, '_idle_since'):
        if (time.time() - self._idle_since) > timeout:
            logger.info(f"Agent {self.agent_id} idle timeout ({timeout}s) reached, stopping")
            await self.stop()
            return
    # Call hookable on_idle for extensibility
    await self.on_idle()

# ... existing STOPPED/SUSPENDED handling ...

# Sleep interval: longer when IDLE to avoid busy-looping
sleep_interval = self.metadata.idle_sleep_interval if self.state == AgentState.IDLE else 0.1
await asyncio.sleep(sleep_interval)
```

**Why no `_idle_step()` or `has_pending_work()`**: The IDLE agent runs the normal `execute_iteration()` → `plan_step()` path. `CacheAwareActionPolicy.plan_step()` calls `super().plan_step()` (EventDrivenActionPolicy) which processes events via registered handlers. If a meaningful event arrives (a new run request, a wake-up signal from another agent), the handler produces an action or creates a new plan — `plan_step()` returns that action, `execute_iteration()` dispatches it and returns `idle=False`, and `run_step()` transitions IDLE → RUNNING. Noisy events that handlers skip produce no action — `plan_step()` returns None with `idle=True` — agent stays IDLE. The policy decides what constitutes real work, not the agent.

**7c.** Add hookable `on_idle()` method:

```python
@hookable
async def on_idle(self) -> None:
    """Called each idle cycle. Hook for background work (memory consolidation, etc.)."""
    pass
```

### 8. Exports

**File**: `colony/python/colony/agents/__init__.py`
- Add `LifecycleMode`, `PlanExhaustionBehavior` to public exports

**File**: `colony/python/colony/agents/patterns/actions/policies.py`
- Add imports: `from ...models import LifecycleMode, PlanExhaustionBehavior, RevisionTrigger, RevisionStrategy` (some may already be imported)

---

## Files Summary

| File | Change |
|------|--------|
| `colony/python/colony/agents/models.py` | Add `PLAN_EXHAUSTED` to `RevisionTrigger`, add `PlanExhaustionBehavior` enum, add `LifecycleMode` enum, add `idle` to `ActionPolicyIterationResult`, add 3 fields to `AgentMetadata` |
| `colony/python/colony/agents/base.py` | Allow IDLE in `run_step()`, handle `idle` result from `execute_iteration()`, add hookable `on_idle()`, configurable sleep interval |
| `colony/python/colony/agents/patterns/actions/policies.py` | Add `idle` check in `BaseActionPolicy.execute_iteration()`; add `plan_exhaustion_behavior` + `max_replan_cycles` params to `CacheAwareActionPolicy.__init__()`, rewrite plan exhaustion block in `plan_step()` |
| `colony/python/colony/agents/__init__.py` | Export new enums |

---

## Data Flow

```
Plan horizon exhausted (current_action_index >= len(actions)):
  │
  ├─ PlanExhaustionBehavior.REPLAN (default for MPC agents):
  │    ├─ replan_cycle_count < max_replan_cycles?
  │    │    YES → _replan_horizon(PLAN_EXHAUSTED, ADD_ACTIONS)
  │    │          → planner.revise_plan() sees full execution_context
  │    │          → revise_plan keeps completed actions, appends new ones
  │    │          → has_remaining_actions()?
  │    │               YES → continue executing (loop back to plan_step)
  │    │               NO  → planner says goal satisfied → fall to completion
  │    │    NO  → budget exceeded → fall to completion
  │    │
  │    └─ True completion:
  │         ├─ LifecycleMode.ONE_SHOT → policy_complete=True → STOPPED
  │         └─ LifecycleMode.CONTINUOUS → idle=True → run_step sets IDLE
  │              → next iteration: execute_iteration → plan_step → events?
  │              → meaningful event → action returned → idle=False → RUNNING
  │              → no work → idle=True → stays IDLE → on_idle() hook
  │              → idle_timeout → STOPPED
  │
  └─ PlanExhaustionBehavior.STOP (legacy):
       └─ policy_complete=True → STOPPED (current behavior)
```

---

## What This Does NOT Change

- `_replan_horizon()` — untouched. Already handles exhausted plans correctly.
- `revise_plan()` — untouched. Already keeps completed actions and appends new ones.
- `BaseActionPolicy.execute_iteration()` — untouched.
- `_run_agent_loop()` — untouched. IDLE agents keep looping through `run_step`.
- `EventDrivenActionPolicy` — untouched.
- `ReplanningPolicy` subclasses — untouched (the decision comes from `plan_step`, not the policy).
- No new files created.
- No cosmetic changes.

---

## Verification

```bash
colony-env down && colony-env up --workers 3 && colony-env run --local-repo /home/anassar/workspace/agents/crewAI/ --config my_analysis.yaml --verbose
```

1. **Default behavior (`REPLAN` + `ONE_SHOT`)**: After plan actions execute, agent should replan. If planner returns more actions, continue. If planner returns empty, agent stops. Watch for `PLAN_EXHAUSTED` trigger in logs.
2. **Legacy behavior (`STOP` + `ONE_SHOT`)**: Set `plan_exhaustion_behavior: stop` in `action_policy_config`. Agent should stop immediately when plan exhausted (same as before).
3. **`REPLAN` + `CONTINUOUS`**: Agent replans when exhausted, and when planner says done, enters IDLE. New events wake it to RUNNING.
4. **Max replan budget**: Set `max_replan_cycles: 2`. Agent should replan at most twice, then complete.
5. **Verify plan context preservation**: In `INFER` spans, check that the replanning LLM prompt includes completed action results from the exhausted plan.

