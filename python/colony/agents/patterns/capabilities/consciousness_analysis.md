# `ConsciousnessCapability` & `AgentSelfConcept` — Analysis and Integration Plan

## Context

The planning pipeline needs agent identity in the LLM prompt. Currently `_build_system_prompt()` in `CacheAwareActionPolicy` manually scrapes `agent.metadata.role`, `agent.__class__.__doc__`, etc. This is ad-hoc — `ConsciousnessCapability` and `AgentSelfConcept` already exist to provide this information.

## `AgentSelfConcept` Field Analysis

### Well-defined, operationally distinct fields (keep as-is)
- **`agent_id`**, **`name`**, **`role`**, **`description`** — Core identity. Clear and distinct.
- **`version`** — Useful for multi-generational agents.
- **`world_model`** — Agent's understanding of its environment (execution context, deployment, etc.)
- **`goals`** — High-level optimization objectives (NOT task assignments).
- **`constraints`** — Hard limits on behavior. Distinct from goals.
- **`capabilities`** — What the agent can do. Maps to concrete action groups.
- **`limitations`** — What the agent cannot do. Opposite of capabilities.
- **`skills`** — More granular than capabilities (name + description pairs).
- **`identity`** — Concise self-authored essence statement. Distinct from `role`.
- **`frame_of_mind`** — Localizes knowledge and allows contradictory beliefs per context. Useful for multi-task agents.

### BDI fields — conceptually sound but some overlap
- **`beliefs`** — What the agent holds to be true. Core BDI.
- **`desires`** — What the agent wants (has meaningful defaults). Core BDI.
- **`intentions`** — What the agent is committed to doing. Uses `AgentIntention` with `CommitmentStrategy`. Core BDI.
- **`commitments`** — Overlaps with `intentions`. An intention IS a commitment to act.
- **`commitment_rules`** — Could be part of `AgentIntention.CommitmentStrategy` (which already exists).

### Overlapping affective/motivational fields
- **`motivations`** — Overlaps with `desires`. A desire IS a motivation.
- **`aspirations`** — Subset of `desires` (long-term ones).
- **`fears`** — Inverse of `desires` (negative valence).
- **`emotional_states`** — No operational semantics defined. No code path consumes them.
- **`moods`** — Overlaps with `emotional_states`, equally undefined operationally.
- **`needs`** — Overlaps with `desires` (Maslow-style vs. BDI).

### Fields lacking operational semantics
- **`biases`** — Useful for self-awareness but no code path consumes them.
- **`mental_models`** — Overlaps with `world_model` and `beliefs`.
- **`values`** / **`value_system`** — Two fields for the same concept.
- **`regimes`** — Undefined. No usage, no clear meaning.
- **`personal_traits`** — No operational integration.
- **`physical_embodiment`** — For software agents, overlaps with `world_model`.

### Evolution tracking (well-designed, keep as-is)
- **`version_history`**, **`last_modified`**, **`evolution_metrics`**, **`evolution_constraints`** — Clear operational semantics via `_update_evolution_metrics`, `_check_evolution_constraints`.

**Decision**: Field overlaps are a known design consideration but NOT changed now — refactoring the model is a breaking change for anyone already persisting self-concepts. Users selectively populate fields that matter to them.

## `ConsciousnessCapability` Plumbing Status

### What works
1. **Capability registration**: In `EXTRA_CAPABILITIES_REGISTRY` in `polymath.py`
2. **Instantiation path**: `_create_agent_instance -> capability_classes -> Agent._create_action_policy() -> ConsciousnessCapability(agent) -> add_capability() -> initialize()`
3. **`initialize()`** calls `_load_self_concept()` from persistent storage
4. **Action executors**: `get_self_concept`, `update_self_concept`, `get_system_documentation`, etc. — all discoverable via MRO fix
5. **`ReflectionCapability`** already does `agent.get_capability(ConsciousnessCapability.get_capability_name())` to pull self-concept

### What's broken / missing
1. **`_build_system_prompt()` is ad-hoc** — Does NOT use `ConsciousnessCapability` or `AgentSelfConcept`.
2. **No self-concept created on first spawn** — `initialize()` tries to load from storage; if nothing stored (first run), silently returns `None`. `_create_default_self_concept()` exists but only called from `update_self_concept()`. First-run agents have no self-concept.
3. **`SystemDocumentation` is never populated** — Accessible via `get_system_documentation()` but nobody writes to it during deployment setup.
4. **Planning pipeline doesn't query `ConsciousnessCapability`** — `_build_system_prompt()` should pull from self-concept when available.

## Fixes

### Fix A: Wire `_build_system_prompt()` to use `ConsciousnessCapability`

In `CacheAwareActionPolicy._build_system_prompt()`:
1. Check if agent has `ConsciousnessCapability` via `agent.get_capability_by_type(ConsciousnessCapability)`
2. If available and self-concept is loaded, build prompt from `AgentSelfConcept` fields (name, role, description, identity, goals, constraints, capabilities, limitations, world_model, frame_of_mind)
3. Fall back to current metadata-based approach when consciousness is not attached

### Fix B: Auto-create default self-concept on first init

In `ConsciousnessCapability.initialize()`:
- When `_load_self_concept()` returns None (first run), call `_create_default_self_concept()` to populate from agent metadata
- This ensures every agent with `ConsciousnessCapability` always has a self-concept after initialization

## Files Modified

| File | Change |
|------|--------|
| `colony/agents/patterns/actions/policies.py` | `_build_system_prompt()` — use `ConsciousnessCapability` when available |
| `colony/agents/patterns/capabilities/consciousness.py` | `initialize()` — auto-create default self-concept on first run |
