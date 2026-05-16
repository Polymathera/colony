# Registering a mission

A *mission* in Colony is a named, parameterised agent-hierarchy invocation handle the [`SessionAgent`](../../src/polymathera/colony/web_ui/backend/chat/__init__.py) dispatches in response to a user request. Each mission declares: a label and description (what the planner sees), a coordinator class (the root agent spawned to run it), its worker class, the capabilities the coordinator and worker compose, the extra per-run parameters the YAML parser surfaces, and a self-concept (the LLM-identity context the coordinator's planner reads at every turn). The schema is one Pydantic model — [`MissionSpec`](../../src/polymathera/colony/agents/configs.py) — and every registration path validates against it.

There are two registration paths. They surface to the same `SessionAgent.available_missions` dict; the only difference is *where* an entry lives.

## Decision tree

```
Does your mission ship as part of a pip-installable Python package?

  ├── YES  →  Path A: polymathera.mission_types entry-point group
  │          (pip-distribution-time; visible everywhere the package
  │           is installed)
  │
  └── NO, it's per-program content in a design monorepo
       →  Path B: .colony/missions/ L1-A discovery
           (per-monorepo; visible only when a session is opened
            against that monorepo)
```

Pick by where the mission logically *lives*, not by what's convenient. A mission whose coordinator class ships in an installed package — and which any cluster running that package should be able to dispatch — uses Path A. A mission whose coordinator class lives only inside one design monorepo's `.colony/agents/` — and which should only surface when a session is opened against that monorepo — uses Path B.

## Path A — `polymathera.mission_types` entry-point group

Declare the entry-point in your distribution's `pyproject.toml`:

```toml
[tool.poetry.plugins."polymathera.mission_types"]
<mission_id> = "<module.path>:<factory_callable>"
```

The entry-point *name* (left side) is the mission key the `SessionAgent` sees. The right-side callable returns a `dict` matching [`MissionSpec`](../../src/polymathera/colony/agents/configs.py):

```python
from typing import Any


def my_mission_entry() -> dict[str, Any]:
    return {
        "label": "My mission",
        "description": "One-line summary the SessionAgent's planner reads.",
        "coordinator_v1": "my_package.coordinator.MyCoordinator",
        "coordinator_v2": "my_package.coordinator.MyCoordinator",
        "worker": "my_package.worker.MyWorker",
        "coordinator_capabilities": ["MyCoordinatorCapability"],
        "worker_capabilities": ["MyWorkerCapability"],
        "extra_metadata_keys": ["target_path"],
        "self_concept": {
            "description": "Coordinates ...",
            "goals": ["..."],
            "constraints": ["..."],
        },
    }
```

`get_mission_registry()` (in [`colony.agents.mission_registry`](../../src/polymathera/colony/agents/mission_registry.py)) walks the group at runtime, validates each entry through `MissionSpec`, and merges with the colony-builtin set. Entries that fail validation are logged and skipped — they don't poison the registry.

**`extra="forbid"`**: A typo'd key (`coordinator_capabilites` instead of `coordinator_capabilities`) fails validation and the entry is skipped, with the typo'd key named in the warning. This is the schema-drift guard.

## Path B — `.colony/missions/` L1-A discovery

Drop a Python file at `<design_monorepo>/.colony/missions/<mission_id>.py` exposing a top-level `mission_entry()` callable that returns the same dict shape Path A uses:

```python
# .colony/missions/<mission_id>.py
from typing import Any


def mission_entry() -> dict[str, Any]:
    return {
        "label": "...",
        "description": "...",
        "coordinator_v1": "<coordinator_module>.<CoordinatorClass>",
        "coordinator_v2": "<coordinator_module>.<CoordinatorClass>",
        "worker": "<worker_module>.<WorkerClass>",
        "coordinator_capabilities": [...],
        "worker_capabilities": [...],
        "extra_metadata_keys": [...],
        "self_concept": {...},
    }
```

The **file stem** is the mission key. The L1-A discoverer ([`discover_missions`](../../src/polymathera/colony/design_monorepo/extensions.py)) loads the file in an isolated module namespace (not via `sys.path`), calls `mission_entry()`, validates through `MissionSpec`, and surfaces the entry under `RepoStateProvider.discovered_extensions.missions[<stem>]`. Same drift discipline: failures are logged and skipped.

Coordinator class strings (`coordinator_v1` / `coordinator_v2`) typically reference an Agent class L1-A discovered under `.colony/agents/` — `AgentPoolCapability.create_agent` resolves the string via `importlib.import_module` + `getattr`, and the agent file's module name (the file stem) is what the import path resolves against. Worker class strings often point at reusable workers shipped in an installed package the monorepo depends on.

## Shared schema, shared discipline

Both paths validate against the same [`MissionSpec`](../../src/polymathera/colony/agents/configs.py) (`extra="forbid"`). The required fields are: `label`, `description`, `coordinator_v1`, `coordinator_v2`, `worker`, `self_concept` (with at least a `description`). The list fields (`coordinator_capabilities`, `worker_capabilities`, `extra_metadata_keys`) default to empty. `self_concept.goals` and `self_concept.constraints` default to empty.

Drift across paths is impossible at registration time: a renamed field, a typo, a removed key — any of them fails `MissionSpec.model_validate()` and the entry is skipped, logged with the validation error so the author can fix it.

## How the `SessionAgent` runs a registered mission

Once a mission is registered (Path A or Path B), the chat user just describes the task; the `SessionAgent` does the rest. The full chain:

```
user types in chat
     │
     ▼
SessionAgent.metadata.parameters["available_missions"]      ← live registry view
   (rebuilt on initialize() and on every user message via
    _refresh_available_missions; union of get_mission_registry()
    + RepoStateProvider.discovered_extensions.missions)
     │
     ▼
LLM planner sees available_missions in its system prompt
+ the MISSION SPAWN PROTOCOL guidance in self_concept
     │
     ▼
LLM calls   await run("spawn_mission", mission_type="<key>")
     │
     ▼
SessionOrchestratorCapability.spawn_mission(mission_type=...)
   looks up <key> in the LIVE merged registry,
   builds AgentMetadata from the entry's self_concept + caller params,
   dispatches AgentPoolCapability.create_agent(agent_type=coordinator_v2)
     │
     ▼
AgentPoolCapability.create_agent
   resolves the class via colony.agents.class_resolver.resolve_class()
   (importlib first; L4-discovered-agent fallback second)
   spawns the coordinator
     │
     ▼
Coordinator's own action policy drives the mission.
SessionAgent relays progress via SessionOrchestratorCapability.respond_to_user.
```

### `spawn_mission` — the action the LLM calls

[`SessionOrchestratorCapability.spawn_mission`](../../src/polymathera/colony/web_ui/backend/chat/session_agent.py) is the mission-aware wrapper over `AgentPoolCapability.create_agent`. The LLM does NOT extract `coordinator_class` from `available_missions` and pass it to `create_agent` directly. It calls `spawn_mission` with just the mission key:

```python
# Inside the SessionAgent's REPL:
r = await run(
    "spawn_mission",
    mission_type="opm_meg",                       # key from available_missions
    mission_params={                              # optional, per-mission
        "noise_floor_target_fT_rt_hz": 12.0,
    },
)
if r.success and r.output["created"]:
    coord_id = r.output["agent_id"]
    await run(
        "respond_to_user",
        content=f"Coordinator {coord_id[:8]} is running.",
    )
else:
    await run(
        "respond_to_user",
        content=f"Spawn failed: {r.error or r.output.get('error')}",
    )
```

What `spawn_mission` does internally:

1. Re-reads the **live** merged registry (`get_mission_registry()` ∪ `RepoStateProvider.discovered_extensions.missions`). A mission added mid-session via L1-E becomes spawnable on the next chat turn — the action does not rely on the static snapshot baked into `metadata.parameters["available_missions"]`.
2. Reads `coordinator_v2` (falling back to `coordinator_v1` if absent) as the class to spawn.
3. Builds [`AgentMetadata`](../../src/polymathera/colony/agents/models.py) populated from the mission entry's `self_concept` plus any caller-supplied `mission_params` (the entry's `extra_metadata_keys` describes which params a given mission expects).
4. Calls `AgentPoolCapability.create_agent(agent_type=<coordinator_v2>, metadata=<above>)`.
5. Returns `{"agent_id", "mission_type", "coordinator_class", "created", "label"}`, with an `error` field on failure.

Unknown `mission_type` returns `{"created": False, "error": "Unknown mission type ... Available: [...]"}` — clean error path, no exception bubbling to the planner.

### The MISSION SPAWN PROTOCOL in the self-concept

The SessionAgent's [`self_concept.description`](../../src/polymathera/colony/web_ui/backend/routers/sessions.py) carries a dedicated `MISSION SPAWN PROTOCOL` section telling the LLM:

- When the user describes a task that matches an `available_missions` entry by `label` / `description`, call `spawn_mission(mission_type=<key>)`.
- Do NOT extract `coordinator_class` and call `create_agent` directly for missions — `spawn_mission` does the lookup.
- Branch on `r.success` and `r.output["created"]`; relay the spawned `agent_id` back to the user.

This explicit protocol exists because the LLM otherwise has to infer the mapping `available_missions[type].coordinator_class → create_agent(agent_type=...)` from a dict-literal in the prompt + a generic "use create_agent" hint. Inferring works for strong models but is brittle; the protocol makes it explicit.

### Dynamic registry refresh — mid-session mission additions

`SessionOrchestratorCapability._refresh_available_missions` re-runs on:

- `initialize()` (after the design-monorepo lazy clone), in a thread executor so the lazy `git clone` doesn't block the event loop.
- Every user message arriving via `handle_user_message`.

The cost is cheap: the underlying `RepoStateProvider.discovered_extensions` cache is mtime-fingerprinted (one `stat` per surface dir; full re-walk only when a mtime bumps). The result: an agent that authored `.colony/missions/synthetic.py` via the L1-E `bootstrap_*` flow can see `synthetic` in `available_missions` on the next user message — no session restart required.

### Class resolution — L4 coordinators authored under `.colony/agents/`

[`colony.agents.class_resolver.resolve_class(fqn, fallback_registry=None)`](../../src/polymathera/colony/agents/class_resolver.py) is the canonical string-to-class resolver every spawn path delegates to (`AgentPoolCapability._resolve_class`, `cli.polymath._resolve_class`, REST `/api/jobs/submit` and `/api/agents/spawn`). Two paths, tried in order:

1. `importlib.import_module(<module_path>)` + `getattr(<class_name>)` — for pip-installed classes.
2. `fallback_registry[<class_name>]` — for L4 classes loaded by L1-A's `discover_agents` from `<monorepo>/.colony/agents/*.py`. L1-A's loader deliberately keeps these modules out of `sys.modules` (extension files are discovery-scoped, re-read for mtime invalidation), so the importlib path cannot find them.

`AgentPoolCapability.create_agent` threads `RepoStateProvider.discovered_extensions.agents` (a `dict[str, type[Agent]]` keyed by class short-name) as `fallback_registry`. The result: a mission whose `coordinator_v2` is `"opm_meg_coordinator.OPMMEGCoordinator"` resolves transparently even though `opm_meg_coordinator` is not on `sys.path` — the importlib path fails, the fallback finds the class by short-name. No `pip install` of the L4 monorepo required.

### End-to-end worked example

A clean smoke-test of the whole chain (assumes a running cluster + OPM-MEG design monorepo configured):

```
# Operator (one-time, via the dashboard):
#   1. Create the colony.
#   2. Settings → Design Monorepo → set origin URL to
#      https://github.com/<org>/monorepo_opm_meg.git

# User opens chat against that colony and types:
> "Run an OPM-MEG noise-floor design analysis."

# What happens (visible in the chat UI):
#   1. SessionAgent: "Starting OPM-MEG noise-floor analysis…"
#   2. (under the hood) spawn_mission(mission_type="opm_meg")
#        → AgentPoolCapability.create_agent(
#              agent_type="opm_meg_coordinator.OPMMEGCoordinator",
#              metadata=<from registry self_concept>)
#        → returns agent_id = "agent_abc123…"
#   3. SessionAgent: "Coordinator agent_abc1 is running. I'll relay updates."
#   4. Coordinator runs; emits results back to the chat via the standard
#      relay (SessionOrchestratorCapability subscribes to its blackboard).
```

The same chain runs for any Path A entry-point mission AND any Path B L4 mission. The author of the mission file does not need to know how `spawn_mission`, `_refresh_available_missions`, or the class-resolver fallback wire together — those are framework-side. The author's surface is the `MissionSpec` dict.
