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
