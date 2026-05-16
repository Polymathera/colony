# Design-monorepo extension authoring (L1-E)

The symmetric write surface for [L1-A discovery](design-monorepo-extensions.md). Where L1-A reads `.colony/{plugins,agents,deployments,tools,profiles}/`, L1-E writes into it — through one `@action_executor` per surface on [`ToolBuilder`](../../src/polymathera/colony/design_monorepo/capabilities.py), each routed through a single internal helper so the validation / commit / audit discipline cannot drift between surfaces.

## TL;DR

```python
from polymathera.colony.design_monorepo import ToolBuilder

builder = ToolBuilder(agent=..., working_dir=repo_root)

await builder.bootstrap_plugin("serf_simulator", description="Toy SERF.")
await builder.bootstrap_agent("opm_meg_planner")
await builder.bootstrap_deployment("scoring_service", deployment_kwargs="num_replicas=2")
await builder.bootstrap_tool_adapter("quspin_adapter")
await builder.bootstrap_profile("fda_510k", tags=["regulatory", "510k"])
```

Each call returns an [`ExtensionAuthoredPayload`](../../src/polymathera/colony/design_monorepo/models.py) carrying `surface`, `name`, `commit_sha`, `session_id`, `authored_at`, and the rendered file path.

## Five bootstrap actions, five surfaces

Surface names come from [`DEFAULT_SURFACE_DIRS`](../../src/polymathera/colony/design_monorepo/manifest.py) — the single source of truth shared with L1-A's `resolve_surface_dirs`. The L1-E scaffold-template map declares an import-time assertion that its key set equals `DEFAULT_SURFACE_DIRS`, so adding a sixth surface fails loudly in both halves until the template ships.

| Action | Surface | Rendered path (under resolved surface dir) | Discovers as |
|---|---|---|---|
| `bootstrap_plugin(name, description)` | `plugins` | `<name>/SKILL.md` | `SkillSpec` via L1-A `discover_plugins` |
| `bootstrap_agent(name, base_class, base_module, class_name, description)` | `agents` | `<name>.py` | `Agent` subclass via `discover_agents` |
| `bootstrap_deployment(name, deployment_kwargs, class_name, description)` | `deployments` | `<name>.py` | class with `__deployment_config__` via `discover_deployments` |
| `bootstrap_tool_adapter(name, tool_spec_var)` | `tools` | `<name>.py` | invokes `register(registry)` via `discover_tools` |
| `bootstrap_profile(name, tags, embedding_strategy, description)` | `profiles` | `<name>.yaml` | parsed mapping via `discover_profiles` |

`class_name` (agents, deployments) defaults to a PascalCased form of `name` — `my_agent` → `MyAgent`.

## The authoring pipeline

Every `bootstrap_*` call executes the same five steps via the private `_author_extension` helper. Six surfaces will not produce six slightly-different pipelines.

1. **Resolve the surface directory** — `resolve_surface_dirs(working_dir, manifest)[surface]`. Honours per-surface `manifest.extensions.<surface>.directory` overrides; falls back to defaults when the manifest is v1 / missing / unparseable.
2. **Render the scaffold** — `scaffolds_module.render_extension_scaffold(surface, surface_dir, name, template_vars=...)` materialises a single file at the canonical destination path. Refuses to overwrite an existing destination.
3. **AST allow-list gate** — for `.py` files (agents / deployments / tools), [`validate_python_file`](../../src/polymathera/colony/design_monorepo/ast_validator.py) parses the freshly-written source and rejects any use of the disallowed surfaces (see below). On rejection, the file is unlinked and a `DesignMonorepoError` is raised.
4. **Commit with agent identity** — `client.commit_with_identity(identity, "bootstrap <surface>/<name> (L1-E)", paths=[relative_path])`. The commit author/committer is the agent's transactional `AgentIdentity` (same pipeline as `DesignCheckpointer` mutations).
5. **Emit the audit event** — write `ExtensionAuthoredPayload.model_dump(mode="json")` to the blackboard under `DesignMonorepoEventProtocol.extension_authored_key(surface, name)`. Best-effort; detached-mode / no-blackboard contexts log at WARNING and the commit on disk remains the durable record.

## Audit trail

The blackboard event answers "which session authored which extension, when, in response to which user message" — Risk #5's symmetric `git blame` for agent output.

```python
from polymathera.colony.agents.blackboard.protocol import DesignMonorepoEventProtocol

key = DesignMonorepoEventProtocol.extension_authored_key("agents", "opm_meg_planner")
entry = await bb.read(key)
# entry.value: dict matching ExtensionAuthoredPayload
```

The payload's `session_id` field is populated automatically from the ambient session context via [`get_current_session_id()`](../../src/polymathera/colony/agents/sessions/context.py); `user_message_id` is reserved for a future provenance plumbing pass (no entry point sets it today).

Subscribe with the wildcard pattern:

```python
@event_handler(pattern=DesignMonorepoEventProtocol.extension_authored_pattern())
async def _on_authored(self, event: BlackboardEvent, repl) -> EventProcessingResult | None:
    surface, name = DesignMonorepoEventProtocol.parse_extension_authored_key(event.key)
    ...
```

## AST allow-list

[`colony.design_monorepo.ast_validator`](../../src/polymathera/colony/design_monorepo/ast_validator.py) ships one module, one validator function, and four `frozenset` constants — `DISALLOWED_IMPORT_MODULES`, `DISALLOWED_FROM_OS`, `DISALLOWED_BUILTIN_CALLS`, `DISALLOWED_ATTRIBUTE_CALLS`. Tests consume the same constants, so the disallow-list never gets enumerated twice.

Rejected at write time:

- `import subprocess` / `from subprocess import ...` (also `ctypes`, `cffi`, `pickle`, `marshal`, `importlib`).
- `from os import system, popen, exec*, spawn*, posix_spawn, fork, forkpty`.
- Builtin calls to `eval`, `exec`, `compile`, `__import__`.
- Attribute calls to `os.system` / `os.popen` / `os.exec*` / `os.spawn*` / `os.fork` / `subprocess.{run,call,check_call,check_output,Popen,getoutput,getstatusoutput}` / `importlib.import_module` / `importlib.reload` / `pickle.{loads,load}` / `marshal.{loads,load}`.

What is **not** rejected: plain `import os` (so `os.path.join` works), and all standard-library / Colony / third-party imports that don't open a shell, spawn a process, or load arbitrary code. The AST allow-list is the cheap static gate; the deeper defence is `SandboxedShellCapability`'s container backend at execution time.

The same validator will fire on L1-F's `src/`/`tests/` writes when PR 3 ships — one uniform pipeline regardless of which half of L4 is being authored, per Risk #5.

## Round-trip with L1-A

[`RepoStateProvider.discovered_extensions`](../../src/polymathera/colony/design_monorepo/capabilities.py) caches discovery results behind an mtime fingerprint that covers `.colony/` and every resolved surface directory. An L1-E `bootstrap_*` call adds a file under one of those dirs, which bumps the dir's mtime, which invalidates the fingerprint, which forces the next `discovered_extensions` access to re-walk. No explicit `invalidate_extensions()` call is needed for the L1-E path.

```python
state = RepoStateProvider(agent=..., working_dir=repo_root)
state.discovered_extensions.agents          # {} — empty monorepo
await builder.bootstrap_agent("planner")
state.discovered_extensions.agents          # {"Planner": <class>} — cache auto-invalidated
```

## Scaffold template files

Templates live under [`scaffolds/monorepo_extensions/`](../../src/polymathera/colony/design_monorepo/scaffolds/monorepo_extensions/) and use `string.Template` `$variable` substitution (matching the existing `render_template` convention — Jinja was the alignment plan's wording, but consistency with the existing renderer wins). They are deliberately minimal: agents and CPS-shaped L2-F overrides fill in real bodies via `ProjectAuthoringCapability` (L1-F, PR 3) once the file is on disk.

## Plugging domain-shaped scaffolds in — `ExtensionScaffold` registry

The blank `bootstrap_*` templates above are intentionally generic — they produce empty stubs. A domain package (`polymathera-cps`, future `polymathera-racer`, third-party extension distributions) registers richer scaffolds — an FDA-regulatory `RegulatoryAgent` subclass instead of a blank `Agent`, a CalculiX tool-adapter instead of a blank `ToolAdapter` stub — through one registry. The caller selects them by id at the same `bootstrap_*` call.

### Registering a scaffold

[`colony.design_monorepo.scaffolds.registry`](../../src/polymathera/colony/design_monorepo/scaffolds/registry.py) holds the process-singleton registry. A domain package registers each of its scaffolds at startup from its [`polymathera.config_components`](../../src/polymathera/colony/distributed/config/extensions.py) entry-point hook:

```python
# polymathera-cps's register_components() hook
from polymathera.colony.design_monorepo.scaffolds import (
    ExtensionScaffold,
    register_extension_scaffold,
)


def register_cps_scaffolds() -> None:
    register_extension_scaffold(
        ExtensionScaffold(
            scaffold_id="agent_regulatory",
            surface="agents",
            template_path=PKG_TEMPLATES / "agent_regulatory.py.tmpl",
            required_vars=frozenset(
                {"class_name", "framework_id", "engine_class"},
            ),
        ),
    )
    # ...register_extension_scaffold(...) per scaffold
```

`ExtensionScaffold` is a frozen dataclass:

| Field | Meaning |
|---|---|
| `scaffold_id` | Caller-facing id (e.g. `"agent_regulatory"`). Single source of truth — referenced by tests, planner guidance, the `bootstrap_*` `scaffold=` kwarg. |
| `surface` | The L1-E surface this scaffold targets (`agents`, `plugins`, `deployments`, `tools`, `profiles`). The bootstrap action validates that `scaffold.surface == <action's surface>` and refuses on mismatch. |
| `template_path` | On-disk path to the template file. `string.Template` `$variable` substitution. |
| `required_vars` | The `template_vars` keys the caller must supply (in addition to the renderer's defaults: `name`, `name_snake`, `name_dash`). Validated before any IO at render time. |

Registration is validated at registration time: duplicate `scaffold_id`, missing template file, or unknown `surface` raises [`ExtensionScaffoldRegistryError`](../../src/polymathera/colony/design_monorepo/scaffolds/registry.py). A startup error beats a render-time error.

### Calling a scaffold

Every `bootstrap_*` action on `ToolBuilder` accepts an optional `scaffold=<id>` plus a `template_vars=<dict>` for the scaffold's `required_vars`. `scaffold=None` (the default) renders the blank L1-E template. With a scaffold id:

```python
await builder.bootstrap_agent(
    "opm_meg_regulatory",
    scaffold="agent_regulatory",
    template_vars={
        "class_name": "OPMMEGRegulatoryAgent",
        "framework_id": "fda_510k",
        "engine_class": "FDA510KFrameworkEngine",
    },
)
```

[`render_extension_scaffold`](../../src/polymathera/colony/design_monorepo/scaffolds/renderer.py) substitutes the variables and writes the result to the same `<surface>/<name>.<ext>` path the blank template would. Everything downstream is identical: same AST allow-list, same `commit_with_identity` attribution, same blackboard event, same L1-A discovery picks it up via `discovered_extensions.<surface>`.

### Where the registration hook fires

Domain packages (`polymathera-cps` and successors) declare a `polymathera.config_components` entry-point pointing at a `register_components()` callable. Colony's `ConfigurationManager.initialize()` walks the entry-point group at startup and calls each registered hook. CPS's hook calls `register_cps_scaffolds()` (and any other domain-specific registration); the scaffolds become available everywhere the package is pip-installed — same distribution mechanism as missions ([`registering-a-mission.md`](../guides/registering-a-mission.md)) and config components. No `polymathera-colony` change is required to add a new scaffold; the operator just installs the domain package (via the L1-G `cluster.extensions.packages` YAML knob in [`architecture/image-extensions.md`](image-extensions.md) or by adding it to the runtime image).

## Trust model

L1-E writes are agent-authored. The AST allow-list is the write-time gate; the sandbox is the run-time gate; the `DesignCheckpointer`-style commit attribution is the audit gate. All three apply uniformly to agent and human authorship — discovery (L1-A) cannot tell them apart, so the write side must reject disallowed surfaces regardless of provenance. PR 3 extends the same pipeline to L1-F (`src/`/`tests/`); PR 5 will route CPS-shaped scaffolds through L1-E's `bootstrap_*` actions via [L2-F](../../../cps/CPS_ALIGNMENT_PLAN.md).
