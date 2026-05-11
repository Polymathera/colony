# Design-monorepo extensions (L1-A)

How a design monorepo declares its own **L4 extensions** — plugins, agents, deployments, tools, profiles — and how Colony discovers them at runtime.

## TL;DR

A design monorepo's `.colony/manifest.json` (`schema_version` 2 and up) carries an optional `extensions` block whose per-surface fields name the directories Colony walks to find extensions:

```json
{
  "schema_version": 2,
  "tenant": "acme", "colony": "...", "program": "...",
  "target_system": "...", "design_repo_url": "...",
  "extensions": {
    "plugins":     { "directory": ".colony/plugins/" },
    "agents":      { "directory": ".colony/agents/" },
    "deployments": { "directory": ".colony/deployments/" },
    "tools":       { "directory": ".colony/tools/" },
    "profiles":    { "directory": ".colony/profiles/" }
  }
}
```

Each surface field is optional. An omitted surface (or an entirely missing `extensions` block, which is what a v1 manifest looks like) falls back to the default in [`DEFAULT_SURFACE_DIRS`](../../src/polymathera/colony/design_monorepo/manifest.py) — for example, plugins always default to `.colony/plugins/` whether or not the manifest mentions them.

A surface directory that doesn't exist on disk is equivalent to an empty one — no error.

## What discovery means per surface

| Surface | File convention | What discovery returns |
|---|---|---|
| `plugins` | `<dir>/<skill-name>/SKILL.md` (Claude-style skills; `plugin.json` optional) | `list[SkillSpec]` — same shape `UserPluginCapability` consumes |
| `agents` | `<dir>/<file>.py`, each declaring `Agent` subclasses | `dict[str, type[Agent]]` keyed by class name |
| `deployments` | `<dir>/<file>.py`, each declaring classes wrapped with `@serving.deployment(...)` | `dict[str, type]` keyed by class name (detection: `__deployment_config__` attribute) |
| `tools` | `<dir>/<file>.py`, each exposing a top-level `register(registry: ToolRegistry) -> None` callable | `ToolRegistry` populated with `ToolAdapter`s |
| `profiles` | `<dir>/<file>.yaml`, top-level mapping | `dict[str, dict]` keyed by filename stem |

The `tools` convention is intentionally different from `agents` / `deployments`: tool adapters carry construction state (DB sessions, HTTP clients, Docker handles) so the file controls its own instantiation via `register(registry)`. Agents and deployments declare classes only — the framework instantiates them later, at agent-spawn / deployment-deploy time.

## Public surface

[`polymathera.colony.design_monorepo.extensions`](../../src/polymathera/colony/design_monorepo/extensions.py) exports:

- `discover_plugins(repo_root, manifest=None) -> list[SkillSpec]`
- `discover_agents(repo_root, manifest=None) -> dict[str, type[Agent]]`
- `discover_deployments(repo_root, manifest=None) -> dict[str, type]`
- `discover_tools(repo_root, manifest=None) -> ToolRegistry`
- `discover_profiles(repo_root, manifest=None) -> dict[str, dict]`
- `discover_all(repo_root, manifest=None) -> DiscoveredExtensions` — convenience bundle of all five.

Pass the loaded manifest to honour per-surface directory overrides; omit it and the defaults apply.

## Wiring on agents

`RepoStateProvider` exposes a lazy, cached `discovered_extensions: DiscoveredExtensions` property. The first access on a given capability instance:

1. Reads `<working_dir>/.colony/manifest.json` if present. Best-effort — a missing or malformed manifest falls back to defaults.
2. Calls `discover_all(working_dir, manifest)`.
3. Memoizes the result for the lifetime of the capability.

```python
cap = RepoStateProvider(agent=..., working_dir=repo_root)
snap = cap.discovered_extensions
# snap.plugins / snap.agents / snap.deployments / snap.tools / snap.profiles
```

Discovery is deliberately not eager at `__init__` because the base capability may defer the actual clone (lazy `git clone` on first client access); reading from a not-yet-cloned `working_dir` would produce empty results that the cache would then stick.

### Cache invalidation

The cache invalidates automatically on a mtime fingerprint covering the manifest, `.colony/`, and the *resolved* surface directories (manifest overrides if any, defaults otherwise — the same paths `discover_all` walks). Mutations that bump those mtimes — the L1-E authoring path's `write_file` calls, `rm`, `mv`, manifest edits — all auto-invalidate.

The one mutation pattern the fingerprint cannot catch is editing an *existing* file's contents in place (Linux doesn't bump parent-dir mtime on child-content modify). Call `RepoStateProvider.invalidate_extensions()` for that case.

## Migration: v1 → v2

[`polymathera.colony.tools.manifest_migrate.migrate_manifest`](../../src/polymathera/colony/tools/manifest_migrate.py) rewrites a v1 manifest in place with `schema_version: 2` and a default `extensions` block. Idempotent — re-running on a v2+ manifest returns `MigrationResult(was_migrated=False)` and makes no disk writes.

```python
from polymathera.colony.tools.manifest_migrate import migrate_manifest

result = migrate_manifest(repo_root)
# result.was_migrated, result.from_version, result.to_version
```

Optional commit hook (uses the same `DesignMonorepoClient.commit_with_identity` machinery `DesignCheckpointer` uses internally):

```python
from polymathera.colony.design_monorepo.identity import AgentIdentity

result = migrate_manifest(
    repo_root,
    commit_identity=AgentIdentity(agent_id="ops", role="migrator", colony_id="local"),
)
# result.commit_sha — populated only when a write actually happened
```

The commit is paths-scoped: only the modified manifest file is staged, so an unrelated dirty working tree doesn't leak into the migration commit.

## Trust model

L1-A is read-only. Discovery walks the surface directories and loads each `*.py` file by reading its source and `exec()`-ing it into a fresh `types.ModuleType` — deliberately bypassing `importlib`'s bytecode cache so an invalidate-and-rediscover sequence always picks up the latest source. A single bad file is logged at WARNING and skipped — broken extensions don't poison the whole monorepo, but they DO execute their top-level code at load time. The deeper write-side validation (AST allow-list at write time, sandbox at execution time) is L1-E's responsibility (PR 2 in the alignment plan) — see [Risk #5 in CPS_ALIGNMENT_PLAN.md](../../../cps/CPS_ALIGNMENT_PLAN.md).

## Forward compatibility

`schema_version` gates incompatible shape changes. Adding a new surface kind to the `extensions` block is an incompatible change because `extra="forbid"` rejects unknown keys at validation time. When the day comes:

1. Bump `MANIFEST_SCHEMA_VERSION` and update `ExtensionsConfig`.
2. Update `migrate_manifest` to bump older manifests, populating the new surface's default directory.
3. Add `discover_<new_surface>` to the discovery module; add it to `DiscoveredExtensions`.
