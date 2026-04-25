# `UserPluginCapability`

Lets users *drop a directory* into a workspace to add a new tool the agent can use. Discovers `SKILL.md` skill bundles and `plugin.json` plugin packages, validates parameters, and runs each skill inside a [`SandboxedShellCapability`](sandboxed-shell-capability.md) container.

Code: `polymathera.colony.agents.patterns.capabilities.UserPluginCapability`. Subpackage: `_plugin/{schema,discovery}.py`. Sample plugin: `colony/src/polymathera/colony/samples/plugins/colony-samples/`.

The on-disk layout deliberately overlaps with [Claude Code's Skills/Plugins](https://code.claude.com/docs/en/skills.md) so users can move directories between Colony and Claude Code with minimal translation.

## Why it exists

Colony will frequently meet domain-specific tools the framework knows nothing about — a particular CFD simulator, a CAD tool, a domain-specific verifier, an in-house data pipeline. Hardcoding adapters for every one is a losing battle. Users add a directory, the agent discovers it. The capability is a *registry + adapter*, never a runner — execution always goes through the sandbox.

## Layout

```
<plugin_root>/
└── colony-samples/
    ├── .claude-plugin/
    │   └── plugin.json
    └── skills/
        ├── code-complexity/
        │   ├── SKILL.md
        │   └── scripts/
        │       ├── run.sh
        │       └── complexity.py
        ├── scientific-debugging/
        │   ├── SKILL.md
        │   └── scripts/{run.sh, worksheet.py}
        └── systemic-vulnerability-scan/
            ├── SKILL.md
            └── scripts/{run.sh, scan.py}
```

A standalone skill (no plugin) sits at `<skill_root>/<name>/SKILL.md`. A plugin namespaces its skills under `<plugin>/<skill>` so two plugins can ship skills with the same local name.

## `SKILL.md` frontmatter

```yaml
---
name: code-complexity
description: |
  Compute per-function cyclomatic complexity for Python source files…
when_to_use: |
  Triggered when Python files are involved and the user wants a
  quantitative read on code complexity, refactor targets, or
  maintenance risk hotspots.
sandbox_image_role: default      # picks the Docker image for execution
script: scripts/run.sh           # path inside the skill dir
params:
  path: { type: string, required: true }
  threshold: { type: integer, required: false }
  top_n: { type: integer, required: false }
timeout_seconds: 120
paths: "**/*.py"                  # optional activation hint
disable-model-invocation: false   # default; true → user-only skill
---

# Skill body

Markdown body shown to the LLM when it `get_skill`-s the skill in detail.
```

The skill ships its own executable scripts under `scripts/`. They run inside the sandbox image declared by `sandbox_image_role` (defaults to the capability's `default_sandbox_image_role`, which is `"default"`).

## Discovery roots

Default search order (highest priority first):

1. `<workspace_root>/.colony/skills` — session/project (when the workspace is mounted)
2. `~/.colony/skills` — per-user
3. `/etc/colony/skills` — operator-managed shared

Same scheme for `plugins/`. Higher-priority sources win on name collisions; the loser is logged at INFO. Plugin-namespaced skills (`<plugin>/<name>`) cannot collide.

`extra_plugin_roots` / `extra_skill_roots` add paths at `SYSTEM` priority — used by the session agent to ship the bundled `colony-samples` plugin without shadowing what the user installed locally.

## Action surface

| Action | Purpose |
|--------|---------|
| `list_skills(source=…)` | Every loaded skill, optionally filtered by source (`session` / `user` / `system` / `plugin`). |
| `get_skill(name)` | Full metadata + body markdown. |
| `search_skills(query, max_results=10)` | Substring match against name / description / `when_to_use`, ranked. |
| `list_plugins` | Loaded plugins with their skill counts. |
| `reload_skills` | Force a rescan — useful when the user just installed something. |
| `run_skill(name, params=…, container_id=…, …)` | Validate + execute. Launches its own container if `container_id` is None and stops it on exit; otherwise runs in the caller's container. |

`get_action_group_description` enumerates every loaded skill with a one-liner so the LLM sees them in the planning prompt without a separate `list_skills` call.

## Execution

```python
async def run_skill(self, name, *, params=None, container_id=None, …):
    sk = self._resolve_skill(name)
    self._validate_params(sk, params)         # required + type check

    sandbox = self.agent.get_capability_by_type(SandboxedShellCapability)
    if container_id is None:
        launched = await sandbox.launch_container(
            image_role=sk.sandbox_image_role or self._default_sandbox_image_role,
            extra_volumes=[{"src": str(sk.directory), "dst": "/skill", "mode": "ro"}],
            …,
        )
        container_id = launched["container_id"]
        owned = True

    try:
        return await sandbox.execute_command(
            container_id=container_id,
            command=self._render_script_command(sk, params),  # bash -lc 'cd /skill && bash <script> --p1 v1 …'
            timeout_seconds=sk.timeout_seconds,
        )
    finally:
        if owned:
            await sandbox.stop_container(container_id)
```

Param validation:

- `required: true` enforced.
- Type strings (`string`, `integer`, `number`, `boolean`, `array`, `object`) checked. `bool` is excluded from `integer` to avoid silent acceptance of `True`.
- Unknown params are passed through (the LLM may add extras).

Command rendering:

- The script path itself supports `{name}` placeholders (rare).
- All other params become `--name value` positional args, individually shell-quoted.
- The whole thing is wrapped as `bash -lc 'cd /skill && bash <script> <args>'`.

## The bundled `colony-samples` plugin

Three skills shipped with the package and auto-discovered by the session agent:

| Skill | What it does |
|-------|--------------|
| `colony-samples/code-complexity` | McCabe cyclomatic complexity for Python files (stdlib only). |
| `colony-samples/scientific-debugging` | Emits a structured RCA worksheet — observe → hypothesise → predict → experiment → conclude. |
| `colony-samples/systemic-vulnerability-scan` | Heuristic scanner for `bare_except`, `shell=True`, mutable defaults, hard-coded secrets, the documented `write_transaction` return-bug pattern, and asserts in non-test files. |

Wired into the session agent via:

```python
UserPluginCapability.bind(
    scope=BlackboardScope.SESSION,
    extra_plugin_roots=[_bundled_samples_plugins_root()],
)
```

The helper resolves to `polymathera.colony.samples.__file__ / .. / plugins`, so the plugin ships with the wheel and is discovered without extra mounts.

## Compatibility with Claude Code

| Field | Claude Code | Colony |
|-------|-------------|--------|
| `name`, `description`, `when_to_use`, `paths`, `disable-model-invocation` | ✓ | ✓ |
| `sandbox_image_role` | — | Colony-only |
| `script` | optional (inline shell injection allowed) | required |
| `params` type validation | loose | strict (rejects type mismatches) |
| Auto-discovery | `~/.claude/skills`, `.claude/skills` | `~/.colony/skills`, `<workspace>/.colony/skills`, `/etc/colony/skills` |

Skills can be copied between the two ecosystems. Inline shell injection (Claude Code's ` !`...` ` syntax) is not supported — every Colony skill must declare a `script`, because every skill execution is containerised.

## Security

- **Sandbox boundary**: every skill runs inside `SandboxedShellCapability`. The capability never executes code in the host process.
- **`disable-model-invocation: true`**: skills with this flag are skipped by `search_skills` and refused by `run_skill` (override with the `allow_model_invocation_override` blueprint kwarg for testing).
- **Image role**: a skill that requests an unknown `sandbox_image_role` fails at `launch_container` time.
- **Param validation**: protects against shell-quoting confusion in `run_skill` since the substituted args are individually quoted.

## Test surface

`tests/test_user_plugin_capability.py` (27 tests). Uses real filesystem `tmp_path` for discovery + a stub `SandboxedShellCapability` for execution. Covers: frontmatter parser (no FM / valid / malformed), source-priority collisions, per-skill error isolation, plugin namespacing, action surface, end-to-end `run_skill` with launch/exec/stop wiring, param required/type validation, `disable-model-invocation` gating, missing-sandbox error, launch failure surfacing, shell-quoting helper.

## Open follow-ups

- **Dynamic per-skill `@action_executor`**: the design proposes registering each discovered skill as its own action key (e.g., `skill.colony-samples.code-complexity`). The dispatcher walks `cls.__dict__`, not instance attrs, so this requires a dispatcher refactor; v1 ships `run_skill(name)` instead.
- **Procedural-memory sync**: when the memory system supports skill storage, add `_sync_to_procedural_memory()` so feedback can refine skills over time.
- **Plugin `agents/` directories**: a plugin could ship its own child-agent classes; out of scope for v1.
- **Settings UI**: a Skills tab listing discovered skills with enable/disable toggles.
