# `VCMCapability`

Agent-facing facade over the [Virtual Context Manager](virtual-context-memory.md). Lets any agent map repositories or blackboard scopes into VCM pages, request and lock pages, watch the filesystem for changes, and inspect VCM state — all as `@action_executor` methods.

Code: `polymathera.colony.agents.patterns.capabilities.VCMCapability`. Whether a given agent gets the capability is a configuration choice (a blueprint entry in the agent's `capability_blueprints`), not a code change.

## Why a capability?

Before this capability, the only callers of `VirtualContextManager.mmap_application_scope` were the FastAPI dashboard router, the `polymath` CLI, and `MemoryCapability.initialize`. Agents couldn't drive the VCM. The session agent's only response to *"map this repo"* was a chat message. Lifting the VCM into a capability turns that into a real action the LLM planner can call.

The capability also owns concerns that don't belong on the VCM deployment itself:

- **Filesystem watching**: react to changes in the directory backing a mapped scope.
- **Lifecycle events**: emit blackboard records when a scope is mapped, unmapped, or re-indexed.

## Action surface

All actions are `@action_executor()` methods returning a uniform `{"status"|"ok", ..., "message"}` shape. Failures degrade to error dicts rather than raising so the LLM observes them as data.

### Mapping lifecycle

| Action | Purpose |
|--------|---------|
| `mmap_repo` | Map a git repository into the VCM. Accepts `origin_url` *or* `local_repo_path` (converted to `file://`); optional `branch` / `commit` / `scope` / explicit `scope_id` / `MmapConfig`. Emits `mapped:` event. |
| `mmap_blackboard_scope` | Map the agent's own (or another) blackboard scope so LLM workers can attend to it. |
| `munmap_scope(scope_id)` | Reverse a mapping. Flushes pending writes; emits `unmapped:`. |
| `is_scope_mapped(scope_id)` | Boolean check via the VCM. |
| `get_scope_status(scope_id)` | Full mapping record: config, syscontext, materialised page count. |
| `list_mapped_scopes` | Every visible mapping. (Privileged on the VCM side; degrades to an error dict when the caller lacks permission.) |

### Page lifecycle

| Action | Purpose |
|--------|---------|
| `request_pages(page_ids, …, lock_duration_s=…)` | Issue a fault and wait. Optional lock pins pages in place during a multi-step reasoning sequence. |
| `lock_pages(page_ids, ttl_s)` | Acquire eviction-prevention locks. Per-page best-effort; reports `{locked, failed}`. |
| `unlock_pages(page_ids)` | Release locks. Distinguishes `unlocked` from `already_unlocked`. |
| `extend_lock(page_id, additional_s)` | Refresh an existing lock. |
| `get_page_graph(max_nodes)` | JSON-serialisable snapshot of the page graph. |
| `list_stored_pages` / `get_pages_for_scope` | Inspection helpers. |
| `get_vcm_stats` | Cluster-wide stats (page table + storage). |

### Filesystem watcher

| Action | Purpose |
|--------|---------|
| `watch_repo(scope_id, paths, on_change=…)` | Start a [`watchfiles`](https://watchfiles.helpmanual.io/) task. `on_change` ∈ `{notify_only, invalidate, reindex}`. Returns a `watch_id`. |
| `unwatch_repo(watch_id)` | Stop a watch and reap its task. |
| `list_watches` | Enumerate active watches with fire counts. |

Watch state is persisted across `serialize_suspension_state` / `deserialize_suspension_state`, so an agent that suspends mid-watch resumes its watchers automatically.

## Event protocol

Lifecycle events go to the capability's own blackboard partition under `VCMEventProtocol`:

```
mapped:{scope_id}            value = {scope_id, source_type, origin_url?, branch?, commit?, ts}
unmapped:{scope_id}          value = {scope_id, ts}
reindexed:{scope_id}         value = {scope_id, watch_id, paths, ts}
page_fault:{fault_id}        value = {fault_id, page_ids, priority, requester, ts}
watch_fired:{watch_id}       value = {watch_id, scope_id, paths, ts}
```

`scope_id` values contain colons; the protocol's `parse_*` helpers split correctly. Other capabilities subscribe with `input_patterns=[VCMEventProtocol.mapped_pattern(), …]`.

## Configuration

```python
from polymathera.colony.agents.patterns.capabilities import VCMCapability
from polymathera.colony.agents.scopes import BlackboardScope

VCMCapability.bind(
    scope=BlackboardScope.SESSION,        # default partition for events
    namespace="vcm",                       # capability sub-namespace
    watch_root="/mnt/shared/filesystem",  # base for relative watch paths
    max_concurrent_watches=16,
)
```

The capability is wired into the session agent in `web_ui/backend/routers/sessions.py`. Coordinator agents that need direct VCM control add it to their `extra_capabilities` in YAML test configs.

## Failure modes

- **`mmap_repo` with no `origin_url` and no `local_repo_path`** → `status="error"`.
- **VCM raises** (Ray timeout, transport failure, missing page graph) → `status="error"` with the exception text in `message`.
- **`watch_repo` against a non-existent path** → registration succeeds (the path may appear later); the background `watchfiles` task fails on its first iteration, logs the error, and stays registered until `unwatch_repo`.
- **KERNEL-ring endpoints called from USER context** (`get_vcm_stats`, `list_mapped_scopes`) → the VCM rejects with an auth error; the capability surfaces `{"stats": None, "message": "<reason>"}` rather than crashing.

## Test surface

Unit tests at `colony/src/polymathera/colony/agents/patterns/capabilities/tests/test_vcm_capability.py` (42 tests) cover every action with a mocked VCM handle, the `VCMEventProtocol` round-trips, and the watcher's start/stop/list/suspension-restore flow.

## Open follow-ups

- **Incremental page-graph rebuild** under `on_change="reindex"` instead of munmap+remap. Requires a new VCM endpoint.
- **Settings UI** for active watches.
- **Auto-detection of the backing path** for a `scope_id` when `paths` is omitted to `watch_repo`. Today the caller must supply paths explicitly.
