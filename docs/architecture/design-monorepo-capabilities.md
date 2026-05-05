# Design-Monorepo Capabilities

Three `AgentCapability` subclasses give an agent first-class access
to the design monorepo. Bind them all at once with
`design_monorepo_capability_blueprints()`; each one resolves a
**per-agent local clone** at agent-init time so multiple agents in
the same process never collide.

## One-time setup — point the colony at your repo

Each colony has **one** design monorepo. Configure it on the
dashboard's **landing page** (the screen you see after login,
before any session is started):

1. In the **Colonies** panel, pick or create a colony with **+ New
   colony**.
2. On that colony's row, click the pencil next to *Design monorepo*,
   paste the GitHub (or any git) URL, optionally change the branch,
   and click **Save**.
3. Click **New Session**.

The landing page is the right home for this gesture because the
dashboard's tabs (including the Design Monorepo inspector) only
appear once a session is active — so the URL has to be configured
*before* the session boots. The save writes onto the colony's row in
the `colonies` Postgres table (`design_monorepo_url`,
`design_monorepo_branch`, `design_monorepo_commit`). Every
subsequent SessionAgent created in that colony reads it at
session-creation time, attaches it to
`AgentMetadata.parameters["design_monorepo_url"]`, and the design-
monorepo capability trio lazy-clones the repo into the per-agent
working directory on first action.

Authentication is the operator's responsibility. The
`docker-compose.yml` already plumbs `GITHUB_TOKEN` into every Ray
service. For `https://github.com/...` clone URLs the framework
rewrites the URL in-process to embed the token
(`https://x-access-token:$GITHUB_TOKEN@github.com/...`) via
`utils.git.utils.inject_github_token`, so non-interactive clones
work for private repos without the user having to embed credentials
in the URL they paste. ssh remotes, GitLab, and URLs that already
carry credentials pass through untouched — git's standard machinery
handles those itself. Whatever access the token has, the capability
has — clone failures (404, auth, etc.) surface verbatim from `git`.

```http
GET  /api/v1/colonies/{colony_id}/design-monorepo
PUT  /api/v1/colonies/{colony_id}/design-monorepo
     {"origin_url": "https://github.com/me/my-design.git",
      "branch": "main", "commit": "HEAD"}
```

## The three capabilities

| Class | Role | Example actions |
|---|---|---|
| `RepoStateProvider` | Read-only query surface | `get_repo_state`, `find_existing_tool`, `list_recent_decisions`, `diff_against_checkpoint`, `get_branch_topology` |
| `DesignCheckpointer` | Write side: branches, checkpoints, merges | `checkpoint_state`, `restore_checkpoint`, `fork_design`, `merge_design`, `cherry_pick_decisions`, `commit_state`, `tag_checkpoint`, `list_checkpoints`, `list_forks`, `diff_design` |
| `ToolBuilder` | Scaffold a new tool into `tools/<purpose>/<name>/` | `bootstrap_repo` |

`DesignCheckpointer` is also event-driven: it auto-tags an
`auto_quiescence_<iso8601>` checkpoint when the convergence runtime
settles with uncommitted changes (`@event_handler` on
`ConvergenceQuiescenceProtocol.quiescence_pattern()`).

## Minimal example — wire all three into an agent

```python
from polymathera.colony.design_monorepo import (
    design_monorepo_capability_blueprints,
)

SessionAgent.bind(
    ...,
    capability_blueprints=[
        ...,
        *design_monorepo_capability_blueprints(),
    ],
)
```

The helper returns three `AgentCapabilityBlueprint`s, in this order:
`RepoStateProvider`, `DesignCheckpointer`, `ToolBuilder`. Pass
`auto_checkpoint_on_quiescence=False` to opt out of auto-checkpointing,
or `read_only_state=True` to point `RepoStateProvider` at the shared
read-only clone (good for agents that only read).

`auto_checkpoint_on_quiescence=False` does two things in lockstep: it
disables the auto-checkpoint behaviour *and* unsubscribes the
capability from `ConvergenceQuiescenceProtocol.quiescence_pattern()`
on the action policy's event queue. The second half matters for
`reactive_only` agents (e.g., `SessionAgent`) — without it, every
episode boundary would wake the LLM planner and the agent would
plan-and-act in a tight loop on infrastructure events. The
remote-change subscription stays active either way; it only fires
when the upstream actually changes.

`RepoStateProvider` and `ToolBuilder` declare no event handlers —
they are pure action surfaces. Their constructors pass
`input_patterns=[]` to `super().__init__`, which is the
[`AgentCapability` convention](../../src/polymathera/colony/agents/base.py)
for "explicit opt-out: do not subscribe to anything on this scope."
Without that opt-out, the base class's legacy empty-patterns
fallback would subscribe to `"*"` and the agent's own
`policy:action_started:*` lifecycle writes would loop back into the
action policy's event queue — in `reactive_only` mode that triggers
a fresh `plan_step` for every action the policy itself dispatched.

## Per-agent clones

By default each capability resolves its working directory to:

```
/mnt/shared/agents/<agent_id>/clones/<scope_id>/
```

The `/mnt/shared` mount is the colony-shared docker volume — the
clone survives Ray actor restarts. Different agents working on the
same scope get distinct directories, so:

- Two agents can simultaneously check out different branches of the
  same design monorepo without colliding.
- A failed merge in one agent's clone does not corrupt another
  agent's view.
- Branch creation (`fork_design`) writes to the per-agent clone
  only; the global VCM mapping (the read-only view of `main`) does
  not see the change until the agent merges back.

Override the path by passing `working_dir=...` to a specific
capability constructor (skip the helper for that one), or change
`COLONY_SHARED_ROOT` to redirect the entire layout.

## Shared read-only clone

For agents that never write — typically planners that just read
state — the read-only flag selects a single per-node clone:

```python
*design_monorepo_capability_blueprints(read_only_state=True),
```

The shared clone lives at `/mnt/shared/shared_clones/<scope_id>/`.
Every `RepoStateProvider(read_only=True)` on the same node opens it,
saving disk and clone time. `DesignCheckpointer` and `ToolBuilder`
ignore the flag — they always use a per-agent writable clone.

## Branch-update events

When the convergence runtime rebuilds a scope's page graph after the
remote-watcher observes upstream commits, it writes a
`VCMEventProtocol.reindexed:<scope_id>` event. `DesignCheckpointer`
subscribes to that pattern and translates it into a coarser
`DesignMonorepoEventProtocol.branch_changed:<scope_id>` event the
agent's planner consumes.

Subscribe in your own capability (or directly in the planning prompt)
to react:

```python
@event_handler(pattern=DesignMonorepoEventProtocol.branch_changed_pattern())
async def on_branch_changed(self, event, scope):
    scope_id = DesignMonorepoEventProtocol.parse_branch_changed_key(event.key)
    # plan a checkout / merge / rebase against the per-agent clone
```

## Reference — clone path

`polymathera.colony.design_monorepo.clones.resolve_clone_path(*, agent, scope_id, read_only)`:

| Arg | Effect |
|---|---|
| `agent` | Owning agent. Required when `read_only=False`. |
| `scope_id` | Per-clone scope key (typically the VCM scope id or a branch name). |
| `read_only` | `False` → `/mnt/shared/agents/<agent_id>/clones/<scope_id>/`; `True` → `/mnt/shared/shared_clones/<scope_id>/`. |

Override the base by exporting `COLONY_SHARED_ROOT`.
