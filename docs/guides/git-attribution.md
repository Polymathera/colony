# Git commit attribution

Every commit Colony agents make to your design monorepo lands with a configurable Git author + co-author. The operator chooses who appears as the commit *principal* (the `Author:` line in `git log`) and the *co-author* (the `Co-authored-by:` trailer in the commit message). Both are per-colony settings stored in the colonies table and respected uniformly by every agent-authored commit path — L1-E meta-tooling writes, L1-F project-substance writes, `DesignCheckpointer` tags, fork merges, cherry-picks, repo-map initialization, every tag operation.

## Why this exists

A regulated-design workflow needs an honest audit trail. "Who authored this commit?" must answer with a person and a system in a way the audit can defend:

- Sometimes the *colony* should appear as the principal (an agent acted within sanctioned scope; the operator is the co-author on the trailer).
- Sometimes the *user* should appear as the principal (the operator drove the change through the chat; the colony agent is the co-author).
- Sometimes neither (a generic system identity for agent-only operations).

The colony's settings UI is the single source of truth; the framework enforces it uniformly so an agent can't (deliberately or accidentally) bypass it.

## The settings — what the operator configures

The attribution surface splits across two layers (see `colony/github_identity_fix_plan.md`):

**Per-colony — the preference**. Two columns on the `colonies` row:

| Column | Type | Used as |
|---|---|---|
| `commit_principal` | `str` (well-known: `"colony"` / `"user"` / `"agent"`; anything else treated as an agent-type label) | Decides who shows as `Author:` (schema default `"colony"`) |
| `commit_co_author` | `str \| None` (same value space) | Decides the `Co-authored-by:` trailer; `None` = no trailer (schema default `"user"`) |

**Per-user — the identity**. The `git_user_name` / `git_user_email` that resolve when either field is `"user"` come from the *user* row, populated by the GitHub OAuth callback (the "Connect GitHub" button on the user profile — see [`connect-github.md`](connect-github.md)). Operators do not type these in; verified values come from GitHub directly to prevent commit impersonation.

When the user hasn't connected GitHub and `"user"` is selected, the trailer (or principal) is dropped silently with a `logger.warning` — the commit succeeds with the colony-side identity only. This is intentional: commit attribution must reflect reality.

## Where the operator sets it

The dashboard's Settings page exposes the four fields. Backend routes (defined in [`colony/web_ui/backend/routers/colonies.py`](../../src/polymathera/colony/web_ui/backend/routers/colonies.py)):

| Method | Path | Body / response |
|---|---|---|
| `GET` | `/api/v1/colonies/{colony_id}/git-attribution` | Returns `{commit_principal, commit_co_author}` as `GitAttributionConfig` |
| `PUT` | `/api/v1/colonies/{colony_id}/git-attribution` | Body: `SetGitAttributionRequest`. Returns the persisted row. |

Operators can drive the PUT directly from a shell:

```bash
curl -X PUT https://<dashboard-host>/api/v1/colonies/<colony_id>/git-attribution \
  -H "Authorization: Bearer <session-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "commit_principal": "user",
    "commit_co_author": "colony"
  }'
```

Setting persists in Postgres on the colony row; agents read it at every commit. For the `"user"` side to render correctly, each user contributing to the colony must connect GitHub via their profile UI.

## How agents read it — the `Identity` union

The framework's commit primitive is `DesignMonorepoClient.commit_with_identity(principal, message, paths=[...])`. The first arg is an [`Identity`](../../src/polymathera/colony/design_monorepo/identity.py) — a tagged union over:

- `AgentIdentity(agent_id, role, colony_id, agent_email_domain="<domain>")` — the transactional agent identity. Used when the colony has `commit_principal == "colony"`.
- `CommitIdentity(name, email)` — a free-form Git identity. Used when the colony has `commit_principal == "user"`; carries the operator's configured `git_user_name` / `git_user_email`.

The capability layer's [`_resolve_attribution`](../../src/polymathera/colony/design_monorepo/capabilities.py) helper reads two metadata blocks populated at session-create time:

- `agent.metadata.parameters["git_attribution"]` — the per-colony preference (`commit_principal`, `commit_co_author`) from [`auth_service.get_git_attribution`](../../src/polymathera/colony/web_ui/backend/auth/service.py).
- `agent.metadata.parameters["github_identity"]` — the per-user OAuth-verified identity (`git_user_name`, `git_user_email`, `user_github_login`, `tenant_installation_id`) from [`auth_service.get_user_github_identity`](../../src/polymathera/colony/web_ui/backend/auth/service.py) + [`auth_service.get_tenant_github_installation`](../../src/polymathera/colony/web_ui/backend/auth/service.py).

The helper returns `(principal, co_author_or_None)`. The capability layer's `_commit_attribution(message)` formats the message with the `Co-authored-by:` trailer when applicable and returns `(principal, decorated_message)`. Every commit-producing action threads through these two helpers.

## What lands in `git log`

With `commit_principal=colony`, `commit_co_author=user` (the default for fresh colonies after operator configures the user identity):

```text
commit 8f3a…
Author: agent_x7k9 <agent_x7k9@<colony-id>.colony.local>
Date:   Thu May 16 14:22:11 2026

    L1-F write_file: src/opm_meg/serf/calibration.py

    Co-authored-by: Jane Doe <jane.doe@example.com>
```

With `commit_principal=user`, `commit_co_author=colony`:

```text
commit 8f3a…
Author: Jane Doe <jane.doe@example.com>
Date:   Thu May 16 14:22:11 2026

    L1-F write_file: src/opm_meg/serf/calibration.py

    Co-authored-by: agent_x7k9 <agent_x7k9@<colony-id>.colony.local>
```

With `commit_principal=colony`, `commit_co_author=None`:

```text
commit 8f3a…
Author: agent_x7k9 <agent_x7k9@<colony-id>.colony.local>
Date:   Thu May 16 14:22:11 2026

    L1-F write_file: src/opm_meg/serf/calibration.py
```

## Which commit paths honor this

Every commit-producing capability action goes through the same `_commit_attribution` → `commit_with_identity` chain. The list (all in [`colony/design_monorepo/capabilities.py`](../../src/polymathera/colony/design_monorepo/capabilities.py)):

| Capability | Actions |
|---|---|
| `DesignCheckpointer` | `tag_checkpoint`, `merge_design` (fast-forward + non-FF), `cherry_pick_decisions`, `commit_state`, `create_tag` |
| `RepoStateProvider` | `initialize_repo_map` (commits the seeded `repo_map.yaml`) |
| `ProjectAuthoringCapability` (L1-F) | `write_file`, `edit_file`, `delete_file`, `move_file`, `insert_lines`, `delete_lines`, `replace_lines`, `make_directory`, `remove_directory`, `copy_file`, `set_file_executable`, plus the four L2-G `scaffold_*` actions |
| `ToolBuilder` (L1-E) | `bootstrap_plugin`, `bootstrap_agent`, `bootstrap_deployment`, `bootstrap_tool_capability`, `bootstrap_profile` |

The framework guarantee: if any of these actions produces a commit, the principal / co-author are the colony's configured pair. There is no agent-side override hatch — the resolver reads the setting and the message decorator runs unconditionally.

## What "agent_email_domain" is

`AgentIdentity` carries an `agent_email_domain` derived from the colony's manifest (defaults to `<colony-id>.colony.local`). The Git author email becomes `<agent_id>@<agent_email_domain>`. This gives every agent commit a deterministic, colony-scoped, machine-distinguishable email — useful for filtering audit history by colony AND distinguishing agent commits from operator commits in `git shortlog`.

## Failure modes

- **Operator sets `commit_principal=user` but the user has not connected GitHub** → `_resolve_attribution` cannot resolve the user-side `git_user_name` / `git_user_email`; `_safe_resolve("user")` catches the `ValueError` from `resolve_commit_identity` and falls through with `co_author=None`. The commit still succeeds with the synthetic `colony:<colony_id>` identity as principal; the trailer is dropped silently with a `logger.warning`. Operators see this as "my commits don't have my name on them" → fix: click "Connect GitHub" on the profile.
- **Colony row doesn't exist (typo'd `colony_id`)** → `set_git_attribution` raises `KeyError`; the dashboard surfaces it as 404.
- **An agent commits before the operator has configured anything** → schema defaults apply (`commit_principal=colony`, `commit_co_author=user`). If the user has not connected GitHub, the `Co-authored-by:` trailer is dropped (only the colony principal commits, no trailer).

## Related

- [`github-app-setup.md`](github-app-setup.md) — operator: register the Colony GitHub App + configure per-tenant installation.
- [`connect-github.md`](connect-github.md) — user: how the "Connect GitHub" button populates the per-user identity this attribution flow reads.
- [`registering-a-mission.md`](registering-a-mission.md) — how the SessionAgent dispatches missions whose spawn paths produce commits.
- [`architecture/project-substance-authoring.md`](../architecture/project-substance-authoring.md) — L1-F write surface; every action listed there respects attribution.
- [`architecture/design-monorepo-authoring.md`](../architecture/design-monorepo-authoring.md) — L1-E bootstrap surface; same attribution chain.
