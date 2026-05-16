# Protected branches and human approval

Colony lets the operator nominate one or more branches of the design monorepo as *protected*. Network-visible and history-rewriting operations targeting a protected branch don't execute autonomously — they pause, post a request to the chat UI, and wait for the operator to approve or reject. Branches not on the protected list run uninterrupted, so agents can iterate on feature branches at full speed and only stop at the merge point.

## When you'd use this

You're authoring a regulatory submission on a `main` branch that an external auditor reviews. The agent may freely edit and commit on any number of `feature/<topic>` branches; the moment it tries to `merge_design` back to `main` or `push_remote` `main` to the remote, the operator must say yes first. The same applies to `cherry_pick_decisions` onto a protected target, `pull_remote(strategy=merge|rebase)`, and `rebase_onto`.

## The setting — `manifest.protected_branches`

[`DesignMonorepoManifest.protected_branches`](../../src/polymathera/colony/design_monorepo/manifest.py) is a tuple of `fnmatch` glob patterns. Default: `("main",)`. Stored in the monorepo's `.colony/manifest.json` under `protected_branches`:

```json
{
  "schema_version": 2,
  "tenant": "acme",
  "colony": "acme-colony",
  "program": "opm_meg_program",
  "target_system": "OPM-MEG Helmet",
  "design_repo_url": "https://github.com/<org>/monorepo_opm_meg.git",
  "protected_branches": ["main", "release/*", "stable-*"]
}
```

[`is_branch_protected(branch_name)`](../../src/polymathera/colony/design_monorepo/manifest.py) matches the current branch against every pattern via `fnmatch.fnmatch`. Empty `branch_name` (detached HEAD shows as `""`) is treated as unprotected. To **disable** the gate entirely, set `"protected_branches": []`.

## Which actions are gated

Two distinct rules apply:

### L1-F project-substance writes — refuse on protected

[`ProjectAuthoringCapability`](../../src/polymathera/colony/design_monorepo/capabilities.py) (`write_file`, `edit_file`, `delete_file`, `move_file`, line-ops, scaffolds, etc.) refuses *outright* on a protected branch. No approval round-trip — the agent gets a clear error and is expected to branch off:

```text
L1-F write_file: refusing to author on protected branch 'main'.
``create_branch`` + ``checkout_branch`` to a non-protected branch
first; merge back via ``merge_design`` when done (which gates
through human approval).
```

Rationale: every L1-F action commits a single file's worth of change. Gating each one through the operator would freeze the agent. The discipline is *branch first, then iterate, then ask once at the merge point*.

### Network / history-rewriting ops — gate through human approval

[`DesignCheckpointer`](../../src/polymathera/colony/design_monorepo/capabilities.py) actions that touch the protected branch route through the operator. They return a typed [`ProtectedOpResult`](../../src/polymathera/colony/design_monorepo/models.py) whose `status` distinguishes the two cases:

| Action | When `status="executed"` | When `status="pending_approval"` |
|---|---|---|
| `merge_design(source, target)` | `target` is not protected | `target` matches a `protected_branches` pattern |
| `push_remote(branch)` | `branch` is not protected | `branch` matches a pattern |
| `pull_remote(branch, strategy="merge"\|"rebase")` | `branch` is not protected OR `strategy="ff-only"` | `branch` matches a pattern AND strategy rewrites history |
| `cherry_pick_decisions(commits, target_branch)` | `target_branch` is not protected | `target_branch` matches a pattern |
| `rebase_onto(target_ref)` | current branch is not protected | current branch matches a pattern |

For the `executed` path the result carries `sha` (the resulting commit). For the `pending_approval` path the result carries `request_id` — the chat UI surfaces a typed `human_approval` message with `approve` / `reject` buttons. The user's click hits `POST /api/v1/sessions/{session_id}/human-approval/{request_id}/respond` (see [`HumanApprovalCapability`](../../src/polymathera/colony/agents/patterns/capabilities/human_approval.py)); the response arrives back to the capability via blackboard event `HumanApprovalProtocol.response_pattern()` and the queued op dispatches.

## The chat round-trip — worked example

User has been working with the agent on a feature branch `feature/serf-calibration`; the agent is now ready to merge back to `main`:

```
agent → run("merge_design", source_branch="feature/serf-calibration",
                            target_branch="main")
                  │
                  ▼
DesignCheckpointer.merge_design sees target_branch="main" matches
the protected pattern; calls _post_protected_approval(...)
                  │
                  ├── writes HumanApprovalRequest at
                  │     human_approval:request:appr_abc123
                  │     {question: "Merge feature/serf-calibration → main?",
                  │      options: ["approve", "reject"], ...}
                  │
                  ├── writes PendingProtectedOp at
                  │     design_monorepo:protected_op_pending:appr_abc123
                  │     {op_kind: "merge_design", target_branch: "main",
                  │      args: {source_branch: "feature/serf-calibration", ...}}
                  │
                  └── returns ProtectedOpResult{status="pending_approval",
                                                request_id="appr_abc123"}

The SessionOrchestratorCapability's HumanApprovalRelay (always running)
sees the request key and surfaces it to the chat UI as a typed
agent_question:

  ┌──────────────────────────────────────────────────────────┐
  │ Merge feature/serf-calibration → main?              │
  │   [approve]  [reject]                               │
  └──────────────────────────────────────────────────────────┘

User clicks [approve]. The frontend POSTs to
/api/v1/sessions/{session_id}/human-approval/appr_abc123/respond
which writes HumanApprovalResponse back to the blackboard.

DesignCheckpointer._on_protected_approval_response fires on the
matching pattern. Reads the PendingProtectedOp record, sees
op_kind="merge_design", dispatches the merge with the saved args.
Writes a ProtectedOpOutcome with the resulting SHA.

The chat UI surfaces the outcome:
  "Merge completed: 8f3a9c2…"
```

If the operator clicks **reject**, the `ProtectedOpOutcome` records `status="rejected"`, the merge does not happen, and the agent's next iteration sees the rejection event.

If nobody responds, the request sits indefinitely — there is no timeout today. The agent can re-prompt the operator via `respond_to_user` if it wants to nudge.

## What the agent's own logic looks like

The agent does NOT need to know about the approval round-trip in detail. It calls the capability action and branches on the typed result:

```python
r = await run(
    "merge_design",
    source_branch="feature/serf-calibration",
    target_branch="main",
)
if not r.success:
    # capability raised — argument shape issue, etc.
    await run("respond_to_user", content=f"merge_design errored: {r.error}")
elif r.output["status"] == "executed":
    await run(
        "respond_to_user",
        content=f"Merge completed: {r.output['sha'][:8]}",
    )
elif r.output["status"] == "pending_approval":
    await run(
        "respond_to_user",
        content="Merge is waiting for your approval in the chat.",
    )
elif r.output["status"] == "rejected":
    await run(
        "respond_to_user",
        content="Merge was rejected — leaving the source branch as-is.",
    )
```

The `pending_approval` arm is a no-op for the agent's state; the actual dispatch happens later from the event handler. The agent's next iteration may see the resulting `ProtectedOpOutcome` if it subscribes to the relevant pattern, or just observes the working-tree change via `git_status` on a later turn.

## Disabling the gate (for hands-off pipelines)

Some pipelines run end-to-end with no human in the loop (a research demo with no auditor, a development sandbox, an automated regression run). Two ways to opt out:

1. Set `"protected_branches": []` in the manifest — no branch is ever protected, every action runs autonomously.
2. Have agents always work on feature branches and route merges via a *separate* approval mechanism outside Colony (a GitHub PR review, a Gerrit CL). The framework provides the protected-branch gate as one option; it does not require it.

The gate's design intent is "interpose a person at the points where damage is widely visible (network-visible push, history rewrite, merge to the audited line)" — not "block all agent work on `main`". An auditor-required workflow sets protected branches non-empty; a research workflow leaves them empty.

## Related

- [`git-attribution.md`](git-attribution.md) — every approved merge / push commit honors the colony's `commit_principal` / `commit_co_author`.
- [`architecture/design-monorepo-capabilities.md`](../architecture/design-monorepo-capabilities.md) — the full action surface of `DesignCheckpointer`.
- [`architecture/project-substance-authoring.md`](../architecture/project-substance-authoring.md) — the L1-F "refuse outright on protected" path.
