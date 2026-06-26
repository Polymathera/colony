# R7-PLAN-KG-PERSIST v3.1 — single-file `.kg.json` snapshot + lazy multi-branch rehydrate + branch-annotated Kuzu

**Status**: PLANNED (v3.1 — surgical edit to v3 incorporating the architect's multi-branch annotated Kuzu design; rehydrate is per-branch, every node/edge carries `branches: set[str]`, Kuzu holds the union, snapshot filters to current branch). Awaiting user review.
**Authored**: 2026-06-25 by Claude.
**Related**: `[[primitives-not-pipelines]]`, `[[adversarial-minimality-audit]]`,
`[[extract-dont-bloat]]`, `[[no-llm-facing-framework-state]]`,
`[[surface-assumptions-in-plans]]`, `[[search-before-writing]]`,
`[[no-bandaids-durable-solutions]]`,
`[[read-own-code-before-coordinating-fix]]`,
`[[surgical-text-edits]]` (v3.1 was a surgical edit per the architect's explicit instruction: "Do not butcher the plan every time I ask for a change. Be surgical and precise.").

---

## 0. What the user fixed in v2 (the 4 NOTE-to-Claude corrections)

| NOTE | What v2 got wrong | v3 design |
|---|---|---|
| **4** | Invented per-source/per-scope partitioning. Nothing in the codebase does this — I fabricated it as a merge-conflict-surface "optimization." | **ONE knowledge graph, ONE file**: `.colony/colony.kg.json` at the design monorepo root. |
| **3** | Hard-wired snapshot into specific actions (`ingest_repo_map_literature`, `materialize_design_context`). | Generic `pre_commit_callbacks` mechanism on the commit-and-push helpers; KB registers a snapshot callback. Any future commit-hook user (audit trail, lockfile updater, etc.) plugs in the same way. |
| **1** | Eager-rehydrate via `@initialize_deployment` hook — would inflate cluster init time. | Lazy, user-initiated, mirroring VCM-mapping: `@action_executor rehydrate_kg()` + dashboard "Rehydrate KG" button. Nothing happens until triggered. |
| **2** | Assumed rehydrate from current clone's view. **Problem**: clones are per-agent (private), Kuzu is shared. Rehydrating from one agent's clone misses commits in other agents' branches that haven't yet pulled. | **Multi-branch annotated Kuzu** (architect's design): every Kuzu node and edge carries a `branches: set[str]` annotation. Rehydrate is per-branch — it loads one branch's `.kg.json` and tags every claim with that branch (adding to the existing set if the node/edge already exists from another branch). Kuzu holds the union across every branch ever hydrated. Queries can pass `branch_filter` to scope results to one branch; omit the filter for the union view. Snapshot writes only claims whose `branches` contains the branch being committed to. Full design in §5. |

## 1. Verified architectural baseline (audits A/B/C/D/E/F)

### 1.1 Producers (audit A)
The KG today has **one production write path**: the `Ingestor →
ClaimExtractor → GraphStore.add_claims` chain. Two extractors:
- `DeterministicClaimExtractor` at `knowledge/extractors/claims.py:66`
- `LLMClaimExtractor` at `knowledge/extractors/claims.py:212`

Ingestor is called from:
- `materialize_knowledge_sources` (PDFs) — `materialize.py:266`
- `materialize_design_context_sources` (design-context text files) — `materialize.py:894`
- `web_ui/backend/routers/kb.py:455` (HTTP upload)
- `MonorepoPersistedIngestor` (document-level cache wrapper)

**There is NO partitioning in the codebase today** (verified). The KG is one logical graph; multiple extractors / multiple sources all write to the same `GraphStore`.

### 1.2 Consumers (audit B)
| Consumer | file:line | reads |
|---|---|---|
| `GraphRetrievalCapability.run` | `knowledge/retrieval/graph.py:43-68` | `query` / `neighbours` |
| `SystemDesignCapability.search_design_context(path='kuzu')` | `design_monorepo/system_design.py:292-544` | `query` filtered by `citation_uri` |
| `SystemDesignCapability.find_inconsistencies` | `design_monorepo/system_design.py:558-680` | scans for contradiction predicates |
| `SystemDesignCapability.audit_hypothesis_coverage` | `design_monorepo/system_design.py:700-860` | scans for hypothesis predicates |
| `SystemDesignAgentBlueprint._find_bottleneck_rules` | `design_monorepo/process.py:3810-3901` | `query` LIMIT |

**Round-trip requirement**: every consumer reads `edge.confidence`, `edge.citation_uri`, `edge.source_id`, `edge.target_id`, `edge.predicate`, and `node.properties["surface"]`. Phase 3 of this plan upgrades the kg-merge schema so all of those survive serialisation.

### 1.3 Kuzu topology (audit D — corrected from v2)
**Kuzu is opened per-Ray-process**, not a Ray deployment. Each agent / deployment / worker calls `KuzuGraphStore.open(graph_db_path)` and opens the database directory locally. They share state via a **single on-disk DB on a shared volume** (e.g. `/mnt/shared/colony-design-graph.kuzu` per `cluster_config.py:256-260`).

Per-process `threading.RLock` (`graph.py:529`) serialises writes within a process. Across processes, Kuzu relies on the embedded DB's file-level locks (SQLite-like semantics).

`graph_store` is **intentionally omitted** from the deployment blueprints (`deps.py:465-469`, `:485-489`) — each process opens its own local handle to the shared DB; the GraphStore is not a remote resource.

**Implication for snapshot**: when an agent action writes claims to `add_claim`, they land in the shared Kuzu immediately — all other agent processes on the same volume see them at once. The `.kg.json` snapshot is NOT what makes claims visible to peers in the live cluster. It's what makes them visible to:
- (a) a fresh deployment on a different machine (cold-start)
- (b) a fork of the design monorepo on someone else's clone
- (c) git history / PR review / blame
- (d) merge resolution across branches

### 1.4 Per-agent clones (audit E)
Each agent gets its own clone at `/mnt/shared/agents/<agent_id>/clones/<scope_id>/` (resolved by `resolve_clone_path` at `design_monorepo/clones.py:54-84`). Lazy clone via `_lazy_clone_from_agent_metadata()` at `capabilities.py:810`.

Commits go to per-agent branches (prefixes `session/`, `agent/`, `tool/`, `fork/` — `client.py:104-115`). `client.push()` at `client.py:1150` pushes the current branch to `origin/<branch>`. Merges back to `main` happen via `merge_design()` (protected-branch gated) or via the auto `_on_remote_change()` handler reacting to upstream events.

### 1.5 Commit-and-push helpers (audit E + direct read)
Module-level functions at `capabilities.py:141-225`:
- `_commit_all_and_push(client, identity, message) -> (sha, push_status)` — calls `client.commit_with_identity(..., all_changes=True)` then `client.push()`.
- `_commit_paths_and_push(client, identity, message, paths, all_changes) -> (sha, push_status)` — paths-variant.

Five callers (audit E inventory):
- `ingest_repo_map_literature` at `capabilities.py:1651`
- `checkpoint_state` at `capabilities.py:2380`
- `commit_state` (protected-op dispatch) at `capabilities.py:2567`
- `_on_episode_checkpoint` (convergence handler) at `capabilities.py:3376`
- `_dispatch_protected_op[commit_state]` at `capabilities.py:3727`

**Existing `fire_post_commit` event** at `capabilities.py:1251-1291` publishes a `MonorepoCommitProtocol` to the colony blackboard AFTER commit succeeds. It is **read-only on the commit, async, informational** — it CANNOT stage files into the commit. So it is NOT the right surface for the snapshot hook; we need a NEW pre-commit mechanism.

### 1.6 VCM-mapping initiation pattern (audit F — the template for rehydrate)
- LLM action `mmap_repo` at `agents/patterns/capabilities/vcm.py:305-417` (`@action_executor(interruptible=True)`)
- UI button "Map to VCM" at `web_ui/frontend/src/components/repo/RepoMapTab.tsx:247` → POST `/vcm/map` → background task
- Lazy: nothing happens until triggered
- Idempotent: re-trigger returns `status="already_mapped"` (mapped scopes recorded in shared `VirtualPageTableState.mapped_scopes`)
- Reconciliation: on replica restart, `_reconcile_scope_mappings()` reads the shared state and auto-materialises any prior mappings locally

**Phase 2 of this plan mirrors this pattern exactly** for the KG rehydrate UX.

## 2. The two-layer model

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Design monorepo (git, versioned, merge-aware)                      │
  │                                                                     │
  │  In each branch's checkout:                                         │
  │    .colony/colony.kg.json   ← THAT branch's view of the KG          │
  │                              schema: {version, namespaces, claims}  │
  │  (The branch is implicit in git context — the file itself           │
  │   does NOT carry a branch field, so merges across branches are      │
  │   handled cleanly by kg-merge.)                                     │
  └────────────────────────────────┬──────────────────┬─────────────────┘
                                   │                  │
              rehydrate(branch)    │                  │ snapshot (pre-commit)
              lazy, user-init;     │                  │ filters Kuzu to claims
              tags every loaded    │                  │ whose `branches` set
              claim with `branch`  │                  │ contains the current
                                   │                  │ commit's branch
                                   ▼                  ▲
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Shared Kuzu (one on-disk DB on a shared volume; opened             │
  │  per-process; live state seen by every agent immediately)           │
  │                                                                     │
  │  Each node and edge carries `branches: set[str]` annotation.        │
  │  Kuzu holds the UNION of every branch ever hydrated into it.        │
  │  Query API supports `branch_filter`; omit filter → union view.      │
  │                                                                     │
  │  Producers: Ingestor → ClaimExtractor → GraphStore.add_claims       │
  │             (auto-tagged with the writing agent's current branch    │
  │             via a contextvar — see §6).                             │
  │  Consumers: GraphRetrievalCapability + 4 SystemDesignCapability     │
  │             actions + _find_bottleneck_rules                        │
  │             (may pass branch_filter; default = union view).         │
  └─────────────────────────────────────────────────────────────────────┘
```

**Hot path** (normal operation): all agents read/write the shared Kuzu directly. Each `add_claim` is auto-tagged with the writing agent's current branch (resolved from a contextvar set by the capability that triggered the action — §6). Other agents see new claims in real-time via Kuzu. The branch-filtered query semantics let an agent on a feature branch see only that branch's view if it asks, or the union view if it doesn't.

**Cold path** (fresh deployment, lost Kuzu volume, fork of the monorepo on a different machine, new colony bootstrap): the operator or session-agent clicks "Rehydrate KG" → loads `<agent-clone>/.colony/colony.kg.json` for whatever branch the clone is on → every loaded claim gets tagged with that branch. To pull multiple branches into Kuzu, repeat per-branch (the LLM action accepts a `branch` parameter; a convenience "rehydrate all active branches" iterates known agent clones).

## 3. The snapshot mechanism (NOTE 3 — generic pre-commit hooks)

### 3.1 New framework: `pre_commit_callbacks` on commit-and-push

Add a generic pre-commit-callback registry. The minimal surface:

```python
# in design_monorepo/commit_hooks.py (new module — extract per
# [[extract-dont-bloat]])

@dataclass(frozen=True)
class PreCommitContext:
    client: DesignMonorepoClient     # for staging additional files
    identity: AgentIdentity | CommitIdentity
    message: str
    branch: str
    paths: list[Path] | None         # None = all_changes mode
    working_dir: Path                # the per-agent clone root

PreCommitCallback = Callable[[PreCommitContext], Awaitable[None]]

class PreCommitRegistry:
    """Process-singleton registry of pre-commit callbacks.

    Callbacks are invoked in registration order BEFORE
    client.commit_with_identity stages files. A callback may write
    new files into the working tree; those files are picked up by
    the subsequent ``git add -A`` (all_changes mode) or must be
    explicitly added to the ``paths`` list (paths mode).
    """
    def register(self, name: str, callback: PreCommitCallback) -> None: ...
    def unregister(self, name: str) -> None: ...
    async def fire_all(self, ctx: PreCommitContext) -> None: ...
```

### 3.2 Helper integration

Promote `_commit_all_and_push` + `_commit_paths_and_push` from module functions to use the registry:

```python
async def _commit_all_and_push(
    client: DesignMonorepoClient,
    identity: AgentIdentity | CommitIdentity,
    message: str,
    *,
    branch: str = "<resolved>",
) -> tuple[str, str]:
    await get_pre_commit_registry().fire_all(
        PreCommitContext(
            client=client, identity=identity, message=message,
            branch=branch, paths=None, working_dir=client.working_dir,
        )
    )
    # … existing commit_with_identity(all_changes=True) + push …
```

Same shape for `_commit_paths_and_push` (paths-mode pre-commit callbacks must either limit themselves to writing into already-staged paths or extend the `paths` list).

**Failure mode**: if a pre-commit callback raises, the commit is **aborted** (the commit-and-push helper re-raises). Loud, not silent. Rationale: if KB snapshot fails, the commit shouldn't proceed with a stale `.kg.json` — that would break the rehydrate-from-main invariant.

**Open Q O1 (§9)**: confirm abort-on-failure vs log-and-continue.

### 3.3 KB subscribes its snapshot callback

In a one-time module-init in `knowledge/persistence.py`:

```python
def register_kg_snapshot_callback() -> None:
    get_pre_commit_registry().register(
        "kb.snapshot_kg",
        snapshot_kg_to_monorepo,  # the writer described in §5
    )
```

Called once per process from `set_knowledge_deps()` (`deps.py:302`) — deps init already runs once per process, so this is the natural one-time registration point.

The KB callback:
1. Reads claims from the shared Kuzu via `GraphStore.export_claims(branch=ctx.branch)` — filters to claims whose `branches` set contains the branch being committed to.
2. Atomically writes `<working_dir>/.colony/colony.kg.json` (temp + `os.replace`).
3. Returns — the surrounding `_commit_all_and_push` then sees the file in `git add -A` and includes it in the commit.

**Important**: the callback runs in the AGENT'S clone for the AGENT'S current branch. The Kuzu it reads from is the SHARED Kuzu, which may also contain claims tagged with OTHER branches (from peer agents). The `branch=ctx.branch` filter is what keeps branch-X commits from leaking branch-Y-only claims into branch X's `.kg.json` file. A claim that exists on BOTH branches (e.g. shared foundational knowledge merged from main) has both branches in its `branches` set and so legitimately appears in both branches' snapshots.

### 3.4 Why not the existing `fire_post_commit`?
`fire_post_commit` runs AFTER the commit and only PUBLISHES an event. We need to STAGE A FILE INTO the commit — that's a different shape of hook (pre, mutating) than `fire_post_commit` (post, observational). **Both coexist**: pre-commit hooks stage, post-commit hooks notify.

## 4. The single-file format (NOTE 4 — ONE KG, ONE file)

`.colony/colony.kg.json` at the design monorepo root. Schema:

```json
{
  "version": "1.0",
  "namespaces": {"colony": "...", "rdf": "...", "rdfs": "...", ...},
  "claims": [
    {
      "subject": "<surface subject string>",
      "predicate": "<predicate>",
      "object": "<surface object string>",
      "confidence": 0.92,
      "citation": {
        "source_uri": "design_context://...",
        "section_path": "...",
        "char_start": 1450,
        "char_end": 1520
      },
      "subject_properties": {"surface": "..."},
      "object_properties": {"surface": "..."},
      "edge_properties": {},
      "provenance": {
        "extractor": "LLMClaimExtractor@<git_sha_short>",
        "extracted_at": "2026-06-25T12:34:56Z",
        "extractor_run_id": "ing-7f3a..."
      }
    }
  ]
}
```

Stable claim ordering: `(subject, predicate, object,
citation.source_uri)`. Unchanged graphs produce byte-identical files → empty git diffs.

**The file does NOT carry a `branch` field.** The branch context is implicit in git: the file in the `main` checkout IS the main view; the file in the `feature/x` checkout IS the feature/x view. When kg-merge merges two versions of the file (e.g. during a PR merge), the result represents whatever branch the merge happened on — no field-level conflict. Branch annotation lives ONLY in Kuzu (as `branches: set[str]` per node/edge), never in the persisted file.

**Phase 3** extends `git_merge/kg_merge.py` to handle this schema (set-merge on `(subject, predicate, object)`; on duplicate keep highest `confidence`; tiebreak on lexicographic `provenance.extractor_run_id`; functional-predicate logic carried forward unchanged).

**Open Q O2 (§9)**: confirm `.colony/colony.kg.json` path. Alt: `.colony/kg.json` doesn't match the merge-driver pattern `**/*.kg.json` (requires at least one char before `.kg.json`).

## 5. The rehydrate mechanism (NOTE 1 — lazy, user-initiated like VCM; NOTE 2 — multi-branch annotated)

### 5.1 Surface — per-branch primitive + multi-branch convenience

Mirror the VCM-mapping pattern (lazy + LLM action + UI button).

**LLM-callable primitives** on `KnowledgeCuratorCapability`:

```python
@action_executor(planning_summary=(
    "Rehydrate one branch's view of the KG into the shared Kuzu. "
    "Loads <clone>/.colony/colony.kg.json from the agent's clone "
    "for the named branch (post `git fetch`) and tags every loaded "
    "claim with that branch in Kuzu's `branches` annotation. "
    "Idempotent: re-rehydrating an unchanged branch is a fast no-op. "
    "If `branch` is omitted, defaults to the agent's current "
    "branch's upstream."
))
async def rehydrate_kg(
    self, *, branch: str | None = None,
) -> dict[str, Any]:
    """Returns: {branch, claims_in_file, claims_newly_added,
    claims_newly_tagged, source_commit_sha}."""

@action_executor(planning_summary=(
    "Convenience: rehydrate every branch currently held by an active "
    "agent clone in the cluster. Iterates the dashboard's known "
    "agent-clone registry and calls `rehydrate_kg(branch=...)` for "
    "each distinct branch. Use after a fresh deployment or to refresh "
    "Kuzu's union view across all in-flight work."
))
async def rehydrate_all_active_branches(self) -> dict[str, Any]:
    """Returns: {branches_rehydrated: list[str], totals: {...}}."""
```

**UI button**: "Rehydrate KG" in the Design Monorepo tab (sibling of "Map to VCM"). POST `/kb/rehydrate?branch=<name>` (or omit `branch` for current-branch default). A dropdown next to the button lets the operator pick "this branch" / "all active branches" / a specific branch from the active list. Mirror VCM's UI flow at `web_ui/backend/routers/vcm.py:229-275`.

### 5.2 Mechanism

`rehydrate_kg(branch=B)`:
1. Resolves `B` (passed value, or the agent's current branch).
2. `await asyncio.to_thread(client._refresh_against_origin)` — `git fetch origin` (no hard-reset on a different branch; we read via `git show origin/B:.colony/colony.kg.json` rather than checkout).
3. Reads the file content via `client._repo.git.show(f"origin/{B}:.colony/colony.kg.json")` (no working-tree mutation; multiple rehydrates can run without disturbing the agent's checkout).
4. Parses as `KgFile` (Pydantic).
5. Iterates claims, calls `GraphStore.import_claims(batch, branch=B)`
   — see §6 for the import semantics:
   - Node/edge new → insert with `branches={B}`.
   - Node/edge exists → add `B` to its `branches` set (no-op if already present).
6. Returns counts (newly-added vs newly-tagged).

### 5.3 How this dissolves the NOTE 2 multi-agent problem

NOTE 2 worried: rehydrating from one agent's clone misses other agents' claims (because clones are private).

**Resolution**: the multi-branch model makes Kuzu the convergence point. Each agent's branch can be hydrated independently; Kuzu holds the UNION; queries can choose per-branch or union view. Concretely:

- Agent A on branch `session/A` hydrates its own branch — Kuzu gains `session/A`-tagged claims.
- Agent B on branch `session/B` hydrates its own branch — Kuzu gains `session/B`-tagged claims.
- A claim that exists on BOTH branches (e.g. a foundational fact inherited from `main`) gets both tags on its single Kuzu entry.
- An operator triggers `rehydrate_all_active_branches` — all known agent branches converge into Kuzu's union view.
- A consumer that wants "only what's on main" passes `branch_filter='main'` (§6). A consumer that wants the union view passes nothing.

**Live cross-agent visibility** during normal operation still happens at the Kuzu level (writes are instantaneous): an agent on `session/A` writing a new claim auto-tags it with `session/A` and Kuzu's union view shows it to everyone. The file-level layer kicks in only at cold-start / fresh-machine / lost-volume / cross-deployment scenarios.

### 5.4 Branch identifier convention

**Recommended**: strip remote prefix. The branch tag stored in Kuzu and accepted by query/import APIs is the canonical branch name — `main`, `session/abc-123`, `agent/xyz`, NOT `origin/main` or `refs/heads/main`. The rehydrate primitive normalizes the input (strips `origin/` if present); the snapshot pre-commit hook reads the agent's branch via `git rev-parse --abbrev-ref HEAD` and uses that stripped name.

Rationale: the same branch in two different remotes is still the same branch. Using the stripped name keeps annotations comparable across mirrors and prevents accidental sharding by remote.

**Open Q O3 (§9)**: confirm stripped-name convention.

### 5.5 Idempotency + tagging vs adding

The `import_claims(claims, branch=B)` semantics:
- "Newly added" = the underlying (subject, predicate, object) triple was not in Kuzu at all before.
- "Newly tagged" = the triple existed but did not yet have `B` in its `branches` set.
- "No-op" = the triple existed AND already carried `B`.

Re-rehydrating an unchanged branch is therefore mostly no-ops (some "newly tagged" if Kuzu happened to acquire those triples from another branch's hydration in the interim). No special skip-if-unchanged logic needed in v1; if measurements warrant it later, add a `last_rehydrated_file_sha` marker keyed by `(branch, file_sha)`.

## 6. ABC additions (branch-aware, per `[[adversarial-minimality-audit]]`)

### 6.1 Schema additions on existing model classes
- `GraphNode` gains `branches: frozenset[str] = frozenset()` —
  per-node branch annotation.
- `GraphEdge` gains `branches: frozenset[str] = frozenset()` —
  per-edge branch annotation.
- `Claim` gains `provenance: dict[str, Any] = {}` (per O4) — the
  round-trip provenance dict that the persisted-file schema needs.

### 6.2 Contextvar for the writing branch
- New module-level `CURRENT_BRANCH_CONTEXT: ContextVar[str | None] =
  ContextVar("colony_current_branch", default=None)` in
  `knowledge/stores/graph.py`.
- Capabilities set it at the start of any action that may write to
  the KG: `with set_current_branch(ctx.branch): await
  action_body(...)`. The contextvar is `asyncio`-safe and
  thread-safe via `contextvars`.
- `GraphStore.add_node` / `add_edge` / `add_claim` / `add_claims`
  read the contextvar if no explicit `branch` kwarg is passed; absent
  contextvar AND absent kwarg → raise (loud, not silent — this
  prevents claims from landing in Kuzu with empty `branches` sets,
  which would make them invisible to every branch-filtered query).

### 6.3 New methods on `GraphStore`
- `export_claims(*, branch: str | None = None) -> AsyncIterator[Claim]` —
  iterate every claim with full provenance + `branches`. If `branch`
  is set, filter to claims whose `branches` contains that branch.
- `import_claims(claims: Sequence[Claim], *, branch: str) -> ImportResult` —
  batch insert; `branch` is REQUIRED (not Optional) because import is
  the only operation where the writing branch can't be inferred from
  capability context (the rehydrate primitive knows the source
  branch, not the contextvar). Per-claim semantics:
  - Triple new → insert with `branches={branch}`.
  - Triple exists, `branch` not in its set → add it (returns "newly
    tagged" in `ImportResult`).
  - Triple exists, `branch` already in set → no-op.

### 6.4 Branch filter on existing read methods
- `query(query: str, *, branch_filter: str | None = None) -> GraphQueryResult`
- `neighbours(node_id, *, predicate=None, depth=1, branch_filter: str | None = None) -> GraphQueryResult`
- `get_node(node_id, *, branch_filter: str | None = None) -> GraphNode | None` —
  returns `None` if node exists but its `branches` doesn't contain
  the filter; identical to "node truly missing" from the consumer's
  perspective (consumers don't need to distinguish).
- `count(*, branch_filter: str | None = None) -> tuple[int, int]`

`branch_filter=None` = union view (default; preserves existing
consumer behavior since today no consumer passes a branch filter).

### 6.5 Kuzu schema changes
- `Entity` table gains a `branches` column (JSON-encoded sorted list
  of strings — kept as a JSON column rather than a separate `Branch`
  table because the cardinality is small per-node and the query DSL
  only filters via substring/array-contains).
- `Relates` table gains the same.
- One-time schema migration on `KuzuGraphStore.open()` adds the
  columns if the table exists without them (Kuzu's `ALTER TABLE …
  ADD COLUMN`).
- Query implementation: filter via Kuzu Cypher's
  `WHERE 'branch_name' IN <column>` on read.

### 6.6 InMemory impl
- `branches: set[str]` on each node/edge entry.
- Filtering via plain Python set membership.

Both impls share the same contract; tests verify behavior on both.

**Open Q O4 (§9)**: `Claim.provenance: dict[str, Any]` field
(recommended) vs sidecar dict — unchanged from v3.

## 7. Phasing

### Phase 1 (~3 days) — Snapshot + commit-hook framework + branch-aware ABC
1. Schema additions: `GraphNode.branches`, `GraphEdge.branches`,
   `Claim.provenance` (per §6.1 + O4); migration on Kuzu schema for
   the new `branches` columns.
2. `CURRENT_BRANCH_CONTEXT` contextvar + `set_current_branch()`
   helper in `knowledge/stores/graph.py`.
3. `add_*` mutators read the contextvar; raise if both contextvar
   and explicit `branch` are absent.
4. New ABC reads: `branch_filter` on `query`, `neighbours`,
   `get_node`, `count`. New ABC writes: `export_claims(branch=...)`,
   `import_claims(claims, branch=...)`. Impls in InMemory + Kuzu.
5. Capability wrappers set the contextvar at action entry:
   `KnowledgeCuratorCapability`, `RepoStateProvider`, any other
   capability whose actions write to the KG (audit A confirmed only
   one production write path today — `Ingestor.ingest_document` —
   but the contextvar pattern is generic).
6. New module `design_monorepo/commit_hooks.py` with
   `PreCommitRegistry` + `PreCommitContext`.
7. Promote `_commit_all_and_push` / `_commit_paths_and_push` to
   `async` and call `fire_all(ctx)` before `commit_with_identity`.
   Update all 5 call sites.
8. New module `knowledge/persistence.py` with `KgFile` (Pydantic),
   `KgSnapshotWriter` (filters via
   `export_claims(branch=ctx.branch)`), `register_kg_snapshot_callback`.
9. Hook `register_kg_snapshot_callback()` into `set_knowledge_deps()`
   (`deps.py:302`).
10. Phase 1 tests (§8).

### Phase 2 (~1.5 days) — Lazy rehydrate (per-branch + multi-branch)
1. `KgRehydrator` in `knowledge/persistence.py` (per-branch loader
   using `git show origin/<branch>:.colony/colony.kg.json` rather
   than working-tree checkout — non-disruptive to the agent's
   current state).
2. `@action_executor rehydrate_kg(branch=None)` on
   `KnowledgeCuratorCapability`.
3. `@action_executor rehydrate_all_active_branches()` — enumerates
   distinct branches across active agent clones (uses the dashboard's
   agent-clone registry; if no registry exists yet, the v1
   implementation walks `/mnt/shared/agents/*/clones/*/.git/HEAD`).
4. Backend route `POST /kb/rehydrate?branch=...` (mirror `/vcm/map`).
5. Frontend "Rehydrate KG" button + branch-selector dropdown in
   `RepoMapTab.tsx`.
6. Phase 2 tests.

### Phase 3 (~1 day) — kg-merge schema upgrade
1. Extend `git_merge/kg_merge.py` to handle the full-claim schema
   (`{"version", "namespaces", "claims": [...]}`).
2. Set-merge on `(subject, predicate, object)`; conflict-resolution
   tiebreaks per §4.
3. Cross-version refusal (loud merge conflict if `version` differs).
4. Extend `test_merge_drivers.py`.

## 8. Tests

**Phase 1 — commit hooks + branch-aware ABC + snapshot**:
1. `test_pre_commit_callback_fires_before_commit` — register a callback that records the call order; assert it ran before `commit_with_identity`.
2. `test_pre_commit_callback_failure_aborts_commit` — callback raises → no commit, no push.
3. `test_pre_commit_callback_can_stage_file` — callback writes `.colony/colony.kg.json`; after commit, the file is in HEAD.
4. `test_add_claim_requires_branch_context_or_kwarg` — calling `add_claim` with neither contextvar set nor explicit `branch` raises (loud failure-mode for the no-branches-set anti-pattern).
5. `test_add_claim_picks_up_contextvar_branch` — set contextvar, call `add_claim` without explicit branch, assert resulting node/edge `branches == {ctx_branch}`.
6. `test_export_import_round_trip_with_branches_inmemory` — add 100 claims under branch A; `export_claims(branch=A)` → fresh store `import_claims(..., branch=A)`; assert identical `count()` + every claim's `branches == {A}`.
7. `test_export_import_round_trip_with_branches_kuzu` — same against Kuzu.
8. `test_import_claims_adds_branch_to_existing_triple` — pre-seed Kuzu with triple `(X,p,Y)` tagged `{A}`; `import_claims([same_triple], branch=B)`; assert resulting tag is `{A,B}` and `claims_newly_added=0, claims_newly_tagged=1`.
9. `test_query_branch_filter_returns_only_matching` — seed mixed-branch claims; `query(..., branch_filter=A)` returns only A-tagged triples; no-filter call returns the union.
10. `test_snapshot_filters_to_current_branch` — Kuzu has triples tagged `{A}`, `{B}`, `{A,B}`; snapshot callback invoked with `ctx.branch=A`; assert resulting `.kg.json` contains the `{A}` and `{A,B}` triples and excludes the `{B}`-only triple.
11. `test_kb_snapshot_callback_round_trip` — extract 100 claims into Kuzu (tagged with branch `session/x`), commit via test stub on `session/x`, assert `.colony/colony.kg.json` parses back to identical claim set when re-loaded with `import_claims(..., branch='session/x')`.
12. `test_ingest_action_produces_kg_file_in_same_commit` — end-to-end with stubbed monorepo + stubbed LLM extractor; assert the resulting commit contains BOTH sidecars AND `.colony/colony.kg.json`.

**Phase 2 — lazy rehydrate**:
13. `test_rehydrate_loads_from_branch_upstream_via_git_show` — pre-seed origin with `.colony/colony.kg.json` on branch A; agent on different branch; call `rehydrate_kg(branch=A)`; assert Kuzu gains A-tagged claims AND the agent's working-tree state is undisturbed (no checkout side-effect).
14. `test_rehydrate_is_idempotent_when_unchanged` — rehydrate twice; second call's `claims_newly_added == 0 AND claims_newly_tagged == 0`.
15. `test_rehydrate_strips_remote_prefix_from_branch_arg` — call `rehydrate_kg(branch='origin/main')`; assert resulting Kuzu tag is `'main'` (not `'origin/main'`).
16. `test_rehydrate_all_active_branches_iterates_distinct_branches` — set up 3 active agent clones on 3 branches; call `rehydrate_all_active_branches`; assert each branch's claims land with the right tag.
17. `test_rehydrate_action_returns_correct_counts`.

**Phase 3 — kg-merge schema upgrade**:
18. `test_kg_merge_full_claims_schema`.
19. `test_kg_merge_refuses_cross_version`.
20. `test_kg_merge_set_merge_dedup_on_subject_predicate_object`.
21. `test_kg_merge_confidence_tiebreak`.

## 9. Open questions for you (must resolve before Phase 1 starts)

**O1**: Pre-commit callback failure mode — **abort the commit** (recommended) or log+continue?
**O2**: KG file path — `.colony/colony.kg.json` (matches `**/*.kg.json`) or different name?
**O3**: Branch-identifier convention — **stripped name** (`main`, `session/abc`, recommended) or full ref (`refs/heads/main`, `origin/main`)? The rehydrate primitive normalizes either way; the contextvar + Kuzu `branches` set hold whichever convention you pick. Stripped is simpler and matches the existing branch-prefix constants at `client.py:104-115`.
**O4**: Add `Claim.provenance: dict[str, Any]` field (recommended) or carry provenance via a sidecar dict?
**O5**: `rehydrate_all_active_branches` in Phase 2 (recommended — the multi-agent value of the design only kicks in when you can hydrate every active branch in one click) or defer to a later phase?
**O6**: Garbage collection of branch tags when a branch is deleted/merged — defer to a later phase (recommended; orphan tags are storage cost only, not correctness) or include in Phase 1?

(No more questions like v2's D1.b/D6/D7/D9 — those were partitioning
artefacts and don't apply.)

## 10. Adversarial pass (`[[adversarial-pass-on-own-code]]`)

1. **Dict-key uniqueness**: `PreCommitRegistry` keys callbacks by name string; collision detection raises on `register("foo", ...)` if `"foo"` already registered. ✓
2. **Field-selection**: `PersistedClaim` round-trips every field every current consumer reads (verified by audit B). ✓
3. **Defaults**: `pre_commit_callbacks` starts empty; the snapshot callback is registered explicitly from `set_knowledge_deps()`. Empty registry = no behavior change (existing commit flows still work). ✓
4. **Duplication**: searched for any existing pre-commit hook mechanism — none. `fire_post_commit` is the only commit-adjacent hook today and it's post + observational. New `PreCommitRegistry` does not duplicate. ✓
5. **Branch-set membership semantics** (new for v3.1): the `branches` field on `GraphNode`/`GraphEdge` is a SET, not a list — addition is idempotent (re-tagging a node with a branch it already has is a no-op). The `add_claim` failure mode (raise if no branch in context AND no explicit kwarg) PREVENTS the failure mode where a claim lands with `branches=∅` and becomes invisible to every branch-filtered query. ✓
6. **Existing-consumer compatibility** (new for v3.1): all 5 current KG-reading call sites in §1.2 pass NO `branch_filter` argument today — they'll continue to work unchanged (union view = today's view because there's only one branch's worth of claims initially). The branch filter is opt-in; existing reads are unaffected. ✓

## 11. Cost

| Phase | Time | LOC | Tests |
|---|---|---|---|
| 1 (commit hooks + branch-aware ABC + snapshot) | ~3d | ~700 | 12 |
| 2 (lazy rehydrate UX — per-branch + multi-branch) | ~1.5d | ~350 | 5 |
| 3 (kg-merge schema upgrade) | ~1d | ~150 | 4 |
| **Total** | **~5.5d** | **~1200** | **~21** |

Across: 2 new modules (`design_monorepo/commit_hooks.py`, `knowledge/persistence.py`), 1 schema upgrade (`git_merge/kg_merge.py`), modifications to `Claim` / `GraphNode` / `GraphEdge` models + Kuzu schema migration (new `branches` columns on `Entity` + `Relates`) + 5 existing commit-and-push call sites + `set_knowledge_deps()` + the `KnowledgeCuratorCapability` (+ contextvar wiring on every capability that issues KG-write actions) + 1 new backend route + 1 new frontend button + a dropdown for branch selection.

v3.1 delta vs v3: +1.5d / +300 LOC / +5 tests for the branch-annotation work (schema migration, contextvar plumbing, multi-branch rehydrate, per-branch snapshot filter, branch-filter on every read method). The architect's design adds real capability (multi-branch union view + branch-scoped queries) at the cost of more schema/test surface.

---

**Reviewer**: please answer O1–O4 in §9 before I start Phase 1.
