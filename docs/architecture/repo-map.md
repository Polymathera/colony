# `.colony/repo_map.yaml` — Repo Map

A design monorepo will usually want to be paged into VCM as **multiple
sources** — code, literature, frozen submodules — not one. The repo
map declares one row per source and is version-controlled inside the
monorepo so the mapping stays consistent across colonies, replicas,
and time.

Lives at `<repo_root>/.colony/repo_map.yaml`. Loaded by
`polymathera.colony.design_monorepo.repo_map.RepoMap.load`. When
absent, the materialiser falls back to a single default `git_repo`
source over the whole tree, preserving the historical one-source
behaviour.

## Schema

```yaml
schema_version: 1

sources:
  # 1) Code subtree, with build artifacts excluded. Override the
  #    deployment's MmapConfig defaults for this row only — finer-
  #    grained pages keep semantic search precise on hot code.
  - name: design-code
    type: git_repo
    start_dir: tools/
    exclude_globs: ["**/build/**", "**/__pycache__/**"]
    binary_policy: skip
    static: false
    flush_threshold: 8        # smaller groups → smaller pages
    flush_token_budget: 2048

  # 2) Literature directory — PDFs go through the chunker.
  - name: literature
    type: literature
    start_dir: literature/
    chunk_target_tokens: 800

  # 3) Frozen external dependency, declared as a submodule. Pinned
  #    pages stay hot in cache; useful when an LLM keeps faulting
  #    these.
  - name: external-foo
    type: git_repo
    submodule: third_party/foo
    start_dir: src/
    static: true
    pinned: true

# Each row carries an explicit destination via ``ingest_to``.
# ``knowledge_base`` (default) ingests via the process-singleton
# Ingestor; ``vcm`` is documentation-only — it records that this
# path has been promoted to VCM by a sources: row above and must
# not be KB-ingested.
knowledge_routing:
  - paths: ["literature/curated/**/*.pdf"]
    ingest_to: knowledge_base    # default; explicit for clarity
    profile: scientific_paper

  - paths: ["standards/**/*.pdf"]
    # ingest_to omitted ⇒ knowledge_base

  - paths: ["literature/promoted/**/*.pdf"]
    ingest_to: vcm               # promoted — KB materialiser skips it
```

`schema_version` is required and currently `1`. Unknown values fail
loud at load time. Extra fields on a source row are rejected
(`extra: forbid`) so a typo is reported instead of silently ignored.

## Source rows

Every row produces one `mmap_application_scope` call. The kwargs are:

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Used as the suffix when composing `scope_id` for multi-source repos: `f"{base_scope_id}:{name}"`. The single-source default fallback (`name == "default"`) keeps using the caller-supplied scope id. |
| `type` | yes | `"git_repo"` or `"literature"`. Custom types register through `ContextPageSourceFactory`. |
| `origin_url` | one of `origin_url` / `submodule` | Git URL. Mutually exclusive with `submodule`. |
| `submodule` | one of `origin_url` / `submodule` | Path under `.gitmodules`. The materialiser resolves it: URL from `.gitmodules`, commit from the parent repo's pinned gitlink. |
| `branch` | no | Defaults to `"main"`. |
| `commit` | no | Defaults to `"HEAD"`. Submodule rows pin to the gitlink commit regardless of this field. |
| `start_dir` | no | Sub-tree restriction (see [`GitRepoContextPageSource`](git-repo-context-page-source.md)). |
| `include_globs` | no | List of gitignore-style patterns. |
| `exclude_globs` | no | List of gitignore-style patterns. |
| `binary_policy` | no | `git_repo` only — `"skip"` (default) / `"include"`. The literature source forces `"include"`. |
| `static` | no | `true` = frozen-commit; `false` = live-watched. |
| `chunk_target_tokens` | no | `literature` only. |
| `chunk_overlap_tokens` | no | `literature` only. |
| `flush_threshold` | no | Max records per VCM page (default `20`). See "Page-size knobs" below. |
| `flush_token_budget` | no | Max tokens per VCM page (default `4096`). |
| `pinned` | no | If `true`, pages from this source are never evicted from the VCM cache (default `false`). |

### Page-size knobs (`flush_threshold`, `flush_token_budget`, `pinned`)

Each source row produces a stream of *records* (one file for
`git_repo`, one prose chunk for `literature`). The ingestion policy
groups records by locality and emits a VCM page whenever either
bound is reached — whichever fires first:

- **`flush_threshold`** — the page is cut after this many records.
- **`flush_token_budget`** — the page is cut once the accumulated
  records would exceed this many tokens (estimated by the project's
  tokenizer over the JSON-serialised record values).

So a page holds at most `flush_threshold` records *and* at most
~`flush_token_budget` tokens. Read the defaults as: ~20 files per
page, capped at ~4096 tokens, whichever comes first.

Picking values:

- Lower both for **hot code that changes often or is queried at
  fine granularity**. Smaller pages mean an edit invalidates less
  surrounding content, and a query that touches one symbol faults
  in a smaller blob. Cost: more pages, more page-table overhead,
  more cache slots used for the same total bytes. A reasonable
  small-page profile is `flush_threshold: 8`, `flush_token_budget:
  2048`.
- Raise both for **large, stable, coherently-read corpora** (frozen
  third-party code, archived docs, generated reference). Bigger
  pages reduce per-fault overhead and let the LLM see more related
  context in one fetch. Cost: each fault drags in more bytes; an
  edit that does land invalidates a bigger page.
- Keep `flush_token_budget` comfortably under the LLM's per-call
  context budget for the worker that will read these pages.
  Doubling it without raising the LLM's context window won't make
  pages bigger — the budget is what *prevents* oversized pages, not
  what targets them.

`pinned: true` is the VCM equivalent of `mlock(2)` — pages produced
by that source are exempt from cache eviction. Use it for small,
high-traffic reference material the agent keeps faulting (a tools
registry, a manifest, a short coding-style guide). Avoid it for
anything large; pinned pages hold cache slots that nothing else can
reclaim, and a pinned literature corpus will drown out everything
else.

Rows that omit any of these three knobs inherit the deployment's
base `MmapConfig` for those fields.

## Knowledge routing rows

| Field | Required | Notes |
|---|---|---|
| `paths` | yes | List of gitignore-style globs, evaluated relative to the repo root. |
| `ingest_to` | no | `"knowledge_base"` (default) or `"vcm"`. See "How `ingest_to` works" below. |
| `profile` | no | Forwarded to `Ingestor.ingest_file(data_type_override=...)` so the chunks land with a meaningful `data_type` (e.g., `scientific_paper`). Ignored when `ingest_to: vcm`. |

`knowledge_routing` is **the user's seed** for the knowledge base —
it ingests files already committed to the monorepo. It does not
overlap with chat-driven acquisition (`BulkAcquisitionCapability`),
which adds *new external* sources at the agent's request.

### How `ingest_to` works

`ingest_to` is the single declarative field that names where a glob
of literature paths should land. There are two values:

- **`knowledge_base`** (default for new literature). The materialiser
  walks every matching file and feeds it through the process-singleton
  `Ingestor` — the same backend curation and retrieval share, so the
  ingested chunks are immediately available to the SessionAgent's
  retrieval surface in the next chat turn.

- **`vcm`**. The materialiser **skips** these rows on the KB side. The
  row exists in `knowledge_routing` only so that a single human-readable
  list of literature paths is the source of truth for routing
  decisions. The actual VCM mapping for those paths still comes from a
  `LiteratureContextPageSource` (or other) row under `sources:` —
  `knowledge_routing` and `sources:` answer different questions:
  *what does the KB ingest?* vs. *what does VCM map?*.

#### Why a row can stay in `knowledge_routing` after promotion

A `vcm` row records intent, not action. The dashboard's "Design
Monorepo" tab promotes a file from KB → VCM in two co-ordinated edits:

1. Flip the row's `ingest_to` from `knowledge_base` to `vcm`.
2. Add the path (or its enclosing directory) to a literature
   `sources:` row so VCM actually maps it.

Both edits are committed as one commit to the design monorepo, so a
reviewer reading the YAML alone can see the routing for every
literature path — there is no implicit-by-list-membership decision
hidden between the two lists.

A file can simultaneously be `ingest_to: knowledge_base` *and* live
under a literature `sources:` row — VCM gets the cache-aware chunks,
KB gets the retrievable chunks, and both share the same
`ProseChunker` boundaries via the
[knowledge deps singleton](knowledge-capabilities.md#process-singleton-deps).

## Default fallback (no map file)

```yaml
schema_version: 1
sources:
  - name: default
    type: git_repo
```

This is what `RepoMap.default_for_unmapped_repo()` returns. The
single source uses the caller-supplied `origin_url`, `branch`,
`commit`, and `scope_id` directly — i.e., the mapping is identical
to the historical single-`mmap_application_scope` behaviour, so
existing call sites do not change.

## End-to-end flow

```
CLI / dashboard
   └─► materialize_repo_map(vcm_handle, origin_url, branch, commit, base_scope_id, mmap_config)
         ├─► clone_or_retrieve_repository(origin_url, branch, commit)
         ├─► RepoMap.load(repo_root)        # parses .colony/repo_map.yaml or returns default
         ├─► for spec, scope_id in zip(repo_map.sources, scope_ids):
         │     await vcm_handle.mmap_application_scope(**spec.to_mmap_kwargs(...))
         ├─► materialize_knowledge_routing(repo_map, repo_root)
         │     # ingest matching files via the process-singleton Ingestor
         └─► returns one MmapResult per source
```

A failure on a single source is logged and skipped — the rest still
materialise — so a typo in one row does not block the whole map.

## Where it plugs in

- **CLI** (`polymath.py:run_integration_test`) — replaces a single
  `mmap_application_scope` call with `materialize_repo_map`.
- **Dashboard** (`/api/v1/vcm/map-repo`) — same.

Other callers of `mmap_application_scope` (e.g., custom CLI tools)
can adopt the materialiser by importing
`polymathera.colony.design_monorepo.materialize.materialize_repo_map`.

## Two ingestion paths into the KB

| Path | Trigger | Source |
|---|---|---|
| `knowledge_routing` rules in `repo_map.yaml` | Materialiser runs (CLI / dashboard `Map Repo`) | Files **already committed** to the design monorepo. Used to seed the KB with literature the user wants the SessionAgent to be able to retrieve from day one. |
| `BulkAcquisitionCapability.acquire_manifest` / `KnowledgeCuratorCapability.ingest_raw` | Agent action triggered from chat | **New external** sources — papers the SessionAgent decides to acquire while answering a question. |

Both write through the same process-singleton `Ingestor`, so chunks
from the two paths coexist in one KB.

## Promoting a file from KB → VCM

The "Design Monorepo" tab edits a single field — `ingest_to` — to
mark a `knowledge_routing` row as `vcm`, and at the same time adds
the path to a literature `sources:` row so the next materialisation
run produces VCM chunks instead of KB chunks. The KB chunks already
ingested under the previous `knowledge_base` route stay until they
are re-ingested or deleted; the dashboard surfaces a per-row
"clear KB chunks" action when the user wants the old chunks gone.
(PUT-and-commit is deferred — see the "Design Monorepo" tab
section below.)

## Git LFS

Design monorepos accumulate large binary blobs — literature PDFs,
CAD outputs, simulation runs, ML weights, scientific datasets — at
a pace that breaks plain git (GitHub rejects pushes of any file >
100 MB; cumulative repo size grows as N × revisions; the framework
clones each repo three times, so worst-case disk usage compounds).
The framework treats LFS as the default for design monorepos and
sets it up automatically.

### What `initialize_repo_map(enable_lfs=True)` does

1. Runs `git lfs install --local` on the working tree to wire the
   clean/smudge filters and the pre-push hook into the per-agent
   clone. Idempotent — re-running is a no-op. Best-effort: if
   `git-lfs` isn't installed (rare; the framework's container
   images ship it), the action logs a warning and continues so the
   bootstrap commit still happens.
2. Writes a default
   [`.gitattributes`](../../src/polymathera/colony/design_monorepo/templates/gitattributes.template)
   declaring LFS patterns for documents, archives, scientific data,
   images, CAD/3D, ML weights, and audio/video — **only when no
   `.gitattributes` exists** (operator edits are never overwritten).
3. Sets `manifest.lfs.mode = "same_remote"` for new manifests, or
   flips the mode from `"disabled"` to `"same_remote"` on
   pre-existing manifests so other clones of this repo activate LFS
   too. The mode is the single source of truth other consumers
   (dashboard cache, sibling agents) read.

Patterns are **format-based, not path-based**. The framework does
not prescribe a directory layout, so `*.pdf` matches PDFs anywhere
in the tree rather than `literature/**/*.pdf`. Edit `.gitattributes`
freely to add domain-specific formats or to scope a pattern to a
subtree.

Per-agent clones go through `_lazy_clone_from_agent_metadata`,
which also runs `git lfs install --local` after each clone so
later commits from those agents route through LFS instead of plain
git.

### Forward-only by default; `migrate_existing_to_lfs=True` rewrites history

LFS is forward-only. Adding a pattern to `.gitattributes` only
affects commits made *after* the pattern is in place. PDFs that
were committed earlier sit as plain git objects in history — the
repo size doesn't shrink retroactively.

To convert already-committed blobs into LFS pointers, run
`initialize_repo_map(migrate_existing_to_lfs=True)`. The action
sequences three steps:

1. Make the bootstrap commit (manifest + repo_map.yaml +
   `.gitattributes`).
2. Run `git lfs migrate import --include=<patterns> --everything`
   where `<patterns>` is parsed from the current `.gitattributes`.
   **This rewrites every commit SHA on the migrated refs.**
3. Push with `--force-with-lease` so the rewritten history reaches
   the upstream. Force is necessary because the rewritten SHAs
   share no commits with the remote's existing branch — a regular
   push fails as non-fast-forward, and `git pull` reports "refusing
   to merge unrelated histories" (there's no common ancestor to
   merge against). `--force-with-lease` is safer than `--force`
   because it checks the remote is in the state we expect and
   refuses if someone else pushed concurrently.

Anyone else who already cloned the repo will need to **re-clone**
afterwards — `git pull --rebase` won't help because pre- and
post-migration histories share no commits. Default is `False`;
opt in only on a fresh repo or when you're sure no one else has a
working clone.

If you've run the migration manually outside this action and hit
the `non-fast-forward` rejection, the fix is `git push
--force-with-lease origin <branch>`. Running this action with
`migrate_existing_to_lfs=True` again would also work — it's
idempotent on the file content and re-publishes the rewritten
history.

### GitHub LFS quota

LFS storage and bandwidth live on the upstream remote, not in the
framework. For self-hosted Gitea / GitLab there's no quota. For
github.com the free tier is 1 GB storage + 1 GB bandwidth per
month, then $5 per 50 GB data pack. A single literature monorepo
will fit comfortably in the free tier; a CAD-heavy design repo
that pushes 50 GB/month of revision data won't. The operator pays
for LFS storage on github.com — flagged here so it's not a
surprise on the first invoice.

## Dashboard tab — "Design Monorepo"

The **Design Monorepo** tab is now both the inspector AND the entry
point for mapping the repo into VCM. It is available once a session
is active; it inspects (and maps from) the design monorepo associated
with the active session's colony.

Configuring the colony's design-monorepo URL is a separate gesture
that lives on the **landing page** (`Colonies` panel → pencil →
paste URL → Save). See
[Design-Monorepo Capabilities — One-time setup](design-monorepo-capabilities.md#one-time-setup-point-the-colony-at-your-repo).
The persistence endpoints used by the landing page also stay
visible here for completeness.

The tab and the landing page together call six endpoints:

| Endpoint | Caller | Purpose |
|---|---|---|
| `GET /api/v1/colonies/{colony_id}/design-monorepo` | landing page | Current persisted URL/branch/commit. |
| `PUT /api/v1/colonies/{colony_id}/design-monorepo` | landing page | Save URL/branch/commit on the colony's row. |
| `GET /api/v1/repo-map` | tab | Parsed `repo_map.yaml` (or default fallback) + raw YAML text. |
| `GET /api/v1/repo-map/tree` | tab | Bounded directory tree (skips `.git/`, capped by `max_depth` + `max_nodes`). |
| `POST /api/v1/repo-map/preview` | tab | Dry-run the materialiser — returns the exact `mmap_application_scope` kwargs the cluster would issue, without executing them. |
| `POST /api/v1/vcm/map` | tab | Trigger the actual mapping in the background. The request body carries `enabled_sources` — a subset of `sources:` rows the user ticked. Paging knobs come from `repo_map.yaml`, not the request. |

In-tab workflow (after a session is active):

1. The form pre-populates from the active colony's persisted URL.
2. Click **Load** to clone (idempotent — same `GitFileStorage` path
   the page sources use) and render:
   - left: the file tree (collapsible);
   - right top: per-source checkboxes (every `sources:` row from the
     parsed `repo_map.yaml`, ticked by default) plus the raw YAML for
     reference. Each checkbox shows the row's type, `start_dir`, and
     any per-source paging knobs (`flush_threshold`, `flush_token_budget`,
     `pinned`, or `chunk_target_tokens` for literature).
   - right bottom: the dry-run preview after **Preview mmap calls**.
3. Untick any source the user does NOT want mapped right now.
4. Click **Preview mmap calls** to see the per-source `scope_id` +
   constructor kwargs the materialiser would feed to VCM.
5. Click **Map to VCM** to start the mapping. A confirmation modal
   summarises the URL, branch, and the selected sources, and points
   the user at the **VCM** tab to track progress and inspect the
   resulting page catalog. The actual mapping runs in a background
   task on the cluster (`POST /vcm/map` returns immediately with a
   `MappingOpStatus.op_id`).

`enabled_sources` semantics: when the user has every row ticked, the
request omits the field entirely (matches the materialiser's "`None`
⇒ map every row" contract). Otherwise the request lists only the
ticked names — rows whose `name` is absent are skipped.

All endpoints are `Ring.USER` and gated by `require_auth`. Cloning
and tree walking happen in the dashboard process (FastAPI worker)
against the colony-shared `/mnt/shared` volume, so a re-render is
fast.

> The "Map Content" dialog that used to live on the **VCM** tab has
> been removed. That dialog asked the user to retype a URL/branch
> with no link to the colony's persisted design-monorepo URL, and
> exposed the paging knobs as deployment-wide globals. Both gestures
> now live in this tab and read from `repo_map.yaml`.
