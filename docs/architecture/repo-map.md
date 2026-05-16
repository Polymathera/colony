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
schema_version: 2

# VCM mapping — declares what the framework pages into the Virtual
# Context Manager. One row per source; each becomes an
# ``mmap_application_scope`` call.
vcm_sources:
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

  # 2) Literature directory — PDFs go through the chunker. Note:
  #    being in vcm_sources only paginates these into VCM; KB
  #    ingestion is a separate concern handled by knowledge_sources
  #    below.
  - name: literature-paged
    type: literature
    start_dir: literature/promoted/
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

# KB ingestion — declares what the framework feeds through the
# process-singleton Ingestor into the knowledge base. Independent of
# vcm_sources: presence here means "ingest these files into the KB";
# absence means "don't." A single path may appear in both lists; VCM
# mapping and KB ingestion are orthogonal.
#
# Each named row is one of two shapes:
#   - LOCAL  (paths set, acquirer unset): glob bundle on disk
#   - REMOTE (acquirer + destination set, paths unset): fetcher
#       writes a file into destination/, materialiser ingests it
knowledge_sources:
  # LOCAL bundle — files already committed under literature/curated/.
  - name: curated-papers
    paths: ["literature/curated/**/*.pdf"]
    profile: scientific_paper
    tier: research_paper

  # LOCAL bundle — standards docs.
  - name: standards
    paths: ["standards/**/*.pdf"]
    profile: standard_clause
    tier: standard

  # REMOTE bundle — arXiv fetch. ``destination`` is the repo-root-
  # relative directory the acquirer writes the fetched PDF into; the
  # written file is committed alongside the sidecar so a re-run
  # avoids the download.
  - name: foundational-paper
    acquirer:
      method: arxiv_id
      args: {arxiv_id: "2407.12345"}
    destination: literature/acquired/
    profile: scientific_paper
    tier: research_paper
```

`schema_version` is required and currently `2`. Unknown values fail
loud at load time. Extra fields on any row are rejected
(`extra: forbid`) so a typo is reported instead of silently ignored.

## VCM source rows (`vcm_sources:`)

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

## Knowledge source rows (`knowledge_sources:`)

Each row is a **named bundle** of either local files (`paths`) or a remote acquirer (`acquirer` + `destination`). One row → one ingestion target the operator can enable / disable from the dashboard's checkbox list.

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Human-readable label. Surfaces as a checkbox in the Design Monorepo tab; also the filter key for `materialize_knowledge_sources(enabled_sources=[…])`. |
| `paths` | one of `paths` / `acquirer` | List of gitignore-style globs, evaluated relative to the repo root. Mutually exclusive with `acquirer`. |
| `acquirer` | one of `paths` / `acquirer` | Remote-source fetcher. Mutually exclusive with `paths`. Shape: `{method: <strategy_key>, args: {…}}`. Strategy keys come from the `AcquirerStrategy` registry (e.g. `arxiv_id`, `doi`, `http_url`). |
| `destination` | required when `acquirer` is set | Repo-root-relative directory the acquirer writes its fetched file into. Forbidden when `acquirer` is unset. |
| `profile` | no | `data_type` label propagated to ingested chunks (e.g., `scientific_paper`, `standard_clause`, `component_datasheet`). Forwarded to `Ingestor.ingest_file(data_type_override=...)`. |
| `tier` | no | Corpus tier label (`research_paper`, `standard`, `datasheet`, `patent`, `textbook`, `untiered`). Used by tiered retrieval profiles. |

`knowledge_sources` is **the user's seed** for the knowledge base — it ingests files already in the monorepo (LOCAL) or pulls new files in (REMOTE). It is independent of `vcm_sources`: a file can appear in both lists (VCM gets cache-aware chunks for runtime planning, KB gets retrievable chunks for retrieval) or neither.

### Triggering ingestion from chat

`RepoStateProvider.ingest_repo_map_literature` is the chat-callable wrapper. The `SessionAgent` picks it when the user says "ingest literature from `repo_map.yaml`" / "process the design monorepo". The action:

1. Refreshes the per-agent clone against `origin` (so operator edits pushed from the host land before ingestion runs).
2. Reads the **enabled subset** of `knowledge_sources:` rows — operators tick rows via the Design Monorepo tab's checkbox list, persisted server-side per colony.
3. For each enabled LOCAL row: walks `paths` globs, ingests every match through the process-singleton `Ingestor`.
4. For each enabled REMOTE row: runs the acquirer (writes the fetched file into `destination/`, commits the file + an `.ingested/<stem>/` sidecar so re-runs avoid the download), then ingests the written file.

Returns `{"ingested": [<source_uri>, …], "count": N, "errors": [...]}`.

`BulkAcquisitionCapability` — the separate "fetch this URL into the KB" surface that used to take a standalone `CorpusManifest` YAML — has been **folded into this schema** (commit `7e54ebb1`). The same `knowledge_sources:` list carries both LOCAL bundles AND REMOTE acquirer bundles; one source of truth, one materialiser, one dashboard checkbox list. No separate manifest file.

## Separation: VCM mapping vs. KB ingestion

The schema split is intentional. `vcm_sources` and `knowledge_sources` answer different questions:

| | `vcm_sources` | `knowledge_sources` |
|---|---|---|
| **Question** | What does VCM page into its cache for runtime planning? | What does the Ingestor add to the searchable knowledge base? |
| **Consumer** | `mmap_application_scope` → VCM page graph → agent's working set | `Ingestor` → vector store → `KnowledgeRetrievalCapability` |
| **Cost shape** | Page faults, KV cache slots | Embedding tokens, vector store size |
| **Triggered by** | `materialize_repo_map` (CLI / dashboard "Map Repo") | `ingest_repo_map_literature` (chat-driven, per the enabled checkbox list) |

A file can appear in either list, both, or neither — the framework does not infer one from the other. Putting a literature directory in `vcm_sources` paginates it for runtime planning; adding the same paths under `knowledge_sources` ingests them for retrieval; doing both gets you both at the same `ProseChunker` boundaries (the [knowledge deps singleton](knowledge-capabilities.md#process-singleton-deps) shares them).

## Default fallback (no map file)

```yaml
schema_version: 2
vcm_sources:
  - name: default
    type: git_repo
knowledge_sources: []
```

This is what `RepoMap.default_for_unmapped_repo()` returns. The single `vcm_sources` row uses the caller-supplied `origin_url`, `branch`, `commit`, and `scope_id` directly — i.e., the mapping is identical to the historical single-`mmap_application_scope` behaviour, so existing call sites do not change. `knowledge_sources` defaults to empty so existing repos without an explicit map do not silently start ingesting.

## End-to-end flow

```
CLI / dashboard
   └─► materialize_repo_map(vcm_handle, origin_url, branch, commit, base_scope_id, mmap_config)
         ├─► clone_or_retrieve_repository(origin_url, branch, commit)
         ├─► RepoMap.load(repo_root)        # parses .colony/repo_map.yaml or returns default
         ├─► for spec, scope_id in zip(repo_map.vcm_sources, scope_ids):
         │     await vcm_handle.mmap_application_scope(**spec.to_mmap_kwargs(...))
         └─► returns one MmapResult per source

chat (SessionAgent calls RepoStateProvider.ingest_repo_map_literature)
   └─► materialize_knowledge_sources(repo_map, repo_root, enabled_sources=[…])
         ├─► for each enabled knowledge_sources row:
         │     - LOCAL  → walk paths globs, ingest each match through Ingestor
         │     - REMOTE → run acquirer (writes to destination/), ingest the file
         └─► returns {"ingested": [...], "count": N, "errors": [...]}
```

A failure on a single source is logged and skipped — the rest still materialise / ingest — so a typo in one row does not block the whole map.

## Where it plugs in

- **CLI** (`polymath.py:run_integration_test`) — replaces a single `mmap_application_scope` call with `materialize_repo_map`.
- **Dashboard** (`/api/v1/vcm/map-repo`) — same.
- **Chat-driven ingestion** (`SessionAgent` → `RepoStateProvider.ingest_repo_map_literature`) — calls `materialize_knowledge_sources` with the user-toggled enabled-sources subset.

Other callers of `mmap_application_scope` (e.g., custom CLI tools) can adopt the materialiser by importing `polymathera.colony.design_monorepo.materialize.materialize_repo_map`.

## Two ingestion paths into the KB

Both write through the same process-singleton `Ingestor`, so chunks from either path coexist in one KB:

| Path | Trigger | Source |
|---|---|---|
| `knowledge_sources:` LOCAL row | `ingest_repo_map_literature` (chat) or `materialize_knowledge_sources` (programmatic) | Files **already committed** to the design monorepo. Used to seed the KB with literature the user wants the `SessionAgent` to be able to retrieve from day one. |
| `knowledge_sources:` REMOTE row (`acquirer` + `destination`) | same | **New external** sources — papers / standards / datasheets pulled in via an `AcquirerStrategy` (`arxiv_id`, `doi`, `http_url`, …). The fetched file is written into `destination/`, committed alongside the `.ingested/<stem>/` sidecar so re-runs avoid the download. |

What used to be a separate `BulkAcquisitionCapability.acquire_manifest(<CorpusManifest>)` surface is now folded into the same `knowledge_sources:` list (commit `7e54ebb1`) — one schema, one materialiser, one dashboard checkbox list.

## Promoting a file from KB-only to KB + VCM

To take a file the KB already retrieves and also paginate it into VCM for runtime planning: keep its existing `knowledge_sources:` row AND add a `vcm_sources:` row whose `start_dir` / `paths` covers the file. The two rows are independent — neither edit removes the other's effect. The next `materialize_repo_map` run paginates the file into VCM; the next `ingest_repo_map_literature` run leaves the KB chunks untouched. A `data_type` mismatch between the two paths cannot happen because each list carries its own `profile` field.

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
| `POST /api/v1/vcm/map` | tab | Trigger the actual mapping in the background. The request body carries `enabled_sources` — a subset of `vcm_sources:` rows the user ticked. Paging knobs come from `repo_map.yaml`, not the request. |

In-tab workflow (after a session is active):

1. The form pre-populates from the active colony's persisted URL.
2. Click **Load** to clone (idempotent — same `GitFileStorage` path
   the page sources use) and render:
   - left: the file tree (collapsible);
   - right top: per-source checkboxes (every `vcm_sources:` row from the
     parsed `repo_map.yaml`, ticked by default) plus the raw YAML for
     reference. Each checkbox shows the row's type, `start_dir`, and
     any per-source paging knobs (`flush_threshold`, `flush_token_budget`,
     `pinned`, or `chunk_target_tokens` for literature). A separate
     `knowledge_sources:` checkbox list (LOCAL + REMOTE rows) governs
     chat-driven KB ingestion via `ingest_repo_map_literature`.
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
