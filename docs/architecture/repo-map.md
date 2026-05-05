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
  # 1) Code subtree, with build artifacts excluded.
  - name: design-code
    type: git_repo
    start_dir: tools/
    exclude_globs: ["**/build/**", "**/__pycache__/**"]
    binary_policy: skip
    static: false

  # 2) Literature directory — PDFs go through the chunker.
  - name: literature
    type: literature
    start_dir: literature/
    chunk_target_tokens: 800

  # 3) Frozen external dependency, declared as a submodule.
  - name: external-foo
    type: git_repo
    submodule: third_party/foo
    start_dir: src/
    static: true

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

## Dashboard tab — "Design Monorepo"

The dashboard ships a read-only inspector at the **Design Monorepo**
tab. The tab is available only once a session is active; it
inspects the design monorepo associated with the active session's
colony.

Configuring the colony's design-monorepo URL is a separate gesture
that lives on the **landing page** (`Colonies` panel → pencil →
paste URL → Save). See
[Design-Monorepo Capabilities — One-time setup](design-monorepo-capabilities.md#one-time-setup-point-the-colony-at-your-repo).
The persistence endpoints used by the landing page also stay
visible here for completeness.

The tab and the landing page together call five endpoints:

| Endpoint | Caller | Purpose |
|---|---|---|
| `GET /api/v1/colonies/{colony_id}/design-monorepo` | landing page | Current persisted URL/branch/commit. |
| `PUT /api/v1/colonies/{colony_id}/design-monorepo` | landing page | Save URL/branch/commit on the colony's row. |
| `GET /api/v1/repo-map` | tab | Parsed `repo_map.yaml` (or default fallback) + raw YAML text. |
| `GET /api/v1/repo-map/tree` | tab | Bounded directory tree (skips `.git/`, capped by `max_depth` + `max_nodes`). |
| `POST /api/v1/repo-map/preview` | tab | Dry-run the materialiser — returns the exact `mmap_application_scope` kwargs the cluster would issue, without executing them. |

In-tab workflow (after a session is active):

1. The form pre-populates from the active colony's persisted URL.
2. Click **Load** to clone (idempotent — same `GitFileStorage` path
   the page sources use) and render:
   - left: the file tree (collapsible);
   - right: the contents of `.colony/repo_map.yaml` in a `<pre>`
     viewer, or the rendered default-fallback summary if the file
     is absent.
3. Click **Preview mmap calls** to see the per-source `scope_id` +
   constructor kwargs the materialiser would feed to VCM. Useful as
   a sanity check before clicking **Map** in the VCM tab.

All endpoints are `Ring.USER` and gated by `require_auth`. Cloning
and tree walking happen in the dashboard process (FastAPI worker)
against the colony-shared `/mnt/shared` volume, so a re-render is
fast.
