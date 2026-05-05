# Design-Monorepo + Knowledge Integration Plan

Author: claude (proposal — not yet implemented)
Status: **For review.** Do not start coding until each section's checkboxes are approved.

---

## 0. Scope

Wire `_DesignMonorepoCapabilityBase` subclasses, knowledge retrieval, multi-part repo mapping, and a "Design Monorepo" UI tab into the rest of the running system. **Re-use existing classes; do not duplicate.** Every section below states what already exists, what changes, and what is genuinely new.

---

## 1. Inventory & inconsistencies found

| Existing | Location | Status |
|---|---|---|
| `ContextPageSource` ABC | `vcm/sources/context_page_source.py:38` | Just refactored, `static` is now an instance arg |
| `GitRepoContextPageSource` | `samples/paging/git_repo_page_source.py:33` | Live; no subdir/include/exclude; binaries silently included as `unknown` mime |
| `BlackboardContextPageSource` | `agents/blackboard/paging/blackboard_page_source.py:754` | Live; uses its own event stream, no watchers |
| `LocalFsWatcher`, `GitRemoteWatcher`, `CompositeWatcher` | `vcm/watchers/` | Live; `GitRepoContextPageSource` composes both |
| `_DesignMonorepoCapabilityBase` + `RepoStateProvider`, `DesignCheckpointer`, `ToolBuilder` | `design_monorepo/capabilities.py:88` | **NOT wired into SessionAgent** |
| `KnowledgeCuratorCapability` | `agents/roles/knowledge_curator.py:113` | Curator-only — no general retrieval surface |
| `BulkAcquisitionCapability` | `knowledge/bulk_acquisition.py:409` | Live; `acquire_manifest` action — drives `Ingestor` over a corpus manifest |
| Retrieval adapters | `knowledge/retrieval/` | Registered as `ToolAdapter`s, not actions |
| PDF readers (`pypdf`, `grobid_pdf` stub) | `knowledge/readers/` | Used by `Ingestor`, NOT by VCM |
| `SessionAgent` blueprint | `web_ui/backend/routers/sessions.py:367` | 9 capabilities; no design-monorepo / knowledge |
| Dashboard tabs | `web_ui/frontend/src/components/layout/AppShell.tsx:23` | 9 tabs; no "Design Monorepo" |
| `.colony/`, `.colonyignore`, repo-map file | nowhere | Does not exist |
| Per-agent local clone | nowhere | One shared clone per `working_dir` |
| `CPSCoordinator` | nowhere | Lives in the **separate CPS repo** that consumes Colony as a library — out of scope for this PR; we expose hooks |

**Inconsistencies to call out before I touch anything:**

1. **`GitRepoContextPageSource` carries a `LocalFsWatcher`** even though every running mapping today is the global main-branch read-only view. Per the design intent (<mark>VCM = global read-only view of `main`; per-agent edits live in private clones</mark>), `LocalFsWatcher` is dead code in production. Keep the class — the watcher abstraction is fine — but make the source default to **`watch_local=False`** and only opt in for tests / unusual single-writer setups. (See §3.)

2. **`_DesignMonorepoCapabilityBase` already comments**: *"Live page-change events for the working tree flow through `GitRepoContextPageSource.watch()` once the working tree is mapped into the VCM"*. That comment is half-right today and fully right after this plan: <mark>the **VCM mapping** (read-only main) gets `GitRemoteWatcher` events; **per-agent clones** are private working trees that don't feed VCM.</mark>

3. **`mimetypes` classifier in `file_grouping.py:462`** silently labels binaries as `'unknown'` and lets them through. Two callers want opposite behaviour: `GitRepoContextPageSource` wants binaries skipped (text only); a new `LiteratureContextPageSource` wants binaries kept and routed to a PDF extractor. <mark>Fix: pass an explicit `binary_policy` into the sharding strategy; don't change the classifier.</mark>

4. **Acquisition + curation already exist as agent capabilities (`BulkAcquisitionCapability`, `KnowledgeCuratorCapability`); retrieval is missing.** The retrieval adapters are tool-adapters (Phase C2 plumbing). For agents to *use* knowledge they need a normal `@action_executor` surface. Adding `KnowledgeRetrievalCapability` is genuinely new, but its body is a thin wrapper over the existing adapters — no new retrieval logic. The full agent-facing trio (acquisition / curation / retrieval) is what the `SessionAgent` invokes from chat for *new external* sources — the CLI gets nothing knowledge-related (cluster ops only). Files **already committed** to the design monorepo are seeded into the KB declaratively via `knowledge_routing` in `repo_map.yaml`; that path is the materialiser's responsibility, not the SessionAgent's. (See §5 + §7.1.)

5. **No `.gitmodules`/`.colony` config story.** Today `mmap_application_scope` takes a single `origin_url`. <mark>We need an explicit, version-controllable manifest *if* we want one repo-map → many sources. Proposing **a single `.colony/repo_map.yaml`** in the design monorepo root, parsed by a thin loader. Submodules are an optional source kind, not the primary mechanism.</mark>

---

## 2. Design overview

```
                 ┌───────────────────────────────────────────────────┐
                 │  design monorepo (git, single source of truth)    │
                 │   .colony/repo_map.yaml  ◄──── version-controlled │
                 │   tools/        code, tests                       │
                 │   literature/   PDFs (route: vcm | kb)            │
                 │   third_party/  submodules                        │
                 └───────────────────────────────────────────────────┘
                          │ clone-once (read-only) per node
                          ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ VCM (global, read-only view of `main`)                        │
   │   GitRepoContextPageSource     scope=tools                    │
   │   GitRepoContextPageSource     scope=third_party/foo (frozen) │
   │   LiteratureContextPageSource  scope=literature               │
   │   BlackboardContextPageSource  scope=session/agent (live)     │
   │   GitRemoteWatcher only — no LocalFsWatcher                   │
   └───────────────────────────────────────────────────────────────┘
                          ▲                          │
                          │ branch-update events     │ pages (cache-aware infer)
                          │                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Agents                                                       │
   │   SessionAgent / coordinators / workers                      │
   │   ├ VCMCapability                  (existing)                │
   │   ├ DesignMonorepoCapability       (NEW — composite of base+ │
   │   │                                  RepoStateProvider +     │
   │   │                                  DesignCheckpointer +    │
   │   │                                  ToolBuilder)            │
   │   ├ KnowledgeRetrievalCapability   (NEW)                     │
   │   ├ … other existing capabilities                            │
   │   each agent: own private working tree at                    │
   │   ~/.colony/agents/<agent_id>/clones/<scope>/                │
   └──────────────────────────────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ Dashboard "Design Monorepo" tab                              │
   │   tree view + per-path source assignment + repo_map editor   │
   └──────────────────────────────────────────────────────────────┘
```

---

## 3. `GitRepoContextPageSource`: subdir mapping + binary policy + watcher policy

**File touched:** `samples/paging/git_repo_page_source.py`, `samples/paging/sharding/strategy.py`, `samples/paging/sharding/file_grouping.py`.

### 3.1 New constructor kwargs

```python
class GitRepoContextPageSource(ContextPageSource):
    def __init__(
        self,
        *,
        scope_id: str,
        mmap_config: MmapConfig,
        origin_url: str,
        branch: str = "main",
        commit: str = "HEAD",
        # --- NEW ---
        start_dir: str | None = None,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        ignore_files: tuple[str, ...] = (".gitignore", ".colonyignore"),
        binary_policy: Literal["skip", "include"] = "skip",
        watch_remote: bool = True,
        watch_local: bool = False,
        static: bool = False,
    ):
        ...
```

- `start_dir` is **repo-relative**. None = repo root (current behaviour).
- `include_globs` / `exclude_globs` use gitignore semantics (not python `fnmatch`); reuse the existing `pathspec` library if it's already a dep, else add `pathspec` (it's small, single-purpose, MIT). **Action: confirm `pathspec` is in `pyproject.toml`; add to `code_analysis` extra if not.**
- `ignore_files` are *filenames within the repo* whose patterns are merged into the exclude set. Default reads `.gitignore` and `.colonyignore`. Operator can pass `()` to disable.
- `binary_policy="skip"` is the new default for code mappings — fixes inconsistency #3. The literature source flips this to `"include"`.
- `watch_local=False` is the new default — fixes inconsistency #1. Existing call sites (`cli/polymath.py:1398`, `web_ui/backend/routers/vcm.py:313/405`) keep working unchanged because they don't set this kwarg.

### 3.2 Sharding-strategy change

`GitRepoShardingStrategy.create_shards_with_graph()` currently calls `repo.head.commit.tree.traverse()` and indiscriminately includes every blob. Add a `_walk()` helper that:

```python
def _walk(self, repo, *, start_dir, include_pathspec, exclude_pathspec, binary_policy):
    root_tree = repo.head.commit.tree
    if start_dir:
        root_tree = root_tree / start_dir
    for blob in root_tree.traverse():
        if blob.type != "blob":
            continue
        rel = Path(blob.path).relative_to(start_dir or "")
        if exclude_pathspec.match_file(str(rel)):
            continue
        if include_pathspec is not None and not include_pathspec.match_file(str(rel)):
            continue
        if binary_policy == "skip" and _is_binary_blob(blob):
            continue
        yield blob, rel
```

`_is_binary_blob` reuses `mimetypes.guess_type` (returns `text/*`/`application/json`/`application/xml` ⇒ text). For ambiguous extensions, peek first 8KB and reject if `b"\x00"` present.

### 3.3 Watcher composition

Today `_setup_composite_watcher()` always builds `LocalFsWatcher + GitRemoteWatcher`. Change to:

```python
watchers = []
if self._watch_local:
    watchers.append(LocalFsWatcher(...))
if self._watch_remote:
    watchers.append(GitRemoteWatcher(...))
self._composite_watcher = CompositeWatcher(tuple(watchers), scope_id=...)
```

If both are off, `watch()` yields nothing (the existing `if self._repo_path is None: return` early-exit covers the empty case after we set `_composite_watcher = None`).

### 3.4 Tests

Extend `vcm/convergence/tests/test_chain_smoke.py` with:
- `start_dir` + `exclude_globs` → only listed files become pages.
- `binary_policy="skip"` → PDFs in tree are absent from `file_to_page`.
- `watch_local=False`, `watch_remote=True` → no `LocalFsWatcher` events propagate even when the working tree is touched.

---

## 4. New `LiteratureContextPageSource`

**File:** `samples/paging/literature_page_source.py` (new).

### 4.1 Why a new class

`GitRepoContextPageSource` walks git blobs as text. Literature wants:
- per-PDF chunked text (extracted via existing `knowledge/readers/pdf.py` — **do not duplicate**),
- one VCM page per chunk so cache-aware inference operates at chunk granularity,
- metadata (title, page numbers, DOI when available) stored on the page,
- the same `GitRemoteWatcher` story for live updates (PDFs land via git commits to the design monorepo).

### 4.2 Sketch

```python
@register_new_source_type(BuilInContextPageSourceType.LITERATURE)
class LiteratureContextPageSource(ContextPageSource):
    """Pages literature artifacts (PDFs, plain-text papers) into the VCM."""

    def __init__(
        self,
        *,
        scope_id: str,
        mmap_config: MmapConfig,
        origin_url: str,
        branch: str = "main",
        commit: str = "HEAD",
        start_dir: str | None = "literature",
        include_globs: list[str] | None = ("**/*.pdf", "**/*.txt", "**/*.md"),
        exclude_globs: list[str] | None = None,
        chunk_tokens: int = 800,
        chunk_overlap_tokens: int = 80,
        extractor: Literal["pypdf", "grobid"] = "pypdf",
        watch_remote: bool = True,
        static: bool = False,
    ): ...

    async def initialize(self) -> None:
        # 1. clone-or-retrieve via GitFileStorage (same path GitRepoContextPageSource uses)
        # 2. walk start_dir with the same _walk() helper from §3.2 (binary_policy="include")
        # 3. for each PDF: call knowledge/readers/pdf.PdfReader.extract_chunks(path, chunk_tokens, overlap)
        # 4. for each chunk: VirtualContextPage with text=chunk.text, metadata={"file": rel, "page_range": ..., "title": ..., "extractor": ...}
        # 5. persist via the same PageStorage instance (file_to_page, page_to_file maps point to chunk_ids)
```

### 4.3 Re-use checklist

- Chunking + extraction: **reuse** `knowledge/readers/pdf.PdfReader` (and `grobid_pdf.GrobidPdfReader` when configured). Add a `chunk(text, n_tokens, overlap)` helper to `knowledge/readers/__init__.py` if it doesn't already exist.
- Repo cloning: **reuse** `polymathera.get_storage().git_storage.clone_or_retrieve_repository()` exactly as `GitRepoContextPageSource` does.
- Watcher: **reuse** `GitRemoteWatcher`; on each PDF mutation, re-extract that file's chunks and emit `PageReplaced` for old chunk ids + `PageAdded` for new ones.
- File walker: **reuse** `_walk()` from §3.2 (extract it to `samples/paging/_walk.py` so both classes call into it).

### 4.4 Tests

Add a tiny PDF fixture (1 page, 1 paragraph), assert `file_to_page` keys are chunk ids and the page text contains the paragraph.

---

## 5. `.colony/repo_map.yaml` — single source of truth

**File:** `design_monorepo/repo_map.py` (new); schema in `design_monorepo/repo_map_schema.py`.

### 5.1 Schema (Pydantic)

```yaml
# .colony/repo_map.yaml
schema_version: 1

# Each entry becomes one VCM mapping (one ContextPageSource).
sources:
  - name: design-code
    type: git_repo               # source_type registered with ContextPageSourceFactory
    start_dir: tools/
    exclude_globs: ["**/*.pdf", "**/build/**", "**/__pycache__/**"]
    binary_policy: skip
    static: false                # follows main

  - name: literature-vcm
    type: literature
    start_dir: literature/in_context/
    chunk_tokens: 800
    static: false

  - name: external-foo
    type: git_repo
    submodule: third_party/foo   # path of a submodule under .gitmodules
    start_dir: src/
    static: true                 # frozen at submodule's pinned commit

# Each row carries an explicit ``ingest_to`` field naming the
# destination store. ``knowledge_base`` (default) ingests via the
# process-singleton Ingestor; ``vcm`` is documentation-only —
# the row records that the path was promoted to VCM and the
# materialiser must skip KB ingestion for it.
knowledge_routing:
  - paths: ["literature/curated/**/*.pdf"]
    ingest_to: knowledge_base    # default; explicit for clarity
    profile: scientific_paper

- paths: ["standards/**/*.pdf"]
    # ingest_to omitted ⇒ knowledge_base
  - paths: ["literature/notes/**/*.md"]
    ingest_to: vcm               # promoted — KB materialiser skips it
```

`knowledge_routing` is the **user-seeded** path into the KB — literature already committed to the monorepo. Chat-driven acquisition of *new external* sources is a separate path through `BulkAcquisitionCapability` / `KnowledgeCuratorCapability` (see §7.1).
The dashboard's "Design Monorepo" tab is the UI for promoting a file from KB → VCM by flipping `ingest_to` on the row + adding a literature source row in one commit.

### 5.2 Loader interface

```python
class RepoMap(BaseModel):
    schema_version: int
    sources: list[SourceSpec]
    knowledge_routing: list[KnowledgeRoute] = []

    @classmethod
    async def load(cls, repo_root: Path) -> "RepoMap":
        path = repo_root / ".colony" / "repo_map.yaml"
        if not path.exists():
            return cls.default_for_unmapped_repo()
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def to_mmap_calls(self, *, colony_id: str, origin_url: str) -> list[MmapCall]:
        """One MmapCall per source — the VCM materialiser issues each."""
```

### 5.3 Where it plugs in

The CLI integration test and the dashboard's `New Session` flow currently issue **one** `mmap_application_scope` call. They become:

```python
repo_map = await RepoMap.load(local_clone_path)
for spec in repo_map.sources:
    await vcm_handle.mmap_application_scope(**spec.to_mmap_kwargs(origin_url, colony_id))
await materialize_knowledge_routing(repo_map=repo_map, repo_root=local_clone_path)
```

(`materialize_knowledge_routing` walks `knowledge_routing` and feeds each matching file through the process-singleton `Ingestor`. Chat-driven ingestion of **new external** sources stays a separate path; see §7.)

`SourceSpec.to_mmap_kwargs` translates the YAML row into the keyword args of the matching `ContextPageSource` constructor. Submodules are resolved by reading the local clone's `.gitmodules` and substituting `origin_url`/`commit` from the submodule entry.

### 5.4 Default for repos without `repo_map.yaml`

`RepoMap.default_for_unmapped_repo()` returns a single `git_repo` source over the whole tree with `binary_policy="skip"`, preserving today's behaviour for existing users.

### 5.5 Test plan

- Round-trip: write a `repo_map.yaml`, load, materialise into a list of mmap calls, assert kwargs match per source row.
- Submodule resolution: stub `.gitmodules`, assert nested origin/commit propagation.
- Default fallback: missing file → exactly one source spec.

---

## 6. `DesignMonorepoCapability` family wiring + per-agent local clones

### 6.1 Composite capability (NEW thin wrapper, not duplicated logic)

`_DesignMonorepoCapabilityBase`, `RepoStateProvider`, `DesignCheckpointer`, `ToolBuilder` already exist and work. We add a single composite blueprint helper so the `SessionAgent` and CPS coordinators don't have to bind three things:

```python
# design_monorepo/blueprints.py (new, ~30 lines)
def design_monorepo_capability_blueprints(
    *,
    working_dir_root: str = "~/.colony/agents",  # base for per-agent clones
    auto_checkpoint_on_quiescence: bool = True,
) -> list[AgentCapabilityBlueprint]:
    return [
        RepoStateProvider.bind(working_dir=_per_agent_clone_dir(working_dir_root)),
        DesignCheckpointer.bind(
            working_dir=_per_agent_clone_dir(working_dir_root),
            auto_checkpoint_on_quiescence=auto_checkpoint_on_quiescence,
        ),
        ToolBuilder.bind(working_dir=_per_agent_clone_dir(working_dir_root)),
    ]
```

`_per_agent_clone_dir` returns a *callable* (or a sentinel) that the capability's async `initialize()` resolves to `Path(working_dir_root) / agent.agent_id / scope_id`. The clone itself is created lazily on first action via `DesignMonorepoClient.clone_or_open()`. This is the surgical change: no class restructuring; only the working-dir resolution becomes per-agent.

**Inconsistency-fix:** `_DesignMonorepoCapabilityBase._client_sync()` today caches by `working_dir`. Keep that — once `working_dir` is per-agent, the cache key is naturally per-agent. Per-agent clones live at `/mnt/shared/agents/<agent_id>/clones/<scope_id>/` so they survive Ray actor restarts (vs. `~/.colony` on the actor's local FS, which is wiped). The shared *read-only* clone for any agent that doesn't write goes to `/mnt/shared/shared_clones/<scope_id>/` (one per node) and is opened with `read_only=True`. New flag `read_only=True` on `RepoStateProvider.bind(...)` selects the shared clone.

### 6.2 `SessionAgent` + CPS coordinator wiring

`web_ui/backend/routers/sessions.py:367` add to `capability_blueprints`:

```python
*design_monorepo_capability_blueprints(),
# Knowledge trio — agent-driven from chat (no CLI / no auto-routing).
BulkAcquisitionCapability.bind(scope=BlackboardScope.SESSION),
KnowledgeCuratorCapability.bind(scope=BlackboardScope.SESSION),
KnowledgeRetrievalCapability.bind(scope=BlackboardScope.SESSION),
```

For coordinators in the CPS repo (out of tree, lives next to Colony), expose a public helper from `polymathera.colony.design_monorepo`:

```python
from polymathera.colony.design_monorepo import design_monorepo_capability_blueprints
```

CPS code adds `*design_monorepo_capability_blueprints()` to its coordinator blueprint. **No code in this PR lives in the CPS repo.**

### 6.3 `GitRemoteWatcher` → action policy event stream

The capability subscribes via the agent's own input-pattern mechanism (existing path) — no new framework piece needed.

`DesignCheckpointer.__init__` is already an `AgentCapability` subclass; it already declares `@event_handler` on `_on_quiescence`. Add a second handler:

```python
@event_handler(input_pattern=VCMEventProtocol.page_changed_pattern())
async def _on_remote_change(self, event: BlackboardEvent, repl: PolicyREPL) -> None:
    """Translate VCM page-change events from the global mapping into a
    coarser branch-update event the agent's action policy can react to.
    Filters by source_id == self._scope_id.
    """
    if event.payload.get("source_id") != self._scope_id:
        return
    await repl.write_event(
        DesignMonorepoEventProtocol.branch_changed_key(self._scope_id),
        {"page_id": event.payload["page_id"], "kind": event.payload["kind"]},
    )
```

Define `DesignMonorepoEventProtocol` next to `VCMEventProtocol` in `agents/blackboard/protocol.py` — one new protocol class, three keys (`branch_changed`, `branch_merged`, `checkpoint_emitted`). The agent's planning prompt sees these as ordinary blackboard events.

### 6.4 Tests

- Per-agent clone isolation: spawn two `RepoStateProvider`s with `agent_id="A"` and `"B"` against the same `origin_url`; assert distinct clone paths and that a write in A is invisible to B.
- Shared read-only: two `RepoStateProvider(read_only=True)` instances on the same node share one clone path.
- `_on_remote_change`: feed a fake `VCMEventProtocol.page_changed` event into the capability's blackboard, assert a `branch_changed` event is written to the agent's scope.

---

## 7. `KnowledgeRetrievalCapability` (NEW, ~80 lines)

**File:** `agents/patterns/capabilities/knowledge_retrieval.py`.

```python
class KnowledgeRetrievalCapability(AgentCapability):
    """Read-side surface over the knowledge store. Wraps the existing
    retrieval adapters in `polymathera.colony.knowledge.retrieval`."""

    def __init__(
        self,
        agent: Agent,
        *,
        scope: BlackboardScope = BlackboardScope.SESSION,
        capability_key: str = "knowledge_retrieval",
        adapter_name: str = "scoped",   # "scoped" | "grounded" | "hybrid" — registered in retrieval/
        app_name: str | None = None,
    ): ...

    async def initialize(self) -> None:
        await super().initialize()
        from polymathera.colony.knowledge.retrieval import get_adapter
        self._adapter = await get_adapter(self._adapter_name)

    @action_executor()
    async def search_knowledge(
        self, *, query: str, scope: str | None = None, top_k: int = 8,
    ) -> dict[str, Any]: ...

    @action_executor()
    async def get_chunk(self, *, chunk_id: str) -> dict[str, Any]: ...

    @action_executor()
    async def list_corpora(self) -> dict[str, Any]: ...
```

**Re-use checklist:**
- All retrieval logic lives in `knowledge/retrieval/*` already. The capability holds *zero* retrieval state.
- The future "agent contributes knowledge" path adds a separate `KnowledgeWriteCapability`. Out of scope here.

### 7.1 Two ingestion paths into the KB

**Declarative (user-seeded).** `knowledge_routing` rows in `repo_map.yaml` (§5.1) seed the KB with literature already committed to the monorepo. The materialiser runs them on every `Map Repo` call. Default for new literature is the KB — users selectively promote files to VCM later by editing `repo_map.yaml` (typically through the dashboard's "Design Monorepo" tab).

**Chat-driven (agent-acquired).** When the user asks the SessionAgent to acquire *new external* literature, the agent invokes `BulkAcquisitionCapability.acquire_manifest` (or `KnowledgeCuratorCapability.ingest_raw` for ad-hoc curation). These do not touch `repo_map.yaml`.

The two paths share the process-singleton `Ingestor` (and therefore embedder + vector store), so chunks from both coexist in one KB. A PDF can simultaneously appear under a `LiteratureContextPageSource` source row (cache-aware VCM pages) and `knowledge_routing` (KB chunks); they are not mutually exclusive.

### 7.2 No CLI ingestion trigger

The CLI is for cluster bring-up, debugging, and maintenance only. All knowledge acquisition / curation / retrieval is exposed exclusively through agent capabilities and reaches users through the dashboard chat surface.

---

## 8. Dashboard "Design Monorepo" tab

**Files:** `web_ui/frontend/src/components/repo/` (new), `web_ui/backend/routers/repo_map.py` (new), `web_ui/frontend/src/components/layout/AppShell.tsx` (one icon entry).

### 8.1 Backend endpoints

```
GET  /api/v1/repo-map                      → current repo_map.yaml content + parsed sources
PUT  /api/v1/repo-map                      → validate + commit to design monorepo (NEW commit on a config branch)
GET  /api/v1/repo-map/tree?ref=main        → repo file tree (calls existing GitFileStorage)
POST /api/v1/repo-map/preview              → dry-run: resolve mmap calls without executing
```

All endpoints `Ring.USER`, require `require_auth`, gated by tenant/colony in execution context.

### 8.2 Frontend

Two-pane layout:
- **Left:** repo file tree (collapsible). Each path annotated with the source it resolves to, colour-coded.
- **Right:** YAML editor (Monaco) for `repo_map.yaml` with schema validation + a form-based "Add source" button. The drag-and-drop the user mentioned is a Phase 2 nicety — skip until the YAML editor lands and is exercised.

Add tab entry to `AppShell.tsx:23` after `vcm`:
```ts
{ id: 'design-monorepo', label: 'Design Monorepo', icon: GitBranch },
```

### 8.3 Cut for Phase 1

Drag-and-drop mapping; live-preview of how chunks will look in VCM. Both deferred.

---

## 9. Implementation order

The whole thing is too big for one PR. Sequence (each numbered step is one mergeable PR with tests green):

1. **§3** `GitRepoContextPageSource` subdir + binary + watcher kwargs. Default-compatible with existing call sites.
2. **§4** `LiteratureContextPageSource` + `_walk()` extraction. Re-uses §3 helpers.
3. **§5** `.colony/repo_map.yaml` loader + `to_mmap_kwargs` + `materialize_knowledge_routing`. Covers default-for-unmapped fallback (empty `knowledge_routing`).
4. **§7** `KnowledgeRetrievalCapability` (read surface; thin wrapper over existing `knowledge.retrieval` adapters).
5. **§6** Per-agent clones + `design_monorepo_capability_blueprints()` + `_on_remote_change` handler + add the design-monorepo trio AND the knowledge trio (`BulkAcquisitionCapability` + `KnowledgeCuratorCapability` + `KnowledgeRetrievalCapability`) to `SessionAgent`. Depends on 4.
6. **Docs.** Update `colony/docs/` with concrete + concise pages for everything above (see §11).
7. **§8** Dashboard tab. Read-only first (GET endpoints + tree/YAML viewer); PUT/preview after that lands. **Deferred until 1–6 ship.**

Each step ships with the tests called out in its section. None of them require rebuilding the Docker image other than 2 (PDF deps already in `knowledge` extra) and 5 (per-agent clone path under `/mnt/shared` needs the existing colony-shared volume).

---

## 11. Documentation deliverables (Phase 7)

For every change in §3–§7 there is a corresponding short, concrete page under `colony/docs/`. Each page follows the same template: *what it is → when to use it → minimal example → reference of every option*. Pages to add or update:

- `docs/architecture/context-page-sources.md` — overview of the source family, when to use which, table of subclasses with their constructor kwargs.
- `docs/architecture/git-repo-context-page-source.md` — subdir/include/exclude, binary policy, watcher policy, frozen vs live (`static`).
- `docs/architecture/literature-context-page-source.md` — PDF chunking, extractor choices, examples mapping a `literature/` directory.
- `docs/architecture/repo-map.md` — `.colony/repo_map.yaml` schema with annotated example, default-for-unmapped behaviour, submodule resolution.
- `docs/architecture/design-monorepo-capabilities.md` — per-agent clone story, the three subclasses and what each is for, the `_on_remote_change` event surface, agent action vocabulary.
- `docs/architecture/knowledge-capabilities.md` — the agent-driven trio (acquisition / curation / retrieval), how the SessionAgent wires them, example chat flows that trigger each.

**Doc rules:** concrete and concise. Each page opens with a one-paragraph summary, follows with a minimal working example, and ends with a reference table. No vague statements; every claim has a code or file pointer. No duplicating master-spec text — link out to it instead.

---

## 10. Open questions for you

1. **`pathspec` dependency** — confirm OK to add. Otherwise we hand-roll gitignore matching, which is annoying. <u>**Approved**</u>
2. **Per-agent clone location** — `~/.colony/agents/<agent_id>/clones/<scope>/` is my proposal. On the Ray workers this is `/home/ray/.colony/...`. Confirm the path lives long enough for an agent's lifetime (Ray actors restart on failure — clone gets re-cloned, fine, but slow for big repos). Alternative: `/mnt/shared/agents/<agent_id>/<scope>/` so the clone survives actor restarts. <u>**Approved use of `/mnt/shared` for the shared read-only clone too.**</u>
3. **Submodules vs. independent sources** — my proposal supports both (`type: git_repo` with either `origin_url` *or* `submodule:` field). You listed both as options. Confirm both should be supported, or pick one. <u>**Approved both.**</u>
4. **`KnowledgeRetrievalCapability.adapter_name`** — pick one default. I have `"scoped"` as a placeholder; tell me which adapter the `SessionAgent` should use out of the box. <u>**Approved "scoped".**</u>
5. **`LocalFsWatcher` removal** — my plan keeps the class (still useful for tests + odd setups) and changes the default to `watch_local=False`. Confirm you don't want it deleted outright. <u>**Approved to keep the class but default it off.**</u>
6. **Knowledge ingestion trigger** — defer the `polymath ingest` CLI subcommand to Phase 2? Or in scope here? <u>**Knowledge acquisition (`BulkAcquisitionCapability`), curation and ingestion should be driven by agent capabilities integrated with the `SessionAgent` that triggers these operations in response to user chat messages in the UI, not from the CLI. The focus of the CLI now should be on cluster bring-up, debugging and maintenance.**</u>

When approved I will execute steps 1–7 in that order, one PR each, with the tests listed under each section.
