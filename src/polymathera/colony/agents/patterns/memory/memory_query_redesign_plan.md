# Plan: Memory Query API Redesign — Typed Queries + ChromaDB Semantic Backend

## Problem

When the LLM planner calls `gather_context` or `recall` with a text query like `"compliance analysis procedures license security quality"`, the pipeline:

1. Wraps it into `MemoryQuery(query="compliance...")`
2. `RecencyRetrieval.retrieve()` calls `backend.query(tags=None, limit=20)` — ignoring the text entirely (line 811 TODO)
3. Returns entries by insertion order, or empty if no entries exist

The LLM planner is misled by descriptions claiming "semantic search" support. The actual backend (`BlackboardStorageBackend`) only supports tag-based and time-range filtering. No semantic search exists anywhere.

**Root cause**: A single polymorphic `MemoryQuery` hides the fact that different query types need different backend capabilities. The LLM doesn't know what's available.

## Design

### Principle: Unified API, Multi-Backend Routing

`MemoryCapability` exposes a **single query API** that supports all query kinds. Internally, it routes to the appropriate backend(s). The LLM planner doesn't need to know about backends — it just constructs a `MemoryQuery` with the fields it needs.

### Query Types

A `MemoryQuery` can express three kinds of retrieval:

1. **Logical** — filter on tags (AND/OR/NOT), time ranges, metadata fields, key patterns. Classical DB query. Routed to `BlackboardStorageBackend`.

2. **Semantic** — vector similarity search on entry content. Routed to `ChromaStorageBackend`. Requires embeddings.

3. **Hybrid** — logical filters + semantic ranking. Both backends queried, results merged.

The query type is **implicit** based on which fields are populated:
- Only `tags`/`time_range`/`key_pattern` → logical
- Only `query` (text) → semantic
- Both → hybrid

### ChromaDB Integration

ChromaDB runs **embedded** (in-process, no container needed). It uses `sentence-transformers` for embeddings, which is already a dependency in the `cpu` extras. ChromaDB persists to a directory on the shared volume (`/mnt/shared/chromadb/<agent_id>/`).

**Why embedded**: Avoids adding another container to the stack. ChromaDB's embedded mode is production-ready for single-process use. Each Ray worker process gets its own ChromaDB client. Data is persisted to disk and survives restarts.

**Embedding model**: Uses `sentence-transformers` `all-MiniLM-L6-v2` (default, ~80MB, CPU-friendly). The model is loaded once per process and shared across all memory capabilities.

---

## Changes

### 1. Add `chromadb` dependency

**File**: `pyproject.toml`

Add to dependencies and create a new extras group or add to `cpu`:
```toml
chromadb = { version = "^1.0.0", optional = true }

[tool.poetry.extras]
cpu = [
    "anthropic",
    "openai",
    "sentence-transformers",
    "chromadb",
]
```

ChromaDB 1.0+ uses `sentence-transformers` internally for default embeddings. Since `sentence-transformers` is already in `cpu`, they share the same model.

**File**: `Dockerfile.local` — no changes needed (already installs `cpu` extras).

### 2. Extend `MemoryQuery` with logical filter expressions

**File**: `colony/python/colony/agents/patterns/memory/types.py`

Current `MemoryQuery` has `query: str | None` and `tags: set[str]`. Extend it:

```python
class TagFilter(BaseModel):
    """Logical filter expression over tags.

    Supports AND (all_of), OR (any_of), NOT (none_of) combinators.
    The LLM planner constructs these to do precise tag-based retrieval.
    """
    all_of: set[str] = Field(default_factory=set, description="Entry must have ALL of these tags")
    any_of: set[str] = Field(default_factory=set, description="Entry must have at least ONE of these tags")
    none_of: set[str] = Field(default_factory=set, description="Entry must have NONE of these tags")

class MemoryQuery(BaseModel):
    """Query parameters for recalling memories.

    Supports three query modes based on which fields are populated:
    - Logical: tag_filter and/or time_range and/or key_pattern (classical DB query)
    - Semantic: query text (vector similarity search)
    - Hybrid: both logical filters + semantic ranking

    The LLM planner constructs MemoryQuery objects to retrieve relevant memories.
    """
    # Semantic search
    query: str | None = Field(
        default=None,
        description="Natural language query for semantic similarity search"
    )

    # Logical filters (backward-compatible: old `tags` field still works)
    tags: set[str] = Field(
        default_factory=set,
        description="Filter by tags (entry must have ALL of these tags). "
                    "Simple shorthand for tag_filter.all_of."
    )
    tag_filter: TagFilter | None = Field(
        default=None,
        description="Advanced tag filter with AND/OR/NOT logic. "
                    "Overrides `tags` if both are provided."
    )

    # Time filtering
    max_age_seconds: float | None = Field(
        default=None,
        description="Only return memories created within this time window"
    )
    time_range: tuple[float, float] | None = Field(
        default=None,
        description="(start_timestamp, end_timestamp) filter"
    )

    # Key pattern
    key_pattern: str | None = Field(
        default=None,
        description="Key glob pattern filter (e.g., 'scope:Action:*')"
    )

    # Result control
    max_results: int = Field(default=10, ge=1, le=100)
    min_relevance: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum relevance score (0-1). Applied to semantic scores."
    )
    include_expired: bool = Field(default=False)
```

**Backward compatibility**: The existing `tags: set[str]` field is kept. If `tag_filter` is also provided, `tag_filter` takes precedence. Callers using `MemoryQuery(tags={"action"})` continue to work.

### 3. Create `ChromaStorageBackend`

**File**: `colony/python/colony/agents/patterns/memory/backends/chroma.py` (new file)

```python
class ChromaStorageBackend:
    """Vector storage backend using ChromaDB (embedded mode).

    Provides semantic similarity search over memory entries.
    Uses sentence-transformers for embeddings (shared model instance).
    Data persists to /mnt/shared/chromadb/<scope_id>/.

    This backend is used as the SECONDARY backend in HybridStorageBackend.
    It does NOT implement the full StorageBackend protocol — only the
    vector-specific operations needed by HybridStorageBackend.
    """

    def __init__(self, scope_id: str, persist_dir: str | None = None):
        ...

    async def add(self, key: str, text: str, metadata: dict, tags: set[str]) -> None:
        """Add or update a document embedding."""
        ...

    async def search(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,  # ChromaDB metadata filters
    ) -> list[tuple[str, float]]:
        """Semantic search. Returns list of (key, similarity_score)."""
        ...

    async def delete(self, key: str) -> bool:
        """Remove a document by key."""
        ...

    async def list_all_tags(self) -> dict[str, int]:
        """Return all unique tags and their counts."""
        ...
```

**Key decisions**:
- ChromaDB operations are CPU-bound (embedding computation). Wrap in `asyncio.to_thread()` to avoid blocking the event loop.
- Each `scope_id` maps to a ChromaDB collection. Collections are isolated.
- Embeddings are computed from a text representation of the entry value (serialized to string).
- The ChromaDB client is created per-process and cached (module-level singleton).
- `persist_dir` defaults to `/mnt/shared/chromadb/` (the shared Docker volume).

### 4. Create `HybridStorageBackend`

**File**: `colony/python/colony/agents/patterns/memory/backends/hybrid.py` (new file)

```python
class HybridStorageBackend:
    """Composite backend: BlackboardStorageBackend + ChromaStorageBackend.

    Implements the full StorageBackend protocol.
    Routes operations to the appropriate backend:
    - write() → writes to BOTH (blackboard for structured data, chroma for embeddings)
    - read() → blackboard (source of truth for full entries)
    - query() → blackboard (logical queries)
    - search_semantic() → chroma (vector similarity)
    - query_hybrid() → chroma filters + blackboard hydration
    - delete() → both
    - stream_events_to_queue() → blackboard (event streaming)

    The blackboard remains the source of truth for all entry data.
    ChromaDB is a secondary index for semantic search only.
    """

    def __init__(
        self,
        blackboard_backend: BlackboardStorageBackend,
        chroma_backend: ChromaStorageBackend,
    ):
        ...

    # --- Full StorageBackend protocol ---

    async def write(self, key, value, metadata, tags=None, ttl_seconds=None):
        """Write to blackboard AND index in ChromaDB."""
        await self.blackboard.write(key, value, metadata, tags, ttl_seconds)
        # Extract text representation for embedding
        text = self._extract_text(value, metadata)
        if text:
            await self.chroma.add(key, text, metadata, tags or set())

    async def query(self, pattern=None, tags=None, time_range=None, limit=100):
        """Logical query — delegates to blackboard."""
        return await self.blackboard.query(pattern, tags, time_range, limit)

    async def search_semantic(self, query_text: str, n_results: int = 10) -> list[ScoredEntry]:
        """Semantic search — delegates to chroma, hydrates from blackboard."""
        results = await self.chroma.search(query_text, n_results)
        entries = []
        for key, score in results:
            entry = await self.blackboard.read(key)
            if entry:
                entries.append(ScoredEntry(entry=entry, score=score, components={"semantic": score}))
        return entries

    async def delete(self, key):
        """Delete from both."""
        await self.chroma.delete(key)
        return await self.blackboard.delete(key)

    # stream_events_to_queue → blackboard
    # read → blackboard
    # count, clear → blackboard + chroma cleanup

    def _extract_text(self, value: dict, metadata: dict) -> str | None:
        """Extract embeddable text from entry value.

        Strategy: Concatenate all string values from the dict.
        For Action entries: action_type + parameters + result summary.
        For MemoryRecord: content text.
        """
        ...
```

**Key design**: Blackboard remains source of truth. ChromaDB is a **secondary index**. If ChromaDB data is lost, it can be rebuilt from blackboard entries. Write operations are **dual-write** (blackboard first, then chroma). This is acceptable because both are in-process and local.

### 5. Create `HybridStorageBackendFactory`

**File**: `colony/python/colony/agents/patterns/memory/backends/hybrid.py` (same file)

```python
class HybridStorageBackendFactory:
    """Factory that creates HybridStorageBackend instances.

    Falls back to BlackboardStorageBackend if ChromaDB is not available
    (e.g., chromadb not installed or import fails).
    """

    def __init__(self, agent: "Agent", chroma_persist_dir: str | None = None):
        self._agent = agent
        self._persist_dir = chroma_persist_dir
        self._chroma_available = self._check_chroma_available()

    @staticmethod
    def _check_chroma_available() -> bool:
        try:
            import chromadb
            return True
        except ImportError:
            return False

    async def create_for_scope(self, scope_id: str) -> StorageBackend:
        blackboard = await self._agent.get_blackboard(scope="shared", scope_id=scope_id)
        bb_backend = BlackboardStorageBackend(
            scope_id=scope_id, blackboard=blackboard, agent_id=self._agent.agent_id
        )

        if not self._chroma_available:
            logger.warning(
                f"ChromaDB not available, falling back to blackboard-only for {scope_id}. "
                "Semantic queries will return empty results."
            )
            return bb_backend

        chroma_backend = ChromaStorageBackend(
            scope_id=scope_id, persist_dir=self._persist_dir
        )
        return HybridStorageBackend(bb_backend, chroma_backend)
```

### 6. Update retrieval to handle query routing

**File**: `colony/python/colony/agents/patterns/memory/protocols.py`

Modify `RecencyRetrieval.retrieve()` to handle the three query modes:

```python
class RecencyRetrieval:
    async def retrieve(self, query, backend, context=None):
        # Determine query mode from populated fields
        has_semantic = bool(query.query)
        has_logical = bool(query.tags or query.tag_filter or query.time_range or query.key_pattern)

        if has_semantic and hasattr(backend, 'search_semantic'):
            # Semantic or hybrid query
            scored = await backend.search_semantic(query.query, query.max_results * 2)

            if has_logical:
                # Hybrid: apply logical filters to semantic results
                scored = self._apply_logical_filters(scored, query)

            # Apply min_relevance threshold
            scored = [s for s in scored if s.score >= query.min_relevance]
            return scored[:query.max_results]

        # Logical-only query (or backend doesn't support semantic)
        # Resolve effective tags from tags field or tag_filter
        effective_tags = self._resolve_tags(query)

        entries = await backend.query(
            pattern=query.key_pattern,
            tags=effective_tags,
            time_range=query.time_range,
            limit=query.max_results * 2,
        )

        # Existing recency scoring logic...
        ...

    def _resolve_tags(self, query: MemoryQuery) -> set[str] | None:
        """Resolve tags from query.tags or query.tag_filter.all_of."""
        if query.tag_filter:
            return query.tag_filter.all_of or None
        return query.tags if query.tags else None

    def _apply_logical_filters(self, scored: list[ScoredEntry], query: MemoryQuery) -> list[ScoredEntry]:
        """Apply tag_filter and time constraints to pre-scored entries."""
        result = []
        for se in scored:
            entry = se.entry
            # Tag filtering
            if query.tag_filter:
                if query.tag_filter.all_of and not query.tag_filter.all_of.issubset(entry.tags):
                    continue
                if query.tag_filter.any_of and not query.tag_filter.any_of.intersection(entry.tags):
                    continue
                if query.tag_filter.none_of and query.tag_filter.none_of.intersection(entry.tags):
                    continue
            elif query.tags and not query.tags.issubset(entry.tags):
                continue
            # Time filtering
            if query.max_age_seconds is not None:
                import time
                if time.time() - entry.created_at > query.max_age_seconds:
                    continue
            result.append(se)
        return result
```

**Note**: `any_of` and `none_of` tag filtering is done in Python (post-filter) because the blackboard backend only supports AND-style tag matching. For semantic queries, ChromaDB's `where` clause can handle these natively.

### 7. Add memory exploration actions to `MemoryCapability`

**File**: `colony/python/colony/agents/patterns/memory/capability.py`

```python
@action_executor(action_key="memory_list_tags",
    planning_summary="List all tags stored in this memory scope with their counts. "
                     "Use this to discover available tags before constructing tag-based queries.")
async def list_tags(self) -> dict[str, int]:
    """List all unique tags and their counts in this scope.

    Returns:
        Dict mapping tag name to count of entries with that tag.
    """
    entries = await self.storage.query(limit=100000)
    tag_counts: dict[str, int] = {}
    for entry in entries:
        for tag in entry.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return dict(sorted(tag_counts.items(), key=lambda x: -x[1]))

@action_executor(action_key="memory_stats",
    planning_summary="Get statistics about this memory scope: entry count, tag distribution, age range.")
async def stats(self) -> dict[str, Any]:
    """Get statistics about stored memories.

    Returns:
        Dict with entry_count, unique_tags, oldest_entry_age_seconds,
        newest_entry_age_seconds, tag_counts (top 20).
    """
    entries = await self.storage.query(limit=100000)
    if not entries:
        return {"entry_count": 0}

    now = time.time()
    tag_counts: dict[str, int] = {}
    for entry in entries:
        for tag in entry.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    ages = [now - e.created_at for e in entries]
    top_tags = dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:20])

    return {
        "entry_count": len(entries),
        "unique_tags": len(tag_counts),
        "oldest_entry_age_seconds": max(ages),
        "newest_entry_age_seconds": min(ages),
        "tag_counts": top_tags,
    }
```

### 8. Add `list_tags` to `AgentContextEngine`

**File**: `colony/python/colony/agents/patterns/memory/context.py`

```python
@action_executor(action_key="list_all_tags",
    planning_summary="List all tags across all memory scopes. "
                     "Returns a dict mapping scope_id to tag counts. "
                     "Use this to discover available tags for constructing queries.")
async def list_all_tags(self) -> dict[str, dict[str, int]]:
    """List all tags across all memory scopes."""
    result = {}
    for cap in self._memory_capabilities:
        tags = await cap.list_tags()
        if tags:
            result[cap.scope_id] = tags
    return result
```

### 9. Fix action group descriptions

**File**: `colony/python/colony/agents/patterns/memory/capability.py`

Update `get_action_group_description()`:

```python
def get_action_group_description(self) -> str:
    return (
        f"Memory Scope ({self.scope_id}) — storage with tag-based and semantic retrieval. "
        "recall retrieves memories using tags (logical filters), natural language (semantic search), or both (hybrid). "
        "Use list_tags to discover available tags before querying. "
        "store adds entries with tags; forget/prune/deduplicate maintain quality. "
        "Subscriptions auto-collect from other scopes; ingest_now processes pending immediately."
    )
```

### 10. Update defaults to use `HybridStorageBackendFactory`

**File**: `colony/python/colony/agents/patterns/memory/defaults.py`

In `create_default_memory_hierarchy()`, pass `HybridStorageBackendFactory` instead of `BlackboardStorageBackendFactory`:

```python
from .backends.hybrid import HybridStorageBackendFactory

# Use hybrid backend (blackboard + chroma) for all memory levels
storage_factory = HybridStorageBackendFactory(agent)

# Pass to each MemoryCapability
working = MemoryCapability(
    agent=agent,
    scope_id=f"agent:{agent.agent_id}:working",
    storage_backend_factory=storage_factory,
    ...
)
```

The `HybridStorageBackendFactory` gracefully falls back to blackboard-only if `chromadb` is not installed.

---

## Files Summary

| File | Change | New? |
|------|--------|------|
| `pyproject.toml` | Add `chromadb` to `cpu` extras | No |
| `types.py` | Add `TagFilter`, extend `MemoryQuery` with `tag_filter`, `time_range`, `key_pattern` | No |
| `backends/chroma.py` | `ChromaStorageBackend` — embedded ChromaDB wrapper | **Yes** |
| `backends/hybrid.py` | `HybridStorageBackend` + `HybridStorageBackendFactory` | **Yes** |
| `protocols.py` | Update `RecencyRetrieval.retrieve()` for query routing + `TagFilter` support | No |
| `capability.py` | Add `list_tags()`, `stats()` actions; fix `get_action_group_description()` | No |
| `context.py` | Add `list_all_tags()` action | No |
| `defaults.py` | Use `HybridStorageBackendFactory` | No |

---

## Data Flow

```
LLM Planner generates MemoryQuery:
  ├─ query="compliance analysis"           → SEMANTIC
  ├─ tags={"action", "success"}            → LOGICAL
  ├─ tag_filter={any_of: {"infer","plan"}} → LOGICAL (advanced)
  ├─ query="security" + tags={"action"}    → HYBRID
  └─ (no query, no tags)                   → ALL (recency-sorted)

MemoryCapability.recall(query)
  └─ RecencyRetrieval.retrieve(query, backend)
       │
       ├─ has_semantic AND backend is HybridStorageBackend?
       │    YES → backend.search_semantic(query.query, n_results)
       │          → ChromaDB vector search
       │          → Hydrate entries from BlackboardStorageBackend
       │          → Apply logical filters (if hybrid)
       │          → Apply min_relevance threshold
       │    NO  → Fall through to logical
       │
       └─ Logical query:
            → Resolve tags from query.tags or query.tag_filter.all_of
            → backend.query(pattern, tags, time_range, limit)
            → BlackboardStorageBackend → EnhancedBlackboard
            → Apply any_of/none_of filters in Python
            → Score by recency
            → Return scored entries
```

---

## What This Does NOT Change

- `BlackboardStorageBackend` — untouched (still the primary structured store)
- `EnhancedBlackboard` — untouched
- `StorageBackend` protocol — untouched (HybridStorageBackend implements it, adds `search_semantic`)
- `BlackboardEntry` model — untouched
- Memory hierarchy structure — untouched (same levels, same subscriptions)
- Producer hooks / extractors — untouched
- Maintenance policies — untouched
- `MemoryLens` — untouched
- No container additions (ChromaDB runs embedded)

---

## Graceful Degradation

If `chromadb` is not installed:
- `HybridStorageBackendFactory` detects the missing import and creates `BlackboardStorageBackend` instead
- Semantic queries return empty results (no crash)
- Logical queries work exactly as before
- A warning is logged once per scope

---

## Verification

```bash
colony-env down && colony-env up --workers 3 && colony-env run --local-repo /home/anassar/workspace/agents/crewAI/ --config my_analysis.yaml --verbose
```

1. **list_tags action**: LLM planner should be able to call `list_tags` and see available tags
2. **Logical query with tags**: `recall(MemoryQuery(tags={"action", "success"}))` should return matching entries
3. **Semantic query**: `recall(MemoryQuery(query="what files were analyzed"))` should return semantically relevant entries
4. **Hybrid query**: `recall(MemoryQuery(query="analysis results", tags={"success"}))` should combine both
5. **Graceful degradation**: Remove chromadb from install, verify logical queries still work
6. **Prompt accuracy**: Check that the action group description in the planning prompt accurately describes available query types
