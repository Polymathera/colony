# Knowledge Capabilities — Acquisition, Curation, Retrieval

The knowledge layer is exposed to agents as a **trio** of
capabilities. Each is a thin agent-facing wrapper over the existing
`knowledge.*` machinery; together they let the SessionAgent (or any
other agent) drive the master §6.3 ingestion pipeline and the
master §6.4 retrieval modes from chat — no CLI, no auto-routing.

| Capability | Role | Key actions |
|---|---|---|
| `BulkAcquisitionCapability` | Acquire and ingest a corpus declared by a manifest. | `acquire_manifest` |
| `KnowledgeCuratorCapability` | Curate single sources; review-queue handling. | `ingest_raw`, `list_review_queue`, `resolve_review_item`, `mirror_to_design_monorepo` |
| `KnowledgeRetrievalCapability` | Search the knowledge base. | `search_knowledge`, `list_modes` |

All three are wired into the `SessionAgent` blueprint by default. A
chat message like *"ingest these two PDFs as scientific papers"*
becomes a `KnowledgeCuratorCapability.ingest_raw` action; *"summarise
what we know about <topic>"* becomes a
`KnowledgeRetrievalCapability.search_knowledge`. The CLI does **not**
expose any of these — knowledge work is agent-driven.

## Process-singleton deps

All three capabilities share the same backends — embedder, vector
store, and (optional) graph store — so curation and retrieval
operate in the same embedding space. The shared bundle lives in
`polymathera.colony.knowledge.deps`:

```python
from polymathera.colony.knowledge.deps import (
    get_default_ingestor,    # Ingestor
    get_knowledge_deps,      # RetrievalDeps (embedder + vector store + graph store)
    set_knowledge_deps,      # production override
)
```

Out-of-the-box defaults are the in-memory embedder + vector store
(no persistence, no external services) — fine for development and
single-process deployments. Production overrides via
`set_knowledge_deps(...)` once during cluster bring-up:

```python
# In polymath / dashboard startup
from polymathera.colony.knowledge.deps import set_knowledge_deps
from polymathera.colony.knowledge.stores.qdrant_vector import QdrantVectorStore
from polymathera.colony.knowledge.embedder import ColonyEmbeddingClient

set_knowledge_deps(
    embedder=ColonyEmbeddingClient(...),
    vector_store=QdrantVectorStore(...),
)
```

Calling `set_knowledge_deps` rebuilds the `Ingestor` so all three
capabilities pick up the new backends on the next blueprint
construction.

## Retrieval modes

`KnowledgeRetrievalCapability.search_knowledge` accepts a `mode`
argument selecting one of the master §6.4 modes:

| Mode | When to use |
|---|---|
| `scoped` (default) | Single-source retrieval; pass `source_prefix=...`. |
| `grounded` | Cross-source retrieval with citations enforced. |
| `graph` | Knowledge-graph queries — pass `graph_query`. |
| `budgeted` | Token-bounded retrieval; pass `max_tokens`. |
| `standards` | Time-versioned regulatory retrieval; pass `effective_at`. |

Example:

```python
result = await session_agent.run(
    'search_knowledge(query="kgrid.makeTime", mode="scoped", '
    'source_prefix="docs:k_wave:", top_k=5)'
)
```

The action returns the typed `RetrievalResult` JSON-serialised — a
dict with `mode`, `total_candidates`, and `hits[]` (each hit carries
a full `Chunk` plus `score`, `rank`, and `explanation`).

## How the SessionAgent wires the trio

`web_ui/backend/routers/sessions.py` adds the three blueprints to
`SessionAgent.capability_blueprints`:

```python
*design_monorepo_capability_blueprints(),
BulkAcquisitionCapability.bind(ingestor=get_default_ingestor()),
KnowledgeCuratorCapability.bind(ingestor=get_default_ingestor()),
KnowledgeRetrievalCapability.bind(
    scope=BlackboardScope.SESSION,
    deps=get_knowledge_deps(),
),
```

`get_default_ingestor()` and `get_knowledge_deps()` return the same
process singletons, so the curator's writes are immediately visible
to the retrieval surface — and a follow-up chat *"now look up …"*
hits the chunks the previous turn just ingested.

## Future — `KnowledgeWriteCapability`

Letting agents *contribute* knowledge to the base (write-side
retrieval / claim extraction / curation policies) is intentionally
out of scope for the v1 trio. Today only the curator capability
writes — and only at the user's explicit request via chat. A
future `KnowledgeWriteCapability` will be added when the
contribution policy is settled.
