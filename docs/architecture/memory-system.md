# Memory System

Colony's memory system implements a cognitive memory architecture where agents reason *about* their memory, not just *with* it. All memory state lives in blackboards, memory levels form a dataflow graph, and the LLM planner has full introspection into the memory layout.

## Unified Storage Principle

**All agent state must be stored in blackboards.** No out-of-band state in instance variables. This single rule enables:

- State changes as observable events via blackboard notifications
- Memory as a bidirectional observer pattern (agents observe memories; memories observe agent behavior)
- Transparent persistence, replication, and recovery
- Cross-agent memory sharing through shared blackboard scopes

## Memory Hierarchy

Memory is organized as a dataflow graph where nodes are memory scopes and edges are subscriptions. Each level is managed by a `MemoryCapability` instance from `polymathera.colony.agents.patterns.memory`.

```mermaid
graph TB
    Sensory["Sensory Memory<br/>(events, observations)"]
    Working["Working Memory<br/>(recent actions, plans)"]
    STM["Short-Term Memory<br/>(consolidated summaries)"]
    LTM_E["LTM: Episodic<br/>(game experiences, task outcomes)"]
    LTM_S["LTM: Semantic<br/>(learned concepts, facts)"]
    LTM_P["LTM: Procedural<br/>(strategies, skills)"]

    Sensory --> Working
    Working --> STM
    STM --> LTM_E
    STM --> LTM_S
    STM --> LTM_P
```

| Level | Scope | TTL | Purpose |
|-------|-------|-----|---------|
| Sensory | `agent:{id}:sensory` | Seconds | Raw events and observations |
| Working | `agent:{id}:working` | ~1 hour | Recent actions, current plan, immediate context |
| Short-Term | `agent:{id}:stm` | ~1 day | Consolidated summaries from working memory |
| LTM Episodic | `agent:{id}:ltm:episodic` | Persistent | Past experiences, task outcomes, game results |
| LTM Semantic | `agent:{id}:ltm:semantic` | Persistent | Learned concepts, domain knowledge |
| LTM Procedural | `agent:{id}:ltm:procedural` | Persistent | Strategies, skills, procedural knowledge |

## MemoryCapability

`polymathera.colony.agents.patterns.memory.MemoryCapability` manages a single memory scope. It handles:

- **Ingestion**: Pull data from subscribed scopes, transform via `ingestion_policy.transformer`
- **Storage**: Write entries to its scope with tags and metadata
- **Retrieval**: Query via `recall()` with semantic, logical, or hybrid queries
- **Maintenance**: Background decay, pruning, deduplication, reindexing

```python
from polymathera.colony.agents.patterns.memory import (
    MemoryCapability, MemorySubscription, MemoryProducerConfig,
)

stm = MemoryCapability(
    agent=agent,
    scope_id=MemoryScope.agent_stm(agent_id),
    ingestion_policy=MemoryIngestPolicy(
        subscriptions=[
            MemorySubscription(source_scope_id=MemoryScope.agent_working(agent_id)),
        ],
        transformer=SummarizingTransformer(agent=agent, prompt="..."),
    ),
    ttl_seconds=86400,
    max_entries=500,
)
```

### Architecture: Pull Model

Each capability only manages its own scope. There is no "push" logic:

1. CapabilityB subscribes to CapabilityA's scope
2. CapabilityB's ingestion transformer consolidates incoming data
3. CapabilityA's maintenance policies clean up old entries independently

## Memory as Observer Pattern

Memory formation uses hook-based producers (`MemoryProducerConfig` in `polymathera.colony.agents.patterns.memory.types`). A producer attaches a hook to a `@hookable` method and extracts storable data when that method executes:

```python
MemoryProducerConfig(
    pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
    extractor=extract_action_from_dispatch,  # (ctx, result) -> (data, tags, metadata)
    ttl_seconds=3600,
)
```

This implements the principle: **memory observes agent behavior**. The agent does not explicitly write to memory after each action -- the memory system captures it automatically via hooks.

## Query System

`MemoryQuery` (in `polymathera.colony.agents.patterns.memory.types`) supports three query modes:

### Semantic Search
Vector similarity via ChromaDB embeddings:
```python
MemoryQuery(query="What authentication approach was used?")
```

### Logical Filtering
Structured filtering with `TagFilter` (AND/OR/NOT combinators):
```python
MemoryQuery(tag_filter=TagFilter(
    all_of={"action", "success"},
    none_of={"action_type:infer"},
))
```

### Hybrid Queries
Combine semantic search with logical filtering -- results are filtered first, then ranked by similarity:
```python
MemoryQuery(
    query="security analysis results",
    tag_filter=TagFilter(any_of={"action_type:infer", "action_type:plan"}),
    max_results=10,
)
```

Additional query controls include `time_range`, `key_pattern`, `max_age_seconds`, `min_relevance`, and `include_expired`.

## Memory Lenses

A `MemoryLens` is a read-only view over memory with custom filtering and ranking, defined in `polymathera.colony.agents.patterns.memory.types`. Lenses do not copy data -- they are configured query interfaces for different contexts:

- **`PLANNING_LENS`**: Recent actions and goals from working/STM (last hour, max 10 results)
- **`REFLECTION_LENS`**: Past experiences from STM/LTM episodic (action results, game experiences)
- **`SKILL_LENS`**: Procedural knowledge from LTM procedural (max 5 results)

## Memory Introspection

The LLM planner can inspect the memory system at runtime through structured types:

- **`MemoryMap`**: Complete layout of all memory scopes, their configurations, entry counts, and dataflow edges
- **`MemoryScopeInfo`**: Per-scope details including capacity, TTL, subscription relationships, and pending ingestion counts
- **`MemoryStatistics`**: Health and usage statistics across all scopes

This introspection is what enables agents to reason *about* their memory -- the LLM can examine what it knows, what it has forgotten, and what is pending consolidation.

!!! important "Agents reason about memory"
    Memory is not a passive store. The LLM planner can query available tags via `list_tags`, inspect the memory map, and construct targeted `MemoryQuery` objects. It can decide to consolidate, forget, or search based on its current goals. Memory is a first-class cognitive resource navigable by the `ActionPolicy`.

## Memory Scopes by Ownership

| Scope Level | Lifetime | Example |
|-------------|----------|---------|
| Agent private | Agent lifetime | Working memory, personal STM |
| Capability-scoped | Capability lifetime | Game protocol state |
| Task-scoped | Task duration | Shared analysis context |
| Collective | Team lifetime | Team-wide knowledge base |
| Global system | Application lifetime | System-wide facts, managed by `MemoryManagementAgent` |

## Maintenance Policies

`MaintenanceConfig` controls background memory maintenance:

- **Decay**: Reduce relevance over time (`decay_rate` per minute)
- **Pruning**: Remove entries below a relevance threshold
- **Deduplication**: Merge entries above a similarity threshold (default 95%)
- **Reindexing**: Periodically update embeddings
- **Access tracking**: Track access counts and timestamps for LRU-style policies

These are subconscious cognitive processes -- they run in the background without LLM involvement, keeping each memory level healthy.
