# Philosophy

Colony is built on a set of unorthodox ideas about how multi-agent systems should work. These ideas challenge conventional assumptions in the LLM application ecosystem -- particularly around retrieval-augmented generation, monolithic agent design, and the treatment of inference infrastructure as an afterthought.

<s>The framework's name during early development was **Syaq** (from the Arabic word for "context"), reflecting the core conviction that *context is everything*. Not retrieved context. Not summarized context. Live, active, exhaustive context -- managed with the same rigor that operating systems bring to virtual memory.</s>


!!! tip "Rewrite this"
    To extract or synthesize novel insights, the current breed of research agents may need to ingest large amounts of context perhaps in multiple passes revisiting the same information repeatedly. The order in which they ingest this context is important because it determines data locality in LLM KV caches. Moreover, in many cases, insights often emerge from unexpected connections between distant pieces of information which requires keeping the entire context live and accessible, not filtering it through retrieval. This is the problem Colony is designed to solve.


## Core Convictions

Three philosophical pillars support Colony's architecture:

1. **The NoRAG Paradigm** -- Colony keeps the full context live and accessible, not filtering it through retrieval. RAG activates sparse subsets and misses the dense, cross-cutting connections where breakthroughs happen.

2. **Cache-Aware Patterns** -- When context spans billions of tokens distributed across a GPU cluster, cache management is not an optimization -- it is the dominant factor in whether reasoning succeeds or fails.

3. **Agents All the Way Down** -- General intelligence is not a property of any single model. It emerges from the right composition of LLM-based reasoning agents that communicate, coordinate, and collaborate across unbounded context.

<s>These are not incremental improvements on existing agent frameworks. They represent a different mental model for what multi-agent systems are *for* and how they should be built.</s>

## The Unifying Idea

Colony reconceptualizes deep research as a **game** with a large number of possible moves available to agents at every step. One class of moves is combinations of currently known facts that offer the smallest leap to new insights. Because the narrowest leap across the **discovery front** is often unpredictable, the entire context must remain live -- not filtered through retrieval -- because breakthroughs emerge from unpredictable connections between distant pieces of information.

A dynamic group of agents iteratively walks a page graph, accumulating state, communicating findings, and coordinating their traversal to maximize KV cache reuse. The page graph itself is built and refined as agents explore, creating a self-improving map of how context relates to itself.

```mermaid
graph TD
    A((Deep Research Task)) --> B[Build Initial Page Graph]
    B --> PG((Page Graph))
    PG --> MAS[Agent Swarm Traverses Graph] --> |New Connections| PG
    MAS --> CAS[Cache-Aware Scheduling] --> MAS
    MAS --> G[Games: Hypothesis, Negotiation, Coalition, Consensus Game] --> MAS
    MAS --> M[Memory Architecture] --> MAS
    MAS --> CAP[Capabilities: Page Attention, Reflection, Refinement, Validation, Grounding] --> MAS
    MAS --> F((Insights Synthesized))
```

## Read More

- [The NoRAG Paradigm](no-rag.md) -- Why retrieval-augmented generation is the wrong abstraction for deep reasoning
- [Agents All the Way Down](agents-all-the-way-down.md) -- How general intelligence emerges from composition
- [Cache-Aware Patterns](cache-awareness.md) -- Why cache management is a first-class concern, not an optimization
- [The Consciousness-Intuition Interface](consciousness-intuition.md) -- How Colony models cognition as layered intuition + deliberate policies
