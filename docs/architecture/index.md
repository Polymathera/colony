# Architecture Overview

!!! bug "Duplicated Docs and Diagrams"
    Unify with `distributed.md`.

Colony is a cache-aware multi-agent framework for reasoning over extremely long context (billion-token scale) without RAG. Instead of retrieving fragments, Colony keeps the entire context "live" through paged distributed KV caching across a GPU cluster, allowing agents to perform <u>*deep, iterative reasoning over the full input*</u>.

## Core Design Principles

1. **Explicit context over implicit context.** LLMs struggle with knowledge buried in training data. Deep reasoning requires explicating implicit context into explicit, in-context material.

2. **The LLM is the planner, not the framework.** Agent control flow and all decisions are driven by a reasoning LLM given sufficient context, not hardcoded logic. The framework provides context, asks "what next?", executes, and feeds back results.

3. **Policy-based design.** Every cognitive process is a pluggable policy with well-defined interfaces and default implementations. Policies compose hierarchically.

4. **All state on the blackboard.** No out-of-band state in instance variables. State changes become observable events via blackboard notifications, enabling the memory-as-observer pattern.

5. **Cache-awareness is emergent.** Cache efficiency is not a property of individual agents but a collective property of the multi-agent system as a whole.

## System Architecture

!!! bug "Add More Details and Shown Design Options"
    This digram does not convey how flexible the architecture is. For example, the memory system is not constrained to the specific hierarchy shown here -- users can design arbitrary memory hierarchies with custom scopes and maintenance policies. The VCM is not constrained to specific page sources. The agent system supports both VCM-bound agents and unbound (**floating**) agents that operate on blackboard state without direct page bindings.

```mermaid
graph TB
    subgraph Cluster["GPU Cluster"]
        VCM["Virtual Context Memory<br/>(VCM)"]
        vLLM["vLLM Replicas"]
    end

    subgraph AgentSystem["Agent System"]
        Agent["Agent<br/>(base.Agent)"]
        AP["ActionPolicy<br/>(CacheAwareActionPolicy)"]
        Cap["AgentCapabilities"]
        Hooks["Hook System<br/>(AOP)"]
    end

    subgraph Memory["Memory System"]
        BB["EnhancedBlackboard<br/>(Redis-backed)"]
        WM["Working Memory"]
        STM["Short-Term Memory"]
        LTM["Long-Term Memory"]
    end

    subgraph Games["Game Engine"]
        GP["GameProtocolCapability"]
        Roles["Proposer / Skeptic /<br/>Grounder / Arbiter"]
    end

    Agent --> AP
    Agent --> Cap
    Agent --> Hooks
    AP --> Cap
    Cap --> BB
    Cap --> VCM
    VCM --> vLLM
    Agent --> BB
    WM --> STM
    STM --> LTM
    BB --- WM
    BB --- STM
    BB --- LTM
    GP --> BB
    GP --> Roles
```

## Subsystems


### [Distributed Architecture](distributed.md)

Colony is natively distributed. This section covers Colony's serving framework, request routing, multi-tenancy, autoscaling, and the heterogeneous LLM cluster.

### [Virtual Context Memory](virtual-context-memory.md)

The VCM manages context pages like an OS manages virtual memory with page tables, page faults, and cache-aware scheduling. It operates at the cluster level (across GPU nodes), unlike vLLM which is node-level. Extended VCM combines immutable read-only input pages with read-write blackboard output.

### [Agent System](agent-system.md)

Agents are autonomous computational entities with lifecycle states, capabilities, and action policies. The framework supports VCM-bound agents (loaded/unloaded with pages), unbound agents, service agents, and supervisor agents. `AgentCapability` provides the extension point -- each capability is an AOP aspect, and the `ActionPolicy` acts as the aspect weaver.

### [Agent Memory System](agent-memory-system.md)

A unified memory architecture where all state lives in blackboards. Memory is organized hierarchically -- sensory, working, short-term, and long-term (episodic, semantic, procedural) -- with each level implemented as a `MemoryCapability` managing a blackboard scope. Agents reason *about* their memory, not just *with* it.

### [Action Policies](action-policies.md)

The decision-making core. The LLM selects actions through a two-phase process (choose action, then parameterize), executes partial plans in a Model-Predictive Control loop, and revises as conditions change. `CacheAwareActionPolicy` is the primary implementation, coordinating planning, execution, and replanning.

### [Blackboard](blackboard.md)

`EnhancedBlackboard` is the single source of truth for all agent state. Redis-backed, event-driven, with optimistic concurrency control. Policy-based design (access, eviction, validation) without inheritance hierarchies.

### [Hook System](hook-system.md)

Aspect-oriented programming for cross-cutting concerns. The `@hookable` decorator marks interception points; `Before`, `After`, and `Around` hooks attach via `Pointcut` expressions. Used for token tracking, rate limiting, checkpointing, and memory capture without polluting core logic.

### [Game Patterns](game-engine.md)

A framework for structured multi-agent deliberation. Four game types -- hypothesis, bidding/contract, negotiation, consensus -- with defined roles and an Agent Communication Language. Games serve as error correction mechanisms: hypothesis games combat hallucination, contract nets combat laziness, objective guards combat goal drift.

### [Observability](observability.md)

Distributed tracing and persistent logging for long-running sessions. Traces capture structured execution spans (LLM calls, agent steps) via `TracingCapability`. Logs capture all Python logging under `polymathera.colony` via `KafkaLogHandler`. Both flow through Kafka to PostgreSQL for durable, queryable post-mortem debugging -- even after agents stop.

### [Training](training.md)

Coming soon.


## Key Classes

| Class | Module | Role |
|-------|--------|------|
| `Agent` | `polymathera.colony.agents.base` | Base agent with lifecycle, capabilities, blackboard access |
| `ActionPolicy` | `polymathera.colony.agents.base` | Abstract base for action selection and planning |
| `CacheAwareActionPolicy` | `polymathera.colony.agents.patterns.actions.policies` | Primary policy: MPC-style planning with cache awareness |
| `AgentCapability` | `polymathera.colony.agents.base` | Extension point for agent functionality (AOP aspect) |
| `EnhancedBlackboard` | `polymathera.colony.agents.blackboard` | Redis-backed shared state with events and OCC |
| `MemoryCapability` | `polymathera.colony.agents.patterns.memory` | Manages one memory scope with ingestion, retrieval, maintenance |
| `GameProtocolCapability` | `polymathera.colony.agents.patterns.games` | Structured multi-agent deliberation protocol |
| `AgentHookRegistry` | `polymathera.colony.agents.patterns.hooks` | Per-agent hook registration and dispatch |
