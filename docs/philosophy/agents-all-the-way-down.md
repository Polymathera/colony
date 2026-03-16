# Agents All the Way Down

Colony is built on a single, testable conjecture:

> **General intelligence is emergent from the right composition of LLM-based reasoning policy and action space.**

The conjecture is bold: compose enough LLM-based agents with the right mix of capabilities, and general intelligence emerges. Colony is the testbed for that conjecture.

This architectural claim has specific consequences for how the framework is built.

!!! danger "Nested Action Policies"

    Add explanation here of how nested recursive action policies enable the emergence of general intelligence, but we can exploit agents to emulate this recursive structure.


!!! danger "Nested Action Policies"

    Add explanation here of how the action policy is an aspect weaver that decides control and data flow inside an agent.


<!-->
## The Argument from Bounded Depth

A single LLM call has finite reasoning depth. No matter how capable the model, its forward pass executes a fixed number of layers, and the chain-of-thought it produces in a single generation has practical limits. This is not a flaw -- it is a fundamental property of any finite computational process.

But many real-world tasks require reasoning depth that exceeds what any single call can produce. Understanding the full implications of a change across a million-line codebase. Tracing a causal chain through hundreds of scientific papers. Synthesizing a legal argument that accounts for thousands of precedents.

Colony's answer: **iterative deepening**. If a single LLM call produces finite-depth reasoning, then iterative deepening of that reasoning -- with reflection, learning, and accumulated context -- produces unbounded-depth reasoning. Each iteration builds on the findings of the previous one, effectively traversing arbitrary path lengths through the implicit knowledge graph.


!!! tip "Unbounded Depth, Unbounded Context"
    Iterative deepening gives unbounded reasoning *depth* (arbitrary path length in the knowledge hypergraph). **Distributed reasoning over partitioned context** gives unbounded reasoning *breadth* (relationships with arbitrary degree in the knowledge hypergraph). Colony combines both.

## The Argument from Bounded Context
-->

## The Virtual Agent

Here is Colony's most provocative architectural idea: a multi-agent system is not a collection of independent agents collaborating on a task. It is the **different cognitive levels of a single virtual agent**.

Consider how human cognition works at different levels:

| Level | Human Cognition | Colony Implementation |
|---|---|---|
| L0: Reflexive | Immediate reactions, pattern matching | Rule-based guards, reactive policies |
| L1: Deliberative | Goal-oriented planning, sequencing | LLM-based action policies, plan generation |
| L2: Reflective | Self-assessment, strategy revision | Reflection capabilities, meta-reasoning agents |
| L3: Meta-cognitive | Reasoning about reasoning itself | Supervisor agents, capability orchestration |

In Colony, each level can be implemented by different agents with different capabilities. The top-level agent has higher-level, more abstract capabilities (strategic planning, meta-reasoning). Lower-level agents have specialized, fine-grained capabilities (page analysis, code inspection, hypothesis testing). Together, they implement the cognitive architecture of a single virtual agent whose reasoning depth and breadth exceed what any individual agent could achieve.

```mermaid
graph TB
    subgraph "Virtual Agent"
        Meta[Meta-Cognitive Agent<br/>L3: Reasoning about reasoning]
        Meta --> Reflect1[Reflective Agent<br/>L2: Strategy revision]
        Meta --> Reflect2[Reflective Agent<br/>L2: Self-assessment]
        Reflect1 --> Delib1[Deliberative Agent<br/>L1: Code analysis plan]
        Reflect1 --> Delib2[Deliberative Agent<br/>L1: Test generation plan]
        Reflect2 --> Delib3[Deliberative Agent<br/>L1: Documentation plan]
        Delib1 --> Page1[Page Agent<br/>L0: Analyze module A]
        Delib1 --> Page2[Page Agent<br/>L0: Analyze module B]
        Delib2 --> Page3[Page Agent<br/>L0: Generate tests]
    end
```


## What This Means in Practice

The "agents all the way down" philosophy produces concrete architectural decisions:
1. **Dynamic hierarchies.** The agent hierarchy is not fixed at design time. Agents spawn sub-agents and agent pools, form teams and coalitions, play games, and dissolve -- all decided at runtime by the action policy based on the task.

