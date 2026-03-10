# Design Insights

Colony is built on a set of design ideas that diverge from mainstream multi-agent frameworks. Where most frameworks treat agents as thin wrappers around LLM calls with tool access, Colony treats multi-agent systems as distributed cognitive architectures with formal coordination, observable state, and emergent behavior.

This section contains deeper dives into three of Colony's most consequential design decisions. They are written for engineers evaluating the framework or considering contributing -- people who want to understand not just *what* Colony does, but *why* it does it this way.

## The Three Pillars

### [AgentCapabilities as AOP Aspects](capabilities-as-aspects.md)

Colony models agent behavior using aspect-oriented programming rather than inheritance hierarchies. Each `AgentCapability` is an aspect; the `ActionPolicy` is the aspect weaver. This produces emergent behavior from capability composition without explicit modeling of all interaction paths.

### [Games as Correctness Mechanisms](game-theoretic-correctness.md)

Game-theoretic protocols in Colony are not coordination overhead -- they are correctness mechanisms that combat specific LLM failure modes. Hallucination maps to hypothesis games, laziness to contract nets, goal drift to objective guard agents. Colony applies formal mechanism design (VCG incentives, no-regret learning, social choice theory) to multi-agent LLM systems.

### [Seven Core Abstraction Patterns](abstraction-patterns.md)

Distilled from analysis of 30+ code analysis strategies, these seven patterns generalize to any domain involving partial knowledge and discovered relationships. They form the backbone of Colony's approach to distributed reasoning under uncertainty.

---

!!! tip "Who should read these?"

    If you are deciding whether Colony's architecture fits your problem, start with **Games as Correctness Mechanisms** -- it addresses the most common objection ("why is multi-agent coordination worth the overhead?"). If you are building custom capabilities, read **AgentCapabilities as AOP Aspects** first. If you are designing analysis or reasoning pipelines, the **Abstraction Patterns** page will save you from reinventing Colony's primitives.
