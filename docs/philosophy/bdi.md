
## The BDI Model

Colony's cognitive architecture maps to the Belief-Desire-Intention (BDI) model from agent theory:

| BDI Component | Colony Implementation |
|--------------|----------------------|
| **Beliefs** | References to blackboard entries the agent considers true. Updated by observation, inference, and peer correction. |
| **Desires** | Explicit `Goal` objects with success criteria and priority. Goals can be hierarchical and can conflict. |
| **Intentions** | Current plans and sub-tasks. The active plan represents the agent's committed course of action. |

The BDI mapping is not decorative. It structures how agents reason about their own state:

- An agent can examine its **beliefs** (blackboard queries) and discover inconsistencies
- An agent can evaluate its **goals** against current progress and adjust priorities
- An agent can inspect its **intentions** (current plan) and decide to revise or abandon them

This self-inspection capability -- reasoning *about* one's own cognitive state -- is what distinguishes Colony's approach from frameworks where agents simply execute a prompt-to-action loop.

## `AgentSelfConcept`

Each agent carries an `AgentSelfConcept` that defines its identity independently of its capabilities:

- **Identity**: Who the agent is (name, description, persona)
- **Goals**: What the agent is trying to achieve
- **Motivations**: Why the agent pursues its goals
- **Values**: Constraints on how the agent should behave

`SelfConcept` is distinct from *role*. An agent's role is defined by its `AgentCapabilities` -- the actions it can perform, the events it can observe, the protocols it can participate in. The `SelfConcept` provides the "why" that guides how those capabilities are used.

## Levels of Cognition

Colony organizes agent behavior into levels, each with distinct processing characteristics:

| Level | Name | Description | Memory Needs | Implementation |
|-------|------|-------------|--------------|----------------|
| L0 | Reflexive | Immediate reactions, pattern matching | Sensory buffer | Rule-based guards, reactive hooks |
| L1 | Deliberative | Goal-oriented planning, action sequencing | Working memory | LLM-based action policies, plan generation |
| L2 | Reflective | Self-assessment, strategy revision | Short-term memory | Reflection capabilities, meta-reasoning |
| L3 | Meta-cognitive | Reasoning about reasoning itself | Long-term memory | Supervisor agents, capability orchestration |

A multi-agent system implements these levels through the **virtual agent** concept: different agents with different capabilities collectively implement the cognitive architecture of a single virtual agent whose reasoning depth and breadth exceed what any individual agent could achieve.

The top-level agent operates at L2-L3 (strategic planning, meta-reasoning). It spawns lower-level agents at L1 (task execution, page analysis). L0 behavior is handled by reactive hooks and rule-based guards that fire automatically without LLM involvement.

!!! tip "Not a metaphor"
    The virtual agent concept is not an analogy. When a supervisor agent spawns child agents, assigns them goals, monitors their progress, and synthesizes their results, it is literally implementing the meta-cognitive level of a single reasoning process distributed across multiple LLM instances. The children are the "hands" and the supervisor is the "executive function."

## How This Differs from Other Frameworks

Most multi-agent frameworks model agents as independent actors that communicate via messages. Colony models a multi-agent system as **the cognitive architecture of a single virtual agent**, where:

- **CrewAI** assigns roles via system prompts. Colony assigns roles via composable capabilities with conscious and subconscious processes.
- **AutoGen** uses conversation turns as the coordination mechanism. Colony uses policy-driven cognitive processes with blackboard-mediated state sharing.
- **LangGraph** encodes agent behavior as explicit state graphs. Colony lets the LLM planner synthesize control flow dynamically from available capabilities.
- **MetaGPT** prescribes Standard Operating Procedures. Colony provides policies with defaults that the LLM can override based on context.

The key difference: in Colony, the cognitive architecture is *layered and introspectable*. An agent can examine its own beliefs, goals, plans, confidence levels, and memory state -- and reason about whether to change them. This self-awareness is not bolted on; it emerges from the policy-based design where every cognitive process is a first-class, queryable component.


!!! info "Intuition vs. Consciousness: A Cognitive Analogy for Agent Architecture"
    Colony's architecture draws an analogy to the distinction between **intuition** and **consciousness**. LLMs provide the "intuition" -- fast, associative, pattern-matching capabilities that can generate ideas, hypotheses, and plans in a single leap but are opaque, inscrutable, and prone to error. The `ActionPolicy` and `AgentCapabilities` provide the "consciousness" -- slower, deliberate, sequential processes that compose, verify, and correct those intuitions into coherent behavior. Just as human consciousness weaves together various cognitive processes (perception, memory, reasoning) with the intuitive leaps of the subconscious mind, Colony's `ActionPolicy` weaves together various `AgentCapabilities` with the intuitive power of the LLM.


!!! bug "Thinking Fast and Slow"
    Should we relate this to Daniel Kahneman's "Thinking, Fast and Slow"? The LLM provides the "fast" intuitive thinking, while the `ActionPolicy` and `AgentCapabilities` provide the "slow" deliberate thinking? Is this a useful analogy or just a superficial one? Are there important aspects of human cognition that this analogy misses? For example, does it capture the role of working memory, attention, or emotion in human thought? Does it oversimplify the relationship between **intuition** and **deliberation**?


| Layer | Cognitive Analogy | Colony Implementation | Properties |
|-------|------------------|-----------------------|------------|
| **Intuition** | Fast, associative, pattern-matching | The LLM itself | Parallel, immediate, capable of remarkable leaps but also prone to hallucination and overconfidence |
| **Consciousness** | Slow, deliberate, sequential | Cognitive policies + action policy | Planning, reflection, error correction, goal tracking |



