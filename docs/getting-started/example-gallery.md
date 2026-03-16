---
hide:
  - toc
---

# Example Gallery

Explore end-to-end examples showing Colony's capabilities across different domains and complexity levels.

## Code Analysis

<div class="grid cards" markdown>

-   :material-code-braces:{ .lg .middle } __Large Codebase Analysis__

    ---

    Analyze a million-line codebase using Colony's no-RAG approach with Virtual Context Memory. Demonstrates page sharding, page graph construction, and cache-aware multi-agent orchestration.

    `beginner` `code-analysis` `vcm`

-   :material-source-branch:{ .lg .middle } __Change Impact Analysis__

    ---

    Given a set of code changes, agents collaboratively trace impact propagation across a large codebase using the `ChangeImpactAnalysisCoordinatorCapability` with multi-hop page graph traversal.

    `intermediate` `impact-analysis` `page-graph`

-   :material-file-document-check:{ .lg .middle } __Contract Inference__

    ---

    Infer function contracts (preconditions, postconditions, invariants) using qualitative LLM reasoning. Demonstrates `ScopeAwareResult`, multi-level validation, and iterative refinement.

    `intermediate` `contracts` `qualitative-analysis`

-   :material-bug:{ .lg .middle } __Automated Bug Localization__

    ---

    Given a bug report, agents narrow down the root cause across a large codebase using hypothesis games for error correction and planning policies for search strategy.

    `advanced` `hypothesis-games` `planning`

</div>

## Multi-Agent Patterns

<div class="grid cards" markdown>

-   :material-account-group:{ .lg .middle } __Hypothesis Game__

    ---

    Multiple agents propose, challenge, and refine hypotheses about code behavior using Colony's game engine with VCG-incentive scoring and no-regret learning.

    `intermediate` `games` `error-correction`

-   :material-handshake:{ .lg .middle } __Contract Net Delegation__

    ---

    A supervisor agent delegates analysis tasks to specialized agents via contract net auctions. Demonstrates `TaskBid`, `ContractAward`, and capability-based agent selection.

    `intermediate` `games` `delegation`

-   :material-brain:{ .lg .middle } __Memory-Driven Analysis__

    ---

    Agents use working memory, short-term memory, and episodic memory to accumulate findings across analysis rounds. Demonstrates `MemoryCapability` subscriptions and consolidation.

    `advanced` `memory` `capabilities`

</div>

## Getting Started

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Hello Colony__

    ---

    Minimal example: create an agent with a single capability, run it against a small codebase, and inspect the blackboard results.

    `beginner` `quickstart`

-   :material-puzzle:{ .lg .middle } __Custom Capability__

    ---

    Build a custom `AgentCapability` with `@action_executor` methods, hook into the agent lifecycle, and compose it with existing capabilities.

    `beginner` `capabilities` `extensibility`

</div>
