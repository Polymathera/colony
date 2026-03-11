# Qualitative Analysis

Colony reframes classical algorithmic analyses as **LLM-driven qualitative reasoning**. The insight: human programmers already perform "fuzzy" versions of formal analyses mentally -- they trace likely execution paths, reason about probable behaviors, and identify patterns without computing exact dataflow equations or building complete call graphs. LLMs can do the same, at scale, across extremely long context.

This reframing is not specific to code analysis. The same patterns generalize to any domain where agents work with partial knowledge and discovered relationships.

## The Reframing Principle

Classical program analyses compute exact results over formal abstractions (control flow graphs, type lattices, points-to sets). These are expensive, often intractable for large systems, and brittle in the face of dynamic behavior. Colony replaces exact computation with **qualitative reasoning** -- confidence-scored, evidence-backed assessments that improve iteratively.

| Classical Analysis | Colony Reframing | What Changes |
|-------------------|-----------------|--------------|
| Symbolic Execution | Execution Narratives | Exact path constraints → natural language path descriptions with risk markers |
| Abstract Interpretation | Lattice Hints | Fixpoint computation → confidence-scored assertions that narrow iteratively |
| Points-To Analysis | Alias Storytelling | Constraint graphs → ownership "stories" with allocation sites and thread touchpoints |
| Taint Analysis | Information Flow Tracking | Propagation rules → qualitative flow tracking with sanitization validation |
| API Misuse Detection | Contract Cards | Pattern matching → contract summaries + usage profiling with consensus validation |
| Architectural Conformance | Policy Narratives | Rule checking → layer intent cards + conformance scouting with breach reporting |

## Execution Narratives

**Classical**: Symbolic execution explores all execution paths by maintaining symbolic state for each variable. Cost: exponential path explosion.

**Colony**: An `ExecutionNarrativeAgent` describes each discovered path in natural language, producing `PathNarrative` artifacts:

- **Entry context**: What conditions lead to this path
- **Guard summaries**: Branch conditions described qualitatively ("requires admin role", "assumes list non-empty")
- **Side effects**: What state changes occur along this path
- **Risk markers**: Where the path may fail or produce unexpected behavior

A `ConstraintSketchBoard` accumulates qualitative predicates across narratives. When two narratives reference the same variable under conflicting constraints, a coordinator asks the LLM whether the paths actually conflict -- replacing formal constraint solving with targeted qualitative reasoning.

## Lattice Hints

**Classical**: Abstract interpretation computes fixpoints over lattices (intervals, nullness, taint) by iterating transfer functions until convergence. Cost: depends on lattice height and program size.

**Colony**: Agents emit `AbstractHint` objects -- likely bounds or invariants plus supporting evidence:

- "Loop counter is bounded by array length" (evidence: loop condition, array allocation)
- "Return value is non-null after this check" (evidence: null guard at line 42)
- "Taint cleared after sanitizer call" (evidence: sanitizer invocation pattern)

A `HintMergePolicy` narrows hints when they are compatible and flags contradictions when they are not. A `FixpointOrchestrator` reruns hint refinement until aggregate confidence exceeds a threshold -- an iterative convergence process analogous to fixpoint computation, but driven by LLM reasoning rather than abstract transfer functions.

## Alias Storytelling

**Classical**: Points-to analysis builds constraint graphs and runs algorithms (Andersen's, Steensgaard's) to determine which pointers may alias. Cost: cubic or worse for flow-sensitive analysis.

**Colony**: An `AliasStoryAgent` produces ownership "stories" tracking:

- Allocation sites and their lifecycles
- Alias sets (which references point to the same object)
- Thread touchpoints (where aliased objects cross thread boundaries)
- Confidence vectors referencing observed patterns (RAII, pooling, singleton)

Stories are indexed by resource ID. A `RelationshipGraphBuilder` converts stories into qualitative alias edges in the page graph, enabling cross-page reasoning about shared state.

## Information Flow Tracking

**Classical**: Taint analysis propagates taint markers along dataflow edges using predefined propagation rules. Cost: linear in program size but requires complete call graph.

**Colony**: An `InformationFlowTracker` traces how untrusted data flows through the system qualitatively:

```python
class TaintFlow(BaseModel):
    source: str                      # Where untrusted data enters
    flow_paths: list[str]            # Qualitative description of flow
    sinks: list[str]                 # Where data reaches sensitive operations
    sanitization_points: list[str]   # Where sanitization is applied
    confidence: float                # How certain is this flow
    vulnerability_risk: str          # Assessment of risk level
```

The LLM validates sanitization adequacy -- not just whether a sanitizer was called, but whether it is the *right* sanitizer for the specific data type and context. A `TaintFlowMergePolicy` combines flows discovered by different agents across different pages.

## Contract Cards

**Classical**: API misuse detection matches call patterns against known rules (e.g., "must call `close()` after `open()`"). Cost: requires manual rule authoring for each API.

**Colony**: A two-step process:

1. **Contract Summarizer** distills reference docs and tests into `ContractCard` artifacts:
    - Preconditions (what must be true before calling)
    - Mandatory sequencing (what calls must happen in what order)
    - Forbidden states (what should never occur)

2. **Usage Profiler** scans call sites and records `ContractDelta` artifacts:
    - "Missing `await` before `close`"
    - "Token reused after revoke"
    - Validated via consensus before raising incidents

Contract Cards are a generic schema -- they work for API contracts, compliance rules, SLO definitions, and any domain where usage must conform to documented expectations.

## Policy Narratives

**Classical**: Architectural conformance checking compares dependency graphs against allowed-dependency rules. Cost: requires explicitly maintained architecture models.

**Colony**: A two-step process:

1. **Layer Intent Agent** ingests architecture decision records (ADRs) and produces `LayerPolicyCard` artifacts:
    - Allowed imports between layers
    - Data ownership rules
    - Communication patterns

2. **Conformance Scout** compares actual dependency summaries with policy cards:
    - "Does this UI module import persistence directly?" (LLM prompt)
    - Creates `PolicyBreach` entries referencing both code locations and policy definitions
    - Enables cross-team remediation

## Dynamic Analysis Reframings

The same qualitative approach extends to dynamic analysis:

| Dynamic Analysis | Colony Reframing |
|-----------------|-----------------|
| Fuzzing & crash triage | `CrashNarrative` artifacts from trace analysis, joined with static contract violations |
| Concurrency analysis | `ScheduleHypothesis` stories describing possible races, requiring corroboration from two independent agents |
| Runtime observability | `PerformanceNarrative` artifacts from telemetry, with "good vs bad" trace comparison |
| Compliance monitoring | Runtime events normalized and judged qualitatively against `PolicyCard` artifacts |

## Cross-Domain Generalization

The reframing patterns are not code-specific. They generalize through seven meta-patterns:

### 1. Flow Tracking

Generalizes taint analysis, data flow, slicing, and memory safety into a single `FlowTracker` abstraction. Applied to: knowledge flow in research, influence in social networks, resource flow in supply chains, causality tracking in incident analysis.

### 2. Constraint Accumulation

Generalizes symbolic execution, abstract interpretation, and type checking. Agents accumulate soft constraints with evidence, using lattice operations for consistency checking. Applied to: belief revision, planning constraint satisfaction, configuration management.

### 3. Incremental Refinement

Tracks partial results and refinement dependencies. Results improve as more context is discovered. Applied to: document understanding, medical diagnosis, translation quality improvement.

### 4. Hierarchical Merge

Merges results from multiple agents with type-specific strategies. Applied to: distributed aggregation, consensus building, multi-document summarization, sensor fusion.

### 5. Query-Driven Discovery

Generates queries from findings to discover relevant context. Applied to: research (following citations), investigation (pursuing leads), medical diagnosis (ordering tests).

### 6. Conflict Resolution

Detects and resolves conflicting results from different agents. Applied to: multi-agent systems, information fusion, distributed databases, negotiation.

### 7. Scope-Aware Communication

Messages include scope metadata enabling incremental discovery. Tracks message dependencies and refinement relationships. Applied to: distributed problem solving, collaborative editing, multi-stage pipelines.

!!! tip "The unifying insight"
    All seven meta-patterns serve the same principle: **the right unit of distributed analysis is not the answer -- it is the partial, confidence-scored, context-aware finding that knows what it does not know.** Systems built on `ScopeAwareResult` can route effort precisely where uncertainty is highest, discover relationships that no single agent could see, and converge on high-confidence results through targeted refinement.

## Narrative-Centric Memory

Qualitative analyses produce natural language artifacts (narratives, stories, sketches) rather than formal data structures. Colony stores these in specialized blackboard namespaces:

- **Execution Narrative Store**: Append-only log of path stories indexed by entity ID
- **Constraint Sketch Board**: Shared ledger of soft constraints with contradiction detection
- **Trace Contrast Memory**: Before/after comparisons for observability and security
- **Policy Cards**: Generic rule/contract schema reusable across domains
- **Evidence Notebook**: Multi-modal provenance (code snippet, log entry, narrative) embedded in `ScopeAwareResult`

These structures are queryable via the standard `MemoryCapability` interface, enabling agents to reason about past qualitative analyses when planning new ones.

## Why This Matters

The qualitative analysis approach enables Colony to tackle analyses that are intractable for classical tools:

1. **Scale**: LLM-based analysis scales to million-line codebases where formal analysis would time out
2. **Completeness**: Qualitative reasoning handles dynamic dispatch, reflection, and other features that defeat static analysis
3. **Cross-domain**: The same patterns work for code, research papers, legal documents, and any domain with partial knowledge
4. **Iterative improvement**: Results improve over successive rounds as the page graph stabilizes and more context is discovered
5. **Human-readable output**: Narratives and stories are directly useful to humans, unlike formal analysis artifacts
