# Unified Planning Capabilities — Design Proposal

## Problem

`CacheAwareActionPolicy` and `CodeGenerationActionPolicy` both need the same **planning infrastructure** — cache analysis, learned patterns, conflict detection, plan coordination, replanning triggers. Currently this infrastructure lives in policy classes (`ActionPlanningCachePolicy`, `ActionPlanLearningPolicy`, `ActionPlanCoordinationPolicy`) that are called by `CacheAwareActionPlanner` in a **hardcoded sequence** BEFORE the LLM is invoked. The LLM never sees these components — they enrich the prompt context silently.

This means:
1. `CodeGenerationActionPolicy` cannot access any of this infrastructure. It would need to duplicate the entire **pre-LLM enrichment pipeline**.
2. The LLM planner in `CacheAwareActionPolicy` cannot proactively request cache analysis or conflict checks — these happen automatically, whether the LLM wants them or not.
3. **The sequence is rigid**: always learning → cache → strategy. The LLM has no choice about ordering, repeating, or skipping steps.

## The Action Policy Space

Action policies sit at the intersection of two independent axes:

```
                            Structure / Guidance
                (Granularity of planning abstractions or primitives
                            provided to the LLM)
                ────────────────────────────────────────────────────►
                    None            Optional           Full
                (LLM decides    (available but     (pre-programmed
                 everything)     not forced)        sequence)

Turing Completeness
    Code Gen    ┌──────────────────┬──────────────────┬──────────────────┐
    (arbitrary  │                  │                  │                  │
     Python)    │     CodeGen      │    CodeGen +     │    CodeGen +     │
         ▲      │     Minimal      │    Planning      │      Full        │
         │      │                  │   Capabilities   │    Pipeline      │
     Execution  │                  │                  │                  │
        Mode    ├──────────────────┼──────────────────┼──────────────────┤
         │      │                  │                  │                  │
         │      │     Minimal      │    Minimal +     │    CacheAware    │
         ▼      │     Action       │    Planning      │      Action      │
    JSON action │     Policy       │   Capabilities   │      Policy      │
    selection   │                  │                  │                  │
                └──────────────────┴──────────────────┴──────────────────┘
```

**Bottom-left**: `MinimalActionPolicy` — LLM picks from `@action_executor` **domain actions**. No planning infrastructure.

**Bottom-middle**: `MinimalActionPolicy` + planning capabilities registered on the agent. The LLM sees `analyze_cache`, `get_learned_patterns`, `check_plan_conflicts` as available actions and MAY use them (its choice).

**Bottom-right**: `CacheAwareActionPolicy` — full pre-programmed pipeline: learning → cache → strategy → replan. LLM doesn't decide when to analyze cache; the planner always does it before prompting.

**Top-left**: `CodeGenerationActionPolicy` with no planning capabilities. LLM writes Python that *parameterizes* and calls **domain actions** only.

**Top-middle**: `CodeGenerationActionPolicy` with planning capabilities. LLM writes Python that can call `self.agent.get_capability_by_type(CacheAnalysisCapability).analyze_cache_requirements(...)` — the full programmatic API, not just `@action_executor` wrappers.

**Top-right**: `CodeGenerationActionPolicy` with full planning pipeline available. LLM can call the entire planner programmatically OR bypass it and call individual components.

## Core Insight

The **planning infrastructure** components solve real problems that ANY action policy needs:

| Component | Problem it solves |
|-----------|-------------------|
| `ActionPlanningCachePolicy` | "Which pages should I prioritize? What's my working set? How do I sequence actions for cache locality?" |
| `ActionPlanLearningPolicy` | "What has worked before for similar goals? What patterns should I reuse?" |
| `ActionPlanCoordinationPolicy` | "Will my plan conflict with other agents? How do I resolve contention?" |
| `PlanBlackboard` | "Where do I publish my plan so others can see it? How do I read sibling plans?" |
| `PlanEvaluator` | "How good is this plan? What's the cost/benefit tradeoff?" |
| `ReplanningPolicy` | "Should I replan now? What went wrong?" |

These are **cognitive capabilities** — they augment an agent's reasoning about its own planning. They should be `AgentCapability` subclasses with `@action_executor` methods, not hidden policy internals.

The planning components (`CacheAnalysisCapability`, `PlanLearningCapability`, `PlanCoordinationCapability`, etc.) must support **three consumption modes**:

| Mode | Consumer | Interface used | Example |
|------|----------|---------------|---------|
| **Pre-programmed** | `CacheAwareActionPlanner` | Programmatic API (complex params) | `cache_cap.analyze_cache_requirements(planning_context)` |
| **LLM-selected** | `MinimalActionPolicy` | `@action_executor` (simple params) | LLM picks `{"action_type": "analyze_cache", "parameters": {"page_ids": [...]}}` |
| **Code-generated** | `CodeGenerationActionPolicy` | Programmatic API via generated code | LLM writes `cache_ctx = await cap.analyze_cache_requirements(ctx)` |

The **programmatic API** serves both pre-programmed and code-generated modes. The `@action_executor` wrappers serve the LLM-selected mode.

## Design: Planning Components as `AgentCapabilities`

### The key question: How do both policies use the same capabilities?

**`CacheAwareActionPolicy`** uses them in a **pre-programmed sequence**:
```
1. learning.get_applicable_patterns(context)    → patterns
2. cache.analyze_cache_requirements(context)    → cache_context
3. strategy.generate_plan(context, patterns, cache_context)  → plan
```

This sequence is encoded in `CacheAwareActionPlanner.create_plan()`. The LLM doesn't decide the order — the planner does.

**`CodeGenerationActionPolicy`** should let the LLM decide when and how to call them:
```python
# LLM generates this code:
patterns = await run("get_learned_patterns", goal="analyze code")
cache_ctx = await run("analyze_cache", page_ids=pages)

# LLM can make decisions the pre-programmed sequence can't:
if cache_ctx.output["min_cache_size"] > 20:
    log("Large working set — splitting into batches")
    batches = await run("optimize_action_sequence", page_ids=pages, batch_size=10)
    for batch in batches.output["batches"]:
        await run("analyze_pages", page_ids=batch)
else:
    await run("analyze_pages", page_ids=pages)
```

### The answer: Dual-mode capabilities

Each planning component becomes an `AgentCapability` with `@action_executor` methods for the LLM. But it ALSO exposes a **programmatic API** (plain async methods without `@action_executor`) that `CacheAwareActionPlanner` and `CodeGenerationActionPolicy` call directly.

The `@action_executor` methods are thin wrappers that:
1. Accept simple, LLM-producible parameters (strings, lists, dicts)
2. Call the programmatic API internally
3. Return structured results the LLM can use

```python
class CacheAnalysisCapability(AgentCapability):
    """Cache-aware planning analysis for VCM-paged context."""

    # ── Programmatic API (used by CacheAwareActionPlanner and CodeGenerationActionPolicy) ──

    async def analyze_cache_requirements(
        self, context: PlanningContext, page_graph: nx.DiGraph | None = None
    ) -> CacheContext:
        """Full cache analysis. Called by CacheAwareActionPlanner.create_plan()."""
        ...

    async def optimize_action_sequence(
        self, actions: list[Action], cache_context: CacheContext
    ) -> list[Action]:
        """Reorder actions for cache locality. Called programmatically."""
        ...

    # ── LLM API (used by MinimalActionPolicy via run()) ──

    @action_executor(
        planning_summary="Analyze cache requirements for the given pages and return working set priorities.",
    )
    async def analyze_cache(self, page_ids: list[str] | None = None) -> dict[str, Any]:
        """Analyze which pages to prioritize for cache-optimal execution.

        Args:
            page_ids: Pages to analyze (defaults to agent's bound pages).

        Returns:
            Dict with working_set, priorities, spatial_locality, sizing info.
        """
        context = await self._build_minimal_context(page_ids)
        cache_ctx = await self.analyze_cache_requirements(context)
        return cache_ctx.model_dump()  # Simple dict the LLM can inspect

    @action_executor(
        planning_summary="Get cache-optimal page ordering for batch processing.",
    )
    async def get_cache_optimal_batches(
        self, page_ids: list[str], batch_size: int = 10
    ) -> dict[str, Any]:
        """Split pages into cache-friendly batches.

        Args:
            page_ids: Pages to batch.
            batch_size: Max pages per batch.

        Returns:
            Dict with batches (list of page_id lists) and locality scores.
        """
        ...
```

### Dual-interface pattern: Programmatic + LLM interfaces on the same capability

```
┌──────────────────────────────────────────────────────────┐
│                    AgentCapability                       │
│                                                          │
│  Programmatic API              LLM API                   │
│  (complex params,              (@action_executor,        │
│   system objects)               simple params)           │
│                                                          │
│  analyze_cache_requirements    analyze_cache             │
│  (PlanningContext)             (page_ids: list[str])     │
│          │                            │                  │
│          └────── shared logic ────────┘                  │
│                                                          │
│  Called by:                    Called by:                │
│  • CacheAwareActionPlanner    • MinimalActionPolicy      │
│    (direct method call)         (via JSON action select) │
│  • CodeGenerationActionPolicy                            │
│    (via generated Python)                                │
└──────────────────────────────────────────────────────────┘
```

The programmatic API methods are NOT `@action_executor` — they accept complex objects (`PlanningContext`, `CacheContext`, `list[Action]`) that the LLM can't produce. They exist for `CacheAwareActionPlanner` which already has these objects.

<mark>The `@action_executor` methods are simplified wrappers that accept LLM-friendly types and call the programmatic API internally.</mark> The `@action_executor` methods are **simplified entry points** for policies that use JSON action selection (like `MinimalActionPolicy`). `CodeGenerationActionPolicy` doesn't need them — it calls the programmatic API directly in generated code.

### `CacheAwareActionPlanner` adds missing capabilities

When `CacheAwareActionPlanner` initializes, it ensures its required capabilities are registered on the agent:

```python
class CacheAwareActionPlanner:
    async def initialize(self):
        # Ensure required capabilities are on the agent
        if not self.agent.get_capability_by_type(CacheAnalysisCapability):
            self.agent.add_capability(CacheAnalysisCapability(
                agent=self.agent,
                scope=BlackboardScope.COLONY,
                namespace="cache_analysis",
            ))
        if not self.agent.get_capability_by_type(PlanLearningCapability):
            self.agent.add_capability(PlanLearningCapability(
                agent=self.agent,
                scope=BlackboardScope.AGENT,
                namespace="plan_learning",
            ))
        # ... same for coordination, evaluation, replanning

        # Now look them up for the pre-programmed pipeline
        self._cache_cap = self.agent.get_capability_by_type(CacheAnalysisCapability)
        self._learning_cap = self.agent.get_capability_by_type(PlanLearningCapability)
        # ...
```

This means:
- Using `CacheAwareActionPlanner` automatically gives the agent planning capabilities.
- Those capabilities become visible to ANY policy that runs later (including `CodeGenerationActionPolicy` if the policy is swapped).
- Users can pre-register customized versions (e.g., `CacheAnalysisCapability(cache_capacity=50)`) and the planner will use them instead of creating defaults.

## Proposed Capability Classes

### 1. `CacheAnalysisCapability` (replaces `ActionPlanningCachePolicy`)

**Problem solved**: "Which pages should I prioritize? How do I sequence actions for cache locality?"

**Programmatic API** (for `CacheAwareActionPlanner` and `CodeGenerationActionPolicy`):
- `analyze_cache_requirements(context: PlanningContext, page_graph: nx.DiGraph | None = None) -> CacheContext`
- `optimize_action_sequence(actions: list[Action], cache_context: CacheContext) -> list[Action]`
- `estimate_working_set(goals: list[str], actions: list[Action], page_graph: nx.DiGraph | None = None) -> list[str]`

**LLM API** (for `MinimalActionPolicy` and other JSON-selecting policies):
```python
@action_executor(planning_summary="Analyze cache requirements and return working set priorities for given pages.")
async def analyze_cache(self, page_ids: list[str] | None = None) -> dict:
    """Returns working set, priorities, spatial locality, and sizing info."""

@action_executor(planning_summary="Get cache-optimal page ordering for batch processing.")
async def get_cache_optimal_batches(self, page_ids: list[str], batch_size: int = 10) -> dict:
    """Returns batches of page IDs optimized for cache locality."""

@action_executor(planning_summary="Get dependency graph information for a page.")
async def get_page_dependencies(self, page_id: str) -> dict:
    """Returns page neighbors, centrality score, and cluster membership."""
```

### 2. `PlanLearningCapability` (replaces `ActionPlanLearningPolicy`)

**Problem solved**: "What has worked before for similar goals?"

**Programmatic API**:
- `get_applicable_patterns(context: PlanningContext) -> list[PlanPattern]`
- `learn_from_execution(plan: ActionPlan, outcome: dict) -> None`
- `get_similar_plans(goals: list[str], context: PlanningContext, limit: int = 5) -> list[PlanExecutionRecord]`

**LLM API**:
```python
@action_executor(planning_summary="Get action patterns that worked for similar goals in the past.")
async def get_learned_patterns(self, goal: str) -> dict:
    """Returns patterns with recommended actions, confidence scores, and applicability."""

@action_executor(planning_summary="Query execution history for past plans matching a goal.")
async def get_execution_history(self, goal: str, outcome: str | None = None, limit: int = 5) -> dict:
    """Returns past execution records with success rates and durations."""

@action_executor(planning_summary="Record the outcome of the current plan for future learning.")
async def record_outcome(self, success: bool, quality_score: float = 0.0) -> dict:
    """Stores execution outcome so future agents can learn from it."""
```

### 3. `PlanCoordinationCapability` (replaces `ActionPlanCoordinationPolicy`)

**Problem solved**: "Will my plan conflict with other agents? How do I coordinate?"

**Programmatic API**:
- `check_conflicts(plan: ActionPlan, other_plans: list[ActionPlan]) -> list[ActionPlanConflict]`
- `resolve_conflict(plan: ActionPlan, conflict: ActionPlanConflict) -> ActionPlan | None`

**LLM API**:
```python
@action_executor(planning_summary="Check if current working set conflicts with other agents' plans.")
async def check_plan_conflicts(self) -> dict:
    """Returns list of conflicts with severity, type, and recommended resolution."""

@action_executor(planning_summary="Get plans of sibling agents in the colony.")
async def get_sibling_plans(self) -> dict:
    """Returns active plans of other agents including their working sets and goals."""

@action_executor(planning_summary="Propose current plan for approval by parent agent.")
async def propose_plan(self, description: str) -> dict:
    """Publishes plan to the colony plan blackboard for coordination."""

@action_executor(planning_summary="Resolve cache contention with another agent by staggering or partitioning.")
async def resolve_contention(self, conflicting_agent_id: str, strategy: str = "stagger") -> dict:
    """Adjusts plan to reduce overlap with the specified agent's working set."""
```

### 4. `PlanEvaluationCapability` (replaces `PlanEvaluator`)

**Problem solved**: "How good is my current plan?"

**Programmatic API**:
- `evaluate(plan: ActionPlan, context: PlanningContext) -> PlanEvaluation`

**LLM API**:
```python
@action_executor(planning_summary="Evaluate the quality of the current plan (cost, benefit, risk).")
async def evaluate_plan(self) -> dict:
    """Returns utility score, cost estimate, benefit estimate, and risk factors."""
```

### 5. `ReplanningCapability` (wraps `CompositeReplanningPolicy`)

**Problem solved**: "Should I revise my plan?"

**Programmatic API**:
- `evaluate_replanning_need(state: ActionPolicyExecutionState, last_result: ActionResult | None) -> ReplanningDecision`
- `reset_state(state: ActionPolicyExecutionState) -> None`

**LLM API**:
```python
@action_executor(planning_summary="Check if the current plan should be revised based on progress and failures.")
async def should_replan(self) -> dict:
    """Returns whether to replan, the triggers, recommended strategy, and reason."""
```

## `MinimalActionPolicy`

The simplest possible policy: gather actions, show to LLM, get selection, dispatch.

```python
class MinimalActionPolicy(BaseActionPolicy):
    """Bare-bones action policy. No planning infrastructure, no event processing.

    The LLM sees all available @action_executor methods and picks which one
    to call next. No pre-programmed enrichment, no hardcoded sequence.

    If planning capabilities (CacheAnalysisCapability, PlanLearningCapability,
    etc.) are registered on the agent, their @action_executor methods appear
    in the action list and the LLM can use them — but it's not forced to.

    This is the baseline for evaluating the value that structure adds.
    """

    def __init__(self, agent, max_actions_per_step: int = 1, **kwargs):
        super().__init__(agent=agent, **kwargs)
        self.max_actions_per_step = max_actions_per_step
        self._iteration_count = 0
        self._max_iterations = agent.metadata.max_iterations or 50

    async def plan_step(self, state: ActionPolicyExecutionState) -> Action | None:
        self._iteration_count += 1
        if self._iteration_count > self._max_iterations:
            state.custom["policy_complete"] = True
            return None

        # 1. Gather action descriptions from all registered capabilities
        action_descriptions = await self.get_action_descriptions()

        # 2. Build a minimal prompt
        prompt = self._build_prompt(state, action_descriptions)

        # 3. Call LLM
        response = await self.agent.infer(
            prompt=prompt,
            max_tokens=512,
            temperature=0.3,
            json_schema=self._response_schema(),
        )

        # 4. Parse and return
        return self._parse_response(response, action_descriptions)
```

`MinimalActionPolicy` has zero planning infrastructure. It doesn't know about cache, learning, coordination, or replanning. But if those capabilities are registered on the agent, their `@action_executor` methods (like `analyze_cache`, `get_learned_patterns`, `check_plan_conflicts`) appear in the action list. The LLM decides whether to use them.

This creates the **middle column** of the policy space: `MinimalActionPolicy` + planning capabilities = unstructured JSON selection with optional planning intelligence available on demand.

## Contrasting the Three Policies

| Aspect | `MinimalActionPolicy` | `CacheAwareActionPolicy` | `CodeGenerationActionPolicy` |
|--------|----------------------|--------------------------|------------------------------|
| **Execution mode** | JSON action selection | JSON action selection | Python code generation |
| **Planning structure** | None (LLM decides everything) | Full pipeline (learning → cache → strategy → replan) | LLM writes code that can call any method |
| **How it uses planning capabilities** | Via `@action_executor` (LLM may or may not call them) | Via programmatic API in hardcoded sequence | Via programmatic API in generated Python |
| **Multi-agent coordination** | Only if LLM calls `check_plan_conflicts` action | Automatic (planner always checks) | LLM writes coordination code |
| **Cache awareness** | Only if LLM calls `analyze_cache` action | Automatic (planner always analyzes) | LLM writes cache-aware batching code |
| **Replanning** | None (LLM must decide on its own to change approach) | Automatic (CompositeReplanningPolicy evaluates) | LLM can call `should_replan` in code |
| **Strengths** | Simplest, most flexible, lowest overhead | Most reliable for standard workflows | Most powerful for complex data flow |
| **Weaknesses** | LLM may not use planning intelligence even when beneficial | Rigid sequence, LLM can't skip/reorder steps | Highest complexity, code generation errors |
| **Best for** | Simple agents, evaluation baseline, agents with custom logic | Production analysis workflows with cache-heavy patterns | Complex coordinators, multi-step conditional logic |

## Why `@action_executor` Wrappers Still Matter

Even though `CodeGenerationActionPolicy` doesn't need them, the simplified `@action_executor` methods serve a critical purpose: they make planning intelligence accessible to **any** JSON-selecting policy without requiring that policy to understand the planning system.

Consider a user who writes a custom `MyPolicy(BaseActionPolicy)` for a specific domain. They don't know about Colony's planning infrastructure. But they register `CacheAnalysisCapability` on their agent (because someone told them it helps). Now their LLM planner sees:

```
analyze_cache — Analyze cache requirements and return working set priorities for given pages.
  Parameters: page_ids?: list[str]
```

The LLM can call it without understanding `PlanningContext`, `CacheContext`, or the page graph. The `@action_executor` wrapper handles all of that internally.

Without the wrapper, the user would need to either:
1. Use `CacheAwareActionPolicy` (full pipeline they may not want), or
2. Write code that calls the programmatic API (requires understanding internal types)

The `@action_executor` wrappers are the **accessibility layer** — they make planning intelligence a zero-knowledge add-on for simple policies.

## What This Design Achieves

1. **Same capabilities, two consumption modes.** `CacheAwareActionPlanner` calls the programmatic API (complex objects, pre-programmed sequence). `MinimalActionPolicy` calls the `@action_executor` wrappers (simple params, LLM-decided sequence).

2. **No duplication.** The actual logic (cache analysis, pattern matching, conflict detection) lives once in the capability. The `@action_executor` methods are thin wrappers.

3. **Multi-agent coordination is accessible.** The LLM can call `check_plan_conflicts`, `get_sibling_plans`, `resolve_cache_contention`, `propose_plan` — primitives that `CacheAwareActionPolicy` calls automatically but `MinimalActionPolicy` exposes to the LLM's judgment.

4. **Progressive adoption.** Existing `CacheAwareActionPolicy` users get the same behavior (capabilities are registered on the agent; the planner discovers and calls them). New users can use `MinimalActionPolicy` and get access to the same planning intelligence via code generation.

5. <mark>**The action vocabulary shapes the intelligence.** By exposing cache analysis, learned patterns, conflict detection, and replanning as first-class actions, <u>*the LLM can reason about its own planning process*</u> — not just execute **domain actions**.</mark>

## Implementation Order

### Phase 1: Create capability classes with dual interfaces
1. `CacheAnalysisCapability` — extract logic from `ActionPlanningCachePolicy`, add `@action_executor` wrappers
2. `PlanLearningCapability` — extract from `ActionPlanLearningPolicy`, add wrappers
3. `PlanCoordinationCapability` — extract from `ActionPlanCoordinationPolicy`, add wrappers
4. `PlanEvaluationCapability` — extract from `PlanEvaluator`, add wrappers
5. `ReplanningCapability` — wrap `CompositeReplanningPolicy`, add wrappers

### Phase 2: Wire `CacheAwareActionPlanner` to add + discover capabilities
6. Planner adds missing capabilities on `initialize()`
7. Planner looks up capabilities via `agent.get_capability_by_type()`
8. Programmatic API call sequence preserved exactly

### Phase 3: Implement `MinimalActionPolicy`
9. Minimal prompt builder (goals + action descriptions + last result)
10. JSON schema for response (action_type + parameters)
11. Action key resolution (reuse existing fuzzy matching)
12. Factory function `create_minimal_action_policy(agent)`

### Phase 4: Update `CodeGenerationActionPolicy`
13. Ensure `browse()` shows planning capability methods
14. Add planning capability usage examples to code generation prompt
15. Test code generation with cache-aware multi-agent coordination

### Phase 5: Documentation and evaluation
16. Update `docs/architecture/action-policies.md` with the policy space diagram
17. Document dual-interface pattern with examples
18. Document how to evaluate policies against each other on the same task

---

## Philosophy: The Agent's Environment as a Programming Environment

### Core thesis

The agent's environment — its capabilities, blackboard, VCM, sibling agents, and the external systems it interacts with — can be modeled as a **programming environment**. The agent's interactions with this environment are most naturally expressed as **dynamically generated code** that manipulates the environment's API surface.

This is not a metaphor. It is a literal architectural claim:

| Environment concept | Programming analogy | Colony implementation |
|---|---|---|
| Agent capabilities | Libraries/modules | `AgentCapability` classes with methods |
| Blackboard | Shared mutable state | `EnhancedBlackboard` (Redis-backed KV store) |
| VCM pages | File system | `PageStorage` + `VirtualContextManager` |
| Sibling agents | Network services | `DeploymentHandle` RPC calls |
| Action executors | Public API functions | `@action_executor` decorated methods |
| Programmatic methods | Internal library functions | Non-decorated async methods on capabilities |
| REPL namespace | Process memory | `PolicyPythonREPL.user_ns` |
| Execution history | Call stack / log | `_execution_history` list |

When the LLM generates code rather than selecting from a JSON menu, it operates as a **programmer writing against an API**, not as a decision-maker picking from a fixed set of options. This is a fundamental shift — it means the LLM can:

1. **Compose** operations with control flow (loops, conditionals, error handling)
2. **Transform** data between operations (parse, filter, aggregate)
3. **Introspect** the API surface at runtime (`browse()`, `inspect`)
4. **Adapt** its approach based on intermediate results (without replanning)

### The semantic gap

The main challenge: how does the LLM know what the environment's API surface *means*? When a developer reads a library's documentation, they understand:

- What each function does (behavior)
- What it expects (preconditions, parameter types and semantics)
- What it returns (postconditions, return type and structure)
- How it relates to other functions (dependencies, typical call sequences)
- When to use it vs. alternatives (design intent)

The LLM needs the same information. Currently, the LLM receives:

- **For `@action_executor` methods**: The `planning_summary` string + compact parameter signature. This is analogous to a function's one-line docstring + type stub — minimal.
- **For programmatic methods**: Nothing, unless `browse("programmatic")` is called, which shows full signatures and docstrings.
- **For the environment itself**: Nothing explicit. The LLM must infer from context (goals, execution history, capability names) what the environment looks like and how to interact with it.

### `CodeInspector`: Extracting and conveying semantics

The `CapabilityBrowser` currently provides three levels of detail: group summaries → action signatures → full docstrings. This is adequate for JSON action selection but insufficient for code generation, where the LLM needs to understand API contracts well enough to write correct calls.

A `CodeInspector` generalizes this by extracting **structured semantics** from any code artifact:

```python
class CodeInspector:
    """Extracts and formats the semantics of code artifacts for LLM consumption.

    Can inspect:
    - @action_executor methods (signature + planning_summary + param descriptions)
    - Programmatic methods on AgentCapability (full docstring + type hints)
    - Pydantic models (field schemas, validation rules)
    - Generated code snippets (infer behavior from structure)
    - External APIs (from OpenAPI specs, CLI help text, library docs)
    """

    async def inspect_method(self, method) -> MethodSemantics:
        """Extract structured semantics from a method."""
        ...

    async def inspect_capability(self, cap: AgentCapability) -> CapabilitySemantics:
        """Extract full API surface of a capability."""
        ...

    async def inspect_model(self, model_cls: type[BaseModel]) -> ModelSemantics:
        """Extract field-level schema from a Pydantic model."""
        ...

    async def suggest_usage(self, goal: str, available_apis: list[MethodSemantics]) -> str:
        """Use LLM to suggest how to combine APIs to achieve a goal."""
        ...
```

**`MethodSemantics`** would include:
- `name`, `signature`, `return_type`
- `description` (full, not truncated)
- `parameters` with per-parameter descriptions and types
- `preconditions` (what must be true before calling)
- `postconditions` (what is guaranteed after calling)
- `related_methods` (typically called before/after this one)
- `example_usage` (a concrete code snippet)
- `side_effects` (what state changes does this method cause)

**`CapabilitySemantics`** would include:
- `name`, `description`
- `public_methods` (list of `MethodSemantics`)
- `state_model` (what persistent state does this capability maintain)
- `typical_workflow` (the usual sequence of method calls)
- `access_pattern` (how to get an instance: `_agent.get_capability_by_type(...)`)

This is not a theoretical exercise. The LLM generating code NEEDS this information to:
- Know that `analyze_cache_requirements` returns a `CacheContext` with fields `working_set`, `spatial_locality`, `working_set_priority`
- Know that `get_applicable_patterns` should be called BEFORE `generate_plan` because patterns inform the prompt
- Know that `check_conflicts` takes a list of OTHER agents' plans, not the current agent's plan
- Know that after `optimize_action_sequence`, the actions should be dispatched in the returned order

Without structured semantics, the LLM must guess from method names — and guesses produce bugs.

### Progressive disclosure of semantics

A full `CodeInspector` dump of all capabilities would exceed the context window. The solution is **progressive disclosure** — the same principle used in `CapabilityBrowser` but applied to semantics:

1. **Level 0 (always in prompt)**: Capability names + one-line descriptions. ~100 tokens per capability.
2. **Level 1 (on demand via `browse()`)**: Method signatures + first paragraph of docstring. ~200 tokens per method.
3. **Level 2 (on demand via `browse("group.method")`)**: Full docstring + parameter descriptions + return type + example. ~500 tokens per method.
4. **Level 3 (on demand via `inspect`)**: `MethodSemantics` with preconditions, postconditions, related methods, side effects. ~1000 tokens per method. This level may require an LLM call to generate (using the source code as input).

The LLM starts at Level 0 and drills down as needed. This is how developers navigate large codebases — they don't read every file; they skim, then drill into what's relevant.

### Self-describing capabilities

The `@action_executor` decorator already captures some semantics (`planning_summary`, `input_schema`, `output_schema`, `reads`, `writes`). To support `CodeInspector`, capabilities should be self-describing at a richer level:

- **`get_typical_workflow()`**: Returns a code snippet showing the typical sequence of method calls. This is the "README example" of the capability.
- **`get_related_capabilities()`**: Returns a list of capabilities that are typically used together (e.g., `CacheAnalysisCapability` → `WorkingSetCapability`).
- **`get_state_description()`**: Describes what persistent state the capability maintains and how it changes.

These methods are optional — the `CodeInspector` can fall back to docstring parsing and type inference when they're not implemented. But when present, they dramatically improve the LLM's ability to generate correct code.

### Interface between generated code and the environment

The current interface is:

```python
# Through dispatcher (tracked, hooked, observable)
result = await run("action_key", param1=val1, ...)

# Through programmatic API (direct, untracked)
cap = _agent.get_capability_by_type(CacheAnalysisCapability)
cache_ctx = await cap.analyze_cache_requirements(context)
```

The `run()` path goes through the `ActionDispatcher`, which provides tracking (execution history), hooks (memory observation), Ref resolution, and error handling. The direct programmatic path bypasses all of this — the generated code is responsible for its own error handling.

For production use, the generated code should prefer `run()` for observability. But for complex multi-step programmatic operations (building a `PlanningContext`, calling `analyze_cache_requirements`, then `optimize_action_sequence`), the direct path is more natural and efficient.

The design decision: **both paths are valid, and the LLM should understand the tradeoff**. The `CodeInspector` should annotate each method with whether it's "tracked" (goes through dispatcher) or "untracked" (direct call), so the LLM can make informed decisions.

### Context window optimization

The fundamental tension: the LLM needs enough information to generate correct code, but showing everything exceeds the context window. The mode system (planning vs. execution) is one dimension of optimization. Others:

1. **Scope selection** (already implemented in `ActionPlanningStrategy`): Before showing detailed action descriptions, ask the LLM which capability groups are relevant. Only expand those.
2. **Progressive disclosure** (described above): Start with summaries, let the LLM request detail.
3. **Execution context pruning**: Only show the last N steps of execution history, not the full trace.
4. **Result compression**: Store full action results in the REPL namespace but show only summaries in the prompt.
5. **Capability-aware caching**: If the LLM called `browse("CacheAnalysisCapability")` in a previous iteration, the response can be cached in the REPL namespace — the LLM can reference it without re-requesting.

These optimizations compose: scope selection reduces the number of groups, progressive disclosure reduces per-group tokens, result compression reduces execution context tokens. Together they keep the prompt within bounds even for agents with 50+ capabilities and 100+ action history entries.
