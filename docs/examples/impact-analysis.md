# Change Impact Analysis

Trace the ripple effects of code changes across a large codebase. This is Colony's most feature-rich example, demonstrating multi-hop dependency propagation, hypothesis games for validating critical impacts, game-theoretic merge policies, and cache-aware page graph traversal.

## What You'll Learn

- How Colony's coordinator-worker hierarchy distributes analysis across pages
- How **hypothesis games** validate high-severity impact claims before acceptance
- How the **ImpactMergePolicy** resolves conflicting assessments from different agents
- How **page graph traversal** traces multi-hop dependency chains
- How **ScopeAwareResult** tracks confidence and missing context

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/impact-analysis.yaml`](https://github.com/polymathera/colony/blob/main/examples/impact-analysis.yaml)

```yaml
analyses:
  - type: impact
    coordinator_version: v2       # cache-aware coordination
    max_agents: 10
    quality_threshold: 0.7
    max_iterations: 10
    batching_policy: hybrid       # balance cache reuse + coverage
    overlap_threshold: 0.3
    batch_size: 5
    prefetch_depth: 2

    changes:
      - file_path: "src/main.py"
        change_type: modification
        description: "Refactored authentication flow"
      - file_path: "src/auth/handler.py"
        change_type: modification
        description: "Updated token validation logic"

hierarchy:
  extra_capabilities:
    - ReflectionCapability        # self-assessment after each action
    - ConsciousnessCapability     # self-awareness via SystemDocumentation
```

The key field is `changes` — a list of files and descriptions of what changed. The coordinator distributes these across worker agents who trace the impact through their assigned pages.

## Agent Hierarchy

When you run this example, Colony spawns the following agent hierarchy:

```
ChangeImpactAnalysisCoordinator (1 agent)
├── Capabilities: ChangeImpactAnalysisCoordinatorCapability,
│                 WorkingSetCapability, AgentPoolCapability,
│                 PageGraphCapability, ResultCapability,
│                 CriticCapability, SynthesisCapability
│
└── ChangeImpactAnalysisAgent (up to 10 agents, 1 per page)
    ├── Capabilities: ChangeImpactAnalysisCapability,
    │                 MergeCapability, GroundingCapability,
    │                 HypothesisGameProtocol
    └── Bound to: exactly ONE VCM page
```

The coordinator **does not poll** worker agents. It subscribes to blackboard events (`*:cluster_analysis_complete`) and reacts when workers report results.

## How It Works

### 1. Page Sharding

Colony shards the repository into VCM pages (typically 20-40K tokens each) using `FileGrouperContextPageSource`. Related files are grouped together using import analysis, git co-change history, and semantic similarity:

```python
# From the CLI — this happens automatically when you run colony-env run
mmap_result: MmapResult = await vcm_handle.mmap_application_scope(
    scope_id="my-project",
    source_type=BuiltInContextPageSourceType.FILE_GROUPER.value,
    config=MmapConfig(),
    repo_path=repo_path,
)
```

### 2. Per-Page Impact Analysis

Each `ChangeImpactAnalysisAgent` analyzes its bound page for impact from the specified changes. The agent uses LLM reasoning to identify directly impacted components:

```python
class ChangeImpactAnalysisPolicy:
    """Uses LLM to:
    1. Understand the semantic intent of changes
    2. Identify directly impacted components
    3. Trace indirect impacts through dependencies
    4. Assess risk and severity
    5. Recommend mitigation strategies
    6. Identify required test updates
    """

    def __init__(self, agent, blackboard, max_depth=5,
                 include_tests=True, include_docs=True):
        self.agent = agent
        self.blackboard = blackboard
        self.timeline = CausalityTimeline(
            blackboard=blackboard,
            namespace="impact_analysis"
        )
```

Each agent produces a `ChangeImpactResult` (a `ScopeAwareResult[ChangeImpactReport]`) that includes impacted components, impact paths, and confidence scores.

### 3. Domain Model

The impact analysis uses a rich domain model to represent findings:

```python
class ImpactedComponent(BaseModel):
    component_id: str          # e.g., "auth.handler.validate_token"
    component_type: str        # "function", "class", "module", "test"
    file_path: str
    impact_types: list[ImpactType]   # FUNCTIONAL, SECURITY, API, ...
    severity: ImpactSeverity         # CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
    description: str
    requires_update: bool
    confidence: float
    evidence: list[str]        # grounded in source code

class ImpactPath(BaseModel):
    """A chain of impact propagation steps."""
    source_change: str         # the change that starts the chain
    steps: list[ImpactStep]    # from_component → to_component via relationship
    final_impact: ImpactedComponent
    total_severity: ImpactSeverity
```

Impact severity is an enum (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`) and impact types include `FUNCTIONAL`, `PERFORMANCE`, `SECURITY`, `API`, `DATA`, `COMPATIBILITY`, `TEST`, and `DEPLOYMENT`.

### 4. Hypothesis Game Validation

Worker agents inherit from `HypothesisGameAgent`, which means high-severity impact claims are challenged before acceptance. When an agent claims a `CRITICAL` impact, other agents can:

1. **Challenge** the claim with counter-evidence
2. **Support** the claim with corroborating evidence from their pages
3. **Refine** the severity based on broader context

This prevents false alarms — a common failure mode in automated impact analysis.

### 5. Game-Theoretic Merge

The coordinator merges results from all page agents using `ImpactMergePolicy`:

```python
class ImpactMergePolicy(MergePolicy[ChangeImpactReport]):
    async def merge(self, results, context) -> ChangeImpactResult:
        # Group impacts by (component_id, file_path)
        # If multiple agents report impact on the same component:
        #   → resolve conflict using agent weights + evidence quality
        # Merge risk assessments across all results
        # Deduplicate impact paths, breaking changes, recommendations
        ...

    async def validate(self, original, merged) -> ValidationResult:
        # Verify no changes were lost during merge
        # Check merged confidence is reasonable
        ...
```

When two agents disagree on the severity of an impact, the merge policy uses agent weights (based on evidence quality and confidence) to resolve the conflict. The merge also produces `coordination_notes` explaining how conflicts were resolved.

### 6. Final Report

The output is a `ChangeImpactReport` containing:

- **impacted_components**: All components affected, ranked by severity
- **impact_paths**: Chains showing how impact propagates through dependencies
- **risk_assessment**: Overall risk score and level
- **breaking_changes**: Identified breaking API changes
- **test_impact**: Which tests need updating
- **recommendations**: Actionable remediation steps
- **timeline**: Temporal ordering of cascading impacts via `CausalityTimeline`

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/impact-analysis.yaml \
  --verbose
```

!!! tip "Explore before running"
    Use `polymath describe impact` to see the full agent hierarchy, capabilities, and execution flow before spending API credits.

!!! tip "Budget control"
    Uncomment `budget_usd: 5.00` in the config to cap spending. The coordinator stops gracefully when the budget is reached.

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Page sharding & VCM | [Virtual Context Memory](../architecture/virtual-context-memory.md) |
| Agent capabilities & composition | [Agent System](../architecture/agent-system.md) |
| Hypothesis games | [Game Patterns](../architecture/game-engine.md) |
| ScopeAwareResult & merge policies | [Abstraction Patterns](../design-insights/abstraction-patterns.md) |
| Page graph traversal | [Page Graphs](../design-insights/page-graphs.md) |
| Cache-aware scheduling | [Cache-Aware Patterns](../philosophy/cache-awareness.md) |

## Going Further

- **Add more changes**: List additional files in the `changes` section to trace broader impact
- **Increase depth**: Raise `max_iterations` and `prefetch_depth` for deeper dependency chains
- **Attach more capabilities**: Add `ValidationCapability` or `ObjectiveGuardCapability` via `extra_capabilities`
- **Combine with other analyses**: See the [Multi-Analysis Workflow](multi-analysis.md) to run impact + compliance + intent simultaneously
