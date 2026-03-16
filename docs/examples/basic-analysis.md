# Codebase Analysis

The simplest Colony example. Analyze a codebase end-to-end using the three-tier agent hierarchy: a coordinator spawns ClusterAnalyzer agents that shard pages into clusters, perform local analysis, resolve cross-page queries, and synthesize a structural report.

## What You'll Learn

- The three-tier agent hierarchy: **Coordinator** → **ClusterAnalyzer** → **PageAnalyzer**
- How V2's LLM-driven reasoning loop (PLAN → ACT → REFLECT → CRITIQUE → ADAPT) replaces hardcoded FSMs
- How **cross-page queries** let agents answer questions that span page boundaries
- How **page keys** (compact summaries) enable efficient attention routing

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/basic-analysis.yaml`](https://github.com/polymathera/colony/blob/main/examples/basic-analysis.yaml)

```yaml
analyses:
  - type: basic
    coordinator_version: v2       # LLM-driven reasoning loop
    max_agents: 10
    quality_threshold: 0.7
    max_iterations: 10
    batching_policy: hybrid       # balance cache reuse + coverage

hierarchy:
  extra_capabilities:
    - ReflectionCapability
```

This is the minimal analysis config — no domain-specific parameters needed.

## Agent Hierarchy

```
CodeAnalysisCoordinatorV2 (1 agent)
├── Capabilities: CodeAnalysisCoordinatorCapability, CriticCapability
│
└── ClusterAnalyzer (up to 10 agents, 1 per cluster)
    ├── Capabilities: ClusterAnalyzerCapabilityV2
    ├── Manages: multiple VCM pages per cluster
    │
    └── [spawns internally] PageAnalyzer (1 per page)
        └── Capabilities: PageAnalyzerCapability
            Bound to: exactly ONE VCM page
```

### Why Three Tiers?

Each tier serves a distinct purpose:

- **PageAnalyzer** — atomic leaf agent. Produces a compact 1-2KB summary ("key") per page. Simple linear workflow: load → analyze → write → stop. No reasoning loop needed.

- **ClusterAnalyzer V2** — multi-page reasoning. Manages a cluster of related pages. Uses an LLM-driven reasoning loop to iteratively analyze pages, generate cross-page queries, route them, and synthesize findings.

- **Coordinator** — orchestration. Spawns ClusterAnalyzers, monitors progress via blackboard events (no polling), and synthesizes a global report from cluster results.

## How It Works

### PageAnalyzer: Compact Summaries as Attention Keys

Each page gets a `PageAnalyzer` that produces a structured summary serving as a "key" in the attention mechanism:

```python
class PageAnalyzerCapability(AgentCapability):
    """Bound to exactly ONE page. Produces compact summary (1-2KB)
    that serves as the 'key' in key-query-value attention.
    Linear: load page → analyze → write result → stop."""

    @action_executor()
    async def analyze_page(self) -> ScopeAwareResult[dict]:
        """Analyze the single bound page and produce compact summary."""
        ...
```

These summaries enable efficient query routing — when a ClusterAnalyzer generates a query, it can find relevant pages by matching against keys rather than loading full page content.

### ClusterAnalyzer V2: Iterative Reasoning

The V2 analyzer replaces the V1 FSM (hardcoded phases) with an LLM-driven reasoning loop:

```python
class ClusterAnalyzerCapabilityV2(AgentCapability):
    """LLM-driven reasoning loop replaces FSM:
    PLAN → ACT → REFLECT → CRITIQUE → ADAPT

    Agent reasons about what to analyze next, generates cross-page
    queries, routes to relevant pages, and iterates until quality
    threshold is met."""

    def get_action_group_description(self) -> str:
        return (
            "Cluster Analysis V2 (Iterative reasoning loop) — "
            "PLAN → ACT → REFLECT → CRITIQUE → ADAPT cycle. "
            "Agent reasons about what to analyze next, generates "
            "cross-page queries, routes to relevant pages, and "
            "iterates until quality threshold is met."
        )
```

The LLM decides what to analyze next, which queries to generate, and when the analysis is complete — rather than following a fixed sequence.

### Coordinator: Event-Driven Monitoring

The coordinator subscribes to blackboard events instead of polling:

```python
class BaseCodeAnalysisCoordinatorCapability(AgentCapability, ABC):

    @event_handler(pattern="*:cluster_analysis_complete")
    async def on_child_complete(self, event, repl):
        """Monitor child agent completion via blackboard events."""
        ...
```

### Configuration Models

All parameters are configurable via Pydantic models:

```python
class ClusterAnalyzerConfig(BaseModel):
    quality_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    max_pages_per_iteration: int = Field(default=5, ge=1, le=20)
    attention_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    num_tokens_context: int = Field(default=8192, ge=1024, le=32768)

class CoordinatorConfig(BaseModel):
    max_cluster_size: int = Field(default=10, ge=2, le=50)
    min_cluster_size: int = Field(default=2, ge=1, le=10)
    monitor_interval_seconds: float = Field(default=5.0, ge=0.5, le=60.0)
```

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/basic-analysis.yaml \
  --verbose
```

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Three-tier agent hierarchy | [Agent System](../architecture/agent-system.md) |
| Page sharding & VCM | [Virtual Context Memory](../architecture/virtual-context-memory.md) |
| Reasoning loops vs FSMs | [Action Policies](../architecture/action-policies.md) |
| Page keys & attention routing | [Page Graphs](../design-insights/page-graphs.md) |
| Event-driven blackboard | [Blackboard Pattern](../architecture/blackboard.md) |

## Going Further

- **Switch to V1**: Set `coordinator_version: v1` to see the FSM-based approach and compare
- **Attach game protocols**: Add `HypothesisGameProtocol` to `extra_capabilities` for claim validation
- **Add domain-specific analyses**: See [Change Impact Analysis](impact-analysis.md) or [Compliance Audit](compliance-audit.md) for richer examples
