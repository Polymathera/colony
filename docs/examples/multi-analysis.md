# Multi-Analysis Workflow

Run multiple analyses simultaneously on the same paged codebase. This example demonstrates Colony's most powerful operational mode: concurrent agent teams sharing the same VCM, page graph, and KV cache — pages loaded for one analysis are reusable by others.

## What You'll Learn

- How Colony runs **multiple coordinator agents concurrently** on the same codebase
- How the **shared VCM** eliminates redundant page loading across analyses
- How **cache-aware scheduling** ensures agent teams benefit from each other's page loads
- How to compose analyses for a comprehensive codebase audit

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/multi-analysis.yaml`](https://github.com/polymathera/colony/blob/main/examples/multi-analysis.yaml)

```yaml
# Three analyses run concurrently on the same paged codebase.
analyses:

  # 1. What breaks when we change the auth flow?
  - type: impact
    coordinator_version: v2
    max_agents: 10
    batching_policy: hybrid
    changes:
      - file_path: "src/auth/handler.py"
        change_type: modification
        description: "Migrated from JWT to session tokens"

  # 2. Are we violating any license or security policies?
  - type: compliance
    coordinator_version: v2
    max_agents: 8
    compliance_types:
      - license
      - security

  # 3. What is each function actually trying to do?
  - type: intent
    coordinator_version: v2
    max_agents: 8
    granularity: "function"
```

## Agent Hierarchy

Three independent coordinator-worker teams operate concurrently:

```
Session
├── ChangeImpactAnalysisCoordinator ──→ up to 10 ChangeImpactAnalysisAgent
├── ComplianceAnalysisCoordinator   ──→ up to 8  ComplianceAnalysisAgent
└── IntentInferenceCoordinator      ──→ up to 8  IntentInferenceAgent
                                          │
                            All share the same VCM
                            and page graph instance
```

## Why This Matters: Shared VCM

In a naive system, each analysis would independently load every page it needs into the LLM context — tripling the cost for three analyses. Colony's VCM changes this:

1. **Shared page table**: All agents reference the same virtual page table. When the impact analysis loads `src/auth/handler.py` into the KV cache, the compliance and intent agents can reuse that cached page at zero additional cost.

2. **Page graph reuse**: The page graph (which pages are related to which) is built once and shared. All three analyses benefit from the same semantic relationship structure.

3. **Cache-aware batching**: The `batching_policy: hybrid` setting ensures that when multiple analyses need overlapping pages, they are scheduled together to maximize cache hits.

This is analogous to how an operating system's shared page cache benefits all processes reading the same files — Colony applies the same principle at the LLM inference level.

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/multi-analysis.yaml \
  --verbose
```

!!! tip "Budget control"
    Multi-analysis workflows consume more API credits. Set `budget_usd: 15.00` to cap total spending across all analyses.

## How Results Are Organized

Each analysis produces an independent report. Results are saved to the output directory:

```
./results/
├── impact_<session_id>.json       # Change impact report
├── compliance_<session_id>.json   # Compliance audit report
└── intent_<session_id>.json       # Intent inference report
```

Colony also provides cost tracking per analysis type, so you can see how much each analysis contributed to total spending.

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Virtual Context Memory | [Virtual Context Memory](../architecture/virtual-context-memory.md) |
| Cache-aware scheduling | [Cache-Aware Patterns](../philosophy/cache-awareness.md) |
| Page graph sharing | [Page Graphs](../design-insights/page-graphs.md) |
| Individual analyses | [Impact](impact-analysis.md), [Compliance](compliance-audit.md), [Intent](intent-inference.md) |

## Going Further

- **Add more analyses**: Include `contracts` or `slicing` for a 5-way concurrent analysis
- **Custom capabilities**: Attach `ObjectiveGuardCapability` to prevent goal drift in long-running multi-analysis sessions
- **Budget partitioning**: Use separate sessions per analysis if you want independent budget tracking
