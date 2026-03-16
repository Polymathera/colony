# Program Slicing

Extract the minimal code subset affecting a target variable or expression. Colony distributes slicing across pages and resolves interprocedural dependencies that span page boundaries — something traditional slicers cannot do at this scale with semantic understanding.

## What You'll Learn

- How Colony performs **LLM-based program slicing** (semantic, not just syntactic)
- How **backward and forward slices** trace data and control dependencies
- How **interprocedural resolution** stitches together slices across page boundaries
- The `ProgramSlice` domain model with dependency edges and reasoning traces

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/program-slicing.yaml`](https://github.com/polymathera/colony/blob/main/examples/program-slicing.yaml)

```yaml
analyses:
  - type: slicing
    coordinator_version: v2
    max_agents: 8

    # What to slice — specify the variable and direction.
    slice_criteria:
      - file_path: "src/main.py"
        line: 42
        variable: "user_token"
        slice_type: "backward"    # trace what affects user_token
```

## Agent Hierarchy

```
ProgramSlicingCoordinator (1 agent)
├── Capabilities: SlicingAnalysisCapability, MergeCapability,
│                 WorkingSetCapability, AgentPoolCapability
│
└── ProgramSlicingAgent (up to 8 agents, 1 per page)
    ├── Capabilities: ProgramSlicingCapability, MergeCapability
    └── Bound to: exactly ONE VCM page
```

## Domain Model

```python
class SliceType(str, Enum):
    BACKWARD = "backward"        # what affects the target?
    FORWARD = "forward"          # what does the target affect?
    CHOPPING = "chopping"        # statements between two points
    DYNAMIC = "dynamic"          # runtime-specific slice
    CONDITIONED = "conditioned"  # slice under specific conditions

class SliceCriterion(BaseModel):
    file_path: str
    line_number: int
    variable: str | None         # target variable name
    expression: str | None       # or target expression
    slice_type: SliceType

class DependencyEdge(BaseModel):
    from_line: int
    to_line: int
    dep_type: str                # "data", "control", "call", ...
    variable: str | None
    condition: str | None        # under what condition
    confidence: float

class ProgramSlice(BaseModel):
    criterion: SliceCriterion
    included_lines: list[int]    # lines in the slice
    excluded_lines: list[int]    # lines explicitly excluded
    dependencies: list[DependencyEdge]
    entry_points: list[str]      # functions that enter the slice
    exit_points: list[str]       # functions that leave the slice
    interprocedural: bool        # does slice cross function boundaries?
    reasoning: list[str]         # LLM reasoning trace for each inclusion
```

## How It Works

1. **Per-page slicing**: Each agent computes a local slice for its page, using LLM reasoning to identify data and control dependencies
2. **Interprocedural resolution**: The coordinator stitches together partial slices from different pages — when a function call crosses a page boundary, the coordinator connects the caller's slice to the callee's slice
3. **External dependency tracking**: Dependencies on code outside the analyzed pages are tracked separately
4. **Result merge**: Partial slices are merged into a complete, minimal slice

The key advantage over traditional slicers: Colony's LLM-based approach understands **semantic dependencies** (e.g., "this string formatting call affects the SQL query because it constructs the WHERE clause") that syntactic analysis would miss.

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/program-slicing.yaml \
  --verbose
```

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Qualitative analysis | [Qualitative Analysis](../design-insights/qualitative-analysis.md) |
| Merge policies | [Abstraction Patterns](../design-insights/abstraction-patterns.md) |
| Page graph traversal | [Page Graphs](../design-insights/page-graphs.md) |

## Going Further

- **Forward slicing**: Change `slice_type` to `"forward"` to trace what a variable *affects* downstream
- **Multiple criteria**: Add more entries to `slice_criteria` to compute slices for several variables simultaneously
- **Chopping**: Use `"chopping"` to find statements between two points — useful for understanding data flow between an input and an output
