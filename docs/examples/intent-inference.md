# Intent Inference

Map code to business purposes. Agents infer the intent behind each function, class, or module, build intent graphs, detect misalignments between stated purpose and actual behavior, and reach consensus via game protocols.

## What You'll Learn

- How Colony infers **business-level purpose** behind code (not just what code does, but *why*)
- How **consensus games** resolve disagreements between agents about intent
- How **intent graphs** map hierarchical relationships between code purposes
- The difference between `ALIGNED`, `MISALIGNED`, and `PARTIALLY_ALIGNED` intent

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/intent-inference.yaml`](https://github.com/polymathera/colony/blob/main/examples/intent-inference.yaml)

```yaml
analyses:
  - type: intent
    coordinator_version: v2
    max_agents: 8

    # "function" = per-function intent
    # "class"    = per-class intent
    # "module"   = per-module intent
    granularity: "function"
```

## Agent Hierarchy

```
IntentInferenceCoordinator (1 agent)
├── Capabilities: IntentAnalysisCapability, MergeCapability,
│                 SynthesisCapability, WorkingSetCapability,
│                 AgentPoolCapability
│
└── IntentInferenceAgent (up to 8 agents, 1 per page)
    ├── Capabilities: IntentInferenceCapability, MergeCapability,
    │                 ConsensusGameProtocol
    └── Bound to: exactly ONE VCM page
```

Note that worker agents include `ConsensusGameProtocol` — when agents disagree about the intent of a code region, they enter a consensus game to resolve the conflict rather than simply averaging confidence scores.

## Domain Model

```python
class IntentCategory(str, Enum):
    BUSINESS_LOGIC = "business_logic"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SECURITY = "security"
    INTEGRATION = "integration"
    USER_INTERFACE = "user_interface"
    INFRASTRUCTURE = "infrastructure"
    TESTING = "testing"
    UTILITY = "utility"

class IntentAlignment(str, Enum):
    ALIGNED = "aligned"                # code does what it's supposed to
    MISALIGNED = "misaligned"          # code diverges from stated intent
    PARTIALLY_ALIGNED = "partially_aligned"
    UNCLEAR = "unclear"                # intent cannot be determined

class CodeIntent(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    primary_intent: str              # natural language description
    secondary_intents: list[str]
    categories: list[IntentCategory]
    business_goals: list[str]        # what business objective this serves
    alignment: IntentAlignment
    issues: list[str]                # misalignment issues found
    confidence: float
    evidence: list[str]
```

The coordinator builds an `IntentGraph` with nodes (code intents), edges (relationships between intents), hierarchies (module → class → function), and detected conflicts.

## How It Works

1. **Per-page inference**: Each agent infers the intent of code in its bound page at the specified granularity
2. **Consensus games**: When multiple agents analyze code that spans page boundaries, they use `ConsensusGameProtocol` to agree on intent
3. **Cross-page hierarchies**: The coordinator builds hierarchical intent maps (module → class → function)
4. **Misalignment detection**: Agents flag code where actual behavior diverges from inferred intent

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/intent-inference.yaml \
  --verbose
```

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Consensus games | [Game Patterns](../architecture/game-engine.md) |
| Qualitative analysis | [Qualitative Analysis](../design-insights/qualitative-analysis.md) |
| Agent capabilities | [Capabilities as AOP Aspects](../design-insights/capabilities-as-aspects.md) |

## Going Further

- **Change granularity**: Try `"class"` or `"module"` for higher-level intent maps
- **Combine with impact**: Intent inference + change impact analysis reveals whether changes align with business goals — see [Multi-Analysis Workflow](multi-analysis.md)
