# Contract Inference

Infer function contracts — preconditions, postconditions, and invariants — using qualitative LLM reasoning. Hypothesis games challenge and validate each contract before acceptance. Specifications can be generated at four formalism levels, from natural language to executable assertions.

## What You'll Learn

- How Colony infers **formal-ish specifications** from code without running tests
- How **hypothesis games** validate inferred contracts — agents challenge each other's claims
- The four **formalism levels**: natural → semi-formal → formal → code
- How `FunctionContract` captures preconditions, postconditions, invariants, purity, and termination

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/contract-inference.yaml`](https://github.com/polymathera/colony/blob/main/examples/contract-inference.yaml)

```yaml
analyses:
  - type: contracts
    coordinator_version: v2
    max_agents: 8

    # Formalism level for generated specs:
    #   "natural"     → English descriptions
    #   "semi_formal" → structured but not machine-checkable
    #   "formal"      → machine-checkable specifications
    #   "code"        → executable assertions (assert statements)
    formalism: "semi_formal"
```

## Agent Hierarchy

```
ContractInferenceCoordinator (1 agent)
├── Capabilities: ContractAnalysisCapability, MergeCapability,
│                 SynthesisCapability, WorkingSetCapability,
│                 AgentPoolCapability
│
└── ContractInferenceAgent (up to 8 agents, 1 per page)
    ├── Capabilities: ContractInferenceCapability, MergeCapability,
    │                 HypothesisGameProtocol
    └── Bound to: exactly ONE VCM page
```

Worker agents include `HypothesisGameProtocol`. When an agent infers a contract, it proposes it as a hypothesis. Other agents can challenge with counterexamples from their pages, leading to refined, higher-confidence contracts.

## Domain Model

```python
class ContractType(str, Enum):
    PRECONDITION = "precondition"    # must hold before call
    POSTCONDITION = "postcondition"  # must hold after call
    INVARIANT = "invariant"          # must hold throughout
    ASSERTION = "assertion"          # must hold at specific point
    ASSUMPTION = "assumption"        # assumed to hold (not verified)

class FormalismLevel(str, Enum):
    NATURAL = "natural"              # "x must be positive"
    SEMI_FORMAL = "semi_formal"      # "x > 0, returns non-null"
    FORMAL = "formal"                # "∀x ∈ ℤ⁺: f(x) > 0"
    CODE = "code"                    # "assert x > 0"

class Contract(BaseModel):
    contract_type: ContractType
    description: str
    formal_spec: str | None          # at requested formalism level
    variables: list[str]
    confidence: float
    counterexamples: list[str]       # found during hypothesis games

class FunctionContract(BaseModel):
    function_name: str
    file_path: str
    line_number: int
    preconditions: list[Contract]
    postconditions: list[Contract]
    invariants: list[Contract]
    modifies: list[str]              # what state is mutated
    pure: bool                       # no side effects?
    termination: str | None          # termination argument
    complexity: str | None           # inferred complexity class
    formalism: FormalismLevel
```

## How It Works

1. **Per-page inference**: Each agent analyzes functions in its page, inferring contracts from code patterns, naming conventions, error handling, and type annotations
2. **Hypothesis validation**: Inferred contracts are proposed as hypotheses and challenged by other agents who may find contradicting patterns in their pages
3. **Cross-page merging**: The coordinator merges contracts, resolving conflicts where the same function appears across pages
4. **Security contract extraction**: The coordinator identifies security-critical contracts (authentication, authorization, input validation)

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/contract-inference.yaml \
  --verbose
```

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Hypothesis games | [Game Patterns](../architecture/game-engine.md) |
| ScopeAwareResult | [Abstraction Patterns](../design-insights/abstraction-patterns.md) |
| Qualitative analysis | [Qualitative Analysis](../design-insights/qualitative-analysis.md) |

## Going Further

- **Increase formalism**: Set `formalism: "code"` to generate executable `assert` statements
- **Validate critical paths**: Add `ValidationCapability` to `extra_capabilities` for multi-level validation of security-critical contracts
