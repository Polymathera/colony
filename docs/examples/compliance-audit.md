# Compliance Audit

Check license, security, and quality compliance across a codebase. Agents build obligation graphs linking requirements to source evidence, detect license compatibility conflicts, and produce actionable remediation guidance.

## What You'll Learn

- How Colony agents check multiple compliance dimensions simultaneously
- How **obligation graphs** link requirements to source evidence
- How **merge policies** reconcile findings from different page agents
- The `ComplianceReport` domain model with violations, licenses, and risk assessment

## Prerequisites

- A running Colony cluster (`colony-env up --workers 3`)
- An Anthropic API key (`ANTHROPIC_API_KEY` environment variable)
- A git repository to analyze

## Configuration

Download: [`examples/compliance-audit.yaml`](https://github.com/polymathera/colony/blob/main/examples/compliance-audit.yaml)

```yaml
analyses:
  - type: compliance
    coordinator_version: v2
    max_agents: 8

    # What to check — pick from:
    #   license, regulatory, security, organizational,
    #   industry, architectural, quality
    compliance_types:
      - license
      - security
      - quality
```

## Agent Hierarchy

```
ComplianceAnalysisCoordinator (1 agent)
├── Capabilities: ComplianceVCMCapability, MergeCapability,
│                 WorkingSetCapability, AgentPoolCapability
│
└── ComplianceAnalysisAgent (up to 8 agents, 1 per page)
    ├── Capabilities: ComplianceAnalysisCapability, MergeCapability
    └── Bound to: exactly ONE VCM page
```

## Domain Model

```python
class ComplianceType(str, Enum):
    LICENSE = "license"             # License obligations and conflicts
    REGULATORY = "regulatory"       # Regulatory requirements (GDPR, HIPAA, ...)
    SECURITY = "security"           # Security policies and vulnerabilities
    ORGANIZATIONAL = "organizational"  # Internal coding standards
    INDUSTRY = "industry"           # Industry-specific requirements
    ARCHITECTURAL = "architectural" # Architecture decision compliance
    QUALITY = "quality"             # Code quality standards

class ComplianceViolation(BaseModel):
    violation_id: str
    type: ComplianceType
    severity: ComplianceSeverity    # CRITICAL, HIGH, MEDIUM, LOW, INFO
    description: str
    location: str                   # file:line
    rule: str                       # which rule was violated
    evidence: list[str]             # grounded in source code
    remediation: str                # actionable fix
    risk: str
    confidence: float

class License(BaseModel):
    name: str
    spdx_id: str                    # e.g., "MIT", "GPL-3.0-only"
    category: str                   # "permissive", "copyleft", "proprietary"
    permissions: list[str]
    conditions: list[str]
    limitations: list[str]
    compatible_with: list[str]
    incompatible_with: list[str]
```

The final `ComplianceReport` includes: violations, requirements checked, licenses found, license conflicts, risk assessment, and recommendations.

## How It Works

1. **Per-page analysis**: Each `ComplianceAnalysisAgent` scans its bound page for license headers, security patterns, and quality issues
2. **Obligation graph construction**: The coordinator builds a graph linking requirements to the source evidence that satisfies (or violates) them
3. **License conflict detection**: Cross-page license analysis identifies incompatible license combinations (e.g., MIT code linking to GPL-only dependencies)
4. **Result merge**: `ComplianceMergePolicy` reconciles findings, deduplicates violations, and produces a unified report

## Running the Example

```bash
colony-env run \
  --local-repo /path/to/your/codebase \
  --config examples/compliance-audit.yaml \
  --verbose
```

## Key Concepts

| Concept | Where to learn more |
|---------|-------------------|
| Merge policies | [Abstraction Patterns](../design-insights/abstraction-patterns.md) |
| ScopeAwareResult | [Abstraction Patterns](../design-insights/abstraction-patterns.md) |
| Agent capabilities | [Agent System](../architecture/agent-system.md) |
| Qualitative LLM analysis | [Qualitative Analysis](../design-insights/qualitative-analysis.md) |

## Going Further

- **Add regulatory checks**: Include `regulatory` in `compliance_types` for GDPR/HIPAA analysis
- **Combine with impact**: Run compliance alongside change impact analysis to catch compliance regressions — see [Multi-Analysis Workflow](multi-analysis.md)
