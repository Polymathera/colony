---
hide:
  - toc
---

# Examples

End-to-end examples demonstrating Colony's multi-agent analysis capabilities. Each example includes a ready-to-run YAML config, an annotated walkthrough of the agent hierarchy, and key code from the Colony codebase.

!!! tip "Prerequisites"
    All examples assume a running Colony cluster. See [Installation](../getting-started/installation.md) and the [colony-env guide](../guides/colony-env.md) for setup.

    ```bash
    colony-env up --workers 3
    colony-env run --local-repo /path/to/codebase --config examples/impact-analysis.yaml
    ```


## Code Analysis

<div class="grid cards" markdown>

-   :material-magnify-scan:{ .lg .middle } __Codebase Analysis__

    ---

    Analyze a large codebase end-to-end. The coordinator spawns ClusterAnalyzer agents that shard the repo into VCM pages, perform local analysis, resolve cross-page queries, and synthesize a structural report.

    [:octicons-arrow-right-24: Run this example](basic-analysis.md)

    `beginner` `page-sharding` `vcm` `cross-page-queries`

-   :material-source-branch:{ .lg .middle } __Change Impact Analysis__

    ---

    Trace the ripple effects of code changes across a codebase. Worker agents use multi-hop dependency propagation, hypothesis games validate critical impacts, and a game-theoretic merge produces a ranked impact report.

    [:octicons-arrow-right-24: Run this example](impact-analysis.md)

    `intermediate` `hypothesis-games` `page-graph` `merge-policies`

-   :material-file-document-check:{ .lg .middle } __Compliance Audit__

    ---

    Check license, security, and quality compliance. Agents build obligation graphs linking requirements to source evidence, detect license compatibility conflicts, and produce actionable remediation guidance.

    [:octicons-arrow-right-24: Run this example](compliance-audit.md)

    `intermediate` `obligation-graphs` `merge-policies`

-   :material-head-cog:{ .lg .middle } __Intent Inference__

    ---

    Map code to business purposes. Agents infer function-level, class-level, or module-level intent, build intent graphs, detect misalignments between stated purpose and actual behavior, and reach consensus via game protocols.

    [:octicons-arrow-right-24: Run this example](intent-inference.md)

    `intermediate` `consensus-games` `intent-graphs`

-   :material-file-document-edit:{ .lg .middle } __Contract Inference__

    ---

    Infer function contracts — preconditions, postconditions, and invariants — at configurable formalism levels. Hypothesis games challenge and validate each contract before acceptance.

    [:octicons-arrow-right-24: Run this example](contract-inference.md)

    `intermediate` `hypothesis-games` `formal-specs`

-   :material-content-cut:{ .lg .middle } __Program Slicing__

    ---

    Extract the minimal code subset affecting a target variable. Supports backward, forward, chopping, and conditioned slices with LLM-based interprocedural dependency reasoning across page boundaries.

    [:octicons-arrow-right-24: Run this example](program-slicing.md)

    `intermediate` `slicing` `interprocedural`

</div>

## Advanced Workflows

<div class="grid cards" markdown>

-   :material-layers-triple:{ .lg .middle } __Multi-Analysis Workflow__

    ---

    Run impact analysis, compliance audit, and intent inference **simultaneously** on the same paged codebase. All agent teams share the VCM and page graph — pages loaded for one analysis are reusable by others via the shared KV cache.

    [:octicons-arrow-right-24: Run this example](multi-analysis.md)

    `advanced` `concurrent-teams` `shared-vcm` `cache-reuse`

</div>

## Running Any Example

Every example follows the same workflow:

=== "Generate a config"

    ```bash
    # Start from the built-in template
    polymath init-config --output my_analysis.yaml

    # Or use one of the example configs directly
    cp examples/impact-analysis.yaml my_analysis.yaml
    ```

=== "Start the cluster"

    ```bash
    colony-env up --workers 3
    ```

=== "Run the analysis"

    ```bash
    colony-env run \
      --local-repo /path/to/codebase \
      --config my_analysis.yaml \
      --verbose
    ```

=== "Inspect results"

    ```bash
    # Results are saved to the output directory
    ls ./results/

    # Or view the web dashboard
    colony-env dashboard
    ```

!!! info "Explore the CLI"
    Use `polymath list analyses` to see all available analysis types, `polymath list capabilities` to see attachable capabilities, and `polymath describe impact` to see the full agent hierarchy for a specific analysis.
