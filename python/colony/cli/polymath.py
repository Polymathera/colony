#!/usr/bin/env python3
"""Polymath: Integration test CLI for the Polymathera multi-agent framework.

This tool runs a complete analysis workflow on a codebase using the Colony
multi-agent framework. It demonstrates:

1. **Context Paging** — Using VirtualContextManager.mmap_application_scope with
   FileGrouperContextPageSource to page a codebase into VCM pages.

2. **Agent Hierarchies** — Spawning coordinator agents that in turn spawn
   teams of specialized page-level analysis agents, creating rich parent-child
   hierarchies with cache-aware scheduling.

3. **Analysis Workflows** — Running one or more analysis types (impact analysis,
   program slicing, compliance checking, intent inference, contract inference)
   each orchestrated by its own coordinator agent with domain-specific
   capabilities, merge policies, and game-theoretic validation.

4. **Capability Composition** — Agents are composed from reusable capabilities
   (WorkingSetCapability, AgentPoolCapability, PageGraphCapability,
   MergeCapability, CriticCapability, HypothesisGameProtocol, etc.) that
   provide @action_executor methods and @event_handler hooks.

Prerequisites:
    - A running Ray cluster (local or AWS) with the Polymathera application deployed
    - The target codebase available on the cluster filesystem (e.g., via EFS or
      shared Docker volume)
    - Python dependencies: typer, rich, pyyaml

Usage:
    # Quick start — run impact analysis on a local codebase
    polymath run /path/to/codebase --analysis impact

    # Run multiple analyses with a YAML config
    polymath run /path/to/codebase --config my_test.yaml

    # List available analyses, agents, and capabilities
    polymath list analyses
    polymath list agents
    polymath list capabilities

    # Generate a sample YAML config
    polymath init-config --output my_test.yaml

Example YAML config (my_test.yaml):
    repo_id: "my-project"
    tenant_id: "test-tenant"
    session_id: "integration-test-001"

    paging:
      flush_threshold: 20
      flush_token_budget: 4096
      pinned: false

    analyses:
      - type: impact
        coordinator_version: v2       # v1 = cache-oblivious, v2 = cache-aware
        max_agents: 10
        quality_threshold: 0.7
        max_iterations: 3
        batching_policy: hybrid
        changes:
          - file_path: "src/main.py"
            change_type: modification
            description: "Refactored authentication flow"

      - type: compliance
        coordinator_version: v2
        max_agents: 8
        compliance_types:
          - license
          - security

    hierarchy:
      # Optional: override default coordinator/worker agent classes
      coordinator_class: null   # uses analysis-specific default
      worker_class: null        # uses analysis-specific default
      extra_capabilities:
        - ReflectionCapability
        - ConsciousnessCapability
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# ---------------------------------------------------------------------------
# YAML import (optional dependency with graceful fallback)
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

console = Console()
err_console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("polymath")


# ===========================================================================
# Registry of available analyses, agents, and capabilities
# ===========================================================================

class AnalysisType(str, Enum):
    """Supported analysis types."""
    IMPACT = "impact"
    SLICING = "slicing"
    COMPLIANCE = "compliance"
    INTENT = "intent"
    CONTRACTS = "contracts"
    BASIC = "basic"  # General code structure analysis


# Maps analysis types to their coordinator and worker agent class paths.
# These are the fully-qualified Python class paths used by the agent system
# to instantiate agent classes dynamically.
ANALYSIS_REGISTRY: dict[str, dict[str, Any]] = {
    "impact": {
        "label": "Change Impact Analysis",
        "description": (
            "Analyzes the ripple effects of code changes across a codebase. "
            "Uses multi-hop dependency propagation, hypothesis games for "
            "validating critical impacts, and game-theoretic merge policies."
        ),
        "coordinator_v1": "colony.samples.code_analysis.impact.coordinator.ChangeImpactAnalysisCoordinator",
        "coordinator_v2": "colony.samples.code_analysis.impact.coordinator.ChangeImpactAnalysisCoordinator",
        "worker": "colony.samples.code_analysis.impact.page_analyzer.ChangeImpactAnalysisAgent",
        "coordinator_capabilities": [
            "ChangeImpactAnalysisCoordinatorCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
            "PageGraphCapability",
            "ResultCapability",
            "CriticCapability",
            "SynthesisCapability",
            "HypothesisGameProtocol",
        ],
        "worker_capabilities": [
            "ChangeImpactAnalysisCapability",
            "MergeCapability",
            "GroundingCapability",
        ],
        "extra_metadata_keys": ["changes", "change_description"],
    },
    "slicing": {
        "label": "Program Slicing",
        "description": (
            "Extracts the minimal code subset affecting a target variable or "
            "expression. Supports backward, forward, chopping, dynamic, and "
            "conditioned slices with LLM-based dependency reasoning."
        ),
        "coordinator_v1": "colony.samples.code_analysis.slicing.agents.ProgramSlicingCoordinator",
        "coordinator_v2": "colony.samples.code_analysis.slicing.agents.ProgramSlicingCoordinator",
        "worker": "colony.samples.code_analysis.slicing.agents.ProgramSlicingAgent",
        "coordinator_capabilities": [
            "SlicingAnalysisCapability",
            "MergeCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
        ],
        "worker_capabilities": [
            "ProgramSlicingCapability",
            "MergeCapability",
        ],
        "extra_metadata_keys": ["slice_criteria"],
    },
    "compliance": {
        "label": "Compliance Analysis",
        "description": (
            "Checks license, regulatory, security, and organizational compliance. "
            "Uses LLM-based semantic understanding of license terms and builds "
            "obligation graphs for tracking compliance requirements."
        ),
        "coordinator_v1": "colony.samples.code_analysis.compliance.agents.ComplianceAnalysisCoordinator",
        "coordinator_v2": "colony.samples.code_analysis.compliance.agents.ComplianceAnalysisCoordinator",
        "worker": "colony.samples.code_analysis.compliance.agents.ComplianceAnalysisAgent",
        "coordinator_capabilities": [
            "ComplianceVCMCapability",
            "MergeCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
        ],
        "worker_capabilities": [
            "ComplianceAnalysisCapability",
            "MergeCapability",
        ],
        "extra_metadata_keys": ["compliance_types"],
    },
    "intent": {
        "label": "Intent Inference",
        "description": (
            "Infers code purpose and developer intentions — business goals vs. "
            "implementation details. Builds intent graphs and detects "
            "misalignments using consensus game protocols."
        ),
        "coordinator_v1": "colony.samples.code_analysis.intent.agents.IntentInferenceCoordinator",
        "coordinator_v2": "colony.samples.code_analysis.intent.agents.IntentInferenceCoordinator",
        "worker": "colony.samples.code_analysis.intent.agents.IntentInferenceAgent",
        "coordinator_capabilities": [
            "IntentAnalysisCapability",
            "MergeCapability",
            "SynthesisCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
        ],
        "worker_capabilities": [
            "IntentInferenceCapability",
            "MergeCapability",
            "ConsensusGameProtocol",
        ],
        "extra_metadata_keys": ["granularity"],
    },
    "contracts": {
        "label": "Contract Inference",
        "description": (
            "Infers function contracts — preconditions, postconditions, and "
            "invariants. Uses hypothesis games to validate contracts and "
            "produces specifications at configurable formalism levels."
        ),
        "coordinator_v1": "colony.samples.code_analysis.contracts.agents.ContractInferenceCoordinator",
        "coordinator_v2": "colony.samples.code_analysis.contracts.agents.ContractInferenceCoordinator",
        "worker": "colony.samples.code_analysis.contracts.agents.ContractInferenceAgent",
        "coordinator_capabilities": [
            "ContractAnalysisCapability",
            "MergeCapability",
            "SynthesisCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
        ],
        "worker_capabilities": [
            "ContractInferenceCapability",
            "MergeCapability",
            "HypothesisGameProtocol",
        ],
        "extra_metadata_keys": ["formalism"],
    },
    "basic": {
        "label": "Basic Code Analysis",
        "description": (
            "General-purpose code structure analysis. The coordinator spawns "
            "ClusterAnalyzer agents that perform key generation, local page "
            "analysis, cross-page query resolution, and cluster-level synthesis."
        ),
        "coordinator_v1": "colony.samples.code_analysis.basic.coordinator.CodeAnalysisCoordinator",
        "coordinator_v2": "colony.samples.code_analysis.basic.coordinator.CodeAnalysisCoordinatorV2",
        "worker": "colony.samples.code_analysis.basic.cluster_analyzer.ClusterAnalyzer",
        "coordinator_capabilities": [
            "CodeAnalysisCoordinatorCapability",
            "CriticCapability",
        ],
        "worker_capabilities": [
            "ClusterAnalyzerCapability",
        ],
        "extra_metadata_keys": [],
    },
}


# Capabilities that can be attached to any agent for cross-cutting concerns.
EXTRA_CAPABILITIES_REGISTRY: dict[str, dict[str, str]] = {
    "ReflectionCapability": {
        "path": "colony.agents.patterns.capabilities.reflection.ReflectionCapability",
        "description": "Enables agent self-reflection on actions with SystemDocumentation support.",
    },
    "ConsciousnessCapability": {
        "path": "colony.agents.patterns.capabilities.consciousness.ConsciousnessCapability",
        "description": "Provides self-awareness via AgentSelfConcept and SystemDocumentation.",
    },
    "ReputationCapability": {
        "path": "colony.agents.patterns.capabilities.reputation.ReputationCapability",
        "description": "No-regret learning using multiplicative weights for agent reputation.",
    },
    "ValidationCapability": {
        "path": "colony.agents.patterns.capabilities.validation.ValidationCapability",
        "description": "Multi-level validation (cross-shard, evidence-based, consensus).",
    },
    "GroundingCapability": {
        "path": "colony.agents.patterns.capabilities.grounding.GroundingCapability",
        "description": "Validates claims against evidence with query expansion.",
    },
    "ObjectiveGuardCapability": {
        "path": "colony.agents.patterns.capabilities.goal_alignment.ObjectiveGuardCapability",
        "description": "Prevents goal drift by monitoring outputs against original goals.",
    },
    "ConsistencyCapability": {
        "path": "colony.agents.patterns.capabilities.consistency.ConsistencyCapability",
        "description": "Detects contradictions using the epistemic layer.",
    },
    "HypothesisTrackingCapability": {
        "path": "colony.agents.patterns.games.hypothesis.tracking.HypothesisTrackingCapability",
        "description": "Tracks hypotheses across agent interactions.",
    },
    "WorkingMemoryCapability": {
        "path": "colony.agents.patterns.memory.working.WorkingMemoryCapability",
        "description": "Token-bounded working memory with compaction.",
    },
    "SessionMemoryCapability": {
        "path": "colony.agents.patterns.memory.session_memory.SessionMemoryCapability",
        "description": "Session-level memory tracking for cross-session continuity.",
    },
}


# ===========================================================================
# Configuration models
# ===========================================================================

@dataclass
class PagingConfig:
    """Configuration for VCM paging of the codebase."""
    flush_threshold: int = 20
    flush_token_budget: int = 4096
    flush_interval_seconds: float = 60.0
    pinned: bool = False
    locality_policy_type: str = "tag"
    flush_policy_type: str = "threshold"


@dataclass
class AnalysisConfig:
    """Configuration for a single analysis run."""
    type: str = "basic"
    coordinator_version: str = "v2"
    max_agents: int = 10
    quality_threshold: float = 0.7
    max_iterations: int = 3
    batching_policy: str = "hybrid"
    overlap_threshold: float = 0.3
    batch_size: int = 5
    prefetch_depth: int = 2
    extra_capabilities: list[str] = field(default_factory=list)
    # Analysis-specific parameters (e.g., changes for impact, criteria for slicing)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyConfig:
    """Configuration for agent hierarchy overrides."""
    coordinator_class: str | None = None
    worker_class: str | None = None
    extra_capabilities: list[str] = field(default_factory=list)


@dataclass
class TestConfig:
    """Complete integration test configuration."""
    repo_id: str = "polymath-test"
    tenant_id: str = "test-tenant"
    session_id: str = field(default_factory=lambda: f"polymath-{uuid.uuid4().hex[:8]}")
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    paging: PagingConfig = field(default_factory=PagingConfig)
    analyses: list[AnalysisConfig] = field(default_factory=lambda: [AnalysisConfig(type="basic")])
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    output_dir: str = "./polymath-results"
    timeout_seconds: int = 600
    verbose: bool = False


def load_config_from_yaml(path: str) -> TestConfig:
    """Load test configuration from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed TestConfig.

    Raises:
        ImportError: If pyyaml is not installed.
        FileNotFoundError: If the config file does not exist.
    """
    if yaml is None:
        raise ImportError(
            "pyyaml is required for YAML config files. Install it with: pip install pyyaml"
        )

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        return TestConfig()

    paging = PagingConfig(**raw.get("paging", {}))

    analyses = []
    for a in raw.get("analyses", []):
        params = {}
        # Extract analysis-specific keys into parameters
        analysis_type = a.get("type", "basic")
        if analysis_type in ANALYSIS_REGISTRY:
            for key in ANALYSIS_REGISTRY[analysis_type].get("extra_metadata_keys", []):
                if key in a:
                    params[key] = a.pop(key)
        # Standard fields
        extra_caps = a.pop("extra_capabilities", [])
        a.pop("type", None)
        analyses.append(AnalysisConfig(
            type=analysis_type,
            extra_capabilities=extra_caps,
            parameters=params,
            **{k: v for k, v in a.items() if k in AnalysisConfig.__dataclass_fields__},
        ))

    if not analyses:
        analyses = [AnalysisConfig(type="basic")]

    hierarchy_raw = raw.get("hierarchy", {})
    hierarchy = HierarchyConfig(
        coordinator_class=hierarchy_raw.get("coordinator_class"),
        worker_class=hierarchy_raw.get("worker_class"),
        extra_capabilities=hierarchy_raw.get("extra_capabilities", []),
    )

    return TestConfig(
        repo_id=raw.get("repo_id", "polymath-test"),
        tenant_id=raw.get("tenant_id", "test-tenant"),
        session_id=raw.get("session_id", f"polymath-{uuid.uuid4().hex[:8]}"),
        run_id=raw.get("run_id", f"run-{uuid.uuid4().hex[:8]}"),
        paging=paging,
        analyses=analyses,
        hierarchy=hierarchy,
        output_dir=raw.get("output_dir", "./polymath-results"),
        timeout_seconds=raw.get("timeout_seconds", 600),
        verbose=raw.get("verbose", False),
    )


def generate_sample_config() -> str:
    """Generate a sample YAML configuration string."""
    return """\
# Polymath Integration Test Configuration
# ========================================
# This file configures a complete analysis workflow using the Polymathera
# multi-agent framework. Edit it to customize agent hierarchies, analysis
# types, capabilities, and paging parameters.

repo_id: "my-project"
tenant_id: "test-tenant"
# session_id: "auto-generated-if-omitted"
# run_id: "auto-generated-if-omitted"

# --- VCM Paging Configuration ---
# Controls how the codebase is partitioned into VCM pages.
paging:
  flush_threshold: 20          # Records per page before flushing
  flush_token_budget: 4096     # Token budget per page
  flush_interval_seconds: 60.0
  pinned: false                # Pin pages to prevent eviction (for small codebases)
  locality_policy_type: "tag"  # "tag" or "temporal"
  flush_policy_type: "threshold"  # "threshold", "periodic", or "immediate"

# --- Analysis Configurations ---
# Each entry spawns a coordinator agent that manages a team of workers.
# Multiple analyses can run concurrently on the same paged codebase.
analyses:

  # 1. Change Impact Analysis
  #    Coordinator → per-page ChangeImpactAnalysisAgent workers
  #    Features: multi-hop propagation, hypothesis game validation,
  #              game-theoretic merge, FeedbackLoopPredictor prefetching
  - type: impact
    coordinator_version: v2    # v1 = cache-oblivious, v2 = cache-aware
    max_agents: 10
    quality_threshold: 0.7
    max_iterations: 3
    batching_policy: hybrid    # "hybrid", "clustering", or "continuous"
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

  # 2. Compliance Analysis
  #    Coordinator → per-page ComplianceAnalysisAgent workers
  #    Features: obligation graphs, license conflict detection, risk assessment
  - type: compliance
    coordinator_version: v2
    max_agents: 8
    compliance_types:
      - license
      - security
      - quality

  # 3. Intent Inference
  #    Coordinator → per-page IntentInferenceAgent workers
  #    Features: intent graphs, misalignment detection, consensus games
  - type: intent
    coordinator_version: v2
    max_agents: 8
    granularity: "function"    # "function", "class", or "module"

  # 4. Contract Inference
  #    Coordinator → per-page ContractInferenceAgent workers
  #    Features: hypothesis games for contract validation, formal specs
  # - type: contracts
  #   coordinator_version: v2
  #   max_agents: 8
  #   formalism: "semi_formal"   # "natural", "semi_formal", "formal", "code"

  # 5. Program Slicing
  #    Coordinator → per-page ProgramSlicingAgent workers
  #    Features: backward/forward slicing, interprocedural resolution
  # - type: slicing
  #   coordinator_version: v2
  #   max_agents: 8
  #   slice_criteria:
  #     - file_path: "src/main.py"
  #       line: 42
  #       variable: "user_token"
  #       slice_type: "backward"

  # 6. Basic Code Structure Analysis
  #    Coordinator → ClusterAnalyzer workers
  #    Features: page key generation, local analysis, cross-page queries, synthesis
  # - type: basic
  #   coordinator_version: v2
  #   max_agents: 10

# --- Agent Hierarchy Overrides ---
# Override default coordinator/worker classes and attach extra capabilities
# to all agents in the hierarchy.
hierarchy:
  # coordinator_class: null     # Use analysis-specific default
  # worker_class: null          # Use analysis-specific default
  extra_capabilities:
    - ReflectionCapability       # Self-reflection on actions
    - ConsciousnessCapability    # Self-awareness via SystemDocumentation
    # - ReputationCapability     # No-regret learning for agent reputation
    # - ValidationCapability     # Multi-level validation
    # - ObjectiveGuardCapability # Goal drift prevention

output_dir: "./polymath-results"
timeout_seconds: 600
verbose: false
"""


# ===========================================================================
# Display helpers
# ===========================================================================

def build_analysis_tree(config: TestConfig) -> Tree:
    """Build a rich Tree showing the planned agent hierarchy."""
    tree = Tree(
        f"[bold cyan]Polymath Integration Test[/bold cyan]  "
        f"[dim](session={config.session_id})[/dim]"
    )

    # Paging info
    paging_node = tree.add("[bold yellow]VCM Paging[/bold yellow]")
    paging_node.add(f"flush_threshold: {config.paging.flush_threshold}")
    paging_node.add(f"flush_token_budget: {config.paging.flush_token_budget}")
    paging_node.add(f"pinned: {config.paging.pinned}")

    # Agent hierarchy
    agents_node = tree.add("[bold green]Agent Hierarchy[/bold green]")

    for i, analysis in enumerate(config.analyses):
        reg = ANALYSIS_REGISTRY.get(analysis.type)
        if not reg:
            continue

        label = reg["label"]
        version = analysis.coordinator_version
        coord_key = f"coordinator_{version}"
        coord_class = (
            config.hierarchy.coordinator_class
            or reg.get(coord_key, reg.get("coordinator_v2", ""))
        )
        worker_class = config.hierarchy.worker_class or reg.get("worker", "")

        # Coordinator node
        coord_name = coord_class.rsplit(".", 1)[-1] if coord_class else "Coordinator"
        coord_node = agents_node.add(
            f"[bold magenta]{label}[/bold magenta] — "
            f"[cyan]{coord_name}[/cyan] ({version})"
        )

        # Coordinator capabilities
        caps_node = coord_node.add("[dim]Capabilities:[/dim]")
        for cap in reg.get("coordinator_capabilities", []):
            caps_node.add(f"[dim]{cap}[/dim]")
        for cap in analysis.extra_capabilities + config.hierarchy.extra_capabilities:
            caps_node.add(f"[dim italic]{cap} (extra)[/dim italic]")

        # Config
        cfg_node = coord_node.add("[dim]Config:[/dim]")
        cfg_node.add(f"max_agents: {analysis.max_agents}")
        cfg_node.add(f"batching: {analysis.batching_policy}")
        cfg_node.add(f"quality_threshold: {analysis.quality_threshold}")

        # Worker agents (shown as template)
        worker_name = worker_class.rsplit(".", 1)[-1] if worker_class else "Worker"
        workers_node = coord_node.add(
            f"[yellow]Workers[/yellow] — [cyan]{worker_name}[/cyan] "
            f"(x{analysis.max_agents} max)"
        )
        worker_caps_node = workers_node.add("[dim]Capabilities:[/dim]")
        for cap in reg.get("worker_capabilities", []):
            worker_caps_node.add(f"[dim]{cap}[/dim]")

        # Analysis-specific parameters
        if analysis.parameters:
            params_node = coord_node.add("[dim]Parameters:[/dim]")
            for k, v in analysis.parameters.items():
                val_str = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                params_node.add(f"[dim]{k}: {escape(val_str)}[/dim]")

    return tree


def display_results_table(results: list[dict[str, Any]]) -> None:
    """Display analysis results in a rich table."""
    table = Table(title="Analysis Results", show_lines=True)
    table.add_column("Analysis", style="bold cyan", min_width=20)
    table.add_column("Status", min_width=10)
    table.add_column("Coordinator", style="dim", min_width=20)
    table.add_column("Agents Spawned", justify="right", min_width=8)
    table.add_column("Duration", justify="right", min_width=10)
    table.add_column("Summary", min_width=30)

    for r in results:
        status = r.get("status", "unknown")
        status_style = {
            "completed": "[bold green]completed[/bold green]",
            "failed": "[bold red]failed[/bold red]",
            "timeout": "[bold yellow]timeout[/bold yellow]",
            "skipped": "[dim]skipped[/dim]",
        }.get(status, f"[dim]{status}[/dim]")

        duration = r.get("duration_seconds", 0)
        duration_str = f"{duration:.1f}s" if duration else "-"

        summary = r.get("summary", "-")
        if len(summary) > 60:
            summary = summary[:57] + "..."

        table.add_row(
            r.get("analysis_type", "-"),
            status_style,
            r.get("coordinator_id", "-"),
            str(r.get("agents_spawned", "-")),
            duration_str,
            summary,
        )

    console.print(table)


# ===========================================================================
# Core integration test logic
# ===========================================================================

async def run_integration_test(
    codebase_path: str,
    config: TestConfig,
    app_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run the integration test workflow.

    This is the main entry point that:
    1. Connects to the Ray cluster and Polymathera application
    2. Pages the codebase into VCM using FileGrouperContextPageSource
    3. Spawns coordinator agents for each configured analysis
    4. Monitors progress and collects results
    5. Returns a list of result dictionaries

    Args:
        codebase_path: Absolute path to the codebase on the cluster filesystem.
        config: Test configuration.
        app_name: Optional Polymathera serving application name. If None,
            uses the current application context.

    Returns:
        List of result dictionaries, one per analysis.
    """
    # -----------------------------------------------------------------------
    # Lazy imports — these require Ray and colony to be available
    # -----------------------------------------------------------------------
    from ..vcm.sources import BuilInContextPageSourceType
    from ..vcm.models import MmapConfig, MmapResult
    from ..agents.models import AgentSpawnSpec, AgentMetadata, AgentResourceRequirements
    from ..system import get_agent_system, get_vcm, spawn_agents

    results: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Step 1: Connect to VCM and page the codebase
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]Step 1:[/bold] Paging codebase into VCM\n"
        f"  Path:   {codebase_path}\n"
        f"  Scope:  {config.repo_id}\n"
        f"  Tenant: {config.tenant_id}",
        title="[bold cyan]VCM Context Paging[/bold cyan]",
        border_style="cyan",
    ))

    vcm_handle = get_vcm(app_name)

    mmap_config = MmapConfig(
        flush_policy_type=config.paging.flush_policy_type,
        flush_threshold=config.paging.flush_threshold,
        flush_token_budget=config.paging.flush_token_budget,
        flush_interval_seconds=config.paging.flush_interval_seconds,
        locality_policy_type=config.paging.locality_policy_type,
        pinned=config.paging.pinned,
    )

    with console.status("[cyan]Mapping codebase into VCM pages..."):
        mmap_result: MmapResult = await vcm_handle.mmap_application_scope(
            scope_id=config.repo_id,
            source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
            config=mmap_config,
            tenant_id=config.tenant_id,
            repo_path=codebase_path,
        )

    if mmap_result.status in ("mapped", "already_mapped"):
        console.print(
            f"  [green]OK[/green] — {mmap_result.status}: {mmap_result.message or 'codebase mapped successfully'}"
        )
    else:
        console.print(f"  [red]FAILED[/red] — {mmap_result.status}: {mmap_result.message}")
        return [{"analysis_type": "paging", "status": "failed", "summary": mmap_result.message}]

    # -----------------------------------------------------------------------
    # Step 2: Spawn coordinator agents for each analysis
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]Step 2:[/bold] Spawning {len(config.analyses)} analysis coordinator(s)",
        title="[bold green]Agent Spawning[/bold green]",
        border_style="green",
    ))

    coordinator_specs: list[tuple[AnalysisConfig, AgentSpawnSpec]] = []

    for analysis in config.analyses:
        reg = ANALYSIS_REGISTRY.get(analysis.type)
        if not reg:
            console.print(f"  [yellow]SKIP[/yellow] Unknown analysis type: {analysis.type}")
            results.append({
                "analysis_type": analysis.type,
                "status": "skipped",
                "summary": f"Unknown analysis type: {analysis.type}",
            })
            continue

        # Resolve coordinator class
        version = analysis.coordinator_version
        coord_key = f"coordinator_{version}"
        coord_class = (
            config.hierarchy.coordinator_class
            or reg.get(coord_key, reg.get("coordinator_v2", ""))
        )

        # Build metadata for the coordinator
        metadata = AgentMetadata(
            role="coordinator",
            tenant_id=config.tenant_id,
            session_id=config.session_id,
            run_id=config.run_id,
            goals=[f"Run {reg['label']} on {config.repo_id}"],
            max_iterations=analysis.max_iterations,
            parameters={
                "repo_id": config.repo_id,
                "max_agents": analysis.max_agents,
                "quality_threshold": analysis.quality_threshold,
                "max_iterations": analysis.max_iterations,
                "batching_policy": {
                    "type": analysis.batching_policy,
                    "overlap_threshold": analysis.overlap_threshold,
                    "batch_size": analysis.batch_size,
                },
                "prefetch_depth": analysis.prefetch_depth,
                "analysis_type": analysis.type,
                **analysis.parameters,
            },
        )

        # Resolve extra capabilities
        all_extra_caps = list(set(
            analysis.extra_capabilities + config.hierarchy.extra_capabilities
        ))
        capability_paths = [
            EXTRA_CAPABILITIES_REGISTRY[c]["path"]
            for c in all_extra_caps
            if c in EXTRA_CAPABILITIES_REGISTRY
        ]

        spec = AgentSpawnSpec(
            agent_type=coord_class,
            capabilities=capability_paths,
            metadata=metadata,
            bound_pages=[],  # Coordinators don't bind to pages
        )

        coordinator_specs.append((analysis, spec))
        coord_name = coord_class.rsplit(".", 1)[-1]
        console.print(
            f"  [green]+[/green] {reg['label']} — [cyan]{coord_name}[/cyan] "
            f"(max_agents={analysis.max_agents}, batching={analysis.batching_policy})"
        )

    if not coordinator_specs:
        console.print("  [red]No valid analyses to run.[/red]")
        return results

    # Spawn all coordinators
    all_specs = [spec for _, spec in coordinator_specs]

    with console.status("[green]Spawning coordinator agents..."):
        coordinator_ids = await spawn_agents(
            agent_specs=all_specs,
            session_id=config.session_id,
            run_id=config.run_id,
            soft_affinity=True,
        )

    console.print(f"\n  Spawned {len(coordinator_ids)} coordinator(s):")
    for cid, (analysis, _) in zip(coordinator_ids, coordinator_specs):
        reg = ANALYSIS_REGISTRY[analysis.type]
        console.print(f"    [dim]{cid}[/dim] — {reg['label']}")

    # -----------------------------------------------------------------------
    # Step 3: Monitor progress
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]Step 3:[/bold] Monitoring analysis progress "
        f"(timeout={config.timeout_seconds}s)",
        title="[bold yellow]Monitoring[/bold yellow]",
        border_style="yellow",
    ))

    agent_system = get_agent_system(app_name)
    start_time = time.time()
    completed: set[str] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Analyses running...",
            total=len(coordinator_ids),
        )

        while len(completed) < len(coordinator_ids):
            elapsed = time.time() - start_time
            if elapsed > config.timeout_seconds:
                console.print(f"\n  [yellow]Timeout reached ({config.timeout_seconds}s)[/yellow]")
                break

            for i, cid in enumerate(coordinator_ids):
                if cid in completed:
                    continue

                try:
                    agent_info = await agent_system.get_agent_info(cid)
                    state = agent_info.get("state", "unknown") if agent_info else "unknown"
                except Exception:
                    state = "unknown"

                if state in ("stopped", "failed"):
                    completed.add(cid)
                    progress.update(task, advance=1)

                    analysis_cfg = coordinator_specs[i][0]
                    reg = ANALYSIS_REGISTRY.get(analysis_cfg.type, {})
                    duration = time.time() - start_time

                    # Attempt to retrieve results from blackboard
                    result_summary = "Analysis complete"
                    agents_spawned = 0
                    try:
                        # The coordinator writes final results to blackboard
                        # under its own agent_id
                        run_info = await agent_system.get_agent_info(cid)
                        if run_info:
                            agents_spawned = run_info.get("children_count", 0)
                    except Exception:
                        pass

                    if state == "failed":
                        result_summary = "Analysis failed — check logs"

                    results.append({
                        "analysis_type": reg.get("label", analysis_cfg.type),
                        "coordinator_id": cid,
                        "status": "completed" if state == "stopped" else "failed",
                        "agents_spawned": agents_spawned,
                        "duration_seconds": duration,
                        "summary": result_summary,
                    })

            await asyncio.sleep(2)

    # Mark timed-out analyses
    for i, cid in enumerate(coordinator_ids):
        if cid not in completed:
            analysis_cfg = coordinator_specs[i][0]
            reg = ANALYSIS_REGISTRY.get(analysis_cfg.type, {})
            results.append({
                "analysis_type": reg.get("label", analysis_cfg.type),
                "coordinator_id": cid,
                "status": "timeout",
                "agents_spawned": 0,
                "duration_seconds": config.timeout_seconds,
                "summary": f"Timed out after {config.timeout_seconds}s",
            })

    return results


# ===========================================================================
# Typer CLI Application
# ===========================================================================

app = typer.Typer(
    name="polymath",
    help=(
        "Integration test CLI for the Polymathera multi-agent framework.\n\n"
        "Runs complete analysis workflows on codebases using distributed "
        "agent hierarchies with cache-aware scheduling, hypothesis games, "
        "and game-theoretic result merging."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

list_app = typer.Typer(help="List available analyses, agents, and capabilities.")
app.add_typer(list_app, name="list")


# ---------------------------------------------------------------------------
# polymath run
# ---------------------------------------------------------------------------

@app.command()
def run(
    codebase_path: str = typer.Argument(
        ...,
        help="Path to the codebase on the Ray cluster filesystem.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to a YAML configuration file.",
    ),
    analysis: Optional[list[str]] = typer.Option(
        None,
        "--analysis", "-a",
        help="Analysis type(s) to run. Can be specified multiple times.",
    ),
    max_agents: int = typer.Option(
        10,
        "--max-agents", "-n",
        help="Maximum number of concurrent worker agents per analysis.",
    ),
    coordinator_version: str = typer.Option(
        "v2",
        "--coordinator-version",
        help="Coordinator version: v1 (cache-oblivious) or v2 (cache-aware).",
    ),
    batching_policy: str = typer.Option(
        "hybrid",
        "--batching",
        help="Batching policy: hybrid, clustering, or continuous.",
    ),
    quality_threshold: float = typer.Option(
        0.7,
        "--quality-threshold",
        help="Minimum quality score for accepting results (0.0-1.0).",
    ),
    repo_id: str = typer.Option(
        "polymath-test",
        "--repo-id",
        help="Unique identifier for the repository scope in VCM.",
    ),
    tenant_id: str = typer.Option(
        "test-tenant",
        "--tenant-id",
        help="Tenant ID for multi-tenant isolation.",
    ),
    timeout: int = typer.Option(
        600,
        "--timeout", "-t",
        help="Maximum time (seconds) to wait for analyses to complete.",
    ),
    output_dir: str = typer.Option(
        "./polymath-results",
        "--output", "-o",
        help="Directory to save result files.",
    ),
    app_name: Optional[str] = typer.Option(
        None,
        "--app-name",
        help="Polymathera serving application name (auto-detected if omitted).",
    ),
    extra_capabilities: Optional[list[str]] = typer.Option(
        None,
        "--capability",
        help="Extra capabilities to attach to all agents. Can be specified multiple times.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the planned agent hierarchy without running the test.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Run an integration test of the multi-agent framework on a codebase.

    This command pages the codebase into VCM, spawns coordinator agents that
    create teams of specialized workers, and runs complete analysis workflows.

    [bold]Quick Start:[/bold]

        polymath run /path/to/codebase --analysis impact --analysis compliance

    [bold]With Config File:[/bold]

        polymath run /path/to/codebase --config test.yaml

    [bold]Dry Run (show hierarchy only):[/bold]

        polymath run /path/to/codebase --analysis impact --dry-run
    """
    if verbose:
        logging.getLogger("polymath").setLevel(logging.DEBUG)
        logging.getLogger("colony").setLevel(logging.DEBUG)

    # Build configuration
    if config:
        try:
            test_config = load_config_from_yaml(config)
            console.print(f"[dim]Loaded config from {config}[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to load config: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Build config from CLI arguments
        analyses_list = []
        analysis_types = analysis or ["basic"]
        for a in analysis_types:
            analyses_list.append(AnalysisConfig(
                type=a,
                coordinator_version=coordinator_version,
                max_agents=max_agents,
                batching_policy=batching_policy,
                quality_threshold=quality_threshold,
            ))

        test_config = TestConfig(
            repo_id=repo_id,
            tenant_id=tenant_id,
            analyses=analyses_list,
            hierarchy=HierarchyConfig(
                extra_capabilities=extra_capabilities or [],
            ),
            output_dir=output_dir,
            timeout_seconds=timeout,
            verbose=verbose,
        )

    # Validate codebase path
    codebase = Path(codebase_path)
    if not codebase.exists():
        console.print(
            f"[yellow]Warning:[/yellow] Codebase path does not exist locally: {codebase_path}\n"
            f"  This is OK if running on a remote Ray cluster where the path is valid."
        )

    # Display planned hierarchy
    console.print()
    tree = build_analysis_tree(test_config)
    console.print(tree)
    console.print()

    if dry_run:
        console.print("[dim]Dry run complete. No agents were spawned.[/dim]")
        raise typer.Exit(0)

    # Run the integration test
    console.print(Panel(
        f"[bold]Codebase:[/bold] {codebase_path}\n"
        f"[bold]Repo ID:[/bold]  {test_config.repo_id}\n"
        f"[bold]Session:[/bold]  {test_config.session_id}\n"
        f"[bold]Run ID:[/bold]   {test_config.run_id}\n"
        f"[bold]Timeout:[/bold]  {test_config.timeout_seconds}s",
        title="[bold]Starting Integration Test[/bold]",
        border_style="bright_blue",
    ))

    try:
        test_results = asyncio.run(run_integration_test(
            codebase_path=str(codebase),
            config=test_config,
            app_name=app_name,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Integration test failed:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    # Display results
    console.print()
    display_results_table(test_results)

    # Save results
    output_path = Path(test_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"results_{test_config.session_id}.json"

    results_payload = {
        "session_id": test_config.session_id,
        "run_id": test_config.run_id,
        "repo_id": test_config.repo_id,
        "codebase_path": str(codebase),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": test_results,
    }

    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    console.print(f"\n[dim]Results saved to {results_file}[/dim]")

    # Exit with error code if any analysis failed
    failures = [r for r in test_results if r.get("status") in ("failed", "timeout")]
    if failures:
        console.print(f"\n[yellow]{len(failures)} analysis(es) failed or timed out.[/yellow]")
        raise typer.Exit(1)

    console.print("\n[bold green]All analyses completed successfully.[/bold green]")


# ---------------------------------------------------------------------------
# polymath list analyses
# ---------------------------------------------------------------------------

@list_app.command("analyses")
def list_analyses() -> None:
    """List all available analysis types."""
    table = Table(title="Available Analysis Types", show_lines=True)
    table.add_column("Type", style="bold cyan", min_width=12)
    table.add_column("Label", min_width=25)
    table.add_column("Description", min_width=40)
    table.add_column("Coordinator Capabilities", style="dim", min_width=30)

    for atype, reg in ANALYSIS_REGISTRY.items():
        caps = ", ".join(reg.get("coordinator_capabilities", []))
        table.add_row(atype, reg["label"], reg["description"], caps)

    console.print(table)


# ---------------------------------------------------------------------------
# polymath list agents
# ---------------------------------------------------------------------------

@list_app.command("agents")
def list_agents() -> None:
    """List all available agent classes for each analysis type."""
    table = Table(title="Available Agent Classes", show_lines=True)
    table.add_column("Analysis", style="bold cyan", min_width=12)
    table.add_column("Role", min_width=12)
    table.add_column("Class Path", style="dim")
    table.add_column("Version", min_width=8)

    for atype, reg in ANALYSIS_REGISTRY.items():
        for version in ("v1", "v2"):
            coord_key = f"coordinator_{version}"
            if coord_key in reg:
                table.add_row(
                    atype,
                    "coordinator",
                    reg[coord_key],
                    version,
                )
        table.add_row(atype, "worker", reg.get("worker", "-"), "-")

    console.print(table)


# ---------------------------------------------------------------------------
# polymath list capabilities
# ---------------------------------------------------------------------------

@list_app.command("capabilities")
def list_capabilities() -> None:
    """List all extra capabilities that can be attached to agents."""
    table = Table(title="Extra Capabilities (Cross-Cutting Concerns)", show_lines=True)
    table.add_column("Name", style="bold cyan", min_width=25)
    table.add_column("Description", min_width=50)

    for name, info in EXTRA_CAPABILITIES_REGISTRY.items():
        table.add_row(name, info["description"])

    console.print(table)
    console.print(
        "\n[dim]Use --capability NAME with 'polymath run' or add to "
        "hierarchy.extra_capabilities in YAML config.[/dim]"
    )


# ---------------------------------------------------------------------------
# polymath init-config
# ---------------------------------------------------------------------------

@app.command("init-config")
def init_config(
    output: str = typer.Option(
        "polymath-config.yaml",
        "--output", "-o",
        help="Output path for the generated YAML config.",
    ),
) -> None:
    """Generate a sample YAML configuration file.

    The generated file includes all available options with documentation.
    Edit it to customize your integration test.
    """
    output_path = Path(output)
    if output_path.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    sample = generate_sample_config()
    output_path.write_text(sample)
    console.print(f"[green]Sample config written to {output}[/green]")
    console.print(f"[dim]Edit the file and run: polymath run /path/to/code --config {output}[/dim]")


# ---------------------------------------------------------------------------
# polymath describe
# ---------------------------------------------------------------------------

@app.command("describe")
def describe(
    analysis_type: str = typer.Argument(
        ...,
        help="Analysis type to describe (e.g., 'impact', 'compliance').",
    ),
) -> None:
    """Show detailed information about an analysis type.

    Displays the full agent hierarchy, capabilities, game protocols,
    and merge policies used by the analysis.
    """
    reg = ANALYSIS_REGISTRY.get(analysis_type)
    if not reg:
        console.print(f"[red]Unknown analysis type: {analysis_type}[/red]")
        console.print(f"[dim]Available: {', '.join(ANALYSIS_REGISTRY.keys())}[/dim]")
        raise typer.Exit(1)

    # Title panel
    console.print(Panel(
        reg["description"],
        title=f"[bold cyan]{reg['label']}[/bold cyan]",
        border_style="cyan",
    ))

    # Agent hierarchy tree
    tree = Tree(f"[bold]{reg['label']} Agent Hierarchy[/bold]")

    # Coordinator
    for version in ("v1", "v2"):
        coord_key = f"coordinator_{version}"
        if coord_key in reg:
            coord_class = reg[coord_key].rsplit(".", 1)[-1]
            coord_node = tree.add(
                f"[magenta]{coord_class}[/magenta] ({version})"
            )
            caps_node = coord_node.add("[dim]Capabilities:[/dim]")
            for cap in reg.get("coordinator_capabilities", []):
                caps_node.add(f"[dim]{cap}[/dim]")

    # Worker
    worker_class = reg.get("worker", "").rsplit(".", 1)[-1]
    worker_node = tree.add(f"[yellow]{worker_class}[/yellow] (per-page)")
    caps_node = worker_node.add("[dim]Capabilities:[/dim]")
    for cap in reg.get("worker_capabilities", []):
        caps_node.add(f"[dim]{cap}[/dim]")

    console.print(tree)

    # Execution flow
    console.print()
    flow_items = {
        "impact": [
            "1. Coordinator maps codebase pages and initializes working set",
            "2. Cache-aware batch scheduling selects pages with highest working set overlap",
            "3. Per-page ChangeImpactAnalysisAgent analyzes local impact using LLM",
            "4. FeedbackLoopPredictor prefetches pages for self-critique and hypothesis games",
            "5. Multi-hop propagation traces impact across dependency graph (BFS)",
            "6. HypothesisGameProtocol validates CRITICAL impacts (proposer/skeptic/arbiter)",
            "7. ImpactMergePolicy consolidates results using game-theoretic coalition weights",
            "8. SynthesisCapability produces unified ChangeImpactReport",
        ],
        "slicing": [
            "1. Coordinator receives slice criteria (file, line, variable, slice type)",
            "2. Per-page ProgramSlicingAgent computes data and control dependencies via LLM",
            "3. Interprocedural resolution resolves cross-page dependencies",
            "4. SliceMergePolicy merges partial slices across pages",
            "5. Final ProgramSlice output contains included/excluded lines and dependency graph",
        ],
        "compliance": [
            "1. Coordinator initializes with compliance types to check",
            "2. Per-page ComplianceAnalysisAgent detects licenses, violations, and risks via LLM",
            "3. Obligation graph tracks compliance requirements across pages",
            "4. ComplianceMergePolicy consolidates violations and license conflicts",
            "5. Final ComplianceReport includes risk assessment and remediation recommendations",
        ],
        "intent": [
            "1. Coordinator sets granularity level (function/class/module)",
            "2. Per-page IntentInferenceAgent infers code intents and business goals via LLM",
            "3. ConsensusGameProtocol resolves disagreements between agents",
            "4. IntentMergePolicy builds cross-page intent graph",
            "5. Coordinator detects misalignments between intent and implementation",
        ],
        "contracts": [
            "1. Coordinator sets formalism level (natural/semi-formal/formal/code)",
            "2. Per-page ContractInferenceAgent infers preconditions, postconditions, invariants via LLM",
            "3. HypothesisGameProtocol validates critical contracts with counterexample search",
            "4. ContractMergePolicy merges cross-page contracts",
            "5. Final output includes FunctionContract objects at requested formalism level",
        ],
        "basic": [
            "1. Coordinator collects all page clusters from context page source",
            "2. Cache-aware scheduling spawns ClusterAnalyzer agents by working set overlap",
            "3. Each ClusterAnalyzer runs 4-phase protocol:",
            "   a. Key generation — structural summaries for all pages in cluster",
            "   b. Local analysis — analyze each page independently (one at a time)",
            "   c. Query processing — cross-page query resolution via attention routing",
            "   d. Synthesis — combine findings into cluster summary (no pages loaded)",
            "4. CriticCapability critiques each cluster's results",
            "5. Coordinator synthesizes global report from all cluster summaries",
        ],
    }

    flow = flow_items.get(analysis_type, ["No execution flow documentation available."])
    console.print(Panel(
        "\n".join(flow),
        title="[bold]Execution Flow[/bold]",
        border_style="dim",
    ))

    # Extra metadata keys
    extra_keys = reg.get("extra_metadata_keys", [])
    if extra_keys:
        console.print(
            f"\n[dim]Analysis-specific parameters: {', '.join(extra_keys)}[/dim]"
        )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Entry point for the polymath CLI."""
    app()


if __name__ == "__main__":
    main()
