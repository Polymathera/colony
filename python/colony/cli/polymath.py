#!/usr/bin/env python3
"""Polymath: Integration test CLI for the Polymathera multi-agent framework.

This tool runs a complete analysis workflow on a codebase using the Polymathera Colony
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
    - A running Ray cluster (local or AWS) — the CLI deploys the Polymathera application
    - The target codebase available on the cluster filesystem (e.g., via EFS or
      shared Docker volume)
    - Python dependencies: typer, rich, pyyaml

Usage:
    # Quick start — run impact analysis on a local codebase
    polymath run --local-repo /path/to/codebase --analysis impact

    # Remote repository
    polymath run --origin-url https://github.com/org/repo --analysis impact

    # Run multiple analyses with a YAML config
    polymath run --local-repo /path/to/codebase --config my_test.yaml

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
#
# NOTE on worker_capabilities: These are DOCUMENTATION-ONLY. Workers are
# self-configuring — each worker agent class adds its own capabilities in its
# initialize() method (e.g., ChangeImpactAnalysisAgent extends HypothesisGameAgent
# which adds HypothesisGameProtocol automatically). The lists here are used by
# the `describe` and `list` commands to show what each worker provides.

# TODO: This registry should be allowed to be injected using a JSON or Markdown file.

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
        ],
        "worker_capabilities": [
            "ChangeImpactAnalysisCapability",
            "MergeCapability",
            "GroundingCapability",
            "HypothesisGameProtocol",  # via HypothesisGameAgent base class
        ],
        "extra_metadata_keys": ["changes", "change_description"],
        "self_concept": {
            "description": (
                "Coordinates change impact analysis across a codebase by spawning "
                "worker agents, propagating dependency chains, and synthesizing "
                "a unified impact report."
            ),
            "goals": [
                "Identify all code regions affected by the specified changes",
                "Propagate impact through multi-hop dependency chains",
                "Validate critical impacts via hypothesis games",
                "Produce a ranked, grounded impact report with confidence scores",
            ],
            "constraints": [
                "Every impact claim must be grounded in source code evidence",
                "Do not hallucinate dependencies that are not in the codebase",
            ],
        },
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
        "self_concept": {
            "description": (
                "Coordinates program slicing by distributing slice criteria across "
                "workers and merging partial slices into minimal complete subsets."
            ),
            "goals": [
                "Extract the minimal code subset that affects the target criteria",
                "Preserve soundness — never omit statements that influence the target",
                "Maximize precision — exclude statements irrelevant to the target",
            ],
            "constraints": [
                "Slices must be complete — every data and control dependency on the target must be included",
                "Do not include code that has no influence path to the slice criterion",
            ],
        },
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
        "self_concept": {
            "description": (
                "Coordinates compliance analysis by distributing license, regulatory, "
                "and security checks across workers and building obligation graphs."
            ),
            "goals": [
                "Identify all license obligations and compatibility conflicts",
                "Detect regulatory and security compliance violations",
                "Build obligation graphs linking requirements to source evidence",
                "Produce actionable compliance reports with remediation guidance",
            ],
            "constraints": [
                "Every compliance finding must cite the specific license clause or regulation",
                "Do not make legal conclusions — report obligations and conflicts factually",
            ],
        },
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
        "self_concept": {
            "description": (
                "Coordinates intent inference by distributing analysis across workers "
                "and building intent graphs that map code to business purposes."
            ),
            "goals": [
                "Infer the business-level purpose behind each code component",
                "Distinguish business logic from implementation scaffolding",
                "Detect misalignments between stated intent and actual behavior",
                "Build intent graphs linking code regions to inferred purposes",
            ],
            "constraints": [
                "Clearly separate high-confidence inferences from speculative ones",
                "Use consensus game validation for contested intent claims",
            ],
        },
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
        "self_concept": {
            "description": (
                "Coordinates contract inference by distributing function analysis "
                "across workers and validating inferred contracts via hypothesis games."
            ),
            "goals": [
                "Infer preconditions, postconditions, and invariants for each function",
                "Validate contracts against actual code behavior",
                "Produce specifications at the requested formalism level",
            ],
            "constraints": [
                "Inferred contracts must be consistent with the code — no aspirational specs",
                "Use hypothesis games to challenge and validate each contract before accepting",
            ],
        },
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
        "self_concept": {
            "description": (
                "Coordinates general-purpose code structure analysis by spawning "
                "cluster analyzers and synthesizing their findings."
            ),
            "goals": [
                "Analyze code structure, patterns, and relationships across the codebase",
                "Synthesize findings from individual cluster analyses into a coherent report",
            ],
            "constraints": [
                "Ground all findings in actual code evidence from the analyzed pages",
            ],
        },
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
class VLLMDeploymentYAMLConfig:
    """YAML-friendly config for a single vLLM deployment (GPU-based).

    Parsed from the YAML cluster.vllm_deployments list and converted to
    LLMDeploymentConfig when building the PolymatheraCluster.
    """
    model_name: str = "meta-llama/Llama-3.1-8B"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    kv_cache_capacity: int | None = None
    num_replicas: int = 2
    quantization: str | None = None
    s3_bucket: str | None = None


@dataclass
class RemoteDeploymentYAMLConfig:
    """YAML-friendly config for a single remote LLM deployment.

    Parsed from the YAML cluster.remote_deployments list and converted to
    RemoteLLMDeploymentConfig when building the PolymatheraCluster.
    """
    model_name: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"  # "anthropic" or "openrouter"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    max_cached_pages: int = 50
    max_cached_tokens: int = 2_000_000
    system_prompt: str | None = None
    ttl: str = "1h"  # "5m" or "1h"
    max_concurrent_requests: int = 10
    num_replicas: int = 1


@dataclass
class RemoteEmbeddingYAMLConfig:
    """YAML-friendly config for a remote API-based embedding deployment.

    Parsed from the YAML cluster.remote_embedding_config section and converted
    to RemoteEmbeddingConfig when building the PolymatheraCluster.
    """
    model_name: str = "text-embedding-3-small"
    provider: str = "openrouter"  # "openai", "gemini", or "openrouter"
    api_key_env_var: str | None = None  # Defaults per provider
    dimensions: int | None = None
    max_batch_size: int = 2048
    max_concurrent_requests: int = 10
    num_replicas: int = 1


@dataclass
class STEmbeddingYAMLConfig:
    """YAML-friendly config for a SentenceTransformer embedding deployment.

    Parsed from the YAML cluster.st_embedding_config section and converted
    to STEmbeddingDeploymentConfig when building the PolymatheraCluster.
    """
    model_name: str = "all-MiniLM-L6-v2"  # STEmbeddingModel enum value
    fallback_model: str = "all-MiniLM-L6-v2"
    enable_gpu: bool = False
    batch_size: int = 32
    max_concurrent_embeddings: int = 8
    max_content_length: int = 2048
    chunk_large_files: bool = True
    max_chunks_per_file: int = 10


@dataclass
class VCMYAMLConfig:
    """YAML-friendly config for VirtualContextManager.

    Parsed from the top-level 'vcm' section of the YAML config and converted
    to VCMConfig when building the PolymatheraCluster.
    """
    caching_policy: str = "LRU"                    # "LRU" or "LFU"
    page_storage_backend_type: str = "efs"         # "efs" or "s3"
    page_storage_path: str = "colony/context_pages"
    page_fault_processing_interval_s: float = 5.0
    metrics_collection_interval_s: float = 30.0
    reconciliation_interval_s: float = 30.0


@dataclass
class AgentSystemYAMLConfig:
    """YAML-friendly config for AgentSystem.

    Parsed from the top-level 'agent_system' section of the YAML config
    and converted to AgentSystemConfig when building the PolymatheraCluster.
    """
    max_retries: int = 3
    enable_sessions: bool = True
    default_session_ttl: float = 86400.0   # 24 hours


@dataclass
class LLMClusterYAMLConfig:
    """YAML-friendly cluster configuration.

    Controls which LLM deployments are created.
    - Cloud mode (GPUs): specify vllm_deployments (and optional embedding_config)
    - Local mode (no GPUs): specify remote_deployments and remote_embedding_config
    - Hybrid: both vllm_deployments and remote_deployments
    """
    app_name: str = "polymathera"
    vllm_deployments: list[VLLMDeploymentYAMLConfig] = field(default_factory=list)
    remote_deployments: list[RemoteDeploymentYAMLConfig] = field(default_factory=list)
    embedding_config: VLLMDeploymentYAMLConfig | None = None
    remote_embedding_config: RemoteEmbeddingYAMLConfig | None = None
    st_embedding_config: STEmbeddingYAMLConfig | None = None
    cleanup_on_init: bool = True


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
    cluster: LLMClusterYAMLConfig = field(default_factory=LLMClusterYAMLConfig)
    vcm: VCMYAMLConfig = field(default_factory=VCMYAMLConfig)
    agent_system: AgentSystemYAMLConfig = field(default_factory=AgentSystemYAMLConfig)
    output_dir: str = "./polymath-results"
    timeout_seconds: int = 600
    verbose: bool = False
    # Optional directory containing user code (custom agents, capabilities, configs)
    # to distribute to Ray worker nodes.  Ray zips this directory on the driver,
    # uploads it to every worker (including autoscaled ones), unpacks it, and
    # adds it to sys.path so its modules are importable.
    #
    # This is NOT for the codebase being analyzed — that lives on shared storage
    # (e.g. EFS) and is accessed via VCM paging.  Use this when your YAML config
    # references custom agent_type or capability paths that aren't in the Docker
    # image or the colony package.
    working_dir: str | None = None
    # Git repository to analyze.  Exactly one of --origin-url or --local-repo
    # must be set.  --local-repo is translated to file://<path> so downstream
    # code always works with a URL (https:// or file://).
    origin_url: str = ""       # Git repository URL (https:// or file://)
    branch: str = "main"       # Git branch to check out
    commit: str = "HEAD"       # Git commit SHA (defaults to branch HEAD)
    # Budget enforcement for remote LLM deployments
    budget_usd: float | None = None  # Maximum budget in USD (None = unlimited)
    warn_budget_pct: float = 0.8     # Warn when cost reaches this fraction of budget


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

    # Parse cluster config (vLLM deployments, remote deployments, embedding)
    cluster_raw = raw.get("cluster", {})

    vllm_deps = []
    for vd in cluster_raw.get("vllm_deployments", []):
        vllm_deps.append(VLLMDeploymentYAMLConfig(
            **{k: v for k, v in vd.items() if k in VLLMDeploymentYAMLConfig.__dataclass_fields__},
        ))

    remote_deps = []
    for rd in cluster_raw.get("remote_deployments", []):
        remote_deps.append(RemoteDeploymentYAMLConfig(
            **{k: v for k, v in rd.items() if k in RemoteDeploymentYAMLConfig.__dataclass_fields__},
        ))

    embedding_raw = cluster_raw.get("embedding_config")
    embedding_cfg = None
    if embedding_raw:
        embedding_cfg = VLLMDeploymentYAMLConfig(
            **{k: v for k, v in embedding_raw.items() if k in VLLMDeploymentYAMLConfig.__dataclass_fields__},
        )

    remote_embedding_raw = cluster_raw.get("remote_embedding_config")
    remote_embedding_cfg = None
    if remote_embedding_raw:
        remote_embedding_cfg = RemoteEmbeddingYAMLConfig(
            **{k: v for k, v in remote_embedding_raw.items()
               if k in RemoteEmbeddingYAMLConfig.__dataclass_fields__},
        )

    st_embedding_raw = cluster_raw.get("st_embedding_config")
    st_embedding_cfg = None
    if st_embedding_raw:
        st_embedding_cfg = STEmbeddingYAMLConfig(
            **{k: v for k, v in st_embedding_raw.items()
               if k in STEmbeddingYAMLConfig.__dataclass_fields__},
        )

    cluster = LLMClusterYAMLConfig(
        app_name=cluster_raw.get("app_name", "polymathera"),
        vllm_deployments=vllm_deps,
        remote_deployments=remote_deps,
        embedding_config=embedding_cfg,
        remote_embedding_config=remote_embedding_cfg,
        st_embedding_config=st_embedding_cfg,
        cleanup_on_init=cluster_raw.get("cleanup_on_init", True),
    )

    # Parse VCM config
    vcm_raw = raw.get("vcm", {})
    vcm_cfg = VCMYAMLConfig(
        **{k: v for k, v in vcm_raw.items() if k in VCMYAMLConfig.__dataclass_fields__},
    )

    # Parse agent system config
    agent_system_raw = raw.get("agent_system", {})
    agent_system_cfg = AgentSystemYAMLConfig(
        **{k: v for k, v in agent_system_raw.items() if k in AgentSystemYAMLConfig.__dataclass_fields__},
    )

    return TestConfig(
        repo_id=raw.get("repo_id", "polymath-test"),
        tenant_id=raw.get("tenant_id", "test-tenant"),
        session_id=raw.get("session_id", f"polymath-{uuid.uuid4().hex[:8]}"),
        run_id=raw.get("run_id", f"run-{uuid.uuid4().hex[:8]}"),
        paging=paging,
        analyses=analyses,
        hierarchy=hierarchy,
        cluster=cluster,
        vcm=vcm_cfg,
        agent_system=agent_system_cfg,
        output_dir=raw.get("output_dir", "./polymath-results"),
        timeout_seconds=raw.get("timeout_seconds", 600),
        verbose=raw.get("verbose", False),
        working_dir=raw.get("working_dir"),
        origin_url=raw.get("origin_url", ""),
        branch=raw.get("branch", "main"),
        commit=raw.get("commit", "HEAD"),
        budget_usd=raw.get("budget_usd"),
        warn_budget_pct=raw.get("warn_budget_pct", 0.8),
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

# --- Repository ---
# Specify the git repository to analyze.  Use origin_url for remote repos
# (https://...) or for local repos as file:///path/to/repo.
# The CLI --origin-url and --local-repo flags override this value.
# origin_url: "https://github.com/org/repo"
# branch: "main"
# commit: "HEAD"

# --- LLM Cluster Configuration ---
# Controls which LLM deployments are created and deployed.
#
# Modes:
#   - Local (no GPUs):  Use remote_deployments only
#   - Cloud (GPUs):     Use vllm_deployments (and optional embedding_config)
#   - Hybrid:           Both vllm_deployments and remote_deployments
cluster:
  app_name: "polymathera"
  cleanup_on_init: true        # Clean up previous deployment state on startup

  # --- Self-hosted vLLM deployments (require GPUs) ---
  # vllm_deployments:
  #   - model_name: "meta-llama/Llama-3.1-8B"
  #     tensor_parallel_size: 1   # GPUs per replica
  #     gpu_memory_utilization: 0.9
  #     max_model_len: null        # Auto from model registry
  #     num_replicas: 2
  #     quantization: null         # "awq", "gptq", "fp8", or null
  #     s3_bucket: null            # S3 bucket for model weights

  # --- Embedding model (GPU-based, requires GPU) ---
  # embedding_config:
  #   model_name: "intfloat/e5-small-v2"
  #   tensor_parallel_size: 1
  #   gpu_memory_utilization: 0.5
  #   num_replicas: 1

  # --- Remote embedding (API-based, no GPU required) ---
  # Mutually exclusive with embedding_config and st_embedding_config.
  # remote_embedding_config:
  #   model_name: "text-embedding-3-small"
  #   provider: "openai"           # "openai", "gemini", or "openrouter"
  #   # api_key_env_var: "OPENAI_API_KEY"  # Auto-detected from provider
  #   # dimensions: null           # null = model default (1536 for small)
  #   # max_batch_size: 2048       # OpenAI: 2048, Gemini: 100
  #   # num_replicas: 1

  # --- SentenceTransformer embedding (CPU or GPU, no API key needed) ---
  # Mutually exclusive with embedding_config and remote_embedding_config.
  # st_embedding_config:
  #   model_name: "all-MiniLM-L6-v2"      # See STEmbeddingModel enum for options
  #   # fallback_model: "all-MiniLM-L6-v2"
  #   # enable_gpu: false
  #   # max_content_length: 2048
  #   # chunk_large_files: true

  # --- Remote LLM deployments (no GPUs required) ---
  remote_deployments:
    # Anthropic Claude — uses prefix caching for VCM page text
    - model_name: "claude-sonnet-4-20250514"
      provider: "anthropic"
      api_key_env_var: "ANTHROPIC_API_KEY"
      max_cached_pages: 50
      max_cached_tokens: 2000000
      ttl: "1h"                # Prefix cache TTL ("5m" or "1h")
      num_replicas: 1

    # OpenRouter — OpenAI-compatible API, supports many providers
    # - model_name: "anthropic/claude-sonnet-4"
    #   provider: "openrouter"
    #   api_key_env_var: "OPENROUTER_API_KEY"
    #   num_replicas: 1

# --- VCM Configuration ---
# Controls VirtualContextManager behavior: page storage, eviction policy, etc.
# vcm:
#   caching_policy: "LRU"              # Page eviction policy: "LRU" or "LFU"
#   page_storage_backend_type: "efs"   # Storage backend: "efs" or "s3"
#   page_storage_path: "colony/context_pages"
#   page_fault_processing_interval_s: 5.0
#   metrics_collection_interval_s: 30.0
#   reconciliation_interval_s: 30.0

# --- Agent System Configuration ---
# Controls agent system behavior: sessions, retries, etc.
# agent_system:
#   max_retries: 3
#   enable_sessions: true
#   default_session_ttl: 86400.0       # 24 hours

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

# --- Ray Worker Code Distribution ---
# Optional directory containing YOUR code (custom agents, capabilities, configs)
# to distribute to all Ray worker nodes.  Ray zips this on the driver, uploads
# it to every worker (including autoscaled ones), and adds it to sys.path.
#
# This is NOT for the codebase being analyzed — that lives on shared storage
# (e.g. EFS) and is accessed via VCM paging.  Use this when your config
# references custom agent_type or capability class paths that aren't already
# in the Docker image or the colony package.
# working_dir: "/path/to/my-custom-agents"

output_dir: "./polymath-results"
timeout_seconds: 600
verbose: false

# --- Budget Enforcement (Remote LLM APIs) ---
# Maximum spend in USD for remote LLM API calls (Anthropic, OpenRouter).
# When the budget is exceeded, remaining analyses are skipped (soft stop).
# Omit or set to null for unlimited spending.
# budget_usd: 5.00
# warn_budget_pct: 0.8    # Warn when cost reaches 80% of budget (default)
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
    # Check if any result has cost data to decide whether to show cost columns
    has_cost_data = any(r.get("cost_usd", 0) > 0 for r in results)
    has_token_data = any(
        r.get("input_tokens", 0) > 0 or r.get("output_tokens", 0) > 0
        for r in results
    )

    table = Table(title="Analysis Results", show_lines=True)
    table.add_column("Analysis", style="bold cyan", min_width=20)
    table.add_column("Status", min_width=10)
    table.add_column("Coordinator", style="dim", min_width=20)
    table.add_column("Agents Spawned", justify="right", min_width=8)
    table.add_column("Duration", justify="right", min_width=10)
    if has_token_data:
        table.add_column("Tokens (in/out)", justify="right", min_width=14)
    if has_cost_data:
        table.add_column("Cost ($)", justify="right", min_width=10)
    table.add_column("Summary", min_width=30)

    for r in results:
        status = r.get("status", "unknown")
        status_style = {
            "completed": "[bold green]completed[/bold green]",
            "failed": "[bold red]failed[/bold red]",
            "timeout": "[bold yellow]timeout[/bold yellow]",
            "skipped": "[dim]skipped[/dim]",
            "budget_exceeded": "[bold red]budget exceeded[/bold red]",
        }.get(status, f"[dim]{status}[/dim]")

        duration = r.get("duration_seconds", 0)
        duration_str = f"{duration:.1f}s" if duration else "-"

        summary = r.get("summary", "-")
        if len(summary) > 60:
            summary = summary[:57] + "..."

        row = [
            r.get("analysis_type", "-"),
            status_style,
            r.get("coordinator_id", "-"),
            str(r.get("agents_spawned", "-")),
            duration_str,
        ]

        if has_token_data:
            in_tok = r.get("input_tokens", 0)
            out_tok = r.get("output_tokens", 0)
            row.append(f"{in_tok:,}/{out_tok:,}" if in_tok or out_tok else "-")

        if has_cost_data:
            cost = r.get("cost_usd", 0.0)
            row.append(f"${cost:.4f}" if cost > 0 else "-")

        row.append(summary)
        table.add_row(*row)

    console.print(table)

    # Print cost summary if there's cost data
    if has_cost_data:
        total_cost = sum(r.get("cost_usd", 0.0) for r in results)
        total_in = sum(r.get("input_tokens", 0) for r in results)
        total_out = sum(r.get("output_tokens", 0) for r in results)
        total_cache_read = sum(r.get("cache_read_tokens", 0) for r in results)
        total_cache_write = sum(r.get("cache_write_tokens", 0) for r in results)
        console.print(
            f"\n  [bold]Total cost:[/bold] ${total_cost:.4f}  "
            f"[dim](tokens: {total_in:,} in / {total_out:,} out, "
            f"cache: {total_cache_read:,} read / {total_cache_write:,} write)[/dim]"
        )


# ===========================================================================
# Core integration test logic
# ===========================================================================

def _resolve_class(fqn: str) -> type:
    """Resolve a fully qualified class name to the actual class."""
    import importlib
    module_path, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def run_integration_test(
    config: TestConfig,
    app_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run the integration test workflow.

    This is the main entry point that:
    1. Connects to the Ray cluster via PolymatheraApp.setup_ray()
    2. Deploys PolymatheraCluster (LLM deployments, VCM, agent system)
    3. Pages the codebase into VCM using FileGrouperContextPageSource
    4. Spawns coordinator agents via AgentHandle.from_blueprint()
    5. Monitors progress via handle.run_streamed() and collects results
    6. Returns a list of result dictionaries

    Args:
        config: Test configuration (includes origin_url, branch, commit).
        app_name: Optional Polymathera serving application name. If None,
            uses the current application context.

    Returns:
        List of result dictionaries, one per analysis.
    """
    # -----------------------------------------------------------------------
    # Lazy imports — these require Ray and colony to be available.
    # We use absolute imports so the CLI can run both as a module
    # (python -m colony.cli.polymath) and as a direct script (./polymath.py).
    # -----------------------------------------------------------------------
    from colony.distributed import get_initialized_polymathera
    from colony.vcm.sources import BuilInContextPageSourceType, ContextPageSourceFactory
    from colony.vcm.models import MmapConfig, MmapResult
    from colony.agents import AgentMetadata, AgentHandle, AgentRunEvent
    from colony.system import (
        PolymatheraCluster, PolymatheraClusterConfig,
        get_vcm, get_session_manager,
    )
    from colony.cluster.config import ClusterConfig, LLMDeploymentConfig
    from colony.cluster.remote_config import RemoteLLMDeploymentConfig
    from colony.vcm.config import VCMConfig
    from colony.agents.config import AgentSystemConfig
    from colony.agents.sessions import AgentRun
    from colony.agents import AgentSelfConcept

    # Import built-in page source modules to trigger their @register decorators.
    # User-defined page sources from working_dir should also be imported here
    # (or in user startup code) before publish_to_env() is called.
    import colony.samples.paging  # noqa: F401 — registers file_grouper
    # blackboard is already registered via colony.agents.blackboard import chain

    # Publish registered page source module paths to an env var so Ray workers
    # can discover and import them (they start with a fresh Python interpreter).
    ContextPageSourceFactory.publish_to_env()

    results: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Step 0: Connect to Ray cluster
    # -----------------------------------------------------------------------
    console.print()
    # Build Step 0 display
    step0_lines = ["[bold]Step 0:[/bold] Connecting to Ray cluster"]
    if config.working_dir:
        step0_lines.append(f"  working_dir: {config.working_dir}")
    console.print(Panel(
        "\n".join(step0_lines),
        title="[bold blue]Ray Cluster Connection[/bold blue]",
        border_style="blue",
    ))

    with console.status("[blue]Connecting to Ray cluster..."):
        polymathera = await get_initialized_polymathera()

        # setup_ray connects to the Ray cluster and propagates env vars to
        # all worker processes (including autoscaled nodes).
        #
        # working_dir (optional): A directory on the driver containing user
        # code — custom agents, capabilities, configs — that workers need to
        # import.  Ray zips it, uploads it to every worker, unpacks it, and
        # prepends it to sys.path.
        #
        # This is NOT for the codebase being analyzed.  The target codebase
        # lives on shared storage (e.g. EFS) and workers access it through
        # VCM paging.  Colony modules are already distributed via py_modules
        # (setup_ray appends the colony package automatically).
        # Collect env vars to propagate — include API keys for remote deployments
        env_prefixes = ("POLYMATH_", "RAY_", "REDIS_", "PYTHONPATH")
        api_key_vars = {
            rd.api_key_env_var
            for rd in config.cluster.remote_deployments
        }
        if config.cluster.remote_embedding_config and config.cluster.remote_embedding_config.api_key_env_var:
            api_key_vars.add(config.cluster.remote_embedding_config.api_key_env_var)
        worker_env_vars = {
            k: v for k, v in os.environ.items()
            if k.startswith(env_prefixes) or k in api_key_vars
        }

        await polymathera.setup_ray(
            worker_env_vars=worker_env_vars,
            working_dir=config.working_dir,
        )

    console.print("  [green]OK[/green] — Connected to Ray cluster")
    if config.working_dir:
        console.print(f"  [dim]User code distributed from: {config.working_dir}[/dim]")

    # Use the deployed app name for subsequent handle lookups
    effective_app_name = app_name or config.cluster.app_name

    # -----------------------------------------------------------------------
    # Step 0.5: Deploy PolymatheraCluster (LLM deployments, VCM, agent system)
    # -----------------------------------------------------------------------
    # Build RemoteLLMDeploymentConfig objects from YAML config
    remote_deployment_configs = []
    for rd in config.cluster.remote_deployments:
        remote_deployment_configs.append(RemoteLLMDeploymentConfig(
            model_name=rd.model_name,
            provider=rd.provider,
            api_key_env_var=rd.api_key_env_var,
            max_cached_pages=rd.max_cached_pages,
            max_cached_tokens=rd.max_cached_tokens,
            system_prompt=rd.system_prompt,
            ttl=rd.ttl,
            max_concurrent_requests=rd.max_concurrent_requests,
            num_replicas=rd.num_replicas,
        ))

    # Build LLMDeploymentConfig objects from YAML vllm_deployments
    vllm_deployment_configs = []
    for vd in config.cluster.vllm_deployments:
        vllm_deployment_configs.append(LLMDeploymentConfig.from_model_registry(
            model_name=vd.model_name,
            tensor_parallel_size=vd.tensor_parallel_size,
            quantization=vd.quantization,
            s3_bucket=vd.s3_bucket,
            gpu_memory_utilization=vd.gpu_memory_utilization,
            max_model_len=vd.max_model_len,
            kv_cache_capacity=vd.kv_cache_capacity,
            num_replicas=vd.num_replicas,
        ))

    # Build embedding config from YAML (if specified)
    embedding_deployment_config = None
    if config.cluster.embedding_config is not None:
        ec = config.cluster.embedding_config
        embedding_deployment_config = LLMDeploymentConfig.from_model_registry(
            model_name=ec.model_name,
            tensor_parallel_size=ec.tensor_parallel_size,
            quantization=ec.quantization,
            s3_bucket=ec.s3_bucket,
            gpu_memory_utilization=ec.gpu_memory_utilization,
            max_model_len=ec.max_model_len,
            kv_cache_capacity=ec.kv_cache_capacity,
            num_replicas=ec.num_replicas,
        )

    # Build remote embedding config from YAML (if specified)
    from colony.cluster.embedding import RemoteEmbeddingConfig, STEmbeddingDeploymentConfig, STEmbeddingModel

    remote_embedding_deployment_config = None
    if config.cluster.remote_embedding_config is not None:
        rec = config.cluster.remote_embedding_config
        remote_embedding_deployment_config = RemoteEmbeddingConfig(
            model_name=rec.model_name,
            provider=rec.provider,
            **({"api_key_env_var": rec.api_key_env_var} if rec.api_key_env_var else {}),
            dimensions=rec.dimensions,
            max_batch_size=rec.max_batch_size,
            max_concurrent_requests=rec.max_concurrent_requests,
            num_replicas=rec.num_replicas,
        )

    # Build SentenceTransformer embedding config from YAML (if specified)
    st_embedding_deployment_config = None
    if config.cluster.st_embedding_config is not None:
        stc = config.cluster.st_embedding_config
        st_embedding_deployment_config = STEmbeddingDeploymentConfig(
            model_name=STEmbeddingModel(stc.model_name),
            fallback_model=STEmbeddingModel(stc.fallback_model),
            enable_gpu=stc.enable_gpu,
            batch_size=stc.batch_size,
            max_concurrent_embeddings=stc.max_concurrent_embeddings,
            max_content_length=stc.max_content_length,
            chunk_large_files=stc.chunk_large_files,
            max_chunks_per_file=stc.max_chunks_per_file,
        )

    cluster_config = ClusterConfig(
        app_name=effective_app_name,
        vllm_deployments=vllm_deployment_configs,
        embedding_config=embedding_deployment_config,
        remote_embedding_config=remote_embedding_deployment_config,
        st_embedding_config=st_embedding_deployment_config,
        remote_deployments=remote_deployment_configs,
        cleanup_on_init=config.cluster.cleanup_on_init,
    )

    vcm_config = VCMConfig(
        caching_policy=config.vcm.caching_policy,
        page_storage_backend_type=config.vcm.page_storage_backend_type,
        page_storage_path=config.vcm.page_storage_path,
        page_fault_processing_interval_s=config.vcm.page_fault_processing_interval_s,
        metrics_collection_interval_s=config.vcm.metrics_collection_interval_s,
        reconciliation_interval_s=config.vcm.reconciliation_interval_s,
    )
    agent_system_config = AgentSystemConfig(
        max_retries=config.agent_system.max_retries,
        enable_sessions=config.agent_system.enable_sessions,
        default_session_ttl=config.agent_system.default_session_ttl,
    )

    polyconfig = PolymatheraClusterConfig(
        app_name=effective_app_name,
        llm_cluster_config=cluster_config,
        vcm_config=vcm_config,
        agent_system_config=agent_system_config,
        cleanup_on_init=config.cluster.cleanup_on_init,
    )

    # Display deployment info
    deploy_lines = [
        f"[bold]Step 0.5:[/bold] Deploying PolymatheraCluster",
        f"  App:    {effective_app_name}",
    ]
    for vd in vllm_deployment_configs:
        deploy_lines.append(
            f"  vLLM:   {vd.model_name} (tp={vd.tensor_parallel_size}, {vd.num_replicas} replica(s))"
        )
    for rd in remote_deployment_configs:
        deploy_lines.append(
            f"  Remote: {rd.model_name} ({rd.provider}, {rd.num_replicas} replica(s))"
        )
    if embedding_deployment_config:
        deploy_lines.append(
            f"  Embed:  {embedding_deployment_config.model_name} ({embedding_deployment_config.num_replicas} replica(s))"
        )
    if remote_embedding_deployment_config:
        deploy_lines.append(
            f"  Embed:  {remote_embedding_deployment_config.model_name} "
            f"({remote_embedding_deployment_config.provider}, API, "
            f"{remote_embedding_deployment_config.num_replicas} replica(s))"
        )
    if st_embedding_deployment_config:
        device = "GPU" if st_embedding_deployment_config.enable_gpu else "CPU"
        deploy_lines.append(
            f"  Embed:  {st_embedding_deployment_config.model_name.value} "
            f"(SentenceTransformer, {device})"
        )
    if not vllm_deployment_configs and not remote_deployment_configs:
        deploy_lines.append("  [yellow]WARNING: No LLM deployments configured[/yellow]")
    console.print()
    console.print(Panel(
        "\n".join(deploy_lines),
        title="[bold magenta]Cluster Deployment[/bold magenta]",
        border_style="magenta",
    ))

    with console.status("[magenta]Deploying cluster..."):
        polycluster = PolymatheraCluster(
            config=polyconfig,
            top_level=True,
        )
        await polycluster.deploy()

    console.print("  [green]OK[/green] — Cluster deployed")

    # -----------------------------------------------------------------------
    # Step 0.75: Create session for this test run
    # -----------------------------------------------------------------------
    if config.agent_system.enable_sessions:
        with console.status("[magenta]Creating session..."):
            sm_handle = get_session_manager(effective_app_name)
            session = await sm_handle.create_session(
                tenant_id=config.tenant_id,
            )
            config.session_id = session.session_id
        console.print(
            f"  [green]OK[/green] — Session created: {config.session_id}"
        )

    # -----------------------------------------------------------------------
    # Step 1: Page the codebase into VCM
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]Step 1:[/bold] Paging codebase into VCM\n"
        f"  Origin: {config.origin_url}\n"
        f"  Branch: {config.branch}  Commit: {config.commit}\n"
        f"  Scope:  {config.repo_id}\n"
        f"  Tenant: {config.tenant_id}",
        title="[bold cyan]VCM Context Paging[/bold cyan]",
        border_style="cyan",
    ))

    vcm_handle = get_vcm(effective_app_name)

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
            group_id=config.repo_id,  # Using repo_id as group_id for simplicity
            tenant_id=config.tenant_id,
            source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
            config=mmap_config,
            origin_url=config.origin_url,
            branch=config.branch,
            commit=config.commit,
        )

    if mmap_result.status in ("mapped", "already_mapped"):
        console.print(
            f"  [green]OK[/green] — {mmap_result.status}: "
            f"{mmap_result.message or 'codebase mapped successfully'}"
        )
    else:
        console.print(
            f"  [red]FAILED[/red] — {mmap_result.status}: {mmap_result.message}"
        )
        return [{"analysis_type": "paging", "status": "failed", "summary": mmap_result.message}]

    # -----------------------------------------------------------------------
    # Step 2: Spawn coordinator agents via AgentHandle
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]Step 2:[/bold] Spawning {len(config.analyses)} analysis coordinator(s)",
        title="[bold green]Agent Spawning[/bold green]",
        border_style="green",
    ))

    coordinator_handles: list[tuple[AnalysisConfig, AgentHandle]] = []

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
        self_concept_config = reg.get("self_concept", {})
        metadata = AgentMetadata(
            role=f"{reg['label']} coordinator",
            tenant_id=config.tenant_id,
            group_id=config.repo_id,
            session_id=config.session_id,
            run_id=config.run_id,
            goals=[f"Run {reg['label']} on {config.repo_id}"],
            max_iterations=analysis.max_iterations,
            self_concept=AgentSelfConcept(
                agent_id="",  # Placeholder — overwritten by ConsciousnessCapability
                name="",      # Placeholder — overwritten by ConsciousnessCapability
                **self_concept_config,
            ) if self_concept_config else None,
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

        # Resolve extra capabilities (with warnings for unknown names)
        # ConsciousnessCapability is always included — agents need identity context for planning.
        all_extra_caps = list(set(
            ["ConsciousnessCapability"]
            + analysis.extra_capabilities
            + config.hierarchy.extra_capabilities
        ))
        capability_paths = []
        for cap_name in all_extra_caps:
            if cap_name in EXTRA_CAPABILITIES_REGISTRY:
                capability_paths.append(EXTRA_CAPABILITIES_REGISTRY[cap_name]["path"])
            else:
                console.print(
                    f"  [yellow]WARNING[/yellow] Extra capability [bold]{cap_name}[/bold] "
                    f"not found in EXTRA_CAPABILITIES_REGISTRY — skipping.\n"
                    f"    Available: {', '.join(sorted(EXTRA_CAPABILITIES_REGISTRY.keys()))}"
                )

        # Resolve coordinator class and build blueprint
        agent_cls = _resolve_class(coord_class)
        cap_blueprints = [
            _resolve_class(path).bind() for path in capability_paths
        ]
        bp = agent_cls.bind(
            agent_type=coord_class,
            metadata=metadata,
            bound_pages=[],  # Coordinators don't bind to pages
            capability_blueprints=cap_blueprints,
        )

        coord_name = coord_class.rsplit(".", 1)[-1]
        console.print(
            f"  [green]+[/green] {reg['label']} — [cyan]{coord_name}[/cyan] "
            f"(max_agents={analysis.max_agents}, batching={analysis.batching_policy})"
        )

        # Spawn via AgentHandle.from_blueprint() — higher-level than raw
        # spawn_agents(), returns a handle for monitoring and communication.
        with console.status(f"  [green]Spawning {coord_name}..."):
            handle = await AgentHandle.from_blueprint(
                agent_blueprint=bp,
                session_id=config.session_id,
                run_id=config.run_id,
                app_name=effective_app_name
            )

        console.print(f"    [dim]agent_id: {handle.agent_id}[/dim]")
        coordinator_handles.append((analysis, handle))

    if not coordinator_handles:
        console.print("  [red]No valid analyses to run.[/red]")
        return results

    console.print(f"\n  Spawned {len(coordinator_handles)} coordinator(s)")

    # -----------------------------------------------------------------------
    # Step 3: Monitor progress via AgentHandle.run_streamed()
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        f"[bold]Step 3:[/bold] Monitoring analysis progress "
        f"(timeout={config.timeout_seconds}s)",
        title="[bold yellow]Monitoring[/bold yellow]",
        border_style="yellow",
    ))

    async def _get_run_cost(run_id: str) -> dict[str, Any]:
        """Query SessionManagerDeployment for accumulated cost data for a run."""
        try:
            sm_handle = get_session_manager(effective_app_name)
            run: AgentRun = await sm_handle.get_run(run_id=run_id)
            if run and run.resource_usage:
                usage = run.resource_usage
                return {
                    "cost_usd": usage.cost_usd,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                }
        except Exception as e:
            logger.debug(f"Could not retrieve cost data for run {run_id}: {e}")
        return {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    async def monitor_coordinator(
        analysis_cfg: AnalysisConfig,
        handle: AgentHandle,
    ) -> dict[str, Any]:
        """Monitor a single coordinator via handle.run_streamed().

        Sends the analysis task to the coordinator and streams events
        until completion, error, or timeout.
        """
        reg = ANALYSIS_REGISTRY[analysis_cfg.type]
        start = time.time()
        event_count = 0
        last_event_type = "unknown"

        def _make_result(
            status: str, agents_spawned: int, summary: str,
        ) -> dict[str, Any]:
            return {
                "analysis_type": reg["label"],
                "coordinator_id": handle.agent_id,
                "status": status,
                "agents_spawned": agents_spawned,
                "duration_seconds": time.time() - start,
                "summary": summary,
                "events": event_count,
            }

        try:
            async for event in handle.run_streamed(
                input_data={
                    "repo_id": config.repo_id,
                    "analysis_type": analysis_cfg.type,
                    **analysis_cfg.parameters,
                },
                timeout=float(config.timeout_seconds),
                session_id=config.session_id,
            ):
                event_count += 1
                last_event_type = event.event_type

                if config.verbose:
                    console.print(
                        f"    [dim]{handle.agent_id[:12]}… "
                        f"[{event.event_type}] {event.data}[/dim]"
                    )

                if event.event_type == "completed":
                    agents_spawned = 0
                    if isinstance(event.data, dict):
                        value = event.data.get("value", {})
                        if isinstance(value, dict):
                            agents_spawned = value.get("agents_spawned", 0)
                    result = _make_result("completed", agents_spawned, "Analysis complete")
                    cost_data = await _get_run_cost(config.run_id)
                    result.update(cost_data)
                    return result

                if event.event_type == "error":
                    error_msg = "Unknown error"
                    if isinstance(event.data, dict):
                        error_msg = event.data.get("error", error_msg)
                    result = _make_result("failed", 0, f"Error: {error_msg}")
                    cost_data = await _get_run_cost(config.run_id)
                    result.update(cost_data)
                    return result

                if event.event_type == "timeout":
                    result = _make_result("timeout", 0, f"Timed out after {config.timeout_seconds}s")
                    cost_data = await _get_run_cost(config.run_id)
                    result.update(cost_data)
                    return result

        except Exception as e:
            logger.error(f"Error monitoring {handle.agent_id}: {e}")
            result = _make_result("failed", 0, f"Monitor error: {e}")
            cost_data = await _get_run_cost(config.run_id)
            result.update(cost_data)
            return result

        # Stream ended without an explicit terminal event
        result = _make_result(
            "completed" if last_event_type != "error" else "failed",
            0,
            f"Stream ended (last event: {last_event_type})",
        )
        cost_data = await _get_run_cost(config.run_id)
        result.update(cost_data)
        return result

    # Run all monitors concurrently with budget tracking
    accumulated_cost_usd = 0.0
    budget_exceeded = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task(
            "Analyses running...",
            total=len(coordinator_handles),
        )

        monitor_tasks = [
            asyncio.create_task(monitor_coordinator(analysis_cfg, handle))
            for analysis_cfg, handle in coordinator_handles
        ]

        ### remaining_tasks = list(monitor_tasks)
        ### while remaining_tasks:
        ###     done, remaining_tasks_set = await asyncio.wait(
        ###         remaining_tasks, return_when=asyncio.FIRST_COMPLETED,
        ###     )
        ###     remaining_tasks = list(remaining_tasks_set)

        ###     for task in done:
        ###         result = task.result()
        for coro in asyncio.as_completed(monitor_tasks):
            result = await coro
            results.append(result)
            progress.update(progress_task, advance=1)

            label = result.get("analysis_type", "?")
            status = result.get("status", "?")
            cost = result.get("cost_usd", 0.0)
            accumulated_cost_usd += cost

            style = {
                "completed": "green",
                "failed": "red",
                "timeout": "yellow",
                "budget_exceeded": "red",
            }.get(status, "dim")

            cost_str = f" (${cost:.4f})" if cost > 0 else ""
            console.print(f"  [{style}]{label}: {status}{cost_str}[/{style}]")

            # Budget enforcement (soft stop)
            if config.budget_usd is not None and not budget_exceeded:
                warn_threshold = config.budget_usd * config.warn_budget_pct
                if accumulated_cost_usd >= config.budget_usd:
                    budget_exceeded = True
                    console.print(
                        f"\n  [bold red]BUDGET EXCEEDED:[/bold red] "
                        f"${accumulated_cost_usd:.4f} >= ${config.budget_usd:.2f} limit. "
                        f"Cancelling remaining analyses."
                    )
                    # Cancel remaining monitor tasks
                    for t in remaining_tasks:
                        t.cancel()
                    # Record skipped analyses
                    for t in remaining_tasks:
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass
                    # Add budget_exceeded entries for cancelled analyses
                    completed_ids = {r["coordinator_id"] for r in results}
                    for analysis_cfg, handle in coordinator_handles:
                        if handle.agent_id not in completed_ids:
                            reg = ANALYSIS_REGISTRY[analysis_cfg.type]
                            results.append({
                                "analysis_type": reg["label"],
                                "coordinator_id": handle.agent_id,
                                "status": "budget_exceeded",
                                "agents_spawned": 0,
                                "duration_seconds": 0,
                                "summary": f"Skipped — budget exceeded (${accumulated_cost_usd:.4f}/${config.budget_usd:.2f})",
                                "events": 0,
                                "cost_usd": 0.0,
                            })
                    remaining_tasks = []
                    break
                elif accumulated_cost_usd >= warn_threshold:
                    console.print(
                        f"  [yellow]WARNING:[/yellow] Cost ${accumulated_cost_usd:.4f} "
                        f"has reached {config.warn_budget_pct*100:.0f}% of "
                        f"${config.budget_usd:.2f} budget"
                    )

    # Print total cost summary
    if accumulated_cost_usd > 0:
        console.print(
            f"\n  [bold]Total accumulated cost:[/bold] ${accumulated_cost_usd:.4f}"
        )

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
    origin_url: Optional[str] = typer.Option(
        None,
        "--origin-url",
        help="Git repository URL (HTTPS) for the codebase to analyze.",
    ),
    local_repo: Optional[str] = typer.Option(
        None,
        "--local-repo",
        help="Path to a local git repository. Equivalent to --origin-url file://<path>.",
    ),
    branch: str = typer.Option(
        "main",
        "--branch",
        help="Git branch to check out.",
    ),
    commit: str = typer.Option(
        "HEAD",
        "--commit",
        help="Git commit SHA to check out (defaults to branch HEAD).",
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
    working_dir: Optional[str] = typer.Option(
        None,
        "--working-dir", "-w",
        help=(
            "Directory containing user code (custom agents, capabilities) to "
            "distribute to Ray worker nodes.  Ray zips this on the driver and "
            "uploads it to every worker so its modules are importable.  This is "
            "NOT for the codebase being analyzed — that stays on shared storage "
            "(EFS) and is accessed via VCM paging."
        ),
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
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        help="Maximum budget in USD for remote LLM API calls. Analyses are stopped when exceeded.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Run an integration test of the multi-agent framework on a codebase.

    This command clones the repository via GitStorage, pages it into VCM,
    spawns coordinator agents that create teams of specialized workers,
    and runs complete analysis workflows.

    Exactly one of --origin-url or --local-repo must be provided (unless
    the URL is specified in the YAML config file via the origin_url field).

    [bold]Quick Start (remote repo):[/bold]

        polymath run --origin-url https://github.com/org/repo --analysis impact

    [bold]Quick Start (local repo):[/bold]

        polymath run --local-repo /path/to/codebase --analysis impact

    [bold]With Config File:[/bold]

        polymath run --local-repo /path/to/codebase --config test.yaml

    [bold]Dry Run (show hierarchy only):[/bold]

        polymath run --origin-url https://github.com/org/repo --dry-run
    """
    if verbose:
        logging.getLogger("polymath").setLevel(logging.DEBUG)
        logging.getLogger("colony").setLevel(logging.DEBUG)

    # Build configuration
    if config:
        try:
            test_config = load_config_from_yaml(config)
            console.print(f"[dim]Loaded config from {config}[/dim]")
            # CLI options override YAML values
            if working_dir is not None:
                test_config.working_dir = working_dir
            if budget is not None:
                test_config.budget_usd = budget
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
            working_dir=working_dir,
            budget_usd=budget,
        )

    # Resolve --origin-url / --local-repo into test_config.origin_url.
    # CLI options override YAML values.
    if origin_url and local_repo:
        console.print("[red]Error:[/red] --origin-url and --local-repo are mutually exclusive.")
        raise typer.Exit(1)

    if origin_url:
        test_config.origin_url = origin_url
    elif local_repo:
        test_config.origin_url = f"file://{Path(local_repo).resolve()}"

    if branch != "main" or not test_config.branch:
        test_config.branch = branch
    if commit != "HEAD" or not test_config.commit:
        test_config.commit = commit

    if not test_config.origin_url:
        console.print(
            "[red]Error:[/red] No repository specified. "
            "Use --origin-url <URL> or --local-repo <path> "
            "(or set origin_url in the YAML config)."
        )
        raise typer.Exit(1)

    # Display planned hierarchy
    console.print()
    tree = build_analysis_tree(test_config)
    console.print(tree)
    console.print()

    if dry_run:
        console.print("[dim]Dry run complete. No agents were spawned.[/dim]")
        raise typer.Exit(0)

    # Run the integration test
    budget_line = (
        f"\n[bold]Budget:[/bold]   ${test_config.budget_usd:.2f}"
        if test_config.budget_usd is not None
        else "\n[bold]Budget:[/bold]   unlimited"
    )
    console.print(Panel(
        f"[bold]Origin:[/bold]   {test_config.origin_url}\n"
        f"[bold]Branch:[/bold]   {test_config.branch}\n"
        f"[bold]Commit:[/bold]   {test_config.commit}\n"
        f"[bold]Repo ID:[/bold]  {test_config.repo_id}\n"
        f"[bold]Session:[/bold]  {test_config.session_id}\n"
        f"[bold]Run ID:[/bold]   {test_config.run_id}\n"
        f"[bold]Timeout:[/bold]  {test_config.timeout_seconds}s"
        f"{budget_line}",
        title="[bold]Starting Integration Test[/bold]",
        border_style="bright_blue",
    ))

    try:
        test_results = asyncio.run(run_integration_test(
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

    total_cost = sum(r.get("cost_usd", 0.0) for r in test_results)
    results_payload = {
        "session_id": test_config.session_id,
        "run_id": test_config.run_id,
        "repo_id": test_config.repo_id,
        "origin_url": test_config.origin_url,
        "branch": test_config.branch,
        "commit": test_config.commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cost_usd": total_cost,
        "budget_usd": test_config.budget_usd,
        "results": test_results,
    }

    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    console.print(f"\n[dim]Results saved to {results_file}[/dim]")

    # Exit with error code if any analysis failed
    failures = [r for r in test_results if r.get("status") in ("failed", "timeout")]
    budget_skipped = [r for r in test_results if r.get("status") == "budget_exceeded"]
    if failures:
        console.print(f"\n[yellow]{len(failures)} analysis(es) failed or timed out.[/yellow]")
        raise typer.Exit(1)
    if budget_skipped:
        console.print(
            f"\n[yellow]{len(budget_skipped)} analysis(es) skipped due to budget limit.[/yellow]"
        )
        raise typer.Exit(2)

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
