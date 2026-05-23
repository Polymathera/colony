"""Typed configuration for the agent layer.

This module hosts ``ConfigComponent``s that the rest of the agent system
reads (mission registry today; plugin/sandbox/job configs in later steps).
The registry-style ``MissionRegistryConfig`` replaces the previously-
hardcoded ``MISSION_REGISTRY`` dict in ``cli/polymath.py``; the dict is
re-exported there so existing call sites keep indexing it as a plain
``dict[str, dict]`` until step 11 lifts ``TestConfig``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..distributed.config import (
    ConfigComponent,
    Mutability,
    Ownership,
    Tier,
    register_polymathera_config,
    tier_metadata,
)
from .sandbox_images import DockerImageSpec, ScriptSpec  # noqa: F401 — re-exported below


# ---------------------------------------------------------------------------
# Mission registry — typed surface
# ---------------------------------------------------------------------------


class MissionSelfConcept(BaseModel):
    """The self-concept fields a coordinator agent reads at spawn time."""

    model_config = ConfigDict(extra="forbid")

    description: str
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class MissionSpec(BaseModel):
    """Definition of one mission type (coordinator + worker classes + metadata).

    Worker capabilities are documentation-only — workers self-configure their
    capabilities in their ``initialize()`` methods. Coordinator capabilities
    are passed to the agent system so it can wire the named capabilities at
    spawn time.

    Schema is the single source of truth for every registration mechanism that
    surfaces a mission to the SessionAgent (the colony-builtin dict, the
    ``polymathera.mission_types`` entry-point group, and L1-A's
    :func:`polymathera.colony.design_monorepo.extensions.discover_missions`).
    ``extra="forbid"`` keeps drift visible: typo'd keys in any of those paths
    surface as validation errors at load time rather than silently passing.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    description: str
    coordinator_v1: str
    coordinator_v2: str
    worker: str
    coordinator_capabilities: list[str] = Field(default_factory=list)
    worker_capabilities: list[str] = Field(default_factory=list)
    extra_metadata_keys: list[str] = Field(default_factory=list)
    self_concept: MissionSelfConcept


# Built-in missions shipped by colony. CPS and other extensions add more via
# the ``polymathera.config_components`` entry-point group (see
# ``distributed.config.extensions``) or via the existing legacy
# ``polymathera.mission_types`` group consumed by
# ``agents.mission_registry.get_mission_registry``.

# Maps mission types to their coordinator and worker agent class paths.
# These are the fully-qualified Python class paths used by the agent system
# to instantiate agent classes dynamically.
#
# NOTE on worker_capabilities: These are DOCUMENTATION-ONLY. Workers are
# self-configuring — each worker agent class adds its own capabilities in its
# initialize() method (e.g., ChangeImpactAnalysisAgent extends HypothesisGameAgent
# which adds HypothesisGameProtocol automatically). The lists here are used by
# the `describe` and `list` commands to show what each worker provides.

_BUILTIN_MISSIONS: dict[str, dict[str, Any]] = {
    "impact": {
        "label": "Change Impact Analysis",
        "description": (
            "Analyzes the ripple effects of code changes across a codebase. "
            "Uses multi-hop dependency propagation, hypothesis games for "
            "validating critical impacts, and game-theoretic merge policies."
        ),
        "coordinator_v1": "polymathera.colony.samples.code_analysis.impact.coordinator.ChangeImpactAnalysisCoordinator",
        "coordinator_v2": "polymathera.colony.samples.code_analysis.impact.coordinator.ChangeImpactAnalysisCoordinator",
        "worker": "polymathera.colony.samples.code_analysis.impact.page_analyzer.ChangeImpactAnalysisAgent",
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
            "DynamicGameCapability",  # via HypothesisGameAgent base class
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
        "coordinator_v1": "polymathera.colony.samples.code_analysis.slicing.agents.ProgramSlicingCoordinator",
        "coordinator_v2": "polymathera.colony.samples.code_analysis.slicing.agents.ProgramSlicingCoordinator",
        "worker": "polymathera.colony.samples.code_analysis.slicing.agents.ProgramSlicingAgent",
        "coordinator_capabilities": [
            "SlicingAnalysisCapability",
            "MergeCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
            "PageGraphCapability",
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
        "coordinator_v1": "polymathera.colony.samples.code_analysis.compliance.agents.ComplianceAnalysisCoordinator",
        "coordinator_v2": "polymathera.colony.samples.code_analysis.compliance.agents.ComplianceAnalysisCoordinator",
        "worker": "polymathera.colony.samples.code_analysis.compliance.agents.ComplianceAnalysisAgent",
        "coordinator_capabilities": [
            "ComplianceVCMCapability",
            "MergeCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
            "PageGraphCapability",
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
        "coordinator_v1": "polymathera.colony.samples.code_analysis.intent.agents.IntentInferenceCoordinator",
        "coordinator_v2": "polymathera.colony.samples.code_analysis.intent.agents.IntentInferenceCoordinator",
        "worker": "polymathera.colony.samples.code_analysis.intent.agents.IntentInferenceAgent",
        "coordinator_capabilities": [
            "IntentAnalysisCapability",
            "MergeCapability",
            "SynthesisCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
            "PageGraphCapability",
        ],
        "worker_capabilities": [
            "IntentInferenceCapability",
            "MergeCapability",
            "DynamicGameCapability",
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
        "coordinator_v1": "polymathera.colony.samples.code_analysis.contracts.agents.ContractInferenceCoordinator",
        "coordinator_v2": "polymathera.colony.samples.code_analysis.contracts.agents.ContractInferenceCoordinator",
        "worker": "polymathera.colony.samples.code_analysis.contracts.agents.ContractInferenceAgent",
        "coordinator_capabilities": [
            "ContractAnalysisCapability",
            "MergeCapability",
            "SynthesisCapability",
            "WorkingSetCapability",
            "AgentPoolCapability",
            "PageGraphCapability",
        ],
        "worker_capabilities": [
            "ContractInferenceCapability",
            "MergeCapability",
            "DynamicGameCapability",
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
        "coordinator_v1": "polymathera.colony.samples.code_analysis.basic.coordinator.CodeAnalysisCoordinator",
        "coordinator_v2": "polymathera.colony.samples.code_analysis.basic.coordinator.CodeAnalysisCoordinatorV2",
        "worker": "polymathera.colony.samples.code_analysis.basic.cluster_analyzer.ClusterAnalyzer",
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


def _builtin_missions() -> dict[str, MissionSpec]:
    """Build typed defaults from ``_BUILTIN_MISSIONS``."""
    return {key: MissionSpec(**value) for key, value in _BUILTIN_MISSIONS.items()}


@register_polymathera_config(path="mission_registry")
class MissionRegistryConfig(ConfigComponent):
    """Registered repository of mission types.

    Defaults are the colony built-ins lifted from the legacy ``MISSION_REGISTRY``
    dict in ``cli/polymath.py``. Extensions add entries either by registering
    additional ``MissionSpec`` values into this component (via the
    ``polymathera.config_components`` entry-point group) or via the legacy
    ``polymathera.mission_types`` discovery (see
    ``agents.mission_registry.get_mission_registry``).
    """

    missions: dict[str, MissionSpec] = Field(
        default_factory=_builtin_missions,
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR,
            ownership=Ownership.COLONY,
            mutability=Mutability.RELOADABLE,
        ),
    )


# Backwards-compatible re-export. ``cli.polymath.MISSION_REGISTRY`` and
# ``agents.mission_registry.get_mission_registry`` index this dict as the
# colony-builtin baseline. Step 11 (lifting ``TestConfig``) removes the dict
# alias and routes everything through ``MissionRegistryConfig``.
MISSION_REGISTRY: dict[str, dict[str, Any]] = _BUILTIN_MISSIONS


# ---------------------------------------------------------------------------
# Plugin / skill discovery roots
# ---------------------------------------------------------------------------

# Defaults must mirror the legacy constants in
# ``agents/patterns/capabilities/user_plugin.py`` so wiring through this
# config does not change the resolved roots in any existing deployment.
_DEFAULT_PLUGINS_SYSTEM_ROOT = "/etc/colony"
_DEFAULT_PLUGINS_USER_ROOT = "~/.colony"
_DEFAULT_PLUGINS_WORKSPACE_ROOT = "/workspace/.colony"


@register_polymathera_config(path="plugins")
class PluginsConfig(ConfigComponent):
    """Skill / plugin discovery roots.

    The three ``*_root`` fields name parent directories under which a
    ``skills/`` or ``plugins/`` subdirectory is searched. Priority is
    fixed: workspace (session) > user > system. ``extra_*`` paths are
    appended at SYSTEM priority — used to ship bundled samples without
    shadowing user-installed skills.

    Tiers vary per field: ``workspace_root`` is per-session, ``user_root``
    per-tenant, the rest operator-set; uniform tier metadata is left for
    a follow-up step that wires per-field overlays.
    """

    workspace_root: str = Field(default=_DEFAULT_PLUGINS_WORKSPACE_ROOT)
    user_root: str = Field(default=_DEFAULT_PLUGINS_USER_ROOT)
    system_root: str = Field(default=_DEFAULT_PLUGINS_SYSTEM_ROOT)
    extra_skill_roots: list[str] = Field(
        default_factory=list,
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )
    extra_plugin_roots: list[str] = Field(
        default_factory=list,
        json_schema_extra=tier_metadata(tier=Tier.L1_OPERATOR),
    )


async def get_plugins_config() -> PluginsConfig:
    """Async fetch of the registered ``PluginsConfig`` — initializes the
    global ``ConfigurationManager`` first so the returned component
    reflects YAML + env-var values, not bare class defaults."""
    return await _get_component_or_default("plugins", PluginsConfig)


# ---------------------------------------------------------------------------
# Sandbox image registry
#
# ``DockerImageSpec`` / ``ScriptSpec`` are the shared schema —
# ``colony.agents.sandbox_images`` is the single source of truth used
# both here (operator-YAML validation) and by
# ``agents.patterns.capabilities._sandbox.registry.DockerImageRegistry``.
# ---------------------------------------------------------------------------


@register_polymathera_config(path="sandbox_images")
class DockerImageRegistryConfig(ConfigComponent):
    """Allowed sandbox images, keyed by ``role``.

    Default is **empty** — preserving the existing
    ``SandboxedShellCapability`` semantics where an unmounted
    ``sandbox-images.yaml`` resolves to an empty registry. Operator
    populates this either via the operator YAML
    (``sandbox_images: {images: [...]}``) or by mounting the legacy
    ``sandbox-images.yaml`` at ``/etc/colony/sandbox-images.yaml`` (the
    capability's fallback path).
    """

    images: list[DockerImageSpec] = Field(
        default_factory=list,
        json_schema_extra=tier_metadata(
            tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE,
        ),
    )


async def get_sandbox_image_registry_config() -> DockerImageRegistryConfig:
    """Async fetch — see :func:`get_plugins_config`."""
    return await _get_component_or_default("sandbox_images", DockerImageRegistryConfig)


# ---------------------------------------------------------------------------
# Capability secrets (env-bound)
# ---------------------------------------------------------------------------


@register_polymathera_config(path="capabilities.web_search")
class WebSearchConfig(ConfigComponent):
    """Configuration for ``WebSearchCapability`` (Tavily backend)."""

    api_key: str = Field(
        default="",
        json_schema_extra={"env": "TAVILY_API_KEY", "optional": True},
    )


async def get_web_search_config() -> WebSearchConfig:
    return await _get_component_or_default("capabilities.web_search", WebSearchConfig)


@register_polymathera_config(path="capabilities.github")
class GitHubAuthConfig(ConfigComponent):
    """Configuration for ``GitHubCapability`` (GitHub App credentials)."""

    app_id: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_APP_ID", "optional": True},
    )
    installation_id: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_INSTALLATION_ID", "optional": True},
    )
    private_key_pem: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_PRIVATE_KEY_PEM", "optional": True},
    )


async def get_github_auth_config() -> GitHubAuthConfig:
    return await _get_component_or_default("capabilities.github", GitHubAuthConfig)


@register_polymathera_config(path="memory.chroma")
class ChromaConfig(ConfigComponent):
    """Configuration for the Chroma memory backend."""

    persist_dir: str = Field(
        default="/tmp/colony_chromadb",
        json_schema_extra={"env": "CHROMA_PERSIST_DIR", "optional": True},
    )


async def get_chroma_config() -> ChromaConfig:
    return await _get_component_or_default("memory.chroma", ChromaConfig)


# ---------------------------------------------------------------------------
# Async-fetch helper. The underlying ``get_component_or_default`` is sync
# and degrades to bare defaults when the global ``ConfigurationManager`` has
# not loaded YAML / env vars yet — which is the state in fresh deployment
# worker processes. This wrapper awaits the shared init coroutine first so
# the returned component reflects the operator's actual configuration.
# ---------------------------------------------------------------------------


async def _get_component_or_default(path: str, cls: type[ConfigComponent]):
    """Forward to the shared helper in ``distributed.config.manager``."""
    from ..distributed import get_initialized_polymathera
    from ..distributed.config import get_component_or_default
    await get_initialized_polymathera()
    return get_component_or_default(path, cls)


__all__ = (
    "MISSION_REGISTRY",
    "MissionRegistryConfig",
    "MissionSelfConcept",
    "MissionSpec",
    "ChromaConfig",
    "GitHubAuthConfig",
    "PluginsConfig",
    "DockerImageSpec",
    "ScriptSpec",
    "DockerImageRegistryConfig",
    "WebSearchConfig",
    "get_chroma_config",
    "get_github_auth_config",
    "get_plugins_config",
    "get_sandbox_image_registry_config",
    "get_web_search_config",
)
