"""Typed configuration for the agent layer.

This module hosts ``ConfigComponent``s that the rest of the agent system
reads (mission registry today; plugin/sandbox/job configs in later steps).
The registry-style ``MissionRegistryConfig`` replaces the previously-
hardcoded ``MISSION_REGISTRY`` dict in ``cli/polymath.py``; the dict is
re-exported there so existing call sites keep indexing it as a plain
``dict[str, dict]`` until step 11 lifts ``TestConfig``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..distributed.config import (
    ConfigComponent,
    Mutability,
    Ownership,
    Tier,
    register_polymathera_config,
    tier_metadata,
)
from .metadata_parameters import ParameterScope, ParameterSpec
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


# ---------------------------------------------------------------------------
# Mission execution policy — declarative guardrails the spawn gate
# enforces deterministically. See
# ``colony/mission_and_action_guardrails_plan.md`` (Part 1) for the
# motivation. The schema lives here so both the static registry
# (``MissionSpec.execution_policy``) and the per-coordinator
# ``MISSION_EXECUTION_POLICY`` ClassVar share one source of truth.
# ---------------------------------------------------------------------------


class MissionConcurrencyScope(str, Enum):
    """Boundary at which ``max_concurrent_instances`` is counted.

    Picked per-mission. Matches the existing scope vocabulary used by
    ``BlackboardScope`` / ``ScopeUtils`` so the same scope names work
    across the codebase.
    """

    GLOBAL  = "global"
    TENANT  = "tenant"
    COLONY  = "colony"
    SESSION = "session"
    AGENT   = "agent"


class MissionExecutionPolicy(BaseModel):
    """Declarative spawn-gate + lifecycle constraints for one mission.

    Read by ``SessionOrchestratorCapability.spawn_mission`` and
    ``AgentPoolCapability.create_agent`` BEFORE the agent system
    instantiates the coordinator. Every field has a conservative
    default so coordinators that do not declare a policy keep the
    pre-policy semantics (single instance per parent agent, no
    auto-chained modes, no cost cap, etc.).

    The recommended source of truth is a
    ``MISSION_EXECUTION_POLICY: ClassVar[MissionExecutionPolicy]`` on
    the coordinator class. The registry's ``MissionSpec.execution_policy``
    overrides the ClassVar when set, so operators can tighten caps
    (e.g. lower ``max_llm_cost_usd``) without subclassing.
    """

    model_config = ConfigDict(extra="forbid")

    # --- concurrency ---------------------------------------------------
    max_concurrent_instances: int | None = Field(
        default=1,
        description=(
            "Maximum number of in-flight coordinator instances per "
            "``concurrency_scope``. ``None`` = unbounded; ``1`` = "
            "singleton. Default ``1`` matches the existing 'unique "
            "child role' framework guard."
        ),
    )
    concurrency_scope: MissionConcurrencyScope = Field(
        default=MissionConcurrencyScope.AGENT,
        description=(
            "Boundary the spawn gate counts ``max_concurrent_instances`` "
            "against. ``AGENT`` matches the legacy guard."
        ),
    )
    on_concurrency_violation: Literal[
        "reject", "queue", "preempt_oldest", "return_existing",
    ] = Field(
        default="reject",
        description=(
            "Action when a new spawn would exceed the cap. "
            "``return_existing`` is the right shape for idempotent "
            "missions; ``reject`` is the safe default for everything "
            "else."
        ),
    )

    # --- preemption ----------------------------------------------------
    preemptible: bool = Field(
        default=False,
        description=(
            "When True, the scheduler may cancel this mission to free "
            "room for a higher-priority spawn. Default False — the "
            "mission runs to natural completion or explicit cancel."
        ),
    )
    preemption_grace_seconds: float = Field(
        default=10.0,
        ge=0.0,
        description=(
            "Seconds to wait for a clean shutdown after a preemption "
            "signal before the runtime hard-cancels the coordinator."
        ),
    )

    # --- interruption --------------------------------------------------
    interruptible: bool = Field(
        default=True,
        description=(
            "When True, the operator can cancel the mission mid-flight "
            "via a chat message. When False, the mission rejects cancel "
            "requests until it reaches a safe boundary."
        ),
    )
    cancel_propagates_to_children: bool = Field(
        default=True,
        description=(
            "When True, cancelling this mission also cancels every "
            "child agent it spawned. Off only for missions that need "
            "to leave their children running (rare)."
        ),
    )

    # --- chaining / sequencing -----------------------------------------
    chains_with_modes: list[str] | None = Field(
        default=None,
        description=(
            "When set, declares 'this mission's modes auto-chain "
            "internally — do NOT spawn one coordinator per mode'. The "
            "spawn gate rejects subsequent spawns of the same mission "
            "type whose ``mission_params['mode']`` differs only by an "
            "already-running mode in the same scope. The exact list of "
            "mode strings declared here is what the gate matches "
            "against."
        ),
    )

    # --- dependencies + ordering ---------------------------------------
    requires_mission_complete: list[str] = Field(
        default_factory=list,
        description=(
            "Mission-type names that must have reached terminal state "
            "in the same scope before this one can spawn. Used for the "
            "'bootstrap before assignments' pattern when modes are NOT "
            "chained internally."
        ),
    )

    # --- resource gates ------------------------------------------------
    max_runtime_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Hard wall-clock cap. ``None`` = no cap.",
    )
    max_llm_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Hard LLM budget cap in USD. ``None`` = no cap.",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Hard cap on coordinator planner iterations. ``None`` = "
            "fall back to ``AgentMetadata.max_iterations``."
        ),
    )

    # --- idempotency / reentrance --------------------------------------
    idempotent: bool = Field(
        default=False,
        description=(
            "Re-running with the same ``mission_params`` is safe. When "
            "True, the spawn gate may return the existing run's handle "
            "instead of spawning a new one (subject to "
            "``on_concurrency_violation``)."
        ),
    )
    reentrant: bool = Field(
        default=False,
        description=(
            "The mission can spawn a sub-mission of its own type. "
            "Default False guards against unbounded recursion when an "
            "LLM planner re-invokes itself."
        ),
    )

    # --- approval gates ------------------------------------------------
    requires_human_approval_before: list[str] = Field(
        default_factory=list,
        description=(
            "Action-key prefixes that MUST be preceded by a "
            "``request_human_approval`` → ``approval_granted`` round "
            "in the same run. Enforced by ``ApprovalRequiredGuardrail`` "
            "(Part 2 of the guardrails plan)."
        ),
    )

    # --- side-effect scope (advisory; informs UI + dry-run) ------------
    mutates_remote: bool = Field(
        default=False,
        description=(
            "True when the mission touches external services (GitHub "
            "REST, etc.). UI hint only."
        ),
    )
    mutates_local_only: bool = Field(
        default=False,
        description=(
            "True when the mission only edits the per-agent clone "
            "without ever pushing or calling external services."
        ),
    )
    pure: bool = Field(
        default=False,
        description=(
            "True when the mission has no side effects (read-only "
            "queries, analyses)."
        ),
    )

    # --- cooldown ------------------------------------------------------
    cooldown_seconds_after_completion: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Rate-limit consecutive spawns of the same mission in the "
            "same scope by N seconds."
        ),
    )

    @model_validator(mode="after")
    def _side_effect_flags_are_consistent(
        self,
    ) -> "MissionExecutionPolicy":
        truthy = [
            name for name in ("mutates_remote", "mutates_local_only", "pure")
            if getattr(self, name)
        ]
        if len(truthy) > 1:
            raise ValueError(
                f"MissionExecutionPolicy: at most one of "
                f"``mutates_remote`` / ``mutates_local_only`` / "
                f"``pure`` may be True; got {truthy!r}.",
            )
        return self

    @model_validator(mode="after")
    def _chains_with_modes_entries_are_non_empty(
        self,
    ) -> "MissionExecutionPolicy":
        if self.chains_with_modes is None:
            return self
        if not self.chains_with_modes:
            raise ValueError(
                "MissionExecutionPolicy.chains_with_modes: empty list "
                "is ambiguous — use ``None`` to disable the gate.",
            )
        offenders = [m for m in self.chains_with_modes if not m]
        if offenders:
            raise ValueError(
                "MissionExecutionPolicy.chains_with_modes: every entry "
                "must be a non-empty mode string.",
            )
        return self


class MissionSpec(BaseModel):
    """Definition of one mission type (coordinator + worker classes + metadata).

    Worker capabilities are documentation-only — workers self-configure their
    capabilities in their ``initialize()`` methods. Coordinator capabilities
    are passed to the agent system so it can wire the named capabilities at
    spawn time.

    ``worker`` is optional (default ``""``): some missions are LLM-planner-
    driven on the coordinator alone and never spawn a separate worker class
    (e.g. ``project_planning`` — the coordinator drives the full
    propose-approve-apply flow via existing action surfaces). Downstream
    readers (``cli/polymath.py``, ``session_agent.py``'s
    ``available_missions``) already default to empty string.

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
    worker: str = ""
    coordinator_capabilities: list[str] = Field(default_factory=list)
    worker_capabilities: list[str] = Field(default_factory=list)
    # CALLER-scoped parameters the spawn caller (LLM planner via
    # ``spawn_mission``, or REST handler via ``/api/jobs/submit``)
    # must supply in ``mission_params``. Replaces the legacy
    # ``extra_metadata_keys: list[str]`` field — the typed form
    # carries descriptions (rendered into the planner prompt by
    # ``SessionOrchestratorCapability._refresh_available_missions``),
    # defaults (Pydantic-style — required iff no default of any
    # kind), and structural shape that makes drift visible at
    # registry-load time. COLONY/SESSION-scoped needs are properties
    # of the mounted capabilities — declared on the capability's
    # ``AGENT_METADATA_PARAMS`` ClassVar — and flow automatically
    # via the inheritance gate in ``AgentPoolCapability.create_agent``.
    # See ``colony/agent_metadata_parameter_spec_plan.md``.
    caller_parameters: list[ParameterSpec] = Field(default_factory=list)
    self_concept: MissionSelfConcept
    # Optional spawn-gate policy override. When ``None`` (the default),
    # the resolved policy is whatever the coordinator class declares
    # via its ``MISSION_EXECUTION_POLICY`` ClassVar — or the
    # ``MissionExecutionPolicy()`` default factory if the class
    # doesn't declare one either. Operators set this to tighten caps
    # (e.g. ``max_llm_cost_usd``) without subclassing the coordinator.
    # See ``colony/mission_and_action_guardrails_plan.md`` Part 1.
    execution_policy: MissionExecutionPolicy | None = None

    @model_validator(mode="after")
    def _caller_parameters_are_caller_scoped(self) -> "MissionSpec":
        offenders = [
            p.name for p in self.caller_parameters
            if p.scope is not ParameterScope.CALLER
        ]
        if offenders:
            raise ValueError(
                f"MissionSpec {self.label!r}: caller_parameters entries "
                f"must have scope=CALLER, got non-CALLER for {offenders!r}. "
                f"COLONY/SESSION-scoped needs belong on the mounted "
                f"capabilities' AGENT_METADATA_PARAMS, not on the mission "
                f"spec."
            )
        return self


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
        "caller_parameters": [
            {
                "name": "changes",
                "scope": "caller",
                "description": (
                    "List of changed files / regions to analyse — "
                    "the unit of work for impact propagation."
                ),
                "json_type": "array",
                "default": None,
            },
            {
                "name": "change_description",
                "scope": "caller",
                "description": (
                    "Free-text explanation of the changes. Hint to "
                    "the worker LLMs when ranking impact severity."
                ),
                "default": None,
            },
        ],
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
        "caller_parameters": [
            {
                "name": "slice_criteria",
                "scope": "caller",
                "description": (
                    "Target variable / expression / statement(s) the "
                    "slice is computed against. The minimal subset "
                    "criterion."
                ),
                "default": None,
            },
        ],
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
        "caller_parameters": [
            {
                "name": "compliance_types",
                "scope": "caller",
                "description": (
                    "List of compliance dimensions to check — e.g. "
                    "license / regulatory / security. Empty / absent "
                    "lets the coordinator infer the relevant set."
                ),
                "json_type": "array",
                "default": None,
            },
        ],
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
        "caller_parameters": [
            {
                "name": "granularity",
                "scope": "caller",
                "description": (
                    "Resolution at which intent is inferred — e.g. "
                    "per-function / per-module / per-file. Selects "
                    "the worker's analysis grain."
                ),
                "default": None,
            },
        ],
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
        "caller_parameters": [
            {
                "name": "formalism",
                "scope": "caller",
                "description": (
                    "Formalism level for the inferred contracts — "
                    "e.g. informal prose / Hoare-style triples / Z. "
                    "Picks the worker's output target."
                ),
                "default": None,
            },
        ],
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
        "caller_parameters": [],
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
    "project_planning": {
        "label": "Project Planning",
        "description": (
            "Bootstrap or revise the design roadmap from the project's "
            "design context (objectives + constraints + requirements), "
            "create matching GitHub issues + Project items, propose "
            "colony/user task assignments, and decompose high-level "
            "GitHub issues into smaller sub-issues — every mutation "
            "gated on a single human-approval round so the user "
            "reviews + approves (or rejects) before anything is "
            "written. Four modes via mission_params['mode']: bootstrap "
            "(initial roadmap from design context), refresh "
            "(bidirectional sync between ROADMAP.md and GitHub "
            "issues), assignments (propose colony-vs-user assignment "
            "per open roadmap-linked issue), decompose (break "
            "high-level issues with many unchecked subtasks into "
            "linked sub-issues)."
        ),
        "coordinator_v1": "polymathera.colony.agents.missions.project_planning.coordinator.ProjectPlanningCoordinator",
        "coordinator_v2": "polymathera.colony.agents.missions.project_planning.coordinator.ProjectPlanningCoordinator",
        # No separate worker class: the LLM planner on the coordinator
        # drives the full propose-approve-apply flow over the existing
        # P5 action surfaces. ``MissionSpec.worker`` defaults to "" for
        # this mission shape.
        "coordinator_capabilities": [
            "DesignProcessCapability",
            "SystemDesignCapability",
            "GitHubCapability",
            "HumanApprovalCapability",
            "RepoStateProvider",
            "DesignCheckpointer",
            "ToolBuilder",
        ],
        "worker_capabilities": [],
        "caller_parameters": [
            {
                "name": "mode",
                "scope": "caller",
                "description": (
                    "Which sub-action to run: 'bootstrap' (initial "
                    "roadmap from design context), 'refresh' "
                    "(bidirectional ROADMAP.md ↔ GitHub sync), "
                    "'assignments' (propose colony/user routing "
                    "per open roadmap-linked issue), 'decompose' "
                    "(break high-level issues with many unchecked "
                    "subtasks into linked sub-issues). No default "
                    "— the planner must pick."
                ),
                "json_type": "string",
            },
            {
                "name": "roadmap_path",
                "scope": "caller",
                "description": "Relative path to ROADMAP.md inside the repo.",
                "default": "docs/ROADMAP.md",
            },
            {
                "name": "user_github_login",
                "scope": "caller",
                "description": (
                    "Override the OAuth-verified user login on "
                    "``github_identity`` when proposing user-side "
                    "task assignments. Almost always absent — "
                    "the coordinator reads from agent metadata."
                ),
                "default": None,
            },
            {
                "name": "direction",
                "scope": "caller",
                "description": (
                    "'refresh' mode only: 'bidirectional' (default) | "
                    "'roadmap_to_github' | 'github_to_roadmap'."
                ),
                "default": "bidirectional",
                # Cross-checked at registry-build time against
                # ``sync_roadmap_with_github.direction``'s
                # ``SyncDirection`` Literal so the spec default can't
                # drift from the action's accepted set without the
                # test suite catching it.
                "validates_against": (
                    "polymathera.colony.design_monorepo.process."
                    "DesignProcessCapability.sync_roadmap_with_github",
                ),
            },
            {
                "name": "decomposition_criteria",
                "scope": "caller",
                "description": (
                    "'decompose' mode only: free-text describing "
                    "what counts as a 'decomposable' issue. Passed "
                    "verbatim to the LLM judge inside "
                    "``classify_issues_decomposability`` and the LLM "
                    "proposer inside ``propose_decompositions``. "
                    "When omitted, the canonical default "
                    "(``DEFAULT_DECOMPOSITION_CRITERIA`` — issues "
                    "that are too high-level, too vague, too big "
                    "for one PR, or describe ongoing work) is used. "
                    "Operators tune this per-call to match their "
                    "team's notion of 'too big' without code "
                    "changes."
                ),
                "json_type": "string",
                "default": None,
            },
            {
                # The SessionAgent extracts ``issue_numbers`` from the
                # user's prompt at ``spawn_mission`` time — same pattern
                # already used for ``user_github_login`` /
                # ``roadmap_path``. The coordinator inherits the scope
                # as typed data; the LLM does NOT redefine scope by
                # giving up early. Per [[no-llm-facing-framework-state]].
                #
                # When omitted (``None``), scope = "all currently-open
                # roadmap issues at mission spawn" — snapshot once at
                # spawn, do NOT re-resolve mid-run. New issues filed
                # mid-run are out-of-scope and surface in a *next*
                # mission. Drain predicate is set-difference of stable
                # GitHub issue numbers.
                "name": "issue_numbers",
                "scope": "caller",
                "description": (
                    "'decompose' mode only: explicit list of GitHub "
                    "issue numbers to decompose. When omitted, scope "
                    "= all open roadmap issues at spawn time. The "
                    "list is the framework's structural drain "
                    "predicate — the validator allows "
                    "signal_completion only after every entry has "
                    "been processed (decomposed, classified as "
                    "non-decomposable, or explicitly early-stopped)."
                ),
                "json_type": "array",
                "default": None,
            },
            {
                # Hard cap on the apply phase. When both
                # ``issue_numbers`` and ``max_parents_per_run`` are
                # set and the list exceeds the cap, the cap wins —
                # remainder is treated as deferred-out-of-scope and
                # surfaces in the mission-final summary.
                "name": "max_parents_per_run",
                "scope": "caller",
                "description": (
                    "'decompose' mode only: hard cap on the number "
                    "of parents the coordinator may apply "
                    "``create_decomposition`` to in this mission run. "
                    "When None, the cap is the size of "
                    "``issue_numbers`` (or the open-issues fallback). "
                    "Caps below the candidate count surface the "
                    "remainder as deferred-out-of-scope in the "
                    "mission summary."
                ),
                "json_type": "integer",
                "default": None,
            },
        ],
        "self_concept": {
            "description": (
                "Propose, gate on user approval, and apply roadmap "
                "edits. The mission orchestrates action surfaces on "
                "DesignProcessCapability + GitHubCapability. For "
                "bootstrap / refresh / assignments modes: each is a "
                "single action call (bootstrap_roadmap_from_objectives "
                "/ sync_roadmap_with_github / propose_task_assignments) "
                "followed by a HumanApprovalRequest and an apply. "
                "For decompose mode: the planner COMPOSES three "
                "primitives — classify_issues_decomposability (LLM-"
                "judged candidate selection), propose_decompositions "
                "(per-parent or joint, returns parent_proposals + "
                "shared_concerns), and create_decomposition (the "
                "ONLY mutating primitive — gated by approval). The "
                "planner picks the strategy based on data: classify "
                "all issues then propose jointly for clusters, or "
                "sample then propose per-parent, or skip "
                "classification when the user pointed at specific "
                "issues. Every apply step is preceded by a "
                "HumanApprovalRequest carrying the proposal as ``extra``."
            ),
            "goals": [
                (
                    "Read mission_params['mode'] to pick the flow: "
                    "bootstrap → bootstrap_roadmap_from_objectives; "
                    "refresh → sync_roadmap_with_github; "
                    "assignments → propose_task_assignments; "
                    "decompose → compose the three decompose "
                    "primitives (classify_issues_decomposability, "
                    "propose_decompositions, create_decomposition)"
                ),
                (
                    "Call the chosen action with dry_run=True to compute "
                    "the proposal; render it concisely in the approval "
                    "question; pass the full proposal dict as ``extra``"
                ),
                (
                    "Post one HumanApprovalRequest via "
                    "request_human_approval(action_type='<short_action_name>', "
                    "question=..., extra={proposal, mode, repo, ...}). "
                    "Set action_type to the short name of the gated "
                    "action (e.g. 'sync_roadmap_with_github', "
                    "'create_decomposition'). The UI then renders "
                    "three choices: reject / approve_once / "
                    "approve_all. WAIT for the user's choice to "
                    "surface as planner context"
                ),
                (
                    "On choice='approve_once' or 'approve_all': re-call "
                    "the gated action with dry_run=False. The guardrail "
                    "consults the persistent approval state on the "
                    "blackboard — no need to re-poll get_response in "
                    "the same iteration as the apply. 'approve_all' "
                    "covers every future dispatch of the same "
                    "action_type this session"
                ),
                (
                    "On choice='reject': exit cleanly without writing; "
                    "report what was proposed and why it was rejected"
                ),
                (
                    "For decompose mode: there is NO single "
                    "'decompose_issues' action — you compose primitives. "
                    "The mission's scope is the typed in-scope set "
                    "(mission_params['issue_numbers'] if set, otherwise "
                    "all open roadmap issues at mission spawn, capped "
                    "by mission_params['max_parents_per_run'] if set). "
                    "The mission is complete only when every in-scope "
                    "issue has been decomposed (via create_decomposition "
                    "with dry_run=False) or classified non-decomposable "
                    "(via classify_issues_decomposability returning "
                    "decomposable=False), OR the user has explicitly "
                    "authorised early stop via "
                    "request_decompose_early_stop (a verbatim user "
                    "quote is required; you cannot self-certify). "
                    "Available primitives: list_issues, "
                    "classify_issues_decomposability, "
                    "propose_decompositions, request_human_approval, "
                    "create_decomposition, request_decompose_early_stop, "
                    "respond_to_user. Compose them as the data warrants "
                    "— discover incrementally or all at once; propose "
                    "per-issue, per-batch, or scope-wide; request "
                    "approval per-decision or batched; apply singly or "
                    "in bulk under an active approve_all. The completion "
                    "validator will not accept signal_completion() while "
                    "the in-scope backlog has unaddressed issues."
                ),
            ],
            "constraints": [
                (
                    "NEVER call any DesignProcessCapability MUTATING "
                    "action (sync_roadmap_with_github with "
                    "dry_run=False, propose_task_assignments with "
                    "dry_run=False, bootstrap_roadmap_from_objectives "
                    "with dry_run=False, create_decomposition with "
                    "dry_run=False) until the guardrail allows it. "
                    "The guardrail reads persistent approval state "
                    "from the blackboard; once the user has chosen "
                    "approve_once or approve_all for the matching "
                    "action_type, the apply can run in any later "
                    "iteration. The READ-ONLY decompose primitives "
                    "(classify_issues_decomposability, "
                    "propose_decompositions) do NOT need approval"
                ),
                (
                    "Always pass mission_params['user_github_login'] as "
                    "the user_github_login arg to "
                    "propose_task_assignments — never hardcode or invent"
                ),
                (
                    "Forward mission_params['roadmap_path'] verbatim to "
                    "the action calls; do not infer it"
                ),
                (
                    "Stamp the colony:roadmap-task marker via the "
                    "existing actions (they already do this); never "
                    "create issues directly through GitHubCapability"
                ),
            ],
        },
    },
}


def _builtin_missions() -> dict[str, MissionSpec]:
    """Build typed defaults from ``_BUILTIN_MISSIONS``.

    Side effect — after constructing each :class:`MissionSpec`, every
    one of its :attr:`caller_parameters` entries with a non-empty
    ``validates_against`` list is cross-checked against the
    referenced action signatures via
    :func:`validate_parameter_spec_against_actions`. Spec drift from
    the action signature surfaces as
    :class:`MissionSpecValidationError` at import time so the test
    suite — and ``colony-env up`` — fails loudly instead of letting
    the LLM planner waste a mission's iteration budget on
    ``{"error": "invalid_<arg>"}`` echoes.
    """

    from .metadata_parameters import validate_parameter_spec_against_actions

    specs: dict[str, MissionSpec] = {}
    for key, value in _BUILTIN_MISSIONS.items():
        spec = MissionSpec(**value)
        for caller_param in spec.caller_parameters:
            validate_parameter_spec_against_actions(caller_param)
        specs[key] = spec
    return specs


def resolve_mission_execution_policy(
    spec: "MissionSpec | dict[str, Any] | None",
    coordinator_class: type | None,
) -> MissionExecutionPolicy:
    """Single resolution point for the in-flight policy on a mission.

    Precedence:

    1. ``spec.execution_policy`` (when ``spec`` is a typed
       :class:`MissionSpec` and the field is set) — operator override
       from the registry config.
    2. ``coordinator_class.MISSION_EXECUTION_POLICY`` ClassVar — the
       coordinator's declared default.
    3. ``MissionExecutionPolicy()`` defaults — pre-policy semantics
       (one instance per parent agent, no auto-chaining, no caps).

    Accepts ``spec`` as either a :class:`MissionSpec` or a plain dict
    (the legacy ``MISSION_REGISTRY`` shape — call sites that haven't
    migrated yet pass the dict verbatim). When ``spec`` is a dict, the
    ``execution_policy`` key is read and coerced through the model;
    typed callers should pass the typed spec to skip the coercion.

    ``coordinator_class`` may be ``None`` for missions registered
    without a class reference (rare; older test fixtures); the
    function falls through to the schema default in that case.
    """

    # Layer 1: operator override on the registry entry.
    if isinstance(spec, MissionSpec) and spec.execution_policy is not None:
        return spec.execution_policy
    if isinstance(spec, dict):
        override = spec.get("execution_policy")
        if isinstance(override, MissionExecutionPolicy):
            return override
        if isinstance(override, dict):
            return MissionExecutionPolicy(**override)

    # Layer 2: coordinator's declared default.
    declared = getattr(
        coordinator_class, "MISSION_EXECUTION_POLICY", None,
    )
    if isinstance(declared, MissionExecutionPolicy):
        return declared

    # Layer 3: schema defaults — singleton per agent, no chaining,
    # no caps. Matches pre-policy semantics so unannotated missions
    # keep working.
    return MissionExecutionPolicy()


def resolve_effective_max_iterations(
    *,
    caller_override: int | None,
    policy: MissionExecutionPolicy,
    schema_default: int = 20,
) -> int:
    """Compute the planner-loop iteration cap for a mission coordinator.

    Precedence:

    1. ``caller_override`` — when the spawn caller (LLM planner via
       ``spawn_mission`` or REST handler via ``/api/jobs/submit``)
       explicitly passed an integer, it wins. Callers that don't
       care pass ``None``.
    2. ``policy.max_iterations`` — the mission's declared shape on
       its coordinator's ``MISSION_EXECUTION_POLICY`` ClassVar.
       Set this when the mission's natural loop shape (e.g.
       propose → request_approval → idle-poll → apply → report)
       doesn't fit the generic 20-iteration default.
    3. ``schema_default`` — :class:`AgentMetadata`'s field default
       (20). Matches the pre-policy behaviour for any mission that
       declares no explicit budget.

    Single resolution point so both spawn paths
    (:meth:`SessionOrchestratorCapability.spawn_mission` and
    :func:`web_ui.backend.routers.jobs._run_job`) agree on the
    precedence rules — adding a new spawn path means routing through
    this helper, not re-deriving the layering.
    """

    if caller_override is not None:
        return caller_override
    if policy.max_iterations is not None:
        return policy.max_iterations
    return schema_default


def build_coordinator_self_concept(
    registry_entry: "MissionSpec | dict[str, Any]",
    *,
    mission_type: str,
) -> "AgentSelfConcept | None":
    """Stamp a runtime :class:`AgentSelfConcept` from a mission entry's
    spec-side :class:`MissionSelfConcept`.

    The two shapes are deliberately distinct: the **spec** side
    (``MissionSelfConcept``, ``extra="forbid"``) carries the static
    description / goals / constraints declared on the mission
    registry entry; the **runtime** side
    (:class:`polymathera.colony.agents.self_concept.AgentSelfConcept`)
    additionally requires ``agent_id`` and ``name`` plus the broader
    bag of fields :class:`ConsciousnessCapability` populates over
    the agent's lifetime.

    Bridging the two is the single fragile step every mission-spawn
    site has to get right. This helper centralises it so both the
    chat-side ``spawn_mission`` and the REST-side ``jobs.start_run``
    use the same convention — ``agent_id=""`` (blank for
    ConsciousnessCapability to overwrite at init) and ``name`` =
    the mission's ``label`` (with ``mission_type`` as fallback when
    the entry has no label). Avoids the bug class where one call
    site stamps the name and the other forgets.

    Returns ``None`` when the registry entry carries no
    ``self_concept`` — both consumers pass that straight through to
    ``AgentMetadata.self_concept`` (which is ``Optional``).

    Accepts either the typed :class:`MissionSpec` (the
    ``MissionRegistryConfig`` path) or the legacy raw dict form
    (the ``_BUILTIN_MISSIONS`` + entry-point group paths) — both
    in-tree call sites still index entries as plain dicts.
    """

    # Lazy import — ``self_concept`` lives in a sibling module that
    # imports nothing from here, so a module-top import would invert
    # an otherwise clean dependency direction without buying anything.
    from .self_concept import AgentSelfConcept

    if isinstance(registry_entry, MissionSpec):
        spec_self_concept = registry_entry.self_concept.model_dump()
        label = registry_entry.label
    else:
        spec_self_concept = dict(registry_entry.get("self_concept") or {})
        label = registry_entry.get("label") or ""
    if not spec_self_concept:
        return None
    return AgentSelfConcept(
        agent_id="",
        name=label or mission_type,
        **spec_self_concept,
    )


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
    # NOTE: per-tenant installation_id no longer lives here. The backend reads it from the database at runtime based on the authenticated user's login, so it is not part of the operator config.
    private_key_pem: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_PRIVATE_KEY_PEM", "optional": True},
    )
    # OAuth client credentials for the same GitHub App's user-to-server
    # flow (the "Connect GitHub" button on the user profile). The App
    # registration exposes these alongside the App ID / private key in
    # the GitHub App settings page. Backend exchanges the authorisation
    # code for a one-shot user token, reads verified login + emails,
    # then discards the token.
    oauth_client_id: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_APP_CLIENT_ID", "optional": True},
    )
    oauth_client_secret: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_APP_CLIENT_SECRET", "optional": True},
    )
    # P9: HMAC-SHA256 key the ``POST /api/v1/github/webhook`` receiver
    # verifies inbound payloads against (operator sets matching value
    # in the GitHub App's webhook secret field). When unset, the
    # receiver rejects every inbound webhook with 503 — webhook mode
    # is opt-in, the operator must wire this AND set
    # ``mode: webhook`` in the colony's
    # ``.colony/github_inbound.yaml`` for the receiver to fire.
    webhook_secret: str = Field(
        default="",
        json_schema_extra={"env": "GITHUB_WEBHOOK_SECRET", "optional": True},
    )

    @field_validator("private_key_pem", mode="after")
    @classmethod
    def _normalize_pem(cls, v: str) -> str:
        """Translate literal ``\\n`` sequences to real newlines.

        ``docker-compose``'s ``environment:`` list truncates env-var
        values at the first real newline (the YAML parser treats the
        next line as the next list item), so multi-line PEM keys must
        be stored on a single line with ``\\n`` escapes in ``.env``.
        We invert that here so pyjwt / cryptography see a valid PEM.
        Idempotent: leaves real-newline-bearing values alone.
        """
        if v and "\\n" in v and "\n" not in v:
            return v.replace("\\n", "\n")
        return v


async def get_github_auth_config() -> GitHubAuthConfig:
    return await _get_component_or_default("capabilities.github", GitHubAuthConfig)


@register_polymathera_config(path="capabilities.gitlab")
class GitLabAuthConfig(ConfigComponent):
    """OAuth-Application credentials for the GitLab ``VcsProvider``.

    Unlike GitHub, GitLab has no per-org App-installation flow — there
    is only the user-to-server OAuth Application registered on
    GitLab. Server-to-server actions ride a Group Access Token the
    tenant admin provisions per-group and pastes into Colony (PR 6's
    encrypted ``tenants.bot_token_encrypted``); there is no
    deploy-wide GitLab private key.
    """

    oauth_client_id: str = Field(
        default="",
        json_schema_extra={"env": "GITLAB_OAUTH_CLIENT_ID", "optional": True},
    )
    oauth_client_secret: str = Field(
        default="",
        json_schema_extra={"env": "GITLAB_OAUTH_CLIENT_SECRET", "optional": True},
    )
    # Self-hosted GitLab (CE/EE) instances expose the same REST + OAuth
    # surfaces at the operator's own URL. Default is gitlab.com; an
    # enterprise deployment overrides this to e.g.
    # ``https://gitlab.acme.internal``.
    base_url: str = Field(
        default="https://gitlab.com",
        json_schema_extra={"env": "GITLAB_BASE_URL", "optional": True},
    )


async def get_gitlab_auth_config() -> GitLabAuthConfig:
    return await _get_component_or_default("capabilities.gitlab", GitLabAuthConfig)


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
    "MissionConcurrencyScope",
    "MissionExecutionPolicy",
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
    "build_coordinator_self_concept",
    "get_chroma_config",
    "get_github_auth_config",
    "get_plugins_config",
    "get_sandbox_image_registry_config",
    "get_web_search_config",
    "resolve_mission_execution_policy",
)
