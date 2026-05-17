"""Typed shapes for the design-time tool layer.

Per master §3.3, the colony tool framework is built around five
ingredients:

- ``ToolSpec`` — frozen description of a tool: who, what version,
  what capabilities it fulfils, what its headless / HITL / licensing
  / determinism / cost profile is, and what container or repo it
  needs.
- Four enums — ``HeadlessReadiness``, ``HITLFrequency``,
  ``Determinism``, ``Licensing`` — that codify the dossier-wide
  Appendix-C / Appendix-D vocabulary.
- ``ToolCall`` / ``ToolResult`` — the typed invocation record.
- ``Preferences`` — typed match expression used by ``ToolRegistry.resolve``
  (see ``registry.py``).
- ``ToolAdapter`` (ABC) — the implementation surface; subclasses ship
  a class-level ``spec`` and override ``invoke``.

This module is colony-generic: the abstractions don't carry any
design-engineering-shaped semantics (master §1.4 boundary rule). CPS
and per-domain tool catalogues add their own ``ToolSpec`` instances
on top.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums (master §3.3 / §4.2)
# ---------------------------------------------------------------------------


class HeadlessReadiness(str, Enum):
    """How agentic-friendly the tool's interface is.

    Values match the Appendix-C tier of every per-domain dossier.
    String values are stable on the wire so JSON / YAML registries
    serialize readably.
    """

    NATIVE = "native"
    """Python or CLI-native, no GUI dependency. The green zone — agent
    runs autonomously."""

    CLI_ONLY = "cli_only"
    """CLI-driven, no Python API but scriptable end-to-end."""

    PARTIAL = "partial"
    """Has GUI residuals or interactive setup steps; can run headless
    with effort."""

    GUI_PRIMARY = "gui_primary"
    """GUI-first; CLI is a thin shadow. Agent prepares; human runs."""

    NONE = "none"
    """No headless story at all. Agent is essentially a notetaker."""

    @property
    def order(self) -> int:
        """Higher = more agent-friendly. Used for ``Preferences.min_headless``."""
        return _HEADLESS_ORDER[self]


_HEADLESS_ORDER: dict[HeadlessReadiness, int] = {
    HeadlessReadiness.NATIVE: 4,
    HeadlessReadiness.CLI_ONLY: 3,
    HeadlessReadiness.PARTIAL: 2,
    HeadlessReadiness.GUI_PRIMARY: 1,
    HeadlessReadiness.NONE: 0,
}


class HITLFrequency(str, Enum):
    """How often a human is in the loop on this tool's outputs."""

    AUTONOMOUS = "autonomous"
    """No human review under normal operation; sampled milestones only."""

    REVIEW_MILESTONES = "review_milestones"
    """Human reviews aggregated milestones (e.g., end of a workflow phase)."""

    APPROVAL_GATES = "approval_gates"
    """Human approval required at specific decision gates
    (publish, push to main, file a regulatory submission)."""

    CO_PILOT = "co_pilot"
    """Human reviews every output before it's accepted."""

    HUMAN_PRIMARY = "human_primary"
    """Human is the primary operator; agent assists."""

    @property
    def order(self) -> int:
        """Higher = more human in the loop. Used for ``Preferences.max_hitl``."""
        return _HITL_ORDER[self]


_HITL_ORDER: dict[HITLFrequency, int] = {
    HITLFrequency.AUTONOMOUS: 0,
    HITLFrequency.REVIEW_MILESTONES: 1,
    HITLFrequency.APPROVAL_GATES: 2,
    HITLFrequency.CO_PILOT: 3,
    HITLFrequency.HUMAN_PRIMARY: 4,
}


class Determinism(str, Enum):
    """Reproducibility profile of the tool's outputs."""

    DETERMINISTIC = "deterministic"
    """Same inputs → same outputs, byte-for-byte."""

    SEEDED = "seeded"
    """Deterministic given an explicit RNG seed."""

    STOCHASTIC = "stochastic"
    """Outputs vary across runs; bound by a confidence interval
    or distribution."""


class ExecutionLocality(str, Enum):
    """Where the tool is allowed to run.

    Distinct from ``ToolSpec.backend``: ``backend`` describes the
    *mechanism* (in-process, docker container, REST call), while
    ``execution_locality`` describes the *physical environment*
    (in-cluster, remote HPC, customer site). A single backend
    (``http_api``) is used by both local sidecars and remote HPC
    dispatchers, so the two axes are orthogonal.
    """

    LOCAL = "local"
    """In-cluster: in-process / cli_subprocess / docker / ray_serve on a
    node that ``colony-env up`` started. The default."""

    HPC = "hpc"
    """Remote large-resource cluster — for CPS today, AWS Batch via the
    REST contract in ``cps/deployment/cdk/`` reached through an
    ``HPCJobAdapter`` subclass."""

    CUSTOMER_SITE = "customer_site"
    """Operator's on-premise environment. Reserved for forward
    compatibility; no adapter ships against it today."""


class GpuRequirement(BaseModel):
    """GPU specification for a tool's per-call resource requirements.

    Omit (leave ``ResourceRequirements.gpu`` as None) when the tool
    runs on CPU only. ``kind`` is a coarse class label, not a vendor
    SKU — the HPC scheduler maps it to an instance type via the
    operator's ``cps.hpc.limits.allowed_gpu_kinds``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["a10g", "a100", "h100", "v100", "any"] = "any"
    count: int = Field(default=1, ge=1)
    memory_gb: float | None = Field(default=None, gt=0)
    """Minimum per-GPU memory. None = no specific minimum."""


class ResourceRequirements(BaseModel):
    """Per-call resource requirements declared on ``ToolSpec``.

    Distinct from ``CostModel``: ``CostModel`` is the *expected per-call
    cost* (used for budgeting + scoring); ``ResourceRequirements`` is
    the *minimum environment* the tool refuses to run without. A tool
    can have low cost (``cost_model.cpu_seconds=2``) and large minimums
    (``min_vcpus=16``) — e.g. a quick parallel CalculiX run.

    The HPC dispatcher (``HPCJobAdapter``) forwards these to AWS Batch
    as the job's ``resourceRequirements``; the registry resolver uses
    them with ``Preferences.max_required_*`` to drop tools the runtime
    can't accommodate.
    """

    model_config = ConfigDict(frozen=True)

    min_vcpus: int = Field(default=1, ge=1)
    min_memory_gb: float = Field(default=1.0, gt=0)
    gpu: GpuRequirement | None = None
    expected_wallclock_seconds: float = Field(default=600.0, gt=0)
    """Wall-clock estimate. The HPC dispatcher uses this for the job's
    timeout (with a safety factor); the resolver uses it with
    ``Preferences.max_required_wallclock_seconds`` to drop tools
    whose runs would exceed the operator's tolerance."""


class Licensing(str, Enum):
    """SPDX-style licence buckets — coarse but sufficient for
    framework-level routing.

    Free-form licence text lives on ``ToolSpec.licensing_notes``;
    this enum is for the framework's gating decisions
    (e.g., 'forbid commercial' at deployment time).
    """

    PUBLIC_DOMAIN = "public_domain"
    MIT = "mit"
    BSD = "bsd"
    APACHE_2_0 = "apache_2_0"
    LGPL = "lgpl"
    GPL = "gpl"
    AGPL = "agpl"
    MOZILLA = "mpl"
    COMMERCIAL = "commercial"
    """Paid / closed source / per-seat licence."""

    RESTRICTED = "restricted"
    """Export-controlled, ITAR, customer-restricted, etc."""

    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------


class CostModel(BaseModel):
    """Per-call cost estimates for a tool (master §3.3)."""

    model_config = ConfigDict(frozen=True)

    cpu_seconds: float = Field(default=0.0, ge=0.0)
    gpu_seconds: float = Field(default=0.0, ge=0.0)
    memory_gb: float = Field(default=0.0, ge=0.0)
    dollars: float = Field(default=0.0, ge=0.0)
    """Direct $$ cost (e.g., commercial licence per-call, paid API)."""

    extra: dict[str, float] = Field(default_factory=dict)
    """Free-form extension (e.g., 'wallclock_seconds', 'requests_per_run')."""


class ToolSpec(BaseModel):
    """Frozen description of a tool's design-time profile.

    Every field master §3.3 lists is present here. The model is frozen
    so a ``ToolSpec`` shared across an in-memory registry cannot be
    mutated by a caller — registry consumers always see a consistent
    snapshot.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    """Stable identifier, unique within a registry. Snake_case is preferred
    (e.g., 'numpy_lstsq', 'openfoam_simple_foam')."""

    version: str = Field(default="0.1.0")
    """Semantic version of the *adapter*, not the underlying library."""

    domain: str = Field(default="general")
    """Coarse domain label ('general', 'cad', 'fem', 'cfd', 'em',
    'optics', 'plasma', 'biomedical', etc.). Free-form."""

    backend: str = Field(default="in_process")
    """How the adapter executes: 'in_process', 'cli_subprocess',
    'docker', 'ray_serve', 'http_api', 'mcp_server', etc."""

    capabilities: tuple[str, ...] = Field(default_factory=tuple)
    """Capability keys this tool fulfils. ``ToolRegistry.resolve``
    matches against these. Sorted at construction for deterministic
    iteration."""

    inputs_schema: Mapping[str, Any] = Field(default_factory=dict)
    """JSON-Schema-like description of ``ToolCall.parameters``."""

    outputs_schema: Mapping[str, Any] = Field(default_factory=dict)
    """JSON-Schema-like description of ``ToolResult.value``."""

    determinism: Determinism = Determinism.DETERMINISTIC
    cost_model: CostModel = Field(default_factory=CostModel)

    licensing: Licensing = Licensing.UNKNOWN
    licensing_notes: str = ""
    """Free-form licensing text the enum doesn't capture (the SPDX
    identifier, attribution requirements, restrictions)."""

    references: tuple[str, ...] = Field(default_factory=tuple)
    """Doc / paper / vendor URLs the tool was built against."""

    headless: HeadlessReadiness = HeadlessReadiness.NATIVE
    hitl_frequency: HITLFrequency = HITLFrequency.AUTONOMOUS
    interruptibility: bool = False
    """Whether the adapter supports cooperative cancellation
    (responds to ``asyncio.Task.cancel`` cleanly, releasing all
    external resources)."""

    extends_repo: str | None = None
    """Repo this adapter lives in, when produced by a tool-building
    pool (master §9). Format: ``subdir:tools/<purpose>/<name>`` for
    in-monorepo tools, ``git:<remote>:<ref>`` for standalone."""

    container_image: str | None = None
    """Docker image tag this adapter requires (master §4.5). When set,
    the dispatcher routes to a host that has the image."""

    execution_locality: ExecutionLocality = ExecutionLocality.LOCAL
    """Where this tool is allowed to run. See ``ExecutionLocality``."""

    resource_requirements: ResourceRequirements = Field(
        default_factory=ResourceRequirements,
    )
    """Minimum environment the tool refuses to run without. Distinct
    from ``cost_model`` (which is the expected per-call cost).
    Single source of truth for both:
      - the registry's ``max_required_*`` hard-filter, and
      - the HPC dispatcher's POST body to AWS Batch.
    """

    extra: dict[str, Any] = Field(default_factory=dict)
    """Free-form metadata for tool-class-specific extensions."""

    @field_validator("capabilities", mode="before")
    @classmethod
    def _normalise_capabilities(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            return (v,)
        return tuple(sorted({str(c) for c in v}))

    def fulfils(self, capability: str) -> bool:
        return capability in self.capabilities


# ---------------------------------------------------------------------------
# Invocation records
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """One invocation of a tool through the registry."""

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    capability: str
    """The capability the *caller* asked for; the registry maps it to
    an adapter."""

    parameters: Mapping[str, Any] = Field(default_factory=dict)
    caller: str = Field(default="")
    """Identifier of the caller: agent_id, capability_key, or the empty
    string for system-level calls."""

    trace_id: str | None = None
    """Distributed-tracing correlation id."""

    started_at: float = Field(default_factory=time.time)


class ToolResult(BaseModel):
    """Result of a ``ToolAdapter.invoke``."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    adapter_name: str
    success: bool
    value: Any = None
    """Adapter-defined payload. The caller (and the typed
    ``ToolCapability`` wrapper) interprets it."""

    error: str | None = None
    cost_observed: CostModel | None = None
    """Actual cost — populated when the adapter measures it."""

    completed_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Preferences (used by ToolRegistry.resolve)
# ---------------------------------------------------------------------------


class Preferences(BaseModel):
    """Typed match expression for adapter resolution.

    All fields optional. ``ToolRegistry.resolve`` runs in two passes:

    1. **Hard filter** — drop adapters that violate any of:
       - ``min_headless`` (adapter's tier < required)
       - ``max_hitl`` (adapter's tier > permitted)
       - ``required_determinism`` (mismatch)
       - ``required_backend`` (mismatch)
       - ``forbid_licences`` (adapter's licence in set)
       - ``max_cpu_seconds`` / ``max_gpu_seconds`` / ``max_memory_gb``
         / ``max_dollars`` (adapter's *cost* above cap — distinct from
         ``max_required_*`` which gate the *resource requirements*)
       - ``require_interruptible`` (adapter's spec doesn't support it)
       - ``allowed_container_images`` (adapter's image not in set,
         when set is non-empty and adapter's image is non-None)
       - ``allowed_localities`` (adapter's ``execution_locality`` not
         in set)
       - ``max_required_vcpus`` / ``max_required_memory_gb`` /
         ``max_required_gpu_count`` / ``max_required_wallclock_seconds``
         (adapter's ``resource_requirements.*`` minimum exceeds cap)
       - ``allowed_required_gpu_kinds`` (adapter's
         ``resource_requirements.gpu.kind`` not in set, when adapter
         declares a GPU requirement)
    2. **Score rank** — among survivors, prefer:
       - higher ``headless`` tier
       - lower ``hitl`` tier
       - matches against ``preferred_backend``
       - lower observed cost on the dimensions the caller cares about

    A caller that doesn't know what to ask for just supplies
    ``Preferences()`` and accepts the registry's default ranking.
    """

    model_config = ConfigDict(frozen=True)

    min_headless: HeadlessReadiness | None = None
    max_hitl: HITLFrequency | None = None
    required_determinism: Determinism | None = None
    required_backend: str | None = None
    preferred_backend: str | None = None
    forbid_licences: frozenset[Licensing] = Field(default_factory=frozenset)
    max_cpu_seconds: float | None = None
    max_gpu_seconds: float | None = None
    max_memory_gb: float | None = None
    """Caps the adapter's ``cost_model.memory_gb`` (estimated cost per
    call). Distinct from ``max_required_memory_gb`` which caps the
    adapter's ``resource_requirements.min_memory_gb`` (refused-without
    minimum)."""

    max_dollars: float | None = None
    require_interruptible: bool = False
    allowed_container_images: frozenset[str] = Field(default_factory=frozenset)

    # ---- Locality + resource-requirements filters ----
    allowed_localities: frozenset[ExecutionLocality] | None = None
    """Restrict adapters by ``ToolSpec.execution_locality``. ``None``
    (the default) accepts every locality; set explicitly to e.g.
    ``frozenset({ExecutionLocality.LOCAL})`` to drop HPC adapters when
    the operator has no HPC endpoint configured."""

    max_required_vcpus: int | None = None
    """Cap on the adapter's ``resource_requirements.min_vcpus``. Set
    from the operator's HPC limits + local cluster capacity."""

    max_required_memory_gb: float | None = None
    """Cap on the adapter's ``resource_requirements.min_memory_gb``."""

    max_required_gpu_count: int | None = None
    """Cap on the adapter's ``resource_requirements.gpu.count``
    (when the adapter declares a GPU requirement)."""

    allowed_required_gpu_kinds: frozenset[str] | None = None
    """Allow-list of adapter GPU kinds (``a10g``, ``a100``, ``h100``,
    ``v100``, ``any``). Applies only when the adapter declares a GPU
    requirement; CPU-only adapters always pass this filter."""

    max_required_wallclock_seconds: float | None = None
    """Cap on the adapter's
    ``resource_requirements.expected_wallclock_seconds`` — drops jobs
    longer than the operator's tolerance before any artifact upload."""

    prefer_lower_cost: bool = True
    """When True (default), the score-rank tier-breaks by cost
    (cpu+gpu seconds, dollars). When False, cost is ignored."""


# ---------------------------------------------------------------------------
# ToolAdapter ABC
# ---------------------------------------------------------------------------


class ToolAdapter(ABC):
    """Implementation surface for a tool.

    Subclasses ship a class-level ``spec`` (so the registry can read
    it without instantiating), an async ``invoke`` (the only required
    method), and optional ``healthcheck`` / ``warmup`` / ``shutdown``
    hooks. The constructor signature is left to the subclass — an
    adapter may need a database session, an HTTP client, a Docker
    runtime handle, etc., depending on its backend.

    Why async: colony's runtime is async-first, and most non-trivial
    tools (a CFD job, an LLM API call, a database query) are
    naturally async. In-process synchronous adapters wrap their work
    with ``asyncio.to_thread`` inside ``invoke``.
    """

    spec: ClassVar[ToolSpec]
    """Class-level description. Subclasses MUST override."""

    @abstractmethod
    async def invoke(self, call: ToolCall) -> ToolResult:
        """Execute the tool against ``call`` and return a ``ToolResult``."""

    async def healthcheck(self) -> bool:
        """Return True if the adapter can serve invocations right now.

        Default: always healthy. Adapters that depend on external
        services (an HTTP API, a database, a Docker image) override.
        """
        return True

    async def warmup(self) -> None:
        """Pre-load expensive resources (a model, a JIT cache, a
        process pool). Default: no-op."""

    async def shutdown(self) -> None:
        """Release adapter resources cleanly. Default: no-op."""

    @property
    def name(self) -> str:
        """Convenience: ``self.spec.name``."""
        return type(self).spec.name


__all__ = (
    "HeadlessReadiness",
    "HITLFrequency",
    "Determinism",
    "Licensing",
    "ExecutionLocality",
    "GpuRequirement",
    "ResourceRequirements",
    "CostModel",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "Preferences",
    "ToolAdapter",
)
