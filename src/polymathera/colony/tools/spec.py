"""Typed metadata vocabulary for tool capabilities.

This module defines the frozen :class:`ToolSpec` description plus the
enums + per-call cost / resource models that
:class:`~polymathera.colony.agents.patterns.capabilities.tool.ToolCapability`
subclasses declare as class-level ``spec``. The spec is the
authoritative description of a tool — version, licence, headless
tier, HITL frequency, cost, resource requirements, execution
locality.

This module is colony-generic: the abstractions don't carry any
design-engineering-shaped semantics (master §1.4 boundary rule). CPS
and per-domain tool capability subclasses declare their own
``ToolSpec`` instances on top.

The on-disk catalog :class:`~polymathera.colony.design_monorepo.models.ToolEntry`
references the implementing class via ``capability_fqn``; ``ToolEntry``
holds only catalog-index fields (name, purpose, location,
capability_fqn, capability, extra). Everything else — version, licence,
container image, headless tier, HITL frequency, cost model, resource
requirements — lives on this ``ToolSpec`` and is read off the
imported class at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

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
        """Higher = more agent-friendly."""
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
        """Higher = more human in the loop."""
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
    ``HPCToolCapability`` subclass."""

    CUSTOMER_SITE = "customer_site"
    """Operator's on-premise environment. Reserved for forward
    compatibility; no capability ships against it today."""


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
# Per-call cost + resource requirements
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

    The HPC dispatcher (``HPCToolCapability``) forwards these to AWS
    Batch as the job's ``resourceRequirements``; the operator's
    ``cps.hpc.limits`` block can cap them at submit time.
    """

    model_config = ConfigDict(frozen=True)

    min_vcpus: int = Field(default=1, ge=1)
    min_memory_gb: float = Field(default=1.0, gt=0)
    gpu: GpuRequirement | None = None
    expected_wallclock_seconds: float = Field(default=600.0, gt=0)
    """Wall-clock estimate. The HPC dispatcher uses this for the job's
    timeout (with a safety factor)."""


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    """Frozen description of a tool capability's design-time profile.

    Every field master §3.3 lists is present here. The model is frozen
    so a ``ToolSpec`` shared across an in-memory registry cannot be
    mutated by a caller — registry consumers always see a consistent
    snapshot.

    Each :class:`~polymathera.colony.agents.patterns.capabilities.tool.ToolCapability`
    subclass declares one as ``spec`` (a class attribute); the ABC
    enforces the declaration at subclass creation.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    """Stable identifier, unique within an agent's mounted capability
    set. Snake_case is preferred (e.g., 'compute_shielding_factor',
    'openems_fdtd')."""

    version: str = Field(default="0.1.0")
    """Semantic version of the *capability subclass*, not the
    underlying library."""

    domain: str = Field(default="general")
    """Coarse domain label ('general', 'cad', 'fem', 'cfd', 'em',
    'optics', 'plasma', 'biomedical', etc.). Free-form."""

    backend: str = Field(default="in_process")
    """How the capability executes: 'in_process', 'cli_subprocess',
    'docker', 'ray_serve', 'http_api', 'mcp_server', etc. Distinct
    from ``execution_locality`` (where the work runs)."""

    capabilities: tuple[str, ...] = Field(default_factory=tuple)
    """Capability keys the tool fulfils. Used by
    :func:`~polymathera.colony.design_monorepo.registry.search` and as
    the cross-reference key in
    :class:`~polymathera.colony.design_monorepo.models.ToolEntry`.
    Sorted at construction for deterministic iteration."""

    inputs_schema: Mapping[str, Any] = Field(default_factory=dict)
    """JSON-Schema-like description of the action's input parameters.
    The concrete signature comes from the ``@action_executor``
    method's kwargs; this field is for tool-author documentation."""

    outputs_schema: Mapping[str, Any] = Field(default_factory=dict)
    """JSON-Schema-like description of the action's typed return."""

    determinism: Determinism = Determinism.DETERMINISTIC
    cost_model: CostModel = Field(default_factory=CostModel)

    licensing: Licensing = Licensing.UNKNOWN
    licensing_notes: str = ""
    """Free-form licensing text the enum doesn't capture (the SPDX
    identifier, attribution requirements, restrictions)."""

    references: tuple[str, ...] = Field(default_factory=tuple)
    """Doc / paper / vendor URLs the capability was built against."""

    headless: HeadlessReadiness = HeadlessReadiness.NATIVE
    hitl_frequency: HITLFrequency = HITLFrequency.AUTONOMOUS
    interruptibility: bool = False
    """Whether the action supports cooperative cancellation
    (responds to ``asyncio.Task.cancel`` cleanly, releasing all
    external resources)."""

    extends_repo: str | None = None
    """Repo this capability lives in, when produced by a tool-building
    pool (master §9). Format: ``subdir:tools/<purpose>/<name>`` for
    in-monorepo tools, ``git:<remote>:<ref>`` for standalone."""

    execution_locality: ExecutionLocality = ExecutionLocality.LOCAL
    """Where this tool is allowed to run. See ``ExecutionLocality``."""

    resource_requirements: ResourceRequirements = Field(
        default_factory=ResourceRequirements,
    )
    """Minimum environment the tool refuses to run without. Distinct
    from ``cost_model`` (which is the expected per-call cost). Forwarded
    to AWS Batch by ``HPCToolCapability``; checked against operator
    limits at planner time."""

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


__all__ = (
    "HeadlessReadiness",
    "HITLFrequency",
    "Determinism",
    "ExecutionLocality",
    "Licensing",
    "CostModel",
    "GpuRequirement",
    "ResourceRequirements",
    "ToolSpec",
)
