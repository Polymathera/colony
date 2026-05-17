"""Design-time tool framework for Polymathera Colony (Phase C2).

Master §3.3 / §4.3. Defines the typed vocabulary the colony tool layer
exposes:

- ``ToolSpec`` — frozen description of a tool, with first-class
  ``HeadlessReadiness`` / ``HITLFrequency`` / ``Determinism`` /
  ``Licensing`` enums + cost / interruptibility / container metadata.
- ``ToolAdapter`` (ABC) — implementation surface; subclasses ship a
  class-level ``spec`` and override ``invoke``.
- ``ToolRegistry`` — capability-keyed index with hard-filter + score-
  rank resolution against typed ``Preferences``.
- ``ToolCapability`` — agent-facing wrapper over a (capability,
  registry, preferences) tuple. Agents never see backend args.
- ``BuildVsBuyAdvisor`` + ``BuildVsBuyContext`` + ``BuildVsBuyVerdict``
  — the master §4.3 six-rule decision policy with C5 augment-vs-build
  refinement.

This package is colony-generic — the abstractions don't carry any
design-engineering-shaped semantics, so any multi-agent system can
reuse them. CPS and per-domain registries layer on top.

The existing ``colony.agents.tools`` runtime layer (deployment-backed
tool execution + result cache) coexists with this design-time layer;
adapters that delegate to a Ray-Serve deployment wire one to the other
inside their ``invoke`` method.
"""

from __future__ import annotations

from .base import (
    CostModel,
    Determinism,
    ExecutionLocality,
    GpuRequirement,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    Preferences,
    ResourceRequirements,
    ToolAdapter,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .build_vs_buy import (
    BuildVsBuyAdvisor,
    BuildVsBuyContext,
    BuildVsBuyDecision,
    BuildVsBuyVerdict,
    INNER_LOOP_FREQUENCY_THRESHOLD,
    RuleEvaluation,
    TeamTrackRecord,
    ToolMatchSummary,
)
from .capability import ToolCapability
from .registry import (
    DuplicateAdapter,
    NoAdapterAvailable,
    ToolRegistry,
    ToolRegistryError,
)


__all__ = (
    # Enums
    "Determinism",
    "ExecutionLocality",
    "HeadlessReadiness",
    "HITLFrequency",
    "Licensing",
    # Models
    "CostModel",
    "GpuRequirement",
    "Preferences",
    "ResourceRequirements",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    # ABC
    "ToolAdapter",
    # Registry
    "ToolRegistry",
    "ToolRegistryError",
    "NoAdapterAvailable",
    "DuplicateAdapter",
    # Capability
    "ToolCapability",
    # Build-vs-buy
    "BuildVsBuyAdvisor",
    "BuildVsBuyContext",
    "BuildVsBuyDecision",
    "BuildVsBuyVerdict",
    "RuleEvaluation",
    "TeamTrackRecord",
    "ToolMatchSummary",
    "INNER_LOOP_FREQUENCY_THRESHOLD",
)
