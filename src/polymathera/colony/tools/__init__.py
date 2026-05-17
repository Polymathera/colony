"""Typed metadata vocabulary for tool capabilities (master §3.3).

Each :class:`~polymathera.colony.agents.patterns.capabilities.tool.ToolCapability`
subclass declares a class-level :class:`ToolSpec`; this package
exports the spec model plus the enums and per-call cost / resource
shapes it composes from.

The tool framework itself lives in
:mod:`polymathera.colony.agents.patterns.capabilities.tool` —
``ToolCapability`` (ABC), ``LocalToolCapability``,
``SandboxToolCapability``. The CPS-side
``HPCToolCapability`` lives in
:mod:`polymathera.cps.tools.hpc.capability`.
"""

from __future__ import annotations

from .spec import (
    CostModel,
    Determinism,
    ExecutionLocality,
    GpuRequirement,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ResourceRequirements,
    ToolSpec,
)


__all__ = (
    # Enums
    "Determinism",
    "ExecutionLocality",
    "HeadlessReadiness",
    "HITLFrequency",
    "Licensing",
    # Per-call models
    "CostModel",
    "GpuRequirement",
    "ResourceRequirements",
    # Spec
    "ToolSpec",
)
