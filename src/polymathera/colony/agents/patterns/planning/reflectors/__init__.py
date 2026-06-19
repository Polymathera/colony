"""Reflectors — pluggable inference + advisory + diagnostic producers
that subsume the prior detector/advisor bus pairs.

Each reflector is a :class:`~polymathera.colony.agents.patterns.planning.reflection.StreamReflector`
the substrate calls at well-defined moments (per-entry, per-iteration
boundary, per-planning step). A reflector returns a typed
:class:`~polymathera.colony.agents.patterns.planning.reflection.StreamReflection`
the policy aggregates into the next planner prompt + the cross-agent
diagnostic blackboard.

This module also exposes stream-builder helpers that wrap each
reflector in a :class:`~polymathera.colony.agents.patterns.planning.streams.ConsciousnessStream`
configured with the right ``iteration_boundary`` filter — call sites
register the stream-list directly via the action policy's
``consciousness_streams=`` kwarg."""

from __future__ import annotations

from typing import Any

from ..streams import ConsciousnessStream
from .approval_advance import ApprovalAdvanceReflector
from .cliff_guard import CliffGuardReflector
from .contract_drift import ContractDriftReflector, ContractRegistration
from .error_rewriter import (
    DEFAULT_RULES,
    ErrorRewriterReflector,
    RewriteRule,
)
from .inconsistency import InconsistencyReflector


def _accept_all_iteration_boundaries(_payload: dict[str, Any]) -> bool:
    return True


def _reflector_stream(
    *,
    name: str,
    reflector: Any,
    max_entries: int = 4,
) -> ConsciousnessStream:
    """Build a :class:`ConsciousnessStream` whose only kind is
    ``iteration_boundary`` and whose reflector is the given inference
    plugin. Small rolling window — these streams reflect on the latest
    boundary only; older boundaries don't need to be re-read."""

    return ConsciousnessStream(
        name=name,
        reflector=reflector,
        iteration_boundary_filter=_accept_all_iteration_boundaries,
        max_entries=max_entries,
    )


def contract_drift_stream(
    *,
    registrations: tuple[ContractRegistration, ...] | None = None,
) -> ConsciousnessStream:
    return _reflector_stream(
        name="contract_drift",
        reflector=ContractDriftReflector(registrations=registrations),
    )


def inconsistency_stream() -> ConsciousnessStream:
    return _reflector_stream(
        name="inconsistency",
        reflector=InconsistencyReflector(),
    )


def cliff_guard_stream(
    *,
    max_iterations: int,
    lead_iterations: int = 2,
) -> ConsciousnessStream:
    return _reflector_stream(
        name="cliff_guard",
        reflector=CliffGuardReflector(
            max_iterations=max_iterations,
            lead_iterations=lead_iterations,
        ),
    )


def approval_advance_stream() -> ConsciousnessStream:
    return _reflector_stream(
        name="approval_advance",
        reflector=ApprovalAdvanceReflector(),
    )


def error_rewriter_stream(
    *,
    rules: list[RewriteRule] | None = None,
) -> ConsciousnessStream:
    return _reflector_stream(
        name="error_rewriter",
        reflector=ErrorRewriterReflector(rules=rules),
    )


__all__ = (
    "ApprovalAdvanceReflector",
    "CliffGuardReflector",
    "ContractDriftReflector",
    "ContractRegistration",
    "DEFAULT_RULES",
    "ErrorRewriterReflector",
    "InconsistencyReflector",
    "RewriteRule",
    "approval_advance_stream",
    "cliff_guard_stream",
    "contract_drift_stream",
    "error_rewriter_stream",
    "inconsistency_stream",
)
