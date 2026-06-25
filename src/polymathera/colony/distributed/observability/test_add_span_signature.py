"""Regression test for the :meth:`TracingFacility.add_span` signature
contract — every field on :class:`Span` that callers populate at
creation time MUST be reachable via an ``add_span`` kwarg, not via
post-hoc attribute mutation. Run 6 forensic surfaced ``output_summary``
as the drift: ``AgentTracingFacility.emit_lifecycle_event`` passed
``output_summary=details`` but ``add_span`` didn't accept the kwarg,
raising ``TypeError`` on every agent stop (orphan asyncio task fired
for both coordinators in the run6 logs)."""

from __future__ import annotations

import inspect
from typing import Any


def test_add_span_signature_accepts_output_summary() -> None:
    """The R6-FIX-2 contract: ``output_summary`` is a first-class
    kwarg on ``add_span``. Pin via signature inspection so a future
    refactor that drops the kwarg surfaces here, not at runtime on
    every agent stop."""

    from polymathera.colony.distributed.observability.facility import (
        TracingFacility,
    )

    params = inspect.signature(TracingFacility.add_span).parameters
    assert "output_summary" in params, (
        "add_span must accept output_summary; see run6 forensic + "
        "R6-FIX-2 in colony/claude_followups.md"
    )
    # Same audit covers ``input_summary``; the deeper invariant is
    # "every Span field caller populates at creation time is reachable
    # via a kwarg, not via post-hoc attribute mutation".
    assert "input_summary" in params


def test_add_span_propagates_output_summary_to_returned_span() -> None:
    """Round-trip: passing ``output_summary=`` to ``add_span`` lands
    on the returned :class:`Span` instance. Uses a minimal concrete
    subclass that stubs the abstract methods + a real
    :class:`TracingConfig` with tracing disabled so no Kafka producer
    starts."""

    from polymathera.colony.distributed.hooks import HookRegistry, HookContext
    from polymathera.colony.distributed.observability.facility import (
        TracingFacility,
        TracingConfig,
    )
    from polymathera.colony.distributed.observability.models import (
        Span,
        SpanKind,
        SpanStatus,
    )

    class _StubFacility(TracingFacility):
        def get_hook_registry(self) -> HookRegistry:  # type: ignore[override]
            return HookRegistry()

        def get_pointcuts(self):  # type: ignore[override]
            return []

        def get_trace_id(self) -> str:  # type: ignore[override]
            return "trace_stub"

        def get_agent_id(self) -> str:  # type: ignore[override]
            return "agent_stub"

        def get_root_span_id(self) -> str | None:  # type: ignore[override]
            return None

        def summarize_input(
            self, ctx: HookContext, kind: SpanKind,
        ) -> dict[str, Any]:
            return {}

        def summarize_output(
            self, join_point: str, kind: SpanKind, result: Any,
        ) -> dict[str, Any]:
            return {}

        def enrich_span(  # type: ignore[override]
            self, span: Span, ctx: HookContext, kind: SpanKind,
        ) -> None:
            return None

    config = TracingConfig(enabled=False)
    facility = _StubFacility(config)

    details = {"stop_reason": "policy_completed", "iterations": 7}
    span = facility.add_span(
        name="lifecycle:stop",
        kind=SpanKind.LIFECYCLE,
        span_id="span_xyz",
        parent_span_id=None,
        status=SpanStatus.OK,
        output_summary=details,
        finish=True,
    )
    assert span.output_summary == details

    # Mutable-default identity pin (also fixed in R6-FIX-2): two
    # add_span calls must not share the same input_summary dict
    # instance — bound by the old signature's `dict = {}` default.
    sibling = facility.add_span(
        name="another",
        kind=SpanKind.LIFECYCLE,
        span_id="span_abc",
        parent_span_id=None,
        status=SpanStatus.OK,
        finish=True,
    )
    assert span.input_summary == {}
    assert sibling.input_summary == {}
    assert span.input_summary is not sibling.input_summary, (
        "Mutable-default identity leak: two add_span calls should "
        "not share the same input_summary dict instance"
    )
