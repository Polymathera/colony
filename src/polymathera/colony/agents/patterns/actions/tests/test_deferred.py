"""Tests for the deferred-closure extraction primitive.

Covers:

- :class:`DeferredClosure` constructor snapshots (capability_fqn,
  tool_name, run_id, context).
- :func:`eager_execution` ContextVar toggle.
- :func:`deferred` decorator dual-mode behaviour: eager (default) ⇒
  invokes the closure; deferred (``eager_execution(False)``) ⇒
  surfaces the closure.

The primitive is generic — these tests use a tiny domain-free
fixture. Integration with any specific downstream extractor
(e.g. trial-runnable dispatch) is tested where that extractor
lives, not here.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from polymathera.colony.agents.patterns.actions import (
    DeferredClosure,
    action_executor,
    deferred,
    eager_execution,
    is_eager_execution,
)
from polymathera.colony.agents.patterns.capabilities.tool import (
    LocalToolCapability,
)
from polymathera.colony.tools import (
    CostModel,
    Determinism,
    ExecutionLocality,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ResourceRequirements,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Module-level fixture: a tiny tool with one @deferred action that
# bundles a constant ``offset`` into the closure.
# ---------------------------------------------------------------------------


class _ScaleClosure(DeferredClosure[dict]):
    """Returns ``{"y": x * scale + offset}``. ``scale`` is captured
    in the closure's context; ``x`` is supplied at call time."""

    def __init__(self, *, capability: Any, offset: float) -> None:
        super().__init__(capability, offset=offset)

    async def __call__(self, *, x: float = 0.0, scale: float = 1.0) -> Any:
        offset = self.get_context_by_key("offset")

        class _Result:
            payload = {"y": x * scale + offset}

        return _Result()


class _ScaleTool(LocalToolCapability):
    """Minimal capability with one ``@deferred`` action."""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="scale_tool",
        domain="testing",
        version="0.0.1",
        capabilities=("apply",),
        backend="in_process",
        execution_locality=ExecutionLocality.LOCAL,
        determinism=Determinism.DETERMINISTIC,
        cost_model=CostModel(cpu_seconds=0.001),
        resource_requirements=ResourceRequirements(
            min_vcpus=1, min_memory_gb=0.1,
        ),
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        licensing=Licensing.MIT,
    )

    @action_executor()
    @deferred
    async def apply(
        self, *, x: float = 0.0, scale: float = 1.0,
    ) -> _ScaleClosure:
        # Action body's only job: assemble context + return the
        # closure. Kwargs all have defaults: in deferred-extraction
        # mode the framework calls with no caller-supplied kwargs
        # (or only a subset); the body doesn't reference them, only
        # the closure does.
        del x, scale
        return _ScaleClosure(capability=self, offset=0.1)


# ---------------------------------------------------------------------------
# eager_execution ContextVar
# ---------------------------------------------------------------------------


def test_is_eager_execution_defaults_true() -> None:
    assert is_eager_execution() is True


def test_eager_execution_toggle_scoped() -> None:
    assert is_eager_execution() is True
    with eager_execution(False):
        assert is_eager_execution() is False
        with eager_execution(True):
            assert is_eager_execution() is True
        assert is_eager_execution() is False
    assert is_eager_execution() is True


# ---------------------------------------------------------------------------
# DeferredClosure snapshots
# ---------------------------------------------------------------------------


def test_deferred_closure_snapshots_tool_name_and_fqn() -> None:
    cap = _ScaleTool(agent=None, scope_id="t")
    closure = _ScaleClosure(capability=cap, offset=0.5)
    assert closure.tool_name == "scale_tool"
    assert closure.capability_fqn.endswith("_ScaleTool")
    # No agent → no run_id.
    assert closure.run_id == ""
    assert closure.get_context_by_key("offset") == 0.5


# ---------------------------------------------------------------------------
# @deferred dual-mode dispatch
# ---------------------------------------------------------------------------


async def test_deferred_eager_mode_invokes_closure() -> None:
    """Default eager mode: calling the action returns the closure's
    result (here, a duck-typed object with a payload)."""
    cap = _ScaleTool(agent=None, scope_id="t")
    result = await cap.apply(x=2.0, scale=3.0)
    # offset=0.1 (in closure), x=2.0, scale=3.0 → y = 6.1.
    assert result.payload["y"] == pytest.approx(6.1)


async def test_deferred_extraction_mode_returns_closure_object() -> None:
    """``eager_execution(False)``: calling the action returns the
    DeferredClosure object itself, not its result."""
    cap = _ScaleTool(agent=None, scope_id="t")
    with eager_execution(False):
        closure = await cap.apply()
    assert isinstance(closure, DeferredClosure)
    assert closure.tool_name == "scale_tool"
    assert closure.get_context_by_key("offset") == 0.1


async def test_deferred_action_must_return_deferred_closure() -> None:
    """A @deferred action whose body returns a non-DeferredClosure
    raises TypeError immediately — never silently passes through."""

    class _Misbehaving(LocalToolCapability):
        spec: ClassVar[ToolSpec] = _ScaleTool.spec.model_copy(
            update={"name": "misbehaving"},
        )

        @action_executor()
        @deferred
        async def apply(self) -> Any:
            return 42  # not a DeferredClosure → must TypeError

    cap = _Misbehaving(agent=None, scope_id="t")
    with pytest.raises(TypeError, match="must return a DeferredClosure"):
        await cap.apply()
