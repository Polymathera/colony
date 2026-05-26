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


# ---------------------------------------------------------------------------
# compile() machinery
# ---------------------------------------------------------------------------


async def test_decorator_auto_compiles_returned_closure() -> None:
    """After the @deferred decorator returns the closure (extraction
    mode), the closure is already compiled — the transient capability
    reference is cleared and ``is_compiled`` is True."""
    cap = _ScaleTool(agent=None, scope_id="t")
    with eager_execution(False):
        closure = await cap.apply()
    assert closure.is_compiled is True
    # Transient capability reference cleared post-compile.
    assert closure._capability_pre_compile is None


async def test_compile_is_idempotent() -> None:
    """Re-calling compile() on an already-compiled closure is a
    no-op; doesn't raise; doesn't re-extract."""
    cap = _ScaleTool(agent=None, scope_id="t")
    closure = _ScaleClosure(capability=cap, offset=0.1)
    await closure.compile()
    assert closure.is_compiled is True
    # Second + later calls don't raise + don't change state.
    await closure.compile()
    await closure.compile()
    assert closure.is_compiled is True
    assert closure._capability_pre_compile is None


async def test_uncompiled_closure_still_callable_if_manually_compiled() -> None:
    """Tests that build closures directly (bypassing the decorator)
    must call ``await closure.compile()`` themselves before invoking."""
    cap = _ScaleTool(agent=None, scope_id="t")
    closure = _ScaleClosure(capability=cap, offset=0.1)
    # Manual compile (what tests do when bypassing the decorator).
    await closure.compile()
    result = await closure(x=2.0, scale=3.0)
    assert result.payload["y"] == pytest.approx(6.1)


# ---------------------------------------------------------------------------
# Composite closures — compile() extracts sub-closures recursively
# ---------------------------------------------------------------------------


class _CompositeClosure(DeferredClosure[dict]):
    """Composite closure: composes two ``apply`` sub-closures from
    the same capability. Demonstrates the compile() pattern for
    sub-closure extraction."""

    def __init__(self, *, capability: Any) -> None:
        super().__init__(capability)

    async def compile(self) -> None:
        if self.is_compiled:
            return
        cap = self._capability_pre_compile
        # Extract sub-closure via the capability's @deferred action.
        # Under ``eager_execution(False)``, the inner @deferred wrapper
        # returns the closure object (already compiled by the inner
        # auto-compile) instead of invoking it.
        with eager_execution(False):
            self.sub = await cap.apply()
        # Sub-closures arrive already compiled; we just need to
        # finalize ourselves.
        await super().compile()

    async def __call__(self, *, x: float = 0.0, scale: float = 1.0) -> Any:
        # Awaits the pre-compiled sub-closure.
        return await self.sub(x=x, scale=scale)


class _CompositeTool(_ScaleTool):
    """Adds a composite @deferred action that wraps ``apply``."""

    spec: ClassVar[ToolSpec] = _ScaleTool.spec.model_copy(
        update={"name": "composite_tool", "capabilities": ("apply", "compose")},
    )

    @action_executor()
    @deferred
    async def compose(self, *, x: float = 0.0, scale: float = 1.0) -> _CompositeClosure:
        del x, scale  # composite knows only the closure class
        return _CompositeClosure(capability=self)


async def test_composite_compile_extracts_sub_closure() -> None:
    """Composite's compile() extracts the sub-closure via the
    capability's @deferred action — sub-closure arrives compiled."""
    cap = _CompositeTool(agent=None, scope_id="t")
    with eager_execution(False):
        closure = await cap.compose()
    assert isinstance(closure, _CompositeClosure)
    assert closure.is_compiled is True
    # Sub-closure was extracted + auto-compiled by the inner @deferred
    # call inside compile().
    assert isinstance(closure.sub, _ScaleClosure)
    assert closure.sub.is_compiled is True
    assert closure.sub._capability_pre_compile is None


async def test_composite_eager_mode_invokes_recursively() -> None:
    """Eager mode on a composite: action body builds composite,
    decorator auto-compiles (extracts sub-closure), then invokes
    composite's __call__ which awaits the sub-closure."""
    cap = _CompositeTool(agent=None, scope_id="t")
    result = await cap.compose(x=2.0, scale=3.0)
    # offset=0.1 (in sub-closure), x=2.0, scale=3.0 → y = 6.1.
    assert result.payload["y"] == pytest.approx(6.1)
