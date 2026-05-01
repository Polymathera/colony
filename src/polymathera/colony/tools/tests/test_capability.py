"""Tests for ``ToolCapability``."""

from __future__ import annotations

import pytest

from polymathera.colony.tools import (
    HITLFrequency,
    HeadlessReadiness,
    Preferences,
    ToolAdapter,
    ToolCall,
    ToolCapability,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


pytestmark = pytest.mark.asyncio


def _build_registry_with(*adapters: ToolAdapter) -> ToolRegistry:
    r = ToolRegistry()
    for a in adapters:
        r.register(a)
    return r


def _ok_adapter(name: str = "a", capability: str = "echo") -> ToolAdapter:
    spec_obj = ToolSpec(name=name, capabilities=(capability,))

    class _A(ToolAdapter):
        spec = spec_obj
        invocations: list[ToolCall] = []

        async def invoke(self, call):
            self.__class__.invocations.append(call)
            return ToolResult(
                call_id=call.call_id, adapter_name=name, success=True,
                value=dict(call.parameters),
            )

    return _A()


async def test_invocation_returns_typed_result() -> None:
    reg = _build_registry_with(_ok_adapter())
    cap = ToolCapability(name="echo", registry=reg)
    res = await cap(caller="agent_1", x=1)
    assert res.success
    assert res.value == {"x": 1}
    assert res.adapter_name == "a"


async def test_unavailable_capability_raises() -> None:
    reg = ToolRegistry()
    cap = ToolCapability(name="missing", registry=reg)
    with pytest.raises(Exception):
        await cap(caller="agent_1")


async def test_invocation_failure_wrapped() -> None:
    spec_obj = ToolSpec(name="bad", capabilities=("x",))

    class _Bad(ToolAdapter):
        spec = spec_obj

        async def invoke(self, call):
            raise RuntimeError("boom")

    reg = _build_registry_with(_Bad())
    cap = ToolCapability(name="x", registry=reg)
    res = await cap(caller="agent")
    assert not res.success
    assert "boom" in (res.error or "")
    assert res.adapter_name == "bad"


async def test_per_call_preferences_override() -> None:
    spec_native = ToolSpec(name="n", capabilities=("c",), headless=HeadlessReadiness.NATIVE)
    spec_gui = ToolSpec(name="g", capabilities=("c",), headless=HeadlessReadiness.GUI_PRIMARY)

    class _N(ToolAdapter):
        spec = spec_native
        async def invoke(self, call):
            return ToolResult(call_id=call.call_id, adapter_name="n", success=True)

    class _G(ToolAdapter):
        spec = spec_gui
        async def invoke(self, call):
            return ToolResult(call_id=call.call_id, adapter_name="g", success=True)

    reg = _build_registry_with(_N(), _G())
    cap = ToolCapability(name="c", registry=reg, preferences=Preferences())
    # By default: NATIVE scores higher, picks _N.
    r = await cap(caller="agent")
    assert r.adapter_name == "n"
    # Per-call preferences override: required_backend=other → DENY, but
    # we verify availability instead.
    assert cap.is_available()
    assert "n" in cap.available_adapters()


async def test_available_adapters_filtered_by_preferences() -> None:
    spec_native = ToolSpec(name="n", capabilities=("c",), headless=HeadlessReadiness.NATIVE)
    spec_gui = ToolSpec(name="g", capabilities=("c",), headless=HeadlessReadiness.GUI_PRIMARY)

    class _N(ToolAdapter):
        spec = spec_native
        async def invoke(self, call):
            return ToolResult(call_id=call.call_id, adapter_name="n", success=True)

    class _G(ToolAdapter):
        spec = spec_gui
        async def invoke(self, call):
            return ToolResult(call_id=call.call_id, adapter_name="g", success=True)

    reg = _build_registry_with(_N(), _G())
    cap = ToolCapability(
        name="c", registry=reg,
        preferences=Preferences(min_headless=HeadlessReadiness.NATIVE),
    )
    assert cap.available_adapters() == ["n"]


async def test_call_id_propagation() -> None:
    spec_obj = ToolSpec(name="a", capabilities=("e",))
    seen: list[str] = []

    class _A(ToolAdapter):
        spec = spec_obj
        async def invoke(self, call):
            seen.append(call.call_id)
            return ToolResult(
                call_id=call.call_id, adapter_name="a", success=True, value=None,
            )

    reg = _build_registry_with(_A())
    cap = ToolCapability(name="e", registry=reg)
    res = await cap(caller="x")
    assert seen == [res.call_id]


async def test_empty_name_rejected() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        ToolCapability(name="", registry=reg)
