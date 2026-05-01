"""Tests for ``ToolSpec`` + enums + value records."""

from __future__ import annotations

import pytest

from polymathera.colony.tools import (
    CostModel,
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def test_headless_order_strictly_decreasing() -> None:
    order = [
        HeadlessReadiness.NATIVE,
        HeadlessReadiness.CLI_ONLY,
        HeadlessReadiness.PARTIAL,
        HeadlessReadiness.GUI_PRIMARY,
        HeadlessReadiness.NONE,
    ]
    for a, b in zip(order, order[1:]):
        assert a.order > b.order


def test_hitl_order_strictly_increasing() -> None:
    order = [
        HITLFrequency.AUTONOMOUS,
        HITLFrequency.REVIEW_MILESTONES,
        HITLFrequency.APPROVAL_GATES,
        HITLFrequency.CO_PILOT,
        HITLFrequency.HUMAN_PRIMARY,
    ]
    for a, b in zip(order, order[1:]):
        assert a.order < b.order


def test_toolspec_capabilities_are_sorted_and_unique() -> None:
    spec = ToolSpec(
        name="x", capabilities=("z", "a", "m", "a"),  # duplicate + unsorted
    )
    assert spec.capabilities == ("a", "m", "z")


def test_toolspec_capabilities_accept_single_string() -> None:
    spec = ToolSpec(name="x", capabilities="solo")  # type: ignore[arg-type]
    assert spec.capabilities == ("solo",)


def test_toolspec_frozen_rejects_mutation() -> None:
    spec = ToolSpec(name="x")
    with pytest.raises(Exception):
        spec.name = "y"  # type: ignore[misc]


def test_toolspec_fulfils() -> None:
    spec = ToolSpec(name="x", capabilities=("a", "b"))
    assert spec.fulfils("a")
    assert not spec.fulfils("c")


def test_costmodel_defaults_zero() -> None:
    c = CostModel()
    assert c.cpu_seconds == 0.0
    assert c.dollars == 0.0


def test_costmodel_validates_negative() -> None:
    with pytest.raises(Exception):
        CostModel(cpu_seconds=-1.0)


def test_toolcall_round_trip() -> None:
    c = ToolCall(capability="x", parameters={"a": 1})
    assert c.capability == "x"
    assert c.parameters == {"a": 1}
    assert c.call_id.startswith("call_")


def test_toolresult_round_trip() -> None:
    r = ToolResult(call_id="cid", adapter_name="ad", success=True, value={"k": "v"})
    assert r.success
    assert r.value == {"k": "v"}


def test_licensing_enum_strings_stable() -> None:
    # The string values are part of the on-wire contract; surface
    # any rename loudly.
    assert Licensing.COMMERCIAL.value == "commercial"
    assert Licensing.RESTRICTED.value == "restricted"
    assert Licensing.MIT.value == "mit"
    assert Licensing.GPL.value == "gpl"
    assert Determinism.DETERMINISTIC.value == "deterministic"
