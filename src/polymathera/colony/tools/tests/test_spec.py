"""Tests for ``ToolSpec`` + enums + per-call cost / resource shapes."""

from __future__ import annotations

import pytest

from polymathera.colony.tools import (
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


def test_licensing_enum_strings_stable() -> None:
    # The string values are part of the on-wire contract; surface
    # any rename loudly.
    assert Licensing.COMMERCIAL.value == "commercial"
    assert Licensing.RESTRICTED.value == "restricted"
    assert Licensing.MIT.value == "mit"
    assert Licensing.GPL.value == "gpl"
    assert Determinism.DETERMINISTIC.value == "deterministic"


def test_execution_locality_enum_strings_stable() -> None:
    # On-wire contract for cps.hpc.* config + REST API.
    assert ExecutionLocality.LOCAL.value == "local"
    assert ExecutionLocality.HPC.value == "hpc"
    assert ExecutionLocality.CUSTOMER_SITE.value == "customer_site"


def test_toolspec_execution_locality_defaults_local() -> None:
    spec = ToolSpec(name="x")
    assert spec.execution_locality == ExecutionLocality.LOCAL


def test_toolspec_resource_requirements_defaults() -> None:
    spec = ToolSpec(name="x")
    req = spec.resource_requirements
    assert req.min_vcpus == 1
    assert req.min_memory_gb == 1.0
    assert req.gpu is None
    assert req.expected_wallclock_seconds == 600.0


def test_toolspec_dropped_container_image_field_absent() -> None:
    """v8 retrofit: ``ToolSpec.container_image`` was removed; image
    routing is the responsibility of ``SandboxToolCapability`` /
    ``HPCToolCapability`` (Batch JobDefinition), not metadata on the
    spec."""
    spec = ToolSpec(name="x")
    assert not hasattr(spec, "container_image")
    # Pydantic silently drops unknown kwargs by default (ToolSpec
    # doesn't set ``extra="forbid"``). The important invariant: a
    # legacy caller passing ``container_image=...`` doesn't end up
    # with a populated attribute on the new model.
    spec2 = ToolSpec(name="x", container_image="img:1")  # type: ignore[call-arg]
    assert not hasattr(spec2, "container_image")


def test_gpu_requirement_defaults() -> None:
    gpu = GpuRequirement()
    assert gpu.kind == "any"
    assert gpu.count == 1
    assert gpu.memory_gb is None


def test_resource_requirements_with_gpu() -> None:
    req = ResourceRequirements(
        min_vcpus=16, min_memory_gb=64.0,
        gpu=GpuRequirement(kind="a100", count=2, memory_gb=40.0),
        expected_wallclock_seconds=3600.0,
    )
    assert req.gpu is not None
    assert req.gpu.kind == "a100"
    assert req.gpu.count == 2
