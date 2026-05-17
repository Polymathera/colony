"""Tests for ``ToolRegistry`` resolution + preferences."""

from __future__ import annotations

import pytest

from polymathera.colony.tools import (
    CostModel,
    Determinism,
    DuplicateAdapter,
    ExecutionLocality,
    GpuRequirement,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    NoAdapterAvailable,
    Preferences,
    ResourceRequirements,
    ToolAdapter,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def _make_adapter_class(
    *,
    name: str,
    capabilities: tuple[str, ...] = ("solve",),
    headless: HeadlessReadiness = HeadlessReadiness.NATIVE,
    hitl: HITLFrequency = HITLFrequency.AUTONOMOUS,
    determinism: Determinism = Determinism.DETERMINISTIC,
    licensing: Licensing = Licensing.MIT,
    backend: str = "in_process",
    interruptibility: bool = False,
    container_image: str | None = None,
    cpu_seconds: float = 0.0,
    dollars: float = 0.0,
    execution_locality: ExecutionLocality = ExecutionLocality.LOCAL,
    resource_requirements: ResourceRequirements | None = None,
) -> type[ToolAdapter]:
    spec_obj = ToolSpec(
        name=name,
        capabilities=capabilities,
        headless=headless,
        hitl_frequency=hitl,
        determinism=determinism,
        licensing=licensing,
        backend=backend,
        interruptibility=interruptibility,
        container_image=container_image,
        cost_model=CostModel(cpu_seconds=cpu_seconds, dollars=dollars),
        execution_locality=execution_locality,
        resource_requirements=resource_requirements or ResourceRequirements(),
    )

    class _A(ToolAdapter):
        spec = spec_obj

        async def invoke(self, call: ToolCall) -> ToolResult:
            return ToolResult(
                call_id=call.call_id, adapter_name=name, success=True,
                value=call.parameters,
            )

    return _A


def test_register_and_resolve() -> None:
    reg = ToolRegistry()
    A = _make_adapter_class(name="a")
    reg.register(A())
    adapter = reg.resolve("solve")
    assert type(adapter).spec.name == "a"


def test_duplicate_register_rejected() -> None:
    reg = ToolRegistry()
    A = _make_adapter_class(name="a")
    reg.register(A())
    with pytest.raises(DuplicateAdapter):
        reg.register(A())


def test_unregister_removes_capability_index() -> None:
    reg = ToolRegistry()
    A = _make_adapter_class(name="a")
    reg.register(A())
    assert reg.unregister("a") is True
    assert list(reg.list_capabilities()) == []
    assert reg.unregister("a") is False


def test_resolve_no_capability() -> None:
    reg = ToolRegistry()
    with pytest.raises(NoAdapterAvailable):
        reg.resolve("missing")


def test_min_headless_filter() -> None:
    reg = ToolRegistry()
    Native = _make_adapter_class(name="n", headless=HeadlessReadiness.NATIVE)
    Gui = _make_adapter_class(name="g", headless=HeadlessReadiness.GUI_PRIMARY)
    reg.register(Native())
    reg.register(Gui())
    chosen = reg.resolve("solve", Preferences(min_headless=HeadlessReadiness.NATIVE))
    assert type(chosen).spec.name == "n"


def test_max_hitl_filter_excludes_human_primary() -> None:
    reg = ToolRegistry()
    Auto = _make_adapter_class(name="a", hitl=HITLFrequency.AUTONOMOUS)
    Hp = _make_adapter_class(name="h", hitl=HITLFrequency.HUMAN_PRIMARY)
    reg.register(Auto())
    reg.register(Hp())
    chosen = reg.resolve(
        "solve", Preferences(max_hitl=HITLFrequency.REVIEW_MILESTONES),
    )
    assert type(chosen).spec.name == "a"


def test_required_determinism_filter() -> None:
    reg = ToolRegistry()
    D = _make_adapter_class(name="d", determinism=Determinism.DETERMINISTIC)
    S = _make_adapter_class(name="s", determinism=Determinism.STOCHASTIC)
    reg.register(D())
    reg.register(S())
    chosen = reg.resolve(
        "solve", Preferences(required_determinism=Determinism.DETERMINISTIC),
    )
    assert type(chosen).spec.name == "d"


def test_forbid_licences() -> None:
    reg = ToolRegistry()
    Mit = _make_adapter_class(name="mit", licensing=Licensing.MIT)
    Comm = _make_adapter_class(name="comm", licensing=Licensing.COMMERCIAL)
    reg.register(Mit())
    reg.register(Comm())
    chosen = reg.resolve(
        "solve",
        Preferences(forbid_licences=frozenset({Licensing.COMMERCIAL})),
    )
    assert type(chosen).spec.name == "mit"


def test_max_dollars_filter() -> None:
    reg = ToolRegistry()
    Cheap = _make_adapter_class(name="cheap", dollars=0.01)
    Expensive = _make_adapter_class(name="expensive", dollars=2.0)
    reg.register(Cheap())
    reg.register(Expensive())
    chosen = reg.resolve("solve", Preferences(max_dollars=0.5))
    assert type(chosen).spec.name == "cheap"


def test_require_interruptible_filter() -> None:
    reg = ToolRegistry()
    NoInt = _make_adapter_class(name="ni", interruptibility=False)
    Int = _make_adapter_class(name="ok", interruptibility=True)
    reg.register(NoInt())
    reg.register(Int())
    chosen = reg.resolve("solve", Preferences(require_interruptible=True))
    assert type(chosen).spec.name == "ok"


def test_required_backend_filter() -> None:
    reg = ToolRegistry()
    Ip = _make_adapter_class(name="ip", backend="in_process")
    Cli = _make_adapter_class(name="cli", backend="cli_subprocess")
    reg.register(Ip())
    reg.register(Cli())
    chosen = reg.resolve(
        "solve", Preferences(required_backend="cli_subprocess"),
    )
    assert type(chosen).spec.name == "cli"


def test_preferred_backend_scores_higher() -> None:
    reg = ToolRegistry()
    Ip = _make_adapter_class(name="ip", backend="in_process")
    Cli = _make_adapter_class(name="cli", backend="cli_subprocess")
    reg.register(Ip())
    reg.register(Cli())
    chosen = reg.resolve(
        "solve", Preferences(preferred_backend="cli_subprocess"),
    )
    assert type(chosen).spec.name == "cli"


def test_resolve_all_returns_score_sorted() -> None:
    reg = ToolRegistry()
    Native = _make_adapter_class(name="native", headless=HeadlessReadiness.NATIVE)
    Cli = _make_adapter_class(name="cli", headless=HeadlessReadiness.CLI_ONLY)
    reg.register(Native())
    reg.register(Cli())
    survivors = reg.resolve_all("solve")
    assert [type(a).spec.name for a in survivors] == ["native", "cli"]


def test_lower_cost_preferred_when_tier_breaks() -> None:
    reg = ToolRegistry()
    Cheap = _make_adapter_class(name="cheap", dollars=0.01)
    Expensive = _make_adapter_class(name="expensive", dollars=2.0)
    reg.register(Cheap())
    reg.register(Expensive())
    chosen = reg.resolve("solve")
    assert type(chosen).spec.name == "cheap"


def test_allowed_container_images() -> None:
    reg = ToolRegistry()
    A = _make_adapter_class(name="a", container_image="colony-base:0.1")
    B = _make_adapter_class(name="b", container_image="colony-cad:0.1")
    reg.register(A())
    reg.register(B())
    chosen = reg.resolve(
        "solve",
        Preferences(allowed_container_images=frozenset({"colony-cad:0.1"})),
    )
    assert type(chosen).spec.name == "b"


def test_allowed_localities_drops_hpc_when_local_only() -> None:
    reg = ToolRegistry()
    Local = _make_adapter_class(name="local", execution_locality=ExecutionLocality.LOCAL)
    Hpc = _make_adapter_class(name="hpc", execution_locality=ExecutionLocality.HPC)
    reg.register(Local())
    reg.register(Hpc())
    chosen = reg.resolve(
        "solve",
        Preferences(allowed_localities=frozenset({ExecutionLocality.LOCAL})),
    )
    assert type(chosen).spec.name == "local"


def test_allowed_localities_none_accepts_all() -> None:
    reg = ToolRegistry()
    Local = _make_adapter_class(name="local", execution_locality=ExecutionLocality.LOCAL)
    Hpc = _make_adapter_class(name="hpc", execution_locality=ExecutionLocality.HPC)
    reg.register(Local())
    reg.register(Hpc())
    survivors = reg.resolve_all("solve", Preferences())
    assert sorted(type(a).spec.name for a in survivors) == ["hpc", "local"]


def test_max_required_vcpus_drops_heavy_tool() -> None:
    reg = ToolRegistry()
    Light = _make_adapter_class(
        name="light",
        resource_requirements=ResourceRequirements(min_vcpus=4),
    )
    Heavy = _make_adapter_class(
        name="heavy",
        resource_requirements=ResourceRequirements(min_vcpus=64),
    )
    reg.register(Light())
    reg.register(Heavy())
    chosen = reg.resolve("solve", Preferences(max_required_vcpus=16))
    assert type(chosen).spec.name == "light"


def test_max_required_memory_gb_drops_heavy_tool() -> None:
    reg = ToolRegistry()
    Light = _make_adapter_class(
        name="light",
        resource_requirements=ResourceRequirements(min_memory_gb=8),
    )
    Heavy = _make_adapter_class(
        name="heavy",
        resource_requirements=ResourceRequirements(min_memory_gb=256),
    )
    reg.register(Light())
    reg.register(Heavy())
    chosen = reg.resolve("solve", Preferences(max_required_memory_gb=64))
    assert type(chosen).spec.name == "light"


def test_max_required_wallclock_drops_long_jobs() -> None:
    reg = ToolRegistry()
    Quick = _make_adapter_class(
        name="quick",
        resource_requirements=ResourceRequirements(expected_wallclock_seconds=60),
    )
    Slow = _make_adapter_class(
        name="slow",
        resource_requirements=ResourceRequirements(expected_wallclock_seconds=36000),
    )
    reg.register(Quick())
    reg.register(Slow())
    chosen = reg.resolve("solve", Preferences(max_required_wallclock_seconds=3600))
    assert type(chosen).spec.name == "quick"


def test_allowed_required_gpu_kinds_drops_disallowed() -> None:
    reg = ToolRegistry()
    A10g = _make_adapter_class(
        name="a10g",
        resource_requirements=ResourceRequirements(
            gpu=GpuRequirement(kind="a10g", count=1),
        ),
    )
    A100 = _make_adapter_class(
        name="a100",
        resource_requirements=ResourceRequirements(
            gpu=GpuRequirement(kind="a100", count=1),
        ),
    )
    Cpu = _make_adapter_class(name="cpu")  # no gpu — always passes
    reg.register(A10g())
    reg.register(A100())
    reg.register(Cpu())
    survivors = reg.resolve_all(
        "solve",
        Preferences(allowed_required_gpu_kinds=frozenset({"a10g"})),
    )
    names = sorted(type(a).spec.name for a in survivors)
    # A100 dropped; A10g + Cpu (no GPU) survive.
    assert names == ["a10g", "cpu"]


def test_max_required_gpu_count_drops_multi_gpu() -> None:
    reg = ToolRegistry()
    Single = _make_adapter_class(
        name="single",
        resource_requirements=ResourceRequirements(
            gpu=GpuRequirement(kind="a100", count=1),
        ),
    )
    Quad = _make_adapter_class(
        name="quad",
        resource_requirements=ResourceRequirements(
            gpu=GpuRequirement(kind="a100", count=4),
        ),
    )
    reg.register(Single())
    reg.register(Quad())
    chosen = reg.resolve("solve", Preferences(max_required_gpu_count=2))
    assert type(chosen).spec.name == "single"


def test_resource_requirement_filters_are_independent_of_cost_model() -> None:
    """Regression: ``max_memory_gb`` (cost) and ``max_required_memory_gb``
    (requirement) are distinct fields with distinct meanings — a tool
    can have low estimated cost AND high minimum-required memory."""
    reg = ToolRegistry()
    spec_obj = ToolSpec(
        name="parallel_calculix",
        capabilities=("solve",),
        cost_model=CostModel(memory_gb=2.0),  # estimated 2 GB per call
        resource_requirements=ResourceRequirements(min_memory_gb=64.0),  # but needs 64 GB minimum
    )

    class _A(ToolAdapter):
        spec = spec_obj

        async def invoke(self, call: ToolCall) -> ToolResult:
            return ToolResult(call_id=call.call_id, adapter_name="parallel_calculix", success=True)

    reg.register(_A())

    # max_memory_gb on cost passes (2.0 ≤ 4); max_required_memory_gb drops it.
    with pytest.raises(NoAdapterAvailable):
        reg.resolve("solve", Preferences(max_memory_gb=4.0, max_required_memory_gb=32.0))
    # Same with just max_required_memory_gb.
    with pytest.raises(NoAdapterAvailable):
        reg.resolve("solve", Preferences(max_required_memory_gb=32.0))
    # Lifting the requirement cap lets it through.
    chosen = reg.resolve("solve", Preferences(max_memory_gb=4.0, max_required_memory_gb=128.0))
    assert type(chosen).spec.name == "parallel_calculix"


def test_list_capabilities() -> None:
    reg = ToolRegistry()
    reg.register(_make_adapter_class(name="a", capabilities=("c1", "c2"))())
    reg.register(_make_adapter_class(name="b", capabilities=("c2",))())
    assert list(reg.list_capabilities()) == ["c1", "c2"]
    assert [type(a).spec.name for a in reg.list_adapters_for("c2")] == ["a", "b"]
