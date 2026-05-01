"""Tests for ``ToolRegistry`` resolution + preferences."""

from __future__ import annotations

import pytest

from polymathera.colony.tools import (
    CostModel,
    Determinism,
    DuplicateAdapter,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    NoAdapterAvailable,
    Preferences,
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


def test_list_capabilities() -> None:
    reg = ToolRegistry()
    reg.register(_make_adapter_class(name="a", capabilities=("c1", "c2"))())
    reg.register(_make_adapter_class(name="b", capabilities=("c2",))())
    assert list(reg.list_capabilities()) == ["c1", "c2"]
    assert [type(a).spec.name for a in reg.list_adapters_for("c2")] == ["a", "b"]
