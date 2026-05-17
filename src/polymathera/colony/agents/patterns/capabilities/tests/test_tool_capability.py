"""Tests for ``ToolCapability``, ``LocalToolCapability``, ``SandboxToolCapability``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.capabilities.tool import (
    TOOL_TAG,
    LocalToolCapability,
    SandboxToolCapability,
    ToolCapability,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
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


@pytest.fixture(autouse=True)
def _exec_ctx():
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _fake_agent() -> MagicMock:
    agent = MagicMock()
    agent.agent_id = "agent-tool-test"
    agent.syscontext.tenant_id = "t1"
    agent.syscontext.colony_id = "c1"
    agent.get_capability_by_type.return_value = None
    return agent


# Reusable spec for the simple-case tests.
_BASIC_SPEC = ToolSpec(
    name="sample_tool",
    domain="general",
    capabilities=("do_thing",),
    backend="in_process",
    execution_locality=ExecutionLocality.LOCAL,
    determinism=Determinism.DETERMINISTIC,
    cost_model=CostModel(cpu_seconds=0.5, memory_gb=0.1),
    resource_requirements=ResourceRequirements(min_vcpus=1, min_memory_gb=0.5),
    headless=HeadlessReadiness.NATIVE,
    hitl_frequency=HITLFrequency.AUTONOMOUS,
    licensing=Licensing.MIT,
)


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement
# ---------------------------------------------------------------------------


def test_subclass_without_spec_raises_at_class_creation() -> None:
    with pytest.raises(TypeError, match=r"does not declare a class-level"):
        class _MissingSpec(LocalToolCapability):
            pass


def test_subclass_with_non_toolspec_spec_raises() -> None:
    with pytest.raises(TypeError, match="must be a ToolSpec instance"):
        class _WrongType(LocalToolCapability):
            spec = "not-a-toolspec"  # type: ignore[assignment]


def test_subclass_inheriting_spec_from_parent_is_allowed() -> None:
    class _ParentTool(LocalToolCapability):
        spec = _BASIC_SPEC

    # Child doesn't declare ``spec`` itself but inherits from _ParentTool.
    class _ChildTool(_ParentTool):
        pass

    assert _ChildTool.spec is _BASIC_SPEC


# ---------------------------------------------------------------------------
# Tag merging
# ---------------------------------------------------------------------------


class _DomainTaggedTool(LocalToolCapability):
    spec = _BASIC_SPEC

    def _domain_tags(self) -> frozenset[str]:
        return frozenset({"em", "fdtd"})


def test_capability_tags_always_include_tool_tag() -> None:
    cap = _DomainTaggedTool(agent=_fake_agent())
    tags = cap.get_capability_tags()
    assert TOOL_TAG in tags
    assert "em" in tags
    assert "fdtd" in tags


def test_capability_tags_without_domain_override_returns_just_tool() -> None:
    class _PlainTool(LocalToolCapability):
        spec = _BASIC_SPEC

    cap = _PlainTool(agent=_fake_agent())
    assert cap.get_capability_tags() == frozenset({TOOL_TAG})


# ---------------------------------------------------------------------------
# Action-group description folds the spec metadata
# ---------------------------------------------------------------------------


def test_action_group_description_renders_spec_metadata() -> None:
    cap = _DomainTaggedTool(agent=_fake_agent())
    desc = cap.get_action_group_description()
    # Spot-check that every spec dimension the planner needs appears.
    assert "sample_tool" in desc
    assert "domain=general" in desc
    assert "backend=in_process" in desc
    assert "locality=local" in desc
    assert "do_thing" in desc
    assert "mit" in desc.lower()
    assert "native" in desc.lower()
    assert "autonomous" in desc.lower()
    assert "deterministic" in desc.lower()
    # Cost + resource numbers.
    assert "cpu=0.5s" in desc
    assert "memory=0.1 GB" in desc
    assert "1 vCPU" in desc
    assert "0.5 GB RAM" in desc


def test_action_group_description_renders_gpu_requirement_when_present() -> None:
    spec_with_gpu = ToolSpec(
        name="gpu_tool",
        capabilities=("solve_em",),
        backend="http_api",
        execution_locality=ExecutionLocality.HPC,
        resource_requirements=ResourceRequirements(
            min_vcpus=8, min_memory_gb=64.0,
            gpu=GpuRequirement(kind="a100", count=2, memory_gb=40.0),
            expected_wallclock_seconds=3600.0,
        ),
    )

    class _GpuTool(LocalToolCapability):
        spec = spec_with_gpu

    cap = _GpuTool(agent=_fake_agent())
    desc = cap.get_action_group_description()
    assert "2× a100" in desc
    assert "40 GB" in desc
    assert "3600s" in desc


def test_describe_tool_extras_hook_extends_description() -> None:
    class _ExtrasTool(LocalToolCapability):
        spec = _BASIC_SPEC

        def _describe_tool_extras(self) -> str:
            return "Extra note: requires the SERF_CALIB env var."

    cap = _ExtrasTool(agent=_fake_agent())
    desc = cap.get_action_group_description()
    assert "requires the SERF_CALIB env var" in desc


# ---------------------------------------------------------------------------
# check_preconditions action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_preconditions_returns_spec_resource_requirements() -> None:
    cap = _DomainTaggedTool(agent=_fake_agent())
    report = await cap.check_preconditions()
    assert report["tool"] == "sample_tool"
    assert report["execution_locality"] == "local"
    assert report["ok"] is True
    assert report["warnings"] == []
    assert report["resource_requirements"]["min_vcpus"] == 1
    assert report["resource_requirements"]["min_memory_gb"] == 0.5


# ---------------------------------------------------------------------------
# Subclass actions are dispatchable
# ---------------------------------------------------------------------------


class _AddTool(LocalToolCapability):
    spec = ToolSpec(
        name="add_tool",
        capabilities=("add",),
        backend="in_process",
    )

    @action_executor()
    async def add(self, *, x: int, y: int) -> dict[str, int]:
        return {"sum": x + y}


@pytest.mark.asyncio
async def test_subclass_action_executor_method_is_invocable() -> None:
    cap = _AddTool(agent=_fake_agent())
    result = await cap.add(x=2, y=40)
    assert result == {"sum": 42}


# ---------------------------------------------------------------------------
# SandboxToolCapability — image_role enforcement + sandbox delegation
# ---------------------------------------------------------------------------


def test_sandbox_subclass_without_image_role_raises() -> None:
    with pytest.raises(TypeError, match="sandbox_image_role"):
        class _MissingRole(SandboxToolCapability):
            spec = _BASIC_SPEC
            # sandbox_image_role intentionally omitted


class _SandboxedSampleTool(SandboxToolCapability):
    spec = ToolSpec(
        name="sandbox_sample_tool",
        capabilities=("echo",),
        backend="docker",
    )
    sandbox_image_role = "code_analysis"

    @action_executor()
    async def echo(self, *, text: str) -> dict[str, Any]:
        return await self._exec_in_sandbox(["echo", text])


@pytest.mark.asyncio
async def test_sandbox_tool_describes_image_role_in_extras() -> None:
    cap = _SandboxedSampleTool(agent=_fake_agent())
    desc = cap.get_action_group_description()
    assert "sandbox_sample_tool" in desc
    assert "code_analysis" in desc


@pytest.mark.asyncio
async def test_sandbox_tool_lazily_launches_and_reuses_container() -> None:
    """First action call launches a container via the agent's
    SandboxedShellCapability; subsequent calls reuse it."""
    from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
        SandboxedShellCapability,
    )

    shell = MagicMock(spec=SandboxedShellCapability)
    shell.launch_container = AsyncMock(return_value={
        "started": True,
        "container_id": "cont_abc",
        "container_name": "test-container",
        "image": "polymathera/code-analysis:0.1",
    })
    shell.execute_command = AsyncMock(return_value={
        "container_id": "cont_abc",
        "exit_code": 0,
        "stdout": "hello\n",
        "stderr": "",
    })
    shell.stop_container = AsyncMock(return_value={"container_id": "cont_abc", "stopped": True})

    agent = _fake_agent()

    def _resolve(typ):
        if typ is SandboxedShellCapability:
            return shell
        return None

    agent.get_capability_by_type.side_effect = _resolve

    cap = _SandboxedSampleTool(agent=agent)
    await cap.echo(text="hello")
    await cap.echo(text="world")

    # launch_container is called once, execute_command twice.
    assert shell.launch_container.call_count == 1
    shell.launch_container.assert_called_with(image_role="code_analysis")
    assert shell.execute_command.call_count == 2

    # Both exec calls target the same container_id returned by launch.
    for call in shell.execute_command.call_args_list:
        assert call.args[0] == "cont_abc"


@pytest.mark.asyncio
async def test_sandbox_tool_raises_when_shell_capability_missing() -> None:
    cap = _SandboxedSampleTool(agent=_fake_agent())
    # _fake_agent's get_capability_by_type returns None for every type.
    with pytest.raises(RuntimeError, match="SandboxedShellCapability"):
        await cap.echo(text="hello")


@pytest.mark.asyncio
async def test_sandbox_tool_raises_when_detached() -> None:
    cap = _SandboxedSampleTool(
        agent=None,
        scope_id="standalone-test",
    )
    with pytest.raises(RuntimeError, match="detached.*SandboxedShellCapability"):
        await cap.echo(text="hello")


@pytest.mark.asyncio
async def test_sandbox_tool_shutdown_stops_container_when_launched() -> None:
    from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
        SandboxedShellCapability,
    )

    shell = MagicMock(spec=SandboxedShellCapability)
    shell.launch_container = AsyncMock(return_value={
        "started": True, "container_id": "cont_xyz",
        "container_name": "n", "image": "i",
    })
    shell.execute_command = AsyncMock(return_value={
        "container_id": "cont_xyz", "exit_code": 0, "stdout": "", "stderr": "",
    })
    shell.stop_container = AsyncMock(return_value={"container_id": "cont_xyz", "stopped": True})

    agent = _fake_agent()
    agent.get_capability_by_type.side_effect = (
        lambda typ: shell if typ is SandboxedShellCapability else None
    )

    cap = _SandboxedSampleTool(agent=agent)
    await cap.echo(text="x")  # triggers launch
    await cap.shutdown()
    shell.stop_container.assert_called_once_with("cont_xyz")


@pytest.mark.asyncio
async def test_sandbox_tool_shutdown_is_noop_when_no_container_launched() -> None:
    """If no action ran, shutdown should not call stop_container."""
    from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
        SandboxedShellCapability,
    )

    shell = MagicMock(spec=SandboxedShellCapability)
    shell.stop_container = AsyncMock()
    agent = _fake_agent()
    agent.get_capability_by_type.side_effect = (
        lambda typ: shell if typ is SandboxedShellCapability else None
    )

    cap = _SandboxedSampleTool(agent=agent)
    await cap.shutdown()
    shell.stop_container.assert_not_called()


# ---------------------------------------------------------------------------
# Capability-key default = spec.name (so multiple tool capabilities mounted
# on the same agent get distinct dispatch keys)
# ---------------------------------------------------------------------------


def test_capability_key_defaults_to_spec_name() -> None:
    cap_a = _AddTool(agent=_fake_agent())
    cap_b = _DomainTaggedTool(agent=_fake_agent())
    assert cap_a._capability_key == "add_tool"
    assert cap_b._capability_key == "sample_tool"
