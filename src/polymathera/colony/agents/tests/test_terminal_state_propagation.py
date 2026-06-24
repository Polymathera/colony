"""Tests for the agent-termination cleanup contract.

Two bugs landed together on 2026-06-22 (one bug, two symptoms):

1. ``Agent.stop()`` only updated ``self.state`` locally; the
   cluster-wide ``AgentSystemDeployment`` registry was never
   informed of natural termination → ``fetch_agent_info`` /
   ``get_agent_status`` returned the spawn-time ``INITIALIZED``
   state forever, misleading the ``no_unverified_agent_state_claims``
   semantic guardrail and the SessionAgent's reuse-vs-spawn logic.

2. The manager's resource accounting (CPU / mem / GPU counters,
   ``_agents`` dict, ``_agent_tasks`` dict, AgentSystem
   registration) was ONLY cleaned up by :meth:`stop_agent`. Self-
   termination (``policy_completed`` / ``max_iterations`` /
   exception → FAILED) bypassed ``stop_agent`` entirely — the agent
   loop's task just ended, and the manager kept thinking the agent
   owned its resources.

Single fix: extract the cleanup tail from ``stop_agent`` into
:meth:`AgentManagerBase._finalize_agent_cleanup` (idempotent via
``dict.pop``), and call it from both ``stop_agent`` (after the
await-task half) and ``_run_agent_loop``'s ``finally`` (every loop
exit path). ``unregister_agent`` deletes the registry entry, so
the ``get_agent_status`` lie disappears as a side effect.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Source-level pins: both call sites invoke _finalize_agent_cleanup
# ---------------------------------------------------------------------------


def test_run_loop_finally_calls_finalize_agent_cleanup() -> None:
    """``AgentManagerBase._run_agent_loop``'s ``finally`` MUST call
    ``self._finalize_agent_cleanup(agent.agent_id)`` after the
    ``agent.stop()`` ensure block. Without this, self-terminating
    agents leak CPU/mem/GPU counters and stay in the registry as
    ``INITIALIZED`` (the 2026-06-22 ``/decompose`` bug)."""

    base_py = (
        Path(__file__).resolve().parents[1] / "base.py"
    ).read_text(encoding="utf-8")
    assert (
        "await self._finalize_agent_cleanup(agent.agent_id)" in base_py
    ), "_run_agent_loop's finally must call _finalize_agent_cleanup."

    ensure_stop_idx = base_py.index(
        "# Ensure stop() is called if the agent hasn't already stopped"
    )
    finalize_idx = base_py.index(
        "await self._finalize_agent_cleanup(agent.agent_id)",
        ensure_stop_idx,
    )
    assert ensure_stop_idx < finalize_idx, (
        "Cleanup must run AFTER the stop() ensure block so agent.state "
        "has reached its terminal value before resources are released."
    )


def test_run_loop_propagates_running_to_registry_after_agent_start() -> None:
    """D3 (2026-06-23): ``AgentManagerBase._run_agent_loop`` MUST
    call ``self._agent_system_handle.update_agent_state(agent_id,
    agent.state)`` immediately after ``await agent.start()`` runs.

    Without this, the registry has only the spawn-time INITIALIZED
    snapshot — every cross-replica reader (``fetch_agent_info``,
    ``AgentPoolCapability.get_agent_status``, the
    ``no_unverified_agent_state_claims`` semantic guardrail's
    verifier) reports INITIALIZED for the entire run of an
    actively-running agent. The 2026-06-23 ``run3`` log shows zero
    ``update_agent_state(RUNNING)`` calls across 8.5 hours; the
    SessionAgent's LLM faithfully reported "coordinator INITIALIZED"
    to the user for a coordinator actively writing chat messages.
    """

    base_py = (
        Path(__file__).resolve().parents[1] / "base.py"
    ).read_text(encoding="utf-8")
    start_idx = base_py.index("await agent.start()")
    finally_idx = base_py.index(
        "# Ensure stop() is called if the agent hasn't already stopped"
    )
    propagate_idx = base_py.index(
        "update_agent_state(\n                        agent.agent_id, "
        "agent.state,\n                    )",
        start_idx,
    )
    assert start_idx < propagate_idx < finally_idx, (
        "D3: RUNNING-state propagation must run AFTER agent.start() "
        "and BEFORE the loop's finally cleanup."
    )


def test_stop_agent_calls_finalize_agent_cleanup() -> None:
    """``AgentManagerBase.stop_agent`` MUST call
    ``self._finalize_agent_cleanup(agent_id)`` instead of inlining
    the cleanup. Single source of truth keeps the self-term path
    (``_run_agent_loop``'s finally) and the external-call path
    (``stop_agent``) from drifting."""

    base_py = (
        Path(__file__).resolve().parents[1] / "base.py"
    ).read_text(encoding="utf-8")
    assert (
        "await self._finalize_agent_cleanup(agent_id)" in base_py
    ), "stop_agent must delegate cleanup to _finalize_agent_cleanup."


def test_inline_resource_release_is_removed_from_stop_agent() -> None:
    """The inlined resource-release block previously in
    ``stop_agent`` (``self._used_cpu_cores -= agent.resource_...``,
    inline ``unregister_agent``, inline ``del self._agents[id]``)
    must be GONE from stop_agent — extracted into
    ``_finalize_agent_cleanup``. If it reappears, the two call
    sites drift and the next regression of this shape lands silently."""

    from polymathera.colony.agents.base import AgentManagerBase
    src = inspect.getsource(AgentManagerBase.stop_agent)
    assert "self._used_cpu_cores -=" not in src, (
        "stop_agent must not inline resource release; delegate to "
        "_finalize_agent_cleanup."
    )
    assert "del self._agents[agent_id]" not in src, (
        "stop_agent must not inline _agents deletion; delegate to "
        "_finalize_agent_cleanup."
    )
    assert "unregister_agent(agent_id)" not in src, (
        "stop_agent must not inline unregister_agent; delegate to "
        "_finalize_agent_cleanup."
    )


# ---------------------------------------------------------------------------
# Runtime: the cleanup method does what its docstring promises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_releases_resources_and_unregisters() -> None:
    """Drive ``_finalize_agent_cleanup`` against a minimal fake
    manager state and verify: counters decremented, dicts cleaned,
    unregister called with the right agent_id."""

    from polymathera.colony.agents.base import AgentManagerBase
    from polymathera.colony.agents.models import AgentResourceRequirements

    manager = AgentManagerBase.__new__(AgentManagerBase)
    manager._agents = {}
    manager._agent_tasks = {}
    manager._used_cpu_cores = 1.5
    manager._used_memory_mb = 2048
    manager._used_gpu_cores = 1.0
    manager._used_gpu_memory_mb = 4096
    manager._agent_system_handle = MagicMock()
    manager._agent_system_handle.unregister_agent = AsyncMock()

    fake_agent = MagicMock()
    fake_agent.resource_requirements = AgentResourceRequirements(
        cpu_cores=0.5, memory_mb=512, gpu_cores=0.25, gpu_memory_mb=1024,
    )
    manager._agents["agent-xyz"] = fake_agent
    manager._agent_tasks["agent-xyz"] = MagicMock()

    await manager._finalize_agent_cleanup("agent-xyz")

    assert "agent-xyz" not in manager._agents
    assert "agent-xyz" not in manager._agent_tasks
    assert manager._used_cpu_cores == pytest.approx(1.0)
    assert manager._used_memory_mb == 1536
    assert manager._used_gpu_cores == pytest.approx(0.75)
    assert manager._used_gpu_memory_mb == 3072
    manager._agent_system_handle.unregister_agent.assert_awaited_once_with(
        "agent-xyz",
    )


@pytest.mark.asyncio
async def test_finalize_is_idempotent_for_double_call() -> None:
    """``_finalize_agent_cleanup`` must be safe to call twice. The
    real-world race: external ``stop_agent`` and the loop's finally
    both call it. With single-threaded asyncio the calls are
    serialised; the second one finds ``_agents.pop`` returns None
    and returns early without double-decrementing counters or
    double-calling ``unregister_agent``."""

    from polymathera.colony.agents.base import AgentManagerBase
    from polymathera.colony.agents.models import AgentResourceRequirements

    manager = AgentManagerBase.__new__(AgentManagerBase)
    manager._agents = {}
    manager._agent_tasks = {}
    manager._used_cpu_cores = 0.5
    manager._used_memory_mb = 512
    manager._used_gpu_cores = 0.0
    manager._used_gpu_memory_mb = 0
    manager._agent_system_handle = MagicMock()
    manager._agent_system_handle.unregister_agent = AsyncMock()

    fake_agent = MagicMock()
    fake_agent.resource_requirements = AgentResourceRequirements(
        cpu_cores=0.5, memory_mb=512, gpu_cores=0.0, gpu_memory_mb=0,
    )
    manager._agents["agent-xyz"] = fake_agent

    await manager._finalize_agent_cleanup("agent-xyz")
    await manager._finalize_agent_cleanup("agent-xyz")  # 2nd call

    assert manager._used_cpu_cores == pytest.approx(0.0)
    assert manager._used_memory_mb == 0
    # unregister_agent must NOT be called twice — the second call
    # would log a spurious "Failed to unregister" if the system has
    # already removed the entry on the first call.
    assert (
        manager._agent_system_handle.unregister_agent.await_count == 1
    )


@pytest.mark.asyncio
async def test_finalize_no_system_handle_still_releases_resources() -> None:
    """A manager without an ``_agent_system_handle`` (early-init or
    test) must still clean up its local counters + dicts. The
    registry call is guarded."""

    from polymathera.colony.agents.base import AgentManagerBase
    from polymathera.colony.agents.models import AgentResourceRequirements

    manager = AgentManagerBase.__new__(AgentManagerBase)
    manager._agents = {}
    manager._agent_tasks = {}
    manager._used_cpu_cores = 0.5
    manager._used_memory_mb = 512
    manager._used_gpu_cores = 0.0
    manager._used_gpu_memory_mb = 0
    manager._agent_system_handle = None

    fake_agent = MagicMock()
    fake_agent.resource_requirements = AgentResourceRequirements(
        cpu_cores=0.5, memory_mb=512, gpu_cores=0.0, gpu_memory_mb=0,
    )
    manager._agents["agent-xyz"] = fake_agent

    await manager._finalize_agent_cleanup("agent-xyz")  # must not raise

    assert "agent-xyz" not in manager._agents
    assert manager._used_cpu_cores == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_finalize_unregister_failure_does_not_block_cleanup() -> None:
    """A registry hiccup on ``unregister_agent`` must NOT block the
    local cleanup that already ran. Resource counters and dicts are
    decremented BEFORE the registry call so a failure leaves a
    consistent local state."""

    from polymathera.colony.agents.base import AgentManagerBase
    from polymathera.colony.agents.models import AgentResourceRequirements

    manager = AgentManagerBase.__new__(AgentManagerBase)
    manager._agents = {}
    manager._agent_tasks = {}
    manager._used_cpu_cores = 0.5
    manager._used_memory_mb = 512
    manager._used_gpu_cores = 0.0
    manager._used_gpu_memory_mb = 0
    manager._agent_system_handle = MagicMock()
    manager._agent_system_handle.unregister_agent = AsyncMock(
        side_effect=RuntimeError("registry down"),
    )

    fake_agent = MagicMock()
    fake_agent.resource_requirements = AgentResourceRequirements(
        cpu_cores=0.5, memory_mb=512, gpu_cores=0.0, gpu_memory_mb=0,
    )
    manager._agents["agent-xyz"] = fake_agent

    await manager._finalize_agent_cleanup("agent-xyz")  # must not raise

    assert "agent-xyz" not in manager._agents
    assert manager._used_cpu_cores == pytest.approx(0.0)
