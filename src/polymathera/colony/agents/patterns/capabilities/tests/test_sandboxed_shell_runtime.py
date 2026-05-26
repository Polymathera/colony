"""Unit tests for ``SandboxedShellRuntime``.

Multi-container behaviour (full action surface, streaming,
ownership, slot accounting) is covered by
:mod:`test_sandboxed_shell_capability`. These tests target the
runtime-specific contracts:

- ``exec_stream`` raises ``RuntimeError`` unconditionally (locked
  design — runtime is serialisable, streaming requires a blackboard
  handle that isn't).
- Construction kwargs are cloudpickle-serialisable end-to-end so
  callers can carry a ``runtime_template`` dict through a worker
  closure + reconstruct the runtime fresh.
- ``exec`` returns an ``ExecResultWithAudit`` whose ``audit_records``
  pull tenant/session/agent identity from the labels snapshot
  (NOT from per-thread contextvars, which don't survive a worker
  boundary).
- The wall-time killer fires + the runtime reports
  ``killed_by_deadline=True``.
- ``stop`` is idempotent + records ``stop_already_gone`` when the
  backend reports the container vanished.
"""

from __future__ import annotations

import asyncio
import pickle
from typing import Any

import pytest

from polymathera.colony.agents.patterns.capabilities._sandbox.backend import (
    ContainerBackend,
    ContainerHandle,
    ContainerSpec,
    ExecResult,
    NoSuchContainer,
)
from polymathera.colony.agents.patterns.capabilities._sandbox.runtime import (
    ExecResultWithAudit,
    SandboxedShellRuntime,
)
from polymathera.colony.agents.patterns.capabilities._sandbox.registry import (
    ScriptSpec,
)


# ---------------------------------------------------------------------------
# Fake backend — module-level so it's picklable from a different module
# import path. Inline locals would fail cloudpickle.
# ---------------------------------------------------------------------------


class _FakeBackend(ContainerBackend):
    """In-memory backend for runtime unit tests.

    Tracks launched containers and the commands they ran. Configure
    ``raise_no_such_on_stop`` to simulate the "container already gone
    on the daemon" race.
    """

    def __init__(self) -> None:
        self.launched: list[ContainerSpec] = []
        self.stopped: list[ContainerHandle] = []
        self.execs: list[tuple[ContainerHandle, list[str]]] = []
        self.copy_ins: list[tuple[str, str]] = []
        self.copy_outs: list[tuple[str, str]] = []
        self.raise_no_such_on_stop: bool = False
        self.exec_stdout: str = "ok\n"
        self.exec_stderr: str = ""
        self.exec_exit_code: int = 0

    async def launch(self, spec: ContainerSpec) -> ContainerHandle:
        self.launched.append(spec)
        return ContainerHandle(
            container_id=f"cid_{len(self.launched)}",
            name=spec.name,
            image=spec.image,
        )

    async def stop(self, handle: ContainerHandle, *, timeout_s: int = 10) -> None:
        if self.raise_no_such_on_stop:
            raise NoSuchContainer(handle.container_id)
        self.stopped.append(handle)

    async def restart(self, handle: ContainerHandle) -> None:
        return None

    async def is_running(self, handle: ContainerHandle) -> bool:
        return True

    async def inspect(self, handle: ContainerHandle) -> dict[str, Any]:
        return {"State": {"Status": "running"}}

    async def exec(
        self, handle: ContainerHandle, cmd: list[str], *,
        timeout_seconds: int, env: dict[str, str] | None = None,
        workdir: str | None = None, stdin: str | None = None,
    ) -> ExecResult:
        self.execs.append((handle, list(cmd)))
        return ExecResult(
            exit_code=self.exec_exit_code,
            stdout=self.exec_stdout,
            stderr=self.exec_stderr,
            wall_time_ms=5,
        )

    async def exec_stream(self, *args: Any, **kwargs: Any):  # noqa: D401
        raise NotImplementedError(
            "_FakeBackend.exec_stream is not used by runtime tests "
            "(streaming is capability-only).",
        )
        yield  # pragma: no cover — make this a generator

    async def copy_in(
        self, handle: ContainerHandle, *,
        src_host_path: str, dst_container_path: str,
    ) -> None:
        self.copy_ins.append((src_host_path, dst_container_path))

    async def copy_out(
        self, handle: ContainerHandle, *,
        src_container_path: str, dst_host_path: str,
    ) -> None:
        self.copy_outs.append((src_container_path, dst_host_path))

    async def list_by_label(
        self, labels: dict[str, str],
    ) -> list[dict[str, Any]]:
        return []


def _build_runtime(
    backend: ContainerBackend | None = None,
    *,
    workspace_path: str = "/tmp/_sandbox_runtime_test",
    max_wall_time_seconds: int = 0,
    capture_max_bytes: int = 1_000_000,
    image: str = "test:image",
    labels: dict[str, str] | None = None,
) -> SandboxedShellRuntime:
    return SandboxedShellRuntime(
        backend=backend or _FakeBackend(),
        image=image,
        workspace_path=workspace_path,
        workdir="/workspace",
        env={"A": "1"},
        cpu_limit=1.0,
        memory_limit_mb=256,
        network_mode="bridge",
        extra_volumes=(),
        labels=labels or {
            "colony.owner_agent_id": "agent-X",
            "colony.tenant_id": "tenant-T",
            "colony.session_id": "session-S",
            "colony.image_role": "test_role",
            "colony.capability_key": "sandboxed_shell",
        },
        max_wall_time_seconds=max_wall_time_seconds,
        capture_max_bytes=capture_max_bytes,
        container_name="colony_sbx_test",
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# exec_stream
# ---------------------------------------------------------------------------


def test_exec_stream_raises_runtime_error() -> None:
    """The runtime never streams (locked design 2026-05-23). The
    placeholder method exists only so accidental calls fail fast."""
    rt = _build_runtime()
    with pytest.raises(RuntimeError, match="does not support streaming"):
        rt.exec_stream()
    with pytest.raises(RuntimeError, match="does not support streaming"):
        rt.exec_stream("ignored", "args")


# ---------------------------------------------------------------------------
# Cloudpickle (runtime-serialisability contract)
# ---------------------------------------------------------------------------


def test_construction_kwargs_are_picklable(tmp_path) -> None:
    """Downstream worker-orchestration callers carry a
    ``runtime_template`` dict (NOT a runtime instance) and
    reconstruct the runtime fresh per worker. Verify the kwargs
    dict pickles cleanly + reconstructs identically.
    """
    template = {
        "backend": _FakeBackend(),
        "image": "test:image",
        "workspace_path": str(tmp_path),
        "workdir": "/workspace",
        "env": {"A": "1"},
        "cpu_limit": 1.0,
        "memory_limit_mb": 512,
        "network_mode": "bridge",
        "extra_volumes": (),
        "labels": {
            "colony.owner_agent_id": "agent-X",
            "colony.tenant_id": "tenant-T",
            "colony.session_id": "session-S",
            "colony.image_role": "test_role",
        },
        "max_wall_time_seconds": 60,
        "capture_max_bytes": 1_000_000,
        "container_name": "colony_sbx_pickle",
    }
    blob = pickle.dumps(template)
    restored = pickle.loads(blob)
    # Construct a runtime from the restored kwargs — must not raise.
    rt = SandboxedShellRuntime(**restored)
    assert rt.workspace_path == str(tmp_path)


def test_runtime_instance_is_picklable_before_launch(tmp_path) -> None:
    """A runtime that hasn't been launched is still picklable — no
    asyncio Task field has been created yet."""
    rt = _build_runtime(workspace_path=str(tmp_path))
    blob = pickle.dumps(rt)
    restored = pickle.loads(blob)
    assert isinstance(restored, SandboxedShellRuntime)
    assert restored.handle is None  # not yet launched


# ---------------------------------------------------------------------------
# exec + audit records
# ---------------------------------------------------------------------------


def test_exec_returns_audit_record_from_labels(tmp_path) -> None:
    """The audit record's tenant/session/agent identity comes from
    the labels snapshot (NOT from per-thread contextvars). This is
    the property that lets the runtime survive crossing a worker
    boundary where the parent's contextvars are unreachable."""
    backend = _FakeBackend()
    rt = _build_runtime(backend=backend, workspace_path=str(tmp_path))
    _run(rt.launch())
    result = _run(rt.exec(["echo", "hi"], timeout_seconds=10))
    assert isinstance(result, ExecResultWithAudit)
    assert len(result.audit_records) == 1
    audit = result.audit_records[0]
    assert audit["tenant_id"] == "tenant-T"
    assert audit["session_id"] == "session-S"
    assert audit["agent_id"] == "agent-X"
    assert audit["image_role"] == "test_role"
    assert audit["image"] == "test:image"
    assert audit["command"] == ["echo", "hi"]
    assert audit["exit_code"] == 0
    assert audit["wall_time_ms"] == 5
    assert audit["stdout_size"] == len(backend.exec_stdout)


def test_exec_raises_when_not_launched(tmp_path) -> None:
    rt = _build_runtime(workspace_path=str(tmp_path))
    with pytest.raises(RuntimeError, match="before launch"):
        _run(rt.exec(["echo"], timeout_seconds=1))


def test_exec_truncates_per_call_override(tmp_path) -> None:
    """``capture_max_bytes`` on exec() overrides the construction-
    time cap (matches the pre-refactor execute_command kwarg)."""
    backend = _FakeBackend()
    backend.exec_stdout = "a" * 5000
    rt = _build_runtime(
        backend=backend, workspace_path=str(tmp_path),
        capture_max_bytes=10_000,
    )
    _run(rt.launch())
    result = _run(rt.exec(
        ["echo"], timeout_seconds=1, capture_max_bytes=100,
    ))
    assert result.exec_result.truncated is True
    assert len(result.exec_result.stdout) == 100


# ---------------------------------------------------------------------------
# exec_script
# ---------------------------------------------------------------------------


def test_exec_script_renders_command_template(tmp_path) -> None:
    backend = _FakeBackend()
    rt = _build_runtime(backend=backend, workspace_path=str(tmp_path))
    _run(rt.launch())
    script = ScriptSpec(
        name="greet", cmd=("echo", "hello {name}"),
        params={"name": {"required": True}},
        timeout_seconds=10,
    )
    _run(rt.exec_script(script, params={"name": "world"}))
    assert backend.execs == [(rt.handle, ["echo", "hello world"])]


def test_exec_script_rejects_missing_required_param(tmp_path) -> None:
    rt = _build_runtime(workspace_path=str(tmp_path))
    _run(rt.launch())
    script = ScriptSpec(
        name="greet", cmd=("echo", "hi {name}"),
        params={"name": {"required": True}},
        timeout_seconds=10,
    )
    with pytest.raises(ValueError, match="missing required param"):
        _run(rt.exec_script(script, params={}))


# ---------------------------------------------------------------------------
# Wall-time killer
# ---------------------------------------------------------------------------


def test_wall_time_killer_fires_and_records_flag(tmp_path) -> None:
    """A 1-second wall-time cap on a runtime that's launched + idle
    for >1s triggers the killer; the runtime flips
    ``killed_by_deadline`` + subsequent exec calls fail fast."""

    async def _exercise() -> SandboxedShellRuntime:
        backend = _FakeBackend()
        rt = _build_runtime(
            backend=backend, workspace_path=str(tmp_path),
            max_wall_time_seconds=1,
        )
        await rt.launch()
        await asyncio.sleep(1.2)
        return rt

    rt = asyncio.new_event_loop().run_until_complete(_exercise())
    assert rt.killed_by_deadline is True


def test_stop_records_already_gone_when_container_vanished(tmp_path) -> None:
    """Pre-refactor capability surfaced "container was already gone
    on the daemon" — runtime preserves that via ``stop_already_gone``."""
    backend = _FakeBackend()
    rt = _build_runtime(backend=backend, workspace_path=str(tmp_path))
    _run(rt.launch())
    backend.raise_no_such_on_stop = True
    _run(rt.stop())
    assert rt.stop_already_gone is True


def test_stop_is_idempotent(tmp_path) -> None:
    backend = _FakeBackend()
    rt = _build_runtime(backend=backend, workspace_path=str(tmp_path))
    _run(rt.launch())
    _run(rt.stop())
    _run(rt.stop())   # second call must not re-stop
    assert len(backend.stopped) == 1


# ---------------------------------------------------------------------------
# copy_in / copy_out
# ---------------------------------------------------------------------------


def test_copy_in_dispatches_to_backend(tmp_path) -> None:
    backend = _FakeBackend()
    rt = _build_runtime(backend=backend, workspace_path=str(tmp_path))
    _run(rt.launch())
    _run(rt.copy_in(src_host_path="/host/x", dst_container_path="/c/x"))
    assert backend.copy_ins == [("/host/x", "/c/x")]


def test_copy_out_dispatches_to_backend(tmp_path) -> None:
    backend = _FakeBackend()
    rt = _build_runtime(backend=backend, workspace_path=str(tmp_path))
    _run(rt.launch())
    _run(rt.copy_out(src_container_path="/c/y", dst_host_path="/host/y"))
    assert backend.copy_outs == [("/c/y", "/host/y")]


# ---------------------------------------------------------------------------
# Workspace mkdir
# ---------------------------------------------------------------------------


def test_launch_creates_workspace_directory(tmp_path) -> None:
    backend = _FakeBackend()
    workspace = tmp_path / "nested" / "workspace"
    assert not workspace.exists()
    rt = _build_runtime(backend=backend, workspace_path=str(workspace))
    _run(rt.launch())
    assert workspace.is_dir()
