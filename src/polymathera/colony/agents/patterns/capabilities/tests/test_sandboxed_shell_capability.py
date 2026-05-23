"""Unit tests for ``SandboxedShellCapability``.

All Docker interactions are routed through a fake ``ContainerBackend``
so these tests run without a daemon. The workspace directory is
overridden per test with ``pytest``'s ``tmp_path`` so nothing touches
``/mnt/shared``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.capabilities._sandbox import (
    ContainerBackend,
    ContainerHandle,
    ContainerSpec,
    ExecResult,
    DockerImageRegistry,
    NoSuchContainer,
)
from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
    SandboxedShellCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    execution_context, Ring,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeBackend(ContainerBackend):
    """In-memory container backend."""

    def __init__(self):
        self.launched: list[ContainerSpec] = []
        self.execs: list[dict] = []
        self.copies: list[tuple[str, dict]] = []
        self.running_state: dict[str, bool] = {}
        self.inspect_state: dict[str, dict] = {}
        self.next_exec_result: ExecResult = ExecResult(
            exit_code=0, stdout="", stderr="", wall_time_ms=5,
        )
        self.raise_on: set[str] = set()
        self.launched_handle: ContainerHandle | None = None
        self._counter = 0

    async def launch(self, spec):
        if "launch" in self.raise_on:
            raise RuntimeError("simulated launch failure")
        self.launched.append(spec)
        self._counter += 1
        handle = ContainerHandle(
            container_id=f"cid_{self._counter}",
            name=spec.name,
            image=spec.image,
        )
        self.running_state[handle.container_id] = True
        self.inspect_state[handle.container_id] = {
            "State": {"Status": "running", "Running": True, "ExitCode": 0,
                       "StartedAt": "0"},
        }
        self.launched_handle = handle
        return handle

    async def stop(self, handle, *, timeout_s=10):
        if "stop" in self.raise_on:
            raise RuntimeError("simulated stop failure")
        if "no_such_on_stop" in self.raise_on:
            raise NoSuchContainer(handle.container_id)
        self.running_state.pop(handle.container_id, None)

    async def restart(self, handle):
        if "restart" in self.raise_on:
            raise RuntimeError("simulated restart failure")

    async def is_running(self, handle):
        return self.running_state.get(handle.container_id, False)

    async def inspect(self, handle):
        return self.inspect_state.get(handle.container_id, {})

    async def exec(self, handle, cmd, *, timeout_seconds,
                   env=None, workdir=None, stdin=None):
        self.execs.append({
            "cmd": list(cmd), "timeout_seconds": timeout_seconds,
            "env": dict(env or {}), "workdir": workdir, "stdin": stdin,
        })
        if "exec" in self.raise_on:
            raise RuntimeError("simulated exec failure")
        return self.next_exec_result

    async def exec_stream(self, handle, cmd, *, timeout_seconds,
                          env=None, workdir=None, stdin=None):
        self.execs.append({
            "cmd": list(cmd), "stream": True,
            "timeout_seconds": timeout_seconds,
        })
        # Yield a stdout then a stderr chunk.
        async def _gen():
            yield "stdout", "hello\n"
            yield "stderr", "warn\n"
        async for item in _gen():
            yield item

    async def copy_in(self, handle, *, src_host_path, dst_container_path):
        self.copies.append(("in", {
            "src": src_host_path, "dst": dst_container_path,
        }))
        if "copy_in" in self.raise_on:
            raise RuntimeError("simulated copy_in failure")

    async def copy_out(self, handle, *, src_container_path, dst_host_path):
        self.copies.append(("out", {
            "src": src_container_path, "dst": dst_host_path,
        }))
        if "copy_out" in self.raise_on:
            raise RuntimeError("simulated copy_out failure")

    async def list_by_label(self, labels):
        return []


class _FakeBlackboard:
    def __init__(self):
        self.writes: list[tuple[str, Any]] = []

    async def write(self, k, v):
        self.writes.append((k, v))


_REGISTRY_YAML = """
images:
  - role: default
    image: python:3.11-slim
    description: Python 3.11 base.
    scripts:
      - name: ok
        description: returns 0
        params:
          foo:
            type: string
            required: true
        cmd: ["echo", "{foo}"]
      - name: notmpl
        description: no params
        params: {}
        cmd: ["echo", "hi"]
  - role: other
    image: alpine:3.19
    description: Alpine.
"""


def _make_cap(
    backend: _FakeBackend | None = None,
    *,
    agent_id: str = "agent-A",
    bb: _FakeBlackboard | None = None,
    workspace: str | None = None,
    max_concurrent: int = 4,
    max_cpu: float = 4.0,
    max_mem: int = 8192,
) -> SandboxedShellCapability:
    agent = MagicMock()
    agent.agent_id = agent_id
    cap = SandboxedShellCapability(
        agent=agent,
        scope=BlackboardScope.SESSION,
        backend=backend or _FakeBackend(),
        registry=DockerImageRegistry.from_yaml_text(_REGISTRY_YAML),
        host_workspace_root=workspace or "/tmp/colony_test_ws",
        max_concurrent_containers=max_concurrent,
        max_total_cpu_cores=max_cpu,
        max_total_memory_mb=max_mem,
    )
    if bb is not None:
        cap._blackboard = bb
    return cap


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _with_context():
    return execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1", session_id="s1",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_parses_yaml_and_reports_summaries():
    reg = DockerImageRegistry.from_yaml_text(_REGISTRY_YAML)
    assert reg.roles() == ["default", "other"]
    assert reg.get("default").image == "python:3.11-slim"
    default_script = reg.get("default").script_by_name("ok")
    assert default_script is not None
    assert default_script.cmd == ("echo", "{foo}")
    summaries = reg.summaries()
    assert summaries[0]["role"] == "default"


def test_registry_skips_malformed_entries():
    bad = """
images:
  - {role: no_image}
  - {image: only_image_no_role}
  - role: ok
    image: x
"""
    reg = DockerImageRegistry.from_yaml_text(bad)
    assert reg.roles() == ["ok"]


def test_registry_empty_for_missing_file(tmp_path):
    reg = DockerImageRegistry.from_path(tmp_path / "does_not_exist.yaml")
    assert len(reg) == 0


def test_registry_find_script_respects_role_scope():
    reg = DockerImageRegistry.from_yaml_text(_REGISTRY_YAML)
    hit = reg.find_script("ok", image_role="default")
    assert hit is not None and hit[1].name == "ok"
    miss = reg.find_script("ok", image_role="other")
    assert miss is None


# ---------------------------------------------------------------------------
# launch / stop / list / status
# ---------------------------------------------------------------------------

def test_launch_container_rejects_unknown_role(tmp_path):
    with _with_context():
        cap = _make_cap(workspace=str(tmp_path))
        result = _run(cap.launch_container(image_role="unknown"))
    assert result["started"] is False
    assert "unknown image_role" in result["message"]


def test_launch_container_sets_labels_and_workspace(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        result = _run(cap.launch_container(
            image_role="default",
            cpu_limit=0.5, memory_limit_mb=256,
            max_wall_time_seconds=0,  # disable killer for test
        ))
    assert result["started"] is True
    assert result["image"] == "python:3.11-slim"
    assert result["workspace_path"].startswith(str(tmp_path))
    [spec] = backend.launched
    # Labels carry ownership information.
    assert spec.labels["colony.owner_agent_id"] == "agent-A"
    assert spec.labels["colony.tenant_id"] == "t1"
    assert spec.labels["colony.session_id"] == "s1"
    assert spec.labels["colony.image_role"] == "default"
    # Workspace bind-mounted rw at workdir.
    assert any(
        host.startswith(str(tmp_path)) and cont == "/workspace" and mode == "rw"
        for host, cont, mode in spec.volumes
    )
    # Cleanup.
    _run(cap.stop_container(result["container_id"]))


def test_launch_container_enforces_concurrent_cap(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(
            backend=backend, workspace=str(tmp_path),
            max_concurrent=1,
        )
        r1 = _run(cap.launch_container(
            image_role="default", cpu_limit=0.1,
            memory_limit_mb=64, max_wall_time_seconds=0,
        ))
        r2 = _run(cap.launch_container(
            image_role="default", cpu_limit=0.1,
            memory_limit_mb=64, max_wall_time_seconds=0,
        ))
        _run(cap.stop_container(r1["container_id"]))
    assert r1["started"] is True
    assert r2["started"] is False
    assert "max_concurrent_containers" in r2["message"]


def test_launch_container_enforces_cpu_cap(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(
            backend=backend, workspace=str(tmp_path),
            max_cpu=1.0, max_mem=4096,
        )
        r1 = _run(cap.launch_container(
            image_role="default", cpu_limit=0.6,
            memory_limit_mb=64, max_wall_time_seconds=0,
        ))
        r2 = _run(cap.launch_container(
            image_role="default", cpu_limit=0.6,
            memory_limit_mb=64, max_wall_time_seconds=0,
        ))
        _run(cap.stop_container(r1["container_id"]))
    assert r1["started"] is True
    assert r2["started"] is False
    assert "max_total_cpu_cores" in r2["message"]


def test_launch_container_rejects_invalid_network_mode(tmp_path):
    with _with_context():
        cap = _make_cap(workspace=str(tmp_path))
        result = _run(cap.launch_container(
            image_role="default", network_mode="wat",  # type: ignore
            max_wall_time_seconds=0,
        ))
    assert result["started"] is False
    assert "network_mode" in result["message"]


def test_stop_container_rejects_non_owner(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap_a = _make_cap(
            backend=backend, agent_id="agent-A",
            workspace=str(tmp_path),
        )
        r = _run(cap_a.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        cap_b = _make_cap(
            backend=backend, agent_id="agent-B",
            workspace=str(tmp_path),
        )
        cap_b._containers = cap_a._containers  # share dict for test
        result = _run(cap_b.stop_container(r["container_id"]))
        _run(cap_a.stop_container(r["container_id"]))
    assert result["stopped"] is False
    assert "only the owner" in result["message"]


def test_stop_container_handles_already_gone(tmp_path):
    backend = _FakeBackend()
    backend.raise_on.add("no_such_on_stop")
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap.stop_container(r["container_id"]))
    assert result["stopped"] is True
    assert "already gone" in result["message"]


def test_list_containers_filters_by_ownership(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap_a = _make_cap(
            backend=backend, agent_id="agent-A",
            workspace=str(tmp_path),
        )
        cap_b = _make_cap(
            backend=backend, agent_id="agent-B",
            workspace=str(tmp_path),
        )
        cap_b._containers = cap_a._containers
        r_a = _run(cap_a.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        r_b = _run(cap_b.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        listed = _run(cap_a.list_containers(owned_by_me=True))
        listed_all = _run(cap_a.list_containers(owned_by_me=False))
        _run(cap_a.stop_container(r_a["container_id"]))
        _run(cap_b.stop_container(r_b["container_id"]))
    assert listed["count"] == 1
    assert listed_all["count"] == 2


def test_get_container_status_reports_running_state(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        status = _run(cap.get_container_status(r["container_id"]))
        _run(cap.stop_container(r["container_id"]))
    assert status["running"] is True
    assert status["known"] is True


# ---------------------------------------------------------------------------
# execute_command / audit / streaming
# ---------------------------------------------------------------------------

def test_execute_command_wraps_string_in_bash(tmp_path):
    backend = _FakeBackend()
    backend.next_exec_result = ExecResult(
        exit_code=0, stdout="hello\n", stderr="", wall_time_ms=7,
    )
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap.execute_command(
            r["container_id"], "echo hello",
        ))
        _run(cap.stop_container(r["container_id"]))
    assert result["exit_code"] == 0
    assert result["stdout"] == "hello\n"
    # String form wrapped as bash -lc.
    assert backend.execs[0]["cmd"] == ["bash", "-lc", "echo hello"]


def test_execute_command_rejects_caller_without_exec_permission(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap_a = _make_cap(
            backend=backend, agent_id="agent-A",
            workspace=str(tmp_path),
        )
        cap_b = _make_cap(
            backend=backend, agent_id="agent-B",
            workspace=str(tmp_path),
        )
        cap_b._containers = cap_a._containers
        r = _run(cap_a.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap_b.execute_command(
            r["container_id"], ["true"],
        ))
        _run(cap_a.stop_container(r["container_id"]))
    assert result["exit_code"] == -1
    assert "not allowed" in result["message"]


def test_execute_command_truncates_large_output(tmp_path):
    backend = _FakeBackend()
    backend.next_exec_result = ExecResult(
        exit_code=0, stdout="x" * 5000, stderr="", wall_time_ms=1,
    )
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap.execute_command(
            r["container_id"], ["true"], capture_max_bytes=100,
        ))
        _run(cap.stop_container(r["container_id"]))
    assert result["truncated"] is True
    assert len(result["stdout"]) == 100


def test_execute_command_writes_audit_record(tmp_path):
    backend = _FakeBackend()
    bb = _FakeBlackboard()
    with _with_context():
        cap = _make_cap(
            backend=backend, workspace=str(tmp_path), bb=bb,
        )
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        _run(cap.execute_command(r["container_id"], ["true"]))
        _run(cap.stop_container(r["container_id"]))
    audit_records = [w for w in bb.writes if w[0].startswith("audit:shell:")]
    assert len(audit_records) == 1
    _, record = audit_records[0]
    assert record["agent_id"] == "agent-A"
    assert record["tenant_id"] == "t1"
    assert record["image_role"] == "default"


def test_execute_command_streaming_emits_chunks_and_complete(tmp_path):
    backend = _FakeBackend()
    bb = _FakeBlackboard()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path), bb=bb)
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap.execute_command(
            r["container_id"], ["true"], stream_to_blackboard=True,
        ))
        _run(cap.stop_container(r["container_id"]))
    chunk_keys = [k for k, _ in bb.writes if k.startswith("shell:stream:")]
    complete_keys = [
        k for k, _ in bb.writes
        if k.startswith("shell:exec:") and k.endswith(":complete")
    ]
    assert len(chunk_keys) == 2
    assert len(complete_keys) == 1
    assert result["stream_key"] is not None
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == "warn\n"


# ---------------------------------------------------------------------------
# execute_script
# ---------------------------------------------------------------------------

def test_execute_script_validates_params(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap.execute_script(
            r["container_id"], "ok",
            image_role="default", args={},  # missing 'foo'
        ))
        _run(cap.stop_container(r["container_id"]))
    assert result["exit_code"] == -1
    assert "missing required param" in result["message"]


def test_execute_script_substitutes_template_args(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        _run(cap.execute_script(
            r["container_id"], "ok",
            image_role="default", args={"foo": "hello world"},
        ))
        _run(cap.stop_container(r["container_id"]))
    # The substituted cmd is recorded as the exec command.
    exec_calls = [e for e in backend.execs if e["cmd"][0] == "echo"]
    assert exec_calls[0]["cmd"] == ["echo", "hello world"]


def test_execute_script_reports_unknown_script(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        result = _run(cap.execute_script(
            r["container_id"], "missing", image_role="default",
        ))
        _run(cap.stop_container(r["container_id"]))
    assert "not in the registry" in result["message"]


# ---------------------------------------------------------------------------
# sharing
# ---------------------------------------------------------------------------

def test_attach_requires_shared_flag(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap_a = _make_cap(
            backend=backend, agent_id="agent-A",
            workspace=str(tmp_path),
        )
        cap_b = _make_cap(
            backend=backend, agent_id="agent-B",
            workspace=str(tmp_path),
        )
        cap_b._containers = cap_a._containers
        r = _run(cap_a.launch_container(
            image_role="default", shared=False,
            max_wall_time_seconds=0,
        ))
        rejected = _run(cap_b.attach_container(r["container_id"]))
        _run(cap_a.stop_container(r["container_id"]))
    assert rejected["attached"] is False
    assert "shared=True" in rejected["message"]


def test_shared_container_allows_attached_agent_to_exec(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap_a = _make_cap(
            backend=backend, agent_id="agent-A",
            workspace=str(tmp_path),
        )
        cap_b = _make_cap(
            backend=backend, agent_id="agent-B",
            workspace=str(tmp_path),
        )
        cap_b._containers = cap_a._containers
        r = _run(cap_a.launch_container(
            image_role="default", shared=True,
            max_wall_time_seconds=0,
        ))
        attached = _run(cap_b.attach_container(r["container_id"]))
        exec_r = _run(cap_b.execute_command(
            r["container_id"], ["true"],
        ))
        _run(cap_a.stop_container(r["container_id"]))
    assert attached["attached"] is True
    assert exec_r["exit_code"] == 0


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------

def test_copy_file_in_and_out_delegate_to_backend(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        in_result = _run(cap.copy_file_in(
            r["container_id"], "/host/a.txt", "/workspace/a.txt",
        ))
        out_result = _run(cap.copy_file_out(
            r["container_id"], "/workspace/a.txt", "/host/b.txt",
        ))
        _run(cap.stop_container(r["container_id"]))
    assert in_result["ok"] is True
    assert out_result["ok"] is True
    assert backend.copies == [
        ("in", {"src": "/host/a.txt", "dst": "/workspace/a.txt"}),
        ("out", {"src": "/workspace/a.txt", "dst": "/host/b.txt"}),
    ]


def test_write_file_streams_content_through_stdin(tmp_path):
    backend = _FakeBackend()
    with _with_context():
        cap = _make_cap(backend=backend, workspace=str(tmp_path))
        r = _run(cap.launch_container(
            image_role="default", max_wall_time_seconds=0,
        ))
        _run(cap.write_file(
            r["container_id"], "/workspace/x.txt", "hello!",
        ))
        _run(cap.stop_container(r["container_id"]))
    write_calls = [
        e for e in backend.execs
        if e["cmd"][:2] == ["bash", "-lc"] and "tee" in e["cmd"][2]
    ]
    assert write_calls[0]["stdin"] == "hello!"


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

def test_bind_round_trips_through_cloudpickle():
    # Ray's vendored cloudpickle — see comment in
    # test_github_capability for why standalone PyPI cloudpickle is
    # not the right import here.
    from ray import cloudpickle
    bp = SandboxedShellCapability.bind(
        scope=BlackboardScope.SESSION,
        max_concurrent_containers=2,
    )
    bp2 = cloudpickle.loads(cloudpickle.dumps(bp))
    assert bp2.cls is SandboxedShellCapability
    assert bp2.kwargs["max_concurrent_containers"] == 2


def test_action_executors_are_registered():
    import inspect
    keys = {
        m._action_key for _, m in inspect.getmembers(
            SandboxedShellCapability, predicate=inspect.isfunction,
        ) if getattr(m, "_action_key", None)
    }
    assert keys == {
        "launch_container", "stop_container", "restart_container",
        "list_containers", "get_container_status",
        "attach_container", "detach_container",
        "execute_command", "execute_script", "run_script",
        "list_scripts", "list_images",
        "list_script_templates", "get_script_template",
        "copy_file_in", "copy_file_out", "read_file", "write_file",
    }
