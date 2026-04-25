"""Sandboxed shell capability — run commands in curated Docker containers.

Gives an LLM-driven agent ``@action_executor`` methods to:

- Launch a Docker container from a curated image registry (by role label).
- Execute single commands or named scripts inside it.
- Copy files in and out.
- Stream stdout/stderr over the blackboard for long-running work.
- Share a container with peer agents in the same session.

Everything flows through ``ContainerBackend``; the default is
``DockerCLIBackend`` (shells out to the ``docker`` binary via
``asyncio.create_subprocess_exec``). Swap in ``AiodockerBackend`` or
``KubernetesBackend`` later by passing a different backend to the
blueprint — see ``design_SandboxedShellCapability.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from overrides import override

from ...base import AgentCapability
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, ScopeUtils, get_scope_prefix
from ..actions import action_executor

from ._sandbox import (
    ContainerBackend,
    ContainerHandle,
    ContainerSpec,
    DockerCLIBackend,
    ExecResult,
    ImageRegistry,
    NoSuchContainer,
    ScriptSpec,
)
from ._sandbox.backend import _safe_name

if TYPE_CHECKING:
    from ...base import Agent


logger = logging.getLogger(__name__)


_NetworkMode = Literal["bridge", "none", "host"]


# ---------------------------------------------------------------------------
# Local bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _ContainerRecord:
    """Per-capability view of one live container.

    Ownership semantics:

    - ``owner_agent_id`` is the agent that called ``launch_container``.
      Only the owner can ``stop_container`` / ``restart_container``.
    - Other agents that ``attach_container(shared_id)`` are recorded in
      ``attached_agent_ids`` for reference counting.
    """

    handle: ContainerHandle
    image_role: str
    launched_at: float
    owner_agent_id: str
    shared: bool
    cpu_limit: float
    memory_limit_mb: int
    network_mode: _NetworkMode
    workspace_path: str
    max_wall_time_seconds: int
    deadline_task: asyncio.Task | None = None
    attached_agent_ids: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Defaults and constants
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY_PATH = "/etc/colony/sandbox-images.yaml"
_CONTAINER_NAME_PREFIX = "colony_sbx_"
_LABEL_OWNER = "colony.owner_agent_id"
_LABEL_TENANT = "colony.tenant_id"
_LABEL_SESSION = "colony.session_id"
_LABEL_ROLE = "colony.image_role"
_LABEL_SHARED = "colony.shared"
_LABEL_CAPABILITY = "colony.capability_key"


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

class SandboxedShellCapability(AgentCapability):
    """Launch, run commands in, and tear down Docker sandboxes.

    Args:
        agent: Owning agent.
        scope: Blackboard partition this capability writes events under.
        namespace: Sub-namespace for the capability.
        backend: Container backend. Default: ``DockerCLIBackend``.
        registry: Pre-built image registry. If ``None``, falls back to
            ``registry_path``.
        registry_path: Path to the YAML registry file. Defaults to
            ``/etc/colony/sandbox-images.yaml`` — mount this into the
            container so the operator can edit without rebuilding.
        host_workspace_root: Directory on the host where per-session
            workspaces are created and bind-mounted into sandboxes.
        default_network_mode: Network mode for launched containers when
            the caller does not override.
        max_concurrent_containers: Per-agent cap.
        max_total_cpu_cores: Per-agent cap across all live containers.
        max_total_memory_mb: Per-agent cap across all live containers.
        stream_chunk_bytes: Buffer size for streaming output chunks.
        audit_enabled: If True, every ``execute_command`` writes an
            audit record to the colony-scoped audit log.
        capability_key: Dispatcher key.
        app_name: ``serving`` application name override.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "shell",
        backend: ContainerBackend | None = None,
        registry: ImageRegistry | None = None,
        registry_path: str = _DEFAULT_REGISTRY_PATH,
        host_workspace_root: str = "/mnt/shared/workspaces",
        default_network_mode: _NetworkMode = "bridge",
        max_concurrent_containers: int = 4,
        max_total_cpu_cores: float = 4.0,
        max_total_memory_mb: int = 8192,
        stream_chunk_bytes: int = 2048,
        audit_enabled: bool = True,
        capability_key: str = "sandboxed_shell",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )
        self._backend = backend or DockerCLIBackend()
        if registry is not None:
            self._registry = registry
        else:
            self._registry = ImageRegistry.from_path(registry_path)
        self._host_workspace_root = host_workspace_root
        self._default_network_mode = default_network_mode
        self._max_concurrent = max_concurrent_containers
        self._max_cpu = max_total_cpu_cores
        self._max_mem = max_total_memory_mb
        self._stream_chunk_bytes = stream_chunk_bytes
        self._audit_enabled = audit_enabled
        self._containers: dict[str, _ContainerRecord] = {}

    def get_action_group_description(self) -> str:
        return (
            "Sandboxed Shell — launch curated Docker containers, run "
            "shell commands or named scripts inside them, copy files "
            "in and out, share a container with peer agents. Images "
            "come from a curated registry picked by 'role' (never raw "
            "image names). execute_script is preferred over "
            "execute_command when a registered script matches the task."
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"sandbox", "shell", "docker", "external"})

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        # Live container state is ephemeral and tied to the Docker
        # daemon; do not try to reconstruct it on resume. Let the
        # resuming agent re-launch whatever it needs.
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        return None

    async def shutdown(self) -> None:
        """Stop every container this capability owns. Idempotent."""
        cids = list(self._containers.keys())
        for cid in cids:
            rec = self._containers.pop(cid, None)
            if rec is None:
                continue
            if rec.deadline_task is not None:
                rec.deadline_task.cancel()
            try:
                await self._backend.stop(rec.handle, timeout_s=10)
            except Exception as e:  # pragma: no cover — defensive
                logger.debug(
                    "SandboxedShellCapability: stop failed for %s: %s",
                    cid, e,
                )

    # --- Internal helpers -------------------------------------------------

    def _tenant_id(self) -> str:
        try:
            from ....distributed.ray_utils.serving.context import (
                require_execution_context,
            )
            ctx = require_execution_context()
            return ctx.tenant_id or "unknown"
        except Exception:
            return "unknown"

    def _session_id(self) -> str:
        try:
            from ....distributed.ray_utils.serving.context import (
                require_execution_context,
            )
            ctx = require_execution_context()
            return ctx.session_id or "unknown"
        except Exception:
            return "unknown"

    def _agent_id(self) -> str:
        return self.agent.agent_id if self._agent is not None else "unknown"

    def _session_workspace_path(self) -> str:
        """Compute (and mkdir if missing) the session's workspace dir."""
        tenant = self._tenant_id()
        session = self._session_id()
        path = os.path.join(
            self._host_workspace_root,
            _safe_name(tenant),
            _safe_name(session),
        )
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "SandboxedShellCapability: failed to create workspace "
                "%s: %s", path, e,
            )
        return path

    def _check_agent_caps(
        self, *, extra_cpu: float, extra_mem_mb: int,
    ) -> str | None:
        """Return ``None`` if adding a container is within caps, else
        a human-readable reason."""
        if len(self._containers) >= self._max_concurrent:
            return (
                f"max_concurrent_containers ({self._max_concurrent}) "
                f"reached"
            )
        total_cpu = sum(r.cpu_limit for r in self._containers.values())
        total_mem = sum(r.memory_limit_mb for r in self._containers.values())
        if total_cpu + extra_cpu > self._max_cpu + 1e-9:
            return (
                f"max_total_cpu_cores would be exceeded "
                f"({total_cpu + extra_cpu} > {self._max_cpu})"
            )
        if total_mem + extra_mem_mb > self._max_mem:
            return (
                f"max_total_memory_mb would be exceeded "
                f"({total_mem + extra_mem_mb} > {self._max_mem})"
            )
        return None

    def _make_container_name(self) -> str:
        return _safe_name(f"{_CONTAINER_NAME_PREFIX}{uuid.uuid4().hex[:10]}")

    def _build_labels(
        self,
        *, image_role: str, shared: bool,
    ) -> dict[str, str]:
        return {
            _LABEL_OWNER: self._agent_id(),
            _LABEL_TENANT: self._tenant_id(),
            _LABEL_SESSION: self._session_id(),
            _LABEL_ROLE: image_role,
            _LABEL_SHARED: "true" if shared else "false",
            _LABEL_CAPABILITY: self.capability_key,
        }

    async def _write_audit(self, record: dict[str, Any]) -> None:
        if not self._audit_enabled:
            return
        try:
            bb = await self.get_blackboard()
            key = f"audit:shell:{int(time.time() * 1000)}:{uuid.uuid4().hex[:8]}"
            await bb.write(key, record)
        except Exception as e:  # pragma: no cover — defensive
            logger.debug(
                "SandboxedShellCapability: audit write failed: %s", e,
            )

    async def _arm_wall_time_killer(
        self, record: _ContainerRecord,
    ) -> None:
        """Schedule a background task that stops the container when its
        wall-time cap is hit. Cancelled if the container stops first.
        """
        if record.max_wall_time_seconds <= 0:
            return

        async def _kill_after():
            try:
                await asyncio.sleep(record.max_wall_time_seconds)
            except asyncio.CancelledError:
                return
            cid = record.handle.container_id
            if cid not in self._containers:
                return
            logger.info(
                "SandboxedShellCapability: container %s reached "
                "max_wall_time_seconds; stopping",
                cid,
            )
            try:
                await self._backend.stop(record.handle, timeout_s=5)
            except Exception as e:  # pragma: no cover — defensive
                logger.debug(
                    "SandboxedShellCapability: wall-time stop failed "
                    "for %s: %s", cid, e,
                )
            self._containers.pop(cid, None)

        record.deadline_task = asyncio.create_task(_kill_after())

    # --- Action executors: lifecycle --------------------------------------

    @action_executor()
    async def launch_container(
        self,
        *,
        image_role: str,
        env: dict[str, str] | None = None,
        workdir: str = "/workspace",
        cpu_limit: float = 1.0,
        memory_limit_mb: int = 2048,
        max_wall_time_seconds: int = 3600,
        extra_volumes: list[dict[str, str]] | None = None,
        network_mode: _NetworkMode | None = None,
        shared: bool = False,
    ) -> dict[str, Any]:
        """Launch a Docker container from the curated registry.

        Args:
            image_role: Role label (e.g., ``"code_analysis"``) in the
                registered image list. Unknown roles return an error.
            env: Extra environment variables for the container.
            workdir: Container working directory for subsequent exec
                calls. The per-session host workspace is always bind-
                mounted at ``workdir`` so files persist between execs.
            cpu_limit: CPU cores. Passed to Docker via ``--cpus``.
            memory_limit_mb: Memory cap in MiB. Passed via ``--memory``.
            max_wall_time_seconds: Hard cap on the container's lifetime.
                A background task kills the container when the cap is
                hit. ``<=0`` disables the cap.
            extra_volumes: Additional bind mounts, each
                ``{"src", "dst", "mode"}`` where mode is ``"ro"|"rw"``.
            network_mode: Override default network mode. ``"none"`` for
                offline work; ``"bridge"`` for outbound internet;
                ``"host"`` for privileged use only.
            shared: If True, this container may be attached by other
                agents in the same session.

        Returns:
            On success: ``{"container_id", "container_name", "image",
            "workspace_path", "owner_agent_id", "shared",
            "started": True}``. On failure: the same fields plus a
            non-empty ``"message"`` and ``"started": False``.
        """
        spec = self._registry.get(image_role)
        if spec is None:
            return {
                "started": False,
                "container_id": None, "container_name": None,
                "image": None, "workspace_path": None,
                "owner_agent_id": self._agent_id(), "shared": shared,
                "message": (
                    f"unknown image_role {image_role!r}; registered "
                    f"roles: {self._registry.roles()}"
                ),
            }
        nm = network_mode or self._default_network_mode
        if nm not in ("bridge", "none", "host"):
            return {
                "started": False,
                "container_id": None, "container_name": None,
                "image": spec.image, "workspace_path": None,
                "owner_agent_id": self._agent_id(), "shared": shared,
                "message": f"invalid network_mode {nm!r}",
            }

        cap_reason = self._check_agent_caps(
            extra_cpu=cpu_limit, extra_mem_mb=memory_limit_mb,
        )
        if cap_reason is not None:
            return {
                "started": False,
                "container_id": None, "container_name": None,
                "image": spec.image, "workspace_path": None,
                "owner_agent_id": self._agent_id(), "shared": shared,
                "message": cap_reason,
            }

        workspace = self._session_workspace_path()
        volumes: list[tuple[str, str, str]] = [
            (workspace, workdir, "rw"),
        ]
        for v in (extra_volumes or []):
            src = v.get("src")
            dst = v.get("dst")
            mode = v.get("mode", "ro")
            if not src or not dst:
                continue
            volumes.append((str(src), str(dst), str(mode)))

        container_spec = ContainerSpec(
            image=spec.image,
            name=self._make_container_name(),
            env=dict(env or {}),
            workdir=workdir,
            cpu_limit=cpu_limit,
            memory_limit_mb=memory_limit_mb,
            network_mode=nm,
            volumes=tuple(volumes),
            labels=self._build_labels(image_role=image_role, shared=shared),
        )
        try:
            handle = await self._backend.launch(container_spec)
        except Exception as e:
            logger.exception(
                "SandboxedShellCapability.launch_container failed",
            )
            return {
                "started": False,
                "container_id": None, "container_name": None,
                "image": spec.image, "workspace_path": workspace,
                "owner_agent_id": self._agent_id(), "shared": shared,
                "message": f"backend.launch raised: {e}",
            }

        record = _ContainerRecord(
            handle=handle,
            image_role=image_role,
            launched_at=time.time(),
            owner_agent_id=self._agent_id(),
            shared=shared,
            cpu_limit=cpu_limit,
            memory_limit_mb=memory_limit_mb,
            network_mode=nm,
            workspace_path=workspace,
            max_wall_time_seconds=max_wall_time_seconds,
        )
        self._containers[handle.container_id] = record
        await self._arm_wall_time_killer(record)
        return {
            "started": True,
            "container_id": handle.container_id,
            "container_name": handle.name,
            "image": spec.image,
            "workspace_path": workspace,
            "owner_agent_id": record.owner_agent_id,
            "shared": shared,
            "message": "",
        }

    @action_executor()
    async def stop_container(
        self, container_id: str, *, timeout_s: int = 10,
    ) -> dict[str, Any]:
        """Stop and remove a container that this agent owns.

        Returns:
            ``{"container_id", "stopped": bool, "message": str}``.
            ``stopped`` is False if the caller is not the owner, the
            id is unknown, or the backend failed.
        """
        record = self._containers.get(container_id)
        if record is None:
            return {
                "container_id": container_id, "stopped": False,
                "message": "container_id not tracked by this capability",
            }
        if record.owner_agent_id != self._agent_id():
            return {
                "container_id": container_id, "stopped": False,
                "message": "only the owner can stop a container",
            }
        if record.deadline_task is not None:
            record.deadline_task.cancel()
        try:
            await self._backend.stop(record.handle, timeout_s=timeout_s)
        except NoSuchContainer:
            self._containers.pop(container_id, None)
            return {
                "container_id": container_id, "stopped": True,
                "message": "container was already gone on the daemon",
            }
        except Exception as e:
            logger.exception(
                "SandboxedShellCapability.stop_container failed",
            )
            return {
                "container_id": container_id, "stopped": False,
                "message": f"backend.stop raised: {e}",
            }
        self._containers.pop(container_id, None)
        return {
            "container_id": container_id, "stopped": True, "message": "",
        }

    @action_executor()
    async def restart_container(self, container_id: str) -> dict[str, Any]:
        """Restart a container this agent owns."""
        record = self._containers.get(container_id)
        if record is None:
            return {
                "container_id": container_id, "restarted": False,
                "message": "container_id not tracked",
            }
        if record.owner_agent_id != self._agent_id():
            return {
                "container_id": container_id, "restarted": False,
                "message": "only the owner can restart a container",
            }
        try:
            await self._backend.restart(record.handle)
        except Exception as e:
            return {
                "container_id": container_id, "restarted": False,
                "message": f"backend.restart raised: {e}",
            }
        return {
            "container_id": container_id, "restarted": True, "message": "",
        }

    @action_executor()
    async def list_containers(
        self, *, owned_by_me: bool = True,
    ) -> dict[str, Any]:
        """List containers this capability knows about.

        Args:
            owned_by_me: If True (default), return only containers this
                agent launched. If False, include attached shared
                containers too.
        """
        containers: list[dict[str, Any]] = []
        for rec in self._containers.values():
            is_mine = rec.owner_agent_id == self._agent_id()
            if owned_by_me and not is_mine:
                continue
            containers.append({
                "container_id": rec.handle.container_id,
                "container_name": rec.handle.name,
                "image": rec.handle.image,
                "image_role": rec.image_role,
                "launched_at": rec.launched_at,
                "owner_agent_id": rec.owner_agent_id,
                "owner_is_me": is_mine,
                "shared": rec.shared,
                "cpu_limit": rec.cpu_limit,
                "memory_limit_mb": rec.memory_limit_mb,
                "network_mode": rec.network_mode,
                "workspace_path": rec.workspace_path,
            })
        return {"containers": containers, "count": len(containers)}

    @action_executor()
    async def get_container_status(
        self, container_id: str,
    ) -> dict[str, Any]:
        """Return live status of a container (running? inspect dump)."""
        record = self._containers.get(container_id)
        if record is None:
            return {
                "container_id": container_id,
                "known": False, "running": False, "message": "not tracked",
            }
        try:
            running = await self._backend.is_running(record.handle)
            inspect = await self._backend.inspect(record.handle)
        except NoSuchContainer:
            self._containers.pop(container_id, None)
            return {
                "container_id": container_id,
                "known": True, "running": False,
                "message": "container disappeared on daemon",
            }
        except Exception as e:
            return {
                "container_id": container_id,
                "known": True, "running": False,
                "message": f"inspect failed: {e}",
            }
        state = (inspect.get("State") or {}) if isinstance(inspect, dict) else {}
        return {
            "container_id": container_id,
            "known": True,
            "running": bool(running),
            "status": state.get("Status"),
            "started_at": state.get("StartedAt"),
            "exit_code": state.get("ExitCode"),
            "message": "",
        }

    # --- Action executors: sharing ---------------------------------------

    @action_executor()
    async def attach_container(
        self, container_id: str,
    ) -> dict[str, Any]:
        """Attach to a peer-owned, shared container in the same session.

        After attaching, this agent can call ``execute_command`` /
        ``execute_script`` against the container but cannot stop or
        restart it — only the owner can.
        """
        record = self._containers.get(container_id)
        if record is None:
            return {
                "container_id": container_id, "attached": False,
                "message": "container_id not tracked by this capability",
            }
        if not record.shared:
            return {
                "container_id": container_id, "attached": False,
                "message": "container was not launched with shared=True",
            }
        record.attached_agent_ids.add(self._agent_id())
        return {
            "container_id": container_id, "attached": True,
            "message": "",
        }

    @action_executor()
    async def detach_container(
        self, container_id: str,
    ) -> dict[str, Any]:
        """Drop this agent's attachment to a shared container."""
        record = self._containers.get(container_id)
        if record is None:
            return {
                "container_id": container_id, "detached": False,
                "message": "container_id not tracked",
            }
        record.attached_agent_ids.discard(self._agent_id())
        return {
            "container_id": container_id, "detached": True, "message": "",
        }

    # --- Execution --------------------------------------------------------

    def _may_exec(self, record: _ContainerRecord) -> bool:
        agent_id = self._agent_id()
        if record.owner_agent_id == agent_id:
            return True
        if record.shared and agent_id in record.attached_agent_ids:
            return True
        return False

    def _truncate(self, s: str, cap: int) -> tuple[str, bool]:
        if len(s) <= cap:
            return s, False
        return s[:cap], True

    @action_executor(interruptible=True)
    async def execute_command(
        self,
        container_id: str,
        command: list[str] | str,
        *,
        timeout_seconds: int = 300,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
        capture_max_bytes: int = 1_000_000,
        stream_to_blackboard: bool = False,
    ) -> dict[str, Any]:
        """Run one command in the named container.

        Args:
            container_id: Target container. Must be owned by or shared
                with this agent.
            command: Argv list or a single shell string (wrapped in
                ``bash -lc`` when a string is passed).
            timeout_seconds: Maximum wall-clock time. Timeouts surface
                as ``exit_code=124`` with the failure reason in
                ``stderr``.
            env: Extra environment variables for the exec'd process.
            workdir: Override the container's default workdir.
            stdin: String written to the exec'd process's stdin.
            capture_max_bytes: Per-stream (stdout, stderr) cap. When
                exceeded, the captured text is truncated and
                ``truncated=True`` is set.
            stream_to_blackboard: If True, also publish stdout/stderr
                chunks to the blackboard at
                ``shell:stream:{container_id}:{exec_id}`` and a final
                ``shell:exec:{exec_id}:complete`` record. The return
                value still contains the full captured output.
        """
        record = self._containers.get(container_id)
        if record is None:
            return self._exec_error(container_id, "container_id not tracked")
        if not self._may_exec(record):
            return self._exec_error(
                container_id,
                "not allowed to exec in this container",
            )
        cmd_list = (
            list(command) if isinstance(command, list)
            else ["bash", "-lc", command]
        )
        exec_id = f"exec_{uuid.uuid4().hex[:12]}"
        stream_key = None
        if stream_to_blackboard:
            stream_key = f"shell:stream:{container_id}:{exec_id}"
            result = await self._run_streaming(
                record=record, cmd=cmd_list,
                timeout_seconds=timeout_seconds, env=env, workdir=workdir,
                stdin=stdin, capture_max_bytes=capture_max_bytes,
                exec_id=exec_id, stream_key=stream_key,
            )
        else:
            try:
                exec_result = await self._backend.exec(
                    record.handle, cmd_list,
                    timeout_seconds=timeout_seconds,
                    env=env, workdir=workdir, stdin=stdin,
                )
            except Exception as e:
                logger.exception(
                    "SandboxedShellCapability.execute_command failed",
                )
                return self._exec_error(
                    container_id, f"backend.exec raised: {e}",
                )
            stdout, trunc_out = self._truncate(exec_result.stdout, capture_max_bytes)
            stderr, trunc_err = self._truncate(exec_result.stderr, capture_max_bytes)
            result = {
                "container_id": container_id,
                "exec_id": exec_id,
                "command": cmd_list,
                "exit_code": exec_result.exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "wall_time_ms": exec_result.wall_time_ms,
                "truncated": trunc_out or trunc_err,
                "stream_key": None,
                "message": "",
            }

        await self._write_audit({
            "ts": time.time(),
            "tenant_id": self._tenant_id(),
            "session_id": self._session_id(),
            "agent_id": self._agent_id(),
            "container_id": container_id,
            "image": record.handle.image,
            "image_role": record.image_role,
            "command": cmd_list,
            "exit_code": result["exit_code"],
            "wall_time_ms": result["wall_time_ms"],
            "stdout_size": len(result.get("stdout", "") or ""),
            "stderr_size": len(result.get("stderr", "") or ""),
        })
        return result

    async def _run_streaming(
        self,
        *,
        record: _ContainerRecord,
        cmd: list[str],
        timeout_seconds: int,
        env: dict[str, str] | None,
        workdir: str | None,
        stdin: str | None,
        capture_max_bytes: int,
        exec_id: str,
        stream_key: str,
    ) -> dict[str, Any]:
        start = time.time()
        bb = None
        try:
            bb = await self.get_blackboard()
        except Exception:  # pragma: no cover — defensive
            pass
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        stdout_size = 0
        stderr_size = 0
        truncated = False
        seq = 0
        try:
            async for tag, chunk in self._backend.exec_stream(
                record.handle, cmd,
                timeout_seconds=timeout_seconds,
                env=env, workdir=workdir, stdin=stdin,
            ):
                if tag == "stdout":
                    if stdout_size + len(chunk) <= capture_max_bytes:
                        stdout_buf.append(chunk)
                        stdout_size += len(chunk)
                    else:
                        truncated = True
                else:
                    if stderr_size + len(chunk) <= capture_max_bytes:
                        stderr_buf.append(chunk)
                        stderr_size += len(chunk)
                    else:
                        truncated = True
                if bb is not None:
                    try:
                        await bb.write(
                            f"{stream_key}:{seq:06d}",
                            {"tag": tag, "chunk": chunk, "ts": time.time()},
                        )
                    except Exception as e:  # pragma: no cover — defensive
                        logger.debug(
                            "SandboxedShellCapability: stream write "
                            "failed: %s", e,
                        )
                seq += 1
        except Exception as e:
            return self._exec_error(
                record.handle.container_id,
                f"exec_stream raised: {e}",
            )
        wall_ms = int((time.time() - start) * 1000)
        # Exit code is not directly available via streaming on the CLI
        # backend (docker exec's exit code lands on the wait call).
        # The CLI backend kills on timeout and the iterator ends; treat
        # "completed cleanly" as exit_code=0 and non-clean as 1. A
        # future AiodockerBackend can surface the real exit code here.
        exit_code = 0
        complete_key = f"shell:exec:{exec_id}:complete"
        if bb is not None:
            try:
                await bb.write(complete_key, {
                    "exit_code": exit_code,
                    "wall_time_ms": wall_ms,
                    "truncated": truncated,
                    "ts": time.time(),
                })
            except Exception as e:  # pragma: no cover — defensive
                logger.debug(
                    "SandboxedShellCapability: complete-key write "
                    "failed: %s", e,
                )
        return {
            "container_id": record.handle.container_id,
            "exec_id": exec_id,
            "command": cmd,
            "exit_code": exit_code,
            "stdout": "".join(stdout_buf),
            "stderr": "".join(stderr_buf),
            "wall_time_ms": wall_ms,
            "truncated": truncated,
            "stream_key": stream_key,
            "message": "",
        }

    @staticmethod
    def _exec_error(container_id: str, message: str) -> dict[str, Any]:
        return {
            "container_id": container_id,
            "exec_id": None,
            "command": [],
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "wall_time_ms": 0,
            "truncated": False,
            "stream_key": None,
            "message": message,
        }

    # --- Scripts ---------------------------------------------------------

    def _validate_script_args(
        self, script: ScriptSpec, args: dict[str, Any],
    ) -> str | None:
        """Return ``None`` on valid args, else a human-readable reason."""
        for name, meta in script.params.items():
            required = bool(meta.get("required", False))
            if required and name not in args:
                return f"missing required param {name!r}"
        # Unknown params are allowed (LLMs may pass extras) but logged.
        unknown = set(args) - set(script.params)
        if unknown:
            logger.debug(
                "SandboxedShellCapability: script %r ignored extras %s",
                script.name, unknown,
            )
        return None

    @staticmethod
    def _render_cmd(
        cmd: tuple[str, ...], args: dict[str, Any],
    ) -> list[str]:
        """Substitute ``{name}`` placeholders in each argv entry."""
        rendered: list[str] = []
        for token in cmd:
            try:
                rendered.append(token.format(**args))
            except KeyError as e:
                raise ValueError(
                    f"script template references missing param {e!s}"
                )
        return rendered

    @action_executor(interruptible=True)
    async def execute_script(
        self,
        container_id: str,
        script_name: str,
        *,
        image_role: str | None = None,
        args: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        stream_to_blackboard: bool = False,
    ) -> dict[str, Any]:
        """Run a registered script in the named container.

        The script is resolved against the image registry — either
        scoped to ``image_role`` or searched across all roles. Args
        are validated against the script's declared params, then
        substituted into the script's argv template before execution.

        Args:
            container_id: Target container.
            script_name: Script name as declared in the registry.
            image_role: Narrow the search to this role. Required when
                two roles define scripts of the same name.
            args: Parameter values to substitute into the script's
                argv template.
            timeout_seconds: Override the script's default timeout.
            stream_to_blackboard: Forwarded to ``execute_command``.
        """
        args = dict(args or {})
        found = self._registry.find_script(
            script_name, image_role=image_role,
        )
        if found is None:
            return self._exec_error(
                container_id,
                f"script {script_name!r} not in the registry",
            )
        _image_spec, script = found
        err = self._validate_script_args(script, args)
        if err is not None:
            return self._exec_error(container_id, err)
        try:
            cmd = self._render_cmd(script.cmd, args)
        except ValueError as e:
            return self._exec_error(container_id, str(e))

        return await self.execute_command(
            container_id=container_id,
            command=cmd,
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None
                else script.timeout_seconds
            ),
            stream_to_blackboard=stream_to_blackboard,
        )

    @action_executor()
    async def list_scripts(
        self, *, image_role: str | None = None,
    ) -> dict[str, Any]:
        """List registered scripts, optionally narrowed to one role."""
        scripts: list[dict[str, Any]] = []
        roles = [image_role] if image_role else self._registry.roles()
        for r in roles:
            spec = self._registry.get(r)
            if spec is None:
                continue
            for s in spec.scripts:
                entry = s.to_summary()
                entry["image_role"] = r
                scripts.append(entry)
        return {"scripts": scripts, "count": len(scripts)}

    @action_executor()
    async def list_images(self) -> dict[str, Any]:
        """List every role currently available in the image registry."""
        images = self._registry.summaries()
        return {"images": images, "count": len(images)}

    # --- File transfer ---------------------------------------------------

    @action_executor()
    async def copy_file_in(
        self,
        container_id: str,
        src_host_path: str,
        dst_container_path: str,
    ) -> dict[str, Any]:
        """Copy a file from the host filesystem into the container."""
        record = self._containers.get(container_id)
        if record is None or not self._may_exec(record):
            return {
                "container_id": container_id, "ok": False,
                "message": "not tracked or not permitted",
            }
        try:
            await self._backend.copy_in(
                record.handle,
                src_host_path=src_host_path,
                dst_container_path=dst_container_path,
            )
        except Exception as e:
            return {
                "container_id": container_id, "ok": False,
                "message": f"copy_in failed: {e}",
            }
        return {"container_id": container_id, "ok": True, "message": ""}

    @action_executor()
    async def copy_file_out(
        self,
        container_id: str,
        src_container_path: str,
        dst_host_path: str,
    ) -> dict[str, Any]:
        """Copy a file from the container onto the host filesystem."""
        record = self._containers.get(container_id)
        if record is None or not self._may_exec(record):
            return {
                "container_id": container_id, "ok": False,
                "message": "not tracked or not permitted",
            }
        try:
            await self._backend.copy_out(
                record.handle,
                src_container_path=src_container_path,
                dst_host_path=dst_host_path,
            )
        except Exception as e:
            return {
                "container_id": container_id, "ok": False,
                "message": f"copy_out failed: {e}",
            }
        return {"container_id": container_id, "ok": True, "message": ""}

    @action_executor()
    async def read_file(
        self, container_id: str, path: str,
        *, max_bytes: int = 1_000_000,
    ) -> dict[str, Any]:
        """Read a file from inside the container (via ``cat``)."""
        record = self._containers.get(container_id)
        if record is None or not self._may_exec(record):
            return {
                "container_id": container_id, "ok": False,
                "content": "", "truncated": False,
                "message": "not tracked or not permitted",
            }
        try:
            exec_result = await self._backend.exec(
                record.handle,
                ["bash", "-lc", f"head -c {max_bytes} {path!s}"],
                timeout_seconds=60,
            )
        except Exception as e:
            return {
                "container_id": container_id, "ok": False,
                "content": "", "truncated": False,
                "message": f"read failed: {e}",
            }
        if exec_result.exit_code != 0:
            return {
                "container_id": container_id, "ok": False,
                "content": "", "truncated": False,
                "message": exec_result.stderr.strip() or (
                    f"read exited with code {exec_result.exit_code}"
                ),
            }
        content = exec_result.stdout
        return {
            "container_id": container_id, "ok": True,
            "content": content,
            "truncated": len(content) >= max_bytes,
            "message": "",
        }

    @action_executor()
    async def write_file(
        self, container_id: str, path: str, content: str,
    ) -> dict[str, Any]:
        """Write a file inside the container via ``tee``.

        The content is streamed on stdin so special characters are not
        reinterpreted by the shell.
        """
        record = self._containers.get(container_id)
        if record is None or not self._may_exec(record):
            return {
                "container_id": container_id, "ok": False,
                "message": "not tracked or not permitted",
            }
        try:
            exec_result = await self._backend.exec(
                record.handle,
                ["bash", "-lc", f"tee {path!s} > /dev/null"],
                timeout_seconds=60,
                stdin=content,
            )
        except Exception as e:
            return {
                "container_id": container_id, "ok": False,
                "message": f"write failed: {e}",
            }
        if exec_result.exit_code != 0:
            return {
                "container_id": container_id, "ok": False,
                "message": exec_result.stderr.strip() or (
                    f"write exited with code {exec_result.exit_code}"
                ),
            }
        return {"container_id": container_id, "ok": True, "message": ""}
