"""``SandboxedShellRuntime`` — one container's full lifecycle.

Extracted from :class:`SandboxedShellCapability`. The split exists
so the polymorphic ``SandboxToolCapability.get_trial_runnable`` can
construct a fresh, **cloudpickle-serializable** runtime per Ray Tune
trial — the multi-container capability (`SandboxedShellCapability`)
can't be serialized because it holds live blackboard handles + an
in-process ``_containers`` accounting dict.

What lives where (split rationale):

- **One container's lifecycle** (launch → exec → stop, workspace
  mkdir, ``ContainerSpec`` construction, label snapshot, wall-time
  killer task) → this class.
- **Multi-container slot accounting** (``max_concurrent_containers``,
  ``max_total_cpu_cores``, ``max_total_memory_mb``), **ownership**
  (``owner_agent_id`` + ``attached_agent_ids``), **live blackboard
  streaming** for ``stream_to_blackboard=True``, **action methods**
  exposed to the LLM planner → stay on
  :class:`SandboxedShellCapability`.

Hard design constraint locked 2026-05-23: **the runtime exposes NO
streaming surface.** Streaming requires a blackboard reference
(Redis client + asyncio pubsub tasks) which is not serializable into
a Ray Tune worker closure. F-2b's trial outputs don't need live
streaming — they flow through the registry-based ``tool_result.json``
contract instead. The placeholder :meth:`exec_stream` exists only to
fail fast if anything accidentally tries to stream through the runtime.

Construction-args contract: every kwarg is plain serializable Python
(strings, numbers, dicts, tuples of dicts). The single class-typed
arg is ``backend`` — :class:`DockerCLIBackend` holds only a string
binary path so it's trivially picklable; future backends must
preserve the same property if they want to be experimentable.

Audit records: the runtime accumulates per-exec audit dicts (using
the labels snapshot it was constructed with for ``tenant_id`` /
``session_id`` / ``agent_id``) and returns them in the
:class:`ExecResultWithAudit`; the parent capability writes them to
the blackboard from agent context.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .backend import (
    ContainerBackend,
    ContainerHandle,
    ContainerSpec,
    ExecResult,
    NoSuchContainer,
)
from .registry import ScriptSpec


logger = logging.getLogger(__name__)


_NetworkMode = Literal["bridge", "none", "host"]

# Label keys (must mirror the constants on ``SandboxedShellCapability``
# so audit records carry the same fields regardless of construction
# path — the capability builds labels via ``_build_labels`` and the
# runtime reads them back here for the audit records).
_LABEL_OWNER = "colony.owner_agent_id"
_LABEL_TENANT = "colony.tenant_id"
_LABEL_SESSION = "colony.session_id"
_LABEL_ROLE = "colony.image_role"


@dataclass(frozen=True)
class ExecResultWithAudit:
    """Wraps :class:`ExecResult` with the audit records the parent
    capability should write to its blackboard.

    Frozen for immutability after construction; the runtime never
    mutates a returned value. The ``audit_records`` tuple is
    JSON-serialisable (the parent writes it to the blackboard via
    ``await bb.write(key, value)``).
    """

    exec_result: ExecResult
    audit_records: tuple[dict[str, Any], ...] = ()


class SandboxedShellRuntime:
    """One container's lifecycle. Disposable, use-once.

    The expected lifecycle from the capability's perspective:

    1. Construct (cheap; no I/O).
    2. ``await runtime.launch()`` — mkdirs workspace, calls
       ``backend.launch``, arms the wall-time killer.
    3. Zero or more ``await runtime.exec(...)`` /
       ``await runtime.exec_script(...)`` /
       ``await runtime.copy_in(...)`` / ``await runtime.copy_out(...)``
       calls.
    4. ``await runtime.stop()`` — cancels the killer, calls
       ``backend.stop``. Safe to call multiple times; second + later
       are no-ops.

    ``SandboxToolCapability.get_trial_runnable`` constructs
    the runtime FRESH per trial in a Ray Tune worker, so the
    use-once contract matches the trial lifecycle.
    """

    def __init__(
        self,
        *,
        backend: ContainerBackend,
        image: str,
        workspace_path: str,
        workdir: str,
        env: dict[str, str] | None,
        cpu_limit: float,
        memory_limit_mb: int,
        network_mode: _NetworkMode,
        extra_volumes: tuple[tuple[str, str, str], ...] | None,
        labels: dict[str, str] | None,
        max_wall_time_seconds: int,
        capture_max_bytes: int,
        container_name: str,
    ) -> None:
        self._backend = backend
        self._image = image
        self._workspace_path = workspace_path
        self._workdir = workdir
        self._env: dict[str, str] = dict(env or {})
        self._cpu_limit = cpu_limit
        self._memory_limit_mb = memory_limit_mb
        self._network_mode = network_mode
        self._extra_volumes: tuple[tuple[str, str, str], ...] = tuple(
            extra_volumes or (),
        )
        self._labels: dict[str, str] = dict(labels or {})
        self._max_wall_time_seconds = max_wall_time_seconds
        self._capture_max_bytes = capture_max_bytes
        self._container_name = container_name
        # Mutable runtime state — NOT part of construction.
        self._handle: ContainerHandle | None = None
        self._deadline_task: asyncio.Task[None] | None = None
        self._killed_by_deadline: bool = False
        self._stopped: bool = False
        self._stop_already_gone: bool = False

    # ------------------------------------------------------------------
    # Read-only state for the parent capability
    # ------------------------------------------------------------------

    @property
    def handle(self) -> ContainerHandle | None:
        """The backend handle after :meth:`launch`; ``None`` before."""
        return self._handle

    @property
    def workspace_path(self) -> str:
        return self._workspace_path

    @property
    def network_mode(self) -> _NetworkMode:
        return self._network_mode

    @property
    def killed_by_deadline(self) -> bool:
        """True if the wall-time killer fired."""
        return self._killed_by_deadline

    @property
    def stop_already_gone(self) -> bool:
        """True if :meth:`stop` caught :class:`NoSuchContainer` (the
        container was already gone on the daemon). Lets the parent
        capability surface a "container was already gone" message in
        its action result without making the caller distinguish
        between "we stopped it" and "it was already stopped."""
        return self._stop_already_gone

    @property
    def deadline_task(self) -> asyncio.Task[None] | None:
        """The wall-time killer task (or ``None`` before launch /
        after stop / when no deadline)."""
        return self._deadline_task

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def launch(self) -> ContainerHandle:
        """Idempotent — returns the existing handle if already launched.

        On first call: mkdirs the workspace, builds the
        :class:`ContainerSpec`, calls ``backend.launch``, arms the
        wall-time killer.
        """
        if self._handle is not None:
            return self._handle
        try:
            Path(self._workspace_path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Match the pre-refactor capability behaviour: log + continue;
            # backend.launch will surface the real error if the missing
            # workspace breaks the bind mount.
            logger.warning(
                "SandboxedShellRuntime: workspace mkdir %s failed: %s",
                self._workspace_path, exc,
            )
        volumes: tuple[tuple[str, str, str], ...] = (
            (self._workspace_path, self._workdir, "rw"),
        ) + self._extra_volumes
        spec = ContainerSpec(
            image=self._image,
            name=self._container_name,
            env=dict(self._env),
            workdir=self._workdir,
            cpu_limit=self._cpu_limit,
            memory_limit_mb=self._memory_limit_mb,
            network_mode=self._network_mode,
            volumes=volumes,
            labels=dict(self._labels),
        )
        self._handle = await self._backend.launch(spec)
        await self._arm_deadline()
        return self._handle

    async def _arm_deadline(self) -> None:
        """Schedule the wall-time killer. ``max_wall_time_seconds <= 0``
        disables the killer entirely (used by long-running interactive
        shells; trial use cases always set a finite cap)."""
        if self._max_wall_time_seconds <= 0:
            return

        async def _kill_after() -> None:
            try:
                await asyncio.sleep(self._max_wall_time_seconds)
            except asyncio.CancelledError:
                return
            if self._handle is None or self._stopped:
                return
            self._killed_by_deadline = True
            logger.info(
                "SandboxedShellRuntime: container %s reached "
                "max_wall_time_seconds=%ds; stopping",
                self._handle.container_id, self._max_wall_time_seconds,
            )
            try:
                await self._backend.stop(self._handle, timeout_s=5)
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.debug(
                    "SandboxedShellRuntime: wall-time stop failed: %s",
                    exc,
                )

        self._deadline_task = asyncio.create_task(_kill_after())

    async def stop(self, *, timeout_s: int = 10) -> None:
        """Cancel the killer + call ``backend.stop``. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._deadline_task is not None:
            self._deadline_task.cancel()
            self._deadline_task = None
        if self._handle is None:
            return
        try:
            await self._backend.stop(self._handle, timeout_s=timeout_s)
        except NoSuchContainer:
            # Container already gone on the daemon — treat as success
            # but mark it so the parent capability can surface the
            # diagnostic message ("container was already gone…").
            self._stop_already_gone = True

    # ------------------------------------------------------------------
    # Exec
    # ------------------------------------------------------------------

    async def exec(
        self,
        cmd: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
        capture_max_bytes: int | None = None,
    ) -> ExecResultWithAudit:
        """Run one command (full-buffered; non-streaming).

        ``capture_max_bytes`` overrides the runtime's default cap for
        this single exec — the pre-refactor capability had it as a
        per-call kwarg, so the override preserves that. When ``None``,
        falls back to the runtime's construction-time cap.

        Streaming output is NOT supported on the runtime — the parent
        capability owns the streaming path because it requires a
        blackboard reference (which isn't serializable into a Ray
        Tune worker closure).
        """
        if self._handle is None:
            raise RuntimeError(
                "SandboxedShellRuntime.exec called before launch()",
            )
        if self._killed_by_deadline:
            raise RuntimeError(
                "SandboxedShellRuntime: container killed by wall-time "
                f"guard (max_wall_time_seconds={self._max_wall_time_seconds}s)",
            )
        merged_env: dict[str, str] = {**self._env, **(env or {})}
        exec_result = await self._backend.exec(
            self._handle,
            list(cmd),
            timeout_seconds=timeout_seconds,
            env=merged_env,
            workdir=workdir or self._workdir,
            stdin=stdin,
        )
        # Truncation matches the pre-refactor capability behaviour:
        # each stream independently capped, ``truncated=True`` set on
        # the result when either stream exceeded the cap.
        cap = (
            capture_max_bytes if capture_max_bytes is not None
            else self._capture_max_bytes
        )
        stdout, t_out = self._truncate(exec_result.stdout, cap)
        stderr, t_err = self._truncate(exec_result.stderr, cap)
        truncated = ExecResult(
            exit_code=exec_result.exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_time_ms=exec_result.wall_time_ms,
            truncated=t_out or t_err,
        )
        audit = self._build_audit_record(cmd=list(cmd), exec_result=truncated)
        return ExecResultWithAudit(
            exec_result=truncated,
            audit_records=(audit,),
        )

    async def exec_script(
        self,
        script: ScriptSpec,
        *,
        params: dict[str, Any],
    ) -> ExecResultWithAudit:
        """Validate ``params`` against ``script.params``, render the
        ``script.cmd`` template, and dispatch through :meth:`exec`.

        Unknown extra params are tolerated (logged at debug) — matches
        pre-refactor capability behaviour where LLM planners
        occasionally pass extras the script ignores.
        """
        err = _validate_script_args(script, params)
        if err is not None:
            raise ValueError(err)
        rendered = _render_cmd(script.cmd, params)
        return await self.exec(
            rendered, timeout_seconds=script.timeout_seconds,
        )

    def exec_stream(self, *args: Any, **kwargs: Any) -> Any:
        """Streaming is NOT supported on the runtime.

        Raises :class:`RuntimeError` unconditionally. The parent
        :class:`SandboxedShellCapability` keeps the streaming path
        (``execute_command(stream_to_blackboard=True)`` →
        ``_run_streaming``); the runtime can't because a blackboard
        handle is not cloudpickle-serializable.
        """
        raise RuntimeError(
            "SandboxedShellRuntime does not support streaming. Use "
            "SandboxedShellCapability.execute_command("
            "stream_to_blackboard=True) at the parent capability "
            "instead. The runtime is serializable; a blackboard "
            "handle is not.",
        )

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    async def copy_in(
        self, *, src_host_path: str, dst_container_path: str,
    ) -> None:
        if self._handle is None:
            raise RuntimeError(
                "SandboxedShellRuntime.copy_in called before launch()",
            )
        await self._backend.copy_in(
            self._handle,
            src_host_path=src_host_path,
            dst_container_path=dst_container_path,
        )

    async def copy_out(
        self, *, src_container_path: str, dst_host_path: str,
    ) -> None:
        if self._handle is None:
            raise RuntimeError(
                "SandboxedShellRuntime.copy_out called before launch()",
            )
        await self._backend.copy_out(
            self._handle,
            src_container_path=src_container_path,
            dst_host_path=dst_host_path,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _truncate(self, s: str, cap: int) -> tuple[str, bool]:
        if len(s) <= cap:
            return s, False
        return s[:cap], True

    def _build_audit_record(
        self, *, cmd: list[str], exec_result: ExecResult,
    ) -> dict[str, Any]:
        """Construct a per-exec audit dict.

        ``tenant_id`` / ``session_id`` / ``agent_id`` come from the
        labels snapshot taken at construction time (matches what
        ``SandboxedShellCapability._build_labels`` already emits). In
        Ray Tune worker contexts the labels are the only source of
        agent identity — context-vars don't propagate across the Ray
        boundary.
        """
        assert self._handle is not None
        return {
            "ts": time.time(),
            "tenant_id": self._labels.get(_LABEL_TENANT, "unknown"),
            "session_id": self._labels.get(_LABEL_SESSION, "unknown"),
            "agent_id": self._labels.get(_LABEL_OWNER, "unknown"),
            "container_id": self._handle.container_id,
            "image": self._handle.image,
            "image_role": self._labels.get(_LABEL_ROLE, "unknown"),
            "command": list(cmd),
            "exit_code": exec_result.exit_code,
            "wall_time_ms": exec_result.wall_time_ms,
            "stdout_size": len(exec_result.stdout),
            "stderr_size": len(exec_result.stderr),
        }


# ---------------------------------------------------------------------------
# Script-arg helpers (extracted from SandboxedShellCapability so the
# runtime can dispatch ``exec_script`` without reaching back to the
# capability)
# ---------------------------------------------------------------------------


def _validate_script_args(
    script: ScriptSpec, args: dict[str, Any],
) -> str | None:
    """Return ``None`` on valid args, else a human-readable reason."""
    for name, meta in script.params.items():
        required = bool(meta.get("required", False))
        if required and name not in args:
            return f"missing required param {name!r}"
    unknown = set(args) - set(script.params)
    if unknown:
        logger.debug(
            "SandboxedShellRuntime: script %r ignored extras %s",
            script.name, unknown,
        )
    return None


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
                f"script template references missing param {e!s}",
            )
    return rendered


__all__ = (
    "ExecResultWithAudit",
    "SandboxedShellRuntime",
)
