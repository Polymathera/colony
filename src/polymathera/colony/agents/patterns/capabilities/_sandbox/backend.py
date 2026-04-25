"""Container backend abstraction + Docker CLI implementation.

The capability talks to ``ContainerBackend``; the default backend,
``DockerCLIBackend``, shells out to the ``docker`` binary using
``asyncio.create_subprocess_exec``. This choice mirrors the existing
``colony/cli/deploy/compose.py`` invocation style and avoids adding
``aiodocker`` as a hard dependency right now — the design doc lists
that as a future swap (§7.3).

All public methods are async. Command failures surface as
``ExecResult`` with a non-zero exit code + stderr; lifecycle failures
raise ``NoSuchContainer`` or ``RuntimeError``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContainerSpec:
    """Everything needed to launch one container."""

    image: str
    name: str
    env: dict[str, str] = field(default_factory=dict)
    workdir: str = "/workspace"
    cpu_limit: float = 1.0
    memory_limit_mb: int = 2048
    network_mode: Literal["bridge", "none", "host"] = "bridge"
    volumes: tuple[tuple[str, str, str], ...] = ()
    """Each entry is ``(host_path, container_path, mode)`` with mode in
    ``{"ro", "rw"}``. The caller is expected to have validated host paths."""
    labels: dict[str, str] = field(default_factory=dict)
    """Applied with ``--label`` — used for ownership tracking and
    filtering (``list_containers``)."""


@dataclass(frozen=True)
class ContainerHandle:
    """Opaque handle for a launched container."""

    container_id: str
    name: str
    image: str


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one ``exec`` call.

    ``truncated`` is set by the capability (not the backend) when the
    stdout or stderr exceeded the byte cap; leave it at the default
    here.
    """

    exit_code: int
    stdout: str
    stderr: str
    wall_time_ms: int
    truncated: bool = False


class NoSuchContainer(RuntimeError):
    """Raised when a container_id does not exist on the backend."""


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class ContainerBackend(ABC):
    """Pluggable container runtime.

    v1 ships ``DockerCLIBackend``. ``aiodocker`` and ``KubernetesBackend``
    are planned per the design doc; both can be dropped in without
    touching the capability by implementing this ABC.
    """

    @abstractmethod
    async def launch(self, spec: ContainerSpec) -> ContainerHandle: ...

    @abstractmethod
    async def stop(
        self, handle: ContainerHandle, *, timeout_s: int = 10,
    ) -> None: ...

    @abstractmethod
    async def restart(self, handle: ContainerHandle) -> None: ...

    @abstractmethod
    async def is_running(self, handle: ContainerHandle) -> bool: ...

    @abstractmethod
    async def inspect(self, handle: ContainerHandle) -> dict: ...

    @abstractmethod
    async def exec(
        self,
        handle: ContainerHandle,
        cmd: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    def exec_stream(
        self,
        handle: ContainerHandle,
        cmd: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
    ) -> AsyncIterator[tuple[Literal["stdout", "stderr"], str]]:
        """Yield (stream, chunk) tuples until the command exits.

        Returning an async iterator (rather than ``async def``) avoids
        buffering the entire output in memory; callers await the
        iterator and can relay chunks to the blackboard."""
        ...

    @abstractmethod
    async def copy_in(
        self, handle: ContainerHandle,
        *, src_host_path: str, dst_container_path: str,
    ) -> None: ...

    @abstractmethod
    async def copy_out(
        self, handle: ContainerHandle,
        *, src_container_path: str, dst_host_path: str,
    ) -> None: ...

    @abstractmethod
    async def list_by_label(
        self, labels: dict[str, str],
    ) -> list[dict]:
        """Return minimal metadata for every container matching every
        label. Used by ``SandboxedShellCapability.list_containers`` to
        filter to the caller's own containers."""
        ...


# ---------------------------------------------------------------------------
# Docker CLI backend
# ---------------------------------------------------------------------------

class DockerCLIBackend(ContainerBackend):
    """Shell out to the ``docker`` binary.

    Requires:
    - ``docker`` on PATH.
    - The caller process has permission to talk to the Docker daemon
      (membership in the ``docker`` group, or a mounted ``docker.sock``).

    The design doc flags the ``/var/run/docker.sock`` mount into
    ``ray-head`` as the dev-only way to give the capability access;
    production is expected to use a remote daemon over TLS, which this
    backend also supports via ``DOCKER_HOST``.
    """

    def __init__(self, *, docker_binary: str = "docker"):
        self._docker = docker_binary

    # --- Utilities --------------------------------------------------------

    async def _run(
        self,
        *args: str,
        stdin: bytes | None = None,
        timeout_s: float = 60.0,
    ) -> tuple[int, bytes, bytes]:
        """Run ``docker <args>`` and return (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            self._docker, *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"docker {' '.join(args[:2])}... timed out after {timeout_s}s"
            )
        return proc.returncode or 0, stdout, stderr

    async def _check(self, *args: str, timeout_s: float = 60.0) -> bytes:
        code, out, err = await self._run(*args, timeout_s=timeout_s)
        if code != 0:
            raise RuntimeError(
                f"docker {args[0]} failed (exit={code}): {err.decode(errors='replace').strip()}"
            )
        return out

    # --- Lifecycle --------------------------------------------------------

    async def launch(self, spec: ContainerSpec) -> ContainerHandle:
        args: list[str] = ["run", "-d", "--name", spec.name]
        # Resource caps.
        args += ["--cpus", f"{spec.cpu_limit}"]
        args += ["--memory", f"{spec.memory_limit_mb}m"]
        # Networking.
        args += ["--network", spec.network_mode]
        # Security posture — minimal capabilities, no privilege escalation.
        args += ["--cap-drop=ALL", "--security-opt", "no-new-privileges"]
        # Volumes.
        for host, cont, mode in spec.volumes:
            if mode not in ("ro", "rw"):
                raise ValueError(f"volume mode must be 'ro'|'rw', got {mode!r}")
            args += ["-v", f"{host}:{cont}:{mode}"]
        # Env vars.
        for k, v in spec.env.items():
            args += ["-e", f"{k}={v}"]
        # Labels.
        for k, v in spec.labels.items():
            args += ["--label", f"{k}={v}"]
        # Workdir + image + sleep (the container must stay alive; the
        # capability uses `docker exec` to run actual work).
        args += ["--workdir", spec.workdir, spec.image]
        args += ["sleep", "infinity"]

        out = await self._check(*args)
        container_id = out.decode().strip()
        return ContainerHandle(
            container_id=container_id, name=spec.name, image=spec.image,
        )

    async def stop(
        self, handle: ContainerHandle, *, timeout_s: int = 10,
    ) -> None:
        code, _, err = await self._run(
            "rm", "-f", "-v", handle.container_id,
            timeout_s=float(timeout_s) + 5.0,
        )
        if code != 0:
            err_text = err.decode(errors="replace").strip()
            if "No such container" in err_text:
                raise NoSuchContainer(handle.container_id)
            raise RuntimeError(
                f"docker rm failed (exit={code}): {err_text}"
            )

    async def restart(self, handle: ContainerHandle) -> None:
        await self._check("restart", handle.container_id)

    async def is_running(self, handle: ContainerHandle) -> bool:
        try:
            out = await self._check(
                "inspect", "-f", "{{.State.Running}}", handle.container_id,
            )
        except RuntimeError:
            return False
        return out.decode().strip().lower() == "true"

    async def inspect(self, handle: ContainerHandle) -> dict:
        try:
            out = await self._check("inspect", handle.container_id)
        except RuntimeError as e:
            if "No such" in str(e):
                raise NoSuchContainer(handle.container_id) from e
            raise
        data = json.loads(out.decode() or "[]")
        return data[0] if data else {}

    # --- Execution --------------------------------------------------------

    async def exec(
        self,
        handle: ContainerHandle,
        cmd: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
    ) -> ExecResult:
        args: list[str] = ["exec"]
        if stdin is not None:
            args.append("-i")
        if workdir:
            args += ["--workdir", workdir]
        if env:
            for k, v in env.items():
                args += ["-e", f"{k}={v}"]
        args += [handle.container_id, *cmd]

        loop = asyncio.get_event_loop()
        start = loop.time()
        try:
            code, out, err = await self._run(
                *args,
                stdin=stdin.encode() if stdin is not None else None,
                timeout_s=float(timeout_seconds) + 5.0,
            )
        except RuntimeError as e:
            # Timeout from _run. Surface as a failing ExecResult rather
            # than raising so the capability can shape the return.
            wall = int((loop.time() - start) * 1000)
            return ExecResult(
                exit_code=124, stdout="", stderr=str(e),
                wall_time_ms=wall,
            )
        wall = int((loop.time() - start) * 1000)
        return ExecResult(
            exit_code=code,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
            wall_time_ms=wall,
        )

    async def exec_stream(
        self,
        handle: ContainerHandle,
        cmd: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
    ):
        args: list[str] = ["exec"]
        if stdin is not None:
            args.append("-i")
        if workdir:
            args += ["--workdir", workdir]
        if env:
            for k, v in env.items():
                args += ["-e", f"{k}={v}"]
        args += [handle.container_id, *cmd]

        proc = await asyncio.create_subprocess_exec(
            self._docker, *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _reader(
            stream: asyncio.StreamReader,
            tag: Literal["stdout", "stderr"],
            queue: asyncio.Queue,
        ):
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                await queue.put((tag, chunk.decode(errors="replace")))
            await queue.put((tag, None))  # sentinel

        if stdin is not None and proc.stdin is not None:
            proc.stdin.write(stdin.encode())
            await proc.stdin.drain()
            proc.stdin.close()

        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(_reader(proc.stdout, "stdout", queue)),
            asyncio.create_task(_reader(proc.stderr, "stderr", queue)),
        ]

        finished = 0
        try:
            while finished < 2:
                tag, chunk = await asyncio.wait_for(
                    queue.get(), timeout=timeout_seconds + 5,
                )
                if chunk is None:
                    finished += 1
                    continue
                yield tag, chunk
        except asyncio.TimeoutError:
            proc.kill()
            yield "stderr", "[stream timed out]"
        finally:
            for t in tasks:
                t.cancel()
            await proc.wait()

    # --- File transfer ----------------------------------------------------

    async def copy_in(
        self, handle: ContainerHandle,
        *, src_host_path: str, dst_container_path: str,
    ) -> None:
        await self._check(
            "cp", src_host_path, f"{handle.container_id}:{dst_container_path}",
        )

    async def copy_out(
        self, handle: ContainerHandle,
        *, src_container_path: str, dst_host_path: str,
    ) -> None:
        await self._check(
            "cp", f"{handle.container_id}:{src_container_path}", dst_host_path,
        )

    # --- Listing ---------------------------------------------------------

    async def list_by_label(
        self, labels: dict[str, str],
    ) -> list[dict]:
        args = ["ps", "-a", "--no-trunc", "--format", "{{json .}}"]
        for k, v in labels.items():
            args += ["--filter", f"label={k}={v}"]
        code, out, err = await self._run(*args)
        if code != 0:
            raise RuntimeError(
                f"docker ps failed: {err.decode(errors='replace').strip()}"
            )
        entries: list[dict] = []
        for line in out.decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("DockerCLIBackend: skipped malformed line %r", line)
        return entries


def _safe_name(stem: str) -> str:
    """Return a Docker-safe container name.

    Docker names must match ``[a-zA-Z0-9_.-]`` and be <= 253 chars; this
    helper normalises anything the capability passes in (scope ids can
    contain ``:``).
    """
    out = []
    for c in stem:
        if c.isalnum() or c in "_.-":
            out.append(c)
        else:
            out.append("_")
    s = "".join(out).strip("_.")
    return s[:253] or "colony_sandbox"


__all__ = [
    "ContainerSpec",
    "ContainerHandle",
    "ExecResult",
    "NoSuchContainer",
    "ContainerBackend",
    "DockerCLIBackend",
    "_safe_name",
]
