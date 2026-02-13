"""Docker Compose deployment provider for colony-env."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from ..config import DeployConfig
from ..env import collect_passthrough_env
from ..health import (
    docker_container_healthy,
    docker_container_running,
    redis_ping,
    wait_until_ready,
)
from .base import DeploymentProvider, ProviderStatus, ServiceInfo

# Path to docker-compose.yml relative to this file
_COMPOSE_FILE = Path(__file__).parent.parent / "docker" / "docker-compose.yml"


class DockerComposeProvider(DeploymentProvider):
    """Manages a local Ray cluster via Docker Compose."""

    def __init__(self, config: DeployConfig) -> None:
        self._config = config

    def _compose_cmd(self, *args: str) -> list[str]:
        """Build a docker compose command with the correct file path."""
        return ["docker", "compose", "-f", str(_COMPOSE_FILE), *args]

    async def _exec(self, *args: str, capture: bool = True) -> tuple[int, str, str]:
        """Run a subprocess and return (returncode, stdout, stderr)."""
        if capture:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout.decode(), stderr.decode()
        else:
            proc = await asyncio.create_subprocess_exec(*args)
            await proc.wait()
            return proc.returncode, "", ""

    async def up(self, build: bool = True, workers: int = 1) -> list[ServiceInfo]:
        """Build image and start Ray cluster + Redis."""
        if build:
            rc, _, stderr = await self._exec(*self._compose_cmd("build"))
            if rc != 0:
                raise RuntimeError(f"Docker Compose build failed:\n{stderr}")

        env = {
            **os.environ,
            "COLONY_DASHBOARD_PORT": str(self._config.ray.dashboard_port),
            "COLONY_CLIENT_PORT": str(self._config.ray.client_port),
            "COLONY_REDIS_PORT": str(self._config.redis.host_port),
        }

        rc, _, stderr = await self._exec(
            *self._compose_cmd(
                "up", "-d",
                "--scale", f"ray-worker={workers}",
            ),
            capture=True,
        )
        if rc != 0:
            raise RuntimeError(f"Docker Compose up failed:\n{stderr}")

        # Wait for services to be ready
        head_ready = await wait_until_ready(
            lambda: docker_container_healthy(self._config.ray.head_container_name),
            timeout=120.0,
            interval=3.0,
            description="ray-head",
        )

        redis_ready = await wait_until_ready(
            lambda: redis_ping("localhost", self._config.redis.host_port),
            timeout=30.0,
            interval=1.0,
            description="redis",
        )

        services = [
            ServiceInfo(
                name="ray-head",
                status=ProviderStatus.RUNNING if head_ready else ProviderStatus.ERROR,
                host="localhost",
                port=self._config.ray.dashboard_port,
                details={"dashboard": f"http://localhost:{self._config.ray.dashboard_port}"},
            ),
            ServiceInfo(
                name="ray-worker",
                status=ProviderStatus.RUNNING if head_ready else ProviderStatus.STARTING,
                details={"replicas": str(workers)},
            ),
            ServiceInfo(
                name="redis",
                status=ProviderStatus.RUNNING if redis_ready else ProviderStatus.ERROR,
                host="localhost",
                port=self._config.redis.host_port,
            ),
        ]
        return services

    async def down(self) -> None:
        """Stop and remove all containers and volumes."""
        rc, _, stderr = await self._exec(
            *self._compose_cmd("down", "--volumes", "--remove-orphans"),
        )
        if rc != 0:
            raise RuntimeError(f"Docker Compose down failed:\n{stderr}")

    async def status(self) -> list[ServiceInfo]:
        """Get status of all services."""
        rc, stdout, _ = await self._exec(
            *self._compose_cmd("ps", "--format", "json"),
        )
        if rc != 0:
            return []

        services = []
        # docker compose ps --format json outputs one JSON object per line
        for line in stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                container = json.loads(line)
            except json.JSONDecodeError:
                continue

            name = container.get("Service", container.get("Name", "unknown"))
            state = container.get("State", "unknown")
            health = container.get("Health", "")

            if state == "running":
                status = ProviderStatus.RUNNING
            elif state == "exited":
                status = ProviderStatus.STOPPED
            else:
                status = ProviderStatus.ERROR

            details = {}
            if health:
                details["health"] = health
            publishers = container.get("Publishers") or []
            for pub in publishers:
                if pub.get("PublishedPort"):
                    details["port"] = f"{pub['PublishedPort']}"

            services.append(ServiceInfo(
                name=name,
                status=status,
                details=details,
            ))

        return services

    async def run(
        self,
        codebase_path: str,
        config_path: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        """Run polymath.py inside the ray-head container.

        Copies the codebase into the shared volume, then executes
        polymath.py inside the running head container.
        """
        head = self._config.ray.head_container_name

        # Verify cluster is running
        if not await docker_container_healthy(head):
            raise RuntimeError(
                f"Container '{head}' is not healthy. Run 'colony-env up' first."
            )

        # Copy codebase into shared volume
        codebase = Path(codebase_path).resolve()
        if not codebase.is_dir():
            raise FileNotFoundError(f"Codebase not found: {codebase}")

        # Clear previous codebase and copy new one
        await self._exec("docker", "exec", head, "rm", "-rf", "/mnt/shared/codebase")
        rc, _, stderr = await self._exec(
            "docker", "cp", f"{codebase}/.", f"{head}:/mnt/shared/codebase/",
        )
        if rc != 0:
            raise RuntimeError(f"Failed to copy codebase:\n{stderr}")

        # Copy config if provided
        container_config_path = None
        if config_path:
            config_file = Path(config_path).resolve()
            if not config_file.is_file():
                raise FileNotFoundError(f"Config not found: {config_file}")
            rc, _, stderr = await self._exec(
                "docker", "cp", str(config_file), f"{head}:/mnt/shared/config.yaml",
            )
            if rc != 0:
                raise RuntimeError(f"Failed to copy config:\n{stderr}")
            container_config_path = "/mnt/shared/config.yaml"

        # Build docker exec command
        cmd = ["docker", "exec"]

        # Pass through API keys and any extra env vars
        passthrough = collect_passthrough_env(self._config)
        if extra_env:
            passthrough.update(extra_env)
        for key, val in passthrough.items():
            cmd.extend(["-e", f"{key}={val}"])

        # Attach TTY if running interactively
        if sys.stdin.isatty():
            cmd.append("-it")

        cmd.append(head)
        cmd.extend(["python", "-m", "colony.cli.polymath", "run", "/mnt/shared/codebase"])

        if container_config_path:
            cmd.extend(["--config", container_config_path])

        if extra_args:
            cmd.extend(extra_args)

        # Stream output directly to terminal
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        return proc.returncode

    async def doctor(self) -> dict[str, bool]:
        """Check prerequisites for Docker Compose deployment."""
        checks = {}

        # Check Docker daemon
        rc, _, _ = await self._exec("docker", "info")
        checks["docker_daemon"] = rc == 0

        # Check docker compose
        rc, _, _ = await self._exec("docker", "compose", "version")
        checks["docker_compose"] = rc == 0

        # Check compose file exists
        checks["compose_file"] = _COMPOSE_FILE.is_file()

        # Check Dockerfile exists
        dockerfile = _COMPOSE_FILE.parent / "Dockerfile.local"
        checks["dockerfile"] = dockerfile.is_file()

        return checks
