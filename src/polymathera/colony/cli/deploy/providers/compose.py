"""Docker Compose deployment provider for colony-env."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from overrides import override

from ..config import DeployConfig
from ..env import collect_passthrough_env, load_dotenv
from ..health import (
    docker_container_healthy,
    docker_container_running,
    redis_ping,
    wait_until_ready,
)
from .base import DeploymentProvider, ProviderStatus, ServiceInfo

# Path to docker-compose.yml relative to this file
_COMPOSE_FILE = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
# User .env file for API keys (same directory as .env.template)
_ENV_FILE = Path(__file__).parent.parent / ".env"


class DockerComposeProvider(DeploymentProvider):
    """Manages a local Ray cluster via Docker Compose."""

    def __init__(self, config: DeployConfig) -> None:
        self._config = config

    def _compose_cmd(self, *args: str) -> list[str]:
        """Build a docker compose command with the correct file path.

        If a .env file exists (next to .env.template), passes it via
        --env-file so Docker Compose can substitute API keys into the
        compose YAML — regardless of whether the user exported them.
        """
        cmd = ["docker", "compose", "-f", str(_COMPOSE_FILE)]
        if _ENV_FILE.is_file():
            cmd.extend(["--env-file", str(_ENV_FILE)])
        cmd.extend(args)
        return cmd

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

    @override
    async def up(
        self,
        build: bool = True,
        workers: int = 1,
        config_path: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> list[ServiceInfo]:
        """Build image and start Ray cluster + Redis."""
        def _log(msg: str) -> None:
            if on_status:
                on_status(msg)

        if build:
            _log("Building colony:local image...")
            # Stream build output to terminal so user sees download/compile progress
            rc, _, stderr = await self._exec(
                *self._compose_cmd("build"), capture=False,
            )
            if rc != 0:
                raise RuntimeError(f"Docker Compose build failed:\n{stderr}")

        _log(f"Starting services (workers={workers})...")
        rc, _, stderr = await self._exec(
            *self._compose_cmd(
                "up", "-d",
                "--scale", f"ray-worker={workers}",
            ),
            capture=True,
        )
        if rc != 0:
            raise RuntimeError(f"Docker Compose up failed:\n{stderr}")

        # Copy cluster config into the shared volume so ray-head's
        # entrypoint can find it during auto-deploy.
        if config_path:
            config_file = Path(config_path).resolve()
            if not config_file.is_file():
                raise FileNotFoundError(f"Config not found: {config_file}")
            head = self._config.ray.head_container_name
            _log(f"Copying config {config_file.name} to cluster...")
            rc, _, stderr = await self._exec(
                "docker", "cp", str(config_file), f"{head}:/mnt/shared/config.yaml",
            )
            if rc != 0:
                _log(f"WARNING: Failed to copy config: {stderr}")

        # Wait for services to be ready
        _log("Waiting for ray-head to become healthy...")
        head_ready = await wait_until_ready(
            lambda: docker_container_healthy(self._config.ray.head_container_name),
            timeout=120.0,
            interval=3.0,
            description="ray-head",
        )
        _log(
            "ray-head: healthy" if head_ready
            else "ray-head: not healthy (timed out after 120s)"
        )

        _log("Waiting for redis...")
        redis_ready = await wait_until_ready(
            lambda: redis_ping("localhost", self._config.redis.host_port),
            timeout=30.0,
            interval=1.0,
            description="redis",
        )
        _log(
            "redis: ready" if redis_ready
            else "redis: not responding (timed out after 30s)"
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

    @override
    async def down(self) -> None:
        """Stop and remove all containers and volumes."""
        rc, _, stderr = await self._exec(
            *self._compose_cmd("down", "--volumes", "--remove-orphans"),
        )
        if rc != 0:
            raise RuntimeError(f"Docker Compose down failed:\n{stderr}")

    @override
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

    @override
    async def run(
        self,
        origin_url: str | None = None,
        local_repo: str | None = None,
        branch: str = "main",
        commit: str = "HEAD",
        config_path: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        """Submit a job via the dashboard API and poll until completion.

        If --local-repo is given, copies the codebase into the shared volume
        and passes it as a file:// URL.  If --origin-url is given, forwards
        it directly (the container clones via GitStorage).
        Then submits the job to the dashboard API and polls for status.

        Falls back to docker exec if the dashboard API is unreachable.
        """
        import json

        head = self._config.ray.head_container_name
        dashboard_port = self._config.dashboard_ui_port

        # Verify cluster is running
        if not await docker_container_healthy(head):
            raise RuntimeError(
                f"Container '{head}' is not healthy. Run 'colony-env up' first."
            )

        # When a local repo is provided, copy it into the shared volume
        effective_origin_url = origin_url
        if local_repo:
            codebase = Path(local_repo).resolve()
            if not codebase.is_dir():
                raise FileNotFoundError(f"Codebase not found: {codebase}")

            # Clear previous codebase and copy new one
            await self._exec("docker", "exec", head, "rm", "-rf", "/mnt/shared/codebase")
            rc, _, stderr = await self._exec(
                "docker", "cp", f"{codebase}/.", f"{head}:/mnt/shared/codebase/",
            )
            if rc != 0:
                raise RuntimeError(f"Failed to copy codebase:\n{stderr}")
            effective_origin_url = "file:///mnt/shared/codebase"

        # Parse config file for analysis specs
        analyses = [{"type": "basic"}]
        if config_path:
            config_file = Path(config_path).resolve()
            if not config_file.is_file():
                raise FileNotFoundError(f"Config not found: {config_file}")

            try:
                import yaml as yaml_mod
                with open(config_file) as f:
                    yaml_config = yaml_mod.safe_load(f) or {}
                if "analyses" in yaml_config:
                    analyses = yaml_config["analyses"]
            except ImportError:
                pass  # No YAML support — use default analysis

        # Try API-based submission first
        api_url = f"http://localhost:{dashboard_port}/api/v1"
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check dashboard is reachable
                health = await client.get(f"{api_url}/infra/status")
                if health.status_code != 200:
                    raise ConnectionError("Dashboard API not reachable")

                # Submit job
                job_request = {
                    "origin_url": effective_origin_url,
                    "branch": branch,
                    "commit": commit,
                    "analyses": analyses,
                }
                resp = await client.post(f"{api_url}/jobs/submit", json=job_request)
                if resp.status_code != 200:
                    raise RuntimeError(f"Job submission failed: {resp.text}")

                result = resp.json()
                job_id = result["job_id"]
                session_id = result["session_id"]
                print(f"Job submitted: {job_id} (session: {session_id})")

                # Poll for completion
                while True:
                    await asyncio.sleep(5)
                    status_resp = await client.get(f"{api_url}/jobs/{job_id}")
                    if status_resp.status_code != 200:
                        print(f"Failed to get job status: {status_resp.text}")
                        return 1

                    status = status_resp.json()
                    completed = status.get("analyses_completed", 0)
                    total = status.get("analyses_total", 0)
                    job_status = status.get("status", "unknown")

                    print(f"  [{job_status}] {completed}/{total} analyses complete")

                    if job_status in ("completed", "failed", "cancelled"):
                        print(f"Job {job_id}: {job_status} — {status.get('message', '')}")
                        return 0 if job_status == "completed" else 1

        except (ImportError, ConnectionError, Exception) as e:
            print(f"API submission unavailable ({e}), falling back to docker exec...")

        # Fallback: docker exec polymath.py run (legacy path)
        cmd = ["docker", "exec"]

        # Pass through API keys and any extra env vars.
        # load_dotenv() reads the .env file directly so keys are available
        # even if the user didn't `export` them in their shell.
        passthrough = load_dotenv(self._config)
        passthrough.update(collect_passthrough_env(self._config))
        if extra_env:
            passthrough.update(extra_env)
        for key, val in passthrough.items():
            cmd.extend(["-e", f"{key}={val}"])

        # Attach TTY if running interactively
        if sys.stdin.isatty():
            cmd.append("-it")

        cmd.append(head)
        cmd.extend(["python", "-m", "polymathera.colony.cli.polymath", "run"])

        if origin_url:
            cmd.extend(["--origin-url", origin_url])
        else:
            # local_repo was copied to /mnt/shared/codebase above
            cmd.extend(["--local-repo", "/mnt/shared/codebase"])
        cmd.extend(["--branch", branch, "--commit", commit])

        if config_path:
            # Copy config to container
            config_file = Path(config_path).resolve()
            rc, _, stderr = await self._exec(
                "docker", "cp", str(config_file), f"{head}:/mnt/shared/config.yaml",
            )
            if rc == 0:
                cmd.extend(["--config", "/mnt/shared/config.yaml"])

        if extra_args:
            cmd.extend(extra_args)

        # Stream output directly to terminal
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        return proc.returncode

    @override
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
