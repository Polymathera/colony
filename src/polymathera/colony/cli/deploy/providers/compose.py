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
from ..extensions import dedupe_packages, lint_setup_commands
from ..health import (
    docker_container_healthy,
    docker_container_running,
    redis_ping,
    wait_until_ready,
)
from ..runtime_writer import (
    write_cluster_runtime,
    write_operator_config,
    write_path_extensions_override,
)
from .base import DeploymentProvider, ProviderStatus, ServiceInfo

# Path to docker-compose.yml relative to this file
_COMPOSE_FILE = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
# User .env file for API keys (same directory as .env.template)
_ENV_FILE = Path(__file__).parent.parent / ".env"
# Docker build artefacts: base/runtime Dockerfiles and the per-run
# .runtime/ directory (cluster-runtime.json + path-extensions override).
_DOCKER_DIR = _COMPOSE_FILE.parent
_DOCKERFILE_BASE = _DOCKER_DIR / "Dockerfile.base"
_BUILD_CONTEXT = _DOCKER_DIR.parents[5]  # .../colony/ — same as compose's context: ../../../../../..
_RUNTIME_DIR = _DOCKER_DIR / ".runtime"
# Tag of the locally-built base image. Match Dockerfile.local's default
# ``COLONY_BASE_IMAGE`` ARG so compose's runtime build picks it up
# without needing --build-arg.
_BASE_IMAGE_TAG = "polymathera/colony-base:local"


class DockerComposeProvider(DeploymentProvider):
    """Manages a local Ray cluster via Docker Compose."""

    def __init__(self, config: DeployConfig) -> None:
        self._config = config

    def _compose_cmd(
        self, *args: str, extra_files: list[Path] | None = None,
    ) -> list[str]:
        """Build a docker compose command with the correct file path.

        Always passes ``--env-file`` when ``deploy/.env`` is present so
        Docker Compose reads the file regardless of the user's CWD.
        Note: ``--env-file`` only affects compose's variable substitution
        and does NOT override shell-exported variables of the same name
        (compose's documented precedence puts shell env above
        ``--env-file``). The subprocess env built by
        :meth:`_compose_subprocess_env` is what makes ``.env``
        authoritative — see that method's docstring.

        ``extra_files`` are additional ``-f`` arguments appended after the
        primary compose file; compose merges them in order. Used by
        :meth:`up` to layer in path-source extension volume mounts.
        """
        cmd = ["docker", "compose", "-f", str(_COMPOSE_FILE)]
        for extra in (extra_files or []):
            cmd.extend(["-f", str(extra)])
        if _ENV_FILE.is_file():
            cmd.extend(["--env-file", str(_ENV_FILE)])
        cmd.extend(args)
        return cmd

    def _compose_subprocess_env(self) -> dict[str, str]:
        """Build the subprocess environment for ``docker compose`` calls.

        Compose substitutes ``${VAR}`` in the YAML from the env of the
        process that runs it. Its documented precedence is:

            shell env  >  --env-file  >  compose ``environment:`` defaults

        That means a stale ``GITHUB_TOKEN`` exported in the user's shell
        silently shadows the value in ``deploy/.env`` — the trap that
        produced "Invalid username or token" failures from valid token
        material on disk.

        We make ``.env`` authoritative by overlaying its values on top
        of ``os.environ`` *before* spawning compose. Compose still
        substitutes from "shell env", but the shell env it sees is the
        one we hand it, with ``.env`` already applied. Only keys
        explicitly listed in :attr:`DeployConfig.api_key_env_vars`
        are overridden — host-level overrides for unrelated variables
        (``HOME``, ``PATH``, ``DOCKER_HOST``, …) flow through
        unchanged.
        """
        env = os.environ.copy()
        env.update(load_dotenv(self._config))
        return env

    async def _exec(
        self, *args: str, capture: bool = True,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run a subprocess and return (returncode, stdout, stderr).

        ``env`` overrides the inherited environment for the spawned
        process. Compose-invoking callers pass
        :meth:`_compose_subprocess_env`'s result so the ``.env`` file
        wins over shell-exported variables.
        """
        if capture:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()
            return proc.returncode, stdout.decode(), stderr.decode()
        else:
            proc = await asyncio.create_subprocess_exec(*args, env=env)
            await proc.wait()
            return proc.returncode, "", ""

    async def _build_base_image(
        self, *, on_status: Callable[[str], None] | None,
    ) -> None:
        """Build ``polymathera/colony-base:local`` from Dockerfile.base.

        Compose's runtime build (Dockerfile.local) does ``FROM
        polymathera/colony-base:local`` — that tag must exist before the
        compose build runs. First-run is slow (Python deps, system
        packages, Node, Linguist); cached layers thereafter.
        """
        if on_status:
            on_status(
                f"Building {_BASE_IMAGE_TAG} (heavy deps; cached on subsequent runs)...",
            )
        rc, _, stderr = await self._exec(
            "docker", "build",
            "-f", str(_DOCKERFILE_BASE),
            "-t", _BASE_IMAGE_TAG,
            str(_BUILD_CONTEXT),
            capture=False, env=self._compose_subprocess_env(),
        )
        if rc != 0:
            raise RuntimeError(f"Base image build failed:\n{stderr}")

    async def _build_bake_image(
        self,
        *,
        runtime_cfg,
        runtime_dir: Path,
        on_status: Callable[[str], None] | None,
    ) -> str:
        """Snapshot the resolved ``cluster.extensions.packages`` into a
        pinned ``colony-local:<hash>`` image. ``FROM colony:local`` (the
        runtime image already built by compose) and pip-installs the
        resolved package list into the same persistent-overlay path the
        container-start hook reads, so the hook's hash check skips the
        install on every boot.

        Returns the bake image tag for compose to pick up via
        ``COLONY_IMAGE``. Path-source packages are not yet supported in
        bake mode (they require copying host paths into the build context;
        deferred). Use the default fast path for path-source workflows.
        """
        pkgs = dedupe_packages(runtime_cfg.extensions.packages)
        path_pkgs = [p for p in pkgs if p.source == "path"]
        if path_pkgs:
            names = ", ".join(p.name for p in path_pkgs)
            raise NotImplementedError(
                f"--bake does not yet support path-source packages ({names}). "
                f"Use the default fast path for path-source workflows.",
            )
        from ..extensions import resolve_pip_args, resolved_hash
        pip_args = resolve_pip_args(pkgs)  # version-only — no yaml_dir needed
        h = resolved_hash(
            pkgs,
            setup_commands=runtime_cfg.setup_commands,
            head_setup_commands=runtime_cfg.head_setup_commands,
            worker_setup_commands=runtime_cfg.worker_setup_commands,
        )
        bake_dir = runtime_dir / f"bake-{h}"
        bake_dir.mkdir(parents=True, exist_ok=True)
        bake_dockerfile = bake_dir / "Dockerfile"
        # Same /opt/colony-overlay layout the container-start hook reads.
        # Writing the hash file under the overlay tells the hook "already
        # installed; skip"; setup_commands still run from the JSON.
        if pip_args:
            install_cmd = "pip install --target=/opt/colony-overlay " + " ".join(
                f'"{arg}"' for arg in pip_args
            )
        else:
            install_cmd = "true"
        bake_dockerfile.write_text(
            "FROM colony:local\n"
            "USER root\n"
            f"RUN {install_cmd} && "
            f'echo "{h}" > /opt/colony-overlay/.installed-hash && '
            "chown -R ray:ray /opt/colony-overlay\n"
            "USER ray\n",
        )
        bake_tag = f"colony-local:{h}"
        if on_status:
            on_status(f"Baking pinned image {bake_tag} ({len(pip_args)} package(s))...")
        rc, _, stderr = await self._exec(
            "docker", "build",
            "-f", str(bake_dockerfile),
            "-t", bake_tag,
            str(bake_dir),
            capture=False, env=self._compose_subprocess_env(),
        )
        if rc != 0:
            raise RuntimeError(f"Bake image build failed:\n{stderr}")
        return bake_tag

    @override
    async def up(
        self,
        build: bool = True,
        workers: int = 1,
        config_path: str | None = None,
        on_status: Callable[[str], None] | None = None,
        bake: bool = False,
    ) -> list[ServiceInfo]:
        """Build image and start Ray cluster + Redis."""
        def _log(msg: str) -> None:
            if on_status:
                on_status(msg)

        # L1-G runtime artefacts: cluster-runtime.json (always written, stub
        # when no extensions), the operator config (always written so the
        # ``/etc/colony/cluster.yaml`` bind mount has a real source —
        # otherwise ray-head's entrypoint hits FileNotFoundError before
        # colony-env's docker-cp can populate /mnt/shared/config.yaml), and
        # the optional path-extensions override. All must exist BEFORE
        # compose up so docker-compose's bind mounts resolve.
        runtime_cfg = write_cluster_runtime(
            config_path=config_path,
            runtime_dir=_RUNTIME_DIR,
            bake_pip_inline=bake,
        )
        write_operator_config(config_path=config_path, runtime_dir=_RUNTIME_DIR)
        for warning in lint_setup_commands(runtime_cfg):
            _log(f"WARN: {warning}")
        yaml_dir = Path(config_path).resolve().parent if config_path else None
        override_path = write_path_extensions_override(
            packages=runtime_cfg.extensions.packages,
            yaml_dir=yaml_dir,
            runtime_dir=_RUNTIME_DIR,
        )
        extra_compose_files = [override_path] if override_path else []

        if build:
            await self._build_base_image(on_status=on_status)
            _log("Building colony:local image (runtime stage)...")
            # Stream build output to terminal so user sees download/compile progress
            rc, _, stderr = await self._exec(
                *self._compose_cmd("build", extra_files=extra_compose_files),
                capture=False,
                env=self._compose_subprocess_env(),
            )
            if rc != 0:
                raise RuntimeError(f"Docker Compose build failed:\n{stderr}")

        compose_env = self._compose_subprocess_env()
        if bake:
            bake_tag = await self._build_bake_image(
                runtime_cfg=runtime_cfg, runtime_dir=_RUNTIME_DIR,
                on_status=on_status,
            )
            compose_env["COLONY_IMAGE"] = bake_tag

        _log(f"Starting services (workers={workers})...")
        rc, _, stderr = await self._exec(
            *self._compose_cmd(
                "up", "-d",
                "--scale", f"ray-worker={workers}",
                extra_files=extra_compose_files,
            ),
            capture=True,
            env=compose_env,
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
            env=self._compose_subprocess_env(),
        )
        if rc != 0:
            raise RuntimeError(f"Docker Compose down failed:\n{stderr}")

    @override
    async def status(self) -> list[ServiceInfo]:
        """Get status of all services."""
        rc, stdout, _ = await self._exec(
            *self._compose_cmd("ps", "--format", "json"),
            env=self._compose_subprocess_env(),
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

        # Check Dockerfile exists (base + runtime)
        checks["dockerfile_base"] = _DOCKERFILE_BASE.is_file()
        checks["dockerfile_local"] = (_DOCKER_DIR / "Dockerfile.local").is_file()

        return checks

    @override
    async def image_info(self) -> dict[str, list[str]]:
        """Inspect the running ray-head: list polymathera-* packages, splitting
        baked (from the runtime image) vs. overlay-installed (from the L1-G
        extensions overlay at /opt/colony-overlay).
        """
        head = self._config.ray.head_container_name
        # ``pip list --path`` shows packages installed in a specific dir.
        # ``--format freeze`` is just "name==version", easy to parse.
        rc_overlay, overlay_out, _ = await self._exec(
            "docker", "exec", head,
            "pip", "list", "--path", "/opt/colony-overlay", "--format", "freeze",
        )
        # All polymathera-* — baked + overlay together. Subtract overlay to
        # get baked. ``pip list`` (no --path) lists everything on sys.path.
        rc_all, all_out, _ = await self._exec(
            "docker", "exec", head, "pip", "list", "--format", "freeze",
        )
        if rc_all != 0:
            raise RuntimeError(
                f"docker exec {head} pip list failed; is the cluster up?",
            )
        overlay_lines = [
            line for line in (overlay_out or "").splitlines()
            if line.strip() and line.lower().startswith("polymathera")
        ]
        all_polymathera = [
            line for line in all_out.splitlines()
            if line.strip() and line.lower().startswith("polymathera")
        ]
        overlay_set = set(overlay_lines)
        baked_lines = [line for line in all_polymathera if line not in overlay_set]
        return {"baked": baked_lines, "overlay": overlay_lines}

    @override
    async def image_build(
        self,
        config_path: str | None = None,
        bake: bool = False,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Build the base + runtime images (and optionally a bake image)
        without bringing the cluster up. Returns the final image tag.
        """
        runtime_cfg = write_cluster_runtime(
            config_path=config_path,
            runtime_dir=_RUNTIME_DIR,
            bake_pip_inline=bake,
        )
        write_operator_config(config_path=config_path, runtime_dir=_RUNTIME_DIR)
        yaml_dir = Path(config_path).resolve().parent if config_path else None
        override_path = write_path_extensions_override(
            packages=runtime_cfg.extensions.packages,
            yaml_dir=yaml_dir,
            runtime_dir=_RUNTIME_DIR,
        )
        extra_compose_files = [override_path] if override_path else []

        await self._build_base_image(on_status=on_status)
        if on_status:
            on_status("Building colony:local image (runtime stage)...")
        rc, _, stderr = await self._exec(
            *self._compose_cmd("build", extra_files=extra_compose_files),
            capture=False, env=self._compose_subprocess_env(),
        )
        if rc != 0:
            raise RuntimeError(f"Docker Compose build failed:\n{stderr}")

        if bake:
            return await self._build_bake_image(
                runtime_cfg=runtime_cfg, runtime_dir=_RUNTIME_DIR,
                on_status=on_status,
            )
        return "colony:local"
