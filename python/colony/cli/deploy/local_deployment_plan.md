# Implementation Plan: Local Test Environment (`colony-env`)

## Context

We need a Python-based tool to start a local Ray cluster + supporting infrastructure (Redis) so that the existing `polymath.py` CLI can run locally for testing the Colony multi-agent framework. This replaces the 1000+ line shell scripts used for cloud deployment with a simple Python tool focused on the local case, while keeping the abstraction extensible for future cloud deployment.

The existing `polymath.py` already handles all app-level concerns (deploying `PolymatheraCluster`, VCM paging, spawning agents). The deployment tool just sets up the environment that `polymath.py` expects.

**Key principle**: "It is up to the users to launch the Ray cluster and set the necessary environment variables and configs on the driver node before running the `polymathera.colony` app." — `colony-env` IS that user-facing tool.

---

## Infrastructure Requirements (Verified)

| Service | Required? | Why | Default |
|---------|-----------|-----|---------|
| **Redis** | YES | `StateManager`, `RedisClient`, caching — all need it | `localhost:6379` |
| **Ray** | YES | `setup_ray()` connects via `ray.init(address="auto")` | autodetect via `/tmp/ray/` |
| **etcd** | NO | `PolymatheraApp(distributed=False)` skips etcd init | — |
| **Postgres/Neo4j** | NO | Optional storage backends, not needed for core flow | — |

**Critical port conflict**: Ray's GCS server defaults to port 6379, same as Redis. The tool MUST start Ray on a different port (e.g., `--port=6380`).

---

## File Structure

```
colony/python/colony/cli/deploy/
├── __init__.py           # Exports DeploymentManager
├── cli.py                # Typer CLI entry point
├── manager.py            # DeploymentManager orchestrator
├── config.py             # DeployConfig, RedisDeployConfig, RayDeployConfig
├── env.py                # Environment variable builder
├── health.py             # Health check utilities (TCP, Redis ping, Ray status)
└── providers/
    ├── __init__.py
    ├── base.py           # InfraProvider ABC + ProviderInfo/ProviderStatus
    ├── redis.py          # DockerRedisProvider (detect-or-start)
    └── ray.py            # LocalRayProvider (ray start --head)
```

---

## Component Design

### 1. `providers/base.py` — Provider Abstraction

```python
class ProviderStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"

@dataclass
class ProviderInfo:
    name: str
    status: ProviderStatus
    host: str | None = None
    port: int | None = None
    details: dict[str, str] = field(default_factory=dict)

class InfraProvider(ABC):
    @abstractmethod
    async def start(self) -> ProviderInfo: ...
    @abstractmethod
    async def stop(self) -> None: ...
    @abstractmethod
    async def status(self) -> ProviderInfo: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

Extensible for cloud: `DockerRedisProvider` → `ElastiCacheProvider`, `LocalRayProvider` → `EKSRayProvider`.

### 2. `providers/redis.py` — `DockerRedisProvider`

Logic:
1. Check if Redis is already reachable on `host:port` (via TCP connect + PING) → if yes, skip start
2. If not, check if Docker container `colony-redis` exists but stopped → `docker start`
3. If no container, `docker run -d --name colony-redis -p {port}:6379 redis:7-alpine`
4. Poll PING until ready (timeout 15s)

All subprocess calls via `asyncio.create_subprocess_exec` — no shell scripts.

Stop: `docker stop colony-redis && docker rm colony-redis` (only if we started it).

### 3. `providers/ray.py` — `LocalRayProvider`

Logic:
1. Check if Ray is already running: `ray status` succeeds → skip start
2. If not, `ray start --head --port=6380 --dashboard-port=8265 --include-dashboard=true --num-cpus={N} --num-gpus=0`
3. Wait for cluster to be reachable: poll `ray status` (timeout 30s)

Port 6380 avoids conflict with Redis on 6379. `ray.init(address="auto")` in `setup_ray()` autodetects via `/tmp/ray/ray_current_cluster` — no need for explicit `RAY_ADDRESS`.

Stop: `ray stop --force` (only if we started it).

### 4. `config.py` — Deployment Configuration

```python
@dataclass
class RedisDeployConfig:
    host: str = "localhost"
    port: int = 6379
    use_docker: bool = True
    docker_image: str = "redis:7-alpine"
    container_name: str = "colony-redis"

@dataclass
class RayDeployConfig:
    gcs_port: int = 6380           # NOT 6379 (Redis conflict)
    dashboard_port: int = 8265
    num_cpus: int | None = None    # None = auto-detect
    num_gpus: int = 0
    include_dashboard: bool = True

@dataclass
class DeployConfig:
    redis: RedisDeployConfig = field(default_factory=RedisDeployConfig)
    ray: RayDeployConfig = field(default_factory=RayDeployConfig)
    extra_env: dict[str, str] = field(default_factory=dict)
```

### 5. `env.py` — Environment Variable Builder

Builds the env vars that `polymath.py` and `PolymatheraApp` need:

```python
def build_local_env(config: DeployConfig) -> dict[str, str]:
    env = {
        "REDIS_HOST": config.redis.host,
        "REDIS_PORT": str(config.redis.port),
        "RAY_ADDRESS": "auto",
        "POLYMATHERA_ENV": "development",
    }
    # Preserve API keys from current environment
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    env.update(config.extra_env)
    return env
```

**Note**: We do NOT set `POLYMATHERA_RUNNING_LOCALLY=true`. The `ConfigurationManager` with `distributed=False` (already hardcoded in `PolymatheraApp.__init__`) properly resolves env vars like `REDIS_HOST`/`REDIS_PORT` via `json_schema_extra={"env": ...}`. Setting `POLYMATHERA_RUNNING_LOCALLY=true` would bypass this and return `RedisConfig()` with `redis_host=None`, which breaks `get_redis_url()`.

### 6. `manager.py` — `DeploymentManager`

```python
class DeploymentManager:
    def __init__(self, config: DeployConfig): ...

    async def up(self) -> dict[str, ProviderInfo]:
        """Start Redis, then Ray. Set env vars. Return status."""

    async def down(self) -> None:
        """Stop Ray, then Redis (reverse order)."""

    async def status(self) -> dict[str, ProviderInfo]:
        """Check status of all components."""

    async def health(self) -> dict[str, bool]:
        """Health check all components."""

    def get_env_vars(self) -> dict[str, str]:
        """Return env vars dict for polymath.py."""

    def print_run_instructions(self) -> None:
        """Print how to run polymath.py after env is up."""
```

### 7. `cli.py` — Typer CLI

```
colony-env up       Start Redis + Ray local cluster
colony-env down     Stop Ray + Redis
colony-env status   Show status of all components
colony-env env      Print sourceable environment variables
colony-env doctor   Check prerequisites (Docker, Ray, Python, redis-py)
```

Options for `up`:
- `--redis-port` (default 6379)
- `--ray-dashboard-port` (default 8265)
- `--num-cpus` (default auto)
- `--skip-redis` / `--skip-ray` (if user manages them separately)
- `--config` (optional YAML file for DeployConfig overrides)

### 8. `health.py` — Health Check Utilities

- `tcp_check(host, port, timeout)` — raw socket connect
- `redis_ping(host, port, timeout)` — Redis PING command
- `ray_status_check(timeout)` — `ray status` subprocess
- `wait_until_ready(check_fn, timeout, interval)` — polling wrapper

---

## No Changes to Existing Code

After analysis, **no modifications to `setup_ray()` or `polymath.py` are needed**:

1. `ray start --head --port=6380` creates a local cluster. `ray.init(address="auto")` in `setup_ray()` finds it via `/tmp/ray/ray_current_cluster`.
2. Setting `REDIS_HOST=localhost` and `REDIS_PORT=6379` env vars satisfies `RedisConfig`, `StateStorageConfig`, and `RedisStateStorageConfig` (all read from env vars via ConfigurationManager).
3. `PolymatheraApp(distributed=False)` already skips etcd — no changes needed.

---

## Entry Point

Add to `colony/pyproject.toml` under `[tool.poetry.scripts]`:
```toml
colony-env = "colony.cli.deploy.cli:app"
```

---

## Usage Flow

```bash
# 1. Start infrastructure
colony-env up
#   Redis ✓  localhost:6379 (Docker: colony-redis)
#   Ray   ✓  localhost:6380 (dashboard: http://localhost:8265)
#
#   Run your analysis:
#     export REDIS_HOST=localhost REDIS_PORT=6379
#     python -m colony.cli.polymath run /path/to/repo --config analysis.yaml

# 2. Or get a sourceable env block
eval $(colony-env env)
python -m colony.cli.polymath run /path/to/repo --config analysis.yaml

# 3. Check status
colony-env status

# 4. Tear down
colony-env down
```

---

## Cloud Extension Points (Future, not implemented now)

```
providers/
    base.py              # InfraProvider (shared)
    redis.py             # DockerRedisProvider (local)
    ray.py               # LocalRayProvider (local)
    aws/                 # Future
        elasticache.py   # ElastiCacheProvider
        eks.py           # EKSRayProvider
```

`DeployConfig` gains a `target: Literal["local", "aws"]` field. `DeploymentManager` selects providers based on target. The CLI gets a `--target` flag.

---

## Implementation Order

| Step | File | Description |
|------|------|-------------|
| 1 | `deploy/__init__.py` | Package init |
| 2 | `deploy/config.py` | `DeployConfig`, `RedisDeployConfig`, `RayDeployConfig` |
| 3 | `deploy/providers/base.py` | `InfraProvider` ABC, `ProviderInfo`, `ProviderStatus` |
| 4 | `deploy/health.py` | `tcp_check`, `redis_ping`, `ray_status_check`, `wait_until_ready` |
| 5 | `deploy/providers/redis.py` | `DockerRedisProvider` |
| 6 | `deploy/providers/ray.py` | `LocalRayProvider` |
| 7 | `deploy/env.py` | `build_local_env()` |
| 8 | `deploy/manager.py` | `DeploymentManager` |
| 9 | `deploy/cli.py` | Typer CLI with `up`, `down`, `status`, `env`, `doctor` |
| 10 | `pyproject.toml` | Add `colony-env` entry point |

---

## Verification

1. **Prerequisites**: `colony-env doctor` — checks Docker daemon running, `ray` CLI available, `redis` Python package installed
2. **Start**: `colony-env up` — Redis container starts, Ray head starts on port 6380, both health checks pass
3. **Status**: `colony-env status` — shows both services running with ports
4. **Integration**: Run `polymath.py` with a simple remote-deployment config targeting Anthropic, verify it connects to Ray and Redis successfully
5. **Teardown**: `colony-env down` — Ray stopped, Redis container removed
6. **Idempotency**: `colony-env up` twice in a row — second call detects already-running services and skips
