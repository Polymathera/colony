# Implementation Plan: Local Test Environment (`colony-env`) — Docker/K8s-Based

## Context

Colony is an open-source multi-agent framework that runs on Ray clusters. Users need a **zero-friction** way to spin up a local test environment and run the `polymath.py` integration test without installing complex dependencies or reading long READMEs. The only prerequisite should be **Docker** (for the default mode).

The existing cloud deployment uses Docker images (`Dockerfile.ray`, `Dockerfile.ray-cpu`) + KubeRay on EKS. The local deployment must mirror this architecture: **all Colony dependencies live inside Docker images**, and Ray runs containerized — not natively on the host.

The existing `polymath.py` already handles all app-level concerns (deploying `PolymatheraCluster`, VCM paging, spawning agents). The deployment tool sets up the infrastructure that `polymath.py` expects.

---

## Architecture Overview

```
User's machine (only Docker required)
│
├── colony-env up          → Builds Colony Docker image, starts containers
├── colony-env run PATH    → Runs polymath.py INSIDE the ray-head container
├── colony-env status      → Shows running services
└── colony-env down        → Tears everything down

Inside Docker:
┌─────────────────────────────────────────────────┐
│  colony-net (Docker bridge network)             │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐      │
│  │ ray-head │  │ray-worker│  │  redis    │      │
│  │ (colony  │  │ (colony  │  │ (7-alpine)│      │
│  │  image)  │  │  image)  │  │           │      │
│  │          │  │          │  │ port 6379 │      │
│  │ GCS:6379 │  │ joins    │  └───────────┘      │
│  │ dash:8265│  │ ray-head │                     │
│  │client:   │  │          │                     │
│  │  10001   │  │          │                     │
│  └──────────┘  └──────────┘                     │
│       │                                         │
│  colony-shared volume (/mnt/shared)             │
│  (codebase mounted here for analysis)           │
└─────────────────────────────────────────────────┘
```

**Two deployment modes** (same CLI interface):

| | Docker Compose (Default) | Kind + KubeRay (Advanced) |
|---|---|---|
| **Prerequisites** | Docker only | Docker + kubectl + kind + helm |
| **Setup time** | ~30s (image cached) | ~3-5 min |
| **Autoscaling** | Manual (`--workers N`) | Full KubeRay autoscaler |
| **Production parity** | Good (same image) | Exact (same K8s manifests) |
| **GPU support** | Yes (nvidia-docker) | No (Kind limitation) |
| **Best for** | 90% of users, daily dev | Testing K8s-specific behavior |

**V1 implements Docker Compose only.** Kind+KubeRay is stubbed with the provider interface for future implementation.

---

## Infrastructure Requirements

| Service | Required? | Why | Container |
|---------|-----------|-----|-----------|
| **Redis** | YES | `StateManager`, `RedisClient`, caching | `redis:7-alpine` on port 6379 |
| **Ray** | YES | `setup_ray()` → `ray.init(address="auto")` | `colony:local` (head + workers) |
| **etcd** | NO | `PolymatheraApp(distributed=False)` skips it | — |
| **Postgres/Neo4j** | NO | Optional storage backends | — |

**Port conflict note**: Ray GCS defaults to port 6379, same as Redis. Inside Docker Compose this is not an issue — Ray and Redis are separate containers with different hostnames. On the host, only Redis maps to `localhost:6379`; Ray dashboard maps to `localhost:8265`.

---

## Key Design Decision: Driver Runs Inside the Container

`polymath.py` (the driver) runs **inside the ray-head container**, not on the host. This is critical because:

1. `ray.init(address="auto")` works natively inside the container (finds the local Ray head via `/tmp/ray/ray_current_cluster`)
2. All Colony dependencies are in the container — no host installation needed
3. This mirrors production where the driver runs on the Ray head node
4. `REDIS_HOST=redis` resolves via Docker DNS inside the network

The `colony-env run` command handles mounting the target codebase into the container and passing through API keys.

---

## File Structure

```
colony/python/colony/cli/deploy/
├── __init__.py                    # Exports DeploymentManager
├── cli.py                         # Typer CLI (colony-env)
├── manager.py                     # DeploymentManager orchestrator
├── config.py                      # DeployConfig, RedisDeployConfig, RayDeployConfig
├── env.py                         # Environment variable builder
├── health.py                      # Health check utilities (TCP, Redis ping, Ray status)
├── docker/
│   ├── Dockerfile.local           # Colony image for local dev (adapted from Dockerfile.ray-cpu)
│   └── docker-compose.yml         # ray-head + ray-worker + redis
├── k8s/                           # Future: Kind + KubeRay
│   ├── kind-config.yaml           # Kind cluster with port mappings
│   └── ray-cluster-local.yaml     # Simplified RayCluster CRD
└── providers/
    ├── __init__.py
    ├── base.py                    # DeploymentProvider ABC
    ├── compose.py                 # DockerComposeProvider (V1)
    └── k8s.py                     # KindKubeRayProvider (stub)
```

---

## Component Design

### 1. `docker/Dockerfile.local` — Colony Docker Image

Adapted from the production `Dockerfile.ray-cpu` (`/home/anassar/workspace/polymathera/deployment/docker/Dockerfile.ray-cpu`). Same base image, same dependency installation, stripped of cloud-specific tooling.

**Base**: `rayproject/ray:2.49.0-py311-cpu` (matches production Ray version)

**KEEP from Dockerfile.ray-cpu:**
- Poetry install with `--only main` + extras `code_analysis,cpu`
- faiss-cpu==1.7.4
- System deps: git, git-lfs, graphviz, graphviz-dev, build-essential, python3-dev, libmagic1, cloc, ruby-dev + linguist gem, pkg-config, cmake, libblas/lapack/openblas-dev
- Ray user setup, /home/ray/app workdir, proper permissions
- `RAY_DISABLE_DOCKER_CPU_WARNING=1`

**STRIP (not needed locally):**
- SSH server/client config (Docker networking, no SSH needed)
- ubuntu user creation
- awscli, nfs-common (no AWS/EFS)
- node_exporter, prometheus monitoring, metrics service
- setup_node.sh, setup_head_node.sh, health_check.sh, cleanup.sh scripts
- systemd service files

**Key structure:**
```dockerfile
FROM rayproject/ray:2.49.0-py311-cpu

ARG RAY_CONTAINER_USER=ray
ARG APP_MOUNT_PATH=/home/ray/app

USER root
WORKDIR ${APP_MOUNT_PATH}

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime

# Poetry install
COPY pyproject.toml poetry.lock ./
ARG POLYMATHERA_PYTHON_CMD=python3.11
RUN ${POLYMATHERA_PYTHON_CMD} -m pip install --no-cache-dir poetry
RUN --mount=type=cache,target=/root/.cache \
    poetry config virtualenvs.create false && \
    poetry config virtualenvs.in-project false && \
    poetry lock --no-ansi --no-interaction && \
    poetry install --only main --extras "code_analysis cpu" --no-interaction --no-ansi --no-root && \
    pip install --no-cache-dir faiss-cpu==1.7.4

# System deps (subset of production)
RUN apt-get update -yq && apt-get install -y --no-install-recommends \
    build-essential git git-lfs graphviz graphviz-dev pkg-config \
    libmagic1 python3-dev libblas-dev liblapack-dev libopenblas-dev \
    cloc ruby-dev libssl-dev libkrb5-dev libicu-dev zlib1g-dev \
    libcurl4-openssl-dev cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN gem install github-linguist

# Copy project code
COPY --chown=${RAY_CONTAINER_USER}:${RAY_CONTAINER_USER} . .

RUN chown -R ${RAY_CONTAINER_USER}:${RAY_CONTAINER_USER} ${APP_MOUNT_PATH} && \
    chmod -R 755 ${APP_MOUNT_PATH} && \
    mkdir -p /tmp/ray && chown -R ${RAY_CONTAINER_USER}:${RAY_CONTAINER_USER} /tmp/ray && \
    chown -R ${RAY_CONTAINER_USER}:users /home/${RAY_CONTAINER_USER}/.config && \
    chown -R ${RAY_CONTAINER_USER}:users /home/${RAY_CONTAINER_USER}/.local

ENV RAY_DISABLE_DOCKER_CPU_WARNING=1

USER ${RAY_CONTAINER_USER}
CMD ["/bin/bash"]
```

### 2. `docker/docker-compose.yml` — Local Cluster

```yaml
services:
  ray-head:
    build:
      context: ../../../../..   # colony repo root (where pyproject.toml lives)
      dockerfile: python/colony/cli/deploy/docker/Dockerfile.local
    image: colony:local
    container_name: colony-ray-head
    command: >
      ray start --head
      --port=6379
      --dashboard-host=0.0.0.0
      --dashboard-port=8265
      --ray-client-server-port=10001
      --num-cpus=0
      --block
    ports:
      - "${COLONY_DASHBOARD_PORT:-8265}:8265"
      - "${COLONY_CLIENT_PORT:-10001}:10001"
    volumes:
      - colony-shared:/mnt/shared
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - RAY_CLUSTER_ENVIRONMENT=true
      - POLYMATHERA_ENV=development
    depends_on:
      redis:
        condition: service_started
    shm_size: '4gb'
    networks:
      - colony-net
    healthcheck:
      test: ["CMD", "ray", "status"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s

  ray-worker:
    image: colony:local
    command: >
      ray start
      --address=ray-head:6379
      --block
    volumes:
      - colony-shared:/mnt/shared
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - RAY_CLUSTER_ENVIRONMENT=true
      - POLYMATHERA_ENV=development
    depends_on:
      ray-head:
        condition: service_healthy
    shm_size: '4gb'
    networks:
      - colony-net

  redis:
    image: redis:7-alpine
    container_name: colony-redis
    command: redis-server --appendonly yes --protected-mode no
    ports:
      - "${COLONY_REDIS_PORT:-6379}:6379"
    volumes:
      - redis-data:/data
    networks:
      - colony-net

volumes:
  colony-shared:
  redis-data:

networks:
  colony-net:
    name: colony-net
    driver: bridge
```

**Notes:**
- `num-cpus=0` on head prevents scheduling tasks on head (coordination only), matching production K8s config
- `shm_size: '4gb'` for Ray object store (production uses 10.24gb but 4gb is sufficient locally)
- `colony-shared` volume provides shared storage between head and workers (replaces EFS)
- Environment variables match what `ConfigurationManager` reads via `json_schema_extra={"env": ...}`
- Head health check uses `ray status` — same as production K8s liveness probe

### 3. `providers/base.py` — Provider Abstraction

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

class ProviderStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"

@dataclass
class ServiceInfo:
    name: str
    status: ProviderStatus
    host: str | None = None
    port: int | None = None
    details: dict[str, str] = field(default_factory=dict)

class DeploymentProvider(ABC):
    """Base class for deployment providers (Compose, K8s, etc.)"""

    @abstractmethod
    async def up(self, build: bool = True, workers: int = 1) -> list[ServiceInfo]:
        """Start all infrastructure. Returns status of each service."""

    @abstractmethod
    async def down(self) -> None:
        """Stop and remove all infrastructure."""

    @abstractmethod
    async def status(self) -> list[ServiceInfo]:
        """Get status of all services."""

    @abstractmethod
    async def run(
        self,
        codebase_path: str,
        config_path: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        """Run polymath.py inside the cluster. Returns exit code."""

    @abstractmethod
    async def doctor(self) -> dict[str, bool]:
        """Check prerequisites. Returns {check_name: passed}."""
```

### 4. `providers/compose.py` — DockerComposeProvider

Core logic for each command:

**`up(build, workers)`:**
1. Resolve path to `docker/docker-compose.yml` (relative to this package)
2. If `build=True`: `docker compose build`
3. `docker compose up -d --scale ray-worker={workers}`
4. Poll health checks: wait for ray-head healthy, redis accepting connections
5. Return service info list

**`down()`:**
1. `docker compose down --volumes --remove-orphans`

**`status()`:**
1. `docker compose ps --format json`
2. Parse container states into `ServiceInfo` objects

**`run(codebase_path, config_path, extra_env, extra_args)`:**
1. Verify cluster is running (check `colony-ray-head` container is healthy)
2. Copy codebase into shared volume: `docker cp {codebase_path}/. colony-ray-head:/mnt/shared/codebase/`
3. If config_path: `docker cp {config_path} colony-ray-head:/mnt/shared/config.yaml`
4. Build `docker exec` command:
   ```
   docker exec -it
     -e ANTHROPIC_API_KEY={from host env}
     -e OPENROUTER_API_KEY={from host env}
     -e HUGGING_FACE_HUB_TOKEN={from host env}
     colony-ray-head
     python -m colony.cli.polymath run /mnt/shared/codebase
       --config /mnt/shared/config.yaml
       {extra_args}
   ```
5. Stream stdout/stderr to terminal in real-time
6. Return exit code

All subprocess calls via `asyncio.create_subprocess_exec`.

### 5. `config.py` — Configuration

```python
@dataclass
class ComposeConfig:
    workers: int = 1
    dashboard_port: int = 8265
    client_port: int = 10001
    redis_port: int = 6379
    shm_size: str = "4gb"
    build: bool = True

@dataclass
class K8sConfig:
    cluster_name: str = "colony-dev"
    min_workers: int = 0
    max_workers: int = 3
    kuberay_version: str = "1.5.1"

@dataclass
class DeployConfig:
    mode: str = "compose"           # "compose" or "k8s"
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    k8s: K8sConfig = field(default_factory=K8sConfig)
    api_key_env_vars: list[str] = field(default_factory=lambda: [
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
    ])
```

### 6. `health.py` — Health Utilities

```python
async def tcp_check(host: str, port: int, timeout: float = 2.0) -> bool: ...
async def redis_ping(host: str, port: int, timeout: float = 2.0) -> bool: ...
async def docker_container_healthy(container_name: str) -> bool: ...
async def wait_until_ready(
    check_fn: Callable[[], Awaitable[bool]],
    timeout: float = 60.0,
    interval: float = 2.0,
    description: str = "",
) -> bool: ...
```

### 7. `manager.py` — DeploymentManager

```python
class DeploymentManager:
    def __init__(self, config: DeployConfig):
        if config.mode == "compose":
            self._provider = DockerComposeProvider(config.compose)
        elif config.mode == "k8s":
            self._provider = KindKubeRayProvider(config.k8s)  # Future
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    async def up(self, **kwargs) -> list[ServiceInfo]: ...
    async def down(self) -> None: ...
    async def status(self) -> list[ServiceInfo]: ...
    async def run(self, **kwargs) -> int: ...
    async def doctor(self) -> dict[str, bool]: ...
```

### 8. `cli.py` — Typer CLI

```
colony-env up [--workers N] [--no-build] [--k8s]
    Build Colony image and start Ray cluster + Redis.
    Default: Docker Compose with 1 worker.

colony-env down [--k8s]
    Stop and remove all containers/resources.

colony-env status
    Show status of all running services.

colony-env run PATH [--config YAML] [-- EXTRA_ARGS...]
    Run polymath.py analysis on a codebase.
    Mounts PATH into the cluster and executes inside ray-head.

colony-env doctor
    Check prerequisites (Docker daemon, docker compose, disk space).
```

### 9. `providers/k8s.py` — Kind+KubeRay Provider (Stub for V1)

Implements `DeploymentProvider` but raises `NotImplementedError` with a message pointing to the GitHub issue for K8s support. The `k8s/` directory contains the YAML templates that will be used when this provider is implemented:

- `kind-config.yaml`: Kind cluster config with extraPortMappings for dashboard (8265) and client (10001)
- `ray-cluster-local.yaml`: Simplified `RayCluster` CRD adapted from the production `deployment/k8s/config/ray-cluster.yaml`, with:
  - `enableInTreeAutoscaling: true`
  - Head: `num-cpus: "0"`, dashboard, client ports
  - Single CPU worker group (no GPU group for local)
  - `imagePullPolicy: IfNotPresent` (loaded via `kind load docker-image`)
  - No EFS, no nodeSelector/tolerations, no ECR
  - Health probes adapted from production (simpler timeouts)

---

## No Changes to Existing Code

1. **`setup_ray()`** — `ray.init(address="auto")` works natively inside the ray-head container because the Ray head process runs in the same container and creates `/tmp/ray/ray_current_cluster`
2. **`polymath.py`** — Runs inside the container with all env vars pre-set
3. **`ConfigurationManager`** — `REDIS_HOST=redis` and `REDIS_PORT=6379` are set in the compose environment, satisfying `RedisConfig`, `StateStorageConfig` via `json_schema_extra={"env": ...}`
4. **`PolymatheraApp(distributed=False)`** — Already skips etcd initialization

---

## Entry Point

Add to `colony/pyproject.toml`:
```toml
[tool.poetry.scripts]
colony-env = "colony.cli.deploy.cli:app"
```

Also add `typer` as a dependency (if not already present):
```toml
typer = {version = "^0.15.0", extras = ["all"]}
```

---

## Usage Flow

```bash
# 1. Start the local cluster (one command)
colony-env up
#   Building colony:local image... done (cached layers: ~30s)
#   Starting redis............... OK  localhost:6379
#   Starting ray-head............ OK  dashboard: http://localhost:8265
#   Starting ray-worker (x1)..... OK
#
#   Ready! Run your analysis:
#     colony-env run /path/to/codebase --config analysis.yaml

# 2. Run an analysis
colony-env run ~/projects/my-repo --config tests/sample_config.yaml

# 3. Scale up workers
colony-env up --workers 3

# 4. Check status
colony-env status
#   ray-head    running   dashboard: http://localhost:8265
#   ray-worker  running   (1 replica)
#   redis       running   localhost:6379

# 5. Tear down
colony-env down
```

---

## Implementation Order

| Step | File | Description |
|------|------|-------------|
| 1 | `deploy/__init__.py` | Package init, exports |
| 2 | `deploy/config.py` | `DeployConfig`, `ComposeConfig`, `K8sConfig` |
| 3 | `deploy/docker/Dockerfile.local` | Colony image adapted from Dockerfile.ray-cpu |
| 4 | `deploy/docker/docker-compose.yml` | ray-head + ray-worker + redis |
| 5 | `deploy/health.py` | `tcp_check`, `redis_ping`, `docker_container_healthy`, `wait_until_ready` |
| 6 | `deploy/providers/__init__.py` | Package init |
| 7 | `deploy/providers/base.py` | `DeploymentProvider` ABC, `ServiceInfo`, `ProviderStatus` |
| 8 | `deploy/providers/compose.py` | `DockerComposeProvider` — full V1 implementation |
| 9 | `deploy/manager.py` | `DeploymentManager` delegates to provider |
| 10 | `deploy/cli.py` | Typer CLI with `up`, `down`, `status`, `run`, `doctor` |
| 11 | `deploy/providers/k8s.py` | `KindKubeRayProvider` — stub with NotImplementedError |
| 12 | `deploy/k8s/kind-config.yaml` | Kind cluster config template |
| 13 | `deploy/k8s/ray-cluster-local.yaml` | Local RayCluster CRD template |
| 14 | `pyproject.toml` | Add `colony-env` script + typer dependency |

---

## Cloud Extension Points (Future)

When K8s mode is implemented:

```
colony-env up --k8s           # Creates Kind cluster, installs KubeRay, deploys RayCluster
colony-env run PATH --k8s     # kubectl exec into head pod
colony-env down --k8s         # Deletes Kind cluster
```

The `KindKubeRayProvider` would:
1. `kind create cluster --config kind-config.yaml`
2. `docker build -t colony:local . && kind load docker-image colony:local`
3. `helm install kuberay-operator kuberay/kuberay-operator`
4. `kubectl apply -f ray-cluster-local.yaml`
5. `kubectl port-forward svc/colony-ray-head-svc 8265:8265 10001:10001`
6. For `run`: `kubectl exec -it {head-pod} -- python -m colony.cli.polymath run ...`

This uses the **same Docker image** as the Compose mode. The only difference is how the cluster is orchestrated.

---

## Verification

1. **Prerequisites**: `colony-env doctor` — checks Docker daemon running, `docker compose` available, sufficient disk space
2. **Build**: `colony-env up` — image builds successfully with all deps (Poetry, faiss-cpu, linguist)
3. **Services**: `colony-env status` — ray-head healthy, ray-worker connected, redis accepting PING
4. **Dashboard**: Open `http://localhost:8265` — shows 1 head + 1 worker, 0 CPUs on head
5. **Integration**: `colony-env run /path/to/test/repo --config test.yaml` — polymath.py connects to Ray, deploys cluster, runs analysis (with Anthropic API key passed through)
6. **Teardown**: `colony-env down` — all containers stopped and removed
7. **Idempotency**: `colony-env up` twice — second call detects running services, skips rebuild
