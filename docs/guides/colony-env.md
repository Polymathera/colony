# colony-env CLI

`colony-env` is Colony's local development tool. It manages a Docker Compose cluster with Ray, Redis, PostgreSQL, Kafka, and the Colony web dashboard.

## Commands

### `colony-env up`

Build the Docker image and start the cluster.

```bash
colony-env up --config my_analysis.yaml              # 1 worker with config
colony-env up --workers 3 --config my_analysis.yaml  # 3 Ray workers with config
colony-env up --no-build --config my_analysis.yaml   # Skip image rebuild
colony-env up                                        # No config (empty cluster)
```

The `--config` flag copies a YAML configuration file into the cluster. The cluster auto-deploys LLM backends, VCM, and the agent system from this config on startup. Without `--config`, the cluster starts with no LLM deployments.

**Services started:**

| Service | Container | Port |
|---------|-----------|------|
| Ray head | `colony-ray-head` | 6379, 8265, 10001 |
| Ray workers | `colony-ray-worker-N` | — |
| Redis | `colony-redis` | 6379 |
| PostgreSQL | `colony-postgres` | 5432 |
| Kafka | `colony-kafka` | 9092 |
| Dashboard | `colony-dashboard` | 8080 |

### `colony-env down`

Stop and remove all containers and volumes.

```bash
colony-env down
```

### `colony-env run`

Submit an analysis job to the running cluster. Tries the dashboard API first, falls back to `docker exec`.

```bash
# Analyze a local codebase
colony-env run --local-repo /path/to/codebase --config analysis.yaml

# With verbose output
colony-env run --origin-url https://github.com/org/repo --config analysis.yaml --verbose
```

For local repos, the codebase is copied to a shared Docker volume and made available to all containers. For most workflows, use the **web dashboard** at [localhost:8080](http://localhost:8080) instead — it provides a richer interface for mapping content, configuring runs, and monitoring agents.

### `colony-env status`

Show running services and their health.

```bash
colony-env status
```

### `colony-env dashboard`

Open the web dashboard in your browser.

```bash
colony-env dashboard              # Opens localhost:8080
colony-env dashboard --port 9090  # Custom port
```

### `colony-env doctor`

Check prerequisites (Docker, Docker Compose, required files).

```bash
colony-env doctor
```

## Typical Workflow

```bash
# Full rebuild + start the always-on cluster with config
colony-env down && colony-env up --workers 3 --config my_analysis.yaml

# Open the dashboard to map content, create sessions, and submit runs
colony-env dashboard
```

## Environment Variables

Pass environment variables to the cluster via a `.env` file or shell exports. Common variables:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for Anthropic Claude |
| `OPENAI_API_KEY` | API key for OpenAI |
| `COLONY_DASHBOARD_UI_PORT` | Dashboard port (default: 8080) |

## Architecture

`colony-env` uses Docker Compose under the hood. The compose file and Dockerfile are bundled inside the `polymathera-colony` package at `polymathera/colony/cli/deploy/docker/`.

The Docker image is based on `rayproject/ray:2.49.0-py311-cpu` and includes all Colony dependencies plus build tools for the web dashboard frontend.
