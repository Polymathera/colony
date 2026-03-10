# Quick Start

This guide walks you through running a Colony agent system on your local machine using `colony-env`.

## Prerequisites

- `polymathera-colony` installed (`pip install polymathera-colony`)
- Docker running on your machine

## Start the Cluster

```bash
# Start a local Ray cluster with 3 workers + Redis
colony-env up --workers 3
```

This builds a Docker image (first run takes a few minutes) and starts:

| Service | Port | Description |
|---------|------|-------------|
| Colony dashboard | `localhost:8080` | Web UI for agents, sessions, VCM |
| Ray dashboard | `localhost:8265` | Cluster monitoring |
| Ray client | `localhost:10001` | Ray client connection |
| Redis | `localhost:6379` | State backend |

## Run an Analysis

```bash
# Run a code analysis over a local codebase
colony-env run --local-repo /path/to/your/codebase --config my_analysis.yaml --verbose
```

## Open the Dashboard

```bash
colony-env dashboard
```

The web dashboard shows:

- **Overview**: cluster health, deployments, stats
- **Agents**: registered agents, their state and capabilities
- **Sessions**: browsable session history with token usage
- **VCM**: page table and virtual context statistics

## Check Status

```bash
colony-env status
```

## Tear Down

```bash
colony-env down
```

## Next Steps

- Read [Key Concepts](concepts.md) to understand Colony's architecture
- Explore the [Philosophy](../philosophy/index.md) to understand why Colony exists
- Check the [Architecture](../architecture/index.md) for technical details
