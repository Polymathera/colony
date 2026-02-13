# Colony

Polymathera's no-RAG, multi-agent framework for extremely long, *dense* contexts (1B+ tokens). It provides:

- A cluster-level **virtual context memory** with user-defined context paging.
- Cache-aware agent action policies.
- Powerful and composable multi-agent patterns.
- Arbitrarily sophisticated memory hierarchies and cognitive processes.

## Quick Start

### Installation

```bash
pip install colony
```

With optional extras:

```bash
pip install colony[code_analysis]    # Code analysis tools
pip install colony[gpu]              # GPU inference (vLLM, PyTorch)
pip install colony[cpu]              # CPU-only inference (Anthropic API)
pip install colony --all-extras      # Everything
```

### Local Test Environment

Colony ships with `colony-env`, a CLI tool that spins up a local Ray cluster + Redis using Docker Compose. The only prerequisite is **Docker**.

```bash
# Start the cluster (builds image on first run)
colony-env up

# Generate a sample analysis config
polymath init-config --output my_analysis.yaml

# Run a code analysis over a local codebase
colony-env run /path/to/codebase --config my_analysis.yaml

# Check service status
colony-env status

# Scale workers
colony-env up --workers 3

# Tear down
colony-env down

# Verify prerequisites
colony-env doctor
```

All Colony dependencies run inside Docker — no local GPU drivers, Ray, or Redis installation required. The `colony-env run` command copies your codebase to be analyzed into the cluster and executes inside the Ray head container with full access to the framework.

**Services started by `colony-env up`:**

| Service | Port | Description |
|---------|------|-------------|
| Ray dashboard | `localhost:8265` | Cluster monitoring UI |
| Ray client | `localhost:10001` | Ray client connection |
| Redis | `localhost:6379` | State management backend |

## Development

```bash
git clone https://github.com/polymathera/colony.git
cd colony
poetry install --all-extras
```

### Optional Dependencies

Dependencies that require system libraries (CUDA, native extensions) are declared as optional extras in `pyproject.toml`:

```toml
[tool.poetry.dependencies]
some-dep = { version = "^1.0", optional = true }

[tool.poetry.extras]
feature_name = ["some-dep"]
```

Guard optional imports in code:

```python
try:
    import heavy_dep
except ImportError:
    heavy_dep = None

def feature_function():
    if heavy_dep is None:
        raise ImportError(
            "Install with: pip install colony[feature_name]"
        )
```
