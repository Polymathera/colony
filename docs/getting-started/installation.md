# Installation

## Requirements

- Python 3.11 or 3.12
- Docker (for `colony-env` local clusters)

## Install from PyPI

```bash
pip install polymathera-colony
```

### Optional Extras

Colony uses optional dependency groups for features that require heavy or platform-specific packages:

```bash
# Code analysis tools (tree-sitter, python-magic, etc.)
pip install polymathera-colony[code_analysis]

# GPU inference (vLLM, PyTorch with CUDA)
pip install polymathera-colony[gpu]

# CPU-only inference (Anthropic API, OpenAI API, ChromaDB)
pip install polymathera-colony[cpu]

# Web dashboard (FastAPI, Uvicorn)
pip install polymathera-colony[dashboard]

# Observability (Kafka-based span streaming)
pip install polymathera-colony[observability]

# Everything
pip install polymathera-colony --all-extras
```

## Development Installation

```bash
git clone https://github.com/polymathera/colony.git
cd colony
poetry install --all-extras
```

## Verify Installation

After installing, the `colony-env` CLI should be available:

```bash
colony-env --help
```

Check that Docker is available for local clusters:

```bash
colony-env doctor
```

## Project Layout

Colony uses a namespace package structure:

```
colony/
├── src/
│   └── polymathera/              # PEP 420 implicit namespace (no __init__.py)
│       └── colony/               # Main package
│           ├── agents/           # Agent system
│           ├── cli/              # CLI tools
│           ├── cluster/          # LLM cluster management
│           ├── distributed/      # Ray + Redis utilities
│           ├── vcm/              # Virtual Context Memory
│           └── web_ui/           # Web dashboard
├── docs/                         # This documentation
├── pyproject.toml                # Package config (Poetry)
└── mkdocs.yml                    # Docs config
```

All imports use the `polymathera.colony` namespace:

```python
from polymathera.colony.agents import Agent, AgentHandle
```
