# Colony

[![PyPI](https://img.shields.io/pypi/v/polymathera-colony)](https://pypi.org/project/polymathera-colony/)
[![Python](https://img.shields.io/pypi/pyversions/polymathera-colony)](https://pypi.org/project/polymathera-colony/)
[![License](https://img.shields.io/github/license/polymathera/colony)](LICENSE)
[![CI](https://github.com/polymathera/colony/actions/workflows/ci.yml/badge.svg)](https://github.com/polymathera/colony/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-polymathera.github.io%2Fcolony-blue)](https://polymathera.github.io/colony)

**A no-RAG, cache-aware multi-agent framework for extremely long, dense contexts (1B+ tokens).**


Colony is a framework for building *tightly-coupled, self-improving, self-aware multi-agent systems* (***agent colonies***) that reason over extremely long context without retrieval-augmented generation (RAG). Instead of fragmenting context into chunks and retrieving snippets, Colony keeps the entire context *live* across a **cluster of LLMs** through a virtual memory system that manages GPU KV caches the same way an operating system manages (almost unlimited) virtual memory over finite physical memory.

!!! tip "Colony's Vision"
    Colony's goal is to be the most efficient *country of geniuses in a datacenter* — the ideal substrate for **civilization-building AI**.


!!! tip "Pre-Alpha Early Access"

    Colony is still in pre-alpha early access. The API is not stable and the framework is under active development. We welcome feedback and contributions, but be aware that breaking changes may occur.


!!! tip "Who should use Colony?"

    Colony is designed for **engineers building complex multi-agent systems** that require reasoning over extremely long contexts. It is not a general-purpose agent framework or a consumer product. If you are looking for a simple agent orchestration tool or a way to add tool use to an LLM, Colony may not be the right fit. It runs over a Ray cluster (local or in the cloud) and it can be resource-intensive and expensive.

## Why Colony?

Most agent frameworks treat context as something to retrieve or manage. Colony treats it as something to be *brought to life*. Certain domains require *reasoning deep and wide*. Examples include:
- *Scientific research*: synthesizing novel insights from a vast literature requires complex integration
- *Cyber-physical systems*: understanding the full context of a complex system (code, physical environment, requirements, regulations) is essential for architecting solutions and identifying edge cases and failure modes
- *Systemic vulnerability analysis*: identifying security risks in a complex system by reasoning over a large attack surface and many potential interactions.
- *Business intelligence*: making strategic decisions based on a wide range of internal and external data, where relevant information may be siloed and require cross-domain reasoning
- *Economic modeling*: simulating and understanding complex economic systems with many interacting agents and factors and long supply chains
- *Long-form content creation*: writing a book or comprehensive report that requires maintaining a coherent narrative across a large amount of information

Colony's core innovations are:
- **NoRAG** -- Colony keeps the full context live and accessible, not filtered through retrieval. Colony manages all kinds of context (code, text, data) through distributed KV cache paging, not vector search.

- **Cache-Aware Agents** -- Agents are aware of what's in GPU memory (at the cluster level) and consciously plan their work to maximize cache reuse.

- **Agents All the Way Down** -- General intelligence emerges from the right composition of *agent capabilities* and *multi-agent patterns*. Every cognitive process -- attention, memory, planning, confidence tracking -- is a pluggable policy with a default implementation.

- **Game-Theoretic Correctness** -- Multi-agent game protocols (hypothesis games, contract nets, negotiation) combat specific LLM failure modes: hallucination, laziness, and goal drift.

Read the full [Philosophy](https://polymathera.github.io/colony/philosophy/) for the ideas behind the framework.


> P.S. Colony does not preclude agents from using retrieval or vector search -- those can be implemented as capabilities that agents use when appropriate. Colony's point is that retrieval is not the only way to manage long context, and for certain domains, it's not the best way.


## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                               Agent Colony                                     │
│                                                                                │
│   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                 │
│   │    Agent 1     │   │    Agent 2     │   │    Agent N     │                 │
│   │ Capabilities   │   │ Capabilities   │   │ Capabilities   │   ...           │
│   │ Action Policy  │   │ Action Policy  │   │ Action Policy  │                 │
│   │ Planner (LLM)  │   │ Planner (LLM)  │   │ Planner (LLM)  │                 │
│   └──────┬─────────┘   └──────┬─────────┘   └──────┬─────────┘                 │
│          │ read/write         │                    │ infer_with_suffix         │
│     ┌────┴────────────────────┴────────────────────┴──────┐                    │
│     ▼                                                     ▼                    │
│  ┌────────────────────────────┐   ┌────────────────────────────────────────┐   │
│  │    Blackboard (Redis)      │   │     Virtual Context Memory (VCM)       │   │
│  │                            │   │                                        │   │
│  │  Shared state & events     │   │  Page Table · Page Graph               │   │
│  │  OCC · Memory scopes       │   │  Cache Scheduling · Page Faults        │   │
│  │  Agent coordination        │   │                                        │   │
│  └─────────────┬──────────────┘   │  ┌──────────┐ ┌──────────┐             │   │
│                │                  │  │ LLM N1   │ │ LLM N2   │   ...       │   │
│                │ mmap             │  │ KV Cache │ │ KV Cache │             │   │
│                └─────────────────►│  └──────────┘ └──────────┘             │   │
│                                   │                                        │   │
│  ┌────────────────────────────┐   │  Context Sources (mapped as pages):    │   │
│  │    External Sources        │   │  ┌────────┐ ┌──────────┐ ┌─────────┐   │   │
│  │  Git repos, documents,     │   │  │ Repos  │ │Knowledge │ │Blackbrd │   │   │
│  │  knowledge bases, data     ├──►│  │        │ │  Bases   │ │  Data   │   │   │
│  └────────────────────────────┘   │  └────────┘ └──────────┘ └─────────┘   │   │
│                                   └────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
```

Each **Agent** composes pluggable [capabilities](src/polymathera/colony/agents/patterns/capabilities) (memory, attention, games, confidence tracking, grounding, reflection, cache awareness, etc.) coordinated by an **`ActionPolicy`** that consults an LLM **Planner**. Agents share state through a Redis-backed **`Blackboard`** with optimistic concurrency control (OCC) and causal ordering. The **Virtual Context Memory** (VCM) manages distributed GPU KV caches as pages, enabling agents to reason over contexts far larger than any single model's window.

See the full [Architecture docs](https://polymathera.github.io/colony/architecture/).

## Quick Start

### Installation

```bash
pip install polymathera-colony
```

With optional extras:

```bash
pip install polymathera-colony[code_analysis]    # Code analysis tools
pip install polymathera-colony[gpu]              # GPU inference (vLLM, PyTorch)
pip install polymathera-colony[cpu]              # CPU-only inference (Anthropic API)
pip install polymathera-colony --all-extras      # Everything
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

# Open the web dashboard
colony-env dashboard

# Scale workers
colony-env up --workers 3

# Tear down
colony-env down

# Verify prerequisites
colony-env doctor
```

All Colony dependencies run inside Docker -- no local GPU drivers, Ray, or Redis installation required. The `colony-env run` command copies your codebase to be analyzed into the cluster and executes inside the Ray head container with full access to the framework.

**Services started by `colony-env up`:**

| Service | Port | Description |
|---------|------|-------------|
| Colony dashboard | `localhost:8080` | Web UI for agents, sessions, VCM |
| Ray dashboard | `localhost:8265` | Cluster monitoring UI |
| Ray client | `localhost:10001` | Ray client connection |
| Redis | `localhost:6379` | State management backend |

### Web Dashboard

The Colony dashboard starts automatically with `colony-env up` at [localhost:8080](http://localhost:8080). It provides:

- **Overview** — cluster health, application deployments, quick stats
- **Agents** — list registered agents, view state, capabilities, and details
- **Sessions** — browse sessions and their agent runs with token usage
- **VCM** — page table, working set, and virtual context statistics
- **Traces** — detailed tracing of agent actions, VCM operations, and system events for debugging and performance analysis

```bash
# Run the agent colony
colony-env down && colony-env up --workers 3 && colony-env run --local-repo /path/to/codebase --config my_analysis.yaml --verbose

# Open the dashboard in your browser
colony-env dashboard

# Use a custom port (must match COLONY_DASHBOARD_UI_PORT)
colony-env dashboard --port 9090
```

For frontend development, run the Vite dev server on the host with hot-reload:

```bash
cd src/polymathera/colony/web_ui/frontend
npm install
npm run dev     # Starts on localhost:5173, proxies /api to localhost:8080
```

## Key Features

| Feature | Description | Docs |
|---------|-------------|------|
| Virtual Context Memory | OS-style virtual memory for LLM KV caches with page tables and cache-aware scheduling | [VCM](https://polymathera.github.io/colony/architecture/virtual-context-memory/) |
| Agent Capabilities | Composable cognitive modules (memory, attention, games, confidence) attached to agents via AOP-inspired patterns | [Agent System](https://polymathera.github.io/colony/architecture/agent-system/) |
| Action Policies | LLM-centric planning with Model Predictive Control -- the LLM is the planner, not the framework | [Action Policies](https://polymathera.github.io/colony/architecture/action-policies/) |
| Blackboard | Redis-backed shared state with optimistic concurrency, causal timelines, and event-driven coordination | [Blackboard](https://polymathera.github.io/colony/architecture/blackboard/) |
| Memory Hierarchies | Unified memory system with sensory, working, short-term, and long-term memory -- all backed by blackboards | [Memory](https://polymathera.github.io/colony/architecture/memory-system/) |
| Game Engine | Hypothesis games, contract nets, negotiation, and consensus protocols for multi-agent coordination | [Games](https://polymathera.github.io/colony/architecture/game-engine/) |
| Hook System | AOP-inspired hooks for cross-cutting concerns (logging, metrics, memory triggers) | [Hooks](https://polymathera.github.io/colony/architecture/hook-system/) |

## Development

```bash
git clone https://github.com/polymathera/colony.git
cd colony
poetry install --all-extras
```

### Running Tests

```bash
pytest src/ --timeout=120 -x -q
```

### Documentation

```bash
poetry run mkdocs serve     # Local docs server at http://127.0.0.1:8000/
poetry run mkdocs build     # Build static site
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code conventions, and the PR process.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
