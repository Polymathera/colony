# Colony Open-Source Release Plan

## Overview

Prepare the Colony framework for public release: packaging for PyPI, community guidelines, and comprehensive documentation that incorporates the rich design philosophy from the internal docs.

---

## Phase 0: Namespace Package + src Layout Refactor

**Decision**: Adopt `polymathera.colony` namespace package with `src/` layout per [packaging-strategy.md](python/colony/cli/deploy/packaging-strategy.md).

### 0.1 Directory Restructure

```
# Before                          # After
colony/                            colony/
├── python/                        ├── src/
│   └── colony/                    │   └── polymathera/          # NO __init__.py (PEP 420)
│       ├── __init__.py            │       └── colony/
│       ├── agents/                │           ├── __init__.py
│       ├── cli/                   │           ├── agents/
│       └── ...                    │           ├── cli/
                                   │           └── ...
```

- [ ] `mkdir -p src/polymathera`
- [ ] `mv python/colony src/polymathera/`
- [ ] `rmdir python`
- [ ] Verify no `__init__.py` in `src/polymathera/` (PEP 420 implicit namespace)

### 0.2 Import Rename (~48 files)

All Python imports change:
```python
# Before                              # After
from colony.agents import ...         from polymathera.colony.agents import ...
import colony.distributed...          import polymathera.colony.distributed...
```

Scope:
- 48 files with `from colony.` imports
- 3 files with `import colony` statements
- String module paths in `polymath.py` (~30 references like `"colony.samples.code_analysis..."`)
- `docker-compose.yml`: `python -m colony.web_ui.backend.main` → `python -m polymathera.colony.web_ui.backend.main`

### 0.3 Config File Updates

- [ ] `pyproject.toml`:
  - `name = "polymathera-colony"`
  - `packages = [{include = "polymathera", from = "src"}]`
  - `colony-env = "polymathera.colony.cli.deploy.cli:app"`
  - Fix license: `"Apache-2.0 license"` → `"Apache-2.0"`
- [ ] `Dockerfile.local`:
  - `RUN cd src/polymathera/colony/web_ui/frontend && ...`
  - `ENV PYTHONPATH=${APP_MOUNT_PATH}/src`
- [ ] `docker-compose.yml`: `python -m polymathera.colony.web_ui.backend.main`
- [ ] `README.md`: `cd src/polymathera/colony/web_ui/frontend`

---

## Phase 1: Packaging & Distribution

### 1.1 `pyproject.toml` Metadata

- [ ] Add metadata fields for PyPI:
  ```toml
  homepage = "https://github.com/polymathera/colony"
  repository = "https://github.com/polymathera/colony"
  documentation = "https://polymathera.github.io/colony"
  keywords = ["multi-agent", "llm", "context", "cache-aware", "agents", "no-rag"]
  classifiers = [
      "Development Status :: 3 - Alpha",
      "Intended Audience :: Developers",
      "License :: OSI Approved :: Apache Software License",
      "Programming Language :: Python :: 3.11",
      "Programming Language :: Python :: 3.12",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
  ]
  ```
- [ ] Add `exclude` patterns for test files, dev configs, docker files, node_modules
- [ ] Consider adding a `colony` CLI entry point (currently only `colony-env`)

### 1.2 Package Structure Verification

- [ ] Ensure all `__init__.py` files exist and export correctly
- [ ] Verify lazy imports in `colony/__init__.py` work for clean `import colony`
- [ ] Test `poetry build` produces a clean wheel/sdist
- [ ] Test `pip install` from the built wheel works in a clean venv
- [ ] Ensure optional extras install correctly (`pip install colony[cpu]`, etc.)

### 1.3 Pre-Release Hygiene

- [ ] Remove/ignore internal plan files (`.md` plans in code dirs)
- [ ] Check for hardcoded paths (e.g., `/mnt/shared/`, `/home/ray/app`)
- [ ] Check for leaked credentials, API keys, internal URLs
- [ ] Add `py.typed` marker for PEP 561 type checking support
- [ ] Review `.gitignore` — ensure build artifacts, editor configs excluded

### 1.4 PyPI Publishing

- [ ] Set up GitHub Actions workflow for automated publishing on tag
- [ ] Test publish to TestPyPI first
- [ ] Reserve the `colony` package name on PyPI (if available, otherwise `colony-agents` or `polymathera-colony`)

---

## Phase 2: Community Guidelines

### 2.1 CODE_OF_CONDUCT.md

- [ ] Adopt Contributor Covenant v2.1 (standard, widely recognized)
- [ ] Customize contact method (email or GitHub discussions)

### 2.2 CONTRIBUTING.md

- [ ] Development setup instructions (Poetry, Python 3.11+)
- [ ] Code style and conventions (from `.CLAUDE.md` — encapsulation discipline, no cosmetic changes, policy-based design)
- [ ] Testing guidelines (pytest, how to run the test suite)
- [ ] PR process and review expectations
- [ ] Architecture overview for new contributors (pointer to docs)
- [ ] How to add new AgentCapabilities, ActionPolicies, storage backends
- [ ] Issue reporting guidelines

### 2.3 SECURITY.md

- [ ] Responsible disclosure policy
- [ ] Contact for security issues

### 2.4 GitHub Repository Setup

- [ ] Issue templates (bug report, feature request, question)
- [ ] PR template
- [ ] GitHub Actions CI (lint, test, type check)
- [ ] Branch protection rules recommendation
- [ ] Labels for issues (bug, enhancement, documentation, good-first-issue, etc.)

### 2.5 LICENSE

- [x] Already present: Apache-2.0

---

## Phase 3: Documentation

### 3.1 Framework Choice: MkDocs Material

**Decision**: MkDocs with Material theme.

**Rationale**:
- Python-native (aligns with project)
- Material theme is polished and widely used
- Supports admonitions, code tabs, search, versioning
- Easy to host on GitHub Pages
- `.gitignore` already has `/site` placeholder for mkdocs
- `mkdocs-autorefs` + `mkdocstrings[python]` for API docs from docstrings
- Mermaid diagrams via `mkdocs-mermaid2-plugin` for architecture diagrams

### 3.2 Documentation Structure

```
docs/
├── index.md                          # Landing page
├── getting-started/
│   ├── installation.md               # pip install, extras, Docker
│   ├── quickstart.md                 # colony-env, first analysis
│   └── concepts.md                   # Key concepts overview
├── philosophy/
│   ├── index.md                      # Why Colony exists
│   ├── no-rag.md                     # The NoRAG paradigm
│   ├── agents-all-the-way-down.md    # Intelligence from composition
│   ├── cache-awareness.md            # Cache-aware multi-agent patterns
│   └── consciousness-intuition.md    # The consciousness-intuition interface
├── architecture/
│   ├── index.md                      # High-level architecture
│   ├── virtual-context-memory.md     # VCM: virtual memory for LLMs
│   ├── agent-system.md               # Agent types, lifecycle, state
│   ├── blackboard.md                 # Blackboard pattern & backends
│   ├── memory-system.md              # Memory hierarchy & capabilities
│   ├── action-policies.md            # Policy-based action selection
│   ├── planning.md                   # LLM-centric planning, MPC
│   ├── hook-system.md                # AOP-inspired hooks
│   ├── game-engine.md                # Multi-agent game-theoretic protocols
│   └── distributed.md                # Ray, Redis, deployment
├── guides/
│   ├── custom-capabilities.md        # Building AgentCapabilities
│   ├── custom-policies.md            # Building ActionPolicies
│   ├── memory-configuration.md       # Configuring memory hierarchies
│   ├── code-analysis.md              # Using the code analysis domain
│   ├── colony-env.md                 # Local development with colony-env
│   └── web-dashboard.md              # Using the web dashboard
├── design-insights/
│   ├── index.md                      # Why this section exists
│   ├── capabilities-as-aspects.md    # AOP analogy for capabilities
│   ├── memory-as-observer.md         # Bidirectional observer pattern
│   ├── game-theoretic-correctness.md # Games as correctness mechanisms
│   ├── page-graphs.md               # Page graphs as fundamental DS
│   ├── abstraction-patterns.md       # 7 core patterns from code analysis
│   └── qualitative-analysis.md       # LLM-driven qualitative reasoning
├── reference/
│   ├── api/                          # Auto-generated API reference
│   ├── configuration.md              # YAML config reference
│   └── cli.md                        # colony-env CLI reference
└── contributing/
    ├── index.md                      # Link to CONTRIBUTING.md
    ├── development-setup.md          # Dev environment setup
    └── design-principles.md          # Principles for contributors
```

### 3.3 Content Plan — Key Pages

#### 3.3.1 Philosophy Section (from PHILOSOPHY.md + SPECS_AGENTS.md)

The most important section for evangelizing Colony's ideas:

**no-rag.md** — Core thesis:
- Explicit (live) context > implicit context
- Why CoT plateaus (cannot reproduce all necessary implicit context)
- Why not RNNs/SSMs (irreversible forgetting)
- Deep research as a game: moves = combinations of facts offering smallest leap to new insights
- Whole context must remain live, not filtered through retrieval
- Source: PHILOSOPHY.md sections on "Explicit Context > Implicit Context"

**agents-all-the-way-down.md** — Composition thesis:
- "General intelligence is emergent from the right composition of LLM-based reasoning agents"
- Iterative deepening of finite-depth reasoning → unbounded depth
- Distributed reasoning over ELC → unbounded context
- Software complexity O(log N) from right abstractions
- The "virtual agent" concept — multi-agent system implementing different cognition levels
- Source: PHILOSOPHY.md core belief + complexity analysis

**cache-awareness.md** — The key differentiator:
- Cache awareness is NOT a property of primitives — it's emergent from the LLM planner composing primitives
- Cache misses dominate execution time for large contexts
- Working sets as resources: pages allocated/coordinated across agents
- Amortized cost: O(N²) → O(N log N) as page graph stabilizes
- Source: CACHE_AWARE_PLANNING.md + PHILOSOPHY.md

**consciousness-intuition.md** — The cognitive model:
- "Intuition" layer = LLM; "Consciousness" layer = cognitive processes/policies
- Subconscious vs conscious processes (capabilities export action_executors for conscious; background hooks for subconscious)
- Policy-based design: every cognitive process is a pluggable policy
- Source: PHILOSOPHY.md consciousness-intuition interface

#### 3.3.2 Architecture Section

**virtual-context-memory.md** (from SPECS.md, SPECS_VCM.md):
- OS virtual memory analogy: page tables, page faults, cache-aware scheduling
- Extended VCM = immutable VCM + read-write blackboard
- Cluster-level memory management (vs node-level vLLM)
- Page groups, agent-page affinity (soft/hard)
- VirtualContextPage as generic abstraction (not git-specific)

**memory-system.md** (from MEMORY_SYSTEM.md, UNIFIED_MEMORY_CAPABILITY.md, MEMORY_MAP.md):
- Unified storage principle: ALL state in blackboards
- Memory hierarchy as dataflow graph of abstraction levels
- Memory levels: sensory → working → STM → LTM (episodic/semantic/procedural)
- Memory scopes: agent-private, capability-scoped, task-scoped, collective, global
- Lens semantics: read-only views with custom filtering/ranking
- Ingestion → Storage → Retrieval → Maintenance pipeline
- Key insight: "An agent should be able to reason *about* (not just *with*) its own knowledge"

**game-engine.md** (from MULTI_AGENT_GAME_ENGINE.md):
- Four game types: hypothesis, bidding/contract, negotiation, consensus
- Roles: Proposer, Skeptic, Grounder, Arbiter, Planner
- ACL: messages have illocutionary force, not just string content
- Failure mode mapping: hallucination → evidence requirements, laziness → contract net, goal drift → objective guards
- Advanced: no-regret learning, VCG incentives, social choice, epistemic logic
- Hybrid architecture: deliberative LLM core + reactive rules

**action-policies.md** (from DATAFLOW.md, LLM_CENTRIC_PLANNING.md):
- LLM is the planner, not the framework
- Two-phase action selection: choose action → parameterize
- Model-Predictive Control: execute partial plan, re-evaluate, adapt
- ActionPolicy I/O contract
- Cache-aware planning context (working_set, access_patterns, prefetch hints)

#### 3.3.3 Design Insights Section (unique evangelism content)

**capabilities-as-aspects.md**:
- AOP analogy: each AgentCapability is an "aspect"
- ActionPolicy is the "aspect weaver" deciding which aspects activate
- Emergent local behaviors from combinatorial explosion of interleavings
- No explicit modeling of all behavior paths — emergence from composition
- Source: PHILOSOPHY.md AOP section

**game-theoretic-correctness.md**:
- Games as correctness mechanisms (not just coordination tools)
- Mapping LLM failure modes to game-theoretic solutions
- VCG-style incentives: reward agents for marginal contribution to global performance
- No-regret algorithms (Exp3/EXP4) to adjust agent/strategy mixtures
- Social choice theory for aggregating evaluator rankings
- Source: MULTI_AGENT_GAME_ENGINE.md advanced mechanisms

**abstraction-patterns.md** (from CODE_ANALYSIS_ABSTRACTION_PATTERNS):
- 7 core patterns distilled from 30+ analysis strategies
- Generalizability: these patterns work for any domain with partial knowledge + discovered relationships
- ScopeAwareResult, MergePolicy, Query-Driven Context Discovery
- Low-confidence stories trigger refinement, not action
- Source: CODE_ANALYSIS_ABSTRACTION_PATTERNS_*.md

### 3.4 Tooling Setup

- [ ] `mkdocs.yml` configuration with Material theme
- [ ] Add dev dependencies: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-mermaid2-plugin`
- [ ] GitHub Pages deployment via GitHub Actions
- [ ] Add `docs/` to `.gitignore` for `site/` output
- [ ] Script to build docs: `mkdocs build` / `mkdocs serve`

### 3.5 API Reference

- [ ] Use `mkdocstrings[python]` to auto-generate from docstrings
- [ ] Key modules to document:
  - `colony.agents.base` — Agent class
  - `colony.agents.models` — Core models (AgentState, AgentMetadata, etc.)
  - `colony.agents.patterns.actions.policies` — ActionPolicy classes
  - `colony.agents.patterns.memory` — Memory system
  - `colony.agents.patterns.capabilities` — Built-in capabilities
  - `colony.agents.blackboard` — Blackboard pattern
  - `colony.vcm` — Virtual Context Memory
  - `colony.cli.deploy` — colony-env CLI

---

## Phase 4: README Refresh

- [ ] Update README with:
  - Badges (PyPI version, license, Python version, CI status)
  - More compelling intro (incorporate philosophy)
  - Link to full documentation site
  - Brief architecture diagram (Mermaid)
  - Feature highlights with links to relevant docs
  - Contributing section (pointer to CONTRIBUTING.md)

---

## Execution Order

1. **Packaging** (Phase 1) — prerequisite for everything else
2. **Community guidelines** (Phase 2) — quick wins, important for first impression
3. **Documentation framework** (Phase 3.4) — set up MkDocs skeleton
4. **Core documentation** (Phase 3.3) — write content, starting with philosophy & architecture
5. **API reference** (Phase 3.5) — auto-generate from docstrings
6. **README refresh** (Phase 4) — final polish
7. **Publish** (Phase 1.4) — PyPI release

---

## Source Material Mapping

| Doc Page | Primary Source(s) |
|----------|-------------------|
| no-rag.md | PHILOSOPHY.md |
| agents-all-the-way-down.md | PHILOSOPHY.md |
| cache-awareness.md | CACHE_AWARE_PLANNING.md, PHILOSOPHY.md |
| consciousness-intuition.md | PHILOSOPHY.md |
| virtual-context-memory.md | SPECS.md, SPECS_VCM.md |
| agent-system.md | SPECS_AGENTS.md, AGENT_FRAMEWORK.md |
| blackboard.md | BLACKBOARD_DESIGN.md |
| memory-system.md | MEMORY_SYSTEM.md, UNIFIED_MEMORY_CAPABILITY.md, MEMORY_MAP.md |
| action-policies.md | DATAFLOW.md, LLM_CENTRIC_PLANNING.md |
| planning.md | LLM_CENTRIC_PLANNING.md, HIERARCHICAL_PLANNING_DESIGN.md |
| hook-system.md | HOOK_SYSTEM.md |
| game-engine.md | MULTI_AGENT_GAME_ENGINE.md |
| capabilities-as-aspects.md | PHILOSOPHY.md |
| game-theoretic-correctness.md | MULTI_AGENT_GAME_ENGINE.md |
| abstraction-patterns.md | CODE_ANALYSIS_ABSTRACTION_PATTERNS_*.md |
