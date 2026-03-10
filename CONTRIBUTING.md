# Contributing to Colony

Thank you for your interest in contributing to Colony! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/) for dependency management
- Docker (for running `colony-env` local clusters)

### Development Setup

```bash
git clone https://github.com/polymathera/colony.git
cd colony
poetry install --all-extras
```

### Running the Local Cluster

```bash
colony-env up --workers 3
colony-env run --local-repo /path/to/codebase --config my_analysis.yaml
colony-env down
```

### Running Tests

```bash
poetry run pytest
```

## Project Structure

```
colony/
├── src/polymathera/colony/       # Main package (polymathera.colony namespace)
│   ├── agents/                   # Agent system
│   │   ├── base.py               # Agent base class
│   │   ├── models.py             # Core models
│   │   ├── patterns/             # Agent patterns (actions, memory, planning, etc.)
│   │   ├── blackboard/           # Blackboard pattern
│   │   └── sessions/             # Session management
│   ├── cli/                      # CLI tools (colony-env)
│   ├── cluster/                  # LLM cluster management
│   ├── distributed/              # Ray + Redis utilities
│   ├── vcm/                      # Virtual Context Memory
│   ├── web_ui/                   # Web dashboard (FastAPI + React)
│   └── samples/                  # Example analysis configurations
├── docs/                         # Documentation (MkDocs)
├── pyproject.toml                # Package configuration
└── README.md
```

## Design Principles

Colony has strong opinions about code quality. Please read these carefully before contributing.

### 1. Surgical Changes Only

Make the minimum change needed to solve the problem. Do not:
- Rename parameters or variables unless directly required
- Reformat code you didn't functionally change
- Add comments to code you didn't write
- "Improve" surrounding code while fixing a bug

### 2. Fix Root Causes, Not Symptoms

Don't add workarounds or band-aids. Understand the full context, find the root cause, and fix it properly.

### 3. Encapsulation Discipline

Never directly manipulate internal state that belongs to another component. If a class owns state, go through its API. For example:
- `plan_step()` must not touch replanning policy internals
- `Agent.run_step()` must not touch action policy internals
- If you need to reset state owned by component X, add a method to X

### 4. Policy-Based Design

Every cognitive process is a pluggable policy with well-defined interfaces. When adding new behavior:
- Define a protocol (not an abstract class)
- Provide a default implementation
- Allow users to substitute their own

### 5. Fail Fast with Clear Messages

Data must be typed correctly. Objects must have expected attributes. If something is wrong, raise an error immediately with a clear message. Do not let bad data propagate through the system.

### 6. No Code Bloat

Every line must have a purpose. Avoid:
- Over-engineering for hypothetical future requirements
- Adding configuration for things that should just be code
- Creating abstractions for one-time operations

### 7. Understand Before Changing

Before modifying any code, read enough of the codebase to understand the full context. Do not guess what you can know by reading. Trace the full path from user-facing entry point down to the code you're changing.

## How to Contribute

### Reporting Issues

- Use GitHub Issues
- Include: what you expected, what happened, steps to reproduce
- Include Colony version, Python version, and OS
- For errors, include the full traceback

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes following the design principles above
4. Write tests for new functionality
5. Ensure all tests pass: `poetry run pytest`
6. Submit a PR with a clear description of what and why

### PR Guidelines

- Keep PRs focused on a single concern
- Write a clear title (under 70 characters)
- Describe the "why" in the PR body, not just the "what"
- Reference related issues

### Adding New Components

**New AgentCapability:**
- Implement the `AgentCapability` protocol
- Export `@action_executor` methods for LLM-plannable actions
- Use `@hookable` for methods that should be extensible
- Add to appropriate `__init__.py` exports

**New ActionPolicy:**
- Extend `BaseActionPolicy`
- Implement `plan_step()` and any required hooks
- Document the policy's decision-making strategy

**New Storage Backend:**
- Implement the `StorageBackend` protocol
- Provide a factory class for scope-based creation
- Handle graceful degradation if dependencies are unavailable

## Optional Dependencies

When adding functionality that requires new packages:

1. Add the dependency as optional in `pyproject.toml`:
   ```toml
   new-dep = { version = "^1.0", optional = true }
   ```

2. Add to an extras group (or create a new one):
   ```toml
   [tool.poetry.extras]
   feature_name = ["new-dep"]
   ```

3. Guard the import in code:
   ```python
   try:
       import new_dep
   except ImportError:
       new_dep = None
   ```

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## License

By contributing to Colony, you agree that your contributions will be licensed under the Apache License 2.0.
