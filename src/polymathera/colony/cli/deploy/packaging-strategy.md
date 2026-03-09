# Colony Packaging Strategy: PyPI Naming & Namespace

## Problem

1. `poetry install` fails — Poetry can't find the package because code lives in `python/colony/` but Poetry expects it at `colony/` (sibling to `pyproject.toml`). Needs a `packages` directive.
2. `colony` is already taken on PyPI — need a unique name.
3. Want consistent naming across PyPI and imports.

## Option A: Different PyPI name, keep `import colony`

```
PyPI:   pip install polymathera-colony
Import: import colony
```

Common pattern — `Pillow -> import PIL`, `scikit-learn -> import sklearn`, `beautifulsoup4 -> import bs4`.
Minimal changes: just rename in `pyproject.toml` and add the `packages` directive. No code changes.

**Downside**: If a user happens to have the other `colony` package installed, there's a namespace collision.

## Option B: Namespace package `polymathera.colony`

```
PyPI:   pip install polymathera-colony
Import: from polymathera.colony import ...
```

This is the Google/Azure/AWS pattern:
- `google-cloud-storage -> import google.cloud.storage`
- `azure-identity -> import azure.identity`

Since Polymathera is a company that may ship multiple packages, this gives a clean namespace: `polymathera.colony`, `polymathera.something-else`, etc.

### Required structural changes

1. **Move the source tree** — add a `polymathera` namespace directory:

```
colony/python/
├── polymathera/            <-- NO __init__.py (implicit namespace package, PEP 420)
│   └── colony/             <-- existing colony/ moves here
│       ├── __init__.py
│       ├── cli/
│       ├── distributed/
│       └── ...
```

<mark>The `polymathera/` directory must **not** have `__init__.py` — this makes it an implicit namespace package, so multiple separate PyPI packages can coexist under `polymathera.*` without conflicting.</mark>

2. **`pyproject.toml`**:

```toml
[tool.poetry]
name = "polymathera-colony"
packages = [{include = "polymathera", from = "python"}]

[tool.poetry.scripts]
colony-env = "polymathera.colony.cli.deploy.cli:app"
```

3. **Every internal import changes** — `from colony.x import y` becomes `from polymathera.colony.x import y` across the entire codebase. This is the big cost.

## Recommendation

Option B is the right long-term choice if Polymathera will ship more than one package. The `polymathera.*` namespace is clean, avoids collisions, and matches industry convention for company-scoped packages.

The import rename is a large mechanical refactor that touches every file. It should be done as a dedicated task, separate from the `colony-env` work.
