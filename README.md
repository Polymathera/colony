# colony
Polymathera's no-RAG, multi-agent framework for extremely long, *dense* contexts (1B+ tokens). It provides:
- A cluster-level **virtual context memory** with user-defined context paging.
- Cache aware agent action policies.
- Powerful and composable multi-agent patterns.
- Arbitrarily sophisticated memory hierarchies and cognitive processes.


### Requirement: Optional Dependencies

Extras are used when you want users who install your package via pip to have access to optional features.

First, mark the dependency as optional under `[tool.poetry.dependencies]`, then list it under `[tool.poetry.extras]`

```toml
[tool.poetry.dependencies]
requests = { version = "^2.25", optional = true }

[tool.poetry.extras]
http = ["requests"]
```

To install the extras:
```bash
poetry install --extras http
# OR
poetry install -E http
# OR (install all)
poetry install --all-extras
```



```bash
# Install the core package only:
pip install my-package

# Install with a single optional feature:
pip install my-package[plot]

# Install with multiple optional features:
pip install my-package[plot,dev]
```


- Handle Optional Imports in Your Code

```python
try:
    import bokeh
except ImportError:
    bokeh = None

def make_interactive_plot():
    if bokeh is None:
        raise ImportError("The 'plot' optional dependency is required for this feature. Install with: pip install my-package[plot]")
    # Your plotting code using bokeh goes here
```



