---
name: code-complexity
description: |
  Compute per-function cyclomatic complexity for Python source files
  and rank the most complex functions. Use this when the user asks
  to find "the most complex code", "functions that are hard to
  maintain", "candidates for refactoring", or to triage a Python
  codebase's hotspots before deeper analysis.
when_to_use: |
  Triggered when Python files are involved and the user wants a
  quantitative read on code complexity, refactor targets, or
  maintenance risk hotspots.
sandbox_image_role: default
script: scripts/run.sh
params:
  path:
    type: string
    required: true
    description: |
      Filesystem path inside the sandbox to analyse. Either a single
      Python file or a directory; directories are scanned
      recursively for ``*.py`` files.
  threshold:
    type: integer
    required: false
    description: |
      Minimum cyclomatic complexity score to include in the output.
      Defaults to 5 (anything below is considered routine).
  top_n:
    type: integer
    required: false
    description: |
      Maximum number of functions to report (after sorting by
      complexity, descending). Defaults to 25.
timeout_seconds: 120
paths: "**/*.py"
---

# Code Complexity

Walks the requested path with the standard library's ``ast`` module
and computes McCabe-style cyclomatic complexity for every function and
method. Each branching construct adds one to the score:

- ``if``, ``elif``, ``while``, ``for``, ``with`` items, ``try``
  handlers, ``and``/``or`` boolean operators, ternary expressions,
  match/case alternatives.

The script runs entirely inside the sandbox using only the Python
standard library, so it works against any image that ships Python ≥
3.10. The output is a small JSON document the LLM can reason about
directly.

## Output shape

```
{
  "files_scanned": 12,
  "functions_found": 173,
  "threshold": 5,
  "top_n": 25,
  "results": [
    {"file": "auth/login.py", "function": "validate_session",
     "lineno": 42, "complexity": 14},
    ...
  ]
}
```

Higher numbers indicate more decision branches. Anything ≥ 10 is
typically a refactor candidate; ≥ 20 strongly suggests the function is
doing too much.
