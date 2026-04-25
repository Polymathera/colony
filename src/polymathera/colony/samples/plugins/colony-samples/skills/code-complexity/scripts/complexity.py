"""Compute per-function cyclomatic complexity for Python sources.

Standard-library only — runs against any sandbox image that ships
Python ≥ 3.10. Designed to be invoked by ``run.sh`` from the skill's
``UserPluginCapability.run_skill`` call; the stdin/stdout contract is
plain JSON so the LLM can consume it directly.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Cyclomatic complexity
# ---------------------------------------------------------------------------

# Each of these AST nodes adds 1 to the function's complexity. The
# starting baseline is 1 (the entry edge).
_BRANCHING_NODES: tuple[type[ast.AST], ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.Assert,
    ast.IfExp,
    ast.comprehension,        # generator/list/set/dict comprehension predicates
    ast.match_case,           # PEP 634
)


def _function_complexity(func: ast.AST) -> int:
    """Return McCabe complexity for one function (or method)."""
    score = 1
    for node in ast.walk(func):
        if isinstance(node, _BRANCHING_NODES):
            score += 1
            continue
        # Boolean operators contribute one per *additional* operand.
        if isinstance(node, ast.BoolOp):
            score += max(0, len(node.values) - 1)
            continue
        # ``with`` adds one per context manager item.
        if isinstance(node, (ast.With, ast.AsyncWith)):
            score += len(node.items)
    return score


def _iter_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------

def _iter_python_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    if not path.is_dir():
        return
    for root, dirs, files in os.walk(path):
        # Skip noisy / unhelpful trees.
        dirs[:] = [d for d in dirs if d not in {
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            "dist", "build",
        }]
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def _scan(path: Path, *, threshold: int, top_n: int) -> dict:
    files_scanned = 0
    rows: list[dict] = []
    for py in _iter_python_files(path):
        files_scanned += 1
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(py))
        except SyntaxError as e:
            rows.append({
                "file": str(py),
                "function": "<syntax error>",
                "lineno": e.lineno or 0,
                "complexity": -1,
                "error": str(e),
            })
            continue
        for func in _iter_functions(tree):
            score = _function_complexity(func)
            if score < threshold:
                continue
            rows.append({
                "file": str(py),
                "function": func.name,
                "lineno": func.lineno,
                "complexity": score,
            })
    rows.sort(key=lambda r: r["complexity"], reverse=True)
    return {
        "files_scanned": files_scanned,
        "functions_found": len(rows),
        "threshold": threshold,
        "top_n": top_n,
        "results": rows[:top_n],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-function cyclomatic complexity scanner.",
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--top_n", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    target = Path(args.path)
    if not target.exists():
        print(json.dumps({
            "error": f"path does not exist: {target}",
        }))
        return 2
    report = _scan(
        target, threshold=args.threshold, top_n=args.top_n,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
