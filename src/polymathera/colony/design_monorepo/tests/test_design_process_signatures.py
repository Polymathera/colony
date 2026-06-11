"""Signature-audit pin — fails CI if any @action_executor in
DesignProcessCapability re-grows a ``repo`` kwarg.

Same shape as ``test_github_capability_signatures.py``: catch the
drift at import time rather than letting the LLM planner discover it
at runtime. The colony has exactly one design monorepo and the
capability resolves it from agent metadata via
``self._resolve_github_repo()``.

[[no-llm-facing-framework-state]]
"""

from __future__ import annotations

import ast
import inspect

from polymathera.colony.design_monorepo import process as proc_mod


FORBIDDEN_KWARG_NAMES = {"repo"}


def _action_executor_method_names() -> set[str]:
    src = inspect.getsource(proc_mod)
    tree = ast.parse(src)
    out: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "DesignProcessCapability"
        ):
            for item in node.body:
                if isinstance(
                    item, (ast.AsyncFunctionDef, ast.FunctionDef),
                ):
                    for deco in item.decorator_list:
                        if (
                            (
                                isinstance(deco, ast.Call)
                                and getattr(deco.func, "id", None)
                                == "action_executor"
                            )
                            or (
                                isinstance(deco, ast.Name)
                                and deco.id == "action_executor"
                            )
                        ):
                            out.add(item.name)
    return out


def test_no_action_executor_takes_repo_kwarg() -> None:
    """No @action_executor in DesignProcessCapability exposes ``repo``
    as an LLM-facing kwarg. The capability resolves it from agent
    metadata via ``self._resolve_github_repo()``."""

    actions = _action_executor_method_names()
    assert actions, (
        "AST walk found zero @action_executor methods — wrong class?"
    )
    offenders: list[str] = []
    for name in actions:
        method = getattr(proc_mod.DesignProcessCapability, name)
        sig = inspect.signature(method)
        forbidden = FORBIDDEN_KWARG_NAMES.intersection(sig.parameters)
        if forbidden:
            offenders.append(f"{name}: {sorted(forbidden)}")
    assert not offenders, (
        "DesignProcessCapability action(s) re-grew a forbidden "
        f"LLM-facing kwarg — see [[no-llm-facing-framework-state]] in "
        f"MEMORY.md: {offenders}. The capability already knows the "
        "colony's design_monorepo_url; resolve via "
        "self._resolve_github_repo(), not via the LLM planner."
    )
