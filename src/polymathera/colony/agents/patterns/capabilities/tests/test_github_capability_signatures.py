"""Signature-audit pin — fails CI if any @action_executor in
GitHubCapability re-grows a ``repo`` kwarg.

Same shape as the spec validators registered in agents/configs.py:
catch the drift at import time rather than letting the LLM planner
discover it at runtime.

[[no-llm-facing-framework-state]]
"""

from __future__ import annotations

import ast
import inspect

from polymathera.colony.agents.patterns.capabilities import github as gh_mod


FORBIDDEN_KWARG_NAMES = {"repo"}
# App-installation actions legitimately take no repo and are exempt;
# add new exempt actions here only after confirming they don't
# operate on a specific repo.
EXEMPT_ACTIONS = {
    "list_repos",
    "whoami",
    "list_project_items",
}


def _action_executor_method_names() -> set[str]:
    src = inspect.getsource(gh_mod)
    tree = ast.parse(src)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GitHubCapability":
            for item in node.body:
                if isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)):
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


def test_signature_audit():
    """No @action_executor in GitHubCapability exposes ``repo`` as an
    LLM-facing kwarg. The colony has exactly one design monorepo and
    the capability resolves it from agent metadata."""

    actions = _action_executor_method_names()
    assert actions, "AST walk found zero @action_executor methods — wrong class?"
    offenders: list[str] = []
    for name in actions:
        if name in EXEMPT_ACTIONS:
            continue
        method = getattr(gh_mod.GitHubCapability, name)
        sig = inspect.signature(method)
        forbidden = FORBIDDEN_KWARG_NAMES.intersection(sig.parameters)
        if forbidden:
            offenders.append(f"{name}: {sorted(forbidden)}")
    assert not offenders, (
        "GitHubCapability action(s) re-grew a forbidden LLM-facing kwarg "
        f"— see [[no-llm-facing-framework-state]] in MEMORY.md: {offenders}. "
        "The capability already knows the colony's design_monorepo_url; "
        "resolve it in _resolve_repo, not via the LLM planner."
    )


def test_design_monorepo_url_parameter_spec_declared():
    """``GitHubCapability`` reads ``design_monorepo_url`` from the
    agent's metadata.parameters bag. The typed-parameters discipline
    (see ``feedback_colony_scoped_params_propagation.md``) requires
    every CONSUMER of a metadata parameter to declare its own
    ``ParameterSpec`` in ``AGENT_METADATA_PARAMS`` — both so the
    central inheritance gate knows the capability needs the value
    propagated to spawned children, and so static auditors can see
    which parameters this capability cares about without grepping
    method bodies. Pinned here so any refactor that drops the spec
    fails CI loudly."""

    specs = gh_mod.GitHubCapability.AGENT_METADATA_PARAMS
    names = {spec.name for spec in specs}
    assert "design_monorepo_url" in names, (
        "GitHubCapability.AGENT_METADATA_PARAMS is missing a spec for "
        "'design_monorepo_url' but _resolve_repo reads it from the "
        "metadata bag. The typed-parameters discipline requires every "
        "consumer to declare its own ParameterSpec — see "
        "feedback_colony_scoped_params_propagation.md."
    )
    spec = next(s for s in specs if s.name == "design_monorepo_url")
    # Must match the canonical scope declared on
    # DesignMonorepoCapabilityBase or the inheritance gate would
    # propagate the value differently across the boundary.
    from polymathera.colony.agents.metadata_parameters import ParameterScope
    assert spec.scope is ParameterScope.COLONY, (
        f"design_monorepo_url spec on GitHubCapability has wrong scope "
        f"{spec.scope!r} — must be COLONY to match the canonical spec "
        f"on DesignMonorepoCapabilityBase."
    )


def test_design_monorepo_url_param_is_the_canonical_spec():
    """``GitHubCapability`` reuses the canonical
    ``DESIGN_MONOREPO_URL_PARAM`` from ``agents.metadata_parameters`` as
    the same object (no copy, no mirror). Single source of truth: if
    someone renames the key or changes the scope at the canonical site,
    both ``GitHubCapability`` and ``DesignMonorepoCapabilityBase`` track
    it for free. Regression pin so the next refactor doesn't reintroduce
    a duplicate ``ParameterSpec`` on either side."""

    from polymathera.colony.agents.metadata_parameters import (
        DESIGN_MONOREPO_URL_PARAM,
    )
    from polymathera.colony.design_monorepo.capabilities import (
        DesignMonorepoCapabilityBase,
    )
    specs = gh_mod.GitHubCapability.AGENT_METADATA_PARAMS
    assert DESIGN_MONOREPO_URL_PARAM in specs, (
        "GitHubCapability.AGENT_METADATA_PARAMS no longer references "
        "metadata_parameters.DESIGN_MONOREPO_URL_PARAM by identity. "
        "Don't duplicate the spec — import the canonical one."
    )
    assert DESIGN_MONOREPO_URL_PARAM in DesignMonorepoCapabilityBase.AGENT_METADATA_PARAMS, (
        "DesignMonorepoCapabilityBase.AGENT_METADATA_PARAMS no longer "
        "references metadata_parameters.DESIGN_MONOREPO_URL_PARAM by "
        "identity. Both capabilities must share the canonical spec."
    )
