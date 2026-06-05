"""Tests for the ``metadata_parameters`` declaration layer.

Pins the typed contract for ``AgentMetadata.parameters``: spec
semantics (required-iff-no-default, default vs default_factory
resolution), registry conflict policy, and — most importantly —
:func:`inherit_scoped_parameters`, the single central inheritance
gate every agent spawn routes through. A regression in these
semantics is a system-wide bug, hence the careful coverage.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from polymathera.colony.agents.metadata_parameters import (
    MISSING,
    MetadataParameterRegistry,
    ParameterScope,
    ParameterSpec,
    inherit_scoped_parameters,
)


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------


def test_spec_required_when_no_default() -> None:
    s = ParameterSpec(
        name="x", scope=ParameterScope.CALLER, description="needed",
    )
    assert s.required is True
    assert s.default is MISSING
    assert s.default_factory is None


def test_spec_optional_with_default() -> None:
    s = ParameterSpec(
        name="x", scope=ParameterScope.CALLER, description="opt",
        default="hello",
    )
    assert s.required is False
    assert s.resolve_default() == "hello"


def test_spec_optional_with_default_none() -> None:
    """``default=None`` IS a declared default — the spec is optional.

    This is the load-bearing distinction from the prior ``required:
    bool`` design: ``default=None`` unambiguously means "optional,
    no value to apply at init."""
    s = ParameterSpec(
        name="x", scope=ParameterScope.CALLER, description="opt",
        default=None,
    )
    assert s.required is False
    assert s.resolve_default() is None


def test_spec_optional_with_factory() -> None:
    s = ParameterSpec(
        name="x", scope=ParameterScope.COLONY, description="opt",
        json_type="object", default_factory=dict,
    )
    assert s.required is False
    d1 = s.resolve_default()
    d2 = s.resolve_default()
    assert d1 == {} and d2 == {}
    # Fresh dict per call — required for mutable defaults.
    assert d1 is not d2


def test_spec_resolve_default_on_required_raises() -> None:
    s = ParameterSpec(
        name="x", scope=ParameterScope.CALLER, description="needed",
    )
    with pytest.raises(RuntimeError, match="required"):
        s.resolve_default()


def test_spec_rejects_both_default_and_factory() -> None:
    with pytest.raises(ValidationError, match="cannot declare both"):
        ParameterSpec(
            name="x", scope=ParameterScope.CALLER, description="d",
            default="hi", default_factory=lambda: "bye",
        )


def test_spec_extra_forbid() -> None:
    """Typo'd keys in spec dict literals (mission registry entries)
    should surface as ValidationError, not silently pass."""
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate({
            "name": "x", "scope": "caller", "description": "d",
            "requierd": True,   # typo
        })


def test_spec_equality_and_hashability() -> None:
    """Equal specs compare equal (frozen + pydantic field equality).
    Required for the registry's idempotent re-registration path."""
    a = ParameterSpec(
        name="x", scope=ParameterScope.COLONY, description="d",
    )
    b = ParameterSpec(
        name="x", scope=ParameterScope.COLONY, description="d",
    )
    assert a == b


def test_spec_frozen() -> None:
    s = ParameterSpec(
        name="x", scope=ParameterScope.CALLER, description="d",
    )
    with pytest.raises(ValidationError):
        s.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MetadataParameterRegistry
# ---------------------------------------------------------------------------


def test_registry_register_get_roundtrip() -> None:
    reg = MetadataParameterRegistry()
    spec = ParameterSpec(
        name="design_monorepo_url",
        scope=ParameterScope.COLONY,
        description="design monorepo origin url",
    )
    reg.register(spec)
    assert reg.get("design_monorepo_url") is spec
    assert reg.get("nope") is None


def test_registry_idempotent_on_equal_redeclaration() -> None:
    """Two capabilities can declare the same spec — as long as the
    declarations agree, the registry accepts the second call."""
    reg = MetadataParameterRegistry()
    a = ParameterSpec(
        name="x", scope=ParameterScope.COLONY, description="d",
    )
    b = ParameterSpec(
        name="x", scope=ParameterScope.COLONY, description="d",
    )
    reg.register(a)
    reg.register(b)
    assert reg.all_specs() == (a,)


def test_registry_raises_on_conflicting_redeclaration() -> None:
    reg = MetadataParameterRegistry()
    reg.register(ParameterSpec(
        name="x", scope=ParameterScope.COLONY, description="d",
    ))
    with pytest.raises(ValueError, match="re-declared"):
        reg.register(ParameterSpec(
            name="x", scope=ParameterScope.SESSION, description="d",
        ))


def test_registry_keys_by_scope_partitions() -> None:
    reg = MetadataParameterRegistry()
    reg.register(ParameterSpec(
        name="colony_a", scope=ParameterScope.COLONY, description="d",
    ))
    reg.register(ParameterSpec(
        name="colony_b", scope=ParameterScope.COLONY, description="d",
    ))
    reg.register(ParameterSpec(
        name="session_a", scope=ParameterScope.SESSION, description="d",
    ))
    reg.register(ParameterSpec(
        name="caller_a", scope=ParameterScope.CALLER, description="d",
    ))
    reg.register(ParameterSpec(
        name="agent_a", scope=ParameterScope.AGENT, description="d",
    ))
    assert reg.keys_by_scope(ParameterScope.COLONY) == frozenset(
        {"colony_a", "colony_b"},
    )
    assert reg.keys_by_scope(ParameterScope.SESSION) == frozenset(
        {"session_a"},
    )
    assert reg.keys_by_scope(ParameterScope.CALLER) == frozenset(
        {"caller_a"},
    )
    assert reg.keys_by_scope(ParameterScope.AGENT) == frozenset(
        {"agent_a"},
    )


# ---------------------------------------------------------------------------
# inherit_scoped_parameters — the central inheritance gate.
# ---------------------------------------------------------------------------


def _seeded_registry() -> MetadataParameterRegistry:
    reg = MetadataParameterRegistry()
    reg.register(ParameterSpec(
        name="design_monorepo_url",
        scope=ParameterScope.COLONY, description="d",
    ))
    reg.register(ParameterSpec(
        name="git_attribution",
        scope=ParameterScope.COLONY, description="d",
        json_type="object", default_factory=dict,
    ))
    reg.register(ParameterSpec(
        name="github_identity",
        scope=ParameterScope.COLONY, description="d",
        json_type="object",
    ))
    reg.register(ParameterSpec(
        name="available_tools",
        scope=ParameterScope.SESSION, description="d",
        json_type="object", default_factory=dict,
    ))
    reg.register(ParameterSpec(
        name="mode",
        scope=ParameterScope.CALLER, description="d",
    ))
    reg.register(ParameterSpec(
        name="agent_local",
        scope=ParameterScope.AGENT, description="d",
    ))
    return reg


def test_inherit_colony_keys_flow_parent_to_child() -> None:
    reg = _seeded_registry()
    parent = {
        "design_monorepo_url": "https://github.com/acme/monorepo.git",
        "git_attribution": {"commit_principal": "colony"},
        "github_identity": {"tenant_installation_id": "100"},
    }
    out = inherit_scoped_parameters(
        child_params={}, parent_params=parent, registry=reg,
    )
    assert out == parent


def test_inherit_session_keys_flow_parent_to_child() -> None:
    reg = _seeded_registry()
    parent = {"available_tools": {"foo": "bar"}}
    out = inherit_scoped_parameters(
        child_params={}, parent_params=parent, registry=reg,
    )
    assert out == {"available_tools": {"foo": "bar"}}


def test_inherit_caller_keys_do_not_flow() -> None:
    """CALLER-scoped keys are supplied by the spawn caller per-spawn
    and MUST NOT be inherited from the parent — that would silently
    rebind missions/actions to the parent's choice."""
    reg = _seeded_registry()
    parent = {"mode": "bootstrap"}
    out = inherit_scoped_parameters(
        child_params={}, parent_params=parent, registry=reg,
    )
    assert out == {}


def test_inherit_agent_keys_do_not_flow() -> None:
    """AGENT-scoped keys are per-agent — inheriting from the parent
    would alias state across the agent boundary."""
    reg = _seeded_registry()
    parent = {"agent_local": "abc"}
    out = inherit_scoped_parameters(
        child_params={}, parent_params=parent, registry=reg,
    )
    assert out == {}


def test_inherit_child_wins_on_collision() -> None:
    """The LLM planner deliberately rebinding ``design_monorepo_url``
    (e.g. a mission probing a different repo) keeps its value."""
    reg = _seeded_registry()
    parent = {"design_monorepo_url": "https://github.com/acme/mono.git"}
    child = {"design_monorepo_url": "https://github.com/acme/probe.git"}
    out = inherit_scoped_parameters(
        child_params=child, parent_params=parent, registry=reg,
    )
    assert out == {"design_monorepo_url": "https://github.com/acme/probe.git"}


def test_inherit_treats_parent_none_as_absent() -> None:
    """A colony with no design monorepo configured surfaces as
    ``design_monorepo_url=None`` on the parent's parameters. That
    should NOT propagate — a None on the child is indistinguishable
    from "never set" downstream, and the capability validation
    treats both the same."""
    reg = _seeded_registry()
    parent = {"design_monorepo_url": None}
    out = inherit_scoped_parameters(
        child_params={}, parent_params=parent, registry=reg,
    )
    assert out == {}


def test_inherit_does_not_mutate_inputs() -> None:
    reg = _seeded_registry()
    parent = {"design_monorepo_url": "u"}
    child = {"mode": "bootstrap"}
    inherit_scoped_parameters(
        child_params=child, parent_params=parent, registry=reg,
    )
    assert parent == {"design_monorepo_url": "u"}
    assert child == {"mode": "bootstrap"}


def test_inherit_with_empty_inputs_returns_fresh_dict() -> None:
    """Empty dicts in → fresh empty dict out. The signature requires
    real dicts (Agent.metadata.parameters is a pydantic
    default_factory field, so callers always have one), and the
    returned dict must be a fresh allocation so callers can mutate
    it without affecting future calls."""

    reg = _seeded_registry()
    out = inherit_scoped_parameters(
        child_params={}, parent_params={}, registry=reg,
    )
    assert out == {}
    out["sentinel"] = 1
    out2 = inherit_scoped_parameters(
        child_params={}, parent_params={}, registry=reg,
    )
    assert "sentinel" not in out2


def test_inherit_with_empty_registry_is_no_op() -> None:
    """Before step 4 of the rollout (capability declarations land)
    the registry is empty and inheritance must be a complete no-op.
    Pins the safe-by-default property that lets the gate be wired
    into create_agent before any capability declares its specs."""
    reg = MetadataParameterRegistry()
    parent = {"design_monorepo_url": "u", "mode": "bootstrap"}
    child = {"k": "v"}
    out = inherit_scoped_parameters(
        child_params=child, parent_params=parent, registry=reg,
    )
    assert out == {"k": "v"}
