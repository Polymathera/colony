"""Tests for the canonical class resolver
(:func:`polymathera.colony.agents.class_resolver.resolve_class`).

Pins the dual-path lookup semantics + the input-validation contract.
Every spawn path in the codebase (CLI ``polymath cluster run-mission``,
REST ``/api/jobs/submit`` and ``/api/agents/spawn``, the chat-driven
``AgentPoolCapability.create_agent``) delegates here, so a drift in
these semantics is a system-wide bug.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.class_resolver import resolve_class
from polymathera.colony.agents.models import AgentMetadata


def test_resolves_pip_installed_class() -> None:
    """The canonical importlib path resolves a fully-qualified
    pip-installed class. No fallback_registry needed."""
    cls = resolve_class("polymathera.colony.agents.models.AgentMetadata")
    assert cls is AgentMetadata


def test_falls_back_to_registry_on_import_failure() -> None:
    """When importlib fails (the L1-A case — module not in
    ``sys.modules``), resolve_class looks up the class short-name in
    ``fallback_registry``."""

    class _SyntheticL4Coordinator:
        pass

    cls = resolve_class(
        "synthetic_l4_coordinator.SyntheticL4Coordinator",
        fallback_registry={
            "SyntheticL4Coordinator": _SyntheticL4Coordinator,
        },
    )
    assert cls is _SyntheticL4Coordinator


def test_falls_back_to_registry_on_missing_attr() -> None:
    """When the module imports but the class isn't an attribute on
    it (typo, removed export), the fallback registry still gets a
    chance via the same code path that catches ImportError."""

    class _SyntheticReplacement:
        pass

    # ``polymathera.colony.agents.models`` is a real module, but
    # ``NoSuchClass`` is not defined on it. The AttributeError
    # triggers the fallback.
    cls = resolve_class(
        "polymathera.colony.agents.models.NoSuchClass",
        fallback_registry={"NoSuchClass": _SyntheticReplacement},
    )
    assert cls is _SyntheticReplacement


def test_importlib_wins_over_fallback() -> None:
    """A name that resolves via importlib AND is present in the
    fallback registry uses the importlib result. Pip-installed classes
    are canonical; the fallback exists only to cover the
    not-in-sys-modules gap, not to override."""

    class _Decoy:
        pass

    cls = resolve_class(
        "polymathera.colony.agents.models.AgentMetadata",
        fallback_registry={"AgentMetadata": _Decoy},
    )
    assert cls is AgentMetadata
    assert cls is not _Decoy


def test_reraises_when_fallback_misses() -> None:
    """Neither path resolves → the original ImportError/AttributeError
    surfaces so the caller sees the precise reason (typo, missing
    extension, etc.)."""
    with pytest.raises((ImportError, AttributeError)):
        resolve_class(
            "definitely.not.a.module.NoSuchClass",
            fallback_registry={"OtherClass": object},
        )


def test_no_fallback_kwarg_is_backwards_compatible() -> None:
    """Pre-Stage-A callers that omit the kwarg keep the legacy
    behaviour: importlib-only, no fallback path."""
    with pytest.raises((ImportError, AttributeError)):
        resolve_class("definitely.not.a.module.NoSuchClass")


@pytest.mark.parametrize("bad", ["", "no_dots", 42, None, object()])
def test_rejects_non_dotted_input(bad) -> None:
    with pytest.raises(ValueError, match="fully qualified"):
        resolve_class(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Single-source-of-truth audit: every existing delegate stays delegating.
# These tests pin the delegation so future refactors that re-introduce a
# parallel resolver implementation fail loudly.
# ---------------------------------------------------------------------------


def test_agent_pool_capability_delegates_to_canonical_resolver() -> None:
    """``AgentPoolCapability._resolve_class`` exists as a thin wrapper
    over :func:`resolve_class` for backwards compatibility with tests
    / callers that already reach it via the staticmethod name. The
    behaviour must be identical."""
    from polymathera.colony.agents.patterns.capabilities.agent_pool import (
        AgentPoolCapability,
    )

    class _SyntheticAgent:
        pass

    cls = AgentPoolCapability._resolve_class(
        "synthetic.SyntheticAgent",
        fallback_registry={"SyntheticAgent": _SyntheticAgent},
    )
    assert cls is _SyntheticAgent


def test_cli_polymath_resolve_class_delegates_to_canonical_resolver() -> None:
    """``polymathera.colony.cli.polymath._resolve_class`` is the
    backwards-compatible name imported by older REST callers; after
    Stage A it is a thin delegate over :func:`resolve_class`. Pins
    the delegation so future refactors don't accidentally restore the
    duplicate implementation."""
    from polymathera.colony.cli.polymath import _resolve_class

    class _SyntheticAgent:
        pass

    cls = _resolve_class(
        "synthetic.SyntheticAgent",
        fallback_registry={"SyntheticAgent": _SyntheticAgent},
    )
    assert cls is _SyntheticAgent
