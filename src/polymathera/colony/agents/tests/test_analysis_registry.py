"""Tests for ``polymathera.colony.agents.analysis_registry``.

Covers the entry-point-group discovery surface that lets domain
packages (e.g. ``polymathera-cps``) register coordinator analyses
without colony having to import them.

Domain coverage:

- The hardcoded ``ANALYSIS_REGISTRY`` is always present in the
  union (regression on the legacy code-analysis sample types).
- Plugin entries appear alongside builtins.
- Plugin entries can shadow builtins (with a warning).
- Plugin load failures (factory raises, missing keys, wrong type)
  are isolated — the rest of the registry survives.
- The module exposes ``ANALYSIS_TYPES_ENTRY_POINT_GROUP`` as a
  stable string constant (cps and other domain packages depend on
  this value).
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any

import pytest

from polymathera.colony.agents.analysis_registry import (
    ANALYSIS_TYPES_ENTRY_POINT_GROUP,
    get_analysis_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ep(name: str, loaded_factory) -> EntryPoint:
    """Build an EntryPoint whose ``.load()`` returns ``loaded_factory``.

    The real ``importlib.metadata.EntryPoint`` resolves a dotted path
    to a callable at ``.load()`` time. Tests can't easily install a
    real package, so we subclass and override ``load`` to return the
    factory directly. The ``name`` / ``value`` / ``group`` fields are
    still rendered in log messages, so we set them to plausible
    values."""

    class _StubEntryPoint(EntryPoint):
        def load(self):  # type: ignore[override]
            return loaded_factory

    return _StubEntryPoint(
        name=name,
        value=f"test_module:{name}",
        group=ANALYSIS_TYPES_ENTRY_POINT_GROUP,
    )


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, eps: tuple[EntryPoint, ...],
) -> None:
    """Patch ``importlib.metadata.entry_points`` (as imported into the
    module under test) to return ``eps`` for our group and an empty
    tuple for any other group."""

    def _fake_entry_points(*, group: str = "") -> tuple[EntryPoint, ...]:
        if group == ANALYSIS_TYPES_ENTRY_POINT_GROUP:
            return eps
        return ()

    monkeypatch.setattr(
        "polymathera.colony.agents.analysis_registry.entry_points",
        _fake_entry_points,
    )


def _opm_meg_entry() -> dict[str, Any]:
    return {
        "label": "OPM-MEG Noise-Floor Design Analysis",
        "description": (
            "Quantum-sensing OPM-MEG (Optically-Pumped Magnetometer "
            "Magnetoencephalography) design analysis."
        ),
        "coordinator_v2": (
            "polymathera.cps.domains.quantum.agents.coordinator.OPMMEGCoordinator"
        ),
        "worker": (
            "polymathera.cps.domains.quantum.agents.atomic_physics.AtomicPhysicsAgent"
        ),
        "coordinator_capabilities": ["OPMMEGAnalysisCapability"],
        "worker_capabilities": ["AtomicPhysicsCapability"],
        "extra_metadata_keys": ["budget_id", "design_repo_path"],
    }


# ---------------------------------------------------------------------------
# Builtin registry — always present
# ---------------------------------------------------------------------------


def test_builtins_always_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """No plugins registered → registry equals the colony-builtin set."""

    _patch_entry_points(monkeypatch, ())
    reg = get_analysis_registry()
    # Builtin sample analyses (declared in colony/cli/polymath.py).
    for builtin in ("impact", "slicing", "compliance", "intent", "contracts", "basic"):
        assert builtin in reg, (
            f"Builtin analysis {builtin!r} disappeared from get_analysis_registry()"
        )
        assert reg[builtin]["label"]
        assert "coordinator_v2" in reg[builtin]


def test_returns_a_copy_not_the_underlying_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating the return value must NOT affect the underlying
    ``ANALYSIS_REGISTRY`` (each call returns a fresh dict)."""

    _patch_entry_points(monkeypatch, ())
    first = get_analysis_registry()
    first["impact"]["label"] = "POISONED"
    second = get_analysis_registry()
    assert second["impact"]["label"] != "POISONED"


# ---------------------------------------------------------------------------
# Plugin discovery — happy path
# ---------------------------------------------------------------------------


def test_plugin_entry_appears_in_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        (_make_ep("opm_meg", _opm_meg_entry),),
    )
    reg = get_analysis_registry()
    assert "opm_meg" in reg
    assert reg["opm_meg"]["label"].startswith("OPM-MEG")
    assert "coordinator_v2" in reg["opm_meg"]


def test_multiple_plugin_entries_all_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _racer_entry() -> dict[str, Any]:
        return {
            "label": "RACER Lap-Time Optimization",
            "description": "Autonomous EV racing analysis.",
            "coordinator_v2": "polymathera.cps.domains.racer.agents.RacerCoordinator",
            "worker": "polymathera.cps.domains.racer.agents.LapTimeAgent",
        }

    _patch_entry_points(
        monkeypatch,
        (
            _make_ep("opm_meg", _opm_meg_entry),
            _make_ep("racer", _racer_entry),
        ),
    )
    reg = get_analysis_registry()
    assert "opm_meg" in reg
    assert "racer" in reg
    # Builtins still here.
    assert "impact" in reg


# ---------------------------------------------------------------------------
# Plugin shadowing
# ---------------------------------------------------------------------------


def test_plugin_can_shadow_builtin_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A plugin entry whose name matches a builtin overrides the
    builtin (last-write-wins) and emits a WARNING log."""

    def _shadow_impact() -> dict[str, Any]:
        return {
            "label": "Custom Impact (overridden)",
            "description": "Plugin shadow of impact.",
            "coordinator_v2": "test.shadow.ImpactCoordinator",
        }

    _patch_entry_points(
        monkeypatch, (_make_ep("impact", _shadow_impact),),
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.agents.analysis_registry"):
        reg = get_analysis_registry()
    assert reg["impact"]["label"] == "Custom Impact (overridden)"
    assert any(
        "shadows a colony-builtin entry" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Plugin failure isolation
# ---------------------------------------------------------------------------


def test_plugin_factory_raise_does_not_break_registry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A plugin whose factory raises is logged + skipped; the rest
    of the registry (builtins + other plugins) survives intact."""

    def _bad_factory() -> dict[str, Any]:
        raise RuntimeError("plugin import side-effect blew up")

    _patch_entry_points(
        monkeypatch,
        (
            _make_ep("broken", _bad_factory),
            _make_ep("opm_meg", _opm_meg_entry),
        ),
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.agents.analysis_registry"):
        reg = get_analysis_registry()
    assert "broken" not in reg
    assert "opm_meg" in reg  # untouched by the failure
    assert "impact" in reg   # builtins survive
    assert any("broken" in r.getMessage() for r in caplog.records)


def test_plugin_returning_non_dict_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def _wrong_shape() -> str:  # type: ignore[return-value]
        return "not a dict"

    _patch_entry_points(
        monkeypatch,
        (_make_ep("badshape", _wrong_shape),),
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.agents.analysis_registry"):
        reg = get_analysis_registry()
    assert "badshape" not in reg
    assert any("expected dict" in r.getMessage() for r in caplog.records)


def test_plugin_missing_required_keys_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def _no_label() -> dict[str, Any]:
        # Missing 'label' (required key).
        return {
            "description": "An analysis without a label.",
            "coordinator_v2": "x.Y",
        }

    _patch_entry_points(
        monkeypatch,
        (_make_ep("no_label", _no_label),),
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.agents.analysis_registry"):
        reg = get_analysis_registry()
    assert "no_label" not in reg
    assert any(
        "missing required keys" in r.getMessage() for r in caplog.records
    )


def test_plugin_without_coordinator_class_warns_but_keeps_entry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Without ``coordinator_v1`` / ``coordinator_v2`` the SessionAgent
    can show the entry but cannot spawn from it. Surface a warning,
    but keep the entry — the operator may want to see it in
    ``available_analyses`` for diagnostic purposes."""

    def _no_coordinator() -> dict[str, Any]:
        return {
            "label": "Documentation-only entry",
            "description": "An analysis stub with no coordinator yet.",
            # No coordinator_v1 / coordinator_v2.
        }

    _patch_entry_points(
        monkeypatch, (_make_ep("docs_only", _no_coordinator),),
    )
    with caplog.at_level("WARNING", logger="polymathera.colony.agents.analysis_registry"):
        reg = get_analysis_registry()
    assert "docs_only" in reg
    assert any(
        "no coordinator_v1 / coordinator_v2" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Stable contract — entry-point group name
# ---------------------------------------------------------------------------


def test_entry_point_group_name_is_stable() -> None:
    """The group name is referenced from every domain package's
    ``pyproject.toml``. Renaming silently breaks all CPS plugins."""

    assert ANALYSIS_TYPES_ENTRY_POINT_GROUP == "polymathera.analysis_types"
