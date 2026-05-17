"""Tests for ``.colony/tool-registry.json`` IO + capability search."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    REGISTRY_RELATIVE_PATH,
    ToolEntry,
    ToolRegistryError,
    load_registry,
    upsert_tool,
    write_registry,
)
from polymathera.colony.design_monorepo.registry import search


def _entry(**kwargs) -> ToolEntry:
    """Catalog-only entry (no ``capability_fqn``) — upsert_tool skips
    the spec-vs-cache validation, so tests don't need a real
    ToolCapability subclass on the import path."""
    base: dict[str, object] = dict(
        name="laptime",
        purpose="racer/laptime",
        location="subdir:tools/racer/laptime",
        capability="simulate_laptime",
    )
    base.update(kwargs)
    return ToolEntry(**base)


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert load_registry(tmp_path) == ()


def test_round_trip(tmp_path: Path) -> None:
    e = _entry()
    write_registry(tmp_path, [e])
    again = load_registry(tmp_path)
    assert again == (e,)
    assert (tmp_path / REGISTRY_RELATIVE_PATH).is_file()


def test_upsert_replaces_by_name_and_purpose(tmp_path: Path) -> None:
    e1 = _entry(location="subdir:tools/racer/laptime")
    e2 = _entry(location="subdir:tools/racer/laptime_v2")
    upsert_tool(tmp_path, e1)
    upsert_tool(tmp_path, e2)
    entries = load_registry(tmp_path)
    assert len(entries) == 1
    assert entries[0].location == "subdir:tools/racer/laptime_v2"


def test_upsert_appends_new(tmp_path: Path) -> None:
    upsert_tool(tmp_path, _entry())
    upsert_tool(tmp_path, _entry(name="hopkins", purpose="duv/hopkins", capability="solve_hopkins"))
    entries = load_registry(tmp_path)
    assert {e.name for e in entries} == {"laptime", "hopkins"}


def test_load_malformed_raises(tmp_path: Path) -> None:
    p = tmp_path / REGISTRY_RELATIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ToolRegistryError):
        load_registry(tmp_path)


def test_search_exact_match() -> None:
    a = _entry(name="laptime", capability="simulate_laptime")
    b = _entry(name="hopkins", purpose="duv/hopkins", capability="solve_hopkins")
    matches = search("simulate_laptime", local_entries=[a, b])
    assert matches[0].entry.name == "laptime"
    assert matches[0].score == 1.0


def test_search_prefix() -> None:
    a = _entry(name="laptime", capability="simulate_laptime")
    matches = search("simulate", local_entries=[a])
    assert matches[0].score == pytest.approx(0.85)


def test_search_word_overlap() -> None:
    a = _entry(name="laptime_jax", capability="simulate_laptime", purpose="racer/laptime")
    matches = search("racer differentiable laptime", local_entries=[a])
    assert matches[0].score > 0
    assert matches[0].score <= 0.7


def test_search_writable_filter() -> None:
    a = _entry(name="laptime", capability="simulate_laptime")
    matches = search(
        "simulate_laptime",
        local_entries=[a],
        remote_entries=[a],
        require_writable=True,
    )
    # Only the local entry survives (writable=True).
    assert len(matches) == 1
    assert matches[0].writable is True


def test_search_below_min_score_dropped() -> None:
    a = _entry(name="laptime", capability="simulate_laptime")
    matches = search(
        "completely_unrelated_capability_xyz",
        local_entries=[a],
        min_score=0.5,
    )
    assert matches == ()


# --- Spec-vs-cache validation (capability denormalisation contract) ---


def test_upsert_validates_capability_against_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``capability_fqn`` is set, upsert imports the class and
    asserts ``entry.capability`` is one of ``cls.spec.capabilities``.
    Drift raises ``ToolEntrySpecMismatch``."""
    import sys
    import types

    from polymathera.colony.design_monorepo.registry import ToolEntrySpecMismatch

    fake_spec = types.SimpleNamespace(capabilities=("compute_shielding_factor",))
    fake_cls = type("FakeShieldCapability", (), {"spec": fake_spec})
    fake_module = types.ModuleType("polymathera.colony.design_monorepo.tests._fake_shield")
    fake_module.FakeShieldCapability = fake_cls
    monkeypatch.setitem(
        sys.modules,
        "polymathera.colony.design_monorepo.tests._fake_shield",
        fake_module,
    )

    # Agreeing cache → upsert succeeds.
    good = _entry(
        name="shield",
        purpose="opm_meg/shield",
        capability="compute_shielding_factor",
        capability_fqn="polymathera.colony.design_monorepo.tests._fake_shield.FakeShieldCapability",
    )
    upsert_tool(tmp_path, good)

    # Drifted cache → ToolEntrySpecMismatch.
    bad = _entry(
        name="shield",
        purpose="opm_meg/shield",
        capability="solve_em_fdtd",  # not in spec.capabilities
        capability_fqn="polymathera.colony.design_monorepo.tests._fake_shield.FakeShieldCapability",
    )
    with pytest.raises(ToolEntrySpecMismatch):
        upsert_tool(tmp_path, bad)

    # Stub entries (no capability_fqn) are exempt.
    stub = _entry(name="future_tool", purpose="opm_meg/future", capability="future_solver")
    upsert_tool(tmp_path, stub)

    # The validate_spec=False escape hatch also skips the check.
    upsert_tool(tmp_path, bad, validate_spec=False)


def test_upsert_raises_when_capability_fqn_is_unimportable(
    tmp_path: Path,
) -> None:
    from polymathera.colony.design_monorepo.registry import ToolRegistryError

    entry = _entry(
        capability_fqn="polymathera.colony.tests._nonexistent.NotAClass",
    )
    with pytest.raises(ToolRegistryError, match="cannot import"):
        upsert_tool(tmp_path, entry)


def test_load_v1_registry_silently_drops_dropped_fields(
    tmp_path: Path,
) -> None:
    """A v1 on-disk registry (with version/license/headless/container_image)
    loads cleanly under v2 — the dropped fields are stripped, and the
    next write upgrades the file's schema_version in place."""
    import json

    v1_payload = {
        "schema_version": 1,
        "tools": [
            {
                "name": "laptime",
                "purpose": "racer/laptime",
                "capability": "simulate_laptime",
                "location": "subdir:tools/racer/laptime",
                "version": "0.1.0",
                "license": "MIT",
                "container_image": "polymathera/cps-base:0.1.0",
                "headless": "native",
                "extra": {},
            },
        ],
    }
    path = tmp_path / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v1_payload), encoding="utf-8")

    entries = load_registry(tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "laptime"
    assert entries[0].capability == "simulate_laptime"
    # Dropped fields are not on the v2 model.
    assert not hasattr(entries[0], "version")
    assert not hasattr(entries[0], "license")
