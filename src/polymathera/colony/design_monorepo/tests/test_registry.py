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
    base: dict[str, object] = dict(
        name="laptime",
        purpose="racer/laptime",
        capability="simulate_laptime",
        version="0.1.0",
        location="subdir:tools/racer/laptime",
        license="MIT",
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
    e1 = _entry(version="0.1.0")
    e2 = _entry(version="0.2.0")
    upsert_tool(tmp_path, e1)
    upsert_tool(tmp_path, e2)
    entries = load_registry(tmp_path)
    assert len(entries) == 1
    assert entries[0].version == "0.2.0"


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
