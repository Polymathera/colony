"""Integration: ``ToolEntry.headless`` round-trips through the
``HeadlessReadiness`` enum + the on-disk JSON registry.

The C5 ``ToolEntry`` was free-form ``str`` in the previous phase; C2
typed it. The contract this test enforces:

- New entries default to ``HeadlessReadiness.NATIVE``.
- An entry serialised with the previous string-only schema (e.g.,
  ``"native"`` / ``"gui_primary"``) loads back without error and
  yields the matching enum value.
- A registry written + read round-trips losslessly.
"""

from __future__ import annotations

import json
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
from polymathera.colony.tools import HeadlessReadiness


def test_default_headless_is_native() -> None:
    e = ToolEntry(
        name="x", purpose="shared/test", capability="cap",
        location="subdir:tools/shared/test/x",
    )
    assert e.headless is HeadlessReadiness.NATIVE


def test_string_headless_coerces_to_enum() -> None:
    e = ToolEntry(
        name="x", purpose="shared/test", capability="cap",
        location="subdir:tools/shared/test/x",
        headless="gui_primary",  # type: ignore[arg-type]
    )
    assert e.headless is HeadlessReadiness.GUI_PRIMARY


def test_invalid_headless_string_rejected() -> None:
    with pytest.raises(Exception):
        ToolEntry(
            name="x", purpose="p", capability="c",
            location="subdir:tools/p/x",
            headless="not-a-tier",  # type: ignore[arg-type]
        )


def test_round_trip_through_registry_file(tmp_path: Path) -> None:
    e1 = ToolEntry(
        name="lt", purpose="racer/laptime", capability="simulate_laptime",
        location="subdir:tools/racer/laptime",
        headless=HeadlessReadiness.NATIVE,
    )
    e2 = ToolEntry(
        name="hop", purpose="duv/hopkins", capability="solve_hopkins",
        location="subdir:tools/duv/hopkins",
        headless=HeadlessReadiness.PARTIAL,
    )
    write_registry(tmp_path, [e1, e2])
    loaded = load_registry(tmp_path)
    assert {x.name: x.headless for x in loaded} == {
        "lt": HeadlessReadiness.NATIVE,
        "hop": HeadlessReadiness.PARTIAL,
    }


def test_legacy_string_registry_loads(tmp_path: Path) -> None:
    """A registry file written by an older colony build (free-form
    string) must still load against the typed schema."""

    payload = {
        "schema_version": 1,
        "tools": [
            {
                "name": "legacy",
                "purpose": "shared/legacy",
                "capability": "do_thing",
                "location": "subdir:tools/shared/legacy",
                "license": "MIT",
                "headless": "cli_only",
                "version": "0.1.0",
                "container_image": None,
                "extra": {},
            },
        ],
    }
    p = tmp_path / REGISTRY_RELATIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    entries = load_registry(tmp_path)
    assert entries[0].headless is HeadlessReadiness.CLI_ONLY


def test_upsert_preserves_typed_headless(tmp_path: Path) -> None:
    e = ToolEntry(
        name="x", purpose="p", capability="c",
        location="subdir:tools/p/x",
        headless=HeadlessReadiness.PARTIAL,
    )
    upsert_tool(tmp_path, e)
    loaded = load_registry(tmp_path)
    assert loaded[0].headless is HeadlessReadiness.PARTIAL
