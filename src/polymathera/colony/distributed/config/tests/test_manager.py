"""Round-trip tests for :class:`ConfigurationManager`.

Covers the YAML-load path and ``set_config_path`` re-load. Uses a test-only
``ConfigComponent`` so the suite does not need to satisfy the env-var
requirements of every infrastructure config registered in
``polymathera.colony.distributed.configs``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from pydantic import Field

from polymathera.colony.distributed.config import (
    ConfigComponent,
    register_polymathera_config,
)
from polymathera.colony.distributed.config.manager import ConfigurationManager


pytestmark = pytest.mark.asyncio


@register_polymathera_config(path="manager_test")
class _ManagerTestConfig(ConfigComponent):
    name: str = Field(default="default-name")
    enabled: bool = Field(default=False)


async def test_yaml_overrides_default(tmp_path: Path) -> None:
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "manager_test:\n"
        "  name: from-yaml\n"
        "  enabled: true\n"
    )

    cm = ConfigurationManager(config_path=str(yaml_file))
    await cm.initialize()

    cfg = cm.get_component("manager_test")
    assert cfg.name == "from-yaml"
    assert cfg.enabled is True


async def test_set_config_path_marks_dirty_and_reloads(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    first.write_text("manager_test:\n  name: first\n")
    second = tmp_path / "second.yaml"
    second.write_text("manager_test:\n  name: second\n")

    cm = ConfigurationManager(config_path=str(first))
    await cm.initialize()
    assert cm.get_component("manager_test").name == "first"

    cm.set_config_path(str(second))
    assert cm.is_initialized is False  # dirty flag flipped by set_config_path
    await cm.initialize()
    assert cm.get_component("manager_test").name == "second"


async def test_set_config_path_no_op_when_unchanged(tmp_path: Path) -> None:
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("manager_test:\n  name: only\n")

    cm = ConfigurationManager(config_path=str(yaml_file))
    await cm.initialize()
    assert cm.is_initialized is True

    cm.set_config_path(str(yaml_file))
    assert cm.is_initialized is True  # unchanged path → no dirty flip



async def test_no_config_path_uses_defaults() -> None:
    cm = ConfigurationManager()
    await cm.initialize()
    cfg = cm.get_component("manager_test")
    assert cfg.name == "default-name"


async def test_initialize_waits_for_late_arriving_config(tmp_path: Path) -> None:
    """Mimics the colony-env up race: file is `docker cp`'d after the manager
    starts initialise. The wait loop must pick it up."""
    yaml_file = tmp_path / "late.yaml"

    async def _drop_file_after(delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        yaml_file.write_text("manager_test:\n  name: late-arrival\n")

    drop = asyncio.create_task(_drop_file_after(0.2))
    cm = ConfigurationManager(
        config_path=str(yaml_file), wait_for_config_seconds=2.0,
    )

    started = time.monotonic()
    await cm.initialize()
    elapsed = time.monotonic() - started
    await drop

    assert 0.15 < elapsed < 1.5, f"unexpected elapsed={elapsed:.3f}s"
    assert cm.get_component("manager_test").name == "late-arrival"


async def test_initialize_falls_through_after_wait_timeout(tmp_path: Path) -> None:
    """If the file never arrives, log a warning and fall through to defaults
    + env vars — never raise. Equivalent to the no-config-path behaviour but
    with a visible warning."""
    missing = tmp_path / "never_arrives.yaml"
    cm = ConfigurationManager(
        config_path=str(missing), wait_for_config_seconds=0.2,
    )

    started = time.monotonic()
    await cm.initialize()
    elapsed = time.monotonic() - started

    assert elapsed >= 0.2
    assert cm.get_component("manager_test").name == "default-name"


async def test_wait_disabled_when_seconds_zero(tmp_path: Path) -> None:
    """Opt-out: callers that intentionally skip the wait (tests, processes
    that boot without a YAML on purpose) get the old fast-path behaviour."""
    missing = tmp_path / "missing.yaml"
    cm = ConfigurationManager(
        config_path=str(missing), wait_for_config_seconds=0.0,
    )

    started = time.monotonic()
    await cm.initialize()
    elapsed = time.monotonic() - started

    assert elapsed < 0.1, f"wait should have been skipped; elapsed={elapsed:.3f}s"
    assert cm.get_component("manager_test").name == "default-name"


async def test_late_registered_component_materializes_from_loaded_yaml(
    tmp_path: Path,
) -> None:
    """A component class that registers AFTER the manager loaded its
    YAML must still materialize from the loaded payload on first
    access.

    Regression: component classes register at module-import time; an
    agent worker loaded its config before importing ``KnowledgeConfig``,
    so the ``knowledge:`` YAML section was silently dropped and the
    ingest ran against in-memory stores (2026-08-03 forensic)."""

    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text(
        "late_registered_test:\n"
        "  endpoint: http://qdrant:6333\n"
    )
    cm = ConfigurationManager(config_path=str(yaml_file))
    await cm.initialize()

    # Simulate the late import: the class registers only now.
    @register_polymathera_config(path="late_registered_test")
    class _LateRegisteredConfig(ConfigComponent):
        endpoint: str = Field(default="")

    cfg = cm.get_component("late_registered_test")
    assert isinstance(cfg, _LateRegisteredConfig)
    assert cfg.endpoint == "http://qdrant:6333"


async def test_get_component_or_default_warns_on_unregistered_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """When the manager is initialized but the requested path resolves
    to nothing of the requested type, the defaults fallback must be
    LOUD — a silently-defaulted config was the root cause of the
    lost-ingest incident."""

    import logging
    from unittest.mock import patch

    from polymathera.colony.distributed.config.manager import (
        get_component_or_default,
    )

    class _UnregisteredConfig(ConfigComponent):
        value: str = Field(default="d")

    cm = ConfigurationManager(config_path=None)
    await cm.initialize()

    class _FakePolymathera:
        config_manager = cm

    with patch(
        "polymathera.colony.distributed.get_polymathera",
        return_value=_FakePolymathera(),
    ), caplog.at_level(logging.WARNING):
        cfg = get_component_or_default("nonexistent_path", _UnregisteredConfig)

    assert isinstance(cfg, _UnregisteredConfig)
    assert any(
        "NOT being honoured" in rec.message for rec in caplog.records
    )
