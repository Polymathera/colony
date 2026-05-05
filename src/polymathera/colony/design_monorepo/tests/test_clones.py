"""Tests for :mod:`polymathera.colony.design_monorepo.clones` and the
per-agent clone resolution path on
:class:`_DesignMonorepoCapabilityBase`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from polymathera.colony.design_monorepo.clones import (
    PER_AGENT_SUBDIR,
    SHARED_CLONES_SUBDIR,
    resolve_clone_path,
    shared_root,
)


def test_per_agent_path_includes_agent_id_and_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLONY_SHARED_ROOT", "/tmp/colony_test_shared")
    agent = MagicMock()
    agent.agent_id = "agent-A"
    path = resolve_clone_path(agent=agent, scope_id="my-design", read_only=False)
    assert path == Path(
        "/tmp/colony_test_shared",
    ) / PER_AGENT_SUBDIR / "agent-A" / "clones" / "my-design"


def test_two_agents_with_same_scope_get_distinct_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLONY_SHARED_ROOT", "/tmp/colony_test_shared")
    a = MagicMock(); a.agent_id = "agent-A"
    b = MagicMock(); b.agent_id = "agent-B"
    pa = resolve_clone_path(agent=a, scope_id="x", read_only=False)
    pb = resolve_clone_path(agent=b, scope_id="x", read_only=False)
    assert pa != pb
    assert pa.parent.parent.name == "agent-A"
    assert pb.parent.parent.name == "agent-B"


def test_shared_read_only_path_does_not_depend_on_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLONY_SHARED_ROOT", "/tmp/colony_test_shared")
    a = MagicMock(); a.agent_id = "agent-A"
    b = MagicMock(); b.agent_id = "agent-B"
    pa = resolve_clone_path(agent=a, scope_id="design", read_only=True)
    pb = resolve_clone_path(agent=b, scope_id="design", read_only=True)
    assert pa == pb
    assert pa == Path("/tmp/colony_test_shared") / SHARED_CLONES_SUBDIR / "design"


def test_writable_clone_requires_an_agent() -> None:
    with pytest.raises(ValueError, match="writable clones require"):
        resolve_clone_path(agent=None, scope_id="x", read_only=False)


def test_shared_root_default_is_colony_shared_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COLONY_SHARED_ROOT", raising=False)
    assert shared_root() == Path("/mnt/shared")
