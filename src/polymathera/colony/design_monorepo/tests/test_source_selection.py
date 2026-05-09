"""Tests for the per-colony source-selection helpers.

Storage flows through ``PolymatheraApp.get_state_manager`` in
production. We stub the chain with an in-memory dict-backed fake
:class:`StateManager` so the unit tests don't need a live state
backend; the real round-trip is covered by the existing
``state_management`` integration tests.
"""

from __future__ import annotations

import pytest

from polymathera.colony.design_monorepo import source_selection as ss_mod
from polymathera.colony.design_monorepo.source_selection import (
    SourceSelection,
)


class _FakeStateManager:
    """Mirrors :class:`StateManager` shape for the two methods the
    helpers call: ``read_transaction`` / ``write_transaction``. Storage
    is a dict shared across instances created with the same key, so
    "set in one fake, read in another" round-trips."""

    def __init__(
        self,
        store: dict[str, dict],
        state_type: type[SourceSelection],
        state_key: str,
    ) -> None:
        self._store = store
        self._state_type = state_type
        self._state_key = state_key

    def _load(self) -> SourceSelection:
        raw = self._store.get(self._state_key)
        if raw is None:
            return self._state_type()
        return self._state_type.model_validate(raw)

    async def read_transaction(self):
        yield self._load()

    async def write_transaction(self):
        state = self._load()
        yield state
        self._store[self._state_key] = state.model_dump()


class _FakePolymathera:
    """Minimal stand-in for :class:`PolymatheraApp` exposing only the
    one method the helpers need. Caches per-key fakes so successive
    calls observe each other's writes (same contract as the real
    :meth:`PolymatheraApp.get_state_manager`)."""

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self._cache: dict[str, _FakeStateManager] = {}

    async def get_state_manager(
        self, *, state_type, state_key: str,
    ) -> _FakeStateManager:
        sm = self._cache.get(state_key)
        if sm is None:
            sm = _FakeStateManager(self.store, state_type, state_key)
            self._cache[state_key] = sm
        return sm


@pytest.fixture
def fake_polymathera(monkeypatch) -> _FakePolymathera:
    fake = _FakePolymathera()
    monkeypatch.setattr(ss_mod, "get_polymathera", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_unset_selection_returns_none(
    fake_polymathera: _FakePolymathera,
) -> None:
    """Default state for a fresh colony: no key in the store →
    ``SourceSelection()`` (``enabled=None``) → "all rows enabled"."""
    assert await ss_mod.list_enabled_vcm_sources("c-1") is None
    assert await ss_mod.list_enabled_knowledge_sources("c-1") is None


@pytest.mark.asyncio
async def test_set_then_list_round_trips(
    fake_polymathera: _FakePolymathera,
) -> None:
    await ss_mod.set_enabled_knowledge_sources("c-1", ["literature", "books"])
    assert (
        await ss_mod.list_enabled_knowledge_sources("c-1")
        == ["literature", "books"]
    )


@pytest.mark.asyncio
async def test_set_none_clears_persisted_value(
    fake_polymathera: _FakePolymathera,
) -> None:
    """Setting ``None`` after a non-None value resets to the default
    ("all enabled")."""
    await ss_mod.set_enabled_vcm_sources("c-1", ["a"])
    await ss_mod.set_enabled_vcm_sources("c-1", None)
    assert await ss_mod.list_enabled_vcm_sources("c-1") is None


@pytest.mark.asyncio
async def test_vcm_and_knowledge_use_independent_keys(
    fake_polymathera: _FakePolymathera,
) -> None:
    """Per the orthogonality contract: setting one side does not
    affect the other."""
    await ss_mod.set_enabled_vcm_sources("c-1", ["v1", "v2"])
    await ss_mod.set_enabled_knowledge_sources("c-1", ["k1"])
    assert await ss_mod.list_enabled_vcm_sources("c-1") == ["v1", "v2"]
    assert await ss_mod.list_enabled_knowledge_sources("c-1") == ["k1"]


@pytest.mark.asyncio
async def test_per_colony_isolation(
    fake_polymathera: _FakePolymathera,
) -> None:
    """Different colonies don't share state."""
    await ss_mod.set_enabled_knowledge_sources("c-1", ["a"])
    await ss_mod.set_enabled_knowledge_sources("c-2", ["b"])
    assert await ss_mod.list_enabled_knowledge_sources("c-1") == ["a"]
    assert await ss_mod.list_enabled_knowledge_sources("c-2") == ["b"]
