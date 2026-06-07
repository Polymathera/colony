"""Tests for the cluster-shared mission spawn-gate ledger.

Exercises the :class:`MissionExecutionLedger` against an in-memory
``StateStorageBackend`` so the StateManager-mediated semantics
(transactional admit, version-stamped writes) get pinned without
spinning up Redis.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from polymathera.colony.agents.configs import (
    MissionConcurrencyScope,
    MissionExecutionPolicy,
)
from polymathera.colony.agents.missions.execution_ledger import (
    AdmissionAllowed,
    AdmissionRejected,
    AdmissionReturnExisting,
    MissionExecutionLedger,
    MissionLedgerState,
    RunningMissionKey,
)
from polymathera.colony.distributed.state_management import StateManager
from polymathera.colony.distributed.stores.state_base import (
    StateStorageBackend,
    StateStorageBackendFactory,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# In-memory StateStorageBackend — mirrors the pattern used by
# distributed/config/tests/test_overlays.py.
# ---------------------------------------------------------------------------


class _InMemConfig(BaseModel):
    pass


class _InMemBackend(StateStorageBackend):
    """Single-process backend that mimics Redis's compare-and-swap
    semantics. Bound to a single test by construction (no module-
    level mutable state)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, int]] = {}
        self._lock = asyncio.Lock()

    async def get_with_version(self, key):
        async with self._lock:
            return self._store.get(key, (None, 0))

    async def compare_and_swap(self, key, value, version):
        async with self._lock:
            entry = self._store.get(key)
            current = entry[1] if entry is not None else 0
            if current != version:
                return False
            self._store[key] = (value, version + 1)
            return True

    async def cleanup(self, key):
        async with self._lock:
            self._store.pop(key, None)


class _InMemFactory(StateStorageBackendFactory):
    def __init__(self) -> None:
        self.backend = _InMemBackend()

    def create_backend(self, config):
        return self.backend


@pytest.fixture
async def ledger() -> MissionExecutionLedger:
    """Fresh, isolated ledger backed by an in-memory storage. Every
    test gets its own factory so state never leaks."""

    sm = StateManager(
        MissionLedgerState,
        state_key=MissionLedgerState.get_state_key("test"),
        config=_InMemConfig(),
        factory=_InMemFactory(),
    )
    await sm.initialize()
    return MissionExecutionLedger(state_manager=sm)


def _key(
    *,
    scope: MissionConcurrencyScope = MissionConcurrencyScope.SESSION,
    scope_id: str = "session_test",
    mission_type: str = "project_planning",
) -> RunningMissionKey:
    return RunningMissionKey(
        scope=scope, scope_id=scope_id, mission_type=mission_type,
    )


# ---------------------------------------------------------------------------
# RunningMissionKey storage round-trip
# ---------------------------------------------------------------------------


def test_running_mission_key_storage_round_trip() -> None:
    """The string-encoding round-trips so the SharedState dict can
    use bare strings as keys (JSON-serialisable) without losing
    typed access on the way back out."""

    original = RunningMissionKey(
        scope=MissionConcurrencyScope.SESSION,
        scope_id="session_abc",
        mission_type="project_planning",
    )
    encoded = original.to_storage_key()
    decoded = RunningMissionKey.from_storage_key(encoded)
    assert decoded == original


def test_running_mission_key_scope_id_may_contain_dashes() -> None:
    """Real scope_ids (``session_<hex>``, ``colony_<hex>``) contain
    underscores and hex — not pipe characters — so the
    pipe-delimited encoding is unambiguous."""

    key = RunningMissionKey(
        scope=MissionConcurrencyScope.COLONY,
        scope_id="colony_a1b2c3d4e5f6",
        mission_type="project_planning",
    )
    assert RunningMissionKey.from_storage_key(key.to_storage_key()) == key


# ---------------------------------------------------------------------------
# chains_with_modes
# ---------------------------------------------------------------------------


async def test_chains_with_modes_returns_existing_on_chained_mode(
    ledger: MissionExecutionLedger,
) -> None:
    """When the second spawn declares a mode the running mission's
    ``chains_with_modes`` enumerates, the gate hands back the
    running agent_id — the deterministic answer to the Q1
    one-coordinator-per-mode LLM drift."""

    policy = MissionExecutionPolicy(
        chains_with_modes=["bootstrap", "refresh", "assignments"],
    )
    d1, r1 = await ledger.try_admit(
        key=_key(), mode="bootstrap", policy=policy,
    )
    assert isinstance(d1, AdmissionAllowed)
    assert r1 is not None
    await ledger.register(
        reservation_id=r1, agent_id="agent-A", mode="bootstrap",
    )

    for mode in ("refresh", "assignments"):
        d, _ = await ledger.try_admit(
            key=_key(), mode=mode, policy=policy,
        )
        assert isinstance(d, AdmissionReturnExisting), (mode, d)
        assert d.agent_id == "agent-A"
        assert "auto-chain" in d.reason


async def test_chains_with_modes_does_not_fire_for_unlisted_mode(
    ledger: MissionExecutionLedger,
) -> None:
    """A mode NOT in ``chains_with_modes`` falls through to the
    concurrency-cap check rather than coupling with the chain."""

    policy = MissionExecutionPolicy(
        chains_with_modes=["bootstrap"],
        max_concurrent_instances=2,
    )
    _, r1 = await ledger.try_admit(
        key=_key(), mode="bootstrap", policy=policy,
    )
    await ledger.register(
        reservation_id=r1, agent_id="agent-A", mode="bootstrap",
    )
    # "audit" mode is not in chains; cap is 2; this is the second
    # in-flight slot → still admitted.
    d2, _ = await ledger.try_admit(
        key=_key(), mode="audit", policy=policy,
    )
    assert isinstance(d2, AdmissionAllowed)


# ---------------------------------------------------------------------------
# max_concurrent_instances + on_concurrency_violation
# ---------------------------------------------------------------------------


async def test_max_concurrent_one_rejects_second_with_reject_policy(
    ledger: MissionExecutionLedger,
) -> None:
    """``max_concurrent_instances=1`` + ``on_concurrency_violation=
    reject`` is the safe default for missions with side effects."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1, on_concurrency_violation="reject",
    )
    _, r = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    await ledger.register(
        reservation_id=r, agent_id="agent-A", mode=None,
    )
    d, _ = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionRejected)
    assert "currently in flight" in d.reason
    assert d.suggested_action  # non-empty hint for the LLM


async def test_max_concurrent_one_returns_existing_with_return_policy(
    ledger: MissionExecutionLedger,
) -> None:
    """``on_concurrency_violation=return_existing`` is the idempotent
    shape: hands back the already-running agent."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1,
        on_concurrency_violation="return_existing",
    )
    _, r = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    await ledger.register(
        reservation_id=r, agent_id="agent-A", mode=None,
    )
    d, _ = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionReturnExisting)
    assert d.agent_id == "agent-A"


async def test_unbounded_cap_admits_unlimited(
    ledger: MissionExecutionLedger,
) -> None:
    """``max_concurrent_instances=None`` lets every spawn through."""

    policy = MissionExecutionPolicy(max_concurrent_instances=None)
    for n in range(5):
        d, r = await ledger.try_admit(
            key=_key(), mode=None, policy=policy,
        )
        assert isinstance(d, AdmissionAllowed), n
        await ledger.register(
            reservation_id=r, agent_id=f"agent-{n}", mode=None,
        )


# ---------------------------------------------------------------------------
# Bucket isolation
# ---------------------------------------------------------------------------


async def test_different_mission_types_have_independent_buckets(
    ledger: MissionExecutionLedger,
) -> None:
    """One running ``project_planning`` doesn't block ``impact``."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    _, r = await ledger.try_admit(
        key=_key(mission_type="project_planning"),
        mode=None, policy=policy,
    )
    await ledger.register(
        reservation_id=r, agent_id="agent-PP", mode=None,
    )
    d, _ = await ledger.try_admit(
        key=_key(mission_type="impact"),
        mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionAllowed)


async def test_different_scope_ids_have_independent_buckets(
    ledger: MissionExecutionLedger,
) -> None:
    """Two sessions can each run their own
    ``project_planning`` coordinator under
    ``concurrency_scope=SESSION``."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    _, r = await ledger.try_admit(
        key=_key(scope_id="session_A"), mode=None, policy=policy,
    )
    await ledger.register(
        reservation_id=r, agent_id="agent-A", mode=None,
    )
    d, _ = await ledger.try_admit(
        key=_key(scope_id="session_B"), mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionAllowed)


# ---------------------------------------------------------------------------
# Reservations + lifecycle
# ---------------------------------------------------------------------------


async def test_reservation_counts_toward_cap_until_register(
    ledger: MissionExecutionLedger,
) -> None:
    """A reservation made but not yet registered still consumes a
    slot; this closes the parallel-spawn race where two callers both
    see an empty bucket and both admit."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    d1, r1 = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d1, AdmissionAllowed)
    assert r1 is not None

    d2, _ = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d2, AdmissionRejected)


async def test_release_reservation_returns_slot_without_registering(
    ledger: MissionExecutionLedger,
) -> None:
    """A failed spawn between admit and register releases the slot."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    _, r = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    snap = await ledger.snapshot()
    assert _key() not in snap or not snap[_key()]
    # The reservation is the only thing holding the slot; releasing
    # it frees the slot for a fresh admit.
    await ledger.release_reservation(r)
    d, _ = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionAllowed)


async def test_release_reservation_is_idempotent(
    ledger: MissionExecutionLedger,
) -> None:
    """Double-release is a no-op (callers don't need to track state)."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    _, r = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    await ledger.release_reservation(r)
    await ledger.release_reservation(r)  # no raise


async def test_unregister_drops_entry_and_frees_slot(
    ledger: MissionExecutionLedger,
) -> None:
    """Coordinator termination releases the bucket entry so the next
    spawn admits."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1, on_concurrency_violation="reject",
    )
    _, r = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    await ledger.register(
        reservation_id=r, agent_id="agent-A", mode=None,
    )
    d, _ = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionRejected)

    await ledger.unregister("agent-A")
    d2, _ = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    assert isinstance(d2, AdmissionAllowed)


async def test_unregister_unknown_agent_id_is_noop(
    ledger: MissionExecutionLedger,
) -> None:
    """Untracked agents (legacy spawns without a mission_type) can
    still call unregister — no raise, no state change."""

    await ledger.unregister("agent-does-not-exist")
    snap = await ledger.snapshot()
    assert snap == {}


async def test_register_after_release_raises(
    ledger: MissionExecutionLedger,
) -> None:
    """A misuse: releasing a reservation then trying to register
    against the same id is a programmer error worth surfacing."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    _, r = await ledger.try_admit(
        key=_key(), mode=None, policy=policy,
    )
    await ledger.release_reservation(r)
    with pytest.raises(KeyError):
        await ledger.register(
            reservation_id=r, agent_id="agent-A", mode=None,
        )


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


async def test_snapshot_exposes_current_state(
    ledger: MissionExecutionLedger,
) -> None:
    """``snapshot`` returns a typed copy of the live ledger."""

    policy = MissionExecutionPolicy(max_concurrent_instances=2)
    _, r1 = await ledger.try_admit(
        key=_key(), mode="bootstrap", policy=policy,
    )
    _, r2 = await ledger.try_admit(
        key=_key(), mode="refresh", policy=policy,
    )
    await ledger.register(
        reservation_id=r1, agent_id="agent-A", mode="bootstrap",
    )
    await ledger.register(
        reservation_id=r2, agent_id="agent-B", mode="refresh",
    )
    snap = await ledger.snapshot()
    assert _key() in snap
    agent_ids = {entry.agent_id for entry in snap[_key()]}
    assert agent_ids == {"agent-A", "agent-B"}


# ---------------------------------------------------------------------------
# Concurrent admits — pin the parallel-spawn race is closed
# ---------------------------------------------------------------------------


async def test_parallel_admits_under_cap_one_serialise(
    ledger: MissionExecutionLedger,
) -> None:
    """Two concurrent ``try_admit`` calls against a cap=1 policy
    end up with at most one ``AdmissionAllowed`` — the
    compare-and-swap inside the write_transaction closes the race.
    """

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1, on_concurrency_violation="reject",
    )

    results = await asyncio.gather(
        ledger.try_admit(key=_key(), mode=None, policy=policy),
        ledger.try_admit(key=_key(), mode=None, policy=policy),
    )
    allowed = [d for d, _ in results if isinstance(d, AdmissionAllowed)]
    rejected = [d for d, _ in results if isinstance(d, AdmissionRejected)]
    assert len(allowed) == 1, [type(d).__name__ for d, _ in results]
    assert len(rejected) == 1
