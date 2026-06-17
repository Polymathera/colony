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
    AdmissionAwait,
    AdmissionRejected,
    AdmissionReturnExisting,
    MissionExecutionLedger,
    MissionLedgerState,
    PendingReservation,
    RunningMissionEntry,
    RunningMissionKey,
    SpawnOutcome,
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
# Deleted-field invariants — pin that the new minimal model stays minimal.
# ---------------------------------------------------------------------------


def test_chains_with_modes_field_does_not_exist() -> None:
    """``chains_with_modes`` was the symptom of "treat mode as a
    separate spawn key" — deleted in favour of mode-agnostic cap +
    return_existing. Pin its absence so a future refactor can't
    quietly resurrect the field."""

    assert "chains_with_modes" not in MissionExecutionPolicy.model_fields


def test_idempotent_field_does_not_exist() -> None:
    """``idempotent`` was advisory only; the cap + on_concurrency_violation
    pair already expresses every gate decision the field claimed to
    drive. Pin its absence."""

    assert "idempotent" not in MissionExecutionPolicy.model_fields


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
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    await ledger.register(
        reservation_id=d.reservation_id, agent_id="agent-A", mode=None,
    )
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d2, AdmissionRejected)
    assert "currently in flight" in d2.reason
    assert d2.suggested_action  # non-empty hint for the LLM


async def test_max_concurrent_one_returns_existing_with_return_policy(
    ledger: MissionExecutionLedger,
) -> None:
    """``on_concurrency_violation=return_existing`` hands back the
    already-running agent."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1,
        on_concurrency_violation="return_existing",
    )
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    await ledger.register(
        reservation_id=d.reservation_id, agent_id="agent-A", mode=None,
    )
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d2, AdmissionReturnExisting)
    assert d2.agent_id == "agent-A"


async def test_return_existing_treats_modes_as_one_coordinator(
    ledger: MissionExecutionLedger,
) -> None:
    """With cap=1 + return_existing, the gate is mode-agnostic: every
    subsequent spawn — regardless of mode — resolves to the one
    running coordinator. Replaces the old chains_with_modes behaviour
    without a per-mode allow-list."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1,
        concurrency_scope=MissionConcurrencyScope.SESSION,
        on_concurrency_violation="return_existing",
    )
    d = await ledger.try_admit(key=_key(), mode="bootstrap", policy=policy)
    assert isinstance(d, AdmissionAllowed)
    await ledger.register(
        reservation_id=d.reservation_id, agent_id="agent-A",
        mode="bootstrap",
    )
    for mode in ("refresh", "assignments", "decompose"):
        d_other = await ledger.try_admit(
            key=_key(), mode=mode, policy=policy,
        )
        assert isinstance(d_other, AdmissionReturnExisting), (mode, d_other)
        assert d_other.agent_id == "agent-A"


async def test_unbounded_cap_admits_unlimited(
    ledger: MissionExecutionLedger,
) -> None:
    """``max_concurrent_instances=None`` lets every spawn through."""

    policy = MissionExecutionPolicy(max_concurrent_instances=None)
    for n in range(5):
        d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
        assert isinstance(d, AdmissionAllowed), n
        await ledger.register(
            reservation_id=d.reservation_id,
            agent_id=f"agent-{n}", mode=None,
        )


# ---------------------------------------------------------------------------
# Bucket isolation
# ---------------------------------------------------------------------------


async def test_different_mission_types_have_independent_buckets(
    ledger: MissionExecutionLedger,
) -> None:
    """One running ``project_planning`` doesn't block ``impact``."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    d = await ledger.try_admit(
        key=_key(mission_type="project_planning"),
        mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionAllowed)
    await ledger.register(
        reservation_id=d.reservation_id, agent_id="agent-PP", mode=None,
    )
    d2 = await ledger.try_admit(
        key=_key(mission_type="impact"), mode=None, policy=policy,
    )
    assert isinstance(d2, AdmissionAllowed)


async def test_different_scope_ids_have_independent_buckets(
    ledger: MissionExecutionLedger,
) -> None:
    """Two sessions can each run their own ``project_planning``
    coordinator under ``concurrency_scope=SESSION``."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    d = await ledger.try_admit(
        key=_key(scope_id="session_A"), mode=None, policy=policy,
    )
    assert isinstance(d, AdmissionAllowed)
    await ledger.register(
        reservation_id=d.reservation_id, agent_id="agent-A", mode=None,
    )
    d2 = await ledger.try_admit(
        key=_key(scope_id="session_B"), mode=None, policy=policy,
    )
    assert isinstance(d2, AdmissionAllowed)


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
    d1 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d1, AdmissionAllowed)
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d2, AdmissionRejected)


async def test_reservation_in_admit_register_window_returns_await(
    ledger: MissionExecutionLedger,
) -> None:
    """Hole 2 — during the admit → register window, a sibling
    try_admit under return_existing policy now returns AdmissionAwait
    (was AdmissionRejected pre-Change-2). The Await carries the
    blocking reservation_id so the caller can correlate."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1,
        on_concurrency_violation="return_existing",
    )
    d1 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d1, AdmissionAllowed)

    # d1 is still a pending reservation — never called register.
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d2, AdmissionAwait)
    assert d2.reservation_id == d1.reservation_id

    # After register lands, the next admit sees the running slot and
    # gets ReturnExisting (not Await).
    await ledger.register(
        reservation_id=d1.reservation_id, agent_id="agent-A", mode=None,
    )
    d3 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d3, AdmissionReturnExisting)
    assert d3.agent_id == "agent-A"


async def test_release_reservation_returns_slot_without_registering(
    ledger: MissionExecutionLedger,
) -> None:
    """A failed spawn between admit and register releases the slot."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    # snapshot excludes pending reservations
    snap = await ledger.snapshot()
    assert _key() not in snap or not snap[_key()]
    await ledger.release_reservation(d.reservation_id)
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d2, AdmissionAllowed)


async def test_release_reservation_is_idempotent(
    ledger: MissionExecutionLedger,
) -> None:
    """Double-release is a no-op (callers don't need to track state)."""

    policy = MissionExecutionPolicy(max_concurrent_instances=1)
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    await ledger.release_reservation(d.reservation_id)
    await ledger.release_reservation(d.reservation_id)  # no raise


async def test_unregister_drops_entry_and_frees_slot(
    ledger: MissionExecutionLedger,
) -> None:
    """Coordinator termination releases the slot."""

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1, on_concurrency_violation="reject",
    )
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    await ledger.register(
        reservation_id=d.reservation_id, agent_id="agent-A", mode=None,
    )
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d2, AdmissionRejected)

    await ledger.unregister("agent-A")
    d3 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d3, AdmissionAllowed)


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
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    await ledger.release_reservation(d.reservation_id)
    with pytest.raises(KeyError):
        await ledger.register(
            reservation_id=d.reservation_id, agent_id="agent-A", mode=None,
        )


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


async def test_snapshot_exposes_current_state(
    ledger: MissionExecutionLedger,
) -> None:
    """``snapshot`` returns a typed copy of the live ledger,
    excluding pending reservations."""

    policy = MissionExecutionPolicy(max_concurrent_instances=2)
    d1 = await ledger.try_admit(key=_key(), mode="bootstrap", policy=policy)
    d2 = await ledger.try_admit(key=_key(), mode="refresh", policy=policy)
    assert isinstance(d1, AdmissionAllowed)
    assert isinstance(d2, AdmissionAllowed)
    await ledger.register(
        reservation_id=d1.reservation_id, agent_id="agent-A",
        mode="bootstrap",
    )
    await ledger.register(
        reservation_id=d2.reservation_id, agent_id="agent-B",
        mode="refresh",
    )
    snap = await ledger.snapshot()
    assert _key() in snap
    agent_ids = {entry.agent_id for entry in snap[_key()]}
    assert agent_ids == {"agent-A", "agent-B"}


async def test_snapshot_excludes_pending_reservations(
    ledger: MissionExecutionLedger,
) -> None:
    """Snapshot is the planner's "what coordinators are alive?" view.
    Pending reservations are a gate internal — never surfaced."""

    policy = MissionExecutionPolicy(max_concurrent_instances=2)
    d1 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    d2 = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d1, AdmissionAllowed)
    assert isinstance(d2, AdmissionAllowed)
    # Only register the second one; first remains pending.
    await ledger.register(
        reservation_id=d2.reservation_id, agent_id="agent-B", mode=None,
    )
    snap = await ledger.snapshot()
    assert _key() in snap
    agent_ids = {entry.agent_id for entry in snap[_key()]}
    assert agent_ids == {"agent-B"}


# ---------------------------------------------------------------------------
# list_for_scope — read primitive backing list_spawned_missions
# ---------------------------------------------------------------------------


async def test_list_for_scope_filters_by_mission_type(
    ledger: MissionExecutionLedger,
) -> None:
    """When ``mission_type`` is provided, only entries whose KEY's
    mission_type matches are returned — even when other mission types
    are live in the same scope."""

    policy = MissionExecutionPolicy(max_concurrent_instances=2)
    d1 = await ledger.try_admit(
        key=_key(mission_type="project_planning"),
        mode="bootstrap", policy=policy,
    )
    d2 = await ledger.try_admit(
        key=_key(mission_type="impact"), mode=None, policy=policy,
    )
    assert isinstance(d1, AdmissionAllowed)
    assert isinstance(d2, AdmissionAllowed)
    await ledger.register(
        reservation_id=d1.reservation_id, agent_id="agent-PP",
        mode="bootstrap",
    )
    await ledger.register(
        reservation_id=d2.reservation_id, agent_id="agent-IM", mode=None,
    )

    pp_only = await ledger.list_for_scope(
        scope=MissionConcurrencyScope.SESSION,
        scope_id="session_test",
        mission_type="project_planning",
    )
    assert len(pp_only) == 1
    key, entry = pp_only[0]
    assert key.mission_type == "project_planning"
    assert entry.agent_id == "agent-PP"
    assert entry.mode == "bootstrap"


async def test_list_for_scope_returns_all_for_scope_when_mission_type_none(
    ledger: MissionExecutionLedger,
) -> None:
    """``mission_type=None`` returns every live entry under
    ``(scope, scope_id)`` regardless of registry key."""

    policy = MissionExecutionPolicy(max_concurrent_instances=2)
    d1 = await ledger.try_admit(
        key=_key(mission_type="project_planning"),
        mode="bootstrap", policy=policy,
    )
    d2 = await ledger.try_admit(
        key=_key(mission_type="impact"), mode=None, policy=policy,
    )
    assert isinstance(d1, AdmissionAllowed)
    assert isinstance(d2, AdmissionAllowed)
    await ledger.register(
        reservation_id=d1.reservation_id, agent_id="agent-PP",
        mode="bootstrap",
    )
    await ledger.register(
        reservation_id=d2.reservation_id, agent_id="agent-IM", mode=None,
    )
    d3 = await ledger.try_admit(
        key=_key(
            scope_id="session_other", mission_type="project_planning",
        ),
        mode=None, policy=policy,
    )
    assert isinstance(d3, AdmissionAllowed)
    await ledger.register(
        reservation_id=d3.reservation_id, agent_id="agent-OTHER",
        mode=None,
    )

    all_in_scope = await ledger.list_for_scope(
        scope=MissionConcurrencyScope.SESSION,
        scope_id="session_test",
        mission_type=None,
    )
    agent_ids = {entry.agent_id for _, entry in all_in_scope}
    assert agent_ids == {"agent-PP", "agent-IM"}
    mission_types = {key.mission_type for key, _ in all_in_scope}
    assert mission_types == {"project_planning", "impact"}


async def test_list_for_scope_empty_when_no_entries(
    ledger: MissionExecutionLedger,
) -> None:
    """A scope with nothing running returns ``[]`` — not a raise."""

    result = await ledger.list_for_scope(
        scope=MissionConcurrencyScope.SESSION,
        scope_id="session_empty",
        mission_type=None,
    )
    assert result == []

    result_filtered = await ledger.list_for_scope(
        scope=MissionConcurrencyScope.SESSION,
        scope_id="session_empty",
        mission_type="project_planning",
    )
    assert result_filtered == []


async def test_list_for_scope_excludes_pending_reservations(
    ledger: MissionExecutionLedger,
) -> None:
    """Pending reservations don't appear in the planner's read view.
    Otherwise the LLM would see "coordinator already running" before
    the spawn actually landed and stop chaining work."""

    policy = MissionExecutionPolicy(max_concurrent_instances=2)
    d = await ledger.try_admit(key=_key(), mode=None, policy=policy)
    assert isinstance(d, AdmissionAllowed)
    # Never call register — leave d as a pending reservation.

    result = await ledger.list_for_scope(
        scope=MissionConcurrencyScope.SESSION,
        scope_id="session_test",
        mission_type=None,
    )
    assert result == []


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
    allowed = [d for d in results if isinstance(d, AdmissionAllowed)]
    rejected = [d for d in results if isinstance(d, AdmissionRejected)]
    assert len(allowed) == 1, [type(d).__name__ for d in results]
    assert len(rejected) == 1


async def test_three_parallel_admits_under_return_existing_converge(
    ledger: MissionExecutionLedger,
) -> None:
    """Three parallel ``try_admit`` calls (e.g. SessionAgent emits
    bootstrap+refresh+assignments concurrently) under cap=1 +
    return_existing converge: exactly one Allowed; the rest are
    Await (slot held by reservation) or ReturnExisting (slot bound
    after register lands). No Rejected — that was the Hole 2 bug.
    """

    policy = MissionExecutionPolicy(
        max_concurrent_instances=1,
        on_concurrency_violation="return_existing",
    )

    results = await asyncio.gather(
        ledger.try_admit(key=_key(), mode="bootstrap", policy=policy),
        ledger.try_admit(key=_key(), mode="refresh", policy=policy),
        ledger.try_admit(key=_key(), mode="assignments", policy=policy),
    )
    allowed = [d for d in results if isinstance(d, AdmissionAllowed)]
    awaits = [d for d in results if isinstance(d, AdmissionAwait)]
    rejected = [d for d in results if isinstance(d, AdmissionRejected)]
    assert len(allowed) == 1
    assert len(rejected) == 0, (
        "Hole 2: parallel admit under return_existing must never "
        "reject — the slot-holder is recoverable."
    )
    # The other two land as Await (no running slot yet) referencing
    # the one allowed reservation.
    assert len(awaits) == 2
    assert all(a.reservation_id == allowed[0].reservation_id for a in awaits)

# ---------------------------------------------------------------------------
# SpawnOutcome typed shape — Bucket A.1 / Fix F1 prevention. Pins the
# discriminator contract returned by spawn_mission / admit_and_spawn
# so the LLM branches on outcome instead of the legacy created /
# mission_gate pair (which were the lie that drove the F1 forensic
# failure — see refine_github_issues_failure_fixes_plan.md).
# ---------------------------------------------------------------------------


def test_spawn_outcome_spawned_dumps_with_computed_fields() -> None:
    out = SpawnOutcome(
        outcome="spawned",
        mission_type="m",
        coordinator_class="pkg.C",
        label="L",
        agent_id="agent-1",
    ).model_dump()
    assert out["outcome"] == "spawned"
    assert out["created"] is True
    assert out["mission_gate"] is None
    assert out["agent_id"] == "agent-1"
    # Per-outcome optional fields are present + None so the schema is
    # uniform across variants — the LLM never has to KeyError-check.
    assert out["error"] is None
    assert out["reason"] is None
    assert out["suggested_action"] is None


def test_spawn_outcome_return_existing_back_compat() -> None:
    out = SpawnOutcome(
        outcome="return_existing",
        mission_type="m",
        coordinator_class="pkg.C",
        label="L",
        agent_id="agent-1",
        reason="cap reached",
    ).model_dump()
    assert out["outcome"] == "return_existing"
    assert out["created"] is False
    assert out["mission_gate"] == "return_existing"
    assert out["agent_id"] == "agent-1"
    assert out["reason"] == "cap reached"


def test_spawn_outcome_rejected_carries_suggested_action() -> None:
    out = SpawnOutcome(
        outcome="rejected",
        mission_type="m",
        coordinator_class="pkg.C",
        label="L",
        error="At most 1 concurrent",
        suggested_action="Wait for the running mission to complete",
    ).model_dump()
    assert out["outcome"] == "rejected"
    assert out["created"] is False
    assert out["mission_gate"] == "rejected"
    assert out["agent_id"] is None
    assert out["error"] == "At most 1 concurrent"
    assert out["suggested_action"] == "Wait for the running mission to complete"


def test_spawn_outcome_error_has_no_mission_gate() -> None:
    """The error outcome is for failures BEFORE the gate (unknown
    mission_type, missing coordinator class, dispatch raised). The
    mission_gate field is None — the gate never decided anything."""
    out = SpawnOutcome(
        outcome="error",
        mission_type="m",
        coordinator_class="",
        label="",
        error="Unknown mission type 'm'",
    ).model_dump()
    assert out["outcome"] == "error"
    assert out["created"] is False
    assert out["mission_gate"] is None
    assert out["agent_id"] is None
    assert out["error"] == "Unknown mission type 'm'"
    assert out["suggested_action"] is None


def test_spawn_outcome_rejects_unknown_discriminator() -> None:
    """The Literal type bans typos in outcome values — a fifth case
    cannot be silently introduced."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        SpawnOutcome(
            outcome="queued",  # not in the Literal
            mission_type="m",
            coordinator_class="",
            label="",
        )


def test_spawn_outcome_is_frozen() -> None:
    """Each outcome is an immutable snapshot — callers cannot mutate
    the discriminator after the gate decided."""
    out = SpawnOutcome(
        outcome="spawned",
        mission_type="m",
        coordinator_class="pkg.C",
        label="L",
        agent_id="agent-1",
    )
    import pydantic
    with pytest.raises((pydantic.ValidationError, TypeError, AttributeError)):
        out.outcome = "rejected"  # type: ignore[misc]

