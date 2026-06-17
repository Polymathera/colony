"""Cluster-shared ledger of in-flight mission instances.

Owns the mission spawn-gate end-to-end:

- :class:`MissionLedgerState` — the :class:`SharedState` model
  persisted by :class:`StateManager` (Redis-backed by default), so
  every Ray worker sees the same view of "what mission coordinators
  are alive in this cluster right now". Covers every
  :class:`MissionConcurrencyScope` uniformly, including the cross-
  worker scopes (``GLOBAL`` / ``TENANT`` / ``COLONY``) that a
  process-local dict cannot enforce.

- :class:`MissionExecutionLedger` — thin wrapper around the
  :class:`StateManager` exposing the admission API
  (:meth:`try_admit` / :meth:`register` / :meth:`release_reservation`
  / :meth:`unregister` / :meth:`snapshot`). All mutations land in a
  single ``write_transaction()`` so the compare-and-swap closes the
  parallel-spawn race; readers use ``read_transaction()`` to avoid
  spurious version bumps.

- :func:`admit_and_spawn` — orchestration helper both spawn paths
  call (the chat-side ``SessionOrchestratorCapability.spawn_mission``
  and the REST-side ``routers/jobs.py::_run_job``). Centralises the
  admit → ``pool.create_agent`` → register flow so the spawn-gate
  concept never has to leak into the generic
  ``AgentPoolCapability.create_agent`` primitive.

Cross-references:

- ``colony/mission_and_action_guardrails_plan.md`` (Part 1) for the
  motivating failure modes and the policy schema.
- :class:`polymathera.colony.agents.configs.MissionExecutionPolicy`
  for the declarative knobs the gate enforces.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ...distributed.state_management import SharedState, StateManager
from ..configs import (
    MissionConcurrencyScope,
    MissionExecutionPolicy,
)

if TYPE_CHECKING:
    from ...distributed.ray_utils.serving.context import ExecutionContext
    from ..base import Agent
    from ..models import AgentMetadata
    from ..patterns.capabilities.agent_pool import AgentPoolCapability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory shapes — passed to + returned from the admission API.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunningMissionKey:
    """The bucket key the ledger groups in-flight missions under.

    Two missions land in the same bucket iff every field matches. The
    spawn gate's "is anything else of this kind running here?" check
    is just "is this bucket non-empty?".

    Frozen dataclass for in-memory ergonomics — the ledger
    serialises it to a string when crossing the
    :class:`SharedState` boundary because Pydantic JSON dicts only
    support string keys.
    """

    scope: MissionConcurrencyScope
    scope_id: str       # Concrete tenant/colony/session/agent id, or "global".
    mission_type: str   # Registry key (``project_planning`` etc.).

    def to_storage_key(self) -> str:
        """Serialised form used as a dict key inside
        :class:`MissionLedgerState`. Format guarantees round-trip
        (:meth:`from_storage_key`)."""

        return f"{self.scope.value}|{self.scope_id}|{self.mission_type}"

    @classmethod
    def from_storage_key(cls, key: str) -> "RunningMissionKey":
        scope_value, scope_id, mission_type = key.split("|", 2)
        return cls(
            scope=MissionConcurrencyScope(scope_value),
            scope_id=scope_id,
            mission_type=mission_type,
        )


class PendingReservation(BaseModel):
    """A slot held by an admit that hasn't yet bound to an agent_id.

    Counts toward ``max_concurrent_instances`` so two parallel
    admit-then-spawn races can't both see an empty slot list."""

    kind: Literal["reservation"] = "reservation"
    reservation_id: str
    reserved_at: float = Field(default_factory=time.time)


class RunningMissionEntry(BaseModel):
    """A slot bound to a live coordinator. Pydantic for JSON
    round-trip through :class:`StateManager`."""

    kind: Literal["running"] = "running"
    agent_id: str
    mode: str | None = None
    started_at: float = Field(default_factory=time.time)


# The unified slot type stored in MissionLedgerState.slots. Indexing
# pending + running in the same list (not two parallel dicts) closes
# the admit-then-register window where the bucket-only cap check
# undercounted in-flight work.
Slot = Annotated[
    PendingReservation | RunningMissionEntry,
    Field(discriminator="kind"),
]


@dataclass(frozen=True)
class AdmissionAllowed:
    """Spawn approved. ``reservation_id`` is what the caller passes
    to :meth:`MissionExecutionLedger.register` (or
    :meth:`release_reservation` on spawn failure)."""

    reservation_id: str
    kind: Literal["spawn"] = "spawn"


@dataclass(frozen=True)
class AdmissionReturnExisting:
    """A live coordinator already covers this spawn; hand back its
    ``agent_id`` instead of spawning anew."""

    agent_id: str
    reason: str
    kind: Literal["return_existing"] = "return_existing"


@dataclass(frozen=True)
class AdmissionAwait:
    """Cap reached, but the only thing holding the slot is a pending
    reservation under ``return_existing`` policy — the caller should
    wait for the reservation to resolve (either to a running
    coordinator the caller can return, or to a freed slot) rather
    than reject. Returned to keep the LLM-facing gate semantics
    monotonic: a race shouldn't surface as a different outcome than
    the deterministic path."""

    reservation_id: str
    reason: str
    kind: Literal["await"] = "await"


@dataclass(frozen=True)
class AdmissionRejected:
    """The spawn gate refuses the spawn."""

    reason: str
    suggested_action: str = ""
    kind: Literal["reject"] = "reject"


AdmissionDecision = (
    AdmissionAllowed
    | AdmissionReturnExisting
    | AdmissionAwait
    | AdmissionRejected
)


# ---------------------------------------------------------------------------
# SpawnOutcome — typed shape of the result that ``spawn_mission`` /
# ``admit_and_spawn`` return to LLM planners. The LLM branches on
# ``outcome``; the legacy ``created`` and ``mission_gate`` fields stay
# as computed properties for back-compat with the dict-shaped consumers
# that predate the typed discriminator.
# ---------------------------------------------------------------------------


class SpawnOutcome(BaseModel):
    """Discriminated-union return shape for ``spawn_mission``.

    The four outcomes map 1:1 onto the four ways a spawn can end
    (post-Change-2 admission gate + Pool dispatch). ``outcome`` is the
    canonical discriminator. The optional per-outcome fields are set
    only when the outcome warrants them; the LLM branches on ``outcome``
    BEFORE reading any of them.

    Legacy ``created`` (bool) and ``mission_gate`` (str | None) are
    computed properties: they appear in :meth:`model_dump` output so
    pre-typed-contract consumers keep working without code changes."""

    model_config = ConfigDict(frozen=True)

    outcome: Literal["spawned", "return_existing", "rejected", "error"]

    mission_type: str
    coordinator_class: str
    label: str

    agent_id: str | None = None
    """Set for ``spawned`` and ``return_existing``. The id the caller
    addresses the coordinator with."""

    reason: str | None = None
    """Set for ``return_existing`` and ``rejected``. Rationale from
    the spawn gate."""

    error: str | None = None
    """Set for ``rejected`` and ``error``. Describes the failure."""

    suggested_action: str | None = None
    """Set for ``rejected``. The gate's hint for the LLM's next step."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def created(self) -> bool:
        """Legacy: True iff the outcome is ``spawned``. Prefer
        branching on ``outcome`` directly."""
        return self.outcome == "spawned"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mission_gate(self) -> Literal["return_existing", "rejected"] | None:
        """Legacy: the gate path when the outcome was gate-driven."""
        if self.outcome == "return_existing":
            return "return_existing"
        if self.outcome == "rejected":
            return "rejected"
        return None


def _spawned_outcome(
    *, agent_id: str, mission_type: str, coordinator_class: str, label: str,
) -> dict[str, Any]:
    return SpawnOutcome(
        outcome="spawned",
        mission_type=mission_type,
        coordinator_class=coordinator_class,
        label=label,
        agent_id=agent_id,
    ).model_dump()


def _return_existing_outcome(
    *, agent_id: str, mission_type: str, coordinator_class: str,
    label: str, reason: str,
) -> dict[str, Any]:
    return SpawnOutcome(
        outcome="return_existing",
        mission_type=mission_type,
        coordinator_class=coordinator_class,
        label=label,
        agent_id=agent_id,
        reason=reason,
    ).model_dump()


def _rejected_outcome(
    *, mission_type: str, coordinator_class: str, label: str,
    error: str, suggested_action: str,
) -> dict[str, Any]:
    return SpawnOutcome(
        outcome="rejected",
        mission_type=mission_type,
        coordinator_class=coordinator_class,
        label=label,
        error=error,
        suggested_action=suggested_action,
    ).model_dump()


def _error_outcome(
    *, mission_type: str, coordinator_class: str, label: str, error: str,
) -> dict[str, Any]:
    return SpawnOutcome(
        outcome="error",
        mission_type=mission_type,
        coordinator_class=coordinator_class,
        label=label,
        error=error,
    ).model_dump()


# ---------------------------------------------------------------------------
# SharedState — what StateManager actually stores and replicates.
# ---------------------------------------------------------------------------


class MissionLedgerState(SharedState):
    """Cluster-wide registry of in-flight mission instances.

    Stored once per Polymathera app (the state_key is keyed off the
    serving app name) so every Ray worker in the same cluster shares
    the same view. ``MissionConcurrencyScope`` is encoded in the
    slot key itself, so one ledger state handles every scope level
    uniformly without needing N separate state managers.

    A single ``slots`` dict holds both pending reservations and
    running coordinators (discriminated by :attr:`Slot.kind`). The
    cap is checked against ``len(slots[storage_key])`` so the
    admit-then-register window can't undercount in-flight work.
    """

    # ``{storage_key: [Slot, ...]}``. Key format:
    # ``RunningMissionKey.to_storage_key()`` —
    # ``"{scope}|{scope_id}|{mission_type}"``.
    slots: dict[str, list[Slot]] = Field(
        default_factory=dict,
        description=(
            "In-flight slot list per bucket. Each slot is either a "
            "PendingReservation (admitted, not yet bound) or a "
            "RunningMissionEntry (bound to a live coordinator)."
        ),
    )

    @classmethod
    def get_state_key(cls, app_name: str | None) -> str:
        """Generate the StateManager state_key for this app's ledger.

        ``app_name`` is the Polymathera serving app name (the one
        every deployment registers under). ``None`` falls back to
        ``"default"`` so tests can instantiate without a live
        deployment.
        """

        return f"polymathera:serving:{app_name or 'default'}:missions:ledger"


# ---------------------------------------------------------------------------
# Ledger — admission API that wraps the StateManager.
# ---------------------------------------------------------------------------


class MissionExecutionLedger:
    """Admission API on top of a :class:`StateManager`-backed
    :class:`MissionLedgerState`.

    All mutations land inside a single ``write_transaction()`` so
    the compare-and-swap atomically closes the parallel-spawn race.
    Snapshot reads use ``read_transaction()`` to avoid bumping the
    version unnecessarily.

    Constructed with a pre-initialised :class:`StateManager`;
    production callers use :func:`get_mission_execution_ledger` to
    fetch the cluster-shared singleton, tests construct one directly
    with an in-memory backend.
    """

    def __init__(
        self,
        state_manager: StateManager[MissionLedgerState],
    ) -> None:
        self._state_manager = state_manager

    # -- admission gate ----------------------------------------------

    async def try_admit(
        self,
        *,
        key: RunningMissionKey,
        mode: str | None,
        policy: MissionExecutionPolicy,
    ) -> AdmissionDecision:
        """Atomically apply ``policy`` to a proposed spawn.

        Returns one of:

        - :class:`AdmissionAllowed` — caller passes its
          ``reservation_id`` to :meth:`register` on success or
          :meth:`release_reservation` on failure.
        - :class:`AdmissionReturnExisting` — a running coordinator
          already covers this spawn; caller returns its ``agent_id``.
        - :class:`AdmissionAwait` — cap is at limit but the only
          slot-holder is a pending reservation under
          ``return_existing`` policy; caller waits + re-calls.
        - :class:`AdmissionRejected` — cap reached under a
          non-recoverable policy.

        Non-blocking: the wait loop lives in :func:`admit_and_spawn`
        for callers that want the spawn-or-return-or-wait flow.

        ``mode`` is recorded on the slot for observability and is
        not consulted by the cap check — every spawn against the
        same key+scope contributes one slot regardless of mode. The
        "one coordinator for all modes" behaviour falls out of
        ``max_concurrent_instances=1`` + ``return_existing``; no
        separate chain knob is needed.
        """

        storage_key = key.to_storage_key()
        decision_holder: AdmissionDecision | None = None

        # IMPORTANT: never ``return`` or ``break`` from inside the
        # write_transaction loop body — Python async generators skip
        # the post-yield compare_and_swap when the caller
        # short-circuits. Assign to the outer holder and let the
        # body complete naturally.
        async for state in self._state_manager.write_transaction():
            slots = state.slots.setdefault(storage_key, [])
            running = [
                s for s in slots if isinstance(s, RunningMissionEntry)
            ]
            reservations = [
                s for s in slots if isinstance(s, PendingReservation)
            ]
            in_flight = len(slots)

            # --- concurrency cap ---------------------------------
            cap = policy.max_concurrent_instances
            if cap is not None and in_flight >= cap:
                if policy.on_concurrency_violation == "return_existing":
                    if running:
                        existing = running[0]
                        decision_holder = AdmissionReturnExisting(
                            agent_id=existing.agent_id,
                            reason=(
                                f"At most {cap} concurrent "
                                f"{key.mission_type!r} per "
                                f"{key.scope.value} ({key.scope_id!r}); "
                                f"returning the running instance "
                                f"{existing.agent_id!r}."
                            ),
                        )
                        continue
                    # Only reservations hold the slot — wait for one
                    # to resolve. The caller will see the outcome
                    # (ReturnExisting or a freed slot) on its next
                    # try_admit. Without this branch, the parallel
                    # admit case under return_existing would surface
                    # as Rejected, which the LLM has no way to retry
                    # against. Hole 2 in fix_plan.md.
                    decision_holder = AdmissionAwait(
                        reservation_id=reservations[0].reservation_id,
                        reason=(
                            f"At most {cap} concurrent "
                            f"{key.mission_type!r} per "
                            f"{key.scope.value} ({key.scope_id!r}); "
                            f"slot held by pending reservation "
                            f"{reservations[0].reservation_id!r} — "
                            f"awaiting resolution."
                        ),
                    )
                    continue
                # ``preempt_oldest`` / ``queue`` not implemented in
                # this layer — fall through to reject with a hint.
                decision_holder = AdmissionRejected(
                    reason=(
                        f"At most {cap} concurrent "
                        f"{key.mission_type!r} per "
                        f"{key.scope.value} ({key.scope_id!r}); "
                        f"currently in flight: {in_flight}."
                    ),
                    suggested_action=(
                        "Wait for the running mission to complete, "
                        "or pick a different mission_type."
                    ),
                )
                continue

            # --- admit -------------------------------------------
            reservation_id = (
                f"resv_{key.mission_type}_{uuid.uuid4().hex[:12]}"
            )
            slots.append(
                PendingReservation(reservation_id=reservation_id),
            )
            decision_holder = AdmissionAllowed(
                reservation_id=reservation_id,
            )

        assert decision_holder is not None, (
            "MissionExecutionLedger.try_admit: state_manager yielded "
            "no state — was ``state_manager.initialize()`` awaited?"
        )
        return decision_holder

    # -- post-admit bookkeeping --------------------------------------

    async def register(
        self,
        *,
        reservation_id: str,
        agent_id: str,
        mode: str | None,
    ) -> None:
        """Replace the pending reservation with a bound running slot.

        Raises :class:`KeyError` if the reservation has already been
        released — that's a programmer error worth surfacing rather
        than silently corrupting the ledger.
        """

        not_found = object()
        outcome: object = not_found
        async for state in self._state_manager.write_transaction():
            # Reset every CAS retry — the prior iteration's
            # mutation rolls back when compare_and_swap fails.
            outcome = None
            for storage_key, slots in state.slots.items():
                for i, slot in enumerate(slots):
                    if (
                        isinstance(slot, PendingReservation)
                        and slot.reservation_id == reservation_id
                    ):
                        slots[i] = RunningMissionEntry(
                            agent_id=agent_id, mode=mode,
                        )
                        outcome = storage_key
                        logger.info(
                            "MissionExecutionLedger: registered "
                            "agent_id=%s storage_key=%s mode=%s "
                            "in_flight=%d",
                            agent_id, storage_key, mode, len(slots),
                        )
                        break
                if outcome is not None:
                    break

        if outcome is not_found:
            # The transaction never yielded — see try_admit's assert.
            raise RuntimeError(
                "MissionExecutionLedger.register: state_manager "
                "yielded no state — initialise it first.",
            )
        if outcome is None:
            raise KeyError(
                f"MissionExecutionLedger.register: unknown "
                f"reservation_id {reservation_id!r}",
            )

    async def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation without registering an agent.

        Called when the spawn fails BETWEEN ``try_admit`` returning
        Allowed and ``register`` landing the agent. Idempotent — a
        no-op if the reservation has already been released."""

        async for state in self._state_manager.write_transaction():
            self._drop_slot(
                state.slots,
                predicate=lambda s: (
                    isinstance(s, PendingReservation)
                    and s.reservation_id == reservation_id
                ),
            )

    async def unregister(self, agent_id: str) -> None:
        """Drop the running slot for ``agent_id``. Idempotent.

        Called when the coordinator terminates (normal completion,
        cancellation, error)."""

        async for state in self._state_manager.write_transaction():
            self._drop_slot(
                state.slots,
                predicate=lambda s: (
                    isinstance(s, RunningMissionEntry)
                    and s.agent_id == agent_id
                ),
            )

    @staticmethod
    def _drop_slot(slots_map, *, predicate) -> None:
        """Remove every slot matching ``predicate`` and clean up
        emptied buckets. Mutates ``slots_map`` in place."""

        for storage_key in list(slots_map.keys()):
            new_slots = [s for s in slots_map[storage_key] if not predicate(s)]
            if len(new_slots) != len(slots_map[storage_key]):
                if new_slots:
                    slots_map[storage_key] = new_slots
                else:
                    del slots_map[storage_key]

    # -- inspection helpers (tests + future observability) -----------

    async def snapshot(self) -> dict[RunningMissionKey, list[RunningMissionEntry]]:
        """Return a copy of the live ledger state. Read-only — does
        not bump the version. Pending reservations are excluded;
        callers reason about coordinators, not gate internals."""

        result: dict[RunningMissionKey, list[RunningMissionEntry]] = {}
        async for state in self._state_manager.read_transaction():
            for storage_key, slots in state.slots.items():
                running = [
                    RunningMissionEntry(**s.model_dump())
                    for s in slots if isinstance(s, RunningMissionEntry)
                ]
                if running:
                    result[
                        RunningMissionKey.from_storage_key(storage_key)
                    ] = running
        return result

    async def list_for_scope(
        self,
        *,
        scope: MissionConcurrencyScope,
        scope_id: str,
        mission_type: str | None = None,
    ) -> list[tuple[RunningMissionKey, RunningMissionEntry]]:
        """Return every live ``(key, entry)`` pair under ``(scope, scope_id)``.

        Filters to :class:`RunningMissionEntry` only — the LLM
        planner's "what's running in my scope?" query doesn't see
        pending reservations. Empty scopes return ``[]`` (no raise).
        """

        result: list[tuple[RunningMissionKey, RunningMissionEntry]] = []
        async for state in self._state_manager.read_transaction():
            for storage_key, slots in state.slots.items():
                key = RunningMissionKey.from_storage_key(storage_key)
                if key.scope is not scope or key.scope_id != scope_id:
                    continue
                if mission_type is not None and key.mission_type != mission_type:
                    continue
                for slot in slots:
                    if isinstance(slot, RunningMissionEntry):
                        result.append(
                            (key, RunningMissionEntry(**slot.model_dump())),
                        )
        return result


# ---------------------------------------------------------------------------
# Singleton accessor + scope resolution helper.
# ---------------------------------------------------------------------------


_LEDGER_CACHE: dict[str, MissionExecutionLedger] = {}


async def get_mission_execution_ledger(
    app_name: str | None = None,
) -> MissionExecutionLedger:
    """Return the cluster-shared ledger singleton for ``app_name``.

    Uses ``polymathera.get_state_manager`` (which itself caches by
    state_key) so every caller in this process gets the same
    StateManager instance — and through it, the same cluster-shared
    state.

    Tests that don't want a live Polymathera should instantiate
    :class:`MissionExecutionLedger` directly with their own
    StateManager + in-memory backend factory.
    """

    cache_key = app_name or "default"
    cached = _LEDGER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from polymathera.colony.distributed import get_initialized_polymathera

    polymathera = await get_initialized_polymathera()
    state_manager = await polymathera.get_state_manager(
        state_type=MissionLedgerState,
        state_key=MissionLedgerState.get_state_key(app_name),
    )
    ledger = MissionExecutionLedger(state_manager=state_manager)
    _LEDGER_CACHE[cache_key] = ledger
    return ledger


def reset_mission_execution_ledger_cache() -> None:
    """Drop the cached ledger references. Tests use this between
    cases; production never needs it."""

    _LEDGER_CACHE.clear()


def resolve_scope_id(
    scope: MissionConcurrencyScope,
    parent_agent: "Agent",
) -> str:
    """Pick the concrete ``scope_id`` for a mission's
    ``concurrency_scope`` against the spawning ``parent_agent``.

    Reads the parent agent's ``agent_id`` and
    ``metadata.{session_id, colony_id, tenant_id}`` typed properties
    directly. A missing field surfaces as ``AttributeError`` — the
    caller violated the ``parent_agent: Agent`` contract and that
    deserves to fail loudly rather than silently fall back to a
    sentinel that masks the bug.
    """

    if scope is MissionConcurrencyScope.GLOBAL:
        return "global"
    if scope is MissionConcurrencyScope.AGENT:
        return parent_agent.agent_id
    if scope is MissionConcurrencyScope.SESSION:
        return parent_agent.metadata.session_id
    if scope is MissionConcurrencyScope.COLONY:
        return parent_agent.metadata.colony_id
    if scope is MissionConcurrencyScope.TENANT:
        return parent_agent.metadata.tenant_id
    raise ValueError(  # pragma: no cover — enum is closed
        f"resolve_scope_id: unsupported scope {scope!r}",
    )


# ---------------------------------------------------------------------------
# Stop-lifecycle callback factory.
#
# Spawn paths attach this to the spawned coordinator's
# ``stop_callbacks`` list, so the agent's stop() invokes it on
# termination — natural completion, error, explicit terminate, or
# cancellation. The agent itself stays mission-unaware: it only
# knows "I have these generic stop callbacks; fire them on stop."
# ---------------------------------------------------------------------------


def mission_stop_callback(app_name: str | None):
    """Build a stop-callback that unregisters the agent from the
    cluster-shared mission ledger when it stops.

    The closure captures only ``app_name`` (a string) so the
    callback round-trips cleanly through cloudpickle on the Ray
    boundary. The actual ledger lookup happens at call-time on the
    worker, using whatever Polymathera deployment is live in that
    process.

    Returns an ``async def (agent, reason)`` callable matching the
    :attr:`Agent.stop_callbacks` shape.
    """

    async def _on_stop(agent, reason):  # noqa: ARG001 — reason unused
        try:
            ledger = await get_mission_execution_ledger(app_name)
            await ledger.unregister(agent.agent_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "mission_stop_callback: ledger unregister failed "
                "for agent %s — bucket slot may remain reserved "
                "until the ledger state expires or a peer "
                "unregisters.", agent.agent_id,
            )

    return _on_stop


# ---------------------------------------------------------------------------
# Orchestration helper — both spawn paths call this.
# ---------------------------------------------------------------------------


async def admit_and_spawn(
    *,
    parent_agent: "Agent",
    pool: "AgentPoolCapability",
    agent_type: str,
    metadata: "AgentMetadata",
    mission_type: str,
    mode: str | None,
    label: str | None = None,
    create_agent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One-shot orchestration: resolve policy → consult ledger →
    spawn-or-return → register.

    Both spawn paths route through here:

    - :meth:`SessionOrchestratorCapability.spawn_mission` (chat).
    - :func:`web_ui.backend.routers.jobs._run_job` (REST).

    Returns :class:`SpawnOutcome`-shaped dict. The canonical
    discriminator is ``result["outcome"]``; callers branch on it:

    - ``outcome="spawned"``: ``agent_id`` set; coordinator started.
    - ``outcome="return_existing"``: ``agent_id`` set; a coordinator
      already exists for this mission_type + scope under
      ``on_concurrency_violation="return_existing"``. ``reason``
      carries the gate's rationale.
    - ``outcome="rejected"``: explicit rejection (cap reached under
      ``reject`` policy, or a sibling reservation never resolved
      within :data:`_AWAIT_RESERVATION_TIMEOUT_S`). ``error`` +
      ``suggested_action`` describe how to recover.
    - ``outcome="error"``: the spawn itself failed (coordinator
      class couldn't be imported, pool dispatch raised, etc.).

    Legacy ``result["created"]`` and ``result["mission_gate"]`` are
    preserved as computed properties on :class:`SpawnOutcome` for
    back-compat with consumers that predate the discriminator; new
    code SHOULD branch on ``outcome``.

    ``create_agent_kwargs`` is the extra-kwargs bag forwarded to
    :meth:`AgentPoolCapability.create_agent`.
    """

    from ..configs import resolve_mission_execution_policy

    agent_cls = pool.resolve_agent_class(agent_type)
    policy = resolve_mission_execution_policy(
        spec=None, coordinator_class=agent_cls,
    )

    scope_id = resolve_scope_id(policy.concurrency_scope, parent_agent)
    key = RunningMissionKey(
        scope=policy.concurrency_scope,
        scope_id=scope_id,
        mission_type=mission_type,
    )

    from polymathera.colony.distributed.ray_utils import serving
    app_name = serving.get_my_app_name()
    ledger = await get_mission_execution_ledger(app_name)

    decision = await _await_admission(
        ledger=ledger, key=key, mode=mode, policy=policy,
    )

    if isinstance(decision, AdmissionReturnExisting):
        logger.info(
            "admit_and_spawn: gate returned existing coordinator "
            "%s for mission %s (%s)",
            decision.agent_id, mission_type, decision.reason,
        )
        return _return_existing_outcome(
            agent_id=decision.agent_id,
            mission_type=mission_type,
            coordinator_class=agent_type,
            label=label or "",
            reason=decision.reason,
        )
    if isinstance(decision, AdmissionRejected):
        logger.info(
            "admit_and_spawn: gate rejected spawn of mission %s (%s)",
            mission_type, decision.reason,
        )
        return _rejected_outcome(
            mission_type=mission_type,
            coordinator_class=agent_type,
            label=label or "",
            error=decision.reason,
            suggested_action=decision.suggested_action,
        )

    # ---- AdmissionAllowed: actually spawn -----------------------
    assert isinstance(decision, AdmissionAllowed)
    # Build the stop-callback list that fires on the spawned
    # coordinator's termination. The list passes through
    # ``pool.create_agent`` → ``agent_cls.bind`` → the spawned
    # agent's ``stop_callbacks`` field, where ``Agent.stop`` picks
    # them up. The agent stays mission-unaware — it only knows it
    # has generic stop callbacks to fire. The user-supplied
    # ``create_agent_kwargs`` can carry additional callbacks; we
    # prepend ours so the ledger unregister fires first.
    reservation_id = decision.reservation_id
    extra_kwargs = dict(create_agent_kwargs or {})
    caller_callbacks = extra_kwargs.pop("stop_callbacks", None) or []
    extra_kwargs["stop_callbacks"] = [
        mission_stop_callback(app_name),
        *caller_callbacks,
    ]
    try:
        spawn_result = await pool.create_agent(
            agent_type=agent_type,
            metadata=metadata,
            label=label,
            **extra_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — surface to caller below
        # Release the slot before propagating so the bucket isn't
        # leaked. Best-effort cleanup; the caller's exception path
        # gets the original error.
        if reservation_id is not None:
            try:
                await ledger.release_reservation(reservation_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "admit_and_spawn: failed to release reservation "
                    "%s after spawn error", reservation_id,
                )
        raise

    if not spawn_result.get("created"):
        # ``create_agent`` returns ``created=False`` on resolution /
        # spawn failure with an ``error`` field. Release the slot and
        # forward the error verbatim.
        if reservation_id is not None:
            await ledger.release_reservation(reservation_id)
        return _error_outcome(
            mission_type=mission_type,
            coordinator_class=agent_type,
            label=spawn_result.get("label") or label or "",
            error=spawn_result.get("error") or "spawn failed",
        )

    spawned_agent_id = spawn_result["agent_id"]
    if reservation_id is not None:
        await ledger.register(
            reservation_id=reservation_id,
            agent_id=spawned_agent_id,
            mode=mode,
        )

    return _spawned_outcome(
        agent_id=spawned_agent_id,
        mission_type=mission_type,
        coordinator_class=agent_type,
        label=spawn_result.get("label") or label or "",
    )


# ---------------------------------------------------------------------------
# Await-admission helper — converts the non-blocking 4-outcome try_admit
# into the 3-outcome shape both spawn paths actually want (Granted /
# ReturnExisting / Rejected). Lives at module scope so both
# admit_and_spawn and admit_mission_spawn share it without a separate
# public concept.
# ---------------------------------------------------------------------------

_AWAIT_RESERVATION_TIMEOUT_S = 30.0
_AWAIT_POLL_INTERVAL_S = 0.05


async def _await_admission(
    *,
    ledger: "MissionExecutionLedger",
    key: RunningMissionKey,
    mode: str | None,
    policy: MissionExecutionPolicy,
) -> AdmissionDecision:
    """Drive :meth:`MissionExecutionLedger.try_admit` to a terminal
    outcome, polling through any :class:`AdmissionAwait` returns.

    Returns Granted / ReturnExisting / Rejected — never Await. If the
    sibling reservation doesn't resolve within
    :data:`_AWAIT_RESERVATION_TIMEOUT_S`, surfaces a Rejected with a
    diagnostic reason so the caller's LLM can decide what to do next.
    """

    import asyncio

    deadline = time.monotonic() + _AWAIT_RESERVATION_TIMEOUT_S
    while True:
        decision = await ledger.try_admit(
            key=key, mode=mode, policy=policy,
        )
        if not isinstance(decision, AdmissionAwait):
            return decision
        if time.monotonic() >= deadline:
            return AdmissionRejected(
                reason=(
                    f"Reservation {decision.reservation_id!r} for "
                    f"{key.mission_type!r} did not resolve within "
                    f"{_AWAIT_RESERVATION_TIMEOUT_S:.0f}s; the holding "
                    f"spawn likely crashed. {decision.reason}"
                ),
                suggested_action=(
                    "Retry the spawn after verifying the previous "
                    "attempt's logs."
                ),
            )
        await asyncio.sleep(_AWAIT_POLL_INTERVAL_S)


async def admit_mission_spawn(
    *,
    agent_cls: type,
    mission_type: str,
    mode: str | None,
    syscontext: "ExecutionContext",
    agent_id_for_agent_scope: str,
    app_name: str,
) -> tuple[AdmissionDecision, MissionExecutionLedger]:
    """Pool-agnostic ``try_admit`` (with await-loop) for spawn paths
    that don't go through :class:`AgentPoolCapability` — the REST
    ``/api/jobs/submit`` path is the main caller.

    Returns ``(decision, ledger)``. On :class:`AdmissionAllowed` the
    caller reads ``decision.reservation_id`` to feed register /
    release_reservation. The decision is never :class:`AdmissionAwait`
    — the helper drives the gate to a terminal outcome.

    The chat-side path uses :func:`admit_and_spawn` instead, which
    bakes the create_agent + register flow into one call. This entry
    point exists for callers that already own their own spawn
    mechanism — keeps ``execution_ledger`` the single home for the
    mission spawn-gate without forcing every caller through a pool.

    ``agent_id_for_agent_scope`` is the synthetic id the REST caller
    uses when the resolved scope is
    :class:`MissionConcurrencyScope.AGENT` (rare for REST batch jobs
    but valid). Pass the job_id, request id, or similar — the bucket
    key just needs to be stable across retries of the same logical
    "spawner".

    ``syscontext`` MUST be a real :class:`ExecutionContext`; missing
    tenant/colony/session ids on it (when the resolved scope needs
    them) surface as ``AttributeError`` so the REST caller fixes its
    context plumbing rather than silently bucketing under a sentinel.
    """

    from ..configs import resolve_mission_execution_policy

    policy = resolve_mission_execution_policy(
        spec=None, coordinator_class=agent_cls,
    )

    scope = policy.concurrency_scope
    if scope is MissionConcurrencyScope.GLOBAL:
        scope_id = "global"
    elif scope is MissionConcurrencyScope.AGENT:
        scope_id = agent_id_for_agent_scope
    elif scope is MissionConcurrencyScope.SESSION:
        scope_id = syscontext.session_id
    elif scope is MissionConcurrencyScope.COLONY:
        scope_id = syscontext.colony_id
    elif scope is MissionConcurrencyScope.TENANT:
        scope_id = syscontext.tenant_id
    else:  # pragma: no cover — enum is closed
        raise ValueError(
            f"admit_mission_spawn: unsupported scope {scope!r}",
        )

    key = RunningMissionKey(
        scope=scope, scope_id=scope_id, mission_type=mission_type,
    )
    ledger = await get_mission_execution_ledger(app_name)
    decision = await _await_admission(
        ledger=ledger, key=key, mode=mode, policy=policy,
    )
    return decision, ledger


__all__ = (
    "AdmissionAllowed",
    "AdmissionAwait",
    "AdmissionDecision",
    "AdmissionRejected",
    "AdmissionReturnExisting",
    "MissionExecutionLedger",
    "MissionLedgerState",
    "PendingReservation",
    "RunningMissionEntry",
    "RunningMissionKey",
    "Slot",
    "SpawnOutcome",
    "admit_and_spawn",
    "admit_mission_spawn",
    "get_mission_execution_ledger",
    "reset_mission_execution_ledger_cache",
    "resolve_scope_id",
)
