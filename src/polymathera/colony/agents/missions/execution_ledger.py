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
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

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


class RunningMissionEntry(BaseModel):
    """One in-flight coordinator's bookkeeping row.

    Pydantic model (not dataclass) so it round-trips cleanly through
    :class:`StateManager`'s JSON serialisation."""

    agent_id: str
    mode: str | None = None # ``mission_params['mode']``, or ``None`` when absent.
    started_at: float = Field(default_factory=time.time)


@dataclass(frozen=True)
class AdmissionAllowed:
    """The spawn gate approves a fresh spawn."""

    kind: Literal["spawn"] = "spawn"


@dataclass(frozen=True)
class AdmissionReturnExisting:
    """Policy is ``return_existing`` and a compatible instance is
    already running — the gate hands back that running coordinator's
    ``agent_id`` instead of spawning a new one. Used both for
    :class:`MissionExecutionPolicy.idempotent` matches and for
    ``chains_with_modes`` collisions where the LLM emitted spawns
    for additional modes that are auto-chained internally."""

    agent_id: str
    reason: str
    kind: Literal["return_existing"] = "return_existing"


@dataclass(frozen=True)
class AdmissionRejected:
    """The spawn gate refuses the spawn."""

    reason: str
    suggested_action: str = ""
    kind: Literal["reject"] = "reject"


AdmissionDecision = (
    AdmissionAllowed | AdmissionReturnExisting | AdmissionRejected
)


# ---------------------------------------------------------------------------
# SharedState — what StateManager actually stores and replicates.
# ---------------------------------------------------------------------------


class MissionLedgerState(SharedState):
    """Cluster-wide registry of in-flight mission instances.

    Stored once per Polymathera app (the state_key is keyed off the
    serving app name) so every Ray worker in the same cluster shares
    the same view. ``MissionConcurrencyScope`` is encoded in the
    bucket key itself, so one ledger state handles every scope level
    uniformly without needing N separate state managers.
    """

    # Bucket key format: ``RunningMissionKey.to_storage_key()`` —
    # ``"{scope}|{scope_id}|{mission_type}"``.
    buckets: dict[str, list[RunningMissionEntry]] = Field(
        default_factory=dict,
        description="Live mission entries grouped by bucket key.",
    )

    # ``{reservation_id: storage_key}``. A reservation counts toward
    # the cap until it is either :meth:`register`-ed against a real
    # agent_id or :meth:`release_reservation`-d, so two parallel
    # spawn attempts can't both see an empty bucket and both admit.
    reservations: dict[str, str] = Field(
        default_factory=dict,
        description="Outstanding admit-but-not-yet-registered reservations.",
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
    ) -> tuple[AdmissionDecision, str | None]:
        """Atomically apply ``policy`` to a proposed spawn.

        Returns ``(decision, reservation_id)``. On
        :class:`AdmissionAllowed` the ``reservation_id`` is the token
        the caller passes to :meth:`register` (or
        :meth:`release_reservation` if the spawn ultimately fails).
        On the other decision shapes, ``reservation_id`` is ``None``.

        Order of checks:

        1. ``chains_with_modes`` — if ``mode`` is in the declared
           list AND an entry of the same mission_type+scope is
           already running, return the existing entry's ``agent_id``.
        2. ``max_concurrent_instances`` against the bucket size +
           outstanding reservations. Reservations count so we don't
           admit beyond the cap during a parallel spawn race.
        3. ``on_concurrency_violation`` shapes the rejection:
           ``reject`` (default), ``return_existing`` (hands back the
           oldest live entry's ``agent_id``), or ``preempt_oldest`` /
           ``queue`` (not yet implemented — fall through to reject).
        """

        storage_key = key.to_storage_key()
        decision_holder: AdmissionDecision | None = None
        reservation_holder: str | None = None

        # IMPORTANT: never ``return`` or ``break`` from inside the
        # write_transaction loop body — Python async generators
        # skip the post-yield compare_and_swap when the caller
        # short-circuits. See the standing rule in this repo's
        # ``.CLAUDE.md`` / memory: assign to outer-scope holders
        # and let the loop body complete naturally.
        async for state in self._state_manager.write_transaction():
            bucket = state.buckets.setdefault(storage_key, [])
            reservation_count = sum(
                1 for k in state.reservations.values() if k == storage_key
            )
            in_flight = len(bucket) + reservation_count

            # ---- chains_with_modes ------------------------------
            if (
                policy.chains_with_modes is not None
                and mode is not None
                and mode in policy.chains_with_modes
                and bucket
            ):
                existing = bucket[0]
                decision_holder = AdmissionReturnExisting(
                    agent_id=existing.agent_id,
                    reason=(
                        f"Mission {key.mission_type!r}'s modes "
                        f"{policy.chains_with_modes!r} auto-chain "
                        f"internally — the running coordinator "
                        f"{existing.agent_id!r} drives the sequence. "
                        f"Re-spawning per-mode would violate the "
                        f"mission's declared control flow."
                    ),
                )
                continue  # falls through to natural loop end

            # --- concurrency cap ---------------------------------
            cap = policy.max_concurrent_instances
            if cap is not None and in_flight >= cap:
                if policy.on_concurrency_violation == "return_existing" and bucket:
                    existing = bucket[0]
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
                # ``preempt_oldest`` and ``queue`` not implemented in
                # this layer — fall through to reject with a clear
                # hint. (Preemption needs the coordinator-side
                # cancel plumbing; queue needs a separate dispatcher
                # process.)
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
            state.reservations[reservation_id] = storage_key
            decision_holder = AdmissionAllowed()
            reservation_holder = reservation_id

        # Belt-and-braces: an empty transaction (no iterations) means
        # the StateManager wasn't initialised — surface that loudly.
        assert decision_holder is not None, (
            "MissionExecutionLedger.try_admit: state_manager yielded "
            "no state — was ``state_manager.initialize()`` awaited?"
        )
        return decision_holder, reservation_holder

    # -- post-admit bookkeeping --------------------------------------

    async def register(
        self,
        *,
        reservation_id: str,
        agent_id: str,
        mode: str | None,
    ) -> None:
        """Bind the reserved slot to the concrete spawned agent.

        Raises :class:`KeyError` if the reservation has already been
        released — that's a programmer error worth surfacing rather
        than silently corrupting the ledger.
        """

        not_found = object()
        outcome: object = not_found
        async for state in self._state_manager.write_transaction():
            storage_key = state.reservations.pop(reservation_id, None)
            if storage_key is None:
                outcome = None  # signal "not found" without raising mid-loop
                continue
            bucket = state.buckets.setdefault(storage_key, [])
            bucket.append(
                RunningMissionEntry(agent_id=agent_id, mode=mode),
            )
            outcome = storage_key
            logger.info(
                "MissionExecutionLedger: registered agent_id=%s "
                "storage_key=%s mode=%s in_flight=%d",
                agent_id, storage_key, mode, len(bucket),
            )

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
            state.reservations.pop(reservation_id, None)

    async def unregister(self, agent_id: str) -> None:
        """Drop the running entry for ``agent_id`` from the ledger.

        Called when the coordinator terminates (normal completion,
        cancellation, error). Idempotent — extra calls for the same
        ``agent_id`` are no-ops, so callers don't have to track
        whether they've already cleaned up.
        """

        async for state in self._state_manager.write_transaction():
            empty_keys: list[str] = []
            for storage_key, entries in state.buckets.items():
                filtered = [e for e in entries if e.agent_id != agent_id]
                if len(filtered) != len(entries):
                    state.buckets[storage_key] = filtered
                if not state.buckets[storage_key]:
                    empty_keys.append(storage_key)
            for k in empty_keys:
                state.buckets.pop(k, None)

    # -- inspection helpers (tests + future observability) -----------

    async def snapshot(self) -> dict[RunningMissionKey, list[RunningMissionEntry]]:
        """Return a copy of the current ledger state. Read-only —
        does not bump the version. Decoded back into the typed
        :class:`RunningMissionKey` shape for ergonomics."""

        result: dict[RunningMissionKey, list[RunningMissionEntry]] = {}
        async for state in self._state_manager.read_transaction():
            for storage_key, entries in state.buckets.items():
                result[
                    RunningMissionKey.from_storage_key(storage_key)
                ] = [
                    RunningMissionEntry(**e.model_dump())
                    for e in entries
                ]
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

    Returns the contract every caller can branch on directly:

    - ``{agent_id, mission_type, coordinator_class, created: True,
      label}`` — spawn succeeded.
    - ``{agent_id: <existing>, ..., created: False,
      mission_gate: "return_existing", reason}`` — the gate handed
      back a running coordinator (chains_with_modes or
      return_existing on cap collision).
    - ``{agent_id: None, ..., created: False,
      mission_gate: "rejected", error, suggested_action}`` —
      explicit rejection. Includes a hint for the LLM's next
      iteration.
    - ``{agent_id: None, ..., created: False, error}`` — the spawn
      itself failed (e.g. the coordinator class couldn't be
      imported). Distinct from a ``mission_gate`` rejection
      because the failure mode is different.

    ``create_agent_kwargs`` is the extra-kwargs bag forwarded to
    :meth:`AgentPoolCapability.create_agent` for callers that need to
    set non-default options (resource requirements, soft_affinity,
    etc.).
    """

    from ..configs import resolve_mission_execution_policy

    # Resolve the coordinator class so we can read its
    # ``MISSION_EXECUTION_POLICY`` ClassVar.
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

    # Canonical serving-framework lookup for the app name. This call
    # raises outside a deployment context, which is exactly what we
    # want — admit_and_spawn must run inside the spawning agent's
    # runtime (a serving deployment), so a missing app_name is a
    # setup bug worth surfacing loudly rather than papering over.
    from polymathera.colony.distributed.ray_utils import serving
    app_name = serving.get_my_app_name()
    ledger = await get_mission_execution_ledger(app_name)

    decision, reservation_id = await ledger.try_admit(
        key=key, mode=mode, policy=policy,
    )

    if isinstance(decision, AdmissionReturnExisting):
        logger.info(
            "admit_and_spawn: gate returned existing coordinator "
            "%s for mission %s (%s)",
            decision.agent_id, mission_type, decision.reason,
        )
        return {
            "agent_id": decision.agent_id,
            "mission_type": mission_type,
            "coordinator_class": agent_type,
            "created": False,
            "label": label or "",
            "mission_gate": "return_existing",
            "reason": decision.reason,
        }
    if isinstance(decision, AdmissionRejected):
        logger.info(
            "admit_and_spawn: gate rejected spawn of mission %s (%s)",
            mission_type, decision.reason,
        )
        return {
            "agent_id": None,
            "mission_type": mission_type,
            "coordinator_class": agent_type,
            "created": False,
            "label": label or "",
            "mission_gate": "rejected",
            "error": decision.reason,
            "suggested_action": decision.suggested_action,
        }

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
        return {
            "agent_id": None,
            "mission_type": mission_type,
            "coordinator_class": agent_type,
            "created": False,
            "label": spawn_result.get("label") or label or "",
            "error": spawn_result.get("error") or "spawn failed",
        }

    spawned_agent_id = spawn_result["agent_id"]
    if reservation_id is not None:
        await ledger.register(
            reservation_id=reservation_id,
            agent_id=spawned_agent_id,
            mode=mode,
        )

    return {
        "agent_id": spawned_agent_id,
        "mission_type": mission_type,
        "coordinator_class": agent_type,
        "created": True,
        "label": spawn_result.get("label") or label or "",
    }


async def admit_mission_spawn(
    *,
    agent_cls: type,
    mission_type: str,
    mode: str | None,
    syscontext: "ExecutionContext",
    agent_id_for_agent_scope: str,
    app_name: str,
) -> tuple[AdmissionDecision, str | None, MissionExecutionLedger]:
    """Pool-agnostic ``try_admit`` for spawn paths that don't go
    through :class:`AgentPoolCapability` (the REST
    ``/api/jobs/submit`` path is the main caller).

    Returns ``(decision, reservation_id, ledger)``. The caller is
    responsible for:

    - On :class:`AdmissionAllowed`: spawning the coordinator, then
      calling ``ledger.register(reservation_id=..., agent_id=...,
      mode=...)`` on success or ``ledger.release_reservation(...)``
      on spawn failure.
    - On :class:`AdmissionRejected` / :class:`AdmissionReturnExisting`:
      skipping the spawn and surfacing the decision to its own
      caller.

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
    decision, reservation_id = await ledger.try_admit(
        key=key, mode=mode, policy=policy,
    )
    return decision, reservation_id, ledger


__all__ = (
    "AdmissionAllowed",
    "AdmissionDecision",
    "AdmissionRejected",
    "AdmissionReturnExisting",
    "MissionExecutionLedger",
    "MissionLedgerState",
    "RunningMissionEntry",
    "RunningMissionKey",
    "admit_and_spawn",
    "admit_mission_spawn",
    "get_mission_execution_ledger",
    "reset_mission_execution_ledger_cache",
    "resolve_scope_id",
)
