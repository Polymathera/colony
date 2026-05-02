"""``ConvergenceRuntime`` — the dispatch loop that turns a static page
graph into a live design substrate.

Per master §5.2, the runtime is the central piece that

- Receives ``PageChangeEvent``s.
- Looks up matching subscriptions.
- Skips dispatches whose declared numeric output is within tolerance
  of the previous run (master §5.2 mechanism 3, via
  ``ConvergenceDamper``).
- Honours a per-episode invocation budget (1000 by default — same
  reference master §5.2 mechanism 4).
- Rate-limits cascading writes per page (master §5.2 mechanism 5,
  via ``WriteRateLimiter``).
- Emits ``convergence:quiescence`` events on the colony scope so
  dependent capabilities observe state transitions.
- Maintains a bounded change-feed (most-recent N dispatches) for the
  user-visible surface (master §5.4).

``VirtualContextManager`` instantiates one ``ConvergenceRuntime`` per
VCM replica; authoritative state lives in
:class:`ConvergencePersistedState`, embedded on
``VirtualPageTableState`` and accessed via the same ``StateManager``
machinery the rest of VCM uses (compare-and-swap across replicas).
The asyncio.Lock per replica only serializes that replica's own
re-entrant access. The blackboard handles (per-subscriber dispatch +
colony-scope quiescence emit) are per-replica connection state.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...agents.blackboard import EnhancedBlackboard
from ...agents.blackboard.protocol import ConvergenceQuiescenceProtocol
from ...agents.scopes import BlackboardScope, get_scope_prefix
from ..page_events import PageChangeEvent, PageChangeKind
from .damping import ConvergenceDamper
from .index import SubscriptionIndex
from .predicates import EdgeReachResolver
from .rate_limit import WriteRateLimiter
from .subscriptions import PageSubscription


if TYPE_CHECKING:
    from ...distributed.state_management import StateManager


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public state types
# ---------------------------------------------------------------------------


ConvergenceState = Literal["converged", "converging"]


class ConvergenceCounters(BaseModel):
    """Episode-level counters surfaced via ``get_status``."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    started_at: datetime
    finished_at: datetime | None = None
    events_received: int = 0
    dispatches_attempted: int = 0
    dispatches_emitted: int = 0
    dispatches_damped: int = 0
    dispatches_rate_limited: int = 0
    budget_exhausted: bool = False


class ConvergenceStatus(BaseModel):
    """Snapshot of the runtime's current state."""

    model_config = ConfigDict(frozen=True)

    state: ConvergenceState
    last_episode: ConvergenceCounters | None = None
    in_flight_episode: ConvergenceCounters | None = None
    last_quiescence_at: datetime | None = None
    subscription_count: int = 0


class ChangeFeedEntry(BaseModel):
    """One dispatch surfaced in the change feed (master §5.4)."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    subscription_id: str
    capability_key: str
    agent_id: str | None = None
    triggering_event_kind: str
    triggering_page_id: str
    triggering_source: str
    dispatched_at: datetime
    skipped_reason: str | None = Field(
        default=None,
        description=(
            "Set when the dispatch was skipped: 'damped' (numeric "
            "tolerance) or 'rate_limited' (page-write throttle)."
        ),
    )


class ConvergencePersistedState(BaseModel):
    """Persisted state for the convergence runtime, embedded as a
    field on ``VirtualPageTableState`` so all VCM replicas see the
    same dispatch decisions via the page-table's ``StateManager``.
    """

    subscriptions: dict[str, PageSubscription] = Field(default_factory=dict)
    """``subscription_id -> PageSubscription``."""

    damper_cache: dict[str, list[float]] = Field(default_factory=dict)
    """Owned by :class:`ConvergenceDamper`; round-tripped via
    ``ConvergenceDamper.dump_cache`` / ``from_cache``."""

    rate_buckets: dict[str, list[float]] = Field(default_factory=dict)
    """Owned by :class:`WriteRateLimiter`; round-tripped via
    ``WriteRateLimiter.dump_buckets`` / ``from_buckets``."""

    last_episode: ConvergenceCounters | None = None
    last_quiescence_at: datetime | None = None
    change_feed: list[ChangeFeedEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Episode:
    episode_id: str
    counters: ConvergenceCounters


@dataclass
class _PendingDispatch:
    """A dispatch decision computed inside the write transaction and
    issued to the blackboard *after* the transaction commits, so a
    CAS retry doesn't double-write."""

    subscription: PageSubscription
    event: PageChangeEvent


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


class ConvergenceRuntime:
    """Per-replica dispatch loop with authoritative state in
    ``VirtualPageTableState.convergence``."""

    DEFAULT_EPISODE_BUDGET = 1000
    DEFAULT_CHANGE_FEED_SIZE = 256
    DEFAULT_RATE_INTERVAL_S = 1.0
    DEFAULT_RATE_BURST = 4

    def __init__(
        self,
        *,
        state_manager: "StateManager",
        app_name: str,
        episode_budget: int = DEFAULT_EPISODE_BUDGET,
        change_feed_size: int = DEFAULT_CHANGE_FEED_SIZE,
        rate_interval_s: float = DEFAULT_RATE_INTERVAL_S,
        rate_burst: int = DEFAULT_RATE_BURST,
        edge_reach_resolver: EdgeReachResolver | None = None,
    ) -> None:
        self._state_manager = state_manager
        self._app_name = app_name
        self._episode_budget = episode_budget
        self._change_feed_size = change_feed_size
        self._rate_interval_s = rate_interval_s
        self._rate_burst = rate_burst
        self._edge_reach_resolver = edge_reach_resolver

        # Per-replica connection caches + intra-replica serialization.
        self._dispatch_blackboards: dict[str, EnhancedBlackboard] = {}
        self._colony_blackboard: EnhancedBlackboard | None = None
        self._lock = asyncio.Lock()
        self._current_episode: _Episode | None = None

    # -- Lifecycle -------------------------------------------------------

    async def initialize(self) -> None:
        """Bring up the colony-scope blackboard handle (used for
        quiescence emit)."""

        if self._colony_blackboard is None:
            self._colony_blackboard = EnhancedBlackboard(
                app_name=self._app_name,
                scope_id=get_scope_prefix(BlackboardScope.COLONY),
            )
            await self._colony_blackboard.initialize()
        logger.info("ConvergenceRuntime ready (app=%s)", self._app_name)

    async def cleanup(self) -> None:
        """Release per-replica blackboard handles. Shared state stays
        in storage — other replicas continue using it."""

        for bb in self._dispatch_blackboards.values():
            try:
                await bb.stop()
            except Exception:  # noqa: BLE001
                logger.exception("dispatch blackboard stop failed")
        self._dispatch_blackboards.clear()
        if self._colony_blackboard is not None:
            try:
                await self._colony_blackboard.stop()
            except Exception:  # noqa: BLE001
                logger.exception("colony blackboard stop failed")
            self._colony_blackboard = None

    # -- Subscription management ----------------------------------------

    async def register(self, subscription: PageSubscription) -> str:
        async for state in self._state_manager.write_transaction():
            state.convergence.subscriptions[subscription.subscription_id] = subscription
        return subscription.subscription_id

    async def unregister(self, subscription_id: str) -> bool:
        result = False
        async for state in self._state_manager.write_transaction():
            if subscription_id in state.convergence.subscriptions:
                del state.convergence.subscriptions[subscription_id]
                # Drop any cached damper entries for this subscription.
                damper = ConvergenceDamper.from_cache(state.convergence.damper_cache)
                damper.reset(subscription_id=subscription_id)
                state.convergence.damper_cache = damper.dump_cache()
                result = True
        return result

    async def get(self, subscription_id: str) -> PageSubscription | None:
        async for state in self._state_manager.read_transaction():
            return state.convergence.subscriptions.get(subscription_id)
        return None

    async def subscription_count(self) -> int:
        async for state in self._state_manager.read_transaction():
            return len(state.convergence.subscriptions)
        return 0

    async def all_subscriptions(self) -> list[PageSubscription]:
        async for state in self._state_manager.read_transaction():
            return list(state.convergence.subscriptions.values())
        return []

    # -- Event ingestion -------------------------------------------------

    async def feed_event(
        self, event: PageChangeEvent, *, source_id: str = "unknown",
    ) -> None:
        """Process a single ``PageChangeEvent``.

        Each call is an *episode boundary* in the simplest framing:
        the runtime resolves all matching subscriptions, dispatches
        them, and emits quiescence at the end. (A more sophisticated
        runtime would batch within an event-arrival window; the doc's
        §5.5 explicitly notes convergence is asymptotic, so a per-event
        dispatch is acceptable for v1.)
        """

        async with self._lock:
            episode_id = f"ep_{uuid.uuid4().hex[:12]}"
            pending: list[_PendingDispatch] = []
            finished: ConvergenceCounters | None = None
            try:
                async for state in self._state_manager.write_transaction():
                    # Reset on each CAS retry — write_transaction may re-enter.
                    episode = self._begin_episode(episode_id)
                    pending = []
                    proceed = await self._maybe_rate_limit(state, event, source_id, episode)
                    if proceed:
                        pending = await self._dispatch_one(state, event, episode)
                    finished = await self._finish_episode(state, episode)
                # Dispatch + emit quiescence INSIDE the in-flight
                # window so ``get_status`` reports ``converging`` for
                # the duration of the wave, but OUTSIDE the write
                # transaction so a CAS retry never re-dispatches.
                for p in pending:
                    try:
                        await self._dispatch_via_blackboard(
                            p.subscription, p.event,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "dispatch failed for subscription %s",
                            p.subscription.subscription_id,
                        )
                if finished is not None:
                    await self._emit_quiescence(finished)
            finally:
                self._current_episode = None

    # -- Manual dispatch (used by tests + ConvergenceCapability.dispatch_change) --

    async def dispatch_change(
        self, event: PageChangeEvent, *, source_id: str = "manual",
    ) -> None:
        await self.feed_event(event, source_id=source_id)

    # -- Status + change feed -------------------------------------------

    async def get_status(self) -> ConvergenceStatus:
        if self._current_episode is not None:
            replica_state: ConvergenceState = "converging"
            in_flight = self._current_episode.counters
        else:
            replica_state = "converged"
            in_flight = None
        async for state in self._state_manager.read_transaction():
            return ConvergenceStatus(
                state=replica_state,
                last_episode=state.convergence.last_episode,
                in_flight_episode=in_flight,
                last_quiescence_at=state.convergence.last_quiescence_at,
                subscription_count=len(state.convergence.subscriptions),
            )
        return ConvergenceStatus(state=replica_state, in_flight_episode=in_flight)

    async def get_change_feed(self, limit: int = 50) -> list[ChangeFeedEntry]:
        if limit <= 0:
            return []
        async for state in self._state_manager.read_transaction():
            return list(state.convergence.change_feed)[-limit:]
        return []

    # ---- Internals -----------------------------------------------------

    def _begin_episode(self, episode_id: str) -> _Episode:
        counters = ConvergenceCounters(
            episode_id=episode_id,
            started_at=datetime.now(timezone.utc),
        )
        episode = _Episode(episode_id=episode_id, counters=counters)
        self._current_episode = episode
        return episode

    async def _finish_episode(self, state, episode: _Episode) -> ConvergenceCounters:
        finished = ConvergenceCounters(
            episode_id=episode.counters.episode_id,
            started_at=episode.counters.started_at,
            finished_at=datetime.now(timezone.utc),
            events_received=episode.counters.events_received,
            dispatches_attempted=episode.counters.dispatches_attempted,
            dispatches_emitted=episode.counters.dispatches_emitted,
            dispatches_damped=episode.counters.dispatches_damped,
            dispatches_rate_limited=episode.counters.dispatches_rate_limited,
            budget_exhausted=episode.counters.budget_exhausted,
        )
        episode.counters = finished
        state.convergence.last_episode = finished
        state.convergence.last_quiescence_at = finished.finished_at
        return finished

    async def _maybe_rate_limit(
        self,
        state,
        event: PageChangeEvent,
        source_id: str,
        episode: _Episode,
    ) -> bool:
        """Return True iff dispatch should proceed."""

        episode.counters = episode.counters.model_copy(
            update={"events_received": episode.counters.events_received + 1}
        )
        # Rate limit applies only to *write* events (master §5.6 item 7).
        if event.kind not in (
            PageChangeKind.PAGE_REPLACED,
            PageChangeKind.PAGE_INVALIDATED,
            PageChangeKind.PAGE_ADDED,
        ):
            return True
        rate = WriteRateLimiter.from_buckets(
            state.convergence.rate_buckets,
            min_interval_s=self._rate_interval_s,
            burst_size=self._rate_burst,
        )
        rate_key = f"page:{event.page_id}"
        allowed = rate.allow(rate_key)
        state.convergence.rate_buckets = rate.dump_buckets()
        if not allowed:
            episode.counters = episode.counters.model_copy(
                update={
                    "dispatches_rate_limited": (
                        episode.counters.dispatches_rate_limited + 1
                    ),
                },
            )
            self._record_change_feed(
                state, episode_id=episode.episode_id,
                subscription=None, event=event,
                skipped_reason="rate_limited",
            )
            return False
        return True

    async def _dispatch_one(
        self,
        state,
        event: PageChangeEvent,
        episode: _Episode,
    ) -> list[_PendingDispatch]:
        """Resolve subscriptions for ``event`` and dispatch each one,
        applying damping. Follow-up events flow back in through a fresh
        ``feed_event`` call from the source."""

        index = SubscriptionIndex()
        for sub in state.convergence.subscriptions.values():
            index.add(sub)
        candidates = index.candidates_for(
            data_type=event.data_type,
            source=event.source,
        )
        # Re-check predicates with the full event.
        matching: list[PageSubscription] = []
        for sub in candidates:
            try:
                if sub.predicate.matches(
                    event,
                    edge_reach_resolver=self._edge_reach_resolver,
                ):
                    matching.append(sub)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "predicate.matches raised for subscription %s",
                    sub.subscription_id,
                )

        if not matching:
            return []

        damper = ConvergenceDamper.from_cache(state.convergence.damper_cache)
        pending: list[_PendingDispatch] = []
        for sub in matching:
            if (
                episode.counters.dispatches_attempted
                >= self._episode_budget
            ):
                episode.counters = episode.counters.model_copy(
                    update={"budget_exhausted": True},
                )
                logger.warning(
                    "Convergence episode %s hit budget %d; halting wave.",
                    episode.episode_id, self._episode_budget,
                )
                break
            result = await self._dispatch_subscription(
                state, sub, event, episode, damper,
            )
            if result is not None:
                pending.append(result)
        state.convergence.damper_cache = damper.dump_cache()
        return pending

    async def _dispatch_subscription(
        self,
        state,
        sub: PageSubscription,
        event: PageChangeEvent,
        episode: _Episode,
        damper: ConvergenceDamper,
    ) -> _PendingDispatch | None:
        episode.counters = episode.counters.model_copy(
            update={
                "dispatches_attempted": (
                    episode.counters.dispatches_attempted + 1
                ),
            },
        )

        # Damping: skip when the triggering event's numeric payload is
        # within the subscription's tolerance of the previous run. The
        # capability publishes the scalar in ``event.extra["value"]``
        # (master §5.2 mechanism 3 — budget propagation, MDO step,
        # confidence interval).
        new_output = event.extra.get("value")
        if sub.tolerance is not None and new_output is not None:
            converged = damper.is_converged(
                subscription_id=sub.subscription_id,
                page_id=event.page_id,
                new_output=new_output,
                tolerance=sub.tolerance,
            )
            if converged:
                episode.counters = episode.counters.model_copy(
                    update={
                        "dispatches_damped": (
                            episode.counters.dispatches_damped + 1
                        ),
                    },
                )
                self._record_change_feed(
                    state, episode_id=episode.episode_id,
                    subscription=sub, event=event,
                    skipped_reason="damped",
                )
                return None

        episode.counters = episode.counters.model_copy(
            update={
                "dispatches_emitted": episode.counters.dispatches_emitted + 1,
            },
        )
        self._record_change_feed(
            state, episode_id=episode.episode_id,
            subscription=sub, event=event,
            skipped_reason=None,
        )
        return _PendingDispatch(subscription=sub, event=event)

    def _record_change_feed(
        self,
        state,
        *,
        episode_id: str,
        subscription: PageSubscription | None,
        event: PageChangeEvent,
        skipped_reason: str | None,
    ) -> None:
        capability_key = subscription.capability_key if subscription else ""
        agent_id = subscription.agent_id if subscription else None
        sub_id = subscription.subscription_id if subscription else "<source>"
        entry = ChangeFeedEntry(
            episode_id=episode_id,
            subscription_id=sub_id,
            capability_key=capability_key,
            agent_id=agent_id,
            triggering_event_kind=event.kind.value,
            triggering_page_id=event.page_id,
            triggering_source=event.source,
            dispatched_at=datetime.now(timezone.utc),
            skipped_reason=skipped_reason,
        )
        state.convergence.change_feed.append(entry)
        overflow = len(state.convergence.change_feed) - self._change_feed_size
        if overflow > 0:
            del state.convergence.change_feed[:overflow]

    async def _dispatch_via_blackboard(
        self, sub: PageSubscription, event: PageChangeEvent,
    ) -> None:
        bb = await self._get_dispatch_blackboard(sub.dispatch_scope)
        await bb.write(
            sub.dispatch_key,
            value=event.model_dump(mode="json"),
            tags={"convergence", "dispatch"},
            metadata={"subscription_id": sub.subscription_id},
        )

    async def _get_dispatch_blackboard(self, scope_id: str) -> EnhancedBlackboard:
        bb = self._dispatch_blackboards.get(scope_id)
        if bb is not None:
            return bb
        bb = EnhancedBlackboard(app_name=self._app_name, scope_id=scope_id)
        await bb.initialize()
        self._dispatch_blackboards[scope_id] = bb
        return bb

    async def _emit_quiescence(self, counters: ConvergenceCounters) -> None:
        if self._colony_blackboard is None:
            return
        try:
            await self._colony_blackboard.write(
                ConvergenceQuiescenceProtocol.quiescence_key(counters.episode_id),
                value=counters.model_dump(mode="json"),
                tags={"convergence", "quiescence"},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "quiescence emit failed for episode %s", counters.episode_id,
            )


__all__ = (
    "ConvergenceRuntime",
    "ConvergencePersistedState",
    "ConvergenceStatus",
    "ConvergenceCounters",
    "ChangeFeedEntry",
    "ConvergenceState",
)
