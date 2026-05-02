"""``ConvergenceRuntime`` — the dispatch loop that turns a static page
graph into a live design substrate.

Per master §5.2, the runtime is the central piece that

- Receives ``PageChangeEvent``s.
- Looks up matching subscriptions and fires their dispatch callback.
- Skips dispatches whose declared numeric output is within tolerance
  of the previous run (master §5.2 mechanism 3, via
  ``ConvergenceDamper``).
- Honours a per-episode invocation budget (1000 by default —
  master §5.2 mechanism 4 runaway protection).
- Rate-limits cascading writes per page (master §5.2 mechanism 5,
  via ``WriteRateLimiter``).
- Emits a ``convergence:quiescence:<episode_id>`` event after each
  episode settles so consumers (``DesignCheckpointer`` and other
  "react when the design has settled" agents) can react.
- Maintains a bounded in-memory change feed (most-recent N
  dispatches) read via the deployment's ``get_change_feed`` endpoint.

This module is pure logic; the ``ConvergenceRuntimeDeployment`` in
``deployment.py`` adapts it to a Ray-serving singleton.

Threading: the runtime is async-driven from one task at a time; the
``feed_event`` API is the entry point. Internal data structures use
plain Python dicts/lists; concurrency control comes from the
calling layer (one event at a time per dispatch wave).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..page_events import PageChangeEvent, PageChangeKind
from .damping import ConvergenceDamper
from .index import SubscriptionIndex
from .predicates import EdgeReachResolver
from .rate_limit import WriteRateLimiter
from .subscriptions import PageSubscription


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public state types
# ---------------------------------------------------------------------------


ConvergenceState = Literal["converged", "converging"]


class ConvergenceCounters(BaseModel):
    """Episode-level counters surfaced via ``get_convergence_status``."""

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
    """Snapshot of the runtime's current state.

    Written to the blackboard at ``convergence:status`` (master §5.4).
    """

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Async callback the runtime invokes for each dispatch decision. The
# implementation lives in the deployment layer (it writes the dispatch
# event to the subscriber's blackboard scope).
DispatchCallback = Callable[[PageSubscription, PageChangeEvent], Awaitable[None]]
QuiescenceEmitCallback = Callable[[ConvergenceCounters], Awaitable[None]]


@dataclass
class _Episode:
    episode_id: str
    counters: ConvergenceCounters


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------


class ConvergenceRuntime:
    """Core dispatch logic for the always-live design context."""

    DEFAULT_EPISODE_BUDGET = 1000
    DEFAULT_CHANGE_FEED_SIZE = 256
    DEFAULT_RATE_INTERVAL_S = 1.0
    DEFAULT_RATE_BURST = 4

    def __init__(
        self,
        *,
        dispatch_callback: DispatchCallback,
        quiescence_emit_callback: QuiescenceEmitCallback | None = None,
        edge_reach_resolver: EdgeReachResolver | None = None,
        episode_budget: int = DEFAULT_EPISODE_BUDGET,
        change_feed_size: int = DEFAULT_CHANGE_FEED_SIZE,
        rate_interval_s: float = DEFAULT_RATE_INTERVAL_S,
        rate_burst: int = DEFAULT_RATE_BURST,
    ) -> None:
        self._index = SubscriptionIndex()
        self._damper = ConvergenceDamper()
        self._rate = WriteRateLimiter(
            min_interval_s=rate_interval_s, burst_size=rate_burst,
        )
        self._dispatch_cb = dispatch_callback
        self._quiescence_emit_cb = quiescence_emit_callback
        self._edge_reach_resolver = edge_reach_resolver
        self._episode_budget = episode_budget
        self._change_feed: deque[ChangeFeedEntry] = deque(maxlen=change_feed_size)
        self._lock = asyncio.Lock()
        self._current_episode: _Episode | None = None
        self._last_episode: ConvergenceCounters | None = None
        self._last_quiescence_at: datetime | None = None
        self._quiescence_waiters: list[asyncio.Future[None]] = []

    # -- Subscription management ----------------------------------------

    def register(self, subscription: PageSubscription) -> str:
        self._index.add(subscription)
        return subscription.subscription_id

    def unregister(self, subscription_id: str) -> bool:
        ok = self._index.remove(subscription_id)
        if ok:
            self._damper.reset(subscription_id=subscription_id)
        return ok

    def get(self, subscription_id: str) -> PageSubscription | None:
        return self._index.get(subscription_id)

    @property
    def subscription_count(self) -> int:
        return len(self._index)

    def all_subscriptions(self) -> list[PageSubscription]:
        return self._index.all()

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
            episode = self._begin_episode()
            try:
                proceed = await self._maybe_rate_limit(event, source_id, episode)
                if proceed:
                    await self._dispatch_one(event, episode)
            finally:
                await self._finish_episode(episode)

    # -- Manual dispatch (used by tests + ConvergenceCapability.dispatch_change) --

    async def dispatch_change(
        self, event: PageChangeEvent, *, source_id: str = "manual",
    ) -> None:
        await self.feed_event(event, source_id=source_id)

    # -- Status + change feed -------------------------------------------

    def get_status(self) -> ConvergenceStatus:
        if self._current_episode is not None:
            state: ConvergenceState = "converging"
            in_flight = self._current_episode.counters
        else:
            state = "converged"
            in_flight = None
        return ConvergenceStatus(
            state=state,
            last_episode=self._last_episode,
            in_flight_episode=in_flight,
            last_quiescence_at=self._last_quiescence_at,
            subscription_count=self.subscription_count,
        )

    def get_change_feed(self, limit: int = 50) -> list[ChangeFeedEntry]:
        if limit <= 0:
            return []
        return list(self._change_feed)[-limit:]

    async def wait_for_quiescence(self, timeout: float | None = None) -> bool:
        """Block until the runtime is in the ``converged`` state.

        Returns True on quiescence, False on timeout. When the runtime
        is already converged, returns True immediately.
        """

        if self._current_episode is None:
            return True
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._quiescence_waiters.append(fut)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            try:
                self._quiescence_waiters.remove(fut)
            except ValueError:
                pass
            return False

    # ---- Internals -----------------------------------------------------

    def _begin_episode(self) -> _Episode:
        ep_id = f"ep_{uuid.uuid4().hex[:12]}"
        counters = ConvergenceCounters(
            episode_id=ep_id,
            started_at=datetime.now(timezone.utc),
        )
        episode = _Episode(episode_id=ep_id, counters=counters)
        self._current_episode = episode
        return episode

    async def _finish_episode(self, episode: _Episode) -> None:
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
        self._last_episode = finished
        self._last_quiescence_at = finished.finished_at
        self._current_episode = None

        # Emit a quiescence event so consumers (DesignCheckpointer, etc.)
        # can react when an episode settles.
        if self._quiescence_emit_cb is not None:
            try:
                await self._quiescence_emit_cb(finished)
            except Exception:  # noqa: BLE001
                logger.exception("quiescence_emit_callback failed")

        # Wake any wait_for_quiescence callers.
        waiters = self._quiescence_waiters
        self._quiescence_waiters = []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    async def _maybe_rate_limit(
        self,
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
        rate_key = f"page:{event.page_id}"
        if not self._rate.allow(rate_key):
            episode.counters = episode.counters.model_copy(
                update={
                    "dispatches_rate_limited": (
                        episode.counters.dispatches_rate_limited + 1
                    ),
                },
            )
            self._record_change_feed(
                episode=episode, subscription=None, event=event,
                skipped_reason="rate_limited",
            )
            return False
        return True

    async def _dispatch_one(
        self,
        event: PageChangeEvent,
        episode: _Episode,
    ) -> None:
        """Resolve subscriptions for ``event`` and dispatch each one,
        applying damping. Follow-up events flow back in through a fresh
        ``feed_event`` call from the source."""

        candidates = self._index.candidates_for(
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
            return

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
                return
            await self._dispatch_subscription(sub, event, episode)

    async def _dispatch_subscription(
        self,
        sub: PageSubscription,
        event: PageChangeEvent,
        episode: _Episode,
    ) -> None:
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
            converged = self._damper.is_converged(
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
                    episode=episode, subscription=sub, event=event,
                    skipped_reason="damped",
                )
                return

        try:
            await self._dispatch_cb(sub, event)
        except Exception:  # noqa: BLE001
            logger.exception(
                "dispatch_callback failed for subscription %s",
                sub.subscription_id,
            )
            return
        episode.counters = episode.counters.model_copy(
            update={
                "dispatches_emitted": episode.counters.dispatches_emitted + 1,
            },
        )
        self._record_change_feed(
            episode=episode, subscription=sub, event=event, skipped_reason=None,
        )

    # ---- Change feed bookkeeping --------------------------------------

    def _record_change_feed(
        self,
        *,
        episode: _Episode,
        subscription: PageSubscription | None,
        event: PageChangeEvent,
        skipped_reason: str | None,
    ) -> None:
        capability_key = subscription.capability_key if subscription else ""
        agent_id = subscription.agent_id if subscription else None
        sub_id = subscription.subscription_id if subscription else "<source>"
        entry = ChangeFeedEntry(
            episode_id=episode.episode_id,
            subscription_id=sub_id,
            capability_key=capability_key,
            agent_id=agent_id,
            triggering_event_kind=event.kind.value,
            triggering_page_id=event.page_id,
            triggering_source=event.source,
            dispatched_at=datetime.now(timezone.utc),
            skipped_reason=skipped_reason,
        )
        self._change_feed.append(entry)


__all__ = (
    "ConvergenceRuntime",
    "ConvergenceStatus",
    "ConvergenceCounters",
    "ChangeFeedEntry",
    "ConvergenceState",
)
