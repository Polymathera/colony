"""``ConvergenceRuntime`` — the dispatch loop that turns a static page
graph into a live design substrate.

Per master §5.2, the runtime is the central piece that

- Receives ``PageChangeEvent``s.
- Looks up matching subscriptions.
- Builds a *dependency-aware* dispatch wave (master §5.2 mechanism 2).
- Skips dispatches whose declared numeric output is within tolerance
  of the previous run (master §5.2 mechanism 3, via
  ``ConvergenceDamper``).
- Detects cycles and breaks them with a deterministic leader pick
  (master §5.2 mechanism 4).
- Honours a per-episode invocation budget (1000 by default — same
  reference master §5.2 mechanism 4).
- Rate-limits cascading writes per page (master §5.2 mechanism 5,
  via ``WriteRateLimiter``).
- Emits ``convergence:status`` and ``convergence:quiescence`` events
  on the colony scope so the SessionAgent + dependent capabilities
  observe state transitions.
- Maintains a bounded change-feed (most-recent N dispatches) for the
  user-visible surface (master §5.4).

This module is pure logic; the ``ConvergenceRuntimeDeployment`` in
``deployment.py`` adapts it to a Ray-serving singleton with a
blackboard-backed event source.

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

from ..page_events import (
    CONVERGENCE_CHANGE_FEED_KEY,
    CONVERGENCE_DISPATCH_PREFIX,
    CONVERGENCE_QUIESCENCE_TOPIC,
    CONVERGENCE_STATUS_KEY,
    PageChangeEvent,
    PageChangeKind,
)
from .damping import ConvergenceDamper
from .index import SubscriptionIndex
from .predicates import EdgeReachResolver
from .rate_limit import WriteRateLimiter
from .subscriptions import PageSubscription


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public state types
# ---------------------------------------------------------------------------


ConvergenceState = Literal["converged", "converging", "cycling"]


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
    cycles_broken: int = 0
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
            "tolerance), 'rate_limited' (page-write throttle), "
            "'cycle_break' (lost the cycle leader pick)."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Async callback the runtime invokes for each dispatch decision. The
# implementation lives in the deployment layer (it writes the dispatch
# event to the subscriber's blackboard scope).
DispatchCallback = Callable[[PageSubscription, PageChangeEvent], Awaitable[None]]
StatusEmitCallback = Callable[[ConvergenceStatus], Awaitable[None]]
QuiescenceEmitCallback = Callable[[ConvergenceCounters], Awaitable[None]]
ChangeFeedEmitCallback = Callable[[list[ChangeFeedEntry]], Awaitable[None]]


@dataclass
class _Episode:
    episode_id: str
    counters: ConvergenceCounters
    invocation_graph: dict[str, set[str]] = field(default_factory=dict)
    """``subscription_id -> set of subscription_ids it caused to fire``."""

    cycle_break_winners: set[str] = field(default_factory=set)
    """``subscription_id``s that won a cycle-break leader pick this
    episode (so subsequent recurrence loops don't keep firing)."""


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
        status_emit_callback: StatusEmitCallback | None = None,
        quiescence_emit_callback: QuiescenceEmitCallback | None = None,
        change_feed_emit_callback: ChangeFeedEmitCallback | None = None,
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
        self._status_emit_cb = status_emit_callback
        self._quiescence_emit_cb = quiescence_emit_callback
        self._change_feed_emit_cb = change_feed_emit_callback
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
                    await self._dispatch_one(event, episode, parent_sub_id=None)
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
            state: ConvergenceState = (
                "cycling"
                if self._current_episode.cycle_break_winners
                else "converging"
            )
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

    def detect_cycle(self) -> bool:
        return bool(
            self._current_episode is not None
            and self._current_episode.cycle_break_winners
        )

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
            cycles_broken=episode.counters.cycles_broken,
            budget_exhausted=episode.counters.budget_exhausted,
        )
        episode.counters = finished
        self._last_episode = finished
        self._last_quiescence_at = finished.finished_at
        self._current_episode = None

        # Emit status + quiescence + change-feed snapshots.
        if self._status_emit_cb is not None:
            try:
                await self._status_emit_cb(self.get_status())
            except Exception:  # noqa: BLE001
                logger.exception("status_emit_callback failed")
        if self._quiescence_emit_cb is not None:
            try:
                await self._quiescence_emit_cb(finished)
            except Exception:  # noqa: BLE001
                logger.exception("quiescence_emit_callback failed")
        if self._change_feed_emit_cb is not None:
            try:
                await self._change_feed_emit_cb(self.get_change_feed())
            except Exception:  # noqa: BLE001
                logger.exception("change_feed_emit_callback failed")

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
        *,
        parent_sub_id: str | None,
    ) -> None:
        """Resolve subscriptions for ``event`` and dispatch each one,
        applying topo-sort, damping, cycle-break.

        Recursive dispatch (a subscription firing produces a follow-up
        event that triggers more subscriptions) is *not* implemented
        here — the runtime layer is event-driven, and follow-up events
        are expected to flow back in through ``feed_event`` from the
        source. ``parent_sub_id`` is reserved for that purpose; this
        v1 sets it to None.
        """

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

        ordered = self._topo_sort(matching)
        for sub in ordered:
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
            await self._dispatch_subscription(
                sub, event, episode, parent_sub_id=parent_sub_id,
            )

    async def _dispatch_subscription(
        self,
        sub: PageSubscription,
        event: PageChangeEvent,
        episode: _Episode,
        *,
        parent_sub_id: str | None,
    ) -> None:
        episode.counters = episode.counters.model_copy(
            update={
                "dispatches_attempted": (
                    episode.counters.dispatches_attempted + 1
                ),
            },
        )

        # Cycle detection: if dispatching ``sub`` would close a cycle in
        # the episode's invocation graph, break the cycle by skipping
        # this dispatch and recording the leader pick.
        if parent_sub_id is not None:
            episode.invocation_graph.setdefault(parent_sub_id, set()).add(
                sub.subscription_id,
            )
        if sub.subscription_id in episode.cycle_break_winners:
            self._record_change_feed(
                episode=episode, subscription=sub, event=event,
                skipped_reason="cycle_break",
            )
            return
        if self._would_close_cycle(
            episode.invocation_graph, sub.subscription_id, parent_sub_id,
        ):
            winner = self._cycle_leader(
                episode.invocation_graph, sub.subscription_id,
            )
            episode.cycle_break_winners.add(winner)
            episode.counters = episode.counters.model_copy(
                update={"cycles_broken": episode.counters.cycles_broken + 1},
            )
            self._record_change_feed(
                episode=episode, subscription=sub, event=event,
                skipped_reason="cycle_break",
            )
            return

        # Damping: skip when the cached output is within tolerance.
        # The "output" we track is the event's payload — the runtime
        # only sees what the source emitted, not what the *capability*
        # produced. For damping to work the capability declares its
        # tolerance and the runtime applies it to the *triggering
        # event's* numeric payload (carried in ``event.extra["value"]``
        # by convention). This is sufficient for the budget-propagation
        # / MDO-step / confidence-interval cases the doc calls out.
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

    # ---- Scheduling helpers --------------------------------------------

    @staticmethod
    def _topo_sort(subscriptions: list[PageSubscription]) -> list[PageSubscription]:
        """Order ``subscriptions`` so that whenever subscription B's
        predicate would match an output predicate of subscription A,
        A runs before B.

        The implementation is intentionally simple: a stable Kahn's
        algorithm over the subset graph. When the dependency graph has
        cycles (rare in practice; the runtime's cycle-detection layer
        handles the dispatch-time variant), we fall back to the input
        order — Kahn's algorithm naturally degrades that way.
        """

        n = len(subscriptions)
        if n <= 1:
            return list(subscriptions)
        # Build outgoing edges: A -> B if any of A.declared_outputs
        # could match B.predicate.
        out: dict[str, set[str]] = {s.subscription_id: set() for s in subscriptions}
        in_deg: dict[str, int] = {s.subscription_id: 0 for s in subscriptions}
        index = {s.subscription_id: s for s in subscriptions}
        for a in subscriptions:
            for b in subscriptions:
                if a.subscription_id == b.subscription_id:
                    continue
                if any(
                    _predicate_dominates(out_pred, b.predicate)
                    for out_pred in a.declared_outputs
                ):
                    if b.subscription_id not in out[a.subscription_id]:
                        out[a.subscription_id].add(b.subscription_id)
                        in_deg[b.subscription_id] += 1
        # Stable Kahn: process roots in input order.
        ordered: list[PageSubscription] = []
        ready = [s.subscription_id for s in subscriptions if in_deg[s.subscription_id] == 0]
        while ready:
            sid = ready.pop(0)
            ordered.append(index[sid])
            for downstream in sorted(out[sid]):
                in_deg[downstream] -= 1
                if in_deg[downstream] == 0:
                    ready.append(downstream)
        if len(ordered) < n:
            # Cycle in the declared dependency graph; append remaining
            # subscriptions in input order.
            placed = {s.subscription_id for s in ordered}
            for s in subscriptions:
                if s.subscription_id not in placed:
                    ordered.append(s)
        return ordered

    @staticmethod
    def _would_close_cycle(
        graph: dict[str, set[str]],
        sub_id: str,
        parent_sub_id: str | None,
    ) -> bool:
        if parent_sub_id is None:
            return False
        # Cycle iff sub_id can reach parent_sub_id via ``graph``.
        seen: set[str] = set()
        stack = [sub_id]
        while stack:
            n = stack.pop()
            if n == parent_sub_id:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(graph.get(n, ()))
        return False

    @staticmethod
    def _cycle_leader(
        graph: dict[str, set[str]], sub_id: str,
    ) -> str:
        """Pick a deterministic leader for the cycle that includes
        ``sub_id``. We choose the lexicographically smallest id in the
        cycle's strongly-connected component."""

        # Build the SCC containing sub_id via Tarjan-style reachability.
        forward = graph
        reverse: dict[str, set[str]] = {}
        for u, vs in forward.items():
            for v in vs:
                reverse.setdefault(v, set()).add(u)
        reach_forward = _reachable(forward, sub_id)
        reach_backward = _reachable(reverse, sub_id)
        scc = reach_forward & reach_backward | {sub_id}
        return min(scc)

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


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _predicate_dominates(a, b) -> bool:
    """Return True if a predicate ``a`` (as a *write* spec) "dominates"
    a predicate ``b`` (as a *read* spec) — i.e., events written under
    ``a`` would in general match ``b``.

    Rough rules (intentionally narrow; conservative on uncertainty):

    - ``a.data_type`` set ⇒ ``b.data_type`` must equal it (or be unset
      meaning "any").
    - ``a.source_prefix`` set ⇒ ``b.source_prefix`` must be a prefix
      of ``a.source_prefix`` (or unset).
    - ``a.scope_id`` set ⇒ ``b.scope_id`` must equal it (or be unset).

    Edge-reachability and effective-at windows are not used in the
    dependency analysis (they're pure read-side constraints).
    """

    if a.data_type is not None:
        if b.data_type is not None and b.data_type != a.data_type:
            return False
    elif b.data_type is not None:
        # ``a`` writes to "any data type"; that *includes* events of
        # b.data_type, so this is a dominance.
        pass
    if a.source_prefix is not None:
        if b.source_prefix is not None and not a.source_prefix.startswith(
            b.source_prefix
        ):
            return False
    elif b.source_prefix is not None:
        pass
    if a.scope_id is not None:
        if b.scope_id is not None and b.scope_id != a.scope_id:
            return False
    return True


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(graph.get(n, ()))
    seen.discard(start)
    return seen


__all__ = (
    "ConvergenceRuntime",
    "ConvergenceStatus",
    "ConvergenceCounters",
    "ChangeFeedEntry",
    "ConvergenceState",
)
