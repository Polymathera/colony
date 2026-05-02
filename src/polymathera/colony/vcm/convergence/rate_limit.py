"""``WriteRateLimiter`` — debouncing throttle on per-page write events.

Per master §5.2 mechanism 5: "Capabilities that mutate (re-page)
widely-subscribed pages — a top-level requirements page that hundreds
of capabilities watch, a top-level budget page — are rate-limited at
the framework level: the source cannot re-page the same page more than
once per N seconds. This is the equivalent of a debouncing throttle in
a UI; without it, a chain of micro-decisions can overwhelm the
supervisor."

The limiter tracks the last accepted write timestamp per
``page_id`` (or per arbitrary string key). ``allow(key)`` returns True
when at least ``min_interval`` seconds have elapsed since the last
acceptance, and updates the bookkeeping. Otherwise returns False.

Two extra controls:

- ``per_source_min_interval`` lets the runtime apply a *separate*
  rate limit on the per-source event topic (so a source that emits
  10k events in a second is throttled before its events reach
  subscribers — master §5.6 item 7 "working-set churn discipline").
- ``burst_size`` allows a small initial burst before throttling kicks
  in (a common-sense token-bucket variation; default 1 means strict
  rate-only).

Time source is ``time.monotonic`` to avoid wall-clock drift; the
limiter is thread-safe.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    last_at: float
    tokens: float


class WriteRateLimiter:
    """Token-bucket-flavoured rate limiter keyed by string id."""

    def __init__(
        self,
        *,
        min_interval_s: float = 1.0,
        burst_size: int = 1,
    ) -> None:
        if min_interval_s <= 0:
            raise ValueError("min_interval_s must be > 0.")
        if burst_size < 1:
            raise ValueError("burst_size must be >= 1.")
        self._min_interval = float(min_interval_s)
        self._burst = float(burst_size)
        self._lock = threading.RLock()
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Return True if a write keyed by ``key`` is allowed *now*.

        On True the limiter records the acceptance.
        On False the caller is expected to drop / debounce the event.

        ``now`` is for testability; defaults to ``time.monotonic()``.
        """

        ts = time.monotonic() if now is None else float(now)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = _Bucket(
                    last_at=ts, tokens=self._burst - 1,
                )
                return True
            elapsed = ts - bucket.last_at
            refill = elapsed / self._min_interval
            tokens = min(self._burst, bucket.tokens + refill)
            if tokens < 1.0:
                # Update the cooldown anchor so we don't double-charge.
                bucket.tokens = tokens
                bucket.last_at = ts
                return False
            bucket.tokens = tokens - 1.0
            bucket.last_at = ts
            return True

    # ---- Shared-state round-trip --------------------------------------
    #
    # The runtime persists rate-bucket state in
    # ``VirtualPageTableState.convergence`` so per-page rate limits
    # apply across all VCM replicas. The runtime constructs a limiter
    # inside each write transaction via ``from_buckets`` and writes
    # the updated buckets back via ``dump_buckets``.

    def dump_buckets(self) -> dict[str, list[float]]:
        """Serialize buckets to a Pydantic-friendly dict
        ``key -> [last_at, tokens]``."""

        with self._lock:
            return {
                key: [bucket.last_at, bucket.tokens]
                for key, bucket in self._buckets.items()
            }

    @classmethod
    def from_buckets(
        cls,
        buckets: dict[str, list[float]],
        *,
        min_interval_s: float = 1.0,
        burst_size: int = 1,
    ) -> "WriteRateLimiter":
        """Reconstruct from a dict produced by ``dump_buckets``."""

        rl = cls(min_interval_s=min_interval_s, burst_size=burst_size)
        for key, (last_at, tokens) in buckets.items():
            rl._buckets[key] = _Bucket(last_at=float(last_at), tokens=float(tokens))
        return rl

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)


__all__ = ("WriteRateLimiter",)
