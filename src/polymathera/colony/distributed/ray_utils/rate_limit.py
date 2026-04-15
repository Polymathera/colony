"""Token-bucket rate limiter for API throttling."""

import asyncio
import logging
import time

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    requests_per_second: float
    burst_size: int
    recovery_time: float = 1.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter with burst handling.

    Uses a short lock only to read/update token count, never sleeps
    while holding the lock.  This prevents serialization when many
    callers wait concurrently.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, requested_tokens: int = 1) -> None:
        """Wait until a token is available, then consume it."""
        while True:
            async with self._lock:
                self._replenish()
                if self._tokens >= requested_tokens:
                    self._tokens -= requested_tokens
                    return
                # Calculate how long to wait for enough tokens
                deficit = requested_tokens - self._tokens
                wait_time = deficit / self.config.requests_per_second

            # Sleep OUTSIDE the lock so other callers can proceed
            await asyncio.sleep(wait_time)

    def _replenish(self) -> None:
        """Add tokens based on elapsed time (call while holding lock)."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            float(self.config.burst_size),
            self._tokens + elapsed * self.config.requests_per_second,
        )
        self._last_update = now
