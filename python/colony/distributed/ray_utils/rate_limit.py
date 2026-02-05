import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    requests_per_second: float
    burst_size: int
    recovery_time: float = 1.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter with burst handling

    Other options:
    1. Fixed Window Rate Limiting
    2. Sliding Window Rate Limiting
    3. Leaky Bucket Algorithm
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_update = datetime.utcnow()
        self.lock = asyncio.Lock()

    async def acquire(self, requested_tokens: int = 1):
        """Acquire a rate limit token"""
        async with self.lock:
            await self._replenish_tokens()

            while self.tokens <= 0:
                wait_time = self._calculate_wait_time(requested_tokens)
                await asyncio.sleep(wait_time)
                await self._replenish_tokens()

            self.tokens -= requested_tokens

    async def _replenish_tokens(self):
        """Replenish tokens based on time passed"""
        now = datetime.utcnow()
        time_passed = (now - self.last_update).total_seconds()
        self.tokens = min(
            self.config.burst_size,
            self.tokens + time_passed * self.config.requests_per_second,
        )
        self.last_update = now

    def _calculate_wait_time(self, requested_tokens: int = 1) -> float:
        """Calculate wait time until next token"""
        tokens_needed = requested_tokens - self.tokens
        return tokens_needed / self.config.requests_per_second
