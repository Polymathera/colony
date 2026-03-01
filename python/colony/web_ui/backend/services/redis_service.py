"""Async Redis service for the dashboard.

Provides direct Redis access for reading blackboard entries,
shared state (ApplicationRegistry, VirtualPageTableState), and metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisService:
    """Async Redis client wrapper for dashboard data access."""

    def __init__(self, host: str = "redis", port: int = 6379):
        self.host = host
        self.port = port
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        # Verify connection
        await self._client.ping()

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis not connected")
        return self._client

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def info(self, section: str | None = None) -> dict[str, Any]:
        """Get Redis INFO."""
        try:
            if section:
                return await self.client.info(section)
            return await self.client.info()
        except Exception as e:
            return {"error": str(e)}

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Get and parse a JSON value from Redis."""
        import json
        raw = await self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def scan_keys(self, pattern: str, count: int = 100) -> list[str]:
        """Scan for keys matching a pattern."""
        keys = []
        async for key in self.client.scan_iter(match=pattern, count=count):
            keys.append(key)
            if len(keys) >= count:
                break
        return keys
