import redis.asyncio as redis
import asyncio
from contextlib import asynccontextmanager

class ReadWriteLock:
    def __init__(self, lock_name: str, redis_url: str):
        self.redis_client = redis.Redis.from_url(redis_url)
        self.lock_name = lock_name

    @asynccontextmanager
    async def read_lock(self):
        # Implement read lock acquisition
        while True:
            if await self.redis_client.get(f"{self.lock_name}:write_lock") is None:
                await self.redis_client.incr(f"{self.lock_name}:read_lock")
                break
            await asyncio.sleep(0.1)
        try:
            yield
        finally:
            await self.redis_client.decr(f"{self.lock_name}:read_lock")

    @asynccontextmanager
    async def write_lock(self):
        # Implement write lock acquisition
        while True:
            if await self.redis_client.get(f"{self.lock_name}:write_lock") is None and \
               await self.redis_client.get(f"{self.lock_name}:read_lock") == b'0':
                await self.redis_client.set(f"{self.lock_name}:write_lock", '1')
                break
            await asyncio.sleep(0.1)
        try:
            yield
        finally:
            await self.redis_client.delete(f"{self.lock_name}:write_lock")
