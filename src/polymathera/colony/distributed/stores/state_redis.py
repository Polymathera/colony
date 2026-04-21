import logging
import redis
import asyncio
from redis.asyncio import Redis
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
import json

from .state_base import StateStorageBackend, StateStorageBackendFactory

logger = logging.getLogger(__name__)

# Define a reasonable timeout for Redis operations
REDIS_TIMEOUT_SECONDS = 10



class RedisStateStorageConfig(BaseModel):
    """Configuration for state storage"""

    namespace: str = Field(default="polymathera")
    ttl: int = 86400  # 24 hour default TTL
    max_retries: int = 3
    retry_delay: float = 0.1  # seconds

    # Redis specific - use environment variables with fallbacks
    redis_host: str = Field(
        default="localhost",
        description="Redis host to connect to",
        json_schema_extra={"env": "REDIS_HOST"},
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port to connect to",
        json_schema_extra={"env": "REDIS_PORT"},
    )
    redis_db: int = 0
    redis_password: str | None = None
    redis_ssl: bool = False



class RedisStorage(StateStorageBackend):
    """Redis-based storage backend with optimistic locking"""

    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        password: str | None = None,
        ssl: bool = False,
        ttl: int = 3600,
    ):
        self.host = host  # Store host and port for debugging
        self.port = port
        self.redis = Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            decode_responses=True,
        )
        self.ttl = ttl

    async def get_with_version(self, key: str) -> tuple[str | None, int]:
        """Get value and version atomically using Redis MULTI with timeout"""
        logger.debug(f"[{datetime.now()}] RedisStorage.get_with_version: START for key: {key}")
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                # Get both value and version in a single transaction
                logger.debug(f"[{datetime.now()}] RedisStorage.get_with_version: Preparing pipe for key {key}")
                await pipe.get(f"{key}:data")
                await pipe.get(f"{key}:version")

                logger.debug(f"[{datetime.now()}] RedisStorage.get_with_version: Executing pipe for key {key} with timeout {REDIS_TIMEOUT_SECONDS}s...")
                start_time = asyncio.get_event_loop().time()
                try:
                    data, version = await asyncio.wait_for(
                        pipe.execute(), timeout=REDIS_TIMEOUT_SECONDS
                    )
                    end_time = asyncio.get_event_loop().time()
                    logger.debug(f"[{datetime.now()}] RedisStorage.get_with_version: Pipe executed for key {key} in {end_time - start_time:.4f}s.\nData:\n{json.dumps(json.loads(data), indent=4, sort_keys=True) if data else 'None'}\n\tVersion: {version}")
                    return data, int(version or 0)
                except asyncio.TimeoutError:
                    end_time = asyncio.get_event_loop().time()
                    logger.error(f"[{datetime.now()}] Redis command pipe.execute() timed out after {end_time - start_time:.4f}s (>{REDIS_TIMEOUT_SECONDS}s) for key: {key}")
                    # Add diagnostic information
                    try:
                        # Check basic connectivity
                        logger.debug(f"[{datetime.now()}] Running PING check after timeout for key {key} (Host: {self.host}, Port: {self.port})")
                        pong = await asyncio.wait_for(self.redis.ping(), timeout=1)
                        logger.debug(f"[{datetime.now()}] Redis PING successful after timeout for key {key}: {pong}")
                    except Exception as ping_err:
                        logger.error(f"[{datetime.now()}] Redis PING failed after timeout for key {key}: {ping_err} (Host: {self.host}, Port: {self.port})")
                    # Re-raise the timeout error to indicate the failure
                    raise TimeoutError(f"Redis operation timed out for key {key}")
                except asyncio.CancelledError:
                    end_time = asyncio.get_event_loop().time()
                    logger.warning(f"[{datetime.now()}] Redis pipe.execute() was cancelled after {end_time - start_time:.4f}s for key: {key}. This might be due to the timeout.")
                    raise # Re-raise CancelledError as it's part of timeout handling

        except Exception as e:
            logger.exception(f"[{datetime.now()}] Unexpected error during Redis get_with_version for key {key}: {e}") # Use logger.exception for stack trace
            raise # Re-raise other exceptions

    async def compare_and_swap(self, key: str, value: str, version: int) -> bool:
        """
        Atomic compare-and-swap using Redis WATCH/MULTI/EXEC.
        Returns True if successful, False if version mismatch.
        """
        logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: START for key: {key}, version: {version}")
        version_key = f"{key}:version"
        data_key = f"{key}:data"

        while True:
            try:
                # Watch the version key for changes
                logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: Watching {version_key}")
                await self.redis.watch(version_key)

                logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: Getting current version for {version_key}")
                current_version = await self.redis.get(version_key)
                current_version = int(current_version or 0)
                logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: Current version: {current_version}, Expected version: {version}")

                if current_version != version:
                    logger.warning(f"[{datetime.now()}] RedisStorage.compare_and_swap: Version mismatch for key {key}. Expected {version}, got {current_version}. Unwatching.")
                    await self.redis.unwatch()
                    return False

                # Start transaction
                logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: Starting pipeline for key {key}")
                async with self.redis.pipeline(transaction=True) as pipe:
                    # Update value and version atomically
                    await pipe.set(data_key, value, ex=self.ttl)
                    await pipe.incr(version_key)
                    await pipe.expire(version_key, self.ttl)

                    logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: Executing pipe for key {key} with timeout {REDIS_TIMEOUT_SECONDS}s...")
                    start_time = asyncio.get_event_loop().time()
                    try:
                        await asyncio.wait_for(pipe.execute(), timeout=REDIS_TIMEOUT_SECONDS)
                        end_time = asyncio.get_event_loop().time()
                        logger.debug(f"[{datetime.now()}] RedisStorage.compare_and_swap: Pipe executed successfully for key {key} in {end_time - start_time:.4f}s")
                        return True
                    except asyncio.TimeoutError:
                        end_time = asyncio.get_event_loop().time()
                        logger.error(f"[{datetime.now()}] Redis compare_and_swap pipe.execute() timed out after {end_time - start_time:.4f}s (>{REDIS_TIMEOUT_SECONDS}s) for key: {key}")
                        raise TimeoutError(f"Redis compare_and_swap operation timed out for key {key}")
                    except asyncio.CancelledError:
                        end_time = asyncio.get_event_loop().time()
                        logger.warning(f"[{datetime.now()}] Redis compare_and_swap pipe.execute() was cancelled after {end_time - start_time:.4f}s for key: {key}.")
                        raise # Re-raise CancelledError

            except redis.WatchError:
                # Version changed while watching
                logger.warning(f"[{datetime.now()}] Redis WATCH error for key {key}, retrying...")
                continue # Retry the operation
            except Exception as e:
                logger.exception(f"[{datetime.now()}] Unexpected error during Redis compare_and_swap for key {key}: {e}") # Use logger.exception
                raise # Re-raise other exceptions

    async def cleanup(self, key: str) -> None:
        """Close Redis connection"""
        await self.redis.delete(key)
        await self.redis.close()  # TODO: Check if this is needed


class RedisStateStorageBackendFactory(StateStorageBackendFactory):
    """Factory for creating RedisStorage instances"""

    def create_backend(self, config: RedisStateStorageConfig) -> StateStorageBackend:
        """Create a RedisStorage instance based on the provided config"""
        logger.debug(
            f"RedisStateStorageBackendFactory: {config.model_dump_json(indent=2)}"
        )
        return RedisStorage(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            ssl=config.redis_ssl,
            ttl=config.ttl,
        )

