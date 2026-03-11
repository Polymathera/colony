"""Tests for distributed cache implementation."""

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..base import DistributedPipeline
from ..cache import DistributedCache


@pytest.fixture
async def redis():
    """Redis client fixture."""
    client = Redis.from_url("redis://localhost")
    try:
        yield client
    finally:
        await client.flushall()
        await client.close()


@pytest.fixture
def cache(redis):
    """Cache fixture with JSON serialization."""
    return DistributedCache(redis, "test", default_ttl=60, use_json=True)


@pytest.fixture
def pickle_cache(redis):
    """Cache fixture with pickle serialization."""
    return DistributedCache(redis, "test", default_ttl=60, use_json=False)


@pytest.mark.asyncio
async def test_set_get_basic(cache):
    """Test basic set/get operations."""
    # Set and get string
    await cache.set("str", "value")
    assert await cache.get("str") == "value"

    # Set and get dict
    data = {"key": "value", "nested": {"x": 1}}
    await cache.set("dict", data)
    assert await cache.get("dict") == data

    # Set and get list
    data = [1, 2, {"x": 3}]
    await cache.set("list", data)
    assert await cache.get("list") == data


@pytest.mark.asyncio
async def test_set_get_with_metadata(cache):
    """Test set/get with metadata."""
    value = {"name": "test"}
    metadata = {"timestamp": 123, "version": 2}

    await cache.set("key", value, metadata=metadata)

    # Get without metadata
    assert await cache.get("key") == value

    # Get with metadata
    result, meta = await cache.get("key", with_metadata=True)
    assert result == value
    assert meta == metadata


@pytest.mark.asyncio
async def test_ttl_handling(cache, redis):
    """Test TTL handling."""
    # Set with default TTL
    await cache.set("default", "value")
    ttl = await redis.ttl(f"{cache.namespace}:default")
    assert 0 < ttl <= 60

    # Set with custom TTL
    await cache.set("custom", "value", ttl=10)
    ttl = await redis.ttl(f"{cache.namespace}:custom")
    assert 0 < ttl <= 10

    # Set without TTL
    await cache.set("no_ttl", "value", ttl=None)
    ttl = await redis.ttl(f"{cache.namespace}:no_ttl")
    assert ttl == -1


@pytest.mark.asyncio
async def test_pickle_serialization(pickle_cache):
    """Test pickle serialization."""

    class CustomClass:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, CustomClass) and self.value == other.value

    obj = CustomClass(42)
    await pickle_cache.set("obj", obj)
    result = await pickle_cache.get("obj")
    assert result == obj


@pytest.mark.asyncio
async def test_delete_exists(cache):
    """Test delete and exists operations."""
    # Test exists
    assert not await cache.exists("key")
    await cache.set("key", "value")
    assert await cache.exists("key")

    # Test delete
    await cache.delete("key")
    assert not await cache.exists("key")
    assert await cache.get("key") is None


@pytest.mark.asyncio
async def test_increment(cache):
    """Test increment operation."""
    # Basic increment
    assert await cache.increment("counter") == 1
    assert await cache.increment("counter") == 2

    # Custom amount
    assert await cache.increment("counter", 3) == 5

    # Negative increment
    assert await cache.increment("counter", -2) == 3


@pytest.mark.asyncio
async def test_pipeline_operations(cache, redis):
    """Test operations within pipeline."""
    async with DistributedPipeline(redis) as pipe:
        await cache.set("key1", "value1")
        await cache.set("key2", "value2", metadata={"meta": "data"})
        await cache.increment("counter")
        results = await pipe.execute()

    assert len(results) > 0
    assert await cache.get("key1") == "value1"
    assert await cache.get("key2") == "value2"
    assert await cache.increment("counter") == 2


@pytest.mark.asyncio
async def test_cleanup(cache):
    """Test cleanup operation."""
    # Set multiple keys
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("other", "value3")

    # Cleanup specific pattern
    await cache.cleanup("key*")
    assert not await cache.exists("key1")
    assert not await cache.exists("key2")
    assert await cache.exists("other")

    # Cleanup all
    await cache.cleanup()
    assert not await cache.exists("other")


@pytest.mark.asyncio
async def test_error_handling(cache, redis):
    """Test error handling."""
    # Simulate Redis error
    await redis.close()

    with pytest.raises(RedisError):
        await cache.set("key", "value")

    with pytest.raises(RedisError):
        await cache.get("key")


@pytest.mark.asyncio
async def test_serialization_error_handling(cache, redis):
    """Test handling of serialization errors."""
    # Invalid JSON
    key = f"{cache.namespace}:invalid"
    await redis.set(key, "invalid json")

    result = await cache.get("invalid")
    assert result is None

    result, meta = await cache.get("invalid", with_metadata=True)
    assert result is None
    assert meta is None

