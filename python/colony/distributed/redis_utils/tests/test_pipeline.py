"""Tests for distributed pipeline implementation."""

import asyncio

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..base import DistributedPipeline
from ..cache import DistributedCache
from ..messaging import MessageBroker
from ..priority_queue import DistributedPriorityQueue
from ..sets import DistributedSet


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
    """Cache fixture."""
    return DistributedCache(redis, "test_cache", default_ttl=60, use_json=True)


@pytest.fixture
def set_store(redis):
    """Set store fixture."""
    return DistributedSet(redis, "test_set")


@pytest.fixture
def queue(redis):
    """Priority queue fixture."""
    return DistributedPriorityQueue(redis, "test_queue", default_ttl=60)


@pytest.fixture
def broker(redis):
    """Message broker fixture."""
    return MessageBroker(redis, "test_broker", lambda x: x, lambda x: x)


@pytest.mark.asyncio
async def test_basic_pipeline(redis):
    """Test basic pipeline functionality."""
    async with DistributedPipeline(redis) as pipe:
        # Queue multiple commands
        await redis.set("key1", "value1")
        await redis.set("key2", "value2")
        await redis.incr("counter")
        results = await pipe.execute()

    assert len(results) == 3
    assert await redis.get("key1") == b"value1"
    assert await redis.get("key2") == b"value2"
    assert await redis.get("counter") == b"1"


@pytest.mark.asyncio
async def test_pipeline_with_cache(cache):
    """Test pipeline with cache operations."""
    async with DistributedPipeline(cache._redis) as pipe:
        await cache.set("key1", {"field": "value1"})
        await cache.set("key2", {"field": "value2"})
        await cache.increment("counter")
        results = await pipe.execute()

    assert len(results) > 0
    assert await cache.get("key1") == {"field": "value1"}
    assert await cache.get("key2") == {"field": "value2"}
    assert await cache.get("counter") == 1


@pytest.mark.asyncio
async def test_pipeline_with_set(set_store):
    """Test pipeline with set operations."""
    async with DistributedPipeline(set_store._redis) as pipe:
        await set_store.add_member("set1", "member1")
        await set_store.add_members("set1", {"member2", "member3"})
        await set_store.remove_member("set1", "member2")
        results = await pipe.execute()

    assert len(results) > 0
    members = await set_store.get_members("set1")
    assert members == {"member1", "member3"}


@pytest.mark.asyncio
async def test_pipeline_with_queue(queue):
    """Test pipeline with priority queue operations."""
    async with DistributedPipeline(queue._redis) as pipe:
        await queue.add_item("queue1", "item1", 1.0)
        await queue.add_item("queue1", "item2", 2.0)
        await queue.update_priority("queue1", "item1", 3.0)
        results = await pipe.execute()

    assert len(results) > 0
    items = await queue.get_items_by_priority("queue1")
    assert items == ["item2", "item1"]


@pytest.mark.asyncio
async def test_pipeline_with_multiple_stores(cache, set_store, queue):
    """Test pipeline with multiple store types."""
    async with DistributedPipeline(cache._redis) as pipe:
        # Cache operations
        await cache.set("cache_key", "value")
        await cache.increment("counter")

        # Set operations
        await set_store.add_member("set1", "member1")
        await set_store.add_member("set1", "member2")

        # Queue operations
        await queue.add_item("queue1", "item1", 1.0)
        await queue.add_item("queue1", "item2", 2.0)

        results = await pipe.execute()

    assert len(results) > 0

    # Verify cache results
    assert await cache.get("cache_key") == "value"
    assert await cache.get("counter") == 1

    # Verify set results
    assert await set_store.get_members("set1") == {"member1", "member2"}

    # Verify queue results
    assert await queue.get_items_by_priority("queue1") == ["item1", "item2"]


@pytest.mark.asyncio
async def test_pipeline_error_handling(redis):
    """Test pipeline error handling."""
    # Close Redis connection to simulate error
    await redis.close()

    with pytest.raises(RedisError):
        async with DistributedPipeline(redis) as pipe:
            await redis.set("key1", "value1")
            await redis.set("key2", "value2")
            await pipe.execute()


@pytest.mark.asyncio
async def test_nested_pipelines(redis):
    """Test nested pipelines are not allowed."""
    async with DistributedPipeline(redis) as pipe1:
        with pytest.raises(RuntimeError):
            async with DistributedPipeline(redis) as pipe2:
                pass


@pytest.mark.asyncio
async def test_pipeline_with_watch(redis):
    """Test pipeline with watched keys."""
    # Set initial value
    await redis.set("counter", "0")

    # Start transaction with watch
    async with DistributedPipeline(redis) as pipe:
        await pipe.watch("counter")
        value = await redis.get("counter")

        # Simulate concurrent modification
        await redis.incr("counter")

        # Try to update based on watched value
        await redis.set("counter", int(value) + 1)
        results = await pipe.execute()

    # Transaction should fail due to modified watched key
    assert results is None
    assert await redis.get("counter") == b"1"


@pytest.mark.asyncio
async def test_pipeline_transaction_atomicity(redis):
    """Test pipeline transaction atomicity."""
    async with DistributedPipeline(redis) as pipe:
        await redis.set("key1", "value1")
        # This command will fail
        await redis.incr("key1")
        await redis.set("key2", "value2")
        # Transaction should fail entirely
        results = await pipe.execute()

    assert results is None
    assert await redis.get("key1") is None
    assert await redis.get("key2") is None


@pytest.mark.asyncio
async def test_pipeline_large_batch(redis):
    """Test pipeline with large batch of operations."""
    num_operations = 1000

    async with DistributedPipeline(redis) as pipe:
        for i in range(num_operations):
            await redis.set(f"key{i}", f"value{i}")
        results = await pipe.execute()

    assert len(results) == num_operations
    for i in range(num_operations):
        assert await redis.get(f"key{i}") == f"value{i}".encode()


@pytest.mark.asyncio
async def test_pipeline_concurrent_access(redis):
    """Test concurrent access to pipeline."""

    async def worker(worker_id: int):
        async with DistributedPipeline(redis) as pipe:
            for i in range(10):
                key = f"worker{worker_id}_key{i}"
                await redis.set(key, f"value{i}")
            await pipe.execute()

    # Run multiple workers concurrently
    workers = [worker(i) for i in range(5)]
    await asyncio.gather(*workers)

    # Verify all keys were set
    for worker_id in range(5):
        for i in range(10):
            key = f"worker{worker_id}_key{i}"
            assert await redis.get(key) == f"value{i}".encode()

