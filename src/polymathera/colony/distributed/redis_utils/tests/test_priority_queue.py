"""Tests for distributed priority queue implementation."""

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..base import DistributedPipeline
from ..priority_queue import DistributedPriorityQueue


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
def queue(redis):
    """Priority queue fixture."""
    return DistributedPriorityQueue(redis, "test", default_ttl=3600)


@pytest.mark.asyncio
async def test_add_get_items(queue):
    """Test adding and getting items."""
    # Single item
    await queue.add_item("queue1", "item1", 1.0)
    item = await queue.get_item("queue1", "item1")
    assert item["priority"] == 1.0
    assert item["metadata"] is None

    # Multiple items with metadata
    items = {"item2": 2.0, "item3": 3.0, "item4": 4.0}
    metadata = {"item2": {"tag": "a"}, "item3": {"tag": "b"}, "item4": {"tag": "c"}}
    await queue.add_items("queue1", items, metadata)

    # Verify items and metadata
    for item_id, priority in items.items():
        item = await queue.get_item("queue1", item_id)
        assert item["priority"] == priority
        assert item["metadata"] == metadata[item_id]


@pytest.mark.asyncio
async def test_get_items_by_priority(queue):
    """Test getting items ordered by priority."""
    # Add items with different priorities
    items = {"item1": 3.0, "item2": 1.0, "item3": 4.0, "item4": 2.0}
    await queue.add_items("queue1", items)

    # Get all items with priorities
    result = await queue.get_items_by_priority("queue1", with_priorities=True)
    assert [item["id"] for item in result] == ["item2", "item4", "item1", "item3"]
    assert [item["priority"] for item in result] == [1.0, 2.0, 3.0, 4.0]

    # Get items without priorities
    result = await queue.get_items_by_priority("queue1")
    assert result == ["item2", "item4", "item1", "item3"]


@pytest.mark.asyncio
async def test_get_items_by_priority_batch(queue):
    """Test getting items in batches."""
    # Add many items
    items = {f"item{i}": float(i) for i in range(250)}
    await queue.add_items("queue1", items)

    # Get in batches
    retrieved = []
    async for batch in queue.get_items_by_priority_batch("queue1", batch_size=100):
        retrieved.extend(batch)

    assert len(retrieved) == 250
    assert retrieved == [f"item{i}" for i in range(250)]


@pytest.mark.asyncio
async def test_update_priority(queue):
    """Test updating item priority."""
    # Setup
    await queue.add_item("queue1", "item1", 1.0)
    await queue.add_item("queue1", "item2", 2.0)

    # Update priority
    await queue.update_priority("queue1", "item1", 3.0)

    # Verify new order
    items = await queue.get_items_by_priority("queue1")
    assert items == ["item2", "item1"]


@pytest.mark.asyncio
async def test_highest_priority(queue):
    """Test getting highest priority item."""
    # Empty queue
    assert await queue.get_highest_priority("queue1") is None

    # Add items
    await queue.add_item("queue1", "item1", 2.0)
    await queue.add_item("queue1", "item2", 1.0)
    await queue.add_item("queue1", "item3", 3.0)

    # Get highest priority (lowest score)
    item = await queue.get_highest_priority("queue1")
    assert item["id"] == "item2"
    assert item["priority"] == 1.0


@pytest.mark.asyncio
async def test_remove_item(queue):
    """Test removing items."""
    # Setup
    await queue.add_item("queue1", "item1", 1.0, {"tag": "a"})
    await queue.add_item("queue1", "item2", 2.0, {"tag": "b"})

    # Remove item
    await queue.remove_item("queue1", "item1")

    # Verify
    assert await queue.get_item("queue1", "item1") is None
    assert await queue.get_items_by_priority("queue1") == ["item2"]


@pytest.mark.asyncio
async def test_ttl_handling(queue, redis):
    """Test TTL handling."""
    # Add item with custom TTL
    await queue.add_item("queue1", "item1", 1.0, ttl=1)

    # Verify TTL is set
    ttl = await redis.ttl(queue._build_namespaced_key("queue1"))
    assert ttl > 0

    # Add item with default TTL
    await queue.add_item("queue2", "item1", 1.0)
    ttl = await redis.ttl(queue._build_namespaced_key("queue2"))
    assert ttl > 0


@pytest.mark.asyncio
async def test_pipeline_operations(queue, redis):
    """Test operations within pipeline."""
    async with DistributedPipeline(redis) as pipe:
        await queue.add_item("queue1", "item1", 1.0)
        await queue.add_item("queue1", "item2", 2.0)
        await queue.update_priority("queue1", "item1", 3.0)
        results = await pipe.execute()

    assert len(results) > 0
    items = await queue.get_items_by_priority("queue1")
    assert items == ["item2", "item1"]


@pytest.mark.asyncio
async def test_count_items(queue):
    """Test counting items."""
    assert await queue.count_items("queue1") == 0

    await queue.add_items("queue1", {"item1": 1.0, "item2": 2.0, "item3": 3.0})
    assert await queue.count_items("queue1") == 3

    await queue.remove_item("queue1", "item2")
    assert await queue.count_items("queue1") == 2


@pytest.mark.asyncio
async def test_error_handling(queue, redis):
    """Test error handling."""
    # Simulate Redis error
    await redis.close()

    with pytest.raises(RedisError):
        await queue.add_item("queue1", "item1", 1.0)

    with pytest.raises(RedisError):
        await queue.get_items_by_priority("queue1")


@pytest.mark.asyncio
async def test_namespace_isolation(redis):
    """Test namespace isolation."""
    queue1 = DistributedPriorityQueue(redis, "ns1")
    queue2 = DistributedPriorityQueue(redis, "ns2")

    # Add same item to both queues
    await queue1.add_item("queue", "item", 1.0)
    await queue2.add_item("queue", "item", 2.0)

    # Verify isolation
    item1 = await queue1.get_item("queue", "item")
    item2 = await queue2.get_item("queue", "item")
    assert item1["priority"] == 1.0
    assert item2["priority"] == 2.0

