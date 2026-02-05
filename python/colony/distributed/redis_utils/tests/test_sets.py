"""Tests for distributed set implementation."""

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..base import DistributedPipeline
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
def set_store(redis):
    """Set store fixture."""
    return DistributedSet(redis, "test")


@pytest.mark.asyncio
async def test_add_get_members(set_store):
    """Test adding and getting members."""
    # Single member
    await set_store.add_member("set1", "member1")
    assert await set_store.get_members("set1") == {"member1"}

    # Multiple members
    members = {"member2", "member3", "member4"}
    await set_store.add_members("set1", members)
    assert await set_store.get_members("set1") == {
        "member1",
        "member2",
        "member3",
        "member4",
    }


@pytest.mark.asyncio
async def test_remove_members(set_store):
    """Test removing members."""
    # Setup
    members = {"member1", "member2", "member3"}
    await set_store.add_members("set1", members)

    # Remove single member
    await set_store.remove_member("set1", "member2")
    assert await set_store.get_members("set1") == {"member1", "member3"}

    # Remove multiple members
    await set_store.remove_members("set1", {"member1", "member3"})
    assert await set_store.get_members("set1") == set()


@pytest.mark.asyncio
async def test_get_members_batch(set_store):
    """Test getting members in batches."""
    # Add many members
    members = {f"member{i}" for i in range(250)}
    await set_store.add_members("set1", members)

    # Get in batches
    retrieved = set()
    async for batch in set_store.get_members_batch("set1", batch_size=100):
        retrieved.update(batch)

    assert retrieved == members


@pytest.mark.asyncio
async def test_set_operations(set_store):
    """Test set operations (union, intersection)."""
    # Setup sets
    await set_store.add_members("set1", {"a", "b", "c"})
    await set_store.add_members("set2", {"b", "c", "d"})
    await set_store.add_members("set3", {"c", "d", "e"})

    # Test union
    union = await set_store.get_union(["set1", "set2"])
    assert union == {"a", "b", "c", "d"}

    # Test intersection
    intersection = await set_store.get_intersection(["set1", "set2", "set3"])
    assert intersection == {"c"}


@pytest.mark.asyncio
async def test_move_member(set_store):
    """Test moving member between sets."""
    # Setup
    await set_store.add_members("source", {"a", "b"})
    await set_store.add_members("dest", {"c"})

    # Move member
    success = await set_store.move_member("source", "dest", "b")
    assert success

    # Verify
    assert await set_store.get_members("source") == {"a"}
    assert await set_store.get_members("dest") == {"b", "c"}

    # Try moving non-existent member
    success = await set_store.move_member("source", "dest", "x")
    assert not success


@pytest.mark.asyncio
async def test_count_and_membership(set_store):
    """Test counting members and checking membership."""
    members = {"a", "b", "c"}
    await set_store.add_members("set1", members)

    # Test count
    assert await set_store.count_members("set1") == 3

    # Test membership
    assert await set_store.is_member("set1", "b")
    assert not await set_store.is_member("set1", "x")


@pytest.mark.asyncio
async def test_pipeline_operations(set_store, redis):
    """Test operations within pipeline."""
    async with DistributedPipeline(redis) as pipe:
        await set_store.add_member("set1", "a")
        await set_store.add_members("set1", {"b", "c"})
        await set_store.remove_member("set1", "b")
        results = await pipe.execute()

    assert len(results) > 0
    assert await set_store.get_members("set1") == {"a", "c"}


@pytest.mark.asyncio
async def test_empty_operations(set_store):
    """Test operations with empty inputs."""
    # Empty members list
    await set_store.add_members("set1", set())
    assert await set_store.get_members("set1") == set()

    # Empty remove list
    await set_store.remove_members("set1", set())
    assert await set_store.get_members("set1") == set()

    # Empty union/intersection
    assert await set_store.get_union([]) == set()
    assert await set_store.get_intersection([]) == set()


@pytest.mark.asyncio
async def test_error_handling(set_store, redis):
    """Test error handling."""
    # Simulate Redis error
    await redis.close()

    with pytest.raises(RedisError):
        await set_store.add_member("set1", "member1")

    with pytest.raises(RedisError):
        await set_store.get_members("set1")


@pytest.mark.asyncio
async def test_namespace_isolation(redis):
    """Test namespace isolation."""
    set1 = DistributedSet(redis, "ns1")
    set2 = DistributedSet(redis, "ns2")

    # Add same member to both sets
    await set1.add_member("set", "member")
    await set2.add_member("set", "member")

    # Modify one set
    await set1.add_member("set", "extra")

    # Verify isolation
    assert await set1.get_members("set") == {"member", "extra"}
    assert await set2.get_members("set") == {"member"}


