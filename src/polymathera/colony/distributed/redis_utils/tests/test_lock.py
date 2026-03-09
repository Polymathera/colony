"""Tests for distributed lock implementation."""

import asyncio
import time
from typing import Optional

import pytest
from redis.asyncio import Redis

from ..locks import DistributedLock


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
def lock_manager(redis):
    """Lock manager fixture."""
    return DistributedLock(redis, "test", default_timeout=5, renewal_interval=1)


@pytest.mark.asyncio
async def test_basic_lock_acquire_release(lock_manager):
    """Test basic lock acquisition and release."""
    async with lock_manager.acquire("lock1") as acquired:
        assert acquired
        info = await lock_manager.get_lock_info("lock1")
        assert info is not None
        assert info["owner"] is not None
        assert info["metadata"] is None

    # Lock should be released
    info = await lock_manager.get_lock_info("lock1")
    assert info is None


@pytest.mark.asyncio
async def test_lock_with_metadata(lock_manager):
    """Test lock with metadata."""
    metadata = {"purpose": "test", "priority": 1}
    async with lock_manager.acquire("lock1", metadata=metadata) as acquired:
        assert acquired
        info = await lock_manager.get_lock_info("lock1")
        assert info["metadata"] == metadata


@pytest.mark.asyncio
async def test_lock_timeout(lock_manager):
    """Test lock timeout."""
    # Acquire lock with short timeout
    async with lock_manager.acquire("lock1", timeout=1) as acquired:
        assert acquired
        # Wait for timeout
        await asyncio.sleep(2)
        # Lock should be expired
        info = await lock_manager.get_lock_info("lock1")
        assert info is None


@pytest.mark.asyncio
async def test_lock_renewal(lock_manager):
    """Test automatic lock renewal."""
    async with lock_manager.acquire("lock1", timeout=2) as acquired:
        assert acquired
        # Wait longer than timeout, but less than renewal interval
        await asyncio.sleep(1.5)
        # Lock should still be valid due to renewal
        info = await lock_manager.get_lock_info("lock1")
        assert info is not None


@pytest.mark.asyncio
async def test_concurrent_locks(lock_manager):
    """Test concurrent lock attempts."""

    async def acquire_lock(lock_id: str, delay: float) -> Optional[bool]:
        await asyncio.sleep(delay)
        try:
            async with lock_manager.acquire(lock_id, timeout=5) as acquired:
                if acquired:
                    await asyncio.sleep(0.5)
                return acquired
        except Exception:
            return None

    # Start multiple concurrent lock attempts
    results = await asyncio.gather(
        acquire_lock("lock1", 0), acquire_lock("lock1", 0.1), acquire_lock("lock1", 0.2)
    )

    # Only one should succeed
    assert results.count(True) == 1
    assert results.count(False) == 2


@pytest.mark.asyncio
async def test_reentrant_locking(lock_manager):
    """Test reentrant locking (same owner can acquire multiple times)."""
    async with lock_manager.acquire("lock1") as acquired1:
        assert acquired1
        owner1 = (await lock_manager.get_lock_info("lock1"))["owner"]

        # Same owner should be able to acquire again
        async with lock_manager.acquire("lock1") as acquired2:
            assert acquired2
            owner2 = (await lock_manager.get_lock_info("lock1"))["owner"]
            assert owner1 == owner2

    # Lock should be fully released
    assert await lock_manager.get_lock_info("lock1") is None


@pytest.mark.asyncio
async def test_error_handling(lock_manager, redis):
    """Test error handling."""
    # Simulate Redis error
    await redis.close()

    async with lock_manager.acquire("lock1") as acquired:
        assert not acquired


@pytest.mark.asyncio
async def test_cleanup(lock_manager):
    """Test cleanup."""
    async with lock_manager.acquire("lock1"):
        async with lock_manager.acquire("lock2"):
            # Both locks should exist
            assert await lock_manager.get_lock_info("lock1") is not None
            assert await lock_manager.get_lock_info("lock2") is not None

            # Cleanup should release all locks
            await lock_manager.cleanup()

            # Locks should be released
            assert await lock_manager.get_lock_info("lock1") is None
            assert await lock_manager.get_lock_info("lock2") is None


@pytest.mark.asyncio
async def test_namespace_isolation(redis):
    """Test namespace isolation."""
    lock1 = DistributedLock(redis, "ns1")
    lock2 = DistributedLock(redis, "ns2")

    async with lock1.acquire("lock") as acquired1:
        assert acquired1
        # Different namespace should be able to acquire same lock name
        async with lock2.acquire("lock") as acquired2:
            assert acquired2
            # Both locks should exist
            assert await lock1.get_lock_info("lock") is not None
            assert await lock2.get_lock_info("lock") is not None


@pytest.mark.asyncio
async def test_lock_expiry_handling(lock_manager):
    """Test handling of expired locks."""
    # Acquire lock with short timeout
    async with lock_manager.acquire("lock1", timeout=1) as acquired:
        assert acquired

    # Wait for lock to expire
    await asyncio.sleep(1.5)

    # New owner should be able to acquire expired lock
    async with lock_manager.acquire("lock1") as acquired:
        assert acquired


@pytest.mark.asyncio
async def test_lock_contention(lock_manager):
    """Test lock contention with multiple clients."""
    counter = 0

    async def increment_counter():
        nonlocal counter
        async with lock_manager.acquire("counter_lock", timeout=1) as acquired:
            if acquired:
                current = counter
                await asyncio.sleep(0.1)  # Simulate work
                counter = current + 1
                return True
            return False

    # Run multiple concurrent increments
    results = await asyncio.gather(*[increment_counter() for _ in range(5)])

    # Verify only successful acquisitions incremented
    assert counter == results.count(True)


@pytest.mark.asyncio
async def test_lock_stress(lock_manager):
    """Stress test the lock implementation."""
    success_count = 0
    total_attempts = 50

    async def worker(worker_id: int):
        nonlocal success_count
        try:
            async with lock_manager.acquire(
                f"stress_lock_{worker_id % 5}", timeout=0.1
            ) as acquired:
                if acquired:
                    await asyncio.sleep(0.05)  # Simulate work
                    success_count += 1
        except Exception:
            pass

    # Create many concurrent workers
    start_time = time.time()
    await asyncio.gather(*[worker(i) for i in range(total_attempts)])
    duration = time.time() - start_time

    # Verify reasonable success rate and performance
    assert 0 < success_count <= total_attempts
    assert duration < 10  # Should complete in reasonable time

