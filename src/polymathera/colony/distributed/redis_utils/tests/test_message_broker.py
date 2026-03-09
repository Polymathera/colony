"""Tests for distributed message broker implementation."""

import asyncio
import json
import pickle
from typing import Any

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..messaging import MessageBroker


def json_serializer(obj: Any) -> str:
    """JSON serializer."""
    return json.dumps(obj)


def json_deserializer(data: str) -> Any:
    """JSON deserializer."""
    return json.loads(data)


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
def broker(redis):
    """Message broker fixture with JSON serialization."""
    return MessageBroker(redis, "test", json_serializer, json_deserializer)


@pytest.fixture
def pickle_broker(redis):
    """Message broker fixture with pickle serialization."""
    return MessageBroker(redis, "test", pickle.dumps, pickle.loads)


@pytest.mark.asyncio
async def test_publish_subscribe(broker):
    """Test basic publish/subscribe functionality."""
    messages = []

    async def subscriber():
        async for message in broker.subscribe("topic1"):
            messages.append(message)
            if len(messages) == 3:
                break

    # Start subscriber
    task = asyncio.create_task(subscriber())

    # Publish messages
    await broker.publish("topic1", "message1")
    await broker.publish("topic1", {"key": "value"})
    await broker.publish("topic1", [1, 2, 3])

    # Wait for messages
    await task

    assert messages == ["message1", {"key": "value"}, [1, 2, 3]]


@pytest.mark.asyncio
async def test_multiple_subscribers(broker):
    """Test multiple subscribers receiving messages."""
    messages1 = []
    messages2 = []

    async def subscriber1():
        async for message in broker.subscribe("topic1"):
            messages1.append(message)
            if len(messages1) == 2:
                break

    async def subscriber2():
        async for message in broker.subscribe("topic1"):
            messages2.append(message)
            if len(messages2) == 2:
                break

    # Start subscribers
    task1 = asyncio.create_task(subscriber1())
    task2 = asyncio.create_task(subscriber2())

    # Publish messages
    await broker.publish("topic1", "message1")
    await broker.publish("topic1", "message2")

    # Wait for both subscribers
    await asyncio.gather(task1, task2)

    assert messages1 == ["message1", "message2"]
    assert messages2 == ["message1", "message2"]


@pytest.mark.asyncio
async def test_message_filtering(broker):
    """Test message filtering."""
    messages = []

    async def subscriber():
        # Only accept even numbers
        async for message in broker.subscribe("numbers", lambda x: x % 2 == 0):
            messages.append(message)
            if len(messages) == 2:
                break

    # Start subscriber
    task = asyncio.create_task(subscriber())

    # Publish numbers
    for i in range(5):
        await broker.publish("numbers", i)

    # Wait for filtered messages
    await task

    assert messages == [0, 2, 4][:2]  # Only first two even numbers


@pytest.mark.asyncio
async def test_publish_batch(broker):
    """Test batch publishing."""
    messages = []

    async def subscriber():
        async for message in broker.subscribe("topic1"):
            messages.append(message)
            if len(messages) == 3:
                break

    # Start subscriber
    task = asyncio.create_task(subscriber())

    # Publish batch of messages
    batch = ["message1", "message2", "message3"]
    await broker.publish_batch("topic1", batch)

    # Wait for messages
    await task

    assert messages == batch


@pytest.mark.asyncio
async def test_pickle_serialization(pickle_broker):
    """Test pickle serialization for complex objects."""

    class ComplexObject:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, ComplexObject) and self.value == other.value

    messages = []

    async def subscriber():
        async for message in pickle_broker.subscribe("topic1"):
            messages.append(message)
            if len(messages) == 1:
                break

    # Start subscriber
    task = asyncio.create_task(subscriber())

    # Publish complex object
    obj = ComplexObject(42)
    await pickle_broker.publish("topic1", obj)

    # Wait for message
    await task

    assert messages == [obj]


@pytest.mark.asyncio
async def test_namespace_isolation(redis):
    """Test namespace isolation."""
    broker1 = MessageBroker(redis, "ns1", json_serializer, json_deserializer)
    broker2 = MessageBroker(redis, "ns2", json_serializer, json_deserializer)

    messages1 = []
    messages2 = []

    async def subscriber1():
        async for message in broker1.subscribe("topic"):
            messages1.append(message)
            if len(messages1) == 1:
                break

    async def subscriber2():
        async for message in broker2.subscribe("topic"):
            messages2.append(message)
            if len(messages2) == 1:
                break

    # Start subscribers
    task1 = asyncio.create_task(subscriber1())
    task2 = asyncio.create_task(subscriber2())

    # Publish to both namespaces
    await broker1.publish("topic", "message1")
    await broker2.publish("topic", "message2")

    # Wait for messages
    await asyncio.gather(task1, task2)

    assert messages1 == ["message1"]
    assert messages2 == ["message2"]


@pytest.mark.asyncio
async def test_error_handling(broker, redis):
    """Test error handling."""
    # Simulate Redis error
    await redis.close()

    with pytest.raises(RedisError):
        await broker.publish("topic1", "message")


@pytest.mark.asyncio
async def test_serialization_error_handling(broker):
    """Test serialization error handling."""
    # Try to publish un-serializable object
    with pytest.raises(TypeError):
        await broker.publish("topic1", object())


@pytest.mark.asyncio
async def test_cleanup(broker):
    """Test cleanup."""
    messages = []

    async def subscriber():
        try:
            async for message in broker.subscribe("topic1"):
                messages.append(message)
        except asyncio.CancelledError:
            pass

    # Start subscriber
    task = asyncio.create_task(subscriber())

    # Publish message
    await broker.publish("topic1", "message1")

    # Wait briefly
    await asyncio.sleep(0.1)

    # Cleanup should cancel subscription
    await broker.cleanup()

    # Cancel subscriber task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert messages == ["message1"]
    # Verify subscription was cleaned up
    assert not broker._pubsub._subscribed

