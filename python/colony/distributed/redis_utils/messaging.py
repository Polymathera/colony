"""Distributed message broker implementation."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from .base import BaseStore, Pipeline

T = TypeVar("T")


class MessageBroker(BaseStore):
    """
    Distributed message broker with support for:
    - Pub/sub messaging
    - Message serialization
    - Topic namespacing
    - Atomic operations
    - Message filtering
    """

    def __init__(
        self,
        redis: Redis,
        namespace: str,
        serializer: Callable[[Any], str] = json.dumps,
        deserializer: Callable[[str], Any] = json.loads,
    ):
        """Initialize broker.

        Args:
            redis: Redis client
            namespace: Namespace for topic isolation
            serializer: Function to serialize messages
            deserializer: Function to deserialize messages
        """
        super().__init__(redis, namespace)
        self.serializer = serializer
        self.deserializer = deserializer
        self._subscribers: dict[str, PubSub] = {}

    async def publish(self, topic: str, message: Any) -> None:
        """Publish message to topic.

        Args:
            topic: Topic to publish to
            message: Message to publish
        """
        full_topic = self._build_key(topic)
        serialized = self.serializer(message)

        async def _publish(client: Redis | Pipeline, *args: Any) -> None:
            await client.publish(full_topic, serialized)

        await self._execute_atomic("publish", _publish)

    async def subscribe(
        self,
        topic: str,
        message_filter: Callable[[Any], bool] | None = None,
    ) -> AsyncIterator[Any]:
        """Subscribe to topic.

        Args:
            topic: Topic to subscribe to
            message_filter: Optional filter function for messages

        Yields:
            Deserialized messages
        """
        full_topic = self._build_key(topic)

        # Create new pubsub client
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(full_topic)
        self._subscribers[topic] = pubsub

        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is None:
                    continue

                try:
                    payload = self.deserializer(message["data"])
                    if message_filter is None or message_filter(payload):
                        yield payload
                except Exception as e:
                    self.logger.error(f"Failed to process message: {e}")
        finally:
            await self.unsubscribe(topic)

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from topic.

        Args:
            topic: Topic to unsubscribe from
        """
        if topic in self._subscribers:
            pubsub = self._subscribers[topic]
            await pubsub.unsubscribe(self._build_key(topic))
            await pubsub.close()
            del self._subscribers[topic]

    async def publish_batch(self, topic: str, messages: list[Any]) -> None:
        """Publish multiple messages atomically.

        Args:
            topic: Topic to publish to
            messages: List of messages to publish
        """
        if not messages:
            return

        full_topic = self._build_key(topic)
        serialized = [self.serializer(msg) for msg in messages]

        async def _publish_batch(client: Redis | Pipeline, *args: Any) -> None:
            for msg in serialized:
                await client.publish(full_topic, msg)

        await self._execute_atomic("publish_batch", _publish_batch)

    async def cleanup(self) -> None:
        """Cleanup all subscriptions."""
        for topic in list(self._subscribers.keys()):
            await self.unsubscribe(topic)
