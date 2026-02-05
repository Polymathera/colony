"""Distributed event bus using Redis pub-sub."""
import asyncio
from typing import Any, Awaitable, Callable

from ...distributed import get_polymathera
from ...distributed.redis_utils.client import RedisClient
from ...distributed.redis_utils.redis_om import RedisOM, DistributedStateSubscriber, DistributedStateUpdate
from ...utils import setup_logger
from .types import BlackboardEvent, EventFilter

logger = setup_logger(__name__)


class EventBus:
    """Distributed event bus using Redis pub-sub.

    Features:
    - TRUE pub-sub via Redis (no polling!)
    - Event filtering
    - Backpressure handling
    - Multiple subscribers per event
    """

    def __init__(
        self,
        app_name: str,
        scope: str,
        scope_id: str,
        max_queue_size: int = 1000,
        distributed: bool = True,
    ):
        self.app_name = app_name
        self.scope = scope
        self.scope_id = scope_id
        self.distributed = distributed
        self.namespace = f"{app_name}:blackboard:events:{scope}:{scope_id}"

        self.listeners: list[tuple[EventFilter | None, Callable[[BlackboardEvent], Awaitable[None]]]] = []
        self.event_queue: asyncio.Queue[BlackboardEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._dispatcher_task: asyncio.Task | None = None
        self._subscriber: DistributedStateSubscriber | None = None
        self._subscriber_task: asyncio.Task | None = None
        self.redis_client: RedisClient | None = None
        self.redis_om: RedisOM | None = None

    async def start(self) -> None:
        """Start event dispatcher and Redis subscriber."""
        # Initialize Redis if distributed
        if self.distributed:
            polymathera = get_polymathera()
            self.redis_client = await polymathera.get_redis_client()
            self.redis_om = RedisOM(
                redis_client=self.redis_client,
                namespace=self.namespace,
            )

            # Initialize events topic
            await self.redis_om.initialize_topics({"events": {}})

            # Subscribe to events topic
            self._subscriber = self.redis_om.subscribe_to_state_updates("events")

            # Start subscriber with callback
            def on_state_update(update: DistributedStateUpdate, ex):
                """Callback for Redis pub-sub events."""
                if ex:
                    logger.error(f"Error in Redis subscriber: {ex}", exc_info=True)
                    return True  # Continue listening

                if update and update.data:
                    # Enqueue event for local dispatch
                    try:
                        event = BlackboardEvent(
                            event_type=update.data.get("event_type"),
                            key=update.data.get("key"),
                            value=update.data.get("value"),
                            old_value=update.data.get("old_value"),
                            timestamp=update.data.get("timestamp"),
                            agent_id=update.data.get("agent_id"),
                            metadata=update.data.get("metadata", {}),
                        )
                        # Non-blocking enqueue (will drop if full)
                        try:
                            self.event_queue.put_nowait(event)
                        except asyncio.QueueFull:
                            logger.warning("Event queue full, dropping distributed event")
                    except Exception as e:
                        logger.error(f"Error processing distributed event: {e}", exc_info=True)

                return True  # Continue listening

            await self._subscriber.start(on_state_update)

        # Start local dispatcher
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop event dispatcher and Redis subscriber."""
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None

        if self._subscriber_task:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
            self._subscriber_task = None

        if self._subscriber:
            await self._subscriber.cancel()
            self._subscriber = None

    async def emit(self, event: BlackboardEvent) -> None:
        """Emit event locally and to distributed subscribers.

        This immediately dispatches to local listeners and publishes to Redis
        for other nodes to receive.

        Events are published via two channels:
        1. Redis pub-sub (via RedisOM) — for standard subscribers
        2. Redis Streams (XADD) — for consumer group subscribers (e.g., VCM replicas)

        The Redis Stream is bounded by MAXLEN to prevent unbounded growth.
        If no consumer group exists for the stream, events are trimmed
        automatically by Redis. This dual-publish enables VCM's
        ``BlackboardContextPageSource`` to use XREADGROUP for exactly-once
        delivery across replicas without requiring changes to event producers.
        """
        try:
            # Enqueue for local dispatch (immediate)
            # TODO: Will this result in double delivery to local subscribers (second time through Redis)?
            self.event_queue.put_nowait(event)

            # Publish to Redis topic for distributed subscribers
            if self.distributed and self.redis_om:
                import json as json_mod

                event_dict = {
                    "event_type": event.event_type,
                    "key": event.key,
                    "value": event.value,
                    "old_value": event.old_value,
                    "timestamp": event.timestamp,
                    "agent_id": event.agent_id,
                    "metadata": event.metadata,
                }

                await self.redis_om.update_state_topic(
                    "events",
                    event_dict,
                    replace_all=True,  # Each event is independent
                )

                # Also write to Redis Stream for consumer group subscribers.
                # Stream name: bb:events:{scope}:{scope_id}
                # Bounded by MAXLEN ~10000 to prevent unbounded growth.
                # This enables VCM's BlackboardContextPageSource to use
                # XREADGROUP for exactly-once delivery across replicas.
                if self.redis_client:
                    stream_name = f"bb:events:{self.scope}:{self.scope_id}"
                    stream_fields = {
                        "event_type": event.event_type or "",
                        "key": event.key or "",
                        "value": json_mod.dumps(event.value, default=str) if event.value is not None else "",
                        "version": str(getattr(event, "version", 1)),
                        "old_value": json_mod.dumps(event.old_value, default=str) if event.old_value is not None else "",
                        "timestamp": str(event.timestamp or 0),
                        "agent_id": event.agent_id or "",
                        "tags": json_mod.dumps(list(event.tags), default=str) if event.tags else "",
                        "metadata": json_mod.dumps(event.metadata, default=str) if event.metadata else "",
                    }
                    try:
                        async with self.redis_client.get_redis_connection() as conn:
                            await conn.xadd(
                                stream_name,
                                stream_fields,
                                maxlen=10000,
                                approximate=True,
                            )
                    except Exception as e:
                        # Stream publishing is best-effort — don't fail the emit
                        logger.debug(
                            f"Failed to XADD event to stream {stream_name}: {e}"
                        )

        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.event_type} {event.key}")
        except Exception as e:
            logger.error(f"Failed to emit event: {e}", exc_info=True)

    def subscribe(
        self,
        callback: Callable[[BlackboardEvent], Awaitable[None]],
        filter: EventFilter | None = None,
    ) -> None:
        """Subscribe to events with optional filter."""
        self.listeners.append((filter, callback))

    def unsubscribe(self, callback: Callable[[BlackboardEvent], Awaitable[None]]) -> None:
        """Unsubscribe from events."""
        self.listeners = [(f, cb) for f, cb in self.listeners if cb != callback]

    async def _dispatch_loop(self) -> None:
        """Dispatch events to local listeners."""
        while True:
            try:
                event = await self.event_queue.get()

                # Dispatch to matching listeners
                # The same callback should only be invoked once per event even if
                # multiple filters match. Sometimes the same callback is registered
                # multiple times with different filters.
                invoked = []
                for filter, callback in self.listeners:
                    if filter is None or (filter.matches(event) and callback not in invoked):
                        try:
                            await callback(event)
                            invoked.append(callback)
                        except Exception as e:
                            logger.error(f"Error in event listener: {e}", exc_info=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error dispatching event: {e}", exc_info=True)

    async def stream_events_via_consumer_group(
        self,
        event_queue: asyncio.Queue[BlackboardEvent],
        consumer_group: str,
        consumer_name: str,
    ) -> bool:
        """Set up Redis Streams consumer group subscription.

        Creates the consumer group if it doesn't exist, then starts a
        background task that reads events via XREADGROUP.

        This is a VCM-only code path — ensures each event is delivered to
        exactly one consumer in the group (across VCM replicas).

        Args:
            event_queue: Queue to deliver events to.
            consumer_group: Consumer group name.
            consumer_name: This consumer's unique name in the group.
        """
        import json as json_mod

        # Derive stream name from key pattern and scope.
        # EventBus writes to: bb:events:{scope}:{scope_id}
        # The scope/scope_id come from the blackboard's EventBus.
        stream_name = f"bb:events:{self.scope}:{self.scope_id}"

        if self.redis_client is None:
            logger.warning(
                "Cannot set up consumer group: EventBus has no Redis client. "
                "Falling back to standard pub-sub."
            )
            return False

        # Create consumer group if it doesn't exist (MKSTREAM creates the stream)
        try:
            async with self.redis_client.get_redis_connection() as conn:
                await conn.xgroup_create(
                    name=stream_name,
                    groupname=consumer_group,
                    id="0",
                    mkstream=True,
                )
            logger.info(
                f"Created consumer group '{consumer_group}' on stream '{stream_name}'"
            )
        except Exception as e:
            # Group already exists — this is expected on subsequent replicas
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Error creating consumer group: {e}")

        # Store for use by the VCM reconciliation (XAUTOCLAIM)
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name

        # Start background task to read from the consumer group
        async def _read_loop() -> None:
            while True:
                try:
                    async with self.redis_client.get_redis_connection() as conn:
                        # Read new messages for this consumer
                        messages = await conn.xreadgroup(
                            groupname=consumer_group,
                            consumername=consumer_name,
                            streams={stream_name: ">"},
                            count=10,
                            block=5000,  # 5 second block
                        )

                    if not messages:
                        continue

                    for _stream, msgs in messages:
                        for msg_id, msg_data in msgs:
                            try:
                                # Parse event from message data
                                event_data = {
                                    k.decode() if isinstance(k, bytes) else k:
                                    v.decode() if isinstance(v, bytes) else v
                                    for k, v in msg_data.items()
                                }
                                event = BlackboardEvent(
                                    event_type=event_data.get("event_type", "write"),
                                    key=event_data.get("key", ""),
                                    value=json_mod.loads(event_data.get("value", "{}")),
                                    version=int(event_data.get("version", 1)),
                                    old_value=(
                                        json_mod.loads(event_data["old_value"])
                                        if "old_value" in event_data and event_data["old_value"]
                                        else None
                                    ),
                                    timestamp=float(event_data.get("timestamp", 0)),
                                    agent_id=event_data.get("agent_id") or None,
                                    tags=(
                                        set(json_mod.loads(event_data["tags"]))
                                        if "tags" in event_data and event_data["tags"]
                                        else None
                                    ),
                                    metadata=(
                                        json_mod.loads(event_data["metadata"])
                                        if "metadata" in event_data and event_data["metadata"]
                                        else None
                                    ),
                                )
                                await event_queue.put(event)

                                # Acknowledge the message
                                async with self.redis_client.get_redis_connection() as ack_conn:
                                    await ack_conn.xack(
                                        stream_name, consumer_group, msg_id,
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Error processing message {msg_id} from "
                                    f"stream {stream_name}: {e}"
                                )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Error in consumer group read loop: {e}")
                    await asyncio.sleep(1.0)  # Back off on errors

        # Start the read loop as a background task
        self._consumer_group_task = asyncio.create_task(_read_loop())
        logger.info(
            f"Started consumer group reader: group={consumer_group}, "
            f"consumer={consumer_name}, stream={stream_name}"
        )
        return True

    async def claim_orphaned_events(self) -> list[tuple[Any, Any]]:
        """Claim orphaned events from crashed replicas via XAUTOCLAIM.

        Events that were delivered to a consumer but not acknowledged
        (because the replica crashed) are claimed by this replica and
        re-enqueued into the page source's event queue.

        Returns:
            List of claimed message IDs and data.
        """
        if not self.redis_client:
            return

        stream_name = self._stream_name
        consumer_group = self._consumer_group
        consumer_name = self._consumer_name

        messages = []
        try:
            async with self.redis_client.get_redis_connection() as conn:
                # Claim messages idle for > 60 seconds
                result = await conn.xautoclaim(
                    name=stream_name,
                    groupname=consumer_group,
                    consumername=consumer_name,
                    min_idle_time=60000,  # 60 seconds
                    start_id="0-0",
                    count=50,
                )
                # result format: (next_start_id, [(msg_id, msg_data), ...], deleted_ids)
                if result and len(result) >= 2:
                    claimed_msgs = result[1]
                    if claimed_msgs:
                        for msg_id, msg_data in claimed_msgs:
                            messages.append((msg_id, msg_data))
                        logger.info(
                            f"XAUTOCLAIM: claimed {len(claimed_msgs)} orphaned events "
                            f"for scope {self.scope_id}"
                        )
        except Exception as e:
            logger.debug(f"XAUTOCLAIM for scope {self.scope_id}: {e}")

        return messages

