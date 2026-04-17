"""Span producer — writes spans to Kafka."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Span

logger = logging.getLogger(__name__)


class SpanProducer:
    """Produces span records to a Kafka topic.

    Each span is keyed by trace_id (= session_id) for partition affinity,
    ensuring all spans for a session land on the same partition in order.
    """

    def __init__(self, kafka_bootstrap: str, topic: str = "colony.spans"):
        self._kafka_bootstrap = kafka_bootstrap
        self._topic = topic
        self._producer = None
        self._started = False

    async def start(self) -> None:
        """Initialize and start the Kafka producer."""
        from aiokafka import AIOKafkaProducer

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=1,
            linger_ms=50,
            max_batch_size=65536,
        )
        await self._producer.start()
        self._started = True
        logger.info("SpanProducer started (bootstrap=%s, topic=%s)", self._kafka_bootstrap, self._topic)

    async def send_spans(self, spans: list[Span]) -> None:
        """Send a batch of spans to Kafka."""
        if not self._started or not self._producer:
            logger.warning("SpanProducer not started, dropping %d spans", len(spans))
            return
        for span in spans:
            try:
                await self._producer.send(
                    self._topic,
                    key=span.trace_id,
                    value=span.model_dump(mode="json"),
                )
            except Exception:
                logger.warning("Failed to send span %s", span.span_id, exc_info=True)

    async def stop(self) -> None:
        """Flush and stop the producer."""
        if self._producer and self._started:
            try:
                await self._producer.stop()
            except Exception:
                logger.warning("Error stopping SpanProducer", exc_info=True)
            self._started = False
            logger.info("SpanProducer stopped")
