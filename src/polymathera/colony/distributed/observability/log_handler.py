"""KafkaLogHandler — Python logging handler that publishes log records to Kafka.

Attach to any logger to capture log records system-wide without modifying
existing log calls. Records are batched and published asynchronously.

Usage:
    handler = KafkaLogHandler(kafka_bootstrap="kafka:9092")
    await handler.start()
    logging.getLogger("polymathera.colony").addHandler(handler)

The handler enriches each record with execution context (tenant_id, colony_id,
session_id, run_id) when available, enabling powerful cross-session log queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any


logger = logging.getLogger(__name__)

# Avoid recursive logging from the handler itself or Kafka internals
_SUPPRESSED_LOGGERS = frozenset({
    "polymathera.colony.agents.observability.log_handler",
    "aiokafka",
    "kafka",
})


class KafkaLogHandler(logging.Handler):
    """Logging handler that publishes records to a Kafka topic.

    Records are queued in-process and flushed asynchronously by a background
    task. This keeps the handler non-blocking — ``emit()`` never awaits I/O.

    Each record is enriched with:
    - Execution context (tenant_id, colony_id, session_id, run_id) if available
    - Ray actor metadata (node_id, pid, actor_class) from the environment
    - Unique log_id for deduplication
    """

    def __init__(
        self,
        kafka_bootstrap: str,
        topic: str = "colony.logs",
        batch_size: int = 50,
        flush_interval: float = 2.0,
        max_queue_size: int = 10000,
        level: int = logging.DEBUG,
    ):
        super().__init__(level=level)
        self._kafka_bootstrap = kafka_bootstrap
        self._topic = topic
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._producer = None
        self._flush_task: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        """Start the Kafka producer and background flush task."""
        from aiokafka import AIOKafkaProducer

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=1,
            linger_ms=100,
            max_batch_size=131072,
        )
        await self._producer.start()
        self._started = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("KafkaLogHandler started (topic=%s)", self._topic)

    def emit(self, record: logging.LogRecord) -> None:
        """Queue a log record for async publishing. Never blocks."""
        # Suppress recursive logging
        if record.name in _SUPPRESSED_LOGGERS or record.name.startswith("aiokafka"):
            return

        if not self._started:
            return

        try:
            entry = self._format_record(record)
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            pass  # Drop under backpressure — never block the caller
        except Exception:
            pass  # Never let handler errors propagate to application code

    def _format_record(self, record: logging.LogRecord) -> dict[str, Any]:
        """Convert a LogRecord to a JSON-serializable dict."""
        # Extract execution context if available (best-effort)
        tenant_id = None
        colony_id = None
        session_id = None
        run_id = None
        trace_id = None
        try:
            from ...distributed.ray_utils.serving.context import get_execution_context
            ctx = get_execution_context()
            if ctx:
                tenant_id = ctx.tenant_id
                colony_id = ctx.colony_id
                session_id = ctx.session_id
                run_id = ctx.run_id
                trace_id = ctx.trace_id
        except Exception:
            pass

        return {
            "log_id": uuid.uuid4().hex[:16],
            "timestamp": record.created,
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func_name": record.funcName,
            "line_no": record.lineno,
            "pid": record.process,
            "thread_name": record.threadName,
            "actor_class": os.environ.get("POLYMATHERA_SERVING_CURRENT_DEPLOYMENT", ""),
            "node_id": os.environ.get("RAY_NODE_ID", ""),
            "tenant_id": tenant_id,
            "colony_id": colony_id,
            "session_id": session_id,
            "run_id": run_id,
            "trace_id": trace_id,
            "exc_info": self._format_exception(record),
        }

    @staticmethod
    def _format_exception(record: logging.LogRecord) -> str | None:
        """Format exception info if present."""
        if record.exc_info and record.exc_info[1] is not None:
            import traceback
            return "".join(traceback.format_exception(*record.exc_info))
        return None

    async def _flush_loop(self) -> None:
        """Background task: drain queue and send batches to Kafka."""
        while self._started:
            try:
                batch: list[dict[str, Any]] = []
                # Wait for first item
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=self._flush_interval
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    continue

                # Drain up to batch_size
                while len(batch) < self._batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                await self._send_batch(batch)

            except asyncio.CancelledError:
                # Flush remaining on shutdown
                await self._drain_remaining()
                return
            except Exception:
                # Never crash the flush loop
                await asyncio.sleep(1)

    async def _send_batch(self, batch: list[dict[str, Any]]) -> None:
        """Send a batch of log entries to Kafka."""
        if not self._producer or not self._started:
            return
        for entry in batch:
            try:
                # Key by session_id for partition affinity (all logs for a
                # session land on the same partition, preserving order)
                key = entry.get("session_id") or entry.get("actor_class") or ""
                await self._producer.send(self._topic, key=key, value=entry)
            except Exception:
                pass  # Best-effort — drop on failure

    async def _drain_remaining(self) -> None:
        """Flush any remaining queued entries on shutdown."""
        batch: list[dict[str, Any]] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._send_batch(batch)

    async def stop(self) -> None:
        """Flush remaining logs and stop the producer."""
        self._started = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        if self._producer:
            try:
                await self._producer.stop()
            except Exception:
                pass
            self._producer = None
        logger.info("KafkaLogHandler stopped")
