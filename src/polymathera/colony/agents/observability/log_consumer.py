"""Log consumer — reads log records from Kafka and sinks to PostgreSQL.

Mirrors the SpanConsumer pattern: background task reads from colony.logs topic,
batch-inserts into the ``logs`` table. Designed for high throughput — batches
up to 100 records every 2 seconds.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class LogConsumer:
    """Consumes log records from Kafka and inserts them into PostgreSQL.

    Runs as a background task in the dashboard container. Each log record
    is inserted (not upserted — log_id is unique, no updates needed).
    """

    def __init__(
        self,
        kafka_bootstrap: str,
        db_pool: Any,  # asyncpg.Pool
        topic: str = "colony.logs",
        kafka_group_id: str = "colony-log-pg-sink",
    ):
        self._kafka_bootstrap = kafka_bootstrap
        self._db_pool = db_pool
        self._topic = topic
        self._kafka_group_id = kafka_group_id
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the consumer loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("LogConsumer started (group=%s, topic=%s)", self._kafka_group_id, self._topic)

    async def _run(self) -> None:
        """Main consume loop."""
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._kafka_bootstrap,
            group_id=self._kafka_group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda v: json.loads(v),
        )
        await consumer.start()
        logger.info("LogConsumer connected to Kafka at %s", self._kafka_bootstrap)
        try:
            while self._running:
                result = await consumer.getmany(timeout_ms=2000, max_records=100)
                batch: list[dict[str, Any]] = []
                for _tp, messages in result.items():
                    for msg in messages:
                        batch.append(msg.value)
                if batch:
                    await self._flush_batch(batch)
                    logger.debug("Flushed %d log records to PostgreSQL", len(batch))
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("LogConsumer error", exc_info=True)
        finally:
            await consumer.stop()
            logger.info("LogConsumer stopped")

    async def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        """Insert a batch of log records into PostgreSQL."""
        try:
            async with self._db_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO logs (
                        log_id, timestamp, level, logger_name, message,
                        module, func_name, line_no, pid, thread_name,
                        actor_class, node_id,
                        tenant_id, colony_id, session_id, run_id, trace_id,
                        exc_info
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8, $9, $10,
                        $11, $12,
                        $13, $14, $15, $16, $17,
                        $18
                    )
                    ON CONFLICT (log_id) DO NOTHING
                    """,
                    [self._record_to_params(entry) for entry in batch],
                )
        except Exception:
            logger.warning("Failed to flush %d log records to PostgreSQL", len(batch), exc_info=True)

    @staticmethod
    def _record_to_params(data: dict[str, Any]) -> tuple:
        """Convert a log record dict to a tuple of SQL parameters."""
        ts = data.get("timestamp")
        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)

        return (
            data.get("log_id"),
            timestamp,
            data.get("level", "INFO"),
            data.get("logger_name", ""),
            data.get("message", ""),
            data.get("module", ""),
            data.get("func_name", ""),
            data.get("line_no"),
            data.get("pid"),
            data.get("thread_name", ""),
            data.get("actor_class", ""),
            data.get("node_id", ""),
            data.get("tenant_id"),
            data.get("colony_id"),
            data.get("session_id"),
            data.get("run_id"),
            data.get("trace_id"),
            data.get("exc_info"),
        )

    async def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
