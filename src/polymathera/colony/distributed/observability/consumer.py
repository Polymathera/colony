"""Span consumer — reads spans from Kafka and sinks to PostgreSQL."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SpanConsumer:
    """Consumes span records from Kafka and upserts them into PostgreSQL.

    Runs as a background task. Spans arrive as JSON from the colony.spans topic.
    Each span is upserted (INSERT ... ON CONFLICT DO UPDATE) so that span
    completion updates (end_time, status, output_summary) are applied.
    """

    def __init__(
        self,
        kafka_bootstrap: str,
        db_pool: Any,  # asyncpg.Pool
        topic: str = "colony.spans",
        kafka_group_id: str = "colony-pg-sink",
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
        logger.info("SpanConsumer started (group=%s, topic=%s)", self._kafka_group_id, self._topic)

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
        logger.info("SpanConsumer connected to Kafka at %s", self._kafka_bootstrap)
        try:
            while self._running:
                # getmany with timeout ensures partial batches get flushed
                result = await consumer.getmany(timeout_ms=2000, max_records=50)
                batch: list[dict[str, Any]] = []
                for _tp, messages in result.items():
                    for msg in messages:
                        batch.append(msg.value)
                if batch:
                    await self._flush_batch(batch)
                    logger.debug("Flushed %d spans to PostgreSQL", len(batch))
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("SpanConsumer error", exc_info=True)
        finally:
            await consumer.stop()
            logger.info("SpanConsumer stopped")

    async def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        """Upsert a batch of spans into PostgreSQL."""
        try:
            async with self._db_pool.acquire() as conn:
                for span_data in batch:
                    await self._upsert_span(conn, span_data)
        except Exception:
            logger.warning("Failed to flush %d spans to PostgreSQL", len(batch), exc_info=True)

    @staticmethod
    async def _upsert_span(conn: Any, data: dict[str, Any]) -> None:
        """Upsert a single span record."""
        start_wall = data.get("start_wall")
        start_wall_ts = datetime.fromtimestamp(start_wall, tz=timezone.utc) if start_wall else None

        end_time = data.get("end_time")
        start_time = data.get("start_time")
        duration_ms = None
        end_wall_ts = None
        if end_time is not None and start_time is not None and start_wall is not None:
            duration_ms = (end_time - start_time) * 1000
            end_wall_ts = datetime.fromtimestamp(start_wall + (end_time - start_time), tz=timezone.utc)

        await conn.execute(
            """
            INSERT INTO spans (
                span_id, trace_id, parent_span_id, run_id,
                agent_id, name, kind,
                start_wall, end_wall, duration_ms,
                status, error,
                input_summary, output_summary,
                input_tokens, output_tokens, cache_read_tokens,
                model_name, context_page_ids,
                ring, service_name,
                tags, metadata
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9, $10,
                $11, $12,
                $13, $14,
                $15, $16, $17,
                $18, $19,
                $20, $21,
                $22, $23
            )
            ON CONFLICT (span_id) DO UPDATE SET
                end_wall = EXCLUDED.end_wall,
                duration_ms = EXCLUDED.duration_ms,
                status = EXCLUDED.status,
                error = EXCLUDED.error,
                output_summary = EXCLUDED.output_summary,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                cache_read_tokens = EXCLUDED.cache_read_tokens,
                ring = COALESCE(EXCLUDED.ring, spans.ring),
                service_name = COALESCE(EXCLUDED.service_name, spans.service_name)
            """,
            data.get("span_id"),
            data.get("trace_id"),
            data.get("parent_span_id"),
            data.get("run_id"),
            data.get("agent_id"),
            data.get("name"),
            data.get("kind"),
            start_wall_ts,
            end_wall_ts,
            duration_ms,
            data.get("status", "running"),
            data.get("error"),
            json.dumps(data.get("input_summary", {})),
            json.dumps(data.get("output_summary", {})),
            data.get("input_tokens"),
            data.get("output_tokens"),
            data.get("cache_read_tokens"),
            data.get("model_name"),
            data.get("context_page_ids"),
            data.get("ring"),
            data.get("service_name"),
            data.get("tags", []),
            json.dumps(data.get("metadata", {})),
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
