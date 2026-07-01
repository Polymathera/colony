"""Trace and span endpoints for observability."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from polymathera.colony.distributed.observability.feedback import (
    SpanFeedbackRatingError,
)

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


class SpanFeedbackRequest(BaseModel):
    rating: str
    note: str | None = None


@router.get("/traces/")
async def list_traces(
    limit: int = Query(100, le=500),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """List all traces (sessions with spans), ordered by most recent."""
    store = colony.get_span_query_store()
    if store is None:
        return []
    try:
        return await store.list_traces(limit=limit)
    except Exception as e:
        logger.warning("Failed to list traces: %s", e)
        return []


@router.get("/traces/{trace_id}/summary")
async def get_trace_summary(
    trace_id: str,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get aggregated summary for a trace."""
    store = colony.get_span_query_store()
    if store is None:
        return {}
    try:
        return await store.get_trace_summary(trace_id)
    except Exception as e:
        logger.warning("Failed to get trace summary: %s", e)
        return {}


@router.get("/traces/{trace_id}/spans")
async def get_trace_spans(
    trace_id: str,
    run_id: str | None = Query(None),
    kind: str | None = Query(None),
    ring: str | None = Query(None, description="Filter by Ring level: USER or KERNEL"),
    service_name: str | None = Query(None, description="Filter by service name"),
    limit: int = Query(5000, le=10000),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Get all spans for a trace, with optional filters.

    Each span carries its human feedback under ``feedback`` (a list of
    ``{author, rating, note, updated_wall}``; empty when unrated)."""
    store = colony.get_span_query_store()
    if store is None:
        return []
    try:
        spans = await store.get_spans(
            trace_id, run_id=run_id, kind=kind,
            ring=ring, service_name=service_name, limit=limit,
        )
    except Exception as e:
        logger.warning("Failed to get spans for trace %s: %s", trace_id, e)
        return []

    feedback_store = colony.get_span_feedback_store()
    feedback: dict[str, Any] = {}
    if feedback_store is not None:
        try:
            feedback = await feedback_store.get_for_trace(trace_id)
        except Exception as e:
            logger.warning("Failed to load span feedback for trace %s: %s", trace_id, e)
    for span in spans:
        span["feedback"] = feedback.get(span["span_id"], [])
    return spans


@router.post("/traces/{trace_id}/spans/{span_id}/feedback")
async def record_span_feedback(
    trace_id: str,
    span_id: str,
    body: SpanFeedbackRequest,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Record (or update) the caller's rating of a single span."""
    store = colony.get_span_feedback_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Feedback store unavailable")
    try:
        await store.record(
            trace_id=trace_id, span_id=span_id, author=_user["sub"],
            rating=body.rating, note=body.note,
        )
    except SpanFeedbackRatingError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"ok": True}


@router.get("/stream/traces/{trace_id}")
async def stream_trace_spans(
    trace_id: str,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> StreamingResponse:
    """SSE stream of spans for a trace.

    1. Sends all existing spans from PostgreSQL (initial load)
    2. Then consumes new spans from Kafka in real-time
    """

    async def event_generator():
        store = colony.get_span_query_store()

        # Phase 1: Send existing spans from PostgreSQL
        if store:
            try:
                existing = await store.get_spans(trace_id, limit=10000)
                for span in existing:
                    yield f"data: {json.dumps(span, default=str)}\n\n"
            except Exception as e:
                logger.warning("Failed to load existing spans: %s", e)

        # Phase 2: Stream new spans from Kafka
        try:
            from aiokafka import AIOKafkaConsumer

            consumer = AIOKafkaConsumer(
                "colony.spans",
                bootstrap_servers=colony._kafka_bootstrap or "kafka:9092",
                group_id=f"colony-sse-{trace_id}-{id(asyncio.current_task())}",
                auto_offset_reset="latest",
                value_deserializer=lambda v: json.loads(v),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )
            await consumer.start()
            try:
                async for msg in consumer:
                    # Filter by trace_id (Kafka key = trace_id)
                    if msg.key == trace_id:
                        yield f"data: {json.dumps(msg.value, default=str)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                await consumer.stop()
        except Exception as e:
            logger.warning("Kafka SSE consumer error: %s", e)
            # Fallback: poll PostgreSQL
            last_count = 0
            while True:
                await asyncio.sleep(2.0)
                if store:
                    try:
                        spans = await store.get_spans(trace_id, limit=10000)
                        if len(spans) > last_count:
                            for span in spans[last_count:]:
                                yield f"data: {json.dumps(span, default=str)}\n\n"
                            last_count = len(spans)
                    except Exception:
                        break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
