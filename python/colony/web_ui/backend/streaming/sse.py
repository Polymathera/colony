"""Server-Sent Events for streaming scraped Prometheus metrics.

Colony runs a prometheus_client HTTP server on each node (port 9090)
that exposes /metrics in Prometheus text format. There is NO Prometheus
server with PromQL — so we scrape the raw text, parse it, and stream
structured JSON snapshots to the frontend via SSE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, Depends, Query
from starlette.responses import StreamingResponse

from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()

# Metrics we care about (prefix match against scraped metric names)
METRIC_PREFIXES = [
    "blackboard_",
    "cache_",
    "redis_",
    "token_manager_",
    "process_",
]

# In-memory rolling buffer shared across SSE connections
_history: deque[dict[str, Any]] = deque(maxlen=720)  # 1h at 5s intervals


class _ParseResult:
    """Parsed Prometheus text: samples + type annotations."""
    __slots__ = ("metrics", "types")

    def __init__(self) -> None:
        self.metrics: dict[str, list[dict[str, Any]]] = {}
        # metric_name → "counter" | "gauge" | "histogram" | "summary" | "untyped"
        self.types: dict[str, str] = {}


def _parse_prometheus_text(text: str) -> _ParseResult:
    """Parse Prometheus text exposition format into structured dicts.

    Reads ``# TYPE`` annotations so the frontend knows how to render each
    metric (counter → rate chart, gauge → value chart, etc.).
    Only includes metrics matching METRIC_PREFIXES.
    """
    result = _ParseResult()
    for line in text.splitlines():
        if not line:
            continue

        # Capture TYPE annotations: # TYPE metric_name type
        if line.startswith("# TYPE "):
            parts = line.split(None, 4)  # ["#", "TYPE", name, type]
            if len(parts) >= 4:
                name, mtype = parts[2], parts[3]
                if any(name.startswith(p) for p in METRIC_PREFIXES):
                    result.types[name] = mtype
            continue

        if line.startswith("#"):
            continue

        # Match: metric_name{label="val",...} value [timestamp]
        # or:    metric_name value [timestamp]
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?(.*?)\}?\s+([\d.eE+\-]+)', line)
        if not m:
            # Try without labels
            m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.eE+\-]+)', line)
            if not m:
                continue
            name = m.group(1)
            labels_str = ""
            val_str = m.group(2)
        else:
            name = m.group(1)
            labels_str = m.group(2)
            val_str = m.group(3)

        # Filter to interesting metrics
        if not any(name.startswith(p) for p in METRIC_PREFIXES):
            continue

        # Parse labels
        labels: dict[str, str] = {}
        if labels_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                labels[lm.group(1)] = lm.group(2)

        try:
            value = float(val_str)
        except ValueError:
            continue

        result.metrics.setdefault(name, []).append({"labels": labels, "value": value})

    return result


def _aggregate_snapshot(parsed: _ParseResult) -> dict[str, Any]:
    """Produce a simplified snapshot for charting.

    For counters (*_total): sum all label combinations.
    For histograms (*_sum, *_count): sum.
    For gauges: sum.
    Skip _bucket and _created suffixes.

    Includes ``_types`` mapping so the frontend can render each metric
    with the appropriate chart type without hardcoding metric names.
    """
    agg: dict[str, Any] = {}
    for name, samples in parsed.metrics.items():
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        total = sum(s["value"] for s in samples)
        agg[name] = round(total, 6)

    # Attach type annotations for frontend rendering
    agg["_types"] = parsed.types
    return agg


async def _scrape_once(colony: ColonyConnection) -> dict[str, Any] | None:
    """Scrape /metrics and return an aggregated snapshot."""
    text = await colony.fetch_prometheus_metrics()
    if not text:
        return None
    parsed = _parse_prometheus_text(text)
    if not parsed.metrics:
        return None
    snapshot = _aggregate_snapshot(parsed)
    snapshot["_timestamp"] = time.time()
    return snapshot


async def _metrics_event_generator(
    colony: ColonyConnection,
    interval: float,
):
    """Scrape /metrics periodically and yield SSE events."""
    while True:
        try:
            snapshot = await _scrape_once(colony)
            if snapshot:
                _history.append(snapshot)
                yield f"data: {json.dumps(snapshot)}\n\n"
            else:
                yield f"data: {json.dumps({'_timestamp': time.time(), '_error': 'no metrics'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'_timestamp': time.time(), '_error': str(e)})}\n\n"

        await asyncio.sleep(interval)


@router.get("/stream/metrics")
async def stream_metrics(
    interval: float = Query(5.0, ge=1.0, le=60.0),
    colony: ColonyConnection = Depends(get_colony),
) -> StreamingResponse:
    """SSE stream of scraped Prometheus metrics snapshots."""
    return StreamingResponse(
        _metrics_event_generator(colony, interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/metrics/scraped")
async def get_scraped_metrics(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Single scrape: return current metric values."""
    snapshot = await _scrape_once(colony)
    return snapshot or {"_error": "no metrics available"}


@router.get("/metrics/history")
async def get_metrics_history(
    last: int = Query(60, ge=1, le=720, description="Number of snapshots to return"),
) -> list[dict[str, Any]]:
    """Return buffered metric history (from SSE scraper)."""
    return list(_history)[-last:]
