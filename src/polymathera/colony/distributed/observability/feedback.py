"""Span feedback store — human ratings on individual trace spans.

A reviewer rates a span up or down (with an optional note) from the
Traces UI. One rating per author per span (re-rating updates in place).
The ratings are a human signal over individual agent steps, used both
for live inspection and as downstream training data.
"""

from __future__ import annotations

from typing import Any

RATINGS = ("up", "down")


class SpanFeedbackRatingError(ValueError):
    """Raised when a rating is not one of :data:`RATINGS`."""


class SpanFeedbackStore:
    """Read/write store for per-span human feedback."""

    def __init__(self, db_pool: Any):
        self._db_pool = db_pool

    async def record(
        self,
        *,
        trace_id: str,
        span_id: str,
        author: str,
        rating: str,
        note: str | None = None,
    ) -> None:
        """Upsert one author's rating of one span."""

        if rating not in RATINGS:
            raise SpanFeedbackRatingError(
                f"rating must be one of {RATINGS}, got {rating!r}"
            )
        query = """
            INSERT INTO span_feedback (span_id, trace_id, author, rating, note, updated_wall)
            VALUES ($1, $2, $3, $4, $5, now())
            ON CONFLICT (span_id, author) DO UPDATE
                SET rating = EXCLUDED.rating,
                    note = EXCLUDED.note,
                    trace_id = EXCLUDED.trace_id,
                    updated_wall = now()
        """
        async with self._db_pool.acquire() as conn:
            await conn.execute(query, span_id, trace_id, author, rating, note)

    async def get_for_trace(self, trace_id: str) -> dict[str, list[dict[str, Any]]]:
        """All feedback for a trace, grouped by ``span_id``."""

        query = """
            SELECT span_id, author, rating, note, updated_wall
            FROM span_feedback
            WHERE trace_id = $1
            ORDER BY updated_wall ASC
        """
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query, trace_id)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            d = dict(row)
            updated = d.get("updated_wall")
            grouped.setdefault(d["span_id"], []).append(
                {
                    "author": d["author"],
                    "rating": d["rating"],
                    "note": d["note"],
                    "updated_wall": updated.timestamp() if updated is not None else None,
                }
            )
        return grouped
