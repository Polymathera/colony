"""Behavioral tests for the span-feedback route handlers.

FastAPI's route decorators return the wrapped function unchanged, so
the handlers are called directly with duck-typed fakes.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from polymathera.colony.distributed.observability.feedback import (
    SpanFeedbackRatingError,
)
from polymathera.colony.web_ui.backend.routers.traces import (
    SpanFeedbackRequest,
    get_trace_spans,
    record_span_feedback,
)

pytestmark = pytest.mark.asyncio


class _FeedbackStore:
    def __init__(self, *, for_trace=None, raises: Exception | None = None):
        self._for_trace = for_trace or {}
        self._raises = raises
        self.record_calls: list[dict[str, Any]] = []

    async def record(self, **kwargs):
        if self._raises:
            raise self._raises
        self.record_calls.append(kwargs)

    async def get_for_trace(self, trace_id: str):
        return self._for_trace


class _SpanStore:
    def __init__(self, spans):
        self._spans = spans

    async def get_spans(self, trace_id, **kwargs):
        return self._spans


class _Colony:
    def __init__(self, *, spans=None, feedback_store=None):
        self._span_store = _SpanStore(spans) if spans is not None else None
        self._feedback_store = feedback_store

    def get_span_query_store(self):
        return self._span_store

    def get_span_feedback_store(self):
        return self._feedback_store


async def test_record_feedback_passes_author_and_payload() -> None:
    store = _FeedbackStore()
    out = await record_span_feedback(
        "t1", "s1", SpanFeedbackRequest(rating="up", note="nice"),
        _user={"sub": "u1"}, colony=_Colony(feedback_store=store),
    )
    assert out == {"ok": True}
    assert store.record_calls == [
        {"trace_id": "t1", "span_id": "s1", "author": "u1", "rating": "up", "note": "nice"},
    ]


async def test_record_feedback_bad_rating_is_422() -> None:
    store = _FeedbackStore(raises=SpanFeedbackRatingError("bad"))
    with pytest.raises(HTTPException) as exc:
        await record_span_feedback(
            "t1", "s1", SpanFeedbackRequest(rating="meh"),
            _user={"sub": "u1"}, colony=_Colony(feedback_store=store),
        )
    assert exc.value.status_code == 422


async def test_record_feedback_no_store_is_503() -> None:
    with pytest.raises(HTTPException) as exc:
        await record_span_feedback(
            "t1", "s1", SpanFeedbackRequest(rating="up"),
            _user={"sub": "u1"}, colony=_Colony(feedback_store=None),
        )
    assert exc.value.status_code == 503


async def test_get_spans_attaches_feedback_per_span() -> None:
    spans = [{"span_id": "s1"}, {"span_id": "s2"}]
    fb = {"s1": [{"author": "u1", "rating": "up", "note": None, "updated_wall": 1.0}]}
    colony = _Colony(spans=spans, feedback_store=_FeedbackStore(for_trace=fb))
    out = await get_trace_spans("t1", _user={"sub": "u1"}, colony=colony)
    by_id = {s["span_id"]: s for s in out}
    assert by_id["s1"]["feedback"] == fb["s1"]
    assert by_id["s2"]["feedback"] == []


async def test_get_spans_tolerates_missing_feedback_store() -> None:
    colony = _Colony(spans=[{"span_id": "s1"}], feedback_store=None)
    out = await get_trace_spans("t1", _user={"sub": "u1"}, colony=colony)
    assert out[0]["feedback"] == []
