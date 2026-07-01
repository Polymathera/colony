"""Behavioral tests for the chat-message feedback route handler.

The thumb resolves the producing INFER span and records to the same
``span_feedback`` store the Traces tab uses.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from polymathera.colony.web_ui.backend.routers.chat import record_message_feedback
from polymathera.colony.web_ui.backend.routers.traces import SpanFeedbackRequest

pytestmark = pytest.mark.asyncio


class _ChatStore:
    def __init__(self, message):
        self._message = message

    async def get_message(self, message_id):
        return self._message


class _SpanStore:
    def __init__(self, span_id):
        self._span_id = span_id
        self.calls: list[dict[str, Any]] = []

    async def get_latest_infer_span_id(self, trace_id, agent_id, before_wall):
        self.calls.append(
            {"trace_id": trace_id, "agent_id": agent_id, "before_wall": before_wall},
        )
        return self._span_id


class _FeedbackStore:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    async def record(self, **kwargs):
        self.records.append(kwargs)


class _Colony:
    def __init__(self, *, span_store=None, feedback_store=None):
        self._span_store = span_store
        self._feedback_store = feedback_store

    def get_span_query_store(self):
        return self._span_store

    def get_span_feedback_store(self):
        return self._feedback_store


def _agent_message(**over):
    msg = {"id": "m1", "session_id": "s1", "role": "agent", "agent_id": "a1", "timestamp": 100.0}
    msg.update(over)
    return msg


async def _call(*, chat_store, colony, session="s1", message="m1"):
    return await record_message_feedback(
        session, message, SpanFeedbackRequest(rating="up", note="good"),
        _user={"sub": "u1"}, colony=colony, chat_store=chat_store,
    )


async def test_happy_path_records_against_resolved_span() -> None:
    feedback = _FeedbackStore()
    span_store = _SpanStore("span-7")
    out = await _call(
        chat_store=_ChatStore(_agent_message()),
        colony=_Colony(span_store=span_store, feedback_store=feedback),
    )
    assert out == {"ok": True, "span_id": "span-7"}
    assert span_store.calls == [{"trace_id": "s1", "agent_id": "a1", "before_wall": 100.0}]
    assert feedback.records == [
        {"trace_id": "s1", "span_id": "span-7", "author": "u1", "rating": "up", "note": "good"},
    ]


async def test_no_chat_store_is_503() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(chat_store=None, colony=_Colony())
    assert exc.value.status_code == 503


async def test_unknown_message_is_404() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(chat_store=_ChatStore(None), colony=_Colony())
    assert exc.value.status_code == 404


async def test_session_mismatch_is_404() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(chat_store=_ChatStore(_agent_message(session_id="other")), colony=_Colony())
    assert exc.value.status_code == 404


async def test_non_agent_message_is_422() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(chat_store=_ChatStore(_agent_message(role="user")), colony=_Colony())
    assert exc.value.status_code == 422


async def test_no_traced_inference_is_422() -> None:
    with pytest.raises(HTTPException) as exc:
        await _call(
            chat_store=_ChatStore(_agent_message()),
            colony=_Colony(span_store=_SpanStore(None), feedback_store=_FeedbackStore()),
        )
    assert exc.value.status_code == 422
