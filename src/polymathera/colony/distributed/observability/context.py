"""Context propagation for tracing spans.

Follows the same pattern as colony.agents.sessions.context:
ContextVars for current span and trace_id, with context managers
for scoping span lifetime.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .models import Span


# Current span in the execution stack
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)

# Current trace_id (= session_id) — set once per agent lifetime
_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def get_current_span() -> Span | None:
    """Get the current span from context."""
    return _current_span.get()


def set_current_span(span: Span) -> Token[Span | None]:
    """Set the current span. Returns token for resetting."""
    return _current_span.set(span)


def get_current_trace_id() -> str | None:
    """Get the current trace_id from context."""
    return _current_trace_id.get()


def set_current_trace_id(trace_id: str | None) -> Token[str | None]:
    """Set the current trace_id. Returns token for resetting."""
    return _current_trace_id.set(trace_id)


@contextmanager
def span_context(span: Span) -> Iterator[Span]:
    """Context manager that sets span as current, restores previous on exit.

    Usage:
        with span_context(new_span):
            # new_span is now the current span
            result = await proceed()
        # previous span restored
    """
    token = _current_span.set(span)
    try:
        yield span
    finally:
        _current_span.reset(token)


@contextmanager
def trace_context(trace_id: str) -> Iterator[str]:
    """Context manager for setting trace_id scope."""
    token = _current_trace_id.set(trace_id)
    try:
        yield trace_id
    finally:
        _current_trace_id.reset(token)
