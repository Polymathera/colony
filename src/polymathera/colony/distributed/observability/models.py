"""Data models for the Colony tracing system."""

from __future__ import annotations

import uuid
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    """Colony-specific span types."""

    # Agent-layer
    RUN = "run"
    AGENT = "agent"
    AGENT_STEP = "agent_step"
    PLAN = "plan"
    ACTION = "action"
    INFER = "infer"
    PAGE_REQUEST = "page_request"
    BLACKBOARD_OP = "blackboard_op"
    CHILD_SPAWN = "child_spawn"
    EVENT_PROCESS = "event_process"
    CAPABILITY = "capability"
    LIFECYCLE = "lifecycle"
    CUSTOM = "custom"

    # Cluster-layer
    API_CALL = "api_call"      # external HTTP call (Anthropic, OpenRouter)
    DEPLOYMENT = "deployment"  # deployment replica lifecycle


class SpanStatus(str, Enum):
    """Span execution status."""

    RUNNING = "running"
    OK = "ok"
    ERROR = "error"


def generate_span_id() -> str:
    """Generate a unique span ID."""
    return f"span_{uuid.uuid4().hex[:12]}"


class Span(BaseModel):
    """A single unit of work in a trace.

    Spans form a tree via parent_span_id. The root of the tree
    is the session-level trace (trace_id = session_id).
    """

    span_id: str = Field(default_factory=generate_span_id)
    trace_id: str  # = session_id
    parent_span_id: str | None = None
    run_id: str | None = None

    # Identity
    agent_id: str
    name: str
    kind: SpanKind

    # Timing
    start_time: float = Field(default_factory=time.monotonic)
    end_time: float | None = None
    start_wall: float = Field(default_factory=time.time)

    # Status
    status: SpanStatus = SpanStatus.RUNNING
    error: str | None = None

    # Data
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)

    # LLM-specific (kind=INFER)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    model_name: str | None = None
    context_page_ids: list[str] | None = None

    # Distributed context
    ring: str | None = None          # "KERNEL" or "USER" (from ExecutionContext.ring)
    service_name: str | None = None  # deployment name, "agent", "vcm", etc.

    # Metadata
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds, or None if still running."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def finish(self, status: SpanStatus = SpanStatus.OK, error: str | None = None) -> None:
        """Mark span as finished."""
        self.end_time = time.monotonic()
        self.status = status
        if error:
            self.error = error
