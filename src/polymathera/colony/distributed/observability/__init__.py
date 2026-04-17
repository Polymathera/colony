"""Colony Observability — LangSmith-inspired tracing for multi-agent systems.

Captures fine-grained execution traces (spans) from agents via the hook system,
streams them through Kafka, and stores them durably in PostgreSQL.

Public API:
    - Span, SpanKind, SpanStatus: Data model
    - TracingConfig: Configuration
    - TracingFacility: Hook-based instrumentation
    - get_current_span, span_context: Context propagation
    - SpanProducer: Agent → Kafka
    - SpanConsumer: Kafka → PostgreSQL
    - SpanQueryStore: PostgreSQL → Dashboard queries
"""

from .config import TracingConfig
from .context import (
    span_context,
    get_current_span,
    set_current_span,
    reset_current_span,
    get_current_trace_id,
    set_current_trace_id,
)
from .models import Span, SpanKind, SpanStatus, generate_span_id
from .producer import SpanProducer
from .facility import TracingFacility

__all__ = [
    "Span",
    "SpanKind",
    "SpanStatus",
    "generate_span_id",
    "TracingConfig",
    "span_context",
    "get_current_span",
    "set_current_span",
    "reset_current_span",
    "get_current_trace_id",
    "set_current_trace_id",
    "SpanProducer",
    "TracingFacility",
]
