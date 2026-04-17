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
from .context import get_current_span, get_current_trace_id, span_context, set_current_trace_id, set_current_span
from .models import Span, SpanKind, SpanStatus, generate_span_id
from .producer import SpanProducer
from .facility import TracingFacility

__all__ = [
    "Span",
    "SpanKind",
    "SpanStatus",
    "generate_span_id",
    "TracingConfig",
    "get_current_span",
    "get_current_trace_id",
    "span_context",
    "set_current_trace_id",
    "set_current_span",
    "SpanProducer",
    "TracingFacility",
]
