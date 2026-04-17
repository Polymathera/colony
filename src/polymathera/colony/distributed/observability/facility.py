"""TracingFacility — hook-based instrumentation for agent observability.

Registers AROUND hooks on all hookable agent methods to capture execution
spans without any changes to agent code. Buffers spans in-memory and
flushes to Kafka in a background task.

Follows the MemoryCapability pattern:
- AgentCapability subclass
- Registers hooks in initialize()
- Background asyncio.Task for flushing
- Cleanup in shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from overrides import override
from collections import deque
from typing import Any, Callable, TYPE_CHECKING

from ..hooks import HookContext, HookType, ErrorMode, Pointcut, HookRegistry
from .config import TracingConfig
from .context import get_current_span, span_context, set_current_trace_id, get_current_trace_id
from .models import Span, SpanKind, SpanStatus, generate_span_id
from .producer import SpanProducer


logger = logging.getLogger(__name__)

# Map join point patterns to SpanKind
_KIND_MAP: dict[str, SpanKind] = {
    "run_step": SpanKind.AGENT_STEP,
    "execute_iteration": SpanKind.PLAN,
    "plan_step": SpanKind.PLAN,
    "dispatch": SpanKind.ACTION,
    "infer": SpanKind.INFER,
    "request_page": SpanKind.PAGE_REQUEST,
    "get_next_event": SpanKind.EVENT_PROCESS,
}


class TracingFacility(ABC):
    """Hook-based tracing facility.

    Captures execution spans by registering AROUND hooks on all
    hookable agent methods. Spans are buffered and flushed to Kafka
    in a background task.

    The trace boundary is the session (trace_id = session_id).
    Each AgentRun becomes a RUN span within the session trace.
    """

    def __init__(self, config: TracingConfig):
        self._config = config
        self._trace_id: str | None = None
        self._buffer: deque[Span] = deque()
        self._producer: SpanProducer | None = None
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._hook_ids: list[str] = []

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._config.enabled

    async def initialize(self) -> None:
        """Initialize tracing: set trace_id, start producer, register hooks."""
        if not self.is_enabled():
            return

        # Start Kafka producer
        kafka_bootstrap = self._config.kafka_bootstrap or os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
        self._producer = SpanProducer(
            kafka_bootstrap=kafka_bootstrap,
            topic=self._config.kafka_topic,
        )
        try:
            await self._producer.start()
        except Exception:
            logger.warning("Failed to start SpanProducer — tracing disabled", exc_info=True)
            self._config.enabled = False
            return

        # Register AROUND hooks
        self._register_hooks()

        # Start background flush task
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    def append_span(self, span: Span) -> None:
        self._buffer.append(span)

    def add_span(
        self,
        name: str,
        kind: SpanKind,
        span_id: str,
        parent_span_id: str | None,
        input_summary: dict[str, Any] = {},
        status: SpanStatus = SpanStatus.RUNNING,
        start_time: float = None,
        start_wall: float = None,
        end_time: float = None,
        finish: bool = False,
    ) -> Span:
        now_mono = time.monotonic()
        now_wall = time.time()
        span = Span(
            span_id=span_id,
            trace_id=self.resolve_trace_id(),
            parent_span_id=parent_span_id,
            run_id=self.get_run_id(),
            agent_id=self.get_agent_id(),
            name=name,
            kind=kind,
            start_time=start_time or now_mono,
            start_wall=start_wall or now_wall,
            end_time=end_time,
            status=status,
            input_summary=input_summary,
        )
        if finish:
            span.finish()
        self.append_span(span)
        return span

    @abstractmethod
    def get_hook_registry(self) -> HookRegistry:
        """Get the HookRegistry to register hooks with."""
        return None

    @abstractmethod
    def get_pointcuts(self) -> list[tuple[str, SpanKind]]:
        """Get the pointcuts to register hooks for."""
        return []

    @abstractmethod
    def get_trace_id(self) -> str:
        """Get the trace_id for this session."""
        return "unknown"

    def _register_hooks(self) -> None:
        pointcuts = self.get_pointcuts()
        hook_registry = self.get_hook_registry()

        for pattern, kind in pointcuts:
            handler = self._make_around_handler(kind)
            hook_id = hook_registry.register(
                pointcut=Pointcut.pattern(pattern),
                handler=handler,
                hook_type=HookType.AROUND,
                priority=-100,  # Outermost wrapper
                on_error=ErrorMode.SUPPRESS,
                owner=self,
            )
            self._hook_ids.append(hook_id)

    def resolve_trace_id(self) -> str:
        """Resolve trace_id from the agent's metadata.session_id.

        AgentMetadata.session_id is set at blueprint creation time (before the
        agent starts), so it's always available when hooks fire.  Falls back to
        agent_id only if metadata.session_id is the default placeholder.
        """
        if self._trace_id is None:
            self._trace_id = self.get_trace_id()
            set_current_trace_id(self._trace_id)
        return self._trace_id

    def get_agent_id(self) -> str:
        return ""

    def get_run_id(self) -> str:
        return ""

    @abstractmethod
    def get_root_span_id(self) -> str:
        return None

    def _make_around_handler(self, kind: SpanKind) -> Callable:
        """Create an AROUND hook handler for a specific SpanKind."""

        async def handler(ctx: HookContext, proceed: Callable) -> Any:
            trace_id = self.resolve_trace_id()
            parent = get_current_span()
            # If no parent from contextvars, use the AGENT span as parent
            parent_id = parent.span_id if parent else self.get_root_span_id()
            span = Span(
                span_id=generate_span_id(),
                trace_id=trace_id,
                parent_span_id=parent_id,
                run_id=self.get_run_id(),
                agent_id=self.get_agent_id(),
                name=ctx.join_point,
                kind=kind,
                start_time=time.monotonic(),
                start_wall=time.time(),
                status=SpanStatus.RUNNING,
                input_summary=self.summarize_input(ctx, kind),
            )
            # Buffer the RUNNING span immediately so child spans can
            # find their parent in the DB.  The consumer upserts, so
            # the completion update below overwrites this entry.
            self.append_span(span)
            with span_context(span):
                try:
                    result = await proceed()
                    span.status = SpanStatus.OK
                    span.output_summary = self.summarize_output(ctx.join_point, kind, result)
                    self.enrich_span(span, kind, ctx, result)
                    return result
                except Exception as e:
                    span.status = SpanStatus.ERROR
                    span.error = f"{type(e).__name__}: {str(e)[:200]}"
                    raise
                finally:
                    span.end_time = time.monotonic()
                    self.append_span(span)

        return handler

    def _get_str_field(self, value: Any, max_chars: int) -> str:
        """Convert a value to string and truncate if it exceeds max_chars."""
        s = str(value)
        if len(s) > max_chars:
            return s[:max_chars] + "..."
        return s

    @abstractmethod
    def summarize_input(self, ctx: HookContext, kind: SpanKind) -> dict[str, Any]:
        """Extract a truncated input summary from hook context."""
        pass

    @abstractmethod
    def summarize_output(self, join_point: str, kind: SpanKind, result: Any) -> dict[str, Any]:
        """Extract a truncated output summary from the result."""
        pass

    @abstractmethod
    def enrich_span(self, span: Span, kind: SpanKind, ctx: HookContext, result: Any) -> None:
        """Enrich span with kind-specific data (tokens, page IDs, etc.)."""
        pass

    async def _flush_loop(self) -> None:
        """Background task: periodically flush buffered spans to Kafka."""
        while self._running:
            try:
                await asyncio.sleep(self._config.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Flush loop error", exc_info=True)
        # Final flush on exit
        await self._flush()

    async def _flush(self) -> None:
        """Flush buffered spans to Kafka."""
        if not self._buffer or not self._producer:
            return

        batch: list[Span] = []
        while self._buffer and len(batch) < self._config.flush_batch_size:
            batch.append(self._buffer.popleft())

        if batch:
            await self._producer.send_spans(batch)

    async def shutdown(self) -> None:
        """Stop the flush task and producer.

        Finalizes the AGENT and RUN spans with the agent's actual state
        (not hardcoded OK) so the dashboard shows the correct outcome.
        """
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self._flush()

        if self._producer:
            await self._producer.stop()
            self._producer = None

        # Unregister hooks
        hook_registry = self.get_hook_registry()
        if self._hook_ids:
            for hook_id in self._hook_ids:
                hook_registry.unregister(hook_id)
            self._hook_ids.clear()


