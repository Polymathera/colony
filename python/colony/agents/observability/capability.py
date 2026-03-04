"""TracingCapability — hook-based instrumentation for agent observability.

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
from collections import deque
from typing import Any, Callable, TYPE_CHECKING

from ..base import AgentCapability
from ..patterns.hooks.types import HookContext, HookType, ErrorMode
from ..patterns.hooks.pointcuts import Pointcut
from .config import TracingConfig
from .context import get_current_span, span_context, set_current_trace_id, get_current_trace_id
from .models import Span, SpanKind, SpanStatus, generate_span_id
from .producer import SpanProducer

if TYPE_CHECKING:
    from ..base import Agent

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


class TracingCapability(AgentCapability):
    """Hook-based tracing capability.

    Captures execution spans by registering AROUND hooks on all
    hookable agent methods. Spans are buffered and flushed to Kafka
    in a background task.

    The trace boundary is the session (trace_id = session_id).
    Each AgentRun becomes a RUN span within the session trace.
    """

    def __init__(
        self,
        agent: Agent,
        config: TracingConfig,
        *,
        scope_id: str | None = None,
    ):
        super().__init__(agent=agent, scope_id=scope_id)
        self._config = config
        self._trace_id: str | None = None
        self._current_run_id: str | None = None
        self._buffer: deque[Span] = deque()
        self._producer: SpanProducer | None = None
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._hook_ids: list[str] = []

    async def initialize(self) -> None:
        """Initialize tracing: set trace_id, start producer, register hooks."""
        if not self._config.enabled:
            return

        # trace_id will be resolved lazily from session context in _resolve_trace_id()
        # At initialize() time, session_id_context may not be set yet

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

        logger.info(
            "TracingCapability initialized (trace_id=%s, agent=%s)",
            self._trace_id, self.agent.agent_id,
        )

    def _register_hooks(self) -> None:
        """Register AROUND hooks for all traced join points."""
        hook_registry = self.agent.hooks

        pointcuts = [
            ("*.run_step", SpanKind.AGENT_STEP),
            ("*.execute_iteration", SpanKind.PLAN),
            ("*.plan_step", SpanKind.PLAN),
            ("*.dispatch", SpanKind.ACTION),
            ("*.infer", SpanKind.INFER),
            ("*.request_page", SpanKind.PAGE_REQUEST),
            ("*.get_next_event", SpanKind.EVENT_PROCESS),
        ]

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

    def _resolve_trace_id(self) -> str:
        """Resolve trace_id from session context (lazy, called inside hooks).

        At hook execution time, session_id_context IS set (run_step wraps
        everything in session_id_context). So we can reliably get the session_id
        here and use it as trace_id, ensuring one trace per session.
        """
        if self._trace_id is None:
            from ..sessions.context import get_current_session_id
            session_id = get_current_session_id()
            if session_id:
                self._trace_id = session_id
            else:
                self._trace_id = getattr(self.agent, "agent_id", "unknown")
            set_current_trace_id(self._trace_id)
            logger.info("TracingCapability resolved trace_id=%s (agent=%s)", self._trace_id, self.agent.agent_id)
        return self._trace_id

    def _make_around_handler(self, kind: SpanKind) -> Callable:
        """Create an AROUND hook handler for a specific SpanKind."""

        async def handler(ctx: HookContext, proceed: Callable) -> Any:
            trace_id = self._resolve_trace_id()
            parent = get_current_span()
            span = Span(
                span_id=generate_span_id(),
                trace_id=trace_id,
                parent_span_id=parent.span_id if parent else None,
                run_id=self._current_run_id,
                agent_id=self.agent.agent_id,
                name=ctx.join_point,
                kind=kind,
                start_time=time.monotonic(),
                start_wall=time.time(),
                status=SpanStatus.RUNNING,
                input_summary=self._summarize_input(ctx, kind),
            )
            with span_context(span):
                try:
                    result = await proceed()
                    span.status = SpanStatus.OK
                    span.output_summary = self._summarize_output(ctx.join_point, kind, result)
                    self._enrich_span(span, kind, ctx, result)
                    return result
                except Exception as e:
                    span.status = SpanStatus.ERROR
                    span.error = f"{type(e).__name__}: {str(e)[:200]}"
                    raise
                finally:
                    span.end_time = time.monotonic()
                    self._buffer.append(span)

        return handler

    def _summarize_input(self, ctx: HookContext, kind: SpanKind) -> dict[str, Any]:
        """Extract a truncated input summary from hook context."""
        max_chars = self._config.max_input_chars
        summary: dict[str, Any] = {}

        if kind == SpanKind.ACTION and ctx.args:
            # dispatch(action_key, ...) — capture the action key
            summary["action"] = str(ctx.args[0])[:max_chars] if ctx.args else None
        elif kind == SpanKind.INFER and self._config.capture_infer_inputs:
            # Include prompt text (expensive, opt-in)
            if ctx.kwargs.get("messages"):
                msgs = ctx.kwargs["messages"]
                summary["message_count"] = len(msgs) if isinstance(msgs, list) else 1

        return summary

    def _summarize_output(self, join_point: str, kind: SpanKind, result: Any) -> dict[str, Any]:
        """Extract a truncated output summary from the result."""
        max_chars = self._config.max_output_chars
        summary: dict[str, Any] = {}

        if kind == SpanKind.ACTION and self._config.capture_action_results:
            summary["result_type"] = type(result).__name__
            result_str = str(result)
            if len(result_str) > max_chars:
                result_str = result_str[:max_chars] + "..."
            summary["result"] = result_str

        return summary

    def _enrich_span(self, span: Span, kind: SpanKind, ctx: HookContext, result: Any) -> None:
        """Enrich span with kind-specific data (tokens, page IDs, etc.)."""
        if kind == SpanKind.INFER:
            # Extract token usage if available
            usage = getattr(result, "usage", None)
            if usage:
                span.input_tokens = getattr(usage, "input_tokens", None)
                span.output_tokens = getattr(usage, "output_tokens", None)
                span.cache_read_tokens = getattr(usage, "cache_read_input_tokens", None)
            span.model_name = getattr(result, "model", None)

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
        """Stop the flush task and producer."""
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
        if self._hook_ids and hasattr(self.agent, "hooks"):
            for hook_id in self._hook_ids:
                self.agent.hooks.unregister(hook_id)
            self._hook_ids.clear()

        logger.info("TracingCapability shut down (trace_id=%s)", self._trace_id)

    async def serialize_suspension_state(self, state: Any) -> Any:
        """No suspension state for tracing."""
        return state

    async def deserialize_suspension_state(self, state: Any) -> None:
        """No suspension state for tracing."""
        pass

    def set_run_id(self, run_id: str) -> None:
        """Update the current run_id. Called when a new run starts."""
        self._current_run_id = run_id
