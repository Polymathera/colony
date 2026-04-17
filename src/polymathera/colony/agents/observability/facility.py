"""AgentTracingFacility — hook-based instrumentation for agent observability.

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

import logging
import time
from overrides import override
from typing import Any, TYPE_CHECKING

from ...distributed.hooks import HookContext, HookRegistry
from ...distributed.observability import (
    TracingConfig,
    Span,
    SpanKind,
    SpanStatus,
    TracingFacility,
    generate_span_id,
)

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



class AgentTracingFacility(TracingFacility):

    def __init__(self, config: TracingConfig, agent: Agent):
        super().__init__(config)
        self.agent = agent
        self._current_run_id: str | None = None
        self._run_span_id: str | None = None   # Deterministic RUN span (shared across agents)
        self._agent_span: Span | None = None   # Per-agent AGENT span (child of RUN)
        self._start_mono: float = 0.0          # Monotonic clock at init (for duration calc)
        self._start_wall: float = 0.0          # Wall clock at init

    def get_run_id(self) -> str:
        return self._current_run_id

    def set_run_id(self, run_id: str) -> None:
        """Update the current run_id. Called when a new run starts."""
        self._current_run_id = run_id

    async def initialize(self) -> None:
        """Initialize tracing: set trace_id, start producer, register hooks."""
        if not self.is_enabled():
            return

        await super().initialize()  # Start producer and resolve trace_id

        now_mono = time.monotonic()
        now_wall = time.time()
        self._start_mono = now_mono
        self._start_wall = now_wall

        # Extract run_id from metadata
        run_id = getattr(self.agent.metadata, "run_id", None)
        if run_id and run_id != "default":
            self.set_run_id(run_id)

        # Resolve trace_id eagerly now (metadata.session_id is set at blueprint time)
        # Emit a deterministic RUN span (shared across all agents in the same run).
        # Every agent emits it; the consumer deduplicates via ON CONFLICT (span_id).
        if self.get_run_id():
            self._run_span_id = f"span_run_{self.get_run_id()}"
            self.add_span(
                name=f"run:{self.get_run_id()}",
                kind=SpanKind.RUN,
                span_id=self._run_span_id,
                parent_span_id=None
            )

        # Create a per-agent AGENT span (child of the RUN span).
        # All run_step spans become children of this.
        self._agent_span = self.add_span(
            span_id=generate_span_id(),
            parent_span_id=self._run_span_id,
            name=f"agent:{self.get_agent_id()}",
            kind=SpanKind.AGENT,
            input_summary={
                "agent_type": self.agent.agent_type,
                "capability_names": self.agent.get_capability_names(),
                "parent_agent_id": getattr(self.agent.metadata, "parent_agent_id", None),
                "bound_pages": list(self.agent.bound_pages) if hasattr(self.agent, "bound_pages") else [],
            },
        )

        logger.info(
            "AgentTracingFacility initialized (trace_id=%s, agent=%s, agent_span=%s)",
            self._trace_id, self.get_agent_id(), self._agent_span.span_id,
        )

    @override
    def get_hook_registry(self) -> HookRegistry:
        return self.agent.hooks

    @override
    def get_pointcuts(self) -> list[tuple[str, SpanKind]]:
        return [
            ("*.run_step", SpanKind.AGENT_STEP),
            ("*.execute_iteration", SpanKind.PLAN),
            ("*.plan_step", SpanKind.PLAN),
            ("*.dispatch", SpanKind.ACTION),
            ("*.infer", SpanKind.INFER),
            ("*.request_page", SpanKind.PAGE_REQUEST),
            ("*.get_next_event", SpanKind.EVENT_PROCESS),
        ]

    @override
    def get_trace_id(self) -> str:
        """Resolve trace_id from the agent's metadata.session_id.

        AgentMetadata.session_id is set at blueprint creation time (before the
        agent starts), so it's always available when hooks fire.  Falls back to
        agent_id only if metadata.session_id is the default placeholder.
        """
        session_id = None
        metadata = getattr(self.agent, "metadata", None)
        if metadata is not None:
            sid = getattr(metadata, "session_id", None)
            if sid and sid != "default":
                session_id = sid
        if session_id:
            trace_id = session_id
        else:
            trace_id = getattr(self.agent, "agent_id", "unknown")
        logger.info("AgentTracingFacility resolved trace_id=%s (agent=%s)", trace_id, self.agent.agent_id)
        return trace_id

    @override
    def summarize_input(self, ctx: HookContext, kind: SpanKind) -> dict[str, Any]:
        """Extract a truncated input summary from hook context."""
        max_chars = self._config.max_input_chars
        summary: dict[str, Any] = {}

        if kind == SpanKind.ACTION and ctx.args:
            # dispatch(action: Action) — extract structured fields
            action = ctx.args[0]
            if hasattr(action, "action_type"):
                summary["action_type"] = self._get_str_field(action.action_type, max_chars)
            if hasattr(action, "parameters"):
                params = action.parameters
                if isinstance(params, dict):
                    # Truncate large param values
                    summary["parameters"] = {
                        k: self._get_str_field(v, max_chars)
                        for k, v in list(params.items())[:10]
                    }
            if hasattr(action, "reasoning") and action.reasoning:
                summary["reasoning"] = self._get_str_field(action.reasoning, max_chars)
        elif kind == SpanKind.INFER and self._config.capture_infer_inputs:
            # Capture prompt text for UI inspection
            max_infer = self._config.max_infer_chars
            prompt = ctx.kwargs.get("prompt")
            if prompt and isinstance(prompt, str):
                summary["prompt"] = prompt[:max_infer]
            if ctx.kwargs.get("messages"):
                msgs = ctx.kwargs["messages"]
                summary["message_count"] = len(msgs) if isinstance(msgs, list) else 1

        return summary

    @override
    def summarize_output(self, join_point: str, kind: SpanKind, result: Any) -> dict[str, Any]:
        """Extract a truncated output summary from the result."""
        max_chars = self._config.max_output_chars
        summary: dict[str, Any] = {}

        if kind == SpanKind.ACTION and self._config.capture_action_results:
            # ActionResult is a Pydantic model — use model_dump for structured data
            if hasattr(result, "model_dump"):
                dumped = result.model_dump(mode="json")
                summary["success"] = dumped.get("success")
                if dumped.get("error"):
                    summary["error"] = self._get_str_field(dumped["error"], max_chars)
                output = dumped.get("output")
                if output is not None:
                    summary["output"] = self._get_str_field(output, max_chars)
                if dumped.get("metrics"):
                    summary["metrics"] = dumped["metrics"]
                if dumped.get("metadata"):
                    summary["metadata"] = dumped["metadata"]
            else:
                summary["result"] = self._get_str_field(result, max_chars)
        elif kind == SpanKind.INFER and self._config.capture_infer_inputs:
            # Capture LLM response text for UI inspection
            max_infer = self._config.max_infer_chars
            generated = getattr(result, "generated_text", None)
            if generated and isinstance(generated, str):
                summary["response"] = generated[:max_infer]
        elif kind == SpanKind.EVENT_PROCESS:
            # get_next_event returns BlackboardEvent or None
            if result is None:
                summary["event_received"] = False
            else:
                summary["event_received"] = True
                summary["event_key"] = self._get_str_field(
                    getattr(result, "key", None), max_chars
                )
                summary["event_scope"] = self._get_str_field(
                    getattr(result, "scope_id", None), max_chars
                )

        return summary

    @override
    def enrich_span(self, span: Span, kind: SpanKind, ctx: HookContext, result: Any) -> None:
        """Enrich span with kind-specific data (tokens, page IDs, etc.)."""
        if kind == SpanKind.INFER:
            # Extract token usage from InferenceResponse.metadata
            # (where RemoteLLMDeployment / AnthropicLLMDeployment stores it)
            meta = getattr(result, "metadata", None) or {}
            if isinstance(meta, dict):
                span.input_tokens = meta.get("input_tokens")
                span.output_tokens = meta.get("output_tokens")
                span.cache_read_tokens = meta.get("cache_read_tokens")
                span.model_name = meta.get("model")

    @override
    def get_agent_id(self) -> str:
        return self.agent.agent_id

    @override
    def get_root_span_id(self) -> str:
        return self._agent_span.span_id if self._agent_span else None

    def emit_lifecycle_event(
        self,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Emit a lifecycle span (stop, fail, cancel, idle_timeout, etc.).

        Called from ``_run_agent_loop`` in the ``finally`` block so the
        event appears in the trace timeline alongside action/infer spans.
        """
        if not self.is_enabled():
            return

        self.add_span(
            span_id=generate_span_id(),
            parent_span_id=self.get_root_span_id(),
            name=f"lifecycle:{event}",
            kind=SpanKind.LIFECYCLE,
            status=SpanStatus.ERROR if event in ("error", "cancelled") else SpanStatus.OK,
            output_summary=details or {},
            finish=True,
        )

    async def shutdown(self) -> None:
        """Stop the flush task and producer.

        Finalizes the AGENT and RUN spans with the agent's actual state
        (not hardcoded OK) so the dashboard shows the correct outcome.
        """
        now = time.monotonic()

        from ..models import AgentState
        # Determine actual status from agent state
        agent_state = getattr(self.agent, "state", None)
        if agent_state == AgentState.FAILED:
            final_status = SpanStatus.ERROR
        else:
            final_status = SpanStatus.OK
        stop_reason = getattr(self.agent, "_stop_reason", "unknown")

        # Close the AGENT span with actual status
        if self._agent_span and self._agent_span.end_time is None:
            self._agent_span.end_time = now
            self._agent_span.status = final_status
            self._agent_span.output_summary = {"stop_reason": stop_reason}
            if final_status == SpanStatus.ERROR:
                self._agent_span.error = stop_reason
            self.append_span(self._agent_span)

        # Close the RUN span (re-emit with end_time; consumer deduplicates via ON CONFLICT)
        if self._run_span_id and self.resolve_trace_id():
            self.add_span(
                span_id=self._run_span_id,
                parent_span_id=None,
                name=f"run:{self.get_run_id()}",
                kind=SpanKind.RUN,
                start_time=self._start_mono,
                start_wall=self._start_wall,
                end_time=now,
                status=final_status,
            )

        await super().shutdown()  # Flush and stop producer

        logger.info("AgentTracingFacility shut down (trace_id=%s)", self._trace_id)

