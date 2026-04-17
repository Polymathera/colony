"""Cluster-layer tracing facility for LLM deployments.

Provides ``ClusterTracingFacility``, a concrete ``TracingFacility`` for
cluster services (``LLMCluster``, ``RemoteLLMDeployment``, ``VLLMDeployment``).
Captures LLM-specific span data: model name, tokens, cost, cache stats.

Usage::

    # In a deployment's initialize():
    self._tracing = ClusterTracingFacility(
        config=TracingConfig(enabled=True),
        owner=self,
        service_name="RemoteLLMDeployment",
        deployment_name=self.config.get_deployment_name(),
    )
    await self._tracing.initialize()
"""

from __future__ import annotations

import logging
from overrides import override
from typing import Any

from ..distributed.hooks import HookContext, HookRegistry, get_hook_registry
from ..distributed.observability import (
    TracingConfig,
    TracingFacility,
    Span,
    SpanKind,
    SpanStatus,
    generate_span_id,
    get_current_trace_id,
)

logger = logging.getLogger(__name__)


class ClusterTracingFacility(TracingFacility):
    """Tracing facility for cluster deployment services.

    Enriches spans with LLM-specific data: model name, tokens,
    cost, cache hit ratios, page IDs.
    """

    def __init__(
        self,
        config: TracingConfig,
        owner: Any,
        service_name: str,
        deployment_name: str,
        pointcuts: list[tuple[str, SpanKind]] | None = None,
    ):
        super().__init__(config)
        self._owner = owner
        self._service_name = service_name
        self._deployment_name = deployment_name
        self._pointcuts = pointcuts or [
            ("*.infer", SpanKind.INFER),
            ("*.infer_with_suffix", SpanKind.INFER),
            ("*._call_api", SpanKind.API_CALL),
            ("*.load_page", SpanKind.PAGE_REQUEST),
        ]
        self._root_span: Span | None = None

    @override
    def get_hook_registry(self) -> HookRegistry:
        domain_key = self._owner.get_hookable_publication_domain_key()
        return get_hook_registry(domain_key)

    @override
    def get_pointcuts(self) -> list[tuple[str, SpanKind]]:
        return self._pointcuts

    @override
    def get_trace_id(self) -> str:
        # Deployment trace_id comes from the execution context of the
        # current request, NOT cached — different requests may belong
        # to different sessions/traces.
        return get_current_trace_id() or "deployment"

    def resolve_trace_id(self) -> str | None:
        """Override to NOT cache — each request may have a different trace_id.

        Returns None for KERNEL/untraced requests so that the around handler
        skips span creation entirely (no orphaned "deployment" trace).
        """
        from ..distributed.ray_utils.serving.context import get_execution_context
        ctx = get_execution_context()
        if ctx and ctx.trace_id:
            return ctx.trace_id
        return get_current_trace_id() or None

    @override
    def get_agent_id(self) -> str:
        return self._deployment_name

    @override
    def get_run_id(self) -> str:
        return ""

    @override
    def get_root_span_id(self) -> str | None:
        # Parent deployment hooks under the caller's span.
        # DeploymentHandle.call_method() injects get_current_span().span_id
        # as parent_span_id into the ExecutionContext on the request.
        # This is per-request (restored by __handle_request__), not a
        # shared contextvar.
        from ..distributed.ray_utils.serving.context import get_execution_context
        ctx = get_execution_context()
        return ctx.parent_span_id if ctx else None

    async def initialize(self) -> None:
        """Initialize tracing: start producer, register hooks."""
        await super().initialize()
        if not self.is_enabled():
            return

        # No root span for deployments — their spans parent under the
        # CLIENT span from the calling agent's trace.  A deployment root
        # span would create a separate trace entry.

        logger.info(
            "ClusterTracingFacility initialized (deployment=%s, service=%s)",
            self._deployment_name, self._service_name,
        )

    @override
    def summarize_input(self, ctx: HookContext, kind: SpanKind) -> dict[str, Any]:
        max_chars = self._config.max_input_chars
        summary: dict[str, Any] = {"service": self._service_name}

        if kind == SpanKind.INFER:
            # InferenceRequest is typically the first positional arg
            request = ctx.args[0] if ctx.args else None
            if request is not None:
                summary["max_tokens"] = getattr(request, "max_tokens", None)
                summary["temperature"] = getattr(request, "temperature", None)
                page_ids = getattr(request, "context_page_ids", None)
                if page_ids:
                    summary["page_count"] = len(page_ids)
                prompt = getattr(request, "prompt", None)
                if prompt and isinstance(prompt, str):
                    summary["prompt_preview"] = prompt[:min(200, max_chars)]
        elif kind == SpanKind.API_CALL:
            # _call_api(messages, max_tokens, temperature, ...)
            messages = ctx.args[0] if ctx.args else ctx.kwargs.get("messages")
            if messages and isinstance(messages, dict):
                msg_list = messages.get("messages", [])
                summary["message_count"] = len(msg_list)
            summary["max_tokens"] = ctx.kwargs.get("max_tokens")
            summary["temperature"] = ctx.kwargs.get("temperature")
        elif kind == SpanKind.PAGE_REQUEST:
            page = ctx.args[0] if ctx.args else None
            if page is not None:
                summary["page_id"] = self._get_str_field(
                    getattr(page, "page_id", None), max_chars
                )
                summary["page_size"] = getattr(page, "size", None)

        return summary

    @override
    def summarize_output(self, join_point: str, kind: SpanKind, result: Any) -> dict[str, Any]:
        max_chars = self._config.max_output_chars
        summary: dict[str, Any] = {}

        if kind == SpanKind.INFER:
            # InferenceResponse
            generated = getattr(result, "generated_text", None)
            if generated and isinstance(generated, str):
                summary["response_preview"] = generated[:min(200, max_chars)]
                summary["response_len"] = len(generated)
        elif kind == SpanKind.API_CALL:
            # APIResponse from _call_api
            content = getattr(result, "content", None)
            if content and isinstance(content, str):
                summary["response_len"] = len(content)
            summary["cost_usd"] = getattr(result, "cost_usd", None)

        return summary

    @override
    def enrich_span(self, span: Span, kind: SpanKind, ctx: HookContext, result: Any) -> None:
        span.service_name = self._service_name

        if kind in (SpanKind.INFER, SpanKind.API_CALL):
            # Extract token usage
            input_tokens = getattr(result, "input_tokens", None)
            output_tokens = getattr(result, "output_tokens", None)
            cache_read = getattr(result, "cache_read_input_tokens", None)
            model_name = getattr(result, "model_name", None) or getattr(
                getattr(self._owner, "config", None), "model_name", None
            )

            if input_tokens is not None:
                span.input_tokens = input_tokens
            if output_tokens is not None:
                span.output_tokens = output_tokens
            if cache_read is not None:
                span.cache_read_tokens = cache_read
            if model_name:
                span.model_name = model_name

    async def shutdown(self) -> None:
        """Finalize deployment root span and flush."""
        if self._root_span and self._root_span.end_time is None:
            self._root_span.finish(SpanStatus.OK)
            self.append_span(self._root_span)

        await super().shutdown()
        logger.info("ClusterTracingFacility shut down (deployment=%s)", self._deployment_name)
