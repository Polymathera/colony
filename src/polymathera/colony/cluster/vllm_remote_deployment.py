"""Remote-client for a self-hosted colony serving *fleet*.

The *remote-client* counterpart to the in-cluster :class:`VLLMDeployment`:
a thin forwarder that runs as a Ray actor in the *local* cluster and talks
to a fine-tuned model served on a separate AWS fleet (its own Ray cluster
running :class:`VLLMDeployment`) behind a **partial-OpenAI** HTTP surface.

- Plain ``infer`` uses the fleet's OpenAI ``/v1/chat/completions`` (the
  inherited base path via ``_call_api``); ``cost_usd`` is 0 (self-hosted).
- ``load_page``/``evict_page``/``infer_with_suffix`` are **forwarded** to
  the fleet's colony paging endpoints, so request-to-page affinity uses
  the fleet's real VCM paging + router (Option A: the fleet owns paging +
  routing; this deployment keeps no local page state).

Requires: pip install openai.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..distributed.hooks import hookable
from ..distributed.ray_utils import serving
from ..vcm.models import ContextPageId, VirtualContextPage
from .models import InferenceRequest, InferenceResponse
from .remote_config import RemoteLLMDeploymentConfig
from .remote_deployment import APIResponse, RemoteLLMDeployment
from .remote_registry import register_remote_llm_provider
from .routing import TargetClientRouter

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert engineering design assistant. Respond to the task "
    "precisely and concisely."
)


@register_remote_llm_provider("vllm")
class VllmRemoteDeployment(RemoteLLMDeployment):
    """A model served by a self-hosted, OpenAI-compatible vLLM endpoint."""

    def __init__(self, config: RemoteLLMDeploymentConfig):
        super().__init__(config)
        self._client = None  # openai.AsyncOpenAI (chat completions)
        self._http = None    # httpx.AsyncClient (colony paging endpoints)

    async def _initialize_client(self) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for VllmRemoteDeployment. "
                "Install it with: pip install openai"
            )
        import httpx

        # Self-hosted vLLM accepts any (or no) key; send a placeholder when
        # the configured env var is unset so the SDK is satisfied.
        api_key = os.environ.get(self.config.api_key_env_var) or "EMPTY"
        self._client = openai.AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=api_key,
            max_retries=2,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.config.api_timeout_seconds,
                write=30.0,
                pool=30.0,
            ),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.config.max_concurrent_requests * 2,
                    max_keepalive_connections=self.config.max_concurrent_requests,
                    keepalive_expiry=30.0,
                ),
            ),
        )
        # Separate httpx client for the fleet's colony paging endpoints
        # (``/load_page`` etc. under the same base_url as ``/v1/chat/...``).
        self._http = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                connect=10.0, read=self.config.api_timeout_seconds,
                write=30.0, pool=30.0,
            ),
        )
        logger.info(
            "Initialized vLLM client for model %s at %s",
            self.config.model_name, self.config.base_url,
        )

    async def _fleet_post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST to a fleet colony endpoint and return the parsed JSON."""
        response = await self._http.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    # ---- colony paging surface: forwarded to the fleet (Option A) --------

    @serving.endpoint(
        router_class=TargetClientRouter,
        router_kwargs={"strip_routing_params": ["target_client_id"]},
    )
    @hookable
    async def load_page(self, page: VirtualContextPage) -> bool:
        """Forward the page load to the fleet, which loads it on a GPU
        replica; the fleet's router then keeps affine requests there."""
        serving.ensure_context(page.page_id, page.syscontext)
        return bool(await self._fleet_post("/load_page", {"page": page.model_dump(mode="json")}))

    @serving.endpoint
    async def evict_page(self, page_id: ContextPageId) -> bool:
        """Forward the page eviction to the fleet."""
        return bool(await self._fleet_post("/evict_page", {"page_id": page_id}))

    @serving.endpoint
    @hookable
    async def infer_with_suffix(
        self,
        base_page_id: ContextPageId,
        request: InferenceRequest,
        suffix_tokens: list[int] | None = None,
        max_concurrent_per_page: int | None = None,
    ) -> InferenceResponse:
        """Forward page-prefixed inference to the fleet, which serves it
        against the cached page on the affine replica."""
        serving.ensure_context(request.request_id, request.syscontext)
        data = await self._fleet_post("/infer_with_suffix", {
            "base_page_id": base_page_id,
            "request": request.model_dump(mode="json"),
            "suffix_tokens": suffix_tokens,
        })
        return InferenceResponse.model_validate(data)

    @hookable
    async def _call_api(
        self,
        messages: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float | None = None,
        json_schema: dict[str, Any] | None = None,
        deadline_s: float | None = None,
        request_id: str | None = None,
    ) -> APIResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages["messages"],
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        if deadline_s is not None:
            kwargs["timeout"] = deadline_s
        # Propagate tenant identity to the fleet's /v1/chat/completions so it
        # can build a valid request context (headless fleet has no ambient one).
        ctx = serving.get_execution_context()
        if ctx is not None and ctx.colony_id and ctx.tenant_id:
            kwargs["extra_headers"] = {
                "X-Colony-Id": ctx.colony_id,
                "X-Tenant-Id": ctx.tenant_id,
            }

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            import httpx as _httpx

            from .errors import LLMCallDeadlineExceeded

            try:
                from openai import APITimeoutError as _OpenAITimeoutError
            except ImportError:  # pragma: no cover — SDK should be present
                _OpenAITimeoutError = ()  # type: ignore[assignment]

            if isinstance(exc, (_OpenAITimeoutError, _httpx.TimeoutException)) and deadline_s is not None:
                raise LLMCallDeadlineExceeded(
                    request_id=request_id or "<unknown>", deadline_s=deadline_s,
                ) from exc
            raise

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cache_read = 0
        if usage and getattr(usage, "prompt_tokens_details", None):
            cache_read = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        return APIResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
            cost_usd=0.0,  # self-hosted — no per-token API cost
            raw_response=response,
        )

    def _build_cached_messages(
        self,
        page_text: str,
        suffix_text: str,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        # cache_control is Anthropic-specific; plain OpenAI messages (vLLM
        # caches prefixes server-side).
        content = f"{page_text}\n\n{suffix_text}" if page_text else suffix_text
        return {
            "messages": [
                {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
