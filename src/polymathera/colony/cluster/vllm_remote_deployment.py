"""Self-hosted remote vLLM deployment — OpenAI-compatible API at a URL.

The *remote-client* counterpart to the in-cluster :class:`VLLMDeployment`:
this talks to a model served on a separate, self-hosted vLLM behind an
OpenAI-compatible ``/v1`` endpoint (e.g. a fine-tuned LoRA adapter). It
sends plain OpenAI messages — Anthropic ``cache_control`` markers don't
apply to the OpenAI API (vLLM does prefix caching server-side when
enabled). Self-hosted, so ``cost_usd`` is modeled as 0 (no per-token API
charge). Requires: pip install openai.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..distributed.hooks import hookable
from .remote_config import RemoteLLMDeploymentConfig
from .remote_deployment import APIResponse, RemoteLLMDeployment
from .remote_registry import register_remote_llm_provider

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
        self._client = None  # openai.AsyncOpenAI

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
        logger.info(
            "Initialized vLLM client for model %s at %s",
            self.config.model_name, self.config.base_url,
        )

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
