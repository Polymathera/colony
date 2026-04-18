"""OpenRouter LLM deployment — uses OpenAI-compatible API with cache passthrough.

OpenRouter provides an OpenAI-compatible API that passes through Anthropic's
cache_control markers for Claude models, enabling prefix caching via the
underlying Anthropic infrastructure.

Requires: pip install openai
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..distributed.hooks import hookable
from .remote_config import RemoteLLMDeploymentConfig, get_pricing_for_model
from .remote_deployment import APIResponse, RemoteLLMDeployment

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default system prompt when none is configured
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software analyst. Analyze the provided code context "
    "and respond to the task precisely and concisely."
)


class OpenRouterLLMDeployment(RemoteLLMDeployment):
    """OpenRouter LLM deployment with cache passthrough for Claude models.

    Uses the OpenAI-compatible API with Anthropic cache_control markers
    passed through for Claude models. For non-Claude models, caching
    is provider-dependent.

    Supports:
    - Provider order preferences for sticky routing
    - Cache passthrough for Claude models
    - OpenAI-compatible response format
    """

    def __init__(self, config: RemoteLLMDeploymentConfig):
        super().__init__(config)
        self._client = None  # openai.AsyncOpenAI
        self._pricing = get_pricing_for_model(config.model_name)
        self._is_claude_model = "claude" in config.model_name.lower()
    async def _initialize_client(self) -> None:
        """Initialize the OpenAI async client pointed at OpenRouter."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenRouterLLMDeployment. "
                "Install it with: pip install openai"
            )

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{self.config.api_key_env_var}'. "
                f"Set it with: export {self.config.api_key_env_var}=your-key"
            )

        import httpx

        self._client = openai.AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
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
            f"Initialized OpenRouter client for model {self.config.model_name} "
            f"(is_claude={self._is_claude_model})"
        )

    @hookable
    async def _call_api(
        self,
        messages: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float | None = None,
        json_schema: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> APIResponse:
        """Call the OpenRouter API (OpenAI-compatible).

        Args:
            messages: Dict with 'messages' key in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            json_schema: Optional JSON schema for structured output

        Returns:
            Normalized APIResponse with usage data
        """
        # Some models reject having both temperature and top_p.
        # Only include top_p when explicitly overridden from the default.
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages["messages"],
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        # Add extra headers for OpenRouter
        extra_headers: dict[str, str] = {}
        if self.config.openrouter_provider_order:
            # Request specific provider ordering for sticky routing
            extra_headers["X-Provider-Order"] = ",".join(
                self.config.openrouter_provider_order
            )

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        # Add response format for structured output
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        logger.info(
            f"OpenRouter API request: model={self.config.model_name}, "
            f"max_tokens={max_tokens}, temp={temperature}, "
            f"num_messages={len(kwargs['messages'])}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            for i, msg in enumerate(kwargs["messages"]):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, list):
                    for j, block in enumerate(content):
                        text = block.get("text", "")
                        has_cache = "cache_control" in block
                        logger.debug(
                            f"  msg[{i}] role={role} block[{j}]: len={len(text)}, "
                            f"cache_control={has_cache}, "
                            f"preview={text[:200]!r}..."
                        )
                else:
                    logger.debug(
                        f"  msg[{i}] role={role}: {str(content)[:200]!r}..."
                    )

        # Timeout is configured on the httpx client (see _initialize_client),
        # NOT via asyncio.wait_for — cancelling a mid-flight httpx request
        # can leave the connection in a dirty state, exhausting the pool.
        response = await self._client.chat.completions.create(**kwargs)

        # Extract usage information
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # OpenRouter may include cache info for Claude models
        cache_read = 0
        cache_write = 0
        if usage and hasattr(usage, "prompt_tokens_details"):
            details = usage.prompt_tokens_details
            if details:
                cache_read = getattr(details, "cached_tokens", 0) or 0

        # Calculate cost
        cost_usd = self._calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

        # Extract content text
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        logger.info(
            f"OpenRouter API response: input={input_tokens}, output={output_tokens}, "
            f"cache_read={cache_read}, cache_write={cache_write}, "
            f"cost=${cost_usd:.6f}, response_len={len(content)}"
        )

        return APIResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
            cost_usd=cost_usd,
            raw_response=response,
        )

    def _build_cached_messages(
        self,
        page_text: str,
        suffix_text: str,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        """Build OpenAI-compatible messages with cache_control passthrough.

        For Claude models on OpenRouter, cache_control markers are passed
        through to Anthropic's infrastructure. For other models, markers
        are included but may be ignored by the provider.

        Args:
            page_text: The page text (cached prefix)
            suffix_text: The suffix text (varies per request)
            system_prompt: Optional system prompt

        Returns:
            Dict with 'messages' key in OpenAI format
        """
        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": prompt,
            },
        ]

        # For Claude models, add cache_control for prefix caching
        if self._is_claude_model:
            messages[0]["cache_control"] = {"type": "ephemeral"}

            # Only include page text block if non-empty
            # (cache_control on empty text blocks is rejected)
            user_content = []
            if page_text:
                user_content.append({
                    "type": "text",
                    "text": page_text,
                    "cache_control": {"type": "ephemeral"},
                })
            user_content.append({
                "type": "text",
                "text": suffix_text,
            })

            messages.append({
                "role": "user",
                "content": user_content,
            })
        else:
            # Non-Claude models: simple text messages (no caching)
            content = f"{page_text}\n\n{suffix_text}" if page_text else suffix_text
            messages.append({
                "role": "user",
                "content": content,
            })

        return {"messages": messages}

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> float:
        """Calculate cost in USD based on model pricing.

        OpenRouter uses provider pricing + markup. For Claude models,
        cache discounts are reflected.

        Args:
            input_tokens: Total input tokens
            output_tokens: Output tokens generated
            cache_read_tokens: Tokens read from cache
            cache_write_tokens: Tokens written to cache

        Returns:
            Estimated cost in USD
        """
        if not self._pricing:
            return 0.0

        uncached_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)

        cost = 0.0
        cost += uncached_input * self._pricing["input"] / 1_000_000
        cost += output_tokens * self._pricing["output"] / 1_000_000
        cost += cache_read_tokens * self._pricing["cache_read"] / 1_000_000

        # OpenRouter uses 5m TTL by default (no 1h option in passthrough)
        cost += cache_write_tokens * self._pricing["cache_write_5m"] / 1_000_000

        return cost
