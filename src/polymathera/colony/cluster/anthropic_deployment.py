"""Anthropic LLM deployment — uses Anthropic's Messages API with prefix caching.

This deployment uses Anthropic's native prompt caching to efficiently reuse
VCM page content across multiple inference requests. Cache control markers
create cached prefix entries that are shared across concurrent requests at
0.1x the normal input cost.

Requires: pip install anthropic
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..distributed.ray_utils.rate_limit import RateLimitConfig, TokenBucketRateLimiter
from .remote_config import RemoteLLMDeploymentConfig, get_pricing_for_model
from .remote_deployment import APIResponse, RemoteLLMDeployment

logger = logging.getLogger(__name__)

# Default system prompt when none is configured
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software analyst. Analyze the provided code context "
    "and respond to the task precisely and concisely."
)


class AnthropicLLMDeployment(RemoteLLMDeployment):
    """Anthropic LLM deployment with prefix caching.

    Uses Anthropic's Messages API with cache_control markers to create
    prefix cache entries. The prompt structure:

        [system prompt + cache_control]  → Breakpoint 1 (stable)
        [page text    + cache_control]  → Breakpoint 2 (stable for same page)
        [task suffix]                   → Varies per agent/request

    This uses 2 of 4 available breakpoints. Multiple agents working on the
    same page share the cached prefix at 0.1x cost.
    """

    def __init__(self, config: RemoteLLMDeploymentConfig):
        super().__init__(config)
        self._client = None  # anthropic.AsyncAnthropic
        self._pricing = get_pricing_for_model(config.model_name)
        self._rate_limiter = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_second=config.throttle_rps,
            burst_size=config.throttle_burst,
        ))

    async def _initialize_client(self) -> None:
        """Initialize the Anthropic async client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicLLMDeployment. "
                "Install it with: pip install anthropic"
            )

        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{self.config.api_key_env_var}'. "
                f"Set it with: export {self.config.api_key_env_var}=your-key"
            )

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info(
            f"Initialized Anthropic client for model {self.config.model_name} "
            f"(ttl={self.config.ttl})"
        )

    async def _call_api(
        self,
        messages: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        json_schema: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Call the Anthropic Messages API.

        Args:
            messages: Dict with 'system' and 'messages' keys in Anthropic format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            json_schema: Accepted for interface compatibility but intentionally
                         unused. Anthropic has no native schema enforcement at the decoding level.
                         Callers include format instructions in the prompt, and
                         responses are validated via model_validate_json().

        Returns:
            Normalized APIResponse with usage and cost data
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages["messages"],
        }

        # Add system prompt if present
        if "system" in messages:
            kwargs["system"] = messages["system"]

        ### # When json_schema is provided, inject it as a constraint instruction
        ### # into the last user message. Anthropic doesn't have native schema
        ### # enforcement like OpenAI's strict mode, but Claude follows schema
        ### # instructions reliably when given explicitly.
        ### if json_schema:
        ###     import json as _json
        ###     schema_instruction = (
        ###         "\n\nYou MUST respond with JSON that conforms to the following JSON schema. "
        ###         "Do NOT include any text outside the JSON object.\n"
        ###         f"```json\n{_json.dumps(json_schema, indent=2)}\n```"
        ###     )
        ###     msgs = kwargs["messages"]
        ###     if msgs and isinstance(msgs[-1].get("content"), list):
        ###         # Append to the last text block of the last user message
        ###         last_content = msgs[-1]["content"]
        ###         for block in reversed(last_content):
        ###             if isinstance(block, dict) and block.get("type") == "text":
        ###                 block["text"] += schema_instruction
        ###                 break
        ###     elif msgs and isinstance(msgs[-1].get("content"), str):
        ###         msgs[-1]["content"] += schema_instruction

        logger.info(
            f"Anthropic API request: model={self.config.model_name}, "
            f"max_tokens={max_tokens}, temp={temperature}, "
            f"system_blocks={len(kwargs.get('system', []))}, "
            f"user_blocks={sum(len(m.get('content', [])) if isinstance(m.get('content'), list) else 1 for m in kwargs['messages'])}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            for i, msg in enumerate(kwargs["messages"]):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for j, block in enumerate(content):
                        text = block.get("text", "")
                        has_cache = "cache_control" in block
                        logger.debug(
                            f"  msg[{i}].content[{j}]: len={len(text)}, "
                            f"cache_control={has_cache}, "
                            f"preview={text[:200]!r}..."
                        )
                else:
                    logger.debug(f"  msg[{i}]: {str(content)[:200]!r}...")

        # Throttle request rate to reduce 429s.  The SDK retries
        # rate-limit errors internally with exponential backoff, so we
        # don't add our own retry loop — this just spaces out requests.
        await self._rate_limiter.acquire()

        response = await self._client.messages.create(**kwargs)

        # Extract usage information
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # Calculate cost
        cost_usd = self._calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

        # Extract content text
        content = ""
        if response.content:
            content = response.content[0].text

        logger.info(
            f"Anthropic API response: input={input_tokens}, output={output_tokens}, "
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
        """Build Anthropic messages with cache_control markers.

        Creates a prompt structure with 2 cache breakpoints:
        - Breakpoint 1: System prompt (stable across all requests)
        - Breakpoint 2: Page content (stable for same page)
        - No cache_control on suffix (varies per request)

        Args:
            page_text: The page text (cached prefix)
            suffix_text: The suffix text (varies per request)
            system_prompt: Optional system prompt

        Returns:
            Dict with 'system' and 'messages' keys in Anthropic format
        """
        ttl_value = self.config.ttl  # "5m" or "1h"
        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Build user content blocks — only include page text if non-empty
        # (Anthropic rejects cache_control on empty text blocks)
        user_content = []
        if page_text:
            user_content.append({
                "type": "text",
                "text": page_text,
                "cache_control": {"type": "ephemeral", "ttl": ttl_value},
            })
        user_content.append({
            "type": "text",
            "text": suffix_text,
        })

        return {
            "system": [
                {
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral", "ttl": ttl_value},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
        }

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> float:
        """Calculate cost in USD based on model pricing.

        Args:
            input_tokens: Regular (non-cached) input tokens
            output_tokens: Output tokens generated
            cache_read_tokens: Tokens read from cache (0.1x cost)
            cache_write_tokens: Tokens written to cache (1.25x or 2x cost)

        Returns:
            Estimated cost in USD
        """
        if not self._pricing:
            return 0.0

        # Token counts from Anthropic API:
        # - input_tokens: total input tokens (includes cache_read + cache_write + uncached)
        # - cache_read_input_tokens: tokens read from cache
        # - cache_creation_input_tokens: tokens written to cache
        # Uncached = input_tokens - cache_read - cache_write
        uncached_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)

        cost = 0.0
        cost += uncached_input * self._pricing["input"] / 1_000_000
        cost += output_tokens * self._pricing["output"] / 1_000_000
        cost += cache_read_tokens * self._pricing["cache_read"] / 1_000_000

        # Cache write cost depends on TTL
        cache_write_key = (
            "cache_write_5m" if self.config.ttl == "5m" else "cache_write_1h"
        )
        cost += cache_write_tokens * self._pricing[cache_write_key] / 1_000_000

        return cost
