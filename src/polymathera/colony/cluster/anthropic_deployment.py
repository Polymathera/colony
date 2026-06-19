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

from ..distributed.hooks import hookable
from .remote_config import RemoteLLMDeploymentConfig, get_pricing_for_model
from .remote_deployment import APIResponse, RemoteLLMDeployment
from .remote_registry import register_remote_llm_provider

logger = logging.getLogger(__name__)

# Default system prompt when none is configured
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert software analyst. Analyze the provided code context "
    "and respond to the task precisely and concisely."
)


@register_remote_llm_provider("anthropic")
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

        import httpx

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            # SDK retries 429 / 529 with exponential backoff and honors
            # ``Retry-After`` natively. Bumped from the default of 2 to
            # absorb per-tier rate-limit spikes when many callers fan
            # out through the same deployment (e.g.,
            # ``materialize_design_context_sources``'s per-file
            # ingestion).
            max_retries=5,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.config.api_timeout_seconds,
                write=30.0,
                pool=30.0,  # Don't wait forever for a connection from the pool
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
            f"Initialized Anthropic client for model {self.config.model_name} "
            f"(ttl={self.config.ttl}, pool={self.config.max_concurrent_requests * 2})"
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
        """Call the Anthropic Messages API.

        Args:
            messages: Dict with 'system' and 'messages' keys in Anthropic format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            json_schema: Optional JSON Schema (e.g. ``Model.model_json_schema()``).
                         When supplied, the call uses Anthropic's dedicated
                         structured-outputs feature via
                         ``output_config={"format":{"type":"json_schema","schema":...}}``,
                         which applies grammar-constrained sampling: the model is
                         decoder-restricted to emit only schema-valid output. The
                         returned ``APIResponse.content`` is the JSON string the model
                         emitted (read from ``response.content[0].text`` — same shape
                         as text mode). This is the idiomatic API for "I just want
                         JSON conforming to my schema" — we are NOT pretending to
                         invoke a tool. Reference:
                         https://platform.claude.com/docs/en/build-with-claude/structured-outputs
                         Sibling deployments (vLLM ``guided_json``, OpenRouter
                         ``response_format``) honor the same field via their
                         provider-native mechanisms; the contract is uniform.

                         JSON Schema limitations enforced by Anthropic (caller
                         responsibility — surfaces as a 400 from the API at request
                         time if violated): no ``minLength``/``maxLength`` (string
                         constraints), no ``minimum``/``maximum``/``multipleOf``
                         (numeric constraints), no recursive schemas, no external
                         ``$ref``, ``additionalProperties`` MUST be ``false`` on
                         every object. The pydantic models we feed in here
                         (``ClaimList`` etc.) are shaped accordingly; value-range
                         enforcement happens AFTER parse via field_validators.

        Returns:
            Normalized APIResponse with usage and cost data
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages["messages"],
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        # Add system prompt if present
        if "system" in messages:
            kwargs["system"] = messages["system"]

        # Structured output via the dedicated structured-outputs API
        # (grammar-constrained sampling). The model is decoder-restricted
        # to emit schema-valid JSON; the response lands in the same
        # text-block surface as plain text mode (no tool_use pretense).
        # Reference:
        # https://platform.claude.com/docs/en/build-with-claude/structured-outputs
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": json_schema,
                },
            }

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

        # Timeout is configured on the httpx client (see _initialize_client),
        # NOT via asyncio.wait_for — cancelling a mid-flight httpx request
        # can leave the connection in a dirty state, exhausting the pool.
        # When the caller supplies ``deadline_s`` we override the client's
        # default timeout on a per-request basis via the SDK's typed
        # ``timeout=`` argument — same mechanism, finer grain. On
        # exhaustion the SDK raises ``anthropic.APITimeoutError`` (or
        # ``httpx.TimeoutException``); we map both to the framework's
        # typed ``LLMCallDeadlineExceeded`` so consumers can count and
        # respond to deadline exhaustion separately from transport
        # failures.
        if deadline_s is not None:
            kwargs["timeout"] = deadline_s
        logger.debug(
            f"[TRACE] AnthropicLLMDeployment._call_api: BEFORE messages.create() "
            f"request_id={request_id} model={self.config.model_name} "
            f"max_tokens={max_tokens} deadline_s={deadline_s}"
        )
        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            import anthropic  # local import — anthropic SDK loaded lazily
            import httpx as _httpx
            from .errors import LLMCallDeadlineExceeded

            is_timeout = isinstance(
                exc,
                (anthropic.APITimeoutError, _httpx.TimeoutException),
            )
            if is_timeout and deadline_s is not None:
                raise LLMCallDeadlineExceeded(
                    request_id=request_id or "<unknown>",
                    deadline_s=deadline_s,
                ) from exc
            raise
        logger.debug(
            f"[TRACE] AnthropicLLMDeployment._call_api: AFTER messages.create() "
            f"request_id={request_id} model={self.config.model_name}"
        )

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

        # Extract content. Both text mode and structured-outputs mode
        # land their payload in the first text block; the difference is
        # that structured-outputs guarantees the text validates against
        # ``json_schema``. The caller's ``model_validate_json`` path is
        # uniform across deployments (vLLM and OpenRouter return JSON
        # strings the same way).
        content = ""
        if response.content:
            content = getattr(response.content[0], "text", "") or ""
        output_mode = "json_schema" if json_schema is not None else "text"

        logger.info(
            f"Anthropic API response: input={input_tokens}, output={output_tokens}, "
            f"cache_read={cache_read}, cache_write={cache_write}, "
            f"cost=${cost_usd:.6f}, response_len={len(content)}, "
            f"mode={output_mode}"
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
