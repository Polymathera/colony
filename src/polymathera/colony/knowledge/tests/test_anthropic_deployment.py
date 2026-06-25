"""Deployment-level contract tests for :class:`AnthropicLLMDeployment`.

The deployment's ``_call_api`` is the seam where the framework's
typed structured-output contract (``InferenceRequest.json_schema``)
meets the provider's native mechanism (Anthropic tool-use). These
tests use a stub Anthropic client whose ``messages.create`` records
the kwargs the deployment dispatches and returns a synthesised
response object — enough to assert:

- the schema becomes ``tools[0].input_schema``;
- ``tool_choice`` is pinned to the schema-derived tool name;
- the returned ``APIResponse.content`` is the JSON string of
  ``tool_use.input`` (NOT a free-text block);
- deadline exhaustion surfaces as the typed
  :class:`LLMCallDeadlineExceeded`, NOT ``asyncio.TimeoutError``.

Real network is never touched — the SDK client is replaced wholesale.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from polymathera.colony.cluster.anthropic_deployment import AnthropicLLMDeployment
from polymathera.colony.cluster.errors import (
    LLMCallDeadlineExceeded,
    LLMInferenceError,
)
from polymathera.colony.cluster.remote_config import RemoteLLMDeploymentConfig


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Stubs for the Anthropic SDK surface the deployment touches
# ---------------------------------------------------------------------------


@dataclass
class _StubUsage:
    input_tokens: int = 10
    output_tokens: int = 20
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _StubTextBlock:
    type: str
    text: str


@dataclass
class _StubResponse:
    usage: _StubUsage
    content: list[Any]


class _StubMessages:
    """Records the most recent ``messages.create`` kwargs and returns
    a configured ``_StubResponse``. Optionally sleeps before returning
    so deadline tests can exercise the timeout path."""

    def __init__(
        self,
        *,
        response: _StubResponse,
        sleep_s: float = 0.0,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._response = response
        self._sleep_s = sleep_s
        self._raise = raise_on_call
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        if self._sleep_s > 0:
            await asyncio.sleep(self._sleep_s)
        if self._raise is not None:
            raise self._raise
        return self._response


class _StubClient:
    def __init__(self, messages: _StubMessages) -> None:
        self.messages = messages


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_deployment(
    *, client: _StubClient,
) -> AnthropicLLMDeployment:
    """Build a deployment instance with a stubbed client, bypassing
    the SDK initialisation path so we don't touch the network."""

    cfg = RemoteLLMDeploymentConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        api_key_env_var="ANTHROPIC_API_KEY",
        api_timeout_seconds=120.0,
        max_concurrent_requests=4,
        ttl="5m",
    )
    deployment = AnthropicLLMDeployment.__new__(AnthropicLLMDeployment)
    # Skip the framework's __init__ side effects — we only test
    # _call_api, which reads self._client / self._pricing.
    deployment.config = cfg
    deployment._client = client
    deployment._pricing = {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.0,
    }
    return deployment


# ---------------------------------------------------------------------------
# Structured-outputs round-trip (output_config.format)
# ---------------------------------------------------------------------------
#
# Reference docs:
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs


async def test_json_schema_dispatches_as_output_config() -> None:
    """A non-None ``json_schema`` produces an ``output_config.format``
    request:
    ``output_config={"format":{"type":"json_schema","schema":...}}``.
    The model is decoder-restricted to emit schema-valid JSON; the
    response is a normal text block whose content IS the JSON. The
    returned ``APIResponse.content`` is read from
    ``response.content[0].text`` (same as plain text mode — there is
    no tool_use pretense)."""

    schema: dict[str, Any] = {
        "type": "object",
        "title": "ClaimList",
        "properties": {
            "claims": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["claims"],
        "additionalProperties": False,
    }
    rendered_json = json.dumps({"claims": [
        {"subject": "X", "predicate": "is_a", "object": "Y", "confidence": 0.8},
    ]})
    stub_response = _StubResponse(
        usage=_StubUsage(),
        content=[_StubTextBlock(type="text", text=rendered_json)],
    )
    messages = _StubMessages(response=stub_response)
    deployment = _build_deployment(client=_StubClient(messages))

    response = await deployment._call_api(
        messages={"messages": [{"role": "user", "content": "extract claims"}]},
        max_tokens=2048,
        temperature=0.0,
        json_schema=schema,
        request_id="req-1",
    )

    assert len(messages.calls) == 1
    sent = messages.calls[0]
    assert sent["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": schema,
        }
    }
    # We are NOT pretending to invoke a tool — neither key should appear.
    assert "tools" not in sent
    assert "tool_choice" not in sent

    # Content lands in the text block; the caller's
    # ``schema.model_validate_json`` round-trips against it.
    assert json.loads(response.content) == json.loads(rendered_json)


async def test_no_json_schema_falls_back_to_text_mode() -> None:
    """Backwards-compatible: a None ``json_schema`` issues a plain
    text-mode request — no ``output_config`` key, content is the
    first text block."""

    stub_response = _StubResponse(
        usage=_StubUsage(),
        content=[_StubTextBlock(type="text", text="hello world")],
    )
    messages = _StubMessages(response=stub_response)
    deployment = _build_deployment(client=_StubClient(messages))

    response = await deployment._call_api(
        messages={"messages": [{"role": "user", "content": "say hi"}]},
        request_id="req-3",
    )

    sent = messages.calls[0]
    assert "output_config" not in sent
    assert "tools" not in sent
    assert response.content == "hello world"


# ---------------------------------------------------------------------------
# Deadline enforcement (Change 8)
# ---------------------------------------------------------------------------


async def test_deadline_threaded_into_sdk_timeout_kwarg() -> None:
    """Per-call ``deadline_s`` becomes the SDK's typed ``timeout=``
    argument — same mechanism the client-level default uses, finer
    grain. NOT ``asyncio.wait_for`` (would leave httpx connection
    dirty)."""

    stub_response = _StubResponse(
        usage=_StubUsage(),
        content=[_StubTextBlock(type="text", text="ok")],
    )
    messages = _StubMessages(response=stub_response)
    deployment = _build_deployment(client=_StubClient(messages))

    await deployment._call_api(
        messages={"messages": [{"role": "user", "content": "."}]},
        deadline_s=12.5,
        request_id="req-4",
    )

    assert messages.calls[0]["timeout"] == 12.5


async def test_anthropic_timeout_error_maps_to_llm_call_deadline_exceeded() -> None:
    """An SDK-level ``APITimeoutError`` raised inside ``messages.create``
    surfaces as the framework's typed
    :class:`LLMCallDeadlineExceeded`, NOT the underlying provider
    exception or a bare ``asyncio.TimeoutError``."""

    import anthropic

    timeout_exc = anthropic.APITimeoutError(request=object())  # type: ignore[arg-type]
    messages = _StubMessages(
        response=_StubResponse(usage=_StubUsage(), content=[]),
        raise_on_call=timeout_exc,
    )
    deployment = _build_deployment(client=_StubClient(messages))

    with pytest.raises(LLMCallDeadlineExceeded) as excinfo:
        await deployment._call_api(
            messages={"messages": [{"role": "user", "content": "."}]},
            deadline_s=0.5,
            request_id="req-5",
        )
    assert excinfo.value.deadline_s == 0.5
    assert excinfo.value.request_id == "req-5"


async def test_non_timeout_error_wraps_as_llm_inference_error() -> None:
    """A non-timeout SDK exception is not swallowed by the deadline
    branch — and is NOT re-raised raw either. The deployment wraps
    every transport-layer exception into
    :class:`LLMInferenceError` carrying an
    :class:`LLMErrorCategory` (TRANSIENT / AUTH / PERMANENT / etc).
    The wrap is the seam the deployment-level circuit breaker reads
    to count failures + decide when to open. Re-raising the raw
    provider/transport exception would defeat the breaker. The
    original exception is preserved as ``__cause__`` for forensics.
    Pins commit 3b0d04d1 (typed-error classification + circuit
    breaker integration)."""

    boom = RuntimeError("network blew up")
    messages = _StubMessages(
        response=_StubResponse(usage=_StubUsage(), content=[]),
        raise_on_call=boom,
    )
    deployment = _build_deployment(client=_StubClient(messages))

    with pytest.raises(LLMInferenceError, match="network blew up") as excinfo:
        await deployment._call_api(
            messages={"messages": [{"role": "user", "content": "."}]},
            deadline_s=10.0,
            request_id="req-6",
        )
    assert excinfo.value.__cause__ is boom


async def test_timeout_without_deadline_wraps_as_llm_inference_error(
) -> None:
    """When no per-call ``deadline_s`` was supplied, a timeout from
    the client-level configuration is NOT promoted to
    :class:`LLMCallDeadlineExceeded` (that subclass is reserved for
    "I asked for X seconds and it fired"). But it IS still wrapped
    into :class:`LLMInferenceError` (category=TRANSIENT) so the
    deployment-level circuit breaker observes repeated background
    timeouts and can open on a downstream outage. Callers that need
    to distinguish "deadline I set" from "background timeout" branch
    on the SUBCLASS (``LLMCallDeadlineExceeded`` vs plain
    ``LLMInferenceError``), not on the raw provider exception type.

    The original ``anthropic.APITimeoutError`` is preserved as
    ``__cause__``."""

    import anthropic

    timeout_exc = anthropic.APITimeoutError(request=object())  # type: ignore[arg-type]
    messages = _StubMessages(
        response=_StubResponse(usage=_StubUsage(), content=[]),
        raise_on_call=timeout_exc,
    )
    deployment = _build_deployment(client=_StubClient(messages))

    with pytest.raises(LLMInferenceError) as excinfo:
        await deployment._call_api(
            messages={"messages": [{"role": "user", "content": "."}]},
            request_id="req-7",
        )
    # NOT the deadline subclass — there was no deadline to exceed.
    assert not isinstance(excinfo.value, LLMCallDeadlineExceeded)
    # Original provider exception preserved on the cause chain so
    # forensics + post-hoc classification still work.
    assert excinfo.value.__cause__ is timeout_exc
