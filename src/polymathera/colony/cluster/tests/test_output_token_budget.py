"""Resolution ladder for the config-declared output-token budget
(OutputTokenBudgetMixin) — explicit caller cap > per-effort map >
deployment default. Replaces the hardcoded per-call-site max_tokens
literals (2026-08-07: a hardcoded 2048 truncated schema-constrained
codegen once adaptive thinking shared the budget)."""

from polymathera.colony.cluster.config import LLMDeploymentConfig
from polymathera.colony.cluster.remote_config import RemoteLLMDeploymentConfig


def _remote(**kw) -> RemoteLLMDeploymentConfig:
    return RemoteLLMDeploymentConfig(
        model_name="claude-sonnet-5", provider="anthropic",
        api_key_env_var="ANTHROPIC_API_KEY", **kw,
    )


def test_default_when_caller_passes_none() -> None:
    assert _remote().resolve_max_output_tokens(None, None) == 8192


def test_explicit_caller_cap_wins_over_everything() -> None:
    cfg = _remote(max_output_tokens_by_effort={"max": 64000})
    assert cfg.resolve_max_output_tokens(1, "max") == 1  # warmup pings


def test_effort_map_overrides_base_default() -> None:
    cfg = _remote(
        max_output_tokens=8192,
        max_output_tokens_by_effort={"high": 16384, "max": 64000},
    )
    assert cfg.resolve_max_output_tokens(None, "high") == 16384
    assert cfg.resolve_max_output_tokens(None, "max") == 64000
    # Unmapped effort falls back to the base default.
    assert cfg.resolve_max_output_tokens(None, "low") == 8192


def test_vllm_side_shares_the_mixin() -> None:
    cfg = LLMDeploymentConfig(model_name="m", max_output_tokens=4096)
    assert cfg.resolve_max_output_tokens(None, None) == 4096


def test_vllm_sampling_kwargs_execute_the_resolution() -> None:
    """EXECUTES ``_request_sampling_kwargs`` (regression: a blind edit
    referenced ``self`` inside what was then a @staticmethod, and the
    suite stayed green because nothing ran the method). Bypasses the
    engine-heavy __init__ via __new__ — the method touches only
    ``self._output_budget``."""
    import pytest

    pytest.importorskip("vllm")  # module imports vllm at top level
    from polymathera.colony.cluster.models import (
        InferenceRequest, OutputTokenBudgetMixin,
    )
    from polymathera.colony.cluster.vllm_deployment import VLLMDeployment

    deployment = VLLMDeployment.__new__(VLLMDeployment)
    deployment._output_budget = OutputTokenBudgetMixin(
        max_output_tokens=4096,
    )
    request = InferenceRequest(
        request_id="r1", prompt="p", temperature=0.1,
    )
    kwargs = deployment._request_sampling_kwargs(request)
    assert kwargs["max_tokens"] == 4096  # None → config budget
    request_explicit = InferenceRequest(
        request_id="r2", prompt="p", temperature=0.1, max_tokens=64,
    )
    assert deployment._request_sampling_kwargs(
        request_explicit,
    )["max_tokens"] == 64  # explicit caller cap wins


import pytest


@pytest.mark.asyncio
async def test_anthropic_call_api_sends_resolved_integer_max_tokens() -> None:
    """EXECUTES ``_call_api`` payload assembly and asserts on the
    kwargs the Anthropic client actually receives. Regression for
    2026-08-07: the budget resolution was inserted AFTER the request
    dict was built, rebinding only the local — the API got
    ``max_tokens=None`` (provider 400 'Input should be a valid
    integer') while the request log printed the resolved value."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from polymathera.colony.cluster.anthropic_deployment import (
        AnthropicLLMDeployment,
    )

    deployment = AnthropicLLMDeployment.__new__(AnthropicLLMDeployment)
    deployment.config = _remote(
        max_output_tokens=8192,
        max_output_tokens_by_effort={"high": 16384},
        effort="high",
    )
    deployment._pricing = None  # _calculate_cost guard path
    response_stub = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
    )
    create = AsyncMock(return_value=response_stub)
    deployment._client = SimpleNamespace(
        messages=SimpleNamespace(create=create),
    )

    result = await deployment._call_api(
        {"messages": [{"role": "user", "content": "hi"}]},
        max_tokens=None,
    )
    sent = create.await_args.kwargs
    assert sent["max_tokens"] == 16384          # by-effort map (config effort=high)
    assert isinstance(sent["max_tokens"], int)  # never None on the wire
    assert sent["output_config"]["effort"] == "high"
    assert result.stop_reason == "end_turn"

    # Explicit caller cap wins end-to-end too.
    await deployment._call_api(
        {"messages": [{"role": "user", "content": "hi"}]},
        max_tokens=64,
    )
    assert create.await_args.kwargs["max_tokens"] == 64
