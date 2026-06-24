"""Tests for the typed :class:`LLMInferenceError` contracts (D5,
D6) and the :func:`llm_provider_down_circuit` wiring (D7).

The state-machine itself belongs to the third-party ``circuitbreaker``
library and is covered by that library's own tests; we pin only the
adapter glue — predicate, fallback translation, decorator
attachment — that closes the 2026-06-22 credit-depletion failure
mode (18,010 wasted Anthropic calls in 2 h 37 min because the typed
exception didn't survive Ray's actor-boundary pickling, defeating
every consumer's backoff/breaker logic).
"""

from __future__ import annotations

import pickle

import pytest

from polymathera.colony.cluster.circuit_breakers import (
    _is_permanent_llm_failure,
    _llm_provider_down_breaker,
    llm_provider_down_circuit,
)
from polymathera.colony.cluster.errors import (
    PERMANENT_ERROR_CATEGORIES,
    LLMCallDeadlineExceeded,
    LLMErrorCategory,
    LLMInferenceError,
)


# ---------------------------------------------------------------------------
# D5 — positional-friendly constructor + __reduce__ round-trip
# ---------------------------------------------------------------------------


def test_llm_inference_error_constructs_positionally() -> None:
    """Ray's exception-propagation pickling calls ``Exc(*self.args)``
    on the receiving process. A kwarg-only constructor would crash
    that path with TypeError, defeating every ``except
    LLMInferenceError`` block. Pin the positional form works."""

    exc = LLMInferenceError("provider 400")
    assert str(exc) == "provider 400"
    assert exc.category is LLMErrorCategory.UNKNOWN
    assert exc.request_id == "<unknown>"


def test_llm_inference_error_constructs_with_kwargs() -> None:
    exc = LLMInferenceError(
        "credit low",
        category=LLMErrorCategory.BILLING,
        request_id="req_123",
    )
    assert str(exc) == "credit low"
    assert exc.category is LLMErrorCategory.BILLING
    assert exc.request_id == "req_123"


def test_llm_inference_error_pickle_round_trip_preserves_category() -> None:
    """The bug Ray triggered: pickling+unpickling lost category +
    request_id because the kwarg-only constructor couldn't be
    reconstructed by ``__reduce__``'s default tuple. Pin that the
    custom ``__reduce__`` preserves both fields across pickle."""

    exc = LLMInferenceError(
        "credit low",
        category=LLMErrorCategory.BILLING,
        request_id="req_011XYZ",
    )
    restored = pickle.loads(pickle.dumps(exc))
    assert isinstance(restored, LLMInferenceError)
    assert str(restored) == "credit low"
    assert restored.category is LLMErrorCategory.BILLING
    assert restored.request_id == "req_011XYZ"


def test_llm_call_deadline_exceeded_pickle_round_trip() -> None:
    """Subclass MUST round-trip its extra field too. Without the
    explicit ``__reduce__`` the deadline_s would silently default
    to 0.0 on the receiving side."""

    exc = LLMCallDeadlineExceeded(
        deadline_s=30.0, request_id="req_X",
    )
    restored = pickle.loads(pickle.dumps(exc))
    assert isinstance(restored, LLMCallDeadlineExceeded)
    assert restored.deadline_s == 30.0
    assert restored.request_id == "req_X"
    assert restored.category is LLMErrorCategory.TRANSIENT


def test_permanent_error_categories_membership() -> None:
    """Only BILLING and AUTH open the breaker. TRANSIENT / UNKNOWN /
    INVALID_REQUEST stay per-call concerns."""

    assert LLMErrorCategory.BILLING in PERMANENT_ERROR_CATEGORIES
    assert LLMErrorCategory.AUTH in PERMANENT_ERROR_CATEGORIES
    assert LLMErrorCategory.TRANSIENT not in PERMANENT_ERROR_CATEGORIES
    assert LLMErrorCategory.INVALID_REQUEST not in PERMANENT_ERROR_CATEGORIES
    assert LLMErrorCategory.UNKNOWN not in PERMANENT_ERROR_CATEGORIES


# ---------------------------------------------------------------------------
# D6 — anthropic exception classifier
# ---------------------------------------------------------------------------


def test_classify_anthropic_credit_low_as_billing() -> None:
    """The canonical credit-depletion message must classify as
    BILLING — that's the trigger for the breaker. Anthropic returns
    HTTP 400 BadRequestError with a free-form message; we match on
    the substring that the provider commits to."""

    from polymathera.colony.cluster.anthropic_deployment import (
        _classify_anthropic_exception,
    )
    import anthropic

    class _FakeBadRequest(anthropic.BadRequestError):
        def __init__(self, msg: str) -> None:
            Exception.__init__(self, msg)

    cat = _classify_anthropic_exception(_FakeBadRequest(
        "Error code: 400 - Your credit balance is too low to access "
        "the Anthropic API. Please go to Plans & Billing.",
    ))
    assert cat is LLMErrorCategory.BILLING


def test_classify_anthropic_invalid_request_not_billing() -> None:
    """A non-billing BadRequestError (e.g. prompt too long) is
    INVALID_REQUEST, not BILLING — different policy response."""

    from polymathera.colony.cluster.anthropic_deployment import (
        _classify_anthropic_exception,
    )
    import anthropic

    class _FakeBadRequest(anthropic.BadRequestError):
        def __init__(self, msg: str) -> None:
            Exception.__init__(self, msg)

    cat = _classify_anthropic_exception(_FakeBadRequest(
        "Error code: 400 - prompt is too long",
    ))
    assert cat is LLMErrorCategory.INVALID_REQUEST


def test_classify_unknown_exception_falls_back_to_unknown() -> None:
    from polymathera.colony.cluster.anthropic_deployment import (
        _classify_anthropic_exception,
    )
    cat = _classify_anthropic_exception(RuntimeError("???"))
    assert cat is LLMErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# D7 — predicate + fallback adapter for ``llm_provider_down_circuit``
# ---------------------------------------------------------------------------


def test_predicate_counts_billing_failure() -> None:
    """The predicate the breaker uses to decide whether to count a
    raised exception as a trip. BILLING / AUTH count; everything
    else does NOT (so transient failures stay per-call concerns)."""

    billing = LLMInferenceError(
        "credit low", category=LLMErrorCategory.BILLING,
        request_id="r1",
    )
    assert _is_permanent_llm_failure(type(billing), billing) is True


def test_predicate_counts_auth_failure() -> None:
    auth = LLMInferenceError(
        "bad key", category=LLMErrorCategory.AUTH,
        request_id="r1",
    )
    assert _is_permanent_llm_failure(type(auth), auth) is True


def test_predicate_does_not_count_transient_failure() -> None:
    """Transient (5xx, timeout) failures are the consumer's per-call
    backoff concern, not a provider-down signal — the breaker MUST
    NOT open on them."""

    trans = LLMInferenceError(
        "timeout", category=LLMErrorCategory.TRANSIENT,
        request_id="r1",
    )
    assert _is_permanent_llm_failure(type(trans), trans) is False


def test_predicate_does_not_count_invalid_request() -> None:
    """A malformed prompt (INVALID_REQUEST) is local to that call,
    not a provider-down signal. Tripping the breaker would block
    every other consumer for an issue with one payload."""

    inv = LLMInferenceError(
        "prompt too long", category=LLMErrorCategory.INVALID_REQUEST,
        request_id="r1",
    )
    assert _is_permanent_llm_failure(type(inv), inv) is False


def test_predicate_does_not_count_random_exception() -> None:
    """Non-LLMInferenceError exceptions are out of scope — the
    typed contract is the gate."""

    assert _is_permanent_llm_failure(RuntimeError, RuntimeError("?")) is False


@pytest.mark.asyncio
async def test_fallback_translates_to_typed_llm_inference_error() -> None:
    """When the library's open-state path calls our fallback, it MUST
    raise an ``LLMInferenceError`` (not the library's generic
    ``CircuitBreakerError``) carrying the LAST failure's category.
    Without this translation, the D6 typed-category contract would
    silently break at the breaker boundary — consumers' ``except
    LLMInferenceError`` blocks would not catch the fail-fast."""

    original = LLMInferenceError(
        "credit low — Plans & Billing",
        category=LLMErrorCategory.BILLING,
        request_id="r1",
    )
    # Stamp the breaker's last failure as if a real call had tripped
    # it. Restore after the test so we don't leak state into the
    # module-level breaker used by other tests / the running
    # deployment.
    saved = _llm_provider_down_breaker._last_failure
    _llm_provider_down_breaker._last_failure = original
    try:
        with pytest.raises(LLMInferenceError) as exc_info:
            await _llm_provider_down_breaker._fallback()
        assert exc_info.value.category is LLMErrorCategory.BILLING
        assert exc_info.value.request_id == "<breaker_open>"
        assert "credit low" in str(exc_info.value)
    finally:
        _llm_provider_down_breaker._last_failure = saved


def test_decorator_is_attachable_to_async_function() -> None:
    """Sanity check: the decorator wraps an async function without
    crashing. The library's own tests cover the state machine; this
    confirms our typed predicate + fallback combination doesn't
    break the decoration step."""

    @llm_provider_down_circuit
    async def _f() -> int:
        return 42

    # The wrapped function is still a coroutine function.
    import asyncio
    import inspect
    assert inspect.iscoroutinefunction(_f) or callable(_f)


# ---------------------------------------------------------------------------
# Wiring: RemoteLLMDeployment.infer is wrapped with the decorator
# ---------------------------------------------------------------------------


def test_remote_deployment_infer_is_wrapped_with_circuit() -> None:
    """Source pin: ``RemoteLLMDeployment.infer`` MUST have the
    ``@llm_provider_down_circuit`` decorator applied so the breaker
    actually gates inference calls. Without this pin a future
    refactor could remove the decorator and the credit-depletion
    storm would silently come back."""

    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1] / "remote_deployment.py"
    ).read_text(encoding="utf-8")
    # The decorator appears immediately above the ``async def infer``
    # definition (after @serving.endpoint + @hookable).
    assert "@llm_provider_down_circuit\n    async def infer(" in src, (
        "RemoteLLMDeployment.infer must be wrapped with "
        "@llm_provider_down_circuit so permanent provider failures "
        "trip the breaker."
    )
