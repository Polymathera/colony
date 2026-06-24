"""Circuit breaker configurations for LLM cluster operations.

This module provides centralized circuit breaker policies for protecting
against cascading failures in distributed LLM operations.
"""

from typing import Type

from circuitbreaker import CircuitBreaker, circuit

from .errors import (
    PERMANENT_ERROR_CATEGORIES,
    LLMErrorCategory,
    LLMInferenceError,
)

# Circuit breaker for critical inference operations
# Higher threshold (10 failures) as inference failures may be transient
# and we want to maintain availability for critical operations
inference_circuit = circuit(
    failure_threshold=10,
    recovery_timeout=30,
    expected_exception=Exception,
    name="vllm_inference"
)

# Circuit breaker for page loading operations
# Moderate tolerance (5 failures) for KV cache page operations
page_loading_circuit = circuit(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=Exception,
    name="vllm_page_loading"
)

# Circuit breaker for S3 operations
# Stricter threshold (5 failures) since S3 issues are often systemic
# Longer recovery timeout (60s) to allow for AWS service recovery
s3_operations_circuit = circuit(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception,
    name="vllm_s3_operations"
)


# ---------------------------------------------------------------------------
# Provider-down circuit (BILLING / AUTH) for remote LLM deployments.
#
# Distinct from ``inference_circuit`` above because the trigger semantic
# is different:
#
# - ``inference_circuit`` opens after N CONSECUTIVE failures (any kind) —
#   right for the in-process vLLM where transient model-load errors
#   should tolerate retries.
# - Below opens on the FIRST permanent-category failure (credit-out,
#   auth) because retrying a billing error 10 times in a row just
#   compounds the wasted spend with no chance of recovery.
#
# The predicate restricts the trigger to permanent categories; transient
# failures stay the consumer's per-call backoff concern
# (``LLMFailureBackoff``). The fallback translates the library's
# generic ``CircuitBreakerError`` back into a typed ``LLMInferenceError``
# carrying the original failure's category so consumers' existing
# ``except LLMInferenceError`` blocks keep working — without this
# translation, the typed contract from D6 would silently break at the
# breaker boundary.
# ---------------------------------------------------------------------------


def _is_permanent_llm_failure(
    _exc_type: Type[BaseException], exc_value: BaseException,
) -> bool:
    """Predicate the breaker uses to decide whether to count a
    raised exception as a "provider is down" failure.

    Only true for typed :class:`LLMInferenceError` whose
    ``.category`` is in :data:`PERMANENT_ERROR_CATEGORIES` (BILLING,
    AUTH). Transient / unknown / invalid-request failures DO NOT
    open the breaker — those are per-call concerns the existing
    backoff handles."""

    return (
        isinstance(exc_value, LLMInferenceError)
        and exc_value.category in PERMANENT_ERROR_CATEGORIES
    )


class _LLMProviderDownBreaker(CircuitBreaker):
    """Breaker subclass whose fallback is a bound method, so it can
    read ``self._last_failure`` directly without reaching into the
    library's private state from module scope. Subclassing is the
    library's documented extension point — the constructor honors
    a ``FALLBACK_FUNCTION`` class attribute and the docstring shows
    callable predicates on ``EXPECTED_EXCEPTION``."""

    FAILURE_THRESHOLD = 1
    RECOVERY_TIMEOUT = 300
    EXPECTED_EXCEPTION = _is_permanent_llm_failure

    async def _fallback(self, *_args, **_kwargs) -> None:
        """Re-raise the last failure as a typed
        :class:`LLMInferenceError` carrying the original category,
        so consumers' ``except LLMInferenceError`` blocks keep
        branching on ``.category`` contract. The library's own
        ``CircuitBreakerError`` would defeat that contract."""

        last = self._last_failure
        category = (
            last.category
            if isinstance(last, LLMInferenceError)
            else LLMErrorCategory.UNKNOWN
        )
        raise LLMInferenceError(
            (
                f"LLM provider circuit breaker OPEN: {last!s}"
                if last else "LLM provider circuit breaker OPEN"
            ),
            category=category,
            request_id="<breaker_open>",
        )

    def __init__(self) -> None:
        # Pass the bound method through the documented
        # ``fallback_function`` constructor arg — no post-construction
        # reach-through into private attributes.
        super().__init__(
            fallback_function=self._fallback,
            name="llm_provider_down",
        )


# Module-level instance so the same breaker state is shared across
# all ``RemoteLLMDeployment`` instances using ``@llm_provider_down_circuit``.
# A single recorded billing failure stops every consumer immediately;
# half-open probe checks recovery every 5 min.
_llm_provider_down_breaker = _LLMProviderDownBreaker()


def llm_provider_down_circuit(func):
    """Decorator wrapper around the module-level breaker. Use as
    ``@llm_provider_down_circuit`` on the decorated function — same
    shape as the existing ``@inference_circuit`` / ``@page_loading_circuit``."""

    return _llm_provider_down_breaker(func)
