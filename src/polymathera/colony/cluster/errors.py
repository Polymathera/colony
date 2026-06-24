"""Typed exception surface for the LLM cluster.

A single :class:`LLMInferenceError` lets callers (action policies,
extractors) distinguish "the LLM cluster failed" from "my own code
raised" so they can apply policy-specific responses — exponential
backoff on transient cluster failures (5xx, timeouts, rate-limits)
without poisoning the agent loop's iteration accounting; fail-fast
on permanent ones (credit-out, auth, invalid request).

The exception MUST be reconstructible from positional args because
Ray's exception-propagation pickling reaches into the receiving
process and calls ``LLMInferenceError(*self.args)`` to re-raise.
A kwarg-only constructor would crash that path with a TypeError that
the caller's ``except LLMInferenceError`` block never sees — silently
defeating every consumer's retry/backoff logic. The 2026-06-22
credit-depletion incident generated 18,010 wasted Anthropic API
calls for exactly this reason. See [[search-before-writing]] for
the broader principle (cross-actor exceptions need a
pickle-friendly constructor).
"""

from __future__ import annotations

from enum import Enum


class LLMErrorCategory(str, Enum):
    """How the cluster classifies an inference failure for the
    consumer to branch on.

    The category is the SINGLE contract every consumer reads. Adding
    a new category means defining what kind of policy response it
    deserves (retry, fail-fast, circuit-break) — NOT scattering
    string matches over error messages.
    """

    TRANSIENT = "transient"
    """5xx, network glitch, rate-limit, provider timeout. Retry with
    backoff. The cluster's exponential-backoff machinery handles
    this category; consumers do not need to do anything special."""

    BILLING = "billing"
    """Provider says the account is out of credit / payment required.
    Permanent until a human acts; NO amount of retrying will fix it.
    Consumers should park (idle-wait) and surface a typed
    AgentDiagnostic so the operator is paged. The deployment-level
    circuit breaker opens on this category."""

    AUTH = "auth"
    """Invalid API key, revoked credentials, organization-suspended.
    Permanent until a human acts; same response as BILLING — park,
    surface diagnostic, open circuit breaker."""

    INVALID_REQUEST = "invalid_request"
    """The provider rejected the REQUEST itself as malformed
    (unknown model, prompt too long, schema violation, etc.). Not
    retryable with the same payload. Consumers should fail-fast on
    this specific call rather than backoff-retry the same broken
    request indefinitely."""

    UNKNOWN = "unknown"
    """The cluster couldn't classify the exception. Treat as
    TRANSIENT by default — better to retry an unknown failure than
    silently swallow it — but emit a loud log so the missing
    classifier surfaces. ``category=UNKNOWN`` in production logs is
    a request to extend the classifier."""


# Categories that the deployment-level circuit breaker treats as
# "this provider is hard down for everyone — fail every subsequent
# call fast until the half-open probe succeeds". Per-call policy
# (backoff vs park vs fail-fast) is the consumer's choice; the
# breaker is just a thundering-herd dampener that stops 18,010
# retries before they happen.
PERMANENT_ERROR_CATEGORIES: frozenset[LLMErrorCategory] = frozenset({
    LLMErrorCategory.BILLING,
    LLMErrorCategory.AUTH,
})


class LLMInferenceError(Exception):
    """Raised by the LLM cluster when an inference request fails.

    The original provider-specific exception is chained via
    ``__cause__`` (``raise LLMInferenceError(...) from e``); the
    string message is the human-readable failure description (typically
    the provider's error body). ``request_id`` is the cluster's
    request identifier — useful for cross-referencing the failure
    against ``Inference failed for request ...`` log lines.
    ``category`` lets consumers branch on the kind of failure without
    string-matching error bodies.

    Constructor accepts BOTH positional and kwarg forms so the
    exception round-trips correctly through Ray's actor-boundary
    pickling. ``LLMInferenceError(message)`` (positional, matching
    Ray's deserializer) and
    ``LLMInferenceError(message, category=..., request_id=...)``
    (named) both work.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        category: LLMErrorCategory = LLMErrorCategory.UNKNOWN,
        request_id: str = "<unknown>",
    ) -> None:
        super().__init__(message or "")
        self.category = category
        self.request_id = request_id

    def __reduce__(self):
        """Pickle-friendly reconstruction. Ray's actor-boundary
        exception propagation re-raises via the tuple returned here,
        preserving category + request_id across processes. Without
        an explicit ``__reduce__``, the kwarg-only fields silently
        drop and the receiver sees ``category=UNKNOWN`` /
        ``request_id='<unknown>'`` regardless of what the producer
        set."""

        return (
            _reconstruct_llm_inference_error,
            (str(self), self.category.value, self.request_id),
        )


def _reconstruct_llm_inference_error(
    message: str,
    category_value: str,
    request_id: str,
) -> LLMInferenceError:
    """Module-level factory for ``LLMInferenceError.__reduce__``.

    Module-level (rather than a classmethod) so unpickling resolves
    the symbol by import path without instantiating the class first
    — matches the standard pickle factory shape."""

    try:
        category = LLMErrorCategory(category_value)
    except ValueError:
        category = LLMErrorCategory.UNKNOWN
    return LLMInferenceError(
        message, category=category, request_id=request_id,
    )


class LLMCallDeadlineExceeded(LLMInferenceError):
    """Raised when an inference call exceeds its per-call wall-clock
    deadline (``InferenceRequest.deadline_s`` or the cluster's
    ``llm_per_call_deadline_s`` default).

    Distinct from the generic :class:`LLMInferenceError` so consumers
    (e.g. :class:`LLMClaimExtractor`) can count and respond to
    deadline exhaustion separately from transport-level failures. The
    deadline is enforced by the deployment at the SDK's per-request
    timeout boundary — NOT via ``asyncio.wait_for``, which would leave
    the HTTP connection in a dirty state and exhaust the pool. Per
    [[no-bandaids-durable-solutions]] the deadline is mandatory: no
    ``0.0 = disable`` escape hatch is provided. If the bound is
    correct, it is not optional.

    Inherits the positional-friendly constructor and ``__reduce__``
    from the base so the deadline value round-trips across the Ray
    actor boundary along with category + request_id.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        deadline_s: float = 0.0,
        request_id: str = "<unknown>",
    ) -> None:
        super().__init__(
            message or (
                f"LLM call exceeded deadline of {deadline_s:.2f}s "
                f"(request_id={request_id})"
            ),
            category=LLMErrorCategory.TRANSIENT,
            request_id=request_id,
        )
        self.deadline_s = deadline_s

    def __reduce__(self):
        return (
            _reconstruct_llm_call_deadline_exceeded,
            (str(self), self.deadline_s, self.request_id),
        )


def _reconstruct_llm_call_deadline_exceeded(
    message: str,
    deadline_s: float,
    request_id: str,
) -> LLMCallDeadlineExceeded:
    return LLMCallDeadlineExceeded(
        message, deadline_s=deadline_s, request_id=request_id,
    )


__all__ = (
    "LLMInferenceError",
    "LLMCallDeadlineExceeded",
    "LLMErrorCategory",
    "PERMANENT_ERROR_CATEGORIES",
)
