"""Typed exception surface for the LLM cluster.

A single :class:`LLMInferenceError` lets callers (action policies,
extractors) distinguish "the LLM cluster failed" from "my own code
raised" so they can apply policy-specific responses — exponential
backoff on transient cluster failures (credit-out, rate-limit, 5xx)
without poisoning the agent loop's iteration accounting.
"""

from __future__ import annotations


class LLMInferenceError(Exception):
    """Raised by the LLM cluster when an inference request fails.

    The original provider-specific exception is chained via
    ``__cause__`` (``raise LLMInferenceError(...) from e``); the
    string message is the human-readable failure description (typically
    the provider's error body). ``request_id`` is the cluster's
    request identifier — useful for cross-referencing the failure
    against ``Inference failed for request ...`` log lines.
    """

    def __init__(self, *, request_id: str, message: str) -> None:
        super().__init__(message)
        self.request_id = request_id


__all__ = ("LLMInferenceError",)
