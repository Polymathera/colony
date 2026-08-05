"""Typed-attribute propagation across the DeploymentResponse boundary.

Deployment errors cross to callers as strings and are re-raised via
``exception_class(error_msg)`` — message-only, so kwarg-carried
attributes reset to defaults at every DeploymentHandle hop. The
2026-08-03 ingest run surfaced the blast radius: a BILLING
``LLMInferenceError`` arrived as UNKNOWN, so the 300s billing backoff
floor never engaged and the claim extractor degraded instead of
raising. :class:`SupportsWireFields` is the structural contract that
carries such attributes across; these tests pin both ends plus the
multi-hop round trip.
"""

from __future__ import annotations

import pytest

from polymathera.colony.cluster.errors import (
    LLMErrorCategory,
    LLMInferenceError,
    PERMANENT_ERROR_CATEGORIES,
)
from polymathera.colony.distributed.ray_utils.serving.models import (
    DeploymentResponse,
    SupportsWireFields,
)


def test_llm_inference_error_implements_protocol() -> None:
    exc = LLMInferenceError(
        "credit balance too low",
        category=LLMErrorCategory.BILLING,
        request_id="req-1",
    )
    assert isinstance(exc, SupportsWireFields)
    assert issubclass(LLMInferenceError, SupportsWireFields)
    assert exc.wire_fields() == {
        "category": "billing", "request_id": "req-1",
    }


def test_with_error_captures_wire_fields() -> None:
    exc = LLMInferenceError(
        "boom", category=LLMErrorCategory.BILLING, request_id="req-2",
    )
    resp = DeploymentResponse.with_error("rid", exc)
    assert resp.error_fields == {
        "category": "billing", "request_id": "req-2",
    }


def test_with_error_plain_exception_has_no_fields() -> None:
    resp = DeploymentResponse.with_error("rid", ValueError("plain"))
    assert resp.error_fields is None


def test_from_wire_round_trip_preserves_category() -> None:
    original = LLMInferenceError(
        "credit balance too low",
        category=LLMErrorCategory.BILLING,
        request_id="req-3",
    )
    rebuilt = LLMInferenceError.from_wire(
        "Error in llm_cluster.infer: credit balance too low",
        original.wire_fields(),
    )
    assert rebuilt.category is LLMErrorCategory.BILLING
    assert rebuilt.request_id == "req-3"
    # The rebuilt category must still trip the permanent-failure
    # consumers (backoff floor, claim-extractor raise).
    assert rebuilt.category in PERMANENT_ERROR_CATEGORIES


def test_from_wire_tolerates_missing_and_unknown_keys() -> None:
    rebuilt = LLMInferenceError.from_wire("msg", {})
    assert rebuilt.category is LLMErrorCategory.UNKNOWN
    assert rebuilt.request_id == "<unknown>"
    rebuilt2 = LLMInferenceError.from_wire(
        "msg", {"category": "not-a-category"},
    )
    assert rebuilt2.category is LLMErrorCategory.UNKNOWN


def test_multi_hop_round_trip_survives() -> None:
    """Producer → response → reconstruct → producer again — the shape
    of a two-deployment chain (AnthropicLLMDeployment → LLMCluster →
    agent). Category must survive every hop."""

    exc: Exception = LLMInferenceError(
        "hop-0", category=LLMErrorCategory.BILLING, request_id="r",
    )
    for hop in range(3):
        resp = DeploymentResponse.with_error(f"rid-{hop}", exc)
        assert resp.error_fields is not None
        exc = LLMInferenceError.from_wire(
            f"Error in hop{hop}.infer: {resp.error}", resp.error_fields,
        )
    assert exc.category is LLMErrorCategory.BILLING  # type: ignore[union-attr]


def test_response_serialization_round_trip() -> None:
    """error_fields must survive the pydantic dump/validate cycle the
    response actually goes through between processes."""

    exc = LLMInferenceError(
        "x", category=LLMErrorCategory.AUTH, request_id="req-4",
    )
    resp = DeploymentResponse.with_error("rid", exc)
    revived = DeploymentResponse.model_validate(resp.model_dump())
    assert revived.error_fields == {
        "category": "auth", "request_id": "req-4",
    }


def test_deadline_subclass_round_trips_deadline_and_type() -> None:
    """The deadline subclass pins its own category and carries
    ``deadline_s`` — its wire pair must round-trip both, and the
    reconstruction must produce the SUBCLASS, not the base."""

    from polymathera.colony.cluster.errors import LLMCallDeadlineExceeded

    exc = LLMCallDeadlineExceeded(deadline_s=12.5, request_id="r5")
    resp = DeploymentResponse.with_error("rid", exc)
    assert resp.error_fields == {"deadline_s": "12.5", "request_id": "r5"}

    rebuilt = LLMCallDeadlineExceeded.from_wire("msg", resp.error_fields)
    assert isinstance(rebuilt, LLMCallDeadlineExceeded)
    assert rebuilt.deadline_s == 12.5
    assert rebuilt.category is LLMErrorCategory.TRANSIENT
    assert rebuilt.request_id == "r5"
