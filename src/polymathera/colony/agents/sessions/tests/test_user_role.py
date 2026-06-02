"""Tests for the ``SessionMetadata.user_role`` field added in P12.

The field is forward-compat plumbing for RBAC (design doc §18 + §12)
— not enforced anywhere in v1. Tests pin the round-trip + default
so a future enforcement PR has a stable shape to read from."""

from __future__ import annotations

from polymathera.colony.agents.sessions.models import SessionMetadata
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


def _make_metadata(**overrides) -> SessionMetadata:
    """Build a SessionMetadata inside an exec context so its
    ``syscontext`` default factory succeeds."""
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="session_test", origin="test",
    ):
        kwargs = dict(created_by="test")
        kwargs.update(overrides)
        return SessionMetadata(**kwargs)


def test_user_role_default_is_none() -> None:
    """``None`` (default) — load-bearing for backward-compat. Every
    SessionMetadata blob persisted to SharedState pre-P12 lacks the
    field; on rehydrate Pydantic uses this default + the partition
    is unchanged."""

    md = _make_metadata()
    assert md.user_role is None


def test_user_role_accepts_single_role() -> None:
    md = _make_metadata(user_role=["operator"])
    assert md.user_role == ["operator"]


def test_user_role_accepts_multiple_roles() -> None:
    md = _make_metadata(user_role=["operator", "reviewer"])
    assert md.user_role == ["operator", "reviewer"]


def test_user_role_accepts_empty_list() -> None:
    """Empty list is distinct from ``None`` — operator explicitly
    cleared roles. The chat-UI filter treats both as "no roles
    declared" (sessions pass through) per the route's any-overlap
    semantics."""

    md = _make_metadata(user_role=[])
    assert md.user_role == []
