"""Tests for the ``Session.session_kind`` discriminator added in P8-0.

The field marks the colony-singleton "system" session apart from
chat-bound "user" sessions. Validator + backward-compat defaults must
hold; otherwise the chat-attach guard and the sessions-list filter
both misclassify sessions persisted before P8-0.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.sessions.models import (
    Session,
    SessionMetadata,
    SessionState,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


def _make_session(**overrides) -> Session:
    """Build a Session with the minimum required fields. Inside a
    serving execution context so the default ``syscontext`` field
    factory succeeds."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="session_abc", origin="test",
    ):
        metadata = SessionMetadata(created_by="test")
        kwargs = dict(
            session_id="session_abc",
            branch_id="branch_abc",
            state=SessionState.ACTIVE,
            metadata=metadata,
        )
        kwargs.update(overrides)
        return Session(**kwargs)


def test_session_kind_default_is_user() -> None:
    """A Session created without ``session_kind`` defaults to ``user``.

    Load-bearing for backward-compat: every Session blob persisted to
    SharedState pre-P8-0 lacks the field; on rehydrate Pydantic uses
    this default + the partition is unchanged."""

    session = _make_session()
    assert session.session_kind == "user"


def test_session_kind_accepts_system() -> None:
    """The system bootstrap path explicitly sets ``session_kind='system'``."""

    session = _make_session(session_kind="system")
    assert session.session_kind == "system"


def test_session_kind_rejects_unknown_value() -> None:
    """Literal validator catches typos so a misspelled value can't
    silently bypass the chat-attach guard or the list filter."""

    with pytest.raises(ValueError):
        _make_session(session_kind="systme")  # typo


def test_session_kind_rejects_empty_string() -> None:
    """Empty string is not a valid kind — would slip through a
    string-equality check against the wrong constant."""

    with pytest.raises(ValueError):
        _make_session(session_kind="")
