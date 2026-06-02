"""Unit tests for the ``_resolve_github_identity`` helper in
``sessions.py``. The wider POST /sessions/ handler needs a live
cluster (Ray + session manager + db pool); we test only the pure
composition helper here, since it's the bit that determines whether
agents see the right identity in metadata.

See P4 of ``colony/github_identity_fix_plan.md``.
"""

from __future__ import annotations

from polymathera.colony.web_ui.backend.routers.sessions import (
    _resolve_github_identity,
)


def test_resolve_with_both_rows_populated() -> None:
    result = _resolve_github_identity(
        tenant_row={"installation_id": "12345"},
        user_row={
            "github_login": "anassar",
            "github_user_id": 42,
            "github_email": "ali@example.com",
            "git_user_name": "Ali Nassar",
        },
    )
    assert result == {
        "tenant_installation_id": "12345",
        "user_github_login": "anassar",
        "user_github_id": 42,
        "git_user_email": "ali@example.com",
        "git_user_name": "Ali Nassar",
    }


def test_resolve_when_user_not_oauthd_keeps_tenant_fields() -> None:
    """Tenant has installed the App, but the human user hasn't run
    the OAuth flow yet — installation_id flows, all user fields are
    ``None``."""

    result = _resolve_github_identity(
        tenant_row={"installation_id": "12345"},
        user_row=None,
    )
    assert result["tenant_installation_id"] == "12345"
    assert result["user_github_login"] is None
    assert result["user_github_id"] is None
    assert result["git_user_email"] is None
    assert result["git_user_name"] is None


def test_resolve_when_tenant_not_configured_keeps_user_fields() -> None:
    """Tenant hasn't pasted in an installation id, but the user has
    OAuth'd their identity — user fields flow, installation_id is
    ``None``."""

    result = _resolve_github_identity(
        tenant_row=None,
        user_row={
            "github_login": "anassar",
            "github_user_id": 42,
            "github_email": "ali@example.com",
            "git_user_name": "Ali Nassar",
        },
    )
    assert result["tenant_installation_id"] is None
    assert result["user_github_login"] == "anassar"


def test_resolve_with_neither_returns_all_none() -> None:
    """Fresh deployment, neither side configured. Every field is
    ``None``; downstream readers handle this explicitly."""

    result = _resolve_github_identity(tenant_row=None, user_row=None)
    assert result == {
        "tenant_installation_id": None,
        "user_github_login": None,
        "user_github_id": None,
        "git_user_email": None,
        "git_user_name": None,
    }


def test_resolve_handles_partial_user_row() -> None:
    """A user row missing optional fields (e.g. ``git_user_name`` is
    ``None`` on GitHub) doesn't raise; the missing fields surface as
    ``None`` in the metadata block."""

    result = _resolve_github_identity(
        tenant_row=None,
        user_row={
            "github_login": "anassar",
            "github_user_id": 42,
            "github_email": "ali@example.com",
            "git_user_name": None,
        },
    )
    assert result["git_user_name"] is None
    assert result["user_github_login"] == "anassar"
