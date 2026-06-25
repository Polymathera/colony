"""Unit tests for the ``_resolve_github_identity`` helper.

Canonical home is
:func:`polymathera.colony.web_ui.backend.chat.user_session_factory._resolve_github_identity`;
the wider POST /sessions/ handler needs a live cluster (Ray + session
manager + db pool) so we test only the pure composition helper here.
The duplicate copy previously in ``routers/sessions.py`` was deleted
in R6-FIX-5 (it was never called from production code and forced
every fix to land in two places). The tests below kept their original
shape; only the import path moved.

See P4 of ``colony/github_identity_fix_plan.md``.
"""

from __future__ import annotations

from polymathera.colony.web_ui.backend.chat.user_session_factory import (
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
    ``None`` on GitHub — the profile-name field is OPTIONAL on
    GitHub) doesn't raise; ``git_user_name`` falls back to
    ``github_login`` (R6-FIX-5) so commit-attribution surfaces still
    have a non-None name even when the user has no display name
    set on their GitHub profile."""

    result = _resolve_github_identity(
        tenant_row=None,
        user_row={
            "github_login": "anassar",
            "github_user_id": 42,
            "github_email": "ali@example.com",
            "git_user_name": None,
        },
    )
    # R6-FIX-5: fallback to login. Previously this asserted
    # ``result["git_user_name"] is None`` which silently dropped the
    # Co-Authored-By: trailer when the user lacked a GitHub display
    # name. ``user.get("git_user_name") or user.get("github_login")``
    # produces the login when the name column is null.
    assert result["git_user_name"] == "anassar"
    assert result["user_github_login"] == "anassar"


def test_resolve_falls_back_to_login_when_git_user_name_is_empty_string(
) -> None:
    """Defensive: an empty-string ``git_user_name`` is as useless for
    commit attribution as ``None`` — Python's ``or`` treats both as
    falsy so the fallback fires the same way. Pin so a future
    re-write that uses ``if ... is None else ...`` ternary form
    (which would NOT trigger on empty string) surfaces here."""

    result = _resolve_github_identity(
        tenant_row=None,
        user_row={
            "github_login": "anassar",
            "github_user_id": 42,
            "github_email": "ali@example.com",
            "git_user_name": "",
        },
    )
    assert result["git_user_name"] == "anassar"


def test_resolve_does_not_fallback_when_login_also_missing(
) -> None:
    """If BOTH ``git_user_name`` and ``github_login`` are missing,
    the result is ``None`` — there's nothing to fall back to. This
    case is unreachable in production (OAuth requires a login) but
    pinning it documents the failure mode and prevents a refactor
    from accidentally fabricating a placeholder name."""

    result = _resolve_github_identity(
        tenant_row=None,
        user_row={
            "github_login": None,
            "github_user_id": None,
            "github_email": None,
            "git_user_name": None,
        },
    )
    assert result["git_user_name"] is None
    assert result["user_github_login"] is None
