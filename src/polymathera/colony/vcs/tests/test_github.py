"""Tests for :class:`GitHubProvider`.

Only the normalisation + delegation logic is exercised — the
underlying ``auth.github_oauth`` helpers have their own coverage in
``web_ui/backend/routers/tests/test_github_oauth.py``. These tests
pin:

- Construction validation (empty creds raise).
- ``build_authorize_url`` passes through with the right scope.
- ``exchange_code_for_token`` delegates with the right args.
- ``fetch_user_identity`` normalises the dict into
  ``VcsUserIdentity`` (the only non-trivial wrapper logic).
- ``VcsProvider`` runtime-protocol check passes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from polymathera.colony.vcs import (
    OAuthExchangeError,
    VcsProvider,
    VcsRepoRef,
    VcsTenantRef,
    VcsUserIdentity,
)
from polymathera.colony.vcs.github import GitHubProvider


def _make_provider() -> GitHubProvider:
    return GitHubProvider(
        oauth_client_id="Iv23liExampleClientId",
        oauth_client_secret="example-secret-not-real",
    )


# ---------------------------------------------------------------------
# Construction


def test_construct_requires_non_empty_client_id() -> None:
    with pytest.raises(ValueError, match="oauth_client_id"):
        GitHubProvider(oauth_client_id="", oauth_client_secret="x")


def test_construct_requires_non_empty_client_secret() -> None:
    with pytest.raises(ValueError, match="oauth_client_secret"):
        GitHubProvider(oauth_client_id="x", oauth_client_secret="")


def test_provider_satisfies_runtime_protocol() -> None:
    """Runtime ``isinstance`` against the ``@runtime_checkable``
    Protocol — guards against silently dropping a required method."""
    assert isinstance(_make_provider(), VcsProvider)


# ---------------------------------------------------------------------
# build_authorize_url


def test_build_authorize_url_includes_user_email_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must call the underlying helper with
    ``scopes=('user:email',)`` so the verified-email fetch works."""
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "https://github.com/login/oauth/authorize?stub"

    monkeypatch.setattr(
        "polymathera.colony.vcs.github._gh_build_authorize_url",
        fake_build,
    )

    p = _make_provider()
    url = p.build_authorize_url(
        state="nonce123",
        redirect_uri="http://localhost:8080/api/v1/auth/github/callback",
    )
    assert url == "https://github.com/login/oauth/authorize?stub"
    assert captured["client_id"] == "Iv23liExampleClientId"
    assert captured["redirect_uri"].endswith("/auth/github/callback")
    assert captured["state"] == "nonce123"
    assert captured["scopes"] == ("user:email",)


# ---------------------------------------------------------------------
# exchange_code_for_token


@pytest.mark.asyncio
async def test_exchange_code_for_token_passes_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_exchange(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "gho_test_token"

    monkeypatch.setattr(
        "polymathera.colony.vcs.github._gh_exchange_code",
        fake_exchange,
    )

    p = _make_provider()
    token = await p.exchange_code_for_token(
        code="abc", redirect_uri="http://localhost:8080/cb",
    )
    assert token == "gho_test_token"
    assert captured["client_id"] == "Iv23liExampleClientId"
    assert captured["client_secret"] == "example-secret-not-real"
    assert captured["code"] == "abc"
    assert captured["redirect_uri"] == "http://localhost:8080/cb"


# ---------------------------------------------------------------------
# fetch_user_identity — the only non-trivial wrapper logic


@pytest.mark.asyncio
async def test_fetch_user_identity_normalises_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polymathera.colony.vcs.github._gh_fetch_identity",
        AsyncMock(return_value={
            "login": "anassar",
            "github_user_id": 123456,
            "name": "Test User",
            "verified_emails": [
                {"email": "a@example.com", "primary": True,  "verified": True},
                {"email": "b@example.com", "primary": False, "verified": True},
            ],
            "primary_email": "a@example.com",
        }),
    )

    identity = await _make_provider().fetch_user_identity(
        access_token="gho_x",
    )
    assert identity == VcsUserIdentity(
        vcs_user_id="123456",           # int -> str so the DTO contract holds across providers
        login="anassar",
        name="Test User",
        primary_email="a@example.com",
        verified_emails=("a@example.com", "b@example.com"),
    )


@pytest.mark.asyncio
async def test_fetch_user_identity_no_verified_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user with no verified emails on file: ``primary_email`` is
    None, ``verified_emails`` is empty. The DTO conveys this faithfully
    so the caller decides whether to reject the sign-in."""
    monkeypatch.setattr(
        "polymathera.colony.vcs.github._gh_fetch_identity",
        AsyncMock(return_value={
            "login": "anonuser",
            "github_user_id": 42,
            "name": None,
            "verified_emails": [],
            "primary_email": None,
        }),
    )

    identity = await _make_provider().fetch_user_identity(
        access_token="gho_x",
    )
    assert identity.primary_email is None
    assert identity.verified_emails == ()
    assert identity.name is None


@pytest.mark.asyncio
async def test_fetch_user_identity_missing_required_fields_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If GitHub returns 200 but the response lacks ``login`` or
    ``github_user_id``, we raise OAuthExchangeError so the router
    returns a clean 4xx instead of trying to persist a half-identity."""
    monkeypatch.setattr(
        "polymathera.colony.vcs.github._gh_fetch_identity",
        AsyncMock(return_value={
            "login": None,  # malformed
            "github_user_id": 123,
            "name": None,
            "verified_emails": [],
            "primary_email": None,
        }),
    )

    with pytest.raises(OAuthExchangeError, match="required fields"):
        await _make_provider().fetch_user_identity(access_token="gho_x")


# ---------------------------------------------------------------------
# list_user_tenants — GET /user/installations


def _make_installations_response(status_code: int, body: dict) -> Any:
    """Build an httpx-Response-shaped fake. The provider's
    ``list_user_tenants`` reads ``.status_code``, ``.text``, and
    ``.json()`` — that's the full contact surface."""
    resp = type("FakeResp", (), {})()
    resp.status_code = status_code
    resp.text = "fake-body"
    resp.json = lambda: body
    return resp


@pytest.mark.asyncio
async def test_list_user_tenants_normalises_installations() -> None:
    """Two installations in the GitHub response → two VcsTenantRef
    DTOs. Verify every field is normalised correctly."""
    payload = {
        "total_count": 2,
        "installations": [
            {
                "id": 42,
                "account": {
                    "id": 1001,
                    "login": "polymathera-inc",
                    "name": "Polymathera Inc.",
                    "type": "Organization",
                },
            },
            {
                "id": 99,
                "account": {
                    "id": 2002,
                    "login": "another-org",
                    "name": None,        # no display name set
                    "type": "Organization",
                },
            },
        ],
    }

    class _FakeClient:
        async def get(self, url: str, headers: dict) -> Any:
            assert url == "https://api.github.com/user/installations"
            assert headers["Authorization"] == "Bearer gho_x"
            return _make_installations_response(200, payload)

    refs = await _make_provider().list_user_tenants(
        access_token="gho_x", http_client=_FakeClient(),
    )
    assert refs == [
        VcsTenantRef(
            vcs_org_id="1001",
            vcs_org_login="polymathera-inc",
            display_name="Polymathera Inc.",
            installation_id="42",
            role_hint="member",
        ),
        # display_name falls back to login when account.name is None.
        VcsTenantRef(
            vcs_org_id="2002",
            vcs_org_login="another-org",
            display_name="another-org",
            installation_id="99",
            role_hint="member",
        ),
    ]


@pytest.mark.asyncio
async def test_list_user_tenants_empty_when_no_installations() -> None:
    """User belongs to no Colony-installed orgs → empty list, NOT an
    error. Sign-in still succeeds; the user sees no tenants."""

    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(
                200, {"total_count": 0, "installations": []},
            )

    refs = await _make_provider().list_user_tenants(
        access_token="gho_x", http_client=_FakeClient(),
    )
    assert refs == []


@pytest.mark.asyncio
async def test_list_user_tenants_4xx_raises() -> None:
    """GitHub-side 4xx (revoked token, scope missing) surfaces as
    OAuthExchangeError so the sign-in handler can return a clean 4xx."""

    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(401, {})

    with pytest.raises(OAuthExchangeError, match="installations failed"):
        await _make_provider().list_user_tenants(
            access_token="gho_revoked", http_client=_FakeClient(),
        )


@pytest.mark.asyncio
async def test_list_user_tenants_skips_malformed_entries() -> None:
    """A partial GitHub response (missing id/account) drops the
    affected installation but lets the rest through — defensive."""
    payload = {
        "installations": [
            {"id": None, "account": {"id": 1, "login": "x"}},  # no install_id
            {"id": 99, "account": {}},                          # empty account
            {
                "id": 42,
                "account": {"id": 7, "login": "good", "name": "Good"},
            },
        ],
    }

    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(200, payload)

    refs = await _make_provider().list_user_tenants(
        access_token="gho_x", http_client=_FakeClient(),
    )
    assert len(refs) == 1
    assert refs[0].vcs_org_login == "good"


# ---------------------------------------------------------------------
# list_tenant_repos — GET /user/installations/{id}/repositories


def _make_tenant_ref(installation_id: str | None = "100") -> VcsTenantRef:
    return VcsTenantRef(
        vcs_org_id="org_id_str",
        vcs_org_login="acme",
        display_name="Acme",
        installation_id=installation_id,
        role_hint="member",
    )


@pytest.mark.asyncio
async def test_list_tenant_repos_normalises_repos() -> None:
    """Three repos with mixed permission shapes → three VcsRepoRef
    DTOs with the right normalised ``user_permission``."""
    payload = {
        "total_count": 3,
        "repositories": [
            {
                "id": 11, "full_name": "acme/admin-repo",
                "default_branch": "main",
                "permissions": {"admin": True, "push": True, "pull": True},
            },
            {
                "id": 22, "full_name": "acme/writer-repo",
                "default_branch": "dev",
                "permissions": {"admin": False, "push": True, "pull": True},
            },
            {
                "id": 33, "full_name": "acme/reader-repo",
                "default_branch": "main",
                "permissions": {"admin": False, "push": False, "pull": True},
            },
        ],
    }

    class _FakeClient:
        async def get(self, url: str, headers: dict) -> Any:
            assert url == (
                "https://api.github.com/user/installations/100/repositories"
            )
            assert headers["Authorization"] == "Bearer gho_x"
            return _make_installations_response(200, payload)

    refs = await _make_provider().list_tenant_repos(
        tenant=_make_tenant_ref(),
        access_token="gho_x",
        http_client=_FakeClient(),
    )
    assert refs == [
        VcsRepoRef(vcs_repo_id="11", full_name="acme/admin-repo",
                   default_branch="main", user_permission="admin"),
        VcsRepoRef(vcs_repo_id="22", full_name="acme/writer-repo",
                   default_branch="dev", user_permission="write"),
        VcsRepoRef(vcs_repo_id="33", full_name="acme/reader-repo",
                   default_branch="main", user_permission="read"),
    ]


@pytest.mark.asyncio
async def test_list_tenant_repos_empty_when_no_installation_id() -> None:
    """Tenant ref without installation_id (non-GitHub origin, or a
    GitHub tenant pre-install) returns [] without hitting the API."""

    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            raise AssertionError("must not call HTTP")

    refs = await _make_provider().list_tenant_repos(
        tenant=_make_tenant_ref(installation_id=None),
        access_token="gho_x",
        http_client=_FakeClient(),
    )
    assert refs == []


@pytest.mark.asyncio
async def test_list_tenant_repos_4xx_raises() -> None:
    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(403, {})

    with pytest.raises(OAuthExchangeError, match="repositories failed"):
        await _make_provider().list_tenant_repos(
            tenant=_make_tenant_ref(),
            access_token="gho_x", http_client=_FakeClient(),
        )


@pytest.mark.asyncio
async def test_list_tenant_repos_skips_malformed_entries() -> None:
    payload = {
        "repositories": [
            {"id": None, "full_name": "acme/x", "default_branch": "main",
             "permissions": {"pull": True}},                # no id
            {"id": 22, "full_name": None, "default_branch": "main",
             "permissions": {"pull": True}},                # no full_name
            {"id": 33, "full_name": "acme/no-branch",
             "default_branch": None,
             "permissions": {"pull": True}},                # no default_branch
            {"id": 44, "full_name": "acme/good",
             "default_branch": "main",
             "permissions": {"pull": True}},
        ],
    }

    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(200, payload)

    refs = await _make_provider().list_tenant_repos(
        tenant=_make_tenant_ref(),
        access_token="gho_x", http_client=_FakeClient(),
    )
    assert [r.full_name for r in refs] == ["acme/good"]


# ---------------------------------------------------------------------
# repo_path_exists — GET /repos/{owner}/{repo}/contents/{path}


def _make_repo_ref(full_name: str = "acme/repo") -> VcsRepoRef:
    return VcsRepoRef(
        vcs_repo_id="42",
        full_name=full_name,
        default_branch="main",
        user_permission="write",
    )


@pytest.mark.asyncio
async def test_repo_path_exists_returns_true_on_200() -> None:
    class _FakeClient:
        async def get(self, url: str, headers: dict) -> Any:
            assert url == (
                "https://api.github.com/repos/acme/repo/contents/.colony"
            )
            assert headers["Authorization"] == "Bearer gho_x"
            return _make_installations_response(200, {})

    assert await _make_provider().repo_path_exists(
        tenant=_make_tenant_ref(),
        repo=_make_repo_ref(),
        path=".colony",
        access_token="gho_x",
        http_client=_FakeClient(),
    ) is True


@pytest.mark.asyncio
async def test_repo_path_exists_returns_false_on_404() -> None:
    """A repo without ``.colony/`` → 404 → False, NOT an exception.
    Discovery walks lots of repos; treating 404 as an error would
    abort the walk on the first non-Colony repo."""
    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(404, {})

    assert await _make_provider().repo_path_exists(
        tenant=_make_tenant_ref(),
        repo=_make_repo_ref(),
        path=".colony",
        access_token="gho_x",
        http_client=_FakeClient(),
    ) is False


@pytest.mark.asyncio
async def test_repo_path_exists_strips_leading_slash() -> None:
    """Caller passes ``"/.colony"`` or ``".colony"`` interchangeably —
    URL renders identically (defensive against caller variance)."""
    captured: dict[str, str] = {}

    class _FakeClient:
        async def get(self, url: str, headers: dict) -> Any:
            captured["url"] = url
            return _make_installations_response(404, {})

    await _make_provider().repo_path_exists(
        tenant=_make_tenant_ref(),
        repo=_make_repo_ref(),
        path="/.colony",
        access_token="gho_x",
        http_client=_FakeClient(),
    )
    assert captured["url"].endswith("/contents/.colony")


@pytest.mark.asyncio
async def test_repo_path_exists_other_status_raises() -> None:
    """Non-200 / non-404 (e.g. 5xx, 403 rate-limit) → exception
    so the caller can decide whether to retry."""
    class _FakeClient:
        async def get(self, *_a, **_kw) -> Any:
            return _make_installations_response(500, {})

    with pytest.raises(OAuthExchangeError, match="500"):
        await _make_provider().repo_path_exists(
            tenant=_make_tenant_ref(),
            repo=_make_repo_ref(),
            path=".colony",
            access_token="gho_x",
            http_client=_FakeClient(),
        )


# ---------------------------------------------------------------------
# repo_clone_url — pure formatting


def test_repo_clone_url_is_github_https() -> None:
    """Pure string formatting; renders ``https://github.com/{full_name}.git``.
    The colony-discovery walker writes this into
    ``colonies.design_monorepo_url`` so the legacy UI textbox + the
    ``materialize_design_context`` caller see a clone-ready URL."""
    p = _make_provider()
    repo = VcsRepoRef(
        vcs_repo_id="1", full_name="Polymathera/monorepo_opm_meg",
        default_branch="main", user_permission="write",
    )
    assert p.repo_clone_url(repo) == (
        "https://github.com/Polymathera/monorepo_opm_meg.git"
    )
