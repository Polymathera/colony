"""Tests for :class:`GitLabProvider`.

Mirrors ``test_github.py``'s shape: stub the httpx response via a
tiny fake client, monkeypatch only when an underlying ``cryptography``
or similar primitive is involved (none here — pure REST). Asserts on
the URL + headers + body the provider sends, and the DTO shape it
returns.
"""

from __future__ import annotations

from typing import Any

import pytest

from polymathera.colony.vcs import (
    OAuthExchangeError,
    VcsProvider,
    VcsRepoRef,
    VcsTenantRef,
    VcsUserIdentity,
)
from polymathera.colony.vcs.gitlab import GitLabProvider


# ---------------------------------------------------------------------
# Fake httpx client + response


def _make_resp(status_code: int, body: Any) -> Any:
    resp = type("FakeResp", (), {})()
    resp.status_code = status_code
    resp.text = str(body)[:200] if not isinstance(body, str) else body
    resp.json = lambda: body
    return resp


class _RecordingClient:
    """Records every call so tests can assert URL + headers + params."""
    def __init__(self, returns: Any) -> None:
        self._returns = returns
        self.calls: list[dict[str, Any]] = []

    async def get(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self._returns

    async def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self._returns


def _make_provider(base_url: str = "https://gitlab.com") -> GitLabProvider:
    return GitLabProvider(
        oauth_client_id="abc123",
        oauth_client_secret="secret-xyz",
        base_url=base_url,
    )


# ---------------------------------------------------------------------
# Construction


def test_construct_requires_client_id() -> None:
    with pytest.raises(ValueError, match="oauth_client_id"):
        GitLabProvider(oauth_client_id="", oauth_client_secret="x")


def test_construct_requires_client_secret() -> None:
    with pytest.raises(ValueError, match="oauth_client_secret"):
        GitLabProvider(oauth_client_id="x", oauth_client_secret="")


def test_construct_strips_trailing_base_url_slash() -> None:
    """Operators paste base URLs with or without a trailing slash;
    the URL joiner shouldn't double up."""
    p = GitLabProvider(
        oauth_client_id="x", oauth_client_secret="y",
        base_url="https://gitlab.acme.com/",
    )
    url = p.build_authorize_url(state="s", redirect_uri="https://cb")
    # No "//" between base and "/oauth/authorize".
    assert "gitlab.acme.com//oauth" not in url
    assert "gitlab.acme.com/oauth/authorize?" in url


def test_provider_satisfies_runtime_protocol() -> None:
    assert isinstance(_make_provider(), VcsProvider)


# ---------------------------------------------------------------------
# build_authorize_url


def test_authorize_url_uses_base_url_and_read_api_scope() -> None:
    p = _make_provider(base_url="https://gitlab.acme.com")
    url = p.build_authorize_url(
        state="nonce-1", redirect_uri="https://colony/cb",
    )
    assert url.startswith("https://gitlab.acme.com/oauth/authorize?")
    assert "client_id=abc123" in url
    assert "redirect_uri=https%3A%2F%2Fcolony%2Fcb" in url
    assert "state=nonce-1" in url
    # ``read_api`` not ``api`` — the walker only needs read access.
    assert "scope=read_api" in url
    # OAuth2 Authorization Code flow.
    assert "response_type=code" in url


# ---------------------------------------------------------------------
# exchange_code_for_token


@pytest.mark.asyncio
async def test_exchange_code_posts_to_token_endpoint() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, {"access_token": "glat_x"}))
    token = await p.exchange_code_for_token(
        code="abc", redirect_uri="https://cb", http_client=client,
    )
    assert token == "glat_x"
    assert client.calls[0]["url"] == "https://gitlab.com/oauth/token"
    sent = client.calls[0]["data"]
    assert sent["grant_type"] == "authorization_code"
    assert sent["client_id"] == "abc123"
    assert sent["client_secret"] == "secret-xyz"
    assert sent["code"] == "abc"
    assert sent["redirect_uri"] == "https://cb"


@pytest.mark.asyncio
async def test_exchange_code_4xx_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(401, "no"))
    with pytest.raises(OAuthExchangeError, match="failed: 401"):
        await p.exchange_code_for_token(
            code="abc", redirect_uri="x", http_client=client,
        )


@pytest.mark.asyncio
async def test_exchange_code_error_field_in_200_raises() -> None:
    """GitLab can return 200 with ``error`` in the body (similar to
    GitHub) — surface it as the same OAuthExchangeError shape."""
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, {
        "error": "invalid_grant",
        "error_description": "Expired code",
    }))
    with pytest.raises(OAuthExchangeError, match="invalid_grant"):
        await p.exchange_code_for_token(
            code="abc", redirect_uri="x", http_client=client,
        )


@pytest.mark.asyncio
async def test_exchange_code_missing_access_token_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, {"token_type": "Bearer"}))
    with pytest.raises(OAuthExchangeError, match="missing access_token"):
        await p.exchange_code_for_token(
            code="abc", redirect_uri="x", http_client=client,
        )


# ---------------------------------------------------------------------
# fetch_user_identity


@pytest.mark.asyncio
async def test_fetch_user_identity_normalises_happy_path() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, {
        "id": 42,
        "username": "anassar",
        "name": "A Nassar",
        "email": "a@example.com",
        "state": "active",
    }))
    identity = await p.fetch_user_identity(
        access_token="glat_x", http_client=client,
    )
    assert identity == VcsUserIdentity(
        vcs_user_id="42",
        login="anassar",
        name="A Nassar",
        primary_email="a@example.com",
        verified_emails=("a@example.com",),
    )
    assert client.calls[0]["url"] == "https://gitlab.com/api/v4/user"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer glat_x"


@pytest.mark.asyncio
async def test_fetch_user_identity_no_email() -> None:
    """Some GitLab accounts have no email surfaced via /user
    (privacy-restricted). DTO conveys ``primary_email=None``."""
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, {
        "id": 7, "username": "private", "name": None, "email": None,
    }))
    identity = await p.fetch_user_identity(
        access_token="glat_x", http_client=client,
    )
    assert identity.primary_email is None
    assert identity.verified_emails == ()


@pytest.mark.asyncio
async def test_fetch_user_identity_missing_required_fields_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, {
        "id": None, "username": "x",  # malformed id
    }))
    with pytest.raises(OAuthExchangeError, match="required fields"):
        await p.fetch_user_identity(
            access_token="glat_x", http_client=client,
        )


@pytest.mark.asyncio
async def test_fetch_user_identity_4xx_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(401, "denied"))
    with pytest.raises(OAuthExchangeError, match="failed: 401"):
        await p.fetch_user_identity(
            access_token="glat_revoked", http_client=client,
        )


# ---------------------------------------------------------------------
# list_user_tenants


@pytest.mark.asyncio
async def test_list_user_tenants_normalises_groups() -> None:
    """Two top-level groups, one Owner one Reporter, return refs
    with the right role_hint normalised to user_tenants CHECK values."""
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, [
        {
            "id": 1001, "path": "polymathera-inc",
            "name": "Polymathera Inc", "access_level": 50,  # Owner
        },
        {
            "id": 2002, "path": "thirdparty", "name": "Third Party",
            "access_level": 20,  # Reporter
        },
    ]))
    refs = await p.list_user_tenants(
        access_token="glat_x", http_client=client,
    )
    assert len(refs) == 2
    assert refs[0] == VcsTenantRef(
        vcs_org_id="1001",
        vcs_org_login="polymathera-inc",
        display_name="Polymathera Inc",
        installation_id=None,
        role_hint="admin",
    )
    assert refs[1].role_hint == "member"  # Reporter < Maintainer
    # Verify the request shape.
    call = client.calls[0]
    assert call["url"] == "https://gitlab.com/api/v4/groups"
    assert call["params"]["min_access_level"] == "20"
    assert call["params"]["top_level_only"] == "true"


@pytest.mark.asyncio
async def test_list_user_tenants_empty_list() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, []))
    refs = await p.list_user_tenants(
        access_token="glat_x", http_client=client,
    )
    assert refs == []


@pytest.mark.asyncio
async def test_list_user_tenants_skips_malformed_entries() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, [
        {"id": None, "path": "x"},                          # no id
        {"id": 9, "path": None},                            # no path
        {"id": 7, "path": "good", "name": "G", "access_level": 50},
    ]))
    refs = await p.list_user_tenants(
        access_token="glat_x", http_client=client,
    )
    assert [r.vcs_org_login for r in refs] == ["good"]


@pytest.mark.asyncio
async def test_list_user_tenants_4xx_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(403, "no"))
    with pytest.raises(OAuthExchangeError, match="/groups failed"):
        await p.list_user_tenants(
            access_token="glat_x", http_client=client,
        )


# ---------------------------------------------------------------------
# list_tenant_repos


def _tenant(org_id: str = "1001") -> VcsTenantRef:
    return VcsTenantRef(
        vcs_org_id=org_id, vcs_org_login="acme",
        display_name="Acme", installation_id=None,
        role_hint="member",
    )


@pytest.mark.asyncio
async def test_list_tenant_repos_normalises_with_permission_tiers() -> None:
    """Three projects with mixed permission tiers → three VcsRepoRef
    DTOs with normalised ``user_permission`` strings."""
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, [
        {
            "id": 11, "path_with_namespace": "acme/admin-repo",
            "default_branch": "main",
            "permissions": {
                "project_access": {"access_level": 50},  # Owner
                "group_access": None,
            },
        },
        {
            "id": 22, "path_with_namespace": "acme/writer-repo",
            "default_branch": "dev",
            "permissions": {
                "project_access": {"access_level": 30},  # Developer
                "group_access": None,
            },
        },
        {
            "id": 33, "path_with_namespace": "acme/reader-repo",
            "default_branch": "main",
            "permissions": {
                "project_access": None,
                "group_access": {"access_level": 20},     # Reporter
            },
        },
    ]))
    refs = await p.list_tenant_repos(
        tenant=_tenant("1001"), access_token="glat_x",
        http_client=client,
    )
    assert refs == [
        VcsRepoRef(vcs_repo_id="11", full_name="acme/admin-repo",
                   default_branch="main", user_permission="admin"),
        VcsRepoRef(vcs_repo_id="22", full_name="acme/writer-repo",
                   default_branch="dev", user_permission="write"),
        VcsRepoRef(vcs_repo_id="33", full_name="acme/reader-repo",
                   default_branch="main", user_permission="read"),
    ]
    # Verify the request shape includes the right group id + the
    # subgroups-excluded flag (matches plan §2 v1 scoping).
    assert client.calls[0]["url"] == (
        "https://gitlab.com/api/v4/groups/1001/projects"
    )
    assert client.calls[0]["params"]["include_subgroups"] == "false"


@pytest.mark.asyncio
async def test_list_tenant_repos_skips_partial_entries() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, [
        {"id": 1, "path_with_namespace": None,         # no full_name
         "default_branch": "main", "permissions": {}},
        {"id": 2, "path_with_namespace": "acme/x",
         "default_branch": None,                        # no branch
         "permissions": {}},
        {"id": 3, "path_with_namespace": "acme/good",
         "default_branch": "main",
         "permissions": {"project_access": {"access_level": 30}}},
    ]))
    refs = await p.list_tenant_repos(
        tenant=_tenant(), access_token="glat_x",
        http_client=client,
    )
    assert [r.full_name for r in refs] == ["acme/good"]


@pytest.mark.asyncio
async def test_list_tenant_repos_4xx_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(403, "denied"))
    with pytest.raises(OAuthExchangeError, match="/projects failed"):
        await p.list_tenant_repos(
            tenant=_tenant(), access_token="glat_x",
            http_client=client,
        )


# ---------------------------------------------------------------------
# repo_path_exists


def _repo() -> VcsRepoRef:
    return VcsRepoRef(
        vcs_repo_id="123", full_name="acme/repo",
        default_branch="main", user_permission="write",
    )


@pytest.mark.asyncio
async def test_repo_path_exists_returns_true_on_200() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(200, []))  # empty tree is fine
    assert await p.repo_path_exists(
        tenant=_tenant(), repo=_repo(),
        path=".colony", access_token="glat_x",
        http_client=client,
    ) is True
    # URL uses project id, NOT path_with_namespace — GitLab requires
    # the id for repository/tree.
    call = client.calls[0]
    assert call["url"] == (
        "https://gitlab.com/api/v4/projects/123/repository/tree"
    )
    assert call["params"]["path"] == ".colony"
    assert call["params"]["ref"] == "main"


@pytest.mark.asyncio
async def test_repo_path_exists_returns_false_on_404() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(404, ""))
    assert await p.repo_path_exists(
        tenant=_tenant(), repo=_repo(),
        path=".colony", access_token="glat_x",
        http_client=client,
    ) is False


@pytest.mark.asyncio
async def test_repo_path_exists_strips_leading_slash() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(404, ""))
    await p.repo_path_exists(
        tenant=_tenant(), repo=_repo(),
        path="/.colony", access_token="glat_x",
        http_client=client,
    )
    assert client.calls[0]["params"]["path"] == ".colony"


@pytest.mark.asyncio
async def test_repo_path_exists_other_status_raises() -> None:
    p = _make_provider()
    client = _RecordingClient(_make_resp(500, "server error"))
    with pytest.raises(OAuthExchangeError, match="500"):
        await p.repo_path_exists(
            tenant=_tenant(), repo=_repo(),
            path=".colony", access_token="glat_x",
            http_client=client,
        )


# ---------------------------------------------------------------------
# repo_clone_url — uses configured base URL


def test_repo_clone_url_uses_default_base_url() -> None:
    p = _make_provider()  # base_url=https://gitlab.com
    repo = VcsRepoRef(
        vcs_repo_id="1", full_name="acme/widget",
        default_branch="main", user_permission="write",
    )
    assert p.repo_clone_url(repo) == "https://gitlab.com/acme/widget.git"


def test_repo_clone_url_uses_self_hosted_base_url() -> None:
    p = _make_provider(base_url="https://gitlab.acme.internal")
    repo = VcsRepoRef(
        vcs_repo_id="1", full_name="acme/widget",
        default_branch="main", user_permission="write",
    )
    assert p.repo_clone_url(repo) == (
        "https://gitlab.acme.internal/acme/widget.git"
    )
