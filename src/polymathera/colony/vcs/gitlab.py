"""GitLab adapter for :class:`VcsProvider`.

Implements the OAuth + discovery surface against GitLab REST v4. Per
``vcs_native_tenancy_plan.md §2``:

- Tenant unit: a top-level **group**. Subgroups are ignored for v1
  (their projects roll up to the parent group's tenant).
- OAuth: standard OAuth 2.0 Application (``/oauth/authorize``).
- User identity: ``GET /api/v4/user`` (no separate ``/emails``
  endpoint — GitLab returns ``email`` on the user payload directly,
  and the public_email/commit_email split is left to the future).
- Tenants: ``GET /api/v4/groups?min_access_level=20`` returns every
  group the user can see at Reporter (20) or higher.
- Repos: ``GET /api/v4/groups/{id}/projects`` returns projects in the
  group.
- Probe ``.colony/``: ``GET /api/v4/projects/{id}/repository/tree?path=.colony``
  returns 200 with a tree array (any rows) when the directory exists,
  404 when absent.

Bot identity (``mint_bot_credentials``) is deferred until an agent-
side GitLab consumer exists — plan §5.2 has the tenant admin paste a
Group Access Token into ``tenants.bot_token_encrypted`` (PR 6
encryption infrastructure already in place); the method gets added
to the protocol + this provider in the PR that introduces a
``GitLabCapability`` agent consumer.
"""

from __future__ import annotations

import httpx

from .provider import (
    OAuthExchangeError,
    VcsRepoRef,
    VcsTenantRef,
    VcsUserIdentity,
)


# ``min_access_level`` values from GitLab's group permission tiers.
# Reporter (20) is the threshold: Guests (10) can't access most APIs
# we care about, so we filter them out of the tenant list.
_REPORTER_ACCESS_LEVEL = 20


def _normalise_gitlab_permission(access_level: int) -> str:
    """Fold GitLab's 5-tier permission scale to Colony's 3-tier
    (``read`` / ``write`` / ``admin``):

    - Owner (50)               → admin
    - Maintainer (40)          → admin   (push, manage repo settings)
    - Developer (30)           → write   (push to non-protected branches)
    - Reporter (20)            → read
    - Guest (10)               → read    (rare; usually filtered out
                                          by ``min_access_level``)
    """
    if access_level >= 40:
        return "admin"
    if access_level >= 30:
        return "write"
    return "read"


def _gl_auth_headers(access_token: str) -> dict[str, str]:
    """Standard auth headers for GitLab REST v4 calls. GitLab accepts
    a bearer token for both OAuth user-to-server tokens and Personal/
    Group Access Tokens, so this same shape works for the walker AND
    the future ``mint_bot_credentials`` consumer."""
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }


class GitLabProvider:
    """``VcsProvider`` implementation for GitLab (gitlab.com + self-
    hosted CE/EE).

    Construct one at dashboard startup with the deploy-wide OAuth
    client credentials + the GitLab base URL (defaults to
    ``https://gitlab.com``; override for self-hosted). Instances are
    stateless and shared across requests.
    """

    provider_id: str = "gitlab"
    display_name: str = "GitLab"

    def __init__(
        self,
        *,
        oauth_client_id: str,
        oauth_client_secret: str,
        base_url: str = "https://gitlab.com",
    ) -> None:
        if not oauth_client_id or not oauth_client_secret:
            raise ValueError(
                "GitLabProvider requires non-empty oauth_client_id "
                "and oauth_client_secret. Set GITLAB_OAUTH_CLIENT_ID "
                "and GITLAB_OAUTH_CLIENT_SECRET in .env.",
            )
        self._oauth_client_id = oauth_client_id
        self._oauth_client_secret = oauth_client_secret
        # Strip a trailing slash so URL joins don't double-up.
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # OAuth trio

    def build_authorize_url(
        self,
        *,
        state: str,
        redirect_uri: str,
    ) -> str:
        """Render the GitLab authorize URL. ``scope=read_api`` lets us
        call ``GET /user`` + ``GET /groups`` + ``GET /projects`` +
        ``GET /repository/tree``; ``api`` would over-grant (write
        access) for what the sign-in walker needs."""
        from urllib.parse import urlencode

        params = {
            "client_id": self._oauth_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": "read_api",
        }
        return f"{self._base_url}/oauth/authorize?" + urlencode(params)

    async def exchange_code_for_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> str:
        """POST to ``/oauth/token`` with the authorization code."""
        payload = {
            "client_id": self._oauth_client_id,
            "client_secret": self._oauth_client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        url = f"{self._base_url}/oauth/token"

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.post(url, data=payload)
        else:
            resp = await http_client.post(url, data=payload)

        if resp.status_code >= 400:
            raise OAuthExchangeError(
                f"GitLab token exchange failed: {resp.status_code} "
                f"{resp.text[:200]}",
            )
        data = resp.json()
        if "error" in data:
            raise OAuthExchangeError(
                f"GitLab token exchange returned error: "
                f"{data.get('error')}: {data.get('error_description', '')}",
            )
        token = data.get("access_token")
        if not token:
            raise OAuthExchangeError(
                "GitLab token exchange response missing access_token",
            )
        return str(token)

    async def fetch_user_identity(
        self,
        *,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> VcsUserIdentity:
        """``GET /api/v4/user`` returns the user's profile. GitLab
        always includes ``email`` for the authenticated user (unlike
        GitHub's separate verified-emails endpoint)."""
        url = f"{self._base_url}/api/v4/user"
        headers = _gl_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(url, headers=headers)
        else:
            resp = await http_client.get(url, headers=headers)

        if resp.status_code >= 400:
            raise OAuthExchangeError(
                f"GitLab /user failed: {resp.status_code} "
                f"{resp.text[:200]}",
            )

        raw = resp.json()
        user_id = raw.get("id")
        login = raw.get("username")
        if not user_id or not login:
            raise OAuthExchangeError(
                "GitLab /user response missing required fields "
                "(id / username).",
            )
        email = raw.get("email") or None
        verified_emails = (email,) if email else ()
        return VcsUserIdentity(
            vcs_user_id=str(user_id),
            login=str(login),
            name=raw.get("name"),
            primary_email=email,
            verified_emails=verified_emails,
        )

    # ------------------------------------------------------------------
    # Tenant + repo discovery

    async def list_user_tenants(
        self,
        *,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[VcsTenantRef]:
        """``GET /api/v4/groups?min_access_level=20&top_level_only=true``.

        ``top_level_only=true`` matches plan §2 "Group (top-level;
        ignore subgroups for v1)". Subgroups would each become a
        separate tenant — workable in v2 but multiplies the row count
        without a clear win.
        """
        url = f"{self._base_url}/api/v4/groups"
        params = {
            "min_access_level": str(_REPORTER_ACCESS_LEVEL),
            "top_level_only": "true",
            "per_page": "100",
        }
        headers = _gl_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(url, headers=headers, params=params)
        else:
            resp = await http_client.get(url, headers=headers, params=params)

        if resp.status_code >= 400:
            raise OAuthExchangeError(
                f"GitLab /groups failed: {resp.status_code} "
                f"{resp.text[:200]}",
            )

        refs: list[VcsTenantRef] = []
        for entry in resp.json() or []:
            org_id = entry.get("id")
            path = entry.get("path") or entry.get("full_path")
            name = entry.get("name") or path
            if org_id is None or not path:
                continue
            # GitLab doesn't have an App-installation concept; bot
            # creds live in tenants.bot_token_encrypted (PR 6).
            # role_hint maps owner/maintainer to "admin", everything
            # else to "member" — matching the user_tenants CHECK.
            access_level = entry.get("access_level")
            if access_level is None:
                # Some payload shapes nest under ``shared_with_groups``;
                # default to member if absent.
                role_hint = "member"
            else:
                role_hint = "admin" if access_level >= 40 else "member"
            refs.append(VcsTenantRef(
                vcs_org_id=str(org_id),
                vcs_org_login=str(path),
                display_name=str(name),
                installation_id=None,
                role_hint=role_hint,
            ))
        return refs

    async def list_tenant_repos(
        self,
        *,
        tenant: VcsTenantRef,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[VcsRepoRef]:
        """``GET /api/v4/groups/{id}/projects`` returns the group's
        projects. Subgroup projects are excluded (parent-group only)
        — matching the plan §2 v1 scoping."""
        url = (
            f"{self._base_url}/api/v4/groups/{tenant.vcs_org_id}/"
            f"projects"
        )
        params = {
            # Match the walker's permission semantics: the user must
            # at least be able to clone the repo.
            "min_access_level": str(_REPORTER_ACCESS_LEVEL),
            "per_page": "100",
            "include_subgroups": "false",
        }
        headers = _gl_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(url, headers=headers, params=params)
        else:
            resp = await http_client.get(url, headers=headers, params=params)

        if resp.status_code >= 400:
            raise OAuthExchangeError(
                f"GitLab /groups/{tenant.vcs_org_id}/projects failed: "
                f"{resp.status_code} {resp.text[:200]}",
            )

        refs: list[VcsRepoRef] = []
        for entry in resp.json() or []:
            repo_id = entry.get("id")
            full_name = entry.get("path_with_namespace")
            default_branch = entry.get("default_branch")
            # ``permissions.project_access`` carries the user's
            # access level on this specific project; falls back to the
            # group-inherited level when absent.
            perms = entry.get("permissions") or {}
            project_access = (perms.get("project_access") or {})
            group_access = (perms.get("group_access") or {})
            access_level = (
                project_access.get("access_level")
                or group_access.get("access_level")
                or 0
            )
            if repo_id is None or not full_name or not default_branch:
                continue
            refs.append(VcsRepoRef(
                vcs_repo_id=str(repo_id),
                full_name=str(full_name),
                default_branch=str(default_branch),
                user_permission=_normalise_gitlab_permission(
                    int(access_level),
                ),
            ))
        return refs

    async def repo_path_exists(
        self,
        *,
        tenant: VcsTenantRef,
        repo: VcsRepoRef,
        path: str,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> bool:
        """``GET /api/v4/projects/{id}/repository/tree?path={path}``
        returns a JSON array of entries when the directory exists,
        404 when absent. An empty array (200 status) means the
        directory exists but has no contents — still True per the
        protocol contract ("path exists")."""

        del tenant  # GitLab encodes the project owner in repo's id.

        url = (
            f"{self._base_url}/api/v4/projects/{repo.vcs_repo_id}/"
            f"repository/tree"
        )
        params = {
            "path": path.lstrip("/"),
            "ref": repo.default_branch,
        }
        headers = _gl_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(url, headers=headers, params=params)
        else:
            resp = await http_client.get(url, headers=headers, params=params)

        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        raise OAuthExchangeError(
            f"GitLab /repository/tree failed for project "
            f"{repo.vcs_repo_id} path={path}: {resp.status_code} "
            f"{resp.text[:200]}",
        )

    def repo_clone_url(self, repo: VcsRepoRef) -> str:
        """``{base_url}/{full_name}.git`` — works for both
        gitlab.com and self-hosted GitLab. Auth is supplied at clone
        time by whatever credential helper the agent process wires
        up (today that's GitHub-only; GitLab git-cred plumbing lands
        when an agent-side GitLab capability arrives — see
        ``vcs_native_tenancy_plan.md §5.2``)."""
        return f"{self._base_url}/{repo.full_name}.git"


__all__ = ("GitLabProvider",)
