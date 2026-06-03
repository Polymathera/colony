"""GitHub adapter for :class:`VcsProvider`.

Wraps the existing OAuth helpers in
``web_ui/backend/auth/github_oauth.py`` — the wrapping is deliberately
thin: the helpers' shape was already provider-shaped (just hardcoded
to GitHub URLs), so this adapter mostly normalises the identity
response into :class:`VcsUserIdentity`.

The bot-credential side of the protocol
(``mint_bot_credentials`` — to be added in PR 6) will wrap
``agents/patterns/capabilities/_github/factory.py::
build_github_client_for_installation``. The two existing call sites
keep using that factory directly until PR 6 lands.
"""

from __future__ import annotations

import httpx

from ..web_ui.backend.auth.github_oauth import (
    build_authorize_url as _gh_build_authorize_url,
    exchange_code_for_token as _gh_exchange_code,
    fetch_authenticated_identity as _gh_fetch_identity,
)
from .provider import (
    OAuthExchangeError,
    VcsRepoRef,
    VcsTenantRef,
    VcsUserIdentity,
)


_GITHUB_API_USER_INSTALLATIONS = "https://api.github.com/user/installations"
_GITHUB_API_BASE = "https://api.github.com"


def _gh_auth_headers(access_token: str) -> dict[str, str]:
    """Standard auth headers for GitHub user-to-server REST calls.
    Centralised so the three GET-shaped methods (``list_user_tenants``,
    ``list_tenant_repos``, ``repo_path_exists``) stay consistent."""
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _normalise_repo_permission(perms: dict) -> str:
    """Fold GitHub's 5-tier permission booleans into the 3-tier shape
    Colony stores (``read`` / ``write`` / ``admin``). ``admin`` wins
    over ``push`` wins over ``pull``."""
    if perms.get("admin"):
        return "admin"
    if perms.get("push") or perms.get("maintain"):
        return "write"
    # ``pull`` covers both ``read`` and ``triage`` since both grant
    # read access; the distinction doesn't matter for Colony scoping.
    if perms.get("pull") or perms.get("triage"):
        return "read"
    # Unknown permission shape — fail closed.
    return "read"


class GitHubProvider:
    """``VcsProvider`` implementation for GitHub (.com and Enterprise
    Cloud — Enterprise Server's different base URL is a follow-up).

    Instances are stateless and shared across requests. Construct one
    at dashboard startup (see ``main.lifespan``) with the deploy-wide
    OAuth client credentials and register it via
    :func:`register_provider`.
    """

    provider_id: str = "github"
    display_name: str = "GitHub"

    def __init__(
        self,
        *,
        oauth_client_id: str,
        oauth_client_secret: str,
    ) -> None:
        if not oauth_client_id or not oauth_client_secret:
            # Construct-time error so the dashboard fails to start
            # with a clear message instead of crashing on first
            # /auth/github/sign-in.
            raise ValueError(
                "GitHubProvider requires non-empty oauth_client_id "
                "and oauth_client_secret. Set GITHUB_APP_CLIENT_ID "
                "and GITHUB_APP_CLIENT_SECRET in .env (see "
                "docs/guides/github-app-setup.md §2).",
            )
        self._oauth_client_id = oauth_client_id
        self._oauth_client_secret = oauth_client_secret

    def build_authorize_url(
        self,
        *,
        state: str,
        redirect_uri: str,
    ) -> str:
        return _gh_build_authorize_url(
            client_id=self._oauth_client_id,
            redirect_uri=redirect_uri,
            state=state,
            # Default scopes match the existing connect-flow caller
            # (``user:email`` — required for the verified-email fetch).
            scopes=("user:email",),
        )

    async def exchange_code_for_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> str:
        # The underlying helper raises ``OAuthExchangeError`` from
        # ``auth.github_oauth`` — re-export via this module's
        # ``OAuthExchangeError`` symbol so consumers only import from
        # ``vcs`` (kept identical class so existing ``except`` blocks
        # in the OAuth router stay correct after PR 3 rewires them
        # through the provider).
        return await _gh_exchange_code(
            client_id=self._oauth_client_id,
            client_secret=self._oauth_client_secret,
            code=code,
            redirect_uri=redirect_uri,
            http_client=http_client,
        )

    async def fetch_user_identity(
        self,
        *,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> VcsUserIdentity:
        raw = await _gh_fetch_identity(
            user_token=access_token, http_client=http_client,
        )
        github_user_id = raw.get("github_user_id")
        login = raw.get("login")
        if not github_user_id or not login:
            # The /user endpoint returned 200 but the response is
            # missing identity fields we require — treat as an
            # auth-flow failure so the caller can surface a clean
            # error to the user.
            raise OAuthExchangeError(
                "GitHub /user response missing required fields "
                "(github_user_id / login).",
            )
        verified_emails = tuple(
            e["email"]
            for e in raw.get("verified_emails", [])
            if e.get("email")
        )
        return VcsUserIdentity(
            vcs_user_id=str(github_user_id),
            login=str(login),
            name=raw.get("name"),
            primary_email=raw.get("primary_email"),
            verified_emails=verified_emails,
        )

    async def list_user_tenants(
        self,
        *,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[VcsTenantRef]:
        """``GET /user/installations`` returns every installation of
        the Colony App that the authenticated user can access.

        Each installation maps to one ``VcsTenantRef``:

        - ``vcs_org_id`` = ``account.id`` (stringified)
        - ``vcs_org_login`` = ``account.login`` (org handle)
        - ``installation_id`` = ``id`` (App installation id, used
          by ``_github/factory.py`` to mint installation tokens for
          server-to-server actions in this tenant).
        - ``role_hint`` defaults to ``"member"``. GitHub doesn't
          expose org-role on the installations endpoint; the walker
          accepts this default and refines it later if/when the user
          performs an admin-only action (which will fail with a clean
          403 that surfaces the real GitHub-side role).
        """

        headers = _gh_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(
                    _GITHUB_API_USER_INSTALLATIONS, headers=headers,
                )
        else:
            resp = await http_client.get(
                _GITHUB_API_USER_INSTALLATIONS, headers=headers,
            )

        if resp.status_code >= 400:
            raise OAuthExchangeError(
                f"GET /user/installations failed: {resp.status_code} "
                f"{resp.text[:200]}",
            )

        payload = resp.json()
        installations = payload.get("installations", [])
        refs: list[VcsTenantRef] = []
        for entry in installations:
            account = entry.get("account") or {}
            org_id = account.get("id")
            org_login = account.get("login")
            installation_id = entry.get("id")
            if org_id is None or not org_login or installation_id is None:
                # Skip malformed entries silently — partial GitHub
                # downtime sometimes returns half-populated rows.
                continue
            refs.append(VcsTenantRef(
                vcs_org_id=str(org_id),
                vcs_org_login=str(org_login),
                display_name=str(account.get("name") or org_login),
                installation_id=str(installation_id),
                role_hint="member",
            ))
        return refs

    async def list_tenant_repos(
        self,
        *,
        tenant: VcsTenantRef,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[VcsRepoRef]:
        """``GET /user/installations/{installation_id}/repositories``
        returns the App-installation-scoped intersection of repos the
        App has access to AND the authenticated user can see.

        Skips entries missing required fields (defensive against
        partial GitHub downtime). The walker iterates over the
        returned list and probes each for a ``.colony/`` directory
        via :meth:`repo_path_exists`.

        Requires ``tenant.installation_id`` — the GitHub-specific
        installation handle. Returns an empty list when absent
        (cleanly handles GitLab/Bitbucket tenants if a future caller
        passes a non-GitHub ``VcsTenantRef`` by mistake; production
        callers funnel through the registry so the right provider
        receives the right tenant)."""

        if not tenant.installation_id:
            return []

        url = (
            f"{_GITHUB_API_BASE}/user/installations/"
            f"{tenant.installation_id}/repositories"
        )
        headers = _gh_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(url, headers=headers)
        else:
            resp = await http_client.get(url, headers=headers)

        if resp.status_code >= 400:
            raise OAuthExchangeError(
                f"GET /user/installations/{tenant.installation_id}/"
                f"repositories failed: {resp.status_code} "
                f"{resp.text[:200]}",
            )

        payload = resp.json()
        repos = payload.get("repositories", [])
        refs: list[VcsRepoRef] = []
        for entry in repos:
            repo_id = entry.get("id")
            full_name = entry.get("full_name")
            default_branch = entry.get("default_branch")
            perms = entry.get("permissions") or {}
            if repo_id is None or not full_name or not default_branch:
                # Skip partial entries (rare; partial-downtime hedge).
                continue
            refs.append(VcsRepoRef(
                vcs_repo_id=str(repo_id),
                full_name=str(full_name),
                default_branch=str(default_branch),
                user_permission=_normalise_repo_permission(perms),
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
        """``GET /repos/{owner}/{repo}/contents/{path}`` — 200 ⇒ path
        exists (file or directory), 404 ⇒ absent. Any other status
        raises :class:`OAuthExchangeError` so the caller can decide
        whether to retry the whole tenant or skip just this repo.

        ``tenant`` is accepted for protocol uniformity (GitLab will
        need it when its adapter lands — group context matters for
        nested-project URLs); GitHub doesn't read it because the
        ``repo.full_name`` already encodes the owner."""

        del tenant  # Unused for GitHub; reserved for future providers.

        url = (
            f"{_GITHUB_API_BASE}/repos/{repo.full_name}/"
            f"contents/{path.lstrip('/')}"
        )
        headers = _gh_auth_headers(access_token)

        if http_client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ) as client:
                resp = await client.get(url, headers=headers)
        else:
            resp = await http_client.get(url, headers=headers)

        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        raise OAuthExchangeError(
            f"GET /repos/{repo.full_name}/contents/{path} failed: "
            f"{resp.status_code} {resp.text[:200]}",
        )

    def repo_clone_url(self, repo: VcsRepoRef) -> str:
        """``https://github.com/{full_name}.git`` — the standard
        HTTPS clone URL. Auth is per-process: the agent-side git
        credential helper (``distributed/git_credentials.py``)
        supplies the installation token at clone time."""
        return f"https://github.com/{repo.full_name}.git"


__all__ = ("GitHubProvider",)
