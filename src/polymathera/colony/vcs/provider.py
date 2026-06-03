"""``VcsProvider`` protocol + small DTOs.

This protocol is grown incrementally — each method lands when a real
consumer needs it. Methods land alongside their consumers — no scaffolding stubs.

- PR 1 (OAuth trio): ``build_authorize_url`` /
  ``exchange_code_for_token`` / ``fetch_user_identity`` — wraps the
  existing ``auth/github_oauth.py`` helpers.
- PR 3 (first-login walker): ``list_user_tenants`` — for the
  ``services/user_tenant_sync.py`` walker that runs on every sign-in.
- PR 4 (colony discovery): ``list_tenant_repos`` +
  ``repo_path_exists`` — for the ``services/colony_discovery.py``
  ``.colony/`` probe.
- Future: ``mint_bot_credentials`` (PR 6).

See ``colony/vcs_native_tenancy_plan.md §1.1`` for the full target shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx


@dataclass(frozen=True)
class VcsUserIdentity:
    """A VCS user's verified identity after an OAuth sign-in.

    Provider-agnostic shape; each ``VcsProvider`` adapter normalises
    its provider-specific identity response into this DTO so callers
    (signup walker, ``users`` row populator, attribution resolver)
    never branch on provider id.
    """

    vcs_user_id: str
    """Provider's stable numeric user id, rendered as a string so
    we can store provider ids that don't fit in 64-bit (e.g. opaque
    UUIDs from a future provider) without re-typing the column."""

    login: str
    """The user's handle on the provider (e.g. GitHub "anassar")."""

    name: str | None
    """Display name from the provider's profile. May be ``None`` —
    not every provider requires a real name."""

    primary_email: str | None
    """The user's verified primary email, or ``None`` when the user
    has no verified email on file. Callers decide whether to refuse
    sign-in in that case."""

    verified_emails: tuple[str, ...] = field(default_factory=tuple)
    """All verified emails the user has on file. Primary first when
    one is marked primary; ordering otherwise per provider."""


@dataclass(frozen=True)
class VcsTenantRef:
    """A VCS organisation / group / workspace the authenticated user
    belongs to. Returned by :meth:`VcsProvider.list_user_tenants`.

    Provider-agnostic shape; the walker
    (``services/user_tenant_sync.py``) uses this to upsert ``tenants``
    rows + ``user_tenants`` membership without branching on provider id.
    """

    vcs_org_id: str
    """Provider's stable identifier for the org. Stringified so opaque
    UUID-style ids from a future provider don't need a column change."""

    vcs_org_login: str
    """The org's handle on the provider (e.g. GitHub "polymathera-inc")."""

    display_name: str
    """Human-facing name. Often the same as ``vcs_org_login`` (GitHub
    rarely sets a distinct display name); GitLab groups can differ."""

    installation_id: str | None
    """GitHub-specific: the App installation id for this org.
    ``None`` on providers that don't have a per-org App-installation
    concept (GitLab / Bitbucket use long-lived group/workspace tokens
    that live in ``tenants.bot_token_encrypted`` instead — see PR 6).
    GitHub providers MUST populate this — without it, server-to-server
    actions in this tenant can't mint a bot token."""

    role_hint: str
    """``"member"`` or ``"admin"``. Coarse role derived from the user's
    VCS-side org permission. ``user_tenants.role`` CHECK-constrains to
    these two strings; future per-colony refinement (read / write /
    admin) lives on the colony row + per-action permission checks."""


@dataclass(frozen=True)
class VcsRepoRef:
    """A repository inside a VCS tenant the authenticated user can
    access. Returned by :meth:`VcsProvider.list_tenant_repos`.

    Provider-agnostic shape; the colony discovery walker
    (``services/colony_discovery.py``) reads this to decide whether
    to auto-create a colony for the repo.
    """

    vcs_repo_id: str
    """Provider's stable repo identifier (stringified)."""

    full_name: str
    """``owner/repo`` form (e.g. ``polymathera-inc/monorepo``). Used as
    the colony's default human-facing name + the URL-template input
    for ``design_monorepo_url``."""

    default_branch: str
    """Repo's default branch (e.g. ``main``). Persisted on the colony
    row so the agent's first clone targets the right branch."""

    user_permission: str
    """``"read"`` / ``"write"`` / ``"admin"`` — the authenticated
    user's permission on this repo, normalised across providers
    (GitHub: read / triage / write / maintain / admin → folded to
    read / write / admin; GitLab Guest/Reporter → read,
    Developer/Maintainer → write, Owner → admin)."""


class OAuthExchangeError(RuntimeError):
    """Raised when a provider rejects an OAuth code exchange or the
    subsequent identity fetch fails. Caught at the router boundary so
    the user sees a clean 400/401 instead of a raw stack trace."""


@runtime_checkable
class VcsProvider(Protocol):
    """Abstract surface every VCS provider implements.

    Implementations are stateless (no per-request state held by the
    instance) so a single provider object lives for the dashboard's
    lifetime and is shared across requests.
    """

    provider_id: str
    """Short stable id used as the DB enum value, the URL path
    segment (``/auth/{provider_id}/sign-in``), and the registry
    key. MUST match the ``vcs_provider`` column values in
    ``users`` and ``tenants``."""

    display_name: str
    """Human-facing name for the UI provider picker."""

    def build_authorize_url(
        self,
        *,
        state: str,
        redirect_uri: str,
    ) -> str:
        """Render the URL the browser is redirected to so the user
        approves at the provider. ``state`` is the CSRF nonce the
        caller will verify in the callback; ``redirect_uri`` is the
        callback URL on Colony's side (must match what's registered
        with the provider byte-for-byte)."""
        ...

    async def exchange_code_for_token(
        self,
        *,
        code: str,
        redirect_uri: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> str:
        """Exchange the OAuth ``code`` returned in the callback for
        a user-to-server access token. Raises
        :class:`OAuthExchangeError` on rejection / malformed
        response. ``http_client`` is injectable for tests."""
        ...

    async def fetch_user_identity(
        self,
        *,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> VcsUserIdentity:
        """Fetch the user's identity using the access token.
        Normalises the provider's identity payload into
        :class:`VcsUserIdentity`. Raises
        :class:`OAuthExchangeError` if the identity-fetch HTTP call
        fails."""
        ...

    async def list_user_tenants(
        self,
        *,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[VcsTenantRef]:
        """List the orgs/groups/workspaces the authenticated user
        belongs to AND that have the Colony App / OAuth Consumer
        configured. For GitHub this is ``GET /user/installations``
        (orgs where Colony's App is installed AND the user can
        access it); for GitLab this will be
        ``GET /groups?min_access_level=20``; for Bitbucket
        ``GET /user/permissions/workspaces``.

        Returns an empty list when the user belongs to no
        Colony-enabled tenants — sign-in still succeeds; the user just
        sees no tenants in the dashboard until an admin installs the
        App in one of their orgs."""
        ...

    async def list_tenant_repos(
        self,
        *,
        tenant: VcsTenantRef,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[VcsRepoRef]:
        """List repos in ``tenant`` the authenticated user can
        access. GitHub: ``GET /user/installations/{id}/repositories``
        (returns the App-installation-scoped intersection of repos
        the App can see AND the user can see). GitLab will be
        ``GET /groups/{id}/projects``; Bitbucket
        ``GET /workspaces/{slug}/repositories``.

        The colony-discovery walker probes each returned repo for a
        ``.colony/`` directory and auto-creates a colony when found
        (see :meth:`repo_path_exists` + ``services/colony_discovery``)."""
        ...

    async def repo_path_exists(
        self,
        *,
        tenant: VcsTenantRef,
        repo: VcsRepoRef,
        path: str,
        access_token: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> bool:
        """Whether ``path`` exists in ``repo``'s default branch.

        For the colony discovery use case ``path`` is ``".colony"``
        — the directory marker that says "this repo is a Colony
        project". Providers MUST return ``False`` for both "file
        absent" and "directory absent at the given path" responses
        (GitHub returns 404 for either)."""
        ...

    def repo_clone_url(self, repo: VcsRepoRef) -> str:
        """Render the HTTPS clone URL for ``repo``. Used by the
        colony-discovery walker to populate
        ``colonies.design_monorepo_url`` so today's design-monorepo
        readers (UI textbox, ``RepoStateProvider``,
        ``materialize_design_context``) see a clone-ready URL without
        requiring the operator to type one. Pure string formatting —
        no network, no auth header — so it's a sync method.

        GitHub: ``https://github.com/{full_name}.git``.
        GitLab: ``{base_url}/{full_name}.git``.
        Bitbucket: ``https://bitbucket.org/{full_name}.git``."""
        ...


__all__ = (
    "OAuthExchangeError",
    "VcsProvider",
    "VcsRepoRef",
    "VcsTenantRef",
    "VcsUserIdentity",
)
