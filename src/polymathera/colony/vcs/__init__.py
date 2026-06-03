"""Provider-agnostic VCS (Version Control System) integration.

The single home for code that abstracts over GitHub, GitLab, Bitbucket
(and future providers). Every dashboard-side surface that needs to
talk to a VCS (OAuth signup, tenant discovery, repo discovery, bot
identity) goes through :class:`VcsProvider` rather than calling a
provider's HTTP endpoints directly.

See ``colony/vcs_native_tenancy_plan.md`` for the architectural plan
this package implements.
"""

from __future__ import annotations

from .provider import (
    OAuthExchangeError,
    VcsProvider,
    VcsRepoRef,
    VcsTenantRef,
    VcsUserIdentity,
)
from .registry import (
    enabled_providers,
    get_provider,
    register_provider,
)


__all__ = (
    "OAuthExchangeError",
    "VcsProvider",
    "VcsRepoRef",
    "VcsTenantRef",
    "VcsUserIdentity",
    "enabled_providers",
    "get_provider",
    "register_provider",
)
